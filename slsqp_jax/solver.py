"""SLSQP Solver implementation using Optimistix.

This module contains the main SLSQP solver class that extends
optimistix.AbstractMinimiser to provide Sequential Quadratic Programming
optimization with support for equality and inequality constraints.

The QP subproblem always uses a frozen L-BFGS Hessian approximation,
which is O(kn) in both memory and per-product cost. When the user
supplies exact HVPs (obj_hvp_fn), these are probed once per main
iteration along the step direction to produce high-quality secant
pairs for the L-BFGS update, but the HVP is never called inside
the QP inner loop.

Gradients and Jacobians can be user-supplied or computed
automatically via jax.grad (reverse-mode) and jax.jacrev.
"""

from collections.abc import Callable
from typing import Any, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optimistix._misc as optx_misc
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.hessian import (
    LBFGSHistory,
    compute_lagrangian_gradient,
    lbfgs_append,
    lbfgs_hvp,
    lbfgs_identity_reset,
    lbfgs_init,
    lbfgs_inverse_hvp,
    lbfgs_reset,
)
from slsqp_jax.merit import (
    backtracking_line_search,
    compute_merit,
    update_penalty_parameter,
)
from slsqp_jax.qp_solver import solve_qp
from slsqp_jax.types import (
    ConstraintFn,
    ConstraintHVPFn,
    GradFn,
    HVPFn,
    JacobianFn,
    Scalar,
    Vector,
)
from slsqp_jax.utils import args_closure

STAGNATION_MESSAGE = (
    "The solver stagnated: no sufficient progress in the merit function "
    "was made for several consecutive iterations. This may indicate "
    "cycling in the QP subproblem or an infeasible/degenerate problem."
)


def _slsqp_verbose(**kwargs: tuple) -> None:
    """Default verbose callback with per-field format specifiers.

    Each kwarg value is either ``(label, value)`` or
    ``(label, value, fmt_spec)``.  The *fmt_spec* string (e.g. ``".3e"``)
    is inserted into the ``jax.debug.print`` format placeholder.
    """
    string_pieces: list[str] = []
    arg_pieces: list[Any] = []
    for entry in kwargs.values():
        if len(entry) == 3:
            name, value, fmt = entry
            string_pieces.append(f"{name}: {{:{fmt}}}")
        else:
            name, value = entry
            string_pieces.append(f"{name}: {{}}")
        arg_pieces.append(value)
    if string_pieces:
        jax.debug.print(", ".join(string_pieces), *arg_pieces)


def _no_verbose(**kwargs: tuple) -> None:
    pass


class SLSQPState(eqx.Module):
    """State for the SLSQP solver.

    This is a JAX PyTree (via eqx.Module) that holds all mutable state
    needed across SLSQP iterations.

    Attributes:
        step_count: Current iteration number.
        f_val: Current objective function value f(x_k).
        grad: Gradient of objective at current point.
        eq_val: Equality constraint values c_eq(x_k).
        ineq_val: Inequality constraint values c_ineq(x_k).
        eq_jac: Jacobian of equality constraints at x_k.
        ineq_jac: Jacobian of inequality constraints at x_k.
        lbfgs_history: L-BFGS history for matrix-free Hessian approximation.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        prev_grad_lagrangian: Previous Lagrangian gradient (for L-BFGS update).
        merit_penalty: Current penalty parameter for L1 merit function.
        bound_jac: Constant Jacobian for bound constraints (computed once in init).
        qp_iterations: Total accumulated QP active-set iterations across all steps.
        qp_converged: Whether the most recent QP solve converged.
        prev_active_set: Active inequality constraint set from the previous QP solve,
            used for warm-starting the next QP subproblem.
    """

    # Iteration tracking
    step_count: Int[Array, ""]

    # Current function values and gradients
    f_val: Scalar
    grad: Vector

    # Constraint information
    eq_val: Float[Array, " m_eq"]
    ineq_val: Float[Array, " m_ineq"]
    eq_jac: Float[Array, "m_eq n"]
    ineq_jac: Float[Array, "m_ineq n"]

    # L-BFGS history for matrix-free Hessian approximation (O(kn) storage)
    lbfgs_history: LBFGSHistory

    # Lagrange multipliers from QP solution
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]

    # Previous Lagrangian gradient for L-BFGS y = grad_L_new - grad_L_old
    prev_grad_lagrangian: Vector

    # Merit function penalty parameter
    merit_penalty: Scalar

    # Bound constraint Jacobian (constant, computed once in init)
    bound_jac: Float[Array, "m_bounds n"]

    # QP solver statistics
    qp_iterations: Int[Array, ""]
    qp_converged: Bool[Array, ""]

    # Active-set warm-starting: carry the QP active set across iterations
    # to promote multiplier stability (Wright, SIAM J. Optim., 2002, Sec. 8)
    prev_active_set: Bool[Array, " m_ineq"]

    # Consecutive QP failure tracking for escalating L-BFGS recovery
    consecutive_qp_failures: Int[Array, ""]

    # Stagnation detection
    prev_merit: Scalar
    stagnation_count: Int[Array, ""]
    last_alpha: Scalar


class QPResult(eqx.Module):
    """Result from solving the QP subproblem.

    Attributes:
        direction: The search direction d from the QP solution.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        active_set: Boolean mask of active inequality constraints at the solution.
        converged: Whether the QP solver converged successfully.
        iterations: Number of active-set iterations taken.
    """

    direction: Vector
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    active_set: Bool[Array, " m_ineq"]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]


class SLSQP(optx.AbstractMinimiser):
    """SLSQP minimizer using Sequential Quadratic Programming.

    This solver implements the SLSQP algorithm for constrained nonlinear
    optimization, designed to scale to large dimensions (n > 5000).

    At each iteration, it:

    1. Constructs a QP subproblem using a frozen L-BFGS Hessian
       approximation and linearized constraints.
    2. Solves the QP using projected conjugate gradient with an active-set
       method for inequality constraints.
    3. Performs a line search using an L1 merit function.
    4. Updates the L-BFGS history with a secant pair: either an exact
       HVP probe (if obj_hvp_fn is provided) or gradient differences.

    The Hessian is never formed as a dense matrix. The QP subproblem
    always uses a frozen L-BFGS approximation accessed through O(kn)
    Hessian-vector products, where k is the L-BFGS memory (typically 10).
    When exact HVPs are available, they are called once per main iteration
    (not inside the QP inner loop) to produce high-quality secant pairs.

    Users can optionally supply their own derivative functions:
    - obj_grad_fn: Gradient of objective (else jax.grad).
    - eq_jac_fn / ineq_jac_fn: Jacobians of constraints (else jax.jacrev).
    - obj_hvp_fn: HVP of objective (probed once per iteration for L-BFGS).
    - eq_hvp_fn / ineq_hvp_fn: HVPs of constraints (else forward-over-reverse AD).

    Box constraints (bounds) play a dual role.  Inside the QP
    subproblem they act as ordinary inequality constraints so that
    the search direction is aware of the feasible box.  After the
    line search, the accepted iterate is *projected* (clipped) onto
    the box, following the projected-SQP methodology
    (Heinkenschloss & Ridzal, *SIAM J. Optim.*, 1996).  This
    guarantees that the objective and constraint functions are never
    evaluated outside the bounds -- essential when those functions
    are undefined or ill-conditioned outside the box (e.g. a log
    likelihood with positivity constraints on its parameters).

    **Convergence criteria.**  The solver checks two conditions each
    iteration (both must be satisfied, and at least ``min_steps``
    iterations must have elapsed):

    1. *Stationarity* -- the Lagrangian gradient is small relative to
       the objective gradient:

       ``‖∇_x L‖ ≤ atol + rtol · max(‖∇f‖, 1)``

       The ``max(‖∇f‖, 1)`` floor prevents the relative term from
       vanishing when the objective gradient is already small, so the
       effective tolerance is never tighter than ``atol + rtol``.

    2. *Primal feasibility* -- constraint violations are within
       absolute tolerance:

       ``max|c_eq(x)| ≤ atol``  and  ``max(0, −c_ineq(x)) ≤ atol``

    ``rtol`` therefore only participates in the stationarity check
    (scaled by the objective gradient norm), while ``atol`` appears in
    both stationarity and feasibility checks.

    Attributes:
        rtol: Relative tolerance for the stationarity convergence check.
            Multiplied by ``max(‖∇f‖, 1)`` and added to ``atol`` to form
            the threshold on the Lagrangian gradient norm.
        atol: Absolute tolerance for convergence.  Used as the base
            threshold for both the stationarity check (on the Lagrangian
            gradient norm) and the feasibility check (on constraint
            violations).
        max_steps: Maximum number of iterations.
        eq_constraint_fn: Function computing equality constraints c_eq(x) = 0.
        ineq_constraint_fn: Function computing inequality constraints c_ineq(x) >= 0.
        n_eq_constraints: Number of equality constraints (static).
        n_ineq_constraints: Number of inequality constraints (static).
        bounds: Optional box constraints with shape (n, 2).  Iterates are
            always projected onto this box so functions are never evaluated
            outside the bounds.
        obj_grad_fn: Optional gradient of objective.
        eq_jac_fn: Optional Jacobian of equality constraints.
        ineq_jac_fn: Optional Jacobian of inequality constraints.
        obj_hvp_fn: Optional HVP of objective (probed once per iteration).
        eq_hvp_fn: Optional per-constraint HVP of equality constraints.
        ineq_hvp_fn: Optional per-constraint HVP of inequality constraints.
        min_steps: Minimum iterations before convergence is allowed (default 1).
        lbfgs_memory: Number of (s, y) pairs to store for L-BFGS (default 10).
        proximal_sigma: Stabilization parameter for the sSQP formulation
            (Hager, 1999; Wright, 2002).  When positive, equality constraints
            are absorbed into the QP objective with penalty weight
            ``1/proximal_sigma``, regularizing the dual solution and preventing
            QP infeasibility at degenerate vertices.  Recommended range:
            ``[1e-4, 1e-1]``.  Default 0.0 disables stabilization.
        use_preconditioner: When True (default), the L-BFGS inverse Hessian
            (two-loop recursion) is used as a preconditioner for the inner
            CG solver, dramatically improving convergence on ill-conditioned
            QP subproblems.
        adaptive_cg_tol: When True, the CG convergence tolerance is adapted
            based on the outer KKT residual (Eisenstat-Walker style).
            Default False preserves baseline behavior.
        cg_regularization: Minimum eigenvalue threshold ``delta^2`` for the
            CG curvature guard.  CG declares "bad curvature" when the
            effective eigenvalue ``p^T B p / ||p||^2`` falls below this
            value.  Prevents CG from terminating prematurely when the
            Hessian has small but positive eigenvalues (e.g. after an
            L-BFGS diagonal reset).  Based on SNOPT Section 4.5
            (Gill, Murray & Saunders, 2005).  Default ``1e-6``
            (delta ~ 1e-3).  Set to ``0.0`` to disable.

    Example:
        >>> import jax.numpy as jnp
        >>> from slsqp_jax import SLSQP
        >>>
        >>> def objective(x, args):
        ...     return jnp.sum(x**2), None
        >>>
        >>> def eq_constraint(x, args):
        ...     return jnp.array([x[0] + x[1] - 1.0])
        >>>
        >>> solver = SLSQP(eq_constraint_fn=eq_constraint, n_eq_constraints=1)
    """

    # Convergence tolerances
    rtol: float = 1e-6
    atol: float = 1e-6

    # Norm function for convergence checking (required by AbstractMinimiser)
    norm: Callable = eqx.field(static=True, default=optx_misc.max_norm)

    # Maximum iterations
    max_steps: int = 100

    # Minimum iterations before convergence is allowed.
    # Prevents premature termination when multipliers are not yet meaningful
    # (e.g. initial point satisfies constraints with zero-initialized multipliers).
    min_steps: int = 1

    # Constraint functions (static - not differentiated by optimistix)
    eq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)
    ineq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)

    # Number of constraints (must be static for JAX compilation)
    n_eq_constraints: int = eqx.field(static=True, default=0)
    n_ineq_constraints: int = eqx.field(static=True, default=0)

    # Box constraints (bounds) on decision variables.
    # Shape (n, 2) where bounds[i, 0] = lower, bounds[i, 1] = upper.
    # Use -jnp.inf / jnp.inf for unbounded dimensions.
    #
    # Bounds serve a dual role: (1) they enter the QP subproblem as
    # inequality constraints so the search direction respects them,
    # and (2) iterates are projected (clipped) onto the feasible box
    # after every line search step, following the projected-SQP
    # approach (Heinkenschloss & Ridzal, SIAM J. Optim., 1996).
    # This guarantees that the objective and constraint functions are
    # never evaluated outside the bounds, which is critical when those
    # functions are undefined or ill-conditioned outside the box.
    bounds: Optional[Float[Array, "n 2"]] = None

    # Masks indicating which bounds are finite (static for compilation)
    # These are computed from bounds at construction time and stored as tuples
    _lower_bound_mask: Optional[tuple[bool, ...]] = eqx.field(static=True, default=None)
    _upper_bound_mask: Optional[tuple[bool, ...]] = eqx.field(static=True, default=None)
    _n_lower_bounds: int = eqx.field(static=True, default=0)
    _n_upper_bounds: int = eqx.field(static=True, default=0)
    _lower_indices: Optional[tuple[int, ...]] = eqx.field(static=True, default=None)
    _upper_indices: Optional[tuple[int, ...]] = eqx.field(static=True, default=None)

    # Optional user-supplied derivative functions
    obj_grad_fn: Optional[GradFn] = eqx.field(static=True, default=None)
    eq_jac_fn: Optional[JacobianFn] = eqx.field(static=True, default=None)
    ineq_jac_fn: Optional[JacobianFn] = eqx.field(static=True, default=None)
    obj_hvp_fn: Optional[HVPFn] = eqx.field(static=True, default=None)
    eq_hvp_fn: Optional[ConstraintHVPFn] = eqx.field(static=True, default=None)
    ineq_hvp_fn: Optional[ConstraintHVPFn] = eqx.field(static=True, default=None)

    # Precomputed derivative callables (always set in __check_init__)
    _grad_impl: Callable = eqx.field(static=True, default=None)
    _eq_jac_impl: Callable = eqx.field(static=True, default=None)
    _ineq_jac_impl: Callable = eqx.field(static=True, default=None)
    _eq_hvp_contrib_impl: Callable = eqx.field(static=True, default=None)
    _ineq_hvp_contrib_impl: Callable = eqx.field(static=True, default=None)

    # L-BFGS parameters
    lbfgs_memory: int = eqx.field(static=True, default=10)

    # Line search parameters
    line_search_max_steps: int = eqx.field(static=True, default=20)
    armijo_c1: float = 1e-4

    # QP solver parameters
    qp_max_iter: int = eqx.field(static=True, default=100)
    qp_max_cg_iter: int = eqx.field(static=True, default=50)

    # Proximal multiplier stabilization (sSQP).
    # When positive, equality constraints are absorbed into the QP
    # objective via an augmented-Lagrangian penalty with weight
    # 1/proximal_sigma.  This regularizes the dual solution and
    # prevents QP infeasibility at degenerate vertices.  Larger
    # values mean more relaxation.  Recommended range: [1e-4, 1e-1].
    # Default 0.0 disables stabilization (standard QP).
    proximal_sigma: float = 0.0

    # Preconditioned CG: use L-BFGS inverse Hessian (two-loop recursion)
    # as preconditioner for the inner CG solver.  Dramatically improves
    # convergence on ill-conditioned QP subproblems, especially when
    # proximal_sigma is active.  Default True; set False to disable.
    use_preconditioner: bool = eqx.field(static=True, default=True)

    # Adaptive CG tolerance (Eisenstat-Walker-inspired).  Instead of
    # a fixed CG tolerance of atol, the tolerance is set to
    # min(0.1, max(atol, ||grad_L||)) so early QPs are solved loosely
    # and accuracy increases as the outer solver converges.  Default
    # False preserves baseline behavior; enable for large-scale problems
    # where early QP over-solving is wasteful.
    adaptive_cg_tol: bool = eqx.field(static=True, default=False)

    # CG regularization: minimum eigenvalue threshold for the CG curvature
    # guard.  CG stops when the effective eigenvalue p^T B p / ||p||^2
    # falls below this value.  Prevents premature termination when Hessian
    # eigenvalues are small but positive (e.g. after a diagonal reset).
    # Based on SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).
    # Default 1e-6 (delta ~ 1e-3). Set to 0.0 to disable.
    cg_regularization: float = 1e-6

    # QP failure patience: after this many consecutive QP failures, the
    # L-BFGS history is hard-reset to identity (B_0 = I) instead of the
    # SNOPT-style diagonal reset, breaking ill-conditioning cycles.
    qp_failure_patience: int = 3

    # Stagnation detection parameters
    stagnation_tol: float = 1e-12
    stagnation_patience: int = 5

    # Verbose output (resolved to Callable[..., None] in __check_init__)
    verbose: Callable = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Post-initialization to precompute bound info and derivative callables.

        Called by equinox after __init__. Precomputes:
        - Bound masks and indices for box constraints.
        - Gradient, Jacobian, and HVP contribution callables that dispatch
          once here rather than branching on every call.
        """
        # --- Verbose callable ---
        if self.verbose is True:
            object.__setattr__(self, "verbose", _slsqp_verbose)
        elif self.verbose is False:
            object.__setattr__(self, "verbose", _no_verbose)
        elif callable(self.verbose):  # pragma: no cover
            user_fn = self.verbose

            def _strip_fmt(**kwargs: tuple) -> None:
                user_fn(**{k: v[:2] for k, v in kwargs.items()})

            object.__setattr__(self, "verbose", _strip_fmt)
        else:  # pragma: no cover
            raise ValueError(
                f"Unrecognized `verbose` of type {type(self.verbose)}. "
                "Expected True, False, or a callable."
            )

        # --- Bound constraint info ---
        if self.bounds is not None:
            bounds_np = np.asarray(self.bounds)

            if np.any(np.isnan(bounds_np)):
                raise ValueError(
                    "bounds must not contain NaN values"
                )  # pragma: no cover

            if np.any(bounds_np[:, 0] > bounds_np[:, 1]):
                raise ValueError(  # pragma: no cover
                    "Lower bounds must be strictly less or equal to upper bounds."
                )
            if np.any(np.isinf(bounds_np[:, 0]) & (bounds_np[:, 0] > 0)) or np.any(
                np.isinf(bounds_np[:, 1]) & (bounds_np[:, 1] < 0)
            ):
                raise ValueError(  # pragma: no cover
                    "Lower bounds cannot be set to +inf and upper bounds cannot be "
                    "set to -inf."
                )

            lower_mask = np.isfinite(bounds_np[:, 0])
            upper_mask = np.isfinite(bounds_np[:, 1])

            object.__setattr__(self, "_lower_bound_mask", tuple(lower_mask.tolist()))
            object.__setattr__(self, "_upper_bound_mask", tuple(upper_mask.tolist()))
            object.__setattr__(self, "_n_lower_bounds", int(np.sum(lower_mask)))
            object.__setattr__(self, "_n_upper_bounds", int(np.sum(upper_mask)))
            object.__setattr__(
                self,
                "_lower_indices",
                tuple(int(i) for i in np.where(lower_mask)[0]),
            )
            object.__setattr__(
                self,
                "_upper_indices",
                tuple(int(i) for i in np.where(upper_mask)[0]),
            )

        # --- Gradient callable ---
        if self.obj_grad_fn is not None:
            user_grad_fn = self.obj_grad_fn

            def grad_impl(fn, y, args):
                return user_grad_fn(y, args)
        else:

            def grad_impl(fn, y, args):
                return jax.grad(lambda x: fn(x, args)[0])(y)

        object.__setattr__(self, "_grad_impl", grad_impl)

        # --- Equality Jacobian callable ---
        m_eq = self.n_eq_constraints
        if self.eq_constraint_fn is not None and m_eq > 0:
            if self.eq_jac_fn is not None:
                user_eq_jac = self.eq_jac_fn

                def eq_jac_impl(y, args):
                    return user_eq_jac(y, args)
            else:
                eq_fn = self.eq_constraint_fn

                def eq_jac_impl(y, args):
                    return jax.jacrev(args_closure(eq_fn, args))(y)
        else:

            def eq_jac_impl(y, args):
                return jnp.zeros((m_eq, y.shape[0]))

        object.__setattr__(self, "_eq_jac_impl", eq_jac_impl)

        # --- Inequality Jacobian callable ---
        m_ineq = self.n_ineq_constraints
        if self.ineq_constraint_fn is not None and m_ineq > 0:
            if self.ineq_jac_fn is not None:
                user_ineq_jac = self.ineq_jac_fn

                def ineq_jac_impl(y, args):
                    return user_ineq_jac(y, args)
            else:
                ineq_fn = self.ineq_constraint_fn

                def ineq_jac_impl(y, args):
                    return jax.jacrev(args_closure(ineq_fn, args))(y)
        else:

            def ineq_jac_impl(y, args):
                return jnp.zeros((m_ineq, y.shape[0]))

        object.__setattr__(self, "_ineq_jac_impl", ineq_jac_impl)

        # --- Equality HVP contribution callable ---
        if self.eq_constraint_fn is not None and m_eq > 0:
            if self.eq_hvp_fn is not None:
                eq_hvp_fn = self.eq_hvp_fn

                def eq_hvp_contrib(y, v, args, multipliers):
                    return multipliers @ eq_hvp_fn(y, v, args)
            else:
                eq_con_fn = self.eq_constraint_fn

                def eq_hvp_contrib(y, v, args, multipliers):
                    def weighted(x):
                        return jnp.dot(multipliers, eq_con_fn(x, args))

                    _, contrib = jax.jvp(jax.grad(weighted), (y,), (v,))
                    return contrib
        else:

            def eq_hvp_contrib(y, v, args, multipliers):
                return jnp.zeros_like(v)

        object.__setattr__(self, "_eq_hvp_contrib_impl", eq_hvp_contrib)

        # --- Inequality HVP contribution callable ---
        if self.ineq_constraint_fn is not None and m_ineq > 0:
            if self.ineq_hvp_fn is not None:
                ineq_hvp_fn = self.ineq_hvp_fn

                def ineq_hvp_contrib(y, v, args, multipliers):
                    return multipliers @ ineq_hvp_fn(y, v, args)
            else:
                ineq_con_fn = self.ineq_constraint_fn

                def ineq_hvp_contrib(y, v, args, multipliers):
                    def weighted(x):
                        return jnp.dot(multipliers, ineq_con_fn(x, args))

                    _, contrib = jax.jvp(jax.grad(weighted), (y,), (v,))
                    return contrib
        else:

            def ineq_hvp_contrib(y, v, args, multipliers):
                return jnp.zeros_like(v)

        object.__setattr__(self, "_ineq_hvp_contrib_impl", ineq_hvp_contrib)

    def _clip_to_bounds(self, y: Vector) -> Vector:
        """Project ``y`` onto the box defined by ``self.bounds``.

        Returns ``y`` unchanged when no finite bounds exist.
        """
        if self.bounds is None:
            return y
        return jnp.clip(y, self.bounds[:, 0], self.bounds[:, 1])

    def _compute_bound_constraint_values(
        self,
        y: Vector,
    ) -> Float[Array, " m_bounds"]:
        """Compute bound constraint values c(x) where c(x) >= 0 means feasible.

        Uses precomputed ``_lower_indices`` / ``_upper_indices`` from
        ``__check_init__`` to avoid recomputing index arrays on every call.
        """
        if self.bounds is None or (
            self._n_lower_bounds == 0 and self._n_upper_bounds == 0
        ):
            return jnp.zeros((0,))

        lower_idx = np.array(self._lower_indices)
        upper_idx = np.array(self._upper_indices)

        lower_vals = (
            y[lower_idx] - self.bounds[lower_idx, 0]
            if len(lower_idx) > 0
            else jnp.zeros((0,))
        )
        upper_vals = (
            self.bounds[upper_idx, 1] - y[upper_idx]
            if len(upper_idx) > 0
            else jnp.zeros((0,))
        )

        return jnp.concatenate([lower_vals, upper_vals])

    def _build_bound_jacobian(
        self,
        n: int,
    ) -> Float[Array, "m_bounds n"]:
        """Build the constant Jacobian matrix for bound constraints.

        Uses precomputed ``_lower_indices`` / ``_upper_indices``.
        Only called once during ``init``; the result is stored in state.
        """
        if self.bounds is None or (
            self._n_lower_bounds == 0 and self._n_upper_bounds == 0
        ):
            return jnp.zeros((0, n))

        lower_idx = np.array(self._lower_indices)
        upper_idx = np.array(self._upper_indices)
        identity = jnp.eye(n)

        J_lower = identity[lower_idx] if len(lower_idx) > 0 else jnp.zeros((0, n))
        J_upper = -identity[upper_idx] if len(upper_idx) > 0 else jnp.zeros((0, n))

        return jnp.concatenate([J_lower, J_upper], axis=0)

    def _build_lagrangian_hvp(
        self,
        state: "SLSQPState",
    ) -> Callable[[Vector], Vector]:
        """Build the frozen L-BFGS Lagrangian HVP for the QP subproblem.

        Always returns a closure v -> B_k @ v using the L-BFGS compact
        representation, regardless of whether exact HVPs are available.
        This ensures the QP subproblem uses a frozen (constant) Hessian
        approximation, which is both mathematically correct (the QP is a
        local quadratic model) and efficient (no expensive HVP calls
        inside the CG inner loop).
        """
        lbfgs_history = state.lbfgs_history

        def lbfgs_lagrangian_hvp(v: Vector) -> Vector:
            return lbfgs_hvp(lbfgs_history, v)

        return lbfgs_lagrangian_hvp

    def _build_preconditioner(
        self,
        state: "SLSQPState",
    ) -> Callable[[Vector], Vector] | None:
        """Build the preconditioner for the PCG inner solver.

        When ``proximal_sigma > 0`` and equality constraints are present,
        the QP system matrix is ``B_tilde = B + (1/sigma) A_eq^T A_eq``,
        so a plain ``B^{-1}`` preconditioner can amplify the proximal
        eigenvalues and make CG *worse*.  In that case the Woodbury
        identity is used to build ``B_tilde^{-1}`` cheaply::

            (B + (1/s) A^T A)^{-1}
              = B^{-1} - B^{-1} A^T (s I + A B^{-1} A^T)^{-1} A B^{-1}

        The inner matrix ``(s I + A B^{-1} A^T)`` is only m_eq x m_eq
        and is factored once per QP solve.

        When ``proximal_sigma == 0`` or there are no equality constraints,
        falls back to the standard ``B^{-1}`` preconditioner.

        Returns ``None`` when preconditioning is disabled.
        """
        if not self.use_preconditioner:
            return None
        lbfgs_history = state.lbfgs_history

        if self.proximal_sigma > 0 and self.n_eq_constraints > 0:
            A_eq = state.eq_jac
            sigma = self.proximal_sigma
            m_eq = A_eq.shape[0]

            Hinv_AT = jax.vmap(
                lambda a: lbfgs_inverse_hvp(lbfgs_history, a),
            )(A_eq)  # (m_eq, n): rows are B^{-1} @ a_i

            gram = Hinv_AT @ A_eq.T  # (m_eq, m_eq): A B^{-1} A^T
            inner = sigma * jnp.eye(m_eq) + gram
            inner_factor = jnp.linalg.cholesky(inner + 1e-10 * jnp.eye(m_eq))

            def preconditioner(v: Vector) -> Vector:
                Hinv_v = lbfgs_inverse_hvp(lbfgs_history, v)
                A_Hinv_v = A_eq @ Hinv_v  # (m_eq,)
                w = jax.scipy.linalg.cho_solve((inner_factor, True), A_Hinv_v)
                correction = Hinv_AT.T @ w  # (n,): B^{-1} A^T w
                return Hinv_v - correction

            return preconditioner
        else:

            def preconditioner(v: Vector) -> Vector:
                return lbfgs_inverse_hvp(lbfgs_history, v)

            return preconditioner

    def _build_exact_lagrangian_hvp(
        self,
        y: Vector,
        args: Any,
        multipliers_eq: Float[Array, " m_eq"],
        multipliers_ineq: Float[Array, " m_ineq"],
    ) -> Callable[[Vector], Vector]:
        """Build exact Lagrangian HVP using precomputed contribution callables.

        Composes H_L v = H_f v - sum lambda_eq_i H_{c_eq_i} v
                                - sum lambda_ineq_j H_{c_ineq_j} v.

        Called once per main iteration to probe the exact Hessian along the
        step direction, producing a high-quality secant pair for L-BFGS.
        The dispatch to user-supplied vs AD-computed HVPs is resolved at
        construction time in ``__check_init__``.
        """
        m_ineq = self.n_ineq_constraints
        obj_hvp_fn = cast(HVPFn, self.obj_hvp_fn)
        eq_hvp_contrib = self._eq_hvp_contrib_impl
        ineq_hvp_contrib = self._ineq_hvp_contrib_impl

        def lagrangian_hvp(v: Vector) -> Vector:
            obj_val = obj_hvp_fn(y, v, args)
            eq_val = eq_hvp_contrib(y, v, args, multipliers_eq)
            ineq_val = ineq_hvp_contrib(y, v, args, multipliers_ineq[:m_ineq])
            return obj_val - eq_val - ineq_val

        return lagrangian_hvp

    def init(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        options: dict[str, Any],
        f_struct: Any,
        aux_struct: Any,
        tags: frozenset[object],
    ) -> SLSQPState:
        """Initialize the SLSQP solver state.

        Computes initial function value, gradient, constraint values,
        and constraint Jacobians. Initializes the L-BFGS history buffer.

        Args:
            fn: Objective function with signature fn(y, args) -> (f_val, aux).
            y: Initial parameter values.
            args: Additional arguments passed to fn.
            options: Runtime options dictionary.
            f_struct: Structure of function output (for type inference).
            aux_struct: Structure of auxiliary output.
            tags: Lineax tags for the problem.

        Returns:
            Initial SLSQPState with all fields populated.
        """
        n = y.shape[0]
        y = self._clip_to_bounds(y)
        m_eq = self.n_eq_constraints
        m_ineq_general = self.n_ineq_constraints
        m_bounds = self._n_lower_bounds + self._n_upper_bounds
        m_ineq_total = m_ineq_general + m_bounds

        # Evaluate objective
        f_val, _aux = fn(y, args)

        # Compute gradient (precomputed callable)
        grad = self._grad_impl(fn, y, args)

        # Evaluate equality constraint values
        if self.eq_constraint_fn is not None and m_eq > 0:
            eq_val = self.eq_constraint_fn(y, args)
        else:
            eq_val = jnp.zeros((m_eq,))

        # Evaluate general inequality constraint values
        if self.ineq_constraint_fn is not None and m_ineq_general > 0:
            ineq_val_general = self.ineq_constraint_fn(y, args)
        else:
            ineq_val_general = jnp.zeros((m_ineq_general,))

        # Evaluate bound constraint values
        bound_vals = self._compute_bound_constraint_values(y)

        # Concatenate general inequality and bound constraints
        ineq_val = jnp.concatenate([ineq_val_general, bound_vals])

        # Compute constraint Jacobians (precomputed callables)
        eq_jac = self._eq_jac_impl(y, args)
        ineq_jac_general = self._ineq_jac_impl(y, args)

        # Build bound Jacobian (constant, computed once here and stored in state)
        bound_jac = self._build_bound_jacobian(n)
        ineq_jac = jnp.concatenate([ineq_jac_general, bound_jac], axis=0)

        # Initialize L-BFGS history (empty, gamma=1 -> B_0 = I)
        lbfgs_history = lbfgs_init(n, self.lbfgs_memory)

        # Initialize equality multipliers via least-squares:
        #   min_lambda ||grad_f - J_eq^T lambda||^2
        # This avoids the pathological case where zero multipliers cause
        # premature termination when the initial point satisfies constraints.
        if m_eq > 0:
            multipliers_eq, _, _, _ = jnp.linalg.lstsq(eq_jac.T, grad)
        else:
            multipliers_eq = jnp.zeros((m_eq,))

        # Inequality multipliers start at zero (we don't know the active set yet)
        multipliers_ineq = jnp.zeros((m_ineq_total,))

        # Compute initial Lagrangian gradient with the estimated multipliers
        prev_grad_lagrangian = compute_lagrangian_gradient(
            grad,
            eq_jac,
            ineq_jac,
            multipliers_eq,
            multipliers_ineq,
        )

        # Initial merit penalty
        merit_penalty = jnp.array(1.0)

        # Compute initial merit for stagnation tracking
        prev_merit = compute_merit(f_val, eq_val, ineq_val, merit_penalty)

        return SLSQPState(
            step_count=jnp.array(0),
            f_val=f_val,
            grad=grad,
            eq_val=eq_val,
            ineq_val=ineq_val,
            eq_jac=eq_jac,
            ineq_jac=ineq_jac,
            lbfgs_history=lbfgs_history,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            prev_grad_lagrangian=prev_grad_lagrangian,
            merit_penalty=merit_penalty,
            bound_jac=bound_jac,
            qp_iterations=jnp.array(0),
            qp_converged=jnp.array(True),
            prev_active_set=jnp.zeros((m_ineq_total,), dtype=bool),
            consecutive_qp_failures=jnp.array(0),
            prev_merit=prev_merit,
            stagnation_count=jnp.array(0),
            last_alpha=jnp.array(1.0),
        )

    def step(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
    ) -> tuple[Vector, SLSQPState, Any]:
        """Perform one SLSQP iteration.

        This method:
        1. Builds the frozen L-BFGS HVP for the QP subproblem.
        2. Solves the QP subproblem via projected CG to find direction d.
        3. Performs line search with L1 merit function to find step size.
        4. Updates x_{k+1} = x_k + alpha * d.
        5. Re-evaluates objective, gradient, constraints, and Jacobians.
        6. Updates L-BFGS history with secant pair (exact HVP probe or
           gradient differences).

        Args:
            fn: Objective function.
            y: Current parameter values.
            args: Additional arguments.
            options: Runtime options.
            state: Current solver state.
            tags: Lineax tags.

        Returns:
            Tuple of (new_y, new_state, aux).
        """
        # Ensure y is within bounds (matters for the first iteration when
        # the user-supplied y0 may violate bounds).
        y = self._clip_to_bounds(y)

        # Step 1: Build the frozen L-BFGS HVP for the QP subproblem
        hvp_fn = self._build_lagrangian_hvp(state)

        # Step 2: Solve QP subproblem for search direction
        qp_result = self._solve_qp_subproblem(state, hvp_fn)

        # Steepest descent fallback when QP fails or returns a zero direction
        # despite a non-zero gradient (defense-in-depth against stagnation).
        fallback_direction = -state.grad
        direction = jnp.where(
            qp_result.converged, qp_result.direction, fallback_direction
        )
        zero_direction = jnp.linalg.norm(direction) < 1e-30
        grad_nonzero = jnp.linalg.norm(state.grad) > self.atol
        direction = jnp.where(
            zero_direction & grad_nonzero, fallback_direction, direction
        )

        # Step 3: Update penalty parameter — only trust multipliers
        # from a converged QP to avoid permanently inflating rho.
        new_penalty = update_penalty_parameter(
            state.merit_penalty,
            qp_result.multipliers_eq,
            qp_result.multipliers_ineq,
        )
        merit_penalty = jnp.where(qp_result.converged, new_penalty, state.merit_penalty)

        # Step 4: Line search with merit function
        ls_result = backtracking_line_search(
            fn=fn,
            eq_constraint_fn=self.eq_constraint_fn,
            ineq_constraint_fn=self.ineq_constraint_fn,
            x=y,
            direction=direction,
            args=args,
            f_val=state.f_val,
            eq_val=state.eq_val,
            ineq_val=state.ineq_val,
            penalty=merit_penalty,
            grad=state.grad,
            c1=self.armijo_c1,
            max_iter=self.line_search_max_steps,
            bounds=self.bounds,
            lower_bound_mask=self._lower_bound_mask,
            upper_bound_mask=self._upper_bound_mask,
        )

        alpha = ls_result.alpha
        y_new = self._clip_to_bounds(y + alpha * direction)
        f_val_new = ls_result.f_val
        eq_val_new = ls_result.eq_val
        ineq_val_new = ls_result.ineq_val  # Includes bounds from line search

        # Get auxiliary output from function evaluation
        _, aux = fn(y_new, args)

        # Step 5: Compute gradient and Jacobians at new point
        grad_new = self._grad_impl(fn, y_new, args)
        eq_jac_new = self._eq_jac_impl(y_new, args)

        # General inequality Jacobian + constant bound Jacobian from state
        ineq_jac_general_new = self._ineq_jac_impl(y_new, args)
        ineq_jac_new = jnp.concatenate([ineq_jac_general_new, state.bound_jac], axis=0)

        # Alpha-scale multipliers: QP multipliers are for the full step;
        # blend with previous multipliers proportional to the accepted step.
        blended_mult_eq = state.multipliers_eq + alpha * (
            qp_result.multipliers_eq - state.multipliers_eq
        )
        blended_mult_ineq = state.multipliers_ineq + alpha * (
            qp_result.multipliers_ineq - state.multipliers_ineq
        )

        # Step 6: Update L-BFGS history
        s = y_new - y  # Step taken

        # Compute gradient of Lagrangian at new point using blended multipliers
        grad_lagrangian_new = compute_lagrangian_gradient(
            grad_new,
            eq_jac_new,
            ineq_jac_new,
            blended_mult_eq,
            blended_mult_ineq,
        )

        if self.obj_hvp_fn is not None:
            exact_hvp_fn = self._build_exact_lagrangian_hvp(
                y,
                args,
                blended_mult_eq,
                blended_mult_ineq,
            )
            y_for_lbfgs = exact_hvp_fn(s)
        else:
            y_for_lbfgs = grad_lagrangian_new - state.prev_grad_lagrangian

        new_lbfgs_history = lbfgs_append(state.lbfgs_history, s, y_for_lbfgs)

        # SNOPT-style diagonal reset on QP failure: preserve per-variable
        # curvature from the current approximation and clear stale pairs.
        new_lbfgs_history = jax.lax.cond(
            qp_result.converged,
            lambda h: h,
            lbfgs_reset,
            new_lbfgs_history,
        )

        # Escalating L-BFGS recovery: after qp_failure_patience consecutive
        # QP failures, the SNOPT diagonal reset is re-extracting the same
        # problematic diagonal. Hard-reset to identity to break the cycle.
        new_consecutive_qp_failures = jnp.where(
            qp_result.converged,
            jnp.array(0),
            state.consecutive_qp_failures + 1,
        )
        new_lbfgs_history = jax.lax.cond(
            new_consecutive_qp_failures >= self.qp_failure_patience,
            lbfgs_identity_reset,
            lambda h: h,
            new_lbfgs_history,
        )

        # Stagnation detection: compare merit at new point vs previous
        merit_new = compute_merit(f_val_new, eq_val_new, ineq_val_new, merit_penalty)
        rel_improvement = jnp.abs(state.prev_merit - merit_new) / jnp.maximum(
            jnp.abs(state.prev_merit), 1.0
        )
        is_stagnant = rel_improvement < self.stagnation_tol
        new_stagnation_count = jnp.where(
            is_stagnant, state.stagnation_count + 1, jnp.array(0)
        )

        new_state = SLSQPState(
            step_count=state.step_count + 1,
            f_val=f_val_new,
            grad=grad_new,
            eq_val=eq_val_new,
            ineq_val=ineq_val_new,
            eq_jac=eq_jac_new,
            ineq_jac=ineq_jac_new,
            lbfgs_history=new_lbfgs_history,
            multipliers_eq=qp_result.multipliers_eq,
            multipliers_ineq=qp_result.multipliers_ineq,
            prev_grad_lagrangian=grad_lagrangian_new,
            merit_penalty=merit_penalty,
            bound_jac=state.bound_jac,
            qp_iterations=state.qp_iterations + qp_result.iterations,
            qp_converged=qp_result.converged,
            prev_active_set=qp_result.active_set,
            consecutive_qp_failures=new_consecutive_qp_failures,
            prev_merit=merit_new,
            stagnation_count=new_stagnation_count,
            last_alpha=alpha,
        )

        # Verbose output
        m_eq = self.n_eq_constraints
        m_ineq_total = (
            self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
        )
        eq_viol = jnp.max(jnp.abs(eq_val_new)) if m_eq > 0 else jnp.array(0.0)
        ineq_viol = (
            jnp.max(jnp.maximum(0.0, -ineq_val_new))
            if m_ineq_total > 0
            else jnp.array(0.0)
        )
        c_viol = jnp.maximum(eq_viol, ineq_viol)
        kkt = jnp.linalg.norm(grad_lagrangian_new)
        dir_norm = jnp.linalg.norm(direction)
        grad_norm = jnp.linalg.norm(grad_new)
        n_active = jnp.sum(qp_result.active_set.astype(jnp.int32))
        diag_cond = jnp.max(new_lbfgs_history.diagonal) / jnp.maximum(
            jnp.min(new_lbfgs_history.diagonal), 1e-30
        )
        self.verbose(
            num_steps=("Step", new_state.step_count),
            objective=("f", f_val_new, ".6e"),
            constraint_violation=("|c|", c_viol, ".3e"),
            kkt_residual=("|∇L|", kkt, ".3e"),
            grad_norm=("|∇f|", grad_norm, ".3e"),
            step_size=("α", alpha, ".3e"),
            direction_norm=("|d|", dir_norm, ".3e"),
            merit=("merit", merit_new, ".6e"),
            merit_improvement=("Δmerit", rel_improvement, ".3e"),
            stagnation_count=("stag", new_stagnation_count),
            penalty=("ρ", merit_penalty, ".3e"),
            lbfgs_gamma=("γ", new_lbfgs_history.gamma, ".3e"),
            lbfgs_diag_cond=("κ_B", diag_cond, ".1e"),
            qp_iters=("QP it", qp_result.iterations),
            qp_converged=("QP ok", qp_result.converged),
            n_active=("#act", n_active),
        )

        return y_new, new_state, aux

    def terminate(
        self,
        fn: Callable,
        y: Vector,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], Any]:
        """Check if the solver should terminate.

        Checks the KKT conditions for convergence:
        1. Gradient of Lagrangian is small (stationarity).
        2. Constraints are satisfied (primal feasibility).

        Args:
            fn: Objective function.
            y: Current parameter values.
            args: Additional arguments.
            options: Runtime options.
            state: Current solver state.
            tags: Lineax tags.

        Returns:
            Tuple of (done, result) where done is a bool indicating
            termination and result is the termination status code.
        """
        m_eq = self.n_eq_constraints
        m_ineq_total = (
            self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
        )

        # Compute gradient of Lagrangian
        grad_lagrangian = compute_lagrangian_gradient(
            state.grad,
            state.eq_jac,
            state.ineq_jac,
            state.multipliers_eq,
            state.multipliers_ineq,
        )

        # Check stationarity: ||nabla L|| <= atol + rtol * max(||nabla f||, 1)
        grad_norm = jnp.linalg.norm(grad_lagrangian)
        # Floor of 1.0 prevents the relative term from vanishing when
        # ||∇f|| is already small, keeping the tolerance ≥ atol + rtol.
        grad_ref = jnp.maximum(jnp.linalg.norm(state.grad), 1.0)
        stationarity = grad_norm <= self.atol + self.rtol * grad_ref

        # Check primal feasibility
        eq_feasible = jnp.array(True)
        if m_eq > 0:
            eq_violation = jnp.max(jnp.abs(state.eq_val))
            eq_feasible = eq_violation <= self.atol

        ineq_feasible = jnp.array(True)
        if m_ineq_total > 0:
            ineq_violation = jnp.max(jnp.maximum(0.0, -state.ineq_val))
            ineq_feasible = ineq_violation <= self.atol

        primal_feasible = eq_feasible & ineq_feasible

        # Check max iterations
        max_iters_reached = state.step_count >= self.max_steps

        # Check stagnation
        stagnation_detected = state.stagnation_count >= self.stagnation_patience

        # Converged if stationary, feasible, and past minimum iterations.
        # The min_steps guard prevents false convergence when multipliers
        # are zero-initialized and haven't been updated by a QP solve yet.
        has_min_steps = state.step_count >= self.min_steps
        converged = stationarity & primal_feasible & has_min_steps

        # Determine result code
        done = converged | max_iters_reached | stagnation_detected

        result = jax.lax.cond(
            converged,
            lambda: optx.RESULTS.successful,
            lambda: jax.lax.cond(
                stagnation_detected,
                lambda: optx.RESULTS.nonlinear_divergence,
                lambda: jax.lax.cond(
                    max_iters_reached,
                    lambda: optx.RESULTS.max_steps_reached,
                    lambda: optx.RESULTS.successful,  # Still running
                ),
            ),
        )

        return done, result

    def postprocess(
        self,
        fn: Callable,
        y: Vector,
        aux: Any,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
        result: Any,
    ) -> tuple[Vector, Any, dict[str, Any]]:
        """Post-process the optimization result.

        Args:
            fn: Objective function.
            y: Final parameter values.
            aux: Auxiliary output from last function evaluation.
            args: Additional arguments.
            options: Runtime options.
            state: Final solver state.
            tags: Lineax tags.
            result: Termination result code.

        Returns:
            Tuple of (y, aux, stats) where stats is a dictionary
            containing solver statistics.
        """
        y = self._clip_to_bounds(y)
        stats = {
            "num_steps": state.step_count,
            "final_objective": state.f_val,
            "final_grad_norm": jnp.linalg.norm(state.grad),
            "merit_penalty": state.merit_penalty,
            "total_qp_iterations": state.qp_iterations,
            "last_qp_converged": state.qp_converged,
            "qp_tolerance": 1e-8,
            "multipliers_eq": state.multipliers_eq,
            "multipliers_ineq": state.multipliers_ineq,
            "stagnation_count": state.stagnation_count,
            "last_step_size": state.last_alpha,
        }

        return y, aux, stats

    def _solve_qp_subproblem(
        self,
        state: SLSQPState,
        hvp_fn: Callable[[Vector], Vector],
    ) -> QPResult:
        """Solve the QP subproblem for the search direction.

        Solves:
            minimize    (1/2) d^T B d + g^T d
            subject to  A_eq d = -c_eq
                        A_ineq d >= -c_ineq

        using the HVP function v -> B @ v for the Hessian.

        The previous iteration's QP active set is passed as a warm-start
        hint, and the outer KKT residual norm is used for adaptive EXPAND
        tolerance scaling.  When ``proximal_sigma > 0``, the equality
        constraints are absorbed into the objective via sSQP.

        When ``use_preconditioner`` is True, the L-BFGS inverse Hessian
        is passed to the QP solver as a CG preconditioner.

        When ``adaptive_cg_tol`` is True, the CG tolerance is adapted
        based on the outer KKT residual (Eisenstat-Walker style).

        Args:
            state: Current solver state.
            hvp_fn: Hessian-vector product function for the Lagrangian.

        Returns:
            QPResult containing the search direction, multipliers, and
            active set.
        """
        g = state.grad

        A_eq = state.eq_jac
        b_eq = -state.eq_val

        A_ineq = state.ineq_jac
        b_ineq = -state.ineq_val

        kkt_residual = jnp.linalg.norm(state.prev_grad_lagrangian)

        initial_active_set = state.prev_active_set

        precond_fn = self._build_preconditioner(state)

        # Eisenstat-Walker adaptive CG tolerance: solve loosely far from
        # optimum (fast), tightly near the solution (accurate).
        # Kept separate from `tol` so feasibility checking stays tight.
        if self.adaptive_cg_tol:
            adaptive_tol = jnp.minimum(0.1, jnp.maximum(self.atol, kkt_residual))
        else:
            adaptive_tol = None

        qp_result = solve_qp(
            hvp_fn=hvp_fn,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            max_iter=self.qp_max_iter,
            max_cg_iter=self.qp_max_cg_iter,
            initial_active_set=initial_active_set,
            kkt_residual=kkt_residual,
            proximal_sigma=self.proximal_sigma,
            prev_multipliers_eq=state.multipliers_eq,
            precond_fn=precond_fn,
            cg_tol=adaptive_tol,
            cg_regularization=self.cg_regularization,
        )

        return QPResult(
            direction=qp_result.d,
            multipliers_eq=qp_result.multipliers_eq,
            multipliers_ineq=qp_result.multipliers_ineq,
            active_set=qp_result.active_set,
            converged=qp_result.converged,
            iterations=qp_result.iterations,
        )
