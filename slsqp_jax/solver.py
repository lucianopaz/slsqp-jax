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
    estimate_hessian_diagonal,
    lbfgs_append,
    lbfgs_hvp,
    lbfgs_identity_reset,
    lbfgs_init,
    lbfgs_inverse_hvp,
    lbfgs_soft_reset,
)
from slsqp_jax.lpeca import compute_lpeca_active_set
from slsqp_jax.merit import (
    backtracking_line_search,
    compute_merit,
    update_penalty_parameter,
)
from slsqp_jax.qp_solver import _solve_projected_cg, solve_qp
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
    "The solver stagnated: the L1 merit function did not improve over "
    "the patience window (max_steps / 10 consecutive iterations). This "
    "may indicate cycling in the QP subproblem or an infeasible/degenerate "
    "problem."
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

    # Consecutive line search failure tracking for escalating L-BFGS recovery
    consecutive_ls_failures: Int[Array, ""]

    # Merit-based stagnation detection
    best_merit: Scalar
    steps_without_improvement: Int[Array, ""]
    stagnation: Bool[Array, ""]
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

    1. *Stationarity* -- the Lagrangian gradient norm is small
       relative to the Lagrangian value:

       ``‖∇_x L‖ ≤ rtol · max(|L|, 1)``

       where ``L = f − λ_eqᵀ c_eq − μ_ineqᵀ c_ineq`` is the
       Lagrangian.  The ``max(|L|, 1)`` safeguard prevents the
       criterion from becoming vacuous when ``L ≈ 0`` and degrades
       gracefully to an absolute tolerance when ``|L| < 1``.  The
       relative form is motivated by floating-point arithmetic: when
       the gradient is negligible relative to ``|L|``, a step of
       that magnitude cannot change ``L`` in finite precision.

    2. *Primal feasibility* -- constraint violations are within
       absolute tolerance:

       ``max|c_eq(x)| ≤ atol``  and  ``max(0, −c_ineq(x)) ≤ atol``

    Attributes:
        rtol: Relative tolerance for the stationarity convergence
            check.  The Lagrangian gradient norm is compared against
            ``rtol · max(|L|, 1)``.  Default ``1e-6``.
        atol: Absolute tolerance for primal feasibility.  Used as
            the threshold for constraint violation checks and for
            internal heuristics (steepest-descent fallback, adaptive
            CG tolerance floor, proximal mu floor).  Default ``1e-6``.
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
        proximal_tau: Exponent for the adaptive proximal parameter mu in the
            sSQP formulation (Wright, 2002, eq 6.6).  The proximal parameter
            is computed as ``mu = clip(kkt_residual^tau, mu_min, mu_max)``
            at each iteration.  Capped at ``mu_max`` (default 0.1) so the
            proximal weight ``1/mu >= 1/mu_max``, ensuring adequate equality
            even far from the solution.  Must be in the half-open interval
            ``[0, 1)`` for the superlinear convergence guarantee.
            Default 0.5.  Set to 0 to disable sSQP proximal stabilization
            entirely: equality constraints are then enforced via direct
            null-space projection instead of the augmented Lagrangian
            penalty, avoiding the ill-conditioning introduced by the
            proximal term.
        proximal_mu_min: Floor on the adaptive proximal parameter mu.
            Prevents ``1/mu`` from exploding and creating a convergence floor
            on the Lagrangian gradient norm.  Default ``None`` resolves to
            ``atol`` in ``__check_init__``, so the proximal perturbation near
            convergence is ``O(atol)`` — consistent with the convergence
            tolerance.
        proximal_mu_max: Ceiling on the adaptive proximal parameter mu.
            Prevents ``1/mu`` from becoming too small far from the solution,
            which would weaken equality-constraint enforcement and sabotage
            the L1 merit function descent.  Wright's local convergence
            theory assumes ``kkt_residual < 1``; this cap handles the
            regime where ``kkt_residual >> 1``.  Default 0.1.
        use_preconditioner: When True (default), a preconditioner is used
            for the inner CG solver, dramatically improving convergence on
            ill-conditioned QP subproblems.
        preconditioner_type: Which preconditioner to use.  ``"lbfgs"``
            (default) uses the L-BFGS inverse Hessian (two-loop recursion).
            ``"diagonal"`` uses a stochastic estimate of the true Hessian
            diagonal (Bekas et al., 2007), requiring an exact HVP (set
            ``use_exact_hvp_in_qp=True`` or provide ``obj_hvp_fn``).
        diagonal_n_probes: Number of Rademacher probes for the stochastic
            diagonal estimator.  Each probe costs one HVP evaluation.  Only
            used when ``preconditioner_type="diagonal"``.  Default 20.
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
        active_set_method: Controls how the QP active set is initialized
            and whether the EXPAND anti-cycling tolerance grows.  One of
            ``"expand"`` (default EXPAND procedure), ``"lpeca_init"``
            (LPEC-A warm-start with EXPAND fallback), or ``"lpeca"``
            (LPEC-A with fixed tolerance, no EXPAND).  See Oberlin &
            Wright (2005).
        lpeca_sigma: Threshold exponent for LPEC-A (``sigma_bar`` in
            the paper).  Must be in (0, 1).  Default 0.9.
        lpeca_beta: Threshold scaling factor for LPEC-A.  Default
            ``None`` uses ``1 / (m_ineq + n + m_eq)``.
        lpeca_use_lp: When True, solve the LPEC-A LP to obtain tighter
            multiplier estimates before the threshold test.  Requires
            the ``mpax`` package.  Default False.

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

    # Relative tolerance for the stationarity convergence check.
    rtol: float = 1e-6

    # Absolute tolerance for primal feasibility and internal heuristics.
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
    _obj_hvp_impl: Optional[Callable] = eqx.field(static=True, default=None)

    # L-BFGS parameters
    lbfgs_memory: int = eqx.field(static=True, default=10)

    # Powell damping threshold for L-BFGS curvature pairs.  Damping
    # ensures s^T y_damped >= threshold * s^T B s, which is needed for
    # positive definiteness in constrained optimization.  For
    # well-conditioned PD objectives where s^T y > 0 naturally holds,
    # setting this to 0.0 avoids corrupting curvature information.
    damping_threshold: float = 0.2

    # Line search parameters
    line_search_max_steps: int = eqx.field(static=True, default=20)
    armijo_c1: float = 1e-4

    # QP solver parameters
    qp_max_iter: int = eqx.field(static=True, default=100)
    qp_max_cg_iter: int = eqx.field(static=True, default=50)

    # Newton-CG mode: use exact Lagrangian HVP (via AD) in the QP inner
    # loop instead of the L-BFGS approximation.  This costs one
    # forward-over-reverse HVP evaluation per CG step but can
    # dramatically improve convergence on ill-conditioned problems.
    # L-BFGS is still updated for preconditioning and as a fallback.
    use_exact_hvp_in_qp: bool = eqx.field(static=True, default=False)

    # Adaptive proximal multiplier stabilization (sSQP).
    # Active for equality-constrained problems when tau > 0.  The proximal
    # parameter mu = clip(kkt_residual^tau, mu_min, mu_max) is computed
    # per iteration (Wright, 2002, eq 6.6).  Must be in [0, 1).
    # Set to 0 to disable sSQP and use direct null-space projection.
    proximal_tau: float = 0.5

    # Floor on adaptive proximal mu.  None defaults to atol in
    # __check_init__ so that the proximal perturbation near convergence
    # is O(atol), consistent with the convergence tolerance.
    proximal_mu_min: Optional[float] = None

    # Ceiling on adaptive proximal mu.  Prevents weak enforcement of
    # equality constraints when the KKT residual is large (far from
    # solution).  With mu_max = 0.1, the proximal weight 1/mu >= 10.
    proximal_mu_max: float = eqx.field(static=True, default=0.1)

    # Resolved proximal_mu_min (always set in __check_init__)
    _proximal_mu_min: float = eqx.field(static=True, default=1e-6)

    # Resolved proximal_mu_max (always set in __check_init__)
    _proximal_mu_max: float = eqx.field(static=True, default=0.1)

    # Preconditioned CG: use L-BFGS inverse Hessian (two-loop recursion)
    # as preconditioner for the inner CG solver.  Dramatically improves
    # convergence on ill-conditioned QP subproblems, especially when
    # the proximal term is active.  Default True; set False to disable.
    use_preconditioner: bool = eqx.field(static=True, default=True)

    # Preconditioner type.  Controls which preconditioner is used for
    # the inner CG solver when ``use_preconditioner=True``.
    #   "lbfgs"    — L-BFGS inverse Hessian (two-loop recursion).
    #                Default; works without an exact HVP.
    #   "diagonal" — stochastic Hessian diagonal estimate (Bekas et al.,
    #                2007).  Requires an exact HVP (``use_exact_hvp_in_qp
    #                =True`` or ``obj_hvp_fn`` provided).  Probes the
    #                true Hessian at the current iterate, immune to the
    #                L-BFGS reset death spiral on ill-conditioned problems.
    #                Uses ``diagonal_n_probes`` Rademacher probes per step.
    preconditioner_type: str = eqx.field(static=True, default="lbfgs")

    # Number of Rademacher probes for the stochastic diagonal estimator.
    # Each probe costs one forward-over-reverse AD pass (same as one
    # gradient evaluation).  More probes reduce variance.  Only used
    # when ``preconditioner_type="diagonal"``.  Default 20.
    diagonal_n_probes: int = eqx.field(static=True, default=20)

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

    # Line search failure patience: after this many consecutive line search
    # failures, the L-BFGS history is hard-reset to identity.  On each
    # individual failure a SNOPT diagonal reset is applied first; the
    # identity escalation fires only after the patience threshold.
    ls_failure_patience: int = 3

    # Stagnation detection: merit-based patience counter.
    # If the L1 merit function does not improve by at least
    # stagnation_tol * max(|best_merit|, 1) for W = max_steps // 10
    # consecutive steps, the solver declares stagnation.
    stagnation_tol: float = 1e-12

    # Stagnation patience (computed in __check_init__)
    _stagnation_window: int = eqx.field(static=True, default=10)

    # LPEC-A active set identification (Oberlin & Wright, 2005).
    # Controls how the QP active set is initialized and whether the
    # EXPAND anti-cycling tolerance grows during the active-set loop.
    #   "expand"     — standard EXPAND procedure (default)
    #   "lpeca_init" — LPEC-A predicted set for warm-start, EXPAND fallback
    #   "lpeca"      — LPEC-A predicted set, fixed tolerance (no EXPAND)
    active_set_method: str = eqx.field(static=True, default="expand")

    # Threshold exponent for LPEC-A (sigma_bar in the paper).
    # Must be in (0, 1).  Default 0.9 per paper recommendation.
    lpeca_sigma: float = eqx.field(static=True, default=0.9)

    # Threshold scaling factor for LPEC-A.  None uses the paper's
    # recommendation 1 / (m_ineq + n + m_eq).
    lpeca_beta: Optional[float] = eqx.field(static=True, default=None)

    # When True, solve the LPEC-A LP (via mpax.r2HPDHG) for tighter
    # multiplier estimates before the threshold test.  Requires mpax.
    lpeca_use_lp: bool = eqx.field(static=True, default=False)

    # Verbose output (resolved to Callable[..., None] in __check_init__)
    verbose: Callable = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Post-initialization to precompute bound info and derivative callables.

        Called by equinox after __init__. Precomputes:
        - Stagnation window size and proximal mu floor.
        - Bound masks and indices for box constraints.
        - Gradient, Jacobian, and HVP contribution callables that dispatch
          once here rather than branching on every call.
        """
        # --- Stagnation window ---
        object.__setattr__(self, "_stagnation_window", max(1, self.max_steps // 10))

        # --- Proximal mu floor ---
        if self.proximal_mu_min is not None:
            object.__setattr__(self, "_proximal_mu_min", self.proximal_mu_min)
        else:
            object.__setattr__(self, "_proximal_mu_min", self.atol)

        # --- Proximal mu ceiling ---
        object.__setattr__(self, "_proximal_mu_max", self.proximal_mu_max)

        if not (0 <= self.proximal_tau < 1):
            raise ValueError(
                f"proximal_tau must be in the half-open interval [0, 1), "
                f"got {self.proximal_tau}"
            )

        # --- Preconditioner type validation ---
        if self.preconditioner_type not in ("lbfgs", "diagonal"):
            raise ValueError(
                f"preconditioner_type must be 'lbfgs' or 'diagonal', "
                f"got {self.preconditioner_type!r}"
            )
        if self.preconditioner_type == "diagonal" and not (
            self.obj_hvp_fn is not None or self.use_exact_hvp_in_qp
        ):
            raise ValueError(
                "preconditioner_type='diagonal' requires an exact HVP: "
                "set use_exact_hvp_in_qp=True or provide obj_hvp_fn"
            )

        # --- LPEC-A validation ---
        if self.active_set_method not in ("expand", "lpeca_init", "lpeca"):
            raise ValueError(
                f"active_set_method must be 'expand', 'lpeca_init', or 'lpeca', "
                f"got {self.active_set_method!r}"
            )
        if not (0 < self.lpeca_sigma < 1):
            raise ValueError(
                f"lpeca_sigma must be in the open interval (0, 1), "
                f"got {self.lpeca_sigma}"
            )

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

        # --- Objective HVP callable (for Newton-CG and secant probes) ---
        if self.obj_hvp_fn is not None:
            user_obj_hvp = self.obj_hvp_fn

            def obj_hvp_impl(fn, y, v, args):
                return user_obj_hvp(y, v, args)

        elif self.use_exact_hvp_in_qp:

            def obj_hvp_impl(fn, y, v, args):
                _, hvp_val = jax.jvp(jax.grad(lambda x: fn(x, args)[0]), (y,), (v,))
                return hvp_val

        else:
            obj_hvp_impl = None

        object.__setattr__(self, "_obj_hvp_impl", obj_hvp_impl)

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
        fn: Callable,
        y: Vector,
        args: Any,
        state: "SLSQPState",
    ) -> Callable[[Vector], Vector]:
        """Build the Lagrangian HVP for the QP subproblem.

        In default mode, returns a closure v -> B_k @ v using the L-BFGS
        compact representation (frozen, constant Hessian approximation).

        In Newton-CG mode (``use_exact_hvp_in_qp=True``), returns the
        exact Lagrangian HVP at the current iterate, computed via AD.
        This costs one forward-over-reverse pass per CG step but can
        dramatically improve convergence on ill-conditioned problems.
        """
        if self.use_exact_hvp_in_qp and self._obj_hvp_impl is not None:
            return self._build_exact_lagrangian_hvp(
                fn,
                y,
                args,
                state.multipliers_eq,
                state.multipliers_ineq,
            )

        lbfgs_history = state.lbfgs_history

        def lbfgs_lagrangian_hvp(v: Vector) -> Vector:
            return lbfgs_hvp(lbfgs_history, v)

        return lbfgs_lagrangian_hvp

    def _build_preconditioner(
        self,
        state: "SLSQPState",
        proximal_mu: Scalar | float = 0.0,
        lagrangian_hvp_fn: Callable[[Vector], Vector] | None = None,
    ) -> Callable[[Vector], Vector] | None:
        """Build the preconditioner for the PCG inner solver.

        Two preconditioner types are supported:

        **L-BFGS** (``preconditioner_type="lbfgs"``, default):
        Uses the L-BFGS inverse Hessian (two-loop recursion).

        **Diagonal** (``preconditioner_type="diagonal"``):
        Estimates diag(H_L) via stochastic Rademacher probing of the
        exact Lagrangian HVP, then uses M^{-1} = diag(1/d_hat).  This
        is independent of L-BFGS history quality and immune to the
        reset death spiral on ill-conditioned problems.  Requires
        ``lagrangian_hvp_fn`` (the exact Lagrangian HVP at the current
        iterate).

        For both types, when equality constraints are present *and*
        sSQP proximal stabilization is active (``proximal_tau > 0``),
        the QP system matrix is ``B_tilde = B + (1/mu) A_eq^T A_eq``.
        The Woodbury identity is used to build ``B_tilde^{-1}``::

            (B + (1/mu) A^T A)^{-1}
              = B^{-1} - B^{-1} A^T (mu I + A B^{-1} A^T)^{-1} A B^{-1}

        The inner matrix ``(mu I + A B^{-1} A^T)`` is only m_eq x m_eq
        and is factored once per QP solve.

        Args:
            state: Current solver state.
            proximal_mu: Adaptive proximal parameter (mu).
            lagrangian_hvp_fn: Exact Lagrangian HVP at the current
                iterate.  Required when ``preconditioner_type="diagonal"``.

        Returns ``None`` when preconditioning is disabled.
        """
        if not self.use_preconditioner:
            return None

        if self.preconditioner_type == "diagonal":
            return self._build_diagonal_preconditioner(
                state, proximal_mu, lagrangian_hvp_fn
            )

        return self._build_lbfgs_preconditioner(state, proximal_mu)

    def _build_lbfgs_preconditioner(
        self,
        state: "SLSQPState",
        proximal_mu: Scalar | float = 0.0,
    ) -> Callable[[Vector], Vector]:
        """L-BFGS inverse Hessian preconditioner (two-loop recursion)."""
        lbfgs_history = state.lbfgs_history

        if self.n_eq_constraints > 0 and self.proximal_tau > 0:
            A_eq = state.eq_jac
            mu = proximal_mu
            m_eq = A_eq.shape[0]

            Hinv_AT = jax.vmap(
                lambda a: lbfgs_inverse_hvp(lbfgs_history, a),
            )(A_eq)  # (m_eq, n): rows are B^{-1} @ a_i

            gram = Hinv_AT @ A_eq.T  # (m_eq, m_eq): A B^{-1} A^T
            inner = mu * jnp.eye(m_eq) + gram
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

    def _build_diagonal_preconditioner(
        self,
        state: "SLSQPState",
        proximal_mu: Scalar | float = 0.0,
        lagrangian_hvp_fn: Callable[[Vector], Vector] | None = None,
    ) -> Callable[[Vector], Vector]:
        """Stochastic diagonal Hessian preconditioner (Bekas et al., 2007).

        Estimates diag(H_L) by probing the exact Lagrangian HVP with
        Rademacher random vectors, then uses M^{-1} = diag(1/d_hat).
        The estimate is recomputed each SLSQP step using a deterministic
        PRNG key derived from the step count.
        """
        assert lagrangian_hvp_fn is not None, (
            "diagonal preconditioner requires an exact Lagrangian HVP"
        )
        n = state.grad.shape[0]
        key = jax.random.fold_in(jax.random.PRNGKey(42), state.step_count)
        diag_est = estimate_hessian_diagonal(
            lagrangian_hvp_fn, n, key, n_probes=self.diagonal_n_probes
        )
        # Safeguard: clamp small/negative entries to a positive floor so
        # the preconditioner is always positive definite.  Use the median
        # absolute value as the scale reference.
        abs_diag = jnp.abs(diag_est)
        floor = jnp.maximum(1e-8, 1e-6 * jnp.median(abs_diag))
        diag_safe = jnp.maximum(abs_diag, floor)
        inv_diag = 1.0 / diag_safe

        if self.n_eq_constraints > 0 and self.proximal_tau > 0:
            A_eq = state.eq_jac
            mu = proximal_mu
            m_eq = A_eq.shape[0]

            # Woodbury: (D + (1/mu) A^T A)^{-1}
            #   = D^{-1} - D^{-1} A^T (mu I + A D^{-1} A^T)^{-1} A D^{-1}
            Dinv_AT = (A_eq * inv_diag[None, :]).T  # (n, m_eq)
            gram = A_eq @ Dinv_AT  # (m_eq, m_eq): A D^{-1} A^T
            inner = mu * jnp.eye(m_eq) + gram
            inner_factor = jnp.linalg.cholesky(inner + 1e-10 * jnp.eye(m_eq))

            def preconditioner(v: Vector) -> Vector:
                Dinv_v = inv_diag * v
                A_Dinv_v = A_eq @ Dinv_v  # (m_eq,)
                w = jax.scipy.linalg.cho_solve((inner_factor, True), A_Dinv_v)
                correction = Dinv_AT @ w  # (n,)
                return Dinv_v - correction

            return preconditioner
        else:

            def preconditioner(v: Vector) -> Vector:
                return inv_diag * v

            return preconditioner

    def _build_exact_lagrangian_hvp(
        self,
        fn: Callable,
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
        Also used in Newton-CG mode for the QP inner loop.  The dispatch
        to user-supplied vs AD-computed HVPs is resolved at construction
        time in ``__check_init__``.
        """
        m_ineq = self.n_ineq_constraints
        obj_hvp_impl = cast(Callable, self._obj_hvp_impl)
        eq_hvp_contrib = self._eq_hvp_contrib_impl
        ineq_hvp_contrib = self._ineq_hvp_contrib_impl

        def lagrangian_hvp(v: Vector) -> Vector:
            obj_val = obj_hvp_impl(fn, y, v, args)
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

        # Initial merit for stagnation tracking
        initial_merit = compute_merit(f_val, eq_val, ineq_val, merit_penalty)

        return SLSQPState(  # ty: ignore[invalid-return-type]  # equinox @dataclass_transform
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
            consecutive_ls_failures=jnp.array(0),
            best_merit=initial_merit,
            steps_without_improvement=jnp.array(0),
            stagnation=jnp.array(False),
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
        1. Builds the Lagrangian HVP for the QP subproblem (L-BFGS or
           exact AD in Newton-CG mode).
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

        # Step 1: Build the Lagrangian HVP for the QP subproblem
        hvp_fn = self._build_lagrangian_hvp(fn, y, args, state)

        # Step 2: Solve QP subproblem for search direction
        qp_result = self._solve_qp_subproblem(state, hvp_fn, y)

        # Projected steepest descent fallback: project -grad_f onto
        # null(J_eq) so that the fallback direction does not violate
        # equality constraints.  Without projection, sum(d) can be O(n)
        # for a simplex constraint, making the L1 merit DD massively
        # positive and preventing the line search from finding a step.
        neg_grad = -state.grad
        if self.n_eq_constraints > 0:
            J = state.eq_jac  # (m_eq, n)
            JJT = J @ J.T
            m_eq = self.n_eq_constraints
            JJT_reg = JJT + 1e-10 * jnp.eye(m_eq)
            Jv = J @ neg_grad
            w = jnp.linalg.solve(JJT_reg, Jv)
            fallback_direction = neg_grad - J.T @ w
        else:
            fallback_direction = neg_grad

        direction = jnp.where(
            qp_result.converged, qp_result.direction, fallback_direction
        )
        zero_direction = jnp.linalg.norm(direction) < 1e-30
        grad_nonzero = jnp.linalg.norm(state.grad) > self.atol
        # Only fall back to steepest descent when the QP failed to converge.
        # When the QP converges but returns a zero direction (e.g. because
        # bound clipping zeroed it out), the zero direction is correct —
        # the iterate is at a bound-constrained optimum and the convergence
        # check will detect it via the bound multipliers.
        direction = jnp.where(
            zero_direction & grad_nonzero & ~qp_result.converged,
            fallback_direction,
            direction,
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
            eq_jac=state.eq_jac if self.n_eq_constraints > 0 else None,
            ineq_jac=state.ineq_jac[: self.n_ineq_constraints]
            if self.n_ineq_constraints > 0
            else None,
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

        if self._obj_hvp_impl is not None:
            exact_hvp_fn = self._build_exact_lagrangian_hvp(
                fn,
                y,
                args,
                blended_mult_eq,
                blended_mult_ineq,
            )
            y_for_lbfgs = exact_hvp_fn(s)
        else:
            # Recompute grad_L at old point x_k with NEW blended multipliers.
            # The secant condition (Nocedal & Wright §18.3) requires both
            # Lagrangian gradients to use the same multipliers:
            #   y_k = grad_L(x_{k+1}, lambda_{k+1}) - grad_L(x_k, lambda_{k+1})
            grad_lagrangian_old = compute_lagrangian_gradient(
                state.grad,
                state.eq_jac,
                state.ineq_jac,
                blended_mult_eq,
                blended_mult_ineq,
            )
            y_for_lbfgs = grad_lagrangian_new - grad_lagrangian_old

        new_lbfgs_history = lbfgs_append(
            state.lbfgs_history,
            s,
            y_for_lbfgs,
            damping_threshold=self.damping_threshold,
        )

        # VARCHEN-style conditioning control: soft reset (keep most recent
        # pair) when the inverse Hessian condition number exceeds kappa_max.
        # This is less aggressive than the old diagonal/identity resets.
        kappa_est = new_lbfgs_history.eig_upper / jnp.maximum(
            new_lbfgs_history.eig_lower, 1e-30
        )
        new_lbfgs_history = jax.lax.cond(
            (new_lbfgs_history.count > 1) & (kappa_est > 1e6),
            lbfgs_soft_reset,
            lambda h: h,
            new_lbfgs_history,
        )

        # On QP failure: soft reset (keep most recent pair) to preserve
        # the newest curvature information while discarding stale pairs.
        new_lbfgs_history = jax.lax.cond(
            qp_result.converged,
            lambda h: h,
            lbfgs_soft_reset,
            new_lbfgs_history,
        )

        # Escalating recovery: after qp_failure_patience consecutive
        # QP failures, soft resets are re-using the same problematic
        # pair. Hard-reset to identity to break the cycle.
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

        # Line search failure recovery: soft reset on each failure,
        # identity escalation after patience threshold.
        ls_failed = ~ls_result.success
        new_consecutive_ls_failures = jnp.where(
            ls_failed,
            state.consecutive_ls_failures + 1,
            jnp.array(0),
        )
        new_lbfgs_history = jax.lax.cond(
            ls_failed & (new_consecutive_ls_failures < self.ls_failure_patience),
            lbfgs_soft_reset,
            lambda h: h,
            new_lbfgs_history,
        )
        new_lbfgs_history = jax.lax.cond(
            new_consecutive_ls_failures >= self.ls_failure_patience,
            lbfgs_identity_reset,
            lambda h: h,
            new_lbfgs_history,
        )

        # Merit-based stagnation detection: track consecutive steps
        # without sufficient improvement in the L1 merit function.
        merit_new = compute_merit(f_val_new, eq_val_new, ineq_val_new, merit_penalty)
        merit_threshold = self.stagnation_tol * jnp.maximum(
            jnp.abs(state.best_merit), 1.0
        )
        improved = merit_new < state.best_merit - merit_threshold
        new_best_merit = jnp.where(improved, merit_new, state.best_merit)
        new_steps_without = jnp.where(
            improved, jnp.array(0), state.steps_without_improvement + 1
        )
        patience = self._stagnation_window
        merit_stagnation = (state.step_count >= patience) & (
            new_steps_without >= patience
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
            consecutive_ls_failures=new_consecutive_ls_failures,
            best_merit=new_best_merit,
            steps_without_improvement=new_steps_without,
            stagnation=merit_stagnation,
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
            num_steps=("Step", new_state.step_count),  # ty: ignore[unresolved-attribute]  # equinox @dataclass_transform
            objective=("f", f_val_new, ".6e"),
            constraint_violation=("|c|", c_viol, ".3e"),
            kkt_residual=("|∇L|", kkt, ".3e"),
            grad_norm=("|∇f|", grad_norm, ".3e"),
            step_size=("α", alpha, ".3e"),
            direction_norm=("|d|", dir_norm, ".3e"),
            merit=("merit", merit_new, ".6e"),
            stag_count=("stag#", new_steps_without),
            stagnation=("stag", merit_stagnation),
            penalty=("ρ", merit_penalty, ".3e"),
            lbfgs_gamma=("γ", new_lbfgs_history.gamma, ".3e"),
            lbfgs_diag_cond=("κ_B", diag_cond, ".1e"),
            qp_iters=("QP it", qp_result.iterations),
            qp_converged=("QP ok", qp_result.converged),
            n_active=("#act", n_active),
            ls_steps=("LS it", ls_result.n_evals),
            ls_success=("LS ok", ls_result.success),
        )

        return y_new, new_state, aux  # ty: ignore[invalid-return-type]  # equinox @dataclass_transform

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

        # Lagrangian value: L = f - lambda_eq^T c_eq - mu_ineq^T c_ineq
        lagrangian_val = state.f_val
        if m_eq > 0:
            lagrangian_val = lagrangian_val - state.multipliers_eq @ state.eq_val
        if m_ineq_total > 0:
            lagrangian_val = lagrangian_val - state.multipliers_ineq @ state.ineq_val

        # Relative stationarity: ||nabla L|| <= rtol * max(|L|, 1)
        grad_norm = jnp.linalg.norm(grad_lagrangian)
        stationarity = grad_norm <= self.rtol * jnp.maximum(
            jnp.abs(lagrangian_val), 1.0
        )

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

        # Check merit-based stagnation
        stagnation_detected = state.stagnation

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
            "stagnation": state.stagnation,
            "last_step_size": state.last_alpha,
            "consecutive_ls_failures": state.consecutive_ls_failures,
        }

        return y, aux, stats

    def _solve_qp_subproblem(
        self,
        state: SLSQPState,
        hvp_fn: Callable[[Vector], Vector],
        y: Vector,
    ) -> QPResult:
        """Solve the QP subproblem for the search direction.

        Solves:
            minimize    (1/2) d^T B d + g^T d
            subject to  A_eq d = -c_eq
                        A_ineq d >= -c_ineq
                        bounds[:, 0] - y <= d <= bounds[:, 1] - y

        using the HVP function v -> B @ v for the Hessian.

        **Bound handling.**  Box constraints are *not* passed to the QP
        solver as general inequality constraints.  Only the
        ``n_ineq_constraints`` general rows enter the QP's constraint
        matrix ``A``.  After the QP solve, the direction is clipped to
        ``[bounds[:, 0] - y, bounds[:, 1] - y]``.  Bound multipliers
        are recovered from the reduced gradient ``Bd + g - A^T lambda``
        at the clipped point.  This avoids forming a large projection
        matrix when many bounds are present: the CG projection cost
        drops from ``O((m_eq + m_gen + m_bounds)^3)`` to
        ``O((m_eq + m_gen)^3)`` per CG step.

        The previous iteration's QP active set is passed as a warm-start
        hint, and the outer KKT residual norm is used for adaptive EXPAND
        tolerance scaling.  For equality-constrained problems, the
        adaptive proximal parameter ``mu = max(kkt^tau, mu_min)`` is
        computed and the equality constraints are absorbed into the
        objective via sSQP (Wright, 2002, eq 6.6).

        When ``active_set_method`` is ``"lpeca_init"`` or ``"lpeca"``,
        the LPEC-A predicted active set (Oberlin & Wright, 2005) is
        computed from the current NLP iterate and multiplier estimates,
        and passed to the QP solver for initialization.

        When ``use_preconditioner`` is True, the L-BFGS inverse Hessian
        (with Woodbury correction for the proximal term) is passed to
        the QP solver as a CG preconditioner.

        When ``adaptive_cg_tol`` is True, the CG tolerance is adapted
        based on the outer KKT residual (Eisenstat-Walker style).

        Args:
            state: Current solver state.
            hvp_fn: Hessian-vector product function for the Lagrangian.
            y: Current iterate (used for computing box constraints on d).

        Returns:
            QPResult containing the search direction, multipliers, and
            active set.
        """
        g = state.grad

        A_eq = state.eq_jac
        b_eq = -state.eq_val

        m_ineq_general = self.n_ineq_constraints
        m_bounds = self._n_lower_bounds + self._n_upper_bounds

        # Only pass general inequality constraints to QP (not bounds).
        # This keeps the projection matrix small: O((m_eq + m_gen_active)^3)
        # instead of O((m_eq + m_gen_active + m_bounds_active)^3) per CG step.
        A_ineq = state.ineq_jac[:m_ineq_general]
        b_ineq = -state.ineq_val[:m_ineq_general]

        kkt_residual = jnp.linalg.norm(state.prev_grad_lagrangian)

        initial_active_set = (
            state.prev_active_set[:m_ineq_general] if m_ineq_general > 0 else None
        )

        # LPEC-A: compute predicted active set from NLP-level data.
        # Uses full ineq data (including bounds) for the prediction,
        # then slices to the general-ineq portion for the QP.
        predicted_active_set = None
        if self.active_set_method in ("lpeca_init", "lpeca"):
            m_ineq_total = m_ineq_general + m_bounds
            if m_ineq_total > 0:
                full_predicted = compute_lpeca_active_set(
                    c_ineq=state.ineq_val,
                    c_eq=state.eq_val,
                    grad=state.grad,
                    A_ineq=state.ineq_jac,
                    A_eq=state.eq_jac,
                    lambda_ineq=state.multipliers_ineq,
                    mu_eq=state.multipliers_eq,
                    sigma=self.lpeca_sigma,
                    beta=self.lpeca_beta,
                    use_lp=self.lpeca_use_lp,
                )
                predicted_active_set = (
                    full_predicted[:m_ineq_general] if m_ineq_general > 0 else None
                )

        use_proximal = self.proximal_tau > 0
        if use_proximal:
            # Adaptive proximal mu (Wright, 2002, eq 6.6):
            # mu = clip(kkt_residual^tau, mu_min, mu_max)
            # Capped at mu_max so the proximal weight 1/mu >= 1/mu_max,
            # ensuring adequate equality enforcement even far from the solution.
            # Wright's local convergence analysis assumes eta < 1; when the
            # KKT residual is large the cap keeps the penalty tight.
            mu = jnp.clip(
                kkt_residual**self.proximal_tau,
                self._proximal_mu_min,
                self._proximal_mu_max,
            )
        else:
            mu = 0.0

        precond_fn = self._build_preconditioner(
            state,
            proximal_mu=mu,
            lagrangian_hvp_fn=(
                hvp_fn if self.preconditioner_type == "diagonal" else None
            ),
        )

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
            proximal_mu=mu,
            prev_multipliers_eq=state.multipliers_eq,
            precond_fn=precond_fn,
            cg_tol=adaptive_tol,
            cg_regularization=self.cg_regularization,
            use_proximal=use_proximal,
            predicted_active_set=predicted_active_set,
            active_set_method=self.active_set_method,
            use_constraint_preconditioner=self.use_exact_hvp_in_qp,
        )

        direction = qp_result.d

        # --- Bound post-processing (iterative Phase 2) ---
        # Iterative bound-fixing active set: fix variables at bounds,
        # re-solve CG in the free subspace, check for new violations
        # and wrong-sign multipliers, repeat.  Each CG re-solve uses
        # only (m_eq + m_gen_active) rows in the projection matrix,
        # keeping cost at O((m_eq + m_gen)^3) instead of
        # O((m_eq + m_gen + m_bounds)^3).
        #
        # The loop terminates when no variables are added to or
        # dropped from the bound-active set.  Typically 2-5 passes.
        if m_bounds > 0:
            assert self.bounds is not None
            n_vars = g.shape[0]
            d_lower = self.bounds[:, 0] - y
            d_upper = self.bounds[:, 1] - y
            finite_lower = jnp.isfinite(d_lower)
            finite_upper = jnp.isfinite(d_upper)

            # Combined constraint matrix: equality + active gen ineq
            A_combined = jnp.concatenate([A_eq, A_ineq], axis=0)
            b_combined = jnp.concatenate([b_eq, b_ineq], axis=0)
            m_eq_static = self.n_eq_constraints
            eq_active = jnp.ones(m_eq_static, dtype=bool)
            combined_active = jnp.concatenate([eq_active, qp_result.active_set])

            inner_cg_tol = adaptive_tol if adaptive_tol is not None else 1e-8

            free_mask = jnp.ones(n_vars, dtype=bool)
            d_fixed = jnp.zeros(n_vars)
            mult_combined = jnp.zeros(A_combined.shape[0])

            bound_fix_tol = 1e-12
            for _bound_pass in range(5):
                # --- Add step: fix free variables that violate bounds ---
                add_lower = (
                    (direction <= d_lower + bound_fix_tol) & finite_lower & free_mask
                )
                add_upper = (
                    (direction >= d_upper - bound_fix_tol) & finite_upper & free_mask
                )
                add_set = add_lower | add_upper

                # --- Drop step: release fixed variables with wrong-sign
                # bound multipliers.  A lower-bound multiplier should be
                # >= 0 (pushing the variable up); if negative, the variable
                # wants to move away from the bound and should be freed.
                # Similarly for upper bounds. ---
                Bd_cur = hvp_fn(direction)
                grad_qp_cur = Bd_cur + g
                cf = jnp.zeros_like(g)
                if m_eq_static > 0:
                    cf = cf + A_eq.T @ mult_combined[:m_eq_static]
                if m_ineq_general > 0:
                    cf = cf + A_ineq.T @ mult_combined[m_eq_static:]
                reduced_grad_cur = grad_qp_cur - cf

                at_lower_cur = ~free_mask & (d_fixed <= d_lower + bound_fix_tol)
                at_upper_cur = ~free_mask & (d_fixed >= d_upper - bound_fix_tol)
                drop_lower = at_lower_cur & (reduced_grad_cur < -bound_fix_tol)
                drop_upper = at_upper_cur & (-reduced_grad_cur < -bound_fix_tol)
                drop_set = drop_lower | drop_upper

                any_change = jnp.any(add_set | drop_set)

                new_free_mask = (free_mask & ~add_set) | drop_set
                new_d_fixed = jnp.where(
                    add_lower,
                    d_lower,
                    jnp.where(add_upper, d_upper, d_fixed),
                )
                new_d_fixed = jnp.where(drop_set, 0.0, new_d_fixed)

                free_mask = jnp.where(any_change, new_free_mask, free_mask)
                d_fixed = jnp.where(any_change, new_d_fixed, d_fixed)

                any_fixed = ~jnp.all(free_mask)

                d_new, mult_new, _ = _solve_projected_cg(
                    hvp_fn,
                    g,
                    A_combined,
                    b_combined,
                    combined_active,
                    self.qp_max_cg_iter,
                    inner_cg_tol,
                    precond_fn=precond_fn,
                    cg_regularization=self.cg_regularization,
                    free_mask=free_mask,
                    d_fixed=d_fixed,
                    use_constraint_preconditioner=self.use_exact_hvp_in_qp,
                )

                use_new = any_change & any_fixed
                direction = jnp.where(use_new, d_new, direction)
                mult_combined = jnp.where(use_new, mult_new, mult_combined)

            # Final bound-active identification from the converged direction
            at_lower_full = (direction <= d_lower + bound_fix_tol) & finite_lower
            at_upper_full = (direction >= d_upper - bound_fix_tol) & finite_upper
            any_bound_active = jnp.any(at_lower_full | at_upper_full)

            mult_eq_final = jnp.where(
                any_bound_active,
                mult_combined[:m_eq_static],
                qp_result.multipliers_eq,
            )
            mult_gen_final = (
                jnp.where(
                    any_bound_active,
                    mult_combined[m_eq_static:],
                    qp_result.multipliers_ineq,
                )
                if m_ineq_general > 0
                else qp_result.multipliers_ineq
            )

            # Recover bound multipliers from the reduced gradient
            lower_idx = np.array(self._lower_indices)
            upper_idx = np.array(self._upper_indices)

            Bd = hvp_fn(direction)
            grad_qp = Bd + g
            constraint_force = jnp.zeros_like(g)
            if self.n_eq_constraints > 0:
                constraint_force = constraint_force + A_eq.T @ mult_eq_final
            if m_ineq_general > 0:
                constraint_force = constraint_force + A_ineq.T @ mult_gen_final
            reduced_grad = grad_qp - constraint_force

            at_lower = (
                at_lower_full[lower_idx]
                if len(lower_idx) > 0
                else jnp.zeros((0,), dtype=bool)
            )
            at_upper = (
                at_upper_full[upper_idx]
                if len(upper_idx) > 0
                else jnp.zeros((0,), dtype=bool)
            )

            bound_mult_lower = (
                jnp.where(at_lower, reduced_grad[lower_idx], 0.0)
                if len(lower_idx) > 0
                else jnp.zeros((0,))
            )
            bound_mult_upper = (
                jnp.where(at_upper, -reduced_grad[upper_idx], 0.0)
                if len(upper_idx) > 0
                else jnp.zeros((0,))
            )

            multipliers_eq = mult_eq_final
            multipliers_ineq = jnp.concatenate(
                [mult_gen_final, bound_mult_lower, bound_mult_upper]
            )
            active_set = jnp.concatenate([qp_result.active_set, at_lower, at_upper])
        else:
            multipliers_eq = qp_result.multipliers_eq
            multipliers_ineq = qp_result.multipliers_ineq
            active_set = qp_result.active_set

        return QPResult(  # ty: ignore[invalid-return-type]  # equinox @dataclass_transform
            direction=direction,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            active_set=active_set,
            converged=qp_result.converged,
            iterations=qp_result.iterations,
        )
