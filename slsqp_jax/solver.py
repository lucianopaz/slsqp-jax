"""SLSQP Solver implementation using Optimistix.

This module contains the main SLSQP solver class that extends
optimistix.AbstractMinimiser to provide Sequential Quadratic Programming
optimization with support for equality and inequality constraints.

The solver supports three modes for the Lagrangian Hessian:
1. L-BFGS (default): Matrix-free quasi-Newton approximation using O(kn)
   storage instead of O(n^2). Suitable for n up to 50,000.
2. User-supplied HVPs: Exact Hessian-vector products composed from
   user-provided objective and constraint HVPs.
3. AD-computed HVPs: Forward-over-reverse automatic differentiation
   when the user supplies obj_hvp_fn but not constraint HVPs.

Similarly, gradients and Jacobians can be user-supplied or computed
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
    lbfgs_init,
)
from slsqp_jax.merit import (
    backtracking_line_search,
    update_penalty_parameter,
)
from slsqp_jax.qp_solver import solve_qp
from slsqp_jax.types import (
    ConstraintFn,
    ConstraintHVPFn,
    GradFn,
    HVPFn,
    JacobianFn,
)
from slsqp_jax.utils import args_closure


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
    """

    # Iteration tracking
    step_count: Int[Array, ""]

    # Current function values and gradients
    f_val: Float[Array, ""]
    grad: Float[Array, " n"]

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
    prev_grad_lagrangian: Float[Array, " n"]

    # Merit function penalty parameter
    merit_penalty: Float[Array, ""]


class QPResult(eqx.Module):
    """Result from solving the QP subproblem.

    Attributes:
        direction: The search direction d from the QP solution.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        converged: Whether the QP solver converged successfully.
    """

    direction: Float[Array, " n"]
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    converged: Bool[Array, ""]


class SLSQP(optx.AbstractMinimiser):
    """SLSQP minimizer using Sequential Quadratic Programming.

    This solver implements the SLSQP algorithm for constrained nonlinear
    optimization, designed to scale to large dimensions (n > 5000).

    At each iteration, it:

    1. Constructs a QP subproblem using the Lagrangian Hessian (via L-BFGS
       or exact HVPs) and linearized constraints.
    2. Solves the QP using projected conjugate gradient with an active-set
       method for inequality constraints.
    3. Performs a line search using an L1 merit function.
    4. Updates the L-BFGS history (or skips if using exact HVPs).

    The Hessian is never formed as a dense matrix. Instead, it is accessed
    only through Hessian-vector products (HVPs), enabling O(kn) memory
    usage where k is the L-BFGS memory (typically 10).

    Users can optionally supply their own derivative functions:
    - obj_grad_fn: Gradient of objective (else jax.grad).
    - eq_jac_fn / ineq_jac_fn: Jacobians of constraints (else jax.jacrev).
    - obj_hvp_fn: HVP of objective (triggers exact Hessian mode).
    - eq_hvp_fn / ineq_hvp_fn: HVPs of constraints (else forward-over-reverse AD).

    When obj_hvp_fn is provided, the solver composes the exact Lagrangian
    HVP from the objective and constraint HVPs. When it is not provided,
    the solver uses L-BFGS to approximate the Lagrangian Hessian.

    Attributes:
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        max_steps: Maximum number of iterations.
        eq_constraint_fn: Function computing equality constraints c_eq(x) = 0.
        ineq_constraint_fn: Function computing inequality constraints c_ineq(x) >= 0.
        n_eq_constraints: Number of equality constraints (static).
        n_ineq_constraints: Number of inequality constraints (static).
        obj_grad_fn: Optional gradient of objective.
        eq_jac_fn: Optional Jacobian of equality constraints.
        ineq_jac_fn: Optional Jacobian of inequality constraints.
        obj_hvp_fn: Optional HVP of objective (enables exact Hessian mode).
        eq_hvp_fn: Optional per-constraint HVP of equality constraints.
        ineq_hvp_fn: Optional per-constraint HVP of inequality constraints.
        lbfgs_memory: Number of (s, y) pairs to store for L-BFGS (default 10).

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

    # Constraint functions (static - not differentiated by optimistix)
    eq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)
    ineq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)

    # Number of constraints (must be static for JAX compilation)
    n_eq_constraints: int = eqx.field(static=True, default=0)
    n_ineq_constraints: int = eqx.field(static=True, default=0)

    # Box constraints (bounds) on decision variables
    # Shape (n, 2) where bounds[i, 0] = lower, bounds[i, 1] = upper
    # Use -jnp.inf / jnp.inf for unbounded dimensions
    bounds: Optional[Float[Array, "n 2"]] = None

    # Masks indicating which bounds are finite (static for compilation)
    # These are computed from bounds at construction time and stored as tuples
    _lower_bound_mask: Optional[tuple[bool, ...]] = eqx.field(static=True, default=None)
    _upper_bound_mask: Optional[tuple[bool, ...]] = eqx.field(static=True, default=None)
    _n_lower_bounds: int = eqx.field(static=True, default=0)
    _n_upper_bounds: int = eqx.field(static=True, default=0)

    # Optional user-supplied derivative functions
    obj_grad_fn: Optional[GradFn] = eqx.field(static=True, default=None)
    eq_jac_fn: Optional[JacobianFn] = eqx.field(static=True, default=None)
    ineq_jac_fn: Optional[JacobianFn] = eqx.field(static=True, default=None)
    obj_hvp_fn: Optional[HVPFn] = eqx.field(static=True, default=None)
    eq_hvp_fn: Optional[ConstraintHVPFn] = eqx.field(static=True, default=None)
    ineq_hvp_fn: Optional[ConstraintHVPFn] = eqx.field(static=True, default=None)

    # L-BFGS parameters
    lbfgs_memory: int = eqx.field(static=True, default=10)

    # Line search parameters
    line_search_max_steps: int = eqx.field(static=True, default=20)
    armijo_c1: float = 1e-4

    # QP solver parameters
    qp_max_iter: int = eqx.field(static=True, default=100)
    qp_max_cg_iter: int = eqx.field(static=True, default=50)

    def __check_init__(self):
        """Post-initialization to compute bound masks.

        This is called by equinox after __init__ to validate/process fields.
        We compute the static bound masks here.
        """
        if self.bounds is not None:
            # Compute masks for finite bounds (must be done with numpy for static values)
            bounds_np = np.asarray(self.bounds)
            lower_mask = np.isfinite(bounds_np[:, 0])
            upper_mask = np.isfinite(bounds_np[:, 1])

            # Store as tuples for static fields (JAX-compatible)
            object.__setattr__(self, "_lower_bound_mask", tuple(lower_mask.tolist()))
            object.__setattr__(self, "_upper_bound_mask", tuple(upper_mask.tolist()))
            object.__setattr__(self, "_n_lower_bounds", int(np.sum(lower_mask)))
            object.__setattr__(self, "_n_upper_bounds", int(np.sum(upper_mask)))

    def _compute_bound_constraint_values(
        self,
        y: Float[Array, " n"],
    ) -> Float[Array, " m_bounds"]:
        """Compute bound constraint values.

        Returns constraint values c(x) where c(x) >= 0 means feasible.
        - Lower bounds: c_i = y_i - lower_i >= 0
        - Upper bounds: c_i = upper_i - y_i >= 0

        The values are concatenated as [lower_bound_values, upper_bound_values].

        Args:
            y: Current point.

        Returns:
            Array of bound constraint values, shape (n_lower + n_upper,).
        """
        if self.bounds is None or (
            self._n_lower_bounds == 0 and self._n_upper_bounds == 0
        ):
            return jnp.zeros((0,))

        # Use static numpy indices (not JAX boolean indexing) for JIT compatibility
        lower_indices = np.array(
            [
                i
                for i, m in enumerate(cast(tuple[bool, ...], self._lower_bound_mask))
                if m
            ]
        )
        upper_indices = np.array(
            [
                i
                for i, m in enumerate(cast(tuple[bool, ...], self._upper_bound_mask))
                if m
            ]
        )

        # Lower bounds: y - lower >= 0
        if len(lower_indices) > 0:
            lower_vals = y[lower_indices] - self.bounds[lower_indices, 0]
        else:
            lower_vals = jnp.zeros((0,))

        # Upper bounds: upper - y >= 0
        if len(upper_indices) > 0:
            upper_vals = self.bounds[upper_indices, 1] - y[upper_indices]
        else:
            upper_vals = jnp.zeros((0,))

        return jnp.concatenate([lower_vals, upper_vals])

    def _build_bound_jacobian(
        self,
        n: int,
    ) -> Float[Array, "m_bounds n"]:
        """Build the Jacobian matrix for bound constraints.

        The Jacobian is:
        - For lower bound y_i >= lower_i: row is e_i (unit vector, +1 at position i)
        - For upper bound y_i <= upper_i: row is -e_i (-1 at position i)

        Args:
            n: Number of decision variables.

        Returns:
            Jacobian matrix of shape (n_lower + n_upper, n).
        """
        if self.bounds is None or (
            self._n_lower_bounds == 0 and self._n_upper_bounds == 0
        ):
            return jnp.zeros((0, n))

        # Use static numpy indices for JIT compatibility
        lower_indices = np.array(
            [
                i
                for i, m in enumerate(cast(tuple[bool, ...], self._lower_bound_mask))
                if m
            ]
        )
        upper_indices = np.array(
            [
                i
                for i, m in enumerate(cast(tuple[bool, ...], self._upper_bound_mask))
                if m
            ]
        )

        # Lower bounds: +1 on diagonal for active indices
        # Select rows from identity matrix
        identity = jnp.eye(n)
        if len(lower_indices) > 0:
            J_lower = identity[lower_indices]  # (n_lower, n)
        else:
            J_lower = jnp.zeros((0, n))

        # Upper bounds: -1 on diagonal for active indices
        if len(upper_indices) > 0:
            J_upper = -identity[upper_indices]  # (n_upper, n)
        else:
            J_upper = jnp.zeros((0, n))

        return jnp.concatenate([J_lower, J_upper], axis=0)

    def _compute_grad(
        self,
        fn: Callable,
        y: Float[Array, " n"],
        args: Any,
    ) -> Float[Array, " n"]:
        """Compute gradient of objective using user-supplied fn or AD."""
        if self.obj_grad_fn is not None:
            return self.obj_grad_fn(y, args)
        return jax.grad(lambda x: fn(x, args)[0])(y)

    def _compute_eq_jac(
        self,
        y: Float[Array, " n"],
        args: Any,
    ) -> Float[Array, "m_eq n"]:
        """Compute equality constraint Jacobian."""
        n = y.shape[0]
        m_eq = self.n_eq_constraints
        if self.eq_constraint_fn is not None and m_eq > 0:
            if self.eq_jac_fn is not None:
                return self.eq_jac_fn(y, args)
            return jax.jacrev(args_closure(self.eq_constraint_fn, args))(y)
        return jnp.zeros((m_eq, n))

    def _compute_ineq_jac(
        self,
        y: Float[Array, " n"],
        args: Any,
    ) -> Float[Array, "m_ineq n"]:
        """Compute inequality constraint Jacobian."""
        n = y.shape[0]
        m_ineq = self.n_ineq_constraints
        if self.ineq_constraint_fn is not None and m_ineq > 0:
            if self.ineq_jac_fn is not None:
                return self.ineq_jac_fn(y, args)
            return jax.jacrev(args_closure(self.ineq_constraint_fn, args))(y)
        return jnp.zeros((m_ineq, n))

    def _build_lagrangian_hvp(
        self,
        fn: Callable,
        y: Float[Array, " n"],
        args: Any,
        state: "SLSQPState",
    ) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
        """Build the Lagrangian HVP function for the QP subproblem.

        Returns a closure v -> H_L(y) @ v where H_L is the Hessian of the
        Lagrangian at the current point y with the current multipliers.

        When obj_hvp_fn is provided, composes the exact Lagrangian HVP:
            H_L v = H_f v - sum_i lambda_eq_i * H_{c_eq_i} v
                         - sum_j lambda_ineq_j * H_{c_ineq_j} v

        When obj_hvp_fn is not provided, uses L-BFGS:
            H_L v approx B_k v  (L-BFGS two-loop recursion)
        """
        use_exact_hvp = self.obj_hvp_fn is not None

        if use_exact_hvp:
            m_eq = self.n_eq_constraints
            m_ineq = self.n_ineq_constraints

            def lagrangian_hvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
                # Objective HVP
                obj_hvp_val = self.obj_hvp_fn(y, v, args)  # type: ignore[misc]

                # Equality constraint HVP contribution
                eq_contribution = jnp.zeros_like(v)
                if self.eq_constraint_fn is not None and m_eq > 0:
                    if self.eq_hvp_fn is not None:
                        # User-supplied per-constraint HVPs: (m_eq, n)
                        eq_hvps = self.eq_hvp_fn(y, v, args)
                        eq_contribution = state.multipliers_eq @ eq_hvps
                    else:
                        # AD fallback: forward-over-reverse on lambda^T c(x)
                        def weighted_eq(x):
                            return jnp.dot(
                                state.multipliers_eq,
                                self.eq_constraint_fn(x, args),  # type: ignore[misc]
                            )

                        _, eq_contribution = jax.jvp(jax.grad(weighted_eq), (y,), (v,))

                # Inequality constraint HVP contribution
                # Note: Only use general inequality multipliers (not bound multipliers)
                # since bounds have zero Hessian and don't contribute to the HVP.
                ineq_contribution = jnp.zeros_like(v)
                if self.ineq_constraint_fn is not None and m_ineq > 0:
                    # Extract only the general inequality multipliers
                    multipliers_ineq_general = state.multipliers_ineq[:m_ineq]

                    if self.ineq_hvp_fn is not None:
                        ineq_hvps = self.ineq_hvp_fn(y, v, args)
                        ineq_contribution = multipliers_ineq_general @ ineq_hvps
                    else:

                        def weighted_ineq(x):
                            return jnp.dot(
                                multipliers_ineq_general,
                                self.ineq_constraint_fn(x, args),  # type: ignore[misc]
                            )

                        _, ineq_contribution = jax.jvp(
                            jax.grad(weighted_ineq), (y,), (v,)
                        )

                # L(x) = f(x) - lambda_eq^T c_eq(x) - lambda_ineq^T c_ineq(x)
                return obj_hvp_val - eq_contribution - ineq_contribution

            return lagrangian_hvp
        else:
            # L-BFGS mode: use stored history
            lbfgs_history = state.lbfgs_history

            def lbfgs_lagrangian_hvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
                return lbfgs_hvp(lbfgs_history, v)

            return lbfgs_lagrangian_hvp

    def init(
        self,
        fn: Callable,
        y: Float[Array, " n"],
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
        m_eq = self.n_eq_constraints
        m_ineq_general = self.n_ineq_constraints
        m_bounds = self._n_lower_bounds + self._n_upper_bounds
        m_ineq_total = m_ineq_general + m_bounds

        # Evaluate objective
        f_val, _aux = fn(y, args)

        # Compute gradient (user-supplied or AD)
        grad = self._compute_grad(fn, y, args)

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

        # Compute constraint Jacobians (user-supplied or AD via jacrev)
        eq_jac = self._compute_eq_jac(y, args)
        ineq_jac_general = self._compute_ineq_jac(y, args)

        # Build bound Jacobian and concatenate
        bound_jac = self._build_bound_jacobian(n)
        ineq_jac = jnp.concatenate([ineq_jac_general, bound_jac], axis=0)

        # Initialize L-BFGS history (empty, gamma=1 -> B_0 = I)
        lbfgs_history = lbfgs_init(n, self.lbfgs_memory)

        # Initialize multipliers to zero
        multipliers_eq = jnp.zeros((m_eq,))
        multipliers_ineq = jnp.zeros((m_ineq_total,))

        # Initial merit penalty
        merit_penalty = jnp.array(1.0)

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
            prev_grad_lagrangian=grad,  # No constraint contribution at first step
            merit_penalty=merit_penalty,
        )

    def step(
        self,
        fn: Callable,
        y: Float[Array, " n"],
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
    ) -> tuple[Float[Array, " n"], SLSQPState, Any]:
        """Perform one SLSQP iteration.

        This method:
        1. Builds the Lagrangian HVP (L-BFGS or exact).
        2. Solves the QP subproblem via projected CG to find direction d.
        3. Performs line search with L1 merit function to find step size.
        4. Updates x_{k+1} = x_k + alpha * d.
        5. Re-evaluates objective, gradient, constraints, and Jacobians.
        6. Updates L-BFGS history (if using L-BFGS mode).

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
        # Step 1: Build the Lagrangian HVP function
        hvp_fn = self._build_lagrangian_hvp(fn, y, args, state)

        # Step 2: Solve QP subproblem for search direction
        qp_result = self._solve_qp_subproblem(state, hvp_fn)
        direction = qp_result.direction

        # Step 3: Update penalty parameter based on new multipliers
        merit_penalty = update_penalty_parameter(
            state.merit_penalty,
            qp_result.multipliers_eq,
            qp_result.multipliers_ineq,
        )

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
        y_new = y + alpha * direction
        f_val_new = ls_result.f_val
        eq_val_new = ls_result.eq_val
        ineq_val_new = ls_result.ineq_val  # Includes bounds from line search

        # Get auxiliary output from function evaluation
        _, aux = fn(y_new, args)

        # Step 5: Compute gradient and Jacobians at new point
        n = y.shape[0]
        grad_new = self._compute_grad(fn, y_new, args)
        eq_jac_new = self._compute_eq_jac(y_new, args)

        # General inequality Jacobian
        ineq_jac_general_new = self._compute_ineq_jac(y_new, args)

        # Bound Jacobian (constant, doesn't depend on y)
        bound_jac = self._build_bound_jacobian(n)

        # Concatenate general + bound Jacobians
        ineq_jac_new = jnp.concatenate([ineq_jac_general_new, bound_jac], axis=0)

        # Step 6: Update L-BFGS history (only in L-BFGS mode)
        s = y_new - y  # Step taken

        # Compute gradient of Lagrangian at new point
        grad_lagrangian_new = compute_lagrangian_gradient(
            grad_new,
            eq_jac_new,
            ineq_jac_new,
            qp_result.multipliers_eq,
            qp_result.multipliers_ineq,
        )

        y_diff = grad_lagrangian_new - state.prev_grad_lagrangian

        # Update L-BFGS history if not using exact HVPs
        if self.obj_hvp_fn is None:
            new_lbfgs_history = lbfgs_append(state.lbfgs_history, s, y_diff)
        else:
            # Exact HVP mode: L-BFGS history is not used, keep unchanged
            new_lbfgs_history = state.lbfgs_history

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
        )

        return y_new, new_state, aux

    def terminate(
        self,
        fn: Callable,
        y: Float[Array, " n"],
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

        # Check stationarity: ||nabla L|| <= atol + rtol * ||nabla f||
        grad_norm = jnp.linalg.norm(grad_lagrangian)
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

        # Converged if stationary and feasible
        converged = stationarity & primal_feasible

        # Determine result code
        done = converged | max_iters_reached

        result = jax.lax.cond(
            converged,
            lambda: optx.RESULTS.successful,
            lambda: jax.lax.cond(
                max_iters_reached,
                lambda: optx.RESULTS.max_steps_reached,
                lambda: optx.RESULTS.successful,  # Still running
            ),
        )

        return done, result

    def postprocess(
        self,
        fn: Callable,
        y: Float[Array, " n"],
        aux: Any,
        args: Any,
        options: dict[str, Any],
        state: SLSQPState,
        tags: frozenset[object],
        result: Any,
    ) -> tuple[Float[Array, " n"], Any, dict[str, Any]]:
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
        stats = {
            "num_steps": state.step_count,
            "final_objective": state.f_val,
            "final_grad_norm": jnp.linalg.norm(state.grad),
            "merit_penalty": state.merit_penalty,
        }

        return y, aux, stats

    def _solve_qp_subproblem(
        self,
        state: SLSQPState,
        hvp_fn: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    ) -> QPResult:
        """Solve the QP subproblem for the search direction.

        Solves:
            minimize    (1/2) d^T B d + g^T d
            subject to  A_eq d = -c_eq
                        A_ineq d >= -c_ineq

        using the HVP function v -> B @ v for the Hessian.

        Args:
            state: Current solver state.
            hvp_fn: Hessian-vector product function for the Lagrangian.

        Returns:
            QPResult containing the search direction and multipliers.
        """
        g = state.grad

        # Linearized constraints (rearranged to standard form)
        A_eq = state.eq_jac
        b_eq = -state.eq_val

        A_ineq = state.ineq_jac
        b_ineq = -state.ineq_val

        qp_result = solve_qp(
            hvp_fn=hvp_fn,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            max_iter=self.qp_max_iter,
            max_cg_iter=self.qp_max_cg_iter,
        )

        return QPResult(
            direction=qp_result.d,
            multipliers_eq=qp_result.multipliers_eq,
            multipliers_ineq=qp_result.multipliers_ineq,
            converged=qp_result.converged,
        )
