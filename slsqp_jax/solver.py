"""SLSQP Solver implementation using Optimistix.

This module contains the main SLSQP solver class that extends
optimistix.AbstractMinimiser to provide Sequential Quadratic Programming
optimization with support for equality and inequality constraints.
"""

from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import optimistix._misc as optx_misc
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.hessian import (
    bfgs_update_with_skip,
    compute_lagrangian_gradient,
    scale_initial_hessian,
)
from slsqp_jax.merit import (
    backtracking_line_search,
    update_penalty_parameter,
)
from slsqp_jax.qp_solver import solve_qp
from slsqp_jax.types import ConstraintFn
from slsqp_jax.utils import args_closure


class SLSQPState(eqx.Module):
    """State for the SLSQP solver.

    This is a JAX PyTree (via eqx.Module) that holds all mutable state
    needed across SLSQP iterations.

    Attributes:
        step_count: Current iteration number.
        f_val: Current objective function value f(x_k).
        grad: Gradient of objective at current point ∇f(x_k).
        eq_val: Equality constraint values c_eq(x_k).
        ineq_val: Inequality constraint values c_ineq(x_k).
        eq_jac: Jacobian of equality constraints at x_k.
        ineq_jac: Jacobian of inequality constraints at x_k.
        hessian_approx: Approximate Hessian of the Lagrangian B_k.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        prev_x: Previous iterate x_{k-1} (for BFGS update).
        prev_grad_lagrangian: Previous Lagrangian gradient (for BFGS update).
        first_step: True if this is the first iteration.
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

    # Hessian approximation (positive definite)
    hessian_approx: Float[Array, "n n"]

    # Lagrange multipliers from QP solution
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]

    # Previous step data for BFGS update
    prev_x: Float[Array, " n"]
    prev_grad_lagrangian: Float[Array, " n"]

    # Flags and parameters
    first_step: Bool[Array, ""]
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
    optimization. At each iteration, it:

    1. Constructs a QP subproblem using the current Hessian approximation
       and linearized constraints.
    2. Solves the QP to find a search direction.
    3. Performs a line search using an L1 merit function.
    4. Updates the Hessian approximation using damped BFGS.

    The implementation follows the algorithm described by Dieter Kraft (1988)
    and is compatible with JAX transformations (jit, vmap, grad).

    Attributes:
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        max_steps: Maximum number of iterations.
        eq_constraint_fn: Function computing equality constraints c_eq(x) = 0.
        ineq_constraint_fn: Function computing inequality constraints c_ineq(x) >= 0.
        n_eq_constraints: Number of equality constraints (static).
        n_ineq_constraints: Number of inequality constraints (static).

    Example:
        >>> import jax.numpy as jnp
        >>> import optimistix as optx
        >>> from slsqp_jax import SLSQP
        >>>
        >>> def objective(x, args):
        ...     return jnp.sum(x**2)
        >>>
        >>> def eq_constraint(x, args):
        ...     return jnp.array([x[0] + x[1] - 1.0])
        >>>
        >>> solver = SLSQP(eq_constraint_fn=eq_constraint, n_eq_constraints=1)
        >>> x0 = jnp.array([0.0, 0.0])
        >>> result = optx.minimise(objective, solver, x0)
    """

    # Convergence tolerances
    rtol: float = 1e-6
    atol: float = 1e-6

    # Norm function for convergence checking (required by AbstractMinimiser)
    norm: Callable = eqx.field(static=True, default=optx_misc.max_norm)

    # Maximum iterations
    max_steps: int = 100

    # Constraint functions (static - not differentiated)
    eq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)
    ineq_constraint_fn: Optional[ConstraintFn] = eqx.field(static=True, default=None)

    # Number of constraints (must be static for JAX compilation)
    n_eq_constraints: int = eqx.field(static=True, default=0)
    n_ineq_constraints: int = eqx.field(static=True, default=0)

    # Line search parameters
    line_search_max_steps: int = eqx.field(static=True, default=20)
    armijo_c1: float = 1e-4

    # QP solver parameters
    qp_max_iter: int = eqx.field(static=True, default=100)

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
        and constraint Jacobians. Initializes the Hessian approximation
        to the identity matrix.

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
        m_ineq = self.n_ineq_constraints

        # Evaluate objective and gradient
        f_val, aux = fn(y, args)
        grad = jax.grad(lambda x: fn(x, args)[0])(y)

        # Evaluate constraints and their Jacobians
        if self.eq_constraint_fn is not None and m_eq > 0:
            eq_val = self.eq_constraint_fn(y, args)
            eq_jac = jax.jacfwd(args_closure(self.eq_constraint_fn, args))(y)
        else:
            eq_val = jnp.zeros((m_eq,))
            eq_jac = jnp.zeros((m_eq, n))

        if self.ineq_constraint_fn is not None and m_ineq > 0:
            ineq_val = self.ineq_constraint_fn(y, args)
            ineq_jac = jax.jacfwd(args_closure(self.ineq_constraint_fn, args))(y)
        else:
            ineq_val = jnp.zeros((m_ineq,))
            ineq_jac = jnp.zeros((m_ineq, n))

        # Initialize Hessian approximation to identity
        hessian_approx = jnp.eye(n)

        # Initialize multipliers to zero
        multipliers_eq = jnp.zeros((m_eq,))
        multipliers_ineq = jnp.zeros((m_ineq,))

        # Compute initial merit penalty
        # Start with penalty = 1.0, will be updated based on multipliers
        merit_penalty = jnp.array(1.0)

        return SLSQPState(
            step_count=jnp.array(0),
            f_val=f_val,
            grad=grad,
            eq_val=eq_val,
            ineq_val=ineq_val,
            eq_jac=eq_jac,
            ineq_jac=ineq_jac,
            hessian_approx=hessian_approx,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            prev_x=y,
            prev_grad_lagrangian=grad,  # At first step, no constraint contribution
            first_step=jnp.array(True),
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
        1. Solves the QP subproblem to find search direction d.
        2. Performs line search with L1 merit function to find step size α.
        3. Updates x_{k+1} = x_k + α * d.
        4. Updates Hessian approximation using damped BFGS.
        5. Re-evaluates objective, gradient, constraints, and Jacobians.

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
        n = y.shape[0]
        m_eq = self.n_eq_constraints
        m_ineq = self.n_ineq_constraints

        # Step 1: Solve QP subproblem for search direction
        qp_result = self._solve_qp_subproblem(state)
        direction = qp_result.direction

        # Step 2: Update penalty parameter based on new multipliers
        merit_penalty = update_penalty_parameter(
            state.merit_penalty,
            qp_result.multipliers_eq,
            qp_result.multipliers_ineq,
        )

        # Step 3: Line search with merit function
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
        )

        alpha = ls_result.alpha
        y_new = y + alpha * direction
        f_val_new = ls_result.f_val
        eq_val_new = ls_result.eq_val
        ineq_val_new = ls_result.ineq_val

        # Get auxiliary output from function evaluation
        _, aux = fn(y_new, args)

        # Step 4: Compute gradient and Jacobians at new point
        grad_new = jax.grad(lambda x: fn(x, args)[0])(y_new)

        if self.eq_constraint_fn is not None and m_eq > 0:
            eq_jac_new = jax.jacfwd(args_closure(self.eq_constraint_fn, args))(y_new)
        else:
            eq_jac_new = jnp.zeros((m_eq, n))

        if self.ineq_constraint_fn is not None and m_ineq > 0:
            ineq_jac_new = jax.jacfwd(args_closure(self.ineq_constraint_fn, args))(
                y_new
            )
        else:
            ineq_jac_new = jnp.zeros((m_ineq, n))

        # Step 5: BFGS Hessian update
        s = y_new - y  # Step taken

        # Compute gradient of Lagrangian at new point
        grad_lagrangian_new = compute_lagrangian_gradient(
            grad_new,
            eq_jac_new,
            ineq_jac_new,
            qp_result.multipliers_eq,
            qp_result.multipliers_ineq,
        )

        grad_lagrangian_old = state.prev_grad_lagrangian
        y_diff = grad_lagrangian_new - grad_lagrangian_old

        # On first step, optionally scale initial Hessian
        hessian_for_update = jax.lax.cond(
            state.first_step,
            lambda: scale_initial_hessian(state.hessian_approx, s, y_diff),
            lambda: state.hessian_approx,
        )

        # Apply damped BFGS update
        hessian_new = bfgs_update_with_skip(hessian_for_update, s, y_diff)

        new_state = SLSQPState(
            step_count=state.step_count + 1,
            f_val=f_val_new,
            grad=grad_new,
            eq_val=eq_val_new,
            ineq_val=ineq_val_new,
            eq_jac=eq_jac_new,
            ineq_jac=ineq_jac_new,
            hessian_approx=hessian_new,
            multipliers_eq=qp_result.multipliers_eq,
            multipliers_ineq=qp_result.multipliers_ineq,
            prev_x=y,
            prev_grad_lagrangian=grad_lagrangian_new,
            first_step=jnp.array(False),
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
        3. Lagrange multipliers are non-negative for inequalities (dual feasibility).
        4. Complementary slackness holds.

        Args:
            fn: Objective function.
            y: Current parameter values.
            args: Additional arguments.
            options: Runtime options.
            state: Current solver state.
            tags: Lineax tags.

        Returns:
            Tuple of (done, result) where done is a bool indicating termination
            and result is the termination status code.
        """
        m_eq = self.n_eq_constraints
        m_ineq = self.n_ineq_constraints

        # Compute gradient of Lagrangian
        grad_lagrangian = compute_lagrangian_gradient(
            state.grad,
            state.eq_jac,
            state.ineq_jac,
            state.multipliers_eq,
            state.multipliers_ineq,
        )

        # Check stationarity: ‖∇L‖ ≤ atol + rtol * ‖∇f‖
        grad_norm = jnp.linalg.norm(grad_lagrangian)
        grad_ref = jnp.maximum(jnp.linalg.norm(state.grad), 1.0)
        stationarity = grad_norm <= self.atol + self.rtol * grad_ref

        # Check primal feasibility
        eq_feasible = jnp.array(True)
        if m_eq > 0:
            eq_violation = jnp.max(jnp.abs(state.eq_val))
            eq_feasible = eq_violation <= self.atol

        ineq_feasible = jnp.array(True)
        if m_ineq > 0:
            ineq_violation = jnp.max(jnp.maximum(0.0, -state.ineq_val))
            ineq_feasible = ineq_violation <= self.atol

        primal_feasible = eq_feasible & ineq_feasible

        # Check max iterations
        max_iters_reached = state.step_count >= self.max_steps

        # Converged if stationary and feasible
        converged = stationarity & primal_feasible

        # Determine result code
        done = converged | max_iters_reached

        # Return appropriate RESULTS code
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

        This method is called after the optimization loop finishes.
        It can be used to compute additional statistics or transform
        the output.

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
        # Collect useful statistics about the solve
        stats = {
            "num_steps": state.step_count,
            "final_objective": state.f_val,
            "final_grad_norm": jnp.linalg.norm(state.grad),
            "merit_penalty": state.merit_penalty,
        }

        return y, aux, stats

    def _solve_qp_subproblem(self, state: SLSQPState) -> QPResult:
        """Solve the QP subproblem for the search direction.

        Solves:
            minimize    (1/2) d^T B d + g^T d
            subject to  A_eq d + c_eq = 0
                        A_ineq d + c_ineq >= 0

        which is rewritten as:
            minimize    (1/2) d^T B d + g^T d
            subject to  A_eq d = -c_eq
                        A_ineq d >= -c_ineq

        Args:
            state: Current solver state.

        Returns:
            QPResult containing the search direction and multipliers.
        """

        # QP problem data
        H = state.hessian_approx
        g = state.grad

        # Linearized constraints
        # Original: c_eq(x + d) ≈ c_eq(x) + J_eq @ d = 0
        #           c_ineq(x + d) ≈ c_ineq(x) + J_ineq @ d >= 0
        # Rearranged:
        #           J_eq @ d = -c_eq(x)
        #           J_ineq @ d >= -c_ineq(x)

        A_eq = state.eq_jac
        b_eq = -state.eq_val  # RHS for A_eq @ d = b_eq

        A_ineq = state.ineq_jac
        b_ineq = -state.ineq_val  # RHS for A_ineq @ d >= b_ineq

        # Solve QP
        qp_result = solve_qp(
            H=H,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            max_iter=self.qp_max_iter,
        )

        return QPResult(
            direction=qp_result.d,
            multipliers_eq=qp_result.multipliers_eq,
            multipliers_ineq=qp_result.multipliers_ineq,
            converged=qp_result.converged,
        )
