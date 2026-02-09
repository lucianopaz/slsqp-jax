"""L1 Merit Function and Line Search for SLSQP.

This module implements the Han-Powell L1-exact penalty merit function
and backtracking line search used to globalize the SLSQP algorithm.

The merit function is:
    φ(x; ρ) = f(x) + ρ * (‖c_eq(x)‖_1 + ‖max(0, -c_ineq(x))‖_1)

where ρ is the penalty parameter, chosen large enough to ensure descent.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np_cpu
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from slsqp_jax.types import Scalar, Vector


class LineSearchResult(NamedTuple):
    """Result from the line search.

    Attributes:
        alpha: The step size found.
        f_val: Function value at new point.
        eq_val: Equality constraint values at new point.
        ineq_val: Inequality constraint values at new point.
        success: Whether the line search succeeded.
        n_evals: Number of function evaluations.
    """

    alpha: Scalar
    f_val: Scalar
    eq_val: Float[Array, " m_eq"]
    ineq_val: Float[Array, " m_ineq"]
    success: Bool[Array, ""]
    n_evals: Int[Array, ""]


@jaxtyped(typechecker=beartype)
def compute_merit(
    f_val: Scalar,
    eq_val: Float[Array, " m_eq"],
    ineq_val: Float[Array, " m_ineq"],
    penalty: Scalar,
) -> Scalar:
    """Compute the L1-exact penalty merit function value.

    The merit function is:
        φ(x; ρ) = f(x) + ρ * (‖c_eq(x)‖_1 + ‖max(0, -c_ineq(x))‖_1)

    Args:
        f_val: Objective function value f(x).
        eq_val: Equality constraint values c_eq(x).
        ineq_val: Inequality constraint values c_ineq(x).
        penalty: Penalty parameter ρ.

    Returns:
        Merit function value φ(x; ρ).
    """
    # Equality constraint violation: sum of absolute values
    eq_violation = jnp.sum(jnp.abs(eq_val))

    # Inequality constraint violation: sum of max(0, -c_ineq)
    # c_ineq >= 0 is required, so violation occurs when c_ineq < 0
    ineq_violation = jnp.sum(jnp.maximum(0.0, -ineq_val))

    return f_val + penalty * (eq_violation + ineq_violation)


@jaxtyped(typechecker=beartype)
def update_penalty_parameter(
    current_penalty: Scalar,
    multipliers_eq: Float[Array, " m_eq"],
    multipliers_ineq: Float[Array, " m_ineq"],
    margin: float = 1.1,
) -> Scalar:
    """Update the penalty parameter based on Lagrange multipliers.

    The penalty should be larger than the maximum absolute multiplier
    to ensure the merit function provides a descent direction.

    ``ρ >= max(abs(λ_i), abs(μ_j)) + margin``

    Args:
        current_penalty: Current penalty parameter.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        margin: Safety margin factor (default 1.1).

    Returns:
        Updated penalty parameter.
    """
    # Find maximum absolute multiplier
    max_mult = jnp.array(0.0)

    # Check equality multipliers
    if multipliers_eq.shape[0] > 0:
        max_mult = jnp.maximum(max_mult, jnp.max(jnp.abs(multipliers_eq)))

    # Check inequality multipliers
    if multipliers_ineq.shape[0] > 0:
        max_mult = jnp.maximum(max_mult, jnp.max(jnp.abs(multipliers_ineq)))

    # Ensure penalty is at least margin times the max multiplier
    # Also ensure it never decreases
    new_penalty = jnp.maximum(current_penalty, margin * max_mult)

    # Minimum penalty of 1.0
    new_penalty = jnp.maximum(new_penalty, 1.0)

    return new_penalty


def backtracking_line_search(
    fn: Callable,
    eq_constraint_fn: Callable | None,
    ineq_constraint_fn: Callable | None,
    x: Vector,
    direction: Vector,
    args: Any,
    f_val: Scalar,
    eq_val: Float[Array, " m_eq"],
    ineq_val: Float[Array, " m_ineq"],
    penalty: Scalar,
    grad: Vector,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20,
    alpha_init: float = 1.0,
    bounds: Float[Array, "n 2"] | None = None,
    lower_bound_mask: tuple[bool, ...] | None = None,
    upper_bound_mask: tuple[bool, ...] | None = None,
) -> LineSearchResult:
    """Perform backtracking line search with the L1 merit function.

    Finds α such that the Armijo condition is satisfied:
        φ(x + α*d; ρ) ≤ φ(x; ρ) + c1 * α * φ'(x; d, ρ)

    where φ is the L1 merit function.

    Args:
        fn: Objective function fn(x, args) -> (f_val, aux).
        eq_constraint_fn: Equality constraint function or None.
        ineq_constraint_fn: Inequality constraint function or None.
        x: Current point.
        direction: Search direction.
        args: Arguments to pass to functions.
        f_val: Current objective value.
        eq_val: Current equality constraint values.
        ineq_val: Current inequality constraint values (including bounds).
        penalty: Penalty parameter.
        grad: Gradient of objective at x.
        c1: Armijo condition parameter (default 1e-4).
        rho: Step reduction factor (default 0.5).
        max_iter: Maximum number of iterations.
        alpha_init: Initial step size (default 1.0).
        bounds: Optional box constraints, shape (n, 2) with [lower, upper] per variable.
        lower_bound_mask: Tuple of bools indicating which lower bounds are finite.
        upper_bound_mask: Tuple of bools indicating which upper bounds are finite.

    Returns:
        LineSearchResult with the found step size and function values.
    """
    m_eq = eq_val.shape[0]
    m_ineq = ineq_val.shape[0]

    # Determine how many general inequality constraints vs bounds
    n_lower = sum(lower_bound_mask) if lower_bound_mask is not None else 0
    n_upper = sum(upper_bound_mask) if upper_bound_mask is not None else 0
    n_bounds = n_lower + n_upper
    m_ineq_general = m_ineq - n_bounds

    # Current merit value
    merit_0 = compute_merit(f_val, eq_val, ineq_val, penalty)

    # Directional derivative of objective
    grad_dot_d = jnp.dot(grad, direction)

    # For the L1 merit function, the directional derivative is approximately:
    # φ'(x; d) ≈ ∇f · d - ρ * (reduction in constraint violation)
    # We use a simplified sufficient decrease condition

    # Initial state for the line search loop
    class LSState(NamedTuple):
        alpha: Scalar
        f_val: Scalar
        eq_val: Float[Array, " m_eq"]
        ineq_val: Float[Array, " m_ineq"]
        merit: Scalar
        iteration: Int[Array, ""]
        done: Bool[Array, ""]

    def evaluate_at_alpha(alpha):
        """Evaluate function and constraints at x + alpha * d."""
        x_new = x + alpha * direction
        f_new, _ = fn(x_new, args)

        if eq_constraint_fn is not None and m_eq > 0:
            eq_new = eq_constraint_fn(x_new, args)
        else:
            eq_new = jnp.zeros((m_eq,))

        # Evaluate general inequality constraints
        if ineq_constraint_fn is not None and m_ineq_general > 0:
            ineq_new_general = ineq_constraint_fn(x_new, args)
        else:
            ineq_new_general = jnp.zeros((m_ineq_general,))

        # Evaluate bound constraints
        if bounds is not None and n_bounds > 0:
            # Use static indexing based on precomputed indices
            # The masks are tuples of bools - convert to numpy indices

            lower_indices = np_cpu.array(
                [i for i, m in enumerate(cast(tuple[bool, ...], lower_bound_mask)) if m]
            )
            upper_indices = np_cpu.array(
                [i for i, m in enumerate(cast(tuple[bool, ...], upper_bound_mask)) if m]
            )

            # Lower bounds: x - lower >= 0
            if len(lower_indices) > 0:
                lower_vals = x_new[lower_indices] - bounds[lower_indices, 0]
            else:
                lower_vals = jnp.zeros((0,))

            # Upper bounds: upper - x >= 0
            if len(upper_indices) > 0:
                upper_vals = bounds[upper_indices, 1] - x_new[upper_indices]
            else:
                upper_vals = jnp.zeros((0,))

            bound_vals = jnp.concatenate([lower_vals, upper_vals])
        else:
            bound_vals = jnp.zeros((0,))

        # Concatenate general + bounds
        ineq_new = jnp.concatenate([ineq_new_general, bound_vals])

        merit_new = compute_merit(f_new, eq_new, ineq_new, penalty)

        return f_new, eq_new, ineq_new, merit_new

    # Evaluate at initial alpha
    f_init, eq_init, ineq_init, merit_init = evaluate_at_alpha(jnp.array(alpha_init))

    init_state = LSState(
        alpha=jnp.array(alpha_init),
        f_val=f_init,
        eq_val=eq_init,
        ineq_val=ineq_init,
        merit=merit_init,
        iteration=jnp.array(0),
        done=jnp.array(False),
    )

    def cond_fn(state: LSState) -> Bool[Array, ""]:
        """Continue while not done and under iteration limit."""
        return ~state.done & (state.iteration < max_iter)

    def body_fn(state: LSState) -> LSState:
        """One iteration of backtracking."""
        # Check Armijo condition
        # φ(x + α*d) ≤ φ(x) + c1 * α * directional_deriv
        # For L1 merit, use simplified condition:
        # We want sufficient decrease in merit

        # Compute the sufficient decrease threshold
        # Use a combination of gradient descent and constraint reduction
        sufficient_decrease = merit_0 + c1 * state.alpha * grad_dot_d

        # Check if current alpha satisfies the condition
        armijo_satisfied = state.merit <= sufficient_decrease

        # Also accept if merit decreased at all (fallback)
        merit_decreased = state.merit < merit_0

        # Accept if Armijo is satisfied, or if we've improved and alpha is small
        accept = armijo_satisfied | (merit_decreased & (state.alpha < 0.1))

        def accept_branch():
            return LSState(
                alpha=state.alpha,
                f_val=state.f_val,
                eq_val=state.eq_val,
                ineq_val=state.ineq_val,
                merit=state.merit,
                iteration=state.iteration + 1,
                done=jnp.array(True),
            )

        def reject_branch():
            # Reduce alpha
            new_alpha = rho * state.alpha
            f_new, eq_new, ineq_new, merit_new = evaluate_at_alpha(new_alpha)

            return LSState(
                alpha=new_alpha,
                f_val=f_new,
                eq_val=eq_new,
                ineq_val=ineq_new,
                merit=merit_new,
                iteration=state.iteration + 1,
                done=jnp.array(False),
            )

        return jax.lax.cond(accept, accept_branch, reject_branch)

    # Run the line search
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # If we exhausted iterations, still return the last result
    # (may not satisfy Armijo, but prevents infinite loop)
    success = final_state.done | (final_state.merit < merit_0)

    return LineSearchResult(
        alpha=final_state.alpha,
        f_val=final_state.f_val,
        eq_val=final_state.eq_val,
        ineq_val=final_state.ineq_val,
        success=success,
        n_evals=final_state.iteration + 1,
    )
