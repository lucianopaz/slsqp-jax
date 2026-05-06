"""Shared add / drop / EXPAND / ping-pong active-set loop.

The legacy ``slsqp_jax/qp_solver.py`` carried three near-identical copies
of this loop body — one in ``_solve_qp_proximal``, one in
``_solve_qp_direct``, and one inlined in ``solve_qp``'s ineq-only path.
This module hosts the *single* implementation; the three QP-strategy
modules now build their own initial state and inner-solve closure and
delegate to :func:`run_active_set_loop`.

The loop is parameterised by the **inner solve closure**:

    inner_solve_fn(state) -> ActiveSetInnerResult

which packages the new direction, equality / inequality multipliers,
projection diagnostics, and the inner-failure flag.  Callers control
how the inner solver is wired (proximal-stabilised HVP vs. direct
projection) by what they put in this closure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.state import QPState
from slsqp_jax.types import Scalar, Vector


class ActiveSetInnerResult(NamedTuple):
    """Per-iteration payload returned by the caller's inner-solve closure.

    Attributes:
        d: New search direction.
        multipliers_eq: Recovered (or absorbed) equality multipliers
            for *this* iteration.  Pass an empty ``(0,)`` vector when no
            equality constraints are present.
        multipliers_ineq: Recovered inequality multipliers (length
            ``m_ineq``); zero on inactive entries.
        inner_failed: Whether the inner solve produced a non-finite
            direction.
        proj_residual: Latest M-metric projection residual from the
            inner solver (``MinresQLPSolver``); always ``0`` for
            null-space CG / CRAIG.
        n_proj_refinements: Refinement rounds applied by the inner
            solver on this call.  Accumulated by the loop into the
            running QPState count.
        projected_grad_norm: Latest projected-gradient norm from the
            inner solver (``HRInexactSTCG`` only; ``inf`` otherwise).
    """

    d: Vector
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    inner_failed: Bool[Array, ""]
    proj_residual: Scalar
    n_proj_refinements: Array
    projected_grad_norm: Scalar


def run_active_set_loop(
    init_state: QPState,
    inner_solve_fn: Callable[[QPState, Scalar], ActiveSetInnerResult],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    max_iter: int,
    base_tol: Scalar,
    effective_tau: Scalar | float,
    drop_floor: Scalar,
    ping_pong_threshold: int,
) -> QPState:
    """Run the shared add / drop / EXPAND / ping-pong active-set loop.

    The body, identical across all three QP strategies:

    1. Compute the working tolerance ``working_tol = base_tol +
       iteration * effective_tau`` (EXPAND ramp).
    2. Call the caller-supplied :func:`inner_solve_fn` with the current
       active set; receive a new direction, multipliers, and
       projection diagnostics.
    3. Compute violation scores ``A_ineq d - b_ineq`` and drop scores
       ``mult_ineq < -max(working_tol, drop_floor)``.
    4. Branch on add / drop / mark_converged with the same ping-pong
       short-circuit logic that the legacy bodies used.

    Args:
        init_state: Initial active-set state.
        inner_solve_fn: Closure performing the per-iteration inner
            equality-constrained QP solve.  Receives the current
            ``QPState`` and the current ``working_tol`` (some callers
            ignore it).
        A_ineq: Inequality constraint matrix used for the violation
            check.  Must match what ``inner_solve_fn`` consumes.
        b_ineq: Inequality RHS for the violation check.
        max_iter: Iteration budget (matches ``while_loop`` cond).
        base_tol: Outer SQP-level base tolerance.
        effective_tau: Per-iteration EXPAND increment.  ``0.0``
            disables the ramp.
        drop_floor: Floor on the drop test so multiplier-recovery noise
            does not flip a negligible negative multiplier into a drop.
        ping_pong_threshold: Threshold for the explicit add/drop
            ping-pong short-circuit (``2**31 - 1`` effectively
            disables it).

    Returns:
        The final ``QPState`` after the active-set loop terminates.
    """

    def cond_fn(state: QPState) -> Bool[Array, ""]:
        return ~state.converged & (state.iteration < max_iter)

    def body_fn(state: QPState) -> QPState:
        working_tol = base_tol + state.iteration * effective_tau
        inner = inner_solve_fn(state, working_tol)
        d_new = inner.d
        mult_eq_new = inner.multipliers_eq
        mult_ineq_new = inner.multipliers_ineq
        new_any_inner_failure = state.any_inner_failure | inner.inner_failed

        proj_residual_new = inner.proj_residual
        n_proj_refinements_new = state.n_proj_refinements + inner.n_proj_refinements
        projected_grad_norm_new = inner.projected_grad_norm

        # Feasibility check with EXPAND-relaxed tolerance.
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -working_tol) & ~state.active_set
        any_violated = jnp.any(violated)

        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        # Noise-aware drop test.
        drop_tol = jnp.maximum(working_tol, drop_floor)
        negative_mult = (mult_ineq_new < -drop_tol) & state.active_set
        any_negative = jnp.any(negative_mult)

        mult_scores = jnp.where(state.active_set, mult_ineq_new, jnp.inf)
        most_negative_idx = jnp.argmin(mult_scores)

        def add_constraint() -> QPState:
            is_pp = (state.last_drop_idx >= 0) & (
                most_violated_idx == state.last_drop_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_violated_idx].set(True),
            )
            return QPState(  # ty: ignore[invalid-return-type]
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=jnp.where(
                    triggered, state.last_add_idx, most_violated_idx
                ),
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
                proj_residual=proj_residual_new,
                n_proj_refinements=n_proj_refinements_new,
                projected_grad_norm=projected_grad_norm_new,
            )

        def drop_constraint() -> QPState:
            is_pp = (state.last_add_idx >= 0) & (
                most_negative_idx == state.last_add_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_negative_idx].set(False),
            )
            return QPState(  # ty: ignore[invalid-return-type]
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=jnp.where(
                    triggered, state.last_drop_idx, most_negative_idx
                ),
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
                proj_residual=proj_residual_new,
                n_proj_refinements=n_proj_refinements_new,
                projected_grad_norm=projected_grad_norm_new,
            )

        def mark_converged() -> QPState:
            return QPState(  # ty: ignore[invalid-return-type]
                d=d_new,
                active_set=state.active_set,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(True),
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=state.ping_pong_count,
                ping_ponged=state.ping_ponged,
                proj_residual=proj_residual_new,
                n_proj_refinements=n_proj_refinements_new,
                projected_grad_norm=projected_grad_norm_new,
            )

        return jax.lax.cond(
            any_violated,
            add_constraint,
            lambda: jax.lax.cond(any_negative, drop_constraint, mark_converged),
        )

    return jax.lax.while_loop(cond_fn, body_fn, init_state)


__all__ = ["ActiveSetInnerResult", "run_active_set_loop"]
