"""Direct null-space projection QP strategy (sSQP disabled).

Equality constraints are enforced exactly through the inner solver's
null-space projector instead of the augmented-Lagrangian penalty.  This
avoids the ill-conditioning introduced by the ``(1/mu) A_eq^T A_eq``
proximal term.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.qp._inner_check import inner_ok
from slsqp_jax.qp.active_set import ActiveSetInnerResult, run_active_set_loop
from slsqp_jax.state import QPSolverResult, QPState
from slsqp_jax.types import Scalar, Vector


def solve_qp_direct(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    m_eq: int,
    m_ineq: int,
    max_iter: int,
    tol: Scalar | float,
    expand_factor: float,
    initial_active_set: Bool[Array, " m_ineq"] | None,
    kkt_residual: Scalar | float,
    inner_solver: AbstractInnerSolver,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
    predicted_active_set: Bool[Array, " m_ineq"] | None = None,
    use_expand: bool = True,
    mult_drop_floor: float = 1e-6,
    ping_pong_threshold: int = 2**31 - 1,
) -> QPSolverResult:
    """Solve the QP with equality constraints enforced via direct projection.

    A combined constraint matrix ``[A_eq; A_ineq]`` is formed.  Equality
    rows are permanently active; the active-set loop only adds/drops
    inequality rows.
    """
    A_combined = jnp.concatenate([A_eq, A_ineq], axis=0)
    b_combined = jnp.concatenate([b_eq, b_ineq], axis=0)
    eq_active = jnp.ones(m_eq, dtype=bool)

    _adaptive_tol: Scalar | float | None = cg_tol

    if m_ineq == 0:
        # Equality-only: single projected solve, no active-set loop.
        result = inner_solver.solve(
            hvp_fn,
            g,
            A_combined,
            b_combined,
            eq_active,
            precond_fn=precond_fn,
            adaptive_tol=_adaptive_tol,
        )
        return QPSolverResult(
            d=result.d,
            multipliers_eq=result.multipliers[:m_eq],
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=inner_ok(result),
            iterations=jnp.array(1),
            ping_ponged=jnp.array(False),
            reached_max_iter=jnp.array(False),
            final_working_tol=jnp.asarray(0.0, dtype=jnp.float64),
            proj_residual=result.proj_residual.astype(jnp.float64),
            n_proj_refinements=result.n_proj_refinements,
            projected_grad_norm=result.projected_grad_norm.astype(jnp.float64),
        )

    # Equality + inequality: active-set loop on the inequality portion.
    kkt_res = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_res, 1.0) * tol

    init_result = inner_solver.solve(
        hvp_fn,
        g,
        A_combined,
        b_combined,
        jnp.concatenate([eq_active, jnp.zeros(m_ineq, dtype=bool)]),
        precond_fn=precond_fn,
        adaptive_tol=_adaptive_tol,
    )
    d_init = init_result.d
    mult_init = init_result.multipliers
    init_inner_failure = ~inner_ok(init_result)

    residuals_init = A_ineq @ d_init - b_ineq
    if predicted_active_set is not None:
        init_ineq_active = predicted_active_set | (residuals_init < -base_tol)
    elif initial_active_set is not None:
        init_ineq_active = initial_active_set | (residuals_init < -base_tol)
    else:
        init_ineq_active = residuals_init < -base_tol  # pragma: no cover
    init_converged = ~jnp.any(init_ineq_active)

    init_state = QPState(  # ty: ignore[invalid-return-type]
        d=d_init,
        active_set=init_ineq_active,
        multipliers_eq=mult_init[:m_eq],
        multipliers_ineq=jnp.zeros((m_ineq,)),
        iteration=jnp.array(0),
        converged=init_converged,
        any_inner_failure=init_inner_failure,
        last_add_idx=jnp.array(-1),
        last_drop_idx=jnp.array(-1),
        ping_pong_count=jnp.array(0),
        ping_ponged=jnp.array(False),
        proj_residual=init_result.proj_residual.astype(jnp.float64),
        n_proj_refinements=init_result.n_proj_refinements,
        projected_grad_norm=init_result.projected_grad_norm.astype(jnp.float64),
    )

    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)
    effective_tau = tau if use_expand else 0.0
    drop_floor = jnp.asarray(mult_drop_floor, dtype=jnp.float64)

    def _inner_solve(state: QPState, _working_tol: Scalar) -> ActiveSetInnerResult:
        combined_active = jnp.concatenate([eq_active, state.active_set])
        result = inner_solver.solve(
            hvp_fn,
            g,
            A_combined,
            b_combined,
            combined_active,
            precond_fn=precond_fn,
            adaptive_tol=_adaptive_tol,
        )
        mult_all = result.multipliers
        return ActiveSetInnerResult(
            d=result.d,
            multipliers_eq=mult_all[:m_eq],
            multipliers_ineq=mult_all[m_eq:],
            inner_failed=~inner_ok(result),
            proj_residual=result.proj_residual.astype(jnp.float64),
            n_proj_refinements=result.n_proj_refinements,
            projected_grad_norm=result.projected_grad_norm.astype(jnp.float64),
        )

    final_state = run_active_set_loop(
        init_state=init_state,  # ty: ignore[invalid-argument-type]
        inner_solve_fn=_inner_solve,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        max_iter=max_iter,
        base_tol=base_tol,
        effective_tau=effective_tau,
        drop_floor=drop_floor,
        ping_pong_threshold=ping_pong_threshold,
    )

    reached_max_iter = final_state.iteration >= max_iter
    final_converged = final_state.converged & ~reached_max_iter
    final_working_tol = base_tol + final_state.iteration * effective_tau

    return QPSolverResult(
        d=final_state.d,
        multipliers_eq=final_state.multipliers_eq,
        multipliers_ineq=final_state.multipliers_ineq,
        active_set=final_state.active_set,
        converged=final_converged,
        iterations=final_state.iteration,
        ping_ponged=final_state.ping_ponged,
        reached_max_iter=reached_max_iter,
        final_working_tol=jnp.asarray(final_working_tol, dtype=jnp.float64),
        proj_residual=final_state.proj_residual,
        n_proj_refinements=final_state.n_proj_refinements,
        projected_grad_norm=final_state.projected_grad_norm,
    )


__all__ = ["solve_qp_direct"]
