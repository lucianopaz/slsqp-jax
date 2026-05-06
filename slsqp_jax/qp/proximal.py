"""Proximal stabilized SQP (sSQP) QP strategy.

Equality constraints are absorbed into the objective via an
augmented-Lagrangian penalty with adaptive parameter ``mu``; see
Hager (1999) and Wright (2002, eq 6.6) for the formulation.  The
active-set loop operates on inequality constraints only.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.krylov import solve_unconstrained_cg
from slsqp_jax.qp._inner_check import inner_ok
from slsqp_jax.qp.active_set import ActiveSetInnerResult, run_active_set_loop
from slsqp_jax.state import QPSolverResult, QPState
from slsqp_jax.types import Scalar, Vector


def solve_qp_proximal(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    m_eq: int,
    m_ineq: int,
    max_iter: int,
    max_cg_iter: int,
    tol: Scalar | float,
    expand_factor: float,
    initial_active_set: Bool[Array, " m_ineq"] | None,
    kkt_residual: Scalar | float,
    proximal_mu: Scalar | float,
    prev_multipliers_eq: Float[Array, " m_eq"] | None,
    inner_solver: AbstractInnerSolver,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
    predicted_active_set: Bool[Array, " m_ineq"] | None = None,
    use_expand: bool = True,
    mult_drop_floor: float = 1e-6,
    ping_pong_threshold: int = 2**31 - 1,
) -> QPSolverResult:
    """Solve the QP via the stabilized SQP (sSQP) formulation.

    Equality constraints are absorbed into the objective via an
    augmented-Lagrangian penalty with weight ``1/mu``; the active-set
    loop operates on inequality constraints only.

    The stabilized objective is::

        (1/2) d^T B_tilde d + g_tilde^T d

    where ``B_tilde(v) = H v + (1/mu) A_eq^T (A_eq v)`` and
    ``g_tilde = g - (1/mu) A_eq^T b_eq - A_eq^T lambda_k``.

    Equality multipliers are recovered from the penalty optimality
    condition: ``lambda = lambda_k - (1/mu)(A_eq d - b_eq)``.
    """
    inv_mu = 1.0 / jnp.maximum(proximal_mu, 1e-10)
    prev_mult_eq = (
        prev_multipliers_eq if prev_multipliers_eq is not None else jnp.zeros((m_eq,))
    )
    inner_cg_tol: Scalar | float = cg_tol if cg_tol is not None else tol

    def stabilized_hvp(v: Vector) -> Vector:
        return hvp_fn(v) + inv_mu * (A_eq.T @ (A_eq @ v))

    g_mod = g - inv_mu * (A_eq.T @ b_eq) - A_eq.T @ prev_mult_eq

    def _recover_mult_eq(d: Vector) -> Float[Array, " m_eq"]:
        return prev_mult_eq - inv_mu * (A_eq @ d - b_eq)

    # Sub-case: no inequality constraints — just unconstrained CG.
    if m_ineq == 0:
        d, _cg_converged = solve_unconstrained_cg(
            stabilized_hvp,
            g_mod,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )
        finite_d = jnp.isfinite(d).all()
        return QPSolverResult(
            d=d,
            multipliers_eq=_recover_mult_eq(d),
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=finite_d,
            iterations=jnp.array(1),
            ping_ponged=jnp.array(False),
            reached_max_iter=jnp.array(False),
            final_working_tol=jnp.asarray(0.0, dtype=jnp.float64),
            proj_residual=jnp.asarray(0.0, dtype=jnp.float64),
            n_proj_refinements=jnp.asarray(0),
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=jnp.float64),
        )

    # Sub-case: inequalities present — active-set loop on A_ineq only.
    kkt_res = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_res, 1.0) * tol
    _adaptive_tol: Scalar | float | None = cg_tol

    # Initial unconstrained solve (equalities absorbed into objective).
    d_init, _ = solve_unconstrained_cg(
        stabilized_hvp,
        g_mod,
        max_cg_iter,
        inner_cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
    )
    init_inner_failure = ~jnp.isfinite(d_init).all()

    # Determine starting active set
    residuals_init = A_ineq @ d_init - b_ineq
    if predicted_active_set is not None:
        init_active = predicted_active_set | (residuals_init < -base_tol)
    elif initial_active_set is not None:
        init_active = initial_active_set | (residuals_init < -base_tol)
    else:
        init_active = residuals_init < -base_tol
    init_converged = ~jnp.any(init_active)

    init_state = QPState(  # ty: ignore[invalid-return-type]
        d=d_init,
        active_set=init_active,
        multipliers_eq=_recover_mult_eq(d_init),
        multipliers_ineq=jnp.zeros((m_ineq,)),
        iteration=jnp.array(0),
        converged=init_converged,
        any_inner_failure=init_inner_failure,
        last_add_idx=jnp.array(-1),
        last_drop_idx=jnp.array(-1),
        ping_pong_count=jnp.array(0),
        ping_ponged=jnp.array(False),
        proj_residual=jnp.asarray(0.0, dtype=jnp.float64),
        n_proj_refinements=jnp.asarray(0),
        projected_grad_norm=jnp.asarray(jnp.inf, dtype=jnp.float64),
    )

    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)
    effective_tau = tau if use_expand else 0.0
    drop_floor = jnp.asarray(mult_drop_floor, dtype=jnp.float64)

    def _inner_solve(state: QPState, _working_tol: Scalar) -> ActiveSetInnerResult:
        result = inner_solver.solve(
            stabilized_hvp,
            g_mod,
            A_ineq,
            b_ineq,
            state.active_set,
            precond_fn=precond_fn,
            adaptive_tol=_adaptive_tol,
        )
        return ActiveSetInnerResult(
            d=result.d,
            multipliers_eq=_recover_mult_eq(result.d),
            multipliers_ineq=result.multipliers,
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


__all__ = ["solve_qp_proximal"]
