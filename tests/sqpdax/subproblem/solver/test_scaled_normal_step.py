"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.scaled_normal_step`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.logging import DEBUG, WARNING, Logger, MemoryHandler
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.subproblem.funnel_barrier import FunnelBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, ScaledNormalStepSolver
from tests.sqpdax.conftest import make_shifted_box_quadratic
from tests.sqpdax.lagrangian.conftest import make_ip_primal
from tests.sqpdax.subproblem.conftest import (
    FUNNEL_PROBLEMS,
    dense_funnel_reference,
    make_funnel_barrier_subproblem,
    make_scaled_barrier_subproblem,
)

from .conftest import make_scaled_normal_state

problem_ids = pytest.mark.parametrize("problem_name", list(FUNNEL_PROBLEMS))
radius_ids = pytest.mark.parametrize("radius", [0.05, 100.0], ids=["small", "large"])
slack_ids = pytest.mark.parametrize("slack_fill", [1.5, 0.05], ids=["wide", "tight"])

KAPPA_FBN = 0.1


def build(
    problem_name: str, *, slack_fill: float | None = None
) -> FunnelBarrierSubProblem:
    """Funnel subproblem on ``problem_name`` with optional uniform slack values."""
    problem = FUNNEL_PROBLEMS[problem_name]()
    primal = make_ip_primal(n=problem.n, mineq=problem.mineq)
    if slack_fill is not None:
        primal = InteriorPointPrimal(
            x=primal.x,
            slack=Slack(
                s=jnp.full((problem.mineq,), slack_fill),
                s_lb=jnp.full((problem.n,), slack_fill),
                s_ub=jnp.full((problem.n,), slack_fill),
            ),
        )
    return make_funnel_barrier_subproblem(
        problem=problem, primal=primal, kappa_fbn=KAPPA_FBN
    )


def build_near_feasible(
    problem_name: str, *, shift: float = 0.2
) -> FunnelBarrierSubProblem:
    """Funnel subproblem at a point whose slacks almost close every constraint."""
    problem = FUNNEL_PROBLEMS[problem_name]()
    x = make_ip_primal(n=problem.n, mineq=problem.mineq).x
    primal = InteriorPointPrimal(
        x=x,
        slack=Slack(
            s=-problem.ineq_fn(x) + shift,
            s_lb=jnp.where(problem.null_lb, 1.0, x - problem.lb + shift),
            s_ub=jnp.where(problem.null_ub, 1.0, problem.ub - x + shift),
        ),
    )
    return make_funnel_barrier_subproblem(
        problem=problem, primal=primal, kappa_fbn=KAPPA_FBN
    )


def solve(sub: FunnelBarrierSubProblem, radius: float, **solver_kwargs):
    solver = ScaledNormalStepSolver(**solver_kwargs)
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    return solver.solve(sub, zero, make_scaled_normal_state(radius))


def range_projection_residual(A_hat: Array, w: Array) -> Array:
    """``‖w − Âᵀ(ÂÂᵀ)⁺Â w‖``: zero iff ``w ∈ range(Âᵀ)``."""
    proj = A_hat.T @ jnp.linalg.pinv(A_hat @ A_hat.T, rtol=1e-5) @ (A_hat @ w)
    return jnp.linalg.norm(w - proj)


# ---------------------------------------------------------------------------
# constraints (3.5) and the state bookkeeping
# ---------------------------------------------------------------------------


@problem_ids
@radius_ids
@slack_ids
def test_step_respects_ball_and_fraction_to_boundary(
    problem_name: str, radius: float, slack_fill: float
):
    """``‖w‖ ≤ δᵛ`` and ``w ≥ lo`` of the normal box; state fields are consistent."""
    sub = build(problem_name, slack_fill=slack_fill)
    (w, dual), state = solve(sub, radius)
    flat = w.flatten()
    lo, _ = sub.primal_box()

    assert state.success
    assert state.status == RESULTS.successful
    assert jnp.all(jnp.isfinite(flat))
    assert jnp.linalg.norm(flat) <= radius * (1.0 + 1e-5)
    assert jnp.all(flat >= lo - 1e-6)
    assert jnp.allclose(dual.flatten(), 0.0)
    assert jnp.allclose(state.step_norm, jnp.linalg.norm(flat))
    assert state.radius == radius
    assert state.n_iter == 1
    assert bool(state.on_boundary) == bool(state.step_norm >= radius * (1 - 1e-6))
    # Native-scale FTB (paper eq. 3.5): s + nˢ ≥ κ_fbn s on live slacks.
    S = sub.lagrangian.slack
    n_orig = sub._slack_to_orig_scale(w.slack)
    assert jnp.all(S.s + n_orig.s >= KAPPA_FBN * S.s - 1e-6)
    live_lb = ~sub.lagrangian.null_lb
    assert jnp.all(
        jnp.where(live_lb, S.s_lb + n_orig.s_lb - KAPPA_FBN * S.s_lb, 0.0) >= -1e-6
    )


@problem_ids
@radius_ids
@slack_ids
def test_decrease_dominates_cauchy_point(
    problem_name: str, radius: float, slack_fill: float
):
    """(3.6): ``Δm_v,n ≥ m_v(0) − m_v(w_C) > 0``; ``Δm_v,n`` matches a recomputation."""
    sub = build(problem_name, slack_fill=slack_fill)
    (w, _), state = solve(sub, radius)
    assert sub.pi_v() > 0.0
    assert state.cauchy_decrease > 0.0
    assert state.dm_v_n >= state.cauchy_decrease * (1.0 - 1e-5)
    recomputed = sub.violation() - sub.model_v(w)
    assert jnp.allclose(state.dm_v_n, recomputed, rtol=1e-4, atol=1e-6)


@problem_ids
@radius_ids
@slack_ids
def test_cauchy_decrease_satisfies_lemma_3_5(
    problem_name: str, radius: float, slack_fill: float
):
    """Lemma 3.5: ``m_v(0) − m_v(w_C) ≥ χᵛ min{πᵛ, δᵛ, 1 − κ_fbn} / (1 + ‖Â‖²)``."""
    sub = build(problem_name, slack_fill=slack_fill)
    _, state = solve(sub, radius)
    _, _, A_hat, _ = dense_funnel_reference(sub)
    kappa_cn = 1.0 / (1.0 + jnp.linalg.norm(A_hat, ord=2) ** 2)
    bound = (
        kappa_cn
        * sub.chi_v()
        * jnp.minimum(sub.pi_v(), jnp.minimum(radius, 1.0 - KAPPA_FBN))
    )
    assert state.cauchy_decrease >= bound * (1.0 - 1e-4)


@problem_ids
@radius_ids
def test_step_lies_in_range_of_scaled_jacobian_transpose(
    problem_name: str, radius: float
):
    """(3.7): the returned step belongs to ``range(Âᵀ)``."""
    sub = build(problem_name)
    (w, _), _ = solve(sub, radius)
    _, _, A_hat, _ = dense_funnel_reference(sub)
    flat = w.flatten()
    assert range_projection_residual(A_hat, flat) <= 1e-4 * (
        1.0 + jnp.linalg.norm(flat)
    )


@problem_ids
def test_unconstrained_cauchy_norm_matches_dense(problem_name: str):
    """``‖P⁻¹ n*‖ = α* πᵛ`` with ``α* = πᵛ² / ‖Â Âᵀ ĉ‖²`` (eq. 3.8)."""
    sub = build(problem_name)
    _, state = solve(sub, 1.0)
    _, _, A_hat, c = dense_funnel_reference(sub)
    d = -A_hat.T @ c
    alpha_star = jnp.dot(d, d) / jnp.dot(A_hat @ d, A_hat @ d)
    assert jnp.allclose(
        state.unconstrained_cauchy_norm, alpha_star * jnp.linalg.norm(d), rtol=1e-4
    )


# ---------------------------------------------------------------------------
# limiting regimes
# ---------------------------------------------------------------------------


@problem_ids
def test_large_radius_recovers_gauss_newton_step(problem_name: str):
    """Far from the boundary the CGLS step is the least-norm solution of ``Âw = −ĉ``."""
    sub = build_near_feasible(problem_name)
    (w, _), state = solve(sub, 1e3, tol=1e-7)
    _, _, A_hat, c = dense_funnel_reference(sub)
    w_gn = -jnp.linalg.pinv(A_hat, rtol=1e-5) @ c
    lo, _ = sub.primal_box()
    assert jnp.all(w_gn >= lo)  # the FTB box is inactive in this regime
    assert jnp.allclose(w.flatten(), w_gn, rtol=1e-3, atol=1e-4)
    assert not state.on_boundary
    assert not state.ftb_truncated
    assert jnp.allclose(sub.model_v(w), 0.0, atol=1e-3)


@problem_ids
def test_small_radius_saturates_ball(problem_name: str):
    """A tiny ``δᵛ`` returns a step on the ball with the FTB rule inactive."""
    sub = build(problem_name)
    radius = 1e-3
    (w, _), state = solve(sub, radius)
    assert state.on_boundary
    assert not state.ftb_truncated
    assert jnp.allclose(jnp.linalg.norm(w.flatten()), radius, rtol=1e-4)


def test_tight_slacks_trigger_fraction_to_boundary():
    """Strongly violated rows push the Gauss–Newton step past the FTB face."""
    problem = FUNNEL_PROBLEMS["eq-ineq-finite"]()
    primal = InteriorPointPrimal(
        x=jnp.array([4.0, 0.75]),  # h₀ = x₀ − 2 > 0 and x₀ > ub₀: rows violated
        slack=Slack(
            s=jnp.full((problem.mineq,), 2.0),
            s_lb=jnp.full((problem.n,), 2.0),
            s_ub=jnp.full((problem.n,), 2.0),
        ),
    )
    sub = make_funnel_barrier_subproblem(
        problem=problem, primal=primal, kappa_fbn=KAPPA_FBN
    )
    (w, _), state = solve(sub, 100.0)
    _, _, A_hat, c = dense_funnel_reference(sub)
    w_gn = -jnp.linalg.pinv(A_hat, rtol=1e-5) @ c
    lo, _ = sub.primal_box()
    assert jnp.any(w_gn < lo)  # the unconstrained step leaves the box
    assert jnp.all(w.flatten() >= lo - 1e-6)
    assert state.ftb_truncated or state.dm_v_n == state.cauchy_decrease
    assert state.dm_v_n >= state.cauchy_decrease * (1.0 - 1e-5)


def test_feasible_point_returns_zero_step():
    """``v = 0`` gives a zero step, zero decreases and a successful state."""
    problem, _, _ = make_shifted_box_quadratic(n=3)
    x = jnp.array([0.5, 0.0, 0.0])
    primal = InteriorPointPrimal(
        x=x,
        slack=Slack(s=-problem.ineq_fn(x), s_lb=x - problem.lb, s_ub=problem.ub - x),
    )
    sub = make_funnel_barrier_subproblem(problem=problem, primal=primal)
    (w, _), state = solve(sub, 1.0)
    assert jnp.allclose(w.flatten(), 0.0)
    assert state.success
    assert state.n_cg_iter == 0
    assert jnp.allclose(state.dm_v_n, 0.0)
    assert jnp.allclose(state.cauchy_decrease, 0.0)
    assert jnp.allclose(state.unconstrained_cauchy_norm, 0.0)
    assert not state.on_boundary


def test_rejects_non_funnel_subproblem():
    """Only a ``FunnelBarrierSubProblem`` carries the normal-step geometry."""
    sub = make_scaled_barrier_subproblem()
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    with pytest.raises(TypeError, match="FunnelBarrierSubProblem"):
        ScaledNormalStepSolver().solve(sub, zero, make_scaled_normal_state(1.0))


def test_logging_emits_debug_summary_and_no_warning_when_finite():
    """One DEBUG summary per solve with the state's numbers; no WARNING when finite."""
    handler = MemoryHandler()
    logger = Logger.from_options({"level": "DEBUG", "handler": handler})
    sub = build("eq-ineq-finite")
    solver = ScaledNormalStepSolver(logger=logger)
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    _, state = solver.solve(sub, zero, make_scaled_normal_state(0.5))

    assert len(handler.records) == 1
    (rec,) = handler.records
    assert rec.levelno == DEBUG
    assert rec.message.startswith("normal step: radius=5.000e-01")
    assert rec.values["cg_iters"] == int(state.n_cg_iter)
    assert rec.values["dm"] == pytest.approx(float(state.dm_v_n), rel=1e-6)
    assert rec.values["cauchy"] == pytest.approx(float(state.cauchy_decrease), rel=1e-6)
    assert rec.values["on_boundary"] == bool(state.on_boundary)
    assert isinstance(rec.values["fallback"], bool)
    assert not any(r.levelno >= WARNING for r in handler.records)


def test_solve_is_jittable():
    """The solver traces under ``jax.jit`` with the state as a pytree carry."""
    sub = build("eq-ineq-finite")
    solver = ScaledNormalStepSolver()
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))

    @jax.jit
    def run(state):
        return solver.solve(sub, zero, state)

    (w, _), state = run(make_scaled_normal_state(0.5))
    (w_ref, _), state_ref = solver.solve(sub, zero, make_scaled_normal_state(0.5))
    assert jnp.allclose(w.flatten(), w_ref.flatten(), atol=1e-6)
    assert jnp.allclose(state.dm_v_n, state_ref.dm_v_n, atol=1e-6)
