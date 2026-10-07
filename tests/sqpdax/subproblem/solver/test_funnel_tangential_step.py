"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.funnel_tangential_step`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.linalg import box_fraction
from slsqp_jax.sqpdax.logging import DEBUG, WARNING, Logger, MemoryHandler
from slsqp_jax.sqpdax.primal import InteriorPointPrimal
from slsqp_jax.sqpdax.subproblem.funnel_barrier import FunnelBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    FunnelTangentialStepSolver,
    KKTMultiplierRecovery,
    ScaledNormalStepSolver,
)
from tests.sqpdax.subproblem.conftest import (
    FUNNEL_PROBLEMS,
    FunnelCase,
    dense_funnel_reference,
    make_funnel_case,
    make_scaled_barrier_subproblem,
)

from .conftest import make_funnel_tangential_state, make_scaled_normal_state

problem_ids = pytest.mark.parametrize("problem_name", list(FUNNEL_PROBLEMS))
pd_ids = pytest.mark.parametrize(
    "primal_dual", [True, False], ids=["primal-dual", "primal"]
)
# Looped inside the tests so each (problem, primal_dual) compiles once.
RADII = {"tight": 0.6, "wide": 5.0}

KAPPA_FBN = 0.1
KAPPA_FBT = 0.1
NORMAL_RADIUS = 0.5
KAPPA_B = 0.9


def build(problem_name: str, primal_dual: bool = True) -> FunnelCase:
    return make_funnel_case(
        problem_name,
        primal_dual=primal_dual,
        kappa_fbn=KAPPA_FBN,
        kappa_fbt=KAPPA_FBT,
    )


def _normal_step_and_multipliers(sub, zero_normal: bool):
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    if zero_normal:
        w_n = zero[0]
    else:
        (w_n, _), _ = ScaledNormalStepSolver().solve(
            sub, zero, make_scaled_normal_state(NORMAL_RADIUS)
        )
    y = KKTMultiplierRecovery(rtol=1e-8, atol=1e-8).recover(sub, None, w_n)
    return w_n, y


def normal_step_and_multipliers(
    case: FunnelCase, *, zero_normal: bool = False
) -> tuple[InteriorPointPrimal, Dual]:
    """``(w_n, y)`` with ``w_n`` from the normal solver and ``y`` from (2.7)."""
    return case.apply(_normal_step_and_multipliers, zero_normal)


def _tangential_solve(sub, solver, warm, state):
    return solver.solve(sub, warm, state)


def solve(case: FunnelCase, w_n, y, radius, **solver_kwargs):
    """Jitted tangential solve; ``radius`` is traced, solver options are static."""
    solver = FunnelTangentialStepSolver(**solver_kwargs)
    state = make_funnel_tangential_state(jnp.asarray(radius))
    return case.apply(_tangential_solve, solver, (w_n, y), state)


def live_mask(sub: FunnelBarrierSubProblem) -> Array:
    """Boolean mask of the live scaled coordinates (null bound slacks are dead)."""
    lag = sub.lagrangian
    return jnp.concatenate(
        [
            jnp.ones((lag.n + lag.mineq,), bool),
            ~lag.null_lb,
            ~lag.null_ub,
        ]
    )


def projected_newton_step(sub: FunnelBarrierSubProblem) -> Array:
    """Dense minimiser of ``m_f(t)`` over ``null(Â)`` restricted to live coordinates."""
    g, H, A, _ = dense_funnel_reference(sub)
    live = live_mask(sub)
    A_l, H_l, g_l = A[:, live], H[live][:, live], g[live]
    _, s, vt = jnp.linalg.svd(A_l, full_matrices=True)
    rank = int(jnp.sum(s > 1e-5 * s[0]))
    Z = vt[rank:].T
    reduced = Z.T @ H_l @ Z
    assert jnp.all(jnp.linalg.eigvalsh(reduced) > 0)  # test-problem precondition
    t_live = Z @ jnp.linalg.solve(reduced, -(Z.T @ g_l))
    return jnp.zeros(A.shape[1]).at[live].set(t_live)


# ---------------------------------------------------------------------------
# constraints (3.19b)–(3.19d) / (3.23b)–(3.23c) and state bookkeeping
# ---------------------------------------------------------------------------


@problem_ids
@pd_ids
def test_step_is_tangential_inside_the_ball_and_box(
    problem_name: str, primal_dual: bool
):
    """``Â t ≈ 0``, ``‖w_n + t‖ ≤ radius``, box faces hold; state is consistent."""
    case = build(problem_name, primal_dual)
    w_n, y = normal_step_and_multipliers(case)
    for radius in RADII.values():
        check_step_is_tangential_inside_the_ball_and_box(case, w_n, y, radius)


def check_step_is_tangential_inside_the_ball_and_box(case, w_n, y, radius):
    sub = case.sub
    (t, dual), state = solve(case, w_n, y, radius)
    flat_t, flat_n = t.flatten(), w_n.flatten()
    _, _, A_hat, _ = dense_funnel_reference(sub)
    lo, _ = sub.tangential_box(w_n)

    assert state.success
    assert state.status == RESULTS.successful
    assert state.n_iter == 1
    assert jnp.allclose(dual.flatten(), 0.0)
    # (3.19d)/(3.23d) by construction: the step lives in null(Â).
    assert jnp.linalg.norm(A_hat @ flat_t) <= 1e-4 * (1.0 + jnp.linalg.norm(flat_t))
    assert jnp.allclose(state.model_v_after, sub.model_v(w_n), rtol=1e-4, atol=1e-5)
    # (3.19c)/(3.23c) and (3.19b)/(3.23b).
    assert jnp.linalg.norm(flat_n + flat_t) <= radius * (1.0 + 1e-5)
    assert jnp.all(flat_t >= lo - 1e-6)
    # Native-scale FTB: s + nˢ + tˢ ≥ κ_fbt (s + nˢ) on live slacks.
    S = sub.lagrangian.slack
    n_orig = sub._slack_to_orig_scale(w_n.slack)
    t_orig = sub._slack_to_orig_scale(t.slack)
    base = S.s + n_orig.s
    assert jnp.all(base + t_orig.s >= KAPPA_FBT * base - 1e-6)
    live_lb = ~sub.lagrangian.null_lb
    base_lb = S.s_lb + n_orig.s_lb
    assert jnp.all(
        jnp.where(live_lb, base_lb + t_orig.s_lb - KAPPA_FBT * base_lb, 0.0) >= -1e-6
    )
    # Bookkeeping.
    assert jnp.allclose(state.step_norm, jnp.linalg.norm(flat_t))
    assert jnp.allclose(state.total_norm, jnp.linalg.norm(flat_n + flat_t))
    assert state.radius == radius
    assert bool(state.on_boundary) == bool(state.total_norm >= radius * (1 - 1e-6))


@problem_ids
def test_step_stays_tangential_with_inexact_multipliers(problem_name: str):
    """The Cauchy direction is ``−proj(r̂)``: an inexact ``y`` (so that ``r̂`` has
    a ``range(Âᵀ)`` component) must not leak violation into the step."""
    case = build(problem_name)
    sub = case.sub
    w_n, y = normal_step_and_multipliers(case)
    y_flat = y.flatten()
    _, _, A_hat, _ = dense_funnel_reference(sub)
    for noise in (1e-3, 1e-1):
        noisy = Dual.from_flat(
            y_flat + noise * jax.random.normal(jax.random.key(0), y_flat.shape),
            sub.lagrangian.n,
            sub.lagrangian.mineq,
            sub.lagrangian.meq,
        )
        # Force the Cauchy fallback by allowing no CG iterations.
        (t, _), state = solve(case, w_n, noisy, 5.0, max_iter=0)
        flat_t = t.flatten()
        assert state.cauchy_decrease > 0.0
        assert state.dm_f_t == pytest.approx(float(state.cauchy_decrease), rel=1e-5)
        assert jnp.linalg.norm(A_hat @ flat_t) <= 1e-4 * (1.0 + jnp.linalg.norm(flat_t))
        assert jnp.allclose(state.model_v_after, sub.model_v(w_n), rtol=1e-4, atol=1e-5)


@problem_ids
@pd_ids
def test_decrease_dominates_cauchy_point(problem_name: str, primal_dual: bool):
    """(3.19a)/(3.23a): ``Δm_f,t ≥ m_f(w_n) − m_f(w_n + t_C) > 0`` when ``χᶠ ≥ κ_χ πᶠ > 0``."""
    case = build(problem_name, primal_dual)
    sub = case.sub
    w_n, y = normal_step_and_multipliers(case)
    pi_f, chi_f = sub.pi_f(w_n, y), sub.chi_f(w_n, y)
    assert pi_f > 0.0 and chi_f >= 0.1 * pi_f
    for radius in RADII.values():
        (t, _), state = solve(case, w_n, y, radius)
        assert state.cauchy_decrease > 0.0
        assert state.dm_f_t >= state.cauchy_decrease * (1.0 - 1e-5)
        recomputed = sub.model_f(w_n) - sub.model_f(
            InteriorPointPrimal.from_flat(
                w_n.flatten() + t.flatten(), sub.lagrangian.n, sub.lagrangian.mineq
            )
        )
        assert jnp.allclose(state.dm_f_t, recomputed, rtol=1e-4, atol=1e-6)


@problem_ids
@pd_ids
def test_cauchy_decrease_satisfies_lemma_3_9(problem_name: str, primal_dual: bool):
    """Lemma 3.9: ``m_f(n) − m_f(n + t_C) ≥ κ_ct πᶠ min{πᶠ, (1−κ_B)δᵗ, (1−κ_fbt)κ_fbn}``.

    ``κ_ct = κ_χ² / (2(1 + ‖Ĥ‖₂))`` with any ``κ_χ ∈ (0, 1)`` satisfying
    ``χᶠ ≥ κ_χ πᶠ``; the radius is chosen so that the gate (3.12)
    ``‖w_n‖ ≤ κ_B δᵗ`` holds (at the gate and ten times wider).
    """
    case = build(problem_name, primal_dual)
    sub = case.sub
    w_n, y = normal_step_and_multipliers(case)
    pi_f, chi_f = sub.pi_f(w_n, y), sub.chi_f(w_n, y)
    kappa_chi = jnp.minimum(0.99, chi_f / pi_f)
    _, H_hat, _, _ = dense_funnel_reference(sub)
    kappa_ct = kappa_chi**2 / (2.0 * (1.0 + jnp.linalg.norm(H_hat, ord=2)))
    for factor in (1.0, 10.0):
        radius = jnp.linalg.norm(w_n.flatten()) / KAPPA_B * factor
        _, state = solve(case, w_n, y, radius)
        bound = (
            kappa_ct
            * pi_f
            * jnp.minimum(
                pi_f,
                jnp.minimum((1.0 - KAPPA_B) * radius, (1.0 - KAPPA_FBT) * KAPPA_FBN),
            )
        )
        assert bound > 0.0
        assert state.cauchy_decrease >= bound * (1.0 - 1e-4)


# ---------------------------------------------------------------------------
# limiting regimes
# ---------------------------------------------------------------------------


@problem_ids
@pd_ids
def test_zero_normal_step_and_large_radius_recover_projected_newton_step(
    problem_name: str, primal_dual: bool
):
    """With ``w_n = 0`` and a huge ball the CG step is the reduced Newton step.

    When that step leaves the FTB box it is returned ray-backtracked onto it
    (``β t_N``), unless the Cauchy point does better.
    """
    case = build(problem_name, primal_dual)
    sub = case.sub
    w_n, y = normal_step_and_multipliers(case, zero_normal=True)
    t_newton = projected_newton_step(sub)
    lo, _ = sub.tangential_box(w_n)
    beta = box_fraction(t_newton, lo)
    (t, _), state = solve(case, w_n, y, 1e3, tol=1e-8)
    assert not state.on_boundary
    assert bool(state.ftb_truncated) == bool(beta < 1.0)
    if state.dm_f_t > state.cauchy_decrease:
        assert jnp.allclose(t.flatten(), beta * t_newton, rtol=1e-3, atol=1e-4)
    else:
        # Cauchy point (1-D null space): both are the exact minimiser.
        assert jnp.allclose(t.flatten(), t_newton, rtol=1e-3, atol=1e-4)


@problem_ids
def test_small_radius_saturates_ball(problem_name: str):
    """A ball barely larger than ``‖w_n‖`` returns ``‖w_n + t‖ = radius``."""
    case = build(problem_name)
    w_n, y = normal_step_and_multipliers(case)
    radius = jnp.linalg.norm(w_n.flatten()) * (1.0 + 1e-3)
    _, unconstrained = solve(case, w_n, y, 1e3)
    assert unconstrained.total_norm > radius  # the ball is genuinely active
    _, state = solve(case, w_n, y, radius)
    assert state.on_boundary
    assert jnp.allclose(state.total_norm, radius, rtol=1e-4)
    assert state.dm_f_t > 0.0


@problem_ids
def test_normal_step_outside_ball_yields_zero_step(problem_name: str):
    """No room inside the ball: zero step, zero decreases, success."""
    case = build(problem_name)
    w_n, y = normal_step_and_multipliers(case)
    radius = jnp.linalg.norm(w_n.flatten()) * 0.5
    (t, _), state = solve(case, w_n, y, radius)
    assert jnp.allclose(t.flatten(), 0.0)
    assert state.success
    assert state.n_cg_iter == 0
    assert jnp.allclose(state.dm_f_t, 0.0)
    assert jnp.allclose(state.cauchy_decrease, 0.0)
    assert jnp.allclose(state.model_v_after, case.sub.model_v(w_n))


@problem_ids
def test_normal_equations_strategies_yield_the_same_step(problem_name: str):
    """``schur``, ``matrix-free`` and ``generic`` projectors agree; ``auto``
    resolves by the number of general rows and reuses a cached factor."""
    case = build(problem_name)
    sub = case.sub
    w_n, y = normal_step_and_multipliers(case)
    results = {
        strategy: solve(case, w_n, y, 5.0, normal_equations=strategy)
        for strategy in ("generic", "schur", "matrix-free")
    }
    (t_ref, _), state_ref = results["generic"]
    for (t, _), state in results.values():
        assert jnp.allclose(t.flatten(), t_ref.flatten(), rtol=1e-4, atol=1e-4)
        assert jnp.allclose(state.dm_f_t, state_ref.dm_f_t, rtol=1e-4, atol=1e-5)
        assert jnp.allclose(state.model_v_after, state_ref.model_v_after, rtol=1e-4)
    m = sub.lagrangian.meq + sub.lagrangian.mineq
    assert FunnelTangentialStepSolver().resolve_normal_equations(sub) == "schur"
    assert (
        FunnelTangentialStepSolver(schur_max_rows=m).resolve_normal_equations(sub)
        == "matrix-free"
    )
    (t_cached, _), _ = case.apply(
        _tangential_solve_with_cached_factor,
        FunnelTangentialStepSolver(),
        (w_n, y),
        make_funnel_tangential_state(5.0),
    )
    assert jnp.allclose(t_cached.flatten(), results["schur"][0][0].flatten())


def _tangential_solve_with_cached_factor(sub, solver, warm, state):
    return solver.solve(sub.with_schur_normal_equations(), warm, state)


def test_rejects_non_funnel_subproblem():
    """Only a ``FunnelBarrierSubProblem`` carries the tangential-step geometry."""
    sub = make_scaled_barrier_subproblem()
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    with pytest.raises(TypeError, match="FunnelBarrierSubProblem"):
        FunnelTangentialStepSolver().solve(sub, zero, make_funnel_tangential_state(1.0))


def test_logging_emits_debug_summary_and_no_warning_when_finite():
    """One DEBUG summary per solve with the state's numbers; no WARNING when finite."""
    handler = MemoryHandler()
    logger = Logger.from_options({"level": "DEBUG", "handler": handler})
    case = build("shifted-box")
    w_n, y = normal_step_and_multipliers(case)
    solver = FunnelTangentialStepSolver(logger=logger)
    _, state = solver.solve(case.sub, (w_n, y), make_funnel_tangential_state(5.0))

    assert len(handler.records) == 1
    (rec,) = handler.records
    assert rec.levelno == DEBUG
    assert rec.message.startswith("tangential step: radius=5.000e+00")
    assert rec.values["cg_iters"] == int(state.n_cg_iter)
    assert rec.values["dm"] == pytest.approx(float(state.dm_f_t), rel=1e-6)
    assert rec.values["cauchy"] == pytest.approx(float(state.cauchy_decrease), rel=1e-6)
    assert rec.values["m_v"] == pytest.approx(float(state.model_v_after), rel=1e-6)
    assert rec.values["on_boundary"] == bool(state.on_boundary)
    assert isinstance(rec.values["fallback"], bool)
    assert not any(r.levelno >= WARNING for r in handler.records)


def test_solve_is_jittable():
    """The eager solve agrees with the jitted one used throughout this module."""
    case = build("shifted-box")
    w_n, y = normal_step_and_multipliers(case)
    solver = FunnelTangentialStepSolver()
    (t, _), state = solve(case, w_n, y, 5.0)
    (t_ref, _), state_ref = solver.solve(
        case.sub, (w_n, y), make_funnel_tangential_state(5.0)
    )
    assert jnp.allclose(t.flatten(), t_ref.flatten(), atol=1e-5)
    assert jnp.allclose(state.dm_f_t, state_ref.dm_f_t, atol=1e-5)
