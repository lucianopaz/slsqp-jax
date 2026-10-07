"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.trust_funnel`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.logging import DEBUG, WARNING, Logger, MemoryHandler
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem.funnel_barrier import FunnelBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    FunnelTangentialStepSolver,
    IterationType,
    KKTMultiplierRecovery,
    LinearForcing,
    MultiplierCase,
    ScaledNormalStepSolver,
    TrustFunnelSolver,
    TrustFunnelSolverState,
)
from tests.sqpdax.subproblem.conftest import (
    FUNNEL_PROBLEMS,
    dense_funnel_reference,
    make_funnel_barrier_subproblem,
    make_scaled_barrier_subproblem,
)

from .conftest import make_scaled_normal_state, make_trust_funnel_state

problem_ids = pytest.mark.parametrize("problem_name", list(FUNNEL_PROBLEMS))
pd_ids = pytest.mark.parametrize(
    "primal_dual", [True, False], ids=["primal-dual", "primal"]
)

# A negligible ω_t keeps (3.15b) from vetoing the tangential step in scenario
# tests that need one regardless of the πᶠ/πᵛ balance at the test iterate.
NEGLIGIBLE_OMEGA_T = LinearForcing(1e-6)

FEASIBLE_X = {
    "eq-ineq-finite": jnp.array([0.4, 0.6]),
    "eq-ineq-mixed-null": jnp.array([0.4, 0.6]),
    "shifted-box": jnp.array([0.5, 0.0, 0.0]),
}


def build(
    problem_name: str,
    primal_dual: bool = True,
    primal: InteriorPointPrimal | None = None,
) -> FunnelBarrierSubProblem:
    return make_funnel_barrier_subproblem(
        problem=FUNNEL_PROBLEMS[problem_name](), primal_dual=primal_dual, primal=primal
    )


def feasible_primal(problem: Problem, x: Array) -> InteriorPointPrimal:
    """Interior primal with ``c(x, s) = 0`` (dead slacks set to one)."""
    s_lb = jnp.where(problem.null_lb, 1.0, x - problem.lb)
    s_ub = jnp.where(problem.null_ub, 1.0, problem.ub - x)
    return InteriorPointPrimal(
        x=x, slack=Slack(s=-problem.ineq_fn(x), s_lb=s_lb, s_ub=s_ub)
    )


def random_primal(problem: Problem, seed: int) -> InteriorPointPrimal:
    kx, ks = jax.random.split(jax.random.key(seed))
    n, mineq = problem.n, problem.mineq
    x = jax.random.normal(kx, (n,))
    s = jax.random.uniform(ks, (mineq + 2 * n,), minval=0.2, maxval=2.0)
    return InteriorPointPrimal(
        x=x, slack=Slack(s=s[:mineq], s_lb=s[mineq : mineq + n], s_ub=s[mineq + n :])
    )


def solve(sub, state, solver: TrustFunnelSolver | None = None):
    solver = TrustFunnelSolver() if solver is None else solver
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    return solver.solve(sub, zero, state)


def to_scaled(sub: FunnelBarrierSubProblem, d: InteriorPointPrimal) -> Array:
    """``w = P⁻¹ d`` for a native step ``d``."""
    w = InteriorPointPrimal(x=d.x, slack=sub._slack_to_ball_scale(d.slack))
    return w.flatten()


def unconstrained_cauchy_norm(sub: FunnelBarrierSubProblem) -> float:
    """``‖P⁻¹ n*‖ = α* πᵛ`` (3.8) from the dense reference."""
    _, _, A, c = dense_funnel_reference(sub)
    d = A.T @ c
    pi_v_sq = float(d @ d)
    curv = float((A @ d) @ (A @ d))
    return pi_v_sq / curv * jnp.sqrt(pi_v_sq)


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #


def test_cold_state_defaults():
    st = TrustFunnelSolverState.cold(0.5, 2.0, 3.0, eps_pi=1e-3, eps_v=1e-4)
    assert float(st.radius_v) == 0.5
    assert float(st.radius_f) == 2.0
    assert float(st.v_max) == 3.0
    assert float(st.eps_pi) == pytest.approx(1e-3)
    assert float(st.eps_v) == pytest.approx(1e-4)
    assert not bool(st.sf_flag)
    assert float(st.pi_f_prev) == 0.0
    assert int(st.n_iter) == 0
    assert st.status == RESULTS.successful


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa_B": 1.0},
        {"kappa_vf": 0.0},
        {"kappa_vv": 1.0},
        {"kappa_tt": 0.4, "kappa_vv": 0.5},
        {"kappa_tg": 1.0},
        {"kappa_cd": 0.95, "kappa_tg": 0.1},
        {"kappa_delta": 0.0},
        {"kappa_tn": 0.0},
        {"kappa_v": 0.0},
        {"kappa_n": -1.0},
        {"kappa_chi": 1.0},
        {"kappa_omega": 1.5},
        {"zero_normal_tangential": "bogus"},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_check_init_rejects_invalid_constants(kwargs):
    with pytest.raises(ValueError):
        TrustFunnelSolver(**kwargs)


def test_rejects_non_funnel_subproblem():
    sub = make_scaled_barrier_subproblem()
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    with pytest.raises(TypeError, match="FunnelBarrierSubProblem"):
        TrustFunnelSolver().solve(sub, zero, make_trust_funnel_state())


# --------------------------------------------------------------------------- #
# Scenario tests on constructed states
# --------------------------------------------------------------------------- #


@problem_ids
def test_feasible_iterate_skips_normal_step(problem_name):
    problem = FUNNEL_PROBLEMS[problem_name]()
    sub = build(problem_name, primal=feasible_primal(problem, FEASIBLE_X[problem_name]))
    assert float(sub.violation()) == pytest.approx(0.0, abs=1e-6)

    (d, _), st = solve(sub, make_trust_funnel_state(1.0, 1.0, v_max=1.0))

    assert not bool(st.normal_computed)
    assert float(st.normal_norm) == 0.0
    assert float(st.dm_v_n) == 0.0
    assert float(st.dm_f_n) == 0.0
    # A tangential step is taken in the null space, so the step is feasible
    # for the linearised constraints and k ∈ D.
    assert bool(st.tangential_computed)
    assert bool(st.in_d)
    _, _, A, _ = dense_funnel_reference(sub)
    assert jnp.allclose(A @ to_scaled(sub, d), 0.0, atol=1e-4)
    assert st.iteration_type == IterationType.f_iteration


@problem_ids
@pd_ids
def test_kkt_tolerances_met_returns_zero_step(problem_name, primal_dual):
    sub = build(problem_name, primal_dual)
    (d, y), st = solve(sub, make_trust_funnel_state(1.0, 1.0, eps_pi=1e3, eps_v=1e3))

    assert bool(st.kkt_satisfied)
    assert st.multiplier_case == MultiplierCase.terminate
    assert st.iteration_type == IterationType.y_iteration
    assert jnp.all(d.flatten() == 0.0)
    assert not bool(st.normal_computed)
    assert not bool(st.tangential_computed)
    assert float(st.normal_norm) == 0.0
    assert float(st.tangential_norm) == 0.0
    for name in ("dm_f_n", "dm_f_t", "dm_v_n", "dm_v_d"):
        assert float(getattr(st, name)) == 0.0
    assert bool(st.in_d)
    assert bool(st.success)
    assert jnp.all(jnp.isfinite(y.flatten()))


@problem_ids
@pd_ids
def test_gate_312_failure_keeps_previous_multipliers(problem_name, primal_dual):
    """``‖P⁻¹n‖ = δᵛ > κ_B min{κ_vf δᵛ, δᶠ}`` ⇒ ``y = y_{k-1}`` and ``t = 0``."""
    sub = build(problem_name, primal_dual)
    solver = TrustFunnelSolver()
    radius_v = 1e-3
    radius_f = 0.5 * radius_v
    (d, y), st = solve(sub, make_trust_funnel_state(radius_v, radius_f), solver)

    assert bool(st.normal_computed)
    assert float(st.normal_norm) == pytest.approx(radius_v, rel=1e-4)
    assert float(st.normal_norm) > solver.kappa_B * min(
        solver.kappa_vf * radius_v, radius_f
    )
    assert not bool(st.tangential_computed)
    assert float(st.tangential_norm) == 0.0
    assert float(st.dm_f_t) == 0.0
    assert jax.tree.all(jax.tree.map(jnp.array_equal, y, sub.lagrangian.dual))
    # πᶠ and χᶠ are still evaluated (Step 34) with the carried multipliers.
    w_n = InteriorPointPrimal.from_flat(
        to_scaled(sub, d), sub.lagrangian.n, sub.lagrangian.mineq
    )
    assert float(st.pi_f) == pytest.approx(float(sub.pi_f(w_n, y)), rel=1e-4)
    assert float(st.chi_f) == pytest.approx(float(sub.chi_f(w_n, y)), rel=1e-4)
    assert st.iteration_type == IterationType.v_iteration
    assert float(st.pi_f_prev) == float(st.pi_f)


@problem_ids
@pytest.mark.parametrize("sf_flag", [True, False], ids=["sf-set", "sf-clear"])
def test_radius_reset_331_fires_only_with_sf_flag(problem_name, sf_flag):
    sub = build(problem_name)
    solver = TrustFunnelSolver(kappa_n=3.0)
    radius_v = 1e-3
    (_, _), st = solve(
        sub, make_trust_funnel_state(radius_v, 1.0, sf_flag=sf_flag), solver
    )

    assert bool(st.normal_computed)
    expected = max(radius_v, solver.kappa_n * unconstrained_cauchy_norm(sub))
    if sf_flag:
        assert float(st.radius_v) == pytest.approx(expected, rel=1e-4)
        assert float(st.radius_v) > radius_v
    else:
        assert float(st.radius_v) == pytest.approx(radius_v, rel=1e-6)
    # The flag is consumed by the first normal step after it was raised.
    assert not bool(st.sf_flag)


def test_sf_flag_survives_when_no_normal_step_is_taken():
    problem = FUNNEL_PROBLEMS["shifted-box"]()
    sub = build(
        "shifted-box", primal=feasible_primal(problem, FEASIBLE_X["shifted-box"])
    )
    (_, _), st = solve(sub, make_trust_funnel_state(1e-3, 1.0, sf_flag=True))
    assert not bool(st.normal_computed)
    assert bool(st.sf_flag)
    assert float(st.radius_v) == pytest.approx(1e-3, rel=1e-6)


@problem_ids
@pytest.mark.parametrize("policy", ["very_relaxed", "relaxed"])
def test_zero_normal_step_policy(problem_name, policy):
    """With ``n = 0`` at an infeasible iterate (3.2 fails) the tangential
    radius follows the configured policy (3.21) or (3.17)."""
    sub = build(problem_name)
    solver = TrustFunnelSolver(
        zero_normal_tangential=policy, omega_t=NEGLIGIBLE_OMEGA_T
    )
    v = float(sub.violation())
    radius_v, radius_f, v_max = 0.4, 1.0, 1e3
    # πᵛ ≤ ω_n(πᶠ_{k-1}) with a huge carried πᶠ and v < κ_vv v_max ⇒ no normal step.
    state = make_trust_funnel_state(radius_v, radius_f, v_max=v_max, pi_f_prev=1e6)
    (d, _), st = solve(sub, state, solver)

    assert v > 0.0
    assert not bool(st.normal_computed)
    assert float(st.normal_norm) == 0.0
    assert bool(st.tangential_computed)
    radius_td = min(solver.kappa_vf * radius_v, radius_f)
    if policy == "very_relaxed":
        assert float(st.radius_t) == pytest.approx(
            min(radius_td, solver.kappa_v * v_max)
        )
        assert not bool(st.in_td)
    else:
        assert float(st.radius_t) == pytest.approx(radius_td)
        assert bool(st.in_td)
    assert bool(st.in_d)
    assert jnp.linalg.norm(to_scaled(sub, d)) <= float(st.radius_t) * (1 + 1e-4)
    # (3.23d) / (3.19d) hold for the projected step: m_v(t) = m_v(0) = v.
    assert float(st.dm_v_d) == pytest.approx(0.0, abs=1e-4)


def test_very_relaxed_radius_binds_on_kappa_v_vmax():
    """At a feasible iterate ``n = 0`` and (3.21) caps the radius by ``κ_v v_max``."""
    problem = FUNNEL_PROBLEMS["shifted-box"]()
    sub = build(
        "shifted-box", primal=feasible_primal(problem, FEASIBLE_X["shifted-box"])
    )
    solver = TrustFunnelSolver(kappa_v=2.0)
    v_max = 1e-2
    (d, _), st = solve(sub, make_trust_funnel_state(10.0, 10.0, v_max=v_max), solver)
    assert not bool(st.normal_computed)
    assert bool(st.tangential_computed)
    assert float(st.radius_t) == pytest.approx(solver.kappa_v * v_max)
    assert float(jnp.linalg.norm(to_scaled(sub, d))) <= float(st.radius_t) * (1 + 1e-4)


# --------------------------------------------------------------------------- #
# Lemma 3.3 properties and Δm consistency over a randomised batch
# --------------------------------------------------------------------------- #


@problem_ids
@pd_ids
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "radii", [(0.05, 0.1), (0.5, 1.0), (5.0, 5.0)], ids=["tiny", "medium", "wide"]
)
def test_lemma_3_3_properties(problem_name, primal_dual, seed, radii):
    problem = FUNNEL_PROBLEMS[problem_name]()
    sub = build(problem_name, primal_dual, primal=random_primal(problem, seed))
    solver = TrustFunnelSolver()
    radius_v, radius_f = radii
    v = float(sub.violation())
    state = make_trust_funnel_state(radius_v, radius_f, v_max=max(1.0, 2.0 * v))
    (d, y), st = solve(sub, state, solver)

    assert bool(st.success)
    assert st.status == RESULTS.successful
    assert jnp.all(jnp.isfinite(d.flatten()))
    assert jnp.all(jnp.isfinite(y.flatten()))
    w = to_scaled(sub, d)
    w_primal = InteriorPointPrimal.from_flat(w, sub.lagrangian.n, sub.lagrangian.mineq)
    tol = 1e-4 * (1.0 + abs(v))

    # Carried quantities.
    assert float(st.violation) == pytest.approx(v)
    assert float(st.pi_f_prev) == float(st.pi_f)
    assert float(st.radius_v) == pytest.approx(radius_v, rel=1e-6)  # no (3.31) reset
    assert not bool(st.sf_flag)
    assert not bool(st.infeasible_stationary)

    # (i) ‖P⁻¹n‖ ≤ δᵛ; (ii) ‖P⁻¹(n + t)‖ ≤ δᵗ.
    assert float(st.normal_norm) <= radius_v * (1 + 1e-4)
    if bool(st.tangential_computed):
        assert float(jnp.linalg.norm(w)) <= float(st.radius_t) * (1 + 1e-4)
        assert float(st.radius_t) <= min(solver.kappa_vf * radius_v, radius_f) + 1e-6
        # (3.12) held and the multipliers are the LS estimate, not y_{k-1}.
        assert float(st.normal_norm) <= solver.kappa_B * float(st.radius_t) + 1e-6
        # (iii) χᶠ ≥ κ_χ πᶠ whenever a tangential step is computed.
        assert float(st.chi_f) >= solver.kappa_chi * float(st.pi_f) * (1 - 1e-4)
        assert st.multiplier_case == MultiplierCase.tangential
    else:
        assert float(st.tangential_norm) == 0.0

    # (iv) t ≠ 0 ⇒ Δm_f,t > 0; (v) the Cauchy bound is matched or exceeded.
    if float(st.tangential_norm) > 0.0:
        assert float(st.dm_f_t) > 0.0
        assert float(st.cauchy_ratio_f) >= 1.0 - 1e-4
    else:
        assert float(st.dm_f_t) == 0.0
    if bool(st.normal_computed):
        assert float(st.dm_v_n) > 0.0
        assert float(st.cauchy_ratio_v) >= 1.0 - 1e-4
    else:
        assert float(st.normal_norm) == 0.0
        assert float(st.dm_v_n) == 0.0

    # (vi) y-iteration ⇔ d = 0.
    is_y = st.iteration_type == IterationType.y_iteration
    assert bool(is_y) == bool(jnp.all(w == 0.0))
    # f-iteration ⇒ t ≠ 0 and (2.10).
    if st.iteration_type == IterationType.f_iteration:
        assert float(st.tangential_norm) > 0.0
        assert bool(st.objective_decrease_ok)

    # (ix) k ∈ D ⇔ (3.19d) for the final step; (x) T_D ⊆ D; (xi) T₀ ⊆ T_D.
    m_v_d = float(sub.model_v(w_primal))
    m_v_n = v - float(st.dm_v_n)
    in_d_expected = m_v_d <= solver.kappa_tg * v + (1 - solver.kappa_tg) * m_v_n + tol
    assert bool(st.in_d) == in_d_expected
    if bool(st.in_td):
        assert bool(st.in_d)
        assert bool(st.tangential_computed)
    if bool(st.tangential_reset):
        assert bool(st.in_td)
        assert float(st.tangential_norm) == 0.0
    # Tangential step and D-membership ⇒ the contraction (2.15) condition.
    if bool(st.in_d):
        assert bool(st.contraction_ok)
        assert float(st.dm_v_d) >= solver.kappa_cd * float(st.dm_v_n) - tol

    # Δm bookkeeping matches a recomputation from the subproblem.
    assert float(st.dm_v_d) == pytest.approx(v - m_v_d, abs=tol)
    assert float(st.dm_f_n) + float(st.dm_f_t) == pytest.approx(
        -float(sub.model_f(w_primal)), abs=1e-3 * (1.0 + abs(float(st.dm_f_n)))
    )
    expected_decrease = float(st.dm_f_n) + float(
        st.dm_f_t
    ) >= solver.kappa_delta * float(st.dm_f_t)
    assert bool(st.objective_decrease_ok) == expected_decrease
    assert int(st.n_iter) == 1
    assert int(st.n_cg_iter) >= (1 if bool(st.normal_computed) else 0)


@problem_ids
def test_always_normal_forces_normal_step(problem_name):
    sub = build(problem_name)
    state = make_trust_funnel_state(0.5, 1.0, v_max=1e3, pi_f_prev=1e6)
    (_, st_default) = solve(sub, state)
    (_, st_forced) = solve(sub, state, TrustFunnelSolver(always_normal=True))
    assert not bool(st_default.normal_computed)
    assert bool(st_forced.normal_computed)
    assert float(st_forced.normal_norm) > 0.0


# --------------------------------------------------------------------------- #
# (3.20) ∧ ¬(2.10) reset with crafted sub-solvers
# --------------------------------------------------------------------------- #


def _null_space_component(A: Array, r: Array) -> Array:
    """Projection of ``r`` onto ``null(A)`` (trace-friendly, no rank needed)."""
    return r - jnp.linalg.pinv(A) @ (A @ r)


class _UphillNormalSolver(ScaledNormalStepSolver):
    """Test double: a tiny ``m_v`` descent plus a ``+ĝ`` null-space component.

    The step reduces ``m_v`` (so ``k ∈ D`` stays reachable) but increases the
    barrier model, ``Δm_f,n < 0``, which is what the (3.20) ∧ ¬(2.10) reset
    needs to fire.
    """

    norm: float = eqx.field(static=True, default=0.05)

    def solve(self, subproblem, x0, initial_state):
        g, _, A, c = dense_funnel_reference(subproblem)
        descent = -(A.T @ c)
        uphill = _null_space_component(A, g)
        w = 1e-3 * descent / jnp.linalg.norm(descent)
        w = w + self.norm * uphill / jnp.linalg.norm(uphill)
        w_n = InteriorPointPrimal.from_flat(
            w, subproblem.lagrangian.n, subproblem.lagrangian.mineq
        )
        dm_v_n = subproblem.violation() - subproblem.model_v(w_n)
        state = eqx.tree_at(
            lambda s: (s.success, s.dm_v_n, s.cauchy_decrease, s.step_norm),
            initial_state,
            (jnp.asarray(True), dm_v_n, dm_v_n, jnp.linalg.norm(w)),
        )
        return (w_n, subproblem._zero_dual()), state


class _LargeNullSpaceTangentialSolver(FunnelTangentialStepSolver):
    """Tangential step of fixed norm in ``null(Â)`` with a tiny reported decrease."""

    norm: float = eqx.field(static=True, default=1.0)
    reported_decrease: float = eqx.field(static=True, default=1e-3)

    def solve(self, subproblem, x0, initial_state):
        w_n, _ = x0
        _, _, A, _ = dense_funnel_reference(subproblem)
        live = jnp.any(A != 0.0, axis=0)
        r = jnp.where(live, jnp.arange(1.0, A.shape[1] + 1.0), 0.0)
        direction = _null_space_component(A, r)
        t = self.norm * direction / jnp.linalg.norm(direction)
        t_p = InteriorPointPrimal.from_flat(
            t, subproblem.lagrangian.n, subproblem.lagrangian.mineq
        )
        total = w_n.flatten() + t
        state = eqx.tree_at(
            lambda s: (
                s.success,
                s.dm_f_t,
                s.cauchy_decrease,
                s.step_norm,
                s.total_norm,
                s.model_v_after,
            ),
            initial_state,
            (
                jnp.asarray(True),
                jnp.asarray(self.reported_decrease),
                jnp.asarray(self.reported_decrease),
                jnp.asarray(self.norm),
                jnp.linalg.norm(total),
                subproblem.model_v(
                    InteriorPointPrimal.from_flat(
                        total, subproblem.lagrangian.n, subproblem.lagrangian.mineq
                    )
                ),
            ),
        )
        return (t_p, subproblem._zero_dual()), state


@problem_ids
def test_t0_reset_discards_large_tangential_step_without_decrease(problem_name):
    sub = build(problem_name)
    solver = TrustFunnelSolver(
        normal_solver=_UphillNormalSolver(),
        tangential_solver=_LargeNullSpaceTangentialSolver(),
        omega_t=NEGLIGIBLE_OMEGA_T,
    )
    (d, _), st = solve(sub, make_trust_funnel_state(1.0, 1.0, v_max=1e3), solver)

    # Preconditions of the scenario: both steps were computed, (3.20) holds
    # and (2.10) fails because the normal step increased the model.
    assert bool(st.normal_computed)
    assert bool(st.tangential_computed)
    assert float(st.dm_f_n) < -(1 - solver.kappa_delta) * 1e-3
    assert bool(st.in_td)

    assert bool(st.tangential_reset)
    assert float(st.tangential_norm) == 0.0
    assert float(st.dm_f_t) == 0.0
    assert st.iteration_type == IterationType.v_iteration
    assert float(jnp.linalg.norm(to_scaled(sub, d))) == pytest.approx(
        float(st.normal_norm), rel=1e-4
    )
    assert bool(st.in_d)


@problem_ids
def test_small_tangential_step_is_kept_despite_no_decrease(problem_name):
    """(3.20) fails when ``‖t‖ ≤ κ_tn ‖n‖`` so the step is kept (k ∉ T₀)."""
    sub = build(problem_name)
    solver = TrustFunnelSolver(
        normal_solver=_UphillNormalSolver(norm=0.5),
        tangential_solver=_LargeNullSpaceTangentialSolver(norm=0.1),
        omega_t=NEGLIGIBLE_OMEGA_T,
    )
    (_, _), st = solve(sub, make_trust_funnel_state(1.0, 1.0, v_max=1e3), solver)
    assert bool(st.tangential_computed)
    assert not bool(st.objective_decrease_ok)
    assert not bool(st.tangential_reset)
    assert float(st.tangential_norm) == pytest.approx(0.1, rel=1e-4)
    assert st.iteration_type == IterationType.v_iteration


class _FunnelBreakingTangentialSolver(_LargeNullSpaceTangentialSolver):
    """Reports a ``m_v(n + t)`` that violates (3.19d)."""

    def solve(self, subproblem, x0, initial_state):
        step, state = super().solve(subproblem, x0, initial_state)
        return step, eqx.tree_at(
            lambda s: s.model_v_after, state, 10.0 * subproblem.violation() + 1.0
        )


def test_tangential_step_violating_319d_is_discarded():
    sub = build("shifted-box")
    handler = MemoryHandler()
    solver = TrustFunnelSolver(
        tangential_solver=_FunnelBreakingTangentialSolver(),
        omega_t=NEGLIGIBLE_OMEGA_T,
        logger=Logger.from_options({"level": "WARNING", "handler": handler}),
    )
    (d, _), st = solve(sub, make_trust_funnel_state(0.5, 5.0, v_max=1e3), solver)
    assert bool(st.tangential_computed)
    assert bool(st.tangential_rejected)
    assert float(st.tangential_norm) == 0.0
    assert not bool(st.in_td)
    assert float(st.dm_v_d) == pytest.approx(float(st.dm_v_n))
    assert float(jnp.linalg.norm(to_scaled(sub, d))) == pytest.approx(
        float(st.normal_norm), rel=1e-4
    )
    assert any("discarded" in r.message for r in handler.records)


# --------------------------------------------------------------------------- #
# Logging and jit
# --------------------------------------------------------------------------- #


def test_logging_emits_debug_and_no_warning_on_clean_step():
    sub = build("shifted-box")
    handler = MemoryHandler()
    solver = TrustFunnelSolver(
        logger=Logger.from_options({"level": "DEBUG", "handler": handler})
    )
    _, st = solve(sub, make_trust_funnel_state(5.0, 5.0, v_max=10.0), solver)
    debug = [r for r in handler.records if r.levelno == DEBUG]
    warnings = [r for r in handler.records if r.levelno >= WARNING]
    assert len(debug) == 1
    (rec,) = debug
    assert rec.message.startswith("funnel step:")
    assert rec.values["cg_iters"] == int(st.n_cg_iter)
    assert rec.values["dm_v_d"] == pytest.approx(float(st.dm_v_d), rel=1e-6)
    assert rec.values["normal"] == bool(st.normal_computed)
    assert rec.values["tangential"] == bool(st.tangential_computed)
    assert warnings == []


@problem_ids
def test_solve_is_jittable_and_matches_eager(problem_name):
    sub = build(problem_name)
    solver = TrustFunnelSolver()
    state = make_trust_funnel_state(0.5, 1.0, v_max=10.0)
    zero = jax.tree.map(jnp.zeros_like, (sub.lagrangian.ref, sub.lagrangian.dual))
    (d_e, y_e), st_e = solver.solve(sub, zero, state)
    (d_j, y_j), st_j = jax.jit(lambda s: solver.solve(sub, zero, s))(state)
    assert jnp.allclose(d_e.flatten(), d_j.flatten(), atol=1e-5)
    assert jnp.allclose(y_e.flatten(), y_j.flatten(), atol=1e-4)
    assert st_e.iteration_type == st_j.iteration_type
    assert float(st_e.dm_v_d) == pytest.approx(float(st_j.dm_v_d), abs=1e-5)


@problem_ids
def test_normal_equations_strategies_yield_the_same_funnel_step(problem_name):
    """The bound-eliminated solves reproduce the generic step and multipliers,
    and the attached factor is reported through the resolved strategy."""
    sub = build(problem_name)
    state = make_trust_funnel_state(0.5, 1.0, v_max=10.0)
    results = {}
    for strategy in ("generic", "schur", "matrix-free"):
        solver = TrustFunnelSolver(
            tangential_solver=FunnelTangentialStepSolver(normal_equations=strategy),
            multiplier_recovery=KKTMultiplierRecovery(normal_equations=strategy),
        )
        attached, resolved, rank = solver._attach_normal_equations(sub)
        assert resolved == strategy
        assert (attached.schur_cache is not None) == (strategy == "schur")
        assert int(rank) == (
            int(attached.schur_cache.rank) if strategy == "schur" else -1
        )
        results[strategy] = solve(sub, state, solver)
    (d_ref, y_ref), st_ref = results["generic"]
    for (d, y), st in results.values():
        assert jnp.allclose(d.flatten(), d_ref.flatten(), rtol=1e-4, atol=1e-5)
        assert jnp.allclose(y.flatten(), y_ref.flatten(), rtol=1e-4, atol=1e-4)
        assert st.iteration_type == st_ref.iteration_type
        assert float(st.dm_v_d) == pytest.approx(float(st_ref.dm_v_d), abs=1e-5)


def test_returned_multipliers_are_least_squares_estimate():
    """When (3.12) holds the dual is the (unsafeguarded) LS estimate (2.7)."""
    sub = build("eq-ineq-finite")
    solver = TrustFunnelSolver()
    (_, y), st = solve(sub, make_trust_funnel_state(5.0, 5.0, v_max=10.0), solver)
    assert float(st.normal_norm) <= solver.kappa_B * float(st.radius_t)
    g, H, A, _ = dense_funnel_reference(sub)
    # w_n is not observable from the returned step; recompute it with the
    # same normal solver and radius the orchestrator used.
    (w_n, _), _ = solver.normal_solver.solve(
        sub, (sub._zero_primal(), sub._zero_dual()), make_scaled_normal_state(5.0)
    )
    y_ls, *_ = jnp.linalg.lstsq(A.T, -(g + H @ w_n.flatten()), rcond=None)
    assert jnp.allclose(y.flatten(), y_ls, atol=1e-3)
    assert isinstance(y, Dual)
