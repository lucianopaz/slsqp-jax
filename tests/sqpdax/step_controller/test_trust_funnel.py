"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.trust_funnel`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.logging import INFO, WARNING, Logger, MemoryHandler
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.step_controller import TrustFunnelManager
from slsqp_jax.sqpdax.subproblem.solver import IterationType
from tests.sqpdax.conftest import make_shifted_box_quadratic

from .conftest import make_funnel_manager, make_funnel_state, make_tr_state

PROBLEM, _, _ = make_shifted_box_quadratic(n=3)
MU = 0.5

# Infeasible interior iterate: c(x, s) ≠ 0 but every slack is positive.
X0 = InteriorPointPrimal(
    x=jnp.array([0.5, 0.0, 0.0]),
    slack=Slack(s=jnp.array([1.0]), s_lb=jnp.ones(3), s_ub=jnp.ones(3)),
)
# Feasible target with the same x: closes every residual, so moving a fraction
# λ of the way reduces v exactly by the factor (1 − λ) (c is affine here).
X_FEAS = InteriorPointPrimal(
    x=X0.x,
    slack=Slack(
        s=-PROBLEM.ineq_fn(X0.x), s_lb=X0.x - PROBLEM.lb, s_ub=PROBLEM.ub - X0.x
    ),
)
C_OBJ = jnp.array([0.95, -2.0, 0.0])  # unconstrained minimiser of the objective

# Target ρ values covering every branch of the update rules: ≥ η2, [η1, η2),
# [0, η1) and negative.
RHO_CASES = pytest.mark.parametrize(
    "rho_target", [0.9, 0.5, 0.05, -0.5], ids=["grow", "keep", "shrink", "shrink-neg"]
)


def scale(tree, factor):
    return jax.tree.map(lambda a: factor * a, tree)


def diff(a, b):
    return jax.tree.map(jnp.subtract, a, b)


def f_direction(sign: float = 1.0) -> InteriorPointPrimal:
    """x-only step towards (``sign > 0``) or away from the objective minimiser."""
    dx = 0.3 * sign * (C_OBJ - X0.x)
    return InteriorPointPrimal(x=dx, slack=jax.tree.map(jnp.zeros_like, X0.slack))


def v_direction(sign: float = 1.0) -> InteriorPointPrimal:
    """Slack-only step towards (``sign > 0``) or away from feasibility."""
    return scale(diff(X_FEAS, X0), 0.5 * sign)


def actual_decreases(manager: TrustFunnelManager, direction):
    x_trial = jax.tree.map(jnp.add, X0, direction)
    df = manager.barrier_merit(X0) - manager.barrier_merit(x_trial)
    dv = manager.violation(X0) - manager.violation(x_trial)
    return float(df), float(dv), x_trial


def expected_radius(radius: float, rho: float, manager: TrustFunnelManager) -> float:
    if rho >= manager.eta2:
        return min(manager.grow_factor * radius, manager.max_radius)
    if rho >= manager.eta1:
        return radius
    return (manager.gamma1 if rho < 0.0 else manager.gamma2) * radius


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta1": 0.0},
        {"eta1": 0.8, "eta2": 0.5},
        {"eta2": 1.0},
        {"gamma1": 0.0},
        {"gamma1": 0.6, "gamma2": 0.5},
        {"gamma2": 1.0},
        {"grow_factor": 1.0},
        {"max_radius": 0.0},
        {"kappa_t1": 1.0},
        {"kappa_t2": 0.0},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_check_init_rejects_invalid_constants(kwargs):
    with pytest.raises(ValueError):
        make_funnel_manager(PROBLEM, **kwargs)


def test_rejects_non_funnel_state():
    manager = make_funnel_manager(PROBLEM)
    with pytest.raises(TypeError, match="TrustFunnelSolverState"):
        manager.step(X0, f_direction(), make_tr_state())


# --------------------------------------------------------------------------- #
# y-iterations (3.24)
# --------------------------------------------------------------------------- #


def test_y_iteration_changes_nothing():
    manager = make_funnel_manager(PROBLEM)
    state = make_funnel_state(0.7, 1.3, 4.0, sf_flag=True, rho=-3.0)
    zero = jax.tree.map(jnp.zeros_like, X0)
    result = manager.step(X0, zero, state)
    st = result.solver_state

    assert not bool(result.accepted)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.x, X0))
    assert float(result.merit_val) == pytest.approx(float(manager.barrier_merit(X0)))
    assert float(result.proposed_step_norm) == 0.0
    assert st.iteration_type == IterationType.y_iteration
    assert float(st.radius_v) == pytest.approx(0.7)
    assert float(st.radius_f) == pytest.approx(1.3)
    assert float(st.v_max) == pytest.approx(4.0)
    assert bool(st.sf_flag)  # untouched
    assert float(st.rho) == 1.0


# --------------------------------------------------------------------------- #
# f-iterations (3.25)–(3.30)
# --------------------------------------------------------------------------- #


@RHO_CASES
@pytest.mark.parametrize("sf_flag", [False, True], ids=["sf-clear", "sf-set"])
def test_f_iteration_updates(rho_target, sf_flag):
    manager = make_funnel_manager(PROBLEM)
    direction = f_direction(sign=1.0 if rho_target > 0 else -1.0)
    df, _, x_trial = actual_decreases(manager, direction)
    assert (df > 0) == (rho_target > 0)
    dm_f_d = df / rho_target  # predicted decrease giving ρᶠ = rho_target
    radius_v, radius_f, v_max = 0.8, 1.5, 1e3
    state = make_funnel_state(
        radius_v,
        radius_f,
        v_max,
        tangential_norm=0.3,
        normal_norm=0.1,
        dm_f_n=0.0,
        dm_f_t=dm_f_d,
        objective_decrease_ok=True,
        dm_v_n=0.0,
        dm_v_d=0.0,
        contraction_ok=False,
        sf_flag=sf_flag,
    )
    result = manager.step(X0, direction, state)
    st = result.solver_state
    accepted = rho_target >= manager.eta1

    assert st.iteration_type == IterationType.f_iteration
    assert float(st.rho) == pytest.approx(rho_target, rel=1e-4)
    assert bool(result.accepted) == accepted
    target = x_trial if accepted else X0
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.x, target))
    assert float(result.merit_val) == pytest.approx(
        float(manager.barrier_merit(target)), rel=1e-6
    )
    assert float(result.step_size) == (1.0 if accepted else 0.0)
    # (3.27) / (3.29) on δᶠ; (3.28) / (3.29) leave δᵛ; (3.30) leaves v_max.
    assert float(st.radius_f) == pytest.approx(
        expected_radius(radius_f, rho_target, manager), rel=1e-6
    )
    assert float(st.radius_v) == pytest.approx(radius_v)
    assert float(st.v_max) == pytest.approx(v_max)
    # S_f-flag is raised on success and never lowered here.
    assert bool(st.sf_flag) == (sf_flag or accepted)


def test_f_candidate_demoted_to_v_when_funnel_violated():
    """(2.11) fails ⇒ v-iteration even though t ≠ 0 and (2.10) holds."""
    manager = make_funnel_manager(PROBLEM)
    direction = f_direction()
    df, dv, _ = actual_decreases(manager, direction)
    v_trial = float(manager.violation(jax.tree.map(jnp.add, X0, direction)))
    state = make_funnel_state(
        1.0,
        1.0,
        0.5 * v_trial,  # v_max below v⁺
        tangential_norm=0.3,
        normal_norm=0.1,
        dm_f_t=df,
        objective_decrease_ok=True,
        dm_v_n=abs(dv) + 1.0,
        dm_v_d=abs(dv) + 1.0,
        contraction_ok=True,
    )
    result = manager.step(X0, direction, state)
    st = result.solver_state
    assert st.iteration_type == IterationType.v_iteration
    rho_v = dv / (abs(dv) + 1.0)
    assert float(st.rho) == pytest.approx(rho_v, rel=1e-4)
    assert bool(result.accepted) == (rho_v >= manager.eta1)
    assert float(st.radius_f) == pytest.approx(1.0)  # (3.37)
    assert not bool(st.sf_flag)


# --------------------------------------------------------------------------- #
# v-iterations (3.32)–(3.37)
# --------------------------------------------------------------------------- #


@RHO_CASES
def test_v_iteration_updates(rho_target):
    manager = make_funnel_manager(PROBLEM)
    direction = v_direction(sign=1.0 if rho_target > 0 else -1.0)
    _, dv, x_trial = actual_decreases(manager, direction)
    assert (dv > 0) == (rho_target > 0)
    dm_v_d = dv / rho_target
    v0 = float(manager.violation(X0))
    v_trial = float(manager.violation(x_trial))
    radius_v, radius_f, v_max = 0.8, 1.5, 2.0 * v0
    state = make_funnel_state(
        radius_v,
        radius_f,
        v_max,
        tangential_norm=0.0,
        normal_norm=0.2,
        dm_v_n=dm_v_d,
        dm_v_d=dm_v_d,
        contraction_ok=True,
        dm_f_n=0.0,
        dm_f_t=0.0,
        objective_decrease_ok=False,
    )
    result = manager.step(X0, direction, state)
    st = result.solver_state
    accepted = rho_target >= manager.eta1

    assert st.iteration_type == IterationType.v_iteration
    assert float(st.rho) == pytest.approx(rho_target, rel=1e-4)
    assert bool(result.accepted) == accepted
    target = x_trial if accepted else X0
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.x, target))
    assert float(result.merit_val) == pytest.approx(
        float(manager.barrier_merit(target)), rel=1e-6
    )
    # (3.34) / (3.36) on δᵛ; (3.37) leaves δᶠ.
    assert float(st.radius_v) == pytest.approx(
        expected_radius(radius_v, rho_target, manager), rel=1e-6
    )
    assert float(st.radius_f) == pytest.approx(radius_f)
    # (3.35) contracts the funnel on success; (3.36) leaves it otherwise.
    if accepted:
        expected = max(
            manager.kappa_t1 * v_max, v_trial + manager.kappa_t2 * (v0 - v_trial)
        )
        assert float(st.v_max) == pytest.approx(expected, rel=1e-5)
        assert float(st.v_max) < v_max
        assert v_trial <= float(st.v_max)
    else:
        assert float(st.v_max) == pytest.approx(v_max)
    assert not bool(st.sf_flag)


@pytest.mark.parametrize(
    "normal_norm, contraction_ok",
    [(0.0, True), (0.2, False)],
    ids=["zero-normal", "no-contraction"],
)
def test_v_iteration_requires_215(normal_norm, contraction_ok):
    """A good ``ρᵛ`` is not enough: (2.15) needs ``n ≠ 0`` and the contraction."""
    manager = make_funnel_manager(PROBLEM)
    direction = v_direction()
    _, dv, _ = actual_decreases(manager, direction)
    state = make_funnel_state(
        1.0,
        1.0,
        10.0,
        normal_norm=normal_norm,
        dm_v_n=dv,
        dm_v_d=dv,  # ρᵛ = 1
        contraction_ok=contraction_ok,
    )
    result = manager.step(X0, direction, state)
    st = result.solver_state
    assert float(st.rho) == pytest.approx(1.0, rel=1e-4)
    assert not bool(result.accepted)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.x, X0))
    assert float(st.radius_v) == pytest.approx(manager.gamma2)  # ρ ≥ 0 shrink
    assert float(st.v_max) == pytest.approx(10.0)


def test_non_positive_predicted_decrease_rejects():
    manager = make_funnel_manager(PROBLEM)
    direction = v_direction()
    state = make_funnel_state(
        1.0, 1.0, 10.0, normal_norm=0.2, dm_v_n=0.0, dm_v_d=0.0, contraction_ok=True
    )
    result = manager.step(X0, direction, state)
    st = result.solver_state
    assert float(st.rho) == -jnp.inf
    assert not bool(result.accepted)
    assert float(st.radius_v) == pytest.approx(manager.gamma1)


def test_v_max_is_nonincreasing_along_successful_v_iterations():
    manager = make_funnel_manager(PROBLEM)
    x = X0
    v_max = 2.0 * float(manager.violation(x))
    state = make_funnel_state(1.0, 1.0, v_max)
    history = [v_max]
    for _ in range(4):
        direction = scale(diff(X_FEAS, x), 0.5)
        x_trial = jax.tree.map(jnp.add, x, direction)
        dv = float(manager.violation(x) - manager.violation(x_trial))
        state = make_funnel_state(
            float(state.radius_v),
            float(state.radius_f),
            float(state.v_max),
            normal_norm=0.2,
            dm_v_n=dv,
            dm_v_d=dv,
            contraction_ok=True,
        )
        result = manager.step(x, direction, state)
        assert bool(result.accepted)
        state = result.solver_state
        x = result.x
        history.append(float(state.v_max))
        assert float(manager.violation(x)) <= history[-1]
    assert all(b <= a for a, b in zip(history, history[1:]))
    assert history[-1] < history[0]


# --------------------------------------------------------------------------- #
# Growth cap, logging, jit
# --------------------------------------------------------------------------- #


def test_growth_is_capped_by_max_radius():
    manager = make_funnel_manager(PROBLEM, max_radius=1.2)
    direction = f_direction()
    df, _, _ = actual_decreases(manager, direction)
    state = make_funnel_state(
        1.0, 1.0, 1e3, tangential_norm=0.3, dm_f_t=df, objective_decrease_ok=True
    )
    result = manager.step(X0, direction, state)
    assert bool(result.accepted)
    assert float(result.solver_state.radius_f) == pytest.approx(1.2)


def test_logging_info_on_every_step_and_warning_on_rejection():
    handler = MemoryHandler()
    manager = make_funnel_manager(
        PROBLEM, logger=Logger.from_options({"level": "INFO", "handler": handler})
    )
    direction = f_direction()
    df, _, _ = actual_decreases(manager, direction)
    good = make_funnel_state(
        1.0, 1.0, 1e3, tangential_norm=0.3, dm_f_t=df, objective_decrease_ok=True
    )
    bad = make_funnel_state(
        1.0, 1.0, 1e3, tangential_norm=0.3, dm_f_t=50 * df, objective_decrease_ok=True
    )
    manager.step(X0, direction, good)
    n_good = len(handler.records)
    assert [r.levelno for r in handler.records] == [INFO]
    manager.step(X0, direction, bad)
    new = handler.records[n_good:]
    assert [r.levelno for r in new] == [INFO, WARNING]
    assert "rejected" in new[1].message


def test_step_is_jittable_and_matches_eager():
    manager = make_funnel_manager(PROBLEM)
    direction = v_direction()
    _, dv, _ = actual_decreases(manager, direction)
    state = make_funnel_state(
        1.0, 1.0, 10.0, normal_norm=0.2, dm_v_n=dv, dm_v_d=dv, contraction_ok=True
    )
    eager = manager.step(X0, direction, state)
    jitted = jax.jit(lambda x, d, s: manager.step(x, d, s))(X0, direction, state)
    assert bool(eager.accepted) == bool(jitted.accepted)
    assert jnp.allclose(eager.x.flatten(), jitted.x.flatten())
    assert float(eager.solver_state.v_max) == pytest.approx(
        float(jitted.solver_state.v_max), rel=1e-6
    )
    assert eager.solver_state.iteration_type == jitted.solver_state.iteration_type
