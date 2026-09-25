"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.working_set_policy`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import (
    SingleExchangeWorkingSetPolicy,
    ThresholdWorkingSetPolicy,
    WorkingSetPolicyState,
)
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import make_evaluated_lagrangian, make_zero_dual

from .conftest import unbounded_box

TOL = 1e-6


def _lagrangian(*, bounded: bool):
    """Default quadratic NLP (``meq=1, mineq=2``) with or without finite bounds."""
    if bounded:
        problem = make_problem()
    else:
        lb, ub = unbounded_box(2)
        problem = make_problem(lb=lb, ub=ub)
    return make_evaluated_lagrangian(
        problem=problem, primal=make_primal(), dual=make_zero_dual(2, 1, 2)
    )


def _active(ineq=(False, False), lb=(False, False), ub=(False, False)) -> ActiveSet:
    return ActiveSet(
        meq=1,
        active_inequalities=jnp.asarray(ineq, bool),
        active_lb=jnp.asarray(lb, bool),
        active_ub=jnp.asarray(ub, bool),
    )


def _step(dx, *, ineq=(0.0, 0.0), lb=(0.0, 0.0), ub=(0.0, 0.0)):
    return (
        Primal(jnp.asarray(dx, float)),
        Dual(
            eq_multipliers=jnp.zeros((1,)),
            ineq_multipliers=jnp.asarray(ineq, float),
            lb_multipliers=jnp.asarray(lb, float),
            ub_multipliers=jnp.asarray(ub, float),
        ),
    )


def _reference_masks(lag, step, current, tol):
    """Fixed-threshold refresh as it was hard-coded before the policy existed."""
    dx = step[0].x
    lam = step[1]
    x_new = lag.ref.x + dx
    ineq_lin = lag.ineq_fn_val + lag.ineq_fn_jac_val @ dx
    ai, alb, aub = current.active_inequalities, current.active_lb, current.active_ub
    return (
        (ai | (ineq_lin > tol)) & ~(ai & (lam.ineq_multipliers < -tol)),
        (~lag.null_lb)
        & ((alb | ((lag.lb - x_new) > tol)) & ~(alb & (lam.lb_multipliers < -tol))),
        (~lag.null_ub)
        & ((aub | ((x_new - lag.ub) > tol)) & ~(aub & (lam.ub_multipliers < -tol))),
    )


@pytest.mark.parametrize("bounded", [True, False], ids=["bounded", "unbounded"])
@pytest.mark.parametrize(
    ("current", "step"),
    [
        (_active(), _step([2.0, -1.0])),  # violates both inequalities
        (_active(ineq=(True, True)), _step([0.0, 0.0], ineq=(-1.0, 1.0))),  # drop 0
        (_active(lb=(True, False)), _step([-5.0, 5.0], lb=(-1.0, 0.0))),  # bounds
    ],
    ids=["add", "drop", "bounds"],
)
def test_default_policy_reproduces_fixed_threshold_refresh(bounded, current, step):
    """Defaults give the classic all-at-once threshold update and never cycle."""
    lag = _lagrangian(bounded=bounded)
    policy = ThresholdWorkingSetPolicy(tol=TOL)
    state = policy.init_state(current)
    next_set, new_state, cycled = policy.update(lag, step, current, state)

    exp_ai, exp_lb, exp_ub = _reference_masks(lag, step, current, TOL)
    assert jnp.array_equal(next_set.active_inequalities, exp_ai)
    assert jnp.array_equal(next_set.active_lb, exp_lb)
    assert jnp.array_equal(next_set.active_ub, exp_ub)
    assert not bool(cycled)
    assert int(new_state.cycle_count) == 0
    assert float(new_state.working_tol) == pytest.approx(TOL)
    assert bool(eqx.tree_equal(new_state.prev_set, current))


@pytest.mark.parametrize("expand_factor", [0.0, 1.0, 2.5])
@pytest.mark.parametrize("max_iter", [1, 4])
def test_expand_ramp_is_linear_in_the_iteration_budget(expand_factor, max_iter):
    """``working_tol_k = tol (1 + expand_factor k / max_iter)`` after ``k`` updates."""
    lag = _lagrangian(bounded=False)
    policy = ThresholdWorkingSetPolicy(
        tol=TOL, max_iter=max_iter, expand_factor=expand_factor
    )
    current = _active()
    state = policy.init_state(current)
    for k in range(1, 4):
        _, state, _ = policy.update(lag, _step([0.0, 0.0]), current, state)
        expected = TOL * (1.0 + expand_factor * k / max_iter)
        assert float(state.working_tol) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("drop_floor", "multiplier", "expect_kept"),
    [(0.0, -1e-3, False), (1e-2, -1e-3, True), (1e-2, -1e-1, False)],
    ids=["no-floor-drops", "floor-keeps-noise", "floor-drops-real-negative"],
)
def test_drop_floor_suppresses_small_negative_multipliers(
    drop_floor, multiplier, expect_kept
):
    """A negative multiplier drops its row only below ``-max(working_tol, drop_floor)``."""
    lag = _lagrangian(bounded=False)
    policy = ThresholdWorkingSetPolicy(tol=TOL, drop_floor=drop_floor)
    current = _active(ineq=(True, False))
    state = policy.init_state(current)
    step = _step([0.0, 0.0], ineq=(multiplier, 0.0))
    next_set, _, _ = policy.update(lag, step, current, state)
    assert bool(next_set.active_inequalities[0]) is expect_kept


@pytest.mark.parametrize(
    "ping_pong_threshold", [None, 2], ids=["guard-off", "threshold-2"]
)
def test_ping_pong_detection_counts_consecutive_revisits(ping_pong_threshold):
    """A -> B -> A -> B alternation is counted; the flag needs the threshold."""
    lag = _lagrangian(bounded=False)
    policy = ThresholdWorkingSetPolicy(tol=TOL, ping_pong_threshold=ping_pong_threshold)
    set_a = _active(ineq=(True, False))
    set_b = _active(ineq=(False, True))
    # From A: drop row 0 (negative multiplier) and add row 1 (violated) -> B.
    step_a_to_b = _step([0.0, -1.0], ineq=(-1.0, 0.0))
    # From B: drop row 1 and add row 0 -> A.
    step_b_to_a = _step([2.0, 0.0], ineq=(0.0, -1.0))

    state = policy.init_state(set_a)
    transitions = [
        (set_a, step_a_to_b, set_b),
        (set_b, step_b_to_a, set_a),
        (set_a, step_a_to_b, set_b),
        (set_b, step_b_to_a, set_a),
    ]
    counts, flags = [], []
    for current, step, expected in transitions:
        next_set, state, cycled = policy.update(lag, step, current, state)
        assert bool(eqx.tree_equal(next_set, expected))
        counts.append(int(state.cycle_count))
        flags.append(bool(cycled))

    # The first proposal (B) is new; every later one revisits a stored set.
    assert counts == [0, 1, 2, 3]
    if ping_pong_threshold is None:
        assert flags == [False, False, False, False]
    else:
        assert flags == [False, False, True, True]


def test_fixed_point_is_not_a_cycle():
    """Re-proposing the current set (convergence) never increments the counter."""
    lag = _lagrangian(bounded=False)
    policy = ThresholdWorkingSetPolicy(tol=TOL, ping_pong_threshold=1)
    current = _active(ineq=(True, False))
    state = policy.init_state(current)
    for _ in range(3):
        next_set, state, cycled = policy.update(
            lag, _step([0.0, 0.0], ineq=(1.0, 0.0)), current, state
        )
        assert bool(eqx.tree_equal(next_set, current))
        assert not bool(cycled)
        assert int(state.cycle_count) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": -1e-8},
        {"max_iter": 0},
        {"expand_factor": -0.1},
        {"drop_floor": -1.0},
        {"ping_pong_threshold": 0},
    ],
    ids=["tol", "max_iter", "expand", "floor", "threshold"],
)
def test_invalid_knobs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ThresholdWorkingSetPolicy(**kwargs)


def test_state_is_a_plain_pytree_of_arrays():
    """The policy carry can live inside ``lax.while_loop``."""
    state = ThresholdWorkingSetPolicy(
        tol=TOL, max_iter=5, expand_factor=1.0
    ).init_state(_active())
    assert isinstance(state, WorkingSetPolicyState)
    assert float(state.ramp_increment) == pytest.approx(TOL / 5)
    assert int(state.cycle_count) == 0


def _mask_count(active: ActiveSet) -> int:
    return int(
        jnp.sum(active.active_inequalities)
        + jnp.sum(active.active_lb)
        + jnp.sum(active.active_ub)
    )


@pytest.mark.parametrize(
    ("current", "step", "expected"),
    [
        # Both inequalities violated (row 1 more): only row 1 is added.
        (_active(), _step([2.0, -3.0]), _active(ineq=(False, True))),
        # Inequality row 0 and lb₀ violated; the bound violation is larger.
        (
            _active(),
            _step([-5.0, 0.0], ineq=(0.0, 0.0)),
            _active(lb=(True, False)),
        ),
        # Nothing violated, two negative multipliers: drop only the most negative.
        (
            _active(ineq=(True, True)),
            _step([0.0, 0.0], ineq=(-1.0, -3.0)),
            _active(ineq=(True, False)),
        ),
        # A violation takes priority over a negative multiplier.
        (
            _active(ineq=(True, False)),
            _step([0.0, -1.0], ineq=(-1.0, 0.0)),
            _active(ineq=(True, True)),
        ),
        # Fixed point: feasible step, non-negative multipliers.
        (
            _active(ineq=(True, False)),
            _step([0.0, 0.0], ineq=(1.0, 0.0)),
            _active(ineq=(True, False)),
        ),
    ],
    ids=["add-most-violated", "add-bound", "drop-most-negative", "add-first", "fixed"],
)
def test_single_exchange_changes_at_most_one_row(current, step, expected):
    """One-at-a-time exchange: add the worst violation, else drop the worst multiplier."""
    lag = _lagrangian(bounded=True)
    policy = SingleExchangeWorkingSetPolicy(tol=TOL)
    next_set, _, cycled = policy.update(lag, step, current, policy.init_state(current))
    assert bool(eqx.tree_equal(next_set, expected))
    assert abs(_mask_count(next_set) - _mask_count(current)) <= 1
    assert not bool(cycled)


def test_single_exchange_inherits_ramp_and_cycle_guard():
    """The subclass reuses the threshold policy's carry, ramp and detector."""
    lag = _lagrangian(bounded=False)
    policy = SingleExchangeWorkingSetPolicy(
        tol=TOL, max_iter=4, expand_factor=1.0, ping_pong_threshold=1
    )
    set_a = _active(ineq=(True, False))
    set_b = _active()
    state = policy.init_state(set_a)
    # A -> B (drop row 0) -> A (add row 0) is a detected 2-cycle.
    next_set, state, cycled = policy.update(
        lag, _step([0.0, 0.0], ineq=(-1.0, 0.0)), set_a, state
    )
    assert bool(eqx.tree_equal(next_set, set_b)) and not bool(cycled)
    assert float(state.working_tol) == pytest.approx(TOL * 1.25)
    next_set, state, cycled = policy.update(lag, _step([2.0, 0.0]), set_b, state)
    assert bool(eqx.tree_equal(next_set, set_a)) and bool(cycled)
