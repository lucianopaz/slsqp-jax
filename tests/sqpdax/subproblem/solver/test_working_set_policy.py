"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.working_set_policy`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import Lagrangian
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem import build_problem
from slsqp_jax.sqpdax.subproblem.solver import (
    SingleExchangeWorkingSetPolicy,
    ThresholdWorkingSetPolicy,
    WorkingSetPolicyState,
)
from slsqp_jax.sqpdax.subproblem.solver.working_set_policy import (
    cap_rows,
    exchange_row,
    independent_rows,
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
    """Without the row cap the defaults give the classic all-at-once refresh."""
    lag = _lagrangian(bounded=bounded)
    policy = ThresholdWorkingSetPolicy(tol=TOL, cap_working_set=False)
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
    policy = SingleExchangeWorkingSetPolicy(tol=TOL, cap_working_set=False)
    next_set, _, cycled = policy.update(lag, step, current, policy.init_state(current))
    assert bool(eqx.tree_equal(next_set, expected))
    assert abs(_mask_count(next_set) - _mask_count(current)) <= 1
    assert not bool(cycled)


# The default NLP has n = 2 and meq = 1, so the row cap is a single row.
@pytest.mark.parametrize(
    "policy_cls", [ThresholdWorkingSetPolicy, SingleExchangeWorkingSetPolicy]
)
@pytest.mark.parametrize(
    ("current", "step", "expected"),
    [
        # Two violated inequalities: only the more violated row 1 fits.
        (_active(), _step([2.0, -3.0]), _active(ineq=(False, True))),
        # Without the exchange, a row already in the set is kept in
        # preference to a new violation.
        (
            _active(ineq=(True, False)),
            _step([0.0, -3.0], ineq=(1.0, 0.0)),
            _active(ineq=(True, False)),
        ),
        # A dropped row (negative multiplier) frees the slot for the violation
        # under the all-at-once refresh; the single exchange drops first.
        (
            _active(ineq=(True, False)),
            _step([0.0, -3.0], ineq=(-1.0, 0.0)),
            None,
        ),
    ],
    ids=["two-violations", "keep-current", "drop-then-add"],
)
def test_cap_limits_the_working_set_to_the_degrees_of_freedom(
    policy_cls, current, step, expected
):
    lag = _lagrangian(bounded=False)
    policy = policy_cls(tol=TOL, exchange_when_full=False)
    assert policy.cap_working_set
    next_set, _, _ = policy.update(lag, step, current, policy.init_state(current))
    assert _mask_count(next_set) <= lag.n - lag.meq == 1
    if expected is None:
        expected = (
            _active(ineq=(False, True))
            if policy_cls is ThresholdWorkingSetPolicy
            else _active()
        )
    assert bool(eqx.tree_equal(next_set, expected))


# ``make_problem``: ``x₀ + x₁ = 1``, ``x₀ ≤ 2``, ``x₁ ≥ 0``, box ``[0, 2] × [-1, 3]``.
@pytest.mark.parametrize("cap", [True, False])
@pytest.mark.parametrize(
    ("meq", "x", "active0", "expected_count"),
    [
        # Both inequalities violated: only the most violated row fits.
        (1, (2.5, -1.0), _active(ineq=(True, True)), 1),
        # Feasible degenerate vertex (no equality): three active rows for two
        # degrees of freedom are consistent with ``d = 0`` and kept whole.
        (0, (2.0, 0.0), _active(ineq=(True, True), ub=(True, False)), 3),
        # Same vertex but the equality is violated: the set is capped.
        (1, (2.0, 0.0), _active(ineq=(True, True), ub=(True, False)), 1),
    ],
    ids=["violated", "feasible-vertex", "violated-equality"],
)
def test_initial_set_is_capped_from_an_infeasible_reference(
    cap, meq, x, active0, expected_count
):
    """The cold start of an infeasible point is cut down to ``n - meq`` rows."""
    problem = make_problem(meq=meq)
    active0 = ActiveSet(
        meq=meq,
        active_inequalities=active0.active_inequalities,
        active_lb=active0.active_lb,
        active_ub=active0.active_ub,
    )
    lag = make_evaluated_lagrangian(
        problem=problem,
        primal=Primal(jnp.asarray(x)),
        dual=make_zero_dual(2, meq, 2),
    )
    policy = ThresholdWorkingSetPolicy(tol=TOL, cap_working_set=cap)
    kept = policy.initial_set(lag, active0)
    if not cap or expected_count == _mask_count(active0):
        assert bool(eqx.tree_equal(kept, active0))
        return
    assert _mask_count(kept) == expected_count == lag.n - lag.meq
    values = jnp.asarray(lag.ineq_fn_val)
    if bool(jnp.any(values > TOL)):
        # The most violated row is the one kept.
        assert bool(kept.active_inequalities[int(jnp.argmax(values))])


def _dependent_rows_lagrangian():
    """``n = 3``, one equality ``x₂ = 1`` and four violated inequalities.

    Row 0 is a near copy of the more violated row 1 and row 3 is parallel to
    the equality Jacobian, so only rows 1 and 2 can enter a non-singular
    working set. The (most violated) lower bound on ``x₂`` is likewise
    parallel to the equality.
    """
    A = jnp.array([[1.0, 0.0, 0.0], [1.0, 1e-6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    e2 = jnp.array([[0.0, 0.0, 1.0]])
    problem = build_problem(
        n=3,
        meq=1,
        mineq=4,
        fn=lambda x: jnp.sum(x**2),
        grad=lambda x: 2 * x,
        hvp=lambda x, v: 2 * v,
        eq_fn=lambda x: e2 @ x - 1.0,
        eq_fn_jac=lambda x: e2,
        eq_fn_hvp=lambda x, v: jnp.zeros((1, 3)),
        ineq_fn=lambda x: A @ x + jnp.array([3.0, 4.0, 2.0, 1.0]),
        ineq_fn_jac=lambda x: A,
        ineq_fn_hvp=lambda x, v: jnp.zeros((4, 3)),
        lb=jnp.array([-jnp.inf, -jnp.inf, 5.0]),
        ub=jnp.array([jnp.inf, jnp.inf, jnp.inf]),
    )
    return Lagrangian(problem)(Primal(jnp.zeros(3)), make_zero_dual(3, 1, 4))


@pytest.mark.parametrize("via_policy", [False, True], ids=["function", "policy"])
def test_independence_screen_skips_dependent_rows(via_policy):
    """Near-duplicate and equality-parallel rows never enter a full set."""
    lag = _dependent_rows_lagrangian()
    cand = jnp.array([True, True, True, True, False, False, True, False, False, False])
    scores = jnp.concatenate([lag.ineq_fn_val, lag.lb - lag.ref.x, lag.ref.x - lag.ub])
    if via_policy:
        policy = ThresholdWorkingSetPolicy(tol=TOL)
        selected = policy._select_rows(lag, jnp.zeros_like(cand), cand, scores)
    else:
        selected = independent_rows(jnp.zeros_like(cand), cand, scores, 2, lag)
    assert selected.tolist() == [False, True, True, False] + [False] * 6
    # Without the screen the cap keeps the two most violated (dependent) rows.
    plain = ThresholdWorkingSetPolicy(tol=TOL, independence_tol=0.0)
    capped = plain._select_rows(lag, jnp.zeros_like(cand), cand, scores)
    assert (
        capped.tolist() == [False, True, False, False, False, False, True] + [False] * 3
    )


@pytest.mark.parametrize("select", [cap_rows, independent_rows], ids=["cap", "screen"])
def test_rows_staying_in_the_set_are_never_capped_or_screened(select):
    """``keep`` passes through whole even when it exceeds the cap or is dependent."""
    lag = _dependent_rows_lagrangian()
    keep = jnp.array([True, True, False, True, False, False, True, False, False, False])
    cand = jnp.array(
        [False, False, True, False, False, False, False, False, False, False]
    )
    scores = jnp.concatenate([lag.ineq_fn_val, lag.lb - lag.ref.x, lag.ref.x - lag.ub])
    args = (keep, cand, scores, 2) + ((lag,) if select is independent_rows else ())
    selected = select(*args)
    # Four rows stay for a cap of two: no slot is left for the candidate.
    assert selected.tolist() == keep.tolist()


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


def _exchange_lagrangian(a_row, *, equality: bool):
    """``n = 2``, one inequality ``a·x + 1 ≤ 0``, ``x ≥ 0``, optionally ``x₁ = 0``.

    At ``x = 0`` the inequality is violated by 1 and both lower bounds are
    active; stacked rows are ``[a | -e₀ | -e₁ | +e₀ | +e₁]``.
    """
    A = jnp.asarray([a_row], float)
    eq = (
        {
            "meq": 1,
            "eq_fn": lambda x: x[1:],
            "eq_fn_jac": lambda x: jnp.array([[0.0, 1.0]]),
            "eq_fn_hvp": lambda x, v: jnp.zeros((1, 2)),
        }
        if equality
        else {}
    )
    problem = build_problem(
        n=2,
        fn=lambda x: jnp.sum(x**2),
        grad=lambda x: 2 * x,
        hvp=lambda x, v: 2 * v,
        ineq_fn=lambda x: A @ x + 1.0,
        ineq_fn_jac=lambda x: A,
        ineq_fn_hvp=lambda x, v: jnp.zeros((1, 2)),
        lb=jnp.zeros(2),
        **eq,
    )
    return Lagrangian(problem)(Primal(jnp.zeros(2)), None)


@pytest.mark.parametrize(
    ("a_row", "equality", "selected", "lam", "expected"),
    [
        # a = e₀·1 + e₁·2 in the (-e₀, -e₁) rows: alpha = (1, 2), min λ/α.
        ((-1.0, -2.0), False, (1, 1), (1.0, 1.0), (1, 1, 0)),
        ((-1.0, -2.0), False, (1, 1), (0.1, 1.0), (1, 0, 1)),
        # alpha = (1, -2): only the first bound can leave, whatever λ says.
        ((-1.0, 2.0), False, (1, 1), (5.0, 0.0), (1, 0, 1)),
        # alpha = (-1, -1): the linearised rows are inconsistent, no exchange.
        ((1.0, 1.0), False, (1, 1), (1.0, 1.0), (0, 1, 1)),
        # a is independent of span{-e₀}: the set is rank-deficient, no exchange.
        ((-1.0, -2.0), False, (1, 0), (1.0, 0.0), (0, 1, 0)),
        # The equality row carries the e₁ part and never leaves.
        ((-1.0, -2.0), True, (1, 0), (1.0, 0.0), (1, 0, 0)),
    ],
    ids=[
        "ratio",
        "ratio-small-lam",
        "one-positive",
        "inconsistent",
        "independent",
        "eq",
    ],
)
def test_exchange_row_follows_the_goldfarb_idnani_ratio(
    a_row, equality, selected, lam, expected
):
    """The leaving row has ``αⱼ > 0`` and the smallest ``λⱼ / αⱼ``."""
    lag = _exchange_lagrangian(a_row, equality=equality)
    sel = jnp.array([False, *map(bool, selected), False, False])
    rejected = jnp.array([True, False, False, False, False])
    values = jnp.array([1.0, 0.0, 0.0, -jnp.inf, -jnp.inf])
    multipliers = jnp.array([0.0, *lam, 0.0, 0.0])
    out = exchange_row(sel, rejected, values, multipliers, lag)
    assert out.tolist() == [*map(bool, expected), False, False]


@pytest.mark.parametrize(
    "exchange", [False, True, None], ids=["cap", "exchange", "default"]
)
def test_full_set_admits_violated_row_only_with_exchange(exchange):
    """A full set keeps a violated row out unless ``exchange_when_full`` (the
    default)."""
    lag = _exchange_lagrangian((-1.0, -2.0), equality=False)
    current = ActiveSet(
        meq=0,
        active_inequalities=jnp.array([False]),
        active_lb=jnp.array([True, True]),
        active_ub=jnp.array([False, False]),
    )
    step = (
        Primal(jnp.zeros(2)),
        Dual(
            eq_multipliers=jnp.zeros((0,)),
            ineq_multipliers=jnp.zeros((1,)),
            lb_multipliers=jnp.ones(2),
            ub_multipliers=jnp.zeros(2),
        ),
    )
    kwargs = {} if exchange is None else {"exchange_when_full": exchange}
    policy = ThresholdWorkingSetPolicy(tol=TOL, **kwargs)
    exchange = policy.exchange_when_full
    proposed = policy.propose(lag, step, current, jnp.asarray(TOL))
    assert bool(proposed.active_inequalities[0]) is exchange
    assert proposed.active_lb.tolist() == ([True, False] if exchange else [True, True])


def test_single_exchange_rejects_full_set_exchange():
    with pytest.raises(ValueError, match="exchange_when_full"):
        SingleExchangeWorkingSetPolicy(exchange_when_full=True)
