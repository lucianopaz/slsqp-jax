"""Unit tests for :mod:`slsqp_jax.sqpdax.active_set_prediction`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.active_set_prediction import (
    LPECAPrediction,
    LPECAPredictor,
    compute_rho_bar,
    solve_lpeca_lp,
)
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import Lagrangian
from slsqp_jax.sqpdax.primal import Primal

from .conftest import make_shifted_box_quadratic


def _evaluate(problem, x, dual):
    return Lagrangian(problem)(Primal(x), dual)


def _kkt_lagrangian(n: int = 3, *, dx=None):
    """Lagrangian at (a perturbation of) the KKT point with exact multipliers."""
    problem, x_star, dual_star = make_shifted_box_quadratic(n)
    x = x_star if dx is None else x_star + jnp.asarray(dx)
    return problem, _evaluate(problem, x, dual_star), x_star, dual_star


def _expected_masks(n: int = 3):
    active_ineq = jnp.array([True])
    active_lb = jnp.zeros(n, bool).at[1].set(True)
    active_ub = jnp.zeros(n, bool)
    return active_ineq, active_lb, active_ub


def _assert_active_set(prediction: LPECAPrediction, ineq, lb, ub):
    assert jnp.array_equal(prediction.active_set.active_inequalities, ineq)
    assert jnp.array_equal(prediction.active_set.active_lb, lb)
    assert jnp.array_equal(prediction.active_set.active_ub, ub)


def test_rho_bar_vanishes_only_at_the_kkt_point():
    """``ρ̄ = 0`` at the exact KKT point and grows with any perturbation."""
    problem, lag_star, x_star, dual_star = _kkt_lagrangian()
    assert float(compute_rho_bar(lag_star)) == pytest.approx(0.0, abs=1e-6)

    moved = _evaluate(problem, x_star + jnp.array([0.0, 0.1, 0.0]), dual_star)
    wrong_dual = _evaluate(
        problem,
        x_star,
        eqx.tree_at(lambda d: d.ineq_multipliers, dual_star, jnp.array([0.2])),
    )
    assert float(compute_rho_bar(moved)) > 0.05
    assert float(compute_rho_bar(wrong_dual)) > 0.05


@pytest.mark.parametrize("dx", [None, (0.01, 0.005, -0.01)], ids=["exact", "nearby"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_prediction_recovers_the_active_set_near_the_solution(dx, jit):
    """Near ``x*`` with good multipliers the predicted set equals the true one."""
    problem, lag, _, _ = _kkt_lagrangian(dx=dx)
    predictor = LPECAPredictor(method="lpeca", warmup_steps=0)
    predict = jax.jit(predictor.predict) if jit else predictor.predict
    prediction = predict(lag, jnp.asarray(5, jnp.int32))

    assert bool(prediction.valid)
    assert not bool(prediction.capped)
    _assert_active_set(prediction, *_expected_masks())
    assert int(prediction.n_bounds_prefixed) == 1


@pytest.mark.parametrize(
    ("kwargs", "step_count"),
    [({"trust_threshold": 0.0}, 5), ({"warmup_steps": 3}, 2)],
    ids=["trust-gate", "warm-up"],
)
def test_trust_gate_and_warm_up_empty_the_prediction(kwargs, step_count):
    """A failed gate leaves ``valid=False`` and an all-inactive set."""
    _, lag, _, _ = _kkt_lagrangian(dx=(0.01, 0.005, -0.01))
    prediction = LPECAPredictor(method="lpeca_init", **kwargs).predict(lag, step_count)
    assert not bool(prediction.valid)
    assert not bool(prediction.capped)
    assert not bool(jnp.any(prediction.active_set.active_inequalities))
    assert not bool(jnp.any(prediction.active_set.active_lb))
    assert int(prediction.n_bounds_prefixed) == 0
    assert float(prediction.rho_bar) > 0.0


def test_rank_cap_keeps_the_most_violated_rows():
    """With ``n=2`` the cap ``n - meq - 1 = 1`` keeps only the most violated row."""
    problem, x_star, dual_star = make_shifted_box_quadratic(2)
    # Push into the lower bound (violation 0.05) while the inequality stays
    # marginally feasible: the bound must win the single slot.
    lag = _evaluate(problem, x_star + jnp.array([-0.01, -0.05]), dual_star)
    prediction = LPECAPredictor(method="lpeca", warmup_steps=0).predict(lag, 5)
    assert bool(prediction.valid)
    assert bool(prediction.capped)
    _assert_active_set(
        prediction, jnp.array([False]), jnp.array([False, True]), jnp.zeros(2, bool)
    )
    assert int(prediction.n_bounds_prefixed) == 1


def test_predict_bounds_false_only_seeds_general_inequalities():
    """``predict_bounds=False`` masks the bound rows out of the prediction."""
    _, lag, _, _ = _kkt_lagrangian()
    prediction = LPECAPredictor(
        method="lpeca", warmup_steps=0, predict_bounds=False
    ).predict(lag, 5)
    assert bool(prediction.valid)
    _assert_active_set(
        prediction, jnp.array([True]), jnp.zeros(3, bool), jnp.zeros(3, bool)
    )
    assert int(prediction.n_bounds_prefixed) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "bogus"},
        {"sigma": 1.0},
        {"sigma": 0.0},
        {"beta": -1.0},
        {"trust_threshold": -0.1},
        {"warmup_steps": -1},
    ],
    ids=["method", "sigma-high", "sigma-low", "beta", "trust", "warmup"],
)
def test_invalid_knobs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LPECAPredictor(**kwargs)


def test_method_flags_and_nested_init():
    """``enabled`` / ``disables_expand`` follow ``method``; ``init`` sets fields."""
    default = LPECAPredictor()
    assert (default.enabled, default.disables_expand) == (False, False)
    init_only = default.init(method="lpeca_init")
    assert (init_only.enabled, init_only.disables_expand) == (True, False)
    full = default.init(method="lpeca", sigma=0.5)
    assert (full.enabled, full.disables_expand, full.sigma) == (True, True, 0.5)


@pytest.mark.parametrize("via_predictor", [False, True], ids=["direct", "use_lp"])
def test_lp_refinement_recovers_the_kkt_multipliers(via_predictor):
    """The mpax LP returns the exact multipliers at ``x*`` (needs ``mpax``)."""
    pytest.importorskip("mpax")
    problem, x_star, dual_star = make_shifted_box_quadratic()
    zero_dual = Dual(
        eq_multipliers=jnp.zeros((0,)),
        ineq_multipliers=jnp.zeros((1,)),
        lb_multipliers=jnp.zeros(3),
        ub_multipliers=jnp.zeros(3),
    )
    lag = _evaluate(problem, x_star, zero_dual)
    if via_predictor:
        prediction = LPECAPredictor(
            method="lpeca", warmup_steps=0, use_lp=True
        ).predict(lag, 5)
        assert bool(prediction.valid)
        _assert_active_set(prediction, *_expected_masks())
    else:
        refined = solve_lpeca_lp(lag, eps=1e-8, max_iter=5000)
        assert jnp.allclose(
            refined.ineq_multipliers, dual_star.ineq_multipliers, atol=1e-3
        )
        assert jnp.allclose(refined.lb_multipliers, dual_star.lb_multipliers, atol=1e-3)
        assert jnp.allclose(refined.ub_multipliers, 0.0, atol=1e-3)
