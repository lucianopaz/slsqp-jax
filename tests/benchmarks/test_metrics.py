"""Tests for ``benchmarks.metrics``."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from benchmarks.metrics import quality_metrics, result_name, stationarity_residual
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.minimiser import ACTIVE_SET_LINE_SEARCH_RESULTS


def test_metrics_vanish_at_kkt_point(kkt_problem):
    problem, x, dual, fstar = kkt_problem
    assert jnp.allclose(stationarity_residual(problem, x, dual), 0.0, atol=1e-12)
    m = quality_metrics(problem, x, dual, fstar=fstar, xstar=x)
    for key in (
        "feas",
        "feas_eq",
        "feas_lb",
        "feas_ub",
        "stat_inf",
        "stat_2",
        "stat_scaled",
        "compl",
        "f_gap",
        "x_err",
    ):
        assert m[key] == pytest.approx(0.0, abs=1e-12), key
    assert m["mult_min"] >= 0.0
    assert m["finite"]
    assert m["objective"] == pytest.approx(fstar)


@pytest.mark.parametrize(
    ("perturb", "expect_positive"),
    [
        ("x", ["feas_eq", "stat_inf"]),  # move off the equality manifold
        ("lam", ["stat_inf", "stat_2", "stat_scaled"]),  # wrong multiplier
        ("zlb", ["stat_inf", "compl"]),  # bound multiplier on an inactive bound
        ("none_dual", []),
    ],
)
def test_metrics_detect_perturbations(kkt_problem, perturb, expect_positive):
    problem, x, dual, fstar = kkt_problem
    if perturb == "x":
        x = x + jnp.array([0.1, 0.0, 0.0])
    elif perturb == "lam":
        dual = Dual(
            dual.eq_multipliers + 1.0,
            dual.ineq_multipliers,
            dual.lb_multipliers,
            dual.ub_multipliers,
        )
    elif perturb == "zlb":
        x = x + jnp.array([0.0, 0.0, 0.2])  # x2 off its bound while z_lb2 stays 1
    elif perturb == "none_dual":
        dual = None
    m = quality_metrics(problem, x, dual, fstar=fstar)
    for key in expect_positive:
        assert m[key] > 1e-3, key
    if dual is None:
        assert all(
            jnp.isnan(m[k])
            for k in ("stat_inf", "stat_2", "stat_scaled", "compl", "mult_min")
        )
        assert m["feas"] == pytest.approx(0.0)


def test_inactive_bound_terms_are_masked(kkt_problem):
    """Multipliers on infinite bounds must not enter stationarity or complementarity."""
    problem, x, dual, _ = kkt_problem
    noisy = Dual(
        dual.eq_multipliers,
        dual.ineq_multipliers,
        dual.lb_multipliers + jnp.array([5.0, 5.0, 0.0]),  # lb0, lb1 are -inf
        dual.ub_multipliers + jnp.array([5.0, 0.0, 5.0]),  # ub0, ub2 are +inf
    )
    m = quality_metrics(problem, x, noisy)
    assert m["stat_inf"] == pytest.approx(0.0, abs=1e-12)
    assert m["compl"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", ["successful", "max_steps_reached"])
def test_result_name_roundtrip(name):
    assert result_name(getattr(ACTIVE_SET_LINE_SEARCH_RESULTS, name)) == name


def test_result_name_fallback():
    assert result_name("timeout") == "timeout"
