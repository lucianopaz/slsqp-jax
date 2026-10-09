"""Tests for the SciPy baselines (``benchmarks.baselines``) and the runner interface."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.baselines import (
    METHODS,
    ScipyBaseline,
    ScipyRunner,
    dual_from_slsqp,
    dual_from_trust_constr,
    scipy_status_name,
)
from benchmarks.configs import CONFIGS
from benchmarks.metrics import quality_metrics
from benchmarks.runners import SolveOutcome, SqpdaxRunner
from benchmarks.worker import _scalar_stats, make_runner
from slsqp_jax.sqpdax.problem import build_problem

BASELINE_CONFIGS = [c.name for c in CONFIGS.values() if c.backend == "scipy"]
SQPDAX_SMOKE_CONFIGS = ["pasls"]


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name", [*BASELINE_CONFIGS, *SQPDAX_SMOKE_CONFIGS])
def test_runner_solves_hs71_with_consistent_kkt(hs71_problem, name):
    """Every backend yields a SolveOutcome whose multipliers satisfy the shared KKT metrics."""
    problem, x0, fstar = hs71_problem
    cfg = CONFIGS[name]
    runner = make_runner(cfg, problem, x0)
    assert isinstance(runner, ScipyRunner if cfg.backend == "scipy" else SqpdaxRunner)

    compile_s = runner.compile()
    assert compile_s > 0
    runner.warmup()
    out = runner.solve(500)

    assert isinstance(out, SolveOutcome)
    assert out.status == "successful" and out.successful
    assert out.steps > 0
    assert out.dual is not None
    stats = _scalar_stats(out.stats)
    assert "stats_num_steps" not in stats and stats  # num_steps lives in ``steps``
    if cfg.backend == "scipy":
        assert {"stats_nfev", "stats_njev"} <= set(stats)

    metrics = quality_metrics(problem, out.x, out.dual, fstar=fstar)
    assert metrics["finite"]
    assert metrics["feas"] < 1e-6
    assert metrics["f_gap"] < 1e-5
    assert metrics["stat_inf"] < 1e-4
    assert metrics["compl"] < 1e-4
    assert metrics["mult_min"] >= -1e-8


@pytest.mark.parametrize("name", BASELINE_CONFIGS)
def test_baseline_max_steps_status(hs71_problem, name):
    problem, x0, _ = hs71_problem
    runner = make_runner(CONFIGS[name], problem, x0)
    out = runner.solve(1)
    assert out.status == "max_steps_reached"
    assert not out.successful
    assert out.steps == 1


@pytest.mark.parametrize("name", BASELINE_CONFIGS)
def test_baseline_bounds_only_problem(name):
    """Problems without general constraints (the ``bounded`` collections) still run."""
    problem = build_problem(
        lambda x: jnp.sum((x - 2.0) ** 2),
        n=3,
        lb=jnp.array([-jnp.inf, 0.0, 3.0]),
        ub=jnp.array([1.0, jnp.inf, jnp.inf]),
        autodiff_mode="jax",
        force_hvp_in_jax_mode=True,
    )
    runner = make_runner(CONFIGS[name], problem, jnp.array([0.0, 0.5, 4.0]))
    runner.compile()
    out = runner.solve(200)
    assert out.successful
    # trust-constr declares success once the Lagrangian gradient is below
    # gtol, whatever the remaining barrier parameter, so its iterate sits
    # O(mu) inside the active bounds (the harness then judges it through
    # feas / f_gap). The adapter's job is the multiplier translation, which
    # must make the stationarity residual vanish for the returned point.
    np.testing.assert_allclose(np.asarray(out.x), [1.0, 2.0, 3.0], atol=1e-3)
    metrics = quality_metrics(problem, out.x, out.dual, fstar=2.0)
    assert metrics["feas"] <= 1e-6
    assert metrics["stat_inf"] < 1e-4 and metrics["mult_min"] >= 0.0
    assert metrics["f_gap"] < 1e-2


@pytest.mark.parametrize(
    ("method", "status", "success", "expected"),
    [
        ("SLSQP", 0, True, "successful"),
        ("SLSQP", 9, False, "max_steps_reached"),
        ("SLSQP", 4, False, "inequality_constraints_incompatible"),
        ("SLSQP", 8, False, "positive_directional_derivative_for_linesearch"),
        ("SLSQP", 42, False, "slsqp_status_42"),
        ("trust-constr", 1, True, "successful"),
        ("trust-constr", 2, True, "successful"),
        ("trust-constr", 0, False, "max_steps_reached"),
        ("trust-constr", 3, False, "callback_raised_stopiteration"),
        ("trust-constr", 4, False, "constraint_violation_exceeds_gtol"),
    ],
)
def test_scipy_status_name(method, status, success, expected):
    assert scipy_status_name(method, status, success) == expected


def test_scipy_baseline_validation():
    assert set(METHODS) == {"SLSQP", "trust-constr"}
    with pytest.raises(ValueError, match="unknown SciPy baseline method"):
        ScipyBaseline("Nelder-Mead")
    with pytest.raises(ValueError, match="unknown SciPy baseline method"):
        scipy_status_name("Nelder-Mead", 0, True)
    assert not ScipyBaseline("SLSQP").exact_curvature
    assert ScipyBaseline("trust-constr").exact_curvature


def test_trust_constr_requires_exact_curvature():
    problem = build_problem(lambda x: jnp.sum(x**2), n=2, autodiff_mode="jax")
    assert not problem.has_exact_curvature
    with pytest.raises(ValueError, match="exact Hessian-vector products"):
        ScipyRunner(problem, jnp.ones(2), ScipyBaseline("trust-constr"))
    # SLSQP does not need them.
    ScipyRunner(problem, jnp.ones(2), ScipyBaseline("SLSQP"))


def test_dual_translations(kkt_problem):
    """Both translators reproduce the known multipliers of the conftest KKT point."""
    problem, x, dual, _ = kkt_problem
    # SLSQP: L = f - m^T c with c_eq = g  ->  m_eq = -lam; bounds recovered from r.
    from_slsqp = dual_from_slsqp(problem, x, -np.asarray(dual.eq_multipliers))
    assert from_slsqp is not None
    np.testing.assert_allclose(from_slsqp.eq_multipliers, dual.eq_multipliers)
    np.testing.assert_allclose(
        from_slsqp.lb_multipliers, dual.lb_multipliers, atol=1e-12
    )
    np.testing.assert_allclose(
        from_slsqp.ub_multipliers, dual.ub_multipliers, atol=1e-12
    )
    assert dual_from_slsqp(problem, x, None) is None
    # trust-constr: v_bounds = -z_lb + z_ub.
    v_bounds = -np.asarray(dual.lb_multipliers) + np.asarray(dual.ub_multipliers)
    from_tc = dual_from_trust_constr(
        problem,
        [np.asarray(dual.eq_multipliers), v_bounds],
        has_eq=True,
        has_ineq=False,
        has_bounds=True,
    )
    np.testing.assert_allclose(from_tc.eq_multipliers, dual.eq_multipliers)
    np.testing.assert_allclose(from_tc.lb_multipliers, dual.lb_multipliers)
    np.testing.assert_allclose(from_tc.ub_multipliers, dual.ub_multipliers)
    assert from_tc.ineq_multipliers.shape == (0,)
