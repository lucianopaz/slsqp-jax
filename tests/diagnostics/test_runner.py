"""Tests for the manual debug-run loop.

Validates that :func:`debug_run` produces a :class:`DebugRunResult`
that (a) records one :class:`StepSummary` per iteration, (b) terminates
at the same iterate as the production ``optimistix.minimise`` driver
on the same problem, and (c) honours the ``max_steps`` override.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax import SLSQP
from slsqp_jax.diagnostics import debug_run
from slsqp_jax.diagnostics.records import DebugRunResult, StepSummary

jax.config.update("jax_enable_x64", True)


def _rosenbrock(x, args):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


@pytest.fixture
def rosenbrock_solver():
    return SLSQP()


@pytest.fixture
def rosenbrock_x0():
    return jnp.array([-1.2, 1.0, -0.5, 1.5])


def test_debug_run_returns_debug_run_result(rosenbrock_solver, rosenbrock_x0):
    run = debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=20)
    assert isinstance(run, DebugRunResult)
    assert run.n_steps == len(run.summaries)
    assert all(isinstance(s, StepSummary) for s in run.summaries)


def test_debug_run_records_one_summary_per_step(rosenbrock_solver, rosenbrock_x0):
    run = debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=10)
    # The loop ran at most ``max_steps`` iterations.
    assert run.n_steps <= 10
    # Step counts inside the summaries are 1-indexed and strictly
    # increasing.
    step_counts = [s.step_count for s in run.summaries]
    assert step_counts == list(range(1, len(step_counts) + 1))


def test_debug_run_max_steps_caps_iterations(rosenbrock_solver, rosenbrock_x0):
    run = debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=3)
    assert run.n_steps == 3
    # On Rosenbrock, 3 iterations is far short of convergence; the
    # ``terminated_successfully`` property must reflect the budget
    # exhaustion rather than the latent default ``successful`` code.
    assert run.max_steps_reached
    assert not run.terminated_successfully


def test_debug_run_matches_optimistix_minimise(rosenbrock_x0):
    """The manual debug loop converges to the same iterate as ``optx.minimise``."""
    # Use a solver with a wider iteration budget than the default so
    # the assertion below holds across hardware / jaxlib versions:
    # the default ``max_steps=100`` is enough on most environments
    # but not all, and a flaky terminal-state assertion would tell us
    # nothing about the runner under test.
    from slsqp_jax import SLSQPConfig
    from slsqp_jax.config import ToleranceConfig

    solver = SLSQP(
        config=SLSQPConfig(tolerance=ToleranceConfig(max_steps=400)),
    )

    def fn(x, args):
        return _rosenbrock(x, args)

    sol = optx.minimise(
        fn,
        solver,
        rosenbrock_x0,
        max_steps=400,
        throw=False,
    )
    run = debug_run(solver, fn, rosenbrock_x0, max_steps=400)

    assert run.terminated_successfully
    assert jnp.allclose(run.final_y, sol.value, atol=1e-6, rtol=1e-6)


def test_debug_run_supports_has_aux(rosenbrock_solver, rosenbrock_x0):
    """When ``has_aux=True`` the runner accepts an objective returning ``(value, aux)``."""

    def fn_with_aux(x, args):
        return _rosenbrock(x, args), {"step_count": 0}

    run = debug_run(
        rosenbrock_solver,
        fn_with_aux,
        rosenbrock_x0,
        max_steps=10,
        has_aux=True,
    )
    assert isinstance(run, DebugRunResult)
    assert run.n_steps > 0


def test_debug_run_invalid_max_steps_raises(rosenbrock_solver, rosenbrock_x0):
    with pytest.raises(ValueError, match="max_steps must be positive"):
        debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=0)


def test_debug_run_defaults_max_steps_to_solver_value(rosenbrock_x0):
    """When ``max_steps`` is omitted the loop should honour ``solver.max_steps``.

    Exercises the ``_resolve_max_steps`` ``max_steps is None`` branch and
    verifies the resulting budget matches the solver's own default.
    """
    from slsqp_jax import SLSQPConfig
    from slsqp_jax.config import ToleranceConfig

    solver = SLSQP(config=SLSQPConfig(tolerance=ToleranceConfig(max_steps=4)))
    run = debug_run(solver, _rosenbrock, rosenbrock_x0)
    # The solver itself terminates at step 4 via max_iters_reached, so
    # the loop exits via ``break`` rather than budget-exhaustion.  What
    # this proves is that the runner *did* honour ``solver.max_steps=4``
    # (otherwise the run would have continued and converged).
    assert run.n_steps == 4
    assert not run.terminated_successfully


def test_debug_run_records_per_step_signal_only_once(rosenbrock_solver, rosenbrock_x0):
    """A per-step evaluator that returns the *same* signal name on every
    step must show up exactly once in ``run.fired_signals`` — the
    runner deduplicates by ``signal.name`` to keep the report tidy."""

    from slsqp_jax.diagnostics.signals import Signal

    fire_count = {"n": 0}

    def always_fires(_ctx, _state, _summary, summaries):
        fire_count["n"] += 1
        return Signal(
            name="dummy_per_step_signal",
            specificity="generic",
            magnitude="marginal",
            confidence="low",
            summary="x",
            detail="x",
            evidence={},
            suggestions=[],
            artifacts={},
            offending_step=int(summaries[-1].step_count),
        )

    run = debug_run(
        rosenbrock_solver,
        _rosenbrock,
        rosenbrock_x0,
        max_steps=4,
        per_step_evaluators=(always_fires,),
    )
    # Evaluator was invoked at every step ...
    assert fire_count["n"] == run.n_steps
    # ... but the dedup-by-name guarantee keeps a single entry on the run.
    names = [s.name for s in run.fired_signals]
    assert names.count("dummy_per_step_signal") == 1


def test_diff_summaries_treats_both_nan_as_equal():
    """``_diff_summaries`` must skip fields where both values are NaN.

    This is the load-bearing NaN-aware branch in the reproducibility
    diff helper: a doubly-NaN field carries no useful diagnostic, so
    flagging it as ``a != b`` would produce a misleading diff in a
    real ``capture_state_at_step`` failure.
    """
    import dataclasses

    from slsqp_jax.diagnostics.records import StepSummary
    from slsqp_jax.diagnostics.runner import _diff_summaries

    base_kwargs: dict = {f.name: f.default for f in dataclasses.fields(StepSummary)}
    # Build two sentinels with identical structure but a NaN in two
    # places: one shared, one differing.
    a_kwargs = {
        **base_kwargs,
        "step_count": 1,
        "f_val": float("nan"),  # both NaN -- must be skipped
        "merit": float("nan"),  # only `a` is NaN -- must be reported
        "last_alpha": 0.5,
        "qp_iterations_total": 0,
        "qp_iterations_step": 0,
        "qp_converged": True,
        "qp_real_failure": False,
        "qp_reached_max_iter": False,
        "qp_ping_ponged": False,
        "ls_success": True,
        "consecutive_qp_failures": 0,
        "consecutive_ls_failures": 0,
        "consecutive_zero_steps": 0,
        "grad_norm": 1.0,
        "grad_lagrangian_norm": 1.0,
        "lagrangian_value": 0.0,
        "rel_kkt": 1.0,
        "gamma": 1.0,
        "min_diag": 1.0,
        "max_diag": 1.0,
        "diag_kappa": 1.0,
        "lbfgs_count": 0,
        "lbfgs_skipped": False,
        "max_abs_mult_eq": 0.0,
        "max_abs_mult_ineq": 0.0,
        "qp_vs_ls_multiplier_ratio": 1.0,
        "n_active_ineq": 0,
        "eq_jac_min_sv_est": float("inf"),
        "projected_grad_norm": float("inf"),
        "merit_penalty": 1.0,
        "max_eq_violation": 0.0,
        "max_ineq_violation": 0.0,
        "proj_residual_high_water": 0.0,
        "diverging": False,
        "blowup_count": 0,
        "merit_regression_step": False,
    }
    b_kwargs = {**a_kwargs, "merit": 1.0}
    a = StepSummary(**a_kwargs)
    b = StepSummary(**b_kwargs)
    diff = _diff_summaries(a, b)
    assert "f_val" not in diff  # both NaN -- skipped
    assert "merit" in diff
    assert diff["merit"][1] == 1.0
