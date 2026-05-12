"""Smoke tests for :class:`DebugReport` rendering and serialisation.

Validates that the renderer produces a plausible string for clean,
budget-exhausted, signal-firing, and out-of-scope cases without
crashing, and that ``to_dict()`` yields a JSON-serialisable structure.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from slsqp_jax import SLSQP, debug_run, diagnose
from slsqp_jax.diagnostics.report import (
    DebugReport,
    _result_message,
    _result_name,
)
from slsqp_jax.diagnostics.signals import Signal
from slsqp_jax.results import RESULTS

jax.config.update("jax_enable_x64", True)


def _rosenbrock(x, args):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def test_report_renders_clean_run():
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0, -0.5, 1.5])
    report = diagnose(solver, _rosenbrock, x0, max_steps=200)
    text = report.render()
    assert "SLSQP-JAX Debug Report" in text
    assert "Termination" in text
    assert "successful" in text
    assert "SLSQPDiagnostics counters" in text
    assert "Trajectory" in text


def test_report_renders_budget_exhausted_run_with_warning():
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0, -0.5, 1.5])
    report = diagnose(solver, _rosenbrock, x0, max_steps=5)
    text = report.render()
    assert "EXHAUSTED" in text
    assert "default" in text  # the warning paragraph mentions the default code


def test_report_renders_with_synthetic_signal():
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0])
    run = debug_run(solver, _rosenbrock, x0, max_steps=10)
    sig = Signal(
        name="eq_jacobian_rank_deficient",
        specificity="specific",
        magnitude="extreme",
        confidence="high",
        summary="J_eq looks suspicious because of X.",
        detail="Detail.",
        evidence={"sigma_min": 1e-12, "threshold": 1e-8},
        suggestions=["try MinresQLPSolver"],
        artifacts={"JJT": jnp.eye(2)},
        offending_step=2,
    )
    run.fired_signals.append(sig)
    report = DebugReport.from_run(run)
    text = report.render()
    # Signals section appears under in-scope partition.
    assert "Fired signals (in scope" in text
    assert "eq_jacobian_rank_deficient" in text
    assert "JJT" in text  # artifact key listed
    assert "try MinresQLPSolver" in text


def test_report_to_dict_is_json_serialisable():
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0])
    report = diagnose(solver, _rosenbrock, x0, max_steps=10)
    payload = report.to_dict()
    # Must round-trip through json without raising.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert "termination" in decoded
    assert "diagnostics" in decoded
    assert "signals" in decoded


def test_result_name_resolves_known_codes():
    assert _result_name(RESULTS.successful) == "successful"
    assert _result_name(RESULTS.merit_stagnation) == "merit_stagnation"


def test_result_message_returns_documented_string():
    msg = _result_message(RESULTS.merit_stagnation)
    assert "L1 merit" in msg
