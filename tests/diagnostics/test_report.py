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


# ---------------------------------------------------------------------------
# Defensive helpers — exercise the ``None`` / non-enum branches
# ---------------------------------------------------------------------------


def test_result_name_handles_none():
    assert _result_name(None) == "<none>"


def test_result_message_handles_none():
    assert _result_message(None) == ""


def test_result_name_falls_back_to_str_for_non_enum():
    """Anything that is not an ``equinox.Enumeration`` item falls
    back to ``str(result)``."""

    class _NotAnEnum:
        def __str__(self) -> str:
            return "fake-result-string"

    assert _result_name(_NotAnEnum()) == "fake-result-string"


def test_result_message_returns_empty_for_non_enum():
    assert _result_message(object()) == ""


def test_enum_value_returns_none_on_non_enum():
    """``_enum_value`` is the load-bearing helper for both name and
    message lookups; non-enum inputs must return ``None`` rather than
    raise."""
    from slsqp_jax.diagnostics.report import _enum_value

    assert _enum_value(object()) is None
    assert _enum_value(None) is None


def test_enum_value_returns_none_on_uncoercible_value():
    class _BogusEnum:
        _value = "not-an-int"

    from slsqp_jax.diagnostics.report import _enum_value

    assert _enum_value(_BogusEnum()) is None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def test_fmt_e_handles_nan_and_inf_and_non_numeric():
    from slsqp_jax.diagnostics.report import _fmt_e

    assert _fmt_e(float("nan")) == "nan"
    assert _fmt_e(float("inf")) == "+inf"
    assert _fmt_e(float("-inf")) == "-inf"
    # Non-numeric inputs degrade gracefully via ``str(value)``.
    assert _fmt_e("hello") == "hello"


def test_fmt_value_handles_each_branch():
    from slsqp_jax.diagnostics.report import _fmt_value

    assert _fmt_value(True) == "True"
    assert _fmt_value(False) == "False"
    assert _fmt_value(7) == "7"
    assert _fmt_value(1.5) == "1.500e+00"
    # Anything else falls back to ``str(value)``.
    assert _fmt_value([1, 2]) == "[1, 2]"


# ---------------------------------------------------------------------------
# Out-of-scope signals + diagnoses rendering
# ---------------------------------------------------------------------------


def test_report_renders_out_of_scope_signals_under_dedicated_section():
    """A signal that is not in scope for the run's termination code
    must still be listed, but under a dedicated "less likely given
    the termination mode" sub-section.
    """
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0])
    run = debug_run(solver, _rosenbrock, x0, max_steps=2)
    # Force the termination_code to ``infeasible`` so the in-scope
    # set is the small {infeasible_termination, eq_jacobian_rank_deficient}
    # bucket.  An unrelated signal must then render in the
    # out-of-scope section.
    import equinox as eqx

    final_state = eqx.tree_at(
        lambda s: s.termination_code, run.final_state, RESULTS.infeasible
    )
    run.final_state = final_state
    run.final_result = RESULTS.infeasible

    # An out-of-scope signal that does NOT match the infeasible scope.
    sig = Signal(
        name="merit_oscillation",
        specificity="generic",
        magnitude="moderate",
        confidence="low",
        summary="Merit kept regressing.",
        detail="Detail.",
        evidence={},
        suggestions=[],
        artifacts={},
        offending_step=None,
    )
    run.fired_signals.append(sig)

    report = DebugReport.from_run(run)
    text = report.render()
    assert "less likely given the termination mode" in text
    assert "merit_oscillation" in text
    # And the in-scope section flagged that nothing in-scope fired.
    assert "no in-scope signals fired" in text


def test_report_renders_diagnoses_block():
    """``_render_diagnoses`` is exercised when the report carries any
    :class:`Diagnosis` instances on it."""
    from slsqp_jax.diagnostics.playbook import Diagnosis

    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0])
    run = debug_run(solver, _rosenbrock, x0, max_steps=4)
    sig = Signal(
        name="lbfgs_conditioning_extreme",
        specificity="ambiguous",
        magnitude="extreme",
        confidence="high",
        summary="L-BFGS conditioning is extreme.",
        detail="Detail.",
        evidence={},
        suggestions=[],
        artifacts={},
        offending_step=None,
    )
    run.fired_signals.append(sig)
    report = DebugReport.from_run(run)
    report.diagnoses = [
        Diagnosis(
            name="example_diagnosis",
            cause="A short reason that something went wrong with X.",
            suggestions=["try Y", "try Z"],
            related_signals=["lbfgs_conditioning_extreme"],
        )
    ]
    text = report.render()
    assert "Candidate diagnoses" in text
    assert "example_diagnosis" in text
    assert "related signals: lbfgs_conditioning_extreme" in text
    assert "try Y" in text


def test_diagnosis_to_dict_round_trips_through_json():
    """``_diagnosis_to_dict`` produces a JSON-serialisable structure."""
    from slsqp_jax.diagnostics.playbook import Diagnosis
    from slsqp_jax.diagnostics.report import _diagnosis_to_dict

    diag = Diagnosis(
        name="example",
        cause="Because X.",
        suggestions=["a", "b"],
        related_signals=["sig1", "sig2"],
    )
    payload = _diagnosis_to_dict(diag)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["name"] == "example"
    assert decoded["related_signals"] == ["sig1", "sig2"]


# ---------------------------------------------------------------------------
# Trajectory / metrics empty-summaries early-return branches
# ---------------------------------------------------------------------------


def test_render_summary_metrics_empty_summaries_returns_quietly():
    """``_render_summary_metrics`` must short-circuit when the run has
    no per-step summaries (e.g. the solver terminated at step 0)."""
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0])
    run = debug_run(solver, _rosenbrock, x0, max_steps=4)
    run.summaries = []
    report = DebugReport.from_run(run)
    text = report.render()
    # Both early-returning sections are skipped silently.
    assert "Final iterate metrics" not in text
    assert "Trajectory" not in text


def test_render_termination_with_distinct_coarse_and_granular_codes():
    """The header should print the coarse code on its own line when
    it differs from the granular code (e.g. ``infeasible`` -> coarse
    ``nonlinear_divergence``)."""
    import equinox as eqx
    import optimistix as optx

    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0])
    run = debug_run(solver, _rosenbrock, x0, max_steps=4)
    final_state = eqx.tree_at(
        lambda s: s.termination_code, run.final_state, RESULTS.infeasible
    )
    run.final_state = final_state
    run.final_result = RESULTS.infeasible
    run.coarse_result = optx.RESULTS.nonlinear_divergence
    report = DebugReport.from_run(run)
    text = report.render()
    assert "granular RESULTS code: infeasible" in text
    assert "coarse optx code:" in text
