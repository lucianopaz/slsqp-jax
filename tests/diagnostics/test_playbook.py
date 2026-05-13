"""Tests for the playbook rule engine, scope filter, and confidence lookup."""

from __future__ import annotations

import jax
import pytest

from slsqp_jax.diagnostics.playbook import (
    SCOPE_BY_TERMINATION,
    Diagnosis,
    confidence_for,
    evaluate_diagnoses,
    magnitude_for,
    signals_in_scope,
    signals_out_of_scope,
)
from slsqp_jax.diagnostics.signals import Signal
from slsqp_jax.results import RESULTS

jax.config.update("jax_enable_x64", True)


def _signal(name: str, **overrides) -> Signal:
    return Signal(
        name=name,
        specificity=overrides.get("specificity", "specific"),
        magnitude=overrides.get("magnitude", "moderate"),
        confidence=overrides.get("confidence", "high"),
        summary="X",
        detail="Y",
        evidence=overrides.get("evidence", {}),
        suggestions=overrides.get("suggestions", []),
        artifacts=overrides.get("artifacts", {}),
        offending_step=overrides.get("offending_step", None),
    )


# ---------------------------------------------------------------------------
# Confidence lookup table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("specificity", "magnitude", "expected"),
    [
        ("specific", "marginal", "medium"),
        ("specific", "moderate", "high"),
        ("specific", "extreme", "high"),
        ("ambiguous", "marginal", "low"),
        ("ambiguous", "moderate", "medium"),
        ("ambiguous", "extreme", "medium"),
        ("generic", "marginal", "low"),
        ("generic", "moderate", "low"),
        ("generic", "extreme", "low"),
    ],
)
def test_confidence_for_table(specificity, magnitude, expected):
    assert confidence_for(specificity, magnitude) == expected


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.0, "marginal"),
        (5.0, "marginal"),
        (10.0, "moderate"),
        (50.0, "moderate"),
        (100.0, "extreme"),
        (1e6, "extreme"),
        (float("nan"), "marginal"),
        (-1.0, "marginal"),
    ],
)
def test_magnitude_for_buckets(ratio, expected):
    assert magnitude_for(ratio) == expected


# ---------------------------------------------------------------------------
# Scope filter
# ---------------------------------------------------------------------------


def test_scope_filter_partitions_signals_by_termination_code():
    fired = {
        "eq_jacobian_rank_deficient",
        "qp_budget_or_pingpong",
        "merit_oscillation",
    }
    in_scope = signals_in_scope(RESULTS.qp_subproblem_failure, fired)
    out_of_scope = signals_out_of_scope(RESULTS.qp_subproblem_failure, fired)
    assert in_scope == {"eq_jacobian_rank_deficient", "qp_budget_or_pingpong"}
    assert out_of_scope == {"merit_oscillation"}
    assert in_scope.isdisjoint(out_of_scope)


def test_scope_filter_treats_empty_scope_as_everything_in():
    """``RESULTS.successful`` and ``..._max_steps_reached`` use empty
    scope as the "everything in scope" sentinel."""
    fired = {"eq_jacobian_rank_deficient", "qp_budget_or_pingpong"}
    assert signals_in_scope(RESULTS.successful, fired) == fired
    assert signals_in_scope(RESULTS.nonlinear_max_steps_reached, fired) == fired
    assert signals_out_of_scope(RESULTS.successful, fired) == set()


def test_scope_table_covers_every_documented_termination_code():
    """Every granular ``slsqp_jax.RESULTS`` failure code should have an
    explicit entry in :data:`SCOPE_BY_TERMINATION`.

    A future addition to the granular RESULTS enum that forgets to
    update :data:`SCOPE_BY_TERMINATION` would silently fall through
    to the "everything in scope" sentinel; this test catches that.
    """
    documented_failure_codes = (
        RESULTS.merit_stagnation,
        RESULTS.line_search_failure,
        RESULTS.qp_subproblem_failure,
        RESULTS.iterate_blowup,
        RESULTS.infeasible,
    )
    from slsqp_jax.diagnostics.playbook import _enum_value

    for code in documented_failure_codes:
        assert _enum_value(code) in SCOPE_BY_TERMINATION


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------


def test_evaluate_diagnoses_empty_when_no_signals():
    assert evaluate_diagnoses([]) == []


def test_stale_lbfgs_curvature_fires_when_lbfgs_extreme_and_ls_collapse():
    sigs = [
        _signal("lbfgs_conditioning_extreme", specificity="ambiguous"),
        _signal("line_search_collapse", specificity="ambiguous"),
    ]
    diags = evaluate_diagnoses(sigs)
    names = [d.name for d in diags]
    assert "stale_lbfgs_curvature" in names


def test_stale_lbfgs_curvature_fires_when_lbfgs_extreme_and_merit_oscillation():
    sigs = [
        _signal("lbfgs_conditioning_extreme"),
        _signal("merit_oscillation", specificity="generic"),
    ]
    diags = evaluate_diagnoses(sigs)
    names = [d.name for d in diags]
    assert "stale_lbfgs_curvature" in names


def test_stale_lbfgs_curvature_does_not_fire_alone():
    diags = evaluate_diagnoses([_signal("lbfgs_conditioning_extreme")])
    assert all(d.name != "stale_lbfgs_curvature" for d in diags)


def test_active_set_churn_fires_on_pair():
    sigs = [
        _signal("eq_jacobian_rank_deficient"),
        _signal("qp_budget_or_pingpong", specificity="generic"),
    ]
    diags = evaluate_diagnoses(sigs)
    names = [d.name for d in diags]
    assert "active_set_churn" in names


def test_noise_floor_stall_fires_alone():
    sigs = [_signal("multiplier_recovery_noise")]
    diags = evaluate_diagnoses(sigs)
    names = [d.name for d in diags]
    assert "noise_floor_stationarity_stall" in names


def test_multiple_diagnoses_can_fire_together():
    sigs = [
        _signal("lbfgs_conditioning_extreme"),
        _signal("merit_oscillation", specificity="generic"),
        _signal("multiplier_recovery_noise"),
    ]
    diags = evaluate_diagnoses(sigs)
    names = {d.name for d in diags}
    assert {"stale_lbfgs_curvature", "noise_floor_stationarity_stall"} <= names


def test_diagnosis_dataclass_carries_related_signals():
    sigs = [
        _signal("lbfgs_conditioning_extreme"),
        _signal("line_search_collapse", specificity="ambiguous"),
    ]
    diags = evaluate_diagnoses(sigs)
    stale = next(d for d in diags if d.name == "stale_lbfgs_curvature")
    assert isinstance(stale, Diagnosis)
    assert "lbfgs_conditioning_extreme" in stale.related_signals
    assert "line_search_collapse" in stale.related_signals


# ---------------------------------------------------------------------------
# Defensive helpers
# ---------------------------------------------------------------------------


def test_enum_value_returns_none_on_non_enum_inputs():
    """``_enum_value`` must degrade gracefully (no exceptions) when the
    input has no ``_value`` attribute or carries a non-int value."""
    from slsqp_jax.diagnostics.playbook import _enum_value

    assert _enum_value(object()) is None
    # ``_value`` exists but cannot be coerced to ``int``.

    class _Bogus:
        _value = "not-int"

    assert _enum_value(_Bogus()) is None


def test_scope_for_returns_none_on_non_enum():
    """``_scope_for`` short-circuits via :func:`_enum_value` -> ``None``."""
    from slsqp_jax.diagnostics.playbook import _scope_for

    assert _scope_for(object()) is None


def test_signals_in_scope_falls_back_to_everything_for_unknown_code():
    """A non-enum termination code must land in the "everything in scope"
    fallback so the renderer can still produce something useful."""
    fired = {"a", "b", "c"}
    assert signals_in_scope(object(), fired) == fired


def test_magnitude_for_handles_non_numeric_input():
    """Strings (or anything that fails ``float(...)`` coercion) must
    fall back to ``"marginal"`` rather than raise."""
    assert magnitude_for("not-a-number") == "marginal"  # type: ignore[arg-type]
