"""Synthetic per-signal tests.

Each test hand-rolls just enough of an :class:`SLSQPState` /
:class:`StepSummary` to trip exactly one signal.  The cap-policy
enforcement test (``test_registry.py``) requires every registered
signal to have a ``test_signal_<name>_synthetic`` here; do not rename
without updating that test.
"""

from __future__ import annotations

import jax
import numpy as np

from slsqp_jax.diagnostics import signals as sig_mod
from slsqp_jax.results import RESULTS

jax.config.update("jax_enable_x64", True)


def _find_evaluator(name: str):
    for reg in sig_mod.SIGNAL_REGISTRY:
        if reg.name == name:
            return reg.evaluator
    raise LookupError(f"no evaluator registered with name {name!r}")


# ---------------------------------------------------------------------------
# Per-step signals
# ---------------------------------------------------------------------------


def test_signal_eq_jacobian_rank_deficient_synthetic(
    make_summary, make_state, make_eval_context
):
    eq_jac = np.array([[1.0, 0.0, 0.0], [1.0, 1e-15, 0.0]])  # near rank-1
    state = make_state(
        n=3,
        m_eq=2,
        eq_jac=eq_jac,
        eq_val=np.zeros(2),
    )
    summary_prev = make_summary(step_count=1, eq_jac_min_sv_est=float("inf"))
    summary_now = make_summary(step_count=2, eq_jac_min_sv_est=1e-12)
    summaries = [summary_prev, summary_now]

    ctx = make_eval_context()
    sig = _find_evaluator("eq_jacobian_rank_deficient")(
        ctx, state, summary_now, summaries
    )
    assert sig is not None
    assert sig.name == "eq_jacobian_rank_deficient"
    assert sig.specificity == "specific"
    assert sig.confidence in {"high", "medium"}
    assert "J_eq" in sig.artifacts
    assert "JJT" in sig.artifacts
    assert "singular_values" in sig.artifacts
    assert sig.offending_step == 2


def test_signal_eq_jacobian_rank_deficient_does_not_fire_when_clean(
    make_summary, make_state, make_eval_context
):
    state = make_state(n=3, m_eq=2, eq_jac=np.eye(3)[:2])
    summary_prev = make_summary(step_count=1, eq_jac_min_sv_est=1.0)
    summary_now = make_summary(step_count=2, eq_jac_min_sv_est=0.5)
    ctx = make_eval_context()
    sig = _find_evaluator("eq_jacobian_rank_deficient")(
        ctx, state, summary_now, [summary_prev, summary_now]
    )
    assert sig is None


def test_signal_lbfgs_conditioning_extreme_synthetic(
    make_summary, make_state, make_eval_context
):
    # Build a 3-summary streak above 1e6 plus a prior clean summary.
    summaries = [
        make_summary(step_count=1, diag_kappa=1e3),
        make_summary(step_count=2, diag_kappa=1e8),
        make_summary(step_count=3, diag_kappa=1e8),
        make_summary(step_count=4, diag_kappa=1e8),
    ]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("lbfgs_conditioning_extreme")(
        ctx, state, summaries[-1], summaries
    )
    assert sig is not None
    assert sig.name == "lbfgs_conditioning_extreme"
    assert "diagonal" in sig.artifacts
    assert sig.offending_step == 4


def test_signal_lbfgs_conditioning_extreme_streak_too_short(
    make_summary, make_state, make_eval_context
):
    summaries = [
        make_summary(step_count=1, diag_kappa=1e3),
        make_summary(step_count=2, diag_kappa=1e8),
    ]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("lbfgs_conditioning_extreme")(
        ctx, state, summaries[-1], summaries
    )
    assert sig is None


# ---------------------------------------------------------------------------
# End-of-run signals
# ---------------------------------------------------------------------------


def test_signal_multiplier_recovery_noise_synthetic(
    make_summary, make_state, make_eval_context
):
    n_steps = 20
    summaries = [
        make_summary(
            step_count=k + 1,
            grad_lagrangian_norm=1e-3,
            lagrangian_value=1.0,
            rel_kkt=1e-3,
            projected_grad_norm=1e-12,
        )
        for k in range(n_steps)
    ]
    state = make_state(
        n=4,
        diagnostics_overrides={
            "min_projected_grad_norm": 1e-12,
            "n_steps_inexact_below_classical": n_steps,
        },
    )
    ctx = make_eval_context(rtol=1e-6)
    sig = _find_evaluator("multiplier_recovery_noise")(
        ctx, state, summaries, RESULTS.merit_stagnation
    )
    assert sig is not None
    assert sig.name == "multiplier_recovery_noise"


def test_signal_multiplier_recovery_noise_does_not_fire_on_clean_run(
    make_summary, make_state, make_eval_context
):
    summaries = [
        make_summary(step_count=k + 1, projected_grad_norm=1.0) for k in range(10)
    ]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("multiplier_recovery_noise")(
        ctx, state, summaries, RESULTS.successful
    )
    assert sig is None


def test_signal_line_search_collapse_synthetic(
    make_summary, make_state, make_eval_context
):
    summaries = [
        make_summary(step_count=1, last_alpha=1.0, ls_success=True),
        make_summary(step_count=2, last_alpha=1e-12, ls_success=False),
    ]
    state = make_state(
        n=4,
        diagnostics_overrides={
            "tail_ls_failures": 5,
            "ls_alpha_min": 1e-12,
        },
    )
    ctx = make_eval_context()
    sig = _find_evaluator("line_search_collapse")(
        ctx, state, summaries, RESULTS.line_search_failure
    )
    assert sig is not None
    assert sig.name == "line_search_collapse"
    assert "alpha_window" in sig.artifacts
    assert "lbfgs_diagonal_final" in sig.artifacts
    assert sig.offending_step == 2


def test_signal_qp_budget_or_pingpong_synthetic(
    make_summary, make_state, make_eval_context
):
    summaries = [make_summary(step_count=k + 1) for k in range(10)]
    state = make_state(
        n=4,
        diagnostics_overrides={
            "n_qp_budget_exhausted": 4,
            "n_qp_ping_pong": 1,
        },
    )
    ctx = make_eval_context()
    sig = _find_evaluator("qp_budget_or_pingpong")(
        ctx, state, summaries, RESULTS.qp_subproblem_failure
    )
    assert sig is not None
    assert sig.name == "qp_budget_or_pingpong"
    assert sig.specificity == "generic"


def test_signal_merit_oscillation_synthetic(
    make_summary, make_state, make_eval_context
):
    summaries = [make_summary(step_count=k + 1) for k in range(20)]
    state = make_state(
        n=4,
        diagnostics_overrides={"n_merit_regressions": 10},
    )
    ctx = make_eval_context()
    sig = _find_evaluator("merit_oscillation")(
        ctx, state, summaries, RESULTS.merit_stagnation
    )
    assert sig is not None
    assert sig.name == "merit_oscillation"


def test_signal_lpeca_overpredicting_synthetic(
    make_summary, make_state, make_eval_context
):
    n_steps = 10
    summaries = [make_summary(step_count=k + 1) for k in range(n_steps)]
    state = make_state(
        n=4,
        diagnostics_overrides={
            "n_lpeca_capped": 8,  # > 50% capped
        },
    )
    ctx = make_eval_context()
    sig = _find_evaluator("lpeca_overpredicting")(
        ctx, state, summaries, RESULTS.qp_subproblem_failure
    )
    assert sig is not None
    assert sig.name == "lpeca_overpredicting"


def test_signal_infeasible_termination_synthetic(
    make_summary, make_state, make_eval_context
):
    summaries = [
        make_summary(step_count=k + 1, max_eq_violation=1e-3) for k in range(5)
    ]
    state = make_state(
        n=4,
        m_eq=2,
        eq_val=np.array([1e-3, 5e-4]),
        state_overrides={"termination_code": RESULTS.infeasible},
    )
    ctx = make_eval_context(atol=1e-6)
    sig = _find_evaluator("infeasible_termination")(
        ctx, state, summaries, RESULTS.infeasible
    )
    assert sig is not None
    assert sig.name == "infeasible_termination"
    assert "eq_values" in sig.artifacts
    assert "ineq_values" in sig.artifacts
