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


# ---------------------------------------------------------------------------
# Regression tests for the predicate fixes
# ---------------------------------------------------------------------------


def test_ls_floor_for_falls_back_when_solver_lacks_attribute(
    make_summary, make_state, make_eval_context
):
    """``_ls_floor_for`` must degrade to :data:`_LS_COLLAPSE_FALLBACK_FLOOR`
    when the solver instance does not expose ``line_search_max_steps``
    (e.g. a synthetic stand-in built without the SLSQP scaffolding).

    Pins the fallback branch in :func:`slsqp_jax.diagnostics.signals._ls_floor_for`.
    """
    from slsqp_jax.diagnostics.signals import (
        _LS_COLLAPSE_FALLBACK_FLOOR,
        EvalContext,
        _ls_floor_for,
    )

    class _BareSolver:
        rtol = 1e-6
        atol = 1e-6
        max_steps = 100
        # No ``line_search_max_steps`` attribute at all.

    ctx = EvalContext(
        solver=_BareSolver(),  # type: ignore[arg-type]
        rtol=1e-6,
        atol=1e-6,
        max_steps=100,
    )
    assert _ls_floor_for(ctx) == _LS_COLLAPSE_FALLBACK_FLOOR

    # And a solver whose attribute is non-int / non-positive must also
    # fall back -- the predicate uses ``isinstance(int) and > 0``.
    class _BogusSolver(_BareSolver):
        line_search_max_steps = 0  # invalid: <= 0 forces fallback

    ctx2 = EvalContext(
        solver=_BogusSolver(),  # type: ignore[arg-type]
        rtol=1e-6,
        atol=1e-6,
        max_steps=100,
    )
    assert _ls_floor_for(ctx2) == _LS_COLLAPSE_FALLBACK_FLOOR


def test_signal_line_search_collapse_synthetic_alpha_floor(
    make_summary, make_state, make_eval_context
):
    """The LS-collapse signal must fire at the LS floor ``2**-20``.

    Previously the hard-coded ``alpha_min < 1e-10`` threshold meant
    the signal could *never* fire for the default solver
    (``LineSearchConfig.max_steps == 20`` puts the floor at
    ``2**-20 ~= 9.5e-7``).  Regression test for the
    ``_ls_floor_for(ctx)`` predicate fix.
    """
    floor = 2.0**-20
    summaries = [
        make_summary(step_count=1, last_alpha=1.0, ls_success=True),
        make_summary(step_count=2, last_alpha=floor, ls_success=False),
    ]
    state = make_state(
        n=4,
        diagnostics_overrides={
            "tail_ls_failures": 4,
            "ls_alpha_min": floor,
        },
    )
    ctx = make_eval_context()
    sig = _find_evaluator("line_search_collapse")(
        ctx, state, summaries, RESULTS.line_search_failure
    )
    assert sig is not None
    assert sig.name == "line_search_collapse"
    # The predicate now keys on the LS floor, not the deprecated
    # ``1e-10`` literal — the evidence dict reflects that.
    assert "ls_floor" in sig.evidence
    assert sig.evidence["ls_alpha_min"] == floor


def test_signal_lbfgs_conditioning_extreme_synthetic_burst(
    make_summary, make_state, make_eval_context
):
    """A single-step kappa burst above ``1e8`` must fire even without a streak.

    Regression test for the burst clause added to
    ``_eval_lbfgs_conditioning_extreme``: a one-shot blow-up to
    ``kappa = 1e9`` (followed by an immediate L-BFGS reset) does
    *not* satisfy the 3-step streak gate, but is the failure mode
    that the diagnostic notes for the feasible-start divergence run
    flagged.
    """
    summaries = [
        make_summary(step_count=1, diag_kappa=1.0),
        make_summary(step_count=2, diag_kappa=1.0),
        make_summary(step_count=3, diag_kappa=1e9),
    ]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("lbfgs_conditioning_extreme")(
        ctx, state, summaries[-1], summaries
    )
    assert sig is not None
    assert sig.name == "lbfgs_conditioning_extreme"
    assert sig.evidence["burst_clause"] == 1
    assert sig.offending_step == 3


# ---------------------------------------------------------------------------
# New end-of-run signals
# ---------------------------------------------------------------------------


def test_signal_divergence_rollback_triggered_synthetic(
    make_summary, make_state, make_eval_context
):
    """The rollback signal fires when ``divergence_triggered`` is latched."""
    summaries = [make_summary(step_count=1)]
    summaries.append(make_summary(step_count=2, blowup_count=1))
    summaries.append(make_summary(step_count=3, blowup_count=2))
    summaries.append(make_summary(step_count=4, blowup_count=3, diverging=True))
    state = make_state(
        n=4,
        diagnostics_overrides={
            "divergence_triggered": True,
            "n_divergence_blowups": 4,
        },
    )
    ctx = make_eval_context()
    sig = _find_evaluator("divergence_rollback_triggered")(
        ctx, state, summaries, RESULTS.iterate_blowup
    )
    assert sig is not None
    assert sig.name == "divergence_rollback_triggered"
    assert sig.specificity == "specific"
    assert sig.evidence["divergence_triggered"] == 1
    assert sig.evidence["n_divergence_blowups"] == 4
    # The first blow-up was at step 2 (``blowup_count`` went 0 -> 1).
    assert sig.offending_step == 2


def test_signal_divergence_rollback_triggered_does_not_fire_when_clean(
    make_summary, make_state, make_eval_context
):
    """Clean runs (no blow-ups) must not trip the rollback signal."""
    summaries = [make_summary(step_count=k + 1) for k in range(5)]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("divergence_rollback_triggered")(
        ctx, state, summaries, RESULTS.successful
    )
    assert sig is None


def test_signal_divergence_rollback_offending_step_via_diverging_flag(
    make_summary, make_state, make_eval_context
):
    """Offending-step locator hits the ``diverging`` short-circuit branch.

    The locator tries the ``diverging`` flag first (immediate exit) and
    only falls back to ``blowup_count > prev_blowup`` when no summary
    has ``diverging`` set.  This test pins the ``diverging`` branch.
    """
    summaries = [
        make_summary(step_count=1),
        make_summary(step_count=2, diverging=True, blowup_count=0),
        make_summary(step_count=3, diverging=True, blowup_count=1),
    ]
    state = make_state(
        n=4,
        diagnostics_overrides={
            "divergence_triggered": True,
            "n_divergence_blowups": 2,
        },
    )
    ctx = make_eval_context()
    sig = _find_evaluator("divergence_rollback_triggered")(
        ctx, state, summaries, RESULTS.iterate_blowup
    )
    assert sig is not None
    assert sig.offending_step == 2


def test_signal_merit_penalty_explosion_synthetic(
    make_summary, make_state, make_eval_context
):
    """``rho`` jumping by 1e8 in one step must trip the explosion signal."""
    rhos = [1.0, 1.0, 1.0, 1.0, 1.0, 1e8]
    summaries = [
        make_summary(step_count=k + 1, merit_penalty=rho) for k, rho in enumerate(rhos)
    ]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("merit_penalty_explosion")(
        ctx, state, summaries, RESULTS.iterate_blowup
    )
    assert sig is not None
    assert sig.name == "merit_penalty_explosion"
    assert sig.magnitude == "extreme"
    # The largest one-step jump is at step 6 (``rhos[5] / rhos[4]``).
    assert sig.evidence["jump_step"] == 6
    assert sig.evidence["rho_max"] == 1e8
    assert "rho_trajectory" in sig.artifacts


def test_signal_merit_penalty_explosion_does_not_fire_on_steady_rho(
    make_summary, make_state, make_eval_context
):
    """A steadily-growing ``rho`` (well below the explosion threshold) is benign."""
    rhos = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    summaries = [
        make_summary(step_count=k + 1, merit_penalty=rho) for k, rho in enumerate(rhos)
    ]
    state = make_state(n=4)
    ctx = make_eval_context()
    sig = _find_evaluator("merit_penalty_explosion")(
        ctx, state, summaries, RESULTS.successful
    )
    assert sig is None


def test_signal_penalty_starvation_synthetic(
    make_summary, make_state, make_eval_context
):
    """``rho`` frozen while feasibility drifts upward must trip starvation."""
    # 9 steps with constant ``rho = 1.0`` while ``max_eq_violation``
    # walks monotonically from ~atol up by ~100x.  The frozen prefix
    # is the entire run, so the prefix length is 9 (>= ``n_steps // 3``
    # and >= 5 absolute floor).
    violations = [1e-7, 2e-7, 4e-7, 8e-7, 2e-6, 4e-6, 8e-6, 1e-5, 2e-5]
    summaries = [
        make_summary(step_count=k + 1, merit_penalty=1.0, max_eq_violation=v)
        for k, v in enumerate(violations)
    ]
    state = make_state(n=4)
    ctx = make_eval_context(atol=1e-7)
    sig = _find_evaluator("penalty_starvation")(
        ctx, state, summaries, RESULTS.infeasible
    )
    assert sig is not None
    assert sig.name == "penalty_starvation"
    assert sig.evidence["frozen_prefix_steps"] == 9
    assert sig.evidence["rho_initial"] == 1.0
    assert "rho_prefix" in sig.artifacts
    assert "violations_prefix" in sig.artifacts


def test_signal_penalty_starvation_does_not_fire_when_rho_grew(
    make_summary, make_state, make_eval_context
):
    """If ``rho`` did grow during the prefix, the starvation signal must not fire."""
    rhos = [1.0, 1.0, 1.0, 5.0, 10.0, 50.0]
    violations = [1e-7, 2e-7, 4e-7, 8e-7, 2e-6, 4e-6]
    summaries = [
        make_summary(step_count=k + 1, merit_penalty=rho, max_eq_violation=v)
        for k, (rho, v) in enumerate(zip(rhos, violations))
    ]
    state = make_state(n=4)
    ctx = make_eval_context(atol=1e-7)
    sig = _find_evaluator("penalty_starvation")(
        ctx, state, summaries, RESULTS.merit_stagnation
    )
    # The frozen prefix is only 3 steps long (rhos[3] = 5.0 breaks
    # it), which is below the ``max(5, n_steps // 3)`` minimum.
    assert sig is None


def test_signal_penalty_starvation_does_not_fire_when_violations_non_monotone(
    make_summary, make_state, make_eval_context
):
    """A non-monotone feasibility trajectory must not trip starvation.

    The signal requires the violation series to be monotone non-
    decreasing across the frozen prefix (the load-bearing notion is
    "feasibility decayed silently").  A trajectory that *bounces*
    isn't a starvation event and should return ``None`` from the
    monotone check before the growth-ratio test fires.
    """
    # Frozen rho prefix of 9 steps; violations rise then fall, so
    # the monotone non-decreasing test fails.
    violations = [1e-7, 2e-7, 4e-7, 8e-7, 4e-7, 2e-6, 4e-6, 8e-6, 1e-5]
    summaries = [
        make_summary(step_count=k + 1, merit_penalty=1.0, max_eq_violation=v)
        for k, v in enumerate(violations)
    ]
    state = make_state(n=4)
    ctx = make_eval_context(atol=1e-7)
    sig = _find_evaluator("penalty_starvation")(
        ctx, state, summaries, RESULTS.merit_stagnation
    )
    assert sig is None
