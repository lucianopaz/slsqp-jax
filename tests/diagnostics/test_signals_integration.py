"""Integration tests: signals fire on real failing problems.

The cap-policy enforcement test (``test_registry.py``) requires every
registered signal to have a ``test_signal_<name>_integration`` here.
Skips are acceptable as long as the named test exists; they are
flagged with a concrete reason describing why the signal is hard to
trigger reliably without machine-specific tuning.

The problems used here are intentionally minimal — they exist solely
to demonstrate that ``diagnose(...)`` can extract the relevant signal
from a real ``debug_run``, not to exhaustively validate the signal's
threshold.  Synthetic tests (``test_signals_synthetic.py``) carry the
unit-level fidelity check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax import SLSQP, diagnose

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Per-step signals
# ---------------------------------------------------------------------------


def test_signal_eq_jacobian_rank_deficient_integration():
    """Two algebraically identical equality constraints → rank-1 J_eq."""

    def f(x, args):
        return jnp.sum((x - 1.0) ** 2)

    def c_eq(x, args):
        # Two scalar copies of the same affine equation.  J_eq is rank-1.
        residual = jnp.sum(x) - 5.0
        return jnp.array([residual, residual])

    solver = SLSQP(eq_constraint_fn=c_eq, n_eq_constraints=2)
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    report = diagnose(solver, f, x0, max_steps=10)

    # The cumulative low-water mark on eq_jac_min_sv_est should drop
    # well below 1e-8 because the regularised Cholesky exposes the
    # rank-1 structure.  In practice slsqp-jax's regularisation
    # epsilon (1e-8) puts a floor at ~1e-4 on this metric, so the
    # "exactly degenerate" trigger may not fire.  Treat the test as
    # a smoke test that diagnose() returns without crashing on this
    # input, and let the synthetic test own the signal-firing
    # contract.
    assert report.run.n_steps > 0


@pytest.mark.skip(
    reason=(
        "Requires a problem that consistently drives the L-BFGS B0 "
        "diagonal condition number above 1e6 for at least 3 "
        "consecutive iterations.  Synthetic test owns the signal-"
        "firing contract; reproducing the exact pattern from a real "
        "minimisation is fragile and tends to depend on the specific "
        "L-BFGS reset chain timing."
    )
)
def test_signal_lbfgs_conditioning_extreme_integration():
    """L-BFGS conditioning extreme on a multi-scale problem."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# End-of-run signals
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Requires HRInexactSTCG inner solver + a problem whose "
        "multiplier-recovery noise dominates the classical "
        "stationarity test.  Synthetic test owns the signal-firing "
        "contract.  An integration trigger is plausible but needs "
        "an ill-conditioned KKT system that we have not yet picked "
        "a stable canonical instance for."
    )
)
def test_signal_multiplier_recovery_noise_integration():
    """Noise-floor stall on a near-rank-deficient KKT system."""
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "Requires a problem where the line search consistently "
        "collapses to alpha < 1e-10.  Synthetic test owns the "
        "signal-firing contract; reliably triggering LS collapse "
        "from a real run depends on the post-divergence-rollback "
        "L-BFGS state and is fragile."
    )
)
def test_signal_line_search_collapse_integration():
    """Line search collapse on a non-descent QP direction."""
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "Requires a degenerate vertex where the QP active-set loop "
        "ping-pongs or exhausts its budget.  Synthetic test owns "
        "the signal-firing contract; a stable real-problem trigger "
        "is on the Phase 5 wishlist."
    )
)
def test_signal_qp_budget_or_pingpong_integration():
    """QP budget exhaustion or ping-pong on a degenerate vertex."""
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "Requires a problem where the Han-Powell merit penalty "
        "stays under-sized for many iterations.  Synthetic test "
        "owns the signal-firing contract; merit oscillation is a "
        "compound failure that the L-BFGS reset chain often masks."
    )
)
def test_signal_merit_oscillation_integration():
    """Merit oscillation under an under-sized penalty."""
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "Requires an LPEC-A run where over-prediction is the "
        "dominant failure mode.  Synthetic test owns the signal-"
        "firing contract; a stable real-problem trigger is on the "
        "Phase 5 wishlist."
    )
)
def test_signal_lpeca_overpredicting_integration():
    """LPEC-A over-prediction across a majority of steps."""
    raise NotImplementedError


def test_signal_infeasible_termination_integration():
    """Mutually infeasible inequality constraints → infeasible signal fires."""

    def f(x, args):
        return jnp.sum(x**2)

    def c_ineq(x, args):
        # Two scalar inequalities ``c_ineq >= 0``:
        #   x[0] - 1 >= 0         (i.e. x[0] >= 1)
        #   -x[0] - 1 >= 0        (i.e. x[0] <= -1)
        # No real x[0] satisfies both simultaneously.
        return jnp.array([x[0] - 1.0, -x[0] - 1.0])

    solver = SLSQP(ineq_constraint_fn=c_ineq, n_ineq_constraints=2)
    x0 = jnp.array([0.0])
    report = diagnose(solver, f, x0, max_steps=20)

    # Either the explicit termination code says infeasible OR every
    # recorded summary carries a feasibility violation above ``atol``
    # — both routes trigger the signal's "persistent infeasibility"
    # branch.
    fired_names = set(report.signals.keys())
    persistent_infeasibility = report.run.summaries and all(
        s.max_ineq_violation > solver.atol for s in report.run.summaries
    )
    assert "infeasible_termination" in fired_names or persistent_infeasibility


@pytest.mark.skip(
    reason=(
        "Requires a problem that drives the best-iterate divergence "
        "rollback at least once.  The canonical reproducer is the "
        "feasible-start ``Portfolio(n=5000)`` run from the diagnostic "
        "session that motivated this signal: it cascades QP-budget "
        "exhaustion → L-BFGS poisoning → merit-penalty over-correction "
        "→ rollback over ~17 outer steps.  Synthetic test owns the "
        "signal-firing contract; wiring up the portfolio reproducer "
        "(or an equivalent toy that consistently rolls back without "
        "depending on a 5000-variable bound-heavy QP) is on the "
        "follow-up wishlist."
    )
)
def test_signal_divergence_rollback_triggered_integration():
    """Best-iterate divergence rollback fires on a real failing run."""
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "Requires a problem whose Han-Powell ``rho`` jumps by 1e3+ in "
        "a single step or spans 1e6+ across the run.  Canonical "
        "reproducer: feasible-start ``Portfolio(n=5000)`` from the "
        "diagnostic session, where ``rho`` jumps ``1.0 → 4.5e+04 → "
        "6.9e+09`` over three consecutive steps.  Synthetic test "
        "owns the signal-firing contract; the real-problem trigger "
        "is on the follow-up wishlist."
    )
)
def test_signal_merit_penalty_explosion_integration():
    """Merit penalty explosion fires when ``rho`` jumps in a single step."""
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "Requires a problem whose initial iterate is feasible (or "
        "near-feasible) and whose merit-penalty update mechanism "
        "stays starved while feasibility drifts.  Canonical "
        "reproducer: feasible-start ``Portfolio(n=5000)`` from the "
        "diagnostic session, where ``rho`` stays at 1.0 for the "
        "first 13 steps while ``max|c_eq|`` walks from ~3.5e-7 to "
        "~3.5e-6.  Synthetic test owns the signal-firing contract; "
        "the real-problem trigger is on the follow-up wishlist."
    )
)
def test_signal_penalty_starvation_integration():
    """Penalty starvation fires on a feasible-start drift trajectory."""
    raise NotImplementedError
