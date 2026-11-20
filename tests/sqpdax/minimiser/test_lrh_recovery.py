"""Inherited recovery behaviour of :class:`LRHLineSearchMinimiser`.

The LRH minimiser overrides none of the parent's recovery machinery
(restoration fallback, decaying merit penalty, plain retry subproblem,
infeasibility certificate, staged secant resets). These tests mirror the
matching cases of ``test_active_set_linesearch.py`` on the LRH model and add
the minimiser-level reset of a model that already holds pairs.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    LRHLineSearchMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
)
from slsqp_jax.sqpdax.step_controller import StepResult
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import RESULTS

from .conftest import (
    _execute_synthetic_step,
    _indefinite_quadratic,
    make_equality_quadratic,
    make_hs71,
    make_infeasible_problem,
    make_scaled_quartic,
    make_shifted_equality_problem,
    make_unconstrained_quadratic,
    make_zero_jacobian_infeasible_problem,
)


@pytest.fixture(autouse=True)
def x64():
    """Run every test in float64 (the tolerances below assume it)."""
    with jax.enable_x64(True):
        yield


# --- restoration fallback ---------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False], ids=["restoration", "off"])
def test_restoration_fallback_runs_only_for_rejected_infeasible_steps(enabled):
    """A rejected LRH QP direction at an infeasible point is replaced by restoration."""
    problem = make_shifted_equality_problem()
    solver = LRHLineSearchMinimiser(restoration=enabled).init(problem, jnp.zeros(2))
    ctx = solver._init_subproblem(problem)
    step_primal, step_dual, solver_state = solver._solve_direction(ctx)
    rejected = StepResult(
        x=solver.iterate,
        accepted=jnp.asarray(False),
        merit_val=jnp.asarray(4.0),
        solver_state=solver_state,
        step_size=jnp.asarray(0.0),
        proposed_step_norm=jnp.linalg.norm(step_primal.x),
        accepted_by_fallback=jnp.asarray(False),
    )
    result, restored, penalty = solver._restoration_fallback(
        ctx, step_dual, solver_state, rejected
    )
    assert bool(restored) is enabled
    assert bool(result.accepted) is enabled
    han_powell = float(solver._next_merit_penalty(step_dual, solver_state))
    # ``∇f = (0, 1)`` is orthogonal to the step: the restoration keeps the
    # Han--Powell penalty of this step, without any extra increase; when off
    # the carried penalty is returned untouched.
    expected = han_powell if enabled else float(solver.merit_penalty)
    assert float(penalty) == pytest.approx(expected)
    assert (float(result.x.x[0]) > 0.0) is enabled


@pytest.mark.parametrize(
    ("step_size", "step_norm", "expect_restored"),
    [(1.0, 1.0, False), (1.0, 0.0, True), (0.0, 1.0, True)],
    ids=["real-step", "null-direction", "null-step-size"],
)
def test_restoration_fallback_overrides_only_accepted_null_steps(
    step_size, step_norm, expect_restored
):
    """An accepted QP step is kept unless it is a null step at an infeasible point."""
    problem = make_shifted_equality_problem()
    solver = LRHLineSearchMinimiser().init(problem, jnp.zeros(2))
    ctx = solver._init_subproblem(problem)
    _, step_dual, solver_state = solver._solve_direction(ctx)
    accepted = StepResult(
        x=solver.iterate,
        accepted=jnp.asarray(True),
        merit_val=jnp.asarray(4.0),
        solver_state=solver_state,
        step_size=jnp.asarray(step_size),
        proposed_step_norm=jnp.asarray(step_norm),
        accepted_by_fallback=jnp.asarray(False),
    )
    result, restored, _ = solver._restoration_fallback(
        ctx, step_dual, solver_state, accepted
    )
    assert bool(restored) is expect_restored
    assert bool(result.accepted)
    assert (float(result.x.x[0]) > 0.0) is expect_restored


def _violation(problem, x):
    """``ℓ¹`` constraint violation (tiny n, reference only)."""
    return float(
        jnp.sum(jnp.abs(problem.eq_fn(x)))
        + jnp.sum(jnp.maximum(problem.ineq_fn(x), 0.0))
    )


@pytest.mark.parametrize("enabled", [True, False], ids=["restoration", "off"])
def test_rejected_directions_at_an_infeasible_start_still_make_progress(enabled):
    """HS71 from ``(4, 4, 4, 4)`` with a one-trial line search.

    The full QP step is rejected (no backtracking is allowed) at an
    infeasible iterate, so progress can only come from the restoration
    fallback: with it the run counts restoration steps and drives the
    violation to rounding level; with ``restoration=False`` the option is
    honoured (no restoration step is ever taken) and the violation stays
    large.
    """
    problem = make_hs71()
    x0 = jnp.array([4.0, 4.0, 4.0, 4.0])
    sol = minimise(
        problem,
        LRHLineSearchMinimiser(
            restoration=enabled, line_search_max_steps=1, rtol=1e-6, atol=1e-6
        ),
        x0,
        max_steps=30,
        throw=False,
    )
    final = _violation(problem, sol.value)
    n_restoration = int(sol.state.n_restoration_steps)
    assert bool(jnp.all(jnp.isfinite(sol.value)))
    assert (n_restoration > 0) is enabled
    assert (final < 1e-6) is enabled


# --- decaying penalty -------------------------------------------------------------


def test_penalty_rule_is_inherited_unchanged():
    """The LRH class has no penalty override: it *is* the parent's function."""
    assert (
        LRHLineSearchMinimiser._next_merit_penalty
        is ActiveSetLineSearchMinimiser._next_merit_penalty
    )


def test_penalty_update_is_decaying_sign_clamped_and_trust_gated():
    """Only finite, successful, dual-feasible multiplier magnitudes move rho."""
    problem = make_unconstrained_quadratic()
    solver = LRHLineSearchMinimiser(
        initial_penalty=2.0, penalty_multiplier_margin=1.1
    ).init(problem, jnp.ones(2))
    state = eqx.tree_at(lambda s: s.success, solver.solver_state, jnp.asarray(True))
    dual = eqx.tree_at(
        lambda d: (d.lb_multipliers, d.ub_multipliers),
        solver.dual,
        (jnp.asarray([-100.0, 4.0]), jnp.asarray([5.0, -200.0])),
    )
    rho = solver._next_merit_penalty(dual, state)
    assert float(rho) == pytest.approx(5.5)

    raised = eqx.tree_at(lambda m: m.merit_penalty, solver, rho)
    smaller = jax.tree.map(lambda leaf: 0.1 * leaf, dual)
    # Powell rule: max(c=0.55, (5.5 + 0.55) / 2, initial=2.0) = 3.025.
    assert float(raised._next_merit_penalty(smaller, state)) == pytest.approx(3.025)

    failed = eqx.tree_at(lambda s: s.success, state, jnp.asarray(False))
    assert float(solver._next_merit_penalty(dual, failed)) == pytest.approx(2.0)
    nonfinite = eqx.tree_at(
        lambda d: d.lb_multipliers, dual, jnp.asarray([jnp.nan, 4.0])
    )
    assert float(solver._next_merit_penalty(nonfinite, state)) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "rho, multiplier",
    [
        (1000.0, 1.0),  # spike: halve the excess over c = 1.1
        (1.0e-3, 10.0),  # growth jumps straight to c
        (2.0, 0.1),  # floored at initial_penalty
    ],
)
def test_penalty_spike_decays_geometrically(rho, multiplier):
    """A transient multiplier spike decays towards c, never below the floor."""
    problem = make_unconstrained_quadratic()
    solver = LRHLineSearchMinimiser(
        initial_penalty=2.0, penalty_multiplier_margin=1.1
    ).init(problem, jnp.ones(2))
    state = eqx.tree_at(lambda s: s.success, solver.solver_state, jnp.asarray(True))
    dual = eqx.tree_at(
        lambda d: d.lb_multipliers, solver.dual, jnp.asarray([multiplier, 0.0])
    )
    solver = eqx.tree_at(lambda m: m.merit_penalty, solver, jnp.asarray(rho))
    expected_rho = max(2.0, 1.1 * multiplier, 0.5 * (rho + 1.1 * multiplier))
    assert float(solver._next_merit_penalty(dual, state)) == pytest.approx(expected_rho)
    for _ in range(40):
        solver = eqx.tree_at(
            lambda m: m.merit_penalty, solver, solver._next_merit_penalty(dual, state)
        )
    assert float(solver.merit_penalty) == pytest.approx(max(2.0, 1.1 * multiplier))


# --- plain retry subproblem -------------------------------------------------------


def test_retry_subproblem_is_plain_whatever_the_failure_count():
    """No Levenberg shift: the model is always a secant, never exact curvature.

    On ``f = 1/2 (x0^2 - 4 x1^2)`` (exact HVPs available) the parent's exact
    model is shifted after four rejections; the LRH model is not: the QP
    curvature along ``e1`` and the whole subproblem type are the same as with
    no failures.
    """
    problem = _indefinite_quadratic()
    base = LRHLineSearchMinimiser().init(problem, jnp.array([1.0, 1.0]))
    failed = eqx.tree_at(
        lambda s: s.consecutive_ls_failures, base, jnp.asarray(4, jnp.int32)
    )
    e1 = jnp.array([0.0, 1.0])
    curvatures, types = [], []
    for solver in (base, failed):
        ctx = solver._init_subproblem(problem)
        types.append(type(ctx.subproblem))
        # ``H e1`` of the QP model through the lagrangian's secant.
        curvatures.append(float(e1 @ ctx.subproblem.lagrangian.secant.hvp(e1)))
    assert types == [ActiveSetSubProblem, ActiveSetSubProblem]
    assert curvatures[0] == curvatures[1]


# --- infeasibility certificate ----------------------------------------------------


def _at(solver, x):
    """Copy of ``solver`` with its iterate placed at ``x`` (outside the box if so)."""
    return eqx.tree_at(lambda m: m.iterate.x, solver, jnp.asarray(x))


@pytest.mark.parametrize(
    ("make_problem_fn", "x", "expected"),
    [
        (make_infeasible_problem, [0.5, 0.0], 0.0),
        (make_infeasible_problem, [0.2, 0.0], 0.6 / (0.68**0.5)),
        (make_infeasible_problem, [1.0, 0.0], 0.0),
        (make_zero_jacobian_infeasible_problem, [0.0, 0.0], 0.0),
        (make_equality_quadratic, [0.5, 0.5], 0.0),
    ],
    ids=["midpoint", "pulled", "projected", "zero-jacobian", "feasible"],
)
def test_violation_stationarity_is_the_normalised_projected_gradient(
    make_problem_fn, x, expected
):
    """``violation_stationarity`` measures first-order progress on ``½‖c⁺‖²``."""
    problem = make_problem_fn()
    # ``init`` projects ``x0`` onto the box, so place the iterate afterwards.
    solver = _at(LRHLineSearchMinimiser().init(problem, jnp.zeros(2)), x)
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    assert float(metrics.violation_stationarity) == pytest.approx(expected, abs=1e-6)


def test_infeasible_stationary_requires_a_stationary_violation_measure():
    """Lagrangian stationarity alone does not certify infeasibility; a run does."""
    problem = make_infeasible_problem()
    solver = _at(
        LRHLineSearchMinimiser(min_steps=0, zero_step_patience=0).init(
            problem, jnp.zeros(2)
        ),
        [0.2, 0.0],
    )
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    stationary = jnp.asarray(True)
    feasible = jnp.asarray(False)
    assert not bool(solver._infeasible_stationary(metrics, stationary, feasible))
    at_minimiser = _at(solver, [0.5, 0.0])
    metrics = at_minimiser.termination_metrics(
        at_minimiser._optimisation_context(problem)
    )
    assert bool(at_minimiser._infeasible_stationary(metrics, stationary, feasible))
    # ``x₀² + 1 = 0`` from ``x = 0``: the Jacobian vanishes, so the violation
    # cannot be reduced to first order and the certificate is legitimate.
    sol = minimise(
        make_zero_jacobian_infeasible_problem(),
        LRHLineSearchMinimiser(),
        jnp.zeros(2),
        max_steps=50,
        throw=False,
    )
    assert sol.stats["sqpdax_result"] == (
        ACTIVE_SET_LINE_SEARCH_RESULTS.infeasible_stationary
    )
    assert float(sol.stats["violation_stationarity"]) <= 1e-3


@pytest.mark.parametrize(
    ("last_ls_success", "certifies"),
    [(True, False), (False, True)],
    ids=["progressing", "stuck"],
)
def test_infeasible_stationary_needs_a_stuck_iterate(last_ls_success, certifies):
    """At a violation-stationary point the certificate also needs ``stuck``.

    ``stuck`` is ``stationary | ~last_ls_success``: with the Lagrangian not
    stationary, a point where the last step was accepted is still progressing
    and must not certify; one where it was rejected must.
    """
    problem = make_infeasible_problem()
    solver = _at(
        LRHLineSearchMinimiser(min_steps=0, zero_step_patience=0).init(
            problem, jnp.zeros(2)
        ),
        [0.5, 0.0],
    )
    solver = eqx.tree_at(
        lambda m: m.last_ls_success, solver, jnp.asarray(last_ls_success)
    )
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    verdict = solver._infeasible_stationary(
        metrics, jnp.asarray(False), jnp.asarray(False)
    )
    assert bool(verdict) == certifies


# --- staged secant reset of a model that already holds pairs (C19 review O2) ------


@pytest.mark.parametrize(
    ("failures", "rank", "sigma_reset", "stage", "n_resets", "fatal"),
    [
        (1, 1, 0.0, 0, [1, 0, 0], False),  # soft: newest direction, sigma kept
        (2, 0, 0.0, 1, [1, 1, 0], False),  # diagonal: empty basis, sigma kept
        (3, 0, 1.0, 2, [1, 1, 1], False),  # identity: sigma = initial_sigma
        (4, 0, 1.0, 2, [1, 1, 1], True),  # fatal threshold, no further reset
    ],
    ids=["soft", "diagonal", "identity", "fatal"],
)
def test_forced_failures_reset_a_model_that_holds_pairs(
    failures, rank, sigma_reset, stage, n_resets, fatal
):
    """Reset severities of PLAN section 3.3 applied to a learned LRH model.

    Four real steps on the scaled quartic give a rank-2 model with a large
    ``sigma``; forced line-search failures then walk the recovery schedule.
    The rank, ``sigma``, the reset statistics and the reported stats all
    follow the reset severities (``sigma`` returns to ``initial_sigma = 1``
    only at the identity stage), and a real step after a non-fatal reset
    still works on the reset model.
    """
    problem = make_scaled_quartic()
    solver = LRHLineSearchMinimiser(rtol=-1.0, stagnation_patience=100).init(
        problem, jnp.array([0.5, 0.3, 0.2])
    )
    for _ in range(4):
        solver = solver.step(problem)
    learned = solver.secant.factors()
    assert int(learned.rank) == 2
    assert float(learned.sigma) > 2.0

    for _ in range(failures):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=False,
            qp_success=True,
            qp_status=RESULTS.successful,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
        )
    factors = solver.secant.factors()
    expected_sigma = sigma_reset * 1.0 + (1.0 - sigma_reset) * float(learned.sigma)
    assert int(factors.rank) == rank
    assert float(factors.sigma) == pytest.approx(expected_sigma)
    assert solver.secant_stats.n_resets.tolist() == n_resets
    assert int(solver.secant_recovery_state.stage) == stage
    assert int(solver.secant_recovery_state.failure_streak) == failures
    assert bool(solver.secant_recovery_state.fatal) is fatal
    done, result = solver.terminate(problem)
    assert bool(done) is fatal
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure) is (
        fatal
    )

    stats = solver._postprocess_stats(problem, result)
    assert int(stats["lrh_rank"]) == rank
    assert float(stats["lrh_sigma"]) == pytest.approx(expected_sigma)

    after = solver.step(problem)
    assert bool(jnp.all(jnp.isfinite(after.iterate.x)))
    assert float(after.secant.factors().sigma) > 0.0


# --- inherited penalty blow-up (C17 review e) -------------------------------------


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, LRHLineSearchMinimiser],
    ids=["parent", "lrh"],
)
def test_penalty_blowup_on_the_zero_jacobian_problem_is_pinned(minimiser_cls):
    """Pin the CURRENT behaviour of ``x₀² + 1 = 0`` from ``(0.5, 0.3)``.

    The linearisation ``2 x₀ d = -(x₀² + 1)`` stays consistent while
    ``x₀ → 0``, so the solver never goes elastic and the equality multiplier
    grows like ``1/x₀``; Powell's rule then raises the merit penalty without
    bound. Observed on the LRH minimiser: ``ρ = 1, 42, 5.6e6, 5.1e22`` over
    the first four steps and constant afterwards (13 steps, final
    ``x₀ ≈ -7e-9``); the parent reaches ``ρ ≈ 4e166``. Both end with the
    ``infeasible`` status and finite iterates, never NaN. This is inherited
    parent design (no cap on ``ρ``); changing it is outside the LRH plan.

    The termination, status and finiteness asserts are the lasting
    requirements. The ``merit_penalty > 1e10`` assert is a deliberate
    tripwire on the current, undesirable magnitude: a fix that bounds ``ρ``
    is meant to replace it with an upper bound, not to be blocked by it.
    """
    problem = make_zero_jacobian_infeasible_problem()
    sol = minimise(
        problem,
        minimiser_cls(),
        jnp.array([0.5, 0.3]),
        max_steps=60,
        throw=False,
    )
    state = sol.state
    assert int(state.step_count) < 60
    assert bool(sol.result == ACTIVE_SET_LINE_SEARCH_RESULTS.infeasible)
    assert bool(jnp.all(jnp.isfinite(sol.value)))
    assert bool(jnp.isfinite(state.merit_penalty))
    assert float(state.merit_penalty) > 1e10
    assert bool(jnp.all(jnp.isfinite(state.dual.eq_multipliers)))
