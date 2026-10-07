"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.active_set_linesearch`."""

from __future__ import annotations

import warnings
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.active_set_prediction import LPECAPredictor
from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    ProximalActiveSetLineSearchMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.secant import FailureRecoverySchedule
from slsqp_jax.sqpdax.step_controller import StepResult
from slsqp_jax.sqpdax.subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    KKT_SOLVER_RESULTS,
    RESULTS,
    ClampSafeguard,
    CraigProjector,
    LeastSquaresMultiplierRecovery,
    MinresQLPSubProblemSolver,
    SingleExchangeWorkingSetPolicy,
    ThresholdWorkingSetPolicy,
)

from ..conftest import make_shifted_box_quadratic
from .conftest import make_equality_quadratic, make_unconstrained_quadratic


def test_init_builds_primal_dual_and_qp_state():
    """``init`` seeds a plain ``Primal``, zero dual, and cold QP carry."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.ones(2))
    assert isinstance(solver.iterate, Primal)
    assert solver.dual is not None
    assert solver.solver_state is not None
    assert int(solver.solver_state.n_iter) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"initial_penalty": 0.0}, "initial_penalty must be positive"),
        (
            {"penalty_multiplier_margin": 1.0},
            "penalty_multiplier_margin must be greater than 1",
        ),
    ],
)
def test_penalty_configuration_is_validated(kwargs, match):
    """Han--Powell options must define a positive strict multiplier margin."""
    with pytest.raises(ValueError, match=match):
        ActiveSetLineSearchMinimiser(**kwargs)


def test_penalty_options_replace_the_stateless_names():
    """The explicit stateful names are accepted and old names are rejected."""
    problem = make_unconstrained_quadratic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = ActiveSetLineSearchMinimiser().init(
            problem,
            jnp.ones(2),
            options={
                "minimiser": {
                    "initial_penalty": 2.0,
                    "penalty_multiplier_margin": 1.25,
                    "penalty_floor": 3.0,
                    "penalty_factor": 4.0,
                }
            },
        )
    messages = " ".join(str(item.message) for item in caught)
    assert "unknown minimiser option 'penalty_floor'" in messages
    assert "unknown minimiser option 'penalty_factor'" in messages
    assert solver.initial_penalty == 2.0
    assert solver.penalty_multiplier_margin == 1.25
    assert float(solver.merit_penalty) == pytest.approx(2.0)


def test_penalty_update_is_monotone_sign_clamped_and_trust_gated():
    """Only finite, successful, dual-feasible multiplier magnitudes ratchet rho."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
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
    assert float(raised._next_merit_penalty(smaller, state)) == pytest.approx(5.5)

    failed = eqx.tree_at(lambda s: s.success, state, jnp.asarray(False))
    assert float(solver._next_merit_penalty(dual, failed)) == pytest.approx(2.0)
    nonfinite = eqx.tree_at(
        lambda d: d.lb_multipliers,
        dual,
        jnp.asarray([jnp.nan, 4.0]),
    )
    assert float(solver._next_merit_penalty(nonfinite, state)) == pytest.approx(2.0)


def test_penalty_increase_revalues_best_iterate_before_progress_comparison():
    """Best-point comparisons use one rho after a multiplier-driven ratchet."""
    problem = make_equality_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        initial_penalty=1.0,
        penalty_multiplier_margin=1.1,
        stagnation_patience=100,
    ).init(problem, jnp.zeros(2))
    ctx = solver._init_subproblem(problem)
    state = eqx.tree_at(lambda s: s.success, solver.solver_state, jnp.asarray(True))
    step_dual = eqx.tree_at(
        lambda d: d.eq_multipliers, solver.dual, jnp.asarray([10.0])
    )
    x_new = Primal(jnp.asarray([0.25, 0.25]))
    controller = solver._step_controller(ctx, step_dual, state)
    merit_new = controller.merit(x_new)
    # With rho=1 the old best merit is 1, while with rho=11 it is 11.
    # The new point is an improvement only after comparing in the new units.
    assert 1.0 < float(merit_new) < 11.0
    result = StepResult(
        x=x_new,
        accepted=jnp.asarray(True),
        merit_val=merit_new,
        solver_state=state,
        step_size=jnp.asarray(1.0),
        proposed_step_norm=jnp.asarray(jnp.sqrt(0.125)),
    )
    advanced = solver._advance_dynamics(ctx, result, step_dual)
    assert float(advanced.merit_penalty) == pytest.approx(11.0)
    assert jnp.array_equal(advanced.best_iterate.x, x_new.x)
    assert float(advanced.best_merit) == pytest.approx(float(merit_new))


def test_step_and_minimise_unconstrained():
    """Outer loop drives the unconstrained quadratic to the origin."""
    problem = make_unconstrained_quadratic()
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(rtol=1e-5, atol=1e-5, min_steps=1),
        jnp.ones(2),
        max_steps=20,
        throw=True,
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, 0.0, atol=1e-4)


def test_minimise_equality_constrained():
    """Equality-constrained quadratic lands on the feasible affine line."""
    problem = make_equality_quadratic()
    # Feasible minimiser of ‖x‖² s.t. x0+x1=1 is x=(0.5, 0.5).
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(rtol=1e-4, atol=1e-4, min_steps=1),
        jnp.array([0.0, 0.0]),
        max_steps=40,
        throw=False,
    )
    assert jnp.allclose(sol.value[0] + sol.value[1], 1.0, atol=5e-3)
    assert jnp.allclose(sol.value, jnp.array([0.5, 0.5]), atol=5e-2)


@pytest.mark.parametrize(
    "options",
    [
        None,
        {
            "minimiser": {"qp_tol": 1e-7},
            "subproblem": {"working_set_policy": {"tol": 1e-9}},
        },
        {
            "minimiser": {"qp_tol": 1e-7},
            "subproblem": {
                "working_set_policy": {"expand_factor": 1.0, "ping_pong_threshold": 3}
            },
        },
    ],
    ids=["default", "with-options", "with-policy-options"],
)
def test_init_accepts_option_bag(options):
    """Recognised minimiser / subproblem options are applied without error."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.ones(2), options=options)
    assert solver.iterate is not None
    if options is not None:
        assert solver.qp_tol == 1e-7
        policy = solver._init_subproblem(problem).solver.working_set_policy
        policy_opts = options["subproblem"]["working_set_policy"]
        # A nested ``tol`` overrides the minimiser's ``qp_tol`` routing.
        assert policy.tol == policy_opts.get("tol", 1e-7)
        assert policy.expand_factor == policy_opts.get("expand_factor", 0.0)
        assert policy.ping_pong_threshold == policy_opts.get("ping_pong_threshold")
        # Exercise the ``subproblem`` option path inside ``_init_subproblem``.
        solver = solver.step(problem)
        assert int(solver.step_count) == 1


def test_global_failure_recovery_schedule_accepts_nested_options():
    """One nested schedule configures recovery across all failure channels."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(
        problem,
        jnp.ones(2),
        options={
            "minimiser": {
                "secant_reset": {
                    "recovery_schedule": {
                        "soft": 2,
                        "diagonal": 4,
                        "identity": 6,
                        "fatal": 8,
                    }
                },
            }
        },
    )
    assert solver.secant_reset.recovery_schedule == FailureRecoverySchedule(
        soft=2, diagonal=4, identity=6, fatal=8
    )


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
@pytest.mark.parametrize(
    ("qp_tol", "atol", "expected_tol"),
    [(None, 1e-4, 1e-4), (1e-7, 1e-4, 1e-7)],
    ids=["routed-from-atol", "explicit"],
)
@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
def test_qp_solver_receives_tolerance_and_warm_start_flag(
    minimiser_cls, qp_tol, atol, expected_tol, qp_warm_start
):
    """``qp_tol=None`` falls back to ``atol``; ``qp_warm_start`` reaches the solver."""
    problem = make_equality_quadratic()
    solver = minimiser_cls(
        qp_tol=qp_tol, atol=atol, qp_warm_start=qp_warm_start, min_steps=1
    ).init(problem, jnp.zeros(2))
    assert solver.effective_qp_tol == expected_tol
    ctx = solver._init_subproblem(problem)
    assert ctx.solver.tol == expected_tol
    assert ctx.solver.working_set_policy.tol == expected_tol
    assert ctx.solver.max_iter == solver.qp_max_iter
    assert ctx.solver.warm_start is qp_warm_start
    assert type(ctx.solver.working_set_policy) is ThresholdWorkingSetPolicy
    # The carried working set / dual are sized for the problem and cold.
    state = solver.solver_state
    assert state.active_set.active_lb.shape == (2,)
    assert not jnp.any(state.active_set.active_lb)
    assert state.dual.eq_multipliers.shape == (1,)
    # A full step still runs end-to-end with the carry threaded through.
    stepped = solver.step(problem)
    assert int(stepped.solver_state.last_n_iter) >= 1
    assert int(stepped.solver_state.n_iter) == int(stepped.solver_state.last_n_iter)


def test_guarded_qp_kkt_success_after_repeated_full_zero_steps():
    """Repeated converged full zero steps provide the guarded success path."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        min_steps=3,
        zero_step_patience=3,
        stagnation_patience=100,
    ).init(problem, jnp.zeros(2))
    for _ in range(3):
        solver = solver.step(problem)
    done, result = solver.terminate(problem)
    assert bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.successful)
    assert bool(solver.qp_optimal)
    assert float(solver.last_step_size) == pytest.approx(1.0)


def test_mu_max_option_changes_kkt_scale_and_exposes_ratio():
    """The optional filterSQP denominator is distinct from the Lagrangian scale."""
    problem = make_equality_quadratic()
    solver = ActiveSetLineSearchMinimiser(use_mu_max=True).init(
        problem, jnp.asarray([0.25, 0.25])
    )
    solver = eqx.tree_at(lambda m: m.dual.eq_multipliers, solver, jnp.asarray([100.0]))
    mu_metrics = solver.termination_metrics(solver._optimisation_context(problem))
    classical = replace(solver, use_mu_max=False)
    classical_metrics = classical.termination_metrics(
        classical._optimisation_context(problem)
    )
    assert float(mu_metrics.stationarity_scale) == pytest.approx(100.0 * jnp.sqrt(2.0))
    assert float(mu_metrics.stationarity_scale) != pytest.approx(
        float(classical_metrics.stationarity_scale)
    )
    assert float(mu_metrics.kkt_ratio) == pytest.approx(
        float(mu_metrics.stationarity) / float(mu_metrics.stationarity_scale)
    )


def test_merit_stagnation_returns_native_fine_grained_result():
    """No merit improvement terminates with the active-set-specific code."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        min_steps=1,
        zero_step_patience=100,
        stagnation_patience=2,
    ).init(problem, jnp.zeros(2))
    solver = solver.step(problem).step(problem)
    done, result = solver.terminate(problem)
    assert bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.merit_stagnation)


def _execute_synthetic_step(
    solver,
    problem,
    *,
    accepted: bool,
    qp_success: bool,
    qp_status,
    x,
    merit: float,
    step_size: float | None = None,
    proposed_step_norm: float = 1.0,
    accepted_by_fallback: bool = False,
):
    """Exercise phase four with a controlled line-search/QP outcome."""
    ctx = solver._init_subproblem(problem)
    qp_result = ACTIVE_SET_QP_RESULTS.where(
        jnp.asarray(qp_success),
        ACTIVE_SET_QP_RESULTS.working_set_converged,
        ACTIVE_SET_QP_RESULTS.where(
            qp_status == RESULTS.max_steps_reached,
            ACTIVE_SET_QP_RESULTS.max_iter_reached,
            ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
        ),
    )
    state = eqx.tree_at(
        lambda s: (s.success, s.status, s.qp_result),
        solver.solver_state,
        (jnp.asarray(qp_success), qp_status, qp_result),
    )
    result = StepResult(
        x=Primal(jnp.asarray(x)),
        accepted=jnp.asarray(accepted),
        merit_val=jnp.asarray(merit),
        solver_state=state,
        step_size=jnp.asarray(
            (1.0 if accepted else 0.0) if step_size is None else step_size
        ),
        proposed_step_norm=jnp.asarray(proposed_step_norm),
        accepted_by_fallback=jnp.asarray(accepted_by_fallback),
    )
    return solver._execute_step(ctx, solver.dual, result)


@pytest.mark.parametrize(
    ("kind", "qp_status", "expected"),
    [
        (
            "qp",
            RESULTS.singular,
            ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure,
        ),
        # A feasibility floor above target (MINRES-QLP ``residual_floor``)
        # surfaces as ``stagnation`` and is a real QP failure too.
        (
            "qp",
            RESULTS.stagnation,
            ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure,
        ),
        (
            "ls",
            RESULTS.successful,
            ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure,
        ),
    ],
    ids=["qp-singular", "qp-stagnation", "line-search"],
)
def test_failure_channels_share_global_recovery_fatal_threshold(
    kind, qp_status, expected
):
    """Each channel advances the same reset lifecycle and fatal threshold."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        curvature="secant",
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))
    for index in range(4):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=kind == "qp",
            qp_success=kind == "ls",
            qp_status=qp_status,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
        )
        done, _ = solver.terminate(problem)
        assert bool(done) is (index == 3)
    _, result = solver.terminate(problem)
    assert bool(result == expected)
    assert int(solver.secant_recovery_state.failure_streak) == 4
    assert int(solver.secant_recovery_state.stage) == 2
    assert bool(solver.secant_recovery_state.fatal)


@pytest.mark.parametrize(
    ("proposed_step_norm", "expect_fatal"),
    [(jnp.nan, True), (jnp.inf, True), (1.0, False)],
    ids=["nan-direction", "inf-direction", "finite-unconverged"],
)
def test_nonfinite_qp_direction_counts_as_qp_failure(proposed_step_norm, expect_fatal):
    """A rejected non-finite QP direction is a real QP failure.

    The QP solver reports a NaN residual as ``max_steps_reached``, which is
    otherwise (correctly) not counted as a failure; a non-finite direction
    must still drive ``qp_subproblem_failure`` because retrying at the same
    ``(x, λ)`` reproduces it.
    """
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        curvature="secant",
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))
    for _ in range(4):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=not expect_fatal,
            qp_success=False,
            qp_status=RESULTS.max_steps_reached,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
            proposed_step_norm=proposed_step_norm,
        )
    done, result = solver.terminate(problem)
    assert bool(done) is expect_fatal
    assert bool(solver.secant_recovery_state.fatal) is expect_fatal
    if expect_fatal:
        assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure)
    else:
        assert int(solver.consecutive_qp_failures) == 0


def test_mixed_failure_channels_do_not_restart_global_recovery():
    """Alternating QP and line-search failures reach identity, then fatal."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        curvature="secant",
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))

    for kind in ("qp", "ls", "qp", "ls"):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=kind == "qp",
            qp_success=kind == "ls",
            qp_status=RESULTS.singular if kind == "qp" else RESULTS.successful,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
        )

    done, result = solver.terminate(problem)
    assert bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.secant_recovery_failure)
    assert jnp.array_equal(solver.secant_stats.n_resets, jnp.ones(3, jnp.int32))


@pytest.mark.parametrize("bad_merit", [100.0, jnp.inf], ids=["growth", "nonfinite"])
def test_merit_blowup_restores_best_iterate(bad_merit):
    """Repeated excessive merit growth rolls back and reports iterate blow-up."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        divergence_factor=1.0,
        divergence_patience=2,
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))
    best_x = solver.iterate.x
    best_dual = solver.dual
    # Pretend the (abandoned) QPs left a non-trivial working set / dual behind.
    solver = eqx.tree_at(
        lambda m: (m.solver_state.active_set.active_lb, m.solver_state.dual),
        solver,
        (
            jnp.array([True, False]),
            jax.tree.map(lambda leaf: leaf + 5.0, best_dual),
        ),
    )
    for _ in range(2):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=True,
            qp_success=True,
            qp_status=RESULTS.successful,
            x=jnp.asarray([10.0, 10.0]),
            merit=bad_merit,
        )
    done, result = solver.terminate(problem)
    assert bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.iterate_blowup)
    assert jnp.allclose(solver.iterate.x, best_x)
    # The rollback also drops the carried QP working set and re-syncs its dual.
    assert not jnp.any(solver.solver_state.active_set.active_lb)
    for carried, best in zip(
        jax.tree.leaves(solver.solver_state.dual), jax.tree.leaves(best_dual)
    ):
        assert jnp.array_equal(carried, best)


def test_tiny_line_search_step_cannot_trigger_qp_kkt_success():
    """A tiny accepted alpha may latch zero steps but cannot certify KKT success."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        atol=1e-2,
        min_steps=1,
        zero_step_patience=1,
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))
    solver = _execute_synthetic_step(
        solver,
        problem,
        accepted=True,
        qp_success=True,
        qp_status=RESULTS.successful,
        x=solver.iterate.x,
        merit=float(solver.best_merit),
        step_size=1e-3,
        proposed_step_norm=1.0,
    )
    done, result = solver.terminate(problem)
    assert bool(solver.qp_optimal)
    assert not bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.running)


def test_acceptance_kind_is_counted_and_exposed():
    """Strict Armijo and fallback-only accepted steps remain distinguishable."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(rtol=-1.0, stagnation_patience=100).init(
        problem, jnp.ones(2)
    )
    for fallback in (False, True):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=True,
            accepted_by_fallback=fallback,
            qp_success=True,
            qp_status=RESULTS.successful,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
        )
    assert bool(solver.last_ls_success)
    assert bool(solver.diagnostics.last_ls_fallback)
    assert int(solver.diagnostics.n_armijo_accepts) == 1
    assert int(solver.diagnostics.n_fallback_accepts) == 1


def test_postprocess_exposes_kkt_dual_qp_and_failure_statistics():
    """Active-set solutions expose the detailed diagnostics promised by the API."""
    problem = make_unconstrained_quadratic()
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(rtol=1e-5, atol=1e-5),
        jnp.ones(2),
        max_steps=20,
    )
    expected = {
        "final_objective",
        "final_grad_norm",
        "final_lagrangian_grad_norm",
        "kkt_scale",
        "kkt_ratio",
        "multipliers_eq",
        "multipliers_ineq",
        "multipliers_lb",
        "multipliers_ub",
        "qp_iterations",
        "qp_cg_iterations",
        "total_qp_iterations",
        "total_qp_cg_iterations",
        "qp_result",
        "qp_final_working_tol",
        "n_qp_anti_cycling",
        "kkt_feasibility_residual",
        "kkt_n_refinements",
        "kkt_reason",
        "n_lpeca_bypassed",
        "n_lpeca_capped",
        "n_lpeca_bounds_prefixed",
        "merit_penalty",
        "last_step_size",
        "last_ls_success",
        "last_ls_fallback",
        "n_armijo_accepts",
        "n_fallback_accepts",
        "consecutive_qp_failures",
        "consecutive_ls_failures",
        "secant_recovery_streak",
        "secant_recovery_stage",
        "secant_recovery_fatal",
        "sqpdax_result",
    }
    assert expected <= set(sol.stats)
    # Per-QP counts never exceed the totals over the nonlinear solve.
    assert 0 < int(sol.stats["qp_iterations"]) <= int(sol.stats["total_qp_iterations"])
    assert int(sol.stats["qp_cg_iterations"]) <= int(
        sol.stats["total_qp_cg_iterations"]
    )
    assert bool(sol.stats["qp_result"] == ACTIVE_SET_QP_RESULTS.working_set_converged)
    assert float(sol.stats["qp_final_working_tol"]) == pytest.approx(1e-5)
    assert int(sol.stats["n_qp_anti_cycling"]) == 0
    assert bool(sol.stats["kkt_reason"] == KKT_SOLVER_RESULTS.converged)
    assert int(sol.stats["kkt_n_refinements"]) == 0
    assert float(sol.stats["kkt_feasibility_residual"]) < 1e-6
    # Predictor off by default: no LPEC-A activity is recorded.
    assert sol.state.active_set_predictor.method == "expand"
    assert int(sol.stats["n_lpeca_bypassed"]) == 0
    assert int(sol.stats["n_lpeca_bounds_prefixed"]) == 0
    metrics = sol.state.termination_metrics(sol.state._optimisation_context(problem))
    assert float(sol.stats["kkt_ratio"]) == pytest.approx(
        float(metrics.stationarity) / float(sol.stats["kkt_scale"])
    )


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
def test_projector_option_selects_the_craig_backend(minimiser_cls):
    """``options['subproblem']['subproblem_solver']['projector']`` swaps the backend."""
    problem, x_star, dual_star = make_shifted_box_quadratic()
    projector = CraigProjector(max_iter=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sol = minimise(
            problem,
            minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=2),
            jnp.array([0.5, 0.5, 0.5]),
            max_steps=40,
            throw=False,
            options={"subproblem": {"subproblem_solver": {"projector": projector}}},
        )
    ctx = sol.state._init_subproblem(problem)
    assert isinstance(ctx.solver.subproblem_solver.projector, CraigProjector)
    assert ctx.solver.subproblem_solver.projector.max_iter == 50
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, x_star, atol=1e-4)
    assert jnp.allclose(
        sol.stats["multipliers_ineq"], dual_star.ineq_multipliers, atol=1e-3
    )
    assert bool(sol.stats["kkt_reason"] == KKT_SOLVER_RESULTS.converged)


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
def test_minres_qlp_option_selects_the_saddle_point_solver(minimiser_cls):
    """``options['subproblem']['subproblem_solver']`` accepts a MINRES-QLP solver."""
    problem, x_star, dual_star = make_shifted_box_quadratic()
    inner = MinresQLPSubProblemSolver(max_iter=50, proj_refine_max_iter=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sol = minimise(
            problem,
            minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=2),
            jnp.array([0.5, 0.5, 0.5]),
            max_steps=40,
            throw=False,
            options={"subproblem": {"subproblem_solver": inner}},
        )
    ctx = sol.state._init_subproblem(problem)
    assert isinstance(ctx.solver.subproblem_solver, MinresQLPSubProblemSolver)
    assert ctx.solver.subproblem_solver.max_iter == 50
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, x_star, atol=1e-4)
    assert jnp.allclose(
        sol.stats["multipliers_ineq"], dual_star.ineq_multipliers, atol=1e-3
    )
    assert bool(sol.stats["kkt_reason"] == KKT_SOLVER_RESULTS.converged)
    assert int(sol.stats["kkt_n_refinements"]) <= 2
    assert float(sol.stats["kkt_feasibility_residual"]) < 1e-6


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
def test_multiplier_recovery_option_is_wired_into_the_qp_solver(minimiser_cls):
    """``options['subproblem']['multiplier_recovery']`` selects the post-loop recovery."""
    problem, x_star, dual_star = make_shifted_box_quadratic()
    recovery = LeastSquaresMultiplierRecovery(safeguard=ClampSafeguard())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sol = minimise(
            problem,
            minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=2),
            jnp.array([0.5, 0.5, 0.5]),
            max_steps=40,
            throw=False,
            options={"subproblem": {"multiplier_recovery": recovery}},
        )
    ctx = sol.state._init_subproblem(problem)
    assert eqx.tree_equal(ctx.solver.multiplier_recovery, recovery)
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, x_star, atol=1e-4)
    # The committed multipliers are the Hessian-free LS estimate at the
    # solution, which coincides with the exact ones on this quadratic.
    assert jnp.allclose(
        sol.stats["multipliers_ineq"], dual_star.ineq_multipliers, atol=1e-3
    )
    assert jnp.allclose(
        sol.stats["multipliers_lb"], dual_star.lb_multipliers, atol=1e-3
    )
    assert jnp.all(sol.stats["multipliers_ineq"] >= 0)
    assert jnp.all(sol.stats["multipliers_lb"] >= 0)


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
@pytest.mark.parametrize("method", ["expand", "lpeca_init", "lpeca"])
def test_active_set_prediction_modes_reach_the_same_solution(minimiser_cls, method):
    """All predictor modes solve the box / inequality quadratic; stats reflect them."""
    problem, x_star, dual_star = make_shifted_box_quadratic()
    sol = minimise(
        problem,
        minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=2),
        jnp.array([0.5, 0.5, 0.5]),
        max_steps=40,
        throw=False,
        options={
            "minimiser": {"active_set_predictor": {"method": method, "warmup_steps": 0}}
        },
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, x_star, atol=1e-4)
    assert jnp.allclose(
        sol.stats["multipliers_ineq"], dual_star.ineq_multipliers, atol=1e-3
    )
    assert sol.state.active_set_predictor.method == method
    n_steps = int(sol.stats["num_steps"])
    assert n_steps >= 2
    if method == "expand":
        assert int(sol.stats["n_lpeca_bypassed"]) == 0
        assert int(sol.stats["n_lpeca_bounds_prefixed"]) == 0
    else:
        # Far from the solution the trust gate bypasses the first prediction;
        # once the iterate sits at ``x*`` with exact multipliers the active
        # bound is seeded on every remaining step.
        assert int(sol.stats["n_lpeca_bypassed"]) == 1
        assert int(sol.stats["n_lpeca_bounds_prefixed"]) == n_steps - 1
        assert int(sol.stats["n_lpeca_capped"]) == 0


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
@pytest.mark.parametrize(
    ("method", "expected_expand"),
    [("expand", 1.0), ("lpeca_init", 1.0), ("lpeca", 0.0)],
)
@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
def test_prediction_seeds_the_qp_carry_and_lpeca_disables_expand(
    minimiser_cls, method, expected_expand, qp_warm_start
):
    """Enabled prediction is written into the carried QP state with the seed forced on."""
    problem, x_star, dual_star = make_shifted_box_quadratic()
    predictor = LPECAPredictor(method=method, warmup_steps=0)
    solver = minimiser_cls(
        rtol=1e-6, atol=1e-6, qp_warm_start=qp_warm_start, min_steps=1
    ).init(
        problem,
        x_star,
        options={
            "minimiser": {"active_set_predictor": predictor},
            "subproblem": {"working_set_policy": {"expand_factor": 1.0}},
        },
    )
    assert solver.active_set_predictor is predictor
    # Exact multipliers make the KKT point exact, so the prediction is the
    # true active set. Pretend the carried state holds a stale bound too.
    solver = eqx.tree_at(lambda m: m.dual, solver, dual_star)
    stale = eqx.tree_at(
        lambda s: s.active_set.active_ub,
        solver.solver_state,
        jnp.array([False, False, True]),
    )
    solver = eqx.tree_at(lambda m: m.solver_state, solver, stale)

    seeded = solver._seed_predicted_active_set(problem)
    ctx = seeded._init_subproblem(problem)
    assert ctx.solver.working_set_policy.expand_factor == expected_expand
    seed = seeded.solver_state.active_set
    if method == "expand":
        assert seeded is solver
        assert ctx.solver.warm_start is qp_warm_start
        assert jnp.array_equal(seed.active_ub, stale.active_set.active_ub)
    else:
        assert ctx.solver.warm_start is True
        assert bool(eqx.tree_equal(ctx.state.active_set, seed))
        assert jnp.array_equal(seed.active_inequalities, jnp.array([True]))
        assert jnp.array_equal(seed.active_lb, jnp.array([False, True, False]))
        # The stale carried bound survives only when warm starts are requested.
        assert bool(seed.active_ub[2]) is qp_warm_start
        if not qp_warm_start:
            assert jnp.array_equal(
                seeded.solver_state.dual.lb_multipliers, dual_star.lb_multipliers
            )
        # Counters are advanced by the seeding itself (valid, uncapped, one bound).
        assert int(seeded.diagnostics.n_lpeca_bypassed) == 0
        assert int(seeded.diagnostics.n_lpeca_capped) == 0
        assert int(seeded.diagnostics.n_lpeca_bounds_prefixed) == 1
    # The seeded solve returns the KKT point immediately.
    stepped = solver.step(problem)
    assert jnp.allclose(stepped.iterate.x, x_star, atol=1e-6)
    if method != "expand":
        assert int(stepped.diagnostics.n_lpeca_bounds_prefixed) == 1
        assert int(stepped.diagnostics.n_lpeca_bypassed) == 0


@pytest.mark.parametrize(
    "minimiser_cls",
    [ActiveSetLineSearchMinimiser, ProximalActiveSetLineSearchMinimiser],
    ids=["active-set", "proximal"],
)
@pytest.mark.parametrize("via_options", [False, True], ids=["field", "options"])
def test_single_exchange_solves_the_inconsistent_refresh_problem(
    minimiser_cls, via_options
):
    """``qp_single_exchange`` swaps the policy class and keeps the ``qp_tol`` routing."""
    problem, x_star, dual_star = make_shifted_box_quadratic(c0=1.5)
    kwargs = dict(rtol=1e-6, atol=1e-6, min_steps=1, qp_tol=1e-7)
    options = None
    if via_options:
        options = {"minimiser": {"qp_single_exchange": True}}
    else:
        kwargs["qp_single_exchange"] = True
    solver = minimiser_cls(**kwargs).init(problem, jnp.array([0.5, 0.5, 0.5]), options)
    assert solver.qp_single_exchange is True
    policy = solver._init_subproblem(problem).solver.working_set_policy
    assert type(policy) is SingleExchangeWorkingSetPolicy
    assert policy.tol == 1e-7
    assert policy.max_iter == solver.qp_max_iter

    sol = minimise(
        problem,
        minimiser_cls(**kwargs),
        jnp.array([0.5, 0.5, 0.5]),
        max_steps=40,
        throw=False,
        options=options,
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, x_star, atol=1e-4)
    assert jnp.allclose(
        sol.stats["multipliers_ineq"], dual_star.ineq_multipliers, atol=1e-3
    )

    # The default all-at-once refresh stalls on this problem.
    default = minimise(
        problem,
        minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=1),
        jnp.array([0.5, 0.5, 0.5]),
        max_steps=40,
        throw=False,
    )
    assert not bool(default.state.result_adapter.is_successful(default.result))
