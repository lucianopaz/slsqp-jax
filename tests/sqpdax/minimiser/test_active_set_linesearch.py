"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.active_set_linesearch`."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    ProximalActiveSetLineSearchMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller import StepResult
from slsqp_jax.sqpdax.subproblem.solver import ACTIVE_SET_QP_RESULTS, RESULTS

from .conftest import make_equality_quadratic, make_unconstrained_quadratic


def test_init_builds_primal_dual_and_qp_state():
    """``init`` seeds a plain ``Primal``, zero dual, and cold QP carry."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.ones(2))
    assert isinstance(solver.iterate, Primal)
    assert solver.dual is not None
    assert solver.solver_state is not None
    assert int(solver.solver_state.n_iter) == 0


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
    )
    return solver._execute_step(ctx, solver.dual, result)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("qp", ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure),
        ("ls", ACTIVE_SET_LINE_SEARCH_RESULTS.line_search_failure),
    ],
)
def test_failure_counters_wait_for_fatal_threshold(kind, expected):
    """QP and line-search failures become fatal only after their patience."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser(
        rtol=-1.0,
        qp_failure_patience=1,
        ls_failure_patience=1,
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))
    for index in range(2):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=kind == "qp",
            qp_success=kind == "ls",
            qp_status=RESULTS.successful if kind == "ls" else RESULTS.singular,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
        )
        done, _ = solver.terminate(problem)
        assert bool(done) is (index == 1)
    _, result = solver.terminate(problem)
    assert bool(result == expected)


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
        qp_failure_patience=1,
        ls_failure_patience=100,
        stagnation_patience=100,
    ).init(problem, jnp.ones(2))
    for _ in range(2):
        solver = _execute_synthetic_step(
            solver,
            problem,
            accepted=False,
            qp_success=False,
            qp_status=RESULTS.max_steps_reached,
            x=solver.iterate.x,
            merit=float(solver.best_merit),
            proposed_step_norm=proposed_step_norm,
        )
    done, result = solver.terminate(problem)
    assert bool(done) is expect_fatal
    assert bool(solver.qp_fatal) is expect_fatal
    if expect_fatal:
        assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure)
    else:
        assert int(solver.consecutive_qp_failures) == 0


def test_nan_curvature_terminates_with_qp_failure_without_moving():
    """End-to-end: a NaN QP direction never moves ``(x, λ)`` and exits cleanly.

    A NaN Hessian-vector product poisons the CG direction only; the objective,
    gradient and constraints stay finite so the termination check does not
    see it directly. The line search must reject without trials, the dual
    must stay at its previous (finite) value, and the run must end with
    ``qp_subproblem_failure`` rather than ``nonfinite``.
    """
    problem = replace(
        make_unconstrained_quadratic(),
        hvp=lambda x, v: jnp.full_like(v, jnp.nan),
    )
    x0 = jnp.ones(2)
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(qp_failure_patience=1, min_steps=1),
        x0,
        max_steps=20,
        throw=False,
    )
    assert bool(
        sol.stats["sqpdax_result"]
        == ACTIVE_SET_LINE_SEARCH_RESULTS.qp_subproblem_failure
    )
    assert jnp.array_equal(sol.value, x0)
    for key in (
        "multipliers_eq",
        "multipliers_ineq",
        "multipliers_lb",
        "multipliers_ub",
    ):
        assert jnp.all(jnp.isfinite(sol.stats[key]))
    assert int(sol.stats["consecutive_qp_failures"]) == 2


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
        "last_step_size",
        "consecutive_qp_failures",
        "consecutive_ls_failures",
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
    metrics = sol.state.termination_metrics(sol.state._optimisation_context(problem))
    assert float(sol.stats["kkt_ratio"]) == pytest.approx(
        float(metrics.stationarity) / float(sol.stats["kkt_scale"])
    )
