"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.active_set_linesearch`."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser import ActiveSetLineSearchMinimiser, minimise
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller import StepResult
from slsqp_jax.sqpdax.subproblem.solver import RESULTS

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
        {"minimiser": {"qp_tol": 1e-7}, "subproblem": {"tol": 1e-7}},
    ],
    ids=["default", "with-options"],
)
def test_init_accepts_option_bag(options):
    """Recognised minimiser / subproblem options are applied without error."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.ones(2), options=options)
    assert solver.iterate is not None
    if options is not None:
        assert solver.qp_tol == 1e-7
        # Exercise the ``subproblem`` option path inside ``_init_subproblem``.
        solver = solver.step(problem)
        assert int(solver.step_count) == 1


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
    state = eqx.tree_at(
        lambda s: (s.success, s.status),
        solver.solver_state,
        (jnp.asarray(qp_success), qp_status),
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
        "last_step_size",
        "consecutive_qp_failures",
        "consecutive_ls_failures",
        "sqpdax_result",
    }
    assert expected <= set(sol.stats)
    metrics = sol.state.termination_metrics(sol.state._optimisation_context(problem))
    assert float(sol.stats["kkt_ratio"]) == pytest.approx(
        float(metrics.stationarity) / float(sol.stats["kkt_scale"])
    )
