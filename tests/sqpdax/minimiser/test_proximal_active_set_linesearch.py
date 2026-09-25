"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.proximal_active_set_linesearch`."""

from __future__ import annotations

import warnings

import equinox as eqx
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
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    ProximalActiveSetQPSolver,
    ProximalActiveSetQPSolverState,
)
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import make_equality_quadratic, make_unconstrained_quadratic


@pytest.mark.parametrize(
    "with_curvature", [True, False], ids=["exact-hvp", "secant-preconditioned"]
)
def test_init_seeds_proximal_state_and_solver(with_curvature: bool):
    """``init`` seeds a proximal carry; the QP solver is the proximal loop."""
    problem = make_problem(with_curvature=with_curvature)
    solver = ProximalActiveSetLineSearchMinimiser().init(problem, jnp.ones(2))
    state = solver.solver_state
    assert isinstance(state, ProximalActiveSetQPSolverState)
    assert jnp.isposinf(state.kkt_residual)
    assert float(state.mu) == pytest.approx(ProximalActiveSetQPSolver().mu_max)
    assert state.eq_center.shape == (problem.meq,)
    assert jnp.allclose(state.eq_center, 0.0)

    ctx = solver._init_subproblem(problem)
    assert isinstance(ctx.solver, ProximalActiveSetQPSolver)
    has_pre = ctx.solver.subproblem_solver.preconditioner is not None
    assert has_pre is (not with_curvature)
    # The outer KKT residual is finite and written into the carry.
    assert jnp.isfinite(ctx.state.kkt_residual)
    assert ctx.state.kkt_residual > 0


def test_init_subproblem_residual_matches_definition():
    """``kkt_residual = max(‖∇_x L(x, λ)‖_∞, feasibility_∞)`` at the current dual."""
    problem = make_equality_quadratic()
    solver = ProximalActiveSetLineSearchMinimiser().init(problem, jnp.array([0.0, 0.0]))
    solver = eqx.tree_at(lambda m: m.dual.eq_multipliers, solver, jnp.asarray([0.5]))
    ctx = solver._init_subproblem(problem)
    lag = ctx.lagrangian(solver.iterate, solver.dual)
    expected = jnp.maximum(
        jnp.max(jnp.abs(lag.x_grad)),
        ProximalActiveSetLineSearchMinimiser._feasibility_from_lagrangian(lag),
    )
    assert jnp.allclose(ctx.state.kkt_residual, expected)


@pytest.mark.parametrize(
    ("problem_factory", "x0", "expected"),
    [
        (make_unconstrained_quadratic, (1.0, 1.0), (0.0, 0.0)),
        (make_equality_quadratic, (0.0, 0.0), (0.5, 0.5)),
        (lambda: make_problem(with_curvature=False), (0.5, 0.5), None),
    ],
    ids=["unconstrained", "equality", "mixed-secant"],
)
def test_minimise_converges_and_matches_plain_minimiser(problem_factory, x0, expected):
    """The proximal outer loop reaches the same solution as the plain active-set one."""
    problem = problem_factory()
    # Arrays are built inside the test so their dtype follows the x64 flag at
    # run time (other modules may enable x64 after this one is collected).
    x0 = jnp.asarray(x0)
    expected = None if expected is None else jnp.asarray(expected)
    kwargs = dict(rtol=1e-6, atol=1e-6, min_steps=1)
    sol = minimise(
        problem,
        ProximalActiveSetLineSearchMinimiser(**kwargs),
        x0,
        max_steps=60,
        throw=False,
    )
    ref = minimise(
        problem,
        ActiveSetLineSearchMinimiser(**kwargs),
        x0,
        max_steps=60,
        throw=False,
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, ref.value, atol=1e-4)
    if expected is not None:
        assert jnp.allclose(sol.value, expected, atol=1e-4)
    assert {"proximal_mu", "proximal_kkt_residual", "multipliers_eq"} <= set(sol.stats)
    assert jnp.isfinite(sol.stats["proximal_mu"])


def test_first_proximal_step_is_stationary_but_not_fatal():
    """A stationary-yet-infeasible iterate does not trip the fatal test early."""
    problem = make_equality_quadratic()
    solver = ProximalActiveSetLineSearchMinimiser(rtol=1e-6, atol=1e-6, min_steps=1)
    solver = solver.init(problem, jnp.array([0.0, 0.0])).step(problem)
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    assert metrics.stationarity <= solver.rtol * metrics.stationarity_scale
    assert metrics.feasibility > solver.atol
    done, result = solver.terminate(problem)
    assert not bool(done)
    assert bool(result == ACTIVE_SET_LINE_SEARCH_RESULTS.running)


def test_mu_floor_survives_tight_atol_and_options_are_accepted():
    """``atol=1e-14`` cannot push ``μ`` below ``mu_min``; option bag is recognised."""
    problem = make_equality_quadratic()
    options = {"subproblem": {"tau": 0.3, "mu_min": 1e-4, "mu_max": 0.5}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        minimiser = ProximalActiveSetLineSearchMinimiser(
            rtol=1e-6, atol=1e-14, min_steps=1
        ).init(problem, jnp.array([0.0, 0.0]), options=options)
    assert float(minimiser.solver_state.mu) == pytest.approx(0.5)
    ctx = minimiser._init_subproblem(problem)
    assert ctx.solver.tau == 0.3
    assert ctx.solver.mu_min == 1e-4
    # ``minimise`` re-runs ``init``, so the option bag must be passed again.
    sol = minimise(
        problem,
        minimiser,
        jnp.array([0.0, 0.0]),
        max_steps=60,
        throw=False,
        options=options,
    )
    assert float(sol.stats["proximal_mu"]) >= 1e-4
    assert jnp.isfinite(1.0 / sol.stats["proximal_mu"])
    assert jnp.allclose(sol.value, jnp.array([0.5, 0.5]), atol=1e-4)


def test_rollback_resyncs_eq_center_with_committed_dual():
    """After a blow-up rollback the proximal centre follows the restored dual."""
    problem = make_equality_quadratic()
    solver = ProximalActiveSetLineSearchMinimiser(
        rtol=-1.0,
        divergence_factor=1.0,
        divergence_patience=1,
        stagnation_patience=100,
    ).init(problem, jnp.array([0.0, 0.0]))
    best_dual = solver.dual
    ctx = solver._init_subproblem(problem)
    bad_state = eqx.tree_at(
        lambda s: (s.success, s.status, s.eq_center),
        solver.solver_state,
        (jnp.asarray(True), RESULTS.successful, jnp.asarray([42.0])),
    )
    result = StepResult(
        x=Primal(jnp.asarray([10.0, 10.0])),
        accepted=jnp.asarray(True),
        merit_val=jnp.asarray(1e6),
        solver_state=bad_state,
        step_size=jnp.asarray(1.0),
        proposed_step_norm=jnp.asarray(1.0),
    )
    diverged_dual = eqx.tree_at(
        lambda d: d.eq_multipliers, solver.dual, jnp.asarray([42.0])
    )
    rolled = solver._execute_step(ctx, diverged_dual, result)
    assert bool(rolled.iterate_blowup)
    assert jnp.allclose(rolled.dual.eq_multipliers, best_dual.eq_multipliers)
    assert jnp.allclose(rolled.solver_state.eq_center, best_dual.eq_multipliers)
    # Without a rollback the centre equals the committed (new) dual.
    accepted = ProximalActiveSetLineSearchMinimiser(rtol=1e-6, atol=1e-6).init(
        problem, jnp.array([0.0, 0.0])
    )
    accepted = accepted.step(problem)
    assert jnp.allclose(accepted.solver_state.eq_center, accepted.dual.eq_multipliers)


@pytest.mark.parametrize("proposed", [7.0, jnp.nan], ids=["finite", "nan"])
def test_rejected_step_commits_dual_and_eq_center_iff_finite(proposed):
    """On a rejected line search the centre follows the committed dual.

    A finite recovered multiplier is committed even though the primal did not
    move (a method-of-multipliers update at fixed ``x``), and the proximal
    centre ``λ_k`` follows it. A NaN proposal (from a non-finite direction)
    is discarded and both stay at the previous multipliers.
    """
    problem = make_equality_quadratic()
    solver = ProximalActiveSetLineSearchMinimiser(rtol=-1.0).init(
        problem, jnp.array([0.0, 0.0])
    )
    solver = solver.step(problem)  # establish a non-trivial (x, λ, centre)
    old_dual = solver.dual
    old_center = solver.solver_state.eq_center
    assert jnp.all(jnp.isfinite(old_center))

    ctx = solver._init_subproblem(problem)
    qp_state = eqx.tree_at(
        lambda s: (s.success, s.status, s.eq_center),
        solver.solver_state,
        (jnp.asarray(False), RESULTS.max_steps_reached, jnp.asarray([proposed])),
    )
    result = StepResult(
        x=solver.iterate,
        accepted=jnp.asarray(False),
        merit_val=jnp.asarray(float(solver.best_merit)),
        solver_state=qp_state,
        step_size=jnp.asarray(0.0),
        proposed_step_norm=jnp.asarray(proposed),
    )
    proposed_dual = eqx.tree_at(
        lambda d: d.eq_multipliers, old_dual, jnp.asarray([proposed])
    )
    rejected = solver._execute_step(ctx, proposed_dual, result)

    expected = jnp.asarray([proposed]) if jnp.isfinite(proposed) else old_center
    assert jnp.array_equal(rejected.dual.eq_multipliers, expected)
    assert jnp.array_equal(rejected.solver_state.eq_center, expected)
    assert jnp.array_equal(rejected.iterate.x, solver.iterate.x)


def test_exports_roundtrip():
    """Public re-exports resolve from the minimiser and sqpdax packages."""
    from slsqp_jax import sqpdax
    from slsqp_jax.sqpdax import minimiser

    assert (
        minimiser.ProximalActiveSetLineSearchMinimiser
        is ProximalActiveSetLineSearchMinimiser
    )
    assert (
        sqpdax.ProximalActiveSetLineSearchMinimiser
        is ProximalActiveSetLineSearchMinimiser
    )
