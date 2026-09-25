"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.active_set_loop`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    RESULTS,
    ActiveSetQPSolver,
    ProjectedCGSubProblemSolver,
)
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import make_zero_dual

from .conftest import (
    make_active_set_qp_state,
    make_projected_cg_state,
    make_qp_subproblem,
    unbounded_box,
)


def make_bound_hit_qp():
    """``min ‖x‖²`` on ``[0.5, 2] × [-1, 3]`` from ``x = (1, 0.75)``.

    The cold working set is empty (no bound is active at ``x``), the
    unconstrained step lands below ``lb₀``, so the loop must add ``lb₀`` and
    re-solve: exactly two working-set iterations to reach ``x + dx = (0.5, 0)``.
    """
    problem = make_problem(
        meq=0, mineq=0, lb=jnp.array([0.5, -1.0]), ub=jnp.array([2.0, 3.0])
    )
    primal = Primal(jnp.array([1.0, 0.75]))
    sub = make_qp_subproblem(problem=problem, primal=primal)
    warm = (Primal(jnp.zeros(2)), make_zero_dual(2, 0, 0))
    return sub, primal, warm


def test_bound_constrained_quadratic_to_origin():
    """Bound-constrained ``min ‖x‖²`` on ``[0, 1]²`` recovers ``x + dx ≈ 0``."""
    n = 2
    problem = make_problem(
        meq=0,
        mineq=0,
        lb=jnp.zeros(n),
        ub=jnp.ones(n),
    )
    primal = make_primal(n=n)
    sub = make_qp_subproblem(problem=problem, primal=primal)
    solver = ActiveSetQPSolver()
    warm = (Primal(jnp.zeros(n)), make_zero_dual(n, 0, 0))
    (dx, _), state = solver.solve(sub, warm, make_active_set_qp_state())

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert bool(state.qp_result == ACTIVE_SET_QP_RESULTS.working_set_converged)
    assert jnp.all(jnp.isfinite(dx.x))
    assert jnp.allclose(primal.x + dx.x, 0.0, atol=1e-5)


def test_equality_matches_bare_projected_cg():
    """Cold active-set QP recovers the bare projected-CG equality step."""
    n = 2
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    primal = make_primal(n=n)
    sub = make_qp_subproblem(problem=problem, primal=primal)
    warm = (Primal(jnp.zeros(n)), make_zero_dual(n, 1, 0))

    (dx_pcg, _), _ = ProjectedCGSubProblemSolver().solve(
        sub, warm, make_projected_cg_state()
    )
    (dx_as, _), state = ActiveSetQPSolver().solve(
        sub, warm, make_active_set_qp_state(n=n, meq=1)
    )

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert jnp.all(jnp.isfinite(dx_as.x))
    assert jnp.allclose(dx_as.x, dx_pcg.x, atol=1e-6)


@pytest.mark.parametrize(
    ("solver_kwargs", "expected_result", "expected_status", "expected_success"),
    [
        (
            {"max_iter": 5},
            ACTIVE_SET_QP_RESULTS.working_set_converged,
            RESULTS.successful,
            True,
        ),
        (
            {"max_iter": 1},
            ACTIVE_SET_QP_RESULTS.max_iter_reached,
            RESULTS.max_steps_reached,
            False,
        ),
        (
            {
                "max_iter": 5,
                "subproblem_solver": ProjectedCGSubProblemSolver(max_iter=0),
            },
            ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
            RESULTS.max_steps_reached,
            False,
        ),
    ],
    ids=["converged", "budget-exhausted", "kkt-failure"],
)
def test_qp_result_distinguishes_termination_reasons(
    solver_kwargs, expected_result, expected_status, expected_success
):
    """``qp_result`` separates outcomes that ``status`` alone conflates."""
    sub, _, warm = make_bound_hit_qp()
    solver = ActiveSetQPSolver(**solver_kwargs)
    _, state = solver.solve(sub, warm, make_active_set_qp_state())

    assert bool(state.qp_result == expected_result)
    assert state.status == expected_status
    assert bool(state.success) is expected_success


def test_iteration_budget_is_per_solve_and_totals_accumulate():
    """A carry that already spent ``max_iter`` iterations does not starve the next solve."""
    sub, primal, warm = make_bound_hit_qp()
    solver = ActiveSetQPSolver(max_iter=2)

    (dx1, _), s1 = solver.solve(sub, warm, make_active_set_qp_state())
    (dx2, _), s2 = solver.solve(sub, warm, s1)

    for state in (s1, s2):
        assert bool(state.success)
        assert int(state.last_n_iter) == 2
    assert int(s1.n_iter) == 2
    assert int(s2.n_iter) == 4
    assert int(s2.n_cg_iter) == int(s1.n_cg_iter) + int(s2.last_n_cg_iter)
    assert int(s1.n_cg_iter) == int(s1.last_n_cg_iter) > 0
    assert jnp.allclose(dx1.x, dx2.x)
    assert jnp.allclose(primal.x + dx2.x, jnp.array([0.5, 0.0]), atol=1e-5)


@pytest.mark.parametrize("warm_start", [False, True], ids=["cold", "warm"])
def test_warm_start_reuses_carried_working_set(warm_start: bool):
    """The carried set / dual are consumed only under ``warm_start``."""
    assert ActiveSetQPSolver().warm_start is False
    sub, primal, warm = make_bound_hit_qp()
    solver = ActiveSetQPSolver(warm_start=warm_start)

    (dx1, dual1), s1 = solver.solve(sub, warm, make_active_set_qp_state())
    assert int(s1.last_n_iter) == 2
    assert jnp.array_equal(s1.active_set.active_lb, jnp.array([True, False]))
    assert not jnp.any(s1.active_set.active_ub)
    for carried, returned in zip(jax.tree.leaves(s1.dual), jax.tree.leaves(dual1)):
        assert jnp.array_equal(carried, returned)

    # Re-solving the same QP from the carried state: a warm start already
    # holds the correct set and settles in one iteration, a cold start redoes
    # the add step.
    (dx2, _), s2 = solver.solve(sub, warm, s1)
    assert int(s2.last_n_iter) == (1 if warm_start else 2)
    assert bool(s2.success)
    assert jnp.allclose(dx2.x, dx1.x, atol=1e-6)
    assert jnp.allclose(primal.x + dx2.x, jnp.array([0.5, 0.0]), atol=1e-5)
