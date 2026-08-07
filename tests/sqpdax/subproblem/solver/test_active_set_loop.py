"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.active_set_loop`."""

from __future__ import annotations

import jax.numpy as jnp

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import (
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
    (dx_as, _), state = ActiveSetQPSolver().solve(sub, warm, make_active_set_qp_state())

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert jnp.all(jnp.isfinite(dx_as.x))
    assert jnp.allclose(dx_as.x, dx_pcg.x, atol=1e-6)
