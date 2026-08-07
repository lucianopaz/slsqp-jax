"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.gradient_projection`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, GradientProjection
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import make_dual, make_zero_dual

from .conftest import make_qp_subproblem, unbounded_box


@pytest.mark.parametrize("cauchy_only", [True, False], ids=["cauchy", "subspace"])
def test_unbounded_cauchy_newton_step(cauchy_only: bool):
    """Unbounded box: Cauchy / free-space step recovers ``d ≈ -x`` for ``‖x‖²``."""
    n = 2
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=0, mineq=0, lb=lb, ub=ub)
    primal = make_primal(n=n)
    sub = make_qp_subproblem(problem=problem, primal=primal)
    solver = GradientProjection(cauchy_only=cauchy_only)
    dual = make_zero_dual(n, 0, 0)
    warm = (Primal(jnp.zeros(n)), dual)
    (dx, out_dual), state = solver.solve(sub, warm, solver.init_state(sub, warm))

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert jnp.allclose(dx.x, -primal.x, atol=1e-6)
    assert jnp.allclose(out_dual.flatten(), dual.flatten())


@pytest.mark.parametrize("cauchy_only", [True, False], ids=["cauchy", "subspace"])
def test_finite_box_feasible_finite(cauchy_only: bool):
    """Finite box: accepted step stays inside ``primal_box`` and is finite."""
    problem = make_problem(meq=0, mineq=0)
    primal = make_primal(n=problem.n)
    sub = make_qp_subproblem(problem=problem, primal=primal)
    solver = GradientProjection(cauchy_only=cauchy_only)
    dual = make_dual(problem.n, 0, 0)
    warm = (Primal(jnp.zeros(problem.n)), dual)
    (dx, out_dual), state = solver.solve(sub, warm, solver.init_state(sub, warm))

    lo, hi = sub.primal_box()
    assert bool(state.success)
    assert jnp.all(jnp.isfinite(dx.x))
    assert jnp.all(dx.x >= lo - 1e-8)
    assert jnp.all(dx.x <= hi + 1e-8)
    assert jnp.allclose(out_dual.flatten(), dual.flatten())


@pytest.mark.parametrize("cauchy_only", [True, False], ids=["cauchy", "subspace"])
def test_find_active_bounds_masks(cauchy_only: bool):
    """``find_active_bounds`` returns boolean masks of length ``n``."""
    problem = make_problem(meq=0, mineq=0)
    sub = make_qp_subproblem(problem=problem)
    solver = GradientProjection(cauchy_only=cauchy_only)
    active_lb, active_ub = solver.find_active_bounds(sub)

    assert active_lb.shape == (problem.n,)
    assert active_ub.shape == (problem.n,)
    assert active_lb.dtype == jnp.bool_
    assert active_ub.dtype == jnp.bool_
