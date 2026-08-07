"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.dogleg`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, DogLegSolver
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import make_zero_dual

from .conftest import make_dogleg_state, make_qp_subproblem, unbounded_box


def _unbounded_eq_problem(*, primal: Primal):
    n = primal.x.shape[0]
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    return make_qp_subproblem(problem=problem, primal=primal)


@pytest.mark.parametrize(
    ("primal", "radius", "expected_dx", "expect_boundary"),
    [
        (make_primal(n=2), 10.0, jnp.zeros(2), False),
        (Primal(jnp.zeros(2)), 10.0, jnp.array([0.5, 0.5]), False),
        (Primal(jnp.zeros(2)), 1e-3, jnp.array([0.5, 0.5]), True),
    ],
    ids=["feasible-large", "infeasible-gn", "infeasible-tiny"],
)
def test_dogleg_normal_step(primal, radius, expected_dx, expect_boundary):
    """Feasible → zero step; infeasible → Gauss–Newton or trust-region boundary."""
    sub = _unbounded_eq_problem(primal=primal)
    solver = DogLegSolver()
    warm = (Primal(jnp.zeros(2)), make_zero_dual(2, 1, 0))
    (dx, dual), state = solver.solve(sub, warm, make_dogleg_state(radius))
    step_norm = jnp.linalg.norm(dx.x)

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert bool(state.on_boundary) is expect_boundary
    assert jnp.allclose(dual.flatten(), 0.0, atol=1e-12)
    # Interior: match expected direction; boundary: saturate the radius.
    assert expect_boundary or jnp.allclose(dx.x, expected_dx, atol=1e-6)
    assert (not expect_boundary) or jnp.isclose(step_norm, radius, rtol=1e-5, atol=1e-8)


def test_active_bounds_none_uses_gradient_projection():
    """``active_bounds=None`` still succeeds (GradientProjection path)."""
    sub = _unbounded_eq_problem(primal=Primal(jnp.zeros(2)))
    solver = DogLegSolver()
    warm = (Primal(jnp.zeros(2)), make_zero_dual(2, 1, 0))
    state0 = make_dogleg_state(1.0, active_bounds=None)
    (dx, _), state = solver.solve(sub, warm, state0)

    assert bool(state.success)
    assert state.active_bounds is not None
    assert state.active_bounds[0].shape == (2,)
    assert jnp.all(jnp.isfinite(dx.x))
