"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.projected_cg`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.preconditioner import IdentityPreconditioner, MatrixPreconditioner
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, ProjectedCGSubProblemSolver
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.preconditioner.conftest import make_spd_matrix
from tests.sqpdax.subproblem.conftest import (
    make_scaled_barrier_subproblem,
    make_zero_dual,
)

from .conftest import make_projected_cg_state, make_qp_subproblem, unbounded_box


@pytest.mark.parametrize(
    "preconditioner",
    [
        None,
        IdentityPreconditioner(jnp.zeros(2)),
        MatrixPreconditioner(make_spd_matrix(2)),
    ],
    ids=["none", "identity", "matrix"],
)
def test_equality_only_analytic_kkt(preconditioner):
    """Feasible equality QP: ``dx ≈ [0.25, -0.25]``, ``λ_eq ≈ -1``."""
    n = 2
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    primal = make_primal(n=n)
    sub = make_qp_subproblem(problem=problem, primal=primal)
    solver = ProjectedCGSubProblemSolver(preconditioner=preconditioner)
    warm = (Primal(jnp.zeros(n)), make_zero_dual(n, 1, 0))
    (dx, lam), state = solver.solve(sub, warm, make_projected_cg_state())

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert jnp.allclose(dx.x, jnp.array([0.25, -0.25]), atol=1e-5)
    assert jnp.allclose(lam.eq_multipliers, jnp.array([-1.0]), atol=1e-4)
    resid_p, resid_d = sub.residual((dx, lam))
    assert jnp.linalg.norm(resid_p.flatten()) < 1e-4
    assert jnp.linalg.norm(resid_d.flatten()) < 1e-4


def test_wrong_subproblem_type_raises():
    """Scaled-barrier models are rejected with ``TypeError``."""
    sub = make_scaled_barrier_subproblem()
    solver = ProjectedCGSubProblemSolver()
    lag = sub.lagrangian
    warm = (
        Primal(jnp.zeros(lag.n)),
        make_zero_dual(lag.n, lag.meq, lag.mineq),
    )
    with pytest.raises(TypeError, match="ActiveSetSubProblem"):
        solver.solve(sub, warm, make_projected_cg_state())


def test_active_lower_bound_fixes_component():
    """Active lower bound pins ``dx_i`` to the bound target from ``kkt_rhs``."""
    n = 2
    lb = jnp.array([0.0, -jnp.inf])
    ub = jnp.full(n, jnp.inf)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    primal = make_primal(n=n)
    sub = make_qp_subproblem(
        problem=problem,
        primal=primal,
        active_lb=(True, False),
        active_ub=(False, False),
    )
    solver = ProjectedCGSubProblemSolver()
    warm = (Primal(jnp.zeros(n)), make_zero_dual(n, 1, 0))
    (dx, _), _state = solver.solve(sub, warm, make_projected_cg_state())

    _sol_p, sol_d = sub.kkt_rhs()
    expected_fixed = -sol_d.lb_multipliers[0]
    assert jnp.all(jnp.isfinite(dx.x))
    assert jnp.allclose(dx.x[0], expected_fixed, atol=1e-6)
