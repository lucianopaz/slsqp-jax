"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.projected_cg`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.preconditioner import IdentityPreconditioner, MatrixPreconditioner
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import (
    KKT_SOLVER_RESULTS,
    RESULTS,
    CraigProjector,
    KKTSolverState,
    ProjectedCGSubProblemSolver,
    SVDProjector,
)
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


class _FailingProjector(SVDProjector):
    """SVD projector that reports its (fictitious) inner solve as failed."""

    def build(self, subproblem, preconditioner=None):
        ctx = super().build(subproblem, preconditioner)
        return eqx.tree_at(
            lambda c: (c.converged, c.n_iter),
            ctx,
            (jnp.asarray(False), jnp.asarray(3, jnp.int32)),
        )


@pytest.mark.parametrize(
    ("solver_kwargs", "x_shift", "reason", "status"),
    [
        # ``tol`` loose enough that an exact 1-D solve meets it in both
        # float32 and float64 rather than freezing on its roundoff floor.
        ({"tol": 1e-6}, 0.0, KKT_SOLVER_RESULTS.converged, RESULTS.successful),
        ({"tol": 0.0}, 0.0, KKT_SOLVER_RESULTS.residual_floor, RESULTS.successful),
        (
            {"max_iter": 0},
            0.0,
            KKT_SOLVER_RESULTS.max_iter_reached,
            RESULTS.max_steps_reached,
        ),
        (
            {"projector": _FailingProjector()},
            0.0,
            KKT_SOLVER_RESULTS.projector_failure,
            RESULTS.max_steps_reached,
        ),
        ({}, jnp.nan, KKT_SOLVER_RESULTS.nonfinite, RESULTS.singular),
    ],
    ids=["converged", "residual-floor", "max-iter", "projector-failure", "nonfinite"],
)
def test_kkt_state_classifies_the_solve(solver_kwargs, x_shift, reason, status):
    """``ProjectedCGState`` carries the standardised KKT diagnostics."""
    n = 2
    lb, ub = unbounded_box(n)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    primal = Primal(make_primal(n=n).x + x_shift)
    sub = make_qp_subproblem(problem=problem, primal=primal)
    solver = ProjectedCGSubProblemSolver(**solver_kwargs)
    warm = (Primal(jnp.zeros(n)), make_zero_dual(n, 1, 0))
    state0 = make_projected_cg_state()
    (dx, _), state = solver.solve(sub, warm, state0)

    assert isinstance(state, KKTSolverState)
    assert state.reason == reason
    assert state.status == status
    assert bool(state.success) == (status == RESULTS.successful)
    assert bool(state.nonfinite) == (reason == KKT_SOLVER_RESULTS.nonfinite)
    assert int(state.n_refinements) == 0
    assert state.feasibility_residual.dtype == state0.feasibility_residual.dtype
    assert state.projected_grad_norm.dtype == state0.projected_grad_norm.dtype
    if reason == KKT_SOLVER_RESULTS.nonfinite:
        assert not bool(jnp.isfinite(dx.x).all())
    else:
        # Structurally feasible step; projected gradient tracks the CG floor.
        assert float(state.feasibility_residual) < 1e-5
        assert bool(jnp.isfinite(state.projected_grad_norm))
        if reason == KKT_SOLVER_RESULTS.max_iter_reached:
            assert float(state.projected_grad_norm) > 1e-3
        else:
            assert float(state.projected_grad_norm) < 1e-4
    # Inner projector work is folded into the iteration count.
    n_inner = 3 if isinstance(solver.projector, _FailingProjector) else 0
    assert int(state.n_iter) >= n_inner


@pytest.mark.parametrize(
    "preconditioner",
    [None, MatrixPreconditioner(make_spd_matrix(2))],
    ids=["none", "matrix"],
)
def test_craig_projector_reproduces_the_svd_step(preconditioner):
    """The matrix-free projector gives the same KKT step and multipliers."""
    n = 2
    lb = jnp.array([0.0, -jnp.inf])
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=jnp.full(n, jnp.inf))
    sub = make_qp_subproblem(
        problem=problem, primal=make_primal(n=n), active_lb=(True, False)
    )
    warm = (Primal(jnp.zeros(n)), make_zero_dual(n, 1, 0))
    svd_solver = ProjectedCGSubProblemSolver(preconditioner=preconditioner)
    craig_solver = ProjectedCGSubProblemSolver(
        preconditioner=preconditioner, projector=CraigProjector()
    )
    (dx_s, lam_s), st_s = svd_solver.solve(sub, warm, make_projected_cg_state())
    (dx_c, lam_c), st_c = craig_solver.solve(sub, warm, make_projected_cg_state())

    assert jnp.allclose(dx_c.x, dx_s.x, atol=1e-5)
    assert jnp.allclose(lam_c.flatten(), lam_s.flatten(), atol=1e-4)
    assert bool(st_c.success)
    assert st_c.reason == st_s.reason
    # CRAIG's bidiagonalisation steps are folded into the iteration count.
    assert int(st_c.n_iter) > int(st_s.n_iter)


def test_craig_breakdown_is_reported_as_projector_failure():
    """Parallel working rows: CRAIG cannot converge and the state says so."""
    sub = make_qp_subproblem(active_inequalities=(False, True), active_lb=(True, False))
    warm = (Primal(jnp.zeros(2)), make_zero_dual(2, 1, 2))
    solver = ProjectedCGSubProblemSolver(projector=CraigProjector())
    (dx, _), state = solver.solve(sub, warm, make_projected_cg_state())
    assert state.reason == KKT_SOLVER_RESULTS.projector_failure
    assert not bool(state.success)
    assert state.status == RESULTS.max_steps_reached
    assert jnp.all(jnp.isfinite(dx.x))
