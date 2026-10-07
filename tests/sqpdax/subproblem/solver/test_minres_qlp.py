"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.minres_qlp`."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.preconditioner import MatrixPreconditioner
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    KKT_SOLVER_RESULTS,
    RESULTS,
    ActiveSetQPSolver,
    CraigProjector,
    KKTSolverState,
    MinresQLPSubProblemSolver,
    ProjectedCGSubProblemSolver,
    SVDProjectionContext,
    SVDProjector,
)
from slsqp_jax.sqpdax.subproblem.solver.minres_qlp import (
    _pminres_qlp_solve,
    _sym_ortho,
)
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.preconditioner.conftest import make_spd_matrix
from tests.sqpdax.subproblem.conftest import (
    make_scaled_barrier_subproblem,
    make_zero_dual,
)

from .conftest import (
    FailingProjector,
    make_active_set_qp_state,
    make_minres_qlp_state,
    make_projected_cg_state,
    make_qp_subproblem,
    unbounded_box,
)

# --------------------------------------------------------------------------- #
# Working-set QPs
# --------------------------------------------------------------------------- #


def _equality_free(x_shift: float = 0.0):
    lb, ub = unbounded_box(2)
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=ub)
    primal = Primal(make_primal(n=2).x + x_shift)
    return make_qp_subproblem(problem=problem, primal=primal)


def _equality_free_nan():
    return _equality_free(x_shift=jnp.nan)


def _equality_with_fixed_variable():
    lb = jnp.array([0.0, -jnp.inf])
    problem = make_problem(meq=1, mineq=0, lb=lb, ub=jnp.full(2, jnp.inf))
    return make_qp_subproblem(
        problem=problem, primal=make_primal(n=2), active_lb=(True, False)
    )


def _equality_and_active_inequality():
    lb, ub = unbounded_box(2)
    problem = make_problem(meq=1, mineq=2, lb=lb, ub=ub)
    return make_qp_subproblem(problem=problem, active_inequalities=(True, False))


def _duplicated_equality_rows():
    """Two identical equality rows: singular but consistent working set."""
    lb, ub = unbounded_box(2)
    problem = make_problem(meq=2, mineq=0, lb=lb, ub=ub)
    problem = replace(
        problem,
        eq_fn=lambda x: jnp.array([x[0] + x[1] - 1.0, x[0] + x[1] - 1.0]),
        eq_fn_jac=lambda x: jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        eq_fn_hvp=lambda x, v: jnp.zeros((2, x.shape[-1]), dtype=x.dtype),
    )
    return make_qp_subproblem(problem=problem, primal=make_primal(n=2))


def _inconsistent_equality_rows():
    """Parallel equality rows with different targets: no feasible step."""
    lb, ub = unbounded_box(2)
    problem = make_problem(meq=2, mineq=0, lb=lb, ub=ub)
    problem = replace(
        problem,
        eq_fn=lambda x: jnp.array([x[0], x[0] - 1.0]),
        eq_fn_jac=lambda x: jnp.array([[1.0, 0.0], [1.0, 0.0]]),
        eq_fn_hvp=lambda x, v: jnp.zeros((2, x.shape[-1]), dtype=x.dtype),
    )
    return make_qp_subproblem(problem=problem, primal=make_primal(n=2))


SUBPROBLEMS = {
    "equality": _equality_free,
    "fixed-variable": _equality_with_fixed_variable,
    "active-inequality": _equality_and_active_inequality,
    "duplicated-rows": _duplicated_equality_rows,
}

PRECONDITIONERS = {
    "none": lambda: None,
    "matrix": lambda: MatrixPreconditioner(make_spd_matrix(2)),
}

PROJECTORS = {
    "svd": SVDProjector,
    "craig": CraigProjector,
}


def _warm(sub):
    lag = sub.lagrangian
    return (Primal(jnp.zeros(lag.n)), make_zero_dual(lag.n, lag.meq, lag.mineq))


class _DampedSVDContext(SVDProjectionContext):
    """Range-space solve scaled by ``damping``: each projection only removes part of the residual."""

    damping: float = 0.5

    def solve_preconditioned_normal(self, rhs):
        return self.damping * super().solve_preconditioned_normal(rhs)


class _DampedProjector(SVDProjector):
    """SVD projector whose feasibility projection converges geometrically."""

    def build(self, subproblem, preconditioner=None):
        ctx = super().build(subproblem, preconditioner)
        fields = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}
        return _DampedSVDContext(**fields)


# --------------------------------------------------------------------------- #
# Krylov kernel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("a", "b"),
    [(3.0, 4.0), (-3.0, 4.0), (4.0, -3.0), (0.0, 2.0), (2.0, 0.0), (0.0, 0.0)],
)
def test_sym_ortho_is_a_rotation(a, b):
    """``(c, s, r)`` satisfy ``r = √(a²+b²)``, ``c a + s b = r`` and ``c² + s² = 1``."""
    c, s, r = _sym_ortho(jnp.asarray(a), jnp.asarray(b))
    assert jnp.isclose(r, jnp.hypot(a, b))
    assert jnp.isclose(c * a + s * b, r)
    assert jnp.isclose(c**2 + s**2, 1.0)
    assert jnp.isclose(-s * a + c * b, 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("matrix", "precondition"),
    [
        (jnp.diag(jnp.array([1.0, -2.0, 3.0])), False),
        (jnp.diag(jnp.array([1.0, 0.0, 2.0])), False),
        (jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]), True),
        (jnp.diag(jnp.array([1.0, 1e-6, 1.0])), True),
    ],
    ids=["indefinite", "singular", "spd-preconditioned", "ill-conditioned"],
)
def test_pminres_qlp_matches_pseudo_inverse(matrix: Array, precondition: bool):
    """On consistent systems the kernel returns the minimum-length solution."""
    x_star = (
        jnp.array([1.0, 0.0, -2.0])
        if matrix[1, 1] == 0
        else jnp.array([1.0, 2.0, -1.0])
    )
    rhs = matrix @ x_star
    precond = (lambda v: v / jnp.diag(matrix)) if precondition else None
    x, converged, n_iter = _pminres_qlp_solve(
        lambda v: matrix @ v, rhs, tol=1e-8, max_iter=50, precond=precond
    )
    assert bool(converged)
    # A 3-D Krylov space is exhausted in at most three steps; the Lanczos
    # breakdown test may need a couple more to notice.
    assert 0 < int(n_iter) <= 10
    assert jnp.allclose(x, jnp.linalg.pinv(matrix, rcond=1e-12) @ rhs, atol=1e-4)


def test_pminres_qlp_zero_rhs_returns_zero_without_iterating():
    """A zero right-hand side is already solved."""
    matrix = jnp.eye(3)
    x, converged, n_iter = _pminres_qlp_solve(
        lambda v: matrix @ v, jnp.zeros(3), tol=1e-10, max_iter=10
    )
    assert bool(converged)
    assert int(n_iter) == 0
    assert jnp.all(x == 0)


# --------------------------------------------------------------------------- #
# KKT solver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("make_sub", SUBPROBLEMS.values(), ids=SUBPROBLEMS.keys())
@pytest.mark.parametrize(
    "make_pre", PRECONDITIONERS.values(), ids=PRECONDITIONERS.keys()
)
@pytest.mark.parametrize("projector_cls", PROJECTORS.values(), ids=PROJECTORS.keys())
def test_agrees_with_projected_cg(make_sub, make_pre, projector_cls):
    """MINRES-QLP reproduces the projected-CG step and multipliers."""
    sub = make_sub()
    pre = make_pre()
    warm = _warm(sub)
    minres = MinresQLPSubProblemSolver(preconditioner=pre, projector=projector_cls())
    pcg = ProjectedCGSubProblemSolver(preconditioner=pre, projector=projector_cls())

    @eqx.filter_jit
    def solve_both(sub, warm):
        return (
            minres.solve(sub, warm, make_minres_qlp_state()),
            pcg.solve(sub, warm, make_projected_cg_state()),
        )

    ((dx_m, lam_m), st_m), ((dx_p, lam_p), st_p) = solve_both(sub, warm)

    assert bool(st_m.success)
    assert st_m.status == RESULTS.successful
    assert st_m.reason == KKT_SOLVER_RESULTS.converged
    assert jnp.allclose(dx_m.x, dx_p.x, atol=1e-5)
    assert jnp.allclose(lam_m.flatten(), lam_p.flatten(), atol=1e-4)
    # The returned step is feasible and stationary on the working set.
    res_p, res_d = sub.residual((dx_m, lam_m))
    free, _ = sub.free_subspace()
    assert jnp.linalg.norm(jnp.where(free, res_p.flatten(), 0.0)) < 1e-5
    active = sub.active_constraint_rows()
    general = jnp.concatenate([res_d.eq_multipliers, res_d.ineq_multipliers])
    assert jnp.linalg.norm(jnp.where(active, general, 0.0)) < 1e-5
    assert float(st_m.feasibility_residual) < 1e-6
    assert float(st_m.projected_grad_norm) < 1e-5
    assert int(st_m.n_iter) > 0


def test_wrong_primal_layout_raises():
    """Slack-augmented models are rejected with ``TypeError``."""
    sub = make_scaled_barrier_subproblem()
    lag = sub.lagrangian
    warm = (Primal(jnp.zeros(lag.n)), make_zero_dual(lag.n, lag.meq, lag.mineq))
    with pytest.raises(TypeError, match="decision variables only"):
        MinresQLPSubProblemSolver().solve(sub, warm, make_minres_qlp_state())


def test_warm_start_at_the_solution_needs_no_lanczos_steps():
    """Seeding with the exact ``(d, λ)`` makes the Krylov phase a no-op."""
    sub = _equality_with_fixed_variable()
    solver = MinresQLPSubProblemSolver()
    step, st_cold = solver.solve(sub, _warm(sub), make_minres_qlp_state())
    step_warm, st_warm = solver.solve(sub, step, make_minres_qlp_state())

    assert int(st_cold.n_iter) > 0
    assert int(st_warm.n_iter) == 0
    assert bool(st_warm.success)
    assert jnp.allclose(step_warm[0].x, step[0].x, atol=1e-8)
    assert jnp.allclose(step_warm[1].flatten(), step[1].flatten(), atol=1e-8)


@pytest.mark.parametrize(
    ("solver_kwargs", "make_sub", "reason", "status", "refinements"),
    [
        ({}, _equality_free, KKT_SOLVER_RESULTS.converged, RESULTS.successful, 0),
        (
            {"max_iter": 1},
            _equality_free,
            KKT_SOLVER_RESULTS.max_iter_reached,
            RESULTS.max_steps_reached,
            0,
        ),
        # One refinement round is attempted, does not help, and is counted.
        (
            {},
            _inconsistent_equality_rows,
            KKT_SOLVER_RESULTS.residual_floor,
            RESULTS.stagnation,
            1,
        ),
        # A single Lanczos step leaves an infeasible iterate; the damped
        # projector halves the residual per round and runs out of budget.
        (
            {
                "max_iter": 1,
                "projector": _DampedProjector(),
                "proj_refine_max_iter": 2,
            },
            _equality_free,
            KKT_SOLVER_RESULTS.residual_floor,
            RESULTS.stagnation,
            2,
        ),
        # ...or reaches the (loosened) target after a handful of rounds; the
        # exhausted Krylov budget is then what is reported.
        (
            {
                "max_iter": 1,
                "projector": _DampedProjector(),
                "proj_refine_max_iter": 60,
                "proj_refine_rtol": 1e-4,
            },
            _equality_free,
            KKT_SOLVER_RESULTS.max_iter_reached,
            RESULTS.max_steps_reached,
            None,
        ),
        (
            {"projector": FailingProjector()},
            _equality_free,
            KKT_SOLVER_RESULTS.projector_failure,
            RESULTS.max_steps_reached,
            0,
        ),
        ({}, _equality_free_nan, KKT_SOLVER_RESULTS.nonfinite, RESULTS.singular, 1),
    ],
    ids=[
        "converged",
        "max-iter",
        "inconsistent-rows",
        "refinement-floor",
        "refinement-converges",
        "projector-failure",
        "nonfinite",
    ],
)
def test_kkt_state_classifies_the_solve(
    solver_kwargs, make_sub, reason, status, refinements
):
    """``MinresQLPState`` carries the standardised KKT diagnostics."""
    sub = make_sub()
    solver = MinresQLPSubProblemSolver(**solver_kwargs)
    state0 = make_minres_qlp_state()
    (dx, _), state = solver.solve(sub, _warm(sub), state0)

    assert isinstance(state, KKTSolverState)
    assert state.reason == reason
    assert state.status == status
    assert bool(state.success) == (status == RESULTS.successful)
    assert bool(state.nonfinite) == (reason == KKT_SOLVER_RESULTS.nonfinite)
    assert state.feasibility_residual.dtype == state0.feasibility_residual.dtype
    assert state.projected_grad_norm.dtype == state0.projected_grad_norm.dtype
    if refinements is None:
        # Geometric refinement: ran some rounds but stopped before the budget.
        assert 0 < int(state.n_refinements) < solver.proj_refine_max_iter
    else:
        assert int(state.n_refinements) == refinements
    if reason == KKT_SOLVER_RESULTS.nonfinite:
        assert not bool(jnp.isfinite(dx.x).all())
    elif reason == KKT_SOLVER_RESULTS.residual_floor:
        assert float(state.feasibility_residual) > 1e-8
        assert jnp.all(jnp.isfinite(dx.x))
    else:
        assert float(state.feasibility_residual) <= 1e-4 * (
            1.0 + float(jnp.linalg.norm(dx.x))
        )
    n_inner = 3 if isinstance(solver.projector, FailingProjector) else 0
    assert int(state.n_iter) >= n_inner


def test_inconsistent_working_set_is_tolerated_by_projected_cg_but_not_minres():
    """Only MINRES-QLP turns an infeasible working set into a KKT failure.

    Documents the deliberate asymmetry: projected CG returns its
    least-squares-feasible iterate as a usable step, MINRES-QLP reports the
    feasibility floor so the outer loop counts a solver failure.
    """
    sub = _inconsistent_equality_rows()
    (_, _), st_p = ProjectedCGSubProblemSolver().solve(
        sub, _warm(sub), make_projected_cg_state()
    )
    (_, _), st_m = MinresQLPSubProblemSolver().solve(
        sub, _warm(sub), make_minres_qlp_state()
    )
    assert bool(st_p.success)
    assert not bool(st_m.success)
    assert jnp.isclose(st_m.feasibility_residual, st_p.feasibility_residual, atol=1e-6)


# --------------------------------------------------------------------------- #
# Inside the active-set loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("make_sub", "expected_result", "expected_status", "expected_reason"),
    [
        (
            _equality_and_active_inequality,
            ACTIVE_SET_QP_RESULTS.working_set_converged,
            RESULTS.successful,
            KKT_SOLVER_RESULTS.converged,
        ),
        (
            _inconsistent_equality_rows,
            ACTIVE_SET_QP_RESULTS.kkt_solver_failure,
            RESULTS.stagnation,
            KKT_SOLVER_RESULTS.residual_floor,
        ),
    ],
    ids=["converged", "residual-floor"],
)
def test_active_set_loop_forwards_minres_diagnostics(
    make_sub, expected_result, expected_status, expected_reason
):
    """The loop accepts MINRES-QLP as inner solver and forwards its state.

    A feasibility floor above the refinement target surfaces as
    ``kkt_solver_failure`` with a ``stagnation`` status, which the outer
    minimiser counts as a real QP failure.
    """
    sub = make_sub()
    lag = sub.lagrangian
    solver = ActiveSetQPSolver(subproblem_solver=MinresQLPSubProblemSolver())
    (dx, lam), state = solver.solve(
        sub, _warm(sub), make_active_set_qp_state(lag.n, lag.meq, lag.mineq)
    )
    assert jnp.all(jnp.isfinite(dx.x))
    assert jnp.all(jnp.isfinite(lam.flatten()))
    assert bool(state.qp_result == expected_result)
    assert state.status == expected_status
    assert state.last_kkt_reason == expected_reason
    assert state.last_kkt_feasibility_residual.dtype == lag.ref.x.dtype
    if expected_result == ACTIVE_SET_QP_RESULTS.working_set_converged:
        (dx_p, lam_p), _ = ActiveSetQPSolver().solve(
            sub, _warm(sub), make_active_set_qp_state(lag.n, lag.meq, lag.mineq)
        )
        assert jnp.allclose(dx.x, dx_p.x, atol=1e-5)
        assert jnp.allclose(lam.flatten(), lam_p.flatten(), atol=1e-4)
        assert float(state.last_kkt_feasibility_residual) < 1e-6
        assert int(state.last_kkt_n_refinements) == 0
    else:
        assert float(state.last_kkt_feasibility_residual) > 1e-8
        assert int(state.last_kkt_n_refinements) == 1
