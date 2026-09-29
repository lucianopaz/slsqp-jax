"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.base`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.preconditioner import (
    DiagonalPreconditioner,
    IdentityPreconditioner,
)
from slsqp_jax.sqpdax.subproblem.solver import (
    KKT_SOLVER_RESULTS,
    RESULTS,
    ActiveSetQPSolver,
    DogLegSolver,
    DogLegSolverState,
    GradientProjection,
    MinresQLPSubProblemSolver,
    ProjectedCGState,
    ProjectedCGSubProblemSolver,
    ProximalActiveSetQPSolver,
    SteihaugTointCGTangentialStepSolver,
    SubProblemSolver,
    TrustRegionInteriorPointSolver,
)

PRECONDITIONABLE_SOLVERS = {
    "projected-cg": ProjectedCGSubProblemSolver,
    "minres-qlp": MinresQLPSubProblemSolver,
    "active-set": ActiveSetQPSolver,
    "active-set-minres": lambda: ActiveSetQPSolver(
        subproblem_solver=MinresQLPSubProblemSolver()
    ),
    "proximal": ProximalActiveSetQPSolver,
}

UNPRECONDITIONED_SOLVERS = {
    "dogleg": DogLegSolver,
    "steihaug-toint": SteihaugTointCGTangentialStepSolver,
    "trust-region": TrustRegionInteriorPointSolver,
    "gradient-projection": GradientProjection,
}


def _leaf(solver: SubProblemSolver) -> SubProblemSolver:
    """Innermost solver reached through ``subproblem_solver`` delegation."""
    while hasattr(solver, "subproblem_solver"):
        solver = solver.subproblem_solver
    return solver


@pytest.mark.parametrize(
    "make_solver",
    PRECONDITIONABLE_SOLVERS.values(),
    ids=PRECONDITIONABLE_SOLVERS.keys(),
)
def test_with_default_preconditioner_fills_only_empty_slots(make_solver):
    """A default fills an empty slot, never overrides a set one, ``None`` is a no-op."""
    solver = make_solver()
    default = DiagonalPreconditioner(jnp.array([1.0, 2.0]))
    assert solver.accepts_preconditioner() is True
    assert eqx.tree_equal(solver.with_default_preconditioner(None), solver)
    assert _leaf(solver).preconditioner is None

    filled = solver.with_default_preconditioner(default)
    assert type(filled) is type(solver)
    assert eqx.tree_equal(_leaf(filled).preconditioner, default)

    kept = filled.with_default_preconditioner(IdentityPreconditioner(jnp.zeros(2)))
    assert eqx.tree_equal(_leaf(kept).preconditioner, default)


@pytest.mark.parametrize(
    "make_solver",
    UNPRECONDITIONED_SOLVERS.values(),
    ids=UNPRECONDITIONED_SOLVERS.keys(),
)
def test_with_default_preconditioner_is_noop_without_slot(make_solver):
    """Solvers without a preconditioner slot decline and return themselves."""
    solver = make_solver()
    default = DiagonalPreconditioner(jnp.array([1.0, 2.0]))
    assert solver.accepts_preconditioner() is False
    assert solver.with_default_preconditioner(default) is solver


def test_requires_secant_false_on_projected_cg():
    """Leaf solvers default ``requires_secant`` to ``False``."""
    assert ProjectedCGSubProblemSolver().requires_secant() is False


def test_concrete_solver_states_construct():
    """Concrete carry types accept the documented required fields."""
    pcg = ProjectedCGState.cold(jnp.float32)
    dogleg = DogLegSolverState(
        n_iter=jnp.asarray(0, jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
        n_cg_iter=jnp.asarray(0, jnp.int32),
        on_boundary=jnp.asarray(False),
        radius=jnp.asarray(1.0),
        active_bounds=None,
    )
    assert pcg.status == RESULTS.successful
    assert pcg.reason == KKT_SOLVER_RESULTS.converged
    assert pcg.feasibility_residual.dtype == jnp.float32
    assert bool(jnp.isinf(pcg.projected_grad_norm))
    assert not bool(pcg.nonfinite)
    assert dogleg.radius.shape == ()
    assert dogleg.active_bounds is None
