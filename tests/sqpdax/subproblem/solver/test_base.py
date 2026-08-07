"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.base`."""

from __future__ import annotations

import jax.numpy as jnp

from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    DogLegSolverState,
    ProjectedCGState,
    ProjectedCGSubProblemSolver,
)


def test_requires_secant_false_on_projected_cg():
    """Leaf solvers default ``requires_secant`` to ``False``."""
    assert ProjectedCGSubProblemSolver().requires_secant() is False


def test_concrete_solver_states_construct():
    """Concrete carry types accept the documented required fields."""
    pcg = ProjectedCGState(
        n_iter=jnp.asarray(0, jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
    )
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
    assert dogleg.radius.shape == ()
    assert dogleg.active_bounds is None
