"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.utils`."""

from __future__ import annotations

from slsqp_jax.sqpdax.minimiser import minimiser_option_keys, solver_option_keys
from slsqp_jax.sqpdax.subproblem.solver import (
    ActiveSetQPSolver,
    ProjectedCGSubProblemSolver,
)

from .conftest import ActiveSetLineSearchStub


def test_minimiser_option_keys_includes_static_and_excludes_options():
    """Static tunables are recognised; ``options`` itself is not a key."""
    keys = minimiser_option_keys(ActiveSetLineSearchStub)
    assert "rtol" in keys
    assert "atol" in keys
    assert "qp_tol" in keys
    assert "options" not in keys


def test_solver_option_keys_exclude_lagrangian():
    """Subproblem option keys omit the per-step ``lagrangian`` injection."""
    keys = solver_option_keys(ProjectedCGSubProblemSolver)
    assert "tol" in keys
    assert "max_iter" in keys
    assert "lagrangian" not in keys
    # Nested ActiveSetQPSolver still exposes its own fields.
    as_keys = solver_option_keys(ActiveSetQPSolver)
    assert "subproblem_solver" in as_keys
    assert "tol" in as_keys
