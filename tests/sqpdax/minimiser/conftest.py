"""Fixtures shared by :mod:`slsqp_jax.sqpdax.minimiser` tests."""

from __future__ import annotations

from slsqp_jax.sqpdax.minimiser import ActiveSetLineSearchMinimiser
from slsqp_jax.sqpdax.problem import Problem
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import unbounded_box

# Back-compat alias used by the existing CommonMinimiser tests.
ActiveSetLineSearchStub = ActiveSetLineSearchMinimiser


def make_unconstrained_quadratic(*, n: int = 2) -> Problem:
    """``f(x) = ‖x‖²`` with exact HVP and no constraints / bounds."""
    lb, ub = unbounded_box(n)
    return make_problem(n=n, meq=0, mineq=0, lb=lb, ub=ub, with_curvature=True)


def make_equality_quadratic(*, n: int = 2) -> Problem:
    """``f(x) = ‖x‖²`` subject to ``x₀ + x₁ = 1`` (unbounded)."""
    lb, ub = unbounded_box(n)
    return make_problem(n=n, meq=1, mineq=0, lb=lb, ub=ub, with_curvature=True)
