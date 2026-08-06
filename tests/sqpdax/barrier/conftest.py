"""Fixtures shared by :mod:`slsqp_jax.sqpdax.barrier` tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.interior_point import InteriorPointLagrangian
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem
from tests.sqpdax.lagrangian.conftest import make_dual, make_ip_primal, make_problem


def make_slack(n: int, mineq: int, *, fill: float = 2.0) -> Slack:
    """Build a strictly positive slack with constant block fills."""
    return Slack(
        s=jnp.full((mineq,), fill),
        s_lb=jnp.full((n,), fill + 1.0),
        s_ub=jnp.full((n,), fill + 2.0),
    )


def make_log_barrier(problem: Problem, *, weight: float = 1.0) -> LogBarrier:
    """Log barrier using the problem's null-bound masks."""
    return LogBarrier(
        weight=jnp.asarray(weight),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )


def make_evaluated_ip_lagrangian(
    *,
    problem: Problem | None = None,
    weight: float = 1.0,
    primal: InteriorPointPrimal | None = None,
    dual: Dual | None = None,
    primal_dual: bool = False,
):
    """Evaluate an :class:`InteriorPointLagrangian` at a default interior point."""
    if problem is None:
        problem = make_problem()
    barrier = make_log_barrier(problem, weight=weight)
    lag = InteriorPointLagrangian(
        problem, secant=None, barrier=barrier, primal_dual=primal_dual
    )
    if primal is None:
        primal = make_ip_primal(n=problem.n, mineq=problem.mineq)
    if dual is None:
        dual = make_dual(problem.n, problem.meq, problem.mineq)
    return lag(primal, dual), barrier


@pytest.fixture
def quadratic_problem() -> Problem:
    """Default 2-variable NLP used by barrier update tests."""
    return make_problem()
