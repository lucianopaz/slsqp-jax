"""Fixtures shared by :mod:`slsqp_jax.sqpdax.subproblem` tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.basic import Lagrangian
from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
from slsqp_jax.sqpdax.lagrangian.interior_point import InteriorPointLagrangian
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.scaled_barrier import ScaledBarrierSubProblem
from tests.sqpdax.lagrangian.conftest import (
    make_dual,
    make_ip_primal,
    make_primal,
    make_problem,
)


def make_active_set(
    meq: int,
    mineq: int,
    n: int,
    *,
    active_inequalities: tuple[bool, ...] | None = None,
    active_lb: tuple[bool, ...] | None = None,
    active_ub: tuple[bool, ...] | None = None,
) -> ActiveSet:
    """Build an :class:`ActiveSet` with optional explicit masks."""
    if active_inequalities is None:
        active_inequalities = tuple(i % 2 == 0 for i in range(mineq))
    if active_lb is None:
        active_lb = tuple(i % 2 == 0 for i in range(n))
    if active_ub is None:
        active_ub = tuple(i % 2 == 1 for i in range(n))
    return ActiveSet(
        meq=meq,
        active_inequalities=jnp.asarray(active_inequalities),
        active_lb=jnp.asarray(active_lb),
        active_ub=jnp.asarray(active_ub),
    )


def make_evaluated_lagrangian(
    *,
    problem: Problem | None = None,
    primal: Primal | None = None,
    dual: Dual | None = None,
) -> EvaluatedLagrangian[Primal]:
    """Evaluate a plain :class:`Lagrangian` at a default point."""
    if problem is None:
        problem = make_problem()
    if primal is None:
        primal = make_primal(n=problem.n)
    if dual is None:
        dual = make_dual(problem.n, problem.meq, problem.mineq)
    return Lagrangian(problem)(primal, dual)


def make_active_set_subproblem(
    *,
    problem: Problem | None = None,
    active_inequalities: tuple[bool, ...] | None = None,
    active_lb: tuple[bool, ...] | None = None,
    active_ub: tuple[bool, ...] | None = None,
) -> ActiveSetSubProblem:
    """Build an :class:`ActiveSetSubProblem` on the default quadratic NLP."""
    lag = make_evaluated_lagrangian(problem=problem)
    active = make_active_set(
        lag.meq,
        lag.mineq,
        lag.n,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    return ActiveSetSubProblem(lag, active)


def make_zero_dual(n: int, meq: int, mineq: int) -> Dual:
    """All-zero dual used as a tangent / RHS block."""
    return Dual(
        eq_multipliers=jnp.zeros((meq,)),
        ineq_multipliers=jnp.zeros((mineq,)),
        lb_multipliers=jnp.zeros((n,)),
        ub_multipliers=jnp.zeros((n,)),
    )


def make_step(
    n: int = 2,
    meq: int = 1,
    mineq: int = 2,
    *,
    dx: Array | None = None,
) -> tuple[Primal, Dual]:
    """Plain primal-dual step for active-set tests."""
    if dx is None:
        dx = jnp.linspace(0.1, -0.2, n)
    return Primal(x=dx), make_dual(n, meq, mineq)


def make_ip_evaluated(
    *,
    problem: Problem | None = None,
    weight: float = 0.5,
    dual_kkt_regularization: float = 0.0,
    primal_dual: bool = True,
):
    """Evaluate an interior-point Lagrangian (optionally primal-dual)."""
    if problem is None:
        problem = make_problem()
    barrier = LogBarrier(
        weight=jnp.asarray(weight),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    lag = InteriorPointLagrangian(
        problem,
        secant=None,
        barrier=barrier,
        dual_kkt_regularization=dual_kkt_regularization,
        primal_dual=primal_dual,
    )
    primal = make_ip_primal(n=problem.n, mineq=problem.mineq)
    dual = make_dual(problem.n, problem.meq, problem.mineq)
    return lag(primal, dual)


def make_scaled_barrier_subproblem(
    *,
    problem: Problem | None = None,
    weight: float = 0.5,
    dual_kkt_regularization: float = 0.0,
    tau: float = 0.995,
) -> ScaledBarrierSubProblem:
    """Build a :class:`ScaledBarrierSubProblem` on a primal-dual IP Lagrangian."""
    evaluated = make_ip_evaluated(
        problem=problem,
        weight=weight,
        dual_kkt_regularization=dual_kkt_regularization,
        primal_dual=True,
    )
    return ScaledBarrierSubProblem(evaluated, tau=tau)


def make_ip_step(
    n: int = 2,
    mineq: int = 2,
    meq: int = 1,
    *,
    fill: float = 0.1,
) -> tuple[InteriorPointPrimal, Dual]:
    """Interior-point primal-dual step with constant slack fills."""
    primal = InteriorPointPrimal(
        x=jnp.full((n,), fill),
        slack=Slack(
            s=jnp.full((mineq,), fill),
            s_lb=jnp.full((n,), fill),
            s_ub=jnp.full((n,), -fill),
        ),
    )
    return primal, make_dual(n, meq, mineq)


@pytest.fixture
def quadratic_problem() -> Problem:
    """Default 2-variable NLP used across subproblem tests."""
    return make_problem()
