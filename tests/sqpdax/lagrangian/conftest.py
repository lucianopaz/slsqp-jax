"""Fixtures shared by :mod:`slsqp_jax.sqpdax.lagrangian` tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem


def obj(x: Array) -> Array:
    return jnp.sum(x**2)


def obj_grad(x: Array) -> Array:
    return 2 * x


def obj_hvp(x: Array, v: Array) -> Array:
    return 2 * v


def eq_fn(x: Array) -> Array:
    return jnp.array([x[0] + x[1] - 1.0])


def eq_jac(x: Array) -> Array:
    return jnp.array([[1.0, 1.0]])


def eq_hvp(x: Array, v: Array) -> Array:
    return jnp.zeros((1, x.shape[-1]), dtype=x.dtype)


def ineq_fn(x: Array) -> Array:
    return jnp.array([x[0] - 2.0, -x[1]])


def ineq_jac(x: Array) -> Array:
    return jnp.array([[1.0, 0.0], [0.0, -1.0]])


def ineq_hvp(x: Array, v: Array) -> Array:
    return jnp.zeros((2, x.shape[-1]), dtype=x.dtype)


def empty_fn(x: Array) -> Array:
    return jnp.zeros((0,), dtype=x.dtype)


def empty_jac(x: Array) -> Array:
    return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)


def empty_hvp(x: Array, v: Array) -> Array:
    return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)


def make_problem(
    *,
    n: int = 2,
    meq: int = 1,
    mineq: int = 2,
    with_curvature: bool = True,
    lb: Array | None = None,
    ub: Array | None = None,
) -> Problem:
    if lb is None:
        lb = jnp.array([0.0, -1.0])
    if ub is None:
        ub = jnp.array([2.0, 3.0])
    use_eq = meq > 0
    use_ineq = mineq > 0
    return Problem(
        fn=obj,
        grad=obj_grad,
        hvp=obj_hvp if with_curvature else None,
        eq_fn=eq_fn if use_eq else empty_fn,
        ineq_fn=ineq_fn if use_ineq else empty_fn,
        eq_fn_jac=eq_jac if use_eq else empty_jac,
        ineq_fn_jac=ineq_jac if use_ineq else empty_jac,
        eq_fn_hvp=(eq_hvp if use_eq else empty_hvp) if with_curvature else None,
        ineq_fn_hvp=(ineq_hvp if use_ineq else empty_hvp) if with_curvature else None,
        lb=lb,
        ub=ub,
        null_lb=jnp.isinf(lb) & (lb < 0),
        null_ub=jnp.isinf(ub) & (ub > 0),
        n=n,
        meq=meq,
        mineq=mineq,
    )


def make_dual(n: int, meq: int, mineq: int) -> Dual:
    return Dual(
        eq_multipliers=jnp.arange(meq, dtype=jnp.float32) + 0.5,
        ineq_multipliers=jnp.arange(mineq, dtype=jnp.float32) + 0.25,
        lb_multipliers=jnp.linspace(0.1, 0.2, n),
        ub_multipliers=jnp.linspace(0.05, 0.15, n),
    )


def make_primal(n: int = 2) -> Primal:
    return Primal(x=jnp.linspace(0.25, 0.75, n))


def make_ip_primal(n: int = 2, mineq: int = 2) -> InteriorPointPrimal:
    return InteriorPointPrimal(
        x=jnp.linspace(0.25, 0.75, n),
        slack=Slack(
            s=jnp.full((mineq,), 1.5),
            s_lb=jnp.full((n,), 1.25),
            s_ub=jnp.full((n,), 1.75),
        ),
    )


@pytest.fixture
def quadratic_problem() -> Problem:
    """Default 2-variable problem with equalities, inequalities, and HVPs."""
    return make_problem()
