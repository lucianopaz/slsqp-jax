"""Fixtures shared by :mod:`slsqp_jax.sqpdax.minimiser` tests."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from slsqp_jax.sqpdax.minimiser import ActiveSetLineSearchMinimiser
from slsqp_jax.sqpdax.problem import Problem
from slsqp_jax.sqpdax.types import Aux
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


def make_scaled_quartic(*, with_curvature: bool = True) -> Problem:
    """Badly scaled ``Σ wᵢ xᵢ² + ¼ x₀⁴`` on ``Σ xᵢ = 1`` with ``x₀ >= 0.2``.

    ``w = (1, 10, 100)`` makes the Hessian diagonal span two orders of
    magnitude and the quartic term makes it iterate-dependent, so exact
    curvature, the secant and both preconditioners behave differently while
    sharing one minimiser (the inequality and the box are inactive there).

    Parameters
    ----------
    with_curvature
        Whether the problem exposes exact objective / constraint HVPs.

    Returns
    -------
    Problem
        Three-variable NLP with one equality and one inequality.
    """
    w = jnp.array([1.0, 10.0, 100.0])
    e0 = jnp.array([1.0, 0.0, 0.0])

    def fn(x: Array) -> tuple[Array, Aux]:
        return (jnp.sum(w * x**2) + 0.25 * x[0] ** 4, None)

    def grad(x: Array) -> Array:
        return 2.0 * w * x + x[0] ** 3 * e0

    def hvp(x: Array, v: Array) -> Array:
        return 2.0 * w * v + 3.0 * x[0] ** 2 * v[0] * e0

    def zero_hvp(m: int):
        return lambda x, v: jnp.zeros((m, x.shape[-1]), dtype=x.dtype)

    lb = jnp.full(3, -5.0)
    ub = jnp.full(3, 5.0)
    return Problem(
        fn=fn,
        grad=grad,
        hvp=hvp if with_curvature else None,
        eq_fn=lambda x: jnp.array([jnp.sum(x) - 1.0]),
        ineq_fn=lambda x: jnp.array([0.2 - x[0]]),
        eq_fn_jac=lambda x: jnp.ones((1, 3), dtype=x.dtype),
        ineq_fn_jac=lambda x: -e0[None, :].astype(x.dtype),
        eq_fn_hvp=zero_hvp(1) if with_curvature else None,
        ineq_fn_hvp=zero_hvp(1) if with_curvature else None,
        lb=lb,
        ub=ub,
        null_lb=jnp.zeros(3, dtype=bool),
        null_ub=jnp.zeros(3, dtype=bool),
        n=3,
        meq=1,
        mineq=1,
    )
