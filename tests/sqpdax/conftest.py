"""sqpdax-specific pytest fixtures.

The root-level ``tests/conftest.py`` is picked up automatically through
pytest's fixture inheritance via the package hierarchy.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.types import Aux


def make_shifted_box_quadratic(
    n: int = 3, *, c0: float = 0.95
) -> tuple[Problem, Array, Dual]:
    """``min ‖x - c‖²`` with one strongly active inequality and one active bound.

    ``c = (c₀, -2, 0, …)``, ``h(x) = x₀ - x₁ - 1.9 <= 0``, ``lb = (0, -1, …)``,
    ``ub = (1, 3, …)``. For ``0.9 < c₀ < 1.9`` the solution is
    ``x* = (0.9, -1, 0, …)`` with ``λ_h = 2c₀ - 1.8`` and
    ``λ_lb = (0, 3.8 - 2c₀, 0, …)``; every other constraint is inactive with
    a positive slack, so the active set is non-degenerate.

    With the default ``c₀ = 0.95`` only ``h`` and ``lb₁`` are violated at the
    unconstrained minimiser ``c``, so the all-at-once working-set refresh
    lands on a consistent set. With ``c₀ > 1`` the upper bound ``ub₀`` is
    violated too, and ``{ub₀, lb₁, h}`` is inconsistent (``x₀ = 1``,
    ``x₁ = -1`` give ``h = 0.1``): adding all three at once produces a KKT
    system without a solution, which only a one-at-a-time exchange avoids.

    Parameters
    ----------
    n
        Dimension (``>= 2``); extra coordinates are free with minimiser ``0``.
    c0
        First coordinate of the unconstrained minimiser.

    Returns
    -------
    problem, x_star, dual_star
        The NLP, its minimiser and the exact KKT multipliers.
    """
    c = jnp.zeros(n).at[0].set(c0).at[1].set(-2.0)
    lb = jnp.full(n, -1.0).at[0].set(0.0)
    ub = jnp.full(n, 3.0).at[0].set(1.0)
    e0 = jnp.zeros(n).at[0].set(1.0)
    e1 = jnp.zeros(n).at[1].set(1.0)

    def fn(x: Array) -> tuple[Array, Aux]:
        return (jnp.sum((x - c) ** 2), None)

    def grad(x: Array) -> Array:
        return 2.0 * (x - c)

    def hvp(x: Array, v: Array) -> Array:
        return 2.0 * v

    def eq_fn(x: Array) -> Array:
        return jnp.zeros((0,), x.dtype)

    def eq_jac(x: Array) -> Array:
        return jnp.zeros((0, n), x.dtype)

    def eq_hvp(x: Array, v: Array) -> Array:
        return jnp.zeros((0, n), x.dtype)

    def ineq_fn(x: Array) -> Array:
        return jnp.array([x[0] - x[1] - 1.9])

    def ineq_jac(x: Array) -> Array:
        return (e0 - e1)[None, :]

    def ineq_hvp(x: Array, v: Array) -> Array:
        return jnp.zeros((1, n), x.dtype)

    problem = Problem(
        fn=fn,
        grad=grad,
        hvp=hvp,
        eq_fn=eq_fn,
        eq_fn_jac=eq_jac,
        eq_fn_hvp=eq_hvp,
        ineq_fn=ineq_fn,
        ineq_fn_jac=ineq_jac,
        ineq_fn_hvp=ineq_hvp,
        lb=lb,
        ub=ub,
        null_lb=jnp.zeros(n, bool),
        null_ub=jnp.zeros(n, bool),
        n=n,
        meq=0,
        mineq=1,
    )
    x_star = jnp.zeros(n).at[0].set(0.9).at[1].set(-1.0)
    dual_star = Dual(
        eq_multipliers=jnp.zeros((0,)),
        ineq_multipliers=jnp.array([2.0 * c0 - 1.8]),
        lb_multipliers=jnp.zeros(n).at[1].set(3.8 - 2.0 * c0),
        ub_multipliers=jnp.zeros(n),
    )
    return problem, x_star, dual_star
