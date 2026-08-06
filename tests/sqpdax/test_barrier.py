"""Unit tests for :mod:`slsqp_jax.sqpdax.barrier`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.barrier import EvaluatedBarrier, LogBarrier
from slsqp_jax.sqpdax.primal import Slack


def _slack(n: int, mineq: int, *, fill: float = 2.0) -> Slack:
    return Slack(
        s=jnp.full((mineq,), fill),
        s_lb=jnp.full((n,), fill + 1.0),
        s_ub=jnp.full((n,), fill + 2.0),
    )


@pytest.mark.parametrize(
    ("n", "mineq", "weight", "null_lb", "null_ub"),
    [
        (2, 1, 1.0, (False, False), (False, False)),
        (2, 2, 0.5, (True, False), (False, True)),
        (3, 0, 2.0, (False, True, False), (True, False, False)),
        (2, 1, 0.0, (False, False), (False, False)),
    ],
    ids=["full", "partial-null", "no-ineq", "zero-weight"],
)
def test_log_barrier_value_grad_hvp(
    n: int,
    mineq: int,
    weight: float,
    null_lb: tuple[bool, ...],
    null_ub: tuple[bool, ...],
):
    """Log barrier value / grad / HVP match the closed form (with null masks)."""
    barrier = LogBarrier(
        weight=jnp.asarray(weight),
        null_lb=jnp.asarray(null_lb),
        null_ub=jnp.asarray(null_ub),
    )
    slack = _slack(n, mineq, fill=2.0)
    evaluated = barrier(slack)

    assert isinstance(evaluated, EvaluatedBarrier)
    assert evaluated.original is barrier
    assert jnp.allclose(evaluated.weight, weight)

    if weight == 0.0:
        expected_val = 0.0
    else:
        expected_val = -weight * (
            jnp.sum(jnp.log(slack.s))
            + jnp.sum(jnp.log(jnp.where(barrier.null_lb, 1.0, slack.s_lb)))
            + jnp.sum(jnp.log(jnp.where(barrier.null_ub, 1.0, slack.s_ub)))
        )
    assert jnp.allclose(evaluated.fn_val, expected_val)
    assert jnp.allclose(barrier.fn(slack), expected_val)

    grad = barrier.grad(slack)
    if weight == 0.0:
        assert jnp.allclose(grad.s, jnp.zeros_like(slack.s))
        assert jnp.allclose(grad.s_lb, jnp.zeros_like(slack.s_lb))
        assert jnp.allclose(grad.s_ub, jnp.zeros_like(slack.s_ub))
    else:
        assert jnp.allclose(grad.s, -weight / slack.s)
        assert jnp.allclose(
            grad.s_lb,
            jnp.where(barrier.null_lb, 0.0, -weight / slack.s_lb),
        )
        assert jnp.allclose(
            grad.s_ub,
            jnp.where(barrier.null_ub, 0.0, -weight / slack.s_ub),
        )

    tangent = Slack(
        s=jnp.ones_like(slack.s),
        s_lb=0.5 * jnp.ones_like(slack.s_lb),
        s_ub=-0.25 * jnp.ones_like(slack.s_ub),
    )
    hvp = barrier.hvp(slack, tangent)
    if weight == 0.0:
        assert jnp.allclose(hvp.s, jnp.zeros_like(slack.s))
        assert jnp.allclose(hvp.s_lb, jnp.zeros_like(slack.s_lb))
        assert jnp.allclose(hvp.s_ub, jnp.zeros_like(slack.s_ub))
    else:
        assert jnp.allclose(hvp.s, weight * tangent.s / slack.s**2)
        assert jnp.allclose(
            hvp.s_lb,
            jnp.where(
                barrier.null_lb,
                0.0,
                weight * tangent.s_lb / slack.s_lb**2,
            ),
        )
        assert jnp.allclose(
            hvp.s_ub,
            jnp.where(
                barrier.null_ub,
                0.0,
                weight * tangent.s_ub / slack.s_ub**2,
            ),
        )


def test_log_barrier_init_updates_weight():
    """:meth:`Barrier.init` replaces the barrier weight."""
    barrier = LogBarrier(
        weight=jnp.asarray(1.0),
        null_lb=jnp.array([False, True]),
        null_ub=jnp.array([True, False]),
    )
    updated = barrier.init(weight=jnp.asarray(0.1))
    assert jnp.allclose(updated.weight, 0.1)
    assert jnp.allclose(barrier.weight, 1.0)
