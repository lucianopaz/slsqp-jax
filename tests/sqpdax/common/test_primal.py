"""Unit tests for :mod:`slsqp_jax.sqpdax.common.primal`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.common.primal import InteriorPointPrimal, Primal, Slack


def _arange(length: int, start: float = 0.0) -> Array:
    return start + jnp.arange(length, dtype=jnp.float32)


@pytest.mark.parametrize(
    ("n",),
    [(1,), (3,), (5,)],
)
def test_primal_sizes_and_roundtrip(n: int):
    """``flatten`` / ``from_flat`` round-trip and ``sizes`` stay consistent."""
    x = _arange(n, start=1.0)
    primal = Primal(x=x)

    assert primal.n == n
    assert primal.sizes == (n,)
    assert jnp.allclose(primal.flatten(), x)

    rebuilt = Primal.from_flat(primal.flatten(), *primal.sizes)
    assert rebuilt.n == n
    assert jnp.allclose(rebuilt.x, x)


@pytest.mark.parametrize(
    "sizes",
    [(), (2, 3), (1, 2, 3)],
)
def test_primal_from_flat_rejects_wrong_arity(sizes: tuple[int, ...]):
    """``from_flat`` requires exactly one size argument."""
    with pytest.raises(ValueError, match="expects 1 size argument"):
        Primal.from_flat(_arange(4), *sizes)


def test_primal_from_flat_uses_leading_slice():
    """``from_flat`` keeps only the leading ``n`` entries of ``arr``."""
    arr = _arange(5, start=10.0)
    primal = Primal.from_flat(arr, 3)
    assert jnp.allclose(primal.x, arr[:3])


@pytest.mark.parametrize(
    ("n", "mineq"),
    [(1, 0), (2, 1), (4, 3)],
)
def test_slack_sizes_and_roundtrip(n: int, mineq: int):
    """``flatten`` / ``from_flat`` round-trip for :class:`Slack`."""
    s = _arange(mineq, start=1.0)
    s_lb = _arange(n, start=10.0)
    s_ub = _arange(n, start=20.0)
    slack = Slack(s=s, s_lb=s_lb, s_ub=s_ub)

    assert slack.n == n
    assert slack.mineq == mineq
    assert slack.sizes == (n, mineq)

    flat = slack.flatten()
    assert flat.shape == (mineq + 2 * n,)
    assert jnp.allclose(flat, jnp.concatenate([s, s_lb, s_ub]))

    rebuilt = Slack.from_flat(flat, *slack.sizes)
    assert rebuilt.sizes == (n, mineq)
    assert jnp.allclose(rebuilt.s, s)
    assert jnp.allclose(rebuilt.s_lb, s_lb)
    assert jnp.allclose(rebuilt.s_ub, s_ub)


@pytest.mark.parametrize(
    "sizes",
    [(), (2,), (1, 2, 3)],
)
def test_slack_from_flat_rejects_wrong_arity(sizes: tuple[int, ...]):
    """``from_flat`` requires exactly ``(n, mineq)``."""
    with pytest.raises(ValueError, match="expects 2 sizes arguments: n, mineq"):
        Slack.from_flat(_arange(8), *sizes)


@pytest.mark.parametrize(
    ("n", "mineq"),
    [(1, 0), (2, 2), (3, 1)],
)
def test_interior_point_primal_sizes_and_roundtrip(n: int, mineq: int):
    """``flatten`` / ``from_flat`` round-trip for :class:`InteriorPointPrimal`."""
    x = _arange(n, start=1.0)
    slack = Slack(
        s=_arange(mineq, start=10.0),
        s_lb=_arange(n, start=20.0),
        s_ub=_arange(n, start=30.0),
    )
    ip = InteriorPointPrimal(x=x, slack=slack)

    assert ip.n == n
    assert ip.mineq == mineq
    assert ip.sizes == (n, mineq)

    flat = ip.flatten()
    assert flat.shape == (3 * n + mineq,)
    assert jnp.allclose(flat, jnp.concatenate([x, slack.flatten()]))

    rebuilt = InteriorPointPrimal.from_flat(flat, *ip.sizes)
    assert isinstance(rebuilt, InteriorPointPrimal)
    assert rebuilt.sizes == (n, mineq)
    assert jnp.allclose(rebuilt.x, x)
    assert jnp.allclose(rebuilt.slack.s, slack.s)
    assert jnp.allclose(rebuilt.slack.s_lb, slack.s_lb)
    assert jnp.allclose(rebuilt.slack.s_ub, slack.s_ub)


@pytest.mark.parametrize(
    "sizes",
    [(), (2,), (1, 2, 3)],
)
def test_interior_point_primal_from_flat_rejects_wrong_arity(sizes: tuple[int, ...]):
    """``from_flat`` requires exactly ``(n, mineq)``."""
    with pytest.raises(
        ValueError,
        match="InteriorPointPrimal.from_flat expects 2 sizes arguments: n, mineq",
    ):
        InteriorPointPrimal.from_flat(_arange(10), *sizes)
