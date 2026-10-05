"""Fixtures shared by :mod:`slsqp_jax.sqpdax.linalg` tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array


def random_ball_point(seed: int, n: int, radius: float, *, fill: float) -> Array:
    """Point with ``‖w‖ = fill · radius`` in a deterministic random direction."""
    direction = jax.random.normal(jax.random.key(seed), (n,))
    return fill * radius * direction / jnp.linalg.norm(direction)


def random_direction(seed: int, n: int) -> Array:
    """Deterministic random direction of unit norm."""
    p = jax.random.normal(jax.random.key(seed), (n,))
    return p / jnp.linalg.norm(p)


@pytest.fixture(params=[2, 5], ids=["n2", "n5"])
def dim(request) -> int:
    """Vector dimension the trust-region tests run at."""
    return request.param
