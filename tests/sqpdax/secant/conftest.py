"""Fixtures for :mod:`slsqp_jax.sqpdax.secant` tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.secant import LBFGS

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def n() -> int:
    return 5


@pytest.fixture
def memory() -> int:
    return 4


@pytest.fixture
def empty_lbfgs(n: int, memory: int) -> LBFGS:
    """Fresh L-BFGS history with identity ``B₀``."""
    return LBFGS(n=n, memory=memory)


def build_lbfgs_with_pairs(
    n: int,
    pairs: list[tuple],
    *,
    memory: int = 10,
    **kwargs,
) -> LBFGS:
    """Append ``(s, y)`` pairs sequentially to a new :class:`LBFGS`."""
    hist = LBFGS(n=n, memory=memory, **kwargs)
    for s, y in pairs:
        hist = hist.append(jnp.asarray(s), jnp.asarray(y))
    return hist


def explicit_diagonal(hist: LBFGS) -> jnp.ndarray:
    """Reference ``diag(B)`` by probing unit vectors."""
    dim = hist.diagonal.shape[0]
    diag = jnp.zeros(dim)
    for i in range(dim):
        e_i = jnp.zeros(dim).at[i].set(1.0)
        diag = diag.at[i].set(jnp.dot(hist.hvp(e_i), e_i))
    return diag
