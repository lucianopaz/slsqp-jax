"""Unit tests for :mod:`slsqp_jax.sqpdax.dual`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual


def _arange(length: int, start: float = 0.0) -> Array:
    return start + jnp.arange(length, dtype=jnp.float32)


@pytest.mark.parametrize(
    ("n", "mineq", "meq"),
    [
        (1, 0, 0),
        (2, 1, 0),
        (3, 0, 2),
        (4, 2, 1),
    ],
)
def test_dual_sizes_and_roundtrip(n: int, mineq: int, meq: int):
    """``flatten`` / ``from_flat`` round-trip and ``sizes`` stay consistent."""
    eq = _arange(meq, start=1.0)
    ineq = _arange(mineq, start=10.0)
    lb = _arange(n, start=20.0)
    ub = _arange(n, start=30.0)
    dual = Dual(
        eq_multipliers=eq,
        ineq_multipliers=ineq,
        lb_multipliers=lb,
        ub_multipliers=ub,
    )

    assert dual.n == n
    assert dual.mineq == mineq
    assert dual.meq == meq
    assert dual.sizes == (n, mineq, meq)

    flat = dual.flatten()
    assert flat.shape == (meq + mineq + 2 * n,)
    assert jnp.allclose(flat, jnp.concatenate([eq, ineq, lb, ub]))

    rebuilt = Dual.from_flat(flat, *dual.sizes)
    assert rebuilt.sizes == (n, mineq, meq)
    assert jnp.allclose(rebuilt.eq_multipliers, eq)
    assert jnp.allclose(rebuilt.ineq_multipliers, ineq)
    assert jnp.allclose(rebuilt.lb_multipliers, lb)
    assert jnp.allclose(rebuilt.ub_multipliers, ub)


@pytest.mark.parametrize(
    "sizes",
    [(), (2,), (1, 2), (1, 2, 3, 4)],
)
def test_dual_from_flat_rejects_wrong_arity(sizes: tuple[int, ...]):
    """``from_flat`` requires exactly ``(n, mineq, meq)``."""
    with pytest.raises(ValueError, match="expects 3 sizes arguments: n, mineq, meq"):
        Dual.from_flat(_arange(12), *sizes)


def test_dual_from_flat_uses_leading_slices():
    """``from_flat`` keeps only the slices implied by ``sizes``."""
    n, mineq, meq = 2, 1, 3
    needed = meq + mineq + 2 * n
    arr = _arange(needed + 5, start=100.0)
    dual = Dual.from_flat(arr, n, mineq, meq)

    assert dual.sizes == (n, mineq, meq)
    assert jnp.allclose(dual.eq_multipliers, arr[:meq])
    assert jnp.allclose(dual.ineq_multipliers, arr[meq : meq + mineq])
    assert jnp.allclose(dual.lb_multipliers, arr[meq + mineq : meq + mineq + n])
    assert jnp.allclose(dual.ub_multipliers, arr[meq + mineq + n : meq + mineq + 2 * n])
