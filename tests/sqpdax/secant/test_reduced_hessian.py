"""Unit tests for :mod:`slsqp_jax.sqpdax.secant.reduced_hessian`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import (
    LBFGS,
    ReducedHessianFactors,
    ReducedHessianSecant,
    Secant,
)

jax.config.update("jax_enable_x64", True)

N = 9
M = 4


def make_factors(rank: int, seed: int, garbage: float = 0.0):
    """Random factors with zero-padded ``Z``; padded ``R`` block set to ``garbage``.

    Returns the factors and the dense ``(B, B⁻¹)`` references (tiny ``n`` only).
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(N, M)))
    z = np.zeros((N, M))
    z[:, :rank] = q[:, :rank]
    r_act = np.triu(rng.normal(size=(rank, rank)), 1) + np.diag(
        rng.uniform(0.5, 2.0, size=rank)
    )
    chol = np.full((M, M), garbage)
    chol[:rank, :rank] = r_act
    chol = np.triu(chol)
    sigma = float(rng.uniform(0.3, 3.0))
    dense_b = (
        sigma * np.eye(N)
        + z[:, :rank] @ (r_act.T @ r_act - sigma * np.eye(rank)) @ z[:, :rank].T
    )
    factors = ReducedHessianFactors(
        basis=jnp.asarray(z),
        cholesky=jnp.asarray(chol),
        sigma=jnp.asarray(sigma),
        rank=jnp.asarray(rank, dtype=jnp.int32),
    )
    return factors, dense_b, np.linalg.inv(dense_b)


@pytest.fixture
def probe() -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(123).normal(size=N))


@pytest.mark.parametrize("rank", range(M + 1))
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("garbage", [0.0, 7.5])
def test_products_match_dense_reference(rank, seed, garbage, probe):
    """``hvp``/``inverse_hvp`` equal the dense formulas; padding is ignored."""
    factors, dense_b, dense_inv = make_factors(rank, seed, garbage)
    np.testing.assert_allclose(factors.hvp(probe), dense_b @ probe, rtol=1e-10)
    np.testing.assert_allclose(
        factors.inverse_hvp(probe), dense_inv @ probe, rtol=1e-10
    )
    np.testing.assert_allclose(
        factors.inverse_hvp(factors.hvp(probe)), probe, rtol=1e-10
    )


@pytest.mark.parametrize("rank", range(M + 1))
def test_column_mask_and_jit(rank, probe):
    factors, dense_b, _ = make_factors(rank, 3)
    mask = np.asarray(factors.column_mask)
    np.testing.assert_array_equal(mask, np.arange(M) < rank)
    assert factors.rank.dtype == jnp.int32
    jitted = jax.jit(lambda f, v: (f.hvp(v), f.inverse_hvp(v)))
    hv, ihv = jitted(factors, probe)
    np.testing.assert_allclose(hv, factors.hvp(probe), rtol=1e-12)
    np.testing.assert_allclose(ihv, factors.inverse_hvp(probe), rtol=1e-12)


@pytest.mark.parametrize("rank", range(M + 1))
def test_active_cholesky_identity_padding(rank):
    factors, _, _ = make_factors(rank, 5, garbage=7.5)
    got = np.asarray(factors.active_cholesky)
    stored = np.asarray(factors.cholesky)
    np.testing.assert_array_equal(got[:rank, :rank], stored[:rank, :rank])
    expected_pad = np.eye(M)[rank:, rank:]
    np.testing.assert_array_equal(got[rank:, rank:], expected_pad)
    np.testing.assert_array_equal(got[:rank, rank:], 0.0)
    np.testing.assert_array_equal(got[rank:, :rank], 0.0)


def test_abstract_class_not_registered_and_not_instantiable():
    assert "factors" in ReducedHessianSecant.__abstractmethods__
    assert "ReducedHessianSecant" not in {c.__name__ for c in Secant._registry.values()}
    assert ReducedHessianSecant not in Secant._registry.values()
    assert "kind" not in ReducedHessianSecant.__dict__
    assert issubclass(ReducedHessianSecant, Secant)
    assert not issubclass(LBFGS, ReducedHessianSecant)
    with pytest.raises(TypeError):
        ReducedHessianSecant()  # ty: ignore[call-non-callable]
