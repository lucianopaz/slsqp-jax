"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.reduced_hessian`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.linalg import orthogonalize_against, positive_qr

jax.config.update("jax_enable_x64", True)


def _random_matrix(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Deterministic Gaussian matrix (numpy reference data)."""
    return np.random.default_rng(seed).standard_normal(shape)


def _padded_orthonormal_basis(seed: int, n: int, m: int, rank: int) -> np.ndarray:
    """``n × m`` basis whose first ``rank`` columns are orthonormal, rest zero."""
    q, _ = np.linalg.qr(_random_matrix(seed, (n, m)))
    basis = np.zeros((n, m))
    basis[:, :rank] = q[:, :rank]
    return basis


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shape", [(6, 3), (4, 4), (3, 5), (15, 5)])
@pytest.mark.parametrize("mode", ["reduced", "complete"])
@pytest.mark.parametrize("even_column_sign", [1.0, -1.0], ids=["as-is", "flip-even"])
def test_positive_qr(seed, shape, mode, even_column_sign):
    """`positive_qr` reconstructs `a`, has orthonormal `q`, triangular `r`, `diag(r) >= 0`."""
    p, q_cols = shape
    k = min(shape)
    # Flipping columns makes the raw Householder diagonal negative in places.
    column_signs = np.where(np.arange(q_cols) % 2 == 0, even_column_sign, 1.0)
    a = _random_matrix(seed, shape) * column_signs
    q, r = positive_qr(jnp.asarray(a), mode=mode)

    expected_q_cols = k if mode == "reduced" else p
    expected_r_rows = k if mode == "reduced" else p
    assert q.shape == (p, expected_q_cols)
    assert r.shape == (expected_r_rows, q_cols)
    np.testing.assert_allclose(q @ r, a, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(expected_q_cols), atol=1e-12)
    np.testing.assert_allclose(np.tril(r, -1), 0.0, atol=1e-14)
    assert np.all(np.diagonal(r) >= 0.0)

    # Uniqueness: with a positive diagonal the factor equals the sign-fixed numpy QR.
    q_ref, r_ref = np.linalg.qr(a, mode=mode)
    sign = np.where(np.diagonal(r_ref) < 0, -1.0, 1.0)
    np.testing.assert_allclose(r[:k], sign[:, None] * r_ref[:k], atol=1e-12)


def test_positive_qr_rank_deficient_has_nonnegative_diagonal():
    """A zero column gives a zero diagonal entry that is kept non-negative."""
    a = jnp.asarray(_random_matrix(3, (5, 3))).at[:, 1].set(0.0)
    q, r = positive_qr(a)
    assert np.all(np.diagonal(r) >= 0.0)
    assert bool(jnp.all(jnp.isfinite(q))) and bool(jnp.all(jnp.isfinite(r)))
    np.testing.assert_allclose(q @ r, a, atol=1e-12)


def test_positive_qr_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        positive_qr(jnp.eye(3), mode="raw")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["reduced", "complete"])
def test_positive_qr_jit_matches_eager(mode):
    a = jnp.asarray(_random_matrix(4, (7, 4)))
    q_e, r_e = positive_qr(a, mode=mode)
    q_j, r_j = jax.jit(positive_qr, static_argnames="mode")(a, mode=mode)
    np.testing.assert_allclose(q_j, q_e, atol=1e-13)
    np.testing.assert_allclose(r_j, r_e, atol=1e-13)


# name -> (builder(in_span, generic) -> v, upper bound on rho)
V_KINDS = {
    "generic": (lambda in_span, generic: generic, np.inf),
    "in_span": (lambda in_span, generic: in_span, 1e-12),
    "nearly_dependent": (lambda in_span, generic: in_span + 1e-10 * generic, 1e-9),
}


# (rank as a fraction of m: 0 = empty basis, 1 = full)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [6, 15])
@pytest.mark.parametrize("m", [1, 3, 5])
@pytest.mark.parametrize("rank_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", V_KINDS.keys())
def test_orthogonalize_against(seed, n, m, rank_fraction, kind):
    """Coefficients/residual reconstruct `v`; the residual is orthogonal to the basis."""
    rank = int(round(rank_fraction * m))
    basis = _padded_orthonormal_basis(seed, n, m, rank)
    rng = np.random.default_rng(100 + seed)
    generic = rng.standard_normal(n)
    in_span = basis @ rng.standard_normal(m)
    build, rho_max = V_KINDS[kind]
    v = build(in_span, generic)

    coeffs, residual, rho = orthogonalize_against(jnp.asarray(basis), jnp.asarray(v))

    assert coeffs.shape == (m,) and residual.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(coeffs))) and bool(jnp.all(jnp.isfinite(residual)))
    np.testing.assert_allclose(basis @ coeffs + residual, v, atol=1e-13)
    np.testing.assert_allclose(basis.T @ residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(rho, np.linalg.norm(residual), rtol=1e-12, atol=1e-15)
    # Padded columns carry a zero coefficient.
    np.testing.assert_array_equal(coeffs[rank:], 0.0)
    # The coefficients are the expansion of the projection onto range(basis).
    np.testing.assert_allclose(coeffs[:rank], basis[:, :rank].T @ v, atol=1e-12)
    assert float(rho) <= rho_max


def test_orthogonalize_against_second_pass_is_needed():
    """CGS2 stays orthogonal to roundoff where a single pass would not."""
    n = 12
    # Single-precision-like loss of orthogonality: a basis that is only
    # orthonormal to ~1e-8, and a vector almost inside its span.
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((n, 3)))
    basis = q + 1e-8 * rng.standard_normal((n, 3))
    v = basis @ rng.standard_normal(3) + 1e-9 * rng.standard_normal(n)
    _, residual, _ = orthogonalize_against(jnp.asarray(basis), jnp.asarray(v))
    single_pass = v - basis @ (basis.T @ v)
    assert np.linalg.norm(basis.T @ residual) < np.linalg.norm(basis.T @ single_pass)
    assert np.linalg.norm(basis.T @ residual) < 1e-15


def test_orthogonalize_against_jit_matches_eager():
    basis = jnp.asarray(_padded_orthonormal_basis(0, 9, 4, 3))
    v = jnp.asarray(_random_matrix(5, (9, 1))[:, 0])
    eager = orthogonalize_against(basis, v)
    jitted = jax.jit(orthogonalize_against)(basis, v)
    for a, b in zip(eager, jitted, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-14)


def test_orthogonalize_against_forms_no_square_n_array():
    """Jaxpr audit: only `n × k` and `n`-sized intermediates (memory budget)."""
    n, k = 40, 3
    basis = jnp.zeros((n, k))
    v = jnp.zeros(n)
    jaxpr = jax.make_jaxpr(orthogonalize_against)(basis, v)
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            shape = getattr(var.aval, "shape", ())
            assert shape.count(n) <= 1, (eqn.primitive, shape)
