"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.reduced_hessian`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.linalg import (
    bfgs_factor_update,
    drop_oldest_basis_vector,
    orthogonalize_against,
    positive_qr,
)

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


# (seed, shape, mode, even_column_sign): every (shape, mode) pair once, with the
# sign and the seed arranged so that every pair of axis values appears.
POSITIVE_QR_CASES = [
    (0, (6, 3), "reduced", 1.0),
    (1, (6, 3), "complete", -1.0),
    (0, (4, 4), "reduced", -1.0),
    (1, (4, 4), "complete", 1.0),
    (1, (3, 5), "reduced", 1.0),
    (0, (3, 5), "complete", -1.0),
    (1, (15, 5), "reduced", -1.0),
    (0, (15, 5), "complete", 1.0),
]


@pytest.mark.parametrize("seed, shape, mode, even_column_sign", POSITIVE_QR_CASES)
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


# name -> (builder(in_span, generic) -> v, upper bound on rho,
#           bound on ||basis^T residual|| / rho)
# The residual of a vector inside the span is pure roundoff, so its direction
# is meaningless and carries no relative-orthogonality bound (``NO_BOUND``;
# finite so that ``NO_BOUND * 0`` is ``0`` and not ``nan``).
NO_BOUND = np.finfo(float).max
V_KINDS = {
    "generic": (lambda in_span, generic: generic, np.inf, 1e-12),
    "in_span": (lambda in_span, generic: in_span, 1e-12, NO_BOUND),
    "nearly_dependent": (
        lambda in_span, generic: in_span + 1e-10 * generic,
        1e-9,
        1e-12,
    ),
}


# (seed, n, m, rank_fraction, kind): a pairwise covering design (every pair of
# values of any two axes appears). rank_fraction: 0 = empty basis, 1 = full.
ORTHOGONALIZE_CASES = [
    (0, 6, 3, 0.5, "nearly_dependent"),
    (0, 6, 3, 0.5, "generic"),
    (1, 6, 1, 0.0, "in_span"),
    (2, 15, 5, 0.5, "in_span"),
    (2, 15, 1, 1.0, "generic"),
    (2, 15, 3, 0.0, "nearly_dependent"),
    (1, 6, 5, 1.0, "nearly_dependent"),
    (0, 15, 3, 1.0, "in_span"),
    (0, 6, 5, 0.0, "generic"),
    (1, 15, 3, 0.5, "generic"),
    (0, 6, 1, 0.5, "nearly_dependent"),
    (2, 6, 1, 0.0, "generic"),
]


@pytest.mark.parametrize("seed, n, m, rank_fraction, kind", ORTHOGONALIZE_CASES)
def test_orthogonalize_against(seed, n, m, rank_fraction, kind):
    """Coefficients/residual reconstruct `v`; the residual is orthogonal to the basis."""
    rank = int(round(rank_fraction * m))
    basis = _padded_orthonormal_basis(seed, n, m, rank)
    rng = np.random.default_rng(100 + seed)
    generic = rng.standard_normal(n)
    in_span = basis @ rng.standard_normal(m)
    build, rho_max, rel_orth = V_KINDS[kind]
    v = build(in_span, generic)

    coeffs, residual, rho = orthogonalize_against(jnp.asarray(basis), jnp.asarray(v))

    assert coeffs.shape == (m,) and residual.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(coeffs))) and bool(jnp.all(jnp.isfinite(residual)))
    np.testing.assert_allclose(basis @ coeffs + residual, v, atol=1e-13)
    np.testing.assert_allclose(basis.T @ residual, 0.0, atol=1e-12)
    # The normalised residual (what the LRH update appends) is orthogonal too.
    assert np.linalg.norm(basis.T @ residual) <= rel_orth * float(rho)
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


def _upper_triangular_factor(seed: int, k: int) -> np.ndarray:
    """Well-conditioned upper-triangular matrix with a positive diagonal."""
    tri = np.triu(_random_matrix(seed, (k, k)), 1) * 0.5
    return tri + np.diag(1.0 + np.random.default_rng(seed + 50).random(k))


def _curvature_pair(seed: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduced step/curvature vectors with ``y^T s > 0`` (sign-fixed, branch-free)."""
    rng = np.random.default_rng(seed + 100)
    s, y = rng.standard_normal(k), rng.standard_normal(k)
    return s, y * np.sign(y @ s)


# (seed, k, n_padded): pairwise covering design.
BFGS_CASES = [
    (0, 1, 0),
    (0, 3, 0),
    (0, 5, 2),
    (1, 5, 0),
    (1, 1, 2),
    (2, 8, 0),
    (2, 3, 2),
    (0, 8, 2),
    (2, 1, 0),
    (1, 3, 0),
    (2, 5, 0),
    (1, 8, 0),
]


@pytest.mark.parametrize("seed, k, n_padded", BFGS_CASES)
def test_bfgs_factor_update(seed, k, n_padded):
    """`R_new^T R_new` is the dense BFGS update; padded identity slots are untouched."""
    r_live = _upper_triangular_factor(seed, k)
    s_live, y_live = _curvature_pair(seed, k)
    size = k + n_padded
    r = np.eye(size)
    r[:k, :k] = r_live
    s = np.zeros(size)
    s[:k] = s_live
    y = np.zeros(size)
    y[:k] = y_live

    r_new = np.asarray(
        bfgs_factor_update(jnp.asarray(r), jnp.asarray(s), jnp.asarray(y))
    )

    b = r.T @ r  # dense reference (tiny)
    bs = b @ s
    b_expected = b - np.outer(bs, bs) / (s @ bs) + np.outer(y, y) / (y @ s)
    assert r_new.shape == (size, size)
    np.testing.assert_allclose(r_new.T @ r_new, b_expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.tril(r_new, -1), 0.0, atol=1e-13)
    assert np.all(np.diagonal(r_new) >= 0.0)
    # Secant equation of the updated model: B_new s = y.
    np.testing.assert_allclose(r_new.T @ (r_new @ s), y, rtol=1e-10, atol=1e-10)
    # Padded slots stay the identity.
    np.testing.assert_allclose(r_new[k:, k:], np.eye(n_padded), atol=1e-13)
    np.testing.assert_allclose(r_new[:k, k:], 0.0, atol=1e-13)


def test_bfgs_factor_update_jit_matches_eager():
    r = jnp.asarray(_upper_triangular_factor(0, 4))
    s, y = (jnp.asarray(v) for v in _curvature_pair(0, 4))
    np.testing.assert_allclose(
        jax.jit(bfgs_factor_update)(r, s, y), bfgs_factor_update(r, s, y), atol=1e-13
    )


@pytest.fixture
def expanded_state():
    """Builder of a full expanded LRH state with `m + 1` slots.

    Returns ``make(n, m, seed=0) -> (steps, basis, tri, chol, sigma)`` where
    ``basis @ tri == steps`` (the last ``m + 1`` steps, oldest first),
    ``basis`` is orthonormal and ``chol`` is an upper-triangular factor.
    """

    def make(n, m, seed=0):
        steps = _random_matrix(seed, (n, m + 1))
        basis, tri = positive_qr(jnp.asarray(steps))
        chol = _upper_triangular_factor(seed, m + 1)
        sigma = 0.5 + np.random.default_rng(seed).random()
        return steps, np.asarray(basis), np.asarray(tri), chol, sigma

    return make


# (seed, n, m): pairwise covering design.
DROP_CASES = [
    (0, 6, 1),
    (1, 15, 1),
    (1, 6, 3),
    (0, 15, 3),
    (2, 6, 5),
    (0, 15, 5),
    (2, 15, 1),
    (2, 6, 3),
    (1, 6, 5),
]


@pytest.mark.parametrize("seed, n, m", DROP_CASES)
def test_drop_oldest_basis_vector(expanded_state, seed, n, m):
    """Dropping the oldest step keeps the newest `m` steps and the projected model."""
    steps, basis, tri, chol, sigma = expanded_state(n, m, seed)

    z_new, t_new, r_new = (
        np.asarray(x)
        for x in drop_oldest_basis_vector(
            jnp.asarray(basis), jnp.asarray(tri), jnp.asarray(chol)
        )
    )

    assert z_new.shape == (n, m) and t_new.shape == (m, m) and r_new.shape == (m, m)
    np.testing.assert_allclose(z_new.T @ z_new, np.eye(m), atol=1e-12)
    np.testing.assert_allclose(z_new @ t_new, steps[:, 1:], atol=1e-12)
    np.testing.assert_allclose(np.tril(t_new, -1), 0.0, atol=1e-13)
    np.testing.assert_allclose(np.tril(r_new, -1), 0.0, atol=1e-13)
    assert np.all(np.diagonal(t_new) >= 0.0) and np.all(np.diagonal(r_new) >= 0.0)
    # Dense reference (tiny n): B'' restricted to the new basis equals R^T R.
    b_full = basis @ chol.T @ chol @ basis.T + sigma * (np.eye(n) - basis @ basis.T)
    np.testing.assert_allclose(
        r_new.T @ r_new, z_new.T @ b_full @ z_new, rtol=1e-10, atol=1e-10
    )


def test_drop_oldest_basis_vector_jit_matches_eager(expanded_state):
    _, basis, tri, chol, _ = expanded_state(9, 3)
    args = tuple(jnp.asarray(x) for x in (basis, tri, chol))
    for a, b in zip(
        drop_oldest_basis_vector(*args),
        jax.jit(drop_oldest_basis_vector)(*args),
        strict=True,
    ):
        np.testing.assert_allclose(b, a, atol=1e-13)


def test_drop_oldest_basis_vector_forms_no_square_n_array():
    """Jaxpr audit: only `n × k` and `n`-sized intermediates (memory budget)."""
    n, m = 40, 3
    basis = jnp.zeros((n, m + 1))
    tri = jnp.eye(m + 1)
    chol = jnp.eye(m + 1)
    jaxpr = jax.make_jaxpr(drop_oldest_basis_vector)(basis, tri, chol)
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            shape = getattr(var.aval, "shape", ())
            assert shape.count(n) <= 1, (eqn.primitive, shape)
