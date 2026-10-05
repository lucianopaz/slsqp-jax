"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.projection`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.linalg import cg_normal_equations, null_space_projector


def random_matrix(seed: int, m: int, n: int, *, zero_rows: tuple[int, ...] = ()):
    M = jax.random.normal(jax.random.key(seed), (m, n))
    for i in zero_rows:
        M = M.at[i].set(0.0)
    return M


MATRICES = {
    "full-row-rank": dict(m=3, n=7, zero_rows=()),
    "zero-row": dict(m=4, n=7, zero_rows=(1,)),
    "square-rank-deficient": dict(m=5, n=5, zero_rows=(0, 3)),
}


@pytest.mark.parametrize("spec", MATRICES.values(), ids=MATRICES.keys())
@pytest.mark.parametrize("seed", [0, 1])
def test_projector_matches_dense_pseudo_inverse(spec, seed):
    """``proj(v) = (I − Aᵀ(AAᵀ)⁺A) v`` including consistent singular ``AAᵀ``."""
    M = random_matrix(seed, spec["m"], spec["n"], zero_rows=spec["zero_rows"])
    proj = null_space_projector(lambda v: M @ v, lambda y: M.T @ y)
    dense = jnp.eye(spec["n"]) - M.T @ jnp.linalg.pinv(M @ M.T, rtol=1e-5) @ M
    v = jax.random.normal(jax.random.key(seed + 7), (spec["n"],))
    out = proj(v)
    assert jnp.allclose(out, dense @ v, atol=1e-4)
    assert jnp.allclose(M @ out, 0.0, atol=1e-4)
    # Idempotent.
    assert jnp.allclose(proj(out), out, atol=1e-4)


def test_projector_respects_free_mask():
    """Frozen columns are zeroed and excluded from the constraint operator."""
    M = random_matrix(3, 2, 5)
    free = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0])
    proj = null_space_projector(lambda v: M @ v, lambda y: M.T @ y, free_mask=free)
    v = jax.random.normal(jax.random.key(9), (5,))
    out = proj(v)
    assert jnp.all(out[free == 0.0] == 0.0)
    assert jnp.allclose(M @ out, 0.0, atol=1e-4)
    # Same as projecting with the restricted matrix.
    M_free = M[:, free > 0.0]
    dense = jnp.eye(3) - M_free.T @ jnp.linalg.pinv(M_free @ M_free.T) @ M_free
    assert jnp.allclose(out[free > 0.0], dense @ v[free > 0.0], atol=1e-4)


def test_regularisation_shrinks_toward_identity():
    """A large Tikhonov term makes the projector approach the identity."""
    M = random_matrix(1, 2, 4)
    v = jnp.arange(1.0, 5.0)
    out = null_space_projector(lambda v: M @ v, lambda y: M.T @ y, reg=1e6)(v)
    assert jnp.allclose(out, v, rtol=1e-3)


@pytest.mark.parametrize("seed", [0, 1])
def test_cg_normal_equations_solves_spd_system(seed):
    B = random_matrix(seed, 6, 4)
    S = B.T @ B + jnp.eye(4)
    rhs = jax.random.normal(jax.random.key(seed + 1), (4,))
    y = cg_normal_equations(lambda z: S @ z, rhs, tol=1e-8, max_iter=20)
    assert jnp.allclose(S @ y, rhs, atol=1e-4)


def test_cg_normal_equations_zero_rhs_is_zero():
    y = cg_normal_equations(lambda z: 2.0 * z, jnp.zeros(3), tol=1e-8, max_iter=5)
    assert jnp.array_equal(y, jnp.zeros(3))
