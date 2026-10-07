"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.operator_norm`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.linalg import power_iteration_norm, spectral_norm_estimate


def symmetric_matrix(seed: int, n: int, *, zero_rows: tuple[int, ...] = ()):
    M = jax.random.normal(jax.random.key(seed), (n, n))
    M = M + M.T
    for i in zero_rows:
        M = M.at[i].set(0.0).at[:, i].set(0.0)
    return M


SYMMETRIC = {
    "dense": dict(n=5, zero_rows=()),
    "dead-coordinates": dict(n=6, zero_rows=(1, 4)),
    "indefinite-diagonal": None,
}


@pytest.mark.parametrize("name", list(SYMMETRIC), ids=list(SYMMETRIC))
@pytest.mark.parametrize("seed", [0, 1])
def test_power_iteration_matches_dominant_eigenvalue_and_bounds_from_below(name, seed):
    """Converged estimate equals ``max |λ|``; every iterate is a lower bound."""
    spec = SYMMETRIC[name]
    if spec is None:
        M = jnp.diag(jnp.array([1.0, -4.0, 2.5, 0.0]) * (seed + 1))
    else:
        M = symmetric_matrix(seed, spec["n"], zero_rows=spec["zero_rows"])
    exact = jnp.max(jnp.abs(jnp.linalg.eigvalsh(M)))
    x0 = jnp.ones(M.shape[0])
    rough = power_iteration_norm(lambda v: M @ v, x0, n_iter=2)
    tight = power_iteration_norm(lambda v: M @ v, x0, n_iter=200)
    assert float(rough) <= float(exact) * (1 + 1e-5)
    assert float(tight) <= float(exact) * (1 + 1e-5)
    assert float(tight) == pytest.approx(float(exact), rel=1e-3)


def test_zero_operator_and_jit():
    """An identically zero operator gives ``0``; the kernel is jittable."""
    x0 = jnp.ones(4)
    assert float(power_iteration_norm(lambda v: 0.0 * v, x0, n_iter=5)) == 0.0
    M = jnp.diag(jnp.array([3.0, 1.0, 1.0, 1.0]))
    est = jax.jit(lambda m: power_iteration_norm(lambda v: m @ v, x0, n_iter=50))(M)
    assert float(est) == pytest.approx(3.0, rel=1e-4)


@pytest.mark.parametrize(
    ("m", "n", "zero_rows"),
    [(3, 7, ()), (6, 4, (2,)), (5, 5, (0, 3))],
    ids=["wide", "tall-zero-row", "square-rank-deficient"],
)
def test_spectral_norm_estimate_matches_largest_singular_value(m, n, zero_rows):
    """``‖A‖₂`` from power iteration on ``AᵀA`` agrees with the SVD."""
    A = jax.random.normal(jax.random.key(3), (m, n))
    for i in zero_rows:
        A = A.at[i].set(0.0)
    exact = jnp.linalg.svd(A, compute_uv=False)[0]
    est = spectral_norm_estimate(
        lambda v: A @ v, lambda y: A.T @ y, jnp.ones(n), n_iter=300
    )
    assert float(est) <= float(exact) * (1 + 1e-5)
    assert float(est) == pytest.approx(float(exact), rel=1e-3)
