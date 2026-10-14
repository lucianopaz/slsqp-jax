"""Fixtures for :mod:`slsqp_jax.sqpdax.secant` tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.secant import LBFGS

# Compiled once per static configuration; eager ``append`` retraces its
# ``lax.cond`` on every call and dominates the runtime of history-building tests.
append_jit = eqx.filter_jit(lambda hist, s, y: hist.append(s, y))


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
        hist = append_jit(hist, jnp.asarray(s), jnp.asarray(y))
    return hist


def explicit_diagonal(hist: LBFGS) -> jnp.ndarray:
    """Reference ``diag(B)`` by probing unit vectors."""
    dim = hist.diagonal.shape[0]
    diag = jnp.zeros(dim)
    for i in range(dim):
        e_i = jnp.zeros(dim).at[i].set(1.0)
        diag = diag.at[i].set(jnp.dot(hist.hvp(e_i), e_i))
    return diag


def dense_hessian(hist: LBFGS) -> jnp.ndarray:
    """Dense ``B`` assembled from ``hvp`` applied to the identity columns."""
    return jax.vmap(hist.hvp)(jnp.eye(hist.n)).T


def random_pairs(
    seed: int, n: int, count: int, *, negative_every: int = 3
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Curvature pairs of a random SPD quadratic, every k-th one with ``sᵀy < 0``.

    ``y = H s`` for a fixed random SPD ``H`` plus a little noise; each
    ``negative_every``-th pair has its sign flipped (negative curvature).
    Pass ``negative_every=0`` for positive curvature only.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    hessian = a @ a.T / n + 0.5 * np.eye(n)
    pairs = []
    for k in range(count):
        s = rng.normal(size=n) * 10.0 ** rng.uniform(-1.0, 0.0)
        y = hessian @ s
        if negative_every and k % negative_every == negative_every - 1:
            y = -y
        y = y + 0.01 * rng.normal(size=n) * np.linalg.norm(y)
        pairs.append((jnp.asarray(s), jnp.asarray(y)))
    return pairs


def pair_with_curvature_ratio(
    hist: LBFGS, ratio: float, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pair ``(s, y)`` with ``sᵀy = ratio · sᵀ B s`` for the model ``B = hist``.

    ``y = ratio · B s + w`` with ``w ⊥ s``, so ``ratio`` is exactly the quantity
    Powell damping compares with the threshold.
    """
    rng = np.random.default_rng(seed)
    s = jnp.asarray(rng.normal(size=hist.n))
    Bs = hist.hvp(s)
    w = jnp.asarray(rng.normal(size=hist.n))
    w = w - jnp.dot(w, s) / jnp.dot(s, s) * s
    return s, ratio * Bs + 0.3 * jnp.linalg.norm(Bs) * w / jnp.linalg.norm(w)


# Narrow-valley quadratic (the LOGROS crawl): soft curvature along ``t``, stiff
# curvature along the normal ``n``; ``H = λ_t t tᵀ + λ_n n nᵀ``.
VALLEY_LAMBDA_T = 0.1
VALLEY_LAMBDA_N = 1.5e5


@pytest.fixture(
    params=[(15.0, 0), (45.0, 1), (75.0, 2)], ids=["15deg", "45deg", "75deg"]
)
def narrow_valley(request) -> tuple[jnp.ndarray, list[tuple]]:
    """Tangent ``t`` of a rotated narrow valley and 30 pairs crawling along it.

    Each step is ``s = a t + b n`` with ``a ∈ [0.1, 1]`` and a normal
    component ``|b| ≈ 0.05 |a|`` of random sign; ``y = H s`` exactly.
    """
    angle_deg, seed = request.param
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    t = np.array([np.cos(theta), np.sin(theta)])
    nrm = np.array([-np.sin(theta), np.cos(theta)])
    hessian = VALLEY_LAMBDA_T * np.outer(t, t) + VALLEY_LAMBDA_N * np.outer(nrm, nrm)
    pairs = []
    for _ in range(30):
        a = 10.0 ** rng.uniform(-1.0, 0.0)
        b = 0.05 * a * rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        s = a * t + b * nrm
        pairs.append((jnp.asarray(s), jnp.asarray(hessian @ s)))
    return jnp.asarray(t), pairs
