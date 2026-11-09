"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.free_set_factor`."""

from __future__ import annotations

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.linalg import FreeSetFactor
from slsqp_jax.sqpdax.secant import LimitedMemoryReducedHessian, ReducedHessianFactors

jax.config.update("jax_enable_x64", True)

N = 10
M = 4
ATOL = 1e-10

# Mask kinds: (free mask, rows the first basis column is supported on). The
# "rank deficient" mask pins rows 0..2 and makes basis column 0 live only on
# them, so that Z_F loses a column.
_ROWS = np.arange(N)
_FREE_MASKS = {
    "all_free": np.ones(N, dtype=bool),
    "all_fixed": np.zeros(N, dtype=bool),
    "random_half": np.random.default_rng(5).uniform(size=N) < 0.5,
    "rank_deficient": _ROWS >= 3,
}
_COL0_SUPPORT = {
    "all_free": np.ones(N, dtype=bool),
    "all_fixed": np.ones(N, dtype=bool),
    "random_half": np.ones(N, dtype=bool),
    "rank_deficient": _ROWS < 3,
}
# Source of the factors; the rank is the number of basis columns (random) or
# of appended pairs (lrh, memory M: six pairs force a drop, so rank == M).
_SOURCES = {"random": [0, 3, M], "lrh": [0, 3, 6]}


class Scenario(NamedTuple):
    """Factors, free mask and the dense reference quantities (tiny ``n``)."""

    factors: ReducedHessianFactors
    free: np.ndarray
    dense_b: np.ndarray  # B (n × n reference)
    dense_b_ff: np.ndarray  # I_F B I_F
    z_f: np.ndarray  # I_F Z (active columns)


def _random_factors(rank: int, support: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((N, M))
    g[~support, 0] = 0.0
    q, _ = np.linalg.qr(g)
    z = np.zeros((N, M))
    z[:, :rank] = q[:, :rank]
    chol = np.full((M, M), 7.5)  # garbage in the padded block
    chol[:rank, :rank] = np.triu(rng.standard_normal((rank, rank)), 1) + np.diag(
        rng.uniform(0.5, 2.0, size=rank)
    )
    chol = np.triu(chol)
    return ReducedHessianFactors(
        basis=jnp.asarray(z),
        cholesky=jnp.asarray(chol),
        sigma=jnp.asarray(float(rng.uniform(0.3, 3.0))),
        rank=jnp.asarray(rank, dtype=jnp.int32),
    )


def _lrh_factors(n_pairs: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((N, N))
    hess = a @ a.T / N + np.eye(N)
    hist = LimitedMemoryReducedHessian(n=N, memory=M)
    for _ in range(n_pairs):
        s = jnp.asarray(rng.standard_normal(N))
        hist = hist.append(s, jnp.asarray(hess @ np.asarray(s)))
    return hist.factors()


_BUILDERS = {
    "random": lambda rank, support, seed: _random_factors(rank, support, seed),
    "lrh": lambda rank, support, seed: _lrh_factors(rank, seed),
}


def _dense_reference(factors: ReducedHessianFactors, free: np.ndarray) -> Scenario:
    r = int(factors.rank)
    z = np.asarray(factors.basis)[:, :r]
    chol = np.asarray(factors.cholesky)[:r, :r]
    sigma = float(factors.sigma)
    dense_b = sigma * np.eye(N) + z @ (chol.T @ chol - sigma * np.eye(r)) @ z.T
    d = np.diag(free.astype(float))
    return Scenario(factors, free, dense_b, d @ dense_b @ d, d @ z)


@pytest.fixture(
    params=[
        (source, mask_kind, rank)
        for source, ranks in _SOURCES.items()
        for mask_kind, rank in itertools.product(_FREE_MASKS, ranks)
    ],
    ids=lambda p: f"{p[0]}-{p[1]}-r{p[2]}",
)
def scenario(request) -> Scenario:
    """Every (factor source, mask kind, rank) combination."""
    source, mask_kind, rank = request.param
    factors = _BUILDERS[source](rank, _COL0_SUPPORT[mask_kind], rank)
    return _dense_reference(factors, _FREE_MASKS[mask_kind])


@pytest.fixture
def factor(scenario: Scenario) -> FreeSetFactor:
    return FreeSetFactor.build(scenario.factors, jnp.asarray(scenario.free))


@pytest.fixture
def probe() -> np.ndarray:
    """Dense vector with non-zero entries on every coordinate."""
    return np.random.default_rng(123).standard_normal(N)


_ID = np.eye(N)  # test-only dense reference probe (tiny N)


def _dense(method, x: np.ndarray) -> np.ndarray:
    return np.asarray(method(jnp.asarray(x)))


_APPLIES = [
    "hvp",
    "solve",
    "apply_c",
    "apply_c_t",
    "apply_c_inv",
    "apply_c_inv_t",
]


def test_c_factor_squares_to_free_block(scenario, factor):
    c = _dense(factor.apply_c_matrix, _ID)
    np.testing.assert_allclose(c.T @ c, scenario.dense_b_ff, atol=ATOL)
    np.testing.assert_allclose(
        _dense(factor.hvp_matrix, _ID), scenario.dense_b_ff, atol=ATOL
    )


def test_c_inverse_is_inverse_on_free_set(scenario, factor):
    d = np.diag(scenario.free.astype(float))
    c = _dense(factor.apply_c_matrix, _ID)
    c_inv = _dense(factor.apply_c_inv_matrix, _ID)
    np.testing.assert_allclose(c_inv @ c, d, atol=ATOL)
    np.testing.assert_allclose(c @ c_inv, d, atol=ATOL)
    # The transposed pair is the transpose of the pair.
    np.testing.assert_allclose(_dense(factor.apply_c_t_matrix, _ID), c.T, atol=ATOL)
    np.testing.assert_allclose(
        _dense(factor.apply_c_inv_t_matrix, _ID), c_inv.T, atol=ATOL
    )


def test_solve_inverts_hvp_on_free_set(scenario, factor, probe):
    expected = np.where(scenario.free, probe, 0.0)
    np.testing.assert_allclose(
        _dense(factor.solve, _dense(factor.hvp, probe)), expected, atol=ATOL
    )
    # Dense reference: B_FF⁻¹ is the pseudo-inverse of the masked block.
    np.testing.assert_allclose(
        _dense(factor.solve_matrix, _ID),
        np.linalg.pinv(scenario.dense_b_ff, rcond=1e-12),
        atol=ATOL,
    )


def test_numerical_rank_of_restricted_basis(scenario, factor):
    # A zero column keeps the array non-empty at rank 0.
    padded = np.hstack([scenario.z_f, np.zeros((N, 1))])
    expected = np.linalg.matrix_rank(padded, tol=1e-8)
    assert int(factor.free_rank) == expected


@pytest.mark.parametrize("name", _APPLIES)
def test_outputs_are_finite_and_vanish_on_fixed_coordinates(
    scenario, factor, probe, name
):
    out = _dense(getattr(factor, name), probe)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[~scenario.free], 0.0)
    block = _dense(getattr(factor, f"{name}_matrix"), np.stack([probe, 2 * probe], 1))
    np.testing.assert_allclose(block[:, 0], out, atol=1e-14)
    np.testing.assert_allclose(block[:, 1], 2 * out, atol=1e-13)


@pytest.mark.parametrize(
    "source,rank", [(s, r) for s, rs in _SOURCES.items() for r in rs]
)
def test_all_free_matches_factors(source, rank, probe):
    factors = _BUILDERS[source](rank, np.ones(N, dtype=bool), rank)
    factor = FreeSetFactor.build(factors, jnp.ones(N, dtype=bool))
    np.testing.assert_allclose(
        _dense(factor.hvp, probe), _dense(factors.hvp, probe), atol=ATOL
    )
    np.testing.assert_allclose(
        _dense(factor.solve, probe), _dense(factors.inverse_hvp, probe), atol=ATOL
    )


def test_jit_with_traced_mask_matches_eager_without_retracing(scenario, probe):
    traces = []

    def run(factors, free, v):
        traces.append(1)
        fs = FreeSetFactor.build(factors, free)
        return jnp.stack([getattr(fs, name)(v) for name in _APPLIES])

    jitted = jax.jit(run)
    v = jnp.asarray(probe)
    for free in (scenario.free, ~scenario.free):
        free = jnp.asarray(free)
        np.testing.assert_allclose(
            np.asarray(jitted(scenario.factors, free, v)),
            np.asarray(run(scenario.factors, free, v)),
            atol=1e-12,
        )
    assert len(traces) == 1 + 2  # one jit trace + two eager reference calls


def test_large_rank_tol_drops_every_direction(probe):
    factors = _random_factors(M, np.ones(N, dtype=bool), seed=1)
    free = np.random.default_rng(2).uniform(size=N) < 0.6
    factor = FreeSetFactor.build(factors, jnp.asarray(free), rank_tol=2.0)
    assert int(factor.free_rank) == 0
    expected = float(factors.sigma) * np.where(free, probe, 0.0)
    np.testing.assert_allclose(_dense(factor.hvp, probe), expected, atol=1e-12)


def test_build_and_applies_form_no_square_n_array():
    """Jaxpr audit: nothing has two `n`-sized axes (memory budget)."""
    n, m, k = 40, M, 3
    factors = ReducedHessianFactors(
        basis=jnp.zeros((n, m)),
        cholesky=jnp.eye(m),
        sigma=jnp.asarray(1.0),
        rank=jnp.asarray(0, dtype=jnp.int32),
    )

    def run(factors, free, v, block):
        fs = FreeSetFactor.build(factors, free)
        outs = [getattr(fs, name)(v) for name in _APPLIES]
        outs += [getattr(fs, f"{name}_matrix")(block) for name in _APPLIES]
        return outs

    jaxpr = jax.make_jaxpr(run)(
        factors, jnp.ones(n, dtype=bool), jnp.zeros(n), jnp.zeros((n, k))
    )
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            shape = getattr(var.aval, "shape", ())
            assert shape.count(n) <= 1, (eqn.primitive, shape)
