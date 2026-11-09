"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.lawson_hanson`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls

from slsqp_jax.sqpdax.linalg import NNLSResult, nnls

jax.config.update("jax_enable_x64", True)

ATOL = 1e-10
KKT_TOL = 1e-9

# Shapes (m, k): (k+1) x k for the sizes the plan names, plus other aspect
# ratios (tall, square, wide-ish, a single column).
SHAPES = [(2, 1), (4, 3), (9, 8), (21, 20), (30, 8), (8, 8), (50, 5), (3, 1)]
SEEDS = [0, 1, 2]


def random_problem(m: int, k: int, seed: int):
    """Random well-posed NNLS data (full column rank with probability one)."""
    rng = np.random.default_rng(1000 * seed + 10 * m + k)
    return rng.standard_normal((m, k)), rng.standard_normal(m)


def assert_kkt(res: NNLSResult, e: np.ndarray, f: np.ndarray, tol: float = KKT_TOL):
    """Check the NNLS optimality conditions on the returned result."""
    x = np.asarray(res.x)
    passive = np.asarray(res.passive)
    r = e @ x - f
    w = -e.T @ r  # dual, w = Eᵀ (f - E x)
    scale = max(1.0, np.linalg.norm(e) * np.linalg.norm(f))
    assert np.all(x >= 0.0)
    assert np.all(x[~passive] == 0.0)  # off the passive set x is pinned at 0
    assert np.all(w[~passive] <= tol * scale)  # dual feasibility on the active set
    assert np.all(np.abs(w[passive]) <= tol * scale)  # stationarity on the passive set
    assert np.all(np.abs(x * w) <= tol * scale * max(1.0, np.max(x, initial=0.0)))
    np.testing.assert_allclose(np.asarray(res.residual), r, atol=1e-12 * scale)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("m, k", SHAPES)
def test_matches_scipy(m, k, seed):
    """Solution, residual norm and KKT conditions agree with SciPy."""
    e, f = random_problem(m, k, seed)
    res = nnls(jnp.asarray(e), jnp.asarray(f))
    x_ref, rnorm_ref = scipy_nnls(e, f)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.x), x_ref, atol=ATOL)
    np.testing.assert_allclose(
        float(jnp.linalg.norm(res.residual)), rnorm_ref, atol=ATOL
    )
    assert_kkt(res, e, f)


def _rank_deficient(seed: int):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((12, 3))
    return a @ rng.standard_normal((3, 7)), rng.standard_normal(12)


def _duplicated_columns(seed: int):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((9, 4))
    return np.hstack([a, a, a[:, :1]]), rng.standard_normal(9)


def _zero_column(seed: int):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((9, 5))
    e[:, 2] = 0.0
    return e, rng.standard_normal(9)


def _tied_duals(seed: int):
    # Orthonormal columns and f with exactly tied entries: all duals tie.
    del seed
    return np.eye(6), np.array([1.0, 1.0, -1.0, 0.0, 1.0, -1.0])


def _tied_duals_dependent(seed: int):
    # Two identical columns have the identical (tied) dual at every step.
    del seed
    e = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    return e, np.array([2.0, 1.0, 3.0])


def _negative_target(seed: int):
    # f in the negative cone of the columns: the optimum is x = 0.
    rng = np.random.default_rng(seed)
    e = np.abs(rng.standard_normal((10, 4)))
    return e, -np.abs(rng.standard_normal(10))


def _zero_target(seed: int):
    del seed
    return np.random.default_rng(0).standard_normal((5, 3)), np.zeros(5)


def _zero_matrix(seed: int):
    del seed
    return np.zeros((5, 3)), np.random.default_rng(0).standard_normal(5)


def _consistent_positive(seed: int):
    # Exactly representable with an interior solution: f = E x*, x* > 0.
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((15, 6))
    return e, e @ rng.uniform(0.5, 2.0, size=6)


DEGENERATE = {
    "rank-deficient": _rank_deficient,
    "duplicated-columns": _duplicated_columns,
    "zero-column": _zero_column,
    "tied-duals": _tied_duals,
    "tied-duals-dependent": _tied_duals_dependent,
    "negative-target": _negative_target,
    "zero-target": _zero_target,
    "zero-matrix": _zero_matrix,
    "consistent-positive": _consistent_positive,
}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("make", DEGENERATE.values(), ids=DEGENERATE.keys())
def test_degenerate_problems(make, seed):
    """Rank-deficient / tied / trivial data: optimal value and KKT hold.

    The minimiser is not unique for dependent columns, so compare the
    optimal objective value (and the residual, which is unique) to SciPy.
    """
    e, f = make(seed)
    res = nnls(jnp.asarray(e), jnp.asarray(f))
    _, rnorm_ref = scipy_nnls(e, f)
    assert bool(res.converged)
    np.testing.assert_allclose(
        float(jnp.linalg.norm(res.residual)), rnorm_ref, atol=ATOL
    )
    assert_kkt(res, e, f)
    assert int(res.n_iter) <= 3 * e.shape[1]


@pytest.mark.parametrize("m, k", [(3, 0), (0, 0), (0, 3)])
def test_zero_size(m, k):
    """Empty problems return empty / zero outputs and converge at once."""
    res = nnls(jnp.ones((m, k)), jnp.ones((m,)))
    assert res.x.shape == (k,)
    assert res.residual.shape == (m,)
    assert res.passive.shape == (k,)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.residual), -1.0)
    assert int(res.n_iter) == 0
    assert bool(res.converged)


@pytest.mark.parametrize("m, k", SHAPES)
def test_jit_and_budget(m, k):
    """``jit`` reproduces the eager result; ``n_iter`` stays within ``3 k``."""
    e, f = map(jnp.asarray, random_problem(m, k, 7))
    eager = nnls(e, f)
    jitted = jax.jit(nnls)(e, f)
    np.testing.assert_allclose(np.asarray(jitted.x), np.asarray(eager.x), atol=ATOL)
    np.testing.assert_array_equal(np.asarray(jitted.passive), np.asarray(eager.passive))
    assert int(jitted.n_iter) == int(eager.n_iter)
    assert bool(jitted.converged)
    assert 0 <= int(jitted.n_iter) <= 3 * k
    # The budget and tolerance may be traced as well.
    traced = jax.jit(nnls)(e, f, 3 * k, 1e-12)
    np.testing.assert_allclose(np.asarray(traced.x), np.asarray(eager.x), atol=ATOL)


@pytest.mark.parametrize("max_iter", [0, 1, 3, 7, 8, 24])
def test_max_iter_and_converged_flag(max_iter):
    """``converged`` is true exactly when the budget sufficed.

    The data is consistent with an all-positive solution, so the method
    performs exactly ``k`` additions and no removal: it needs a budget of
    ``k`` and stops with ``n_iter == min(max_iter, k)``.
    """
    k = 8
    rng = np.random.default_rng(11)
    e = rng.standard_normal((20, k))
    f = e @ rng.uniform(0.5, 2.0, size=k)
    res = nnls(jnp.asarray(e), jnp.asarray(f), max_iter=max_iter)
    assert bool(res.converged) == (max_iter >= k)
    assert int(res.n_iter) == min(max_iter, k)
    # Whatever the budget, the iterate is feasible, with a consistent residual.
    assert np.all(np.asarray(res.x) >= 0.0)
    np.testing.assert_allclose(
        np.asarray(res.residual), e @ np.asarray(res.x) - f, atol=1e-10
    )
    assert float(jnp.linalg.norm(res.residual)) <= np.linalg.norm(f) + 1e-12
    if max_iter >= k:
        assert_kkt(res, e, f)


@pytest.mark.parametrize(
    "tol, keep_solution",
    [(1e-14, 1.0), (1e-8, 1.0), (1e8, 0.0)],
    ids=["tiny", "loose", "huge-stops-at-zero"],
)
def test_tol_override(tol, keep_solution):
    """A huge ``tol`` stops at ``x = 0``; a tiny one still reaches the optimum."""
    e, f = random_problem(9, 4, 5)
    res = nnls(jnp.asarray(e), jnp.asarray(f), tol=tol)
    x_ref, _ = scipy_nnls(e, f)
    np.testing.assert_allclose(np.asarray(res.x), keep_solution * x_ref, atol=1e-8)
    assert bool(res.converged)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype_follows_input(dtype):
    """Outputs carry the input float dtype (counters stay ``int32``)."""
    e, f = random_problem(9, 4, 2)
    res = nnls(jnp.asarray(e, dtype), jnp.asarray(f, dtype))
    assert res.x.dtype == dtype
    assert res.residual.dtype == dtype
    assert res.n_iter.dtype == jnp.int32
    assert res.passive.dtype == jnp.bool_
    x_ref, _ = scipy_nnls(e, f)
    atol = 1e-4 if dtype == jnp.float32 else ATOL
    np.testing.assert_allclose(np.asarray(res.x), x_ref, atol=atol)
    assert bool(res.converged)


RHO = 10.0  # strength of the shared column component (correlated columns)


def ldp_shaped_problem(k: int, seed: int):
    """``E = [Gᵀ; hᵀ]``, ``f = e_last`` with ``G`` of correlated rows (LDP form)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((k, k)) + RHO * rng.standard_normal(k)[:, None]
    h = rng.standard_normal(k)
    f = np.zeros(k + 1)
    f[-1] = 1.0
    return np.vstack([g.T, h[None, :]]), f


def mixed_sign_problem(k: int, seed: int):
    """``f = E x_true + noise`` with mixed-sign ``x_true`` and correlated columns."""
    m = 3 * k // 2
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((m, k)) + RHO * rng.standard_normal(m)[:, None]
    return e, e @ rng.standard_normal(k) + rng.standard_normal(m)


# (generator, k, seed): seeds were chosen so that the solver performs at least
# one interpolation step that releases an index (asserted in the test), which
# plain random data almost never does.
REMOVAL_CASES = {
    f"{gen.__name__}-k{k}-s{seed}": (gen, k, seed)
    for gen, k, seeds in [
        (ldp_shaped_problem, 8, (1, 2, 5)),
        (ldp_shaped_problem, 20, (2, 4, 5)),
        (ldp_shaped_problem, 40, (0, 2, 4)),
        (mixed_sign_problem, 8, (12, 13, 15)),
        (mixed_sign_problem, 20, (0, 3, 5)),
        (mixed_sign_problem, 40, (1, 3, 4)),
    ]
    for seed in seeds
}


@pytest.mark.parametrize(
    "gen, k, seed", REMOVAL_CASES.values(), ids=REMOVAL_CASES.keys()
)
def test_removal_heavy_problems(gen, k, seed):
    """Interpolation / release path: SciPy agreement and KKT with removals.

    ``n_iter > |passive|`` proves that at least one index entered the passive
    set and later left it (every addition either stays passive or is undone
    by a skip or an interpolation release).
    """
    e, f = gen(k, seed)
    res = nnls(jnp.asarray(e), jnp.asarray(f))
    x_ref, rnorm_ref = scipy_nnls(e, f, maxiter=100 * k)
    assert bool(res.converged)
    assert int(res.n_iter) > int(jnp.sum(res.passive))
    np.testing.assert_allclose(
        float(jnp.linalg.norm(res.residual)), rnorm_ref, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(res.x), x_ref, rtol=1e-7, atol=1e-8)
    assert_kkt(res, e, f)


def ldp_rank_deficient_problem(k: int, seed: int):
    """LDP form with ``k`` columns but only ``k // 3 + 1`` rows (dependent columns)."""
    nx = max(2, k // 3)
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((k, nx))
    h = 0.5 * rng.standard_normal(k)
    f = np.zeros(nx + 1)
    f[-1] = 1.0
    return np.vstack([g.T, h[None, :]]), f


# Found by a mutation search: with a wrong interpolation step length or a
# wrong release rule the solver stops at a suboptimal point (or fails to
# converge) on exactly these problems.
RANK_DEFICIENT_CASES = {
    f"k{k}-s{seed}": (k, seed) for k, seed in [(8, 140), (20, 258), (8, 3), (20, 5)]
}


@pytest.mark.parametrize(
    "k, seed", RANK_DEFICIENT_CASES.values(), ids=RANK_DEFICIENT_CASES.keys()
)
def test_removal_heavy_rank_deficient(k, seed):
    """Dependent columns: the minimiser is not unique, the optimum value is.

    Compare the optimal value with SciPy and check KKT. The first two seeds
    are sensitive to the interpolation step length and to the release rule.
    """
    e, f = ldp_rank_deficient_problem(k, seed)
    res = nnls(jnp.asarray(e), jnp.asarray(f))
    _, rnorm_ref = scipy_nnls(e, f, maxiter=100 * k)
    assert bool(res.converged)
    np.testing.assert_allclose(
        float(jnp.linalg.norm(res.residual)), rnorm_ref, rtol=1e-9, atol=1e-10
    )
    assert_kkt(res, e, f)


SKIP_CASES = {
    f"{m}x{k}-s{seed}": (m, k, seed)
    for m, k in [(4, 3), (9, 8), (21, 20), (30, 8)]
    for seed in (0, 1, 3)
}


@pytest.mark.parametrize("m, k, seed", SKIP_CASES.values(), ids=SKIP_CASES.keys())
def test_skip_rule_with_negative_tol(m, k, seed):
    """A negative ``tol`` forces additions of indices that cannot enter.

    With ``tol = -1.5 · min |w_j|`` over the active set at the optimum, the
    method adds the index with the least negative dual; its least-squares
    coefficient is negative, so the skipping rule has to drop it again
    (otherwise it would cycle until the budget). The result must still be the
    SciPy solution and ``n_iter`` must count the skipped additions.
    """
    e, f = random_problem(m, k, seed)
    x_ref, _ = scipy_nnls(e, f)
    w_ref = e.T @ (f - e @ x_ref)
    active = x_ref == 0.0
    assert np.any(active)  # the chosen seeds all have a non-empty active set
    tol = -1.5 * np.min(np.abs(w_ref[active]))
    res = nnls(jnp.asarray(e), jnp.asarray(f), tol=tol)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.x), x_ref, atol=1e-9)
    assert int(res.n_iter) > int(jnp.sum(res.passive))
