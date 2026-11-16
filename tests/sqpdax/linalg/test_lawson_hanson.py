"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.lawson_hanson`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import nnls as scipy_nnls

from slsqp_jax.sqpdax.linalg import LDP_RESULTS, LDPResult, NNLSResult, ldp, nnls

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


# --------------------------------------------------------------------------
# ldp: min ||x|| s.t. g x >= h
# --------------------------------------------------------------------------

LDP_SHAPES = [(k, n) for k in (1, 3, 8, 20) for n in (1, 2, 5, 12)]
LDP_SEEDS = (0, 1)
LDP_TOL = 1e-9


def _feasible_slack(rng, k):
    return rng.uniform(0.0, 1.0, size=k)


def ldp_random(k, n, seed):
    """Feasible by construction (``x0`` satisfies ``g x0 - h = slack >= 0``)."""
    rng = np.random.default_rng(100 * seed + 7 * k + n)
    g = rng.standard_normal((k, n))
    return g, g @ (2.0 * rng.standard_normal(n)) - _feasible_slack(rng, k)


def ldp_rank_deficient(k, n, seed):
    """Rows of ``g`` are multiples of two directions: ``rank(g) <= 2``."""
    rng = np.random.default_rng(100 * seed + 7 * k + n)
    basis = rng.standard_normal((2, n))
    g = rng.standard_normal((k, 2)) @ basis
    return g, g @ (2.0 * rng.standard_normal(n)) - _feasible_slack(rng, k)


def ldp_tied(k, n, seed):
    """Unit rows ``e_{i mod n}`` with ``h = 1``: duplicated, tied constraints."""
    del seed
    g = np.eye(n)[np.arange(k) % n]
    return g, np.ones(k)


def ldp_origin_feasible(k, n, seed):
    """``h <= 0``: the origin is feasible, hence optimal."""
    rng = np.random.default_rng(100 * seed + 7 * k + n)
    return rng.standard_normal((k, n)), -rng.uniform(0.0, 1.0, size=k)


LDP_GENERATORS = {
    "random": ldp_random,
    "rank-deficient": ldp_rank_deficient,
    "tied": ldp_tied,
    "origin-feasible": ldp_origin_feasible,
}


def scipy_ldp(g, h):
    """Reference: SLSQP on ``½‖x‖²`` with the linear inequality constraints."""
    n = g.shape[1]
    out = scipy_minimize(
        lambda x: 0.5 * x @ x,
        np.zeros(n),
        jac=lambda x: x,
        constraints=[{"type": "ineq", "fun": lambda x: g @ x - h, "jac": lambda x: g}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert out.success
    return out.x


def assert_ldp_kkt(res: LDPResult, g, h, tol: float = LDP_TOL, scale=None):
    """KKT conditions of ``min ½‖x‖²  s.t.  g x ≥ h`` (certify optimality)."""
    x = np.asarray(res.x)
    mu = np.asarray(res.multipliers)
    slack = g @ x - h
    if scale is None:
        scale = max(1.0, np.linalg.norm(g), np.linalg.norm(h))
    assert np.all(mu >= 0.0)
    np.testing.assert_allclose(x, g.T @ mu, atol=tol * scale)  # stationarity
    assert np.all(slack >= -tol * scale)  # primal feasibility
    assert np.all(np.abs(mu * slack) <= tol * scale * max(1.0, np.max(mu)))


@pytest.mark.parametrize("seed", LDP_SEEDS)
@pytest.mark.parametrize("k, n", LDP_SHAPES)
@pytest.mark.parametrize("make", LDP_GENERATORS.values(), ids=LDP_GENERATORS.keys())
def test_ldp_feasible_kkt_and_scipy(make, k, n, seed):
    """Feasible problems: status, KKT certificate and SLSQP agreement."""
    g, h = make(k, n, seed)
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    assert res.x.shape == (n,) and res.multipliers.shape == (k,)
    assert 0 <= int(res.n_iter) <= 3 * k
    assert_ldp_kkt(res, g, h)
    np.testing.assert_allclose(np.asarray(res.x), scipy_ldp(g, h), atol=1e-6)


@pytest.mark.parametrize("k, n", LDP_SHAPES)
def test_ldp_origin_feasible_is_zero(k, n):
    """``h ≤ 0``: ``x = 0`` and no multiplier is active."""
    g, h = ldp_origin_feasible(k, n, 0)
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.multipliers), 0.0)


def ldp_opposite_half_spaces(k, n, seed):
    """Rows 0 and 1 are ``x_0 ≥ 1`` and ``-x_0 ≥ 1``; the rest is random."""
    rng = np.random.default_rng(100 * seed + 7 * k + n)
    g = rng.standard_normal((k, n))
    h = -rng.uniform(0.0, 1.0, size=k)
    g[0], h[0] = np.eye(n)[0], 1.0
    g[1], h[1] = -np.eye(n)[0], 1.0
    return g, h


@pytest.mark.parametrize("seed", LDP_SEEDS)
@pytest.mark.parametrize("k, n", [(k, n) for k, n in LDP_SHAPES if k >= 2])
def test_ldp_infeasible(k, n, seed):
    """An infeasible pair of opposite half-spaces is detected."""
    g, h = ldp_opposite_half_spaces(k, n, seed)
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.infeasible)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.multipliers), 0.0)


NONFINITE = {
    "nan-g": (lambda g, h: (g.at[0, 0].set(jnp.nan), h)),
    "nan-h": (lambda g, h: (g, h.at[1].set(jnp.nan))),
    "inf-h": (lambda g, h: (g, h.at[2].set(jnp.inf))),
    "inf-g": (lambda g, h: (g.at[1, 1].set(-jnp.inf), h)),
}


@pytest.mark.parametrize("corrupt", NONFINITE.values(), ids=NONFINITE.keys())
def test_ldp_nonfinite(corrupt):
    """Non-finite data gives ``nonfinite`` (not ``max_iter_reached``)."""
    g, h = map(jnp.asarray, ldp_random(8, 5, 0))
    res = ldp(*corrupt(g, h))
    assert bool(res.status == LDP_RESULTS.nonfinite)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.multipliers), 0.0)


@pytest.mark.parametrize("max_iter", [0, 1, 3, 7, 8])
def test_ldp_max_iter(max_iter):
    """``g = I``, ``h = 1`` needs 8 NNLS additions: a smaller budget stops early."""
    k = 8
    g, h = np.eye(k), np.ones(k)
    res = ldp(jnp.asarray(g), jnp.asarray(h), max_iter=max_iter)
    assert bool(res.status == LDP_RESULTS.max_iter_reached) == (max_iter < k)
    assert bool(res.status == LDP_RESULTS.feasible) == (max_iter >= k)
    assert int(res.n_iter) == min(max_iter, k)
    np.testing.assert_allclose(np.asarray(res.x), (max_iter >= k) * np.ones(k))


@pytest.mark.parametrize(
    "h, expected",
    [
        (np.array([-1.0, 0.0, -0.5]), LDP_RESULTS.feasible),
        (np.array([-1.0, 1e-3, -0.5]), LDP_RESULTS.infeasible),
    ],
    ids=["feasible", "infeasible"],
)
def test_ldp_no_unknowns(h, expected):
    """``n = 0``: the constraints read ``0 ≥ h``."""
    res = ldp(jnp.zeros((3, 0)), jnp.asarray(h))
    assert bool(res.status == expected)
    assert res.x.shape == (0,)
    assert res.multipliers.shape == (3,)


@pytest.mark.parametrize("n", [0, 1, 4])
def test_ldp_no_constraints(n):
    """``k = 0`` (static shape): the origin, feasible, no iterations."""
    res = ldp(jnp.zeros((0, n)), jnp.zeros((0,)))
    assert bool(res.status == LDP_RESULTS.feasible)
    assert res.x.shape == (n,)
    assert res.multipliers.shape == (0,)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    assert int(res.n_iter) == 0


@pytest.mark.parametrize(
    "make", [ldp_random, ldp_opposite_half_spaces], ids=["feasible", "infeasible"]
)
def test_ldp_jit(make):
    """``jit`` (also with traced budgets) reproduces the eager result."""
    g, h = map(jnp.asarray, make(8, 5, 1))
    eager = ldp(g, h)
    jitted = jax.jit(ldp)(g, h)
    traced = jax.jit(ldp)(g, h, 24, 1e-12, 1e-12)
    for res in (jitted, traced):
        assert bool(res.status == eager.status)
        assert int(res.n_iter) == int(eager.n_iter)
        np.testing.assert_allclose(np.asarray(res.x), np.asarray(eager.x), atol=ATOL)
        np.testing.assert_allclose(
            np.asarray(res.multipliers), np.asarray(eager.multipliers), atol=ATOL
        )


@pytest.mark.parametrize(
    "ldp_tol, expected",
    [(1e-12, LDP_RESULTS.feasible), (1e3, LDP_RESULTS.infeasible)],
    ids=["tight", "huge"],
)
def test_ldp_tol_override(ldp_tol, expected):
    """A huge ``ldp_tol`` reports even a feasible problem as infeasible."""
    g, h = map(jnp.asarray, ldp_random(5, 3, 0))
    assert bool(ldp(g, h, ldp_tol=ldp_tol).status == expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ldp_dtype_follows_input(dtype):
    """Outputs carry the input float dtype."""
    g, h = ldp_random(5, 3, 0)
    res = ldp(jnp.asarray(g, dtype), jnp.asarray(h, dtype))
    assert res.x.dtype == dtype and res.multipliers.dtype == dtype
    assert res.n_iter.dtype == jnp.int32
    assert bool(res.status == LDP_RESULTS.feasible)
    atol = 1e-4 if dtype == jnp.float32 else 1e-8
    np.testing.assert_allclose(np.asarray(res.x), scipy_ldp(g, h), atol=atol)


def test_ldp_results_enumeration():
    """The documented codes exist; ``inconsistent_equalities`` is never produced."""
    names = ("feasible", "infeasible", "inconsistent_equalities")
    names += ("max_iter_reached", "nonfinite")
    assert all(hasattr(LDP_RESULTS, name) for name in names)
    g, h = map(jnp.asarray, ldp_random(5, 3, 0))
    assert bool(ldp(g, h).status != LDP_RESULTS.inconsistent_equalities)


SCALES = [1.0, 1e3, 1e6, 1e9, 1e12]


@pytest.mark.parametrize("c", SCALES)
def test_ldp_half_space_scale(c):
    """One active half-space plus an inactive row, ``h`` scaled by ``c``.

    ``x* = c g₀ / ‖g₀‖²`` has norm ``∝ c``. The result must stay accurate
    (relative error of order ``eps``) and feasible, however large ``‖x*‖``.
    """
    g = np.array([[1.2, 1.6], [0.0, 1.0]])
    h = np.array([c, -c])
    x_star = c * g[0] / (g[0] @ g[0])
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(np.asarray(res.x), x_star, rtol=1e-12, atol=0.0)
    # x* = μ₀ g₀ with μ₀ = c / ‖g₀‖²; the inactive row has no multiplier.
    np.testing.assert_allclose(
        np.asarray(res.multipliers), [c / (g[0] @ g[0]), 0.0], rtol=1e-12, atol=0.0
    )
    assert_ldp_kkt(res, g, h, tol=1e-11, scale=np.linalg.norm(x_star))


def test_ldp_feasible_status_implies_formed_solution():
    """The ``g = [[.6, .8], [0, 1]]``, ``h = [c, -c]`` case at ``c = 1e9``.

    The status must match the outputs: ``feasible`` with ``x ≈ c (.6, .8)``,
    never ``feasible`` with a zeroed ``x``.
    """
    c = 1e9
    g = np.array([[0.6, 0.8], [0.0, 1.0]])
    res = ldp(jnp.asarray(g), jnp.asarray([c, -c]))
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(np.asarray(res.x), c * g[0], rtol=1e-12)


@pytest.mark.parametrize("eps_par", [1e-2, 1e-4, 1e-6, 1e-8])
def test_ldp_near_parallel_constraints(eps_par):
    """``x₁ ≥ 1`` and ``-x₁ + ε x₂ ≥ 0`` meet at ``x* = (1, 1/ε)``.

    The conditioning degrades as ``ε → 0`` (``‖x*‖ ~ 1/ε``); the solution must
    still be feasible and accurate to ``1e3 · eps · ‖x*‖`` (relative).
    """
    g = np.array([[1.0, 0.0], [-1.0, eps_par]])
    h = np.array([1.0, 0.0])
    x_star = np.array([1.0, 1.0 / eps_par])
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    bound = 1e3 * np.finfo(np.float64).eps * np.linalg.norm(x_star)
    err = np.linalg.norm(np.asarray(res.x) - x_star) / np.linalg.norm(x_star)
    assert err <= bound
    # KKT residuals are only small relative to ‖x*‖ (the problem is ill conditioned).
    assert_ldp_kkt(res, g, h, tol=1e-7, scale=np.linalg.norm(x_star))


@pytest.mark.parametrize(
    "ldp_tol, expected",
    [(None, LDP_RESULTS.max_iter_reached), (1e3, LDP_RESULTS.infeasible)],
    ids=["default-tol", "residual-certifies-infeasible"],
)
def test_ldp_budget_exhausted_with_small_residual(ldp_tol, expected):
    """Budget exhausted: ``‖r‖ ≤ ldp_tol`` is already an infeasibility certificate.

    ``g = I``, ``h = 1`` needs 8 NNLS additions; with one step the residual is
    still large for the default tolerance (``max_iter_reached``) but below a
    generous ``ldp_tol`` (``infeasible``).
    """
    g, h = np.eye(8), np.ones(8)
    res = ldp(jnp.asarray(g), jnp.asarray(h), max_iter=1, ldp_tol=ldp_tol)
    assert bool(res.status == expected)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)


def test_ldp_nonfinite_precedes_budget_and_residual():
    """``nonfinite`` wins over both ``max_iter_reached`` and ``infeasible``."""
    g, h = np.eye(8), np.ones(8)
    h[3] = np.nan
    res = ldp(jnp.asarray(g), jnp.asarray(h), max_iter=1, ldp_tol=1e3)
    assert bool(res.status == LDP_RESULTS.nonfinite)


def ldp_scaled(k, n, seed, s):
    """Feasible problem with uneven row scales and ``h`` of magnitude ``∝ s``.

    ``h_i = s (g_i·x0 - slack_i ‖g_i‖)`` so that ``x0 · s`` is feasible.
    """
    rng = np.random.default_rng(10_000 * seed + 100 * k + n)
    g = rng.standard_normal((k, n)) * 10.0 ** rng.uniform(-1.0, 1.0, size=(k, 1))
    x0 = 2.0 * rng.standard_normal(n)
    slack = rng.uniform(0.0, 1.0, size=k) * np.linalg.norm(g, axis=1)
    return g, s * (g @ x0 - slack)


SCALED_SIZES = [(14, 7), (8, 5), (5, 3)]
SCALED_MAGNITUDES = [1e-3, 1.0, 1e4, 7e5, 1e8, 1e10]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("s", SCALED_MAGNITUDES)
@pytest.mark.parametrize("k, n", SCALED_SIZES)
def test_ldp_scale_invariance_random(k, n, s, seed):
    """Random feasible problems keep a valid KKT certificate for any ``‖h‖``.

    The NNLS dual tolerance grows with ``‖E‖_F`` (hence with ``‖h‖``) while the
    true duals shrink like ``1/‖x‖``; without internal normalisation NNLS stops
    early and reports a wrong ``feasible`` for ``‖h‖ ≳ 1e6``. KKT residuals are
    measured in row-normalised units and relative to
    ``max(‖x‖, max_i |h_i| / ‖g_i‖)``.
    """
    g, h = ldp_scaled(k, n, seed, s)
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    x, mu = np.asarray(res.x), np.asarray(res.multipliers)
    rn = np.linalg.norm(g, axis=1)
    scale = max(np.linalg.norm(x), np.max(np.abs(h) / rn))
    tol = 1e-9 * scale
    slack = (g @ x - h) / rn  # row-normalised slack
    mu_hat = mu * rn  # multipliers of the row-normalised constraints
    assert np.all(mu >= 0.0)
    assert np.all(slack >= -tol)  # primal feasibility
    np.testing.assert_allclose(x, (g / rn[:, None]).T @ mu_hat, atol=tol, rtol=0.0)
    assert np.all(np.abs(mu_hat * slack) <= tol * max(1.0, np.max(mu_hat)))


# --------------------------------------------------------------------------
# dep_tol: dependent columns are frozen, never added (C11b)
# --------------------------------------------------------------------------


def near_duplicate_problem(m: int, spread: float, noise: float, seed: int):
    """``E = [a0 a1 a2 a0']`` with ``a0' = a0 + spread · noise``; ``f ≈ a0+a1+a2``.

    The fourth column lies within relative distance about ``spread`` of the
    span of the others, so at the optimum it is redundant: with the first
    column passive its dual is only ``O(spread · ‖r‖)``, noise above the
    default dual tolerance.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, 3))
    e = np.column_stack([a, a[:, 0] + spread * rng.standard_normal(m)])
    return e, a.sum(axis=1) + noise * rng.standard_normal(m)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("noise", [0.0, 0.3])
@pytest.mark.parametrize("spread", [1e-9, 1e-12])
def test_nnls_dep_tol_freezes_dependent_column(spread, noise, seed):
    """Only one of two dependent columns enters; the optimum is unchanged.

    The frozen column is implied by the passive ones up to its distance to
    their span, so the optimal value moves by at most about ``spread``.
    """
    e, f = near_duplicate_problem(8, spread, noise, seed)
    res = nnls(jnp.asarray(e), jnp.asarray(f), dep_tol=1e-6)
    passive = np.asarray(res.passive)
    assert bool(res.converged)
    assert passive.sum() == 3
    assert not (passive[0] and passive[3])
    assert np.all(np.asarray(res.x)[passive] > 0.0)
    assert np.all(np.asarray(res.x)[~passive] == 0.0)
    _, rnorm_ref = scipy_nnls(e, f)
    assert abs(float(jnp.linalg.norm(res.residual)) - rnorm_ref) <= 20 * spread + 1e-12


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("m, k", SHAPES)
def test_nnls_dep_tol_only_removes_dependent_columns(m, k, seed):
    """``dep_tol = 0`` is plain Lawson-Hanson, and independent columns are
    never frozen: a positive ``dep_tol`` gives the same iterates on random
    full-rank data (every shape here has ``m ≥ k``)."""
    e, f = random_problem(m, k, seed)
    plain = nnls(jnp.asarray(e), jnp.asarray(f))
    zero = nnls(jnp.asarray(e), jnp.asarray(f), dep_tol=0.0)
    guarded = nnls(jnp.asarray(e), jnp.asarray(f), dep_tol=1e-8)
    for res in (zero, guarded):
        np.testing.assert_array_equal(np.asarray(res.x), np.asarray(plain.x))
        np.testing.assert_array_equal(
            np.asarray(res.passive), np.asarray(plain.passive)
        )
        assert int(res.n_iter) == int(plain.n_iter)
        assert bool(res.converged)


def test_nnls_dep_tol_zero_default_reproduces_c08_degenerate_battery():
    """The degenerate C08 problems solve identically with ``dep_tol = 0``."""
    for make in DEGENERATE.values():
        e, f = make(0)
        a = nnls(jnp.asarray(e), jnp.asarray(f))
        b = nnls(jnp.asarray(e), jnp.asarray(f), dep_tol=0.0)
        np.testing.assert_array_equal(np.asarray(a.x), np.asarray(b.x))
        assert int(a.n_iter) == int(b.n_iter)


@pytest.mark.parametrize("dep_tol", [1e-8, 1e-6])
def test_nnls_dep_tol_zero_columns_and_empty_passive_set(dep_tol):
    """The first addition is never frozen, and a zero column cannot enter."""
    e = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    f = np.array([1.0, 0.5, 0.0])
    res = nnls(jnp.asarray(e), jnp.asarray(f), dep_tol=dep_tol)
    _, rnorm_ref = scipy_nnls(e, f)
    assert bool(res.converged)
    assert not bool(res.passive[1])
    np.testing.assert_allclose(
        float(jnp.linalg.norm(res.residual)), rnorm_ref, atol=1e-9
    )


# --------------------------------------------------------------------------
# ldp: feasible must never come with violated constraints (C11b)
# --------------------------------------------------------------------------

COARSE_PROBLEM_SIZES = [(4, 3), (8, 5), (20, 6)]


def coarse_ldp_problem(k: int, n: int, seed: int):
    """Feasible data; a coarse NNLS dual tolerance then stops far from optimal."""
    rng = np.random.default_rng(100 * seed + 7 * k + n)
    g = rng.standard_normal((k, n))
    y0 = 2.0 * rng.standard_normal(n)
    h = g @ y0 + np.abs(rng.standard_normal(k)) * (rng.random(k) < 0.5) + 1.0
    return g, h


def violation(g, h, x):
    """Largest violation of ``g x ≥ h`` in row-normalised units."""
    return float(np.max(np.maximum((h - g @ x) / np.linalg.norm(g, axis=1), 0.0)))


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("tol", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("k, n", COARSE_PROBLEM_SIZES)
def test_ldp_feasible_status_means_constraints_hold(k, n, tol, seed):
    """A garbage NNLS solution (premature stop) is reported ``infeasible``.

    ``feasible`` holds exactly when the returned point satisfies the
    constraints; otherwise the status is ``infeasible`` with zero outputs.
    """
    g, h = coarse_ldp_problem(k, n, seed)
    res = ldp(jnp.asarray(g), jnp.asarray(h), tol=tol)
    feasible = bool(res.status == LDP_RESULTS.feasible)
    x, mu = np.asarray(res.x), np.asarray(res.multipliers)
    assert bool(res.status == LDP_RESULTS.infeasible) != feasible
    holds = violation(g, h, x) <= 1e-6 * max(1.0, np.linalg.norm(x))
    assert feasible == holds
    assert feasible or not (np.any(x) or np.any(mu))


@pytest.mark.parametrize(
    "check_tol, expected",
    [(None, LDP_RESULTS.infeasible), (float("inf"), LDP_RESULTS.feasible)],
    ids=["default", "disabled"],
)
def test_ldp_check_tol_override(check_tol, expected):
    """``check_tol = inf`` switches the primal check off (violated ``x`` shows)."""
    g, h = coarse_ldp_problem(8, 5, 0)
    kwargs = {} if check_tol is None else {"check_tol": check_tol}
    res = ldp(jnp.asarray(g), jnp.asarray(h), tol=0.5, **kwargs)
    assert bool(res.status == expected)
    x = np.asarray(res.x)
    feasible = expected == LDP_RESULTS.feasible
    assert bool(np.any(x)) == feasible  # zero outputs unless feasible
    assert violation(g, h, x) > 1e-3 or not feasible  # the unchecked x is garbage


# --------------------------------------------------------------------------
# ldp: Farkas certificates of infeasibility (C11c)
# --------------------------------------------------------------------------


def farkas_infeasible(k, n, seed, s=1.0, decades=0.0):
    """Infeasible ``g x ≥ h`` by construction: ``y > 0`` with ``gᵀy = 0``, ``hᵀy > 0``.

    The last row of ``g`` is chosen so that ``gᵀy = 0`` exactly, ``h`` is
    shifted to give ``hᵀy = 1``. Then ``yᵀ(g x - h) = -1 < 0`` for every
    ``x``, whatever the row scales (``10^{±decades}`` per row, applied to
    ``g`` and ``h`` together) and the magnitude ``s`` of ``h``.
    """
    rng = np.random.default_rng(10_000 * seed + 100 * k + n)
    g = rng.standard_normal((k, n))
    y = rng.uniform(0.5, 1.5, size=k)
    g[-1] = -(y[:-1] @ g[:-1]) / y[-1]
    h = rng.standard_normal(k)
    h[-1] += (1.0 - h @ y) / y[-1]
    scale = 10.0 ** rng.uniform(-decades, decades, size=k)
    return g * scale[:, None], s * h * scale


def assert_ldp_certificate(res: LDPResult, g, h):
    """``c ≥ 0``, ``‖c‖_∞ = 1``, ``gᵀc ≈ 0`` and ``hᵀc > 0``."""
    c = np.asarray(res.certificate)
    assert c.shape == (g.shape[0],)
    assert np.all(c >= 0.0)
    np.testing.assert_allclose(np.max(c), 1.0, rtol=1e-12)
    assert np.linalg.norm(g.T @ c) <= 1e-8 * np.linalg.norm(g)
    assert h @ c > 0.0
    # the sign is not a rounding artefact: it is a sizeable part of |h|ᵀc
    assert h @ c >= 1e-6 * (np.abs(h) @ c)


INFEASIBLE_GENERATORS = {
    "opposite": ldp_opposite_half_spaces,
    "farkas": farkas_infeasible,
    "farkas-uneven-rows": lambda k, n, seed: farkas_infeasible(k, n, seed, decades=3.0),
    "farkas-large-h": lambda k, n, seed: farkas_infeasible(k, n, seed, s=1e6),
    "farkas-small-h": lambda k, n, seed: farkas_infeasible(k, n, seed, s=1e-4),
}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("k, n", [(k, n) for k, n in LDP_SHAPES if k >= 2])
@pytest.mark.parametrize(
    "make", INFEASIBLE_GENERATORS.values(), ids=INFEASIBLE_GENERATORS.keys()
)
def test_ldp_certificate_proves_infeasibility(make, k, n, seed):
    """An infeasible exit carries a valid Farkas certificate."""
    g, h = make(k, n, seed)
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.infeasible)
    assert_ldp_certificate(res, g, h)


def ldp_nnls_residual_norm(g, h) -> float:
    """``‖r‖`` of the NNLS that :func:`ldp` solves (NumPy copy of its normalisation).

    Rows of unit norm, then ``h`` scaled by the largest violation at the origin
    (``s = max_i max(ĥ_i, 0)``, 1 if zero); the NNLS data are ``E = [ĝᵀ; (ĥ/s)ᵀ]``
    with unit-norm columns, and ``f = e_{n+1}``. Reference only: it lets a test
    place ``ldp_tol`` relative to the residual that the code under test sees.
    """
    k, n = g.shape
    row_norm = np.linalg.norm(g, axis=1)
    g_n = g / row_norm[:, None]
    h_hat = h / row_norm
    h_scale = np.max(np.maximum(h_hat, 0.0))
    h_n = h_hat / (h_scale if h_scale > 0.0 else 1.0)
    e = np.concatenate([g_n.T, h_n[None, :]], axis=0)
    e = e / np.linalg.norm(e, axis=0)
    f = np.zeros(n + 1)
    f[n] = 1.0
    return float(jnp.linalg.norm(nnls(jnp.asarray(e), jnp.asarray(f)).residual))


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("k, n", [(k, n) for k, n in LDP_SHAPES if k >= 2])
@pytest.mark.parametrize(
    "make", INFEASIBLE_GENERATORS.values(), ids=INFEASIBLE_GENERATORS.keys()
)
def test_ldp_post_check_exit_keeps_a_verified_certificate(make, k, n, seed):
    """A residual just above the noise threshold still certifies infeasibility.

    ``ldp_tol`` is set to half of the NNLS residual, i.e. the residual test
    fails by a factor of two (the situation of a rounding-level residual that
    lands a little above ``10·eps·…``), so the ``infeasible`` status comes from
    the primal check. (The ``-1e-300`` keeps the residual test failing for an
    NNLS residual of exactly zero.) The NNLS solution verifies as a Farkas certificate
    (``gᵀc ≈ 0``, ``hᵀc > 0``) and is the very one of the residual-test exit;
    status, ``x`` and ``multipliers`` are those of a certificate-less exit.
    """
    g, h = make(k, n, seed)
    g_j, h_j = jnp.asarray(g), jnp.asarray(h)
    by_residual = ldp(g_j, h_j)
    r_norm = ldp_nnls_residual_norm(g, h)
    res = ldp(g_j, h_j, ldp_tol=0.5 * r_norm - 1e-300)
    assert bool(by_residual.status == LDP_RESULTS.infeasible)
    assert bool(res.status == LDP_RESULTS.infeasible)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.multipliers), 0.0)
    assert_ldp_certificate(res, g, h)
    np.testing.assert_array_equal(
        np.asarray(res.certificate), np.asarray(by_residual.certificate)
    )


@pytest.mark.parametrize("seed", LDP_SEEDS)
@pytest.mark.parametrize("k, n", LDP_SHAPES)
@pytest.mark.parametrize("make", LDP_GENERATORS.values(), ids=LDP_GENERATORS.keys())
def test_ldp_certificate_is_zero_on_feasible(make, k, n, seed):
    g, h = make(k, n, seed)
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    assert res.certificate.shape == (k,)
    np.testing.assert_array_equal(np.asarray(res.certificate), 0.0)


def _post_check_infeasible():
    g, h = coarse_ldp_problem(8, 5, 0)
    return g, h, {"tol": 0.5}, LDP_RESULTS.infeasible


def _budget_exhausted():
    g, h = ldp_random(8, 5, 0)
    return g, h, {"max_iter": 0}, LDP_RESULTS.max_iter_reached


def _nonfinite():
    g, h = ldp_random(8, 5, 0)
    g = g.copy()
    g[0, 0] = np.nan
    return g, h, {}, LDP_RESULTS.nonfinite


@pytest.mark.parametrize(
    "make", [_post_check_infeasible, _budget_exhausted, _nonfinite]
)
def test_ldp_certificate_is_zero_without_a_residual_infeasible_exit(make):
    """No certificate without a verified one: the coarse-``tol`` primal-check
    ``infeasible`` (its NNLS solution is garbage, not a certificate; see
    ``test_ldp_post_check_exit_keeps_a_verified_certificate`` for the one
    that verifies), ``max_iter_reached`` and ``nonfinite`` carry none."""
    g, h, kwargs, expected = make()
    res = ldp(jnp.asarray(g), jnp.asarray(h), **kwargs)
    assert bool(res.status == expected)
    np.testing.assert_array_equal(np.asarray(res.certificate), 0.0)


@pytest.mark.parametrize("n", [1, 4])
def test_ldp_certificate_without_constraints(n):
    res = ldp(jnp.zeros((0, n)), jnp.zeros((0,)))
    assert res.certificate.shape == (0,)


def test_ldp_certificate_jit_matches_eager():
    g, h = farkas_infeasible(8, 5, 0, decades=2.0)
    eager = ldp(jnp.asarray(g), jnp.asarray(h))
    jitted = jax.jit(ldp)(jnp.asarray(g), jnp.asarray(h))
    assert bool(jitted.status == LDP_RESULTS.infeasible)
    np.testing.assert_allclose(
        np.asarray(jitted.certificate), np.asarray(eager.certificate), atol=1e-12
    )


# --------------------------------------------------------------------------
# ldp: far slack rows (C16d)
# --------------------------------------------------------------------------
#
# A row that holds at the origin (``h_i <= 0`` in ``g x >= h``) can lie
# arbitrarily far away. If such a slack row set the scale of the
# normalisation (``s = max_i |h_i| / ‖g_i‖``), the normalised solution would
# be tiny, ``‖y‖ = ‖x‖ / s << 1``, and the post-check and the NNLS dual
# tolerance would be far too coarse for the near rows: ``ldp`` returned
# ``feasible`` with violated rows. The scale is now set by the rows violated at
# the origin only, and the NNLS runs on unit-norm columns of ``E``.


def assert_farkas_certificate(res: LDPResult, g, h):
    """``c ≥ 0``, ``‖c‖_∞ = 1``, ``gᵀc ≈ 0`` and ``hᵀc > 0`` (no relative size).

    Weaker than :func:`assert_ldp_certificate`, which also requires ``hᵀc`` to
    be a sizeable part of ``|h|ᵀc``: with a near-infeasibility of relative size
    ``1e-6`` the cancellation of ``hᵀc`` is inherent to the problem.
    """
    c = np.asarray(res.certificate)
    assert c.shape == (g.shape[0],)
    assert np.all(c >= 0.0)
    np.testing.assert_allclose(np.max(c), 1.0, rtol=1e-12)
    assert np.linalg.norm(g.T @ c) <= 1e-8 * np.linalg.norm(g)
    assert h @ c > 0.0


def row_violation(g, h, x):
    """Largest violation of ``g x ≥ h`` in units of the rows' own norms."""
    return float(np.max(np.maximum(h - g @ x, 0.0) / np.linalg.norm(g, axis=1)))


FAR_SLACK_1D_GAPS = [1e-2, 1e-4, 1e-6]
FAR_SLACK_1D_DISTANCES = [1e2, 1e6, 1e10]
FAR_SLACK_1D_ROW_NORMS = [1.0, 1e-6]


@pytest.mark.parametrize("t", FAR_SLACK_1D_ROW_NORMS)
@pytest.mark.parametrize("far", FAR_SLACK_1D_DISTANCES)
@pytest.mark.parametrize("gap", FAR_SLACK_1D_GAPS)
def test_ldp_far_slack_row_does_not_hide_infeasibility(gap, far, t):
    """``x ≥ 1`` and ``x ≤ 1 - gap`` with a far slack row ``-t x ≥ -far·t``.

    The problem is infeasible by ``gap``, whatever the distance ``far`` of the
    slack row (``x ≤ far``) and its norm ``t``. Before C16d ``ldp`` returned
    ``feasible`` with the near rows violated by ``gap`` (``gap = 1e-2`` for
    ``far ≥ 1e6``, ``gap = 1e-4`` for ``far ≥ 1e4``, ``gap = 1e-6`` already at
    ``far = 1e2``) or an uncertified ``infeasible``.
    """
    g = np.array([[1.0], [-1.0], [-t]])
    h = np.array([1.0, -(1.0 - gap), -far * t])
    res = ldp(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.infeasible)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.multipliers), 0.0)
    assert_farkas_certificate(res, g, h)
    # The certificate is the pair of near rows: the slack row is not needed.
    c = np.asarray(res.certificate)
    np.testing.assert_allclose(c[:2], 1.0, atol=1e-6)
    assert h @ c == pytest.approx(gap, rel=1e-4)


FAR_SLACK_DISTANCES = [1e2, 1e5, 1e8, 1e11]
FAR_SLACK_GAPS = [1e-1, 1e-3, 1e-5]
FAR_SLACK_SEEDS = range(6)


def far_slack_problem(far, gap, sign, seed, n=4, k_near=5, k_far=2):
    """Near rows with a slab of signed relative width ``sign · gap``, plus far rows.

    The near rows ``g x ≥ h`` are satisfied by ``x0`` (some of them with zero
    slack). Rows 0 and ``k_near - 1`` bound a slab around ``x0``: row 0 is
    tight at ``x0`` and the last row is ``g_0·x ≤ g_0·x0 + sign · gap ·
    ‖g_0‖ max(1, ‖x0‖)``, so the near problem is feasible for ``sign = +1`` and
    infeasible (``y = (1, ..., 1)`` is a Farkas certificate) for ``sign = -1``.
    The ``k_far`` slack rows have norms in ``1e-8 .. 1`` and lie at distance
    ``far`` (``h_i = -far · ‖g_i‖ · (1 + u)``, ``u ∈ [0, 1)``): they hold at
    ``x0``, at the origin and at the minimum-norm point of the near rows.
    Returns ``(g_near, h_near, g_far, h_far)``.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((k_near, n))
    x0 = rng.standard_normal(n)
    h = g @ x0 - np.abs(rng.standard_normal(k_near)) * (rng.random(k_near) < 0.5)
    h[0] = g[0] @ x0
    g[-1] = -g[0]
    h[-1] = -h[0] - sign * gap * np.linalg.norm(g[0]) * max(1.0, np.linalg.norm(x0))
    g_far = rng.standard_normal((k_far, n)) * 10.0 ** rng.uniform(-8, 0, (k_far, 1))
    h_far = -far * np.linalg.norm(g_far, axis=1) * (1.0 + rng.random(k_far))
    return g, h, g_far, h_far


def lp_is_feasible(g, h):
    """Reference: HiGHS feasibility of ``g x ≥ h`` (zero objective)."""
    n = g.shape[1]
    lp = linprog(
        np.zeros(n), A_ub=-g, b_ub=-h, bounds=[(None, None)] * n, method="highs"
    )
    return lp.status == 0


@pytest.mark.parametrize("seed", FAR_SLACK_SEEDS)
@pytest.mark.parametrize("gap", FAR_SLACK_GAPS)
@pytest.mark.parametrize("far", FAR_SLACK_DISTANCES)
def test_ldp_far_slack_battery_feasible(far, gap, seed):
    """Feasible near rows plus far slack rows: a correct ``feasible`` exit.

    The rows are met to ``1e-6 · max(1, ‖x‖)`` (in the rows' own units), the
    solution matches the SciPy reference of the near rows to ``1e-5``, the far
    rows stay inactive (zero multipliers) and the LP agrees on feasibility.
    """
    g_near, h_near, g_far, h_far = far_slack_problem(far, gap, +1, seed)
    g, h = np.vstack([g_near, g_far]), np.concatenate([h_near, h_far])
    assert lp_is_feasible(g, h)
    res = jax.jit(ldp)(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.feasible)
    x = np.asarray(res.x)
    assert row_violation(g, h, x) <= 1e-6 * max(1.0, np.linalg.norm(x))
    ref = scipy_ldp(g_near, h_near)
    np.testing.assert_allclose(x, ref, atol=1e-5 * max(1.0, np.linalg.norm(ref)))
    np.testing.assert_array_equal(np.asarray(res.multipliers)[len(h_near) :], 0.0)
    np.testing.assert_array_equal(np.asarray(res.certificate), 0.0)


@pytest.mark.parametrize("seed", FAR_SLACK_SEEDS)
@pytest.mark.parametrize("gap", FAR_SLACK_GAPS)
@pytest.mark.parametrize("far", FAR_SLACK_DISTANCES)
def test_ldp_far_slack_battery_infeasible(far, gap, seed):
    """Near rows infeasible by a relative gap, plus far slack rows: certified.

    Before C16d ``ldp`` returned a wrong ``feasible`` on every such seed for
    ``far ≥ 1e8`` and on most of them for ``gap ≤ 1e-3`` at ``far = 1e5``.
    """
    g_near, h_near, g_far, h_far = far_slack_problem(far, gap, -1, seed)
    g, h = np.vstack([g_near, g_far]), np.concatenate([h_near, h_far])
    assert not lp_is_feasible(g, h)
    res = jax.jit(ldp)(jnp.asarray(g), jnp.asarray(h))
    assert bool(res.status == LDP_RESULTS.infeasible)
    np.testing.assert_array_equal(np.asarray(res.x), 0.0)
    np.testing.assert_array_equal(np.asarray(res.multipliers), 0.0)
    assert_farkas_certificate(res, g, h)


@pytest.mark.parametrize("seed", FAR_SLACK_SEEDS)
@pytest.mark.parametrize("far", FAR_SLACK_DISTANCES)
@pytest.mark.parametrize("make", LDP_GENERATORS.values(), ids=LDP_GENERATORS.keys())
def test_ldp_far_slack_row_leaves_feasible_solution_unchanged(make, far, seed):
    """Appending far slack rows changes neither ``x`` nor the near multipliers."""
    k, n = 8, 5
    g, h = make(k, n, seed)
    rng = np.random.default_rng(seed)
    g_far = rng.standard_normal((2, n)) * 10.0 ** rng.uniform(-8, 0, (2, 1))
    h_far = -far * np.linalg.norm(g_far, axis=1) * (1.0 + rng.random(2))
    base = ldp(jnp.asarray(g), jnp.asarray(h))
    res = ldp(
        jnp.asarray(np.vstack([g, g_far])), jnp.asarray(np.concatenate([h, h_far]))
    )
    assert bool(base.status == LDP_RESULTS.feasible)
    assert bool(res.status == LDP_RESULTS.feasible)
    x_ref = np.asarray(base.x)
    scale = max(1.0, np.linalg.norm(x_ref))
    np.testing.assert_allclose(np.asarray(res.x), x_ref, atol=1e-8 * scale, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(res.multipliers)[:k],
        np.asarray(base.multipliers),
        atol=1e-8 * max(1.0, np.max(np.asarray(base.multipliers))),
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(res.multipliers)[k:], 0.0)
