"""Unit tests for :mod:`slsqp_jax.sqpdax.linalg.lsei`.

Convention under test: ``min ½‖w‖²  s.t.  G_E w = h_E,  G_I w ≤ h_I``, with
``g_t = Gᵀ`` (columns ordered ``[E, I]``), stationarity ``w + Gᵀλ = 0`` and
``λ_I ≥ 0``.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.linalg import (
    LDP_RESULTS,
    ElasticLSEIResult,
    LSEIResult,
    elastic_delta_column,
    elastic_least_distance_lsei,
    elastic_weight_scale,
    least_distance_lsei,
)

jax.config.update("jax_enable_x64", True)

EQ_TOL = 1e-8
REF_TOL = 1e-7
KKT_TOL = 1e-9

# (m_E, m_I, dup): ``dup`` makes the last equality row a copy of the first.
CASES = [
    (0, 0, False),
    (2, 0, False),
    (0, 3, False),
    (2, 3, False),
    (3, 2, True),
]
CASE_IDS = ["0-0", "2-0", "0-3", "2-3", "3dup-2"]
DIMS = [4, 10]
SEEDS = [0, 1, 2, 3]


def make_problem(n, m_e, m_i, dup, seed, *, row_scale=None, h_scale=1.0, violate=0.0):
    """Feasible random problem returned as ``(g_t, h)`` in NumPy.

    ``w0`` is feasible by construction; about half of the inequalities pass
    exactly through ``w0`` and the rest have positive slack, so the
    minimum-norm point (nearer the origin) typically activates some of them.
    ``row_scale`` (shape ``(m,)``) rescales rows of ``G`` and ``h`` together
    (the feasible set does not change), ``h_scale`` multiplies ``h``.
    ``violate > 0`` moves the first inequality so that ``w0`` violates it by
    exactly ``violate`` (in the units of ``w``; not feasible by construction).
    """
    rng = np.random.default_rng(10_000 * seed + 100 * n + 10 * m_e + m_i)
    g = rng.standard_normal((m_e + m_i, n))
    if dup:
        g[m_e - 1] = g[0]
    w0 = 3.0 * rng.standard_normal(n)
    h = g @ w0
    h[m_e:] += np.abs(rng.standard_normal(m_i)) * (rng.random(m_i) < 0.5)
    if violate > 0.0:
        h[m_e] = g[m_e] @ w0 - violate * np.linalg.norm(g[m_e])
    if row_scale is not None:
        g, h = g * row_scale[:, None], h * row_scale
    return g.T.copy(), h_scale * h


def dense_reference(g_t, h, m_eq):
    """Dense KKT enumeration over the inequality working sets (tiny sizes only).

    Returns ``(w, lam_eq, lam_ineq)``: for every subset ``S`` of inequalities
    solve ``[[I, G_aᵀ], [G_a, 0]] [w; λ] = [0; h_a]`` (minimum-norm in the
    multipliers when dependent) and keep the first one that is primal and dual
    feasible. Reference only: never used by the code under test.
    """
    g = g_t.T
    m = g.shape[0]
    m_i = m - m_eq
    n = g.shape[1]
    for size in range(m_i + 1):
        for subset in itertools.combinations(range(m_i), size):
            rows = list(range(m_eq)) + [m_eq + j for j in subset]
            ga, ha = g[rows], h[rows]
            k = len(rows)
            kkt = np.block([[np.eye(n), ga.T], [ga, np.zeros((k, k))]])
            sol = np.linalg.lstsq(kkt, np.concatenate([np.zeros(n), ha]), rcond=None)[0]
            w, lam = sol[:n], sol[n:]
            if np.linalg.norm(kkt @ sol - np.concatenate([np.zeros(n), ha])) > 1e-9:
                continue
            slack = g[m_eq:] @ w - h[m_eq:]
            lam_i = np.zeros(m_i)
            lam_i[list(subset)] = lam[m_eq:]
            if np.all(slack <= 1e-9) and np.all(lam_i >= -1e-9):
                return w, lam[:m_eq], lam_i
    raise AssertionError("no KKT point found (infeasible reference problem)")


def assert_lsei_kkt(res: LSEIResult, g_t, h, m_eq, tol=KKT_TOL):
    """KKT certificate of ``res``, measured relative to the solution scale.

    ``scale = ‖w‖ + max_i |h_i| / ‖g_i‖`` (the size of the data in the
    coordinates of ``w``); every multiplier enters as ``‖g_i‖ λ_i``, which has
    the units of ``w``.
    """
    g = np.asarray(g_t).T
    h = np.asarray(h)
    w = np.asarray(res.w)
    lam = np.concatenate([np.asarray(res.lam_eq), np.asarray(res.lam_ineq)])
    row_norm = np.where(np.linalg.norm(g, axis=1) > 0, np.linalg.norm(g, axis=1), 1.0)
    scale = np.linalg.norm(w) + np.max(np.abs(h) / row_norm, initial=0.0)
    scaled_lam = row_norm * np.abs(lam)
    resid = g @ w - h
    # stationarity w + Gᵀλ = 0
    assert np.linalg.norm(w + g.T @ lam) <= tol * (scale + scaled_lam.sum()) + 1e-300
    # primal feasibility, row-normalised
    assert np.all(np.abs(resid[:m_eq]) / row_norm[:m_eq] <= tol * scale + 1e-300)
    assert np.all(resid[m_eq:] / row_norm[m_eq:] <= tol * scale + 1e-300)
    # dual feasibility and complementarity
    lam_i = lam[m_eq:]
    assert np.all(row_norm[m_eq:] * lam_i >= -tol * scale)
    assert np.all(
        np.abs(lam_i * resid[m_eq:]) <= tol * scale * scaled_lam[m_eq:] + 1e-300
    )


def run(g_t, h, m_eq, eq_tol=EQ_TOL, **kwargs):
    return least_distance_lsei(jnp.asarray(g_t), jnp.asarray(h), m_eq, eq_tol, **kwargs)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("m_e, m_i, dup", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("n", DIMS)
def test_matches_dense_reference_and_kkt(n, m_e, m_i, dup, seed):
    """Primal and dual solution agree with the KKT enumeration; KKT holds."""
    g_t, h = make_problem(n, m_e, m_i, dup, seed)
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert res.w.shape == (n,)
    assert res.lam_eq.shape == (m_e,)
    assert res.lam_ineq.shape == (m_i,)
    w_ref, lam_e_ref, lam_i_ref = dense_reference(g_t, h, m_e)
    scale = max(1.0, np.linalg.norm(w_ref))
    np.testing.assert_allclose(np.asarray(res.w), w_ref, atol=REF_TOL * scale)
    np.testing.assert_allclose(
        np.asarray(res.lam_ineq), lam_i_ref, atol=REF_TOL * scale
    )
    # With dependent equality rows the multipliers are not unique, but the
    # combination G_Eᵀ λ_E is, and both sides return the minimum-norm λ_E.
    np.testing.assert_allclose(
        g_t[:, :m_e] @ np.asarray(res.lam_eq),
        g_t[:, :m_e] @ lam_e_ref,
        atol=REF_TOL * scale,
    )
    np.testing.assert_allclose(np.asarray(res.lam_eq), lam_e_ref, atol=REF_TOL * scale)
    assert_lsei_kkt(res, g_t, h, m_e)


@pytest.mark.parametrize("m_e, m_i, dup", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("n", DIMS)
def test_problem_generator_activates_inequalities(n, m_e, m_i, dup):
    """Across seeds some inequality is active at the solution (``m_I > 0``)."""
    n_active = 0
    for seed in range(8):
        g_t, h = make_problem(n, m_e, m_i, dup, seed)
        res = run(g_t, h, m_e)
        slack = g_t[:, m_e:].T @ np.asarray(res.w) - h[m_e:]
        n_active += int(np.sum(np.abs(slack) < 1e-9))
    assert n_active >= min(1, m_i)


def assert_zero_outputs(res: LSEIResult):
    """A non-feasible result carries no ``w`` and no multipliers."""
    np.testing.assert_array_equal(np.asarray(res.w), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_eq), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_ineq), 0.0)


@pytest.mark.parametrize("m_i", [0, 2])
@pytest.mark.parametrize("n", DIMS)
@pytest.mark.parametrize("perturb", [0.0, 1e-12], ids=["consistent", "noise"])
def test_dependent_consistent_equalities(n, m_i, perturb):
    """A duplicated equality row with (numerically) equal data is solved exactly."""
    g_t, h = make_problem(n, 3, m_i, True, seed=0)
    h = h.copy()
    h[2] += perturb
    res = run(g_t, h, 3)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert_lsei_kkt(res, g_t, h, 3, tol=1e-8)
    # exact: the equality residual of both copies is rounding level
    resid = g_t[:, :3].T @ np.asarray(res.w) - h[:3]
    assert np.max(np.abs(resid[:2])) <= 1e-9 * (1.0 + np.linalg.norm(h))


@pytest.mark.parametrize("m_i", [0, 2])
@pytest.mark.parametrize("n", DIMS)
@pytest.mark.parametrize("perturb", [1.0, 1e-3], ids=["large", "small"])
def test_dependent_inconsistent_equalities(n, m_i, perturb):
    """A duplicated equality row with different data is flagged, outputs zero."""
    g_t, h = make_problem(n, 3, m_i, True, seed=0)
    h = h.copy()
    h[2] += perturb
    res = run(g_t, h, 3)
    assert bool(res.status == LDP_RESULTS.inconsistent_equalities)
    assert_zero_outputs(res)


@pytest.mark.parametrize("n", DIMS)
def test_inconsistent_equalities_take_precedence_over_infeasible(n):
    """Both defects present: ``inconsistent_equalities`` is reported."""
    g_t, h = make_problem(n, 3, 0, True, seed=0)
    h = h.copy()
    h[2] += 1.0
    a = np.zeros((n, 1))
    a[0, 0] = 1.0
    g_t = np.concatenate([g_t, a, -a], axis=1)  # a·w ≤ -1 and -a·w ≤ -1
    h = np.concatenate([h, [-1.0, -1.0]])
    res = run(g_t, h, 3)
    assert bool(res.status == LDP_RESULTS.inconsistent_equalities)
    assert_zero_outputs(res)


@pytest.mark.parametrize("m_e", [0, 2])
@pytest.mark.parametrize("n", DIMS)
def test_infeasible_inequalities(n, m_e):
    """Opposite half-spaces ``a·w ≤ -1`` and ``-a·w ≤ -1`` are infeasible."""
    g_t, h = make_problem(n, m_e, 0, False, seed=0)
    a = np.zeros((n, 1))
    a[0, 0] = 1.0
    g_t = np.concatenate([g_t, a, -a], axis=1)
    h = np.concatenate([h, [-1.0, -1.0]])
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.infeasible)
    np.testing.assert_array_equal(np.asarray(res.w), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_eq), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_ineq), 0.0)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda g_t, h: (g_t.at[0, 0].set(jnp.nan), h),
        lambda g_t, h: (g_t, h.at[0].set(jnp.nan)),
        lambda g_t, h: (g_t, h.at[-1].set(jnp.inf)),
    ],
    ids=["nan-g", "nan-h-eq", "inf-h-ineq"],
)
def test_nonfinite(corrupt):
    """Non-finite data gives ``nonfinite`` and zero outputs."""
    g_t, h = make_problem(5, 2, 3, False, seed=0)
    res = least_distance_lsei(*corrupt(jnp.asarray(g_t), jnp.asarray(h)), 2, EQ_TOL)
    assert bool(res.status == LDP_RESULTS.nonfinite)
    np.testing.assert_array_equal(np.asarray(res.w), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_eq), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_ineq), 0.0)


@pytest.mark.parametrize(
    "max_iter, expected",
    [(0, LDP_RESULTS.max_iter_reached), (None, LDP_RESULTS.feasible)],
)
def test_ldp_kwargs_are_forwarded(max_iter, expected):
    """``max_iter`` reaches the LDP and a non-feasible status zeroes ``w``."""
    n, m_i = 6, 6
    g_t = np.eye(n)[:, :m_i]
    h = -np.ones(m_i)  # w_i ≤ -1 for all i: active, needs NNLS additions
    res = run(g_t, h, 0, max_iter=max_iter)
    assert bool(res.status == expected)
    np.testing.assert_allclose(
        np.asarray(res.w), (expected == LDP_RESULTS.feasible) * -np.ones(n)
    )


@pytest.mark.parametrize("m_e, m_i, dup", CASES, ids=CASE_IDS)
def test_jit_matches_eager(m_e, m_i, dup):
    """The solver is jit-able (``m_eq`` static) and gives the eager result."""
    g_t, h = make_problem(6, m_e, m_i, dup, seed=1)
    eager = run(g_t, h, m_e)
    jitted = jax.jit(least_distance_lsei, static_argnums=(2, 3))(
        jnp.asarray(g_t), jnp.asarray(h), m_e, EQ_TOL
    )
    assert bool(jitted.status == eager.status)
    np.testing.assert_allclose(np.asarray(jitted.w), np.asarray(eager.w), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jitted.lam_eq), np.asarray(eager.lam_eq), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(jitted.lam_ineq), np.asarray(eager.lam_ineq), atol=1e-12
    )


@pytest.mark.parametrize("n, m_e, m_i", [(5, 1, 0), (5, 0, 2), (3, 0, 0)])
def test_zero_constraints_and_shapes(n, m_e, m_i):
    """Zero columns or zero rows of one kind give empty multiplier vectors."""
    m = m_e + m_i
    g_t = np.ones((n, m))
    h = -np.ones(m)
    res = run(g_t, h, m_e)
    assert res.lam_eq.shape == (m_e,)
    assert res.lam_ineq.shape == (m_i,)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert_lsei_kkt(res, g_t, h, m_e)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype_follows_input(dtype):
    """Outputs take the input float dtype (at least float32)."""
    g_t, h = make_problem(5, 2, 2, False, seed=0)
    res = least_distance_lsei(jnp.asarray(g_t, dtype), jnp.asarray(h, dtype), 2, 1e-4)
    assert res.w.dtype == dtype
    assert res.lam_eq.dtype == dtype
    assert res.lam_ineq.dtype == dtype
    assert bool(res.status == LDP_RESULTS.feasible)


@pytest.mark.parametrize("m_eq", [-1, 4])
def test_invalid_arguments(m_eq):
    """A static ``m_eq`` outside ``[0, m]`` raises."""
    with pytest.raises(ValueError, match="m_eq"):
        least_distance_lsei(jnp.ones((5, 3)), jnp.ones(3), m_eq, EQ_TOL)
    with pytest.raises(ValueError, match="h must have shape"):
        least_distance_lsei(jnp.ones((5, 3)), jnp.ones(2), 1, EQ_TOL)


# --- scale battery -----------------------------------------------------------

SCALE_CASES = [(10, 2, 5, False), (10, 3, 4, True), (4, 2, 3, False)]
SCALE_MAGNITUDES = [1e-3, 1.0, 1e4, 1e8, 1e12]
GLOBAL_G_FACTORS = [1e-6, 1.0, 1e6]


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("g_factor", GLOBAL_G_FACTORS)
@pytest.mark.parametrize("s", SCALE_MAGNITUDES)
@pytest.mark.parametrize("n, m_e, m_i, dup", SCALE_CASES)
def test_scale_battery(n, m_e, m_i, dup, s, g_factor, seed):
    """KKT holds relative to the solution scale for ``h`` over 15 decades.

    The problem is homogeneous in ``h``: scaling ``h`` by ``s`` scales ``w``
    and ``λ`` by ``s``. Rows are also scaled unevenly (``10^U(-1, 1)``) and
    all of ``G`` by a global factor, which leave ``w`` unchanged and divide
    ``λ_i`` by the row scale. ``G_Eᵀ λ_E`` is unique even for dependent
    equality rows, and is checked as well.
    """
    rng = np.random.default_rng(seed + 31 * n + m_i)
    row_scale = g_factor * 10.0 ** rng.uniform(-1.0, 1.0, size=m_e + m_i)
    if dup:
        row_scale[m_e - 1] = row_scale[0]  # keep the duplicate exact
    g_t0, h0 = make_problem(n, m_e, m_i, dup, seed)
    g_t, h = make_problem(n, m_e, m_i, dup, seed, row_scale=row_scale, h_scale=s)
    base = run(g_t0, h0, m_e)
    res = run(g_t, h, m_e)
    assert bool(base.status == LDP_RESULTS.feasible)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert_lsei_kkt(res, g_t, h, m_e)
    w, w0 = np.asarray(res.w), np.asarray(base.w)
    np.testing.assert_allclose(w, s * w0, rtol=1e-7, atol=1e-7 * s * np.linalg.norm(w0))
    lam_e_vec = g_t[:, :m_e] @ np.asarray(res.lam_eq)
    lam_e_vec0 = g_t0[:, :m_e] @ np.asarray(base.lam_eq)
    np.testing.assert_allclose(
        lam_e_vec,
        s * lam_e_vec0,
        rtol=1e-6,
        atol=1e-6 * s * max(1.0, np.abs(lam_e_vec0).max()),
    )
    lam_i = np.asarray(res.lam_ineq) * row_scale[m_e:]
    lam_i0 = np.asarray(base.lam_ineq)
    np.testing.assert_allclose(
        lam_i,
        s * lam_i0,
        rtol=1e-6,
        atol=1e-6 * s * max(1.0, np.abs(lam_i0).max(initial=0.0)),
    )


# --- equality decisions independent of row scaling ----------------------------


def row_scale_decades(m, seed, decades):
    """Row scales ``10^e`` with ``e`` drawn from ``±decades`` and ``0``."""
    rng = np.random.default_rng(seed + 977 * m)
    return 10.0 ** rng.choice([-decades, 0.0, decades], size=m)


DETERMINED_DIMS = [2, 3, 5]
DETERMINED_EXTRA = [0, 1]  # m_E = n + extra: the equalities determine w
DETERMINED_DECADES = [0.0, 4.0]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("decades", DETERMINED_DECADES)
@pytest.mark.parametrize("extra", DETERMINED_EXTRA)
@pytest.mark.parametrize("n", DETERMINED_DIMS)
def test_determined_equalities_with_active_inequalities(n, extra, decades, seed):
    """``m_E ≥ n``: the equalities fix ``w = w0``; active inequalities stay feasible.

    Half of the inequalities pass exactly through ``w0``. Their rows lie in
    the span of the equality rows, so the reduced LDP rows are rounding noise
    that must not be amplified into a spurious infeasibility.
    """
    m_e, m_i = n + extra, 3
    row_scale = row_scale_decades(m_e + m_i, seed, decades)
    g_t, h = make_problem(n, m_e, m_i, False, seed, row_scale=row_scale)
    w0 = np.linalg.lstsq(g_t[:, :m_e].T, h[:m_e], rcond=None)[0]
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(
        np.asarray(res.w), w0, atol=1e-7 * (1 + np.linalg.norm(w0))
    )
    assert_lsei_kkt(res, g_t, h, m_e)


@pytest.mark.parametrize("violate", [1e-6, 1.0])
@pytest.mark.parametrize("decades", DETERMINED_DECADES)
@pytest.mark.parametrize("extra", DETERMINED_EXTRA)
@pytest.mark.parametrize("n", DETERMINED_DIMS)
def test_determined_point_violating_an_inequality_is_infeasible(
    n, extra, decades, violate
):
    """A genuine violation at the determined point is not hidden by noise rows."""
    m_e, m_i = n + extra, 3
    row_scale = row_scale_decades(m_e + m_i, 0, decades)
    g_t, h = make_problem(n, m_e, m_i, False, 0, row_scale=row_scale, violate=violate)
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.infeasible)
    assert_zero_outputs(res)


def own_row_residual(g_t, h, w, rows):
    """``|g_i·w - h_i| / (‖g_i‖ ‖w‖ + |h_i|)`` for the given rows."""
    g = g_t.T[rows]
    return np.abs(g @ w - h[rows]) / (
        np.linalg.norm(g, axis=1) * np.linalg.norm(w) + np.abs(h[rows])
    )


def near_dependent_pair(n, ratio, angle, seed):
    """Equality rows ``(R a, a + θ b)`` with ``a ⟂ b`` unit vectors, consistent ``h``."""
    rng = np.random.default_rng(seed)
    a, b = np.linalg.qr(rng.standard_normal((n, 2)))[0].T
    g = np.stack([ratio * a, a + angle * b])
    return g, g @ (3.0 * rng.standard_normal(n))


@pytest.mark.parametrize("m_i", [0, 2])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "ratio, angle", [(1.0, 1e-7), (1e6, 1e-8), (1e8, 1e-6), (1e-6, 1e-8)]
)
def test_near_dependent_equalities_are_solved_in_their_own_scale(
    ratio, angle, seed, m_i
):
    """Near-dependent rows of very different norms: each is met in its own scale.

    The rank cut and consistency test must not be dominated by the big row, nor
    treat the angle-``θ`` pair as dependent.
    """
    n = 6
    g_e, h_e = near_dependent_pair(n, ratio, angle, seed)
    g_i, h_i = (
        make_problem(n, 0, m_i, False, seed)[0].T,
        make_problem(n, 0, m_i, False, seed)[1],
    )
    g_t = np.concatenate([g_e, g_i]).T.copy()
    h = np.concatenate([h_e, h_i])
    res = run(g_t, h, 2)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert np.all(own_row_residual(g_t, h, np.asarray(res.w), [0, 1]) <= 1e-12)
    assert_lsei_kkt(res, g_t, h, 2)


@pytest.mark.parametrize("m_i", [0, 2])
@pytest.mark.parametrize("alpha", [1e-6, 1e6])
@pytest.mark.parametrize(
    "perturb, expected",
    [(0.0, LDP_RESULTS.feasible), (0.1, LDP_RESULTS.inconsistent_equalities)],
    ids=["consistent", "inconsistent-10pct"],
)
def test_dependent_copy_in_another_scale(perturb, expected, alpha, m_i):
    """A copy ``α a`` of a row with data ``α h (1 + perturb)``: judged in its own scale.

    ``eq_tol = 1e-6``: the 10 % inconsistency of the ``α = 1e-6`` copy is then
    about ``1e-6`` in absolute terms, below the tolerance when it is (wrongly)
    measured in the units of the big row (``eq_tol · ‖h_E‖``).
    """
    n = 6
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n)
    h0 = float(a @ (3.0 * rng.standard_normal(n)))
    g_i, h_i = (
        make_problem(n, 0, m_i, False, 3)[0].T,
        make_problem(n, 0, m_i, False, 3)[1],
    )
    g_t = np.concatenate([np.stack([a, alpha * a]), g_i]).T.copy()
    h = np.concatenate([[h0, alpha * h0 * (1.0 + perturb)], h_i])
    res = run(g_t, h, 2, eq_tol=1e-6)
    assert bool(res.status == expected)
    assert bool(np.any(np.asarray(res.w) != 0.0)) == (expected == LDP_RESULTS.feasible)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("slack", [0.0, 0.5], ids=["active", "slack"])
@pytest.mark.parametrize("n", [6, 10])
def test_inequality_in_span_of_equalities_with_free_directions(n, slack, seed):
    """An inequality row that combines the equality rows leaves ``w`` unchanged.

    ``m_E < n`` so the LDP has genuine unknowns. The first inequality row is
    ``a g_0 + b g_1`` with a right-hand side ``a h_0 + b h_1 + slack``: it is
    constant on the equality manifold, so it can only be active or inactive
    as a whole and must not cut the other constraints (its reduced row is
    rounding noise). ``w`` is compared with the dense reference; the
    multipliers of the spanning row are not unique and are not compared.
    """
    m_e, m_i = 2, 3
    g_t, h = make_problem(n, m_e, m_i, False, seed)
    rng = np.random.default_rng(seed)
    coeff = rng.standard_normal(2)
    g_t, h = g_t.copy(), h.copy()
    g_t[:, m_e] = g_t[:, :2] @ coeff
    h[m_e] = h[:2] @ coeff + slack
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.feasible)
    w_ref = dense_reference(g_t, h, m_e)[0]
    np.testing.assert_allclose(
        np.asarray(res.w), w_ref, atol=REF_TOL * max(1.0, np.linalg.norm(w_ref))
    )
    assert_lsei_kkt(res, g_t, h, m_e)


# --------------------------------------------------------------------------
# Degenerate polytopes (empty interior) and zero rows (C11b)
# --------------------------------------------------------------------------


def singleton_vertex_problem(m_e, p, extra, cond, spread, seed):
    """Problem whose feasible set is the single point ``w0``.

    ``m_E`` equalities leave a ``p``-dimensional null space; ``p + 1``
    inequality rows ``dirs @ Nᵀ`` (plus row-space components) positively span
    it (``dirs`` has a positive linear dependency), so ``G_I (w - w0) ≤ 0``
    forces ``w = w0``. All rows are passed through the ill-conditioned map
    ``C⁻¹`` (``C`` symmetric with ``cond(C) = cond``), the way the reduced
    system of a stiff reduced-Hessian model looks, and then rescaled by
    ``10^U(-spread, spread)`` row by row. ``extra`` further inequalities have
    positive slack at ``w0``. Dense NumPy: reference / generator only.
    """
    rng = np.random.default_rng(1000 * seed + 100 * m_e + 10 * p + extra)
    n = m_e + p
    a_e = rng.standard_normal((m_e, n))
    null = np.linalg.svd(a_e)[2][m_e:].T if m_e else np.eye(n)
    dirs = rng.standard_normal((p + 1, p))
    dirs[-1] = -(rng.uniform(0.5, 2.0, p) @ dirs[:-1])
    a_i = dirs @ null.T + rng.standard_normal((p + 1, m_e)) @ a_e
    a_x = rng.standard_normal((extra, n))
    a = np.concatenate([a_e, a_i, a_x])
    u = np.linalg.qr(rng.standard_normal((n, n)))[0]
    c_inv = u @ np.diag(np.logspace(0.0, -np.log10(cond), n)) @ u.T
    g = a @ c_inv
    w0 = rng.standard_normal(n) * 10.0 ** rng.uniform(-2.0, 3.0)
    h = g @ w0
    h[m_e + p + 1 :] += np.linalg.norm(g[m_e + p + 1 :], axis=1) * rng.uniform(
        0.1, 1.0, extra
    )
    row_scale = 10.0 ** rng.uniform(-spread, spread, len(g))
    return (g * row_scale[:, None]).T.copy(), h * row_scale, w0


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("extra", [0, 2])
@pytest.mark.parametrize("spread", [0.0, 2.0])
@pytest.mark.parametrize("cond", [1e4, 1e5])
@pytest.mark.parametrize("m_e, p", [(0, 1), (0, 3), (2, 1), (2, 3)])
def test_singleton_vertex_is_found(m_e, p, cond, spread, extra, seed):
    """``p + 1`` positively spanning active rows in a ``p``-dim null space.

    The feasible set is one point, so the minimum-norm solution is ``w0``;
    NNLS sees dependent columns, which a plain Lawson-Hanson turns into a
    false ``infeasible`` (or a garbage ``w``) once the rows are
    ill-conditioned.
    """
    g_t, h, w0 = singleton_vertex_problem(m_e, p, extra, cond, spread, seed)
    res = run(g_t, h, m_e, eq_tol=1.5e-8)
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(
        np.asarray(res.w), w0, rtol=0.0, atol=1e-8 * np.linalg.norm(w0)
    )
    assert_lsei_kkt(res, g_t, h, m_e, tol=1e-7)


def zero_row_problem(n_zero, n_other, rhs, seed):
    """``n_zero`` exactly zero ``Gᵀ`` columns with right-hand side ``rhs``."""
    rng = np.random.default_rng(seed + 10 * n_zero + n_other)
    n = 5
    g = np.concatenate([np.zeros((n_zero, n)), rng.standard_normal((n_other, n))])
    h = g @ rng.standard_normal(n)
    h[:n_zero] = rhs
    h[n_zero:] += 0.3
    return g, h


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("rhs", [0.0, -1e-16, -1e-12, -1e-10])
@pytest.mark.parametrize("n_other", [0, 3])
@pytest.mark.parametrize("n_zero", [1, 2])
def test_zero_rows_without_equalities(n_zero, n_other, rhs, seed):
    """Inequalities supported only on pinned variables are exactly zero rows.

    Without equalities (``m_E = 0``) such a row (``0 ≤ rhs``) must not be
    amplified by the LDP normalisation into a random constraint: it is
    satisfied up to the ``eq_tol`` floor (``rhs ≥ -eq_tol``) and the other
    rows are solved as if the zero rows were absent.
    """
    g, h = zero_row_problem(n_zero, n_other, rhs, seed)
    res = run(g.T.copy(), h, 0)
    assert bool(res.status == LDP_RESULTS.feasible)
    w_ref, _, lam_ref = dense_reference(g[n_zero:].T.copy(), h[n_zero:], 0)
    np.testing.assert_allclose(np.asarray(res.w), w_ref, atol=REF_TOL)
    np.testing.assert_allclose(np.asarray(res.lam_ineq)[n_zero:], lam_ref, atol=REF_TOL)
    np.testing.assert_array_equal(np.asarray(res.lam_ineq)[:n_zero], 0.0)


@pytest.mark.parametrize("n_other", [0, 3])
@pytest.mark.parametrize("n_zero", [1, 2])
def test_zero_row_violation_is_infeasible(n_zero, n_other):
    """A zero row with a genuinely violated right-hand side stays ``infeasible``."""
    g, h = zero_row_problem(n_zero, n_other, -1e-3, 0)
    res = run(g.T.copy(), h, 0)
    assert bool(res.status == LDP_RESULTS.infeasible)
    assert_zero_outputs(res)


# --------------------------------------------------------------------------
# Farkas certificates of infeasibility (C11c)
#
# Convention: y = [cert_eq; cert_ineq] with y_I >= 0, Gᵀ y = 0, hᵀ y < 0,
# ‖y‖_∞ = 1 (original, un-equilibrated units).
# --------------------------------------------------------------------------


def _e0(n):
    a = np.zeros((n, 1))
    a[0, 0] = 1.0
    return a


def _contradictory_inequalities(n, m_e, seed):
    """Random feasible equalities plus ``a·w ≤ -1`` and ``-a·w ≤ -1``."""
    g_t, h = make_problem(n, m_e, 0, False, seed)
    a = _e0(n)
    return np.concatenate([g_t, a, -a], axis=1), np.concatenate([h, [-1.0, -1.0]]), m_e


def _inconsistent_dependent_equalities(n, m_e, seed):
    """A duplicated equality row with a different right-hand side (+ ineqs)."""
    g_t, h = make_problem(n, 3, 2, True, seed)
    h = h.copy()
    h[2] += 1.0
    return g_t, h, 3


def _inconsistent_independent_equalities(n, m_e, seed):
    """``n + 1`` generic equality rows in ``R^n``: inconsistent."""
    del m_e
    rng = np.random.default_rng(seed + n)
    g_t = rng.standard_normal((n, n + 1))
    return g_t, rng.standard_normal(n + 1), n + 1


def _infeasible_after_elimination(n, m_e, seed):
    """``w_0 + w_1 = 0`` with ``w_0 ≤ -1`` and ``w_1 ≤ -1``: a joint defect.

    Neither inequality is infeasible alone; only after the equality is
    eliminated do they contradict each other.
    """
    del m_e
    rng = np.random.default_rng(seed + n)
    extra = rng.standard_normal((n, 2))
    e0, e1 = np.zeros((n, 1)), np.zeros((n, 1))
    e0[0, 0], e1[1, 0] = 1.0, 1.0
    g_t = np.concatenate([e0 + e1, e0, e1, extra], axis=1)
    h = np.concatenate([[0.0, -1.0, -1.0], 5.0 * rng.random(2)])
    return g_t, h, 1


def _zero_row_violation(n, m_e, seed):
    """An inequality row in the span of the equality rows with a violated rhs."""
    del m_e
    rng = np.random.default_rng(seed + n)
    row = rng.standard_normal((n, 1))
    g_t = np.concatenate([row, 2.0 * row, rng.standard_normal((n, 1))], axis=1)
    h = np.array([1.0, 2.0 - 0.5, 0.3])  # row·w = 1 forces 2 row·w = 2 > 1.5
    return g_t, h, 1


INFEASIBLE, INCONSISTENT = LDP_RESULTS.infeasible, LDP_RESULTS.inconsistent_equalities
INFEASIBLE_PROBLEMS = {
    "contradictory-ineq": (_contradictory_inequalities, INFEASIBLE),
    "inconsistent-dependent-eq": (_inconsistent_dependent_equalities, INCONSISTENT),
    "inconsistent-independent-eq": (
        _inconsistent_independent_equalities,
        INCONSISTENT,
    ),
    "after-elimination": (_infeasible_after_elimination, INFEASIBLE),
    "ineq-in-equality-span": (_zero_row_violation, INFEASIBLE),
}


def assert_lsei_certificate(res: LSEIResult, g_t, h, m_eq, status):
    """``y_I ≥ 0``, ``Gᵀ y ≈ 0``, ``hᵀ y < 0``, ``‖y‖_∞ = 1``; zero outputs."""
    assert bool(res.status == status)
    assert bool(res.has_certificate)
    y_e, y_i = np.asarray(res.cert_eq), np.asarray(res.cert_ineq)
    assert y_e.shape == (m_eq,) and y_i.shape == (g_t.shape[1] - m_eq,)
    y = np.concatenate([y_e, y_i])
    assert np.all(np.isfinite(y))
    assert np.all(y_i >= 0.0)
    np.testing.assert_allclose(np.max(np.abs(y)), 1.0, rtol=1e-12)
    g_norm = np.linalg.norm(g_t)
    assert np.linalg.norm(g_t @ y) <= 1e-8 * g_norm
    assert h @ y < 0.0
    # the sign is not a rounding artefact
    assert h @ y <= -1e-6 * (np.abs(h) @ np.abs(y))
    assert_zero_outputs(res)


def scale_problem(g_t, h, row, col):
    """Rescale constraint rows (columns of ``g_t``, with ``h``) and variables."""
    return col[:, None] * g_t * row[None, :], h * row


SCALINGS = {
    "none": (0.0, 0.0),
    "rows": (3.0, 0.0),
    "variables": (0.0, 3.0),
    "both": (3.0, 3.0),
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scaling", SCALINGS.values(), ids=SCALINGS.keys())
@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize("m_e", [0, 2])
@pytest.mark.parametrize(
    "make, expected",
    INFEASIBLE_PROBLEMS.values(),
    ids=INFEASIBLE_PROBLEMS.keys(),
)
def test_certificate_proves_infeasibility(make, expected, m_e, n, scaling, seed):
    """Every infeasible kind carries a valid certificate, also when the rows
    and the variables are rescaled over several decades (the equilibration
    and the normalisation are undone correctly)."""
    g_t, h, m_eq = make(n, m_e, seed)
    rng = np.random.default_rng(seed)
    row_dec, col_dec = scaling
    row = 10.0 ** rng.uniform(-row_dec, row_dec, size=g_t.shape[1])
    col = 10.0 ** rng.uniform(-col_dec, col_dec, size=n)
    g_t, h = scale_problem(g_t, h, row, col)
    res = run(g_t, h, m_eq)
    assert_lsei_certificate(res, g_t, h, m_eq, expected)


def assert_no_certificate(res: LSEIResult, m_eq: int, m: int):
    assert not bool(res.has_certificate)
    assert res.cert_eq.shape == (m_eq,) and res.cert_ineq.shape == (m - m_eq,)
    np.testing.assert_array_equal(np.asarray(res.cert_eq), 0.0)
    np.testing.assert_array_equal(np.asarray(res.cert_ineq), 0.0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", DIMS)
@pytest.mark.parametrize("m_e, m_i, dup", CASES, ids=CASE_IDS)
def test_certificate_is_zero_on_feasible_problems(n, m_e, m_i, dup, seed):
    g_t, h = make_problem(n, m_e, m_i, dup, seed)
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert_no_certificate(res, m_e, m_e + m_i)


def _post_check_infeasible():
    """A coarse NNLS tolerance: the LDP's primal check reports ``infeasible``.

    ``w_i ≤ -(1 + noise)`` style rows (as in the LDP tests' coarse problem,
    in the LSEI sign convention) that need several NNLS additions.
    """
    rng = np.random.default_rng(0)
    n, m = 5, 8
    g = rng.standard_normal((m, n))
    y0 = 2.0 * rng.standard_normal(n)
    # LDP form g x >= h  <=>  (-g) x <= -h
    h_ldp = g @ y0 + np.abs(rng.standard_normal(m)) * (rng.random(m) < 0.5) + 1.0
    return -g.T, -h_ldp, 0, {"tol": 0.5}, LDP_RESULTS.infeasible


def _budget_exhausted():
    n = 6
    return np.eye(n), -np.ones(n), 0, {"max_iter": 0}, LDP_RESULTS.max_iter_reached


def _nonfinite():
    g_t, h = make_problem(5, 2, 3, False, seed=0)
    g_t = g_t.copy()
    g_t[0, 0] = np.nan
    return g_t, h, 2, {}, LDP_RESULTS.nonfinite


@pytest.mark.parametrize(
    "make", [_post_check_infeasible, _budget_exhausted, _nonfinite]
)
def test_no_certificate_without_a_certified_exit(make):
    """The coarse-``tol`` primal-check ``infeasible`` (whose NNLS solution does
    not verify as a certificate), ``max_iter_reached`` and ``nonfinite`` carry
    no certificate."""
    g_t, h, m_eq, kwargs, expected = make()
    res = run(g_t, h, m_eq, **kwargs)
    assert bool(res.status == expected)
    assert_no_certificate(res, m_eq, g_t.shape[1])


@pytest.mark.parametrize("n, m_e, m_i", [(1, 0, 0), (4, 0, 0), (3, 0, 2), (3, 2, 0)])
def test_certificate_shapes_and_zero_size(n, m_e, m_i):
    g_t, h = make_problem(n, m_e, m_i, False, seed=0)
    res = run(g_t, h, m_e)
    assert res.cert_eq.shape == (m_e,) and res.cert_ineq.shape == (m_i,)
    assert res.has_certificate.shape == ()


@pytest.mark.parametrize(
    "make, expected", INFEASIBLE_PROBLEMS.values(), ids=INFEASIBLE_PROBLEMS.keys()
)
def test_certificate_jit_matches_eager(make, expected):
    g_t, h, m_eq = make(6, 2, 0)
    eager = run(g_t, h, m_eq)
    jitted = jax.jit(least_distance_lsei, static_argnums=(2,))(
        jnp.asarray(g_t), jnp.asarray(h), m_eq, EQ_TOL
    )
    assert bool(jitted.status == expected) and bool(jitted.has_certificate)
    np.testing.assert_allclose(
        np.asarray(jitted.cert_eq), np.asarray(eager.cert_eq), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(jitted.cert_ineq), np.asarray(eager.cert_ineq), atol=1e-10
    )


# --- elastic mode ------------------------------------------------------------
#
# The relaxed problem is ``min ½‖w‖² + ½ρδ²`` s.t. ``G_E w + δ c_E = h_E``,
# ``G_I w + δ c_I ≤ h_I``, ``0 ≤ δ ≤ 1``. In these tests ``c`` is
# ``elastic_delta_column(h, m_eq)``: ``h`` plays the role of the unshifted
# right-hand side of a QP with ``B = I``, ``g = 0`` and no pinned variable, so
# ``w = d`` and the problem is exactly the elastic QP of the plan (§2.5).

ELASTIC_RHOS = [1.0, 1e3]


def run_elastic(g_t, h, m_eq, rho, delta_col=None, eq_tol=EQ_TOL, **kwargs):
    g_t, h = jnp.asarray(g_t), jnp.asarray(h)
    if delta_col is None:
        delta_col = elastic_delta_column(h, m_eq)
    return elastic_least_distance_lsei(
        g_t, h, m_eq, jnp.asarray(delta_col), rho, eq_tol, **kwargs
    )


def assert_elastic_kkt(
    res: ElasticLSEIResult, g_t, h, c, m_eq, rho, tol=1e-8, shape_check=True
):
    """KKT of the augmented problem in the original variables ``(w, δ)``.

    Stationarity ``w + Gᵀλ = 0`` and ``ρδ + cᵀλ - μ_lo + μ_hi = 0``,
    primal feasibility of the relaxed rows and of ``0 ≤ δ ≤ 1``, ``λ_I, μ ≥ 0``
    and complementarity. The multiplier vectors of the general rows have
    exactly ``m_E`` / ``m_I`` entries: ``δ``'s rows are reported apart.
    """
    g_t, h, c = np.asarray(g_t), np.asarray(h), np.asarray(c)
    n, m = g_t.shape
    w, delta = np.asarray(res.w), float(res.delta)
    lam = np.concatenate([np.asarray(res.lam_eq), np.asarray(res.lam_ineq)])
    mu_lo, mu_hi = np.asarray(res.lam_delta)
    if shape_check:
        assert w.shape == (n,)
        assert res.lam_eq.shape == (m_eq,) and res.lam_ineq.shape == (m - m_eq,)
        assert res.lam_delta.shape == (2,)
    row_norm = np.sqrt(np.sum(g_t**2, axis=0) + c**2 / rho)
    row_norm = np.where(row_norm > 0, row_norm, 1.0)
    scale = np.linalg.norm(w) + rho * delta + np.max(np.abs(h) / row_norm, initial=0.0)
    scale_lam = np.sum(row_norm * np.abs(lam)) + mu_lo + mu_hi
    assert 0.0 <= delta <= 1.0
    resid = g_t.T @ w + delta * c - h
    assert np.linalg.norm(w + g_t @ lam) <= tol * (scale + scale_lam) + 1e-300
    assert abs(rho * delta + c @ lam - mu_lo + mu_hi) <= tol * (scale + scale_lam)
    assert np.all(np.abs(resid[:m_eq]) / row_norm[:m_eq] <= tol * scale + 1e-300)
    assert np.all(resid[m_eq:] / row_norm[m_eq:] <= tol * scale + 1e-300)
    lam_i = lam[m_eq:]
    assert np.all(row_norm[m_eq:] * lam_i >= -tol * scale)
    assert mu_lo >= -tol * scale and mu_hi >= -tol * scale
    assert np.all(np.abs(lam_i * resid[m_eq:]) <= tol * scale * scale_lam + 1e-300)
    assert abs(mu_lo * delta) <= tol * scale * scale_lam + 1e-300
    assert abs(mu_hi * (delta - 1.0)) <= tol * scale * scale_lam + 1e-300


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rho", ELASTIC_RHOS)
@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize("m_e", [0, 2])
@pytest.mark.parametrize(
    "make, expected", INFEASIBLE_PROBLEMS.values(), ids=INFEASIBLE_PROBLEMS.keys()
)
def test_elastic_solves_infeasible_problems(make, expected, m_e, n, rho, seed):
    """Every plain-infeasible system has a finite relaxed solution with
    ``0 < δ ≤ 1`` that satisfies the relaxed rows and the KKT conditions of
    the augmented problem; the extra variable never enters ``w`` or the
    general multipliers, and there is no certificate."""
    g_t, h, m_eq = make(n, m_e, seed)
    plain = run(g_t, h, m_eq)
    assert bool(plain.status == expected)
    res = run_elastic(g_t, h, m_eq, rho)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert np.all(np.isfinite(np.asarray(res.w)))
    assert 1e-8 < float(res.delta) <= 1.0
    assert not bool(res.has_certificate)
    c = np.asarray(elastic_delta_column(jnp.asarray(h), m_eq))
    assert_elastic_kkt(res, g_t, h, c, m_eq, rho)
    # the relaxed system is met strictly better than the plain one at w = 0
    # (δ = 1, w = 0 is always feasible): the objective cannot exceed it.
    assert float(res.delta) <= 1.0 and (
        0.5 * float(res.w @ res.w) + 0.5 * rho * float(res.delta) ** 2
        <= 0.5 * rho + 1e-9
    )


def _relax_nothing(lam_ineq):
    return np.zeros_like(lam_ineq)


def _relax_inactive(lam_ineq):
    return (lam_ineq == 0.0).astype(float)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("relax", [_relax_nothing, _relax_inactive])
@pytest.mark.parametrize("m_e, m_i, dup", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("n", DIMS)
def test_elastic_leaves_consistent_problems_unchanged(n, m_e, m_i, dup, relax, seed):
    """With no active relaxed row the elastic problem is the plain one:
    ``δ = 0`` and the same ``w`` and multipliers (relaxing only inequality
    rows that are inactive at the plain solution, or none)."""
    g_t, h = make_problem(n, m_e, m_i, dup, seed)
    plain = run(g_t, h, m_e)
    c = np.zeros_like(h)
    c[m_e:] = -relax(np.asarray(plain.lam_ineq))  # relaxes slack rows
    res = run_elastic(g_t, h, m_e, 10.0, delta_col=c)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert float(res.delta) <= 1e-12
    np.testing.assert_allclose(np.asarray(res.w), np.asarray(plain.w), atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(res.lam_eq), np.asarray(plain.lam_eq), atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(res.lam_ineq), np.asarray(plain.lam_ineq), atol=1e-8
    )
    assert_elastic_kkt(res, g_t, h, c, m_e, 10.0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rho", [1e2, 1e4, 1e10])
@pytest.mark.parametrize("m_e, m_i, dup", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("n", DIMS)
def test_elastic_on_consistent_problems_is_a_small_perturbation(
    n, m_e, m_i, dup, rho, seed
):
    """When a relaxed row is active the first-order gain makes ``δ`` positive
    on a consistent problem, but ``δ ≤ max(0, -cᵀλ)/ρ`` (convexity of the value in
    ``δ``, at the plain multipliers), the relaxed objective is no worse than the plain
    one, and ``w`` approaches the plain solution as ``ρ`` grows."""
    g_t, h = make_problem(n, m_e, m_i, dup, seed)
    plain = run(g_t, h, m_e)
    c = np.asarray(elastic_delta_column(jnp.asarray(h), m_e))
    lam = np.concatenate([np.asarray(plain.lam_eq), np.asarray(plain.lam_ineq)])
    res = run_elastic(g_t, h, m_e, rho)
    assert bool(res.status == LDP_RESULTS.feasible)
    delta = float(res.delta)
    gain = max(0.0, -float(c @ lam))
    assert delta <= gain / rho * (1.0 + 1e-6) + 1e-12
    w, w0 = np.asarray(res.w), np.asarray(plain.w)
    assert 0.5 * w @ w + 0.5 * rho * delta**2 <= 0.5 * w0 @ w0 * (1 + 1e-9) + 1e-12
    assert_elastic_kkt(res, g_t, h, c, m_e, rho)
    if rho >= 1e10:
        assert np.linalg.norm(w - w0) <= 1e-3 * max(1.0, np.linalg.norm(w0))


@pytest.mark.parametrize("global_factor", [1e-4, 1.0, 1e4])
@pytest.mark.parametrize("decades", [0.0, 3.0, 6.0])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize(
    "make, expected", INFEASIBLE_PROBLEMS.values(), ids=INFEASIBLE_PROBLEMS.keys()
)
def test_elastic_is_invariant_to_row_scaling(
    make, expected, n, seed, decades, global_factor
):
    """Rescaling the rows of ``(Gᵀ, h)`` together (so the feasible set and
    the relaxation are unchanged) leaves ``w`` and ``δ`` unchanged over many
    decades: the equilibration of the LSEI reduction also covers the extra
    column."""
    g_t, h, m_eq = make(n, 2, seed)
    rng = np.random.default_rng(seed)
    row = global_factor * 10.0 ** rng.uniform(-decades, decades, size=g_t.shape[1])
    ref = run_elastic(g_t, h, m_eq, 10.0)
    res = run_elastic(g_t * row[None, :], h * row, m_eq, 10.0)
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(float(res.delta), float(ref.delta), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(res.w), np.asarray(ref.w), atol=1e-6 * (1 + np.linalg.norm(ref.w))
    )
    c = np.asarray(elastic_delta_column(jnp.asarray(h * row), m_eq))
    assert_elastic_kkt(res, g_t * row[None, :], h * row, c, m_eq, 10.0, tol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize("m_e", [0, 2])
@pytest.mark.parametrize(
    "make, expected", INFEASIBLE_PROBLEMS.values(), ids=INFEASIBLE_PROBLEMS.keys()
)
def test_elastic_without_relaxed_rows_keeps_the_plain_certificate(
    make, expected, m_e, n, seed
):
    """With ``c = 0`` the elastic problem is infeasible exactly as the plain
    one; the certificate then covers ``[E, I, δ rows]``: ``y_I, y_δ ≥ 0``,
    ``Gᵀy_gen = 0``, the ``δ`` coordinate ``cᵀy_gen - y_lo + y_hi = 0``,
    ``hᵀy_gen + y_hi < 0``, ``‖y‖_∞ = 1``, zero outputs."""
    g_t, h, m_eq = make(n, m_e, seed)
    res = run_elastic(g_t, h, m_eq, 10.0, delta_col=np.zeros_like(h))
    assert bool(res.status == expected)
    assert bool(res.has_certificate)
    y_gen = np.concatenate([np.asarray(res.cert_eq), np.asarray(res.cert_ineq)])
    y_lo, y_hi = np.asarray(res.cert_delta)
    y_i = np.asarray(res.cert_ineq)
    assert y_gen.shape == (g_t.shape[1],) and res.cert_delta.shape == (2,)
    assert np.all(y_i >= 0.0) and y_lo >= 0.0 and y_hi >= 0.0
    np.testing.assert_allclose(max(np.max(np.abs(y_gen)), y_lo, y_hi), 1.0, rtol=1e-12)
    assert np.linalg.norm(g_t @ y_gen) <= 1e-8 * np.linalg.norm(g_t)
    assert abs(-y_lo + y_hi) <= 1e-8
    assert h @ y_gen + y_hi < -1e-6 * (np.abs(h) @ np.abs(y_gen) + y_hi)
    np.testing.assert_array_equal(np.asarray(res.w), 0.0)
    assert float(res.delta) == 0.0
    np.testing.assert_array_equal(np.asarray(res.lam_eq), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_ineq), 0.0)
    np.testing.assert_array_equal(np.asarray(res.lam_delta), 0.0)


@pytest.mark.parametrize(
    "make, expected", INFEASIBLE_PROBLEMS.values(), ids=INFEASIBLE_PROBLEMS.keys()
)
def test_elastic_jit_matches_eager(make, expected):
    """jit-able with ``m_eq`` and ``eq_tol`` static and a traced ``ρ``."""
    g_t, h, m_eq = make(6, 2, 0)
    c = elastic_delta_column(jnp.asarray(h), m_eq)
    eager = run_elastic(g_t, h, m_eq, 5.0)
    jitted = jax.jit(elastic_least_distance_lsei, static_argnums=(2, 5))(
        jnp.asarray(g_t), jnp.asarray(h), m_eq, c, jnp.asarray(5.0), EQ_TOL
    )
    assert bool(jitted.status == eager.status == LDP_RESULTS.feasible)
    for name in ["w", "delta", "lam_eq", "lam_ineq", "lam_delta"]:
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, name)),
            np.asarray(getattr(eager, name)),
            atol=1e-10,
        )


@pytest.mark.parametrize(
    "g_t, h, m_eq, kwargs, expected",
    [
        (*make_problem(5, 2, 3, False, seed=0), 2, {}, LDP_RESULTS.feasible),
        (np.eye(6), -np.ones(6), 0, {"max_iter": 0}, LDP_RESULTS.max_iter_reached),
        (
            np.where(
                np.arange(5)[:, None] == 0, np.nan, make_problem(5, 2, 3, False, 0)[0]
            ),
            make_problem(5, 2, 3, False, seed=0)[1],
            2,
            {},
            LDP_RESULTS.nonfinite,
        ),
    ],
    ids=["feasible", "budget", "nonfinite"],
)
def test_elastic_statuses_zero_outputs_and_shapes(g_t, h, m_eq, kwargs, expected):
    """Statuses are those of the augmented LSEI; unless feasible every output
    is zero; shapes are those of the general rows (the extra variable and its
    two rows are reported separately)."""
    n, m = g_t.shape
    res = run_elastic(g_t, h, m_eq, 10.0, **kwargs)
    assert bool(res.status == expected)
    assert res.w.shape == (n,) and res.delta.shape == ()
    assert res.lam_eq.shape == (m_eq,) and res.lam_ineq.shape == (m - m_eq,)
    assert res.lam_delta.shape == (2,)
    assert res.cert_eq.shape == (m_eq,) and res.cert_ineq.shape == (m - m_eq,)
    assert res.cert_delta.shape == (2,)
    if expected != LDP_RESULTS.feasible:
        for name in ["w", "delta", "lam_eq", "lam_ineq", "lam_delta"]:
            np.testing.assert_array_equal(np.asarray(getattr(res, name)), 0.0)


@pytest.mark.parametrize("n, m_e, m_i", [(1, 0, 0), (4, 0, 0), (3, 0, 2), (3, 2, 0)])
def test_elastic_zero_size_and_consistent_shapes(n, m_e, m_i):
    """No general rows (or rows of only one kind) work; feasible at ``δ = 0``."""
    m = m_e + m_i
    g_t = np.ones((n, m))
    h = -np.ones(m)
    res = run_elastic(g_t, h, m_e, 10.0, delta_col=np.zeros(m))
    assert bool(res.status == LDP_RESULTS.feasible)
    assert res.lam_eq.shape == (m_e,) and res.lam_ineq.shape == (m_i,)
    assert float(res.delta) == 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_elastic_dtype_follows_input(dtype):
    g_t, h = make_problem(5, 2, 2, False, seed=0)
    res = elastic_least_distance_lsei(
        jnp.asarray(g_t, dtype),
        jnp.asarray(h, dtype),
        2,
        elastic_delta_column(jnp.asarray(h, dtype), 2),
        10.0,
        1e-4,
    )
    for name in ["w", "delta", "lam_eq", "lam_ineq", "lam_delta"]:
        assert getattr(res, name).dtype == dtype
    assert bool(res.status == LDP_RESULTS.feasible)


def test_elastic_invalid_arguments():
    with pytest.raises(ValueError, match="h must have shape"):
        elastic_least_distance_lsei(
            jnp.ones((5, 3)), jnp.ones(2), 1, jnp.ones(3), 1.0, EQ_TOL
        )
    with pytest.raises(ValueError, match="delta_col"):
        elastic_least_distance_lsei(
            jnp.ones((5, 3)), jnp.ones(3), 1, jnp.ones(2), 1.0, EQ_TOL
        )
    with pytest.raises(ValueError, match="m_eq"):
        elastic_least_distance_lsei(
            jnp.ones((5, 3)), jnp.ones(3), 4, jnp.ones(3), 1.0, EQ_TOL
        )


@pytest.mark.parametrize("m_eq", [0, 1, 2])
def test_elastic_delta_column(m_eq):
    """``b`` on the equalities; ``b_I`` where negative, else ``0``."""
    b = jnp.array([2.0, -1.0, 3.0, -0.5, 0.0])
    expected = np.concatenate(
        [np.asarray(b[:m_eq]), np.minimum(np.asarray(b[m_eq:]), 0.0)]
    )
    np.testing.assert_array_equal(np.asarray(elastic_delta_column(b, m_eq)), expected)


# --- elastic weight scale and the weight rule (plan §2.5 rev 12) -------------


def _tau2_reference(g_t, h, c, m_eq):
    """Plain NumPy ``τ²`` (loop over rows), for the special cases below."""
    best = 0.0
    for j in range(g_t.shape[1]):
        norm = np.linalg.norm(g_t[:, j])
        if norm == 0.0:
            continue
        h_term = abs(h[j]) if j < m_eq else max(-h[j], 0.0)
        best = max(best, max(h_term, abs(c[j])) / norm)
    return (best if best > 0.0 else 1.0) ** 2


def _scale_w(g_t, h, c, m_eq, kappa=3.0):
    return g_t / kappa, h, c, m_eq  # w' = κ w


def _scale_rows(g_t, h, c, m_eq):
    row = 10.0 ** np.linspace(-4, 4, g_t.shape[1])
    return g_t * row[None, :], h * row, c * row, m_eq


def _add_slack_rows(g_t, h, c, m_eq):
    """Far-away slack inequality rows (``h > 0``, not relaxed)."""
    extra = np.ones((g_t.shape[0], 2))
    return (
        np.concatenate([g_t, extra], axis=1),
        np.concatenate([h, [1e9, 1e12]]),
        np.concatenate([c, [0.0, 0.0]]),
        m_eq,
    )


@pytest.mark.parametrize(
    "transform, factor",
    [(_scale_w, 9.0), (_scale_rows, 1.0), (_add_slack_rows, 1.0)],
    ids=["scale-w", "scale-rows", "slack-rows"],
)
@pytest.mark.parametrize("m_eq", [0, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_elastic_weight_scale_invariances(transform, factor, m_eq, seed):
    """Scaling ``w`` by κ multiplies ``τ²`` by κ², rescaling rows (``Gᵀ``, ``h``
    and ``c`` together) leaves it unchanged, and slack inequality rows
    do not change it."""
    rng = np.random.default_rng(seed)
    g_t = rng.standard_normal((5, 4))
    h = rng.standard_normal(4)
    h[m_eq:] = -np.abs(h[m_eq:])  # violated inequalities
    c = elastic_delta_column(jnp.asarray(h), m_eq)
    base = float(elastic_weight_scale(g_t, h, c, m_eq))
    g2, h2, c2, m2 = transform(g_t, h, np.asarray(c), m_eq)
    scaled = float(elastic_weight_scale(g2, h2, c2, m2))
    np.testing.assert_allclose(base, _tau2_reference(g_t, h, np.asarray(c), m_eq))
    np.testing.assert_allclose(scaled, factor * base, rtol=1e-12)


@pytest.mark.parametrize(
    "g_t, h, c, m_eq",
    [
        (np.eye(3), np.zeros(3), np.zeros(3), 1),  # homogeneous: τ = 1
        (
            np.array([[0.0, 1.0]]),
            np.array([-100.0, -2.0]),
            np.zeros(2),
            0,
        ),  # ‖G_0‖ = 0, violated
        (
            np.array([[2.0, 1.0]]),
            np.array([-4.0, 3.0]),
            np.zeros(2),
            1,
        ),  # |h_E|, slack I
        (np.array([[2.0, 1.0]]), np.array([4.0, -3.0]), np.array([0.0, -9.0]), 1),
        (np.array([[1.0, 1.0]]), np.array([1.0, -1.0]), np.array([5.0, 0.0]), 0),
    ],
    ids=[
        "homogeneous",
        "zero-column",
        "eq-abs-slack-ineq",
        "c-dominates",
        "c-on-slack",
    ],
)
def test_elastic_weight_scale_special_cases(g_t, h, c, m_eq):
    """Homogeneous data give 1; zero columns and slack inequalities are left
    out; ``|c_j|`` counts on every row."""
    got = float(elastic_weight_scale(g_t, h, c, m_eq))
    np.testing.assert_allclose(got, _tau2_reference(g_t, h, c, m_eq), rtol=1e-12)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda g_t, h, c: (g_t.at[0, 0].set(jnp.nan), h, c),
        lambda g_t, h, c: (g_t, h.at[0].set(jnp.inf), c),
        lambda g_t, h, c: (g_t, h, c.at[1].set(jnp.nan)),
        lambda g_t, h, c: (g_t.at[:, 0].set(0.0), h.at[0].set(jnp.nan), c),
    ],
    ids=["nan-g", "inf-h", "nan-c", "nan-h-on-zero-column"],
)
def test_elastic_weight_scale_propagates_nonfinite_and_jits(corrupt):
    g_t = jnp.arange(1.0, 7.0).reshape(3, 2)
    h, c = jnp.array([1.0, -2.0]), jnp.array([0.5, 0.25])
    assert np.isfinite(float(elastic_weight_scale(g_t, h, c, 1)))
    bad = corrupt(g_t, h, c)
    assert np.isnan(float(elastic_weight_scale(*bad, 1)))
    jitted = jax.jit(elastic_weight_scale, static_argnums=(3,))
    assert np.isnan(float(jitted(*bad, 1)))
    np.testing.assert_allclose(
        float(jitted(g_t, h, c, 1)), float(elastic_weight_scale(g_t, h, c, 1))
    )


def test_elastic_weight_scale_invalid_arguments():
    with pytest.raises(ValueError, match="delta_col"):
        elastic_weight_scale(jnp.ones((5, 3)), jnp.ones(3), jnp.ones(2), 1)
    with pytest.raises(ValueError, match="m_eq"):
        elastic_weight_scale(jnp.ones((5, 3)), jnp.ones(3), jnp.ones(3), 4)


def sqp_elastic_problem(n, s, g_scale, delta_min, seed, with_equality=True):
    """SQP-like elastic data with ``B = I``: ``G = A``, ``h = b + A g``.

    Rows ``[E, I]``: (``with_equality``) an equality with ``b = 0``; the contradictory pair
    ``a·d ≤ -s`` and ``-a·d ≤ s(1 - δ_min)`` (``δ`` must reach ``δ_min``);
    two slack rows; one far slack row (distance ``1e6 s``). ``g`` has size
    ``g_scale`` (independent of the data scale ``s``). Returns
    ``(g_t, h, c, m_eq)``.
    """
    rng = np.random.default_rng(seed)
    m_eq = int(with_equality)
    a = rng.standard_normal(n)
    a_mat = np.vstack(
        [rng.standard_normal((m_eq, n)), a, -a, rng.standard_normal((3, n))]
    )
    b = s * np.concatenate(
        [np.zeros(m_eq), [-1.0, 1.0 - delta_min, 0.5 + rng.random(), 0.7, 1e6]]
    )
    g = g_scale * rng.standard_normal(n)
    c = np.asarray(elastic_delta_column(jnp.asarray(b), m_eq))
    return a_mat.T.copy(), b + a_mat @ g, c, m_eq


def relative_row_violation(g_t, h, c, m_eq, w, delta):
    """Largest row-normalised violation of the relaxed rows, over ``‖w‖``."""
    resid = (g_t.T @ w + delta * c - h) / np.linalg.norm(g_t, axis=0)
    viol = np.maximum(resid, 0.0)
    viol[:m_eq] = np.abs(resid[:m_eq])
    return np.max(viol) / np.linalg.norm(w)


# (data scale s, scale of g): g scaled with the data, g = 0, g = O(1).
SCALE_AND_G = (
    [(s, s) for s in [1e-8, 1e-4, 1.0, 1e4]]
    + [(s, 0.0) for s in [1e-8, 1e-4, 1.0, 1e4]]
    + [(s, 1.0) for s in [1e-4, 1.0, 1e4]]
)


# (δ_min, dimensionless weight). The quadratic penalty is exact only when
# ``δ_min`` exceeds the unconstrained trade-off ``≈ 1/(1 + weight·c)``: with
# weight 1e3 that holds for δ_min = 0.3, while δ_min = 1e-3 needs 1e5 (at 1e3
# it settles up to 5e-4 above δ_min). A larger weight costs accuracy of ``w``.
DELTA_AND_WEIGHT = [(0.3, 1e3), (1e-3, 1e5)]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("delta_min, weight", DELTA_AND_WEIGHT)
@pytest.mark.parametrize("s, g_scale", SCALE_AND_G)
@pytest.mark.parametrize("with_equality", [False, True], ids=["no-eq", "eq"])
def test_elastic_scale_battery_with_the_weight_rule(
    with_equality, s, g_scale, delta_min, weight, seed
):
    """With ``ρ_e = weight · τ²`` the relaxation settles at ``δ_min`` and the
    relaxed rows hold to ``1e-8 ‖w‖``, whatever the data scale (``1e-8`` to
    ``1e4``), also for ``g = 0``, ``g = O(1)`` and with a far slack row (a
    fixed ``ρ_e`` fails here)."""
    g_t, h, c, m_eq = sqp_elastic_problem(8, s, g_scale, delta_min, seed, with_equality)
    rho = weight * elastic_weight_scale(g_t, h, c, m_eq)
    res = elastic_least_distance_lsei(g_t, h, m_eq, c, rho, EQ_TOL)
    assert bool(res.status == LDP_RESULTS.feasible)
    assert abs(float(res.delta) - delta_min) <= 1e-8
    w = np.asarray(res.w)
    assert relative_row_violation(g_t, h, c, m_eq, w, float(res.delta)) <= 1e-8


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("delta_min", [1e-3, 0.1, 0.5])
def test_elastic_interior_delta_approaches_delta_min(delta_min, seed):
    """An infeasible problem with an interior ``0 < δ_min < 1`` and ``c ≠ h``
    (``g ≠ 0``): ``δ ≥ δ_min`` always, and ``δ → δ_min`` (non-increasing error) as the
    dimensionless weight grows; KKT holds at every weight (a small weight
    over-relaxes, the quadratic penalty is not exact)."""
    g_t, h, c, m_eq = sqp_elastic_problem(8, 1.0, 0.1, delta_min, seed)
    tau2 = float(elastic_weight_scale(g_t, h, c, m_eq))
    assert not np.allclose(c, h)
    errors = []
    for weight in [1e-2, 1.0, 1e5]:
        res = elastic_least_distance_lsei(g_t, h, m_eq, c, weight * tau2, EQ_TOL)
        assert bool(res.status == LDP_RESULTS.feasible)
        assert_elastic_kkt(res, g_t, h, c, m_eq, weight * tau2)
        errors.append(float(res.delta) - delta_min)
    assert all(e >= -1e-9 for e in errors)
    assert errors[0] >= errors[1] - 1e-9 and errors[1] >= errors[2] - 1e-9
    assert errors[0] > 1e-2 and errors[2] <= 1e-8


@pytest.mark.parametrize(
    "g_t, h, c, rho, delta, w, lam, mu",
    [
        # δ ≥ 0 active: c·λ = 1 > 0, so relaxing the row would cost; δ = 0
        # and μ_lo = cᵀλ.
        ([[1.0], [0.0]], [-1.0], [1.0], 1e-3, 0.0, [-1.0, 0.0], [1.0], (1.0, 0.0)),
        # δ ≤ 1 active: unconstrained δ* = 11/(1 + ρ) > 1.
        ([[1.0]], [-11.0], [-1.0], 1.0, 1.0, [-10.0], [10.0], (0.0, 9.0)),
    ],
    ids=["mu-lo", "mu-hi"],
)
def test_elastic_multipliers_of_delta_rows(g_t, h, c, rho, delta, w, lam, mu):
    """Exact values with ``μ_lo > 0`` and with ``μ_hi > 0`` (in that order)."""
    g_t, h, c = (np.asarray(v, dtype=float) for v in (g_t, h, c))
    res = elastic_least_distance_lsei(g_t, h, 0, c, rho, EQ_TOL)
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(float(res.delta), delta, atol=1e-9)
    np.testing.assert_allclose(np.asarray(res.w), w, atol=1e-9)
    np.testing.assert_allclose(np.asarray(res.lam_delta), mu, atol=1e-8)
    np.testing.assert_allclose(np.asarray(res.lam_ineq), lam, atol=1e-8)
    assert_elastic_kkt(res, g_t, h, c, 0, rho)


@pytest.mark.parametrize(
    "sign, y_expected, delta_dual",
    [(1.0, [-1.0, 1.0], (0.0, 1.0)), (-1.0, [1.0, 1.0], (1.0, 0.0))],
    ids=["y_hi", "y_lo"],
)
@pytest.mark.parametrize("n", [4, 9])
@pytest.mark.parametrize("rho", [1e-2, 1.0, 1e4])
def test_elastic_certificate_with_delta_rows(sign, y_expected, delta_dual, n, rho):
    """Certificates in which ``δ``'s rows participate: ``e·w + δ = 3 sign``
    with ``sign · e·w ≤ 0`` forces ``δ ≥ 3`` (``y_hi``) or ``δ ≤ -3``
    (``y_lo``). Entries, the stacked identities and zero outputs."""
    e = np.zeros((n, 1))
    e[0] = 1.0
    g_t = np.concatenate([e, sign * e], axis=1)
    h = np.array([3.0 * sign, 0.0])
    c = np.array([1.0, 0.0])
    res = elastic_least_distance_lsei(g_t, h, 1, c, rho, EQ_TOL)
    assert bool(res.status == LDP_RESULTS.infeasible)
    assert bool(res.has_certificate)
    y_gen = np.concatenate([np.asarray(res.cert_eq), np.asarray(res.cert_ineq)])
    y_lo, y_hi = np.asarray(res.cert_delta)
    np.testing.assert_allclose(y_gen, y_expected, atol=1e-8)
    np.testing.assert_allclose((y_lo, y_hi), delta_dual, atol=1e-8)
    assert np.linalg.norm(g_t @ y_gen) <= 1e-8
    assert abs(c @ y_gen - y_lo + y_hi) <= 1e-8
    assert h @ y_gen + y_hi < -1.0
    np.testing.assert_array_equal(np.asarray(res.w), 0.0)
    assert float(res.delta) == 0.0


# --- rounding noise in the reduction rows (C16c) --------------------------------
#
# When the equalities determine, or nearly determine, the space the inequality
# rows live in, the projected inequality rows are rounding noise (the unpivoted
# QR, ``eps·cond(R_EE)``), and the least-distance solver must not read them as
# constraints: a wrong ``feasible`` answer violates rows by O(1) at a huge ``w``.
# The contract checked here: no ``feasible`` exit violates a row by more than
# ``1e-6·max(1, ‖ĥ‖∞)`` (hat: unit rows), and every ``infeasible`` exit is
# certified.


def assert_exit_contract(res, g_t, h, m_eq, c=None):
    """``feasible`` ⇒ rows met (relaxed by ``δ c`` if ``c``); else a certificate."""
    g_t, h = np.asarray(g_t), np.asarray(h)
    row_norm = np.linalg.norm(g_t, axis=0)
    row_norm = np.where(row_norm > 0, row_norm, 1.0)
    if bool(res.status == LDP_RESULTS.feasible):
        delta = 0.0 if c is None else float(res.delta)
        shift = 0.0 if c is None else delta * np.asarray(c)
        resid = (g_t.T @ np.asarray(res.w) + shift - h) / row_norm
        viol = np.concatenate([np.abs(resid[:m_eq]), np.maximum(resid[m_eq:], 0.0)])
        assert np.all(viol <= 1e-6 * max(1.0, np.max(np.abs(h) / row_norm)))
    else:
        assert bool(res.has_certificate)


def determined_equalities_problem(theta, violated, seed, n=6, k=3, m_i=3):
    """``k`` equalities that determine a ``k``-dim span of condition ``~1/θ``.

    The inequalities lie in the same span (so they are determined too: their
    projected rows are noise) and are all slack (feasible) or all violated at
    the point ``w0`` the equalities fix, by at least ``0.1``.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((n, k)))[0]
    coef = rng.standard_normal((k, k))
    coef[:, 1] = coef[:, 0] + theta * rng.standard_normal(k)
    w0 = basis @ rng.standard_normal(k)
    g_e = basis @ coef
    g_i = basis @ rng.standard_normal((k, m_i))
    slack = (0.1 + np.abs(rng.standard_normal(m_i))) * (-1.0 if violated else 1.0)
    g_t = np.hstack([g_e, g_i])
    return g_t, np.concatenate([g_e.T @ w0, g_i.T @ w0 + slack]), k, w0


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("violated", [False, True], ids=["satisfied", "violated"])
@pytest.mark.parametrize("theta", [1e-1, 1e-2, 1e-3, 1e-4])
def test_determined_ill_conditioned_equalities_with_inequalities_in_the_span(
    theta, violated, seed
):
    """The plain probe: noise in the rows of an in-span inequality is not a row.

    The point the equalities fix is the only candidate. The solve is
    ``feasible`` when the inequalities hold there and ``infeasible``
    (certified) when they do not; never a ``feasible`` answer with violated
    rows. Before C16c the wrong ``feasible`` rate grew from 1/50 at
    ``θ = 1e-1`` to 42/50 at ``θ = 1e-4``.
    """
    g_t, h, m_eq, _ = determined_equalities_problem(theta, violated, seed)
    res = run(g_t, h, m_eq)
    assert_exit_contract(res, g_t, h, m_eq)
    assert bool(res.status == LDP_RESULTS.feasible) == (not violated)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("theta", [1e-1, 1e-2, 1e-3, 1e-4])
def test_determined_point_is_returned_when_the_inequalities_hold(theta, seed):
    """With the in-span inequalities slack, ``w`` is the point the equalities fix."""
    g_t, h, m_eq, w0 = determined_equalities_problem(theta, False, seed)
    res = run(g_t, h, m_eq)
    assert bool(res.status == LDP_RESULTS.feasible)
    np.testing.assert_allclose(np.asarray(res.w), w0, atol=1e-6)


@pytest.mark.parametrize("m_e", [0, 1])
@pytest.mark.parametrize("e", [1e-6, 1e-8])
def test_wedge_of_near_parallel_rows_is_not_truncated(e, m_e):
    """Rows ``(±1, e)·w ≥ 1`` meet at ``w = (0, 1/e)``, ``‖w‖ = 1/e``.

    The two rows differ only by ``2e`` in the first coordinate, so ``σ_min /
    σ_max = e`` in the projected system: a genuine direction that the
    rank-revealing truncation of the reduction rows (floor ``eps^(2/3)``) must
    keep. An over-aggressive floor (e.g. ``√eps``) drops it and the problem
    turns ``infeasible`` or returns a wrong point. An optional equality on a
    third coordinate (``w_3 = 1/2``) must leave the wedge solution untouched.
    """
    n = 3
    g_t = np.zeros((n, m_e + 2))
    g_t[2, :m_e] = 1.0  # equality w_3 = 1/2 (columns are ordered [E, I])
    g_t[:2, m_e] = [-1.0, -e]  # -(w_1 + e w_2) <= -1
    g_t[:2, m_e + 1] = [1.0, -e]  # -(-w_1 + e w_2) <= -1
    h = np.concatenate([0.5 * np.ones(m_e), -np.ones(2)])
    res = run(g_t, h, m_e)
    assert bool(res.status == LDP_RESULTS.feasible)
    expected = np.array([0.0, 1.0 / e, 0.5 * m_e])
    # ``w`` is accurate relative to its own norm ``1/e``.
    np.testing.assert_allclose(
        np.asarray(res.w), expected, rtol=0, atol=1e-6 * np.linalg.norm(expected)
    )


def pinned_elastic_problem(n_free, seed, weight, n=8, m_e=2, m_i=4, spread=2.0):
    """The elastic LSEI data of a pinned QP whose relaxation is infeasible.

    ``a`` is the constraint matrix in the variables of ``w`` (``B = I``), rows
    scaled over ``spread`` decades; every variable but ``n_free`` of them is
    pinned at ``d_fix``, which shifts the right-hand side (``h = b - A d_fix``)
    but not the column of ``δ`` (``c = elastic_delta_column(b)``). The pinned
    columns of ``Gᵀ`` are zero rows. With ``n_free ≤ m_E`` the two equalities
    fix ``(w_free, δ)`` and the relaxed rows (``δ`` is then ``> 1`` for most
    seeds) are infeasible. Returns ``(g_t, h, c, rho)``.
    """
    rng = np.random.default_rng(seed)
    m = m_e + m_i
    free = rng.choice(n, n_free, replace=False)
    a = rng.standard_normal((m, n)) * 10.0 ** rng.uniform(-spread, 0, size=(m, 1))
    b = 3.0 * rng.standard_normal(m)
    b[m_e:] -= 1.0
    d_fix = 3.0 * rng.standard_normal(n)
    d_fix[free] = 0.0
    h = b - a @ d_fix
    g_t = np.zeros((n, m))
    g_t[free] = a[:, free].T
    c = np.asarray(elastic_delta_column(jnp.asarray(b), m_e))
    tau2 = float(
        elastic_weight_scale(jnp.asarray(g_t), jnp.asarray(h), jnp.asarray(c), m_e)
    )
    return g_t, h, c, weight * tau2


def delta_lp_is_feasible(g_t, h, c, m_eq):
    """LP reference: do the relaxed rows hold for some ``w`` and ``δ ∈ [0, 1]``?"""
    from scipy.optimize import linprog

    n, m = g_t.shape
    a = g_t.T
    out = linprog(
        np.zeros(n + 1),
        A_ub=np.hstack([a[m_eq:], c[m_eq:, None]]),
        b_ub=h[m_eq:],
        A_eq=np.hstack([a[:m_eq], c[:m_eq, None]]),
        b_eq=h[:m_eq],
        bounds=[(None, None)] * n + [(0.0, 1.0)],
        method="highs",
    )
    return out.status == 0


@jax.jit
def _elastic_two_equalities(g_t, h, c, rho):
    return elastic_least_distance_lsei(g_t, h, 2, c, rho, EQ_TOL)


# ``(spread, n_free, weight, seed)``: the grid at the default spread of 2
# decades, and three wide-spread reproducers that only the ``eps^(2/3)`` floor
# of the SVD truncation of the reduction rows resolves (a floor of ``rank_tol ·
# σ_max`` returns ``feasible`` with ``‖w‖ ≈ 1e16`` on each of them). The six
# ``spread = 6`` reproducers (seeds 41, 69, 290) put a tiny-norm slack row of
# the reduced LDP far from the solution: only the scale of the ``ldp``
# normalisation (rows violated at the origin) with unit-column NNLS resolves
# them (before C16d: a wrong ``feasible`` on 41 and 290, an LP mismatch on 69).
ELASTIC_FEW_FREE_CASES = (
    [
        (2.0, n_free, weight, seed)
        for n_free, weight, seed in itertools.product([1, 2], [1e3, 1e4], range(25))
    ]
    + [(2.0, 2, 1e4, 185), (4.0, 2, 1e3, 271), (4.0, 2, 1e4, 36)]
    + [
        (6.0, 2, weight, seed)
        for weight, seed in itertools.product([1e3, 1e4], [41, 69, 290])
    ]
)
ELASTIC_FEW_FREE_IDS = [
    f"spread{spread:g}-{'one' if n_free == 1 else 'two'}-free-{weight:g}-{seed}"
    for spread, n_free, weight, seed in ELASTIC_FEW_FREE_CASES
]


@pytest.mark.parametrize(
    "spread, n_free, weight, seed",
    ELASTIC_FEW_FREE_CASES,
    ids=ELASTIC_FEW_FREE_IDS,
)
def test_elastic_with_two_equalities_and_few_free_variables(
    spread, n_free, weight, seed
):
    """The two defect reproducers at the LSEI level (``m_E = 2``).

    With one free variable the two equality columns span the whole augmented
    space ``(w_free, δ)`` and the rows of ``δ`` lie in it: their projections
    are noise just above ``rank_tol`` (growing like ``eps·cond(R_EE)``). With
    two free variables the augmented data have rank 3 and the rows carry
    ``eps/pivot`` noise in the rank-deficient directions of a genuinely 1-D,
    infeasible, problem. Before C16c both gave a ``feasible`` exit with
    ``‖w‖ ≈ 1e15`` that violated the relaxed rows by O(1). The exit must now
    agree with the LP reference (``feasible`` iff some ``δ ∈ [0, 1]`` relaxes
    the rows), meet the rows, and be certified when infeasible; the jitted and
    eager solves agree.
    """
    g_t, h, c, rho = pinned_elastic_problem(n_free, seed, weight, spread=spread)
    args = (jnp.asarray(g_t), jnp.asarray(h), jnp.asarray(c), rho)
    res = _elastic_two_equalities(*args)
    assert_exit_contract(res, g_t, h, 2, c=c)
    assert bool(res.status == LDP_RESULTS.feasible) == delta_lp_is_feasible(
        g_t, h, c, 2
    )
    eager = run_elastic(g_t, h, 2, rho, delta_col=c)
    assert bool(eager.status == res.status)
    assert bool(eager.has_certificate == res.has_certificate)
