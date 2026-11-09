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
    LSEIResult,
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
