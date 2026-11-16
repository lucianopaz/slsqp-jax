"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds`.

Convention under test (SQP view): ``min gᵀd + ½dᵀBd`` s.t. ``A_E d = b_E``,
``A_I d ≤ b_I``, ``d_W = d_fix``; stationarity
``B d + g + A_Eᵀλ_E + A_Iᵀλ_I - λ_lb + λ_ub = 0``.

The dense references (``n × n`` NumPy / SciPy, tiny ``n``) live only in the
tests; the code under test never forms an ``n × n`` object.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
from slsqp_jax.sqpdax.linalg import LDP_RESULTS
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import EvaluatedProblem
from slsqp_jax.sqpdax.secant import ReducedHessianFactors
from slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds import (
    FixedBoundsQPResult,
    elastic_penalty,
    solve_fixed_bounds_qp,
)
from tests.sqpdax.subproblem.solver.conftest import (
    RandomQP,
    assert_elastic_kkt,
    delta_min_lp,
    elastic_column_np,
    index_mask,
    make_random_lrh_qp,
    walk_avals,
    with_rows,
)


@pytest.fixture(autouse=True)
def _x64():
    """Run in float64 without leaking the flag to the float32 solver tests."""
    with jax.enable_x64(True):
        yield


REF_TOL = 1e-7
KKT_TOL = 1e-8

_solve = jax.jit(solve_fixed_bounds_qp)

# (m_E, m_I): the last equality row duplicates the first when m_E == 3.
MIXES = [(0, 0), (2, 0), (0, 3), (2, 3), (3, 2)]
MIX_IDS = ["0-0", "2-0", "0-3", "2-3", "3dup-2"]
WORKING_SETS = ["none", "random", "all"]


class Case(NamedTuple):
    """A random QP, the working-set masks and the dense reference."""

    qp: RandomQP
    lower: np.ndarray
    upper: np.ndarray


def working_set(kind: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` masks of the pinned variables (disjoint)."""
    rng = np.random.default_rng(77 + 13 * seed + n)
    fixed = {
        "none": np.zeros(n, dtype=bool),
        "random": rng.random(n) < 0.5,
        "all": np.ones(n, dtype=bool),
    }[kind]
    lower = fixed & (rng.random(n) < 0.5)
    return lower, fixed & ~lower


def build_case(n, rank, mix, kind, seed=0, tight=False) -> Case:
    lower, upper = working_set(kind, n, seed)
    rng = np.random.default_rng(5 + seed)
    infinite = ~(lower | upper) & (rng.random(n) < 0.3)
    qp = make_random_lrh_qp(
        n,
        rank,
        *mix,
        seed=seed,
        dup=mix[0] == 3,
        tight=tight,
        lower=lower,
        upper=upper,
        infinite=infinite,
    )
    return Case(qp, lower, upper)


@pytest.fixture(
    params=[
        (n, rank, mix, kind)
        for n in (6, 12)
        for rank in (0, 2, 4)
        for mix in MIXES
        for kind in WORKING_SETS
    ],
    ids=lambda p: f"n{p[0]}-r{p[1]}-{MIX_IDS[MIXES.index(p[2])]}-{p[3]}",
)
def case(request) -> Case:
    """Every (n, LRH rank, constraint mix, working set) combination."""
    return build_case(*request.param)


def run(case: Case, lower=None, upper=None, **kwargs) -> FixedBoundsQPResult:
    lower = case.lower if lower is None else lower
    upper = case.upper if upper is None else upper
    return _solve(
        case.qp.lagrangian,
        case.qp.factors,
        jnp.asarray(lower),
        jnp.asarray(upper),
        **kwargs,
    )


def d_fix_of(qp: RandomQP, lower, upper) -> np.ndarray:
    return np.where(lower, qp.lb - qp.x, np.where(upper, qp.ub - qp.x, 0.0))


def dense_reference(qp: RandomQP, lower, upper):
    """Dense KKT enumeration of the equality-fixed QP (reference, tiny ``n``).

    For every subset ``S`` of inequalities solve
    ``[[B_FF, A_aᵀ], [A_a, 0]] [d_F; λ] = [-g̃; b̃_a]`` (minimum-norm in the
    multipliers when singular) and keep the first one that is primal and dual
    feasible. Returns ``(d, lam_eq, lam_ineq)``.
    """
    m_e, m_i = qp.a_eq.shape[0], qp.a_ineq.shape[0]
    free = ~(lower | upper)
    d_fix = d_fix_of(qp, lower, upper)
    g_t = np.where(free, qp.g + qp.dense_b @ d_fix, 0.0)[free]
    b_ff = qp.dense_b[np.ix_(free, free)]
    a_e = qp.a_eq[:, free]
    a_i = qp.a_ineq[:, free]
    bt_e = qp.b_eq - qp.a_eq @ d_fix
    bt_i = qp.b_ineq - qp.a_ineq @ d_fix
    nf = int(free.sum())
    for size in range(m_i + 1):
        for subset in itertools.combinations(range(m_i), size):
            a_a = np.concatenate([a_e, a_i[list(subset)]], axis=0)
            k = a_a.shape[0]
            kkt = np.block([[b_ff, a_a.T], [a_a, np.zeros((k, k))]])
            rhs = np.concatenate([-g_t, bt_e, bt_i[list(subset)]])
            sol = scipy.linalg.lstsq(kkt, rhs)[0] if nf + k else np.zeros(0)
            d_f, lam = sol[:nf], sol[nf:]
            lam_i = np.zeros(m_i)
            lam_i[list(subset)] = lam[m_e:]
            feasible = np.all(a_i @ d_f - bt_i <= 1e-9) if m_i else True
            if feasible and np.all(lam_i >= -1e-9):
                d = d_fix.copy()
                d[free] += d_f
                return d, lam[:m_e], lam_i
    raise AssertionError("no feasible working set found")  # pragma: no cover


def stationarity_residual(qp: RandomQP, res: FixedBoundsQPResult) -> np.ndarray:
    d = np.asarray(res.step)
    dual = res.dual
    return (
        qp.dense_b @ d
        + qp.g
        + qp.a_eq.T @ np.asarray(dual.eq_multipliers)
        + qp.a_ineq.T @ np.asarray(dual.ineq_multipliers)
        - np.asarray(dual.lb_multipliers)
        + np.asarray(dual.ub_multipliers)
    )


def multipliers_unique(qp: RandomQP, free, lam_i_ref, dup: bool) -> bool:
    """Whether the multipliers are determined (independent active free rows)."""
    m_e = qp.a_eq.shape[0]
    active = np.concatenate([np.ones(m_e, dtype=bool), lam_i_ref > 1e-9])
    rows = np.concatenate([qp.a_eq, qp.a_ineq], axis=0)[active][:, free]
    rank = np.linalg.matrix_rank(rows) if rows.size else 0
    return rank == rows.shape[0] - (1 if dup else 0)


def assert_matches_reference(case: Case) -> None:
    """Status, step and (when determined) inequality multipliers vs the dense QP."""
    qp = case.qp
    res = run(case)
    assert bool(res.ldp_status == LDP_RESULTS.feasible)
    d_ref, _, lam_i_ref = dense_reference(qp, case.lower, case.upper)
    free = ~(case.lower | case.upper)
    np.testing.assert_allclose(np.asarray(res.step), d_ref, atol=REF_TOL)
    if multipliers_unique(qp, free, lam_i_ref, qp.a_eq.shape[0] == 3):
        np.testing.assert_allclose(
            np.asarray(res.dual.ineq_multipliers), lam_i_ref, atol=REF_TOL
        )


class TestFixedBoundsQP:
    def test_matches_dense_reference(self, case: Case):
        assert_matches_reference(case)

    @pytest.mark.parametrize("kind", ["none", "random"])
    @pytest.mark.parametrize("mix", [(0, 3), (2, 3)], ids=["0-3", "2-3"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_tight_inequality_matches_dense_reference(self, mix, kind, seed):
        """One inequality active at the feasible ``d0`` (degenerate vertex)."""
        assert_matches_reference(build_case(12, 3, mix, kind, seed, tight=True))

    def test_kkt_and_structure(self, case: Case):
        qp = case.qp
        res = run(case)
        d = np.asarray(res.step)
        dual = res.dual
        pinned = case.lower | case.upper
        # Stationarity on every coordinate, pinned variables included.
        np.testing.assert_allclose(stationarity_residual(qp, res), 0.0, atol=KKT_TOL)
        # Pinned variables sit exactly at their step; free ones are unrestricted.
        np.testing.assert_array_equal(
            d[pinned], d_fix_of(qp, case.lower, case.upper)[pinned]
        )
        # Multipliers vanish off the working set; the inequality ones are >= 0.
        lb_m, ub_m = np.asarray(dual.lb_multipliers), np.asarray(dual.ub_multipliers)
        assert np.all(lb_m[~case.lower] == 0.0)
        assert np.all(ub_m[~case.upper] == 0.0)
        lam_i = np.asarray(dual.ineq_multipliers)
        assert np.all(lam_i >= -1e-10)
        # Primal feasibility and complementarity of the general constraints.
        np.testing.assert_allclose(qp.a_eq @ d - qp.b_eq, 0.0, atol=KKT_TOL)
        slack = qp.b_ineq - qp.a_ineq @ d
        assert np.all(slack >= -KKT_TOL)
        np.testing.assert_allclose(lam_i * slack, 0.0, atol=KKT_TOL)
        assert float(res.feasibility_residual) <= KKT_TOL
        assert int(res.n_nnls_iter) >= 0

    def test_no_working_set_is_unbounded_qp(self, random_lrh_qp):
        """With ``W = ∅`` the bounds (finite here) play no role."""
        qp = random_lrh_qp(6, 3, 2, 3, seed=4)
        none = jnp.zeros(6, dtype=bool)
        res = _solve(qp.lagrangian, qp.factors, none, none)
        d_ref, lam_e_ref, lam_i_ref = dense_reference(
            qp, np.zeros(6, bool), np.zeros(6, bool)
        )
        np.testing.assert_allclose(np.asarray(res.step), d_ref, atol=REF_TOL)
        np.testing.assert_allclose(
            np.asarray(res.dual.eq_multipliers), lam_e_ref, atol=REF_TOL
        )
        np.testing.assert_allclose(
            np.asarray(res.dual.ineq_multipliers), lam_i_ref, atol=REF_TOL
        )
        assert np.all(np.asarray(res.dual.lb_multipliers) == 0.0)
        assert np.all(np.asarray(res.dual.ub_multipliers) == 0.0)


class TestWorkingSetRules:
    @pytest.mark.parametrize("mix", [(0, 0), (2, 3)], ids=["0-0", "2-3"])
    def test_infinite_bounds_never_pinned(self, mix):
        n = 8
        infinite = np.arange(n) % 2 == 0
        lower, upper = np.zeros(n, bool), np.zeros(n, bool)
        lower[1], upper[3] = True, True
        qp = make_random_lrh_qp(
            n, 3, *mix, seed=2, lower=lower, upper=upper, infinite=infinite
        )
        base = Case(qp, lower, upper)
        flagged = run(base, lower | infinite, upper | infinite)
        reference = run(base)
        np.testing.assert_array_equal(
            np.asarray(flagged.step), np.asarray(reference.step)
        )
        np.testing.assert_array_equal(
            np.asarray(flagged.dual.lb_multipliers),
            np.asarray(reference.dual.lb_multipliers),
        )
        assert np.all(np.asarray(flagged.dual.lb_multipliers)[infinite] == 0.0)
        assert np.all(np.asarray(flagged.dual.ub_multipliers)[infinite] == 0.0)

    @pytest.mark.parametrize("mix", [(0, 0), (2, 3)], ids=["0-0", "2-3"])
    def test_lower_face_wins_when_both_flagged(self, mix):
        n = 8
        equal = np.zeros(n, bool)
        equal[[0, 4]] = True  # lb == ub == x + d0 there
        qp = make_random_lrh_qp(n, 2, *mix, seed=3, lower=equal, upper=equal)
        base = Case(qp, equal, equal)
        both = run(base)
        lower_only = run(base, equal, np.zeros(n, bool))
        np.testing.assert_array_equal(
            np.asarray(both.step), np.asarray(lower_only.step)
        )
        assert np.all(np.asarray(both.dual.ub_multipliers) == 0.0)
        np.testing.assert_array_equal(
            np.asarray(both.dual.lb_multipliers),
            np.asarray(lower_only.dual.lb_multipliers),
        )
        np.testing.assert_allclose(stationarity_residual(qp, both), 0.0, atol=KKT_TOL)
        np.testing.assert_allclose(
            np.asarray(both.step)[equal], (qp.lb - qp.x)[equal], atol=0.0
        )

    def test_traced_masks_compile_once(self):
        traces = []

        def counted(lagrangian, factors, active_lb, active_ub):
            traces.append(1)
            return solve_fixed_bounds_qp(lagrangian, factors, active_lb, active_ub)

        solve = jax.jit(counted)
        for kind in WORKING_SETS:
            case = build_case(6, 2, (2, 3), kind)
            res = solve(
                case.qp.lagrangian,
                case.qp.factors,
                jnp.asarray(case.lower),
                jnp.asarray(case.upper),
            )
            assert bool(res.ldp_status == LDP_RESULTS.feasible)
        assert len(traces) == 1


def _break_equalities(qp: RandomQP) -> RandomQP:
    """Duplicated equality rows with different right-hand sides."""
    ev = qp.lagrangian.evaluated
    b_eq = qp.b_eq.copy()
    b_eq[-1] -= 1.0  # eq_fn_val = -b_eq
    return qp._replace(
        lagrangian=eqx.tree_at(
            lambda lag: lag.evaluated.eq_fn_val,
            qp.lagrangian,
            ev.eq_fn_val.at[-1].add(1.0),
        ),
        b_eq=b_eq,
    )


def _opposite_halfspaces(qp: RandomQP) -> RandomQP:
    """``aᵀd ≤ -1`` and ``-aᵀd ≤ -1``: no step satisfies both."""
    n = qp.x.shape[0]
    a = jnp.ones((1, n))
    lag = eqx.tree_at(
        lambda lg: (lg.evaluated.ineq_fn_jac_val, lg.evaluated.ineq_fn_val),
        qp.lagrangian,
        (jnp.concatenate([a, -a]), jnp.ones(2)),
    )
    return qp._replace(
        lagrangian=lag,
        a_ineq=np.concatenate([np.ones((1, n)), -np.ones((1, n))]),
        b_ineq=-np.ones(2),
    )


def _nonfinite_gradient_at(qp: RandomQP, index: int) -> RandomQP:
    lag = eqx.tree_at(
        lambda lg: lg.evaluated.grad_val,
        qp.lagrangian,
        qp.lagrangian.evaluated.grad_val.at[index].set(jnp.nan),
    )
    return qp._replace(lagrangian=lag)


def _nonfinite_gradient(qp: RandomQP) -> RandomQP:
    return _nonfinite_gradient_at(qp, 0)


def _status_qp(mix, mutate, lower, upper) -> RandomQP:
    qp = make_random_lrh_qp(
        6, 2, *mix, seed=0, dup=mix[0] == 3, lower=lower, upper=upper
    )
    return mutate(qp)


STATUS_CASES = {
    "inconsistent": ((3, 0), _break_equalities, LDP_RESULTS.inconsistent_equalities),
    "infeasible": ((0, 2), _opposite_halfspaces, LDP_RESULTS.infeasible),
    "nonfinite": ((2, 3), _nonfinite_gradient, LDP_RESULTS.nonfinite),
}


class TestStatusPropagation:
    @pytest.mark.parametrize("kind", WORKING_SETS)
    @pytest.mark.parametrize("name", list(STATUS_CASES))
    def test_status_and_fallback(self, name, kind):
        mix, mutate, expected = STATUS_CASES[name]
        lower, upper = working_set(kind, 6, 0)
        qp = _status_qp(mix, mutate, lower, upper)
        res = run(Case(qp, lower, upper))
        assert bool(res.ldp_status == expected)
        d_fix = d_fix_of(qp, lower, upper)
        # Documented fallback: the pinned step, zero dual, no NaN anywhere.
        np.testing.assert_array_equal(np.asarray(res.step), d_fix)
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)
        # The residual is that of the fallback step (finite, non-negative).
        a_all = np.concatenate([qp.a_eq, qp.a_ineq])
        m_e = qp.a_eq.shape[0]
        resid = a_all @ d_fix - np.concatenate([qp.b_eq, qp.b_ineq])
        expected_resid = max(
            np.max(np.abs(resid[:m_e]), initial=0.0),
            np.max(np.maximum(resid[m_e:], 0.0), initial=0.0),
        )
        assert np.isfinite(float(res.feasibility_residual))
        np.testing.assert_allclose(
            float(res.feasibility_residual), expected_resid, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Redundant linear equality under a stiff model (C24).
#
# LSEI sees ``Gᵀ = C_F⁻ᵀ A_Fᵀ``: a redundant equality (``a₃ = a₁ + a₂``) is
# exactly dependent in ``A`` but its rounding is amplified by the metric
# (about ``eps · κ`` for rows in the stiff part of ``range(Z)``). The default
# rank cut must still classify the row as dependent and return the step of the
# two independent rows; too small a cut returns ``feasible`` with a wrong step
# and multipliers of order 1e16 and above. ``B = (I - ZZᵀ) σ + ZRᵀRZᵀ`` is
# only ever applied through ``Z``: the reference is an O(n r) Woodbury solve.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, kappa", [(100, 1e2), (1000, 1e3)])
def test_redundant_equality_under_stiff_model_matches_woodbury(n, kappa):
    r = 10
    rng = np.random.default_rng(0)
    z = np.linalg.qr(rng.standard_normal((n, r)))[0]
    a_ind = rng.standard_normal((2, r)) @ z.T  # rows in range(Z)
    a_eq = np.vstack([a_ind, a_ind.sum(axis=0, keepdims=True)])
    b_eq = a_eq @ rng.standard_normal(n)
    g = rng.standard_normal(n)

    # Reference: B⁻¹ v = v - Z Zᵀ v + κ⁻² Z Zᵀ v (σ = 1, R = κ I), the two
    # independent rows, and their 2 x 2 Schur system.
    def b_inv(v):
        zv = z.T @ v
        return v - z @ zv + z @ zv / kappa**2

    binv_rows = np.stack([b_inv(row) for row in a_ind], axis=1)
    lam_ref = np.linalg.solve(a_ind @ binv_rows, -(b_eq[:2] + a_ind @ b_inv(g)))
    step_ref = -b_inv(g + a_ind.T @ lam_ref)

    evaluated = EvaluatedProblem(
        ref=Primal(x=jnp.zeros(n)),
        fn_val=jnp.zeros(()),
        grad_val=jnp.asarray(g),
        fn_qvp=None,
        eq_fn_val=jnp.asarray(-b_eq),
        eq_fn_jac_val=jnp.asarray(a_eq),
        eq_fn_qvp=None,
        ineq_fn_val=jnp.zeros((0,)),
        ineq_fn_jac_val=jnp.zeros((0, n)),
        ineq_fn_qvp=None,
        lb=jnp.full(n, -jnp.inf),
        ub=jnp.full(n, jnp.inf),
        null_lb=jnp.ones(n, dtype=bool),
        null_ub=jnp.ones(n, dtype=bool),
        aux_val=None,
    )
    lagrangian = EvaluatedLagrangian(
        evaluated=evaluated,
        secant=None,
        dual=Dual(jnp.zeros(3), jnp.zeros(0), jnp.zeros(n), jnp.zeros(n)),
    )
    factors = ReducedHessianFactors(
        basis=jnp.asarray(z),
        cholesky=jnp.asarray(kappa * np.eye(r)),
        sigma=jnp.asarray(1.0),
        rank=jnp.asarray(r, dtype=jnp.int32),
    )
    res = _solve(lagrangian, factors, jnp.zeros(n, bool), jnp.zeros(n, bool))

    assert bool(res.ldp_status == LDP_RESULTS.feasible)
    step = np.asarray(res.step)
    np.testing.assert_allclose(
        np.linalg.norm(step - step_ref) / np.linalg.norm(step_ref), 0.0, atol=1e-8
    )
    lam = np.asarray(res.dual.eq_multipliers)
    assert np.max(np.abs(lam)) <= 10.0 * np.max(np.abs(lam_ref))


class TestMemoryBudget:
    @pytest.mark.parametrize("elastic", [False, True])
    def test_no_n_by_n_intermediate(self, elastic):
        """No array of the jaxpr has two axes as large as ``n``."""
        n, m_c = 30, 5  # n is distinct from every other dimension (m=4, m_c=5)
        lower, upper = working_set("random", n, 0)
        qp = make_random_lrh_qp(n, 3, 2, 3, seed=1, lower=lower, upper=upper)
        jaxpr = jax.make_jaxpr(
            lambda *a: solve_fixed_bounds_qp(*a, elastic_rho=1.0 if elastic else None)
        )(
            qp.lagrangian,
            qp.factors,
            jnp.asarray(lower),
            jnp.asarray(upper),
        )
        shapes = set(walk_avals(jaxpr.jaxpr))
        offenders = [s for s in shapes if sum(1 for d in s if d >= n) >= 2]
        assert not offenders, offenders
        # The elastic QP adds one coordinate and two general rows: the
        # ``(n + 1) × (m_c + 2)`` matrix of the augmented least-distance problem.
        e = int(elastic)
        assert (
            max(int(np.prod(s)) for s in shapes) <= (n + e) * (max(m_c, 4) + 2 * e) + n
        )


# ---------------------------------------------------------------------------
# Degenerate feasible sets (empty interior): C11b.
#
# At an SQP vertex the active rows can outnumber the free dimensions, or two
# rows can be opposite, so the polytope of the reduced LDP is a single point or
# a flat set. NNLS then sees dependent columns, which plain Lawson-Hanson
# turns into a false ``infeasible`` or into a garbage step reported
# ``feasible``. A row supported only on pinned variables is a zero column of
# the reduced system (rounding noise instead of an exact zero).
# ---------------------------------------------------------------------------


def with_stiff_model(qp: RandomQP, factor: float) -> RandomQP:
    """Scale the Cholesky factor of the model: curvature ``factor²`` in ``Z``.

    ``dense_b`` is rebuilt from the factors (reference only, tiny ``n``).
    """
    fac = qp.factors
    rank = int(fac.rank)
    chol = np.asarray(fac.cholesky) * factor
    fac = eqx.tree_at(lambda f: f.cholesky, fac, jnp.asarray(chol))
    z = np.asarray(fac.basis)[:, :rank]
    r = chol[:rank, :rank]
    sigma = float(fac.sigma)
    n = qp.x.shape[0]
    dense_b = sigma * np.eye(n) + z @ (r.T @ r - sigma * np.eye(rank)) @ z.T
    return qp._replace(factors=fac, dense_b=dense_b)


def _scenario_row_on_pinned_variable(seed: int, scale: float):
    """One inequality supported only on a pinned variable, others general."""
    n = 6
    upper = index_mask(n, [0])
    qp = make_random_lrh_qp(n, 2, 0, 3, seed=seed, upper=upper)
    a_in, b_in = qp.a_ineq.copy(), qp.b_ineq.copy()
    a_in[0] = 0.0
    a_in[0, 0] = scale
    b_in[0] = scale * qp.d0[0]  # active at the pinned step
    qp = with_rows(qp, qp.a_eq, qp.b_eq, a_in, b_in)
    return qp, index_mask(n, []), upper, None


def _scenario_all_rows_on_pinned_variables(seed: int, scale: float):
    """Every inequality is supported only on pinned variables (zero columns)."""
    n = 6
    rng = np.random.default_rng(seed)
    upper = index_mask(n, [0, 1])
    qp = make_random_lrh_qp(n, 2, 0, 2, seed=seed, upper=upper)
    a_in = np.zeros((2, n))
    a_in[:, :2] = rng.standard_normal((2, 2)) * scale
    qp = with_rows(qp, np.zeros((0, n)), np.zeros(0), a_in, a_in @ qp.d0)
    return qp, index_mask(n, []), upper, None


def _scenario_two_opposite_inequalities(seed: int, scale: float):
    """3 free variables, 2 equalities, 2 opposite inequalities: singleton set."""
    n = 6
    rng = np.random.default_rng(seed)
    lower = index_mask(n, [0, 1, 2])
    qp = make_random_lrh_qp(n, 2, 2, 1, seed=seed, lower=lower)
    a_e = qp.a_eq * scale
    free = ~lower
    v = np.linalg.svd(a_e[:, free])[2][2]  # null direction of the free equalities
    a_i = np.zeros((2, n))
    a_i[:, free] = np.stack([v, -v]) * rng.uniform(0.5, 2, (2, 1)) + (
        rng.standard_normal((2, 2)) @ a_e[:, free]
    )
    a_i[:, lower] = rng.standard_normal((2, 3))
    a_i *= scale
    return (
        with_rows(qp, a_e, a_e @ qp.d0, a_i, a_i @ qp.d0),
        lower,
        index_mask(n, []),
        "d0",
    )


def _scenario_vertex_in_a_null_space(seed: int, scale: float):
    """1 equality, 3 inequalities positively spanning the 2-D null space."""
    n = 6
    rng = np.random.default_rng(seed)
    lower = index_mask(n, [0, 1, 2])
    qp = make_random_lrh_qp(n, 2, 1, 1, seed=seed, lower=lower)
    a_e = qp.a_eq[:1] * scale
    free = ~lower
    null = np.linalg.svd(a_e[:, free])[2][1:].T  # 3 x 2
    dirs = rng.standard_normal((3, 2))
    dirs[2] = -(rng.uniform(0.5, 2, 2) @ dirs[:2])  # positively spanning
    a_i = np.zeros((3, n))
    a_i[:, free] = dirs @ null.T + rng.standard_normal((3, 1)) @ a_e[:, free]
    a_i[:, lower] = rng.standard_normal((3, 3))
    a_i *= scale
    return (
        with_rows(qp, a_e, a_e @ qp.d0, a_i, a_i @ qp.d0),
        lower,
        index_mask(n, []),
        "d0",
    )


DEGENERATE_SCENARIOS = {
    "row-on-pinned-variable": _scenario_row_on_pinned_variable,
    "all-rows-on-pinned-variables": _scenario_all_rows_on_pinned_variables,
    "two-opposite-inequalities": _scenario_two_opposite_inequalities,
    "vertex-in-null-space": _scenario_vertex_in_a_null_space,
}


class TestDegenerateFeasibleSets:
    @pytest.mark.parametrize("curvature", [1.0, 1e6], ids=["soft", "stiff"])
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("name", list(DEGENERATE_SCENARIOS))
    def test_step_matches_reference(self, name, seed, curvature):
        """``feasible`` and the exact step, whatever the row scale or model.

        The singleton scenarios have ``d0`` as their only feasible step; for
        the others the dense KKT enumeration gives the reference.
        """
        scale = 10 ** np.random.default_rng(100 + seed).uniform(-2, 4)
        qp, lower, upper, expected = DEGENERATE_SCENARIOS[name](seed, scale)
        qp = with_stiff_model(qp, np.sqrt(curvature))
        case = Case(qp, lower, upper)
        d_ref = qp.d0 if expected == "d0" else dense_reference(qp, lower, upper)[0]
        res = run(case)
        assert bool(res.ldp_status == LDP_RESULTS.feasible)
        np.testing.assert_allclose(
            np.asarray(res.step), d_ref, rtol=1e-7, atol=1e-7 * np.linalg.norm(d_ref)
        )
        assert float(res.feasibility_residual) <= 1e-8 * max(1.0, scale)

    @pytest.mark.parametrize("curvature", [1.0, 1e6], ids=["soft", "stiff"])
    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("m_e", [0, 1])
    @pytest.mark.parametrize("tight", [2, 3, 4])
    @pytest.mark.parametrize("n_free", [2, 3])
    def test_several_tight_inequalities(self, n_free, tight, m_e, seed, curvature):
        """Several inequalities active at ``d0`` in a small free subspace.

        With ``tight > n_free`` the active rows can positively span the free
        subspace, leaving ``d0`` as the only feasible step.
        """
        n = 8
        lower = index_mask(
            n, list(range(n_free, n))
        )  # the first n_free variables are free
        qp = make_random_lrh_qp(
            n, 3, m_e, tight + 1, seed=seed, tight=tight, lower=lower
        )
        qp = with_stiff_model(qp, np.sqrt(curvature))
        res = run(Case(qp, lower, index_mask(n, [])))
        assert bool(res.ldp_status == LDP_RESULTS.feasible)
        # Convex QP: a KKT point is the solution (the enumeration reference
        # cannot pick non-negative multipliers of a degenerate vertex).
        d = np.asarray(res.step)
        dual = res.dual
        lam_e, lam_i = (
            np.asarray(dual.eq_multipliers),
            np.asarray(dual.ineq_multipliers),
        )
        terms = [
            np.linalg.norm(qp.dense_b, 2) * d,  # rounding scale of B d
            qp.g,
            qp.a_eq.T @ lam_e,
            qp.a_ineq.T @ lam_i,
            np.asarray(dual.lb_multipliers),
        ]
        scale = sum(np.linalg.norm(t) for t in terms)
        assert np.linalg.norm(stationarity_residual(qp, res)) <= 1e-8 * scale
        np.testing.assert_allclose(qp.a_eq @ d - qp.b_eq, 0.0, atol=1e-8)
        slack = qp.b_ineq - qp.a_ineq @ d
        assert np.all(slack >= -1e-8)
        assert np.all(lam_i >= -1e-10 * scale)
        assert np.all(np.abs(lam_i * slack) <= 1e-8 * scale)


# ---------------------------------------------------------------------------
# Farkas certificates and rays (C11c).
#
# When the pinned QP is infeasible the result carries y = [y_E; y_I] with
# y_I >= 0, A_Fᵀ y ≈ 0, b̃ᵀ y < 0 (b̃ = b - A d_fix), ‖y‖_∞ = 1, and the ray
# ρ_y = A_Eᵀ y_E + A_Iᵀ y_I of the bound multipliers:
# ray_lb = ρ_y on lower pins, ray_ub = -ρ_y on upper pins.
# ---------------------------------------------------------------------------

NOT_FEASIBLE = (LDP_RESULTS.infeasible, LDP_RESULTS.inconsistent_equalities)


def assert_ray_certificate(case: Case, res: FixedBoundsQPResult) -> None:
    """The certificate and ray identities, measured in the data's own units."""
    qp = case.qp
    lower, upper = case.lower, case.upper
    pin_lb = lower & np.isfinite(qp.lb)
    pin_ub = upper & ~lower & np.isfinite(qp.ub)
    free = ~(pin_lb | pin_ub)
    assert bool(res.has_ray)
    assert any(bool(res.ldp_status == s) for s in NOT_FEASIBLE)

    m_e = qp.a_eq.shape[0]
    a_all = np.concatenate([qp.a_eq, qp.a_ineq])
    b_all = np.concatenate([qp.b_eq, qp.b_ineq])
    y_e, y_i = np.asarray(res.certificate_eq), np.asarray(res.certificate_ineq)
    assert y_e.shape == (m_e,) and y_i.shape == (qp.a_ineq.shape[0],)
    y = np.concatenate([y_e, y_i])
    assert np.all(np.isfinite(y)) and np.all(y_i >= 0.0)
    np.testing.assert_allclose(np.max(np.abs(y)), 1.0, rtol=1e-12)

    d_fix = np.where(pin_lb, qp.lb - qp.x, np.where(pin_ub, qp.ub - qp.x, 0.0))
    b_tilde = b_all - a_all @ d_fix
    a_norm = np.linalg.norm(a_all)
    assert np.linalg.norm(a_all[:, free].T @ y) <= 1e-8 * a_norm
    assert b_tilde @ y < 0.0
    assert b_tilde @ y <= -1e-6 * (np.abs(b_tilde) @ np.abs(y))
    # Growth rate of the dual objective along the ray, and no relaxation.
    np.testing.assert_allclose(float(res.ray_rate), -(b_tilde @ y), rtol=1e-12)
    assert float(res.delta) == 0.0

    ray_lb, ray_ub = np.asarray(res.ray_lb), np.asarray(res.ray_ub)
    rho = a_all.T @ y
    np.testing.assert_allclose(ray_lb, np.where(pin_lb, rho, 0.0), atol=1e-12 * a_norm)
    np.testing.assert_allclose(ray_ub, np.where(pin_ub, -rho, 0.0), atol=1e-12 * a_norm)
    # the ray identity on all n coordinates (free ones: ρ_y itself vanishes)
    assert np.linalg.norm(rho - ray_lb + ray_ub) <= 1e-8 * a_norm


def assert_no_ray(qp: RandomQP, res: FixedBoundsQPResult) -> None:
    m_e, m_i = qp.a_eq.shape[0], qp.a_ineq.shape[0]
    assert not bool(res.has_ray)
    assert float(res.ray_rate) == 0.0 and float(res.delta) == 0.0
    assert res.certificate_eq.shape == (m_e,)
    assert res.certificate_ineq.shape == (m_i,)
    assert res.ray_lb.shape == res.ray_ub.shape == qp.x.shape
    for leaf in (res.certificate_eq, res.certificate_ineq, res.ray_lb, res.ray_ub):
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)


def _over_pinned(mix, kind, seed):
    """A QP with a feasible box but pins at bounds that miss ``A d = b``."""
    n = 6
    qp = make_random_lrh_qp(n, 2, *mix, seed=seed, dup=mix[0] == 3)
    pinned = {
        "all": index_mask(n, range(n)),
        "all-but-one": index_mask(n, range(1, n)),
    }[kind]
    upper = pinned & (np.random.default_rng(seed).random(n) < 0.5)
    return Case(qp, pinned & ~upper, upper)


def _box_too_tight(mix, kind, seed):
    """An inequality supported only on a variable whose pin overshoots it."""
    n = 6
    scale = 10.0 ** {"unit": 0.0, "large": 3.0, "small": -3.0}[kind]
    upper = index_mask(n, [0])  # not passed to the generator: ub = x + d0 + (0.5..1.5)
    qp = make_random_lrh_qp(n, 2, *mix, seed=seed, dup=mix[0] == 3)
    a_in, b_in = qp.a_ineq.copy(), qp.b_ineq.copy()
    a_in[0] = 0.0
    a_in[0, 0] = scale
    b_in[0] = scale * (qp.d0[0] + 0.1)  # the pin sits at d0[0] + (0.5..1.5)
    qp = with_rows(qp, qp.a_eq, qp.b_eq, a_in, b_in)
    return Case(qp, index_mask(n, []), upper)


CERTIFICATE_SCENARIOS = {
    "over-pinned-all": (_over_pinned, "all", [(2, 0), (2, 3), (3, 2)]),
    "over-pinned-all-but-one": (_over_pinned, "all-but-one", [(2, 0), (2, 3), (3, 2)]),
    "box-too-tight-unit": (_box_too_tight, "unit", [(0, 3), (2, 3)]),
    "box-too-tight-large": (_box_too_tight, "large", [(0, 3), (2, 3)]),
    "box-too-tight-small": (_box_too_tight, "small", [(0, 3), (2, 3)]),
}
CERTIFICATE_PARAMS = [
    (make, kind, mix)
    for make, kind, mixes in CERTIFICATE_SCENARIOS.values()
    for mix in mixes
]
CERTIFICATE_IDS = [
    f"{name}-{mix[0]}-{mix[1]}"
    for name, (_, _, mixes) in CERTIFICATE_SCENARIOS.items()
    for mix in mixes
]


class TestFarkasCertificates:
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("make, kind, mix", CERTIFICATE_PARAMS, ids=CERTIFICATE_IDS)
    def test_over_pinned_and_tight_box_certificates(self, make, kind, mix, seed):
        case = make(mix, kind, seed)
        assert_ray_certificate(case, run(case))

    @pytest.mark.parametrize("kind", WORKING_SETS)
    @pytest.mark.parametrize("name", ["inconsistent", "infeasible"])
    def test_status_cases_carry_certificates(self, name, kind):
        """Duplicated equalities with clashing rhs, opposite half-spaces."""
        mix, mutate, expected = STATUS_CASES[name]
        lower, upper = working_set(kind, 6, 0)
        case = Case(_status_qp(mix, mutate, lower, upper), lower, upper)
        res = run(case)
        assert bool(res.ldp_status == expected)
        assert_ray_certificate(case, res)

    def test_feasible_problems_have_no_ray(self, case: Case):
        res = run(case)
        assert bool(res.ldp_status == LDP_RESULTS.feasible)
        assert_no_ray(case.qp, res)

    def test_nonfinite_has_no_ray(self):
        mix, mutate, expected = STATUS_CASES["nonfinite"]
        lower, upper = working_set("random", 6, 0)
        qp = _status_qp(mix, mutate, lower, upper)
        res = run(Case(qp, lower, upper))
        assert bool(res.ldp_status == expected)
        assert_no_ray(qp, res)

    @pytest.mark.parametrize("mix", [(2, 0), (2, 3)])
    def test_ray_follows_the_face_of_the_pin(self, mix):
        """A variable flagged on both faces is pinned at the lower one, so the
        ray lives in ``ray_lb`` and ``ray_ub`` stays zero there."""
        case = _over_pinned(mix, "all", 0)
        both = np.ones(6, dtype=bool)
        res = run(case, lower=both, upper=both)
        both_case = Case(case.qp, both, np.zeros(6, dtype=bool))
        assert_ray_certificate(both_case, res)
        assert np.any(np.asarray(res.ray_lb) != 0.0)
        np.testing.assert_array_equal(np.asarray(res.ray_ub), 0.0)


# ---------------------------------------------------------------------------
# Elastic variant (T09, C16): ``elastic_rho`` relaxes the QP by one scalar
# δ ∈ [0, 1] with the UNSHIFTED column ``c = [b_E, min(b_I, 0)]`` and the
# frozen absolute weight ρ_e (plan 2.5 rev 11/12).
# ---------------------------------------------------------------------------

WEIGHT = 1e3


def scaled_qp(qp: RandomQP, kappa: float) -> RandomQP:
    """``qp`` with ``g`` and ``b`` multiplied by ``kappa`` (``w`` scales by ``kappa``)."""
    scaled = with_rows(qp, qp.a_eq, kappa * qp.b_eq, qp.a_ineq, kappa * qp.b_ineq)
    lag = eqx.tree_at(
        lambda lg: lg.evaluated.grad_val,
        scaled.lagrangian,
        kappa * scaled.lagrangian.evaluated.grad_val,
    )
    return scaled._replace(lagrangian=lag, g=kappa * qp.g)


def tau_squared_reference(qp: RandomQP) -> float:
    """Dense ``τ²`` of the unpinned data (plan 2.5 rev 12), reference formula.

    ``‖G₀_j‖² = a_jᵀ B⁻¹ a_j``, ``h₀ = b + A B⁻¹ g``, column ``c = b`` on the
    equalities and ``min(b, 0)`` on the inequalities.
    """
    m_e = qp.a_eq.shape[0]
    a_all = np.concatenate([qp.a_eq, qp.a_ineq])
    b_all = np.concatenate([qp.b_eq, qp.b_ineq])
    if a_all.shape[0] == 0:
        return 1.0
    b_inv_a = np.linalg.solve(qp.dense_b, a_all.T)
    norm = np.sqrt(np.einsum("ij,ji->i", a_all, b_inv_a))
    h0 = b_all + a_all @ np.linalg.solve(qp.dense_b, qp.g)
    h_term = np.concatenate([np.abs(h0[:m_e]), np.maximum(-h0[m_e:], 0.0)])
    dist = np.maximum(h_term, np.abs(elastic_column_np(qp))) / norm
    tau = dist.max()
    return float(tau**2 if tau > 0 else 1.0)


class TestElasticPenalty:
    @pytest.mark.parametrize("kappa", [1e-6, 1.0, 1e3])
    @pytest.mark.parametrize("weight", [1e-2, 1e3])
    @pytest.mark.parametrize("mix", MIXES[1:], ids=MIX_IDS[1:])
    @pytest.mark.parametrize("seed", range(2))
    def test_matches_the_dense_formula(self, mix, weight, kappa, seed):
        qp = scaled_qp(
            make_random_lrh_qp(8, 3, *mix, seed=seed, dup=mix[0] == 3), kappa
        )
        rho = float(elastic_penalty(qp.lagrangian, qp.factors, weight))
        np.testing.assert_allclose(rho, weight * tau_squared_reference(qp), rtol=1e-8)

    @pytest.mark.parametrize("kappa", [1e-6, 1e3])
    @pytest.mark.parametrize(
        "mix", [(2, 0), (2, 3), (3, 2)], ids=["2-0", "2-3", "3dup-2"]
    )
    def test_scales_with_the_square_of_the_data(self, mix, kappa):
        """Scaling ``g`` and ``b`` by ``κ`` scales ``w`` by ``κ``, ``τ²`` by ``κ²``.

        (Needs an active scale: with only slack inequalities ``τ = 1``.)
        """
        base = make_random_lrh_qp(8, 3, *mix, seed=0, dup=mix[0] == 3)
        qp = scaled_qp(base, kappa)
        ratio = elastic_rho_of(qp) / elastic_rho_of(base)
        np.testing.assert_allclose(ratio, kappa**2, rtol=1e-8)

    def test_only_slack_inequalities_give_unit_scale(self):
        qp = make_random_lrh_qp(8, 3, 0, 3, seed=0)
        assert float(elastic_penalty(qp.lagrangian, qp.factors, 5.0)) == 5.0

    def test_no_general_constraints_gives_unit_scale(self):
        qp = make_random_lrh_qp(6, 2, 0, 0, seed=0)
        assert float(elastic_penalty(qp.lagrangian, qp.factors, 7.0)) == 7.0

    def test_jit_and_non_finite_data(self):
        qp = make_random_lrh_qp(6, 2, 2, 3, seed=0)
        jitted = jax.jit(elastic_penalty, static_argnums=(2,))
        np.testing.assert_allclose(
            float(jitted(qp.lagrangian, qp.factors, WEIGHT)),
            float(elastic_penalty(qp.lagrangian, qp.factors, WEIGHT)),
        )
        lag = eqx.tree_at(
            lambda lg: lg.evaluated.grad_val,
            qp.lagrangian,
            qp.lagrangian.evaluated.grad_val.at[0].set(jnp.nan),
        )
        assert np.isnan(float(elastic_penalty(lag, qp.factors, WEIGHT)))


def elastic_rho_of(qp: RandomQP, weight: float = WEIGHT) -> float:
    return float(elastic_penalty(qp.lagrangian, qp.factors, weight))


class TestElasticKernel:
    def test_solves_the_relaxed_qp_on_every_working_set(self, case: Case):
        """KKT of the relaxed QP (so the minimiser), ``d_W = d_fix``, ``δ`` bound.

        The plain QP of these cases is feasible; with ``λ`` its multipliers,
        ``0 ≤ δ ≤ max(-cᵀλ, 0)/ρ_e`` (plan rev 12), and ``δ`` is not in the dual.
        """
        qp = case.qp
        rho = elastic_rho_of(qp)
        res = run(case, elastic_rho=rho)
        plain = run(case)
        pinned = case.lower | case.upper
        assert bool(res.ldp_status == LDP_RESULTS.feasible)
        assert not bool(res.has_ray)
        delta = float(res.delta)
        assert 0.0 <= delta <= 1.0
        d = np.asarray(res.step)
        np.testing.assert_allclose(
            d[pinned], d_fix_of(qp, case.lower, case.upper)[pinned], atol=1e-12
        )
        assert_elastic_kkt(qp, rho, d, delta, res.dual, pinned=pinned)
        lam = np.concatenate(
            [
                np.asarray(plain.dual.eq_multipliers),
                np.asarray(plain.dual.ineq_multipliers),
            ]
        )
        bound = max(-(elastic_column_np(qp) @ lam), 0.0) / rho
        assert delta <= bound + 1e-9
        assert res.dual.eq_multipliers.shape == plain.dual.eq_multipliers.shape
        assert res.dual.ineq_multipliers.shape == plain.dual.ineq_multipliers.shape

    @pytest.mark.parametrize("kind", WORKING_SETS)
    def test_nonfinite_data_gives_the_fallback_outputs(self, kind):
        """A non-finite gradient: ``nonfinite``, ``d_fix`` step, zero dual and ``δ``."""
        mix, mutate, expected = STATUS_CASES["nonfinite"]
        lower, upper = working_set(kind, 6, 0)
        qp = _status_qp(mix, mutate, lower, upper)
        res = run(Case(qp, lower, upper), elastic_rho=1.0)
        assert bool(res.ldp_status == expected)
        np.testing.assert_array_equal(np.asarray(res.step), d_fix_of(qp, lower, upper))
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)
        assert float(res.delta) == 0.0
        assert not bool(res.has_ray) and float(res.ray_rate) == 0.0

    @pytest.mark.parametrize("weight", [1e-1, 1e2, 1e5])
    def test_relaxation_shrinks_towards_the_plain_step_with_the_weight(self, weight):
        """A larger ``ρ_e`` pulls ``δ`` and the step towards the plain solution."""
        case = build_case(8, 3, (2, 3), "random")
        plain = np.asarray(run(case).step)
        errs = []
        for w in (weight, 10 * weight):
            res = run(case, elastic_rho=elastic_rho_of(case.qp, w))
            errs.append(np.abs(np.asarray(res.step) - plain).max())
        assert errs[1] <= errs[0] + 1e-9

    def test_traced_weight_compiles_once(self):
        traces = []

        def counted(lagrangian, factors, active_lb, active_ub, rho):
            traces.append(1)
            return solve_fixed_bounds_qp(
                lagrangian, factors, active_lb, active_ub, elastic_rho=rho
            )

        solve = jax.jit(counted)
        case = build_case(6, 2, (2, 3), "random")
        for weight in (1e1, 1e3, 1e5):
            res = solve(
                case.qp.lagrangian,
                case.qp.factors,
                jnp.asarray(case.lower),
                jnp.asarray(case.upper),
                jnp.asarray(elastic_rho_of(case.qp, weight)),
            )
            assert bool(res.ldp_status == LDP_RESULTS.feasible)
        assert len(traces) == 1


def pinned_elastic_data(make, kind, mix, seed):
    """``(case, δ_min)``: the LP gives the smallest ``δ`` of the pinned elastic QP."""
    case = make(mix, kind, seed)
    qp = case.qp
    d_fix = d_fix_of(qp, case.lower, case.upper)
    pin = (case.lower & np.isfinite(qp.lb)) | (case.upper & np.isfinite(qp.ub))
    lo = np.where(pin, d_fix, qp.lb - qp.x)
    hi = np.where(pin, d_fix, qp.ub - qp.x)
    return case, delta_min_lp(qp, lo, hi)


PINNED_ELASTIC = [(params, seed) for params in CERTIFICATE_PARAMS for seed in range(4)]
PINNED_ELASTIC_IDS = [f"{cid}-s{seed}" for cid in CERTIFICATE_IDS for seed in range(4)]
_PINNED_DELTA_MIN = {
    key: pinned_elastic_data(*params, seed)[1]
    for key, (params, seed) in zip(PINNED_ELASTIC_IDS, PINNED_ELASTIC)
}
PINNED_FEASIBLE = [
    pytest.param(*p[0], p[1], id=i)
    for p, i in zip(PINNED_ELASTIC, PINNED_ELASTIC_IDS)
    if _PINNED_DELTA_MIN[i] is not None
]
PINNED_INFEASIBLE = [
    pytest.param(*p[0], p[1], id=i)
    for p, i in zip(PINNED_ELASTIC, PINNED_ELASTIC_IDS)
    if _PINNED_DELTA_MIN[i] is None
]


def assert_elastic_ray(case: Case, res: FixedBoundsQPResult, rho: float) -> None:
    """Certificate of an infeasible pinned elastic problem (rev 12 ray rule).

    ``y`` (rows of ``A``) has ``y_I ≥ 0``, ``A_Fᵀ y ≈ 0`` and, with the rows of
    ``δ ∈ [0, 1]`` priced in, ``b̃ᵀy + max(-cᵀy, 0) < 0``: no ``(d_F, δ)`` meets
    ``A_F d_F + δ c = b̃`` / ``≤ b̃``. The growth rate along the ray is
    ``-(b̃ᵀy_gen + y_hi)`` with ``y_hi ≥ max(-cᵀy, 0)``.
    """
    qp = case.qp
    pin_lb = case.lower & np.isfinite(qp.lb)
    pin_ub = case.upper & ~case.lower & np.isfinite(qp.ub)
    free = ~(pin_lb | pin_ub)
    assert bool(res.has_ray)
    assert any(bool(res.ldp_status == s) for s in NOT_FEASIBLE)
    a_all = np.concatenate([qp.a_eq, qp.a_ineq])
    b_all = np.concatenate([qp.b_eq, qp.b_ineq])
    y_e, y_i = np.asarray(res.certificate_eq), np.asarray(res.certificate_ineq)
    y = np.concatenate([y_e, y_i])
    assert np.all(np.isfinite(y)) and np.all(y_i >= 0.0)
    # The augmented certificate [y; y_lo; y_hi] has unit sup-norm, so ‖y‖ ≤ 1.
    assert 0.0 < np.max(np.abs(y)) <= 1.0 + 1e-12
    d_fix = np.where(pin_lb, qp.lb - qp.x, np.where(pin_ub, qp.ub - qp.x, 0.0))
    b_tilde = b_all - a_all @ d_fix
    a_norm = np.linalg.norm(a_all)
    assert np.linalg.norm(a_all[:, free].T @ y) <= 1e-8 * a_norm * np.max(np.abs(y))
    c = elastic_column_np(qp)
    margin = b_tilde @ y + max(-(c @ y), 0.0)
    assert margin < 0.0
    assert margin <= -1e-6 * (np.abs(b_tilde) @ np.abs(y) + np.abs(c) @ np.abs(y))
    rate = float(res.ray_rate)
    assert rate > 0.0 and rate <= -margin + 1e-9 * max(1.0, abs(margin))
    # The ray identity on all coordinates, and the outputs of a failed solve.
    rho_y = a_all.T @ y
    ray_lb, ray_ub = np.asarray(res.ray_lb), np.asarray(res.ray_ub)
    np.testing.assert_allclose(
        ray_lb, np.where(pin_lb, rho_y, 0.0), atol=1e-12 * a_norm
    )
    np.testing.assert_allclose(
        ray_ub, np.where(pin_ub, -rho_y, 0.0), atol=1e-12 * a_norm
    )
    assert np.linalg.norm(rho_y - ray_lb + ray_ub) <= 1e-8 * a_norm
    assert float(res.delta) == 0.0
    np.testing.assert_array_equal(np.asarray(res.step), np.where(free, 0.0, d_fix))
    for leaf in jax.tree.leaves(res.dual):
        assert np.all(np.asarray(leaf) == 0.0)


class TestElasticOverPinned:
    """Over-pinned and tight-box QPs, solved or certified by the elastic variant.

    The LP of ``delta_min_lp`` (pins as ``lo == hi``) decides which of the two
    outcomes a pin set has; both occur in the scenarios below.
    """

    def test_both_outcomes_occur(self):
        assert PINNED_FEASIBLE and PINNED_INFEASIBLE

    @pytest.mark.parametrize("make, kind, mix, seed", PINNED_FEASIBLE)
    def test_feasible_pinned_elastic_problem_is_solved(self, make, kind, mix, seed):
        case, delta_min = pinned_elastic_data(make, kind, mix, seed)
        qp = case.qp
        rho = elastic_rho_of(qp)
        res = run(case, elastic_rho=rho)
        assert bool(res.ldp_status == LDP_RESULTS.feasible) and not bool(res.has_ray)
        delta = float(res.delta)
        assert delta_min is not None and delta >= delta_min - 1e-7
        pinned = (case.lower & np.isfinite(qp.lb)) | (
            case.upper & ~case.lower & np.isfinite(qp.ub)
        )
        assert_elastic_kkt(
            qp, rho, np.asarray(res.step), delta, res.dual, pinned=pinned
        )

    @pytest.mark.parametrize("make, kind, mix, seed", PINNED_FEASIBLE)
    def test_nonfinite_gradient_on_a_pinned_variable_zeroes_delta(
        self, make, kind, mix, seed
    ):
        """The reduction does not see a pinned coordinate of ``g``: the solve
        succeeds with ``δ > 0`` until the finiteness check flips the status, which
        must zero ``δ`` as well as the step and dual."""
        case, _ = pinned_elastic_data(make, kind, mix, seed)
        rho = elastic_rho_of(case.qp)
        assert float(run(case, elastic_rho=rho).delta) > 0.0
        pinned = (case.lower & np.isfinite(case.qp.lb)) | (
            case.upper & np.isfinite(case.qp.ub)
        )
        bad = _nonfinite_gradient_at(case.qp, int(np.flatnonzero(pinned)[0]))
        res = run(Case(bad, case.lower, case.upper), elastic_rho=rho)
        assert bool(res.ldp_status == LDP_RESULTS.nonfinite)
        assert float(res.delta) == 0.0 and not bool(res.has_ray)
        np.testing.assert_array_equal(
            np.asarray(res.step), d_fix_of(bad, case.lower, case.upper)
        )
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)

    @pytest.mark.parametrize("make, kind, mix, seed", PINNED_INFEASIBLE)
    def test_infeasible_pinned_elastic_problem_has_a_certificate(
        self, make, kind, mix, seed
    ):
        case, delta_min = pinned_elastic_data(make, kind, mix, seed)
        assert delta_min is None
        rho = elastic_rho_of(case.qp)
        assert_elastic_ray(case, run(case, elastic_rho=rho), rho)
