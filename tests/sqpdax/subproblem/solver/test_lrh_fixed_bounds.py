"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds`.

Convention under test (SQP view): ``min gᵀd + ½dᵀBd`` s.t. ``A_E d = b_E``,
``A_I d ≤ b_I``, ``d_W = d_fix``; stationarity
``B d + g + A_Eᵀλ_E + A_Iᵀλ_I - λ_lb + λ_ub = 0``.

The dense references (``n × n`` NumPy / SciPy, tiny ``n``) live only in the
tests; the code under test never forms an ``n × n`` object.
"""

from __future__ import annotations

import itertools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from slsqp_jax.sqpdax.linalg import LDP_RESULTS
from slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds import (
    FixedBoundsQPResult,
    solve_fixed_bounds_qp,
)
from tests.sqpdax.subproblem.solver.conftest import RandomQP, make_random_lrh_qp


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


def _nonfinite_gradient(qp: RandomQP) -> RandomQP:
    lag = eqx.tree_at(
        lambda lg: lg.evaluated.grad_val,
        qp.lagrangian,
        qp.lagrangian.evaluated.grad_val.at[0].set(jnp.nan),
    )
    return qp._replace(lagrangian=lag)


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


def _walk_avals(jaxpr: Any):
    """Yield the shape of every variable of a jaxpr, sub-jaxprs included."""
    for var in itertools.chain(jaxpr.invars, jaxpr.constvars):
        yield var.aval.shape
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            yield getattr(var.aval, "shape", ())
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _walk_avals(inner)


class TestMemoryBudget:
    def test_no_n_by_n_intermediate(self):
        """No array of the jaxpr has two axes as large as ``n``."""
        n, m_c = 30, 5  # n is distinct from every other dimension (m=4, m_c=5)
        lower, upper = working_set("random", n, 0)
        qp = make_random_lrh_qp(n, 3, 2, 3, seed=1, lower=lower, upper=upper)
        jaxpr = jax.make_jaxpr(solve_fixed_bounds_qp)(
            qp.lagrangian,
            qp.factors,
            jnp.asarray(lower),
            jnp.asarray(upper),
        )
        shapes = set(_walk_avals(jaxpr.jaxpr))
        offenders = [s for s in shapes if sum(1 for d in s if d >= n) >= 2]
        assert not offenders, offenders
        assert max(int(np.prod(s)) for s in shapes) <= n * max(m_c, 4) + n
