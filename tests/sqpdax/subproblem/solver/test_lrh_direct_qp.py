"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.lrh_direct_qp`.

Convention under test (SQP view): ``min gᵀd + ½dᵀBd`` s.t. ``A_E d = b_E``,
``A_I d ≤ b_I``, ``lb ≤ x + d ≤ ub``; stationarity
``B d + g + A_Eᵀλ_E + A_Iᵀλ_I - λ_lb + λ_ub = 0`` with ``λ_I, λ_lb, λ_ub ≥ 0``.

The dense references (``n × n`` NumPy / SciPy, tiny ``n``) live only in the
tests; the code under test never forms an ``n × n`` object.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
from slsqp_jax.sqpdax.linalg import LDP_RESULTS
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.secant import LBFGS, LimitedMemoryReducedHessian
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    KKT_SOLVER_RESULTS,
    RESULTS,
    LRHDirectQPSolver,
    LRHQPSolverState,
    lrh_direct_qp,
)
from slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds import solve_fixed_bounds_qp
from tests.sqpdax.subproblem.solver.conftest import (
    RandomQP,
    index_mask,
    make_empty_active_set,
    make_random_lrh_qp,
    walk_avals,
    with_rows,
)


@pytest.fixture(autouse=True)
def _x64():
    """Run in float64 without leaking the flag to the float32 solver tests."""
    with jax.enable_x64(True):
        yield


MEMORY = 4
REF_TOL = 1e-6
KKT_TOL = 1e-7

# (m_E, m_I): the last equality row duplicates the first when m_E == 3.
MIXES = [(0, 0), (2, 0), (0, 3), (2, 3), (3, 2)]
MIX_IDS = ["0-0", "2-0", "0-3", "2-3", "3dup-2"]

CONVERGED = ACTIVE_SET_QP_RESULTS.working_set_converged
FAILURE = ACTIVE_SET_QP_RESULTS.kkt_solver_failure
BUDGET = ACTIVE_SET_QP_RESULTS.max_iter_reached
STALL = ACTIVE_SET_QP_RESULTS.anti_cycling

_oracle = jax.jit(solve_fixed_bounds_qp)
_unbounded = _oracle


# ---------------------------------------------------------------------------
# QP construction
# ---------------------------------------------------------------------------


def with_bounds_and_secant(qp: RandomQP, lb: np.ndarray, ub: np.ndarray) -> RandomQP:
    """Replace the bounds of ``qp`` and attach an LRH secant holding its factors."""
    n = qp.x.shape[0]
    ev = eqx.tree_at(
        lambda e: (e.lb, e.ub, e.null_lb, e.null_ub),
        qp.lagrangian.evaluated,
        (
            jnp.asarray(lb),
            jnp.asarray(ub),
            jnp.asarray(~np.isfinite(lb)),
            jnp.asarray(~np.isfinite(ub)),
        ),
    )
    f = qp.factors
    secant = eqx.tree_at(
        lambda s: (s.basis, s.cholesky, s.sigma, s.rank),
        LimitedMemoryReducedHessian(n=n, memory=MEMORY),
        (f.basis, f.cholesky, f.sigma, f.rank),
    )
    lag = EvaluatedLagrangian(evaluated=ev, secant=secant, dual=qp.lagrangian.dual)
    return qp._replace(lagrangian=lag, lb=lb, ub=ub)


def make_case(
    n: int,
    frac: float,
    x_on_bound: bool,
    mix: tuple[int, int],
    seed: int = 0,
    *,
    n_fixed: int = 0,
    one_sided: bool = False,
    rank: int = 3,
    tight: bool | int = False,
) -> RandomQP:
    """Random feasible convex QP with bounds, equalities and inequalities.

    ``x`` lies inside the box (as the minimiser guarantees). The general
    constraints are feasible at the anchor ``d_a = 0.1 d0`` (inside the box).
    The bounds cut the solution ``d_u`` of the QP without bounds on a
    fraction ``frac`` of the variables (a face between the anchor interval
    and ``d_u``, so the bound tends to be active at the solution) and are
    loose elsewhere. ``x_on_bound`` puts ``x`` exactly on one face on
    ``n // 4`` of the remaining variables, ``n_fixed`` variables get
    ``lb == ub``, and ``one_sided`` removes the non-cutting faces (and some of
    the loose variables' faces entirely).
    """
    m_e, m_i = mix
    qp = make_random_lrh_qp(
        n, rank, m_e, m_i, seed=seed, memory=MEMORY, dup=m_e == 3, tight=tight
    )
    d_a = 0.1 * qp.d0
    slack = qp.b_ineq - qp.a_ineq @ qp.d0
    qp = with_rows(
        qp, qp.a_eq, qp.a_eq @ d_a, qp.a_ineq, qp.a_ineq @ d_a + slack
    )._replace(d0=d_a)
    none = jnp.zeros(n, dtype=bool)
    d_u = np.asarray(_unbounded(qp.lagrangian, qp.factors, none, none).step)
    rng = np.random.default_rng(31 * seed + n)
    perm = rng.permutation(n)
    n_cut = int(round(frac * n))
    cut, rest = perm[:n_cut], perm[n_cut:]
    ref_lo, ref_hi = np.minimum(d_a, 0.0), np.maximum(d_a, 0.0)
    lo = np.minimum(ref_lo, d_u) - (0.5 + rng.random(n))
    hi = np.maximum(ref_hi, d_u) + (0.5 + rng.random(n))
    cut_up = d_u > ref_hi + 0.05
    cut_down = d_u < ref_lo - 0.05
    for i in cut:
        if cut_up[i]:
            hi[i] = ref_hi[i] + 0.5 * (d_u[i] - ref_hi[i])
        elif cut_down[i]:
            lo[i] = ref_lo[i] + 0.5 * (d_u[i] - ref_lo[i])
    n_on = n // 4 if x_on_bound else 0
    on_bound, rest = rest[:n_on], rest[n_on:]
    for i in on_bound:
        if d_a[i] >= 0:
            lo[i] = 0.0
        else:
            hi[i] = 0.0
    fixed, loose = rest[:n_fixed], rest[n_fixed:]
    lo[fixed] = d_a[fixed]
    hi[fixed] = d_a[fixed]
    if one_sided:
        for i in cut:
            if cut_up[i]:
                lo[i] = -np.inf
            elif cut_down[i]:
                hi[i] = np.inf
        for k, i in enumerate(loose):
            if k % 3 == 0:
                lo[i] = -np.inf
            elif k % 3 == 1:
                hi[i] = np.inf
            else:
                lo[i], hi[i] = -np.inf, np.inf
    return with_bounds_and_secant(qp, qp.x + lo, qp.x + hi)


# ---------------------------------------------------------------------------
# Running the solver
# ---------------------------------------------------------------------------


class Run(NamedTuple):
    step: np.ndarray
    dual: Dual
    state: LRHQPSolverState


def make_subproblem(qp: RandomQP) -> ActiveSetSubProblem:
    lag = qp.lagrangian
    return ActiveSetSubProblem(lag, make_empty_active_set(lag.n, lag.meq, lag.mineq))


def cold_state(qp: RandomQP) -> LRHQPSolverState:
    lag = qp.lagrangian
    return LRHQPSolverState.cold(lag.n, lag.meq, lag.mineq, lag.x_ref.dtype)


@eqx.filter_jit
def _solve(solver, sub, state):
    lag = sub.lagrangian
    x0 = (
        Primal(jnp.zeros_like(lag.x_ref)),
        Dual(
            jnp.zeros_like(lag.evaluated.eq_fn_val),
            jnp.zeros_like(lag.evaluated.ineq_fn_val),
            jnp.zeros_like(lag.x_ref),
            jnp.zeros_like(lag.x_ref),
        ),
    )
    return solver.solve(sub, x0, state)


def run(qp: RandomQP, solver: LRHDirectQPSolver | None = None, state=None) -> Run:
    solver = LRHDirectQPSolver() if solver is None else solver
    state = cold_state(qp) if state is None else state
    (step, dual), new_state = _solve(solver, make_subproblem(qp), state)
    return Run(np.asarray(step.x), dual, new_state)


# ---------------------------------------------------------------------------
# Dense references and KKT checks (tiny n)
# ---------------------------------------------------------------------------


def objective(qp: RandomQP, d: np.ndarray) -> float:
    return float(qp.g @ d + 0.5 * d @ qp.dense_b @ d)


def scipy_qp(
    qp: RandomQP, g: np.ndarray | None = None, with_bounds: bool = True
) -> np.ndarray:
    """SLSQP on the dense QP (reference, tiny ``n``), optionally without the box."""
    g = qp.g if g is None else g
    m_e = qp.a_eq.shape[0]
    # A duplicated equality row (consistent by construction) is dropped: SLSQP
    # does not accept dependent equalities, and the QP is the same.
    keep = m_e - 1 if m_e == 3 else m_e
    a_e, b_e = qp.a_eq[:keep], qp.b_eq[:keep]
    cons = []
    if keep:
        cons.append(
            {"type": "eq", "fun": lambda d: a_e @ d - b_e, "jac": lambda d: a_e}
        )
    if qp.a_ineq.shape[0]:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda d: qp.b_ineq - qp.a_ineq @ d,
                "jac": lambda d: -qp.a_ineq,
            }
        )
    lo, hi = qp.lb - qp.x, qp.ub - qp.x
    bounds = [
        (None if np.isneginf(a) else a, None if np.isposinf(b) else b)
        for a, b in zip(lo, hi)
    ]
    out = minimize(
        lambda d: g @ d + 0.5 * d @ qp.dense_b @ d,
        qp.d0,
        jac=lambda d: g + qp.dense_b @ d,
        bounds=bounds if with_bounds else None,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert out.success, out.message
    return out.x


def scipy_reference(qp: RandomQP) -> np.ndarray:
    """SLSQP on the full QP with ``bounds=``."""
    return scipy_qp(qp)


def theta(qp: RandomQP, dual: Dual) -> float:
    """Dual function of the bounds alone at the bound multipliers of ``dual``.

    ``θ(μ) = min_{d ∈ G} q(d) + μ_lbᵀ(lo - d) + μ_ubᵀ(d - hi)`` with ``G`` the
    set of steps satisfying the general constraints (dense, tiny ``n``).
    """
    mu_lb, mu_ub = np.asarray(dual.lb_multipliers), np.asarray(dual.ub_multipliers)
    lo, hi = qp.lb - qp.x, qp.ub - qp.x
    d = scipy_qp(qp, g=qp.g - mu_lb + mu_ub, with_bounds=False)
    const = np.sum(mu_lb[np.isfinite(lo)] * lo[np.isfinite(lo)]) - np.sum(
        mu_ub[np.isfinite(hi)] * hi[np.isfinite(hi)]
    )
    return float((qp.g - mu_lb + mu_ub) @ d + 0.5 * d @ qp.dense_b @ d + const)


def assert_kkt(qp: RandomQP, d: np.ndarray, dual: Dual, tol: float = KKT_TOL) -> int:
    """Primal feasibility, stationarity, dual signs and complementarity.

    Returns the number of bounds that carry a positive multiplier.
    """
    lam_e, lam_i = np.asarray(dual.eq_multipliers), np.asarray(dual.ineq_multipliers)
    lam_lb, lam_ub = np.asarray(dual.lb_multipliers), np.asarray(dual.ub_multipliers)
    bd = qp.dense_b @ d
    scale = max(1.0, np.linalg.norm(qp.g), np.linalg.norm(bd))
    resid = bd + qp.g + qp.a_eq.T @ lam_e + qp.a_ineq.T @ lam_i - lam_lb + lam_ub
    assert np.max(np.abs(resid), initial=0.0) <= tol * scale
    # Primal feasibility (box, equalities, inequalities).
    gap_lo, gap_hi = qp.x + d - qp.lb, qp.ub - qp.x - d
    assert np.all(gap_lo >= -1e-9) and np.all(gap_hi >= -1e-9)
    np.testing.assert_allclose(qp.a_eq @ d - qp.b_eq, 0.0, atol=1e-8)
    slack = qp.b_ineq - qp.a_ineq @ d
    assert np.all(slack >= -1e-8)
    # Signs: every multiplier is non-negative, also on lb == ub (split pair).
    assert np.all(lam_i >= -tol * scale)
    assert np.all(lam_lb >= -tol * scale)
    assert np.all(lam_ub >= -tol * scale)
    # Complementarity (an infinite face has gap inf and multiplier 0).
    assert np.all(lam_lb[np.isinf(gap_lo)] == 0.0)
    assert np.all(lam_ub[np.isinf(gap_hi)] == 0.0)
    finite_lo, finite_hi = np.isfinite(gap_lo), np.isfinite(gap_hi)
    assert np.all(np.abs(lam_lb[finite_lo] * gap_lo[finite_lo]) <= tol * scale)
    assert np.all(np.abs(lam_ub[finite_hi] * gap_hi[finite_hi]) <= tol * scale)
    assert np.all(np.abs(lam_i * slack) <= tol * scale)
    return int(np.sum(lam_lb > 1e-9) + np.sum(lam_ub > 1e-9))


def assert_converged(res: Run) -> None:
    state = res.state
    assert bool(state.qp_result == CONVERGED)
    assert bool(state.success)
    assert bool(state.status == RESULTS.successful)
    assert bool(state.last_ldp_status == LDP_RESULTS.feasible)
    assert bool(state.last_kkt_reason == KKT_SOLVER_RESULTS.converged)
    assert int(state.last_n_iter) >= 1


def assert_in_box(qp: RandomQP, d: np.ndarray, atol: float = 1e-12) -> None:
    assert np.all(qp.x + d >= qp.lb - atol) and np.all(qp.x + d <= qp.ub + atol)


# ---------------------------------------------------------------------------
# Plain-Python reference of the PLAN 3.5 rev 11 loop (same jitted oracle)
# ---------------------------------------------------------------------------


class RefLoop(NamedTuple):
    qp_result: object
    n_iter: int
    active_lb: np.ndarray
    active_ub: np.ndarray
    step: np.ndarray


def reference_loop(qp: RandomQP, max_iter: int = 100, tol: float = 1e-8) -> RefLoop:
    """The dual bound loop of PLAN 3.5 rev 11 in NumPy, one oracle call per pass.

    Asserts its own invariants: ``μ ≥ 0`` off the ``lb == ub`` pins after every
    dual step, and the values ``q(d̂)`` of the full steps never decrease.
    """
    lag, factors = qp.lagrangian, qp.factors
    n = qp.x.shape[0]
    tau = max(tol, 100.0 * float(np.finfo(qp.x.dtype).eps))
    lo = np.where(np.isfinite(qp.lb), qp.lb - qp.x, -np.inf)
    hi = np.where(np.isfinite(qp.ub), qp.ub - qp.x, np.inf)
    fixed = np.isfinite(qp.lb) & np.isfinite(qp.ub) & (qp.lb == qp.ub)
    a_lb, a_ub = fixed.copy(), np.zeros(n, bool)
    mu_lb, mu_ub = np.zeros(n), np.zeros(n)
    fresh_lb, fresh_ub = np.zeros(n, bool), np.zeros(n, bool)
    p, p_ub = -1, False
    d_full, last_q = np.zeros(n), -np.inf

    def done(result, it):
        return RefLoop(result, it, a_lb, a_ub, np.clip(d_full, lo, hi))

    for it in range(1, max_iter + 1):
        res = _oracle(lag, factors, jnp.asarray(a_lb), jnp.asarray(a_ub))
        drop_lb, drop_ub = a_lb & ~fixed, a_ub.copy()
        if bool(res.ldp_status == LDP_RESULTS.feasible):
            mh_lb = np.asarray(res.dual.lb_multipliers)
            mh_ub = np.asarray(res.dual.ub_multipliers)
            scale = max(
                1.0, np.abs(np.r_[mh_lb[drop_lb], mh_ub[drop_ub]]).max(initial=0.0)
            )
            neg_lb = drop_lb & (mh_lb < -tau * scale)
            neg_ub = drop_ub & (mh_ub < -tau * scale)
            if not (neg_lb.any() or neg_ub.any()):  # full step
                d = np.asarray(res.step)
                d_full = d
                q = objective(qp, d)
                assert q >= last_q - 1e-9 * max(1.0, abs(q)), "full-step value fell"
                last_q = q
                mu_lb = np.where(
                    a_lb, np.where(fixed, mh_lb, np.maximum(mh_lb, 0)), 0.0
                )
                mu_ub = np.where(a_ub, np.maximum(mh_ub, 0), 0.0)
                free = ~a_lb & ~a_ub
                viol_lb = np.where(free & np.isfinite(lo), lo - d, -np.inf)
                viol_ub = np.where(free & np.isfinite(hi), d - hi, -np.inf)
                v_lb = viol_lb > tau * np.maximum(1.0, np.abs(lo))
                v_ub = (viol_ub > tau * np.maximum(1.0, np.abs(hi))) & ~v_lb
                if not (v_lb.any() or v_ub.any()):
                    return done(CONVERGED, it)
                m_lb = np.where(v_lb, viol_lb, -np.inf)
                m_ub = np.where(v_ub, viol_ub, -np.inf)
                p_ub = bool(m_ub.max() > m_lb.max())
                p = int(np.argmax(m_ub if p_ub else m_lb))
                a_lb, a_ub = a_lb | v_lb, a_ub | v_ub
                fresh_lb, fresh_ub = v_lb.copy(), v_ub.copy()
                continue
            dir_lb, dir_ub = mh_lb - mu_lb, mh_ub - mu_ub
        elif bool(res.has_ray):
            dir_lb, dir_ub = np.asarray(res.ray_lb), np.asarray(res.ray_ub)
            r_scale = max(
                np.abs(dir_lb[drop_lb]).max(initial=0.0),
                np.abs(dir_ub[drop_ub]).max(initial=0.0),
            )
            neg_lb = drop_lb & (dir_lb < -tau * r_scale)
            neg_ub = drop_ub & (dir_ub < -tau * r_scale)
            if not (neg_lb.any() or neg_ub.any()):
                return done(FAILURE, it)  # certified infeasible
        else:
            return done(FAILURE, it)  # oracle failure
        # Dual step along ``dir`` until the first negative droppable pin hits 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_lb = np.where(neg_lb, mu_lb / -dir_lb, np.inf)
            ratio_ub = np.where(neg_ub, mu_ub / -dir_ub, np.inf)
        t = min(ratio_lb.min(), ratio_ub.min())
        blk_lb = neg_lb & (ratio_lb <= t * (1 + tau))
        blk_ub = neg_ub & (ratio_ub <= t * (1 + tau))
        if t > 0:
            mu_lb = np.where(a_lb, mu_lb + t * dir_lb, 0.0)
            mu_ub = np.where(a_ub, mu_ub + t * dir_ub, 0.0)
            fresh_lb, fresh_ub = np.zeros(n, bool), np.zeros(n, bool)
            p = -1
        else:  # degenerate: keep the most violated pin of the last addition
            if p >= 0 and (blk_ub if p_ub else blk_lb)[p]:
                blk_lb, blk_ub = fresh_lb.copy(), fresh_ub.copy()
                (blk_ub if p_ub else blk_lb)[p] = False
            if not (blk_lb.any() or blk_ub.any()):
                return done(STALL, it)
        assert np.all(mu_lb[a_lb & ~fixed] >= -1e-9 * max(1.0, np.abs(mu_lb).max()))
        assert np.all(mu_ub[a_ub] >= -1e-9 * max(1.0, np.abs(mu_ub).max()))
        a_lb, a_ub = a_lb & ~blk_lb, a_ub & ~blk_ub
        mu_lb = np.where(a_lb, np.where(fixed, mu_lb, np.maximum(mu_lb, 0)), 0.0)
        mu_ub = np.where(a_ub, np.maximum(mu_ub, 0), 0.0)
        fresh_lb, fresh_ub = fresh_lb & a_lb, fresh_ub & a_ub
    return done(BUDGET, max_iter)


def assert_matches_reference(qp: RandomQP) -> None:
    """The solver takes the reference's path: outcome, iterations, set, step."""
    ref = reference_loop(qp)
    res = run(qp)
    assert bool(res.state.qp_result == ref.qp_result)
    assert int(res.state.last_n_iter) == ref.n_iter
    np.testing.assert_array_equal(
        np.asarray(res.state.active_set.active_lb), ref.active_lb
    )
    np.testing.assert_array_equal(
        np.asarray(res.state.active_set.active_ub), ref.active_ub
    )
    np.testing.assert_allclose(res.step, ref.step, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Reference agreement and KKT
# ---------------------------------------------------------------------------


def check_against_reference(qp: RandomQP) -> Run:
    res = run(qp)
    assert_converged(res)
    d_ref = scipy_reference(qp)
    np.testing.assert_allclose(
        res.step, d_ref, atol=REF_TOL * max(1.0, np.linalg.norm(d_ref))
    )
    assert_kkt(qp, res.step, res.dual)
    return res


class TestAgainstDenseReference:
    @pytest.mark.parametrize("mix", MIXES, ids=MIX_IDS)
    @pytest.mark.parametrize("x_on_bound", [False, True], ids=["interior", "on-bound"])
    @pytest.mark.parametrize("frac", [0.0, 0.3, 0.8])
    @pytest.mark.parametrize("n", [8, 20])
    @pytest.mark.parametrize("seed", [1, 2])
    def test_matches_scipy_and_kkt(self, n, frac, x_on_bound, mix, seed):
        qp = make_case(n, frac, x_on_bound, mix, seed=seed)
        res = check_against_reference(qp)
        n_active = assert_kkt(qp, res.step, res.dual)
        # The cutting bounds are (mostly) active; without them (and without
        # variables starting on a bound) nothing is.
        assert (n_active >= 1) == (frac > 0) or x_on_bound

    @pytest.mark.parametrize(
        "variant",
        [
            dict(n_fixed=2),
            dict(one_sided=True),
            dict(n_fixed=2, one_sided=True),
        ],
        ids=["lb==ub", "one-sided", "lb==ub+one-sided"],
    )
    @pytest.mark.parametrize(
        "mix", [(0, 0), (2, 3), (3, 2)], ids=["0-0", "2-3", "3dup-2"]
    )
    @pytest.mark.parametrize("x_on_bound", [False, True], ids=["interior", "on-bound"])
    @pytest.mark.parametrize("frac", [0.3, 0.8])
    @pytest.mark.parametrize("n", [8, 20])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_bound_variants(self, n, frac, x_on_bound, mix, variant, seed):
        """``lb == ub`` pins, one-sided and infinite bounds."""
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        res = check_against_reference(qp)
        fixed = qp.lb == qp.ub
        # A variable with lb == ub is never released and sits at its value.
        np.testing.assert_allclose((qp.x + res.step)[fixed], qp.lb[fixed], atol=1e-12)
        active = res.state.active_set
        assert np.all(np.asarray(active.active_lb)[fixed])
        # The split multipliers of lb == ub pins are non-negative and net to rho.
        lam_lb, lam_ub = (
            np.asarray(res.dual.lb_multipliers),
            np.asarray(res.dual.ub_multipliers),
        )
        assert np.all(lam_lb[fixed] >= 0) and np.all(lam_ub[fixed] >= 0)
        assert np.all(lam_lb[fixed] * lam_ub[fixed] == 0)
        # Infinite bounds are never pinned.
        assert not np.any(np.asarray(active.active_lb)[~np.isfinite(qp.lb)])
        assert not np.any(np.asarray(active.active_ub)[~np.isfinite(qp.ub)])

    @pytest.mark.parametrize("tight", [1, 3])
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("n", [8, 20])
    def test_degenerate_vertex_from_pinned_bounds(self, n, seed, tight):
        """All but 3 variables pinned by ``lb == ub``; tight rows in the rest.

        The free subspace has dimension 3 with 2 equalities and ``tight``
        inequalities active at the pinned point ``d0``: the feasible set of
        the reduced LDP has an empty interior (a point or a half-line).
        """
        lower = index_mask(n, list(range(3, n)))
        qp = make_random_lrh_qp(
            n,
            3,
            2,
            3,
            seed=seed,
            memory=MEMORY,
            tight=tight,
            lower=lower,
            upper=lower,
        )
        # The free variables get loose bounds around ``x`` (``x`` is in the box).
        loose = 4.0 + np.random.default_rng(seed).random(n)
        qp = with_bounds_and_secant(
            qp,
            np.where(lower, qp.lb, qp.x - loose),
            np.where(lower, qp.ub, qp.x + loose),
        )
        res = run(qp)
        assert_converged(res)
        # Convex QP: a KKT point is the solution.
        assert_kkt(qp, res.step, res.dual)
        np.testing.assert_allclose(
            res.step[lower], (qp.lb - qp.x)[lower], atol=1e-12, rtol=0
        )

    def test_unconstrained_step_is_newton_step(self):
        """No bound, no constraint: ``d = -B⁻¹ g`` in one iteration."""
        qp = make_case(8, 0.0, False, (0, 0), seed=3, one_sided=True)
        qp = with_bounds_and_secant(qp, np.full(8, -np.inf), np.full(8, np.inf))
        res = run(qp)
        assert_converged(res)
        np.testing.assert_allclose(
            res.step, -np.linalg.solve(qp.dense_b, qp.g), atol=1e-10
        )
        assert int(res.state.last_n_iter) == 1


# ---------------------------------------------------------------------------
# Batteries of the dual bound loop (PLAN 3.5 rev 11): no false failure on a
# feasible QP, certified infeasibility, x outside the box, few iterations.
# ---------------------------------------------------------------------------


def lp_feasible(qp: RandomQP) -> bool:
    """Whether ``{d : general rows, lo <= d <= hi}`` is non-empty (LP oracle)."""
    n = qp.x.shape[0]
    out = linprog(
        np.zeros(n),
        A_ub=qp.a_ineq if qp.a_ineq.size else None,
        b_ub=qp.b_ineq if qp.a_ineq.size else None,
        A_eq=qp.a_eq if qp.a_eq.size else None,
        b_eq=qp.b_eq if qp.a_eq.size else None,
        bounds=[
            (None if np.isneginf(a) else a, None if np.isposinf(b) else b)
            for a, b in zip(qp.lb - qp.x, qp.ub - qp.x)
        ],
        method="highs",
    )
    return out.status == 0


def near_infeasible_case(n: int, seed: int, feasible: bool) -> RandomQP:
    """Heavy-cut QP plus a row ``cᵀd ≥ c_max ∓ 5 %`` (feasible / infeasible).

    ``c_max`` is the LP maximum of ``cᵀd`` over the feasible set, so the extra
    row cuts the polytope to a sliver (or to nothing).
    """
    qp = make_case(n, 0.8, seed % 2 == 0, (2, 3), seed=seed)
    c = np.random.default_rng(seed).standard_normal(n)
    out = linprog(
        -c,
        A_eq=qp.a_eq,
        b_eq=qp.b_eq,
        A_ub=qp.a_ineq,
        b_ub=qp.b_ineq,
        bounds=list(zip(qp.lb - qp.x, qp.ub - qp.x)),
        method="highs",
    )
    c_max = -out.fun
    target = c_max + (-0.05 if feasible else 0.05) * (1 + abs(c_max))
    return with_rows(
        qp,
        qp.a_eq,
        qp.b_eq,
        np.vstack([qp.a_ineq, -c[None]]),
        np.concatenate([qp.b_ineq, [-target]]),
    )


def x_outside_case(n: int, mix: tuple[int, int], seed: int) -> RandomQP:
    """A QP whose box excludes ``x`` on ``n // 5`` coordinates (still feasible).

    The box of a coordinate with anchor ``d_a`` away from 0 is moved so that
    ``lo = d_a / 2 > 0`` (``x`` below its lower bound) or ``hi = d_a / 2 < 0``.
    """
    qp = make_case(n, 0.5, False, mix, seed=seed)
    d_a = qp.d0
    cand = np.flatnonzero(np.abs(d_a) > 0.05)
    k = max(1, n // 5)
    idx = np.random.default_rng(seed).choice(cand, size=k, replace=False)
    lb, ub = qp.lb.copy(), qp.ub.copy()
    for i in idx:
        if d_a[i] > 0:
            lb[i] = qp.x[i] + 0.5 * d_a[i]
        else:
            ub[i] = qp.x[i] + 0.5 * d_a[i]
    out = with_bounds_and_secant(qp, lb, ub)
    assert np.any(out.x < out.lb) or np.any(out.x > out.ub)
    return out


class TestBatteries:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("x_on_bound", [False, True], ids=["interior", "on-bound"])
    @pytest.mark.parametrize(
        "mix", [(2, 3), (3, 2), (0, 3), (2, 0)], ids=["2-3", "3dup-2", "0-3", "2-0"]
    )
    @pytest.mark.parametrize("n", [8, 20])
    def test_heavy_cut_has_no_false_failure(self, n, mix, x_on_bound, seed):
        """80 % of the bounds cut the unconstrained solution (incl. the seeds
        that over-pinned the former primal loop)."""
        check_against_reference(make_case(n, 0.8, x_on_bound, mix, seed=seed))

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("n", [8, 20])
    def test_near_infeasible_feasible_side_converges(self, n, seed):
        """The LP says feasible: the QP converges with KKT multipliers."""
        qp = near_infeasible_case(n, seed, True)
        assert lp_feasible(qp)
        res = run(qp)
        assert_converged(res)
        assert_kkt(qp, res.step, res.dual)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("n", [8, 20])
    def test_near_infeasible_infeasible_side_is_certified(self, n, seed):
        """The LP says infeasible: a certified ``kkt_solver_failure``."""
        qp = near_infeasible_case(n, seed, False)
        assert not lp_feasible(qp)
        res = run(qp)
        state = res.state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert bool(state.status == RESULTS.singular)
        assert bool(
            (state.last_ldp_status == LDP_RESULTS.infeasible)
            | (state.last_ldp_status == LDP_RESULTS.inconsistent_equalities)
        )
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)
        assert_in_box(qp, res.step)

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize(
        "mix", [(2, 3), (3, 2), (0, 3)], ids=["2-3", "3dup-2", "0-3"]
    )
    @pytest.mark.parametrize("n", [8, 20])
    def test_x_outside_the_box(self, n, mix, seed):
        """``x`` below / above some bounds: the step must reach into the box."""
        qp = x_outside_case(n, mix, seed)
        res = check_against_reference(qp)
        assert_in_box(qp, res.step)
        # Coordinates whose box excludes x are forced to move inside.
        outside = (qp.x < qp.lb) | (qp.x > qp.ub)
        assert np.all(res.step[outside] != 0.0)

    @pytest.mark.parametrize("frac", [0.3, 0.8])
    @pytest.mark.parametrize("n", [200])
    def test_iteration_count_is_independent_of_active_bounds(self, n, frac):
        """Batch addition: a handful of oracle calls at ``n = 200`` and
        ``m_c = 13`` (one pin per iteration would need about 30-150)."""
        qp = make_case(n, frac, True, (5, 8), seed=0, rank=4)
        res = run(qp)
        assert_converged(res)
        assert int(res.state.last_n_iter) <= 10
        assert_kkt(qp, res.step, res.dual)

    @pytest.mark.parametrize("seed", [0, 1])
    @pytest.mark.parametrize("x_on_bound", [False, True], ids=["interior", "on-bound"])
    @pytest.mark.parametrize(
        "mix", [(2, 3), (3, 2), (0, 3), (2, 0)], ids=["2-3", "3dup-2", "0-3", "2-0"]
    )
    def test_float32_heavy_cut_converges(self, mix, x_on_bound, seed):
        """``τ = max(tol, 100 eps)``: float32 does not see rounding noise."""
        ref = scipy_reference(make_case(8, 0.8, x_on_bound, mix, seed=seed))
        with jax.enable_x64(False):
            qp = make_case(8, 0.8, x_on_bound, mix, seed=seed)
            assert qp.lagrangian.x_ref.dtype == jnp.float32
            res = run(qp)
            assert_converged(res)
            assert res.step.dtype == np.float32
        np.testing.assert_allclose(
            res.step, ref, atol=1e-4 * max(1.0, np.linalg.norm(ref))
        )


class TestReferenceLoop:
    """Pins the *path* of the loop (the dual-step arithmetic included).

    A solver whose outcome is right but whose multiplier updates are wrong
    (for instance ``μ`` not moved on partial or ray steps) reaches the same
    answer in a different number of iterations; only a path-level comparison
    with the plan's loop notices.
    """

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("x_on_bound", [False, True], ids=["interior", "on-bound"])
    @pytest.mark.parametrize(
        "mix", [(2, 3), (3, 2), (0, 3), (2, 0)], ids=["2-3", "3dup-2", "0-3", "2-0"]
    )
    @pytest.mark.parametrize("n", [8, 20])
    def test_heavy_cut_path(self, n, mix, x_on_bound, seed):
        assert_matches_reference(make_case(n, 0.8, x_on_bound, mix, seed=seed))

    @pytest.mark.parametrize("seed", range(40))
    @pytest.mark.parametrize("n", [8, 20])
    @pytest.mark.parametrize("feasible", [True, False], ids=["feasible", "infeasible"])
    def test_near_infeasible_path(self, n, seed, feasible):
        """Includes the infeasible-side seeds 24 and 29 at ``n = 20``."""
        assert_matches_reference(near_infeasible_case(n, seed, feasible))

    @pytest.mark.parametrize("budget", [1, 2, 3])
    def test_budget_exit_path(self, budget):
        """A budget exit matches the reference truncated at the same budget."""
        qp = make_case(20, 0.8, True, (2, 3), seed=1)
        ref = reference_loop(qp, max_iter=budget)
        res = run(qp, LRHDirectQPSolver(max_iter=budget))
        assert bool(res.state.qp_result == ref.qp_result)
        assert int(res.state.last_n_iter) == ref.n_iter
        np.testing.assert_allclose(res.step, ref.step, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Bookkeeping of the state
# ---------------------------------------------------------------------------


class TestStateFields:
    @pytest.mark.parametrize("mix", [(0, 0), (2, 3), (0, 3)], ids=["0-0", "2-3", "0-3"])
    def test_inherited_fields_and_counters(self, mix):
        qp = make_case(8, 0.8, True, mix, seed=4)
        first = run(qp)
        assert_converged(first)
        state = first.state
        n_iter = int(state.last_n_iter)
        n_nnls = int(state.last_n_cg_iter)
        assert int(state.n_iter) == n_iter and int(state.n_cg_iter) == n_nnls
        assert int(state.n_nnls_iter) == n_nnls
        assert float(state.final_working_tol) == LRHDirectQPSolver().tol
        assert int(state.n_anti_cycling) == 0 and int(state.last_kkt_n_refinements) == 0
        # The carried dual is the returned one; inequalities in the set are active.
        for a, b in zip(jax.tree.leaves(state.dual), jax.tree.leaves(first.dual)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(
            np.asarray(state.active_set.active_inequalities),
            np.asarray(first.dual.ineq_multipliers) > 0,
        )
        pinned = np.asarray(state.active_set.active_lb | state.active_set.active_ub)
        lb_m, ub_m = (
            np.asarray(first.dual.lb_multipliers),
            np.asarray(first.dual.ub_multipliers),
        )
        assert np.all((lb_m == 0) & (ub_m == 0) | pinned)
        # Residual of the general constraints at the returned step.
        resid_e = qp.a_eq @ first.step - qp.b_eq
        resid_i = np.maximum(qp.a_ineq @ first.step - qp.b_ineq, 0.0)
        expected = np.max(np.abs(resid_e), initial=0.0) + np.max(resid_i, initial=0.0)
        np.testing.assert_allclose(
            float(state.last_kkt_feasibility_residual), expected, atol=1e-12
        )
        # Counters accumulate over solves that consume the same carry.
        second = run(qp, state=state)
        assert int(second.state.n_iter) == 2 * n_iter
        assert int(second.state.n_cg_iter) == 2 * n_nnls
        assert int(second.state.n_nnls_iter) == 2 * n_nnls
        assert int(second.state.last_n_iter) == n_iter


# ---------------------------------------------------------------------------
# Monotonicity and budget exhaustion
# ---------------------------------------------------------------------------


class TestBudgetSweep:
    @pytest.mark.parametrize(
        "n, mix, seed",
        [(8, (2, 3), 0), (20, (2, 3), 1), (20, (0, 3), 2), (8, (3, 2), 3)],
    )
    def test_dual_value_is_monotone_over_budgets(self, n, mix, seed):
        """Run the loop with budgets ``1, 2, ...``: the paths are identical.

        Every short budget exhausts (``max_iter_reached``, ``success`` false)
        with a step inside the box and finite multipliers; the dual value
        ``θ(μ_k)`` of the returned bound multipliers never decreases; the full
        budget converges to the same step.
        """
        qp = make_case(n, 0.8, True, mix, seed=seed)
        full = run(qp)
        assert_converged(full)
        k_full = int(full.state.last_n_iter)
        assert k_full >= 2
        thetas = []
        for k in range(1, k_full + 1):
            res = run(qp, LRHDirectQPSolver(max_iter=k))
            assert_in_box(qp, res.step)
            for leaf in jax.tree.leaves(res.dual):
                assert np.all(np.isfinite(np.asarray(leaf)))
            assert int(res.state.last_n_iter) == k
            exhausted = k < k_full
            assert bool(res.state.qp_result == (BUDGET if exhausted else CONVERGED))
            assert bool(res.state.success) == (not exhausted)
            assert bool(
                res.state.status
                == (RESULTS.max_steps_reached if exhausted else RESULTS.successful)
            )
            thetas.append(theta(qp, res.dual))
        np.testing.assert_allclose(res.step, full.step, atol=1e-12)
        scale = max(1.0, *map(abs, thetas))
        assert np.all(np.diff(thetas) >= -1e-8 * scale)
        # At the solution the dual value equals the primal objective.
        np.testing.assert_allclose(
            thetas[-1], objective(qp, full.step), atol=1e-6 * scale
        )

    def test_zero_budget_returns_clipped_origin(self):
        qp = make_case(8, 0.8, True, (2, 3), seed=0)
        res = run(qp, LRHDirectQPSolver(max_iter=0))
        assert bool(res.state.qp_result == BUDGET) and not bool(res.state.success)
        assert int(res.state.last_n_iter) == 0
        assert_in_box(qp, res.step)
        np.testing.assert_array_equal(
            res.step, np.clip(0.0, qp.lb - qp.x, qp.ub - qp.x)
        )
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)


# ---------------------------------------------------------------------------
# success == (qp_result == working_set_converged) on every exit path
# ---------------------------------------------------------------------------


def _contradictory_inequalities() -> RandomQP:
    """``aᵀd ≤ -1`` and ``-aᵀd ≤ -1``."""
    n = 6
    qp = make_case(n, 0.3, False, (0, 2), seed=0)
    a = np.ones((1, n))
    return with_rows(
        qp, qp.a_eq, qp.b_eq, np.concatenate([a, -a]), -np.ones(2)
    )._replace(lb=qp.lb, ub=qp.ub)


def _contradictory_equalities() -> RandomQP:
    """``aᵀd = 1`` and ``aᵀd = 2``."""
    n = 6
    qp = make_case(n, 0.3, False, (2, 0), seed=0)
    a = np.ones((1, n))
    return with_rows(
        qp, np.concatenate([a, a]), np.array([1.0, 2.0]), qp.a_ineq, qp.b_ineq
    )


def _box_too_tight() -> RandomQP:
    """``aᵀd = 10`` with every variable confined to ``|d| ≤ 0.1``."""
    n = 6
    qp = make_case(n, 0.0, False, (1, 0), seed=0)
    qp = with_rows(qp, np.ones((1, n)), np.array([10.0]), qp.a_ineq, qp.b_ineq)
    return with_bounds_and_secant(qp, qp.x - 0.1, qp.x + 0.1)


def _nonfinite_gradient() -> RandomQP:
    qp = make_case(6, 0.3, False, (2, 3), seed=0)
    lag = eqx.tree_at(
        lambda lg: lg.evaluated.grad_val,
        qp.lagrangian,
        qp.lagrangian.evaluated.grad_val.at[0].set(jnp.nan),
    )
    return qp._replace(lagrangian=lag)


def _hard_case() -> RandomQP:
    return make_case(8, 0.8, True, (2, 3), seed=0)


# name -> (QP factory, solver, expected qp_result, expected status)
EXIT_PATHS: dict[
    str, tuple[Callable[[], RandomQP], LRHDirectQPSolver, object, object]
] = {
    "converged": (_hard_case, LRHDirectQPSolver(), CONVERGED, RESULTS.successful),
    "budget": (
        _hard_case,
        LRHDirectQPSolver(max_iter=1),
        BUDGET,
        RESULTS.max_steps_reached,
    ),
    "zero-budget": (
        _hard_case,
        LRHDirectQPSolver(max_iter=0),
        BUDGET,
        RESULTS.max_steps_reached,
    ),
    "contradictory-inequalities": (
        _contradictory_inequalities,
        LRHDirectQPSolver(),
        FAILURE,
        RESULTS.singular,
    ),
    "contradictory-equalities": (
        _contradictory_equalities,
        LRHDirectQPSolver(),
        FAILURE,
        RESULTS.singular,
    ),
    "box-too-tight": (_box_too_tight, LRHDirectQPSolver(), FAILURE, RESULTS.singular),
    "nonfinite": (_nonfinite_gradient, LRHDirectQPSolver(), FAILURE, RESULTS.singular),
    "nnls-budget": (
        lambda: make_case(8, 0.0, False, (0, 3), seed=5, tight=3),
        LRHDirectQPSolver(nnls_max_iter=1),
        FAILURE,
        RESULTS.max_steps_reached,
    ),
}


class TestExitPaths:
    @pytest.mark.parametrize("name", list(EXIT_PATHS))
    def test_success_iff_converged(self, name):
        make_qp, solver, expected_result, expected_status = EXIT_PATHS[name]
        qp = make_qp()
        res = run(qp, solver)
        state = res.state
        converged = bool(state.qp_result == CONVERGED)
        assert bool(state.success) == converged
        assert bool(state.qp_result == expected_result)
        assert bool(state.status == expected_status)
        assert bool(state.status == RESULTS.successful) == converged
        # Every exit hands back a finite step inside the box (``clip(d_full)``).
        assert np.all(np.isfinite(res.step))
        assert_in_box(qp, res.step)
        assert int(state.n_anti_cycling) == 0
        assert np.all(np.isfinite(float(state.last_kkt_feasibility_residual)))

    @pytest.mark.parametrize(
        "name",
        ["contradictory-inequalities", "contradictory-equalities", "box-too-tight"],
    )
    def test_infeasible_qp_is_a_kkt_solver_failure(self, name):
        make_qp, solver, *_ = EXIT_PATHS[name]
        res = run(make_qp(), solver)
        state = res.state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert bool(
            (state.last_ldp_status == LDP_RESULTS.infeasible)
            | (state.last_ldp_status == LDP_RESULTS.inconsistent_equalities)
        )
        assert bool(state.last_kkt_reason == KKT_SOLVER_RESULTS.projector_failure)
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)
        assert not np.any(np.asarray(state.active_set.active_inequalities))

    def test_non_finite_data_is_reported(self):
        res = run(_nonfinite_gradient())
        assert bool(res.state.last_ldp_status == LDP_RESULTS.nonfinite)
        assert bool(res.state.last_kkt_reason == KKT_SOLVER_RESULTS.nonfinite)


def _one_bound_qp() -> RandomQP:
    """``n = 2``, ``B = σ I``, ``g = (10, 0)``, ``d_0 ≥ -0.5``: one active bound."""
    n = 2
    qp = make_random_lrh_qp(n, 0, 0, 0, seed=0, memory=MEMORY)
    g = np.array([10.0, 0.0])
    ev = eqx.tree_at(lambda e: e.grad_val, qp.lagrangian.evaluated, jnp.asarray(g))
    lag = eqx.tree_at(lambda lg: lg.evaluated, qp.lagrangian, ev)
    qp = qp._replace(lagrangian=lag, g=g)
    return with_bounds_and_secant(
        qp, qp.x + np.array([-0.5, -np.inf]), np.full(n, np.inf)
    )


def _patch_first_pin_multiplier(monkeypatch, transform) -> None:
    """Make the oracle report ``transform(λ_lb)`` as its bound multipliers."""
    real = lrh_direct_qp.solve_fixed_bounds_qp

    def patched(*args, **kwargs):
        res = real(*args, **kwargs)
        return eqx.tree_at(
            lambda r: r.dual.lb_multipliers,
            res,
            transform(res.dual.lb_multipliers),
        )

    monkeypatch.setattr(lrh_direct_qp, "solve_fixed_bounds_qp", patched)


def _solve_unjitted(qp: RandomQP, solver=None, state=None):
    # Not jitted: a cached compilation would not see the patched oracle.
    solver = LRHDirectQPSolver() if solver is None else solver
    state = cold_state(qp) if state is None else state
    x0 = (Primal(jnp.zeros_like(qp.lagrangian.x_ref)), qp.lagrangian.dual)
    return solver.solve(make_subproblem(qp), x0, state)


class TestStall:
    def test_degenerate_step_without_release_is_anti_cycling(self, monkeypatch):
        """A degenerate step whose only blocker is the pin just added.

        Impossible in exact arithmetic, so the oracle is made to report a
        wrong-sign multiplier for the first pin: the loop must stop with
        ``anti_cycling`` instead of looping, count it, and return the last
        full step (clipped to the box) with that step's dual.
        """
        qp = _one_bound_qp()
        _patch_first_pin_multiplier(monkeypatch, lambda m: -m)
        (step, dual), state = _solve_unjitted(qp)
        assert bool(state.qp_result == STALL)
        assert bool(state.status == RESULTS.max_steps_reached)
        assert not bool(state.success)
        assert int(state.n_anti_cycling) == 1 and int(state.last_n_iter) == 2
        # Last full step: the unconstrained step, clipped to the box.
        d_free = -np.linalg.solve(qp.dense_b, qp.g)
        np.testing.assert_allclose(
            np.asarray(step.x), np.maximum(d_free, qp.lb - qp.x), atol=1e-12
        )
        assert np.all(np.asarray(dual.lb_multipliers) == 0.0)
        assert bool(np.asarray(state.active_set.active_lb)[0])

    def test_converged_bound_multipliers_are_non_negative(self, monkeypatch):
        """A converged multiplier of ``-1e-9`` (inside ``τ``) is returned as 0."""
        qp = _one_bound_qp()
        _patch_first_pin_multiplier(
            monkeypatch, lambda m: jnp.where(m != 0, -1e-9, 0.0)
        )
        (_, dual), state = _solve_unjitted(qp)
        assert bool(state.qp_result == CONVERGED)
        assert float(dual.lb_multipliers[0]) == 0.0
        assert np.all(np.asarray(dual.lb_multipliers) >= 0)
        assert np.all(np.asarray(dual.ub_multipliers) >= 0)

    @pytest.mark.parametrize(
        "x64, expected",
        [(True, STALL), (False, CONVERGED)],
        ids=["float64", "float32"],
    )
    def test_sign_tolerance_has_a_rounding_floor(self, monkeypatch, x64, expected):
        """A multiplier of ``-1e-6`` is rounding noise in float32 only.

        ``τ = max(tol, 100 eps)`` is ``1e-8`` in float64 (the wrong-sign pin
        is released, which stalls) and ``1.2e-5`` in float32 (it is ignored).
        """
        with jax.enable_x64(x64):
            qp = _one_bound_qp()
            _patch_first_pin_multiplier(
                monkeypatch, lambda m: jnp.where(m != 0, -1e-6, 0.0)
            )
            _, state = _solve_unjitted(qp)
            assert bool(state.qp_result == expected)


# ---------------------------------------------------------------------------
# Warm start (C13)
# ---------------------------------------------------------------------------

WARM = LRHDirectQPSolver(warm_start=True)


def with_warm_set(state: LRHQPSolverState, lb_mask, ub_mask) -> LRHQPSolverState:
    """``state`` with its bound working set replaced by the given masks."""
    return eqx.tree_at(
        lambda s: (s.active_set.active_lb, s.active_set.active_ub),
        state,
        (jnp.asarray(lb_mask, bool), jnp.asarray(ub_mask, bool)),
    )


def trial_is_rejected(qp: RandomQP, lb_mask, ub_mask) -> bool:
    """Classify the warm trial independently, with the standalone oracle.

    Mirrors the solver's filtering of the warm set and its full-step test: the
    trial is rejected unless the oracle solved the pinned problem and no
    droppable multiplier is below ``-τ max(1, ‖μ̂_D‖∞)``. It depends on
    ``ldp_status`` and the multipliers only, never on ``has_ray``.
    """
    lag = qp.lagrangian
    lo, hi = make_subproblem(qp).primal_box()
    fixed = ~lag.null_lb & ~lag.null_ub & (lag.lb == lag.ub)
    warm_lb = jnp.asarray(lb_mask, bool) & jnp.isfinite(lo) & ~fixed
    warm_ub = jnp.asarray(ub_mask, bool) & jnp.isfinite(hi) & ~warm_lb & ~fixed
    a_lb = fixed | warm_lb
    res = _oracle(lag, lag.secant.factors(), a_lb, warm_ub)
    tau = max(1e-8, 100.0 * float(jnp.finfo(lag.x_ref.dtype).eps))
    m_lb = np.where(np.asarray(a_lb & ~fixed), np.asarray(res.dual.lb_multipliers), 0.0)
    m_ub = np.where(np.asarray(warm_ub), np.asarray(res.dual.ub_multipliers), 0.0)
    scale = max(
        1.0, np.max(np.abs(m_lb), initial=0.0), np.max(np.abs(m_ub), initial=0.0)
    )
    negative = np.any(m_lb < -tau * scale) | np.any(m_ub < -tau * scale)
    return bool((res.ldp_status != LDP_RESULTS.feasible) | negative)


def same_outcome_as(res: Run, cold: Run, extra_iters: int) -> bool:
    """``res`` is ``cold`` plus ``extra_iters`` iterations, otherwise identical."""
    a, b = res.state, cold.state
    return bool(
        (int(a.last_n_iter) == int(b.last_n_iter) + extra_iters)
        & (a.qp_result == b.qp_result)
        & (a.status == b.status)
        & (a.last_ldp_status == b.last_ldp_status)
        & np.allclose(res.step, cold.step, atol=1e-12, rtol=0)
        & np.all(
            [
                np.array_equal(np.asarray(u), np.asarray(v))
                for u, v in zip(
                    jax.tree.leaves(a.active_set), jax.tree.leaves(b.active_set)
                )
            ]
        )
        & np.all(
            [
                np.allclose(np.asarray(u), np.asarray(v), atol=1e-12, rtol=0)
                for u, v in zip(jax.tree.leaves(res.dual), jax.tree.leaves(cold.dual))
            ]
        )
    )


# Perturbations of the optimal warm set (lb, ub) -> (lb, ub), one per way a
# warm set can be wrong; ``rng`` is seeded per case.
def _keep(lb, ub, rng):
    return lb, ub


def _extra_pins(lb, ub, rng):
    return lb | (rng.random(lb.shape) < 0.4), ub | (rng.random(ub.shape) < 0.4)


def _missing_pins(lb, ub, rng):
    return lb & (rng.random(lb.shape) < 0.5), ub & (rng.random(ub.shape) < 0.5)


def _wrong_face(lb, ub, rng):
    return ub.copy(), lb.copy()


def _everything_pinned(lb, ub, rng):
    return np.ones_like(lb), np.ones_like(ub)


def _both_faces_marked(lb, ub, rng):
    return lb | ub, lb | ub


def _empty(lb, ub, rng):
    return np.zeros_like(lb), np.zeros_like(ub)


PERTURBATIONS = [
    _extra_pins,
    _missing_pins,
    _wrong_face,
    _everything_pinned,
    _both_faces_marked,
    _empty,
]


def _warm_cases():
    for n, frac, x_on_bound, mix, seed, variant in [
        (8, 0.8, False, (2, 3), 1, {}),
        (8, 0.8, True, (3, 2), 2, {}),
        (20, 0.3, False, (0, 3), 3, {}),
        (20, 0.8, True, (2, 0), 4, {}),
        (8, 0.8, False, (2, 3), 5, dict(n_fixed=2)),
        (20, 0.8, True, (2, 3), 6, dict(one_sided=True)),
        (8, 0.3, False, (0, 0), 7, dict(n_fixed=2, one_sided=True)),
    ]:
        yield pytest.param(
            n,
            frac,
            x_on_bound,
            mix,
            seed,
            variant,
            id=f"{n}-{frac}-{x_on_bound}-{seed}",
        )


class TestWarmStart:
    @pytest.mark.parametrize("n, frac, x_on_bound, mix, seed, variant", _warm_cases())
    def test_optimal_set_converges_in_one_iteration(
        self, n, frac, x_on_bound, mix, seed, variant
    ):
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        assert_converged(cold)
        res = run(qp, WARM, cold.state)
        assert_converged(res)
        assert int(res.state.last_n_iter) == 1
        np.testing.assert_allclose(res.step, cold.step, atol=1e-10, rtol=0)
        for a, b in zip(jax.tree.leaves(res.dual), jax.tree.leaves(cold.dual)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)
        np.testing.assert_array_equal(
            np.asarray(res.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_lb),
        )
        # Counters advance by the one trial iteration on top of the cold solve.
        assert int(res.state.n_iter) == int(cold.state.n_iter) + 1

    @pytest.mark.parametrize("perturb", PERTURBATIONS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("n, frac, x_on_bound, mix, seed, variant", _warm_cases())
    def test_wrong_set_still_reaches_the_solution(
        self, n, frac, x_on_bound, mix, seed, variant, perturb
    ):
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        lb, ub = perturb(
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
            np.random.default_rng(seed),
        )
        state = with_warm_set(cold_state(qp), lb, ub)
        res = run(qp, WARM, state)
        assert_converged(res)
        np.testing.assert_allclose(res.step, cold.step, atol=1e-6, rtol=0)
        assert_kkt(qp, res.step, res.dual)
        assert_in_box(qp, res.step)
        # A rejected trial is discarded: the cold run plus one iteration.
        rejected = trial_is_rejected(qp, lb, ub)
        assert (not rejected) or same_outcome_as(res, cold, 1)

    @pytest.mark.parametrize("n, frac, x_on_bound, mix, seed, variant", _warm_cases())
    def test_cold_solver_ignores_the_state_working_set(
        self, n, frac, x_on_bound, mix, seed, variant
    ):
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        junk = with_warm_set(
            cold_state(qp),
            np.ones(n, bool),
            np.random.default_rng(seed).random(n) < 0.5,
        )
        res = run(qp, LRHDirectQPSolver(warm_start=False), junk)
        np.testing.assert_array_equal(res.step, cold.step)
        assert int(res.state.last_n_iter) == int(cold.state.last_n_iter)
        assert int(res.state.last_n_cg_iter) == int(cold.state.last_n_cg_iter)
        for a, b in zip(jax.tree.leaves(res.dual), jax.tree.leaves(cold.dual)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    @pytest.mark.parametrize(
        "perturb", [_keep, *PERTURBATIONS], ids=lambda f: f.__name__
    )
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("n", [8, 20])
    def test_infeasible_qp_gives_the_cold_certified_outcome(self, n, seed, perturb):
        """The base set is the cold run's final (certifying) working set."""
        qp = near_infeasible_case(n, seed, False)
        cold = run(qp)
        assert bool(cold.state.qp_result == FAILURE)
        lb, ub = perturb(
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
            np.random.default_rng(seed),
        )
        res = run(qp, WARM, with_warm_set(cold_state(qp), lb, ub))
        state = res.state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert bool(state.status == cold.state.status)
        assert bool(state.last_ldp_status == cold.state.last_ldp_status)
        for leaf in jax.tree.leaves(res.dual):
            assert np.all(np.asarray(leaf) == 0.0)
        assert_in_box(qp, res.step)
        rejected = trial_is_rejected(qp, lb, ub)
        assert (not rejected) or same_outcome_as(res, cold, 1)

    @pytest.mark.parametrize("budget", [0, 1, 2])
    def test_trial_consumes_budget(self, budget):
        qp = _hard_case()
        cold = run(qp)
        res = run(qp, LRHDirectQPSolver(warm_start=True, max_iter=budget), cold.state)
        assert int(res.state.last_n_iter) == min(budget, 1)
        assert bool(res.state.success) == (budget >= 1)
        assert_in_box(qp, res.step)

    @pytest.mark.parametrize("perturb", PERTURBATIONS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("n, frac, x_on_bound, mix, seed, variant", _warm_cases())
    def test_rejected_trial_with_one_iteration_budget(
        self, n, frac, x_on_bound, mix, seed, variant, perturb
    ):
        """A rejected trial uses the whole budget: clipped origin, no dual."""
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        lb, ub = perturb(
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
            np.random.default_rng(seed),
        )
        res = run(
            qp,
            LRHDirectQPSolver(warm_start=True, max_iter=1),
            with_warm_set(cold_state(qp), lb, ub),
        )
        state = res.state
        rejected = trial_is_rejected(qp, lb, ub)
        zero_dual = all(
            np.all(np.asarray(leaf) == 0.0) for leaf in jax.tree.leaves(res.dual)
        )
        origin = np.clip(0.0, qp.lb - qp.x, qp.ub - qp.x)
        budget_exit = (
            bool(state.qp_result == BUDGET)
            and bool(state.status == RESULTS.max_steps_reached)
            and not bool(state.success)
            and int(state.last_n_iter) == 1
            and zero_dual
            and np.array_equal(res.step, origin)
        )
        assert (not rejected) or budget_exit
        assert_in_box(qp, res.step)

    def test_both_outcomes_occur_in_the_battery(self):
        """Guards the implications above against being vacuous."""
        flags = []
        for case in _warm_cases():
            n, frac, x_on_bound, mix, seed, variant = case.values
            qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
            cold = run(qp)
            base = (
                np.asarray(cold.state.active_set.active_lb),
                np.asarray(cold.state.active_set.active_ub),
            )
            for perturb in PERTURBATIONS:
                lb, ub = perturb(*base, np.random.default_rng(seed))
                flags.append(trial_is_rejected(qp, lb, ub))
        assert any(flags) and not all(flags)

    @pytest.mark.parametrize("n, frac, x_on_bound, mix, seed, variant", _warm_cases())
    def test_oracle_failure_of_the_trial_is_not_reported(
        self, monkeypatch, n, frac, x_on_bound, mix, seed, variant
    ):
        """An oracle failure on the warm set must not become a QP failure.

        The oracle is made to fail (``max_iter_reached``, no certificate) on
        the everything-pinned set, which the cold path never visits.
        """
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        real = lrh_direct_qp.solve_fixed_bounds_qp

        def patched(lag, factors, a_lb, a_ub, **kwargs):
            res = real(lag, factors, a_lb, a_ub, **kwargs)
            bad = jnp.all(a_lb)
            return eqx.tree_at(
                lambda r: (r.ldp_status, r.has_ray),
                res,
                (
                    LDP_RESULTS.where(
                        bad, LDP_RESULTS.max_iter_reached, res.ldp_status
                    ),
                    res.has_ray & ~bad,
                ),
            )

        monkeypatch.setattr(lrh_direct_qp, "solve_fixed_bounds_qp", patched)
        state = with_warm_set(cold_state(qp), np.ones(n, bool), np.zeros(n, bool))
        (step, dual), out = _solve_unjitted(qp, WARM, state)
        assert_converged(Run(np.asarray(step.x), dual, out))
        np.testing.assert_allclose(np.asarray(step.x), cold.step, atol=1e-9, rtol=0)
        assert int(out.last_n_iter) == int(cold.state.last_n_iter) + 1

    @pytest.mark.parametrize("n, frac, x_on_bound, mix, seed, variant", _warm_cases())
    def test_warm_filters_keep_the_optimal_set_optimal(
        self, n, frac, x_on_bound, mix, seed, variant
    ):
        """Infinite faces are never pinned and the lower face wins ties.

        The optimal set plus marks on every infinite face and on the upper
        face of every lower-pinned variable is still accepted as is, so it
        converges in one iteration.
        """
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        lb = np.asarray(cold.state.active_set.active_lb)
        ub = np.asarray(cold.state.active_set.active_ub)
        noisy_lb = lb | ~np.isfinite(qp.lb)
        noisy_ub = ub | ~np.isfinite(qp.ub) | lb
        res = run(qp, WARM, with_warm_set(cold_state(qp), noisy_lb, noisy_ub))
        assert_converged(res)
        assert int(res.state.last_n_iter) == 1
        np.testing.assert_allclose(res.step, cold.step, atol=1e-10, rtol=0)
        # The working set is the filtered one, not the noisy one.
        for face in ("active_lb", "active_ub"):
            np.testing.assert_array_equal(
                np.asarray(getattr(res.state.active_set, face)),
                np.asarray(getattr(cold.state.active_set, face)),
            )

    def test_single_compilation_across_warm_sets(self):
        traces = []

        @jax.jit
        def solve(sub, state):
            traces.append(1)
            lag = sub.lagrangian
            return WARM.solve(sub, (Primal(jnp.zeros_like(lag.x_ref)), lag.dual), state)

        for seed in range(3):
            qp = make_case(8, 0.8, True, (2, 3), seed=seed)
            state = with_warm_set(
                cold_state(qp),
                np.random.default_rng(seed).random(8) < 0.5,
                np.random.default_rng(seed + 9).random(8) < 0.5,
            )
            (step, _), out = solve(make_subproblem(qp), state)
            assert bool(out.success)
            np.testing.assert_allclose(
                np.asarray(step.x), run(qp).step, atol=1e-6, rtol=0
            )
        assert len(traces) == 1


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class NewtonViewQP(ActiveSetSubProblem):
    """Working-set QP declaring the Newton view (dual block is ``Δλ``)."""

    @property
    def is_kkt_dual_increment(self) -> bool:
        return True


class TestValidation:
    def test_rejects_newton_view(self):
        qp = make_case(6, 0.3, False, (2, 3))
        sub = make_subproblem(qp)
        newton = NewtonViewQP(sub.lagrangian, sub.active_set)
        with pytest.raises(TypeError, match="SQP-view"):
            LRHDirectQPSolver().solve(
                newton, (Primal(jnp.zeros(6)), sub.lagrangian.dual), cold_state(qp)
            )

    @pytest.mark.parametrize(
        "replace_secant",
        [
            lambda lag: lag.without_secant(),
            lambda lag: eqx.tree_at(lambda lg: lg.secant, lag, LBFGS(n=6, memory=3)),
        ],
        ids=["none", "lbfgs"],
    )
    def test_rejects_non_reduced_hessian_secant(self, replace_secant):
        qp = make_case(6, 0.3, False, (2, 3))
        lag = replace_secant(qp.lagrangian)
        sub = ActiveSetSubProblem(lag, make_empty_active_set(6, 2, 3))
        with pytest.raises(TypeError, match="ReducedHessianSecant"):
            LRHDirectQPSolver().solve(
                sub, (Primal(jnp.zeros(6)), lag.dual), cold_state(qp)
            )

    def test_requires_secant(self):
        assert LRHDirectQPSolver().requires_secant()


# ---------------------------------------------------------------------------
# jit, dtype stability, memory
# ---------------------------------------------------------------------------


def _state_dtypes(state) -> list:
    return [leaf.dtype for leaf in jax.tree.leaves(state)]


class TestJit:
    @pytest.mark.parametrize(
        "x64, dtype, atol",
        [(True, jnp.float64, 1e-10), (False, jnp.float32, 1e-5)],
        ids=["float64", "float32"],
    )
    def test_inside_while_loop_and_dtype_stable(self, x64, dtype, atol):
        """``solve`` inside a ``lax.while_loop`` whose carry is the state.

        ``while_loop`` rejects a carry whose dtypes or structure change, so a
        successful run proves the state is dtype-stable; the result must
        also match a direct solve.
        """
        with jax.enable_x64(x64):
            qp = make_case(8, 0.8, True, (2, 3), seed=6)
            sub = make_subproblem(qp)
            state0 = cold_state(qp)
            solver = LRHDirectQPSolver()
            assert qp.lagrangian.x_ref.dtype == dtype

            @jax.jit
            def looped(sub, state):
                lag = sub.lagrangian
                x0 = (
                    Primal(jnp.zeros_like(lag.x_ref)),
                    jax.tree.map(jnp.zeros_like, lag.dual),
                )

                def body(carry):
                    i, st, _ = carry
                    (step, _dual), st = solver.solve(sub, x0, st)
                    return i + 1, st, step.x

                def cond(carry):
                    return carry[0] < 2

                return jax.lax.while_loop(
                    cond, body, (jnp.asarray(0), state, jnp.zeros_like(lag.x_ref))
                )

            _, state_out, step_out = looped(sub, state0)
            assert _state_dtypes(state_out) == _state_dtypes(state0)
            assert step_out.dtype == dtype
            direct = run(qp, solver)
            assert _state_dtypes(direct.state) == _state_dtypes(state0)
            np.testing.assert_allclose(np.asarray(step_out), direct.step, atol=atol)
            assert int(state_out.n_iter) == 2 * int(direct.state.last_n_iter)

    def test_single_compilation_across_data(self):
        traces = []

        @jax.jit
        def solve(sub, state):
            traces.append(1)
            lag = sub.lagrangian
            x0 = (jnp.zeros_like(lag.x_ref), None)
            return LRHDirectQPSolver().solve(sub, (Primal(x0[0]), lag.dual), state)

        for seed in range(3):
            qp = make_case(8, 0.8, True, (2, 3), seed=seed)
            (step, _), state = solve(make_subproblem(qp), cold_state(qp))
            assert bool(state.success)
        assert len(traces) == 1


class TestMemoryBudget:
    def test_no_n_by_n_or_bound_stacked_intermediate(self):
        """No array of the jaxpr has two axes as large as ``n``.

        ``n`` is distinct from every other dimension (memory 4, ``m_c = 5``)
        and the largest array stays ``n × max(m_c, m + 1)``: the ``2n``
        bounds are never stacked into a matrix.
        """
        n, m_c = 30, 5
        qp = make_case(n, 0.5, True, (2, 3), seed=1, one_sided=True)
        sub = make_subproblem(qp)
        state = cold_state(qp)
        x0 = (Primal(jnp.zeros(n)), qp.lagrangian.dual)
        solver = LRHDirectQPSolver()
        jaxpr = jax.make_jaxpr(lambda: solver.solve(sub, x0, state))()
        shapes = set(walk_avals(jaxpr.jaxpr))
        offenders = [s for s in shapes if sum(1 for d in s if d >= n) >= 2]
        assert not offenders, offenders
        largest = max(int(np.prod(s)) for s in shapes)
        assert largest <= n * max(m_c, MEMORY + 1) + n
