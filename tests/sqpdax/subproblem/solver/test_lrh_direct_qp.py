"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.lrh_direct_qp`.

Convention under test (SQP view): ``min gᵀd + ½dᵀBd`` s.t. ``A_E d = b_E``,
``A_I d ≤ b_I``, ``lb ≤ x + d ≤ ub``; stationarity
``B d + g + A_Eᵀλ_E + A_Iᵀλ_I - λ_lb + λ_ub = 0`` with ``λ_I, λ_lb, λ_ub ≥ 0``.

The dense references (``n × n`` NumPy / SciPy, tiny ``n``) live only in the
tests; the code under test never forms an ``n × n`` object.
"""

from __future__ import annotations

import functools
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
from slsqp_jax.sqpdax.logging import Logger, MemoryDiagnosticsHandler, MemoryHandler
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
from slsqp_jax.sqpdax.subproblem.solver.lrh_fixed_bounds import (
    elastic_penalty,
    solve_fixed_bounds_qp,
)
from tests.sqpdax.subproblem.solver.conftest import (
    RandomQP,
    assert_elastic_kkt,
    delta_min_lp,
    elastic_column_np,
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

# The C12-C14 tests pin the plain loop: an infeasible QP is a failure, not an
# elastic solve (the elastic mode has its own tests at the end of the file).
PLAIN = LRHDirectQPSolver(elastic=False)

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
    n_ray: int  # iterations that take a ray step (and do not stall)
    n_safeguard: int  # degenerate steps resolved by the safeguard (not stalls)


def reference_loop(
    qp: RandomQP,
    max_iter: int = 100,
    tol: float = 1e-8,
    oracle=None,
    value=None,
) -> RefLoop:
    """The dual bound loop of PLAN 3.5 rev 11 in NumPy, one oracle call per pass.

    Asserts its own invariants: ``μ ≥ 0`` off the ``lb == ub`` pins after every
    dual step, and the values ``q(d̂)`` of the full steps never decrease. It also
    counts the ray steps and the safeguard firings, neither of them including the
    stalling iteration (nor the terminal certified-infeasible one).

    ``oracle`` (default the plain jitted kernel) and ``value`` (default the plain
    objective ``q(d̂)``, called as ``value(qp, result)``) let the same loop run
    on another fixed-bounds oracle, e.g. the elastic one.
    """
    oracle = _oracle if oracle is None else oracle
    value = (lambda q, r: objective(q, np.asarray(r.step))) if value is None else value
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
    n_ray = n_safeguard = 0

    def done(result, it):
        return RefLoop(
            result, it, a_lb, a_ub, np.clip(d_full, lo, hi), n_ray, n_safeguard
        )

    for it in range(1, max_iter + 1):
        res = oracle(lag, factors, jnp.asarray(a_lb), jnp.asarray(a_ub))
        drop_lb, drop_ub = a_lb & ~fixed, a_ub.copy()
        is_ray = False
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
                q = value(qp, res)
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
            is_ray = True
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
                safeguard = True
            else:
                safeguard = False
            if not (blk_lb.any() or blk_ub.any()):
                return done(STALL, it)
            n_safeguard += safeguard
        n_ray += is_ray
        assert np.all(mu_lb[a_lb & ~fixed] >= -1e-9 * max(1.0, np.abs(mu_lb).max()))
        assert np.all(mu_ub[a_ub] >= -1e-9 * max(1.0, np.abs(mu_ub).max()))
        a_lb, a_ub = a_lb & ~blk_lb, a_ub & ~blk_ub
        mu_lb = np.where(a_lb, np.where(fixed, mu_lb, np.maximum(mu_lb, 0)), 0.0)
        mu_ub = np.where(a_ub, np.maximum(mu_ub, 0), 0.0)
        fresh_lb, fresh_ub = fresh_lb & a_lb, fresh_ub & a_ub
    return done(BUDGET, max_iter)


@functools.cache
def _shared_logged_solver() -> tuple[LRHDirectQPSolver, LogView]:
    """One logged solver for the batteries (one compilation per shape)."""
    return with_logger(PLAIN)


def assert_matches_reference(qp: RandomQP) -> None:
    """The solver takes the reference's path: outcome, iterations, set, step.

    The ``"qp"`` record's ray-step and safeguard counts equal the reference's.
    """
    ref = reference_loop(qp)
    solver, logged = _shared_logged_solver()
    res = run(qp, solver)
    values = logged.records[-1].values
    assert values["n_ray_steps"] == ref.n_ray
    assert values["n_safeguard"] == ref.n_safeguard
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


def near_infeasible_pinned_battery(
    size: int = 80, min_free: int | None = None
) -> list[tuple[RandomQP, np.ndarray, np.ndarray]]:
    """``size`` pinned problems whose elastic relaxation is itself infeasible.

    Each is ``(qp, lower, upper)`` from :func:`near_infeasible_case` (infeasible
    side) with a random pin set of fraction 0.3 to 0.9 of the variables (each
    on a random finite face). It is kept when the LP (:func:`delta_min_lp`,
    pins as ``lo == hi``, free variables unbounded) declares the relaxed rows
    infeasible for every ``δ ∈ [0, 1]``, so that the kernel must answer with a
    Farkas ray, and when at least ``min_free`` variables stay free (default
    ``m_E + 1``: with fewer, the pins nearly determine the equality rows and a
    solve may report ``feasible`` with a huge step, a regime that is not the
    subject of the batteries that use the default). Deterministic.
    """
    battery = []
    for seed in range(1000):
        for n in (8, 20):
            qp = near_infeasible_case(n, seed, False)
            n_free_min = qp.a_eq.shape[0] + 1 if min_free is None else min_free
            for frac in (0.3, 0.5, 0.7, 0.9):
                rng = np.random.default_rng(1000 * n + 10 * seed + int(10 * frac))
                pin = rng.random(n) < frac
                lower = pin & (rng.random(n) < 0.5) & np.isfinite(qp.lb)
                upper = pin & ~lower & np.isfinite(qp.ub)
                pinned = lower | upper
                if int((~pinned).sum()) < n_free_min:
                    continue
                d_fix = np.where(
                    lower, qp.lb - qp.x, np.where(upper, qp.ub - qp.x, 0.0)
                )
                lo = np.where(pinned, d_fix, -np.inf)
                hi = np.where(pinned, d_fix, np.inf)
                if delta_min_lp(qp, lo, hi) is None:
                    battery.append((qp, lower, upper))
                    if len(battery) == size:
                        return battery
    raise AssertionError("not enough pinned infeasible problems")


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
        res = run(qp, PLAIN)
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
        solver, logged = with_logger(LRHDirectQPSolver(max_iter=budget))
        res = run(qp, solver)
        values = logged.records[-1].values
        assert values["n_ray_steps"] == ref.n_ray
        assert values["n_safeguard"] == ref.n_safeguard
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
        PLAIN,
        FAILURE,
        RESULTS.singular,
    ),
    "contradictory-equalities": (
        _contradictory_equalities,
        PLAIN,
        FAILURE,
        RESULTS.singular,
    ),
    "box-too-tight": (_box_too_tight, PLAIN, FAILURE, RESULTS.singular),
    "nonfinite": (_nonfinite_gradient, PLAIN, FAILURE, RESULTS.singular),
    "nnls-budget": (
        lambda: make_case(8, 0.0, False, (0, 3), seed=5, tight=3),
        LRHDirectQPSolver(nnls_max_iter=1, elastic=False),
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
        res = run(_nonfinite_gradient(), PLAIN)
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
WARM_PLAIN = LRHDirectQPSolver(warm_start=True, elastic=False)


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
        cold = run(qp, PLAIN)
        assert bool(cold.state.qp_result == FAILURE)
        lb, ub = perturb(
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
            np.random.default_rng(seed),
        )
        res = run(qp, WARM_PLAIN, with_warm_set(cold_state(qp), lb, ub))
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
        # Infinite lower faces are never pinned, so "everything pinned" means
        # every variable with a finite lower bound.
        lo, _ = make_subproblem(qp).primal_box()

        def patched(lag, factors, a_lb, a_ub, **kwargs):
            res = real(lag, factors, a_lb, a_ub, **kwargs)
            bad = jnp.all(a_lb | ~jnp.isfinite(lo))
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
# Logging and the "qp" diagnostic record (C14)
# ---------------------------------------------------------------------------

RECORD_KEYS = {
    # Keys of the ActiveSetQPSolver "qp" record.
    "qp_result",
    "status",
    "success",
    "iters",
    "cg_iters",
    "step",
    "dual",
    "active_set",
    "qp_dual",
    "final_working_tol",
    "n_anti_cycling",
    "kkt_feasibility_residual",
    "kkt_n_refinements",
    "kkt_reason",
    # LRH-specific keys.
    "n_nnls_iter",
    "last_ldp_status",
    "warm_trial_accepted",
    "warm_trial_rejected",
    "n_ray_steps",
    "n_safeguard",
    "stalled",
    "n_pinned",
    "elastic",
    "last_elastic_delta",
    "n_elastic",
}


def with_logger(solver: LRHDirectQPSolver) -> tuple[LRHDirectQPSolver, LogView]:
    """``solver`` with a DEBUG logger and diagnostics sent to memory handlers."""
    handler, diag = MemoryHandler(), MemoryDiagnosticsHandler()
    logger = Logger.from_options(
        {"level": "DEBUG", "handler": handler, "diagnostics": diag}
    ).child("subproblem")
    return solver.init(logger=logger), LogView(handler, diag)


class LogView:
    """Live view of the handlers (callbacks fire after the solve returns)."""

    def __init__(self, handler, diag):
        self._handler, self._diag = handler, diag

    @property
    def lines(self) -> list[tuple[str, str]]:
        jax.effects_barrier()
        return [(r.levelname, r.message) for r in self._handler.records]

    @property
    def records(self) -> list:
        jax.effects_barrier()
        return list(self._diag.records)


QP_RESULT_NAMES = (
    "working_set_converged",
    "kkt_solver_failure",
    "max_iter_reached",
    "anti_cycling",
)
STATUS_NAMES = ("successful", "singular", "max_steps_reached")
LDP_NAMES = (
    "feasible",
    "infeasible",
    "inconsistent_equalities",
    "nonfinite",
    "max_iter_reached",
)
LINE_PREFIX = {"INFO": "qp solved", "WARNING": "qp failed"}


def item_name(item, enum, names) -> str:
    """Name of an enumeration item, as the record payload spells it."""
    return next(n for n in names if bool(item == getattr(enum, n)))


def leaves_equal(a, b) -> bool:
    return all(
        np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)
        for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


class TestLogging:
    @pytest.mark.parametrize("name", list(EXIT_PATHS))
    def test_record_keys_and_values(self, name):
        make_qp, solver, *_ = EXIT_PATHS[name]
        qp = make_qp()
        solver, logged = with_logger(solver)
        res = run(qp, solver)
        state = res.state
        (rec,) = logged.records
        assert rec.kind == "qp" and rec.name == "minimiser.subproblem"
        v = rec.values
        assert set(v) == RECORD_KEYS
        assert v["iters"] == int(state.last_n_iter)
        assert v["cg_iters"] == int(state.last_n_cg_iter) == v["n_nnls_iter"]
        assert v["success"] == bool(state.success)
        assert v["qp_result"] == item_name(
            state.qp_result, ACTIVE_SET_QP_RESULTS, QP_RESULT_NAMES
        )
        assert v["status"] == item_name(state.status, RESULTS, STATUS_NAMES)
        assert v["last_ldp_status"] == item_name(
            state.last_ldp_status, LDP_RESULTS, LDP_NAMES
        )
        assert v["n_anti_cycling"] == int(state.n_anti_cycling)
        assert v["stalled"] == bool(state.qp_result == STALL)
        np.testing.assert_array_equal(np.asarray(v["step"].x), res.step)
        assert v["n_pinned"] == int(
            np.sum(np.asarray(state.active_set.active_lb))
            + np.sum(np.asarray(state.active_set.active_ub))
        )
        assert 0 <= v["n_ray_steps"] <= v["iters"]
        assert 0 <= v["n_safeguard"] <= v["iters"]
        # No warm start: neither trial flag is set.
        assert not v["warm_trial_accepted"] and not v["warm_trial_rejected"]

    @pytest.mark.parametrize("name", list(EXIT_PATHS))
    def test_lines_fire_on_the_exit_taken(self, name):
        make_qp, solver, *_ = EXIT_PATHS[name]
        solver, logged = with_logger(solver)
        res = run(make_qp(), solver)
        converged = bool(res.state.success)
        level = "INFO" if converged else "WARNING"
        result = item_name(res.state.qp_result, ACTIVE_SET_QP_RESULTS, QP_RESULT_NAMES)
        ((got_level, msg),) = logged.lines
        assert got_level == level
        assert msg.startswith(LINE_PREFIX[level])
        assert f"iters={int(res.state.last_n_iter)}" in msg
        assert (f"result={result}" in msg) == (not converged)

    @pytest.mark.parametrize("seed", [3, 5])
    def test_ray_and_safeguard_counters_match_the_reference(self, seed):
        """Near-infeasible QPs take ray steps and degenerate (safeguard) steps."""
        qp = near_infeasible_case(8, seed, False)
        ref = reference_loop(qp)
        solver, logged = with_logger(PLAIN)
        run(qp, solver)
        (rec,) = logged.records
        assert rec.values["n_ray_steps"] == ref.n_ray > 0
        assert rec.values["n_safeguard"] == ref.n_safeguard
        assert ref.n_ray + ref.n_safeguard <= ref.n_iter

    def test_safeguard_counter_is_exercised(self):
        qp = near_infeasible_case(8, 3, False)
        assert reference_loop(qp).n_safeguard > 0

    @pytest.mark.parametrize("feasible", [True, False], ids=["feasible", "infeasible"])
    @pytest.mark.parametrize("seed", [5, 6, 7])
    def test_rejected_ray_trial_is_not_counted(self, seed, feasible):
        """A warm trial that is a ray step but is discarded counts nothing.

        With everything pinned the oracle returns a decreasing certificate ray
        on these QPs, so the trial would be a ray step if it were counted.
        """
        qp = near_infeasible_case(20, seed, feasible)
        cold_solver, cold_logged = with_logger(PLAIN)
        cold = run(qp, cold_solver)
        n = qp.x.shape[0]
        state = with_warm_set(cold_state(qp), np.ones(n, bool), np.zeros(n, bool))
        warm_solver, warm_logged = with_logger(WARM_PLAIN)
        warm = run(qp, warm_solver, state)
        cold_values = cold_logged.records[-1].values
        warm_values = warm_logged.records[-1].values
        assert warm_values["warm_trial_rejected"]
        assert int(warm.state.last_n_iter) == int(cold.state.last_n_iter) + 1
        assert warm_values["n_ray_steps"] == cold_values["n_ray_steps"]
        assert warm_values["n_safeguard"] == cold_values["n_safeguard"]

    def test_no_trial_no_warm_flags(self):
        qp = near_infeasible_case(8, 0, True)
        solver, logged = with_logger(LRHDirectQPSolver(warm_start=True, max_iter=0))
        res = run(qp, solver)
        (rec,) = logged.records
        assert int(res.state.last_n_iter) == 0
        assert not rec.values["warm_trial_accepted"]
        assert not rec.values["warm_trial_rejected"]

    def test_stall_is_logged_and_counted(self, monkeypatch):
        qp = _one_bound_qp()
        _patch_first_pin_multiplier(monkeypatch, lambda m: -m)
        solver, logged = with_logger(LRHDirectQPSolver())
        _, state = _solve_unjitted(qp, solver)
        assert bool(state.qp_result == STALL)
        (rec,) = logged.records
        assert rec.values["stalled"] and rec.values["n_anti_cycling"] == 1
        # The stalling iteration takes no step, so it is neither a ray step nor
        # a safeguard firing.
        assert rec.values["n_safeguard"] == 0 and rec.values["n_ray_steps"] == 0
        msgs = [m for _, m in logged.lines]
        assert any(m.startswith("qp bound loop stalled") for m in msgs)
        assert any(m.startswith("qp failed") for m in msgs)
        assert not any(m.startswith("qp solved") for m in msgs)

    @pytest.mark.parametrize("perturb", [_keep, _wrong_face, _everything_pinned])
    @pytest.mark.parametrize(
        "n, frac, x_on_bound, mix, seed, variant", list(_warm_cases())[:3]
    )
    def test_warm_trial_flags(self, n, frac, x_on_bound, mix, seed, variant, perturb):
        qp = make_case(n, frac, x_on_bound, mix, seed=seed, **variant)
        cold = run(qp)
        lb, ub = perturb(
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
            np.random.default_rng(seed),
        )
        solver, logged = with_logger(WARM)
        run(qp, solver, with_warm_set(cold_state(qp), lb, ub))
        (rec,) = logged.records
        rejected = trial_is_rejected(qp, lb, ub)
        assert rec.values["warm_trial_rejected"] == rejected
        assert rec.values["warm_trial_accepted"] == (not rejected)

    @pytest.mark.parametrize("name", list(EXIT_PATHS))
    def test_logging_does_not_change_the_result(self, name):
        make_qp, solver, *_ = EXIT_PATHS[name]
        qp = make_qp()
        plain = run(qp, solver)
        logged_solver, _ = with_logger(solver)
        logged = run(qp, logged_solver)
        assert leaves_equal(plain.step, logged.step)
        assert leaves_equal(plain.dual, logged.dual)
        assert leaves_equal(plain.state, logged.state)

    @pytest.mark.parametrize("perturb", [_keep, _everything_pinned])
    def test_logging_does_not_change_the_warm_result(self, perturb):
        qp = make_case(8, 0.8, False, (2, 3), seed=1)
        cold = run(qp)
        lb, ub = perturb(
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
            np.random.default_rng(1),
        )
        state = with_warm_set(cold_state(qp), lb, ub)
        plain = run(qp, WARM, state)
        logged_solver, _ = with_logger(WARM)
        logged = run(qp, logged_solver, state)
        assert leaves_equal(plain.step, logged.step)
        assert leaves_equal(plain.dual, logged.dual)
        assert leaves_equal(plain.state, logged.state)


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
    @pytest.mark.parametrize("elastic", [False, True])
    def test_no_n_by_n_or_bound_stacked_intermediate(self, elastic):
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
        solver = LRHDirectQPSolver(elastic=elastic)
        jaxpr = jax.make_jaxpr(lambda: solver.solve(sub, x0, state))()
        shapes = set(walk_avals(jaxpr.jaxpr))
        offenders = [s for s in shapes if sum(1 for d in s if d >= n) >= 2]
        assert not offenders, offenders
        largest = max(int(np.prod(s)) for s in shapes)
        # The elastic QP adds one coordinate and two general rows: the
        # ``(n + 1) × (m_c + 2)`` matrix of the augmented least-distance problem.
        e = int(elastic)
        assert largest <= (n + e) * (max(m_c, MEMORY + 1) + 2 * e) + n


# ---------------------------------------------------------------------------
# Elastic mode (plan 2.5 rev 11/12, T09): inconsistent linearisations
# ---------------------------------------------------------------------------


def contradictory_margin(n: int, seed: int, mix: tuple[int, int]) -> RandomQP:
    """``aᵀd ≤ -1`` (violated at 0) and ``-aᵀd ≤ 0.9``: ``δ_min = 0.1``."""
    qp = make_case(n, 0.3, False, mix, seed=seed)
    a = np.random.default_rng(seed).standard_normal((1, n))
    return with_rows(
        qp, qp.a_eq, qp.b_eq, np.concatenate([a, -a]), np.array([-1.0, 0.9])
    )


def equality_vs_inequality(n: int, seed: int) -> RandomQP:
    """``aᵀd = 1`` against ``aᵀd ≤ 0.5``: ``δ_min = 0.5``."""
    qp = make_case(n, 0.3, False, (0, 0), seed=seed)
    a = np.random.default_rng(seed).standard_normal((1, n))
    return with_rows(qp, a, np.array([1.0]), a.copy(), np.array([0.5]))


def over_fixed_bounds(n: int, seed: int, mix: tuple[int, int]) -> RandomQP:
    """``d_0 + d_{n-1}/2 = 1`` with ``d_0`` fixed (``lb == ub = x``) and ``d_{n-1} ≤ 0.4``.

    The fixed variable cannot move and the other one cannot reach: ``δ_min = 0.8``.
    """
    qp = make_case(n, 0.3, False, mix, seed=seed)
    lb, ub = qp.lb.copy(), qp.ub.copy()
    lb[0] = ub[0] = qp.x[0]
    ub[n - 1] = qp.x[n - 1] + 0.4
    row = np.zeros((1, n))
    row[0, 0], row[0, n - 1] = 1.0, 0.5
    qp = with_rows(qp, row, np.array([1.0]), qp.a_ineq, qp.b_ineq)
    return with_bounds_and_secant(qp, lb, ub)


def clashing_equalities(n: int, seed: int) -> RandomQP:
    """``aᵀd = 1`` and ``aᵀd = 2``: every relaxed row scales with ``1 - δ``, so ``δ = 1``."""
    qp = make_case(n, 0.3, False, (2, 0), seed=seed)
    a = np.random.default_rng(seed).standard_normal((1, n))
    return with_rows(
        qp, np.concatenate([a, a]), np.array([1.0, 2.0]), qp.a_ineq, qp.b_ineq
    )


# name -> QP factory. Every QP has ``x`` inside its box and an infeasible
# linearisation (the LP of ``delta_min_lp`` gives the smallest relaxation).
INCONSISTENT: dict[str, Callable[[], RandomQP]] = {
    **{
        f"near-infeasible-{n}-{seed}": functools.partial(
            near_infeasible_case, n, seed, False
        )
        for n, seed in [(8, 0), (8, 2), (8, 5), (20, 0), (20, 4), (20, 5)]
    },
    **{
        f"margin-{n}-{mix[0]}{mix[1]}": functools.partial(
            contradictory_margin, n, 1, mix
        )
        for n, mix in [(8, (0, 2)), (20, (0, 2)), (8, (2, 3)), (20, (2, 3))]
    },
    "equality-vs-inequality-8": functools.partial(equality_vs_inequality, 8, 0),
    "equality-vs-inequality-20": functools.partial(equality_vs_inequality, 20, 0),
    "over-fixed-8-03": functools.partial(over_fixed_bounds, 8, 0, (0, 3)),
    "over-fixed-20-23": functools.partial(over_fixed_bounds, 20, 1, (2, 3)),
    "box-too-tight": _box_too_tight,
    "clashing-equalities-8": functools.partial(clashing_equalities, 8, 0),
    "clashing-equalities-20": functools.partial(clashing_equalities, 20, 1),
}
# Data of the C15 kind (the relaxation trade-off ``κ`` is small): the excess
# of ``δ`` over ``δ_min`` stays within the plan's ``2 / elastic_weight``.
WELL_SCALED = [name for name in INCONSISTENT if not name.startswith("near-infeasible")]


def merit_directional_derivative(qp: RandomQP, d: np.ndarray, rho_m: float) -> float:
    """One-sided derivative at ``t = 0`` of the L1 merit along ``d``.

    ``φ(x + t d) = f + ρ_m (‖c_E‖₁ + ‖max(c_I, 0)‖₁)`` with the constraints
    linearised (``c_E = -b_E``, ``c_I = -b_I``, Jacobians ``A``).
    """
    c_e, c_i = -qp.b_eq, -qp.b_ineq
    j_e, j_i = qp.a_eq @ d, qp.a_ineq @ d
    d_eq = np.where(c_e != 0, np.sign(c_e) * j_e, np.abs(j_e)).sum()
    d_in = np.where(c_i > 0, j_i, np.where(c_i == 0, np.maximum(j_i, 0.0), 0.0)).sum()
    return float(qp.g @ d + rho_m * (d_eq + d_in))


class Elastic(NamedTuple):
    """An elastic solve with the quantities the tests read."""

    qp: RandomQP
    run: Run
    delta: float
    rho: float
    weight: float
    lam: np.ndarray  # [λ_E, λ_I]
    column: np.ndarray  # the column c of δ


def elastic_rho_of(qp: RandomQP, weight: float = 1e3) -> float:
    return float(elastic_penalty(qp.lagrangian, qp.factors, weight))


def solve_elastic(qp: RandomQP, **kwargs) -> Elastic:
    solver = LRHDirectQPSolver(**kwargs)
    res = run(qp, solver)
    return Elastic(
        qp,
        res,
        float(res.state.last_elastic_delta),
        float(elastic_penalty(qp.lagrangian, qp.factors, solver.elastic_weight)),
        solver.elastic_weight,
        np.concatenate(
            [np.asarray(res.dual.eq_multipliers), np.asarray(res.dual.ineq_multipliers)]
        ),
        elastic_column_np(qp),
    )


@pytest.fixture(params=list(INCONSISTENT))
def inconsistent(request) -> Elastic:
    """The default solver (elastic mode on) run on each inconsistent QP."""
    return solve_elastic(INCONSISTENT[request.param]())


class TestElasticMode:
    def test_inconsistent_linearisation_gives_a_relaxed_step(self, inconsistent):
        el = inconsistent
        qp, res = el.qp, el.run
        state = res.state
        assert_converged(res)
        assert bool(state.qp_result == CONVERGED) and bool(state.success)
        assert bool(state.status == RESULTS.successful)
        assert int(state.n_elastic) == 1
        # Both loops ran: at least the plain loop's certifying iteration plus one
        # elastic oracle call (the counters are cumulative over the two).
        plain_state = run(el.qp, PLAIN).state
        assert int(state.last_n_iter) > int(plain_state.last_n_iter)
        assert int(state.last_n_cg_iter) >= int(plain_state.last_n_cg_iter)
        assert int(state.n_nnls_iter) == int(state.last_n_cg_iter)
        delta_min = delta_min_lp(qp)
        assert delta_min is not None and 0.0 < delta_min <= 1.0
        # A finite step inside the box, 0 < δ ≤ 1, and δ_min ≤ δ.
        assert np.all(np.isfinite(res.step))
        assert_in_box(qp, res.step)
        assert 0.0 < el.delta <= 1.0 + 1e-12
        assert el.delta >= delta_min - 1e-8
        # The relaxed QP is solved to its KKT point (convex: the minimiser), and
        # δ never appears in the returned dual.
        assert_elastic_kkt(qp, el.rho, res.step, el.delta, res.dual)
        assert res.dual.eq_multipliers.shape == (qp.a_eq.shape[0],)
        assert res.dual.ineq_multipliers.shape == (qp.a_ineq.shape[0],)
        assert res.dual.lb_multipliers.shape == res.dual.ub_multipliers.shape
        assert np.all(np.isfinite(el.lam)) and bool(
            state.last_ldp_status == LDP_RESULTS.feasible
        )
        assert bool(state.last_kkt_reason == KKT_SOLVER_RESULTS.converged)

    def test_excess_of_delta_is_the_tradeoff_over_the_weight(self, inconsistent):
        """``δ = max(δ_min, κ/w)`` with ``κ = -cᵀλ/τ²`` and ``ρ_e = w τ²``.

        At an interior ``δ`` the stationarity in ``δ`` is ``ρ_e δ = -cᵀλ``, so
        the excess over ``δ_min`` is at most ``κ/w``: the plan's
        ``2 / elastic_weight`` bound is the case ``κ ≤ 2``.
        """
        el = inconsistent
        delta_min = delta_min_lp(el.qp)
        tau2 = el.rho / el.weight
        kappa = -(el.column @ el.lam) / tau2
        assert el.delta - delta_min <= max(kappa, 0.0) / el.weight + 1e-8

    def test_step_is_a_merit_descent_direction(self, inconsistent):
        """``Dφ ≤ -dᵀBd - (1-δ)(ρ_m - ‖λ‖∞)‖viol‖₁ < 0`` for ``ρ_m > ‖λ‖∞``."""
        el = inconsistent
        qp, d = el.qp, el.run.step
        lam_max = np.abs(el.lam).max(initial=0.0)
        rho_m = 1.1 * lam_max + 1e-6
        dphi = merit_directional_derivative(qp, d, rho_m)
        viol = np.abs(qp.b_eq).sum() + np.maximum(-qp.b_ineq, 0.0).sum()
        bound = -d @ qp.dense_b @ d - (1.0 - el.delta) * (rho_m - lam_max) * viol
        assert dphi < 0.0
        assert dphi <= bound + 1e-8 * max(1.0, abs(bound))

    @pytest.mark.parametrize("name", WELL_SCALED)
    def test_default_weight_bound_on_delta(self, name):
        """At the default weight: ``δ_min ≤ δ ≤ δ_min + 2/elastic_weight``."""
        el = solve_elastic(INCONSISTENT[name]())
        delta_min = delta_min_lp(el.qp)
        assert el.weight == 1e3
        assert delta_min - 1e-8 <= el.delta <= delta_min + 2.0 / el.weight

    @pytest.mark.parametrize("name", ["near-infeasible-20-5", "near-infeasible-8-5"])
    def test_excess_shrinks_with_the_weight(self, name):
        """The excess of ``δ`` over ``δ_min`` is ``O(1/w)``: monotone in the weight."""
        qp = INCONSISTENT[name]()
        delta_min = delta_min_lp(qp)
        excess = []
        for weight in (1e1, 1e2, 1e3):
            el = solve_elastic(qp, elastic_weight=weight)
            assert bool(el.run.state.success)
            assert_elastic_kkt(qp, el.rho, el.run.step, el.delta, el.run.dual)
            excess.append(el.delta - delta_min)
        assert excess[0] >= excess[1] >= excess[2] >= -1e-8
        assert excess[0] > 10 * max(excess[2], 1e-12)

    @pytest.mark.parametrize("name", list(INCONSISTENT)[:6] + ["box-too-tight"])
    def test_elastic_false_reproduces_the_failure(self, name):
        qp = INCONSISTENT[name]()
        plain = run(qp, PLAIN)
        state = plain.state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert bool(state.status == RESULTS.singular)
        assert int(state.n_elastic) == 0 and float(state.last_elastic_delta) == 0.0
        for leaf in jax.tree.leaves(plain.dual):
            assert np.all(np.asarray(leaf) == 0.0)
        # ... and the same QP is solved with elastic mode on.
        assert bool(run(qp).state.success)

    @pytest.mark.parametrize("x_on_bound", [False, True], ids=["interior", "on-bound"])
    @pytest.mark.parametrize(
        "mix", [(0, 0), (2, 3), (0, 3), (3, 2)], ids=["0-0", "2-3", "0-3", "3dup-2"]
    )
    @pytest.mark.parametrize("n", [8, 20])
    @pytest.mark.parametrize("seed", range(2))
    def test_consistent_qps_never_enter_elastic_mode(self, n, mix, x_on_bound, seed):
        """Heavy-cut QPs: ``n_elastic`` unchanged, bit-identical to ``elastic=False``."""
        qp = make_case(n, 0.8, x_on_bound, mix, seed=seed)
        on, off = run(qp), run(qp, PLAIN)
        assert bool(on.state.success)
        assert int(on.state.n_elastic) == 0 and float(on.state.last_elastic_delta) == 0
        assert leaves_equal(on.step, off.step)
        assert leaves_equal(on.dual, off.dual)
        assert leaves_equal(on.state, off.state)

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("n", [8, 20])
    def test_near_infeasible_feasible_side_is_not_relaxed(self, n, seed):
        qp = near_infeasible_case(n, seed, True)
        on, off = run(qp), run(qp, PLAIN)
        assert int(on.state.n_elastic) == 0
        assert leaves_equal(on.step, off.step) and leaves_equal(on.state, off.state)

    def test_counters_accumulate_and_delta_resets(self):
        """``n_elastic`` is cumulative; ``last_elastic_delta`` is that of the last solve."""
        bad = near_infeasible_case(8, 0, False)
        good = near_infeasible_case(8, 0, True)
        first = run(bad)
        second = run(bad, state=first.state)
        assert int(first.state.n_elastic) == 1 and int(second.state.n_elastic) == 2
        np.testing.assert_array_equal(
            np.asarray(second.state.last_elastic_delta),
            np.asarray(first.state.last_elastic_delta),
        )
        assert int(second.state.n_iter) == 2 * int(first.state.last_n_iter)
        third = run(good, state=second.state)
        assert int(third.state.n_elastic) == 2
        assert float(third.state.last_elastic_delta) == 0.0
        assert bool(third.state.success)

    def test_cold_state_has_zero_elastic_fields(self):
        state = LRHQPSolverState.cold(5, 2, 3, jnp.float64)
        assert float(state.last_elastic_delta) == 0.0
        assert state.last_elastic_delta.dtype == jnp.float64
        assert int(state.n_elastic) == 0 and state.n_elastic.dtype == jnp.int32

    def test_elastic_loop_that_fails_reports_failure(self):
        """No relaxation helps: ``x`` below every lower bound (``d ≥ 0.2``), ``Σ d = 1``.

        The relaxed row is ``Σ d = 1 - δ ≤ 1 < 1.2`` for every ``δ ∈ [0, 1]``, so
        the elastic QP is infeasible too (its unpinned relaxation is not, so the
        elastic loop takes a full step with ``δ > 0`` first): the exit is the
        elastic loop's certified-infeasible row, with ``success`` false, zero
        dual and ``δ`` reported as 0 (the earlier full step is not a QP
        solution), and elastic mode entered.
        """
        n = 6
        qp = make_case(n, 0.0, False, (1, 0), seed=1)
        qp = with_rows(qp, np.ones((1, n)), np.ones(1), qp.a_ineq, qp.b_ineq)
        qp = with_bounds_and_secant(qp, qp.x + 0.2, qp.x + 1.0)
        assert not lp_feasible(qp) and delta_min_lp(qp) is None
        # The first elastic full step (nothing pinned) already has δ > 0.
        none = jnp.zeros(n, bool)
        first = solve_fixed_bounds_qp(
            qp.lagrangian, qp.factors, none, none, elastic_rho=elastic_rho_of(qp)
        )
        assert float(first.delta) > 0.0
        state = run(qp).state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert bool(state.status == RESULTS.singular)
        assert int(state.n_elastic) == 1 and float(state.last_elastic_delta) == 0.0
        assert bool(
            (state.last_ldp_status == LDP_RESULTS.infeasible)
            | (state.last_ldp_status == LDP_RESULTS.inconsistent_equalities)
        )

    def test_without_general_constraints_elastic_mode_is_not_entered(self):
        """A failing solve with no rows (non-finite gradient): nothing to relax."""
        qp = make_case(6, 0.3, False, (0, 0), seed=0)
        lag = eqx.tree_at(
            lambda lg: lg.evaluated.grad_val,
            qp.lagrangian,
            qp.lagrangian.evaluated.grad_val.at[0].set(jnp.nan),
        )
        qp = qp._replace(lagrangian=lag)
        state = run(qp).state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert int(state.n_elastic) == 0 and float(state.last_elastic_delta) == 0.0
        assert leaves_equal(state, run(qp, PLAIN).state)

    def test_non_finite_data_enters_and_fails(self):
        state = run(_nonfinite_gradient()).state
        assert bool(state.qp_result == FAILURE) and not bool(state.success)
        assert int(state.n_elastic) == 1 and float(state.last_elastic_delta) == 0.0
        assert bool(state.last_ldp_status == LDP_RESULTS.nonfinite)

    @pytest.mark.parametrize("n, seed", [(8, 2), (20, 4)])
    def test_default_weight_certifies_without_a_retry(self, n, seed):
        """The augmented LDP's infeasible exit carries a certificate by itself.

        At the default weight the pinned elastic problem of these QPs has an
        NNLS residual just above the noise threshold of the infeasibility
        test, so the LDP status is ``infeasible`` by its primal check. That
        exit keeps the verified Farkas certificate (C16b), so the elastic loop
        takes its ray step and the solve succeeds with no second solve of the
        pinned problem at a smaller weight (the C16 kernel retry, removed).
        """
        qp = near_infeasible_case(n, seed, False)
        _, state = _solve_unjitted(qp)
        assert bool(state.success) and bool(state.qp_result == CONVERGED)
        assert int(state.n_elastic) == 1

    @pytest.mark.parametrize("weight", [1e3, 1e4])
    def test_pinned_infeasible_elastic_solves_are_certified_jit_and_eager(self, weight):
        """Every infeasible pinned elastic solve carries a ray, jitted or not.

        The NNLS residual of an infeasible augmented problem can land a little
        above the noise threshold of the LDP's infeasibility test, so that the
        ``infeasible`` status comes from its primal check; which side of the
        threshold it lands on can differ between a compiled and an op-by-op
        run. Such an exit must keep its certificate (``has_ray``) in both.
        """
        battery = near_infeasible_pinned_battery()
        assert len(battery) == 80
        jitted = jax.jit(solve_fixed_bounds_qp)
        outcomes = []
        for qp, lower, upper in battery:
            args = (qp.lagrangian, qp.factors, jnp.asarray(lower), jnp.asarray(upper))
            rho = elastic_rho_of(qp, weight)
            outcomes.append(
                tuple(
                    (
                        bool(res.has_ray),
                        bool(res.ldp_status == LDP_RESULTS.infeasible),
                    )
                    for res in (
                        jitted(*args, elastic_rho=rho),
                        solve_fixed_bounds_qp(*args, elastic_rho=rho),
                    )
                )
            )
        jit_out, eager_out = zip(*outcomes)
        # Jit and eager agree on every problem, ...
        assert jit_out == eager_out
        # ... the pinned problems exercise the infeasible exit, ...
        assert sum(infeasible for _, infeasible in jit_out) >= 60
        # ... and each of them carries a certificate.
        assert all(has_ray for has_ray, _ in jit_out)

    @pytest.mark.parametrize("weight", [1e3, 1e4])
    def test_every_infeasible_exit_of_a_pinned_elastic_solve_has_a_ray(self, weight):
        """The contract at any number of free variables, jitted or not.

        With as few free variables as equality rows the pinned augmented
        problem is infeasible by a margin that is small in the normalised
        units of the LDP, and its NNLS residual often sits just above the noise
        threshold: the ``infeasible`` status then comes from the primal check.
        Whatever else such a solve reports, an ``infeasible`` exit must keep
        its certificate.
        """
        battery = near_infeasible_pinned_battery(min_free=1)
        jitted = jax.jit(solve_fixed_bounds_qp)
        n_infeasible = 0
        for qp, lower, upper in battery:
            args = (qp.lagrangian, qp.factors, jnp.asarray(lower), jnp.asarray(upper))
            rho = elastic_rho_of(qp, weight)
            for res in (
                jitted(*args, elastic_rho=rho),
                solve_fixed_bounds_qp(*args, elastic_rho=rho),
            ):
                if bool(res.ldp_status == LDP_RESULTS.infeasible):
                    n_infeasible += 1
                    assert bool(res.has_ray)
        assert n_infeasible >= 100

    @pytest.mark.parametrize(
        "name",
        [
            "near-infeasible-8-0",
            "near-infeasible-8-2",
            "near-infeasible-20-4",
            "margin-8-23",
            "over-fixed-8-03",
            "equality-vs-inequality-8",
        ],
    )
    def test_path_is_the_plain_loop_then_the_elastic_loop(self, name):
        """Counters are cumulative over the two loops, the step is the second's.

        The NumPy reference runs the loop of the plan on the plain oracle, then
        on the elastic oracle with the frozen ``ρ_e`` (separate budgets). The
        solver must take the same iterations, ray steps, safeguard firings and
        NNLS steps in total and end on the elastic loop's working set and step.
        """
        qp = INCONSISTENT[name]()
        rho = elastic_rho_of(qp)
        elastic_kernel = jax.jit(
            functools.partial(solve_fixed_bounds_qp, elastic_rho=rho)
        )
        nnls = {"plain": 0, "elastic": 0}

        def counted(kind, kernel):
            def oracle(*args):
                res = kernel(*args)
                nnls[kind] += int(res.n_nnls_iter)
                return res

            return oracle

        def elastic_value(q, res):
            return (
                objective(q, np.asarray(res.step)) + 0.5 * rho * float(res.delta) ** 2
            )

        plain = reference_loop(qp, oracle=counted("plain", _oracle))
        elastic = reference_loop(
            qp, oracle=counted("elastic", elastic_kernel), value=elastic_value
        )
        assert bool(plain.qp_result == FAILURE) and bool(elastic.qp_result == CONVERGED)
        solver, logged = with_logger(LRHDirectQPSolver())
        res = run(qp, solver)
        state = res.state
        v = logged.records[-1].values
        assert int(state.last_n_iter) == plain.n_iter + elastic.n_iter
        assert int(state.last_n_cg_iter) == nnls["plain"] + nnls["elastic"]
        assert v["n_ray_steps"] == plain.n_ray + elastic.n_ray
        assert v["n_safeguard"] == plain.n_safeguard + elastic.n_safeguard
        np.testing.assert_array_equal(
            np.asarray(state.active_set.active_lb), elastic.active_lb
        )
        np.testing.assert_array_equal(
            np.asarray(state.active_set.active_ub), elastic.active_ub
        )
        np.testing.assert_allclose(res.step, elastic.step, atol=1e-12, rtol=0)

    @pytest.mark.parametrize(
        "name", ["near-infeasible-8-0", "margin-8-23", "over-fixed-8-03"]
    )
    def test_warm_start_reaches_the_same_elastic_solution(self, name):
        """The elastic loop restarts from the warm trial, like the plain one."""
        qp = INCONSISTENT[name]()
        cold = run(qp)
        assert bool(cold.state.success) and int(cold.state.n_elastic) == 1
        state = with_warm_set(
            cold_state(qp),
            np.asarray(cold.state.active_set.active_lb),
            np.asarray(cold.state.active_set.active_ub),
        )
        warm = run(qp, LRHDirectQPSolver(warm_start=True), state)
        assert bool(warm.state.success) and int(warm.state.n_elastic) == 1
        np.testing.assert_allclose(warm.step, cold.step, atol=1e-8)
        np.testing.assert_allclose(
            float(warm.state.last_elastic_delta),
            float(cold.state.last_elastic_delta),
            atol=1e-10,
        )


class TestElasticJit:
    def test_single_compilation_whether_or_not_elastic_mode_is_entered(self):
        traces = []

        @jax.jit
        def solve(sub, state):
            traces.append(1)
            lag = sub.lagrangian
            x0 = (Primal(jnp.zeros_like(lag.x_ref)), lag.dual)
            return LRHDirectQPSolver().solve(sub, x0, state)

        entered = []
        for seed, feasible in [(0, False), (1, True), (2, False), (3, True)]:
            qp = near_infeasible_case(8, seed, feasible)
            _, state = solve(make_subproblem(qp), cold_state(qp))
            entered.append(int(state.n_elastic))
            assert bool(state.success)
        assert entered == [1, 0, 1, 0]
        assert len(traces) == 1

    @pytest.mark.parametrize(
        "x64, dtype",
        [(True, jnp.float64), (False, jnp.float32)],
        ids=["float64", "float32"],
    )
    def test_inside_while_loop_and_dtype_stable(self, x64, dtype):
        """An elastic solve inside a ``lax.while_loop`` whose carry is the state."""
        with jax.enable_x64(x64):
            qp = near_infeasible_case(8, 0, False)
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

                return jax.lax.while_loop(
                    lambda c: c[0] < 2,
                    body,
                    (jnp.asarray(0), state, jnp.zeros_like(lag.x_ref)),
                )

            _, state_out, step_out = looped(sub, state0)
            assert _state_dtypes(state_out) == _state_dtypes(state0)
            assert step_out.dtype == dtype
            assert int(state_out.n_elastic) == 2
            assert np.all(np.isfinite(np.asarray(step_out)))

    def test_vmapped_over_data(self):
        """``vmap`` of the solve: the elastic ``cond`` becomes a select."""
        qps = [near_infeasible_case(8, s, f) for s, f in [(0, False), (1, True)]]
        subs = jax.tree.map(
            lambda *xs: jnp.stack(xs), *[make_subproblem(q) for q in qps]
        )
        states = jax.tree.map(lambda *xs: jnp.stack(xs), *[cold_state(q) for q in qps])

        def solve(sub, state):
            lag = sub.lagrangian
            x0 = (Primal(jnp.zeros_like(lag.x_ref)), lag.dual)
            (step, _), new = LRHDirectQPSolver().solve(sub, x0, state)
            return step.x, new

        _, new = jax.jit(jax.vmap(solve))(subs, states)
        assert np.asarray(new.n_elastic).tolist() == [1, 0]
        assert bool(np.all(np.asarray(new.success)))


class TestElasticLogging:
    @pytest.mark.parametrize("feasible", [False, True], ids=["elastic", "plain"])
    def test_record_and_line(self, feasible):
        qp = near_infeasible_case(8, 0, feasible)
        solver, logged = with_logger(LRHDirectQPSolver())
        res = run(qp, solver)
        (rec,) = logged.records
        v = rec.values
        assert set(v) == RECORD_KEYS
        assert v["elastic"] == (not feasible)
        assert v["n_elastic"] == int(res.state.n_elastic) == (0 if feasible else 1)
        np.testing.assert_allclose(
            v["last_elastic_delta"], float(res.state.last_elastic_delta), rtol=1e-6
        )
        assert (v["last_elastic_delta"] > 0.0) == (not feasible)
        msgs = [m for _, m in logged.lines]
        assert any(m.startswith("qp elastic mode") for m in msgs) == (not feasible)
        assert any(m.startswith("qp solved") for m in msgs)
        assert not any(m.startswith("qp failed") for m in msgs)

    def test_elastic_record_counts_the_cumulative_solves(self):
        qp = near_infeasible_case(8, 0, False)
        solver, logged = with_logger(LRHDirectQPSolver())
        first = run(qp, solver)
        run(qp, solver, first.state)
        assert [r.values["n_elastic"] for r in logged.records] == [1, 2]

    def test_logging_does_not_change_the_elastic_result(self):
        qp = near_infeasible_case(8, 0, False)
        plain = run(qp)
        logged_solver, _ = with_logger(LRHDirectQPSolver())
        logged = run(qp, logged_solver)
        assert leaves_equal(plain.step, logged.step)
        assert leaves_equal(plain.dual, logged.dual)
        assert leaves_equal(plain.state, logged.state)
