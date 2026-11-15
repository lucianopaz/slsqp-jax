"""Fixtures shared by :mod:`slsqp_jax.sqpdax.subproblem.solver` tests."""

from __future__ import annotations

import itertools
from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import EvaluatedProblem, Problem
from slsqp_jax.sqpdax.secant import ReducedHessianFactors
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    KKT_SOLVER_RESULTS,
    RESULTS,
    ActiveSetQPSolverState,
    DogLegSolverState,
    FunnelTangentialStepState,
    MinresQLPState,
    ProjectedCGState,
    ProximalActiveSetQPSolverState,
    ScaledNormalStepState,
    SteihaugTointCGTangentialStepSolverState,
    SVDProjector,
    TrustFunnelSolverState,
    TrustRegionSolverState,
)
from tests.sqpdax.lagrangian.conftest import make_primal, make_problem
from tests.sqpdax.subproblem.conftest import make_evaluated_lagrangian, make_zero_dual


def make_qp_subproblem(
    *,
    problem: Problem | None = None,
    primal: Primal | None = None,
    active_inequalities: tuple[bool, ...] = (),
    active_lb: tuple[bool, ...] | None = None,
    active_ub: tuple[bool, ...] | None = None,
) -> ActiveSetSubProblem:
    """ActiveSetSubProblem evaluated at zero dual (clean QP)."""
    if problem is None:
        problem = make_problem()
    if primal is None:
        primal = make_primal(n=problem.n)
    if active_lb is None:
        active_lb = tuple(False for _ in range(problem.n))
    if active_ub is None:
        active_ub = tuple(False for _ in range(problem.n))
    if not active_inequalities and problem.mineq > 0:
        active_inequalities = tuple(False for _ in range(problem.mineq))
    lag = make_evaluated_lagrangian(
        problem=problem,
        primal=primal,
        dual=make_zero_dual(problem.n, problem.meq, problem.mineq),
    )
    active = ActiveSet(
        meq=lag.meq,
        active_inequalities=jnp.asarray(active_inequalities, dtype=bool),
        active_lb=jnp.asarray(active_lb, dtype=bool),
        active_ub=jnp.asarray(active_ub, dtype=bool),
    )
    return ActiveSetSubProblem(lag, active)


def make_projected_cg_state() -> ProjectedCGState:
    """Cold :class:`ProjectedCGState`."""
    return ProjectedCGState.cold(jnp.zeros(()).dtype)


def make_minres_qlp_state() -> MinresQLPState:
    """Cold :class:`MinresQLPState`."""
    return MinresQLPState.cold(jnp.zeros(()).dtype)


class FailingProjector(SVDProjector):
    """SVD projector that reports its (fictitious) inner solve as failed."""

    def build(self, subproblem, preconditioner=None):
        ctx = super().build(subproblem, preconditioner)
        return eqx.tree_at(
            lambda c: (c.converged, c.n_iter),
            ctx,
            (jnp.asarray(False), jnp.asarray(3, jnp.int32)),
        )


def make_dogleg_state(
    radius: float | Array = 10.0,
    *,
    active_bounds: tuple[Array, Array] | None = None,
) -> DogLegSolverState:
    """Cold :class:`DogLegSolverState` with the given trust-region radius."""
    return DogLegSolverState(
        n_iter=jnp.zeros((), jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
        n_cg_iter=jnp.zeros((), jnp.int32),
        on_boundary=jnp.asarray(False),
        radius=jnp.asarray(radius),
        active_bounds=active_bounds,
    )


def make_steihaug_state(
    radius: float | Array = 1.0,
    *,
    active_bounds: tuple[Array, Array] | None = None,
) -> SteihaugTointCGTangentialStepSolverState:
    """Cold :class:`SteihaugTointCGTangentialStepSolverState`."""
    return SteihaugTointCGTangentialStepSolverState(
        n_iter=jnp.zeros((), jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
        n_cg_iter=jnp.zeros((), jnp.int32),
        on_boundary=jnp.asarray(False),
        radius=jnp.asarray(radius),
        active_bounds=active_bounds,
    )


def make_scaled_normal_state(radius: float | Array = 1.0) -> ScaledNormalStepState:
    """Cold :class:`ScaledNormalStepState` with the given normal radius."""
    return ScaledNormalStepState.cold(radius)


def make_funnel_tangential_state(
    radius: float | Array = 1.0,
) -> FunnelTangentialStepState:
    """Cold :class:`FunnelTangentialStepState` with the given composite radius."""
    return FunnelTangentialStepState.cold(radius)


def make_trust_funnel_state(
    radius_v: float | Array = 1.0,
    radius_f: float | Array = 1.0,
    v_max: float | Array = 10.0,
    *,
    eps_pi: float = 1e-6,
    eps_v: float = 1e-6,
    pi_f_prev: float | Array = 0.0,
    sf_flag: bool = False,
) -> TrustFunnelSolverState:
    """Cold :class:`TrustFunnelSolverState` with optional carried fields set."""
    state = TrustFunnelSolverState.cold(
        radius_v, radius_f, v_max, eps_pi=eps_pi, eps_v=eps_v
    )
    return eqx.tree_at(
        lambda s: (s.pi_f_prev, s.sf_flag),
        state,
        (jnp.asarray(pi_f_prev, state.radius_v.dtype), jnp.asarray(sf_flag)),
    )


def make_empty_active_set(n: int = 2, meq: int = 0, mineq: int = 0) -> ActiveSet:
    """All-inactive :class:`ActiveSet` for the given sizes."""
    return ActiveSet(
        meq=meq,
        active_inequalities=jnp.zeros((mineq,), bool),
        active_lb=jnp.zeros((n,), bool),
        active_ub=jnp.zeros((n,), bool),
    )


def _cold_active_set_fields(n: int, meq: int, mineq: int) -> dict:
    """Shared cold fields of the active-set QP carries."""
    return dict(
        n_iter=jnp.zeros((), jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
        n_cg_iter=jnp.zeros((), jnp.int32),
        last_n_iter=jnp.zeros((), jnp.int32),
        last_n_cg_iter=jnp.zeros((), jnp.int32),
        qp_result=ACTIVE_SET_QP_RESULTS.working_set_converged,
        active_set=make_empty_active_set(n, meq, mineq),
        dual=make_zero_dual(n, meq, mineq),
        final_working_tol=jnp.asarray(0.0),
        n_anti_cycling=jnp.zeros((), jnp.int32),
        last_kkt_feasibility_residual=jnp.asarray(0.0),
        last_kkt_n_refinements=jnp.zeros((), jnp.int32),
        last_kkt_reason=KKT_SOLVER_RESULTS.converged,
    )


def make_active_set_qp_state(
    n: int = 2, meq: int = 0, mineq: int = 0
) -> ActiveSetQPSolverState:
    """Cold :class:`ActiveSetQPSolverState` sized for ``(n, meq, mineq)``."""
    return ActiveSetQPSolverState(**_cold_active_set_fields(n, meq, mineq))


def make_proximal_state(
    meq: int,
    *,
    n: int = 2,
    mineq: int = 0,
    kkt_residual: float | Array = jnp.inf,
    eq_center: Array | None = None,
) -> ProximalActiveSetQPSolverState:
    """Cold :class:`ProximalActiveSetQPSolverState` with the given residual / centre."""
    if eq_center is None:
        eq_center = jnp.zeros((meq,))
    return ProximalActiveSetQPSolverState(
        **_cold_active_set_fields(n, meq, mineq),
        kkt_residual=jnp.asarray(kkt_residual, dtype=float),
        mu=jnp.asarray(0.0),
        eq_center=jnp.asarray(eq_center),
    )


def make_trust_region_state(
    radius: float | Array = 1.0,
    *,
    merit_penalty: float | Array = 1.0,
) -> TrustRegionSolverState:
    """Cold :class:`TrustRegionSolverState` with the given radius / penalty."""
    return TrustRegionSolverState(
        n_iter=jnp.zeros((), jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
        radius=jnp.asarray(radius),
        predicted_reduction=jnp.asarray(0.0),
        merit_penalty=jnp.asarray(merit_penalty),
        n_cg_iter=jnp.zeros((), jnp.int32),
        on_boundary=jnp.asarray(False),
    )


def unbounded_box(n: int = 2) -> tuple[Array, Array]:
    """Componentwise ``(-inf, +inf)`` bounds of length ``n``."""
    return jnp.full(n, -jnp.inf), jnp.full(n, jnp.inf)


class RandomQP(NamedTuple):
    """Random convex QP of the SQP step with an LRH model (see :func:`make_random_lrh_qp`).

    The QP is ``min gᵀd + ½dᵀBd  s.t.  A_E d = b_E, A_I d ≤ b_I, lb ≤ x + d ≤ ub``
    with the dense NumPy arrays below (tiny ``n``, reference use only) and the
    same data packaged as an ``EvaluatedLagrangian`` plus the LRH factors.
    ``d0`` satisfies every general constraint and has ``x + d0`` on the
    ``lower`` / ``upper`` bounds.
    """

    lagrangian: EvaluatedLagrangian
    factors: ReducedHessianFactors
    dense_b: np.ndarray
    g: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray
    a_ineq: np.ndarray
    b_ineq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    x: np.ndarray
    d0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def make_random_lrh_qp(
    n: int,
    rank: int,
    m_e: int,
    m_i: int,
    *,
    seed: int = 0,
    memory: int = 4,
    dup: bool = False,
    tight: bool | int = False,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    infinite: np.ndarray | None = None,
) -> RandomQP:
    """Random feasible convex QP with LRH factors, bounds, equalities, inequalities.

    ``B = σ I + Z (RᵀR - σ I) Zᵀ`` has an orthonormal ``Z`` (``rank`` columns
    out of ``memory``, the rest zero) and garbage in the padded block of ``R``.
    A point ``d0`` is feasible for the general constraints by construction;
    every inequality has a positive slack at ``d0`` except the first
    ``tight`` ones, which are active at ``d0`` (``True`` means one; any number
    up to ``m_i`` is allowed). Several active inequalities can make the
    feasible set of a small free subspace degenerate (empty interior, e.g. a
    single point), which the LDP chain must still solve. The bounds are
    ``x + d0`` on the ``lower`` (resp. ``upper``) coordinates, so that pinning
    exactly those faces keeps the QP feasible; elsewhere they are a random
    distance away, or infinite on the ``infinite`` coordinates. ``dup`` makes
    the last equality row a copy of the first.
    """
    rng = np.random.default_rng(1000 * seed + 100 * n + 10 * m_e + m_i + rank)
    false = np.zeros(n, dtype=bool)
    lower = false if lower is None else np.asarray(lower, dtype=bool)
    upper = false if upper is None else np.asarray(upper, dtype=bool)
    infinite = false if infinite is None else np.asarray(infinite, dtype=bool)

    q, _ = np.linalg.qr(rng.standard_normal((n, memory)))
    basis = np.zeros((n, memory))
    basis[:, :rank] = q[:, :rank]
    chol = np.triu(rng.standard_normal((memory, memory)), 1)
    chol[np.arange(memory), np.arange(memory)] = rng.uniform(0.5, 2.0, memory)
    sigma = float(rng.uniform(0.5, 2.0))
    r_act = chol[:rank, :rank]
    dense_b = (
        sigma * np.eye(n)
        + basis[:, :rank] @ (r_act.T @ r_act - sigma * np.eye(rank)) @ basis[:, :rank].T
    )
    chol_padded = chol.copy()
    chol_padded[rank:, :] = 7.5  # garbage: the factors ignore the padded block
    chol_padded[:, rank:] = 7.5
    factors = ReducedHessianFactors(
        basis=jnp.asarray(basis),
        cholesky=jnp.asarray(np.triu(chol_padded)),
        sigma=jnp.asarray(sigma),
        rank=jnp.asarray(rank, dtype=jnp.int32),
    )

    x = rng.standard_normal(n)
    d0 = rng.standard_normal(n)
    g = rng.standard_normal(n)
    a_eq = rng.standard_normal((m_e, n))
    if dup and m_e > 1:
        a_eq[m_e - 1] = a_eq[0]
    a_ineq = rng.standard_normal((m_i, n))
    b_eq = a_eq @ d0
    slack = 0.05 + 0.5 * np.abs(rng.standard_normal(m_i))
    slack[: int(tight)] = 0.0
    b_ineq = a_ineq @ d0 + slack

    lb = x + d0 - (0.5 + rng.random(n))
    ub = x + d0 + (0.5 + rng.random(n))
    lb[lower] = (x + d0)[lower]
    ub[upper] = (x + d0)[upper]
    lb[infinite & ~lower] = -np.inf
    ub[infinite & ~upper] = np.inf

    evaluated = EvaluatedProblem(
        ref=Primal(x=jnp.asarray(x)),
        fn_val=jnp.zeros(()),
        grad_val=jnp.asarray(g),
        fn_qvp=None,
        eq_fn_val=jnp.asarray(-b_eq),
        eq_fn_jac_val=jnp.asarray(a_eq),
        eq_fn_qvp=None,
        ineq_fn_val=jnp.asarray(-b_ineq),
        ineq_fn_jac_val=jnp.asarray(a_ineq),
        ineq_fn_qvp=None,
        lb=jnp.asarray(lb),
        ub=jnp.asarray(ub),
        null_lb=jnp.asarray(~np.isfinite(lb)),
        null_ub=jnp.asarray(~np.isfinite(ub)),
        aux_val=None,
    )
    lagrangian = EvaluatedLagrangian(
        evaluated=evaluated,
        secant=None,
        dual=Dual(
            eq_multipliers=jnp.zeros((m_e,)),
            ineq_multipliers=jnp.zeros((m_i,)),
            lb_multipliers=jnp.zeros((n,)),
            ub_multipliers=jnp.zeros((n,)),
        ),
    )
    return RandomQP(
        lagrangian=lagrangian,
        factors=factors,
        dense_b=dense_b,
        g=g,
        a_eq=a_eq,
        b_eq=b_eq,
        a_ineq=a_ineq,
        b_ineq=b_ineq,
        lb=lb,
        ub=ub,
        x=x,
        d0=d0,
        lower=lower,
        upper=upper,
    )


@pytest.fixture
def random_lrh_qp() -> Callable[..., RandomQP]:
    """Factory of random convex LRH QPs (:func:`make_random_lrh_qp`)."""
    return make_random_lrh_qp


def with_rows(qp: RandomQP, a_eq, b_eq, a_ineq, b_ineq) -> RandomQP:
    """Replace the general constraint rows of ``qp`` (NumPy and Lagrangian)."""
    ev = qp.lagrangian.evaluated
    ev = eqx.tree_at(
        lambda e: (e.eq_fn_val, e.eq_fn_jac_val, e.ineq_fn_val, e.ineq_fn_jac_val),
        ev,
        (
            jnp.asarray(-b_eq),
            jnp.asarray(a_eq),
            jnp.asarray(-b_ineq),
            jnp.asarray(a_ineq),
        ),
    )
    lag = eqx.tree_at(
        lambda lg: (lg.evaluated, lg.dual.eq_multipliers, lg.dual.ineq_multipliers),
        qp.lagrangian,
        (ev, jnp.zeros(len(b_eq)), jnp.zeros(len(b_ineq))),
    )
    return qp._replace(
        lagrangian=lag, a_eq=a_eq, b_eq=b_eq, a_ineq=a_ineq, b_ineq=b_ineq
    )


def index_mask(n: int, idx) -> np.ndarray:
    """Boolean mask of length ``n`` that is true on ``idx``."""
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def walk_avals(jaxpr: Any):
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
                    yield from walk_avals(inner)


# ---------------------------------------------------------------------------
# Dense references of the elastic (relaxed) QP (tiny ``n``, reference use only)
# ---------------------------------------------------------------------------


def elastic_column_np(qp: RandomQP) -> np.ndarray:
    """Column ``c`` of ``δ``: ``b_E`` on the equalities, ``min(b_I, 0)`` on the rest.

    The relaxed rows are ``A_E d + δ b_E = b_E`` and ``A_I d + δ c_I ≤ b_I``.
    """
    return np.concatenate([qp.b_eq, np.minimum(qp.b_ineq, 0.0)])


def _relaxed_rows(qp: RandomQP):
    a_all = np.concatenate([qp.a_eq, qp.a_ineq])
    b_all = np.concatenate([qp.b_eq, qp.b_ineq])
    return a_all, b_all, qp.a_eq.shape[0]


def delta_min_lp(
    qp: RandomQP, lo: np.ndarray | None = None, hi: np.ndarray | None = None
) -> float | None:
    """Smallest ``δ ∈ [0, 1]`` with a step ``lo ≤ d ≤ hi`` meeting the relaxed rows.

    Solved as a small LP over ``(d, δ)``. ``lo`` / ``hi`` default to the box of
    ``qp`` (``lb - x``, ``ub - x``); pinned variables are expressed as
    ``lo == hi``. Returns ``None`` when no ``δ ∈ [0, 1]`` makes the relaxed
    rows feasible.
    """
    from scipy.optimize import linprog

    lo = qp.lb - qp.x if lo is None else lo
    hi = qp.ub - qp.x if hi is None else hi
    n = qp.x.shape[0]
    m_e, m_i = qp.a_eq.shape[0], qp.a_ineq.shape[0]
    c_i = np.minimum(qp.b_ineq, 0.0)
    bounds = [
        (None if np.isneginf(a) else a, None if np.isposinf(b) else b)
        for a, b in zip(lo, hi)
    ] + [(0.0, 1.0)]
    cost = np.zeros(n + 1)
    cost[-1] = 1.0
    out = linprog(
        cost,
        A_ub=np.hstack([qp.a_ineq, c_i[:, None]]) if m_i else None,
        b_ub=qp.b_ineq if m_i else None,
        A_eq=np.hstack([qp.a_eq, qp.b_eq[:, None]]) if m_e else None,
        b_eq=qp.b_eq if m_e else None,
        bounds=bounds,
        method="highs",
    )
    return float(out.fun) if out.status == 0 else None


def assert_elastic_kkt(
    qp: RandomQP,
    rho: float,
    d: np.ndarray,
    delta: float,
    dual,
    *,
    pinned: np.ndarray | None = None,
    tol: float = 1e-7,
) -> None:
    """KKT conditions of the elastic QP (convex, so they certify the minimiser).

    ``min gᵀd + ½dᵀBd + ½ρδ²`` s.t. ``A_E d + δ b_E = b_E``,
    ``A_I d + δ c_I ≤ b_I``, ``0 ≤ δ ≤ 1`` (``c`` = :func:`elastic_column_np`)
    and, unless ``pinned`` is given, the box ``lb ≤ x + d ≤ ub``. With
    ``pinned`` (a boolean mask of variables held at ``d``), stationarity is
    required on the other coordinates only and the box is not checked. The
    multipliers of ``δ``'s rows are not returned by the code under test; the
    sign pattern of ``ρδ + cᵀλ`` (``= μ_lo - μ_hi``) stands in for them.
    """
    lam_e = np.asarray(dual.eq_multipliers)
    lam_i = np.asarray(dual.ineq_multipliers)
    lam_lb, lam_ub = np.asarray(dual.lb_multipliers), np.asarray(dual.ub_multipliers)
    n = qp.x.shape[0]
    free = np.ones(n, bool) if pinned is None else ~pinned
    c = elastic_column_np(qp)
    c_e, c_i = c[: qp.a_eq.shape[0]], c[qp.a_eq.shape[0] :]
    scale = max(1.0, np.linalg.norm(qp.g), np.linalg.norm(qp.dense_b @ d))
    # Primal feasibility of the relaxed rows and of the box.
    assert -tol <= delta <= 1.0 + tol
    np.testing.assert_allclose(
        qp.a_eq @ d + delta * c_e - qp.b_eq,
        0.0,
        atol=tol * max(1.0, abs(qp.b_eq).max(initial=0.0)),
    )
    slack = qp.b_ineq - qp.a_ineq @ d - delta * c_i
    assert np.all(slack >= -tol * max(1.0, np.abs(qp.b_ineq).max(initial=0.0)))
    gap_lo, gap_hi = qp.x + d - qp.lb, qp.ub - qp.x - d
    if pinned is None:
        assert np.all(gap_lo >= -1e-9) and np.all(gap_hi >= -1e-9)
    # Stationarity in d (free coordinates) and the multiplier signs.
    resid = qp.dense_b @ d + qp.g + qp.a_eq.T @ lam_e + qp.a_ineq.T @ lam_i
    resid = resid - lam_lb + lam_ub
    assert np.max(np.abs(resid[free]), initial=0.0) <= tol * scale * 10
    assert np.all(lam_i >= -tol * scale)
    assert np.all(np.abs(lam_i * slack) <= tol * scale * 10)
    if pinned is None:
        assert np.all(lam_lb >= -tol * scale) and np.all(lam_ub >= -tol * scale)
        lo_fin, hi_fin = np.isfinite(gap_lo), np.isfinite(gap_hi)
        assert np.all(np.abs(lam_lb[lo_fin] * gap_lo[lo_fin]) <= tol * scale * 10)
        assert np.all(np.abs(lam_ub[hi_fin] * gap_hi[hi_fin]) <= tol * scale * 10)
    # Stationarity in δ: ρδ + cᵀλ = μ_lo - μ_hi with μ ≥ 0, complementary to δ.
    r_delta = rho * delta + c_e @ lam_e + c_i @ lam_i
    r_tol = tol * max(1.0, abs(rho * delta), abs(c) @ np.abs(np.r_[lam_e, lam_i])) * 10
    assert (r_delta >= -r_tol) | (delta > 1e-9)  # δ = 0: μ_lo ≥ 0
    assert (r_delta <= r_tol) | (delta < 1.0 - 1e-9)  # δ = 1: μ_hi ≥ 0
    assert (abs(r_delta) <= r_tol) | (delta < 1e-9) | (delta > 1.0 - 1e-9)
