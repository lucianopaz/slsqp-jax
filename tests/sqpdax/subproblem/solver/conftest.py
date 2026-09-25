"""Fixtures shared by :mod:`slsqp_jax.sqpdax.subproblem.solver` tests."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    ACTIVE_SET_QP_RESULTS,
    RESULTS,
    ActiveSetQPSolverState,
    DogLegSolverState,
    ProjectedCGState,
    ProximalActiveSetQPSolverState,
    SteihaugTointCGTangentialStepSolverState,
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
    return ProjectedCGState(
        n_iter=jnp.zeros((), jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
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
