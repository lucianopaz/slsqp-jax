"""Fixtures shared by :mod:`slsqp_jax.sqpdax.subproblem.solver` tests."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    ActiveSetQPSolverState,
    DogLegSolverState,
    ProjectedCGState,
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


def make_active_set_qp_state() -> ActiveSetQPSolverState:
    """Cold :class:`ActiveSetQPSolverState`."""
    return ActiveSetQPSolverState(
        n_iter=jnp.zeros((), jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
        n_cg_iter=jnp.zeros((), jnp.int32),
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
