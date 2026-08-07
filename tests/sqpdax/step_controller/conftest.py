"""Fixtures shared by :mod:`slsqp_jax.sqpdax.step_controller` tests."""

from __future__ import annotations

import jax.numpy as jnp

from slsqp_jax.sqpdax.merit import NormMerit
from slsqp_jax.sqpdax.problem import Problem
from slsqp_jax.sqpdax.step_controller import ArmijoLineSearch, TrustRegionManager
from slsqp_jax.sqpdax.subproblem.solver import TrustRegionSolverState
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import (
    make_trust_region_state,
    unbounded_box,
)


def make_unconstrained_quadratic(*, n: int = 2) -> Problem:
    """``f(x) = ‖x‖²`` with no constraints / bounds."""
    lb, ub = unbounded_box(n)
    return make_problem(n=n, meq=0, mineq=0, lb=lb, ub=ub)


def make_armijo(*, max_steps: int = 20, backtrack: float = 0.5) -> ArmijoLineSearch:
    """Armijo line search on the unconstrained quadratic merit."""
    return ArmijoLineSearch(
        merit=NormMerit(problem=make_unconstrained_quadratic()),
        max_steps=max_steps,
        backtrack=jnp.asarray(backtrack),
    )


def make_tr_manager() -> TrustRegionManager:
    """Trust-region manager on the unconstrained quadratic merit."""
    return TrustRegionManager(merit=NormMerit(problem=make_unconstrained_quadratic()))


def make_tr_state(
    *,
    radius: float = 1.0,
    predicted_reduction: float = 1.0,
    on_boundary: bool = False,
) -> TrustRegionSolverState:
    """Cold trust-region state with prescribed prediction / boundary flag."""
    state = make_trust_region_state(radius)
    return TrustRegionSolverState(
        n_iter=state.n_iter,
        success=state.success,
        status=state.status,
        radius=state.radius,
        predicted_reduction=jnp.asarray(predicted_reduction),
        merit_penalty=state.merit_penalty,
        n_cg_iter=state.n_cg_iter,
        on_boundary=jnp.asarray(on_boundary),
    )
