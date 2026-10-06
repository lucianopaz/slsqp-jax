"""Fixtures shared by :mod:`slsqp_jax.sqpdax.step_controller` tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.merit import ConstraintViolation, NormMerit
from slsqp_jax.sqpdax.problem import Problem
from slsqp_jax.sqpdax.step_controller import (
    ArmijoLineSearch,
    TrustFunnelManager,
    TrustRegionManager,
)
from slsqp_jax.sqpdax.subproblem.solver import (
    TrustFunnelSolverState,
    TrustRegionSolverState,
)
from tests.sqpdax.lagrangian.conftest import make_problem
from tests.sqpdax.subproblem.solver.conftest import (
    make_trust_funnel_state,
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


def make_barrier_merit(problem: Problem, mu: float = 0.5) -> NormMerit:
    """Barrier function ``f(x, s) = f(x) − μ Σ log s`` as a feasibility-free merit."""
    barrier = LogBarrier(
        weight=jnp.asarray(mu), null_lb=problem.null_lb, null_ub=problem.null_ub
    )
    return NormMerit(
        problem=problem,
        barrier=barrier,
        feasibility_weight=jnp.asarray(0.0),
        norm=2,
    )


def make_funnel_manager(
    problem: Problem, *, mu: float = 0.5, **kwargs
) -> TrustFunnelManager:
    """Trust-funnel manager on ``problem`` with the barrier merit and ``v``."""
    return TrustFunnelManager(
        barrier_merit=make_barrier_merit(problem, mu),
        violation=ConstraintViolation(problem=problem, norm=2),
        **kwargs,
    )


def make_funnel_state(
    radius_v: float = 1.0,
    radius_f: float = 1.0,
    v_max: float = 10.0,
    **fields,
) -> TrustFunnelSolverState:
    """Cold funnel state with arbitrary fields overridden (``name=value``)."""
    state = make_trust_funnel_state(radius_v, radius_f, v_max)
    if not fields:
        return state
    names = tuple(fields)
    values = tuple(
        jnp.asarray(fields[k], getattr(state, k).dtype)
        if hasattr(getattr(state, k), "dtype")
        else fields[k]
        for k in names
    )
    return eqx.tree_at(lambda s: tuple(getattr(s, k) for k in names), state, values)


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
        rho=jnp.asarray(1.0),
    )
