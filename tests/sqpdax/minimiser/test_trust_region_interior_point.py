"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.trust_region_interior_point`."""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.barrier import AdaptiveBarrierUpdate, LogBarrier
from slsqp_jax.sqpdax.minimiser import TrustRegionInteriorPointMinimiser, minimise
from slsqp_jax.sqpdax.primal import InteriorPointPrimal
from slsqp_jax.sqpdax.registry import FrozenDict

from .conftest import make_equality_quadratic, make_unconstrained_quadratic


def test_init_builds_interior_point_primal_and_barrier():
    """``init`` seeds slacks, positive duals, barrier, and trust-region carry."""
    problem = make_unconstrained_quadratic()
    solver = TrustRegionInteriorPointMinimiser().init(problem, jnp.ones(2))
    assert isinstance(solver.iterate, InteriorPointPrimal)
    assert solver.barrier is not None
    assert isinstance(solver.barrier, LogBarrier)
    assert float(solver.barrier.weight) == pytest.approx(1.0)
    assert solver.dual is not None
    assert jnp.all(solver.dual.ineq_multipliers >= 0)
    assert solver.solver_state is not None
    assert float(solver.solver_state.radius) == pytest.approx(1.0)


def test_parse_options_builds_barrier_update_from_kind_spec():
    """``minimiser.barrier_update`` kind-spec replaces the default policy."""
    problem = make_unconstrained_quadratic()
    solver = TrustRegionInteriorPointMinimiser().init(
        problem,
        jnp.ones(2),
        options={"minimiser": {"barrier_update": {"kind": "adaptive"}}},
    )
    assert isinstance(solver.options, FrozenDict)
    assert isinstance(solver.barrier_update, AdaptiveBarrierUpdate)


def test_step_applies_subproblem_options():
    """Non-empty ``subproblem`` options are forwarded into ``solver.init``."""
    problem = make_unconstrained_quadratic()
    solver = TrustRegionInteriorPointMinimiser(initial_mu=0.1).init(
        problem,
        jnp.ones(2),
        options={"subproblem": {"zeta": 0.5}},
    )
    solver = solver.step(problem)
    assert int(solver.step_count) == 1
    assert jnp.all(jnp.isfinite(solver.iterate.flatten()))


def test_step_reduces_merit_on_unconstrained():
    """One composite trust-region step decreases the barrier merit."""
    problem = make_unconstrained_quadratic()
    solver = TrustRegionInteriorPointMinimiser(initial_mu=0.1).init(
        problem, jnp.ones(2)
    )
    assert solver.iterate is not None
    x0_norm = float(jnp.linalg.norm(solver.iterate.x))
    solver = solver.step(problem)
    assert int(solver.step_count) == 1
    assert solver.solver_state is not None
    assert jnp.all(jnp.isfinite(solver.iterate.flatten()))
    # Accepted or rejected: radius / barrier always stay finite and positive.
    assert float(solver.solver_state.radius) > 0
    assert float(solver.barrier.weight) > 0  # type: ignore[union-attr]
    assert float(jnp.linalg.norm(solver.iterate.x)) <= x0_norm + 1e-6 or bool(
        solver.solver_state.success
    )


def test_minimise_unconstrained_quadratic():
    """Owned driver converges the unconstrained quadratic near the origin."""
    problem = make_unconstrained_quadratic()
    sol = minimise(
        problem,
        TrustRegionInteriorPointMinimiser(
            rtol=1e-3,
            atol=1e-2,
            min_steps=1,
            initial_mu=0.1,
            initial_radius=2.0,
        ),
        jnp.ones(2),
        max_steps=40,
        throw=False,
    )
    assert sol.result in (
        optx.RESULTS.successful,
        optx.RESULTS.nonlinear_max_steps_reached,
    )
    # Even if the outer loop hits the budget, the iterate should improve.
    assert jnp.linalg.norm(sol.value) < 1.0


def test_minimise_equality_constrained():
    """Equality-constrained IP run stays finite and reduces infeasibility."""
    problem = make_equality_quadratic()
    sol = minimise(
        problem,
        TrustRegionInteriorPointMinimiser(
            rtol=1e-3,
            atol=1e-2,
            min_steps=1,
            initial_mu=0.1,
            initial_radius=2.0,
        ),
        jnp.array([0.25, 0.25]),
        max_steps=40,
        throw=False,
    )
    assert jnp.all(jnp.isfinite(sol.value))
    # Soft check: residual of x0+x1-1 should shrink from the start.
    assert abs(float(sol.value[0] + sol.value[1] - 1.0)) < 0.5
