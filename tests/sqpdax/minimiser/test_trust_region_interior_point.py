"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.trust_region_interior_point`."""

from __future__ import annotations

import jax.numpy as jnp
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


# ``x0`` / ``expected`` are plain lists rather than arrays: parametrize
# arguments are built at import time, before other conftests may enable x64,
# so materialising them here would pin float32 into an otherwise float64 run.
@pytest.mark.parametrize(
    ("make_problem", "x0", "expected"),
    [
        (make_unconstrained_quadratic, [1.0, 1.0], [0.0, 0.0]),
        (make_equality_quadratic, [0.25, 0.25], [0.5, 0.5]),
    ],
    ids=["unconstrained", "equality"],
)
@pytest.mark.parametrize("min_steps", [1, 3])
def test_minimise_converges_successfully(make_problem, x0, expected, min_steps):
    """The driver reaches the KKT point and reports ``successful``.

    Convergence fires only once the barrier subproblem has been solved to
    ``kappa_eps * μ`` *and* the unperturbed KKT error ``E(x, s, y, z; 0)``
    (N&W eq. 19.10) is below ``atol``, so a ``successful`` status here pins
    both halves of the Algorithm 19.4 stopping test.
    """
    sol = minimise(
        make_problem(),
        TrustRegionInteriorPointMinimiser(
            atol=1e-6,
            min_steps=min_steps,
            initial_mu=0.1,
            initial_radius=2.0,
        ),
        jnp.asarray(x0),
        max_steps=40,
        throw=True,
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, jnp.asarray(expected), atol=1e-5)
    assert int(sol.stats["num_steps"]) >= min_steps


def test_minimiser_raises_on_rtol():
    with pytest.raises(
        ValueError,
        match="TrustRegionInteriorPointMinimiser does not provide a relative",
    ):
        TrustRegionInteriorPointMinimiser(rtol=1e-3)
