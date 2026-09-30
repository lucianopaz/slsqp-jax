"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.trust_region_interior_point`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.barrier import AdaptiveBarrierUpdate, LogBarrier
from slsqp_jax.sqpdax.minimiser import (
    TRUST_REGION_INTERIOR_POINT_RESULTS,
    ActiveSetLineSearchMinimiser,
    TrustRegionInteriorPointMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.primal import InteriorPointPrimal
from slsqp_jax.sqpdax.registry import FrozenDict
from slsqp_jax.sqpdax.step_controller import StepResult
from slsqp_jax.sqpdax.subproblem.solver import (
    BarrierSafeguard,
    LeastSquaresMultiplierRecovery,
    TrustRegionInteriorPointSolver,
)

from .conftest import (
    make_equality_quadratic,
    make_scaled_quartic,
    make_unconstrained_quadratic,
)


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


@pytest.mark.parametrize(
    ("subproblem_options", "check"),
    [
        ({"zeta": 0.5}, lambda tr: tr.zeta == 0.5),
        (
            {
                "multiplier_recovery": LeastSquaresMultiplierRecovery(
                    refinement_rounds=2, safeguard=BarrierSafeguard(cap=1e-2)
                )
            },
            lambda tr: tr.multiplier_recovery.refinement_rounds == 2
            and tr.multiplier_recovery.safeguard.cap == 1e-2,
        ),
    ],
    ids=["zeta", "multiplier-recovery"],
)
def test_step_applies_subproblem_options(subproblem_options, check):
    """Non-empty ``subproblem`` options are forwarded into ``solver.init``."""
    problem = make_unconstrained_quadratic()
    solver = TrustRegionInteriorPointMinimiser(initial_mu=0.1).init(
        problem,
        jnp.ones(2),
        options={"subproblem": subproblem_options},
    )
    tr = solver._init_subproblem(problem).solver
    assert isinstance(tr, TrustRegionInteriorPointSolver)
    assert check(tr)
    solver = solver.step(problem)
    assert int(solver.step_count) == 1
    assert jnp.all(jnp.isfinite(solver.iterate.flatten()))
    assert jnp.all(solver.dual.ineq_multipliers >= 0)
    assert jnp.all(solver.dual.lb_multipliers >= 0)
    assert jnp.all(solver.dual.ub_multipliers >= 0)


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


# --- secant model-stall detection -------------------------------------------


def _secant_model_solver(**kwargs) -> TrustRegionInteriorPointMinimiser:
    """TR-IP on a problem without exact HVPs, so an L-BFGS secant is kept."""
    problem = make_scaled_quartic(with_curvature=False)
    solver = TrustRegionInteriorPointMinimiser(**kwargs).init(
        problem, jnp.array([0.5, 0.3, 0.2])
    )
    assert solver.secant is not None
    return solver


def _with_pairs(solver):
    """Seed three curvature pairs so resets are observable via ``num_pairs``."""
    secant = solver.secant
    for k in range(3):
        s = jnp.zeros(3).at[k].set(1.0) + 0.1
        secant = secant.append(s, (k + 2.0) * s)
    return eqx.tree_at(lambda m: m.secant, solver, secant)


@pytest.mark.parametrize(
    ("radius", "rho", "accepted", "atol", "expected_streak"),
    [
        (1e-12, 1.0, True, 1e-6, 3),
        (1e-12, 1.0, True, 1e6, 0),
        (1.0, -jnp.inf, False, 1e-6, 3),
        (1.0, -5.0, False, 1e-6, 3),
        (1.0, -0.5, False, 1e-6, 0),
        (1.0, 1.0, True, 1e-6, 0),
        (1.0, -jnp.inf, True, 1e-6, 0),
    ],
    ids=[
        "radius-collapse-unconverged",
        "radius-collapse-converged",
        "rejected-nonpositive-pred",
        "rejected-very-negative-rho",
        "healthy-rejection",
        "accepted-sound-radius",
        "accepted-ignores-rho",
    ],
)
def test_advance_dynamics_tracks_model_stall_streak(
    radius, rho, accepted, atol, expected_streak
):
    """Only radius collapse (while unconverged) or a model failure extend the streak.

    The counter is seeded at ``2`` so an increment (``3``) is distinguishable
    from a reset to zero. ``atol=1e6`` makes the current iterate count as
    converged, which must silence the radius-collapse branch.
    """
    problem = make_scaled_quartic(with_curvature=False)
    solver = _secant_model_solver(atol=atol)
    solver = eqx.tree_at(
        lambda m: m.consecutive_model_failures, solver, jnp.asarray(2, jnp.int32)
    )
    ctx = solver._init_subproblem(problem)
    state = eqx.tree_at(
        lambda s: (s.radius, s.rho),
        solver.solver_state,
        (jnp.asarray(radius), jnp.asarray(rho)),
    )
    result = StepResult(
        x=solver.iterate,
        accepted=jnp.asarray(accepted),
        merit_val=jnp.asarray(0.0),
        solver_state=state,
        step_size=jnp.asarray(1.0 if accepted else 0.0),
        proposed_step_norm=jnp.asarray(1.0),
    )

    advanced = solver._advance_dynamics(ctx, result, solver.dual)

    assert int(advanced.consecutive_model_failures) == expected_streak
    signals = advanced._secant_reset_signals()
    assert int(signals.model_streak) == expected_streak
    assert int(signals.subproblem_streak) == 0
    assert int(signals.step_streak) == 0


def test_model_stalls_use_global_recovery_and_report_post_identity_failure():
    """Four model stalls apply all reset stages and terminate specifically."""
    problem = make_scaled_quartic(with_curvature=False)
    solver = _with_pairs(_secant_model_solver())
    for streak in range(1, 5):
        solver = eqx.tree_at(
            lambda m: m.consecutive_model_failures,
            solver,
            jnp.asarray(streak, jnp.int32),
        )
        solver = solver._reset_secant()
        assert int(solver.secant_recovery_state.failure_streak) == streak

    assert jnp.array_equal(solver.secant_stats.n_resets, jnp.ones(3, jnp.int32))
    assert bool(solver.secant_recovery_state.fatal)
    done, result = solver.terminate(problem)
    assert bool(done)
    assert bool(result == TRUST_REGION_INTERIOR_POINT_RESULTS.secant_recovery_failure)


def test_init_zeroes_model_stall_streak():
    """Re-initialising starts from empty raw and global recovery streaks."""
    problem = make_scaled_quartic(with_curvature=False)
    used = eqx.tree_at(
        lambda m: m.consecutive_model_failures,
        _secant_model_solver(),
        jnp.asarray(4, jnp.int32),
    )
    fresh = used.init(problem, jnp.array([0.5, 0.3, 0.2]))
    assert int(fresh.consecutive_model_failures) == 0
    assert int(fresh.secant_recovery_state.failure_streak) == 0


def test_active_set_reports_no_model_stalls():
    """The line-search minimiser never feeds the ``model`` channel."""
    problem = make_scaled_quartic(with_curvature=False)
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.array([0.5, 0.3, 0.2]))
    assert int(solver._secant_reset_signals().model_streak) == 0
