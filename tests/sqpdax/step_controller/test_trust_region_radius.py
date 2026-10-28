"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.trust_region_radius`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.merit import NormMerit
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller import TrustRegionManager
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, SubProblemSolverState

from .conftest import (
    NeverValidCorrection,
    bind_active_set_correction,
    bind_interior_point_correction,
    make_circle_problem,
    make_tr_manager,
    make_tr_state,
    maratos_step,
    tangent_step,
)


@pytest.mark.parametrize(
    (
        "direction",
        "predicted_reduction",
        "on_boundary",
        "expect_accepted",
        "radius_mode",
    ),
    [
        # actual = 1 - 0.25 = 0.75, pred = 0.75 → ρ = 1 (keep; not on boundary).
        (jnp.array([-0.5, 0.0]), 0.75, False, True, "keep"),
        # Same ρ > grow_threshold with on_boundary → grow.
        (jnp.array([-0.5, 0.0]), 0.75, True, True, "grow"),
        # Ascent step → ρ < 0 → reject and shrink.
        (jnp.array([1.0, 0.0]), 1.0, False, False, "shrink"),
        # Non-positive predicted reduction → ρ = -∞ → reject and shrink.
        (jnp.array([-0.5, 0.0]), 0.0, False, False, "shrink"),
        # ρ ≈ 0.5 ∈ [shrink, grow] → accept and keep.
        (jnp.array([-0.5, 0.0]), 1.5, False, True, "keep"),
    ],
    ids=[
        "accept-keep",
        "accept-grow",
        "reject-ascent",
        "reject-bad-pred",
        "accept-mid-rho",
    ],
)
def test_trust_region_accept_and_radius(
    direction,
    predicted_reduction,
    on_boundary,
    expect_accepted,
    radius_mode,
):
    """Acceptance and radius update follow N&W ρ thresholds."""
    mgr = make_tr_manager()
    x0 = Primal(jnp.array([1.0, 0.0]))
    radius0 = 2.0
    state0 = make_tr_state(
        radius=radius0,
        predicted_reduction=predicted_reduction,
        on_boundary=on_boundary,
    )
    result = mgr.step(x0, Primal(direction), state0)

    assert bool(result.accepted) is expect_accepted
    assert result.solver_state is not None
    assert bool(result.solver_state.success) is expect_accepted

    if expect_accepted:
        assert jnp.allclose(result.x.x, x0.x + direction, atol=1e-6)
        assert jnp.isclose(result.merit_val, mgr.merit(result.x))
    else:
        assert jnp.allclose(result.x.x, x0.x, atol=1e-6)
        assert jnp.isclose(result.merit_val, mgr.merit(x0))
    assert jnp.allclose(result.proposed_step_norm, jnp.linalg.norm(result.x.x - x0.x))
    assert jnp.allclose(
        result.step_size, result.accepted.astype(result.step_size.dtype)
    )

    # ``rho`` is written back to the state: actual / predicted reduction, or
    # ``-inf`` when the model predicted no decrease.
    x_trial = Primal(x0.x + direction)
    actual = mgr.merit(x0) - mgr.merit(x_trial)
    expected_rho = actual / predicted_reduction if predicted_reduction > 0 else -jnp.inf
    assert jnp.allclose(result.solver_state.rho, expected_rho)

    new_radius = float(result.solver_state.radius)
    if radius_mode == "keep":
        assert jnp.isclose(new_radius, radius0)
    elif radius_mode == "grow":
        assert jnp.isclose(new_radius, mgr.grow_factor * radius0)
    else:
        assert jnp.isclose(new_radius, mgr.shrink_factor * radius0)


def test_trust_region_grow_respects_max_radius():
    """Grown radius is capped at ``max_radius``."""
    mgr = make_tr_manager()
    mgr = mgr.__class__(
        merit=mgr.merit,
        grow_factor=10.0,
        max_radius=3.0,
    )
    x0 = Primal(jnp.array([1.0, 0.0]))
    state0 = make_tr_state(
        radius=1.0,
        predicted_reduction=0.75,
        on_boundary=True,
    )
    result = mgr.step(x0, Primal(jnp.array([-0.5, 0.0])), state0)
    assert bool(result.accepted)
    assert jnp.isclose(result.solver_state.radius, 3.0)


@pytest.mark.parametrize(
    ("predicted_in_eps", "expected"),
    [(10.0, True), (1e6, False)],
    ids=["rounding-level", "resolvable"],
)
def test_trust_region_flags_rounding_level_predictions(predicted_in_eps, expected):
    """A tiny predicted reduction is reported as ``rounding_level`` (merit ``1``)."""
    mgr = make_tr_manager()
    x0 = Primal(jnp.array([1.0, 0.0]))
    eps = float(jnp.finfo(x0.x.dtype).eps)
    state0 = make_tr_state(radius=1.0, predicted_reduction=predicted_in_eps * eps)
    result = mgr.step(x0, Primal(jnp.array([1e-3, 0.0])), state0)
    assert not bool(result.accepted)
    assert bool(result.rounding_level) is expected


def test_trust_region_requires_trust_region_state():
    """Wrong / missing solver state raises ``TypeError``."""
    mgr = make_tr_manager()
    x0 = Primal(jnp.ones(2))
    direction = Primal(-jnp.ones(2))
    with pytest.raises(TypeError, match="TrustRegionSolverState"):
        mgr.step(x0, direction, None)
    bad = SubProblemSolverState(
        n_iter=jnp.asarray(0, jnp.int32),
        success=jnp.asarray(False),
        status=RESULTS.successful,
    )
    with pytest.raises(TypeError, match="TrustRegionSolverState"):
        mgr.step(x0, direction, bad)  # type: ignore[arg-type]


# --- second-order correction stage -------------------------------------------

ON_CIRCLE = jnp.array([np.cos(0.3), np.sin(0.3)])
PENALTY = 10.0


def _active_set_case():
    """Maratos step on the equality circle, ℓ1 merit."""
    problem = make_circle_problem()
    x0, least_norm = bind_active_set_correction(problem, ON_CIRCLE)
    merit = NormMerit(
        problem,
        barrier=None,
        problem_weight=jnp.asarray(1.0),
        feasibility_weight=jnp.asarray(PENALTY),
        norm=1,
    )
    return merit, x0, maratos_step(x0), least_norm


def _interior_point_case():
    """Tangent step on the slack-augmented inequality circle, barrier ℓ2 merit."""
    problem = make_circle_problem(inequality=True)
    x0, least_norm = bind_interior_point_correction(problem, np.sqrt(0.5) * ON_CIRCLE)
    barrier = LogBarrier(
        weight=jnp.asarray(0.1), null_lb=problem.null_lb, null_ub=problem.null_ub
    )
    merit = NormMerit(
        problem,
        barrier=barrier,
        problem_weight=jnp.asarray(1.0),
        barrier_weight=jnp.asarray(1.0),
        feasibility_weight=jnp.asarray(PENALTY),
        norm=2,
    )
    # Along the tangent that increases x₁ (descent for f).
    return merit, x0, tangent_step(x0, -0.2), least_norm


CASES = {"active-set": _active_set_case, "interior-point": _interior_point_case}


@pytest.mark.parametrize("family", CASES)
@pytest.mark.parametrize(
    ("correction", "corrected"),
    [("none", False), ("never-valid", False), ("least-norm", True)],
)
def test_maratos_step_is_accepted_only_with_a_correction(family, correction, corrected):
    merit, x0, d, least_norm = CASES[family]()
    x_soc = jax.tree.map(lambda a, b, c: a + b + c, x0, d, least_norm(x0, d).step)
    # The model predicts exactly the decrease the corrected point achieves,
    # while the uncorrected trial increases the merit.
    pred = merit(x0) - merit(x_soc)
    x_plain = jax.tree.map(jnp.add, x0, d)
    assert pred > 0.0 and merit(x_plain) > merit(x0)

    corrections = {
        "none": None,
        "never-valid": NeverValidCorrection(),
        "least-norm": least_norm,
    }
    manager = TrustRegionManager(merit=merit, correction=corrections[correction])
    result = manager.step(x0, d, make_tr_state(radius=1.0, predicted_reduction=pred))

    assert bool(result.accepted) is corrected
    assert bool(result.corrected) is corrected
    expected_x = x_soc if corrected else x0
    assert jnp.allclose(result.x.flatten(), expected_x.flatten(), atol=1e-6)
    expected_radius = 1.0 if corrected else manager.shrink_factor
    assert float(result.solver_state.radius) == pytest.approx(expected_radius)
