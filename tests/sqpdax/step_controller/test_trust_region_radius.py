"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.trust_region_radius`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, SubProblemSolverState

from .conftest import make_tr_manager, make_tr_state


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
