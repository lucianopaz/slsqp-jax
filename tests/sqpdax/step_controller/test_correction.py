"""Tests for :mod:`slsqp_jax.sqpdax.step_controller.correction`."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller import LeastNormCorrection
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem

from .conftest import (
    bind_active_set_correction,
    bind_interior_point_correction,
    make_circle_problem,
    tangent_step,
)

THETA = 0.3


def _on_circle(radius: float) -> jax.Array:
    return radius * jnp.array([jnp.cos(THETA), jnp.sin(THETA)])


# Each family binds a correction at a point where the modelled rows vanish:
# the active-set QP on the equality circle (radius 1), the barrier subproblem
# on the inequality circle with slack ``s = 0.5`` (radius √0.5, so that
# ``|x|² − 1 + s = 0``).
FEASIBLE_RADIUS = {"active-set": 1.0, "interior-point": float(np.sqrt(0.5))}
BINDERS: dict[str, Callable] = {
    "active-set": lambda x, **kw: bind_active_set_correction(
        make_circle_problem(), x, **kw
    ),
    "interior-point": lambda x, **kw: bind_interior_point_correction(
        make_circle_problem(inequality=True), x, **kw
    ),
}
FAMILIES = list(BINDERS)


def _bind(family: str, *, radius_scale: float = 1.0, **kw):
    return BINDERS[family](_on_circle(radius_scale * FEASIBLE_RADIUS[family]), **kw)


def _add(*trees):
    return jax.tree.map(lambda *leaves: sum(leaves), *trees)


def _violation(correction, point) -> float:
    """``‖c‖_∞`` over the rows the correction targets, at ``point``."""
    sub = correction.subproblem
    residual = sub.constraint_residual(correction.lagrangian(point, correction.dual))
    return float(jnp.max(jnp.abs(residual.flatten())))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("t", [0.2, 0.05])
def test_correction_restores_feasibility_to_higher_order(family: str, t: float):
    x0, correction = _bind(family)
    d = tangent_step(x0, t)
    out = correction(x0, d)
    assert bool(out.valid)
    assert float(out.fraction) == pytest.approx(1.0)
    before = _violation(correction, _add(x0, d))
    after = _violation(correction, _add(x0, d, out.step))
    assert before == pytest.approx(t**2, rel=1e-4)
    assert float(out.residual_norm) == pytest.approx(before, rel=1e-6)
    assert after <= t**3


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    ("case", "radius_scale", "t", "enabled"),
    [
        ("disabled", 1.0, 0.1, False),
        ("nothing-to-correct", 1.0, 0.0, True),
        ("far-from-feasible", 1.3, 0.05, True),
    ],
)
def test_invalid_correction_is_zero(
    family: str, case: str, radius_scale: float, t: float, enabled: bool
):
    x0, correction = _bind(family, radius_scale=radius_scale, enabled=enabled)
    out = correction(x0, tangent_step(x0, t))
    assert not bool(out.valid), case
    assert all(bool(jnp.all(leaf == 0.0)) for leaf in jax.tree.leaves(out.step))


def test_box_face_blocks_the_correction():
    # The trial point sits on the lower bound of x₁ and the correction points
    # below it, so no fraction of the correction stays in the box.
    x0 = _on_circle(1.0)
    d = tangent_step(bind_active_set_correction(make_circle_problem(), x0)[0], 0.2)
    lb = jnp.array([x0[0] + d.x[0], -jnp.inf])
    problem = make_circle_problem(lb=lb, ub=jnp.full(2, jnp.inf))
    primal, correction = bind_active_set_correction(problem, x0)
    out = correction(primal, d)
    assert float(out.fraction) == 0.0
    assert not bool(out.valid)


def test_unbound_correction_raises():
    x0, _ = _bind("active-set")
    with pytest.raises(ValueError, match="bound"):
        LeastNormCorrection()(x0, x0)


@pytest.mark.parametrize("soc_threshold", [0.0, -1.0])
def test_soc_threshold_must_be_positive(soc_threshold: float):
    with pytest.raises(ValueError, match="soc_threshold"):
        LeastNormCorrection(soc_threshold=soc_threshold)


def _landing_on_a_bound(**kwargs):
    """Tangent step that lands x₁ on its lower bound, which is in the working set."""
    x0 = _on_circle(1.0)
    d = tangent_step(bind_active_set_correction(make_circle_problem(), x0)[0], 0.05)
    lb = jnp.array([x0[0] + d.x[0], -jnp.inf])
    problem = make_circle_problem(lb=lb, ub=jnp.full(2, jnp.inf))
    primal, correction = bind_active_set_correction(
        problem, x0, active_lb=(True, False), **kwargs
    )
    return primal, correction, d


def test_reaching_a_working_set_bound_is_not_infeasibility():
    # The distance of x₁ to its face is a third of the step: as residual it
    # would fail the near-feasibility gate, as a satisfied bound it must not.
    x0, correction, d = _landing_on_a_bound()
    out = correction(x0, d)
    assert bool(out.valid)
    assert float(out.fraction) == pytest.approx(1.0)
    assert float(out.step.x[0]) == 0.0
    # With x₁ held on its face only x₂ corrects: still third order in |d|.
    t = float(jnp.linalg.norm(d.x))
    before = _violation(correction, _add(x0, d))
    after = _violation(correction, _add(x0, d, out.step))
    assert after < before
    assert after <= 4.0 * t**3


class _NoisyActiveSetSubProblem(ActiveSetSubProblem):
    """Least-norm steps carry round-off on x₁, which the working set fixes."""

    def least_norm_step(self, rhs, *, tol=1e-10, max_iter=None):
        w = super().least_norm_step(rhs, tol=tol, max_iter=max_iter)
        return Primal(w.x.at[0].add(-1e-14))


def test_round_off_on_a_fixed_coordinate_does_not_block_the_correction():
    # An outward speck of 1e-14 on the face x₁ sits on would leave no room in
    # the box; it is below the accuracy of the solve and must be ignored.
    x0, correction, d = _landing_on_a_bound(
        correction=LeastNormCorrection(soc_threshold=1e6),
        subproblem_type=_NoisyActiveSetSubProblem,
    )
    out = correction(x0, d)
    assert bool(out.valid)
    assert float(out.fraction) == pytest.approx(1.0)
    assert float(out.step.x[0]) == 0.0
