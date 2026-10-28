"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.base`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.merit import NormMerit
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller.base import (
    ROUNDING_LEVEL_FACTOR,
    MeritStepController,
    StepController,
    StepResult,
    at_rounding_level,
)
from slsqp_jax.sqpdax.subproblem.solver.base import SubProblemSolverState
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import (
    bind_active_set_correction,
    make_circle_problem,
    make_funnel_manager,
    tangent_step,
)


class _BareController(StepController[Primal, SubProblemSolverState]):
    """Merit-free controller used to check the base no longer requires ``merit``."""

    def step(self, x0, direction, solver_state=None):
        raise NotImplementedError


class _MeritOnly(MeritStepController[Primal, SubProblemSolverState]):
    def step(self, x0, direction, solver_state=None):
        raise NotImplementedError


def test_step_controller_constructs_without_merit():
    """The merit-agnostic base is constructible with only a logger default."""
    ctrl = _BareController()
    assert not hasattr(ctrl, "merit") or getattr(ctrl, "merit", None) is None
    assert ctrl.logger is not None


def test_merit_step_controller_requires_merit():
    """Merit-driven controllers still require a :class:`Merit` instance."""
    with pytest.raises(TypeError):
        _MeritOnly()
    problem = make_problem(meq=0, mineq=0)
    ctrl = _MeritOnly(merit=NormMerit(problem=problem))
    assert ctrl.merit is not None


@pytest.mark.parametrize(
    ("predicted_in_eps", "merit0", "expected"),
    [
        # Below ``factor * eps * max(1, |merit0|)`` in both scale regimes.
        (0.5 * ROUNDING_LEVEL_FACTOR, 0.0, True),
        (0.5 * ROUNDING_LEVEL_FACTOR * 1e3, 1e3, True),
        # Above it: the sign of the decrease does not matter, only its size.
        (2.0 * ROUNDING_LEVEL_FACTOR, 0.0, False),
        (-2.0 * ROUNDING_LEVEL_FACTOR * 1e3, 1e3, False),
        # A non-finite prediction is a model failure, never rounding.
        (jnp.nan, 1.0, False),
        (-jnp.inf, 1.0, False),
    ],
    ids=["tiny-unit-scale", "tiny-large-merit", "resolvable", "negative", "nan", "inf"],
)
def test_at_rounding_level_compares_with_the_merit_scale(
    predicted_in_eps, merit0, expected
):
    """``|predicted| <= factor * eps * max(1, |merit0|)`` for finite predictions."""
    merit0 = jnp.asarray(merit0)
    eps = jnp.finfo(merit0.dtype).eps
    flag = at_rounding_level(jnp.asarray(predicted_in_eps * eps, merit0.dtype), merit0)
    assert bool(flag) is expected


def test_step_result_rounding_level_defaults_to_false():
    """Controllers that do not compute a prediction report a resolvable step."""
    result = StepResult(
        x=Primal(jnp.zeros(2)), accepted=jnp.asarray(True), merit_val=jnp.asarray(0.0)
    )
    assert not bool(result.rounding_level)


class _Correctable(StepController[Primal, SubProblemSolverState]):
    """Controller that declares a correction stage (for the base mechanics)."""

    supports_correction = True

    def step(self, x0, direction, solver_state=None):
        raise NotImplementedError


def _bound_correction():
    x0 = jnp.array([jnp.cos(0.3), jnp.sin(0.3)])
    return bind_active_set_correction(make_circle_problem(), x0)


@pytest.mark.parametrize(
    "build",
    [
        lambda correction: _BareController(correction=correction),
        lambda correction: make_funnel_manager(make_problem(), correction=correction),
    ],
    ids=["bare", "trust-funnel"],
)
def test_controller_without_a_stage_refuses_a_correction(build):
    _, correction = _bound_correction()
    with pytest.raises(ValueError, match="no second-order correction stage"):
        build(correction)
    assert build(None).correction is None


def test_corrected_point_adds_the_correction():
    x0, correction = _bound_correction()
    d = tangent_step(x0, 0.1)
    x, result = _Correctable(correction=correction)._corrected_point(x0, d)
    assert bool(result.valid)
    expected = jax.tree.map(lambda a, b, c: a + b + c, x0, d, result.step)
    assert jnp.allclose(x.x, expected.x)


def test_corrected_point_needs_a_correction():
    x0, _ = _bound_correction()
    with pytest.raises(ValueError, match="carries no correction"):
        _Correctable()._corrected_point(x0, x0)


def test_step_result_is_uncorrected_by_default():
    result = StepResult(x=Primal(jnp.zeros(2)), accepted=True, merit_val=0.0)
    assert not bool(result.corrected)
