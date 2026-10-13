"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.base`."""

from __future__ import annotations

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
