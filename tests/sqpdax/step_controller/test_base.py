"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.base`."""

from __future__ import annotations

import pytest

from slsqp_jax.sqpdax.merit import NormMerit
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller.base import MeritStepController, StepController
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
