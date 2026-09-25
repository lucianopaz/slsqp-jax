"""Unit tests for :mod:`slsqp_jax.sqpdax.step_controller.line_search`."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.merit import NormMerit
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.step_controller import ArmijoLineSearch, LineSearchState
from slsqp_jax.sqpdax.subproblem.solver import RESULTS, SubProblemSolverState
from tests.sqpdax.lagrangian.conftest import obj

from .conftest import make_armijo, make_unconstrained_quadratic


def test_line_search_state_x_scales_step():
    """``LineSearchState.x`` is ``x0 + alpha * step``."""
    state = LineSearchState(
        x0=Primal(jnp.array([1.0, 2.0])),
        step=Primal(jnp.array([2.0, -4.0])),
        merit0_val=jnp.asarray(0.0),
        merit0_grad=Primal(jnp.zeros(2)),
        merit0_grad_dot_step=jnp.asarray(0.0),
        merit_val=jnp.asarray(0.0),
        merit_grad=Primal(jnp.zeros(2)),
        alpha=jnp.asarray(0.5),
    )
    assert jnp.allclose(state.x.x, jnp.array([2.0, 0.0]))


@pytest.mark.parametrize(
    ("x0", "direction", "expect_accepted", "expect_x", "expect_full_step"),
    [
        # Steepest-ish unit step on ‖x‖² reaches the origin at α = 1.
        (
            Primal(jnp.ones(2)),
            Primal(-jnp.ones(2)),
            True,
            jnp.zeros(2),
            True,
        ),
        # Oversized descent direction needs geometric backtracking.
        (
            Primal(jnp.array([1.0, 0.0])),
            Primal(jnp.array([-10.0, 0.0])),
            True,
            None,  # checked qualitatively below
            False,
        ),
        # Ascent direction cannot satisfy Armijo within the budget.
        (
            Primal(jnp.ones(2)),
            Primal(jnp.ones(2)),
            False,
            None,
            False,
        ),
    ],
    ids=["unit-descent", "backtrack", "ascent"],
)
def test_armijo_step_acceptance(
    x0, direction, expect_accepted, expect_x, expect_full_step
):
    """Accept good / backtracked descents; reject ascent within ``max_steps``."""
    ls = make_armijo(max_steps=20)
    result = ls.step(x0, direction)

    assert bool(result.accepted) is expect_accepted
    assert jnp.isfinite(result.merit_val)
    if expect_x is not None:
        assert jnp.allclose(result.x.x, expect_x, atol=1e-6)
    if expect_full_step:
        assert jnp.allclose(result.x.x, x0.x + direction.x, atol=1e-6)
        assert jnp.allclose(result.step_size, 1.0)
        assert jnp.allclose(result.proposed_step_norm, jnp.linalg.norm(direction.x))
    if expect_accepted and not expect_full_step:
        # Backtracking must land strictly between x0 and x0 + direction.
        assert not jnp.allclose(result.x.x, x0.x + direction.x, atol=1e-6)
        assert float(result.merit_val) < float(ls.merit(x0))
    if not expect_accepted:
        assert float(result.merit_val) >= float(ls.merit(x0)) - 1e-12


@pytest.mark.parametrize(
    ("x0", "direction", "max_steps"),
    [
        # Ascent direction: no trial can satisfy Armijo within the budget.
        (Primal(jnp.ones(2)), Primal(jnp.ones(2)), 20),
        # Zero budget: the search loop never runs a single trial.
        (Primal(jnp.ones(2)), Primal(-jnp.ones(2)), 0),
        # Non-finite directions are rejected up front.
        (Primal(jnp.ones(2)), Primal(jnp.array([jnp.nan, -1.0])), 20),
        (Primal(jnp.ones(2)), Primal(jnp.array([-1.0, -jnp.inf])), 20),
    ],
    ids=["ascent", "zero-budget", "nan-direction", "inf-direction"],
)
def test_armijo_rejection_retains_x0(x0, direction, max_steps):
    """A rejected search returns ``x0`` and the merit *at* ``x0``.

    This is the :class:`~slsqp_jax.sqpdax.step_controller.base.StepResult`
    contract the outer loop relies on: ``_execute_step`` commits ``result.x``
    unconditionally, so a rejected step must not move the iterate, and
    ``merit_val`` must describe the point actually returned. This holds for
    a non-finite direction as well, whose merit at every trial would be NaN.
    """
    ls = make_armijo(max_steps=max_steps)
    carry = SubProblemSolverState(
        n_iter=jnp.asarray(3, jnp.int32),
        success=jnp.asarray(True),
        status=RESULTS.successful,
    )
    result = ls.step(x0, direction, carry)

    assert not bool(result.accepted)
    assert jnp.allclose(result.x.x, x0.x)
    # Merit is unchanged by a rejected step, and matches the returned iterate.
    assert float(result.merit_val) == pytest.approx(float(ls.merit(x0)))
    assert float(result.merit_val) == pytest.approx(float(ls.merit(result.x)))
    # The subproblem carry is threaded through by value.
    assert int(result.solver_state.n_iter) == 3
    assert bool(result.solver_state.success)
    assert jnp.all(jnp.isfinite(result.x.x))
    assert jnp.isfinite(result.merit_val)


def _counting_armijo(*, max_steps: int):
    """Armijo search whose merit records every evaluation at run time."""
    calls: list[int] = []

    def counting_obj(x):
        jax.debug.callback(lambda: calls.append(1))
        return obj(x)

    problem = replace(make_unconstrained_quadratic(), fn=counting_obj)
    ls = ArmijoLineSearch(
        merit=NormMerit(problem=problem),
        max_steps=max_steps,
        backtrack=jnp.asarray(0.5),
    )
    return ls, calls


def _merit_evaluations(ls, calls, x0, direction) -> int:
    """Run one search and return the number of merit evaluations it made."""
    calls.clear()
    result = ls.step(x0, direction)
    jax.effects_barrier()
    assert not bool(result.accepted)
    return len(calls)


@pytest.mark.parametrize(
    "direction",
    [jnp.array([jnp.nan, -1.0]), jnp.array([-jnp.inf, -1.0])],
    ids=["nan", "inf"],
)
def test_nonfinite_direction_skips_trial_evaluations(direction):
    """A non-finite direction exits before a single trial merit is evaluated.

    The number of merit calls equals that of a zero-budget search on a finite
    direction (only the ``x0`` evaluation), whereas an ascent direction on
    the same budget burns every trial.
    """
    x0 = Primal(jnp.ones(2))
    ls, calls = _counting_armijo(max_steps=8)
    baseline_ls, baseline_calls = _counting_armijo(max_steps=0)

    baseline = _merit_evaluations(baseline_ls, baseline_calls, x0, Primal(-x0.x))
    nonfinite = _merit_evaluations(ls, calls, x0, Primal(direction))
    ascent = _merit_evaluations(ls, calls, x0, Primal(x0.x))

    assert nonfinite == baseline
    assert ascent == baseline + ls.max_steps


def test_armijo_small_alpha_decrease_fallback():
    """Strict ``c1`` can fail Armijo while the α < 0.1 decrease fallback accepts."""
    ls = make_armijo(max_steps=20)
    # Equinox modules are frozen; rebuild with a large Armijo constant.
    ls = ls.__class__(
        merit=ls.merit,
        max_steps=ls.max_steps,
        c1=jnp.asarray(0.9),
        backtrack=ls.backtrack,
    )
    x0 = Primal(jnp.array([1.0, 0.0]))
    direction = Primal(jnp.array([-10.0, 0.0]))
    result = ls.step(x0, direction)

    assert bool(result.accepted)
    # Fallback only fires once α has contracted below 0.1.
    alpha = float(jnp.linalg.norm(result.x.x - x0.x) / jnp.linalg.norm(direction.x))
    assert alpha < 0.1
    assert float(result.merit_val) < float(ls.merit(x0))

    # Pin the logical branch explicitly: Armijo fails, fallback succeeds.
    state = LineSearchState(
        x0=x0,
        step=direction,
        merit0_val=jnp.asarray(1.0),
        merit0_grad=Primal(jnp.array([2.0, 0.0])),
        merit0_grad_dot_step=jnp.asarray(-20.0),
        merit_val=jnp.asarray(0.5),
        merit_grad=Primal(jnp.zeros(2)),
        alpha=jnp.asarray(0.05),
    )
    assert bool(ls.stop_search(state))
