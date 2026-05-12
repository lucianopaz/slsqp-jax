"""Tests for the manual debug-run loop.

Validates that :func:`debug_run` produces a :class:`DebugRunResult`
that (a) records one :class:`StepSummary` per iteration, (b) terminates
at the same iterate as the production ``optimistix.minimise`` driver
on the same problem, and (c) honours the ``max_steps`` override.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax import SLSQP
from slsqp_jax.diagnostics import debug_run
from slsqp_jax.diagnostics.records import DebugRunResult, StepSummary

jax.config.update("jax_enable_x64", True)


def _rosenbrock(x, args):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


@pytest.fixture
def rosenbrock_solver():
    return SLSQP()


@pytest.fixture
def rosenbrock_x0():
    return jnp.array([-1.2, 1.0, -0.5, 1.5])


def test_debug_run_returns_debug_run_result(rosenbrock_solver, rosenbrock_x0):
    run = debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=20)
    assert isinstance(run, DebugRunResult)
    assert run.n_steps == len(run.summaries)
    assert all(isinstance(s, StepSummary) for s in run.summaries)


def test_debug_run_records_one_summary_per_step(rosenbrock_solver, rosenbrock_x0):
    run = debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=10)
    # The loop ran at most ``max_steps`` iterations.
    assert run.n_steps <= 10
    # Step counts inside the summaries are 1-indexed and strictly
    # increasing.
    step_counts = [s.step_count for s in run.summaries]
    assert step_counts == list(range(1, len(step_counts) + 1))


def test_debug_run_max_steps_caps_iterations(rosenbrock_solver, rosenbrock_x0):
    run = debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=3)
    assert run.n_steps == 3
    # On Rosenbrock, 3 iterations is far short of convergence; the
    # ``terminated_successfully`` property must reflect the budget
    # exhaustion rather than the latent default ``successful`` code.
    assert run.max_steps_reached
    assert not run.terminated_successfully


def test_debug_run_matches_optimistix_minimise(rosenbrock_solver, rosenbrock_x0):
    """The manual debug loop converges to the same iterate as ``optx.minimise``."""

    def fn(x, args):
        return _rosenbrock(x, args)

    sol = optx.minimise(
        fn,
        rosenbrock_solver,
        rosenbrock_x0,
        max_steps=200,
        throw=False,
    )
    run = debug_run(rosenbrock_solver, fn, rosenbrock_x0, max_steps=200)

    assert run.terminated_successfully
    assert jnp.allclose(run.final_y, sol.value, atol=1e-6, rtol=1e-6)


def test_debug_run_supports_has_aux(rosenbrock_solver, rosenbrock_x0):
    """When ``has_aux=True`` the runner accepts an objective returning ``(value, aux)``."""

    def fn_with_aux(x, args):
        return _rosenbrock(x, args), {"step_count": 0}

    run = debug_run(
        rosenbrock_solver,
        fn_with_aux,
        rosenbrock_x0,
        max_steps=10,
        has_aux=True,
    )
    assert isinstance(run, DebugRunResult)
    assert run.n_steps > 0


def test_debug_run_invalid_max_steps_raises(rosenbrock_solver, rosenbrock_x0):
    with pytest.raises(ValueError, match="max_steps must be positive"):
        debug_run(rosenbrock_solver, _rosenbrock, rosenbrock_x0, max_steps=0)
