"""Tests for :func:`capture_state_at_step` and the reproducibility check.

The reproducibility check is load-bearing: without it the diagnostics
layer can silently lie about which iterate it is showing.  These tests
exercise both the success path (digest matches) and the failure path
(digest mismatch raises ``RuntimeError``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax import SLSQP
from slsqp_jax.diagnostics import capture_state_at_step, debug_run
from slsqp_jax.diagnostics.records import StepSummary

jax.config.update("jax_enable_x64", True)


def _rosenbrock(x, args):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


@pytest.fixture
def rosenbrock_run():
    solver = SLSQP()
    x0 = jnp.array([-1.2, 1.0, -0.5, 1.5])
    run = debug_run(solver, _rosenbrock, x0, max_steps=20)
    return solver, x0, run


def test_capture_recovers_state_at_step(rosenbrock_run):
    solver, x0, run = rosenbrock_run
    target = max(1, run.n_steps // 2)
    expected_summary = run.summaries[target - 1]
    state = capture_state_at_step(
        solver,
        _rosenbrock,
        x0,
        step=target,
        expected_summary=expected_summary,
    )
    # Recovered state's summary digest must match the expected one
    # (the assertion inside ``capture_state_at_step`` would have
    # raised otherwise).
    recovered = StepSummary.from_state(state)
    # Compare a few load-bearing fields directly.
    assert recovered.step_count == expected_summary.step_count
    assert recovered.f_val == pytest.approx(expected_summary.f_val, rel=1e-12)


def test_capture_reproducibility_check_detects_mismatch(rosenbrock_run):
    solver, x0, run = rosenbrock_run
    target = max(1, run.n_steps // 2)
    original = run.summaries[target - 1]
    # Tamper with the expected summary so the digest no longer matches
    # what the recovered state will produce.
    tampered = dataclasses.replace(original, f_val=original.f_val + 1.0e6)
    with pytest.raises(RuntimeError, match="not reproducible"):
        capture_state_at_step(
            solver,
            _rosenbrock,
            x0,
            step=target,
            expected_summary=tampered,
        )


def test_capture_invalid_step_raises(rosenbrock_run):
    solver, x0, _ = rosenbrock_run
    with pytest.raises(ValueError, match="step must be positive"):
        capture_state_at_step(solver, _rosenbrock, x0, step=0)


def test_capture_without_expected_summary_returns_state(rosenbrock_run):
    solver, x0, _ = rosenbrock_run
    state = capture_state_at_step(solver, _rosenbrock, x0, step=2)
    # No reproducibility check requested: state is returned without
    # raising.  Sanity-check it is a proper SLSQPState.
    assert int(state.step_count) == 2
