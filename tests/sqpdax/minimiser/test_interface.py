"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.interface`."""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.minimiser import minimise

from .conftest import ActiveSetLineSearchStub, make_unconstrained_quadratic


def test_minimise_unconstrained_quadratic():
    """Owned driver converges the unconstrained quadratic to the origin."""
    problem = make_unconstrained_quadratic()
    sol = minimise(
        problem,
        ActiveSetLineSearchStub(rtol=1e-5, atol=1e-5, min_steps=1),
        jnp.ones(2),
        max_steps=20,
        throw=True,
    )
    assert sol.result == optx.RESULTS.successful
    assert jnp.allclose(sol.value, 0.0, atol=1e-4)
    assert int(sol.stats["num_steps"]) >= 1


def test_minimise_throw_false_on_budget_exhaustion():
    """With ``max_steps=0`` and ``throw=False`` the status is max-steps."""
    problem = make_unconstrained_quadratic()
    sol = minimise(
        problem,
        ActiveSetLineSearchStub(min_steps=1),
        jnp.ones(2),
        max_steps=0,
        throw=False,
    )
    assert sol.result == optx.RESULTS.nonlinear_max_steps_reached


def test_minimise_throw_true_raises_on_failure():
    """``throw=True`` surfaces non-convergence via ``equinox.error_if``."""
    problem = make_unconstrained_quadratic()
    with pytest.raises(Exception):
        # Force an immediate failure path: max_steps=0 keeps done=False at x≠* .
        minimise(
            problem,
            ActiveSetLineSearchStub(min_steps=1),
            jnp.ones(2),
            max_steps=0,
            throw=True,
        )
