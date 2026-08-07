"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.optimistix_compat`."""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx

from slsqp_jax.sqpdax.minimiser import as_optimistix_minimiser

from .conftest import ActiveSetLineSearchStub, make_unconstrained_quadratic


def test_as_optimistix_minimiser_runs_through_optimistix():
    """Adapter is usable with ``optimistix.minimise`` on a trivial objective."""
    problem = make_unconstrained_quadratic()
    adapter = as_optimistix_minimiser(
        ActiveSetLineSearchStub(rtol=1e-5, atol=1e-5, min_steps=1),
        problem,
    )
    assert isinstance(adapter, optx.AbstractMinimiser)

    # Dummy scalar objective: optimistix requires fn(y, args) -> scalar, but
    # our adapter ignores fn and drives the constrained problem instead.
    def fn(y, args):
        return jnp.sum(y**2), None

    sol = optx.minimise(
        fn,
        adapter,
        jnp.ones(2),
        max_steps=20,
        throw=False,
        has_aux=True,
    )
    assert sol.result == optx.RESULTS.successful
    assert jnp.allclose(sol.value, 0.0, atol=1e-3)
