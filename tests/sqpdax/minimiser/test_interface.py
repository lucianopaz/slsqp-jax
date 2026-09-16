"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.interface`."""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.minimiser import TrustRegionInteriorPointMinimiser, minimise

from .conftest import (
    ActiveSetLineSearchStub,
    make_equality_quadratic,
    make_unconstrained_quadratic,
)


@pytest.mark.parametrize(
    "make_solver",
    [
        lambda: ActiveSetLineSearchStub(rtol=1e-5, atol=1e-5, min_steps=1),
        lambda: TrustRegionInteriorPointMinimiser(
            atol=1e-6, min_steps=1, initial_mu=0.1, initial_radius=2.0
        ),
    ],
    ids=["active-set-linesearch", "trust-region-interior-point"],
)
# ``x0`` / ``expected`` are plain lists rather than arrays: parametrize
# arguments are built at import time, before other conftests may enable x64,
# so materialising them here would pin float32 into an otherwise float64 run.
@pytest.mark.parametrize(
    ("make_problem", "x0", "expected"),
    [
        (make_unconstrained_quadratic, [1.0, 1.0], [0.0, 0.0]),
        (make_equality_quadratic, [0.25, 0.25], [0.5, 0.5]),
    ],
    ids=["unconstrained", "equality"],
)
def test_minimise_reports_successful_on_quadratics(
    make_solver, make_problem, x0, expected
):
    """Every minimiser converges these quadratics and reports ``successful``.

    ``throw=True`` would raise on any non-successful status, so this also
    pins that the shared ``classify_termination`` path returns
    :attr:`optimistix.RESULTS.successful` rather than merely exhausting the
    step budget.
    """
    sol = minimise(
        make_problem(), make_solver(), jnp.asarray(x0), max_steps=40, throw=True
    )
    assert sol.result == optx.RESULTS.successful
    assert jnp.allclose(sol.value, jnp.asarray(expected), atol=1e-4)
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
