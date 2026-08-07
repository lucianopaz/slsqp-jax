"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.active_set_linesearch`."""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.minimiser import ActiveSetLineSearchMinimiser, minimise
from slsqp_jax.sqpdax.primal import Primal

from .conftest import make_equality_quadratic, make_unconstrained_quadratic


def test_init_builds_primal_dual_and_qp_state():
    """``init`` seeds a plain ``Primal``, zero dual, and cold QP carry."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.ones(2))
    assert isinstance(solver.iterate, Primal)
    assert solver.dual is not None
    assert solver.solver_state is not None
    assert int(solver.solver_state.n_iter) == 0


def test_step_and_minimise_unconstrained():
    """Outer loop drives the unconstrained quadratic to the origin."""
    problem = make_unconstrained_quadratic()
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(rtol=1e-5, atol=1e-5, min_steps=1),
        jnp.ones(2),
        max_steps=20,
        throw=True,
    )
    assert sol.result == optx.RESULTS.successful
    assert jnp.allclose(sol.value, 0.0, atol=1e-4)


def test_minimise_equality_constrained():
    """Equality-constrained quadratic lands on the feasible affine line."""
    problem = make_equality_quadratic()
    # Feasible minimiser of ‖x‖² s.t. x0+x1=1 is x=(0.5, 0.5).
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(rtol=1e-4, atol=1e-4, min_steps=1),
        jnp.array([0.0, 0.0]),
        max_steps=40,
        throw=False,
    )
    assert jnp.allclose(sol.value[0] + sol.value[1], 1.0, atol=5e-3)
    assert jnp.allclose(sol.value, jnp.array([0.5, 0.5]), atol=5e-2)


@pytest.mark.parametrize(
    "options",
    [
        None,
        {"minimiser": {"qp_tol": 1e-7}, "subproblem": {"tol": 1e-7}},
    ],
    ids=["default", "with-options"],
)
def test_init_accepts_option_bag(options):
    """Recognised minimiser / subproblem options are applied without error."""
    problem = make_unconstrained_quadratic()
    solver = ActiveSetLineSearchMinimiser().init(problem, jnp.ones(2), options=options)
    assert solver.iterate is not None
    if options is not None:
        assert solver.qp_tol == 1e-7
        # Exercise the ``subproblem`` option path inside ``_init_subproblem``.
        solver = solver.step(problem)
        assert int(solver.step_count) == 1
