"""Fixtures for the ``benchmarks`` package tests.

``import sif2jax`` takes ~10 s, so everything touching sif2jax is marked
``slow`` and shares a session-scoped import.
"""

from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.problem import Problem, build_problem


@pytest.fixture(scope="session")
def sif2jax():
    return pytest.importorskip("sif2jax")


@pytest.fixture
def kkt_problem() -> tuple[Problem, Array, Dual, float]:
    """``min (x0-1)^2 + (x1-2)^2 + x2^2`` with ``x0 + x1 = 1``, ``x2 >= 0.5``, ``x1 <= 1``.

    Solution: equality gives ``x1 = 1 - x0``; minimising
    ``(x0-1)^2 + (x0+1)^2`` gives ``x0 = 0``, ``x1 = 1`` (bound active),
    ``x2 = 0.5`` (bound active). Multipliers: ``lam = 2`` on the equality,
    ``z_ub1 = 0`` (since ``x1 = 1`` is enforced by the equality and
    ``grad1 + lam = -2 + 2 = 0``), ``z_lb2 = 1``.
    """
    problem = build_problem(
        lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + x[2] ** 2,
        n=3,
        meq=1,
        eq_fn=lambda x: jnp.array([x[0] + x[1] - 1.0]),
        lb=jnp.array([-jnp.inf, -jnp.inf, 0.5]),
        ub=jnp.array([jnp.inf, 1.0, jnp.inf]),
        autodiff_mode="jax",
    )
    x = jnp.array([0.0, 1.0, 0.5])
    dual = Dual(
        eq_multipliers=jnp.array([2.0]),
        ineq_multipliers=jnp.zeros(0),
        lb_multipliers=jnp.array([0.0, 0.0, 1.0]),
        ub_multipliers=jnp.zeros(3),
    )
    return problem, x, dual, 2.25


@pytest.fixture
def hs71_problem() -> tuple[Problem, Array, float]:
    """Hock-Schittkowski 71 written directly in sqpdax form, with exact HVPs.

    ``min x0 x3 (x0 + x1 + x2) + x2`` subject to ``sum x^2 = 40``,
    ``prod x >= 25`` and ``1 <= x <= 5``; the lower bound on ``x0`` is active
    at the optimum (``f* = 17.0140173``), so every multiplier block of
    :class:`~slsqp_jax.sqpdax.dual.Dual` is exercised.
    """
    problem = build_problem(
        lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        n=4,
        meq=1,
        mineq=1,
        eq_fn=lambda x: jnp.array([jnp.sum(x**2) - 40.0]),
        ineq_fn=lambda x: jnp.array([25.0 - jnp.prod(x)]),
        lb=jnp.ones(4),
        ub=5.0 * jnp.ones(4),
        autodiff_mode="jax",
        force_hvp_in_jax_mode=True,
    )
    return problem, jnp.array([1.0, 5.0, 5.0, 1.0]), 17.0140173


@pytest.fixture
def toy_results() -> pd.DataFrame:
    """Two instances x three configs covering every outcome class."""
    return pd.DataFrame(
        {
            "problem": ["A", "A", "A", "B", "B", "B"],
            "y0_iD": 0,
            "collection": "constrained",
            "n": [3, 3, 3, 10, 10, 10],
            "config": ["s1", "s2", "s3"] * 2,
            "status": [
                "successful",
                "successful",
                "max_steps_reached",
                "successful",
                "timeout",
                "successful",
            ],
            "successful": [True, True, False, True, False, True],
            "feas": [0.0, 1e-3, 0.0, 1e-9, None, 0.0],
            "f_gap": [1e-9, 1e-9, 1e-9, 1e-2, None, 0.0],
            "has_fstar": True,
            "finite": True,
            "time_median_s": [1.0, 2.0, 3.0, 4.0, None, 2.0],
            "steps": [5, 6, 500, 10, None, 20],
            "compile_s": 1.0,
        }
    )
