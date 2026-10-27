"""``step_correction`` wiring of the minimisers (second-order correction)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    ProximalActiveSetLineSearchMinimiser,
    TrustFunnelInteriorPointMinimiser,
    TrustRegionInteriorPointMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.results import is_successful
from slsqp_jax.sqpdax.step_controller import LeastNormCorrection
from tests.sqpdax.step_controller.conftest import make_circle_problem

MINIMISERS = {
    "asls": ActiveSetLineSearchMinimiser,
    "pasls": ProximalActiveSetLineSearchMinimiser,
    "trip": TrustRegionInteriorPointMinimiser,
}
DEFAULT_CORRECTION = {"asls": None, "pasls": None, "trip": None}
THETA = 0.3


@pytest.mark.parametrize("name", MINIMISERS)
def test_default_step_correction(name: str):
    assert MINIMISERS[name]().step_correction == DEFAULT_CORRECTION[name]


@pytest.mark.parametrize("name", MINIMISERS)
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, None),
        ({"soc_threshold": 0.2}, LeastNormCorrection(soc_threshold=0.2)),
        (LeastNormCorrection(cg_tol=1e-6), LeastNormCorrection(cg_tol=1e-6)),
    ],
    ids=["none", "mapping", "instance"],
)
def test_step_correction_option(name: str, spec, expected):
    parsed = MINIMISERS[name]()._parse_options({"minimiser": {"step_correction": spec}})
    assert parsed.step_correction == expected


def test_trust_funnel_refuses_a_step_correction():
    with pytest.raises(ValueError, match="no second-order correction stage"):
        TrustFunnelInteriorPointMinimiser()._parse_options(
            {"minimiser": {"step_correction": {}}}
        )


def _solve(solver, problem, spec):
    """Float64 run from the Maratos start (the benchmark's precision)."""
    x0 = jnp.array([jnp.cos(THETA), jnp.sin(THETA)], jnp.float64)
    return minimise(
        problem,
        solver,
        x0,
        max_steps=100,
        throw=False,
        options={"minimiser": {"step_correction": spec}},
    )


def test_active_set_correction_shortens_the_maratos_run():
    # Secant model from the Maratos start: the corrected full steps save
    # outer iterations against the plain backtracking.
    with jax.enable_x64(True):
        problem = make_circle_problem()
        solver = ActiveSetLineSearchMinimiser(curvature="secant", rtol=1e-8, atol=1e-8)
        plain = _solve(solver, problem, None)
        corrected = _solve(solver, problem, {})
    assert bool(is_successful(plain.result))
    assert bool(is_successful(corrected.result))
    assert int(corrected.stats["num_steps"]) < int(plain.stats["num_steps"])
    assert int(plain.stats["n_soc_accepts"]) == 0
    assert int(corrected.stats["n_soc_accepts"]) > 0
    np.testing.assert_allclose(np.asarray(corrected.value), [1.0, 0.0], atol=1e-6)


def test_trust_region_interior_point_runs_with_a_correction():
    # Smoke test of the wiring: the inequality circle's minimiser (0.25, 0) is
    # interior, so the correction must not change where the run ends.
    with jax.enable_x64(True):
        problem = make_circle_problem(inequality=True)
        sol = _solve(TrustRegionInteriorPointMinimiser(atol=1e-8), problem, {})
    assert bool(is_successful(sol.result))
    np.testing.assert_allclose(np.asarray(sol.value), [0.25, 0.0], atol=1e-6)
