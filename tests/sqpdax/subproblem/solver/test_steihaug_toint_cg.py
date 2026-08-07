"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.steihaug_toint_cg`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    SteihaugTointCGTangentialStepSolver,
)
from tests.sqpdax.subproblem.conftest import (
    make_scaled_barrier_subproblem,
    make_zero_dual,
)

from .conftest import make_qp_subproblem, make_steihaug_state


def _zero_warm(sub):
    lag = sub.lagrangian
    return (
        jax.tree.map(jnp.zeros_like, lag.ref),
        make_zero_dual(lag.n, lag.meq, lag.mineq),
    )


@pytest.mark.parametrize(
    ("radius", "expect_boundary"),
    [(1.0, False), (1e-4, True)],
    ids=["unit-radius", "tiny-radius"],
)
def test_scaled_barrier_trust_region(radius: float, expect_boundary: bool):
    """Zero warm-start: finite success, ``‖w‖ ≤ radius``, dual ≈ 0."""
    sub = make_scaled_barrier_subproblem()
    solver = SteihaugTointCGTangentialStepSolver()
    warm = _zero_warm(sub)
    (step_p, step_d), state = solver.solve(sub, warm, make_steihaug_state(radius))

    w_norm = jnp.linalg.norm(step_p.flatten())
    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert jnp.all(jnp.isfinite(step_p.flatten()))
    assert w_norm <= radius * (1.0 + 1e-6) + 1e-8
    assert jnp.allclose(step_d.flatten(), 0.0, atol=1e-12)
    assert (not expect_boundary) or bool(state.on_boundary)


def test_scaled_barrier_with_supplied_active_bounds():
    """Caller-supplied ``active_bounds`` skips gradient-projection identification."""
    sub = make_scaled_barrier_subproblem()
    n = sub.lagrangian.n
    solver = SteihaugTointCGTangentialStepSolver()
    warm = _zero_warm(sub)
    active = (jnp.zeros(n, dtype=bool), jnp.zeros(n, dtype=bool))
    (step_p, _), state = solver.solve(
        sub, warm, make_steihaug_state(1.0, active_bounds=active)
    )

    assert bool(state.success)
    assert state.active_bounds is not None
    assert jnp.array_equal(state.active_bounds[0], active[0])
    assert jnp.array_equal(state.active_bounds[1], active[1])
    assert jnp.all(jnp.isfinite(step_p.flatten()))


def test_active_set_subproblem_raises():
    """Active-set models are rejected with ``TypeError``."""
    sub = make_qp_subproblem()
    solver = SteihaugTointCGTangentialStepSolver()
    warm = _zero_warm(sub)
    with pytest.raises(TypeError, match="ScaledBarrierSubProblem"):
        solver.solve(sub, warm, make_steihaug_state(1.0))


def test_large_radius_model_descent():
    """Returned step is a descent direction for the quadratic model."""
    sub = make_scaled_barrier_subproblem()
    solver = SteihaugTointCGTangentialStepSolver()
    warm = _zero_warm(sub)
    step, state = solver.solve(sub, warm, make_steihaug_state(10.0))

    g = sub.primal_grad().flatten()
    w = step[0].flatten()
    Hw = sub.kkt_mvp_primal((step[0], jax.tree.map(jnp.zeros_like, step[1]))).flatten()
    model = jnp.dot(g, w) + 0.5 * jnp.dot(w, Hw)
    assert bool(state.success)
    assert model <= 1e-8


@pytest.mark.parametrize(
    ("w", "slack_mask", "n", "tau", "expected"),
    [
        # Empty slack block (n == len(w)) → β = 1 by definition.
        (jnp.array([0.5, -0.25]), jnp.array([0.0, 0.0]), 2, 0.995, 1.0),
        # Negative active slack step is clipped by the FTB rule.
        (
            jnp.array([0.0, -0.5]),
            jnp.array([0.0, 1.0]),
            1,
            0.5,
            1.0,  # τ / 0.5 = 1 → still β = 1
        ),
        (
            jnp.array([0.0, -1.0]),
            jnp.array([0.0, 1.0]),
            1,
            0.5,
            0.5,  # τ / 1.0 = 0.5
        ),
    ],
    ids=["empty-slack", "ftb-inactive", "ftb-clip"],
)
def test_fraction_to_boundary_beta(w, slack_mask, n, tau, expected):
    """``_fraction_to_boundary_beta`` covers empty-slack and clipping paths."""
    beta = SteihaugTointCGTangentialStepSolver._fraction_to_boundary_beta(
        w, slack_mask, n, tau
    )
    assert jnp.isclose(beta, expected, atol=1e-8)
