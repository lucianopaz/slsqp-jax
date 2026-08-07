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
