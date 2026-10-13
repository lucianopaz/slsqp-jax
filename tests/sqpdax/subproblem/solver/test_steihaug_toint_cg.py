"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.steihaug_toint_cg`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.linalg import null_space_projector
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    SteihaugTointCGTangentialStepSolver,
)
from slsqp_jax.sqpdax.subproblem.solver.steihaug_toint_cg import _refined_projector
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


@pytest.mark.parametrize(
    "active_lb",
    [None, jnp.array([True, False])],
    ids=["no-active-bounds", "x0-lower-active"],
)
def test_normal_equations_strategies_agree_with_active_bounds(active_lb):
    """All three normal-equations strategies return the same masked step."""
    sub = make_scaled_barrier_subproblem()
    n = sub.lagrangian.n
    warm = _zero_warm(sub)
    active = None
    if active_lb is not None:
        active = (active_lb, jnp.zeros(n, dtype=bool))
    steps = {}
    for strategy in ("generic", "schur", "matrix-free"):
        solver = SteihaugTointCGTangentialStepSolver(normal_equations=strategy)
        (step_p, _), state = solver.solve(
            sub, warm, make_steihaug_state(1.0, active_bounds=active)
        )
        assert bool(state.success)
        steps[strategy] = step_p.flatten()
        if active_lb is not None:
            # Frozen coordinates stay put under every strategy.
            assert jnp.allclose(steps[strategy][:n][active_lb], 0.0, atol=1e-7)
    for strategy in ("schur", "matrix-free"):
        assert jnp.allclose(steps[strategy], steps["generic"], rtol=1e-4, atol=1e-5)
    assert (
        SteihaugTointCGTangentialStepSolver().resolve_normal_equations(sub) == "schur"
    )


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


@pytest.mark.parametrize("scale_error", [1e-1, 1e-2, 1e-3])
def test_refined_projector_reduces_null_space_leakage(scale_error: float):
    """A leaky projector is refined; an accurate one is left untouched.

    ``v = Aᵀλ + δ t`` is almost entirely in range(Aᵀ).  An inexact inner solve
    (scaled by ``1 - scale_error``) leaks ``scale_error * ||A Aᵀ λ||`` out of
    null(A) on one pass and ``scale_error`` times less on the second pass.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    M = jax.random.normal(k1, (3, 8))
    lam = jax.random.normal(k2, (3,))
    t = jax.random.normal(k3, (8,))
    t = t - M.T @ jnp.linalg.solve(M @ M.T, M @ t)  # exactly in null(M)
    v = M.T @ lam + 1e-9 * t
    G = M @ M.T

    def A(x):
        return M @ x

    def inexact_solve(b):
        return (1.0 - scale_error) * jnp.linalg.solve(G, b)

    proj1 = null_space_projector(A, lambda y: M.T @ y, solve=inexact_solve)
    leak1 = jnp.linalg.norm(M @ proj1(v))
    leak2 = jnp.linalg.norm(M @ _refined_projector(proj1, A, 1e-10)(v))
    assert leak1 > 0.0
    assert leak2 <= 2.0 * scale_error * leak1

    # Leak below the (large) relative threshold: the second pass is skipped.
    assert jnp.array_equal(
        _refined_projector(proj1, A, 1e6)(v),
        proj1(v),
    )
