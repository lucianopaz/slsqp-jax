"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.trust_region`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.subproblem.solver import (
    RESULTS,
    TrustRegionInteriorPointSolver,
)
from tests.sqpdax.subproblem.conftest import (
    make_scaled_barrier_subproblem,
    make_zero_dual,
)

from .conftest import make_qp_subproblem, make_trust_region_state


def _zero_warm(sub):
    lag = sub.lagrangian
    return (
        jax.tree.map(jnp.zeros_like, lag.ref),
        make_zero_dual(lag.n, lag.meq, lag.mineq),
    )


def _scaled_from_native(sub, step_p: InteriorPointPrimal) -> InteriorPointPrimal:
    """Invert ``p_s = S p̃_s`` to recover the scaled tangential step."""
    lag = sub.lagrangian
    s_ref = lag.slack
    inv_s = jnp.where(s_ref.s > 0, 1.0 / s_ref.s, 0.0)
    inv_lb = jnp.where((~lag.null_lb) & (s_ref.s_lb > 0), 1.0 / s_ref.s_lb, 0.0)
    inv_ub = jnp.where((~lag.null_ub) & (s_ref.s_ub > 0), 1.0 / s_ref.s_ub, 0.0)
    return InteriorPointPrimal(
        x=step_p.x,
        slack=Slack(
            s=step_p.slack.s * inv_s,
            s_lb=step_p.slack.s_lb * inv_lb,
            s_ub=step_p.slack.s_ub * inv_ub,
        ),
    )


@pytest.mark.parametrize(
    ("radius", "expect_boundary"),
    [(1.0, False), (1e-4, True)],
    ids=["unit-radius", "tiny-radius"],
)
def test_composite_step_finite_and_reports_pred(radius: float, expect_boundary: bool):
    """Composite step is finite; state carries pred / ν / success."""
    sub = make_scaled_barrier_subproblem()
    solver = TrustRegionInteriorPointSolver()
    warm = _zero_warm(sub)
    (step_p, step_d), state = solver.solve(
        sub, warm, make_trust_region_state(radius, merit_penalty=1.0)
    )

    assert bool(state.success)
    assert state.status == RESULTS.successful
    assert jnp.all(jnp.isfinite(step_p.flatten()))
    assert jnp.all(jnp.isfinite(step_d.flatten()))
    assert jnp.isfinite(state.predicted_reduction)
    assert state.merit_penalty >= 1.0 - 1e-12
    assert state.n_iter >= 1
    # Inequality / bound multipliers stay non-negative (eq. 19.38 safeguard).
    assert jnp.all(step_d.ineq_multipliers >= -1e-12)
    assert jnp.all(step_d.lb_multipliers >= -1e-12)
    assert jnp.all(step_d.ub_multipliers >= -1e-12)
    assert (not expect_boundary) or bool(state.on_boundary)


def test_predicted_reduction_matches_formula():
    """``predicted_reduction`` equals ``-q(w) + ν (m(0) - m(w))`` in scaled space."""
    sub = make_scaled_barrier_subproblem()
    solver = TrustRegionInteriorPointSolver()
    warm = _zero_warm(sub)
    nu0 = 2.5
    (step_p, _), state = solver.solve(
        sub, warm, make_trust_region_state(5.0, merit_penalty=nu0)
    )

    assert state.merit_penalty >= nu0 - 1e-12
    ghat = sub.primal_grad().flatten()
    chat = sub.dual_grad().flatten()
    w_primal = _scaled_from_native(sub, step_p)
    w = w_primal.flatten()
    zero_dual = make_zero_dual(
        sub.lagrangian.n, sub.lagrangian.meq, sub.lagrangian.mineq
    )
    Hw = sub.kkt_mvp_primal((w_primal, zero_dual)).flatten()
    obj_model = jnp.dot(ghat, w) + 0.5 * jnp.dot(w, Hw)
    m0 = jnp.linalg.norm(chat)
    mp = jnp.linalg.norm(
        chat + sub.kkt_mvp_lower_offdiag((w_primal, zero_dual)).flatten()
    )
    expected = -obj_model + state.merit_penalty * (m0 - mp)
    assert jnp.allclose(state.predicted_reduction, expected, rtol=1e-4, atol=1e-5)


def test_active_set_subproblem_raises():
    """Active-set models are rejected with ``TypeError``."""
    sub = make_qp_subproblem()
    solver = TrustRegionInteriorPointSolver()
    warm = _zero_warm(sub)
    with pytest.raises(TypeError, match="ScaledBarrierSubProblem"):
        solver.solve(sub, warm, make_trust_region_state(1.0))


def test_exports_roundtrip():
    """Public re-exports resolve from solver / subproblem / sqpdax packages."""
    from slsqp_jax import sqpdax
    from slsqp_jax.sqpdax import subproblem
    from slsqp_jax.sqpdax.subproblem import solver

    assert solver.TrustRegionInteriorPointSolver is TrustRegionInteriorPointSolver
    assert subproblem.TrustRegionInteriorPointSolver is TrustRegionInteriorPointSolver
    assert sqpdax.TrustRegionInteriorPointSolver is TrustRegionInteriorPointSolver
    assert solver.TrustRegionSolverState is sqpdax.TrustRegionSolverState
