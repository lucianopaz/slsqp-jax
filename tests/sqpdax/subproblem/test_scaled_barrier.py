"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.scaled_barrier`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.subproblem.scaled_barrier import ScaledBarrierSubProblem
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import (
    make_ip_evaluated,
    make_ip_step,
    make_scaled_barrier_subproblem,
)


def test_requires_primal_dual_lagrangian():
    """Construction rejects a non-primal-dual interior-point Lagrangian."""
    evaluated = make_ip_evaluated(primal_dual=False)
    with pytest.raises(ValueError, match="primal-dual"):
        ScaledBarrierSubProblem(evaluated)


@pytest.mark.parametrize("tau", [0.5, 0.995])
def test_primal_box_fraction_to_boundary(tau: float):
    """Scaled box: free ``x``, FTB lower faces on slacks, null bounds free."""
    problem = make_problem(
        lb=jnp.array([0.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, 3.0]),
    )
    sub = make_scaled_barrier_subproblem(problem=problem, tau=tau)
    lo, hi = sub.primal_box()
    n, mineq = problem.n, problem.mineq

    assert lo.shape == (n + mineq + 2 * n,)
    assert jnp.all(jnp.isneginf(lo[:n]))
    assert jnp.all(jnp.isposinf(hi[:n]))
    assert jnp.allclose(lo[n : n + mineq], -tau)
    assert jnp.all(jnp.isposinf(hi[n : n + mineq]))

    lo_lb = lo[n + mineq : n + mineq + n]
    lo_ub = lo[n + mineq + n :]
    assert jnp.isneginf(lo_lb[1])  # null lower bound
    assert jnp.allclose(lo_lb[0], -tau)
    assert jnp.isneginf(lo_ub[0])  # null upper bound
    assert jnp.allclose(lo_ub[1], -tau)
    assert jnp.all(jnp.isposinf(hi[n + mineq :]))


def test_active_bounds_on_ftb_faces():
    """Bound faces activate when bound-slacks sit on ``-τ`` (nulls excluded)."""
    problem = make_problem(
        lb=jnp.array([0.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, 3.0]),
    )
    sub = make_scaled_barrier_subproblem(problem=problem, tau=0.8)
    n, mineq = problem.n, problem.mineq
    flat = jnp.zeros((n + mineq + 2 * n,))
    # Put both bound-slack blocks on the FTB face; null masks still suppress.
    flat = flat.at[n + mineq :].set(-0.8)
    primal = InteriorPointPrimal.from_flat(flat, n, mineq)
    active_lb, active_ub = sub.active_bounds(primal, tol=0.0)
    assert jnp.array_equal(active_lb, jnp.array([True, False]))
    assert jnp.array_equal(active_ub, jnp.array([False, True]))


@pytest.mark.parametrize("reg", [0.0, 0.1])
def test_dual_regularization_flag_and_operator(reg: float):
    """``is_kkt_dual_regularized`` mirrors the Lagrangian; operator folds it in."""
    sub = make_scaled_barrier_subproblem(dual_kkt_regularization=reg)
    assert sub.is_kkt_dual_regularized is (reg > 0.0)

    step = make_ip_step(sub.lagrangian.n, sub.lagrangian.mineq, sub.lagrangian.meq)
    # Zero the primal so the dual row of ``kkt_operator`` is pure dual-dual.
    zero_primal = InteriorPointPrimal(
        x=jnp.zeros_like(step[0].x),
        slack=Slack(
            s=jnp.zeros_like(step[0].slack.s),
            s_lb=jnp.zeros_like(step[0].slack.s_lb),
            s_ub=jnp.zeros_like(step[0].slack.s_ub),
        ),
    )
    dual_only = (zero_primal, step[1])
    _, op_d = sub.kkt_operator(dual_only)
    lower = sub.kkt_mvp_lower_offdiag(dual_only)
    dual_dual = sub.kkt_mvp_dual(dual_only)
    if reg > 0.0:
        assert jnp.allclose(op_d.flatten(), (lower.flatten() + dual_dual.flatten()))
        assert jnp.allclose(dual_dual.eq_multipliers, -reg * step[1].eq_multipliers)
    else:
        assert jnp.allclose(op_d.flatten(), lower.flatten())
        assert jnp.allclose(dual_dual.flatten(), 0.0)


def test_primal_grad_is_objective_and_minus_mu():
    """``primal_grad`` is ``∇f`` in ``x`` and ``-μ`` on active slacks."""
    weight = 0.5
    problem = make_problem(
        lb=jnp.array([0.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, 3.0]),
    )
    sub = make_scaled_barrier_subproblem(problem=problem, weight=weight)
    g = sub.primal_grad()
    x = sub.lagrangian.ref.x
    assert jnp.allclose(g.x, 2 * x)  # objective is ||x||^2
    assert jnp.allclose(g.slack.s, -weight)
    assert jnp.allclose(g.slack.s_lb[0], -weight)
    assert jnp.allclose(g.slack.s_lb[1], 0.0)  # null lower bound
    assert jnp.allclose(g.slack.s_ub[0], 0.0)  # null upper bound
    assert jnp.allclose(g.slack.s_ub[1], -weight)


def test_slack_scale_roundtrip_and_null_mask():
    """Ball ↔ original slack scaling round-trips; null bounds stay zero."""
    problem = make_problem(
        lb=jnp.array([0.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, 3.0]),
    )
    sub = make_scaled_barrier_subproblem(problem=problem)
    slack = Slack(
        s=jnp.array([0.4, -0.2]),
        s_lb=jnp.array([0.3, 0.9]),
        s_ub=jnp.array([-0.5, 0.1]),
    )
    ball = sub._slack_to_ball_scale(slack)
    back = sub._slack_to_orig_scale(ball)
    S = sub.lagrangian.slack
    assert jnp.allclose(ball.s, slack.s / S.s)
    assert jnp.allclose(back.s, slack.s)
    assert jnp.allclose(ball.s_lb[1], 0.0)
    assert jnp.allclose(ball.s_ub[0], 0.0)
    assert jnp.allclose(back.s_lb[0], slack.s_lb[0])
    assert jnp.allclose(back.s_lb[1], 0.0)
    assert jnp.allclose(back.s_ub[0], 0.0)
    assert jnp.allclose(back.s_ub[1], slack.s_ub[1])

    step = make_ip_step(problem.n, problem.mineq, problem.meq)
    # Plant original-scale slacks, then round-trip through the step helpers.
    orig_step = (
        InteriorPointPrimal(x=step[0].x, slack=slack),
        step[1],
    )
    ball_step = sub._to_ball_scale(orig_step)
    assert jnp.allclose(ball_step[0].slack.s, ball.s)
    assert jnp.allclose(sub._to_orig_scale(ball_step)[0].slack.s, slack.s)


def test_residual_and_shared_base_helpers():
    """Residual is ``kkt_operator - rhs``; unflatten / step_norm use IP layout."""
    sub = make_scaled_barrier_subproblem()
    lag = sub.lagrangian
    step = make_ip_step(lag.n, lag.mineq, lag.meq, fill=0.05)
    res_p, res_d = sub.residual(step)
    op_p, op_d = sub.kkt_operator(step)
    rhs_p, rhs_d = sub.kkt_rhs()
    assert jnp.allclose(res_p.flatten(), op_p.flatten() - rhs_p.flatten())
    assert jnp.allclose(res_d.flatten(), op_d.flatten() - rhs_d.flatten())

    flat = jnp.linspace(-0.2, 0.3, lag.n + lag.mineq + 2 * lag.n)
    rebuilt = sub.unflatten_primal(flat)
    assert isinstance(rebuilt, InteriorPointPrimal)
    assert jnp.allclose(rebuilt.flatten(), flat)
    assert jnp.allclose(sub.step_norm(step), jnp.linalg.norm(step[0].flatten()))
    assert jnp.allclose(sub.nonbound_constraint_jac(), lag.nonbound_constraint_jac)


def test_dual_grad_matches_lagrangian():
    """Dual residual is forwarded from the IP Lagrangian."""
    sub = make_scaled_barrier_subproblem()
    assert jnp.allclose(sub.dual_grad().flatten(), sub.lagrangian.dual_grad.flatten())
