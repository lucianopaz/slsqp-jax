"""Unit tests for interior-point Lagrangian variants."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.interior_point import InteriorPointLagrangian
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem

from .conftest import make_dual, make_ip_primal, make_problem


@pytest.mark.parametrize("primal_dual", [False, True])
def test_interior_point_lagrangian_value_includes_barrier(primal_dual: bool):
    """IP Lagrangian value is NLP Lagrangian plus barrier."""
    problem = make_problem()
    barrier = LogBarrier(
        weight=jnp.asarray(0.5),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    lag = InteriorPointLagrangian(
        problem, secant=None, barrier=barrier, primal_dual=primal_dual
    )
    x = make_ip_primal(mineq=problem.mineq)
    dual = make_dual(problem.n, problem.meq, problem.mineq)
    evaluated = lag(x, dual)

    barrier_val = barrier(x.slack).fn_val
    # Manual NLP Lagrangian with slack-augmented inequalities / bounds.
    nlp_val = (
        problem.fn(x.x)
        + dual.eq_multipliers @ problem.eq_fn(x.x)
        + dual.ineq_multipliers @ (problem.ineq_fn(x.x) + x.slack.s)
        + dual.lb_multipliers
        @ jnp.where(problem.null_lb, 0.0, problem.lb - x.x + x.slack.s_lb)
        + dual.ub_multipliers
        @ jnp.where(problem.null_ub, 0.0, x.x - problem.ub + x.slack.s_ub)
    )
    assert jnp.allclose(evaluated.value, nlp_val + barrier_val)
    assert jnp.allclose(lag.value(x, dual), nlp_val + barrier_val)
    assert evaluated.primal_dual is primal_dual


@pytest.mark.parametrize("primal_dual", [False, True])
def test_interior_point_kkt_mvp_slack_block(primal_dual: bool):
    """Slack-slack KKT block matches barrier HVP or ``Λ/S`` product."""
    problem = make_problem()
    barrier = LogBarrier(
        weight=jnp.asarray(1.0),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    lag = InteriorPointLagrangian(
        problem,
        secant=None,
        barrier=barrier,
        dual_kkt_regularization=0.1,
        primal_dual=primal_dual,
    )
    x = make_ip_primal(mineq=problem.mineq)
    dual = make_dual(problem.n, problem.meq, problem.mineq)
    evaluated = lag(x, dual)
    assert evaluated.is_kkt_dual_regularized

    ds = Slack(
        s=jnp.ones((problem.mineq,)),
        s_lb=0.5 * jnp.ones((problem.n,)),
        s_ub=-0.25 * jnp.ones((problem.n,)),
    )
    tangent = (
        InteriorPointPrimal(x=jnp.zeros((problem.n,)), slack=ds),
        Dual(
            eq_multipliers=jnp.zeros((problem.meq,)),
            ineq_multipliers=jnp.zeros((problem.mineq,)),
            lb_multipliers=jnp.zeros((problem.n,)),
            ub_multipliers=jnp.zeros((problem.n,)),
        ),
    )
    got = evaluated.kkt_mvp_slack_slack(tangent)
    if primal_dual:
        expected = Slack(
            s=dual.ineq_multipliers / x.slack.s * ds.s,
            s_lb=jnp.where(
                problem.null_lb, 0.0, dual.lb_multipliers / x.slack.s_lb * ds.s_lb
            ),
            s_ub=jnp.where(
                problem.null_ub, 0.0, dual.ub_multipliers / x.slack.s_ub * ds.s_ub
            ),
        )
    else:
        expected = barrier.hvp(x.slack, ds)
    assert jnp.allclose(got.s, expected.s)
    assert jnp.allclose(got.s_lb, expected.s_lb)
    assert jnp.allclose(got.s_ub, expected.s_ub)

    # Dual regularization appears on the equality block of kkt_mvp.
    dlam_eq = jnp.ones((problem.meq,))
    tangent_reg = (
        InteriorPointPrimal(
            x=jnp.zeros((problem.n,)),
            slack=Slack(
                s=jnp.zeros((problem.mineq,)),
                s_lb=jnp.zeros((problem.n,)),
                s_ub=jnp.zeros((problem.n,)),
            ),
        ),
        Dual(
            eq_multipliers=dlam_eq,
            ineq_multipliers=jnp.zeros((problem.mineq,)),
            lb_multipliers=jnp.zeros((problem.n,)),
            ub_multipliers=jnp.zeros((problem.n,)),
        ),
    )
    _, dual_row = evaluated.kkt_mvp(tangent_reg)
    assert jnp.allclose(dual_row.eq_multipliers, -0.1 * dlam_eq)


def test_interior_point_rejects_negative_regularization(quadratic_problem: Problem):
    """Negative dual KKT regularization is rejected."""
    barrier = LogBarrier(
        weight=jnp.asarray(1.0),
        null_lb=quadratic_problem.null_lb,
        null_ub=quadratic_problem.null_ub,
    )
    with pytest.raises(ValueError, match="non-negative"):
        InteriorPointLagrangian(
            quadratic_problem,
            secant=None,
            barrier=barrier,
            dual_kkt_regularization=-1.0,
        )


def test_interior_point_primal_dual_requires_log_barrier(quadratic_problem: Problem):
    """``primal_dual=True`` requires a log barrier at evaluation time."""
    from slsqp_jax.sqpdax.barrier import Barrier

    class DummyBarrier(Barrier):
        def fn(self, slack, *args, **kwargs):
            return jnp.asarray(0.0)

    barrier = DummyBarrier(
        weight=jnp.asarray(1.0),
        null_lb=quadratic_problem.null_lb,
        null_ub=quadratic_problem.null_ub,
    )
    lag = InteriorPointLagrangian(
        quadratic_problem, secant=None, barrier=barrier, primal_dual=True
    )
    x = make_ip_primal(mineq=quadratic_problem.mineq)
    dual = make_dual(
        quadratic_problem.n, quadratic_problem.meq, quadratic_problem.mineq
    )
    with pytest.raises(TypeError, match="LogBarrier"):
        lag(x, dual)


def test_interior_point_evaluated_requires_ip_primal(quadratic_problem: Problem):
    """Evaluated IP Lagrangian rejects a plain :class:`Primal` reference."""
    from slsqp_jax.sqpdax.lagrangian.evaluated import InteriorPointEvaluatedLagrangian

    barrier = LogBarrier(
        weight=jnp.asarray(1.0),
        null_lb=quadratic_problem.null_lb,
        null_ub=quadratic_problem.null_ub,
    )
    plain = Primal(x=jnp.array([0.25, 0.75]))
    dual = make_dual(
        quadratic_problem.n, quadratic_problem.meq, quadratic_problem.mineq
    )
    with pytest.raises(TypeError, match="InteriorPointPrimal"):
        InteriorPointEvaluatedLagrangian(
            evaluated=quadratic_problem(plain),
            secant=None,
            dual=dual,
            barrier=barrier(
                Slack(
                    s=jnp.ones((quadratic_problem.mineq,)),
                    s_lb=jnp.ones((quadratic_problem.n,)),
                    s_ub=jnp.ones((quadratic_problem.n,)),
                )
            ),
        )


def test_interior_point_facades_and_grads(quadratic_problem: Problem):
    """IP unevaluated facades and evaluated primal/dual grads are consistent."""
    barrier = LogBarrier(
        weight=jnp.asarray(0.5),
        null_lb=quadratic_problem.null_lb,
        null_ub=quadratic_problem.null_ub,
    )
    lag = InteriorPointLagrangian(
        quadratic_problem, secant=None, barrier=barrier, primal_dual=False
    )
    x = make_ip_primal(mineq=quadratic_problem.mineq)
    dual = make_dual(
        quadratic_problem.n, quadratic_problem.meq, quadratic_problem.mineq
    )
    evaluated = lag(x, dual)

    assert jnp.allclose(lag.objective_fn(x), quadratic_problem.fn(x.x))
    assert jnp.allclose(lag.objective_grad(x), quadratic_problem.grad(x.x))
    assert jnp.allclose(lag.x_grad(x, dual), evaluated.x_grad)
    assert jnp.allclose(evaluated.x_grad, lag.primal_grad(x, dual).x)

    pg, dg = lag.grad(x, dual)
    assert jnp.allclose(pg.x, evaluated.primal_grad.x)
    assert jnp.allclose(pg.slack.s, evaluated.slack_grad.s)
    assert jnp.allclose(dg.eq_multipliers, evaluated.dual_grad.eq_multipliers)
    assert jnp.allclose(lag.dual_grad(x, dual).ineq_multipliers, evaluated.ineq_fn_val)

    # evaluated.grad property (tuple).
    g_p, g_d = evaluated.grad
    assert jnp.allclose(g_p.slack.s_lb, evaluated.slack_grad.s_lb)
    assert jnp.allclose(g_d.lb_multipliers, evaluated.dual_grad.lb_multipliers)

    dx = InteriorPointPrimal(
        x=jnp.array([0.1, -0.05]),
        slack=Slack(
            s=jnp.ones((quadratic_problem.mineq,)),
            s_lb=0.2 * jnp.ones((quadratic_problem.n,)),
            s_ub=-0.1 * jnp.ones((quadratic_problem.n,)),
        ),
    )
    dlam = Dual(
        eq_multipliers=jnp.ones((quadratic_problem.meq,)),
        ineq_multipliers=jnp.full((quadratic_problem.mineq,), 0.3),
        lb_multipliers=jnp.linspace(0.01, 0.02, quadratic_problem.n),
        ub_multipliers=jnp.linspace(0.02, 0.03, quadratic_problem.n),
    )
    tangent = (dx, dlam)
    assert jnp.allclose(
        lag.kkt_mvp_primal(x, dual, tangent).x, evaluated.kkt_mvp_primal(tangent).x
    )
    assert jnp.allclose(
        lag.kkt_mvp_upper_offdiag(x, dual, tangent).slack.s,
        evaluated.kkt_mvp_upper_offdiag(tangent).slack.s,
    )
    assert jnp.allclose(
        lag.kkt_mvp_lower_offdiag(x, dual, tangent).eq_multipliers,
        evaluated.kkt_mvp_lower_offdiag(tangent).eq_multipliers,
    )
    mvp_p, mvp_d = lag.kkt_mvp(x, dual, tangent)
    assert jnp.allclose(mvp_p.x, evaluated.kkt_mvp(tangent)[0].x)
    assert jnp.allclose(
        mvp_d.eq_multipliers, evaluated.kkt_mvp(tangent)[1].eq_multipliers
    )

    x1 = InteriorPointPrimal(
        x=x.x + jnp.array([0.05, -0.02]),
        slack=x.slack,
    )
    y = lag.curvature_estimate(x1, evaluated)
    assert jnp.allclose(y, 2.0 * (x1.x - x.x))


def test_interior_point_requires_secant_without_curvature():
    """IP Lagrangian without exact HVPs demands a secant."""
    problem = make_problem(with_curvature=False)
    barrier = LogBarrier(
        weight=jnp.asarray(1.0),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    with pytest.raises(TypeError, match="secant is required"):
        InteriorPointLagrangian(problem, secant=None, barrier=barrier)
