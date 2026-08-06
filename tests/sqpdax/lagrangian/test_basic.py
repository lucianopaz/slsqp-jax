"""Unit tests for :mod:`slsqp_jax.sqpdax.lagrangian.basic` and evaluated forms."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.basic import Lagrangian
from slsqp_jax.sqpdax.lagrangian.evaluated import EvaluatedLagrangian
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem

from .conftest import make_dual, make_primal, make_problem


@pytest.mark.parametrize(
    ("meq", "mineq"),
    [(0, 0), (1, 0), (0, 2), (1, 2)],
    ids=["empty", "eq-only", "ineq-only", "both"],
)
def test_lagrangian_value_and_grad_match_formula(meq: int, mineq: int):
    """``L`` and ``∇_x L`` match the explicit multiplier formula."""
    problem = make_problem(meq=meq, mineq=mineq)
    lag = Lagrangian(problem)
    x = make_primal()
    dual = make_dual(problem.n, meq, mineq)
    evaluated = lag(x, dual)

    assert isinstance(evaluated, EvaluatedLagrangian)
    assert evaluated.has_exact_curvature
    assert evaluated.n == problem.n
    assert evaluated.meq == meq
    assert evaluated.mineq == mineq

    expected_val = (
        problem.fn(x.x)
        + dual.eq_multipliers @ problem.eq_fn(x.x)
        + dual.ineq_multipliers @ problem.ineq_fn(x.x)
        + dual.lb_multipliers @ jnp.where(problem.null_lb, 0.0, problem.lb - x.x)
        + dual.ub_multipliers @ jnp.where(problem.null_ub, 0.0, x.x - problem.ub)
    )
    assert jnp.allclose(evaluated.value, expected_val)
    assert jnp.allclose(lag.value(x, dual), expected_val)

    expected_grad = (
        problem.grad(x.x)
        + dual.eq_multipliers @ problem.eq_fn_jac(x.x)
        + dual.ineq_multipliers @ problem.ineq_fn_jac(x.x)
        + jnp.where(problem.null_lb, 0.0, -dual.lb_multipliers)
        + jnp.where(problem.null_ub, 0.0, dual.ub_multipliers)
    )
    assert jnp.allclose(evaluated.x_grad, expected_grad)
    assert jnp.allclose(lag.x_grad(x, dual), expected_grad)
    assert jnp.allclose(evaluated.primal_grad.x, expected_grad)


@pytest.mark.parametrize("meq,mineq", [(1, 2), (0, 0)])
def test_lagrangian_kkt_mvp_and_hvp(meq: int, mineq: int):
    """HVP and KKT MVP assemble consistently for the quadratic model."""
    problem = make_problem(meq=meq, mineq=mineq)
    lag = Lagrangian(problem)
    x = make_primal()
    dual = make_dual(problem.n, meq, mineq)
    evaluated = lag(x, dual)

    dx = Primal(x=jnp.array([0.1, -0.2]))
    dlam = Dual(
        eq_multipliers=jnp.ones((meq,)),
        ineq_multipliers=jnp.full((mineq,), 0.5),
        lb_multipliers=jnp.array([0.05, -0.05]),
        ub_multipliers=jnp.array([-0.02, 0.03]),
    )
    tangent = (dx, dlam)

    # Objective is ||x||^2 with linear constraints → L_xx = 2 I.
    assert jnp.allclose(evaluated.hvp(dx.x), 2 * dx.x)
    assert jnp.allclose(evaluated.primal_hvp(dx).x, 2 * dx.x)

    primal_row, dual_row = evaluated.kkt_mvp(tangent)
    expected_primal = (
        evaluated.kkt_mvp_primal(tangent).x + evaluated.kkt_mvp_upper_offdiag(tangent).x
    )
    assert jnp.allclose(primal_row.x, expected_primal)
    assert jnp.allclose(
        dual_row.eq_multipliers,
        evaluated.kkt_mvp_lower_offdiag(tangent).eq_multipliers,
    )

    # Facades on the unevaluated Lagrangian agree.
    assert jnp.allclose(lag.kkt_mvp(x, dual, tangent)[0].x, primal_row.x)


def test_lagrangian_requires_secant_without_curvature():
    """Construction without exact HVPs demands a secant."""
    problem = make_problem(with_curvature=False)
    with pytest.raises(TypeError, match="secant is required"):
        Lagrangian(problem)


def test_lagrangian_curvature_estimate_shared_multipliers(quadratic_problem: Problem):
    """Curvature estimate uses a shared ``λ`` (bound terms cancel)."""
    lag = Lagrangian(quadratic_problem)
    dual = make_dual(
        quadratic_problem.n, quadratic_problem.meq, quadratic_problem.mineq
    )
    x0 = make_primal()
    x1 = Primal(x=x0.x + jnp.array([0.1, -0.05]))
    prev = lag(x0, dual)
    y = lag.curvature_estimate(x1, prev)

    # For this quadratic + linear-constraint problem, ∇_x L(x, λ) = 2x + J^T λ + …
    # with λ shared, the difference is exactly 2 (x1 - x0).
    assert jnp.allclose(y, 2 * (x1.x - x0.x))


def test_evaluated_hvp_without_curvature_or_secant_raises(
    quadratic_problem: Problem,
):
    """Exact HVP path errors clearly when QVPs are missing."""
    problem = make_problem(with_curvature=False)
    # Bypass Lagrangian.__init__ by building EvaluatedLagrangian directly.
    x = make_primal()
    dual = make_dual(problem.n, problem.meq, problem.mineq)
    evaluated = EvaluatedLagrangian(
        evaluated=problem(x),
        secant=None,
        dual=dual,
    )
    with pytest.raises(TypeError, match="exact curvature"):
        evaluated.hvp(jnp.ones(problem.n))
