"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.base` helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import (
    make_active_set_subproblem,
    make_step,
    make_zero_dual,
)


@pytest.mark.parametrize(
    ("meq", "mineq"),
    [(0, 0), (1, 0), (0, 2), (1, 2)],
    ids=["empty", "eq-only", "ineq-only", "both"],
)
def test_kkt_rhs_and_operator_match_blocks(meq: int, mineq: int):
    """``kkt_rhs`` / ``kkt_operator`` assemble from the abstract blocks."""
    problem = make_problem(meq=meq, mineq=mineq)
    sub = make_active_set_subproblem(
        problem=problem,
        active_inequalities=tuple(True for _ in range(mineq)),
        active_lb=tuple(True for _ in range(problem.n)),
        active_ub=tuple(True for _ in range(problem.n)),
    )
    step = make_step(problem.n, meq, mineq)

    rhs_p, rhs_d = sub.kkt_rhs()
    assert jnp.allclose(rhs_p.x, -sub.primal_grad().x)
    assert jnp.allclose(rhs_d.flatten(), -sub.dual_grad().flatten())

    op_p, op_d = sub.kkt_operator(step)
    expected_p = jax_tree_add(sub.kkt_mvp_primal(step), sub.kkt_mvp_upper_offdiag(step))
    assert jnp.allclose(op_p.x, expected_p.x)
    # Active-set Lagrangians are not dual-regularized.
    assert not sub.is_kkt_dual_regularized
    assert jnp.allclose(op_d.flatten(), sub.kkt_mvp_lower_offdiag(step).flatten())


def jax_tree_add(a: Primal, b: Primal) -> Primal:
    return Primal(x=a.x + b.x)


@pytest.mark.parametrize("penalty", [0.0, 1.0, 2.5])
def test_model_value_and_predicted_reduction(penalty: float):
    """Quadratic model and N&W predicted reduction match closed forms."""
    sub = make_active_set_subproblem(
        active_inequalities=(True, True),
        active_lb=(True, True),
        active_ub=(True, True),
    )
    n, meq, mineq = sub.n, sub.meq, sub.mineq
    dx = jnp.array([0.2, -0.1])
    step = (Primal(x=dx), make_zero_dual(n, meq, mineq))

    g = sub.primal_grad().x
    Hp = sub.kkt_mvp_primal(step).x
    expected_model = g @ dx + 0.5 * (dx @ Hp)
    assert jnp.allclose(sub.model_value(step), expected_model)

    m0 = jnp.linalg.norm(sub.kkt_rhs()[1].flatten())
    lin_inf = sub.linearized_infeasibility(step)
    expected_pred = -expected_model + penalty * (m0 - lin_inf)
    assert jnp.allclose(
        sub.predicted_reduction(step, jnp.asarray(penalty)), expected_pred
    )
    assert jnp.allclose(lin_inf, jnp.linalg.norm(sub.residual(step)[1].flatten()))


def test_step_norm_and_to_native_identity():
    """Default geometry: Euclidean primal norm and identity unscaling."""
    sub = make_active_set_subproblem()
    step = make_step(sub.n, sub.meq, sub.mineq, dx=jnp.array([3.0, 4.0]))
    assert jnp.allclose(sub.step_norm(step), 5.0)
    native = sub.to_native_step(step)
    assert jnp.allclose(native[0].x, step[0].x)
    assert jnp.allclose(native[1].flatten(), step[1].flatten())


@pytest.mark.parametrize(
    ("lb", "ub"),
    [
        (jnp.array([0.0, -1.0]), jnp.array([2.0, 3.0])),
        (jnp.array([-jnp.inf, 0.0]), jnp.array([1.0, jnp.inf])),
    ],
    ids=["finite", "partial-null"],
)
def test_primal_box_active_bounds_and_unflatten(lb: jnp.ndarray, ub: jnp.ndarray):
    """Default box is ``lb-x`` / ``ub-x``; active faces and unflatten agree."""
    problem = make_problem(lb=lb, ub=ub)
    sub = make_active_set_subproblem(
        problem=problem,
        active_lb=tuple(True for _ in range(problem.n)),
        active_ub=tuple(True for _ in range(problem.n)),
    )
    x = sub.lagrangian.ref.x
    lo, hi = sub.primal_box()
    inf = jnp.asarray(jnp.inf, dtype=x.dtype)
    assert jnp.allclose(lo, jnp.where(problem.null_lb, -inf, lb - x))
    assert jnp.allclose(hi, jnp.where(problem.null_ub, inf, ub - x))

    # Sit on the lower face where finite, interior elsewhere.
    on_lower = jnp.where(problem.null_lb, 0.0, lo)
    primal = Primal(x=on_lower)
    active_lb, active_ub = sub.active_bounds(primal, tol=0.0)
    assert jnp.array_equal(active_lb, ~problem.null_lb)
    assert jnp.array_equal(active_ub, jnp.zeros((problem.n,), dtype=bool))

    flat = jnp.linspace(-0.3, 0.4, problem.n)
    rebuilt = sub.unflatten_primal(flat)
    assert isinstance(rebuilt, Primal)
    assert jnp.allclose(rebuilt.x, flat)


def test_unflatten_uses_reference_sizes(quadratic_problem: Problem):
    """``unflatten_primal`` reads sizes from ``lagrangian.ref``, not the grad."""
    sub = make_active_set_subproblem(problem=quadratic_problem)
    flat = jnp.array([0.5, -0.25])
    assert sub.unflatten_primal(flat).sizes == sub.lagrangian.ref.sizes
