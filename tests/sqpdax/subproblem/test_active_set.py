"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.active_set`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.subproblem.active_set import ActiveSetSubProblem
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import (
    make_active_set,
    make_active_set_subproblem,
    make_evaluated_lagrangian,
    make_step,
)


@pytest.mark.parametrize(
    ("meq", "mineq", "active_inequalities", "active_lb", "active_ub"),
    [
        (0, 0, (), (False, False), (False, False)),
        (1, 0, (), (True, False), (False, True)),
        (0, 2, (True, False), (True, True), (False, False)),
        (1, 2, (False, True), (True, False), (True, False)),
    ],
    ids=["empty", "eq-only", "ineq-only", "mixed"],
)
def test_sizes_and_working_set_count(
    meq: int,
    mineq: int,
    active_inequalities: tuple[bool, ...],
    active_lb: tuple[bool, ...],
    active_ub: tuple[bool, ...],
):
    """Sizes mirror the Lagrangian; ``m`` counts equalities plus active rows."""
    problem = make_problem(meq=meq, mineq=mineq)
    sub = make_active_set_subproblem(
        problem=problem,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    assert sub.n == problem.n
    assert sub.meq == meq
    assert sub.mineq == mineq
    assert jnp.allclose(sub.x_k.x, sub.lagrangian.ref.x)
    assert jnp.allclose(sub.d_k.flatten(), sub.lagrangian.dual.flatten())

    expected_m = meq + sum(active_inequalities) + sum(active_lb) + sum(active_ub)
    assert int(sub.m) == expected_m


def test_masked_lagrangian_and_delegation():
    """``L_k`` is the masked Lagrangian; KKT methods / jac match ``L_k``."""
    lag = make_evaluated_lagrangian()
    active = make_active_set(
        lag.meq,
        lag.mineq,
        lag.n,
        active_inequalities=(True, False),
        active_lb=(True, False),
        active_ub=(False, True),
    )
    sub = ActiveSetSubProblem(lag, active)
    expected_L_k = active.mask_lagrangian(lag)

    assert jnp.allclose(sub.L_k.ineq_fn_val, expected_L_k.ineq_fn_val)
    assert jnp.allclose(sub.L_k.ineq_multipliers, expected_L_k.ineq_multipliers)
    assert jnp.array_equal(sub.L_k.null_lb, expected_L_k.null_lb)
    assert jnp.array_equal(sub.L_k.null_ub, expected_L_k.null_ub)

    assert jnp.allclose(sub.primal_grad().x, expected_L_k.primal_grad.x)
    assert jnp.allclose(sub.dual_grad().flatten(), expected_L_k.dual_grad.flatten())
    assert jnp.allclose(
        sub.nonbound_constraint_jac(), expected_L_k.nonbound_constraint_jac
    )

    step = make_step(lag.n, lag.meq, lag.mineq)
    assert jnp.allclose(sub.kkt_mvp_primal(step).x, expected_L_k.kkt_mvp_primal(step).x)
    assert jnp.allclose(
        sub.kkt_mvp_upper_offdiag(step).x, expected_L_k.kkt_mvp_upper_offdiag(step).x
    )
    assert jnp.allclose(
        sub.kkt_mvp_lower_offdiag(step).flatten(),
        expected_L_k.kkt_mvp_lower_offdiag(step).flatten(),
    )
    assert jnp.allclose(
        sub.kkt_mvp_dual(step).flatten(), expected_L_k.kkt_mvp_dual(step).flatten()
    )
    mvp_p, mvp_d = sub.kkt_mvp(step)
    exp_p, exp_d = expected_L_k.kkt_mvp(step)
    assert jnp.allclose(mvp_p.x, exp_p.x)
    assert jnp.allclose(mvp_d.flatten(), exp_d.flatten())


def test_residual_matches_kkt_mvp_minus_rhs():
    """``residual`` is ``kkt_mvp(step) - kkt_rhs()``."""
    sub = make_active_set_subproblem(active_inequalities=(True, False))
    step = make_step(sub.n, sub.meq, sub.mineq)
    res_p, res_d = sub.residual(step)
    mvp_p, mvp_d = sub.kkt_mvp(step)
    rhs_p, rhs_d = sub.kkt_rhs()
    assert jnp.allclose(res_p.x, mvp_p.x - rhs_p.x)
    assert jnp.allclose(res_d.flatten(), mvp_d.flatten() - rhs_d.flatten())


def test_inactive_rows_drop_from_dual_grad_and_jac():
    """Inactive inequalities are zero in the masked dual grad / Jacobian."""
    lag = make_evaluated_lagrangian()
    active = ActiveSet(
        meq=lag.meq,
        active_inequalities=jnp.array([False, True]),
        active_lb=jnp.array([False, False]),
        active_ub=jnp.array([False, False]),
    )
    sub = ActiveSetSubProblem(lag, active)

    assert jnp.allclose(sub.dual_grad().ineq_multipliers[0], 0.0)
    assert jnp.allclose(sub.dual_grad().ineq_multipliers[1], lag.ineq_fn_val[1])
    assert jnp.allclose(sub.nonbound_constraint_jac()[lag.meq], 0.0)
    assert jnp.allclose(
        sub.nonbound_constraint_jac()[lag.meq + 1], lag.ineq_fn_jac_val[1]
    )
