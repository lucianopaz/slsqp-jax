"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.proximal`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.subproblem import ActiveSetSubProblem, ProximalActiveSetSubProblem
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import make_active_set, make_evaluated_lagrangian, make_step


def _make_proximal(
    *,
    meq: int,
    mineq: int = 2,
    active_inequalities=(True, False),
    active_lb=(True, False),
    active_ub=(False, True),
    mu: float = 0.25,
):
    problem = make_problem(meq=meq, mineq=mineq)
    lag = make_evaluated_lagrangian(problem=problem)
    active = make_active_set(
        meq,
        mineq,
        problem.n,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    eq_center = jnp.linspace(0.3, -0.7, meq)
    prox = ProximalActiveSetSubProblem(lag, active, mu, eq_center)
    plain = ActiveSetSubProblem(lag, active)
    return lag, active, plain, prox


@pytest.mark.parametrize("meq", [0, 1], ids=["no-eq", "one-eq"])
@pytest.mark.parametrize(
    ("active_inequalities", "active_lb", "active_ub"),
    [
        ((False, False), (False, False), (False, False)),
        ((True, False), (True, False), (False, True)),
    ],
    ids=["empty-ws", "mixed-ws"],
)
def test_stabilised_operator_and_gradient(
    meq: int, active_inequalities, active_lb, active_ub
):
    """HVP / gradient are the masked ones plus the proximal terms; eq rows vanish."""
    lag, _active, plain, prox = _make_proximal(
        meq=meq,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    A = lag.eq_fn_jac_val
    c = lag.eq_fn_val
    mu = prox.mu
    step = make_step(lag.n, meq, lag.mineq)
    v = step[0].x

    # Equality rows are zeroed in the masked Lagrangian and everything it feeds.
    assert jnp.allclose(prox.L_k.eq_fn_val, 0.0)
    assert jnp.allclose(prox.L_k.eq_fn_jac_val, 0.0)
    assert jnp.allclose(prox.L_k.eq_multipliers, 0.0)
    assert jnp.allclose(prox.dual_grad().eq_multipliers, 0.0)
    assert jnp.allclose(prox.nonbound_constraint_jac()[:meq], 0.0)
    assert jnp.allclose(
        prox.nonbound_constraint_jac()[meq:], plain.nonbound_constraint_jac()[meq:]
    )
    assert jnp.allclose(
        prox.dual_grad().ineq_multipliers, plain.dual_grad().ineq_multipliers
    )

    # Stabilised HVP: masked HVP + (1/μ) Aᵀ A v.
    expected_hvp = prox.L_k.kkt_mvp_primal(step).x + (A.T @ (A @ v)) / mu
    assert jnp.allclose(prox.kkt_mvp_primal(step).x, expected_hvp)
    assert jnp.allclose(prox.stabilisation_hvp(v), (A.T @ (A @ v)) / mu)

    # Stabilised gradient: masked gradient + Aᵀ (λ_k + c/μ).
    expected_grad = prox.L_k.primal_grad.x + A.T @ (prox.eq_center + c / mu)
    assert jnp.allclose(prox.primal_grad().x, expected_grad)

    # Full product / residual routed through the stabilised blocks.
    mvp_p, mvp_d = prox.kkt_mvp(step)
    assert jnp.allclose(mvp_p.x, expected_hvp + prox.kkt_mvp_upper_offdiag(step).x)
    assert jnp.allclose(mvp_d.flatten(), prox.kkt_mvp_lower_offdiag(step).flatten())
    res_p, res_d = prox.residual(step)
    rhs_p, rhs_d = prox.kkt_rhs()
    assert jnp.allclose(res_p.x, mvp_p.x - rhs_p.x)
    assert jnp.allclose(res_d.flatten(), mvp_d.flatten() - rhs_d.flatten())

    # Multiplier recovery formula.
    assert jnp.allclose(
        prox.recover_eq_multipliers(v), prox.eq_center + (A @ v + c) / mu
    )
    if meq == 0:
        assert prox.recover_eq_multipliers(v).shape == (0,)
        assert jnp.allclose(prox.kkt_mvp_primal(step).x, plain.kkt_mvp_primal(step).x)
        assert jnp.allclose(prox.primal_grad().x, plain.primal_grad().x)


def test_with_active_set_preserves_proximal_data():
    """Refreshing the working set keeps ``mu`` / ``eq_center`` and the class."""
    lag, _active, _plain, prox = _make_proximal(meq=1)
    new_active = make_active_set(
        lag.meq,
        lag.mineq,
        lag.n,
        active_inequalities=(False, True),
        active_lb=(False, False),
        active_ub=(True, False),
    )
    refreshed = prox.with_active_set(new_active)
    assert isinstance(refreshed, ProximalActiveSetSubProblem)
    assert jnp.allclose(refreshed.mu, prox.mu)
    assert jnp.allclose(refreshed.eq_center, prox.eq_center)
    assert jnp.array_equal(
        refreshed.active_set.active_inequalities, new_active.active_inequalities
    )
    assert jnp.allclose(refreshed.L_k.eq_fn_jac_val, 0.0)
    # Plain subproblems rebuild as plain subproblems.
    plain = ActiveSetSubProblem(lag, new_active).with_active_set(prox.active_set)
    assert type(plain) is ActiveSetSubProblem


def test_exports_roundtrip():
    """Public re-exports resolve from the subproblem and sqpdax packages."""
    from slsqp_jax import sqpdax
    from slsqp_jax.sqpdax import subproblem

    assert subproblem.ProximalActiveSetSubProblem is ProximalActiveSetSubProblem
    assert sqpdax.ProximalActiveSetSubProblem is ProximalActiveSetSubProblem
