"""Least-norm geometry of the subproblems (used by the second-order correction)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.subproblem.base import SubProblem
from slsqp_jax.sqpdax.subproblem.proximal import ProximalActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.scaled_barrier import ScaledBarrierSubProblem

from .conftest import (
    make_active_set,
    make_active_set_subproblem,
    make_evaluated_lagrangian,
    make_funnel_barrier_subproblem,
    make_ip_evaluated,
)

# Working set: the equality, the first inequality and the lower bound of x0.
WORKING_SET = {
    "active_inequalities": (True, False),
    "active_lb": (True, False),
    "active_ub": (False, False),
}


def _proximal() -> ProximalActiveSetSubProblem:
    lag = make_evaluated_lagrangian()
    active = make_active_set(lag.meq, lag.mineq, lag.n, **WORKING_SET)
    return ProximalActiveSetSubProblem(
        lag, active, jnp.asarray(0.5), jnp.zeros(lag.meq)
    )


SUBPROBLEMS: dict[str, Callable[[], SubProblem]] = {
    "active-set": lambda: make_active_set_subproblem(**WORKING_SET),
    "proximal": _proximal,
    "scaled-barrier": lambda: ScaledBarrierSubProblem(make_ip_evaluated()),
    "funnel-barrier": make_funnel_barrier_subproblem,
}
WORKING_SET_SUBPROBLEMS = ["active-set", "proximal"]


@pytest.fixture(params=list(SUBPROBLEMS))
def subproblem(request) -> SubProblem:
    return SUBPROBLEMS[request.param]()


def materialise(sub: SubProblem) -> np.ndarray:
    """Dense ``A`` of the dual-primal block on flattened model coordinates."""
    zero_dual = jax.tree.map(jnp.zeros_like, sub.dual_grad())

    def apply(v):
        return sub.kkt_mvp_lower_offdiag((sub.unflatten_primal(v), zero_dual)).flatten()

    return np.asarray(jax.jacfwd(apply)(jnp.zeros_like(sub.lagrangian.ref.flatten())))


def as_dual(sub: SubProblem, flat: np.ndarray) -> Dual:
    dtype = sub.lagrangian.ref.flatten().dtype
    return Dual.from_flat(jnp.asarray(flat, dtype), *sub.dual_grad().sizes)


def test_least_norm_step_is_the_pseudo_inverse_solution(subproblem: SubProblem):
    a = materialise(subproblem)
    rhs = a @ np.random.default_rng(0).normal(size=a.shape[1])
    w = np.asarray(subproblem.least_norm_step(as_dual(subproblem, rhs)).flatten())
    np.testing.assert_allclose(w, np.linalg.pinv(a) @ rhs, atol=1e-5)
    np.testing.assert_allclose(a @ w, rhs, atol=1e-5)


def test_constraint_residual_at_the_reference_is_dual_grad(subproblem: SubProblem):
    np.testing.assert_allclose(
        subproblem.constraint_residual(subproblem.lagrangian).flatten(),
        subproblem.dual_grad().flatten(),
    )


@pytest.mark.parametrize("name", WORKING_SET_SUBPROBLEMS)
def test_least_norm_step_keeps_active_bounds_fixed(name: str):
    sub = SUBPROBLEMS[name]()
    a = materialise(sub)
    # A target that leaves the bound rows unchanged: the step must not move x0.
    v = np.array([0.0, 1.3])
    w = np.asarray(sub.least_norm_step(as_dual(sub, a @ v)).flatten())
    assert w[0] == pytest.approx(0.0, abs=1e-7)


def test_schur_factor_matches_the_matrix_free_solve():
    sub = ScaledBarrierSubProblem(make_ip_evaluated())
    a = materialise(sub)
    rhs = as_dual(sub, a @ np.random.default_rng(1).normal(size=a.shape[1]))
    np.testing.assert_allclose(
        sub.with_schur_normal_equations().least_norm_step(rhs).flatten(),
        sub.least_norm_step(rhs).flatten(),
        atol=1e-5,
    )
