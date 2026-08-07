"""Unit tests for :mod:`slsqp_jax.sqpdax.active_set`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.active_set import ActiveSet
from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian.basic import Lagrangian
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem


def _obj(x: Array) -> Array:
    return jnp.sum(x**2)


def _obj_grad(x: Array) -> Array:
    return 2 * x


def _obj_hvp(x: Array, v: Array) -> Array:
    return 2 * v


def _eq(x: Array) -> Array:
    return jnp.array([x[0] + x[1] - 1.0])


def _eq_jac(x: Array) -> Array:
    return jnp.array([[1.0, 1.0]])


def _eq_hvp(x: Array, v: Array) -> Array:
    return jnp.zeros((1, x.shape[-1]), dtype=x.dtype)


def _ineq(x: Array) -> Array:
    return jnp.array([x[0] - 2.0, -x[1]])


def _ineq_jac(x: Array) -> Array:
    return jnp.array([[1.0, 0.0], [0.0, -1.0]])


def _ineq_hvp(x: Array, v: Array) -> Array:
    return jnp.zeros((2, x.shape[-1]), dtype=x.dtype)


def _empty(x: Array) -> Array:
    return jnp.zeros((0,), dtype=x.dtype)


def _empty_jac(x: Array) -> Array:
    return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)


def _empty_hvp(x: Array, v: Array) -> Array:
    return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)


def _make_problem(
    *,
    n: int = 2,
    meq: int = 1,
    mineq: int = 2,
    lb: Array | None = None,
    ub: Array | None = None,
) -> Problem:
    if lb is None:
        lb = jnp.array([0.0, -1.0])
    if ub is None:
        ub = jnp.array([2.0, 3.0])
    use_eq = meq > 0
    use_ineq = mineq > 0
    return Problem(
        fn=_obj,
        grad=_obj_grad,
        hvp=_obj_hvp,
        eq_fn=_eq if use_eq else _empty,
        ineq_fn=_ineq if use_ineq else _empty,
        eq_fn_jac=_eq_jac if use_eq else _empty_jac,
        ineq_fn_jac=_ineq_jac if use_ineq else _empty_jac,
        eq_fn_hvp=_eq_hvp if use_eq else _empty_hvp,
        ineq_fn_hvp=_ineq_hvp if use_ineq else _empty_hvp,
        lb=lb,
        ub=ub,
        null_lb=jnp.isinf(lb) & (lb < 0),
        null_ub=jnp.isinf(ub) & (ub > 0),
        n=n,
        meq=meq,
        mineq=mineq,
    )


def _make_dual(n: int, meq: int, mineq: int) -> Dual:
    return Dual(
        eq_multipliers=jnp.arange(meq, dtype=jnp.float32) + 1.0,
        ineq_multipliers=jnp.arange(mineq, dtype=jnp.float32) + 10.0,
        lb_multipliers=jnp.arange(n, dtype=jnp.float32) + 20.0,
        ub_multipliers=jnp.arange(n, dtype=jnp.float32) + 30.0,
    )


def _make_active_set(
    meq: int,
    mineq: int,
    n: int,
    *,
    active_inequalities: tuple[bool, ...] | None = None,
    active_lb: tuple[bool, ...] | None = None,
    active_ub: tuple[bool, ...] | None = None,
) -> ActiveSet:
    if active_inequalities is None:
        active_inequalities = tuple(i % 2 == 0 for i in range(mineq))
    if active_lb is None:
        active_lb = tuple(i % 2 == 0 for i in range(n))
    if active_ub is None:
        active_ub = tuple(i % 2 == 1 for i in range(n))
    return ActiveSet(
        meq=meq,
        active_inequalities=jnp.asarray(active_inequalities),
        active_lb=jnp.asarray(active_lb),
        active_ub=jnp.asarray(active_ub),
    )


@pytest.mark.parametrize(
    ("meq", "mineq", "active_inequalities"),
    [
        (0, 0, ()),
        (2, 0, ()),
        (0, 3, (True, False, True)),
        (1, 2, (False, True)),
        (3, 2, (True, True)),
    ],
    ids=["empty", "eq-only", "ineq-only", "mixed", "all-ineq-active"],
)
def test_active_gen(
    meq: int,
    mineq: int,
    active_inequalities: tuple[bool, ...],
):
    """``active_gen`` is all-True equalities followed by inequality masks."""
    n = 2
    active = _make_active_set(
        meq,
        mineq,
        n,
        active_inequalities=active_inequalities,
        active_lb=(False,) * n,
        active_ub=(False,) * n,
    )
    expected = jnp.concatenate(
        [jnp.ones((meq,), dtype=bool), jnp.asarray(active_inequalities, dtype=bool)]
    )
    assert active.active_gen.shape == (meq + mineq,)
    assert jnp.array_equal(active.active_gen, expected)


@pytest.mark.parametrize(
    ("meq", "mineq", "n", "active_inequalities", "active_lb", "active_ub"),
    [
        (0, 0, 2, (), (True, False), (False, True)),
        (1, 0, 2, (), (False, False), (True, True)),
        (0, 2, 2, (True, False), (True, False), (False, True)),
        (2, 3, 3, (True, False, True), (True, False, True), (False, True, False)),
    ],
    ids=["no-constraints", "eq-only", "ineq-only", "full"],
)
def test_mask_dual_zeroes_inactive_rows(
    meq: int,
    mineq: int,
    n: int,
    active_inequalities: tuple[bool, ...],
    active_lb: tuple[bool, ...],
    active_ub: tuple[bool, ...],
):
    """Inactive inequality / bound multipliers become zero; equalities stay."""
    active = _make_active_set(
        meq,
        mineq,
        n,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    dual = _make_dual(n, meq, mineq)
    masked = active.mask_dual(dual)

    assert jnp.allclose(masked.eq_multipliers, dual.eq_multipliers)
    assert jnp.allclose(
        masked.ineq_multipliers,
        jnp.where(jnp.asarray(active_inequalities), dual.ineq_multipliers, 0.0),
    )
    assert jnp.allclose(
        masked.lb_multipliers,
        jnp.where(jnp.asarray(active_lb), dual.lb_multipliers, 0.0),
    )
    assert jnp.allclose(
        masked.ub_multipliers,
        jnp.where(jnp.asarray(active_ub), dual.ub_multipliers, 0.0),
    )


@pytest.mark.parametrize(
    ("meq", "mineq", "active_inequalities", "active_lb", "active_ub", "lb", "ub"),
    [
        (
            1,
            2,
            (True, False),
            (True, False),
            (False, True),
            jnp.array([0.0, -1.0]),
            jnp.array([2.0, 3.0]),
        ),
        (
            0,
            2,
            (False, False),
            (False, False),
            (False, False),
            jnp.array([-jnp.inf, 0.0]),
            jnp.array([1.0, jnp.inf]),
        ),
        (
            1,
            0,
            (),
            (True, True),
            (True, False),
            jnp.array([0.0, -2.0]),
            jnp.array([1.0, 4.0]),
        ),
    ],
    ids=["partial", "all-inactive-with-nulls", "eq-only-bounds"],
)
def test_mask_problem_zeroes_inactive_and_marks_null_bounds(
    meq: int,
    mineq: int,
    active_inequalities: tuple[bool, ...],
    active_lb: tuple[bool, ...],
    active_ub: tuple[bool, ...],
    lb: Array,
    ub: Array,
):
    """Inactive ineq / bound rows are zeroed; inactive bounds become null."""
    n = lb.shape[0]
    problem = _make_problem(n=n, meq=meq, mineq=mineq, lb=lb, ub=ub)
    evaluated = problem(Primal(x=jnp.linspace(0.25, 0.75, n)))
    active = _make_active_set(
        meq,
        mineq,
        n,
        active_inequalities=active_inequalities,
        active_lb=active_lb,
        active_ub=active_ub,
    )
    masked = active.mask_problem(evaluated)

    ai = jnp.asarray(active_inequalities)
    alb = jnp.asarray(active_lb)
    aub = jnp.asarray(active_ub)

    assert jnp.allclose(masked.fn_val, evaluated.fn_val)
    assert jnp.allclose(masked.grad_val, evaluated.grad_val)
    assert jnp.allclose(masked.eq_fn_val, evaluated.eq_fn_val)
    assert jnp.allclose(masked.eq_fn_jac_val, evaluated.eq_fn_jac_val)

    assert jnp.allclose(
        masked.ineq_fn_val,
        jnp.where(ai, evaluated.ineq_fn_val, 0.0),
    )
    assert jnp.allclose(
        masked.ineq_fn_jac_val,
        jnp.where(ai[:, None], evaluated.ineq_fn_jac_val, 0.0),
    )
    assert jnp.allclose(masked.lb, jnp.where(alb, evaluated.lb, 0.0))
    assert jnp.allclose(masked.ub, jnp.where(aub, evaluated.ub, 0.0))
    assert jnp.array_equal(masked.null_lb, evaluated.null_lb | ~alb)
    assert jnp.array_equal(masked.null_ub, evaluated.null_ub | ~aub)
    # Static QVP closures are preserved (cannot be tree_at-replaced).
    assert masked.ineq_fn_qvp is evaluated.ineq_fn_qvp
    assert masked.eq_fn_qvp is evaluated.eq_fn_qvp
    assert masked.fn_qvp is evaluated.fn_qvp


@pytest.mark.parametrize(
    ("meq", "mineq"),
    [(0, 0), (1, 0), (0, 2), (1, 2)],
    ids=["empty", "eq-only", "ineq-only", "both"],
)
def test_mask_lagrangian_applies_problem_and_dual_masks(meq: int, mineq: int):
    """``mask_lagrangian`` matches composing ``mask_problem`` and ``mask_dual``."""
    n = 2
    problem = _make_problem(n=n, meq=meq, mineq=mineq)
    dual = _make_dual(n, meq, mineq)
    lag = Lagrangian(problem)(Primal(x=jnp.array([0.3, 0.4])), dual)
    active = _make_active_set(meq, mineq, n)

    masked = active.mask_lagrangian(lag)
    expected_problem = active.mask_problem(lag.evaluated)
    expected_dual = active.mask_dual(lag.dual)

    assert jnp.allclose(masked.ineq_fn_val, expected_problem.ineq_fn_val)
    assert jnp.allclose(masked.ineq_fn_jac_val, expected_problem.ineq_fn_jac_val)
    assert jnp.allclose(masked.lb, expected_problem.lb)
    assert jnp.allclose(masked.ub, expected_problem.ub)
    assert jnp.array_equal(masked.null_lb, expected_problem.null_lb)
    assert jnp.array_equal(masked.null_ub, expected_problem.null_ub)
    assert jnp.allclose(masked.eq_multipliers, expected_dual.eq_multipliers)
    assert jnp.allclose(masked.ineq_multipliers, expected_dual.ineq_multipliers)
    assert jnp.allclose(masked.lb_multipliers, expected_dual.lb_multipliers)
    assert jnp.allclose(masked.ub_multipliers, expected_dual.ub_multipliers)
    assert masked.secant is lag.secant
