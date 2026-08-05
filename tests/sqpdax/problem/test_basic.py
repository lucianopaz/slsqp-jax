"""Unit tests for :mod:`slsqp_jax.sqpdax.problem.basic`."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import EvaluatedProblem, Problem, ProblemProtocol


def _obj(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return a * b * jnp.sum(x**2)


def _obj_grad(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return 2 * a * b * x


def _obj_hvp(
    x: Array, v: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0
) -> Array:
    return 2 * a * b * v


def _eq(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return jnp.array([a * b * (x[0] + x[1] - 1.0)])


def _eq_jac(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return jnp.array([[a * b, a * b]])


def _eq_hvp(
    x: Array, v: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0
) -> Array:
    return jnp.zeros((1, x.shape[-1]), dtype=x.dtype)


def _ineq(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return jnp.array([a * b * (x[0] - 2.0), -a * b * x[1]])


def _ineq_jac(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return jnp.array([[a * b, 0.0], [0.0, -a * b]])


def _ineq_hvp(
    x: Array, v: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0
) -> Array:
    return jnp.zeros((2, x.shape[-1]), dtype=x.dtype)


def _empty(x: Array, *args, **kwargs) -> Array:
    return jnp.zeros((0,), dtype=x.dtype)


def _empty_jac(x: Array, *args, **kwargs) -> Array:
    return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)


def _empty_hvp(x: Array, v: Array, *args, **kwargs) -> Array:
    return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)


def _make_problem(
    *,
    n: int = 2,
    meq: int = 1,
    mineq: int = 2,
    hvp: Callable | None = _obj_hvp,
    eq_fn: Callable = _eq,
    eq_fn_jac: Callable = _eq_jac,
    eq_fn_hvp: Callable | None = _eq_hvp,
    ineq_fn: Callable = _ineq,
    ineq_fn_jac: Callable = _ineq_jac,
    ineq_fn_hvp: Callable | None = _ineq_hvp,
    lb: Array | None = None,
    ub: Array | None = None,
) -> Problem:
    if lb is None:
        lb = jnp.array([0.0, -1.0])
    if ub is None:
        ub = jnp.array([2.0, 3.0])
    return Problem(
        fn=_obj,
        grad=_obj_grad,
        hvp=hvp,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        eq_fn_jac=eq_fn_jac,
        ineq_fn_jac=ineq_fn_jac,
        eq_fn_hvp=eq_fn_hvp,
        ineq_fn_hvp=ineq_fn_hvp,
        lb=lb,
        ub=ub,
        null_lb=jnp.isinf(lb) & (lb < 0),
        null_ub=jnp.isinf(ub) & (ub > 0),
        n=n,
        meq=meq,
        mineq=mineq,
    )


@pytest.mark.parametrize(
    ("meq", "mineq", "eq_fn", "eq_jac", "ineq_fn", "ineq_jac"),
    [
        (0, 0, _empty, _empty_jac, _empty, _empty_jac),
        (1, 0, _eq, _eq_jac, _empty, _empty_jac),
        (0, 2, _empty, _empty_jac, _ineq, _ineq_jac),
        (1, 2, _eq, _eq_jac, _ineq, _ineq_jac),
    ],
    ids=["empty", "eq-only", "ineq-only", "both"],
)
@pytest.mark.parametrize("with_curvature", [True, False])
def test_problem_call_values_and_qvps(
    meq: int,
    mineq: int,
    eq_fn: Callable,
    eq_jac: Callable,
    ineq_fn: Callable,
    ineq_jac: Callable,
    with_curvature: bool,
):
    """``Problem(x)`` matches callables; QVPs appear only with full HVPs."""
    n = 2
    problem = _make_problem(
        n=n,
        meq=meq,
        mineq=mineq,
        hvp=_obj_hvp if with_curvature else None,
        eq_fn=eq_fn,
        eq_fn_jac=eq_jac,
        eq_fn_hvp=_eq_hvp
        if (with_curvature and meq > 0)
        else (_empty_hvp if with_curvature else None),
        ineq_fn=ineq_fn,
        ineq_fn_jac=ineq_jac,
        ineq_fn_hvp=_ineq_hvp
        if (with_curvature and mineq > 0)
        else (_empty_hvp if with_curvature else None),
    )
    x = Primal(x=jnp.array([0.5, 0.25]))
    v = jnp.array([0.1, -0.2])
    evaluated = problem(x)

    assert evaluated.ref is x or jnp.allclose(evaluated.ref.x, x.x)
    assert evaluated.n == n
    assert evaluated.meq == meq
    assert evaluated.mineq == mineq
    assert jnp.allclose(evaluated.fn_val, _obj(x.x))
    assert jnp.allclose(evaluated.grad_val, _obj_grad(x.x))
    assert jnp.allclose(evaluated.eq_fn_val, eq_fn(x.x))
    assert jnp.allclose(evaluated.eq_fn_jac_val, eq_jac(x.x))
    assert jnp.allclose(evaluated.ineq_fn_val, ineq_fn(x.x))
    assert jnp.allclose(evaluated.ineq_fn_jac_val, ineq_jac(x.x))
    assert jnp.allclose(evaluated.lb, problem.lb)
    assert jnp.allclose(evaluated.ub, problem.ub)
    assert jnp.array_equal(evaluated.null_lb, problem.null_lb)
    assert jnp.array_equal(evaluated.null_ub, problem.null_ub)

    assert problem.has_exact_curvature is with_curvature
    assert evaluated.has_exact_curvature is with_curvature
    if with_curvature:
        assert evaluated.fn_qvp is not None
        assert evaluated.eq_fn_qvp is not None
        assert evaluated.ineq_fn_qvp is not None
        assert jnp.allclose(evaluated.fn_qvp(v), _obj_hvp(x.x, v))
        assert evaluated.eq_fn_qvp(v).shape == (meq, n)
        assert evaluated.ineq_fn_qvp(v).shape == (mineq, n)
    else:
        assert evaluated.fn_qvp is None
        assert evaluated.eq_fn_qvp is None
        assert evaluated.ineq_fn_qvp is None


def test_problem_call_forwards_args_kwargs():
    """Extra positional / keyword arguments reach every callable."""
    problem = _make_problem()
    x = Primal(x=jnp.array([0.5, 0.25]))
    a = jnp.asarray(2.0)
    evaluated = problem(x, a, b=3.0)
    v = jnp.array([0.25, -0.5])

    assert jnp.allclose(evaluated.fn_val, _obj(x.x, a, b=3.0))
    assert jnp.allclose(evaluated.grad_val, _obj_grad(x.x, a, b=3.0))
    assert jnp.allclose(evaluated.eq_fn_val, _eq(x.x, a, b=3.0))
    assert jnp.allclose(evaluated.eq_fn_jac_val, _eq_jac(x.x, a, b=3.0))
    assert jnp.allclose(evaluated.ineq_fn_val, _ineq(x.x, a, b=3.0))
    assert jnp.allclose(evaluated.ineq_fn_jac_val, _ineq_jac(x.x, a, b=3.0))
    assert evaluated.fn_qvp is not None
    assert jnp.allclose(evaluated.fn_qvp(v), _obj_hvp(x.x, v, a, b=3.0))


@pytest.mark.parametrize(
    ("hvp", "eq_hvp", "ineq_hvp", "expected"),
    [
        (_obj_hvp, _eq_hvp, _ineq_hvp, True),
        (None, _eq_hvp, _ineq_hvp, False),
        (_obj_hvp, None, _ineq_hvp, False),
        (_obj_hvp, _eq_hvp, None, False),
        (None, None, None, False),
    ],
)
def test_problem_has_exact_curvature(
    hvp: Callable | None,
    eq_hvp: Callable | None,
    ineq_hvp: Callable | None,
    expected: bool,
):
    """Exact curvature requires all three HVPs to be non-``None``."""
    problem = _make_problem(hvp=hvp, eq_fn_hvp=eq_hvp, ineq_fn_hvp=ineq_hvp)
    assert problem.has_exact_curvature is expected


@pytest.mark.parametrize(
    ("fn_qvp", "eq_qvp", "ineq_qvp", "expected"),
    [
        (lambda p: p, lambda p: jnp.zeros((1, 2)), lambda p: jnp.zeros((2, 2)), True),
        (None, lambda p: jnp.zeros((1, 2)), lambda p: jnp.zeros((2, 2)), False),
        (lambda p: p, None, lambda p: jnp.zeros((2, 2)), False),
        (lambda p: p, lambda p: jnp.zeros((1, 2)), None, False),
    ],
)
def test_evaluated_problem_has_exact_curvature(
    fn_qvp: Callable | None,
    eq_qvp: Callable | None,
    ineq_qvp: Callable | None,
    expected: bool,
):
    """``EvaluatedProblem.has_exact_curvature`` mirrors QVP availability."""
    n, meq, mineq = 2, 1, 2
    evaluated = EvaluatedProblem(
        ref=Primal(x=jnp.zeros(n)),
        fn_val=jnp.asarray(0.0),
        grad_val=jnp.zeros(n),
        fn_qvp=fn_qvp,
        eq_fn_val=jnp.zeros(meq),
        eq_fn_jac_val=jnp.zeros((meq, n)),
        eq_fn_qvp=eq_qvp,
        ineq_fn_val=jnp.zeros(mineq),
        ineq_fn_jac_val=jnp.zeros((mineq, n)),
        ineq_fn_qvp=ineq_qvp,
        lb=jnp.full((n,), -jnp.inf),
        ub=jnp.full((n,), jnp.inf),
        null_lb=jnp.array([True, True]),
        null_ub=jnp.array([True, True]),
    )
    assert evaluated.n == n
    assert evaluated.meq == meq
    assert evaluated.mineq == mineq
    assert evaluated.has_exact_curvature is expected


def test_problem_satisfies_protocol():
    """:class:`Problem` is structurally a :class:`ProblemProtocol`."""
    problem = _make_problem()
    assert isinstance(problem, ProblemProtocol)
