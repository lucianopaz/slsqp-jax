"""Unit tests for :mod:`slsqp_jax.sqpdax.problem.builder`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.problem.builder import build_problem


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


def _call_build(
    *,
    n: int = 2,
    meq: int | None = None,
    mineq: int | None = None,
    fn: Callable = _obj,
    grad: Callable | None = _obj_grad,
    hvp: Callable | None = None,
    eq_fn: Callable | None = None,
    ineq_fn: Callable | None = None,
    eq_fn_jac: Callable | None = None,
    ineq_fn_jac: Callable | None = None,
    eq_fn_hvp: Callable | None = None,
    ineq_fn_hvp: Callable | None = None,
    lb: Array | None = None,
    ub: Array | None = None,
    autodiff_mode: Literal["jax", "custom", "none"] = "none",
    force_hvp_in_jax_mode: bool = False,
) -> Problem:
    return build_problem(
        n=n,
        meq=meq,
        mineq=mineq,
        fn=fn,
        grad=grad,  # ty: ignore[arg-type]
        hvp=hvp,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        eq_fn_jac=eq_fn_jac,
        ineq_fn_jac=ineq_fn_jac,
        eq_fn_hvp=eq_fn_hvp,
        ineq_fn_hvp=ineq_fn_hvp,
        lb=lb,
        ub=ub,
        autodiff_mode=autodiff_mode,
        force_hvp_in_jax_mode=force_hvp_in_jax_mode,
    )


@pytest.mark.parametrize(
    ("eq_fn", "eq_fn_jac", "meq", "ineq_fn", "ineq_fn_jac", "mineq"),
    [
        (None, None, None, None, None, None),
        (_eq, _eq_jac, 1, None, None, None),
        (None, None, None, _ineq, _ineq_jac, 2),
        (_eq, _eq_jac, 1, _ineq, _ineq_jac, 2),
    ],
    ids=["unconstrained", "eq-only", "ineq-only", "eq-and-ineq"],
)
def test_build_problem_dimensions_and_stubs(
    eq_fn: Callable | None,
    eq_fn_jac: Callable | None,
    meq: int | None,
    ineq_fn: Callable | None,
    ineq_fn_jac: Callable | None,
    mineq: int | None,
):
    """Dimensions match inputs; omitted constraints become empty callables."""
    n = 2
    problem = _call_build(
        n=n,
        meq=meq,
        mineq=mineq,
        eq_fn=eq_fn,
        eq_fn_jac=eq_fn_jac,
        ineq_fn=ineq_fn,
        ineq_fn_jac=ineq_fn_jac,
    )
    x = jnp.array([0.5, 0.25])
    expected_meq = 0 if eq_fn is None else meq
    expected_mineq = 0 if ineq_fn is None else mineq

    assert isinstance(problem, Problem)
    assert problem.n == n
    assert problem.meq == expected_meq
    assert problem.mineq == expected_mineq
    assert jnp.allclose(problem.fn(x), _obj(x))
    assert jnp.allclose(problem.grad(x), _obj_grad(x))

    eq_val = problem.eq_fn(x)
    eq_jac = problem.eq_fn_jac(x)
    assert eq_val.shape == (expected_meq,)
    assert eq_jac.shape == (expected_meq, n)
    if eq_fn is None:
        assert jnp.allclose(eq_val, jnp.zeros((0,)))
        assert jnp.allclose(eq_jac, jnp.zeros((0, n)))
        assert problem.eq_fn_hvp is not None
        assert problem.eq_fn_hvp(x, x).shape == (0, n)
    else:
        assert jnp.allclose(eq_val, _eq(x))
        assert jnp.allclose(eq_jac, _eq_jac(x))

    ineq_val = problem.ineq_fn(x)
    ineq_jac = problem.ineq_fn_jac(x)
    assert ineq_val.shape == (expected_mineq,)
    assert ineq_jac.shape == (expected_mineq, n)
    if ineq_fn is None:
        assert jnp.allclose(ineq_val, jnp.zeros((0,)))
        assert jnp.allclose(ineq_jac, jnp.zeros((0, n)))
        assert problem.ineq_fn_hvp is not None
        assert problem.ineq_fn_hvp(x, x).shape == (0, n)
    else:
        assert jnp.allclose(ineq_val, _ineq(x))
        assert jnp.allclose(ineq_jac, _ineq_jac(x))


@pytest.mark.parametrize(
    ("lb", "ub", "expect_null_lb", "expect_null_ub"),
    [
        (None, None, [True, True], [True, True]),
        (jnp.array([0.0, -jnp.inf]), None, [False, True], [True, True]),
        (None, jnp.array([jnp.inf, 1.0]), [True, True], [True, False]),
        (
            jnp.array([-1.0, 0.0]),
            jnp.array([1.0, 2.0]),
            [False, False],
            [False, False],
        ),
    ],
    ids=["both-none", "partial-lb", "partial-ub", "finite"],
)
def test_build_problem_bounds_and_null_masks(
    lb: Array | None,
    ub: Array | None,
    expect_null_lb: list[bool],
    expect_null_ub: list[bool],
):
    """``None`` bounds become ±inf; null masks flag infinite sides."""
    n = 2
    problem = _call_build(n=n, lb=lb, ub=ub)
    expected_lb = lb if lb is not None else jnp.full((n,), -jnp.inf)
    expected_ub = ub if ub is not None else jnp.full((n,), jnp.inf)

    assert jnp.array_equal(jnp.isneginf(problem.lb), jnp.isneginf(expected_lb))
    assert jnp.array_equal(jnp.isposinf(problem.ub), jnp.isposinf(expected_ub))
    assert jnp.allclose(
        jnp.where(jnp.isinf(expected_lb), 0.0, problem.lb),
        jnp.where(jnp.isinf(expected_lb), 0.0, expected_lb),
    )
    assert jnp.allclose(
        jnp.where(jnp.isinf(expected_ub), 0.0, problem.ub),
        jnp.where(jnp.isinf(expected_ub), 0.0, expected_ub),
    )
    assert jnp.array_equal(problem.null_lb, jnp.asarray(expect_null_lb))
    assert jnp.array_equal(problem.null_ub, jnp.asarray(expect_null_ub))


@pytest.mark.parametrize("autodiff_mode", ["jax", "custom", "none"])
@pytest.mark.parametrize("with_hvp", [True, False])
@pytest.mark.parametrize("force_hvp_in_jax_mode", [False, True])
def test_build_problem_autodiff_modes(
    autodiff_mode: Literal["jax", "custom", "none"],
    with_hvp: bool,
    force_hvp_in_jax_mode: bool,
):
    """Objective derivatives resolve consistently across autodiff modes."""
    if force_hvp_in_jax_mode and autodiff_mode != "jax":
        pytest.skip("force_hvp_in_jax_mode only affects jax mode")

    x = jnp.array([1.0, -0.5])
    v = jnp.array([0.25, 0.5])
    # In jax mode a non-None hvp is only a placeholder that forces building a
    # JAX HVP; the callable itself is not returned.
    hvp = _obj_hvp if with_hvp else None
    grad = None if autodiff_mode == "jax" else _obj_grad
    problem = _call_build(
        grad=grad,
        hvp=hvp,
        autodiff_mode=autodiff_mode,
        force_hvp_in_jax_mode=force_hvp_in_jax_mode,
    )

    assert jnp.allclose(problem.fn(x), _obj(x))
    assert jnp.allclose(problem.grad(x), _obj_grad(x))

    if autodiff_mode == "jax":
        expect_hvp = force_hvp_in_jax_mode or with_hvp
        if expect_hvp:
            assert problem.hvp is not None
            assert jnp.allclose(problem.hvp(x, v), _obj_hvp(x, v))
            # Empty constraint stubs always supply HVPs, so exact curvature
            # follows the objective HVP alone in this unconstrained build.
            assert problem.has_exact_curvature
        else:
            assert problem.hvp is None
            assert not problem.has_exact_curvature
    elif with_hvp:
        assert problem.hvp is _obj_hvp
        assert problem.has_exact_curvature
    else:
        assert problem.hvp is None
        assert not problem.has_exact_curvature

    if autodiff_mode == "custom":
        assert jnp.allclose(eqx.filter_grad(problem.fn)(x), _obj_grad(x))


def test_build_problem_jax_mode_builds_constraint_derivatives():
    """``autodiff_mode='jax'`` differentiates provided constraint callables."""

    def eq_fn(x: Array) -> Array:
        return jnp.array([x[0] ** 2 + x[1] - 1.0])

    def ineq_fn(x: Array) -> Array:
        return jnp.array([x[0] ** 2 - 2.0, -(x[1] ** 2)])

    problem = _call_build(
        meq=1,
        mineq=2,
        grad=None,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        autodiff_mode="jax",
        force_hvp_in_jax_mode=True,
    )
    x = jnp.array([0.5, 0.25])
    v = jnp.array([1.0, -1.0])

    assert jnp.allclose(problem.eq_fn_jac(x), jax.jacrev(eq_fn)(x))
    assert jnp.allclose(problem.ineq_fn_jac(x), jax.jacrev(ineq_fn)(x))
    assert problem.eq_fn_hvp is not None
    assert problem.ineq_fn_hvp is not None
    expected_eq_hvp = jax.jvp(lambda z: jax.jacrev(eq_fn)(z), (x,), (v,))[1]
    expected_ineq_hvp = jax.jvp(lambda z: jax.jacrev(ineq_fn)(z), (x,), (v,))[1]
    assert jnp.allclose(problem.eq_fn_hvp(x, v), expected_eq_hvp)
    assert jnp.allclose(problem.ineq_fn_hvp(x, v), expected_ineq_hvp)
    assert problem.has_exact_curvature


def test_build_problem_evaluates_via_call():
    """Built problems evaluate cleanly through :meth:`Problem.__call__`."""
    problem = _call_build(
        meq=1,
        mineq=2,
        hvp=_obj_hvp,
        eq_fn=_eq,
        eq_fn_jac=_eq_jac,
        eq_fn_hvp=_eq_hvp,
        ineq_fn=_ineq,
        ineq_fn_jac=_ineq_jac,
        ineq_fn_hvp=_ineq_hvp,
        lb=jnp.array([0.0, -1.0]),
        ub=jnp.array([2.0, 3.0]),
    )
    x = Primal(x=jnp.array([0.5, 0.25]))
    evaluated = problem(x)
    v = jnp.array([0.1, -0.2])

    assert evaluated.n == 2
    assert evaluated.meq == 1
    assert evaluated.mineq == 2
    assert evaluated.has_exact_curvature
    assert jnp.allclose(evaluated.fn_val, _obj(x.x))
    assert jnp.allclose(evaluated.grad_val, _obj_grad(x.x))
    assert evaluated.fn_qvp is not None
    assert jnp.allclose(evaluated.fn_qvp(v), _obj_hvp(x.x, v))
    assert jnp.allclose(evaluated.eq_fn_val, _eq(x.x))
    assert jnp.allclose(evaluated.ineq_fn_val, _ineq(x.x))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"eq_fn": _eq, "eq_fn_jac": _eq_jac, "meq": None},
            "number of equality constraint functions, meq",
        ),
        (
            {"ineq_fn": _ineq, "ineq_fn_jac": _ineq_jac, "mineq": None},
            "number of inequality constraint functions, mineq",
        ),
    ],
)
def test_build_problem_requires_constraint_counts(kwargs: dict, match: str):
    """Providing a constraint callable without its count raises."""
    with pytest.raises(AssertionError, match=match):
        _call_build(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"eq_fn": _eq, "meq": 1, "eq_fn_jac": None, "autodiff_mode": "custom"},
            "Failed to autodiff equality constraint",
        ),
        (
            {
                "ineq_fn": _ineq,
                "mineq": 2,
                "ineq_fn_jac": None,
                "autodiff_mode": "custom",
            },
            "Failed to autodiff inequality constraint",
        ),
    ],
)
def test_build_problem_wraps_constraint_autodiff_errors(kwargs: dict, match: str):
    """Constraint autodiff failures are re-raised with a block-specific prefix."""
    with pytest.raises(ValueError, match=match):
        _call_build(**kwargs)


def test_build_problem_objective_autodiff_error_propagates():
    """Objective autodiff errors propagate without a constraint prefix."""
    with pytest.raises(ValueError, match="grad must be provided"):
        _call_build(grad=None, autodiff_mode="custom")
