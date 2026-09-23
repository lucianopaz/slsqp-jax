"""Tests for the sqpdax SciPy compatibility boundary."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)

from slsqp_jax.sqpdax import (
    build_problem_from_scipy_specification,
    minimise_and_return_scipy_result,
    minimize_like_scipy,
    parse_constraints,
)
from slsqp_jax.sqpdax.compat import _convert_bounds, _prepare_options


def test_dict_constraints_are_mixed_and_apply_per_constraint_args() -> None:
    constraints = [
        {
            "type": "eq",
            "fun": lambda x, target: x[0] + x[1] - target,
            "jac": lambda x, target: jnp.array([[1.0, 1.0]]),
            "args": (3.0,),
        },
        {
            "type": "ineq",
            "fun": lambda x, floor: x[0] - floor,
            "jac": lambda x, floor: jnp.array([1.0, 0.0]),
            "args": (0.5,),
        },
    ]
    x = jnp.array([1.0, 2.0])
    parsed = parse_constraints(constraints, x)
    assert (parsed.meq, parsed.mineq) == (1, 1)
    np.testing.assert_allclose(parsed.eq_fn(x), [0.0])
    np.testing.assert_allclose(parsed.ineq_fn(x), [-0.5])
    np.testing.assert_allclose(parsed.eq_fn_jac(x), [[1.0, 1.0]])
    np.testing.assert_allclose(parsed.ineq_fn_jac(x), [[-1.0, 0.0]])


def test_dict_constraint_without_jac_disables_combined_jacobian() -> None:
    parsed = parse_constraints(
        [
            {"type": "eq", "fun": lambda x: x[:1], "jac": lambda x: jnp.ones((1, 2))},
            {"type": "eq", "fun": lambda x: x[1:]},
        ],
        jnp.ones(2),
    )
    assert parsed.eq_fn_jac is None


def test_dict_constraint_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown constraint type"):
        parse_constraints({"type": "bad", "fun": lambda x: x[0]}, jnp.ones(1))


def test_mixed_tuple_accepts_all_constraint_types() -> None:
    constraints = (
        {"type": "eq", "fun": lambda x: x[0]},
        LinearConstraint([[0.0, 1.0]], 0.0, np.inf),
        NonlinearConstraint(lambda x: x[0] ** 2, -np.inf, 4.0),
    )
    parsed = parse_constraints(constraints, jnp.array([1.0, 1.0]))
    assert (parsed.meq, parsed.mineq) == (1, 2)


@pytest.mark.parametrize(
    ("constraint", "expected_eq", "expected_ineq", "expected_jac"),
    [
        (
            LinearConstraint([[1.0, 2.0]], 5.0, 5.0),
            [0.0],
            None,
            [[1.0, 2.0]],
        ),
        (
            LinearConstraint([[1.0, 2.0]], 6.0, np.inf),
            None,
            [1.0],
            [[-1.0, -2.0]],
        ),
        (
            LinearConstraint([[1.0, 2.0]], -np.inf, 4.0),
            None,
            [1.0],
            [[1.0, 2.0]],
        ),
        (
            LinearConstraint([[1.0, 2.0]], 4.0, 6.0),
            None,
            [-1.0, -1.0],
            [[-1.0, -2.0], [1.0, 2.0]],
        ),
    ],
)
def test_linear_constraint_split_and_signs(
    constraint: LinearConstraint,
    expected_eq: list[float] | None,
    expected_ineq: list[float] | None,
    expected_jac: list[list[float]],
) -> None:
    x = jnp.array([1.0, 2.0])
    parsed = parse_constraints(constraint, x)
    if expected_eq is not None:
        np.testing.assert_allclose(parsed.eq_fn(x), expected_eq)
        np.testing.assert_allclose(parsed.eq_fn_jac(x), expected_jac)
        np.testing.assert_allclose(parsed.eq_fn_hvp(x, x), np.zeros((1, 2)))
    else:
        np.testing.assert_allclose(parsed.ineq_fn(x), expected_ineq)
        np.testing.assert_allclose(parsed.ineq_fn_jac(x), expected_jac)
        np.testing.assert_allclose(
            parsed.ineq_fn_hvp(x, x), np.zeros((len(expected_ineq), 2))
        )


def test_nonlinear_constraint_split_jacobian_and_hvp_signs() -> None:
    def fun(x):
        return jnp.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

    def jac(x):
        return jnp.array([[2 * x[0], 0.0], [0.0, 2 * x[1]], [x[1], x[0]]])

    def hess(x, weights):
        del x
        matrices = jnp.array(
            [
                [[2.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 2.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )
        return jnp.tensordot(weights, matrices, axes=1)

    constraint = NonlinearConstraint(
        fun, [1.0, 1.0, -np.inf], [1.0, np.inf, 3.0], jac=jac, hess=hess
    )
    x = jnp.array([1.0, 2.0])
    direction = jnp.array([3.0, 4.0])
    parsed = parse_constraints(constraint, x)
    assert (parsed.meq, parsed.mineq) == (1, 2)
    np.testing.assert_allclose(parsed.eq_fn(x), [0.0])
    np.testing.assert_allclose(parsed.ineq_fn(x), [-3.0, -1.0])
    np.testing.assert_allclose(parsed.eq_fn_jac(x), [[2.0, 0.0]])
    np.testing.assert_allclose(parsed.ineq_fn_jac(x), [[0.0, -4.0], [2.0, 1.0]])
    np.testing.assert_allclose(parsed.eq_fn_hvp(x, direction), [[6.0, 0.0]])
    np.testing.assert_allclose(
        parsed.ineq_fn_hvp(x, direction), [[0.0, -8.0], [4.0, 3.0]]
    )


def test_nonlinear_hessp_precedes_hess_and_is_cached() -> None:
    calls = {"fun": 0, "jac": 0, "hess": 0, "hessp": 0}

    def fun(x):
        calls["fun"] += 1
        return jnp.array([x[0] ** 2, x[1] ** 2])

    def jac(x):
        calls["jac"] += 1
        return jnp.array([[2 * x[0], 0.0], [0.0, 2 * x[1]]])

    def hess(x, weights):
        del x, weights
        calls["hess"] += 1
        return jnp.eye(2)

    def hessp(x, p):
        del x
        calls["hessp"] += 1
        return jnp.stack([jnp.array([2 * p[0], 0.0]), jnp.array([0.0, 2 * p[1]])])

    constraint = NonlinearConstraint(fun, [1.0, 0.0], [1.0, np.inf], jac=jac, hess=hess)
    constraint.hessp = hessp
    x = jnp.array([1.0, 2.0])
    parsed = parse_constraints(constraint, x)
    calls.update(fun=0, jac=0, hess=0, hessp=0)
    parsed.eq_fn(x)
    parsed.ineq_fn(x)
    parsed.eq_fn_jac(x)
    parsed.ineq_fn_jac(x)
    direction = jnp.ones(2)
    parsed.eq_fn_hvp(x, direction)
    parsed.ineq_fn_hvp(x, direction)
    assert calls == {"fun": 1, "jac": 1, "hess": 0, "hessp": 1}


@pytest.mark.parametrize(
    "bad_hessp",
    [lambda x: x, lambda x, p, extra: x + p + extra],
)
def test_nonlinear_hessp_validates_arity(bad_hessp: Callable) -> None:
    constraint = NonlinearConstraint(lambda x: x, 0.0, 1.0)
    constraint.hessp = bad_hessp
    with pytest.raises(TypeError, match="exactly two positional"):
        parse_constraints(constraint, jnp.ones(1))


@pytest.mark.parametrize(
    "bounds",
    [
        [(np.nan, 1.0)],
        [(2.0, 1.0)],
        [(np.inf, np.inf)],
        [(-np.inf, -np.inf)],
        Bounds(np.nan, 1.0),
        Bounds(2.0, 1.0),
        Bounds(np.inf, np.inf),
        Bounds(-np.inf, -np.inf),
    ],
)
def test_bounds_reject_invalid_endpoints_before_jax(bounds) -> None:
    with pytest.raises(ValueError, match="index 0"):
        _convert_bounds(bounds, 1)


def test_bounds_convert_none_endpoints_broadcast_and_fixed_values() -> None:
    lb, ub = _convert_bounds([(None, 2.0), (1.0, None), (3.0, 3.0)], 3)
    np.testing.assert_array_equal(lb, [-np.inf, 1.0, 3.0])
    np.testing.assert_array_equal(ub, [2.0, np.inf, 3.0])
    lb, ub = _convert_bounds(Bounds(0.0, [1.0, 2.0, 3.0]), 3)
    np.testing.assert_array_equal(lb, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(ub, [1.0, 2.0, 3.0])


def test_bounds_reject_wrong_length() -> None:
    with pytest.raises(ValueError, match="2 entries"):
        _convert_bounds([(0.0, 1.0), (0.0, 1.0)], 1)


def test_constraint_bounds_are_also_validated_at_conversion_boundary() -> None:
    with pytest.raises(ValueError, match="LinearConstraint"):
        parse_constraints(LinearConstraint([[1.0]], 2.0, 1.0), jnp.ones(1))
    with pytest.raises(ValueError, match="NonlinearConstraint"):
        parse_constraints(NonlinearConstraint(lambda x: x, np.nan, 1.0), jnp.ones(1))


def test_minimize_supports_callable_jac_hessp_and_args() -> None:
    hessp_calls = 0

    def fun(x, scale):
        return scale * jnp.sum(x**2)

    def jac(x, scale):
        return 2 * scale * x

    def hessp(x, p, scale):
        nonlocal hessp_calls
        del x
        hessp_calls += 1
        return 2 * scale * p

    solution = minimize_like_scipy(
        fun,
        jnp.array([2.0, -1.0]),
        args=(2.0,),
        jac=jac,
        hessp=hessp,
        options={"maxiter": 20},
    )
    np.testing.assert_allclose(solution.x, 0.0, atol=1e-5)
    assert hessp_calls > 0


def test_minimize_preserves_constraint_hessp_when_jac_is_autodiffed() -> None:
    constraint_hessp_calls = 0

    def constraint_hessp(x, p):
        nonlocal constraint_hessp_calls
        del x
        constraint_hessp_calls += 1
        return jnp.zeros((1, p.shape[0]))

    constraint = NonlinearConstraint(lambda x: x[:1], 0.0, np.inf)
    constraint.hessp = constraint_hessp
    solution = minimize_like_scipy(
        lambda x: jnp.sum(x**2),
        jnp.array([1.0]),
        hessp=lambda x, p: 2 * p,
        constraints=constraint,
        options={"maxiter": 20},
    )
    np.testing.assert_allclose(solution.x, 0.0, atol=1e-5)
    assert constraint_hessp_calls > 0


def test_minimize_supports_jac_true_and_aux() -> None:
    def fun(x):
        value = jnp.sum(x**2)
        return (value, 2 * x), {"norm": jnp.linalg.norm(x)}

    solution = minimize_like_scipy(
        fun,
        jnp.array([2.0, -1.0]),
        jac=True,
        has_aux=True,
        options={"maxiter": 20},
    )
    np.testing.assert_allclose(solution.x, 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.aux["norm"], 0.0, atol=1e-5)


def test_minimize_forwards_native_nested_options() -> None:
    solution = minimize_like_scipy(
        lambda x: jnp.sum(x**2),
        jnp.array([2.0]),
        options={
            "maxiter": 20,
            "minimiser": {"atol": 2e-6, "secant_memory": 4},
            "subproblem": {
                "tol": 3e-9,
                "subproblem_solver": {"max_iter": 7},
            },
        },
    )
    assert solution.success


def test_build_and_solve_steps_are_available_separately() -> None:
    def fun(x, target):
        return jnp.sum((x - target) ** 2)

    problem = build_problem_from_scipy_specification(fun, jnp.ones(2))
    target = jnp.array([0.25, -0.75])
    result = minimise_and_return_scipy_result(
        problem,
        jnp.ones(2),
        args=(target,),
        options={"maxiter": 40},
    )
    assert isinstance(result, OptimizeResult)
    assert result.success
    np.testing.assert_allclose(result.x, target, atol=1e-4)
    np.testing.assert_allclose(result.jac, 0.0, atol=1e-4)
    assert result.nit >= 1


def test_scipy_result_maps_iteration_budget_failure() -> None:
    result = minimize_like_scipy(
        lambda x: jnp.sum(x**2),
        jnp.ones(2),
        options={"maxiter": 0},
        throw=False,
    )
    assert not result.success
    assert result.status == 1
    assert result.nit == 0
    assert "maximum number" in result.message.lower()


def test_prepare_options_only_consumes_maxiter() -> None:
    supplied = {
        "maxiter": 12,
        "atol": 2e-6,
        "minimiser": {"secant_memory": 4},
        "subproblem": {"tol": 3e-9},
    }
    max_steps, options = _prepare_options(supplied)
    assert max_steps == 12
    assert options == {
        "atol": 2e-6,
        "minimiser": {"secant_memory": 4},
        "subproblem": {"tol": 3e-9},
    }


def test_prepare_options_does_not_consume_removed_aliases() -> None:
    aliases = {"max_steps": 5, "lbfgs_memory": 4, "qp_max_cg_iter": 7}
    max_steps, options = _prepare_options(aliases)
    assert max_steps == 100
    assert options == aliases


def test_minimize_rejects_bad_bounds_before_building_problem(monkeypatch) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr("slsqp_jax.sqpdax.compat.build_problem", fail_if_called)
    with pytest.raises(ValueError, match="NaN"):
        minimize_like_scipy(lambda x: x[0] ** 2, [1.0], bounds=[(np.nan, 2.0)])
    assert not called


@pytest.mark.parametrize(
    ("bounds", "constraints", "x0"),
    [
        ([(0.0, 3.0)], (), np.array([1.0])),
        (None, LinearConstraint([[1.0]], 1.0, np.inf), np.array([2.0])),
        (
            None,
            NonlinearConstraint(lambda x: x[0] ** 2, 1.0, np.inf),
            np.array([2.0]),
        ),
    ],
)
def test_end_to_end_matches_scipy(bounds, constraints, x0) -> None:
    def objective(x):
        return (x[0] - 0.25) ** 2

    reference = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    solution = minimize_like_scipy(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100},
        throw=False,
    )
    assert isinstance(solution, OptimizeResult)
    np.testing.assert_allclose(solution.x, reference.x, atol=2e-3)
