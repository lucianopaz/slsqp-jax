"""Unit tests for :mod:`slsqp_jax.sqpdax.merit`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.barrier import LogBarrier
from slsqp_jax.sqpdax.merit import Merit, NormMerit, safe_norm
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Primal
from slsqp_jax.sqpdax.problem.basic import Problem
from tests.sqpdax.lagrangian.conftest import make_ip_primal, make_primal, make_problem


def _expected_exterior_merit(
    problem: Problem,
    x: Array,
    *,
    norm: int,
    problem_weight: float,
    feasibility_weight: float,
) -> Array:
    """Closed-form exterior :class:`NormMerit` value at array ``x``."""
    f = problem.fn(x)
    eq = problem.eq_fn(x)
    ineq = jnp.concatenate(
        [
            jnp.maximum(0.0, problem.ineq_fn(x)),
            jnp.maximum(0.0, jnp.where(problem.null_lb, 0.0, problem.lb - x)),
            jnp.maximum(0.0, jnp.where(problem.null_ub, 0.0, x - problem.ub)),
        ]
    )
    return problem_weight * f + feasibility_weight * (
        jnp.linalg.norm(eq, ord=norm) + jnp.linalg.norm(ineq, ord=norm)
    )


def _expected_interior_merit(
    problem: Problem,
    barrier: LogBarrier,
    primal: InteriorPointPrimal,
    *,
    norm: int,
    problem_weight: float,
    barrier_weight: float,
    feasibility_weight: float,
) -> Array:
    """Closed-form interior-point :class:`NormMerit` value."""
    x = primal.x
    slack = primal.slack
    f = problem.fn(x)
    eq = problem.eq_fn(x)
    ineq = jnp.concatenate(
        [
            problem.ineq_fn(x) + slack.s,
            jnp.where(problem.null_lb, 0.0, problem.lb - x + slack.s_lb),
            jnp.where(problem.null_ub, 0.0, x - problem.ub + slack.s_ub),
        ]
    )
    return (
        problem_weight * f
        + barrier_weight * barrier.fn(slack)
        + feasibility_weight
        * (jnp.linalg.norm(eq, ord=norm) + jnp.linalg.norm(ineq, ord=norm))
    )


@pytest.mark.parametrize("ord_", [1, 2])
@pytest.mark.parametrize(
    "v",
    [
        jnp.zeros(3),
        jnp.array([3.0, 4.0]),
        jnp.array([-1.0, 2.0, -2.0]),
        jnp.zeros(0),
    ],
    ids=["zero", "pythagorean", "mixed", "empty"],
)
def test_safe_norm_value_and_zero_subgradient(ord_: int, v: Array):
    """``safe_norm`` matches ``linalg.norm`` away from 0; zero at the origin."""
    got = safe_norm(v, ord=ord_)
    if v.size == 0 or jnp.all(v == 0):
        assert jnp.allclose(got, 0.0)
        if v.size > 0:
            assert jnp.allclose(jax.grad(lambda z: safe_norm(z, ord=ord_))(v), 0.0)
    else:
        assert jnp.allclose(got, jnp.linalg.norm(v, ord=ord_))


@pytest.mark.parametrize("with_barrier", [False, True])
def test_merit_has_interior_point(with_barrier: bool):
    """``has_interior_point`` tracks whether a barrier is attached."""
    problem = make_problem()
    barrier = (
        LogBarrier(
            weight=jnp.asarray(1.0),
            null_lb=problem.null_lb,
            null_ub=problem.null_ub,
        )
        if with_barrier
        else None
    )
    merit = NormMerit(problem=problem, barrier=barrier)
    assert isinstance(merit, Merit)
    assert merit.has_interior_point is with_barrier


@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize(
    ("x", "problem_weight", "feasibility_weight"),
    [
        (jnp.array([0.25, 0.75]), 1.0, 1.0),  # feasible
        (jnp.array([2.5, -0.5]), 1.0, 1.0),  # violated
        (jnp.array([2.5, -0.5]), 2.0, 0.5),  # weights
    ],
    ids=["feasible", "violated", "weights"],
)
def test_norm_merit_exterior_value(
    norm: int,
    x: Array,
    problem_weight: float,
    feasibility_weight: float,
):
    """Exterior :class:`NormMerit` matches the closed-form penalty."""
    problem = make_problem()
    merit = NormMerit(
        problem=problem,
        norm=norm,
        problem_weight=jnp.asarray(problem_weight),
        feasibility_weight=jnp.asarray(feasibility_weight),
    )
    got = merit(Primal(x=x))
    expected = _expected_exterior_merit(
        problem,
        x,
        norm=norm,
        problem_weight=problem_weight,
        feasibility_weight=feasibility_weight,
    )
    assert jnp.allclose(got, expected)


@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize(
    ("barrier_weight", "problem_weight", "feasibility_weight"),
    [
        (1.0, 1.0, 1.0),
        (0.5, 2.0, 0.25),
    ],
    ids=["unit-weights", "scaled-weights"],
)
def test_norm_merit_interior_value(
    norm: int,
    barrier_weight: float,
    problem_weight: float,
    feasibility_weight: float,
):
    """Interior-point :class:`NormMerit` includes barrier and slack residuals."""
    problem = make_problem()
    barrier = LogBarrier(
        weight=jnp.asarray(0.75),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    primal = make_ip_primal(n=problem.n, mineq=problem.mineq)
    merit = NormMerit(
        problem=problem,
        barrier=barrier,
        norm=norm,
        problem_weight=jnp.asarray(problem_weight),
        barrier_weight=jnp.asarray(barrier_weight),
        feasibility_weight=jnp.asarray(feasibility_weight),
    )
    got = merit(primal)
    expected = _expected_interior_merit(
        problem,
        barrier,
        primal,
        norm=norm,
        problem_weight=problem_weight,
        barrier_weight=barrier_weight,
        feasibility_weight=feasibility_weight,
    )
    assert jnp.allclose(got, expected)


def test_norm_merit_exterior_grad_at_feasible_point():
    """At a feasible point, exterior merit gradient is ``ω_f ∇f``."""
    problem = make_problem()
    problem_weight = 3.0
    merit = NormMerit(
        problem=problem,
        norm=1,
        problem_weight=jnp.asarray(problem_weight),
        feasibility_weight=jnp.asarray(1.0),
    )
    primal = make_primal(n=problem.n)
    # Sanity: closed form has zero violation at this point.
    assert jnp.allclose(
        _expected_exterior_merit(
            problem,
            primal.x,
            norm=1,
            problem_weight=problem_weight,
            feasibility_weight=1.0,
        ),
        problem_weight * problem.fn(primal.x),
    )
    grad = eqx.filter_grad(merit)(primal)
    assert jnp.allclose(grad.x, problem_weight * problem.grad(primal.x))


def test_norm_merit_respects_null_bounds():
    """Inactive (``±inf``) bounds do not contribute to exterior violations."""
    problem = make_problem(
        lb=jnp.array([-jnp.inf, 0.0]),
        ub=jnp.array([jnp.inf, 1.0]),
    )
    # x[0] is far outside any finite bound; only x[1] can violate.
    x = jnp.array([100.0, 2.0])
    merit = NormMerit(problem=problem, norm=1)
    got = merit(Primal(x=x))
    expected = _expected_exterior_merit(
        problem, x, norm=1, problem_weight=1.0, feasibility_weight=1.0
    )
    assert jnp.allclose(got, expected)
    # Explicit check: upper violation on x[1] only (x[1]-1 = 1).
    assert jnp.allclose(
        got,
        problem.fn(x)
        + jnp.linalg.norm(problem.eq_fn(x), ord=1)
        + jnp.sum(jnp.maximum(0.0, problem.ineq_fn(x)))
        + 1.0,  # x[1] - ub[1]
    )


def test_norm_merit_jittable():
    """Merit evaluation is JIT-compatible for exterior and interior forms."""
    problem = make_problem()
    exterior = NormMerit(problem=problem, norm=1)
    p = make_primal()
    assert jnp.allclose(eqx.filter_jit(exterior)(p), exterior(p))

    barrier = LogBarrier(
        weight=jnp.asarray(1.0),
        null_lb=problem.null_lb,
        null_ub=problem.null_ub,
    )
    interior = NormMerit(problem=problem, barrier=barrier, norm=2)
    ip = make_ip_primal(n=problem.n, mineq=problem.mineq)
    assert jnp.allclose(eqx.filter_jit(interior)(ip), interior(ip))


def test_norm_merit_empty_constraints():
    """Merit with no equalities / inequalities reduces to weighted objective."""
    problem = make_problem(
        meq=0,
        mineq=0,
        lb=jnp.full(2, -jnp.inf),
        ub=jnp.full(2, jnp.inf),
    )
    merit = NormMerit(
        problem=problem,
        norm=1,
        problem_weight=jnp.asarray(2.0),
        feasibility_weight=jnp.asarray(5.0),
    )
    p = Primal(x=jnp.array([1.0, -2.0]))
    assert jnp.allclose(merit(p), 2.0 * problem.fn(p.x))
