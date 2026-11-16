"""Memory-budget audit of the LRH minimiser (``.claude/rules/memory-budget.md``).

Every component is traced with :func:`jax.make_jaxpr` (nothing is executed) at
``n = 4096`` with all ``2n`` bounds finite, and every intermediate of the
jaxpr, sub-jaxprs of ``while`` / ``cond`` / ``scan`` / ``pjit`` / custom
derivative calls included, is checked against the budget:

* no array has two or more axes of length ``>= n / 2`` (no ``n x n``), and
* no array is larger than ``4 n max(memory + 1, m_c + 2)``; the ``+ 2`` is the
  elastic augmentation (an ``(n + 1) x (m_c + 2)`` least-distance system).

The synthetic problem's own callables are ``O(n)`` (elementwise objective,
dense ``m_c x n`` Jacobians), so any violation comes from the code under test.
The negative controls check that the walker does flag the forbidden patterns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.minimiser import LRHLineSearchMinimiser
from slsqp_jax.sqpdax.problem import Problem
from slsqp_jax.sqpdax.secant import LimitedMemoryReducedHessian
from slsqp_jax.sqpdax.subproblem.solver import LRHDirectQPSolver

N = 4096
MEQ = 3
MINEQ = 4
M_C = MEQ + MINEQ
MEMORY = 8
# ``+ 2``: the elastic QP adds one coordinate and two general rows.
SIZE_LIMIT = 4 * N * max(MEMORY + 1, M_C + 2)


@pytest.fixture(autouse=True)
def x64():
    """Run every test in float64."""
    with jax.enable_x64(True):
        yield


# --- jaxpr walker -----------------------------------------------------------------


class Aval(NamedTuple):
    """One variable of a (sub-)jaxpr."""

    shape: tuple[int, ...]
    where: str


def _nested_jaxprs(param: Any):
    """Yield every jaxpr inside ``param`` (containers and dicts, any depth)."""
    inner = getattr(param, "jaxpr", param)
    if hasattr(inner, "eqns"):
        yield inner
    elif isinstance(param, dict):
        for sub in param.values():
            yield from _nested_jaxprs(sub)
    elif isinstance(param, (tuple, list)):  # includes named tuples
        for sub in param:
            yield from _nested_jaxprs(sub)


def _sub_jaxprs(eqn: Any):
    """Yield the jaxprs held by the parameters of ``eqn`` (any nesting)."""
    for param in eqn.params.values():
        yield from _nested_jaxprs(param)


def walk(jaxpr: Any, where: str = "top"):
    """Yield an :class:`Aval` for every variable, recursing into sub-jaxprs."""
    for var in (*jaxpr.invars, *jaxpr.constvars):
        yield Aval(tuple(getattr(var.aval, "shape", ())), f"{where}:input")
    for eqn in jaxpr.eqns:
        here = f"{where}/{eqn.primitive.name}"
        for var in eqn.outvars:
            yield Aval(tuple(getattr(var.aval, "shape", ())), here)
        for inner in _sub_jaxprs(eqn):
            yield from walk(inner, here)


def violations(closed_jaxpr: Any, *, n: int = N, size_limit: int = SIZE_LIMIT):
    """Return the avals that break the budget (empty when the trace is clean)."""
    bad = []
    for aval in walk(closed_jaxpr.jaxpr):
        wide_axes = sum(1 for d in aval.shape if d >= n // 2)
        # A matrix with an axis beyond ``n + m_c + 2`` is bounds stacked as
        # rows (``m_c + 2n``); 1-D arrays (multiplier vectors) are exempt.
        too_long = len(aval.shape) >= 2 and max(aval.shape) > n + M_C + 2
        if (
            wide_axes >= 2
            or too_long
            or int(np.prod(aval.shape, dtype=np.int64)) > size_limit
        ):
            bad.append(aval)
    return bad


def largest(closed_jaxpr: Any) -> Aval:
    """The largest intermediate of the trace (by element count)."""
    return max(
        walk(closed_jaxpr.jaxpr),
        key=lambda a: int(np.prod(a.shape, dtype=np.int64)),
    )


def has_primitive(closed_jaxpr: Any, name: str) -> bool:
    """Whether any equation (sub-jaxprs included) is a ``name`` primitive."""

    def search(jaxpr: Any) -> bool:
        return any(
            e.primitive.name == name or any(search(s) for s in _sub_jaxprs(e))
            for e in jaxpr.eqns
        )

    return search(closed_jaxpr.jaxpr)


# --- the audited problem and components --------------------------------------------


@pytest.fixture(scope="module")
def problem() -> Problem:
    """``n = 4096`` NLP with 3 equalities, 4 inequalities and finite bounds.

    The objective is elementwise and the Jacobians are constant dense
    ``m x n`` matrices: nothing here is ``O(n^2)``.
    """
    rng = np.random.default_rng(0)
    a_eq = jnp.asarray(rng.standard_normal((MEQ, N)) / np.sqrt(N))
    a_in = jnp.asarray(rng.standard_normal((MINEQ, N)) / np.sqrt(N))
    w = jnp.asarray(1.0 + rng.random(N))
    c = jnp.asarray(rng.standard_normal(N))
    lb, ub = jnp.full(N, -2.0), jnp.full(N, 2.0)

    def fn(x):
        return 0.5 * jnp.sum(w * (x - c) ** 2) + 0.1 * jnp.sum(x**4), None

    return Problem(
        fn=fn,
        grad=lambda x: w * (x - c) + 0.4 * x**3,
        hvp=None,
        eq_fn=lambda x: a_eq @ x - 0.5,
        eq_fn_jac=lambda x: a_eq.astype(x.dtype),
        eq_fn_hvp=None,
        ineq_fn=lambda x: a_in @ x - 0.5,
        ineq_fn_jac=lambda x: a_in.astype(x.dtype),
        ineq_fn_hvp=None,
        lb=lb,
        ub=ub,
        null_lb=jnp.zeros(N, dtype=bool),
        null_ub=jnp.zeros(N, dtype=bool),
        n=N,
        meq=MEQ,
        mineq=MINEQ,
    )


@pytest.fixture(scope="module")
def x0() -> jax.Array:
    return jnp.asarray(np.random.default_rng(1).standard_normal(N))


@pytest.fixture(scope="module")
def minimiser() -> LRHLineSearchMinimiser:
    # ``restoration`` is on by default; the fallback sits inside the step trace.
    return LRHLineSearchMinimiser(secant_memory=MEMORY, restoration=True)


@pytest.fixture(scope="module")
def initial(minimiser, problem, x0) -> LRHLineSearchMinimiser:
    """The initialised minimiser (the concrete input of the traced ``step``)."""
    with jax.enable_x64(True):
        return minimiser.init(problem, x0)


@pytest.fixture(scope="module")
def traces(minimiser, initial, problem, x0) -> dict[str, Any]:
    """Jaxprs of every audited function (traced once, shared by the tests)."""

    def solve():
        ctx = initial._init_subproblem(problem)
        solver = LRHDirectQPSolver(elastic=True)
        return solver.solve(ctx.subproblem, ctx.warm, ctx.state)

    def restoration():
        ctx = initial._init_subproblem(problem)
        step, dual, state = initial._solve_direction(ctx)
        result = initial._assess_direction(ctx, step, dual, state)
        return initial._restoration_fallback(ctx, dual, state, result)

    secant = LimitedMemoryReducedHessian(n=N, memory=MEMORY)

    with jax.enable_x64(True):
        return {
            "init": jax.make_jaxpr(lambda: minimiser.init(problem, x0))(),
            "step": jax.make_jaxpr(lambda: initial.step(problem))(),
            "solve": jax.make_jaxpr(solve)(),
            "restoration": jax.make_jaxpr(restoration)(),
            "append": jax.make_jaxpr(lambda s, y: secant.append(s, y))(x0, 0.5 * x0),
        }


@pytest.mark.parametrize("name", ["init", "step", "solve", "restoration", "append"])
def test_trace_respects_the_memory_budget(traces, name):
    """No ``n x n``-like array and nothing above ``4 n max(m + 1, m_c + 2)``."""
    jaxpr = traces[name]
    assert not violations(jaxpr), (name, violations(jaxpr)[:5], largest(jaxpr))
    assert int(np.prod(largest(jaxpr).shape)) <= SIZE_LIMIT


def test_audit_covers_the_elastic_and_restoration_paths(traces):
    """The traces are meaningful: control flow, the elastic system, restoration."""
    # The elastic QP has ``n + 1`` unknowns: its arrays show up in the solve.
    solve_shapes = {a.shape for a in walk(traces["solve"].jaxpr)}
    assert any(N + 1 in s for s in solve_shapes)
    # The step contains the solver loops, the restoration cond, and the
    # restoration's own CGLS (a while loop).
    for prim in ("while", "cond"):
        assert has_primitive(traces["step"], prim)
    assert has_primitive(traces["restoration"], "cond")
    # The step trace includes the restoration step (it is on by default):
    # the restoration trace is a strict part of what ``step`` already walks.
    step_where = {a.where for a in walk(traces["step"].jaxpr)}
    assert any("cond" in w for w in step_where)


def test_bounds_are_all_finite_in_the_audited_problem(problem):
    """All ``2n`` bounds are finite (the worst case for the bound handling)."""
    assert bool(jnp.all(jnp.isfinite(problem.lb)))
    assert bool(jnp.all(jnp.isfinite(problem.ub)))
    assert not bool(jnp.any(problem.null_lb)) and not bool(jnp.any(problem.null_ub))


# --- negative controls -------------------------------------------------------------


def _eye(x):
    return jnp.eye(N, dtype=x.dtype) @ x


def _outer(x):
    return jnp.outer(x, x).sum(axis=0)


def _diag_matrix(x):
    return jnp.diag(x) @ x


def _stacked_bounds(x):
    # Bounds stacked into an ``(m_c + 2n) x n`` constraint matrix.
    rows = jnp.concatenate([jnp.zeros((M_C, N), x.dtype), jnp.eye(N), -jnp.eye(N)])
    return rows @ x


def _too_many_columns(x):
    # n x (>> memory): a basis far beyond the LRH memory.
    return jnp.ones((N, 4 * (MEMORY + M_C)), x.dtype).T @ x


def _stacked_bounds_reduced(x):
    # Bounds stacked as rows of a reduced (memory + 1)-column LDP matrix.
    g = jnp.concatenate(
        [jnp.zeros((M_C, MEMORY + 1), x.dtype), jnp.ones((2 * N, MEMORY + 1), x.dtype)]
    )
    return g.sum() * x


@jax.custom_vjp
def _custom_vjp_outer(x):
    return jnp.outer(x, x).sum(axis=0)


_custom_vjp_outer.defvjp(lambda x: (_custom_vjp_outer(x), x), lambda r, g: (g,))


@pytest.mark.parametrize(
    "bad",
    [
        _eye,
        _outer,
        _diag_matrix,
        _stacked_bounds,
        _too_many_columns,
        _stacked_bounds_reduced,
        jax.checkpoint(_eye),
        jax.checkpoint(_stacked_bounds_reduced),
        _custom_vjp_outer,
    ],
)
def test_walker_flags_deliberately_bad_functions(bad: Callable):
    """The walker fails on ``eye(n)``, outer products and stacked bounds."""
    jaxpr = jax.make_jaxpr(bad)(jnp.ones(N))
    assert violations(jaxpr)


def test_walker_exempts_long_vectors_but_not_long_matrices():
    """The ``m_c + 2n`` multiplier vector is allowed; a matrix of it is not."""
    vector = jax.make_jaxpr(lambda x: jnp.zeros(M_C + 2 * N, x.dtype).sum() * x)
    assert not violations(vector(jnp.ones(N)))


def test_walker_finds_jaxprs_in_nested_containers_and_dicts():
    """Jaxprs are found through dict values and containers nested in tuples."""
    inner = jax.make_jaxpr(lambda x: x + 1.0)(jnp.ones(2))
    found = list(_nested_jaxprs({"a": ({"b": [inner]},), "c": 3}))
    assert len(found) == 1 and found[0] is inner.jaxpr


def test_walker_flags_a_violation_inside_nested_control_flow():
    """Violations are found inside ``while`` / ``cond`` / ``pjit`` bodies."""

    @jax.jit
    def inner(x):
        return jnp.eye(N, dtype=x.dtype).sum() * x

    def nested(x):
        def body(carry):
            i, v = carry
            return i + 1, jax.lax.cond(i < 1, inner, lambda z: z, v)

        return jax.lax.while_loop(lambda c: c[0] < 2, body, (0, x))[1]

    jaxpr = jax.make_jaxpr(nested)(jnp.ones(N))
    bad = violations(jaxpr)
    assert bad and any("while" in a.where and "cond" in a.where for a in bad)


def test_walker_accepts_an_allowed_function():
    """``n x r``, ``m_c x n`` and ``n``-vector arrays are within budget."""

    def fine(x):
        basis = jnp.ones((N, MEMORY), x.dtype)
        jac = jnp.ones((M_C, N), x.dtype)
        return jac @ x, basis.T @ x, jnp.cumsum(x)

    assert not violations(jax.make_jaxpr(fine)(jnp.ones(N)))
