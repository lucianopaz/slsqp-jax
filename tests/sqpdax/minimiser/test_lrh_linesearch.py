"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.lrh_linesearch`."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.logging import MemoryHandler
from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    LRHLineSearchMinimiser,
    as_optimistix_minimiser,
    minimise,
)
from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
)
from slsqp_jax.sqpdax.secant import LimitedMemoryReducedHessian
from slsqp_jax.sqpdax.subproblem import ActiveSetSubProblem
from slsqp_jax.sqpdax.subproblem.solver import LRHDirectQPSolver, LRHQPSolverState
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import (
    CONVERGENCE_CASES,
    make_bound_only_quadratic,
    make_equality_quadratic,
    make_infeasible_problem,
    make_unconstrained_quadratic,
)


@pytest.fixture(autouse=True)
def x64():
    """Run every test in float64 (the tolerances below assume it)."""
    with jax.enable_x64(True):
        yield


def _solve(problem, x0, *, max_steps: int = 80, options=None, **kwargs):
    kwargs.setdefault("rtol", 1e-6)
    kwargs.setdefault("atol", 1e-6)
    return minimise(
        problem,
        LRHLineSearchMinimiser(**kwargs),
        jnp.asarray(x0),
        max_steps=max_steps,
        throw=False,
        options=options,
    )


# --- construction and configuration ---------------------------------------------


def test_exact_curvature_is_rejected():
    """The LRH model is always a secant: ``curvature='exact'`` raises."""
    with pytest.raises(ValueError, match="curvature='exact'"):
        LRHLineSearchMinimiser(curvature="exact")


def test_defaults_and_inheritance():
    """``qp_max_iter`` defaults to 100 and the parent loop is inherited as is."""
    assert LRHLineSearchMinimiser().qp_max_iter == 100
    assert issubclass(LRHLineSearchMinimiser, ActiveSetLineSearchMinimiser)
    # None of these are overridden (the restoration fallback, decaying penalty,
    # plain subproblem and dynamics all come from the parent).
    for name in (
        "step",
        "_init_subproblem",
        "_retry_subproblem",
        "_restoration_fallback",
        "_restoration_step",
        "_next_merit_penalty",
        "_advance_dynamics",
        "termination_flags",
        "termination_metrics",
        "_make_working_set_policy",
    ):
        assert name not in vars(LRHLineSearchMinimiser), name


@pytest.mark.parametrize("with_curvature", [True, False], ids=["exact-hvp", "no-hvp"])
@pytest.mark.parametrize("memory", [3, 10])
@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize(
    ("qp_tol", "atol", "expected_tol"),
    [(None, 1e-4, 1e-4), (1e-7, 1e-4, 1e-7)],
    ids=["routed-from-atol", "explicit"],
)
def test_init_builds_lrh_model_state_and_direct_solver(
    with_curvature, memory, qp_warm_start, qp_tol, atol, expected_tol
):
    """Whatever the problem offers, init seeds an LRH secant, state and direct solver."""
    problem = make_problem(with_curvature=with_curvature)
    solver = LRHLineSearchMinimiser(
        secant_memory=memory, qp_warm_start=qp_warm_start, qp_tol=qp_tol, atol=atol
    ).init(problem, jnp.ones(2))
    assert isinstance(solver.secant, LimitedMemoryReducedHessian)
    assert solver.secant.memory == memory
    assert solver.secant.n == problem.n
    state = solver.solver_state
    assert type(state) is LRHQPSolverState
    assert state.dual.eq_multipliers.shape == (problem.meq,)
    assert state.active_set.active_lb.shape == (problem.n,)
    assert not bool(jnp.any(state.active_set.active_lb))

    ctx = solver._init_subproblem(problem)
    assert type(ctx.solver) is LRHDirectQPSolver
    assert type(ctx.subproblem) is ActiveSetSubProblem
    assert ctx.solver.max_iter == 100
    assert ctx.solver.tol == expected_tol
    assert ctx.solver.warm_start is qp_warm_start
    assert not hasattr(ctx.solver, "working_set_policy")
    # The model that the QP sees is the LRH secant, not exact curvature.
    assert ctx.subproblem.lagrangian.secant is not None


@pytest.mark.parametrize(
    ("spec", "memory", "sigma_rule"),
    [
        ({"kind": "lrh"}, 7, "sy_over_ss"),
        ({"kind": "lrh", "memory": 4}, 4, "sy_over_ss"),
        ({"kind": "lrh", "sigma_rule": "yy_over_sy"}, 7, "yy_over_sy"),
    ],
    ids=["default-memory", "spec-memory", "sigma-rule"],
)
def test_reduced_hessian_secant_spec_is_honoured(spec, memory, sigma_rule):
    """A ``minimiser.secant`` spec producing an LRH secant replaces the default."""
    solver = LRHLineSearchMinimiser(secant_memory=7).init(
        make_equality_quadratic(),
        jnp.zeros(2),
        {"minimiser": {"secant": spec}},
    )
    assert isinstance(solver.secant, LimitedMemoryReducedHessian)
    assert solver.secant.memory == memory
    assert solver.secant.sigma_rule == sigma_rule


def test_non_reduced_hessian_secant_spec_raises_type_error():
    """An L-BFGS spec cannot feed the direct solver."""
    with pytest.raises(TypeError, match="ReducedHessianSecant"):
        LRHLineSearchMinimiser().init(
            make_equality_quadratic(),
            jnp.zeros(2),
            {"minimiser": {"secant": {"kind": "lbfgs"}}},
        )


def make_placeholder_bounds_problem():
    """Unconstrained quadratic whose first variable has *null* bounds that hold
    finite junk (``lb = 5``, ``ub = -5``): only the null masks make them inert.
    The second variable is genuinely boxed in ``[-2, 3]``.
    """
    problem = make_unconstrained_quadratic()
    return eqx.tree_at(
        lambda p: (p.lb, p.ub, p.null_lb, p.null_ub),
        problem,
        (
            jnp.array([5.0, -2.0]),
            jnp.array([-5.0, 3.0]),
            jnp.array([True, False]),
            jnp.array([True, False]),
        ),
    )


@pytest.mark.parametrize(
    ("problem_factory", "x0", "expected"),
    [
        (make_bound_only_quadratic, [5.0, -5.0], [2.0, -1.0]),
        (make_bound_only_quadratic, [1.0, 1.0], [1.0, 1.0]),
        # Only the finite (non-null) bound of ``make_infeasible_problem`` applies.
        (make_infeasible_problem, [0.0, 7.0], [1.0, 7.0]),
        (make_unconstrained_quadratic, [3.0, -4.0], [3.0, -4.0]),
        # Finite placeholder values under a null mask must be ignored.
        (make_placeholder_bounds_problem, [0.0, -9.0], [0.0, -2.0]),
        (make_placeholder_bounds_problem, [9.0, 9.0], [9.0, 3.0]),
    ],
    ids=[
        "clipped",
        "inside",
        "null-bounds",
        "unbounded",
        "placeholder-low",
        "placeholder-high",
    ],
)
def test_init_projects_x0_onto_the_box(problem_factory, x0, expected):
    """The starting point is clipped to the (non-null) bounds."""
    solver = LRHLineSearchMinimiser().init(problem_factory(), jnp.asarray(x0))
    assert jnp.allclose(solver.iterate.x, jnp.asarray(expected))


@pytest.mark.parametrize(
    ("kwargs", "options", "n_expected"),
    [
        ({"qp_single_exchange": True}, None, 1),
        ({}, {"minimiser": {"qp_single_exchange": True}}, 1),
        ({}, {"subproblem": {"working_set_policy": {"cap_working_set": False}}}, 1),
        (
            {"qp_single_exchange": True},
            {"subproblem": {"working_set_policy": {"cap_working_set": False}}},
            2,
        ),
    ],
    ids=["field", "minimiser-option", "subproblem-option", "both"],
)
def test_working_set_options_are_ignored_with_a_warning(kwargs, options, n_expected):
    """``qp_single_exchange`` / ``working_set_policy`` warn once each and change nothing."""
    problem = make_equality_quadratic()
    with pytest.warns(UserWarning) as record:
        solver = LRHLineSearchMinimiser(**kwargs).init(problem, jnp.zeros(2), options)
    ignored = [w for w in record if "is ignored" in str(w.message)]
    assert len(ignored) == n_expected
    # The warning is the only diagnostic: no duplicate "unknown option" report.
    assert len(record) == n_expected
    ctx = solver._init_subproblem(problem)
    assert type(ctx.solver) is LRHDirectQPSolver


def test_valid_options_do_not_warn_and_reach_the_solver():
    """``options['subproblem']`` is applied to the direct solver; unknown keys still warn."""
    problem = make_equality_quadratic()
    options = {"subproblem": {"elastic": False, "elastic_weight": 50.0, "max_iter": 7}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver = LRHLineSearchMinimiser().init(problem, jnp.zeros(2), options)
    qp = solver._init_subproblem(problem).solver
    assert qp.elastic is False
    assert qp.elastic_weight == 50.0
    assert qp.max_iter == 7

    with pytest.warns(UserWarning, match="unknown subproblem option 'bogus'"):
        LRHLineSearchMinimiser().init(
            problem, jnp.zeros(2), {"subproblem": {"bogus": 1}}
        )


def test_logger_is_injected_into_the_direct_solver():
    """The ``minimiser.subproblem`` child logger reaches the QP solver."""
    handler = MemoryHandler()
    sol = _solve(
        make_equality_quadratic(),
        [0.25, 0.25],
        options={"logging": {"level": "DEBUG", "handler": handler}},
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    names = {r.name for r in handler.records}
    assert any(name.startswith("minimiser.subproblem") for name in names), names


# --- solving ---------------------------------------------------------------------


@pytest.mark.parametrize("secant_memory", [2, 5], ids=lambda m: f"m{m}")
@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize(
    ("problem_factory", "x0", "x_star"),
    list(CONVERGENCE_CASES.values()),
    ids=list(CONVERGENCE_CASES),
)
def test_minimise_solves_the_shared_problems(
    problem_factory, x0, x_star, qp_warm_start, secant_memory
):
    """Every conftest problem converges to its known solution on the LRH model."""
    sol = _solve(
        problem_factory(),
        x0,
        max_steps=100,
        qp_warm_start=qp_warm_start,
        secant_memory=secant_memory,
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result)), sol.result
    assert jnp.allclose(sol.value, jnp.asarray(x_star), atol=1e-4)
    assert type(sol.state.solver_state) is LRHQPSolverState
    assert isinstance(sol.state.secant, LimitedMemoryReducedHessian)
    assert bool(jnp.all(jnp.isfinite(sol.value)))


def test_minimise_matches_the_lbfgs_active_set_minimiser():
    """On a problem with exact HVPs the LRH run agrees with the parent's solution."""
    problem = make_equality_quadratic()
    kwargs = dict(rtol=1e-6, atol=1e-6)
    ref = minimise(
        problem,
        ActiveSetLineSearchMinimiser(**kwargs),
        jnp.zeros(2),
        max_steps=60,
        throw=False,
    )
    sol = _solve(problem, [0.0, 0.0])
    assert jnp.allclose(sol.value, ref.value, atol=1e-5)


def test_runs_through_as_optimistix_minimiser():
    """The adapter drives the LRH minimiser from ``optimistix.minimise``."""
    problem = make_equality_quadratic()
    adapter = as_optimistix_minimiser(
        LRHLineSearchMinimiser(rtol=1e-6, atol=1e-6), problem
    )
    assert isinstance(adapter, optx.AbstractMinimiser)

    def fn(y, args):
        return jnp.sum(y**2), None

    sol = optx.minimise(
        fn, adapter, jnp.zeros(2), max_steps=60, throw=False, has_aux=True
    )
    assert sol.result == optx.RESULTS.successful
    assert sol.stats["sqpdax_result"] == ACTIVE_SET_LINE_SEARCH_RESULTS.successful
    assert jnp.allclose(sol.value, jnp.array([0.5, 0.5]), atol=1e-4)


@pytest.mark.parametrize("qp_warm_start", [False, True], ids=["cold", "warm"])
def test_whole_solve_compiles_once(qp_warm_start):
    """A second call with another starting point reuses the single trace."""
    problem = make_equality_quadratic()
    minimiser = LRHLineSearchMinimiser(
        rtol=1e-6, atol=1e-6, min_steps=1, qp_warm_start=qp_warm_start
    )
    traces: list[None] = []

    @eqx.filter_jit
    def run(x0):
        traces.append(None)
        return minimise(problem, minimiser, x0, max_steps=40, throw=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = run(jnp.array([0.0, 0.0]))
        second = run(jnp.array([2.0, -1.0]))

    assert len(traces) == 1
    for sol in (first, second):
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert jnp.allclose(sol.value, jnp.array([0.5, 0.5]), atol=1e-4)


# --- elastic mode (C16 review obligations) --------------------------------------


@pytest.mark.parametrize("x0", [[1.0, 0.0], [2.0, 0.0]], ids=["on-bound", "inside"])
def test_locally_infeasible_problem_ends_infeasible_with_a_bounded_penalty(x0):
    """Elastic QPs with ``δ ≈ 1`` never read as converged and keep the penalty bounded.

    ``x₀ ≤ 0`` and ``x₀ ≥ 1`` have no common point, so every linearisation is
    inconsistent and the QP is solved in elastic mode with ``δ = 1`` (a zero
    step). The relaxed QP succeeds, so after ``zero_step_patience`` zero steps
    the parent's ``qp_optimal`` flag is armed at an infeasible iterate (a run
    with ``min_steps = 5`` keeps stepping long enough to see it); only the
    ``feasible`` factor of the parent's QP-KKT shortcut keeps that from being
    reported as convergence. The relaxed-QP multipliers feed the merit-penalty
    update; they are of the order of the elastic penalty
    (``elastic_weight · τ² · δ``) and the penalty never grows after the first
    step.
    """
    problem = make_infeasible_problem()
    solver = LRHLineSearchMinimiser(rtol=1e-6, atol=1e-6, min_steps=5).init(
        problem, jnp.asarray(x0)
    )
    penalties, deltas, armed_while_infeasible = [], [], []
    for _ in range(30):
        done, result = solver.terminate(problem)
        if bool(done):
            break
        solver = solver.step(problem)
        penalties.append(float(solver.merit_penalty))
        deltas.append(float(solver.solver_state.last_elastic_delta))
        infeasible = float(jnp.max(problem.ineq_fn(solver.iterate.x))) > 1e-6
        armed_while_infeasible.append(bool(solver.qp_optimal) and infeasible)
    else:
        pytest.fail("the minimiser did not terminate on an infeasible problem")

    assert len(penalties) >= 5
    assert int(solver.solver_state.n_elastic) >= 1
    assert max(deltas) == pytest.approx(1.0, abs=1e-2)
    # The QP-optimality flag was armed at an infeasible iterate ...
    assert any(armed_while_infeasible)
    # ... yet the outcome is never the QP-KKT "converged", even though the
    # relaxed QP succeeded.
    assert not bool(solver.result_adapter.is_successful(result))
    assert result in (
        ACTIVE_SET_LINE_SEARCH_RESULTS.infeasible_stationary,
        ACTIVE_SET_LINE_SEARCH_RESULTS.infeasible,
    )
    # Bounded: finite, below the elastic multiplier scale (margin *
    # elastic_weight ~ 1.1e3) by an order of magnitude, and non-increasing
    # once the first relaxed multipliers have been consumed.
    assert all(jnp.isfinite(jnp.asarray(penalties)))
    assert max(penalties) <= 1e4
    assert all(b <= a * (1.0 + 1e-12) for a, b in zip(penalties[1:], penalties[2:]))
    assert bool(jnp.all(jnp.isfinite(solver.iterate.x)))
    assert bool(jnp.all(jnp.isfinite(solver.dual.ineq_multipliers)))
