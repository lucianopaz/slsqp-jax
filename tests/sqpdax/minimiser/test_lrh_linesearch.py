"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.lrh_linesearch`."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.logging import MemoryDiagnosticsHandler, MemoryHandler
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
    _hs_problem,
    make_bound_only_quadratic,
    make_equality_quadratic,
    make_infeasible_problem,
    make_scaled_quartic,
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


# --- LRH stats / log fields (C18) -----------------------------------------------

PARENT_STAT_KEYS = (
    "final_objective",
    "kkt_ratio",
    "qp_iterations",
    "total_qp_iterations",
    "last_qp_converged",
)
LRH_STAT_KEYS = (
    "total_nnls_iters",
    "lrh_rank",
    "lrh_sigma",
    "nnls_iters",
    "ldp_status",
    "elastic_delta",
    "n_elastic",
)
STEP_KEYS = tuple(k for k in LRH_STAT_KEYS if not k.startswith("total_"))


def _run_logged(problem, x0, **kwargs):
    """Solve with both log channels attached; return solution, lines, records."""
    text = MemoryHandler()
    diagnostics = MemoryDiagnosticsHandler()
    solver = LRHLineSearchMinimiser(**kwargs).init(
        problem,
        jnp.asarray(x0),
        {"logging": {"level": "INFO", "handler": text, "diagnostics": diagnostics}},
    )
    while not bool(solver.terminate(problem)[0]) and int(solver.step_count) < 30:
        solver = solver.step(problem)
    jax.effects_barrier()
    lines = [r.message for r in text.records if r.message.startswith("step=")]
    return solver, lines, diagnostics.select(name="minimiser", kind="step")


def _line_values(lines, key):
    return [int(line.split(f"{key}=")[1].split()[0]) for line in lines]


@pytest.mark.parametrize("memory", [2, 5], ids=lambda m: f"m{m}")
@pytest.mark.parametrize(
    ("problem_factory", "x0", "_x_star"),
    list(CONVERGENCE_CASES.values()),
    ids=list(CONVERGENCE_CASES),
)
def test_stats_report_lrh_fields_read_from_state(problem_factory, x0, _x_star, memory):
    """Final stats carry the LRH keys, the parent keys, and the state's own values."""
    problem = problem_factory()
    sol = _solve(problem, x0, max_steps=100, secant_memory=memory)
    stats = sol.state._postprocess_stats(problem, sol.result)
    for key in PARENT_STAT_KEYS + LRH_STAT_KEYS:
        assert key in stats, key
    factors = sol.state.secant.factors()
    state = sol.state.solver_state
    assert 0 <= int(stats["lrh_rank"]) <= memory
    assert int(stats["lrh_rank"]) == int(factors.rank)
    assert float(stats["lrh_sigma"]) > 0
    assert float(stats["lrh_sigma"]) == float(factors.sigma)
    assert int(stats["total_nnls_iters"]) == int(state.n_nnls_iter) >= 0
    assert int(stats["nnls_iters"]) == int(state.last_n_cg_iter)
    assert int(stats["total_nnls_iters"]) == int(stats["total_qp_cg_iterations"])
    assert int(stats["n_elastic"]) == int(state.n_elastic) >= 0
    assert stats["ldp_status"] == state.last_ldp_status
    assert float(stats["elastic_delta"]) == float(state.last_elastic_delta)


def test_elastic_case_reports_elastic_fields():
    """On the C17 infeasible case the stats show elastic solves with ``δ ≈ 1``."""
    problem = make_infeasible_problem()
    solver = LRHLineSearchMinimiser(rtol=1e-6, atol=1e-6, min_steps=5).init(
        problem, jnp.array([2.0, 0.0])
    )
    for _ in range(6):
        solver = solver.step(problem)
    stats = solver._postprocess_stats(
        problem, ACTIVE_SET_LINE_SEARCH_RESULTS.successful
    )
    assert int(stats["n_elastic"]) >= 1
    assert float(stats["elastic_delta"]) == pytest.approx(1.0, abs=1e-2)
    assert int(stats["n_elastic"]) == int(solver.solver_state.n_elastic)


def test_step_log_line_and_diagnostic_record_include_lrh_fields():
    """Both the text line and the diagnostic record carry the new fields."""
    text = MemoryHandler()
    diagnostics = MemoryDiagnosticsHandler()
    sol = _solve(
        make_equality_quadratic(),
        [0.25, 0.25],
        options={
            "logging": {
                "level": "INFO",
                "handler": text,
                "diagnostics": diagnostics,
            }
        },
    )
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    lines = [r.message for r in text.records if r.message.startswith("step=")]
    assert lines
    for key in STEP_KEYS:
        assert all(f"{key}=" in line for line in lines), key
    records = diagnostics.select(name="minimiser", kind="step")
    assert len(records) == len(lines)
    for record in records:
        for key in STEP_KEYS:
            assert key in record.values, key
        # Parent payload is still there.
        assert "merit_penalty" in record.values
        assert "qp_result" in record.values
        assert float(record.values["lrh_sigma"]) > 0
        assert float(record.values["n_elastic"]) >= 0


def test_per_step_values_match_the_state_on_the_elastic_run():
    """Per-step records and log lines track the carried solver state, step by step."""
    problem = make_infeasible_problem()
    solver, lines, records = _run_logged(
        problem, [2.0, 0.0], rtol=1e-6, atol=1e-6, min_steps=5
    )
    state = solver.solver_state
    assert len(records) == len(lines) >= 5
    nnls = [int(r.values["nnls_iters"]) for r in records]
    n_elastic = [int(r.values["n_elastic"]) for r in records]
    # Per-step NNLS counts add up to the cumulative counter.
    assert sum(nnls) == int(state.n_nnls_iter) > 0
    assert all(b > a for a, b in zip(n_elastic, n_elastic[1:]))
    assert n_elastic[-1] == int(state.n_elastic)
    delta = float(records[-1].values["elastic_delta"])
    assert delta == float(state.last_elastic_delta)
    assert delta == pytest.approx(1.0, abs=1e-2)
    # The text channel reports the very same numbers.
    assert _line_values(lines, "nnls_iters") == nnls
    assert _line_values(lines, "n_elastic") == n_elastic


def test_logged_model_is_the_one_after_the_secant_update():
    """After every step the newest record's rank and sigma equal the live secant's."""
    problem = make_scaled_quartic()
    text = MemoryHandler()
    diagnostics = MemoryDiagnosticsHandler()
    solver = LRHLineSearchMinimiser(rtol=1e-6, atol=1e-6).init(
        problem,
        jnp.array([0.5, 0.3, 0.2]),
        {"logging": {"level": "INFO", "handler": text, "diagnostics": diagnostics}},
    )
    sigmas = []
    for k in range(1, 7):
        solver = solver.step(problem)
        jax.effects_barrier()
        records = diagnostics.select(name="minimiser", kind="step")
        assert len(records) == k
        factors = solver.secant.factors()
        assert int(records[-1].values["lrh_rank"]) == int(factors.rank)
        assert float(records[-1].values["lrh_sigma"]) == float(factors.sigma)
        sigmas.append(float(factors.sigma))
    # The check is only meaningful if the model actually moves between steps.
    assert len(set(sigmas)) > 1


@pytest.mark.parametrize(
    "n, theta", [(10_000, 0.1), (10_000, 0.01)], ids=["theta0.1", "theta0.01"]
)
def test_float32_equalities_at_large_n_converge_without_elastic_mode(n, theta):
    """Two unit equalities at angle ``theta`` are not read as dependent in float32.

    ``0.5 ‖x − c‖²`` on ``[−1, 1]ⁿ`` with two equalities at angle ``theta``
    (condition ``≈ 2/theta``). The LSEI default rank cut used to grow like
    ``10·eps·n`` (``1.2e-2`` in float32 at ``n = 1e4``), declared the rows
    dependent and inconsistent, and the solve ended in elastic mode and
    failed; float64 converges in one step. The float32 solve must converge
    too, with no elastic entry. Only ``2 × n`` dense data.
    """
    with jax.enable_x64(False):
        rng = np.random.default_rng(0)
        a1 = rng.standard_normal(n)
        a1 /= np.linalg.norm(a1)
        u = rng.standard_normal(n)
        u -= (u @ a1) * a1
        u /= np.linalg.norm(u)
        a2 = np.cos(theta) * a1 + np.sin(theta) * u
        a_mat = jnp.asarray(np.stack([a1, a2]), jnp.float32)
        c = jnp.asarray(0.5 * np.cos(np.arange(n)), jnp.float32)
        b = jnp.asarray([0.1, 0.2], jnp.float32)
        problem = _hs_problem(
            fn=lambda x: 0.5 * jnp.sum((x - c) ** 2),
            grad=lambda x: x - c,
            eq=(lambda x: a_mat @ x - b, lambda x: a_mat),
            lb=[-1.0] * n,
            ub=[1.0] * n,
        )
        sol = minimise(
            problem,
            LRHLineSearchMinimiser(rtol=1e-4, atol=1e-4, secant_memory=10),
            jnp.zeros(n, jnp.float32),
            max_steps=200,
            throw=False,
            options={"minimiser": {"secant": {"kind": "lrh"}}},
        )
        assert sol.value.dtype == jnp.float32
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert int(sol.stats["n_elastic"]) == 0
        assert float(jnp.max(jnp.abs(problem.eq_fn(sol.value)))) <= 1e-4
