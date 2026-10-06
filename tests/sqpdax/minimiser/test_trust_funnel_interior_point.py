"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.trust_funnel_interior_point`."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from slsqp_jax.sqpdax.barrier import FunnelBarrierUpdate, LogBarrier
from slsqp_jax.sqpdax.logging import MemoryHandler
from slsqp_jax.sqpdax.minimiser import (
    TRUST_FUNNEL_INTERIOR_POINT_RESULTS,
    InteriorPointMinimiser,
    TrustFunnelInteriorPointMinimiser,
    TrustFunnelTerminationMetrics,
    TrustRegionInteriorPointMinimiser,
    as_optimistix_minimiser,
    minimise,
)
from slsqp_jax.sqpdax.primal import InteriorPointPrimal, Slack
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.subproblem import FunnelBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    IterationType,
    TrustFunnelSolver,
    TrustFunnelSolverState,
)
from tests.sqpdax.conftest import make_shifted_box_quadratic
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import (
    make_equality_quadratic,
    make_scaled_quartic,
    make_unconstrained_quadratic,
)

QUARTIC_X_STAR = [0.8684468143545903, 0.11959380513219049, 0.01195938051321905]


# --- problems ----------------------------------------------------------------


def make_box_quadratic() -> Problem:
    """``min ‖x − c‖²`` with one active inequality and one active bound."""
    return make_shifted_box_quadratic(n=3)[0]


def make_bound_only_quadratic() -> Problem:
    """``min ‖x‖²`` on ``[0.5, 2] × [−1, 3]``; solution ``(0.5, 0)``."""
    return make_problem(
        n=2,
        meq=0,
        mineq=0,
        lb=jnp.array([0.5, -1.0]),
        ub=jnp.array([2.0, 3.0]),
        with_curvature=True,
    )


def make_infeasible_problem() -> Problem:
    """``min ‖x‖²`` with ``x₀ ≤ 0`` and ``x₀ ≥ 1``: no feasible point."""
    n = 2

    def zero_rows(m: int):
        return lambda x, *a: jnp.zeros((m, n), x.dtype)

    return Problem(
        fn=lambda x: (jnp.sum(x**2), None),
        grad=lambda x: 2.0 * x,
        hvp=lambda x, v: 2.0 * v,
        eq_fn=lambda x: jnp.zeros((0,), x.dtype),
        eq_fn_jac=zero_rows(0),
        eq_fn_hvp=zero_rows(0),
        ineq_fn=lambda x: x[:1],
        ineq_fn_jac=lambda x: jnp.array([[1.0, 0.0]], x.dtype),
        ineq_fn_hvp=zero_rows(1),
        lb=jnp.array([1.0, -jnp.inf]),
        ub=jnp.array([jnp.inf, jnp.inf]),
        null_lb=jnp.array([False, True]),
        null_ub=jnp.array([True, True]),
        n=n,
        meq=0,
        mineq=1,
    )


# ``x0`` / ``x_star`` are plain lists: parametrize arguments are built at
# import time, before the x64 contexts below are entered.
CONVERGENCE_CASES = {
    "box": (make_box_quadratic, [0.5, 0.0, 0.5], [0.9, -1.0, 0.0]),
    "quartic": (make_scaled_quartic, [0.5, 0.3, 0.2], QUARTIC_X_STAR),
    "equality": (make_equality_quadratic, [0.25, 0.25], [0.5, 0.5]),
    "bound": (make_bound_only_quadratic, [1.0, 1.0], [0.5, 0.0]),
}


def constraint_residual(problem: Problem, primal: InteriorPointPrimal) -> jax.Array:
    """``c(x, s)`` stacked over inequalities and live bounds."""
    x, slack = primal.x, primal.slack
    return jnp.concatenate(
        [
            problem.ineq_fn(x) + slack.s,
            jnp.where(problem.null_lb, 0.0, problem.lb - x + slack.s_lb),
            jnp.where(problem.null_ub, 0.0, x - problem.ub + slack.s_ub),
        ]
    )


def funnel_minimiser(**kwargs) -> TrustFunnelInteriorPointMinimiser:
    kwargs.setdefault("atol", 1e-6)
    kwargs.setdefault("initial_mu", 0.1)
    return TrustFunnelInteriorPointMinimiser(**kwargs)


def run(problem, x0, max_steps=120, options=None, **kwargs):
    return minimise(
        problem,
        funnel_minimiser(**kwargs),
        jnp.asarray(x0),
        max_steps=max_steps,
        throw=False,
        options=options,
    )


# --- construction / validation -----------------------------------------------


def test_is_an_interior_point_minimiser_sharing_the_trust_region_scaffolding():
    """Both IP loops derive from the shared base and seed the same primal."""
    assert issubclass(TrustFunnelInteriorPointMinimiser, InteriorPointMinimiser)
    assert issubclass(TrustRegionInteriorPointMinimiser, InteriorPointMinimiser)
    problem = make_box_quadratic()
    x0 = jnp.array([0.5, 0.0, 0.5])
    funnel = funnel_minimiser().init(problem, x0)
    tr = TrustRegionInteriorPointMinimiser(initial_mu=0.1).init(problem, x0)
    assert jnp.array_equal(funnel.iterate.flatten(), tr.iterate.flatten())
    assert jnp.array_equal(funnel.dual.flatten(), tr.dual.flatten())


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"rtol": 1e-3}, ValueError, "does not provide a relative tolerance"),
        ({"barrier_update": None}, TypeError, "requires a FunnelBarrierUpdate"),
        ({"kappa_ca": 0.0}, ValueError, "kappa_ca"),
        ({"kappa_cr": -1.0}, ValueError, "kappa_cr"),
        ({"initial_radius_v": 0.0}, ValueError, "initial radii"),
        ({"initial_radius_f": -1.0}, ValueError, "initial radii"),
        ({"infeasibility_tol": 0.0}, ValueError, "infeasibility_tol"),
        ({"stall_steps": 0}, ValueError, "stall_steps"),
    ],
    ids=[
        "rtol",
        "barrier-update-kind",
        "kappa_ca",
        "kappa_cr",
        "radius_v",
        "radius_f",
        "infeasibility_tol",
        "stall_steps",
    ],
)
def test_invalid_configuration_is_rejected(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        TrustFunnelInteriorPointMinimiser(**kwargs)


def test_non_funnel_barrier_update_spec_is_rejected():
    """``options['minimiser']['barrier_update']`` must build a funnel policy."""
    problem = make_unconstrained_quadratic()
    with pytest.raises(TypeError, match="requires a FunnelBarrierUpdate"):
        funnel_minimiser().init(
            problem,
            jnp.ones(2),
            options={"minimiser": {"barrier_update": {"kind": "monotone"}}},
        )


def test_funnel_barrier_update_spec_and_subproblem_options_are_applied():
    """Kind-specs and solver options reach the policy and the funnel solver."""
    problem = make_unconstrained_quadratic()
    solver = funnel_minimiser().init(
        problem,
        jnp.ones(2),
        options={
            "minimiser": {"barrier_update": {"kind": "funnel", "gamma_mu": 0.5}},
            "subproblem": {"kappa_B": 0.7},
        },
    )
    assert isinstance(solver.barrier_update, FunnelBarrierUpdate)
    assert solver.barrier_update.gamma_mu == 0.5
    sub_solver = solver._init_subproblem(problem).solver
    assert isinstance(sub_solver, TrustFunnelSolver)
    assert sub_solver.kappa_B == 0.7


def test_validate_options_warns_on_unknown_keys():
    problem = make_unconstrained_quadratic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        funnel_minimiser().init(
            problem,
            jnp.ones(2),
            options={
                "minimiser": {"not_a_field": 1.0},
                "subproblem": {"not_a_solver_field": 0.0},
            },
        )
    messages = " ".join(str(w.message) for w in caught)
    assert "unknown minimiser option 'not_a_field'" in messages
    assert "unknown subproblem option 'not_a_solver_field'" in messages


# --- init invariants -----------------------------------------------------------


@pytest.mark.parametrize(
    ("make_problem_fn", "x0"),
    [
        (make_box_quadratic, [0.5, 0.0, 0.5]),
        (make_box_quadratic, [3.0, -5.0, 0.0]),
        (make_scaled_quartic, [0.5, 0.3, 0.2]),
        (make_bound_only_quadratic, [1.0, 1.0]),
        (make_unconstrained_quadratic, [1.0, 1.0]),
    ],
    ids=["box-interior", "box-far", "quartic", "bound", "unconstrained"],
)
def test_init_seeds_funnel_state_and_invariant(make_problem_fn, x0):
    """``init`` gives ``s > 0``, ``c(x, s) ≥ 0``, ``v₀ ≤ v_max₀`` and ``μ₀`` tolerances."""
    problem = make_problem_fn()
    solver = funnel_minimiser(kappa_ca=0.5, kappa_cr=3.0).init(problem, jnp.asarray(x0))
    assert isinstance(solver.iterate, InteriorPointPrimal)
    assert isinstance(solver.barrier, LogBarrier)
    assert float(solver.barrier.weight) == pytest.approx(0.1)
    assert jnp.all(solver.iterate.slack.flatten() > 0)
    assert jnp.all(constraint_residual(problem, solver.iterate) >= 0)
    state = solver.solver_state
    assert isinstance(state, TrustFunnelSolverState)
    v0 = float(jnp.linalg.norm(constraint_residual(problem, solver.iterate)))
    assert float(state.v_max) == pytest.approx(max(0.5, 3.0 * v0), rel=1e-6)
    assert v0 <= float(state.v_max)
    assert float(state.radius_v) == pytest.approx(1.0)
    assert float(state.radius_f) == pytest.approx(1.0)
    update = solver.barrier_update
    assert float(state.eps_pi) == pytest.approx(float(update.eps_pi(0.1)), rel=1e-6)
    assert float(state.eps_v) == pytest.approx(float(update.eps_v(0.1)), rel=1e-6)
    assert not bool(state.sf_flag)
    assert int(solver.consecutive_y_iterations) == 0
    assert int(solver.consecutive_model_failures) == 0


def test_subproblem_uses_mu_dependent_fraction_to_boundary_constants():
    """``κ_fbn(μ)`` / ``κ_fbt(μ)`` of the policy are frozen into the model."""
    problem = make_box_quadratic()
    solver = funnel_minimiser(initial_mu=0.01).init(problem, jnp.array([0.5, 0.0, 0.5]))
    sub = solver._init_subproblem(problem).subproblem
    assert isinstance(sub, FunnelBarrierSubProblem)
    assert float(sub.kappa_fbn) == pytest.approx(
        float(solver.barrier_update.kappa_fbn(jnp.asarray(0.01)))
    )
    assert float(sub.kappa_fbt) == pytest.approx(
        float(solver.barrier_update.kappa_fbt(jnp.asarray(0.01)))
    )


# --- single steps --------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_problem_fn", "x0"),
    [
        (make_box_quadratic, [0.5, 0.0, 0.5]),
        (make_box_quadratic, [3.0, -5.0, 0.0]),
        (make_scaled_quartic, [0.5, 0.3, 0.2]),
        (make_bound_only_quadratic, [1.0, 1.0]),
    ],
    ids=["box-interior", "box-far", "quartic", "bound"],
)
def test_steps_preserve_funnel_invariants(make_problem_fn, x0):
    """Every committed iterate keeps ``s > 0``, ``c(x, s) ≥ 0`` and ``v ≤ v_max``."""
    problem = make_problem_fn()
    solver = funnel_minimiser().init(problem, jnp.asarray(x0))
    for _ in range(4):
        solver = solver.step(problem)
        iterate = solver.iterate
        residual = constraint_residual(problem, iterate)
        assert jnp.all(jnp.isfinite(iterate.flatten()))
        assert jnp.all(iterate.slack.flatten() > 0)
        assert jnp.all(residual >= -1e-6)
        assert float(jnp.linalg.norm(residual)) <= float(solver.solver_state.v_max) * (
            1 + 1e-6
        )
        # ``y`` are equality multipliers of ``c(x, s) = 0`` (sign-free in the
        # paper), so only finiteness is required of them.
        assert jnp.all(jnp.isfinite(solver.dual.flatten()))


def test_slack_reset_only_raises_slacks_and_lowers_f_and_v():
    """(3.26)/(3.33): negative residual rows get ``sᵢ ← −cᵢ(x)``; others stay."""
    problem = make_box_quadratic()
    solver = funnel_minimiser().init(problem, jnp.array([0.5, 0.0, 0.5]))
    lag_module = solver._lagrangian_module(problem)
    iterate = solver.iterate
    # Shrink one inequality slack and one bound slack below the feasible value.
    bad = InteriorPointPrimal(
        x=iterate.x,
        slack=Slack(
            s=iterate.slack.s * 0.1,
            s_lb=iterate.slack.s_lb.at[1].multiply(0.1),
            s_ub=iterate.slack.s_ub,
        ),
    )
    before = constraint_residual(problem, bad)
    assert jnp.sum(before < 0) == 2
    reset, moved = TrustFunnelInteriorPointMinimiser._reset_slacks(
        bad, lag_module(bad, solver.dual)
    )
    after = constraint_residual(problem, reset)
    assert bool(moved)
    assert jnp.array_equal(reset.x, bad.x)
    assert jnp.all(reset.slack.flatten() >= bad.slack.flatten())
    assert jnp.all(after >= -1e-6)
    assert jnp.allclose(after[before < 0], 0.0, atol=1e-6)
    assert jnp.array_equal(after[before >= 0], before[before >= 0])
    assert float(jnp.linalg.norm(after)) < float(jnp.linalg.norm(before))
    f_bad = (
        lag_module(bad, solver.dual).fn_val
        + lag_module(bad, solver.dual).barrier.fn_val
    )
    f_ok = (
        lag_module(reset, solver.dual).fn_val
        + lag_module(reset, solver.dual).barrier.fn_val
    )
    assert float(f_ok) < float(f_bad)
    # Already feasible slacks are a fixed point.
    again, moved_again = TrustFunnelInteriorPointMinimiser._reset_slacks(
        reset, lag_module(reset, solver.dual)
    )
    assert not bool(moved_again)
    assert jnp.array_equal(again.flatten(), reset.flatten())


def test_y_iteration_commits_multipliers_without_moving():
    """At the exact barrier solution the step is a y-iteration that updates ``y``."""
    problem = make_equality_quadratic()
    x_star = jnp.array([0.5, 0.5])
    solver = funnel_minimiser().init(problem, x_star)
    assert float(solver.dual.eq_multipliers[0]) == 0.0
    solver = solver.step(problem)
    assert bool(solver.solver_state.iteration_type == IterationType.y_iteration)
    assert jnp.allclose(solver.iterate.x, x_star)
    assert float(solver.dual.eq_multipliers[0]) == pytest.approx(-1.0, abs=1e-4)
    assert int(solver.consecutive_y_iterations) == 0  # μ was reduced


def test_mu_reduction_restarts_the_funnel():
    """Solving BSP(μ) reduces ``μ``, re-seeds radii / ``v_max`` and the tolerances."""
    problem = make_equality_quadratic()
    solver = funnel_minimiser(
        initial_radius_v=0.3, initial_radius_f=0.7, kappa_ca=5.0
    ).init(problem, jnp.array([0.5, 0.5]))
    solver = eqx.tree_at(
        lambda m: (m.solver_state.radius_v, m.solver_state.radius_f),
        solver,
        (jnp.asarray(9.0), jnp.asarray(9.0)),
    )
    solver = solver.step(problem)
    assert bool(solver.barrier_updated)
    update = solver.barrier_update
    mu = float(solver.barrier.weight)
    assert mu == pytest.approx(0.1 * update.gamma_mu, rel=1e-6)
    state = solver.solver_state
    assert float(state.radius_v) == pytest.approx(0.3, rel=1e-6)
    assert float(state.radius_f) == pytest.approx(0.7, rel=1e-6)
    assert float(state.v_max) == pytest.approx(5.0, rel=1e-6)
    assert float(state.eps_pi) == pytest.approx(float(update.eps_pi(mu)), rel=1e-5)
    assert float(state.eps_v) == pytest.approx(float(update.eps_v(mu)), rel=1e-5)
    assert not bool(state.sf_flag)
    assert float(state.pi_f_prev) == 0.0


# --- termination ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("y_streak", "violation", "expected"),
    [
        (0, 0.0, None),
        (5, 0.0, "stationarity_stall"),
        (5, 10.0, "infeasible_stationary_point"),
        (4, 10.0, None),
    ],
    ids=["no-streak", "stall-feasible", "stall-infeasible", "short-streak"],
)
def test_termination_classifies_y_iteration_streaks(y_streak, violation, expected):
    """A fixed point of Algorithm 2 stops the run, labelled by feasibility."""
    problem = make_box_quadratic()
    x0 = jnp.array([0.5, 0.0, 0.5])
    solver = funnel_minimiser(min_steps=0, stall_steps=5).init(problem, x0)
    # Interior iterate with ``c(x, s) = 0``, then optionally pushed far out of
    # the feasible region through the inequality slack.
    feasible = Slack(
        s=-problem.ineq_fn(x0) + violation,
        s_lb=x0 - problem.lb,
        s_ub=problem.ub - x0,
    )
    solver = eqx.tree_at(
        lambda m: (m.consecutive_y_iterations, m.iterate.slack),
        solver,
        (jnp.asarray(y_streak, jnp.int32), feasible),
    )
    done, result = solver.terminate(problem)
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    assert isinstance(metrics, TrustFunnelTerminationMetrics)
    assert bool(metrics.stalled) == (y_streak >= 5)
    if expected is None:
        assert not bool(done)
    else:
        assert bool(done)
        assert result == getattr(TRUST_FUNNEL_INTERIOR_POINT_RESULTS, expected)


def test_infeasible_problem_terminates_as_infeasible_stationary():
    """Incompatible constraints end at Step 8 rather than exhausting the budget."""
    problem = make_infeasible_problem()
    sol = run(problem, [0.5, 0.3], max_steps=80)
    assert sol.result == TRUST_FUNNEL_INTERIOR_POINT_RESULTS.infeasible_stationary_point
    assert int(sol.stats["num_steps"]) < 80
    # The violation's stationary point splits the two incompatible rows evenly.
    assert float(sol.value[0]) == pytest.approx(0.5, abs=1e-3)
    assert sol.state.result_adapter.to_optimistix(sol.result) == (
        optx.RESULTS.nonlinear_divergence
    )


def test_result_adapter_maps_native_codes():
    adapter = TrustFunnelInteriorPointMinimiser().result_adapter
    R = TRUST_FUNNEL_INTERIOR_POINT_RESULTS
    assert adapter.to_optimistix(R.successful) == optx.RESULTS.successful
    assert adapter.to_optimistix(R.running) == optx.RESULTS.successful
    assert adapter.to_optimistix(R.max_steps_reached) == (
        optx.RESULTS.nonlinear_max_steps_reached
    )
    assert adapter.to_optimistix(R.nonfinite) == optx.RESULTS.nonfinite
    assert adapter.to_optimistix(R.subproblem_nonfinite) == optx.RESULTS.nonfinite_input
    assert adapter.to_optimistix(R.multiplier_solve_failure) == optx.RESULTS.singular
    assert adapter.to_optimistix(R.stationarity_stall) == (
        optx.RESULTS.nonlinear_divergence
    )


# --- convergence ---------------------------------------------------------------


@pytest.mark.parametrize("primal_dual", [True, False], ids=["primal-dual", "primal"])
@pytest.mark.parametrize("case", list(CONVERGENCE_CASES), ids=list(CONVERGENCE_CASES))
def test_minimise_converges_to_known_kkt_points(case, primal_dual):
    """The driver reaches the analytic minimiser with exact curvature."""
    make_problem_fn, x0, x_star = CONVERGENCE_CASES[case]
    # f ≈ 1 at the solutions, so the actual-over-predicted ratios need x64 to
    # resolve reductions at the 1e-6 KKT level.
    with jax.enable_x64(True):
        sol = run(make_problem_fn(), x0, primal_dual=primal_dual)
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert jnp.allclose(sol.value, jnp.asarray(x_star), atol=1e-5)
        metrics = sol.state.termination_metrics(
            sol.state._optimisation_context(make_problem_fn())
        )
        assert float(metrics.optimality_residual) <= 1e-6


@pytest.mark.parametrize(
    ("make_problem_fn", "case"),
    [
        (make_box_quadratic, "box"),
        (make_scaled_quartic, "quartic"),
        (lambda: make_scaled_quartic(with_curvature=False), "quartic"),
    ],
    ids=["box", "quartic-with-hvp", "quartic-no-hvp"],
)
def test_minimise_converges_with_secant_model(make_problem_fn, case):
    """The L-BFGS model drives the funnel to the same KKT points."""
    _, x0, x_star = CONVERGENCE_CASES[case]
    with jax.enable_x64(True):
        sol = run(make_problem_fn(), x0, curvature="secant")
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert jnp.allclose(sol.value, jnp.asarray(x_star), atol=1e-5)
        assert "secant_n_appends" in sol.stats


@pytest.mark.parametrize(
    ("make_problem_fn", "x0"),
    [
        (make_equality_quadratic, [0.25, 0.25]),
        (make_unconstrained_quadratic, [1.0, 1.0]),
    ],
    ids=["equality", "unconstrained"],
)
def test_agrees_with_trust_region_interior_point(make_problem_fn, x0):
    """Both interior-point loops land on the same point.

    Restricted to the problems the trust-region loop currently solves to
    ``atol = 1e-6``; the funnel's bound-active cases are checked against the
    analytic minimisers above.
    """
    with jax.enable_x64(True):
        funnel = run(make_problem_fn(), x0)
        tr = minimise(
            make_problem_fn(),
            TrustRegionInteriorPointMinimiser(atol=1e-6, initial_mu=0.1),
            jnp.asarray(x0),
            max_steps=200,
            throw=False,
        )
        assert bool(funnel.state.result_adapter.is_successful(funnel.result))
        assert bool(tr.state.result_adapter.is_successful(tr.result))
        assert jnp.allclose(funnel.value, tr.value, atol=1e-4)


def test_float32_converges_at_loose_tolerance():
    """In float32 the ratio tests resolve ``f`` only to ~1e-7, so ``atol`` is looser.

    With ``atol = 1e-3`` the run stops once ``μ ≈ 1e-3``, i.e. within the
    barrier's ``O(μ)`` offset of the constrained minimiser.
    """
    with jax.enable_x64(False):
        sol = run(make_box_quadratic(), [0.5, 0.0, 0.5], atol=1e-3)
        assert sol.value.dtype == jnp.float32
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert jnp.allclose(sol.value, jnp.asarray([0.9, -1.0, 0.0]), atol=1e-2)


def test_unconstrained_and_equality_converge_in_float32():
    for make_problem_fn, x0, x_star in (
        CONVERGENCE_CASES["equality"],
        (make_unconstrained_quadratic, [1.0, 1.0], [0.0, 0.0]),
    ):
        sol = run(make_problem_fn(), x0, max_steps=40)
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert jnp.allclose(sol.value, jnp.asarray(x_star), atol=1e-5)


# --- logging / integration -------------------------------------------------------


def test_step_logging_reports_funnel_columns_and_stalls():
    """INFO rows carry the funnel measures; a y-streak logs a WARNING.

    In float32 the infeasible problem ends through the fixed-point detector
    (the slacks cannot shrink far enough for ``χᵛ`` to vanish), which is the
    route that emits the streak warning; float64 reaches Step 8 directly.
    """
    problem = make_infeasible_problem()
    handler = MemoryHandler()
    with jax.enable_x64(False):
        sol = run(
            problem,
            [0.5, 0.3],
            max_steps=80,
            options={"logging": {"level": "INFO", "handler": handler}},
        )
        metrics = sol.state.termination_metrics(
            sol.state._optimisation_context(problem)
        )
    assert sol.result == TRUST_FUNNEL_INTERIOR_POINT_RESULTS.infeasible_stationary_point
    assert bool(metrics.stalled)
    steps = [r.message for r in handler.records if r.message.startswith("step=")]
    assert len(steps) == int(sol.stats["num_steps"])
    for column in (
        "pi_f=",
        "v=",
        "v_max=",
        "radius_v=",
        "radius_f=",
        "type=",
        "y_streak=",
    ):
        assert column in steps[-1]
    warnings_ = [r.message for r in handler.records if r.levelno >= 30]
    assert any("consecutive y-iterations" in m for m in warnings_)


def test_optimistix_driver_compatibility():
    problem = make_unconstrained_quadratic()
    adapter = as_optimistix_minimiser(funnel_minimiser(), problem)
    assert isinstance(adapter, optx.AbstractMinimiser)

    def fn(y, args):
        return jnp.sum(y**2), None

    sol = optx.minimise(
        fn, adapter, jnp.ones(2), max_steps=40, throw=False, has_aux=True
    )
    assert sol.result == optx.RESULTS.successful
    assert sol.stats["sqpdax_result"] == TRUST_FUNNEL_INTERIOR_POINT_RESULTS.successful
    assert jnp.allclose(sol.value, 0.0, atol=1e-4)
