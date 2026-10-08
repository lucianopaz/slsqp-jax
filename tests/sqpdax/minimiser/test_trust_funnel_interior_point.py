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
from slsqp_jax.sqpdax.subproblem import FunnelBarrierSubProblem
from slsqp_jax.sqpdax.subproblem.solver import (
    IterationType,
    TrustFunnelSolver,
    TrustFunnelSolverState,
)

from .conftest import (
    CONVERGENCE_CASES,
    constraint_residual,
    funnel_minimiser,
    make_bound_only_quadratic,
    make_box_quadratic,
    make_equality_quadratic,
    make_infeasible_problem,
    make_scaled_quartic,
    make_unconstrained_quadratic,
    make_zero_jacobian_infeasible_problem,
)


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


def check_funnel_invariants(problem, iterate, dual, solver_state) -> None:
    """``s > 0``, ``c(x, s) ≥ 0`` and ``v ≤ v_max`` at a committed iterate."""
    residual = constraint_residual(problem, iterate)
    assert jnp.all(jnp.isfinite(iterate.flatten()))
    assert jnp.all(iterate.slack.flatten() > 0)
    assert jnp.all(residual >= -1e-6)
    assert float(jnp.linalg.norm(residual)) <= float(solver_state.v_max) * (1 + 1e-6)
    # ``y`` are equality multipliers of ``c(x, s) = 0`` (sign-free in the
    # paper), so only finiteness is required of them.
    assert jnp.all(jnp.isfinite(dual.flatten()))


@pytest.mark.parametrize("case", list(CONVERGENCE_CASES), ids=list(CONVERGENCE_CASES))
def test_steps_preserve_funnel_invariants(funnel_run, case):
    """Every committed iterate of a full run keeps the funnel invariants
    (``box-far`` starts outside the box, so slacks are reset on the way)."""
    run_ = funnel_run(case)
    assert run_.n_steps > 0
    with jax.enable_x64(True):
        for record in run_.steps:
            payload = record.values
            check_funnel_invariants(
                run_.problem, payload["x"], payload["dual"], payload["solver_state"]
            )


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


def check_converged(sol, x_star, metrics=None) -> None:
    """Successful result at the analytic minimiser (and KKT residual, if given)."""
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert jnp.allclose(sol.value, jnp.asarray(x_star), atol=1e-5)
    if metrics is not None:
        assert float(metrics.optimality_residual) <= 1e-6


@pytest.mark.parametrize("case", list(CONVERGENCE_CASES), ids=list(CONVERGENCE_CASES))
def test_minimise_converges_to_known_kkt_points(funnel_run, case):
    """The driver reaches the analytic minimiser with exact curvature.

    ``f ≈ 1`` at the solutions, so the actual-over-predicted ratios need x64
    to resolve reductions at the 1e-6 KKT level: the shared runs are float64.
    """
    run_ = funnel_run(case)
    with jax.enable_x64(True):
        check_converged(run_.sol, run_.x_star, run_.metrics)
    assert run_.sol.value.dtype == jnp.float64


@pytest.mark.parametrize(
    "case",
    [
        case if case in ("box", "bound") else pytest.param(case, marks=pytest.mark.slow)
        for case in CONVERGENCE_CASES
    ],
    ids=list(CONVERGENCE_CASES),
)
def test_primal_barrier_model_converges_to_known_kkt_points(case):
    """The primal (non primal-dual) barrier Hessian reaches the same points.

    The two models differ only in the barrier Hessian, which is exercised
    by the bound-active problems; the remaining cases are slow duplicates.
    """
    make_problem_fn, x0, x_star = CONVERGENCE_CASES[case]
    with jax.enable_x64(True):
        problem = make_problem_fn()
        sol = run(problem, x0, primal_dual=False)
        metrics = sol.state.termination_metrics(
            sol.state._optimisation_context(problem)
        )
        check_converged(sol, x_star, metrics)


@pytest.mark.parametrize("case", ["box", "quartic"])
def test_normal_equations_strategies_converge_alike(funnel_run, case):
    """Forcing each ``(Â Âᵀ)⁺`` realisation through the options reaches the
    same KKT point in a comparable number of steps, and the configured
    strategy reaches both the tangential solver and the multiplier recovery.

    The shared run uses the default ``"auto"`` strategy, which resolves to
    ``"schur"`` on these small problems and is the reference step count.
    """
    make_problem_fn, x0, x_star = CONVERGENCE_CASES[case]
    reference = funnel_run(case)
    steps = {"schur": reference.n_steps}
    with jax.enable_x64(True):
        sub_solver = reference.sol.state._init_subproblem(reference.problem).solver
        assert sub_solver.tangential_solver.normal_equations == "auto"
        for strategy in ("generic", "matrix-free"):
            options = {
                "subproblem": {
                    "tangential_solver": {"normal_equations": strategy},
                    "multiplier_recovery": {"normal_equations": strategy},
                }
            }
            problem = make_problem_fn()
            sol = run(problem, x0, options=options)
            sub_solver = sol.state._init_subproblem(problem).solver
            assert sub_solver.tangential_solver.normal_equations == strategy
            assert sub_solver.multiplier_recovery.normal_equations == strategy
            check_converged(sol, x_star)
            steps[strategy] = int(sol.stats["num_steps"])
    for strategy in ("generic", "matrix-free"):
        assert abs(steps[strategy] - steps["schur"]) <= max(3, steps["schur"] // 4)


@pytest.mark.parametrize(
    ("make_problem_fn", "case"),
    [
        (make_box_quadratic, "box"),
        pytest.param(make_scaled_quartic, "quartic", marks=pytest.mark.slow),
        (lambda: make_scaled_quartic(with_curvature=False), "quartic"),
    ],
    ids=["box", "quartic-with-hvp", "quartic-no-hvp"],
)
def test_minimise_converges_with_secant_model(make_problem_fn, case):
    """The L-BFGS model drives the funnel to the same KKT points.

    Whether the problem exposes an HVP or not only changes how the secant
    is selected, so the ``with_curvature=True`` variant is a slow duplicate
    of the automatic selection exercised by ``quartic-no-hvp``.
    """
    _, x0, x_star = CONVERGENCE_CASES[case]
    with jax.enable_x64(True):
        sol = run(make_problem_fn(), x0, curvature="secant")
        check_converged(sol, x_star)
        assert "secant_n_appends" in sol.stats


@pytest.mark.slow
@pytest.mark.parametrize("case", ["equality", "unconstrained"])
def test_agrees_with_trust_region_interior_point(funnel_run, case):
    """Both interior-point loops land on the same point.

    Restricted to the problems the trust-region loop currently solves to
    ``atol = 1e-6``; the funnel's bound-active cases are checked against the
    analytic minimisers above (which also cover these two problems, so the
    cross-check is a slow test).
    """
    funnel = funnel_run(case)
    with jax.enable_x64(True):
        tr = minimise(
            CONVERGENCE_CASES[case][0](),
            TrustRegionInteriorPointMinimiser(atol=1e-6, initial_mu=0.1),
            jnp.asarray(funnel.x0),
            max_steps=200,
            throw=False,
        )
        assert bool(tr.state.result_adapter.is_successful(tr.result))
        assert jnp.allclose(funnel.sol.value, tr.value, atol=1e-4)


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


@pytest.mark.slow
@pytest.mark.parametrize("case", ["equality", "unconstrained"])
def test_unconstrained_and_equality_converge_in_float32(case):
    """Without active bounds float32 reaches the tight tolerance as well.

    A slow duplicate of the float64 convergence tests; the float32 code
    path itself is covered by ``test_float32_converges_at_loose_tolerance``.
    """
    make_problem_fn, x0, x_star = CONVERGENCE_CASES[case]
    with jax.enable_x64(False):
        sol = run(make_problem_fn(), x0, max_steps=40)
        check_converged(sol, x_star)


# --- logging / integration -------------------------------------------------------


def test_step_logging_reports_funnel_columns(infeasible_funnel_run):
    """INFO rows carry the funnel measures, one row per outer step.

    See :class:`InfeasibleFunnelRun` for why the shared run is float32 and
    which of its properties are platform independent.
    """
    sol, handler = infeasible_funnel_run.sol, infeasible_funnel_run.handler
    assert sol.result == TRUST_FUNNEL_INTERIOR_POINT_RESULTS.infeasible_stationary_point
    assert int(sol.stats["num_steps"]) < 80
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


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_y_iteration_streak_logs_warning_and_stalls(x64):
    """A run of ``stall_steps`` y-iterations logs a WARNING and ends the run.

    At ``x = 0`` of :func:`make_zero_jacobian_infeasible_problem` both
    ``πᵛ`` and ``πᶠ`` vanish exactly while ``v = 1``, so Algorithm 2 sits at
    a fixed point from the first step in either precision; ``min_steps``
    keeps the loop alive past Step 8 long enough for the streak to form.
    """
    problem = make_zero_jacobian_infeasible_problem()
    handler = MemoryHandler()
    with jax.enable_x64(x64):
        sol = run(
            problem,
            [0.0, 0.0],
            max_steps=20,
            options={"logging": {"level": "INFO", "handler": handler}},
            min_steps=3,
            stall_steps=2,
        )
        metrics = sol.state.termination_metrics(
            sol.state._optimisation_context(problem)
        )
    assert sol.result == TRUST_FUNNEL_INTERIOR_POINT_RESULTS.infeasible_stationary_point
    assert int(sol.stats["num_steps"]) == 3
    assert int(sol.state.consecutive_y_iterations) == 3
    assert bool(metrics.stalled)
    steps = [r.message for r in handler.records if r.message.startswith("step=")]
    assert all("type=y_iteration" in m for m in steps)
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
