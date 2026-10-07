"""Unit tests for :mod:`slsqp_jax.sqpdax.minimiser.diagnostics.trust_funnel` and the
diagnostics wiring of the trust-funnel interior-point minimiser."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox._enum import EnumerationItem

from slsqp_jax.sqpdax.barrier import FunnelBarrierUpdate
from slsqp_jax.sqpdax.logging import MemoryDiagnosticsHandler, MemoryHandler
from slsqp_jax.sqpdax.minimiser import (
    TRUST_FUNNEL_INTERIOR_POINT_RESULTS,
    FunnelDiagnostics,
    TrustFunnelInteriorPointMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.primal import Slack
from slsqp_jax.sqpdax.subproblem.solver import (
    IterationType,
    MultiplierCase,
    TrustFunnelSolver,
    TrustFunnelSolverState,
)
from tests.sqpdax.minimiser.test_trust_funnel_interior_point import (
    CONVERGENCE_CASES,
    constraint_residual,
    make_box_quadratic,
    make_infeasible_problem,
)

R = TRUST_FUNNEL_INTERIOR_POINT_RESULTS
SOLVER = TrustFunnelSolver()
COUNTERS = [
    "n_y",
    "n_f",
    "n_v",
    "n_sf",
    "n_sv",
    "n_t0",
    "n_demoted",
    "n_rejected_f",
    "n_rejected_v_rho",
    "n_rejected_v_contraction",
    "n_normal_skipped",
    "n_multiplier_skipped",
    "n_tangential_rejected",
    "n_ftb_normal",
    "n_ftb_tangential",
    "n_multiplier_bound",
    "n_d_cap_hits",
    "n_outside_d_small_radius",
    "n_slack_resets",
    "n_mu_updates",
]
STEP_KEYS = {
    "funnel_diagnostics",
    "kappa_v_threshold",
    "ftb_normal_rate",
    "ftb_tangential_rate",
    "normal_skip_rate",
    "multiplier_skip_rate",
    "multiplier_norm",
    "multiplier_min",
    "complementarity",
    "kappa_y",
    "kappa_d",
    "d_cap_hits",
    "barrier_weight",
    "radius_v",
    "radius_f",
    "v_max",
    "rho",
    "iteration_type",
    "slack",
}
FUNNEL_STEP_KEYS = {
    "violation",
    "pi_v",
    "chi_v",
    "pi_f",
    "chi_f",
    "pi_f_prev",
    "radius_v",
    "radius_f",
    "radius_t",
    "v_max",
    "eps_pi",
    "eps_v",
    "gate_normal",
    "gate_multiplier",
    "normal_computed",
    "tangential_computed",
    "very_relaxed",
    "tangential_rejected",
    "tangential_reset",
    "in_td",
    "in_d",
    "multiplier_case",
    "multiplier_acceptable",
    "kkt_satisfied",
    "infeasible_stationary",
    "iteration_type",
    "dm_f_n",
    "dm_f_t",
    "dm_v_n",
    "dm_v_d",
    "normal_norm",
    "tangential_norm",
    "normal_ratio",
    "cauchy_decrease_v",
    "cauchy_decrease_f",
    "cauchy_ratio_v",
    "cauchy_ratio_f",
    "cauchy_ok",
    "a_norm",
    "h_norm",
    "cauchy_bound_v",
    "cauchy_bound_f",
    "cauchy_bound_ratio_v",
    "cauchy_bound_ratio_f",
    "ftb_normal",
    "ftb_tangential",
    "normal_step",
    "tangential_step",
    "step",
    "multipliers",
    "normal_state",
    "tangential_state",
    "cg_iters",
    "finite",
}
BARRIER_UPDATE_KEYS = {
    "step",
    "mu_old",
    "mu_new",
    "pi_f",
    "violation",
    "eps_pi_old",
    "eps_v_old",
    "eps_pi_new",
    "eps_v_new",
    "v_max_new",
    "diagnostics",
}


# --- helpers -------------------------------------------------------------------


def make_state(**overrides: Any) -> TrustFunnelSolverState:
    """Cold state with open gates, overridden field by field."""
    state = TrustFunnelSolverState.cold(1.0, 1.0, 10.0, eps_pi=1e-3, eps_v=1e-3)
    base = dict(gate_normal=True, gate_multiplier=True)
    base.update(overrides)

    def convert(name: str, value: Any):
        if isinstance(value, EnumerationItem):
            return value
        return jnp.asarray(value, getattr(state, name).dtype)

    return eqx.tree_at(
        lambda s: tuple(getattr(s, k) for k in base),
        state,
        tuple(convert(k, v) for k, v in base.items()),
    )


def update(
    diag: FunnelDiagnostics, state: TrustFunnelSolverState, **kwargs: Any
) -> FunnelDiagnostics:
    """``diag.update`` with healthy defaults."""
    values = dict(
        accepted=False,
        solved=False,
        reset=False,
        slack_positive=True,
        residual_min=0.0,
        violation=0.0,
        v_max_next=state.v_max,
        x_norm=1.0,
        multiplier_norm=0.0,
        kappa_y=1e3,
        d_cap_hits=0,
    )
    values.update(kwargs)
    statics = dict(radius_floor=1e-8, chi_tol=1e-8, tol=1e-6)
    for key in list(values):
        if key in statics:
            statics[key] = values.pop(key)
    arrays = {k: jnp.asarray(v) for k, v in values.items()}
    arrays["d_cap_hits"] = arrays["d_cap_hits"].astype(jnp.int32)
    return diag.update(state, SOLVER, **arrays, **statics)


def counters(diag: FunnelDiagnostics) -> dict[str, int]:
    return {name: int(getattr(diag, name)) for name in COUNTERS}


def funnel_minimiser(**kwargs: Any) -> TrustFunnelInteriorPointMinimiser:
    kwargs.setdefault("atol", 1e-6)
    kwargs.setdefault("initial_mu", 0.1)
    return TrustFunnelInteriorPointMinimiser(**kwargs)


# --- FunnelDiagnostics.zero / update ---------------------------------------------


def test_zero_carry_is_healthy():
    diag = FunnelDiagnostics.zero(7.0)
    assert all(value == 0 for value in counters(diag).values())
    assert float(diag.v_max_prev) == 7.0
    assert float(diag.kappa_g) == float(diag.kappa_c) == 0.0
    assert not bool(diag.invariant_violated) and not bool(diag.cauchy_violated)
    assert bool(diag.slack_positive) and bool(diag.in_funnel)
    assert float(diag.kappa_v_threshold(SOLVER)) == jnp.inf


COUNTER_CASES = {
    "y": (
        dict(iteration_type=IterationType.y_iteration),
        {},
        {"n_y": 1},
    ),
    "f-accepted": (
        dict(
            iteration_type=IterationType.f_iteration,
            tangential_norm=1.0,
            objective_decrease_ok=True,
        ),
        dict(accepted=True),
        {"n_f": 1, "n_sf": 1},
    ),
    "f-rejected": (
        dict(iteration_type=IterationType.f_iteration, tangential_norm=1.0),
        {},
        {"n_f": 1, "n_rejected_f": 1},
    ),
    "v-accepted": (
        dict(
            iteration_type=IterationType.v_iteration,
            normal_norm=1.0,
            contraction_ok=True,
        ),
        dict(accepted=True),
        {"n_v": 1, "n_sv": 1},
    ),
    "v-rejected-rho": (
        dict(
            iteration_type=IterationType.v_iteration,
            normal_norm=1.0,
            contraction_ok=True,
        ),
        {},
        {"n_v": 1, "n_rejected_v_rho": 1},
    ),
    "v-rejected-contraction": (
        dict(
            iteration_type=IterationType.v_iteration,
            normal_norm=1.0,
            contraction_ok=False,
        ),
        {},
        {"n_v": 1, "n_rejected_v_contraction": 1},
    ),
    "solver-flags": (
        dict(
            iteration_type=IterationType.v_iteration,
            normal_norm=1.0,
            contraction_ok=True,
            tangential_reset=True,
            demoted=True,
            tangential_rejected=True,
            ftb_normal=True,
            ftb_tangential=True,
            gate_normal=False,
            gate_multiplier=False,
        ),
        {},
        {
            "n_v": 1,
            "n_rejected_v_rho": 1,
            "n_t0": 1,
            "n_demoted": 1,
            "n_tangential_rejected": 1,
            "n_ftb_normal": 1,
            "n_ftb_tangential": 1,
            "n_normal_skipped": 1,
            "n_multiplier_skipped": 1,
        },
    ),
    "outer-loop": (
        dict(iteration_type=IterationType.y_iteration),
        dict(solved=True, reset=True, multiplier_norm=10.0, kappa_y=1.0, d_cap_hits=3),
        {
            "n_y": 1,
            "n_mu_updates": 1,
            "n_slack_resets": 1,
            "n_multiplier_bound": 1,
            "n_d_cap_hits": 3,
        },
    ),
}


@pytest.mark.parametrize(
    ("overrides", "kwargs", "expected"),
    list(COUNTER_CASES.values()),
    ids=list(COUNTER_CASES),
)
def test_update_increments_exactly_the_expected_counters(overrides, kwargs, expected):
    """Each scenario touches its counters once and leaves every other at zero."""
    diag = update(FunnelDiagnostics.zero(10.0), make_state(**overrides), **kwargs)
    observed = counters(diag)
    assert observed == {name: expected.get(name, 0) for name in COUNTERS}
    # Counters accumulate.
    again = update(diag, make_state(**overrides), **kwargs)
    assert counters(again) == {name: 2 * expected.get(name, 0) for name in COUNTERS}
    assert not bool(diag.invariant_violated)
    assert not bool(diag.cauchy_violated)
    assert bool(diag.multiplier_bound_exceeded) == bool(
        kwargs.get("multiplier_norm", 0.0) > kwargs.get("kappa_y", 1e3)
    )


def test_update_tracks_model_error_constants_and_degeneracy_proxy():
    """``κ_G``, ``κ_C`` are running maxima of the normalised model errors;
    ``‖P⁻¹n‖ / πᵛ`` a running maximum; y-iterations leave them untouched."""
    diag = FunnelDiagnostics.zero(10.0)
    state = make_state(
        iteration_type=IterationType.v_iteration,
        normal_norm=1.0,
        tangential_norm=1.0,
        model_error_f=0.5,
        model_error_v=1.0,
        normal_computed=True,
        pi_v=0.25,
    )
    diag = update(diag, state)
    assert float(diag.kappa_g) == pytest.approx(0.25)
    assert float(diag.kappa_c) == pytest.approx(0.5)
    assert float(diag.normal_ratio_max) == pytest.approx(4.0)
    assert float(diag.kappa_v_threshold(SOLVER)) == pytest.approx(
        (1.0 - SOLVER.kappa_tt) / (0.5 * SOLVER.kappa_v)
    )
    smaller = make_state(
        iteration_type=IterationType.f_iteration,
        tangential_norm=2.0,
        model_error_f=0.4,
        model_error_v=0.4,
    )
    diag = update(diag, smaller)
    assert float(diag.kappa_g) == pytest.approx(0.25)
    assert float(diag.kappa_c) == pytest.approx(0.5)
    y_state = make_state(
        iteration_type=IterationType.y_iteration,
        model_error_f=100.0,
        model_error_v=100.0,
    )
    diag = update(diag, y_state)
    assert float(diag.kappa_g) == pytest.approx(0.25)
    assert float(diag.kappa_c) == pytest.approx(0.5)
    assert float(diag.normal_ratio_max) == pytest.approx(4.0)


INVARIANT_CASES = {
    "slack": ({}, dict(slack_positive=False), "slack_positive"),
    "residual": ({}, dict(residual_min=-1.0), "residual_nonnegative"),
    "funnel": ({}, dict(violation=11.0), "in_funnel"),
    "v_max-growth": (dict(v_max=20.0), {}, "v_max_monotone"),
    "controller": (dict(funnel_violated=True), {}, None),
}


@pytest.mark.parametrize(
    ("overrides", "kwargs", "flag"),
    list(INVARIANT_CASES.values()),
    ids=list(INVARIANT_CASES),
)
def test_invariant_failures_set_the_sticky_flag(overrides, kwargs, flag):
    """Per-step flags report the failing check; ``invariant_violated`` sticks."""
    state = make_state(iteration_type=IterationType.y_iteration, **overrides)
    diag = update(FunnelDiagnostics.zero(10.0), state, **kwargs)
    for name in (
        "slack_positive",
        "residual_nonnegative",
        "in_funnel",
        "v_max_monotone",
    ):
        assert bool(getattr(diag, name)) == (name != flag)
    assert bool(diag.invariant_violated)
    healthy = update(diag, make_state(iteration_type=IterationType.y_iteration))
    assert bool(healthy.slack_positive) and bool(healthy.in_funnel)
    assert bool(healthy.invariant_violated)


def test_invariant_checks_tolerate_roundoff():
    """Violations within ``tol`` of the thresholds are not flagged."""
    state = make_state(
        iteration_type=IterationType.y_iteration, v_max=10.0 * (1 + 5e-7)
    )
    diag = update(
        FunnelDiagnostics.zero(10.0),
        state,
        violation=10.0 * (1 + 5e-7),
        residual_min=-5e-7,
    )
    assert bool(diag.in_funnel) and bool(diag.v_max_monotone)
    assert bool(diag.residual_nonnegative)
    assert not bool(diag.invariant_violated)


def test_cauchy_failure_is_sticky():
    bad = make_state(iteration_type=IterationType.v_iteration, cauchy_ok=False)
    diag = update(FunnelDiagnostics.zero(10.0), bad)
    assert bool(diag.cauchy_violated)
    good = make_state(iteration_type=IterationType.v_iteration)
    assert bool(update(diag, good).cauchy_violated)
    assert not bool(update(FunnelDiagnostics.zero(10.0), good).cauchy_violated)


Y_CASES = {
    "healthy-3.15b": (
        dict(multiplier_case=MultiplierCase.skip_tangential, pi_f=0.4),
        dict(),
        0,
        False,
    ),
    "3.15b-no-contraction": (
        dict(multiplier_case=MultiplierCase.skip_tangential, pi_f=0.6),
        dict(),
        1,
        False,
    ),
    "tangential-came-back-empty": (
        dict(
            multiplier_case=MultiplierCase.tangential,
            tangential_computed=True,
            pi_f=0.1,
        ),
        dict(),
        1,
        False,
    ),
    "terminating": (
        dict(multiplier_case=MultiplierCase.terminate, kkt_satisfied=True, pi_f=0.0),
        dict(),
        0,
        False,
    ),
    "solved": (
        dict(multiplier_case=MultiplierCase.tangential, pi_f=0.1),
        dict(solved=True),
        0,
        False,
    ),
    "collapsed-radius": (
        dict(
            multiplier_case=MultiplierCase.tangential,
            tangential_computed=True,
            pi_f=0.1,
            radius_f=1e-12,
        ),
        dict(),
        0,
        True,
    ),
}


@pytest.mark.parametrize(
    ("overrides", "kwargs", "streak", "collapse"),
    list(Y_CASES.values()),
    ids=list(Y_CASES),
)
def test_y_iteration_health_feeds_multiplier_and_collapse_streaks(
    overrides, kwargs, streak, collapse
):
    """Only (3.15b) y-iterations with ``πᶠ ≤ κ_ω πᶠ_prev`` (``κ_ω = 0.5``) or
    terminating ones are healthy; a collapsed ``δᶠ`` is a model problem."""
    assert SOLVER.kappa_omega == 0.5
    previous = eqx.tree_at(
        lambda d: d.pi_f_last, FunnelDiagnostics.zero(10.0), jnp.asarray(1.0)
    )
    state = make_state(iteration_type=IterationType.y_iteration, **overrides)
    diag = update(previous, state, **kwargs)
    assert int(diag.multiplier_failure_streak) == streak
    assert bool(diag.unhealthy_y_iteration) == (streak > 0)
    assert bool(diag.criticality_collapse) == collapse
    assert int(diag.criticality_collapse_streak) == int(collapse)
    assert float(diag.pi_f_last) == pytest.approx(float(state.pi_f))
    # A primal move resets the multiplier streak.
    moved = update(
        diag, make_state(iteration_type=IterationType.f_iteration, tangential_norm=1.0)
    )
    assert int(moved.multiplier_failure_streak) == 0


COLLAPSE_CASES = {
    "f-collapsed": (
        dict(iteration_type=IterationType.f_iteration, radius_f=1e-10, pi_f=0.01),
        True,
    ),
    "f-collapsed-but-critical": (
        dict(iteration_type=IterationType.f_iteration, radius_f=1e-10, pi_f=1e-4),
        False,
    ),
    "f-large-radius": (
        dict(iteration_type=IterationType.f_iteration, radius_f=0.1, pi_f=0.01),
        False,
    ),
    "v-collapsed": (
        dict(
            iteration_type=IterationType.v_iteration,
            radius_v=1e-10,
            violation=0.5,
            chi_v=0.1,
        ),
        True,
    ),
    "v-collapsed-but-stationary": (
        dict(
            iteration_type=IterationType.v_iteration,
            radius_v=1e-10,
            violation=0.5,
            chi_v=1e-12,
        ),
        False,
    ),
    "v-collapsed-but-feasible": (
        dict(
            iteration_type=IterationType.v_iteration,
            radius_v=1e-10,
            violation=1e-5,
            chi_v=0.1,
        ),
        False,
    ),
}


@pytest.mark.parametrize(
    ("overrides", "expected"), list(COLLAPSE_CASES.values()), ids=list(COLLAPSE_CASES)
)
def test_criticality_collapse_requires_both_small_radius_and_criticality(
    overrides, expected
):
    """Lemmas 4.8–4.10: the governing radius below ``radius_floor · max{1, ‖x‖}``
    with the matching measure above tolerance, and the streak resets otherwise."""
    diag = update(FunnelDiagnostics.zero(10.0), make_state(**overrides), x_norm=3.0)
    assert bool(diag.criticality_collapse) == expected
    assert int(diag.criticality_collapse_streak) == int(expected)
    # The floor scales with ``max{1, ‖x‖}``: at ``‖x‖ = 1e3`` it is 1e-5,
    # which still separates the collapsed (1e-10) from the open (0.1) radii.
    scaled = update(FunnelDiagnostics.zero(10.0), make_state(**overrides), x_norm=1e3)
    assert bool(scaled.criticality_collapse) == expected
    diag = update(diag, make_state(**overrides), x_norm=3.0)
    assert int(diag.criticality_collapse_streak) == 2 * int(expected)
    diag = update(diag, make_state(iteration_type=IterationType.y_iteration))
    assert int(diag.criticality_collapse_streak) == 0


def test_lemma_4_6_flags_v_iterations_outside_d_below_the_threshold():
    """Once ``κ_C`` is known, a v-iteration with
    ``min{δᵗ, κ_v v_max} ≤ κ_V`` must lie in ``D``."""
    seed = make_state(
        iteration_type=IterationType.v_iteration,
        normal_norm=1.0,
        model_error_v=1.0,  # κ_C = 1 → κ_V = (1 − κ_tt) / κ_v
        in_d=True,
    )
    diag = update(FunnelDiagnostics.zero(10.0), seed)
    kappa_v = float(diag.kappa_v_threshold(SOLVER))
    assert kappa_v == pytest.approx((1.0 - SOLVER.kappa_tt) / SOLVER.kappa_v)
    outside = make_state(
        iteration_type=IterationType.v_iteration,
        normal_norm=1.0,
        radius_t=0.5 * kappa_v,
        in_d=False,
    )
    flagged = update(diag, outside)
    assert bool(flagged.outside_d_small_radius)
    assert int(flagged.n_outside_d_small_radius) == 1
    large = eqx.tree_at(lambda s: s.radius_t, outside, jnp.asarray(2.0 * kappa_v))
    assert not bool(update(diag, large).outside_d_small_radius)
    inside = eqx.tree_at(lambda s: s.in_d, outside, jnp.asarray(True))
    assert not bool(update(diag, inside).outside_d_small_radius)


# --- minimiser wiring --------------------------------------------------------------


def test_init_seeds_the_carry_from_the_funnel_radius():
    problem = make_box_quadratic()
    solver = funnel_minimiser().init(problem, jnp.array([0.5, 0.0, 0.5]))
    diag = solver.funnel_diagnostics
    assert isinstance(diag, FunnelDiagnostics)
    assert float(diag.v_max_prev) == float(solver.solver_state.v_max)
    assert all(value == 0 for value in counters(diag).values())


@pytest.mark.parametrize(
    "kwargs",
    [dict(multiplier_failure_steps=0), dict(check_tol=-1.0)],
    ids=["failure-steps", "check-tol"],
)
def test_invalid_diagnostic_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        funnel_minimiser(**kwargs)


@pytest.mark.parametrize("case", ["box", "quartic"])
def test_run_emits_documented_diagnostic_records(case):
    """``step``, ``funnel_step`` and ``barrier_update`` records carry every
    documented key, and the counters are consistent with the run."""
    make_problem_fn, x0, _ = CONVERGENCE_CASES[case]
    handler = MemoryDiagnosticsHandler()
    with jax.enable_x64(True):
        sol = minimise(
            make_problem_fn(),
            funnel_minimiser(),
            jnp.asarray(x0),
            max_steps=120,
            throw=False,
            options={
                "logging": {"diagnostics": handler},
                "subproblem": {"norm_estimate_iters": 50},
            },
        )
    handler.close()
    assert sol.result == R.successful
    n_steps = int(sol.stats["num_steps"])

    steps = handler.select(name="minimiser", kind="step")
    assert len(steps) == n_steps
    payload = steps[-1].values
    assert STEP_KEYS <= set(payload)
    diag = payload["funnel_diagnostics"]
    assert type(diag).__name__ == "FunnelDiagnostics"
    for name in COUNTERS:
        assert isinstance(getattr(diag, name), (int, np.integer))
    assert isinstance(payload["multiplier_norm"], float)
    assert isinstance(payload["d_cap_hits"], (int, np.integer))
    assert payload["complementarity"] >= 0.0
    assert 0.0 <= payload["ftb_normal_rate"] <= 1.0

    # Counter consistency (Theorem 4.30 bookkeeping).
    final = sol.state.funnel_diagnostics
    c = counters(final)
    assert c["n_y"] + c["n_f"] + c["n_v"] == n_steps
    assert c["n_sf"] + c["n_rejected_f"] == c["n_f"]
    assert c["n_sv"] + c["n_rejected_v_rho"] + c["n_rejected_v_contraction"] == c["n_v"]
    assert c["n_sf"] > 0 or c["n_sv"] > 0
    # Monotone accumulation across step records.
    series = [counters(r.values["funnel_diagnostics"]) for r in steps]
    for name in COUNTERS:
        values = [s[name] for s in series]
        assert values == sorted(values)
    assert not bool(final.invariant_violated)
    assert not bool(final.cauchy_violated)

    funnel_steps = handler.select(name="minimiser.subproblem", kind="funnel_step")
    assert len(funnel_steps) == n_steps
    values = funnel_steps[-1].values
    assert FUNNEL_STEP_KEYS <= set(values)
    assert isinstance(values["cauchy_ok"], bool)
    assert isinstance(values["multiplier_case"], str)
    assert values["a_norm"] >= 0.0 and values["h_norm"] >= 0.0
    assert all(bool(r.values["cauchy_ok"]) for r in funnel_steps)
    # Lemma 3.5 / 3.9 lower bounds hold wherever they apply. The operator
    # norms are power-iteration estimates *from below*, so the bounds are
    # slightly overestimated (a few percent here); decreases at the
    # roundoff floor of ``f ≈ 1`` are not informative.
    checked = 0
    for r in funnel_steps:
        for key, bound in (
            ("cauchy_bound_ratio_v", "cauchy_bound_v"),
            ("cauchy_bound_ratio_f", "cauchy_bound_f"),
        ):
            ratio = r.values[key]
            if np.isnan(ratio) or r.values[bound] < 1e-12:
                continue
            checked += 1
            assert ratio >= 0.9
    assert checked > 0

    updates = handler.select(name="minimiser", kind="barrier_update")
    assert len(updates) == c["n_mu_updates"] > 0
    for r in updates:
        assert BARRIER_UPDATE_KEYS <= set(r.values)
        assert r.values["mu_new"] < r.values["mu_old"]
        assert r.values["pi_f"] <= r.values["eps_pi_old"]
        assert r.values["violation"] <= r.values["eps_v_old"]
        assert r.values["eps_pi_new"] < r.values["eps_pi_old"]


def test_no_diagnostics_traced_without_a_handler():
    problem = make_box_quadratic()
    sol = minimise(
        problem,
        funnel_minimiser(),
        jnp.array([0.5, 0.0, 0.5]),
        max_steps=5,
        throw=False,
    )
    assert sol.state.logger.diagnostics_enabled is False
    assert isinstance(sol.state.funnel_diagnostics, FunnelDiagnostics)


# --- termination and warnings -------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "lenient"])
@pytest.mark.parametrize(
    ("flag", "code"),
    [
        ("invariant_violated", "funnel_invariant_violation"),
        ("cauchy_violated", "cauchy_decrease_violation"),
    ],
    ids=["invariant", "cauchy"],
)
def test_strict_checks_promote_sticky_flags_to_fatal_results(strict, flag, code):
    problem = make_box_quadratic()
    solver = funnel_minimiser(min_steps=0, strict_checks=strict).init(
        problem, jnp.array([0.5, 0.0, 0.5])
    )
    solver = eqx.tree_at(
        lambda m: getattr(m.funnel_diagnostics, flag), solver, jnp.asarray(True)
    )
    done, result = solver.terminate(problem)
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    assert bool(getattr(metrics, flag))
    assert bool(done) == strict
    if strict:
        assert result == getattr(R, code)


@pytest.mark.parametrize(
    ("streak", "violation", "expected"),
    [
        (5, 0.0, "multiplier_solve_failure"),
        (4, 0.0, None),
        (5, 10.0, "infeasible_stationary_point"),
    ],
    ids=["streak", "short-streak", "infeasible-precedence"],
)
def test_multiplier_failure_streak_terminates(streak, violation, expected):
    problem = make_box_quadratic()
    x0 = jnp.array([0.5, 0.0, 0.5])
    solver = funnel_minimiser(
        min_steps=0, multiplier_failure_steps=5, stall_steps=5
    ).init(problem, x0)
    feasible = Slack(
        s=-problem.ineq_fn(x0) + violation,
        s_lb=x0 - problem.lb,
        s_ub=problem.ub - x0,
    )
    solver = eqx.tree_at(
        lambda m: (
            m.funnel_diagnostics.multiplier_failure_streak,
            m.consecutive_y_iterations,
            m.iterate.slack,
        ),
        solver,
        (jnp.asarray(streak, jnp.int32), jnp.asarray(streak, jnp.int32), feasible),
    )
    done, result = solver.terminate(problem)
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    if expected is None:
        assert not bool(done)
        assert not bool(metrics.multiplier_failure)
    else:
        assert bool(done)
        assert result == getattr(R, expected)
        assert bool(metrics.multiplier_failure) == (
            expected == "multiplier_solve_failure"
        )
        assert not bool(metrics.stationarity_stall)


def test_secant_model_channel_includes_the_criticality_collapse_streak():
    problem = make_box_quadratic()
    solver = funnel_minimiser().init(problem, jnp.array([0.5, 0.0, 0.5]))
    solver = eqx.tree_at(
        lambda m: (
            m.consecutive_model_failures,
            m.funnel_diagnostics.criticality_collapse_streak,
        ),
        solver,
        (jnp.asarray(2, jnp.int32), jnp.asarray(7, jnp.int32)),
    )
    assert int(solver._secant_reset_signals().model_streak) == 7


def test_injected_violations_fire_warnings_on_a_real_step():
    """A funnel radius below ``v₀`` and a tiny ``κ_y`` trip the invariant and
    multiplier-bound warnings and flags on the next committed step.

    Starting outside the funnel, either the committed iterate stays outside
    (``in_funnel`` fails) or (3.35) has to *grow* ``v_max`` to admit it
    (``v_max_monotone`` fails); both are invariant violations.
    """
    problem = make_box_quadratic()
    handler = MemoryHandler()
    solver = funnel_minimiser(
        barrier_update=FunnelBarrierUpdate(kappa_y_scale=1e-12)
    ).init(
        problem,
        jnp.array([3.0, -5.0, 0.0]),
        options={"logging": {"level": "WARNING", "handler": handler}},
    )
    v0 = float(jnp.linalg.norm(constraint_residual(problem, solver.iterate)))
    assert v0 > 1e-3
    solver = eqx.tree_at(
        lambda m: (m.solver_state.v_max, m.funnel_diagnostics.v_max_prev),
        solver,
        (jnp.asarray(1e-3, dtype=float), jnp.asarray(1e-3, dtype=float)),
    )
    solver = solver.step(problem)
    diag = solver.funnel_diagnostics
    assert not (bool(diag.in_funnel) and bool(diag.v_max_monotone))
    assert bool(diag.invariant_violated)
    assert bool(diag.multiplier_bound_exceeded)
    messages = [r.message for r in handler.records]
    assert any(m.startswith("funnel invariant violated") for m in messages)
    assert any(m.startswith("multiplier estimate exceeds kappa_y") for m in messages)
    metrics = solver.termination_metrics(solver._optimisation_context(problem))
    assert bool(metrics.invariant_violated)
    # Lenient by default: the run is not declared fatal.
    done, _ = solver.terminate(problem)
    assert not bool(done)


def test_infeasible_run_reports_unhealthy_y_iterations_and_bound_hits():
    """On the incompatible problem the multiplier estimate cannot satisfy
    (3.15): the carry records unhealthy y-iterations and ``κ_y`` hits, yet
    Step 8 keeps precedence in the result.

    Run in float32, where the loop ends through the y-iteration fixed point
    (float64 reaches ``χᵛ ≤ infeasibility_tol`` before any y-iteration).
    """
    handler = MemoryHandler()
    with jax.enable_x64(False):
        sol = minimise(
            make_infeasible_problem(),
            funnel_minimiser(),
            jnp.array([0.5, 0.3]),
            max_steps=80,
            throw=False,
            options={"logging": {"level": "WARNING", "handler": handler}},
        )
    assert sol.result == R.infeasible_stationary_point
    diag = sol.state.funnel_diagnostics
    assert int(diag.multiplier_failure_streak) >= 1
    assert int(diag.n_multiplier_bound) >= 1
    assert any(r.message.startswith("unhealthy y-iteration") for r in handler.records)
