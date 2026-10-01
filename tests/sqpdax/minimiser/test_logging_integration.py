"""End-to-end tests of ``options['logging']`` on the constrained minimisers."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.sqpdax.logging import (
    INFO,
    WARNING,
    Logger,
    MemoryDiagnosticsHandler,
    MemoryHandler,
)
from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    ProximalActiveSetLineSearchMinimiser,
    TrustRegionInteriorPointMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.subproblem.solver import ProjectedCGSubProblemSolver

from .conftest import make_equality_quadratic

_MINIMISERS = [
    pytest.param(lambda: ActiveSetLineSearchMinimiser(), id="active-set"),
    pytest.param(lambda: ProximalActiveSetLineSearchMinimiser(), id="proximal"),
    pytest.param(
        lambda: TrustRegionInteriorPointMinimiser(
            atol=1e-6, initial_mu=0.1, initial_radius=2.0
        ),
        id="trust-region-ip",
    ),
]
# Kind of the per-solve subproblem record each minimiser's solver emits.
_SOLVE_KIND = {
    ActiveSetLineSearchMinimiser: "qp",
    ProximalActiveSetLineSearchMinimiser: "qp",
    TrustRegionInteriorPointMinimiser: "tr_step",
}


def _run(
    make_solver, logging_spec, *, max_steps: int = 40, problem=None, **extra_options
):
    problem = make_equality_quadratic() if problem is None else problem
    options = dict(extra_options)
    if logging_spec is not None:
        options["logging"] = logging_spec
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol = minimise(
            problem,
            make_solver(),
            jnp.asarray([0.25, 0.25]),
            max_steps=max_steps,
            throw=False,
            options=options or None,
        )
    return sol, [str(w.message) for w in caught]


def _by_name(handler: MemoryHandler, name: str) -> list:
    return [r for r in handler.records if r.name == name]


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_logging_section_is_a_recognised_option(make_solver):
    handler = MemoryHandler()
    sol, messages = _run(make_solver, {"level": "INFO", "handler": handler})
    assert not any("unknown option section" in m for m in messages)
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert sol.state.logger.level == INFO
    assert sol.state.logger.handler is handler


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_disabled_logger_emits_nothing(make_solver):
    handler = MemoryHandler()
    sol, _ = _run(make_solver, None)
    assert sol.state.logger is Logger.disabled()
    # A disabled logger given explicitly also stays silent.
    sol, _ = _run(make_solver, {"level": "off", "handler": handler})
    assert handler.records == []


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_info_summary_once_per_step_and_run_bookends(make_solver):
    handler = MemoryHandler()
    sol, _ = _run(make_solver, {"level": "INFO", "handler": handler})
    root = _by_name(handler, "minimiser")
    assert root[0].levelno == INFO
    assert root[0].message.startswith(type(sol.state).__name__)
    assert (
        root[-1].message
        == f"finished: result=successful steps={int(sol.stats['num_steps'])}"
    )

    summaries = [r for r in root if r.message.startswith("step=")]
    assert len(summaries) == int(sol.stats["num_steps"])
    assert [r.values["step"] for r in summaries] == list(range(1, len(summaries) + 1))
    for key in ("f", "merit", "alpha", "dnorm", "accepted", "sub_ok", "sub_status"):
        assert key in summaries[0].values
    # Enumeration values are rendered by item name.
    assert summaries[-1].values["sub_status"] in {"successful", "max_steps_reached"}
    assert isinstance(summaries[-1].values["accepted"], bool)
    # Components run at INFO too: nothing below INFO leaks through.
    assert all(r.levelno >= INFO for r in handler.records)


@pytest.mark.parametrize("make_solver", _MINIMISERS)
@pytest.mark.parametrize(
    ("child", "sibling"),
    [("subproblem", "step_controller"), ("step_controller", "subproblem")],
)
def test_child_levels_are_honoured_independently(make_solver, child, sibling):
    handler = MemoryHandler()
    _run(make_solver, {"level": "WARNING", "handler": handler, child: "DEBUG"})
    child_records = _by_name(handler, f"minimiser.{child}")
    sibling_records = _by_name(handler, f"minimiser.{sibling}")
    # The child runs at DEBUG while the root and sibling stay at WARNING. Not
    # every component emits DEBUG records (the trust-region manager only logs
    # at INFO and above), so the check is "below the root level".
    assert any(r.levelno < WARNING for r in child_records)
    assert all(r.levelno >= WARNING for r in sibling_records)
    assert all(r.levelno >= WARNING for r in _by_name(handler, "minimiser"))
    assert all(r.depth == 1 for r in child_records)
    assert all(
        line.startswith("  minimiser.")
        for line in handler.lines
        if f"minimiser.{child}" in line
    )


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_child_loggers_inherit_root_level(make_solver):
    handler = MemoryHandler()
    _run(make_solver, {"level": "DEBUG", "handler": handler})
    names = {r.name for r in handler.records}
    assert {"minimiser", "minimiser.subproblem", "minimiser.step_controller"} <= names


def test_max_steps_exhaustion_is_reported_as_warning():
    handler = MemoryHandler()
    sol, _ = _run(
        lambda: ActiveSetLineSearchMinimiser(),
        {"level": "WARNING", "handler": handler},
        max_steps=0,
    )
    assert not bool(sol.state.result_adapter.is_successful(sol.result))
    assert handler.records[-1].levelno == WARNING
    assert handler.records[-1].message.startswith(
        "finished without success: result=max_steps_reached"
    )


# --- diagnostics channel ------------------------------------------------------


@pytest.mark.parametrize(
    "spec_for",
    [lambda h: h, lambda h: {"level": "WARNING", "diagnostics": h}],
    ids=["bare-handler", "mapping"],
)
@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_diagnostics_records_per_step_solve_and_run(make_solver, spec_for):
    handler = MemoryDiagnosticsHandler()
    sol, messages = _run(make_solver, spec_for(handler))
    assert not any("unknown option section" in m for m in messages)
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    assert sol.state.logger.diagnostics_handler is handler
    handler.close()
    n_steps = int(sol.stats["num_steps"])
    n_before = len(handler)

    steps = handler.select(name="minimiser", kind="step")
    assert [r.values["step"] for r in steps] == list(range(1, n_steps + 1))
    payload = steps[-1].values
    assert set(payload) >= {
        "x",
        "dual",
        "f",
        "grad",
        "grad_lagrangian",
        "eq_val",
        "eq_jac",
        "ineq_val",
        "ineq_jac",
        "merit",
        "alpha",
        "dnorm",
        "accepted",
        "solver_state",
        "metrics",
    }
    assert np.asarray(payload["x"].x).shape == (2,)
    assert payload["eq_jac"].shape == (1, 2)
    assert payload["grad"].shape == (2,)
    assert isinstance(payload["accepted"], bool)
    assert isinstance(payload["f"], float)
    assert type(payload["metrics"]).__name__.endswith("TerminationMetrics")

    solve_kind = _SOLVE_KIND[type(sol.state)]
    solves = handler.select(name="minimiser.subproblem", kind=solve_kind)
    assert len(solves) >= n_steps
    assert isinstance(solves[0].values["status"], str)

    runs = handler.select(kind="run")
    assert len(runs) == 1 and runs[0] is handler.records[-1]
    assert runs[0].values["result"] == "successful"
    assert runs[0].values["successful"] is True
    assert runs[0].values["steps"] == n_steps
    # Every record came from the root or one of its children, in run order.
    assert {r.name for r in handler} <= {"minimiser", "minimiser.subproblem"}
    assert all(r.run is None for r in handler)
    # Closed: the content is frozen and further emission is refused.
    assert len(handler) == n_before
    with pytest.raises(RuntimeError, match="closed"):
        handler.emit(handler.records[0])


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_algorithm_specific_step_fields(make_solver):
    handler = MemoryDiagnosticsHandler()
    sol, _ = _run(make_solver, handler)
    handler.close()
    payload = handler.select(kind="step")[-1].values
    expected = {
        ActiveSetLineSearchMinimiser: {"merit_penalty", "qp_result", "qp_active_set"},
        ProximalActiveSetLineSearchMinimiser: {"merit_penalty", "prox_mu", "eq_center"},
        TrustRegionInteriorPointMinimiser: {"barrier_weight", "radius", "rho", "slack"},
    }[type(sol.state)]
    assert expected <= set(payload)


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_secant_fields_are_reported_when_curvature_is_inexact(make_solver):
    """Without an exact HVP the minimiser carries an L-BFGS secant, and both
    the INFO summary and the diagnostics payload report its state."""
    text = MemoryHandler()
    diag = MemoryDiagnosticsHandler()
    sol, _ = _run(
        make_solver,
        {"level": "INFO", "handler": text, "diagnostics": diag},
        problem=make_equality_quadratic(with_curvature=False),
    )
    diag.close()
    assert sol.state.secant is not None
    n_steps = int(sol.stats["num_steps"])
    summaries = [
        r for r in _by_name(text, "minimiser") if r.message.startswith("step=")
    ]
    assert len(summaries) == n_steps
    assert all(isinstance(r.values["pairs"], int) for r in summaries)
    steps = diag.select(name="minimiser", kind="step")
    assert len(steps) == n_steps
    payload = steps[-1].values
    assert payload["pairs"] == summaries[-1].values["pairs"]
    assert (
        type(payload["secant_stats"]).__name__ == type(sol.state.secant_stats).__name__
    )
    assert type(payload["secant_recovery"]) is type(sol.state.secant_recovery_state)


@pytest.mark.parametrize(
    "spec_for",
    [
        lambda: None,
        lambda: {"level": "DEBUG", "handler": MemoryHandler()},
        lambda: True,
    ],
    ids=["default", "text-only", "true"],
)
@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_no_diagnostics_without_a_handler(make_solver, spec_for, capsys):
    sol, _ = _run(make_solver, spec_for())
    assert sol.state.logger.diagnostics_enabled is False
    assert sol.state.logger.diagnostics_handler is None
    # Text output (when any) is unchanged by the diagnostics machinery.
    out = capsys.readouterr().out
    assert ("step=" in out) is (spec_for() is True)


@pytest.mark.parametrize(
    ("root_enabled", "child_flag", "expected_names"),
    [
        (False, True, {"minimiser.subproblem"}),
        (True, False, {"minimiser"}),
        (True, True, {"minimiser", "minimiser.subproblem"}),
    ],
    ids=["subproblem-only", "all-but-subproblem", "both"],
)
@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_per_component_diagnostics_toggle(
    make_solver, root_enabled, child_flag, expected_names
):
    handler = MemoryDiagnosticsHandler()
    sol, _ = _run(
        make_solver,
        {
            "diagnostics": {"handler": handler, "enabled": root_enabled},
            "subproblem": {"diagnostics": child_flag},
        },
    )
    handler.close()
    assert len(handler) > 0
    assert {r.name for r in handler} == expected_names
    kinds = {r.kind for r in handler}
    assert ("step" in kinds) is root_enabled and ("run" in kinds) is root_enabled
    assert (_SOLVE_KIND[type(sol.state)] in kinds) is child_flag


def test_qp_failure_record_is_emitted_at_the_failing_inner_iteration():
    """A CG with no iteration budget fails on every working set; each failure
    is captured at the inner iteration where it is detected.

    The proximal solver is used because its stabilised system is full-space,
    so zero CG iterations can never satisfy the tolerance (the plain
    active-set loop on this equality-only problem is solved exactly by the
    projector alone).
    """
    handler = MemoryDiagnosticsHandler()
    max_steps = 3
    _run(
        lambda: ProximalActiveSetLineSearchMinimiser(),
        handler,
        max_steps=max_steps,
        subproblem={"subproblem_solver": ProjectedCGSubProblemSolver(max_iter=0)},
    )
    handler.close()
    sub = handler.select(name="minimiser.subproblem")
    failures = [r for r in sub if r.kind == "qp_iter_failure"]
    solves = [r for r in sub if r.kind == "qp"]
    assert len(solves) == max_steps and len(failures) == max_steps
    for failure in failures:
        values = failure.values
        assert values["iter"] == 0
        assert values["kkt_state"].success is False
        assert values["kkt_state"].reason == "max_iter_reached"
        assert np.all(np.isfinite(np.asarray(values["step"].x)))
        assert set(values) >= {"active_set", "next_set", "dual", "changed", "cycled"}
    # Each failure record precedes the per-solve summary of its own solve.
    assert [r.kind for r in sub] == ["qp_iter_failure", "qp"] * max_steps
    assert all(r.values["success"] is False for r in solves)
    assert all(r.values["qp_result"] == "kkt_solver_failure" for r in solves)


@pytest.mark.parametrize("make_solver", _MINIMISERS)
def test_successful_run_emits_no_solver_failure_records(make_solver):
    handler = MemoryDiagnosticsHandler()
    sol, _ = _run(make_solver, handler)
    handler.close()
    assert bool(sol.state.result_adapter.is_successful(sol.result))
    failure_kinds = {
        "normal_step_failure",
        "tangential_step_failure",
        "recovery_failure",
        "secant_reset",
    }
    assert not any(r.kind in failure_kinds for r in handler)
