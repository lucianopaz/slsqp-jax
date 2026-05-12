"""Post-mortem diagnostics for SLSQP-JAX runs.

This sub-package layers a Python-driven debug runner, a per-step
scalar trajectory recorder, an artifact-eager signal pipeline, and a
small explicit playbook on top of the existing :class:`SLSQP` API.
Nothing here lives on the hot path: the additions are computed
post-hoc from existing state fields and never reached by
``step()`` / ``terminate()`` / verbose output.

Public entry points:

- :func:`debug_run` — re-execute a solver under a manual loop and
  return a :class:`DebugRunResult`.
- :func:`diagnose` — convenience wrapper that runs ``debug_run`` and
  builds a :class:`DebugReport`.
- :func:`capture_state_at_step` — re-run the solver to a specific
  step and return the live ``SLSQPState`` (with a mandatory
  reproducibility hash check).

The signal-evaluator tuples are exposed as
:data:`PER_STEP_EVALUATORS` and :data:`END_OF_RUN_EVALUATORS`; both
are empty in Phase 1 and populated by Phase 2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from slsqp_jax.diagnostics.intercept import (
    DiagnosticContext,
    diagnose_minimize_like_scipy,
    diagnostic_run,
)
from slsqp_jax.diagnostics.playbook import (
    SCOPE_BY_TERMINATION,
    Diagnosis,
    confidence_for,
    evaluate_diagnoses,
    magnitude_for,
    signals_in_scope,
    signals_out_of_scope,
)
from slsqp_jax.diagnostics.records import DebugRunResult, StepSummary
from slsqp_jax.diagnostics.report import DebugReport
from slsqp_jax.diagnostics.runner import capture_state_at_step, debug_run
from slsqp_jax.diagnostics.signals import (
    END_OF_RUN_EVALUATORS,
    PER_STEP_EVALUATORS,
    SIGNAL_REGISTRY,
    EvalContext,
    Signal,
    SignalRegistration,
    register_evaluator,
)

if TYPE_CHECKING:
    from slsqp_jax.slsqp import SLSQP


def diagnose(
    solver: "SLSQP",
    fn: Callable,
    x0: Any,
    *,
    args: Any = None,
    max_steps: Optional[int] = None,
    has_aux: bool = False,
) -> DebugReport:
    """Run ``solver`` under a manual loop and return a :class:`DebugReport`.

    Convenience wrapper around :func:`debug_run` + :class:`DebugReport`.
    Uses the registered :data:`PER_STEP_EVALUATORS` and
    :data:`END_OF_RUN_EVALUATORS`; the report is always produced (even
    on a successful run), so the user can request a diagnosis when
    they suspect slow-but-converged behaviour, not just on hard
    failures.
    """
    run = debug_run(
        solver,
        fn,
        x0,
        args=args,
        max_steps=max_steps,
        has_aux=has_aux,
        per_step_evaluators=PER_STEP_EVALUATORS,
        end_of_run_evaluators=END_OF_RUN_EVALUATORS,
    )
    diagnoses = evaluate_diagnoses(run.fired_signals)
    report = DebugReport.from_run(run)
    report.diagnoses = list(diagnoses)
    return report


__all__ = [
    "DebugReport",
    "DebugRunResult",
    "Diagnosis",
    "DiagnosticContext",
    "END_OF_RUN_EVALUATORS",
    "EvalContext",
    "PER_STEP_EVALUATORS",
    "SCOPE_BY_TERMINATION",
    "SIGNAL_REGISTRY",
    "Signal",
    "SignalRegistration",
    "StepSummary",
    "capture_state_at_step",
    "confidence_for",
    "debug_run",
    "diagnose",
    "diagnose_minimize_like_scipy",
    "diagnostic_run",
    "evaluate_diagnoses",
    "magnitude_for",
    "register_evaluator",
    "signals_in_scope",
    "signals_out_of_scope",
]
