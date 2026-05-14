"""Diagnostics-specific pytest fixtures and synthetic-state factories.

The synthetic factories build minimal :class:`SLSQPState` /
:class:`StepSummary` instances rigged to trip a particular signal
without needing to actually run SLSQP on a problem that exhibits the
pattern.  They are the load-bearing test surface for the
"every signal has a synthetic test" cap-policy enforcement.

The root-level ``tests/conftest.py`` is *not* re-imported here —
pytest's fixture inheritance picks it up automatically through the
package hierarchy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax.diagnostics.records import DebugRunResult, StepSummary
from slsqp_jax.diagnostics.signals import EvalContext
from slsqp_jax.hessian import lbfgs_init
from slsqp_jax.results import RESULTS
from slsqp_jax.state import SLSQPState, _init_diagnostics

jax.config.update("jax_enable_x64", True)


def _zero_summary(
    *,
    step_count: int = 1,
    f_val: float = 1.0,
    rel_kkt: float = 0.5,
    eq_jac_min_sv_est: float = float("inf"),
    diag_kappa: float = 1.0,
    last_alpha: float = 1.0,
    ls_success: bool = True,
    qp_converged: bool = True,
    max_eq_violation: float = 0.0,
    max_ineq_violation: float = 0.0,
    projected_grad_norm: float = float("inf"),
    grad_lagrangian_norm: float = 0.5,
    lagrangian_value: float = 1.0,
    **overrides,
) -> StepSummary:
    """Build a clean :class:`StepSummary` with sensible defaults.

    Tests override only the fields they care about for a particular
    signal (e.g. ``eq_jac_min_sv_est=1e-12`` to trip
    ``eq_jacobian_rank_deficient``); the rest stay benign so unrelated
    signals do not also fire.
    """
    base = dict(
        step_count=step_count,
        f_val=f_val,
        merit=f_val,
        last_alpha=last_alpha,
        qp_iterations_total=step_count,
        qp_iterations_step=1,
        qp_converged=qp_converged,
        qp_real_failure=False,
        qp_reached_max_iter=False,
        qp_ping_ponged=False,
        ls_success=ls_success,
        consecutive_qp_failures=0,
        consecutive_ls_failures=0,
        consecutive_zero_steps=0,
        grad_norm=grad_lagrangian_norm,
        grad_lagrangian_norm=grad_lagrangian_norm,
        lagrangian_value=lagrangian_value,
        rel_kkt=rel_kkt,
        gamma=1.0,
        min_diag=1.0,
        max_diag=1.0,
        diag_kappa=diag_kappa,
        lbfgs_count=1,
        lbfgs_skipped=False,
        max_abs_mult_eq=0.0,
        max_abs_mult_ineq=0.0,
        qp_vs_ls_multiplier_ratio=1.0,
        n_active_ineq=0,
        eq_jac_min_sv_est=eq_jac_min_sv_est,
        projected_grad_norm=projected_grad_norm,
        merit_penalty=1.0,
        max_eq_violation=max_eq_violation,
        max_ineq_violation=max_ineq_violation,
        proj_residual_high_water=0.0,
        diverging=False,
        blowup_count=0,
        merit_regression_step=False,
    )
    base.update(overrides)
    return StepSummary(**base)


@pytest.fixture
def make_summary():
    """Factory fixture: build a :class:`StepSummary` with overrides."""
    return _zero_summary


def _synthetic_state(
    *,
    n: int = 4,
    m_eq: int = 0,
    m_ineq: int = 0,
    eq_jac: np.ndarray | None = None,
    ineq_jac: np.ndarray | None = None,
    eq_val: np.ndarray | None = None,
    ineq_val: np.ndarray | None = None,
    diagnostics_overrides: dict | None = None,
    state_overrides: dict | None = None,
) -> SLSQPState:
    """Build a minimal :class:`SLSQPState` for synthetic signal tests.

    Most fields default to "uninteresting" (zeros, identity-like
    structures); tests pass overrides only for the fields whose value
    matters for the signal they are exercising.
    """
    diag = _init_diagnostics()
    if diagnostics_overrides:
        diag = type(diag)(  # rebuild via dataclass-style replace
            **{
                f.name: diagnostics_overrides.get(f.name, getattr(diag, f.name))
                for f in diag.__dataclass_fields__.values()  # type: ignore[attr-defined]
            }
        )

    if eq_jac is None:
        eq_jac = np.zeros((m_eq, n))
    if ineq_jac is None:
        ineq_jac = np.zeros((m_ineq, n))
    if eq_val is None:
        eq_val = np.zeros(m_eq)
    if ineq_val is None:
        ineq_val = np.zeros(m_ineq)

    base = dict(
        step_count=jnp.array(1),
        f_val=jnp.array(1.0),
        grad=jnp.zeros(n),
        eq_val=jnp.asarray(eq_val),
        ineq_val=jnp.asarray(ineq_val),
        eq_jac=jnp.asarray(eq_jac),
        ineq_jac=jnp.asarray(ineq_jac),
        lbfgs_history=lbfgs_init(n, memory=10),
        multipliers_eq_qp=jnp.zeros(m_eq),
        multipliers_ineq_qp=jnp.zeros(m_ineq),
        multipliers_eq_ls=jnp.zeros(m_eq),
        multipliers_ineq_ls=jnp.zeros(m_ineq),
        kkt_residual_grad=jnp.zeros(n),
        grad_lagrangian=jnp.zeros(n),
        merit_penalty=jnp.array(1.0),
        bound_jac=jnp.zeros((0, n)),
        qp_iterations=jnp.array(1),
        qp_converged=jnp.array(True),
        prev_active_set=jnp.zeros(m_ineq, dtype=bool),
        consecutive_qp_failures=jnp.array(0),
        consecutive_ls_failures=jnp.array(0),
        consecutive_zero_steps=jnp.array(0),
        qp_optimal=jnp.array(False),
        best_merit=jnp.array(1.0),
        steps_without_improvement=jnp.array(0),
        stagnation=jnp.array(False),
        last_alpha=jnp.array(1.0),
        last_projected_grad_norm=jnp.array(jnp.inf),
        ls_success=jnp.array(True),
        ls_fatal=jnp.array(False),
        qp_fatal=jnp.array(False),
        termination_code=RESULTS.successful,
        best_x=jnp.zeros(n),
        blowup_count=jnp.array(0),
        diverging=jnp.array(False),
        diagnostics=diag,
    )
    if state_overrides:
        base.update(state_overrides)
    return SLSQPState(**base)


@pytest.fixture
def make_state():
    """Factory fixture: build a synthetic :class:`SLSQPState`."""
    return _synthetic_state


@pytest.fixture
def make_run_result():
    """Factory fixture: build a synthetic :class:`DebugRunResult`."""

    def _build(
        *,
        summaries: list[StepSummary],
        final_state: SLSQPState,
        coarse_result=RESULTS.successful,
        final_result=RESULTS.successful,
        max_steps_reached: bool = False,
    ) -> DebugRunResult:
        return DebugRunResult(
            solver=None,  # Synthetic results don't carry a real solver.
            fn=None,
            x0=None,
            args=None,
            has_aux=False,
            final_state=final_state,
            final_y=None,
            final_result=final_result,
            coarse_result=coarse_result,
            summaries=list(summaries),
            terminated_at_step=len(summaries) - 1,
            max_steps_reached=max_steps_reached,
            fired_signals=[],
        )

    return _build


@pytest.fixture
def fake_solver():
    """Build a tiny stand-in object exposing ``rtol`` / ``atol``.

    Used by the synthetic signal tests to construct an
    :class:`EvalContext` without instantiating a real
    :class:`SLSQP`.  Real integration tests use a real solver.
    """

    class _FakeSolver:
        rtol = 1e-6
        atol = 1e-6
        max_steps = 100
        line_search_max_steps = 20

    return _FakeSolver()


@pytest.fixture
def make_eval_context(fake_solver):
    """Factory fixture: build an :class:`EvalContext` for synthetic tests."""

    def _build(
        *, rtol: float = 1e-6, atol: float = 1e-6, max_steps: int = 100
    ) -> EvalContext:
        return EvalContext(
            solver=fake_solver,  # type: ignore[arg-type]
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )

    return _build
