"""Per-step scalar summaries and aggregate run records.

The diagnostics layer never stores per-step ``SLSQPState`` (or any
device-resident array) because doing so would add hundreds of MB of
memory for large ``n``.  Instead it records a tiny scalar
:class:`StepSummary` per iteration plus the *final* ``SLSQPState`` in
full.  All signal evaluators that need to inspect a non-final iterate
either compute their evidence at the moment the signal fires (when
the live state is still in scope) or trigger a re-run via
:func:`slsqp_jax.diagnostics.runner.capture_state_at_step`.

The fields on :class:`StepSummary` are deliberately host-resident
plain Python scalars so the diagnostics loop can branch on them
without forcing a device sync per access.  ``StepSummary.from_state``
performs the (single) device → host transfer per step.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import jax.numpy as jnp

from slsqp_jax.merit import compute_merit
from slsqp_jax.results import RESULTS

if TYPE_CHECKING:
    from slsqp_jax.slsqp import SLSQP
    from slsqp_jax.state import SLSQPDiagnostics, SLSQPState

# Number of decimal digits the reproducibility hash retains.  fp64
# arithmetic across two runs of an identical computation should agree
# bit-for-bit on the same hardware/jax-config; rounding to 12 digits
# absorbs the rare last-bit difference (e.g. when a fused-multiply-add
# is reordered) without inviting actual divergence.
_HASH_PRECISION_DECIMALS = 12


@dataclass(frozen=True)
class StepSummary:
    """Host-resident scalar summary of a single SLSQP iteration.

    Every field is a plain Python ``int`` / ``float`` / ``bool`` so the
    diagnostics loop can branch on it without an extra device sync.
    The only device → host transfer happens once inside
    :meth:`StepSummary.from_state`.

    Attributes:
        step_count: Iteration number after this step (1-indexed).
        f_val: Objective value at the post-step iterate.
        merit: L1 merit value at the post-step iterate.
        last_alpha: Line-search step size accepted at this step.
        qp_iterations_total: Cumulative QP active-set iterations across
            all steps so far (matches ``state.qp_iterations``).
        qp_iterations_step: QP active-set iterations consumed by *this*
            step alone (delta of ``qp_iterations_total``).
        qp_converged: Whether the QP solver reported success this step.
        qp_real_failure: ``True`` iff the QP did not converge AND did
            not exhaust its iteration budget (the "real failure"
            distinction the L-BFGS reset chain uses; see ``AGENTS.md``
            "Real-vs-budget QP failure distinction").
        qp_reached_max_iter: Whether the QP active-set loop exhausted
            ``qp_max_iter`` this step (delta of
            ``diagnostics.n_qp_budget_exhausted``).
        qp_ping_ponged: Whether the QP ping-pong short-circuit fired
            this step (delta of ``diagnostics.n_qp_ping_pong``).
        ls_success: Whether the line search reported success.
        consecutive_qp_failures: Outer-loop failure-streak counter.
        consecutive_ls_failures: Outer-loop failure-streak counter.
        consecutive_zero_steps: Zero-step convergence-detection counter.
        grad_norm: Euclidean norm of the objective gradient at the
            post-step iterate.
        grad_lagrangian_norm: Euclidean norm of the Lagrangian gradient
            at the post-step iterate (used by the classical
            stationarity criterion).
        lagrangian_value: Value of the Lagrangian
            ``L = f - lambda_eq^T c_eq - mu_ineq^T c_ineq`` at the
            post-step iterate.
        rel_kkt: ``grad_lagrangian_norm / max(|L|, 1)`` — exactly the
            ratio the convergence check compares against ``rtol``.
        gamma: Scalar L-BFGS initial-Hessian scaling.
        min_diag: Minimum entry of the L-BFGS per-variable diagonal.
        max_diag: Maximum entry of the L-BFGS per-variable diagonal.
        diag_kappa: ``max_diag / max(min_diag, 1e-30)``, the L-BFGS
            initial-Hessian condition-number proxy.
        lbfgs_count: Number of curvature pairs currently stored in the
            ring buffer.
        lbfgs_skipped: Whether the L-BFGS append skipped its curvature
            pair this step (delta of ``diagnostics.n_lbfgs_skips``).
        max_abs_mult_eq: ``max(|lambda_eq|)`` at the post-step iterate.
        max_abs_mult_ineq: ``max(|mu_ineq|)`` at the post-step iterate.
        n_active_ineq: Number of active inequality constraints (general
            + bounds) at the post-step iterate.
        eq_jac_min_sv_est: Lower bound on the smallest singular value
            of ``J_eq`` estimated from the Cholesky of ``J_eq J_eq^T``
            (cumulative low-water mark from
            ``diagnostics.eq_jac_min_sv_est``).
        projected_grad_norm: Norm of the inner solver's projected
            gradient (``HRInexactSTCG`` only; ``inf`` otherwise).
        merit_penalty: L1 merit penalty parameter ``rho`` after this
            step.
        max_eq_violation: ``max|c_eq|`` at the post-step iterate.
        max_ineq_violation: ``max(0, -c_ineq)`` at the post-step
            iterate.
        proj_residual_high_water: Cumulative high-water mark of the
            ``MinresQLPSolver`` M-metric projection residual.  Always
            ``0.0`` for null-space solvers.
        diverging: Whether the best-iterate divergence rollback fired
            this step.
        blowup_count: Consecutive blowup events at this step.
        merit_regression_step: Whether ``compute_merit`` exceeded the
            best merit despite the line search reporting success
            (delta of ``diagnostics.n_merit_regressions``).
    """

    step_count: int
    f_val: float
    merit: float
    last_alpha: float

    qp_iterations_total: int
    qp_iterations_step: int
    qp_converged: bool
    qp_real_failure: bool
    qp_reached_max_iter: bool
    qp_ping_ponged: bool

    ls_success: bool
    consecutive_qp_failures: int
    consecutive_ls_failures: int
    consecutive_zero_steps: int

    grad_norm: float
    grad_lagrangian_norm: float
    lagrangian_value: float
    rel_kkt: float

    gamma: float
    min_diag: float
    max_diag: float
    diag_kappa: float
    lbfgs_count: int
    lbfgs_skipped: bool

    max_abs_mult_eq: float
    max_abs_mult_ineq: float
    n_active_ineq: int

    eq_jac_min_sv_est: float
    projected_grad_norm: float
    merit_penalty: float
    max_eq_violation: float
    max_ineq_violation: float
    proj_residual_high_water: float

    diverging: bool
    blowup_count: int
    merit_regression_step: bool

    @classmethod
    def from_state(
        cls,
        state: "SLSQPState",
        *,
        prev_state: Optional["SLSQPState"] = None,
    ) -> "StepSummary":
        """Materialise a :class:`StepSummary` from a live ``SLSQPState``.

        This is the *single* device → host transfer per debug-loop
        iteration.  Everything the runner inspects for control flow
        below this point reads from the returned :class:`StepSummary`.

        ``prev_state`` is the ``SLSQPState`` from the previous iteration
        (or the initial-state from ``solver.init`` for step 1).  It is
        used solely to compute single-step deltas of the cumulative
        diagnostics counters (``n_qp_budget_exhausted``,
        ``n_qp_ping_pong``, ``n_lbfgs_skips``,
        ``n_merit_regressions``) and ``state.qp_iterations``.  Passing
        ``None`` treats every cumulative counter as starting from 0,
        which matches the behaviour of ``solver.init`` (which produces
        a state with all-zero diagnostics).
        """
        diag: SLSQPDiagnostics = state.diagnostics
        prev_diag: Optional[SLSQPDiagnostics] = (
            prev_state.diagnostics if prev_state is not None else None
        )

        f_val = float(state.f_val)
        eq_val = state.eq_val
        ineq_val = state.ineq_val
        merit_penalty = float(state.merit_penalty)
        merit_arr = compute_merit(state.f_val, eq_val, ineq_val, state.merit_penalty)
        merit = float(merit_arr)

        m_eq = int(eq_val.shape[0])
        m_ineq = int(ineq_val.shape[0])

        max_eq_violation = float(jnp.max(jnp.abs(eq_val))) if m_eq > 0 else 0.0
        max_ineq_violation = (
            float(jnp.max(jnp.maximum(0.0, -ineq_val))) if m_ineq > 0 else 0.0
        )

        lagrangian_value = f_val
        if m_eq > 0:
            lagrangian_value -= float(jnp.dot(state.multipliers_eq, eq_val))
        if m_ineq > 0:
            lagrangian_value -= float(jnp.dot(state.multipliers_ineq, ineq_val))

        grad_lagrangian_norm = float(jnp.linalg.norm(state.grad_lagrangian))
        rel_kkt = grad_lagrangian_norm / max(abs(lagrangian_value), 1.0)

        diagonal = state.lbfgs_history.diagonal
        min_diag = float(jnp.min(diagonal))
        max_diag = float(jnp.max(diagonal))
        diag_kappa = max_diag / max(min_diag, 1e-30)

        max_abs_mult_eq = (
            float(jnp.max(jnp.abs(state.multipliers_eq))) if m_eq > 0 else 0.0
        )
        max_abs_mult_ineq = (
            float(jnp.max(jnp.abs(state.multipliers_ineq))) if m_ineq > 0 else 0.0
        )
        n_active_ineq = int(jnp.sum(state.prev_active_set.astype(jnp.int32)))

        qp_iterations_total = int(state.qp_iterations)
        prev_qp_iters = int(prev_state.qp_iterations) if prev_state is not None else 0
        qp_iterations_step = qp_iterations_total - prev_qp_iters

        prev_n_budget = int(prev_diag.n_qp_budget_exhausted) if prev_diag else 0
        qp_reached_max_iter = (int(diag.n_qp_budget_exhausted) - prev_n_budget) > 0

        prev_n_pingpong = int(prev_diag.n_qp_ping_pong) if prev_diag else 0
        qp_ping_ponged = (int(diag.n_qp_ping_pong) - prev_n_pingpong) > 0

        prev_n_lbfgs_skips = int(prev_diag.n_lbfgs_skips) if prev_diag else 0
        lbfgs_skipped = (int(diag.n_lbfgs_skips) - prev_n_lbfgs_skips) > 0

        prev_n_merit_regressions = (
            int(prev_diag.n_merit_regressions) if prev_diag else 0
        )
        merit_regression_step = (
            int(diag.n_merit_regressions) - prev_n_merit_regressions
        ) > 0

        qp_converged = bool(state.qp_converged)
        qp_real_failure = (not qp_converged) and (not qp_reached_max_iter)

        return cls(
            step_count=int(state.step_count),
            f_val=f_val,
            merit=merit,
            last_alpha=float(state.last_alpha),
            qp_iterations_total=qp_iterations_total,
            qp_iterations_step=qp_iterations_step,
            qp_converged=qp_converged,
            qp_real_failure=qp_real_failure,
            qp_reached_max_iter=qp_reached_max_iter,
            qp_ping_ponged=qp_ping_ponged,
            ls_success=bool(state.ls_success),
            consecutive_qp_failures=int(state.consecutive_qp_failures),
            consecutive_ls_failures=int(state.consecutive_ls_failures),
            consecutive_zero_steps=int(state.consecutive_zero_steps),
            grad_norm=float(jnp.linalg.norm(state.grad)),
            grad_lagrangian_norm=grad_lagrangian_norm,
            lagrangian_value=lagrangian_value,
            rel_kkt=rel_kkt,
            gamma=float(state.lbfgs_history.gamma),
            min_diag=min_diag,
            max_diag=max_diag,
            diag_kappa=diag_kappa,
            lbfgs_count=int(state.lbfgs_history.count),
            lbfgs_skipped=lbfgs_skipped,
            max_abs_mult_eq=max_abs_mult_eq,
            max_abs_mult_ineq=max_abs_mult_ineq,
            n_active_ineq=n_active_ineq,
            eq_jac_min_sv_est=float(diag.eq_jac_min_sv_est),
            projected_grad_norm=float(state.last_projected_grad_norm),
            merit_penalty=merit_penalty,
            max_eq_violation=max_eq_violation,
            max_ineq_violation=max_ineq_violation,
            proj_residual_high_water=float(diag.max_proj_residual),
            diverging=bool(state.diverging),
            blowup_count=int(state.blowup_count),
            merit_regression_step=merit_regression_step,
        )

    def reproducibility_digest(self) -> str:
        """Return a short hex digest used by ``capture_state_at_step``.

        Two ``StepSummary`` instances produced by independent runs of
        the same ``(solver, x0, args)`` should hash to the same digest
        on the same hardware / ``jax.config`` settings.  Floats are
        rounded to ``_HASH_PRECISION_DECIMALS`` decimals before hashing
        so a last-bit difference (e.g. from a fused-multiply-add
        reorder) does not false-positive on nondeterminism.
        """
        parts: list[str] = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, float):
                if val != val or val in (float("inf"), float("-inf")):
                    parts.append(f"{f.name}={val!r}")
                else:
                    parts.append(f"{f.name}={round(val, _HASH_PRECISION_DECIMALS)!r}")
            else:
                parts.append(f"{f.name}={val!r}")
        joined = "|".join(parts).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()[:16]


@dataclass
class DebugRunResult:
    """Aggregate result of a manual-loop debug run.

    Carries everything a downstream signal evaluator or report
    renderer needs to do its job, plus the handles
    ``(solver, fn, x0, args, has_aux)`` required to re-execute the
    run for ad-hoc inspection via
    :func:`slsqp_jax.diagnostics.runner.capture_state_at_step`.

    Attributes:
        solver: The :class:`slsqp_jax.SLSQP` instance the run used.
        fn: The objective callable passed to ``debug_run``.
        x0: The initial iterate.
        args: Extra positional payload threaded through ``fn`` /
            constraint callables.
        has_aux: Whether ``fn`` returns ``(value, aux)``.
        final_state: The terminal ``SLSQPState`` after the manual
            loop exited.  Carries the full ``SLSQPDiagnostics``
            accumulator and every device array the end-of-run
            evaluators need.
        final_y: The terminal iterate ``y`` returned by ``step()``.
        final_result: The granular ``slsqp_jax.RESULTS`` termination
            code stored on ``final_state.termination_code``.
        coarse_result: The coarse ``optimistix.RESULTS`` code that
            ``terminate()`` returned (always a member of the parent
            enum, suitable for the optimistix driver-style check).
        summaries: Per-step :class:`StepSummary` records, one per
            iteration the loop actually executed.
        terminated_at_step: Index of the iteration where the loop
            exited (0-based; ``len(summaries) - 1`` on success).
        max_steps_reached: ``True`` iff the loop ran out of iterations
            without ``terminate()`` returning ``done``.
        fired_signals: Signals emitted by the per-step + end-of-run
            evaluators.  Empty list when the diagnostics layer is run
            without signals (Phase 1 default).
    """

    solver: "SLSQP"
    fn: Any
    x0: Any
    args: Any
    has_aux: bool

    final_state: "SLSQPState"
    final_y: Any
    final_result: Any
    coarse_result: Any

    summaries: list[StepSummary]
    terminated_at_step: int
    max_steps_reached: bool

    fired_signals: list[Any] = field(default_factory=list)

    @property
    def diagnostics(self) -> "SLSQPDiagnostics":
        """Convenience accessor for ``final_state.diagnostics``."""
        return self.final_state.diagnostics

    @property
    def n_steps(self) -> int:
        """Number of iterations the loop actually executed."""
        return len(self.summaries)

    @property
    def terminated_successfully(self) -> bool:
        """``True`` iff the run actually converged.

        Both conditions must hold: the granular ``final_result`` is
        :attr:`RESULTS.successful` *and* the manual loop did not run
        out of its iteration budget without ``terminate()`` returning
        ``done=True``.

        The second clause matters because :attr:`RESULTS.successful`
        is the *default* termination_code on a fresh ``SLSQPState``
        (it means "no failure has latched yet") — so a truncated run
        with no failure flag set will report ``successful`` even
        though it never actually converged.
        """
        if self.max_steps_reached:
            return False
        try:
            promoted = (
                self.final_result
                if isinstance(self.final_result, RESULTS)
                else RESULTS.promote(self.final_result)
            )
            return bool(promoted == RESULTS.successful)
        except Exception:  # pragma: no cover  -- defensive
            return False


__all__ = [
    "StepSummary",
    "DebugRunResult",
]
