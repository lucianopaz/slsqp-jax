"""Manual-loop debug runner and ad-hoc state-snapshot helper.

This module replaces the on-device ``jax.lax.while_loop`` of
``optimistix.iterative_solve`` with a host-driven Python ``for`` loop
calling ``jit(step)``.  Every iteration ends with a single
device → host transfer that materialises a :class:`StepSummary` from
the live ``SLSQPState``.  The signal pipeline (Phase 2) layers
per-step + end-of-run evaluators on top of this skeleton.

Performance contract (load-bearing): the runner is a *debug* tool,
not a production loop.  On GPU the host sync per iteration can turn a
3 s production run into a 30-60 s diagnose run.  That cost is
acceptable because the alternative is the user reading verbose output
by eye for hours.  See ``AGENTS.md`` and the plan's "Performance
contract" section for the discipline that keeps it bounded (cheap
predicates over scalar :class:`StepSummary` fields, expensive
``build_artifacts`` only invoked when a signal actually fires).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import jax

from slsqp_jax.diagnostics.records import DebugRunResult, StepSummary
from slsqp_jax.diagnostics.signals import EvalContext
from slsqp_jax.results import RESULTS

if TYPE_CHECKING:
    from slsqp_jax.slsqp import SLSQP
    from slsqp_jax.state import SLSQPState


def _wrap_fn_with_aux(fn: Callable, has_aux: bool) -> Callable:
    """Coerce ``fn`` to the ``(x, args) -> (value, aux)`` signature.

    ``SLSQP`` (mirroring ``optimistix.minimise(... has_aux=True)``)
    always calls the objective as ``f, aux = fn(x, args)``.  Users
    coming from a SciPy-style ``f = fn(x, *args)`` interface may not
    have packaged their function that way; this wrapper bridges the
    two without requiring them to think about it.
    """
    if has_aux:
        return fn

    def _wrapped(x: Any, args: Any) -> tuple[Any, None]:
        return fn(x, args), None

    return _wrapped


def _eval_fn_struct(fn: Callable, x0: Any, args: Any) -> tuple[Any, Any]:
    """Trace ``fn(x0, args)`` once to recover ``(f_struct, aux_struct)``.

    ``SLSQP.init`` requires ``ShapeDtypeStruct`` placeholders matching
    the abstract shape of ``fn``'s return value.  Optimistix derives
    them in its driver before calling ``init``; we replicate that here
    so the runner has the same calling convention as the stock loop.
    """
    return jax.eval_shape(fn, x0, args)


def _resolve_max_steps(solver: "SLSQP", max_steps: Optional[int]) -> int:
    """Pick the iteration budget for the manual loop.

    Defaults to ``solver.max_steps`` (i.e. the same budget
    ``optimistix.minimise`` would honour) so the runner reproduces the
    original run's terminal step count when no override is given.
    """
    if max_steps is None:
        return int(solver.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    return int(max_steps)


def _build_jitted_solver_callbacks(
    solver: "SLSQP",
    fn: Callable,
    f_struct: Any,
    aux_struct: Any,
) -> tuple[Callable, Callable, Callable]:
    """Return jit-compiled ``(init, step, terminate)`` for the manual loop.

    Capturing ``solver`` and ``fn`` inside the closure means each
    invocation of ``debug_run`` triggers a fresh JIT compile (no
    cross-run cache hits).  That is acceptable: the runner is a
    diagnose-then-fix tool, not a hot path.  Re-using a previously
    jitted ``step`` across runs of *different* solvers would require
    retracing anyway because ``solver`` is part of the closure.
    """

    @jax.jit
    def jit_init(x0: Any, args: Any) -> tuple[Any, "SLSQPState"]:
        state = solver.init(fn, x0, args, {}, f_struct, aux_struct, frozenset())
        return x0, state

    @jax.jit
    def jit_step(
        y: Any, args: Any, state: "SLSQPState"
    ) -> tuple[Any, "SLSQPState", Any]:
        return solver.step(fn, y, args, {}, state, frozenset())

    @jax.jit
    def jit_terminate(y: Any, args: Any, state: "SLSQPState") -> tuple[Any, Any]:
        return solver.terminate(fn, y, args, {}, state, frozenset())

    return jit_init, jit_step, jit_terminate


def debug_run(
    solver: "SLSQP",
    fn: Callable,
    x0: Any,
    *,
    args: Any = None,
    max_steps: Optional[int] = None,
    has_aux: bool = False,
    per_step_evaluators: tuple = (),
    end_of_run_evaluators: tuple = (),
) -> DebugRunResult:
    """Run ``solver`` under a manual Python loop and return a
    :class:`DebugRunResult`.

    The loop reproduces what ``optimistix.minimise`` would do,
    iterating ``solver.init`` → ``solver.step`` → ``solver.terminate``
    on each step.  After every step the live ``SLSQPState`` is
    summarised into a :class:`StepSummary` and (when Phase 2 wires
    them) the per-step evaluators in ``per_step_evaluators`` are run
    against ``(state, summary, summaries_so_far)``.  After the loop
    exits, ``end_of_run_evaluators`` are run against
    ``(final_state, summaries, final_result)``.

    Args:
        solver: The :class:`slsqp_jax.SLSQP` instance to run.
        fn: Objective callable.  ``(x, args) -> value`` by default,
            or ``(x, args) -> (value, aux)`` when ``has_aux=True``.
        x0: Initial iterate.
        args: Extra payload threaded through ``fn`` and the constraint
            callables on ``solver``.
        max_steps: Optional iteration budget override.  Defaults to
            ``solver.max_steps``.
        has_aux: Whether ``fn`` returns ``(value, aux)``.
        per_step_evaluators: Tuple of callables
            ``(state, summary, summaries) -> Signal | None`` invoked
            after each step.  Empty by default (Phase 1 ships without
            signals).
        end_of_run_evaluators: Tuple of callables
            ``(final_state, summaries, result) -> Signal | None``
            invoked once after the loop exits.  Empty by default.

    Returns:
        :class:`DebugRunResult` with the per-step ``StepSummary``
        trajectory, the terminal state, the granular and coarse
        termination codes, and any signals fired during the run.
    """
    wrapped_fn = _wrap_fn_with_aux(fn, has_aux)
    f_struct, aux_struct = _eval_fn_struct(wrapped_fn, x0, args)
    jit_init, jit_step, jit_terminate = _build_jitted_solver_callbacks(
        solver, wrapped_fn, f_struct, aux_struct
    )

    budget = _resolve_max_steps(solver, max_steps)

    ctx = EvalContext(
        solver=solver,
        rtol=float(solver.rtol),
        atol=float(solver.atol),
        max_steps=budget,
    )

    y, state = jit_init(x0, args)
    initial_state = state

    summaries: list[StepSummary] = []
    fired: list[Any] = []
    seen_signal_names: set[str] = set()
    prev_state: "SLSQPState" = initial_state

    coarse_result = RESULTS.successful
    terminated_at_step = 0
    max_steps_reached = True

    for k in range(budget):
        y, state, _aux = jit_step(y, args, state)
        summary = StepSummary.from_state(state, prev_state=prev_state)
        summaries.append(summary)

        for evaluator in per_step_evaluators:
            sig = evaluator(ctx, state, summary, summaries)
            if sig is not None and sig.name not in seen_signal_names:
                fired.append(sig)
                seen_signal_names.add(sig.name)

        done, coarse_result = jit_terminate(y, args, state)
        # ``done`` is a 0-d JAX boolean — the host-sync was already
        # paid by ``StepSummary.from_state``, so this read is free
        # in practice (the value is already on the host's L1 path).
        if bool(done):
            terminated_at_step = k
            max_steps_reached = False
            break
        prev_state = state
    else:  # for-else: budget exhausted without `break`
        terminated_at_step = budget - 1

    for evaluator in end_of_run_evaluators:
        sig = evaluator(ctx, state, summaries, coarse_result)
        if sig is not None and sig.name not in seen_signal_names:
            fired.append(sig)
            seen_signal_names.add(sig.name)

    return DebugRunResult(
        solver=solver,
        fn=fn,
        x0=x0,
        args=args,
        has_aux=has_aux,
        final_state=state,
        final_y=y,
        final_result=state.termination_code,
        coarse_result=coarse_result,
        summaries=summaries,
        terminated_at_step=terminated_at_step,
        max_steps_reached=max_steps_reached,
        fired_signals=fired,
    )


def capture_state_at_step(
    solver: "SLSQP",
    fn: Callable,
    x0: Any,
    step: int,
    *,
    args: Any = None,
    has_aux: bool = False,
    expected_summary: Optional[StepSummary] = None,
) -> "SLSQPState":
    """Re-run ``solver`` to step ``step`` and return the live ``SLSQPState``.

    This is the public ad-hoc inspection tool: signals already build
    their artifacts inline at the moment they fire, so most users will
    never call this.  It exists for the case where the user wants to
    poke at a step their signals did not preserve a snapshot of.

    Args:
        solver: The :class:`slsqp_jax.SLSQP` instance to re-run.
        fn: Objective callable (same shape as for :func:`debug_run`).
        x0: Initial iterate.
        step: Target iteration to stop at (1-indexed; ``step=k``
            returns the state immediately after the ``k``-th call to
            ``solver.step``).
        args: Extra payload threaded through ``fn``.
        has_aux: Whether ``fn`` returns ``(value, aux)``.
        expected_summary: Optional :class:`StepSummary` from the
            *original* :func:`debug_run` at the same step.  When
            supplied, the recovered state's summary is hashed and
            compared against it; a mismatch raises ``RuntimeError``.
            This is the load-bearing reproducibility check that
            prevents the tool from silently lying about which iterate
            it is showing.

    Returns:
        The live ``SLSQPState`` immediately after step ``step``.

    Raises:
        ValueError: If ``step`` is non-positive.
        RuntimeError: If ``expected_summary`` is supplied and the
            recovered state's reproducibility digest does not match.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    wrapped_fn = _wrap_fn_with_aux(fn, has_aux)
    f_struct, aux_struct = _eval_fn_struct(wrapped_fn, x0, args)
    jit_init, jit_step, _ = _build_jitted_solver_callbacks(
        solver, wrapped_fn, f_struct, aux_struct
    )

    y, state = jit_init(x0, args)
    prev_state = state
    last_summary: Optional[StepSummary] = None
    for _ in range(step):
        y, state, _aux = jit_step(y, args, state)
        last_summary = StepSummary.from_state(state, prev_state=prev_state)
        prev_state = state

    if expected_summary is not None and last_summary is not None:
        recovered_digest = last_summary.reproducibility_digest()
        expected_digest = expected_summary.reproducibility_digest()
        if recovered_digest != expected_digest:
            diverging = _diff_summaries(last_summary, expected_summary)
            raise RuntimeError(
                "debug-run trajectory is not reproducible: "
                f"recovered digest {recovered_digest!r} != expected "
                f"{expected_digest!r} at step={step}.  Diverging fields: "
                f"{diverging}.  This usually indicates a JAX nondeterminism "
                "(e.g. GPU XLA reductions) between the original "
                "debug_run and this capture_state_at_step call."
            )

    return state


def _diff_summaries(a: StepSummary, b: StepSummary) -> dict[str, tuple[Any, Any]]:
    """Return the fields of ``a`` and ``b`` whose values disagree.

    Used only by :func:`capture_state_at_step` when the
    reproducibility hash check fails, to give the user a concrete
    diagnostic.  Tolerant of NaN/inf comparisons so the diff can be
    computed even on degenerate runs.
    """
    out: dict[str, tuple[Any, Any]] = {}
    import dataclasses

    for f in dataclasses.fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        # NaN-aware equality.
        if isinstance(va, float) and isinstance(vb, float):
            both_nan = (va != va) and (vb != vb)
            if both_nan:
                continue
        if va != vb:
            out[f.name] = (va, vb)
    return out


__all__ = [
    "debug_run",
    "capture_state_at_step",
]
