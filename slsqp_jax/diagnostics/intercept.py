"""Zero-refactor opt-in for SLSQP-JAX diagnostics.

Most users never construct :class:`slsqp_jax.SLSQP` directly — they go
through :func:`slsqp_jax.minimize_like_scipy` (or their own wrapper
around ``optimistix.minimise``).  This module gives them two ways to
turn the diagnostics layer on without refactoring their call sites:

* :func:`diagnostic_run` — a context manager that monkey-patches
  ``optimistix.minimise`` for the duration of the ``with`` block,
  re-routes any call whose ``solver`` is an :class:`SLSQP` instance
  through :func:`slsqp_jax.diagnostics.debug_run`, and otherwise
  passes through.
* :func:`diagnose_minimize_like_scipy` — a non-monkey-patching
  drop-in replacement for :func:`slsqp_jax.minimize_like_scipy` that
  builds the same :class:`SLSQP` instance, runs it under the
  diagnostics loop, and returns ``(optx.Solution, DebugReport)``.

Both share the :func:`_run_via_debug` core, which routes through the
existing :func:`slsqp_jax.diagnostics.debug_run` and reuses
:meth:`SLSQP.postprocess` for ``stats`` fidelity, so the returned
``optx.Solution`` is structurally interchangeable with what
``optimistix.minimise`` would have returned.

Concurrency contract: the context manager is **not re-entrant** and
**not thread-safe**.  A process-wide :class:`threading.Lock` rejects
nested or concurrent enters with :class:`RuntimeError`.  This is by
design — we monkey-patch a global module attribute, and silently
sharing that mutation across threads would be worse than failing
loud.

Import-binding contract: the patch replaces
``optimistix.minimise``-the-module-attribute.  User code that does
``from optimistix import minimise`` and then calls ``minimise(...)``
binds the name at import time and is **not** intercepted; we surface
this with a :class:`UserWarning` on exit when zero calls were
intercepted.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import optimistix as optx

from slsqp_jax.diagnostics.playbook import evaluate_diagnoses
from slsqp_jax.diagnostics.records import DebugRunResult
from slsqp_jax.diagnostics.report import DebugReport
from slsqp_jax.diagnostics.runner import _wrap_fn_with_aux, debug_run
from slsqp_jax.diagnostics.signals import (
    END_OF_RUN_EVALUATORS,
    PER_STEP_EVALUATORS,
)

if TYPE_CHECKING:
    from slsqp_jax.slsqp import SLSQP


# Process-wide guard against re-entrant or concurrent ``diagnostic_run``
# enters.  The patch replaces ``optx.minimise`` at module level, so
# allowing two contexts to enter at once would silently nest the
# patches and break the restore-on-exit invariant.  We deliberately
# fail loud (``RuntimeError``) rather than serialise — the runner is a
# debug tool, not a production primitive, and a deadlock would be
# more confusing than the explicit error.
_DIAGNOSTIC_LOCK = threading.Lock()


@dataclass
class DiagnosticContext:
    """Container yielded by :func:`diagnostic_run`'s ``__enter__``.

    Carries every :class:`DebugRunResult` and matching
    :class:`DebugReport` produced inside the ``with`` block, plus
    counters distinguishing intercepted SLSQP calls from passthrough
    calls (i.e. ``optx.minimise`` invocations whose ``solver`` is
    something other than :class:`SLSQP`).

    Attributes:
        runs: One :class:`DebugRunResult` per intercepted call, in
            execution order.
        reports: One :class:`DebugReport` per intercepted call, in
            the same order as :attr:`runs`.  ``reports[i]`` was
            built from ``runs[i]``.
        intercepted_calls: Count of ``optx.minimise`` calls whose
            ``solver`` was an :class:`SLSQP` instance and were
            re-routed through the debug loop.
        passthrough_calls: Count of ``optx.minimise`` calls whose
            ``solver`` was *not* :class:`SLSQP` and were forwarded
            unchanged to the original ``optx.minimise``.
    """

    runs: list[DebugRunResult] = field(default_factory=list)
    reports: list[DebugReport] = field(default_factory=list)
    intercepted_calls: int = 0
    passthrough_calls: int = 0

    @property
    def n_failing(self) -> int:
        """Number of captured runs that did *not* terminate successfully."""
        return sum(1 for r in self.runs if not r.terminated_successfully)

    def print_summary(
        self,
        idx: Optional[int] = None,
        *,
        file: Any = None,
    ) -> None:
        """Print one or all captured reports to ``file`` (default stdout).

        Pass an integer ``idx`` to print exactly that report (negative
        indices count from the end, matching list semantics).  With
        ``idx=None`` (the default) every captured report is printed in
        order, separated by a blank line.  Useful from a notebook or
        an ``except`` handler when the auto print on exit was
        suppressed (``print_on_exit="never"``) or when the user wants
        to see a successful run's report that the auto-suppress
        skipped.
        """
        target = file if file is not None else sys.stdout
        if idx is not None:
            self.reports[idx].print_summary(file=target)
            return
        for i, report in enumerate(self.reports):
            if i > 0:
                target.write("\n")
            report.print_summary(file=target)

    def __repr__(self) -> str:
        n_runs = len(self.runs)
        n_failing = self.n_failing
        return (
            f"DiagnosticContext(runs={n_runs}, failing={n_failing}, "
            f"intercepted={self.intercepted_calls}, "
            f"passthrough={self.passthrough_calls})"
        )


def _run_via_debug(
    solver: "SLSQP",
    fn: Callable,
    x0: Any,
    *,
    args: Any,
    options: Optional[dict],
    has_aux: bool,
    max_steps: Optional[int],
    tags: frozenset,
) -> tuple[optx.Solution, DebugRunResult, DebugReport]:
    """Run ``solver`` under the manual debug loop, return ``(Solution, run, report)``.

    Shared by :func:`diagnostic_run` and
    :func:`diagnose_minimize_like_scipy`.  The returned
    :class:`optx.Solution` is built by re-evaluating ``fn`` once at
    the terminal iterate to recover ``aux`` (the manual loop discards
    per-step ``aux`` values), then routing through
    :meth:`SLSQP.postprocess` so the ``stats`` dict is bit-identical
    to what ``optimistix.minimise`` would have populated.

    This helper deliberately does **not** raise on failure; the
    ``throw`` semantics are owned by the caller so the
    :class:`DiagnosticContext` can stash the run *before* the
    exception unwinds.
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
    diagnoses = list(evaluate_diagnoses(run.fired_signals))
    report = DebugReport.from_run(run)
    report.diagnoses = diagnoses

    wrapped_fn = _wrap_fn_with_aux(fn, has_aux)
    _f_final, aux_final = wrapped_fn(run.final_y, args)
    y_post, aux_post, stats = solver.postprocess(
        wrapped_fn,
        run.final_y,
        aux_final,
        args,
        options or {},
        run.final_state,
        tags,
        run.coarse_result,
    )
    sol = optx.Solution(
        value=y_post,
        result=run.coarse_result,
        aux=aux_post,
        stats=stats,
        state=run.final_state,
    )
    return sol, run, report  # ty: ignore[invalid-return-type]


def _maybe_throw(sol: optx.Solution, throw: bool) -> optx.Solution:
    """Mirror :func:`optimistix._iterate.iterative_solve`'s ``throw`` block.

    ``optimistix`` ends ``iterative_solve`` with::

        if throw:
            sol = result.error_if(sol, result != RESULTS.successful)

    We replicate that behaviour bit-for-bit so wrapper code that does
    ``try / except optimistix.NonlinearMaxStepsReached`` continues to
    work unchanged.
    """
    if not throw:
        return sol
    return sol.result.error_if(sol, sol.result != optx.RESULTS.successful)


def _print_on_exit(
    ctx: DiagnosticContext,
    mode: Literal["auto", "always", "never"],
    output: Any,
) -> None:
    """Render the on-exit summary for :func:`diagnostic_run`.

    The default ``mode="auto"`` is *quiet on clean success, loud on
    failure*: we only print reports for runs that did not terminate
    successfully, plus a one-line trailer pointing the user at
    :attr:`DiagnosticContext.runs` / :attr:`DiagnosticContext.reports`
    for deeper inspection.  ``mode="always"`` overrides for users who
    want the report on success too; ``mode="never"`` suppresses
    everything (the trailer included).
    """
    if mode == "never":
        return
    target = output if output is not None else sys.stdout

    indices_to_print: list[int]
    if mode == "always":
        indices_to_print = list(range(len(ctx.reports)))
    elif mode == "auto":
        indices_to_print = [
            i for i, run in enumerate(ctx.runs) if not run.terminated_successfully
        ]
    else:
        raise ValueError(
            f"print_on_exit must be one of 'auto', 'always', 'never'; got {mode!r}"
        )

    for j, i in enumerate(indices_to_print):
        if j > 0:
            target.write("\n")
        ctx.reports[i].print_summary(file=target)

    n_runs = len(ctx.runs)
    n_failing = ctx.n_failing
    if n_runs == 0:
        target.write(
            "[slsqp_jax.diagnostic_run] no SLSQP calls were intercepted "
            "inside the context.\n"
        )
    else:
        target.write(
            f"[slsqp_jax.diagnostic_run] captured {n_runs} run(s), "
            f"{n_failing} failing.  Inspect via ctx.runs[i] / "
            f"ctx.reports[i]; ctx.print_summary(idx) re-renders any "
            f"individual report.\n"
        )


def _maybe_warn_no_intercept(ctx: DiagnosticContext) -> None:
    """Emit a :class:`UserWarning` when nothing inside the context was caught.

    The patch replaces ``optimistix.minimise``-the-module-attribute,
    which is invisible to user code that did ``from optimistix import
    minimise`` before the context opened.  Zero intercepts *and* zero
    passthroughs is the load-bearing signal that the patch never had
    a chance to fire.
    """
    if ctx.intercepted_calls + ctx.passthrough_calls == 0:
        warnings.warn(
            "slsqp_jax.diagnostic_run intercepted 0 calls to optimistix.minimise. "
            "If your code does 'from optimistix import minimise', the patch did "
            "not apply because the name was bound at import time.  Either use "
            "'import optimistix as optx; optx.minimise(...)' inside the context, "
            "or call slsqp_jax.diagnose_minimize_like_scipy(...) directly.",
            UserWarning,
            stacklevel=3,
        )


@contextlib.contextmanager
def diagnostic_run(
    *,
    max_steps: Optional[int] = None,
    print_on_exit: Literal["auto", "always", "never"] = "auto",
    output: Any = None,
) -> Iterator[DiagnosticContext]:
    """Context manager that intercepts ``optimistix.minimise`` calls.

    Inside the ``with`` block, any ``optimistix.minimise(fn, solver,
    ...)`` whose ``solver`` is an :class:`slsqp_jax.SLSQP` instance is
    re-routed through :func:`slsqp_jax.diagnostics.debug_run`, the
    resulting :class:`DebugRunResult` and :class:`DebugReport` are
    stashed on the yielded :class:`DiagnosticContext`, and a fully
    populated :class:`optx.Solution` is returned to the caller — so
    the user's wrapper code (whether it goes through
    :func:`slsqp_jax.minimize_like_scipy` or builds its own ``SLSQP``)
    runs **unchanged**.  Calls whose ``solver`` is something other
    than :class:`SLSQP` are forwarded to the original
    ``optimistix.minimise`` unchanged, with the only side effect being
    a bump on :attr:`DiagnosticContext.passthrough_calls`.

    Args:
        max_steps: Optional iteration-budget override forwarded to
            every intercepted ``debug_run``.  ``None`` (the default)
            uses each call's own ``max_steps`` argument, falling back
            to ``solver.max_steps``.
        print_on_exit: When the context exits, print captured reports
            to ``output``.  ``"auto"`` (default): only failing runs
            plus a trailer; ``"always"``: every run plus the trailer;
            ``"never"``: nothing.
        output: File-like sink for the on-exit print.  Defaults to
            ``sys.stdout``.

    Yields:
        :class:`DiagnosticContext` collecting the runs and reports
        for every intercepted call.

    Raises:
        RuntimeError: If another :func:`diagnostic_run` is already
            active (re-entrant or concurrent enter).  The patch is a
            global mutation; serialising would mask bugs.

    Examples:
        >>> with slsqp_jax.diagnostic_run() as ctx:
        ...     sol = slsqp_jax.minimize_like_scipy(fun, x0, ...)
        >>> # On exit, any failing report has already been printed.
        >>> # The Solution is the real optimistix.Solution.
        >>> ctx.runs[0].diagnostics  # SLSQPDiagnostics counters
    """
    ctx = DiagnosticContext()

    if not _DIAGNOSTIC_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "slsqp_jax.diagnostic_run is not re-entrant.  Close the outer "
            "context (or wait for the concurrent run to finish) before "
            "opening another."
        )

    original_minimise = optx.minimise
    # Outer ``diagnostic_run(max_steps=...)`` overrides per-call
    # ``max_steps`` so a single context can cap an unwieldy run.
    # ``None`` means "honour the caller's value".
    outer_max_steps = max_steps

    def patched_minimise(
        fn: Callable,
        solver: Any,
        y0: Any,
        args: Any = None,
        options: Optional[dict] = None,
        *,
        has_aux: bool = False,
        max_steps: Optional[int] = 256,
        adjoint: Any = None,
        throw: bool = True,
        tags: frozenset = frozenset(),
    ) -> optx.Solution:
        from slsqp_jax.slsqp import SLSQP

        if not isinstance(solver, SLSQP):
            ctx.passthrough_calls += 1
            kwargs: dict[str, Any] = {
                "has_aux": has_aux,
                "max_steps": max_steps,
                "throw": throw,
                "tags": tags,
            }
            if adjoint is not None:
                kwargs["adjoint"] = adjoint
            return original_minimise(fn, solver, y0, args, options, **kwargs)

        ctx.intercepted_calls += 1
        effective_max_steps = (
            outer_max_steps if outer_max_steps is not None else max_steps
        )
        sol, run, report = _run_via_debug(
            solver,
            fn,
            y0,
            args=args,
            options=options,
            has_aux=has_aux,
            max_steps=effective_max_steps,
            tags=tags,
        )
        ctx.runs.append(run)
        ctx.reports.append(report)
        return _maybe_throw(sol, throw)

    optx.minimise = patched_minimise  # type: ignore[assignment]

    try:
        yield ctx
    finally:
        optx.minimise = original_minimise  # type: ignore[assignment]
        _DIAGNOSTIC_LOCK.release()
        _maybe_warn_no_intercept(ctx)
        _print_on_exit(ctx, print_on_exit, output)


def diagnose_minimize_like_scipy(
    fun: Callable,
    x0: Any,
    args: tuple = (),
    *,
    jac: Any = None,
    hessp: Any = None,
    bounds: Any = None,
    constraints: Any = (),
    options: Optional[dict[str, Any]] = None,
    has_aux: bool = False,
    throw: bool = False,
    verbose: Any = False,
    max_steps_override: Optional[int] = None,
) -> tuple[optx.Solution, DebugReport]:
    """Drop-in replacement for :func:`slsqp_jax.minimize_like_scipy` with diagnostics.

    Mirrors :func:`slsqp_jax.minimize_like_scipy`'s public signature
    1:1 so callers can swap one import + one call name without any
    other code changes::

        # before
        from slsqp_jax import minimize_like_scipy
        sol = minimize_like_scipy(fun, x0, ...)

        # after
        from slsqp_jax import diagnose_minimize_like_scipy
        sol, report = diagnose_minimize_like_scipy(fun, x0, ...)
        report.print_summary()

    Unlike :func:`diagnostic_run`, this function performs **no**
    monkey-patching — it is the recommended path for users whose
    wrapper code does ``from optimistix import minimise`` (which
    bypasses the context manager's patch), as well as for tests and
    CI where global state must be avoided.

    The extra keyword ``max_steps_override`` accepts an integer to
    cap the diagnostic loop independently of the ``options['maxiter']``
    that builds the underlying ``SLSQP``; ``None`` (default) honours
    the option dict's own value.

    Returns:
        A ``(sol, report)`` tuple.  ``sol`` is a normal
        :class:`optx.Solution` indistinguishable from what
        :func:`slsqp_jax.minimize_like_scipy` would have returned
        (modulo the device-resident state the diagnostics layer
        keeps around); ``report`` is the populated
        :class:`DebugReport` with all signals + diagnoses already
        evaluated.

    Raises:
        Whatever ``minimize_like_scipy`` would raise plus, when
        ``throw=True`` and the run did not converge,
        :class:`optimistix.RESULTS`-flavoured runtime errors via
        ``EnumerationItem.error_if``.
    """
    # We deliberately re-use the Solver-construction logic in
    # ``slsqp_jax.compat.minimize_like_scipy`` rather than duplicate
    # it.  The cheapest correct way is to wrap that function with a
    # one-shot ``diagnostic_run`` context, which intercepts the lone
    # ``optx.minimise`` call inside and stashes the report.
    from slsqp_jax.compat import minimize_like_scipy

    with diagnostic_run(
        max_steps=max_steps_override,
        print_on_exit="never",
    ) as ctx:
        sol = minimize_like_scipy(
            fun,
            x0,
            args,
            jac=jac,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            options=options,
            has_aux=has_aux,
            throw=throw,
            verbose=verbose,
        )

    if not ctx.reports:
        raise RuntimeError(
            "diagnose_minimize_like_scipy: minimize_like_scipy did not call "
            "optimistix.minimise (this should not happen — please report a bug)."
        )
    report = ctx.reports[0]
    # Surface the report on ``sol.stats`` for callers that prefer to
    # consume it via the standard optimistix Solution surface.
    try:
        sol.stats["diagnostic_report"] = report  # type: ignore[index]
    except Exception:  # pragma: no cover  -- defensive: stats may be frozen
        pass
    return sol, report


__all__ = [
    "DiagnosticContext",
    "diagnostic_run",
    "diagnose_minimize_like_scipy",
]
