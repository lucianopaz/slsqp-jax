"""Tests for :func:`slsqp_jax.diagnostic_run` and :func:`diagnose_minimize_like_scipy`.

Validates the zero-refactor opt-in surface: the context manager
intercepts ``optimistix.minimise`` calls inside the ``with`` block,
re-routes SLSQP-flavoured ones through the manual debug loop, returns
a real :class:`optimistix.Solution` so user wrapper code is undisturbed,
and exposes the captured runs / reports on the yielded
:class:`DiagnosticContext`.

The tests cover:

* happy-path interception of a :func:`slsqp_jax.minimize_like_scipy`
  call (Rosenbrock, fast-converging),
* non-SLSQP solver passthrough (``solver`` is anything else),
* the re-entrancy guard (nested ``with`` raises),
* ``throw=True`` propagation: the run is *still* captured even when
  the patched ``optx.minimise`` raises,
* the three ``print_on_exit`` modes,
* the import-binding-leak ``UserWarning``,
* :func:`diagnose_minimize_like_scipy` drop-in equivalence with the
  context manager.
"""

from __future__ import annotations

import io
import warnings

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import slsqp_jax
from slsqp_jax import (
    DiagnosticContext,
    diagnose_minimize_like_scipy,
    diagnostic_run,
    minimize_like_scipy,
)
from slsqp_jax.diagnostics.records import DebugRunResult
from slsqp_jax.diagnostics.report import DebugReport

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------


def _quadratic(x):
    """A trivially convergent quadratic: minimum at (1, 2.5)."""
    return (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2


def _rosenbrock(x):
    """4-variable Rosenbrock — used when we need a problem that
    *cannot* converge in 1-2 SLSQP iterations so the budget-exhaustion
    paths can be exercised reliably."""
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


@pytest.fixture
def quad_x0():
    return jnp.array([0.0, 0.0])


@pytest.fixture
def rosenbrock_x0():
    return jnp.array([-1.2, 1.0, -0.5, 1.5])


# ---------------------------------------------------------------------------
# Happy path: SLSQP interception
# ---------------------------------------------------------------------------


def test_diagnostic_run_intercepts_minimize_like_scipy(quad_x0):
    """The context manager must intercept the ``optx.minimise`` call inside
    ``minimize_like_scipy`` and stash exactly one run + report."""
    out = io.StringIO()
    with diagnostic_run(print_on_exit="auto", output=out) as ctx:
        sol = minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})

    assert isinstance(ctx, DiagnosticContext)
    assert ctx.intercepted_calls == 1
    assert ctx.passthrough_calls == 0
    assert len(ctx.runs) == 1 == len(ctx.reports)
    assert isinstance(ctx.runs[0], DebugRunResult)
    assert isinstance(ctx.reports[0], DebugReport)

    # The returned Solution must be a real optimistix.Solution that
    # solved the problem to convergence (within the loose tolerance
    # SLSQP uses).
    assert isinstance(sol, optx.Solution)
    assert jnp.allclose(sol.value, jnp.array([1.0, 2.5]), atol=1e-4)

    # ``print_on_exit="auto"`` on a clean success: only the trailer
    # prints, no per-run report.
    rendered = out.getvalue()
    assert "captured 1 run(s), 0 failing" in rendered
    assert "SLSQP-JAX Debug Report" not in rendered


def test_diagnostic_run_returned_solution_has_stats(quad_x0):
    """The intercepted ``Solution.stats`` must come from
    :meth:`SLSQP.postprocess`, exactly like the production path."""
    with diagnostic_run(print_on_exit="never") as ctx:
        sol = minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})

    assert ctx.intercepted_calls == 1
    # ``slsqp_result`` is the granular RESULTS code threaded through
    # postprocess; absence here would mean we bypassed postprocess.
    assert "slsqp_result" in sol.stats
    assert "num_steps" in sol.stats
    assert "final_objective" in sol.stats


# ---------------------------------------------------------------------------
# Passthrough: non-SLSQP solver
# ---------------------------------------------------------------------------


def test_diagnostic_run_passthrough_for_non_slsqp_solver(quad_x0):
    """``optx.minimise`` calls with a non-SLSQP solver must run unchanged."""
    other_solver = optx.BFGS(rtol=1e-6, atol=1e-6)

    def fn(x, args):
        return _quadratic(x)

    with diagnostic_run(print_on_exit="never") as ctx:
        sol = optx.minimise(fn, other_solver, quad_x0, max_steps=100, throw=False)

    assert ctx.intercepted_calls == 0
    assert ctx.passthrough_calls == 1
    assert len(ctx.runs) == 0
    assert isinstance(sol, optx.Solution)
    # BFGS still solves a quadratic to the optimum.
    assert jnp.allclose(sol.value, jnp.array([1.0, 2.5]), atol=1e-4)


# ---------------------------------------------------------------------------
# Re-entrancy guard
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:slsqp_jax.diagnostic_run intercepted 0")
def test_diagnostic_run_is_not_reentrant():
    """Nested ``with diagnostic_run(): ...`` must raise ``RuntimeError``."""
    with diagnostic_run(print_on_exit="never"):
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with diagnostic_run(print_on_exit="never"):
                pass


@pytest.mark.filterwarnings("ignore:slsqp_jax.diagnostic_run intercepted 0")
def test_diagnostic_run_releases_lock_on_exception():
    """An exception inside the ``with`` block must still release the
    re-entrancy lock so a subsequent ``diagnostic_run`` can enter."""

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with diagnostic_run(print_on_exit="never"):
            raise _Boom("user code blew up")

    # Lock released — second context can enter without RuntimeError.
    with diagnostic_run(print_on_exit="never") as ctx:
        assert ctx.intercepted_calls == 0


# ---------------------------------------------------------------------------
# ``throw=True`` propagation
# ---------------------------------------------------------------------------


def test_diagnostic_run_throw_true_still_captures_run(rosenbrock_x0):
    """When the underlying call raises (because the solver did not
    converge AND ``throw=True``), the run + report must already be
    stashed on the context so the user can inspect them after the
    ``except``.  Rosenbrock with ``maxiter=2`` reliably exhausts the
    budget."""

    captured_runs: list[DebugRunResult] = []
    captured_reports: list[DebugReport] = []
    with pytest.raises(Exception):  # noqa: BLE001 — optx wraps RESULTS errors
        with diagnostic_run(print_on_exit="never") as ctx:
            try:
                minimize_like_scipy(
                    _rosenbrock,
                    rosenbrock_x0,
                    options={"maxiter": 2},
                    throw=True,
                )
            finally:
                captured_runs.extend(ctx.runs)
                captured_reports.extend(ctx.reports)

    # The stash happened before the exception unwound the context.
    assert len(captured_runs) == 1
    assert len(captured_reports) == 1
    assert not captured_runs[0].terminated_successfully


def test_diagnostic_run_throw_false_returns_failed_solution(rosenbrock_x0):
    """``throw=False`` must keep the captured failure surface and not raise."""
    with diagnostic_run(print_on_exit="never") as ctx:
        sol = minimize_like_scipy(
            _rosenbrock,
            rosenbrock_x0,
            options={"maxiter": 2},
            throw=False,
        )

    assert ctx.intercepted_calls == 1
    assert ctx.n_failing == 1
    assert isinstance(sol, optx.Solution)
    assert sol.result != optx.RESULTS.successful


# ---------------------------------------------------------------------------
# print_on_exit modes
# ---------------------------------------------------------------------------


def test_print_on_exit_never_suppresses_all_output(quad_x0):
    out = io.StringIO()
    with diagnostic_run(print_on_exit="never", output=out) as ctx:
        minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})

    assert ctx.intercepted_calls == 1
    assert out.getvalue() == ""


def test_print_on_exit_always_prints_every_report(quad_x0):
    out = io.StringIO()
    with diagnostic_run(print_on_exit="always", output=out) as ctx:
        minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})

    assert ctx.intercepted_calls == 1
    rendered = out.getvalue()
    # The full report header is in the output now (unlike ``auto``).
    assert "SLSQP-JAX Debug Report" in rendered
    assert "captured 1 run(s)" in rendered


def test_print_on_exit_auto_prints_only_failing(rosenbrock_x0):
    out = io.StringIO()
    with diagnostic_run(print_on_exit="auto", output=out) as ctx:
        minimize_like_scipy(
            _rosenbrock,
            rosenbrock_x0,
            options={"maxiter": 2},
            throw=False,
        )

    assert ctx.intercepted_calls == 1
    assert ctx.n_failing == 1
    rendered = out.getvalue()
    # The failing run printed its full report.
    assert "SLSQP-JAX Debug Report" in rendered
    assert "captured 1 run(s), 1 failing" in rendered


def test_print_on_exit_invalid_mode_raises(quad_x0):
    with pytest.raises(ValueError, match="print_on_exit must be one of"):
        with diagnostic_run(print_on_exit="bogus"):  # type: ignore[arg-type]
            minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})


# ---------------------------------------------------------------------------
# Import-binding leak warning
# ---------------------------------------------------------------------------


def test_diagnostic_run_warns_when_no_calls_intercepted():
    """Empty context (no ``optx.minimise`` call inside) should emit the
    documented ``UserWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with diagnostic_run(print_on_exit="never"):
            pass

    matching = [w for w in caught if "intercepted 0 calls" in str(w.message)]
    assert len(matching) == 1
    assert issubclass(matching[0].category, UserWarning)


def test_diagnostic_run_does_not_warn_when_calls_intercepted(quad_x0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with diagnostic_run(print_on_exit="never") as ctx:
            minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})

    assert ctx.intercepted_calls == 1
    matching = [w for w in caught if "intercepted 0 calls" in str(w.message)]
    assert matching == []


# ---------------------------------------------------------------------------
# diagnose_minimize_like_scipy drop-in
# ---------------------------------------------------------------------------


def test_diagnose_minimize_like_scipy_returns_sol_and_report(quad_x0):
    sol, report = diagnose_minimize_like_scipy(
        _quadratic, quad_x0, options={"maxiter": 50}
    )
    assert isinstance(sol, optx.Solution)
    assert isinstance(report, DebugReport)
    assert jnp.allclose(sol.value, jnp.array([1.0, 2.5]), atol=1e-4)
    # The report must be reachable through ``sol.stats`` for callers
    # that prefer the optimistix Solution surface.
    assert sol.stats.get("diagnostic_report") is report


def test_diagnose_minimize_like_scipy_matches_minimize_like_scipy(quad_x0):
    """Running ``diagnose_minimize_like_scipy`` and ``minimize_like_scipy``
    on the same problem must yield identical ``sol.value`` (modulo
    floating-point noise) — proving the diagnostic loop is a faithful
    re-execution and not a different algorithm."""
    sol_diag, _report = diagnose_minimize_like_scipy(
        _quadratic, quad_x0, options={"maxiter": 50}
    )
    sol_prod = minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})

    assert jnp.allclose(sol_diag.value, sol_prod.value, atol=1e-6)
    assert sol_diag.result == sol_prod.result


def test_diagnose_minimize_like_scipy_does_not_leak_lock(quad_x0):
    """``diagnose_minimize_like_scipy`` opens an inner ``diagnostic_run``;
    after it returns, the global lock must be released so a subsequent
    explicit ``diagnostic_run`` can still enter."""
    sol, _report = diagnose_minimize_like_scipy(
        _quadratic, quad_x0, options={"maxiter": 50}
    )
    assert isinstance(sol, optx.Solution)

    with diagnostic_run(print_on_exit="never") as ctx:
        minimize_like_scipy(_quadratic, quad_x0, options={"maxiter": 50})
    assert ctx.intercepted_calls == 1


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_top_level_exports_present():
    assert hasattr(slsqp_jax, "diagnostic_run")
    assert hasattr(slsqp_jax, "diagnose_minimize_like_scipy")
    assert hasattr(slsqp_jax, "DiagnosticContext")
