"""Tests for the best-iterate divergence rollback in :class:`SLSQP`.

Two layers of behaviour are exercised:

1. ``terminate()`` routes ``state.diverging`` to
   ``RESULTS.nonlinear_divergence``.
2. ``step()`` detects merit blowup (or NaN/Inf merit), increments the
   blowup counter, and after ``divergence_patience`` consecutive
   blowups overwrites the returned iterate with ``state.best_x`` and
   latches ``state.diverging``.

The setup deliberately uses NaN injection in the objective rather than
penalty-driven merit growth so the test does not depend on the
adaptive penalty schedule (which can also legitimately inflate the
merit by an order of magnitude on the first few SQP steps).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from slsqp_jax import SLSQP, MinresQLPSolver, get_diagnostics

jax.config.update("jax_enable_x64", True)


def _run_loop(solver, objective, x0, args=None, max_steps=None):
    if max_steps is None:
        max_steps = solver.max_steps
    state = solver.init(objective, x0, args, {}, None, None, frozenset())
    y = x0
    last_result = optx.RESULTS.successful
    for _ in range(max_steps):
        done, last_result = solver.terminate(objective, y, args, {}, state, frozenset())
        if done:
            break
        y, state, _ = solver.step(objective, y, args, {}, state, frozenset())
    return y, state, last_result


# ---------------------------------------------------------------------------
# Routing: ``state.diverging`` -> ``RESULTS.nonlinear_divergence``
# ---------------------------------------------------------------------------


class TestTerminateRoutesDiverging:
    """``terminate()`` must report ``done=True`` and route to
    ``nonlinear_divergence`` whenever ``state.diverging`` is latched,
    regardless of whether stationarity / feasibility happen to hold.
    """

    def test_diverging_flag_routes_to_nonlinear_divergence(self):
        def objective(x, args):
            return jnp.sum(x**2), None

        solver = SLSQP(atol=1e-8, max_steps=10)
        x0 = jnp.array([1.0, -2.0])
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # Manually latch the divergence flag (this is what step() does
        # once the merit-blowup patience runs out).
        state = eqx.tree_at(
            lambda s: s.diverging,
            state,
            jnp.array(True),
        )

        done, result = solver.terminate(objective, x0, None, {}, state, frozenset())
        assert bool(done)
        assert result == optx.RESULTS.nonlinear_divergence


# ---------------------------------------------------------------------------
# End-to-end: NaN merit triggers rollback
# ---------------------------------------------------------------------------


class TestNanMeritRollsBackToBest:
    """An objective that returns NaN once the iterate strays far from
    the origin must:

    * make the merit non-finite, so ``blowup_now`` fires;
    * after ``divergence_patience`` consecutive non-finite merits,
      latch ``state.diverging`` and roll back ``y`` to ``state.best_x``;
    * leave the optimistix result code at ``nonlinear_divergence``;
    * mark ``divergence_triggered=True`` and ``n_divergence_blowups``
      at least equal to ``divergence_patience``.
    """

    def _make_problem(self, nan_radius=5.0):
        def objective(x, args):
            f = -jnp.sum(x**2)  # gradient pushes outward
            f = f + jnp.where(jnp.sum(x**2) > nan_radius**2, jnp.nan, 0.0)
            return f, None

        return objective

    @pytest.mark.slow
    def test_rollback_to_best_x(self):
        objective = self._make_problem(nan_radius=5.0)
        solver = SLSQP(
            atol=1e-8,
            max_steps=30,
            divergence_factor=10.0,
            divergence_patience=3,
        )
        x0 = jnp.array([0.5, -0.5])
        y, state, result = _run_loop(solver, objective, x0)

        # Sanity: divergence fired, not "max steps".
        assert bool(state.diverging), (
            "expected divergence detector to latch, but state.diverging is False; "
            f"result={result}, blowup_count={int(state.blowup_count)}, "
            f"step_count={int(state.step_count)}"
        )
        assert result == optx.RESULTS.nonlinear_divergence

        # Returned iterate must be the best-merit one (which has finite
        # merit by construction — we never accept a NaN as an
        # "improvement").
        np.testing.assert_array_equal(np.asarray(y), np.asarray(state.best_x))
        assert jnp.all(jnp.isfinite(y)), (
            f"rollback should produce a finite iterate, got {y}"
        )

        diag = get_diagnostics(state)
        assert bool(diag.divergence_triggered)
        assert int(diag.n_divergence_blowups) >= solver.divergence_patience


# ---------------------------------------------------------------------------
# Negative test: well-behaved problem must not trip the guardrail
# ---------------------------------------------------------------------------


class TestBenignProblemDoesNotDiverge:
    """On standard test problems the merit-blowup detector must stay
    quiet.  Otherwise the guardrail would either degrade convergence
    (by truncating successful runs) or produce confusing
    ``nonlinear_divergence`` codes when the solver is in fact
    converging.
    """

    def test_simple_quadratic_does_not_trigger_divergence(self):
        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        solver = SLSQP(atol=1e-8, max_steps=50)
        x0 = jnp.array([3.0, -2.0, 0.5])
        y, state, result = _run_loop(solver, objective, x0)

        assert not bool(state.diverging)
        diag = get_diagnostics(state)
        assert not bool(diag.divergence_triggered)
        assert int(diag.n_divergence_blowups) == 0
        # Sanity: the run actually converged.
        assert result == optx.RESULTS.successful
        np.testing.assert_allclose(np.asarray(y), [1.0, 1.0, 1.0], atol=1e-5)

    @pytest.mark.slow
    def test_constrained_problem_does_not_trigger_divergence(self):
        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 3.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.zeros(3)
        y, state, result = _run_loop(solver, objective, x0)

        assert not bool(state.diverging)
        diag = get_diagnostics(state)
        assert not bool(diag.divergence_triggered)
        assert result == optx.RESULTS.successful
        np.testing.assert_allclose(float(jnp.sum(y)), 3.0, atol=1e-6)

    def test_minres_qlp_inner_solver_quiet_on_benign_run(self):
        """Smoke test that the new diagnostic accumulators
        (``n_proj_refinements``, ``max_proj_residual``) do not blow up
        on a normal MinresQLPSolver run.
        """

        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 3.0])

        solver = SLSQP(
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            inner_solver=MinresQLPSolver(max_iter=200, tol=1e-10),
        )
        x0 = jnp.zeros(3)
        _, state, result = _run_loop(solver, objective, x0)

        diag = get_diagnostics(state)
        assert int(diag.n_proj_refinements) >= 0
        assert float(diag.max_proj_residual) >= 0.0
        assert jnp.isfinite(diag.max_proj_residual)
        assert result == optx.RESULTS.successful
