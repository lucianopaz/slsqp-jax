"""Regression tests for the post-v0.9.2 hardening revert.

These tests pin the behaviour restored from v0.9.2 after the two
hardening commits (``c342509`` "Harden QP convergence" and ``3a6f141``
"LPEC-A determines active bounds as well as inequalities") introduced
production regressions:

1. LPEC-A bound prediction no longer crashes with a scatter dtype
   mismatch under ``jax_enable_x64=False``.
2. The QP ping-pong short-circuit is disabled by default, so
   MINRES-QLP solves cannot leak rogue coupled multipliers into the
   outer Lagrangian gradient / merit penalty.
3. The QP ``final_converged`` flag is independent of transient inner
   CG/MINRES imprecision: a non-strictly-converged inner iterate is
   not promoted to a QP failure.
4. The EXPAND ``working_tol`` ramp follows the v0.9.2 formula
   ``working_tol = base_tol + k * tau``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import optimistix as optx

from slsqp_jax import SLSQP, MinresQLPSolver, ProjectedCGCraig
from slsqp_jax.inner_solver import craig_solve
from slsqp_jax.qp_solver import solve_qp

jax.config.update("jax_enable_x64", True)


def _run_loop(solver, objective, x0, max_steps=None):
    if max_steps is None:
        max_steps = solver.max_steps
    state = solver.init(objective, x0, None, {}, None, None, frozenset())
    y = x0
    last_result = optx.RESULTS.successful
    for _ in range(max_steps):
        done, last_result = solver.terminate(objective, y, None, {}, state, frozenset())
        if done:
            break
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
    return y, state, last_result


# ---------------------------------------------------------------------------
# R1: LPEC-A scatter dtype under x32
# ---------------------------------------------------------------------------


class TestLPECABoundScatterX32:
    """The LPEC-A bound extension must not crash under x32 (the default
    JAX mode in most production notebooks).

    The bug was ``np.array(self._lower_indices)`` yielding ``int64``
    while ``jnp.zeros(n, dtype=bool)`` expected an ``int32`` index for
    the ``scatter`` HLO op, producing an ``s32[] vs s64[]`` accumulator
    mismatch.  Fixed by explicitly casting the index arrays to
    ``jnp.int32`` at the scatter sites in ``solver.py``.

    We run this as a subprocess so the global ``jax_enable_x64=True``
    set at the top of every test module does not leak in.
    """

    def test_compiles_under_x32(self):
        script = textwrap.dedent(
            """
            import jax
            jax.config.update("jax_enable_x64", False)
            import jax.numpy as jnp
            import numpy as np
            import optimistix as optx
            from slsqp_jax import SLSQP

            def objective(x, args):
                return jnp.sum(x ** 2), None

            n = 8
            x0 = jnp.ones(n, dtype=jnp.float32)
            bounds = np.stack([np.full(n, -0.5), np.full(n, 0.5)], axis=1)

            solver = SLSQP(
                atol=1e-6,
                rtol=1e-6,
                max_steps=20,
                bounds=bounds,
                active_set_method="lpeca_init",
                lpeca_predict_bounds=True,
                lpeca_warmup_steps=0,
            )
            sol = optx.minimise(
                objective, solver, x0, has_aux=True, throw=False
            )
            assert jnp.all(jnp.isfinite(sol.value))
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"x32 LPEC-A bound run failed:\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
        assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# R2: MINRES-QLP ping-pong default does not leak rogue multipliers
# ---------------------------------------------------------------------------


class TestMinresQLPPingPongDisabled:
    """The ping-pong detector must be disabled by default so
    MINRES-QLP cannot short-circuit with ill-conditioned coupled
    multipliers that then explode the L1 merit penalty.
    """

    def test_default_threshold_is_effectively_off(self):
        solver = SLSQP(
            atol=1e-6,
            rtol=1e-6,
            max_steps=5,
            inner_solver=MinresQLPSolver(tol=1e-8),
        )
        # The threshold is exposed as a static field; the baseline
        # must be large enough that no real QP ever ping-pongs by
        # default.
        assert solver.qp_ping_pong_threshold >= 2**30

    def test_merit_stays_finite_on_portfolio_shaped_problem(self):
        """Minimise a quadratic subject to a simplex-like constraint
        where active-set oscillations are the historical
        MINRES-QLP failure mode.
        """

        n = 6

        def objective(x, args):
            return 0.5 * jnp.dot(x, x) - jnp.sum(x) * 0.1, None

        x0 = jnp.full(n, 1.0 / n)

        def eq_constraint(x, args):
            return jnp.array([jnp.sum(x) - 1.0])

        def ineq_constraint(x, args):
            return x  # x_i >= 0

        solver = SLSQP(
            atol=1e-6,
            rtol=1e-6,
            max_steps=15,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=n,
            inner_solver=MinresQLPSolver(tol=1e-8),
        )

        _, state, _ = _run_loop(solver, objective, x0)
        # Objective and current best merit stayed finite.
        assert jnp.isfinite(state.f_val)
        assert jnp.isfinite(state.best_merit)
        # Penalty did not blow up into the 1e+20 regime we observed in
        # production when ping-pong leaked MINRES-QLP multipliers.
        assert float(state.merit_penalty) < 1e6


# ---------------------------------------------------------------------------
# R3: QP.converged is independent of inner CG/MINRES imprecision
# ---------------------------------------------------------------------------


class TestQPConvergedIgnoresInnerImprecision:
    """``any_inner_failure`` is a diagnostic only.  A QP whose
    active-set loop stabilises must report ``converged=True`` even
    when the inner CG ran at a loose tolerance.
    """

    def test_loose_cg_still_converges(self):
        n = 4

        def hvp(v):
            return jnp.array([1.0, 2.0, 3.0, 4.0]) * v

        g = jnp.array([1.0, -1.0, 1.0, -1.0])
        # No constraints at all -> trivial QP path but exercises the
        # same converged-flag pipeline.
        result = solve_qp(
            hvp,
            g,
            jnp.zeros((0, n)),
            jnp.zeros(0),
            jnp.zeros((0, n)),
            jnp.zeros(0),
            tol=1e-8,
            max_iter=20,
            cg_tol=1e-2,  # intentionally loose
        )
        assert bool(result.converged)
        assert jnp.all(jnp.isfinite(result.d))


# ---------------------------------------------------------------------------
# R4: EXPAND working_tol follows the v0.9.2 formula
# ---------------------------------------------------------------------------


class TestExpandRampV092Formula:
    """``working_tol = base_tol + k * tau``; with ``expand_factor=1``
    the final tolerance is in ``[base_tol, 2 * base_tol]``.  In
    particular the *starting* tolerance must NOT be ``0.5 * base_tol``.
    """

    def test_working_tol_not_below_base_tol_at_iter0(self):
        n = 4
        diag = jnp.array([1.0, 2.0, 3.0, 4.0])

        def hvp(v):
            return diag * v

        g = jnp.array([-1.0, -2.0, 1.0, 2.0])
        A_ineq = jnp.eye(n)
        b_ineq = jnp.zeros(n)
        tol = 1e-8

        result = solve_qp(
            hvp,
            g,
            jnp.zeros((0, n)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            tol=tol,
            max_iter=20,
        )
        assert bool(result.converged)
        wt = float(result.final_working_tol)
        # v0.9.2 formula: working_tol >= base_tol = tol * (1 + min(kkt, 1))
        # So at the very least it must be >= tol itself (never
        # 0.5 * tol from the stricter hardened ramp).
        assert wt >= tol * 0.999, (
            f"final_working_tol {wt:e} below base_tol lower bound {tol:e}"
        )


# ---------------------------------------------------------------------------
# R5 (preexisting): CRAIG must not return a non-finite iterate on breakdown
# ---------------------------------------------------------------------------


class TestCraigBreakdownIsFinite:
    """``craig_solve`` must never propagate the ``1 / alpha_safe``
    amplification (where ``alpha_safe = max(alpha, 1e-30)``) into the
    returned iterate.

    Both the initial step and ``do_step`` previously stored
    ``x = state.x + s * v`` with ``|s * v| ≈ 1e60`` when ``alpha``
    dropped below the breakdown threshold, even though the breakdown
    flag was set correctly.  Downstream consumers (HVP calls in the
    projected-CG loop) then overflowed and propagated NaNs across the
    whole SLSQP run.  These tests pin the guard.
    """

    def test_init_breakdown_returns_zero(self):
        # Rank-deficient ``A`` with only a zero row: ``A^T u = 0`` for
        # any ``u``, so alpha1 = 0 and the initial step must be
        # breakdown-guarded.
        A = jnp.zeros((3, 5))
        rhs = jnp.array([1.0, 0.0, 0.0])
        x, converged = craig_solve(A, rhs, tol=1e-10, max_iter=10)
        assert jnp.all(jnp.isfinite(x))
        # The guard specifically returns the zero vector.
        assert float(jnp.linalg.norm(x)) == 0.0
        # Breakdown is signalled to the caller.
        assert not bool(converged)

    def test_step_breakdown_returns_finite(self):
        # A 2x3 matrix with strongly rank-deficient columns.  The
        # Golub-Kahan bidiagonalisation will run one normal step and
        # then hit a near-zero alpha on the second pass.
        A = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0 + 1e-16],
            ]
        )
        rhs = jnp.array([1.0, 1.0 + 1e-12])
        x, _ = craig_solve(A, rhs, tol=1e-12, max_iter=20)
        assert jnp.all(jnp.isfinite(x)), (
            f"CRAIG returned non-finite x on rank-deficient A: {x}"
        )
        # The iterate must stay bounded far below the ``1 / 1e-30``
        # amplification regime.
        assert float(jnp.linalg.norm(x)) < 1e6

    def test_projected_cg_craig_finite_on_rank_deficient_active_set(self):
        """End-to-end: ``ProjectedCGCraig`` must produce a finite
        direction even when the active Jacobian is rank-deficient.
        """
        n = 4

        def hvp(v):
            return v  # identity Hessian

        g = jnp.array([1.0, -1.0, 2.0, -2.0])
        # Two duplicate active rows → rank-deficient.
        A_ineq = jnp.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],  # duplicate
                [0.0, 0.0, 1.0, -1.0],
            ]
        )
        b_ineq = jnp.array([0.5, 0.5, 0.0])

        result = solve_qp(
            hvp,
            g,
            jnp.zeros((0, n)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            tol=1e-8,
            max_iter=30,
            inner_solver=ProjectedCGCraig(max_cg_iter=30, cg_tol=1e-8),
        )
        assert jnp.all(jnp.isfinite(result.d)), (
            f"ProjectedCGCraig returned non-finite d on rank-deficient "
            f"active set: {result.d}"
        )
        assert jnp.all(jnp.isfinite(result.multipliers_ineq))
