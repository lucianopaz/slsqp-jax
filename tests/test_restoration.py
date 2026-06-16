"""Tests for the feasibility-restoration fallback mode.

The restoration mode (Curtis-Johnson-Robinson-Wächter 2014, the FP/FQP
fallback only) reuses the existing L1 merit with an *objective weight*
``ω``: ``φ(x; ρ, ω) = ω·f(x) + ρ·v(x)``.  Normal mode is ``ω = 1``
(classical Han-Powell); restoration mode is ``ω = 0`` (``φ ∝ v``, the
feasibility problem ``min v(x)``).

These tests cover:

* ``compute_merit`` / ``backtracking_line_search`` backward-compat and the
  ``obj_weight`` semantics,
* ``RestorationConfig`` defaults and the ``SLSQP`` property accessors,
* the deterministic two-way state switch (entry / recoverable exit),
* the window-separation guarantee (no L-BFGS reset / append on the entry
  step or while in restoration),
* the end-to-end ``RESULTS.infeasible_stationary`` outcome on a locally
  infeasible problem (vs the generic ``infeasible`` when disabled).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from slsqp_jax import RESULTS, RestorationConfig, get_diagnostics
from slsqp_jax.merit import backtracking_line_search, compute_merit
from tests.conftest import _make_slsqp

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Merit / line-search obj_weight semantics + backward compatibility.
# ----------------------------------------------------------------------
class TestMeritObjWeight:
    def test_compute_merit_default_is_classical(self):
        """Default ``obj_weight=1.0`` reproduces ``f + ρ·v`` exactly."""
        f_val = jnp.array(2.0)
        eq_val = jnp.array([0.5, -0.25])
        ineq_val = jnp.array([0.1, -0.3])  # second row violated (-0.3 -> 0.3)
        penalty = jnp.array(4.0)

        merit = compute_merit(f_val, eq_val, ineq_val, penalty)
        v = jnp.sum(jnp.abs(eq_val)) + jnp.sum(jnp.maximum(0.0, -ineq_val))
        expected = f_val + penalty * v
        np.testing.assert_allclose(merit, expected)

    def test_compute_merit_omega_zero_drops_objective(self):
        """``obj_weight=0.0`` reduces the merit to ``ρ·v`` (no ``f`` term)."""
        eq_val = jnp.array([0.5, -0.25])
        ineq_val = jnp.array([0.1, -0.3])
        penalty = jnp.array(4.0)
        v = jnp.sum(jnp.abs(eq_val)) + jnp.sum(jnp.maximum(0.0, -ineq_val))

        # Two very different objective values must give the same merit.
        m1 = compute_merit(jnp.array(2.0), eq_val, ineq_val, penalty, obj_weight=0.0)
        m2 = compute_merit(jnp.array(999.0), eq_val, ineq_val, penalty, obj_weight=0.0)
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_allclose(m1, penalty * v)

    def test_compute_merit_omega_scales_objective_linearly(self):
        f_val = jnp.array(3.0)
        eq_val = jnp.array([0.2])
        ineq_val = jnp.array([0.0])
        penalty = jnp.array(2.0)
        v = jnp.sum(jnp.abs(eq_val))

        merit = compute_merit(f_val, eq_val, ineq_val, penalty, obj_weight=0.5)
        np.testing.assert_allclose(merit, 0.5 * f_val + penalty * v)

    def test_line_search_default_obj_weight_matches_explicit_one(self):
        """The line search default must be a no-op vs ``obj_weight=1.0``."""

        def fn(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        x = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 1.0])
        f_val = jnp.array(2.0)
        eq_val = jnp.zeros((0,))
        ineq_val = jnp.zeros((0,))
        penalty = jnp.array(1.0)
        grad = jax.grad(lambda z: jnp.sum((z - 1.0) ** 2))(x)

        common = dict(
            fn=fn,
            eq_constraint_fn=None,
            ineq_constraint_fn=None,
            x=x,
            direction=direction,
            args=None,
            f_val=f_val,
            eq_val=eq_val,
            ineq_val=ineq_val,
            penalty=penalty,
            grad=grad,
        )
        default = backtracking_line_search(**common)
        explicit = backtracking_line_search(**common, obj_weight=1.0)
        np.testing.assert_allclose(default.alpha, explicit.alpha)
        assert bool(default.success)
        # alpha = 1 reaches the minimum of the 1-D restriction.
        np.testing.assert_allclose(default.alpha, 1.0)

    def test_line_search_omega_zero_ignores_objective_descent(self):
        """With ``ω = 0`` and no constraints the directional derivative is
        zero, so the Armijo test accepts the full step trivially."""

        def fn(x, args):
            # An objective for which ``d`` is an *ascent* direction; with
            # ω = 0 the line search must ignore it (merit is constant).
            return jnp.sum((x + 5.0) ** 2), None

        x = jnp.array([0.0, 0.0])
        direction = jnp.array([1.0, 1.0])
        grad = jax.grad(lambda z: jnp.sum((z + 5.0) ** 2))(x)
        res = backtracking_line_search(
            fn=fn,
            eq_constraint_fn=None,
            ineq_constraint_fn=None,
            x=x,
            direction=direction,
            args=None,
            f_val=fn(x, None)[0],
            eq_val=jnp.zeros((0,)),
            ineq_val=jnp.zeros((0,)),
            penalty=jnp.array(1.0),
            grad=grad,
            obj_weight=0.0,
        )
        assert bool(res.success)
        np.testing.assert_allclose(res.alpha, 1.0)


# ----------------------------------------------------------------------
# Config + property accessors.
# ----------------------------------------------------------------------
class TestRestorationConfig:
    def test_defaults(self):
        cfg = RestorationConfig()
        assert cfg.enabled is True
        assert cfg.patience == 3
        assert cfg.cooldown is None
        assert cfg.max_entries == 5
        assert cfg.exit_tol_factor == 1.0

    def test_property_accessors(self):
        solver = _make_slsqp(
            rtol=1e-6,
            atol=1e-6,
            max_steps=50,
            restoration_enabled=True,
            restoration_patience=4,
            restoration_max_entries=2,
            restoration_exit_tol_factor=2.0,
        )
        assert solver.enable_restoration is True
        assert solver.restoration_patience == 4
        assert solver.restoration_max_entries == 2
        assert solver.restoration_exit_tol_factor == 2.0

    def test_cooldown_none_resolves_to_stagnation_window(self):
        # cooldown None -> max(1, max_steps // 10).
        solver = _make_slsqp(rtol=1e-6, atol=1e-6, max_steps=50)
        assert solver.restoration_cooldown == 5

    def test_cooldown_explicit_is_honoured(self):
        solver = _make_slsqp(
            rtol=1e-6, atol=1e-6, max_steps=50, restoration_cooldown=17
        )
        assert solver.restoration_cooldown == 17


# ----------------------------------------------------------------------
# State seeding + deterministic two-way switch.
# ----------------------------------------------------------------------
def _unconstrained_solver(**kw):
    return _make_slsqp(rtol=1e-6, atol=1e-6, max_steps=50, **kw)


class TestRestorationStateMachine:
    def test_init_seeds_normal_mode(self):
        def objective(x, args):
            return jnp.sum((x - 1.0) ** 2), None

        x0 = jnp.array([0.0, 0.0])
        solver = _unconstrained_solver()
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        np.testing.assert_allclose(state.omega, 1.0)
        assert not bool(state.restoration)
        assert int(state.infeasible_stall_count) == 0
        assert int(state.restoration_cooldown) == 0
        assert int(state.restoration_entries) == 0

    def test_recoverable_exit_resumes_optimization(self):
        """Force the solver into restoration at a feasible iterate; the next
        step must exit (``ω 0->1``), set the cooldown, and normal
        minimisation must then converge."""

        def objective(x, args):
            return jnp.sum((x - 3.0) ** 2), None

        x0 = jnp.array([0.0, 0.0])
        solver = _unconstrained_solver()
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        state = eqx.tree_at(
            lambda s: (s.omega, s.restoration, s.restoration_entries),
            state,
            (jnp.array(0.0), jnp.array(True), jnp.array(1)),
        )

        y = x0
        y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
        # Feasible (no constraints) -> restoration exits immediately.
        np.testing.assert_allclose(state.omega, 1.0)
        assert not bool(state.restoration)
        assert int(state.restoration_cooldown) == solver.restoration_cooldown

        for _ in range(40):
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
        np.testing.assert_allclose(y, jnp.array([3.0, 3.0]), atol=1e-4)

    def test_lbfgs_frozen_while_in_restoration(self):
        """A high-condition diagonal that WOULD trip the ``kappa>1e6``
        soft-reset is preserved untouched while in restoration."""

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq(x, args):
            # Inconsistent -> the iterate is infeasible.
            return jnp.array([x[0] - 0.0, x[0] - 1.0])

        x0 = jnp.array([0.5])
        solver = _make_slsqp(
            rtol=1e-6,
            atol=1e-6,
            eq_constraint_fn=eq,
            n_eq_constraints=2,
            max_steps=50,
        )
        state = solver.init(objective, x0, None, {}, None, None, frozenset())

        # Seed an ill-conditioned diagonal (kappa = 1e8 > 1e6) and mark the
        # solver as already in restoration.
        bad_diag = jnp.array([1e8])
        state = eqx.tree_at(
            lambda s: (s.omega, s.restoration, s.lbfgs_history.diagonal),
            state,
            (jnp.array(0.0), jnp.array(True), bad_diag),
        )

        _, new_state, _ = solver.step(objective, x0, None, {}, state, frozenset())
        # Frozen: diagonal and count unchanged despite the bad conditioning.
        np.testing.assert_array_equal(
            np.asarray(new_state.lbfgs_history.diagonal), np.asarray(bad_diag)
        )
        assert int(new_state.lbfgs_history.count) == int(state.lbfgs_history.count)


# ----------------------------------------------------------------------
# End-to-end behaviour on a locally-infeasible problem.
# ----------------------------------------------------------------------
def _infeasible_problem():
    def objective(x, args):
        return jnp.sum(x**2), None

    def eq(x, args):
        return jnp.array([x[0] - 0.0, x[0] - 1.0])

    return objective, eq, jnp.array([0.5])


class TestRestorationLive:
    def test_locally_infeasible_reports_infeasible_stationary(self):
        objective, eq, x0 = _infeasible_problem()
        solver = _make_slsqp(
            rtol=1e-6,
            atol=1e-6,
            eq_constraint_fn=eq,
            n_eq_constraints=2,
            max_steps=50,
            restoration_enabled=True,
        )
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        assert sol.stats["slsqp_result"] == RESULTS.infeasible_stationary
        diag = get_diagnostics(sol.state)
        assert bool(diag.restoration_triggered)
        assert int(diag.n_restoration_entries) >= 1

    def test_disabled_reverts_to_generic_infeasible(self):
        objective, eq, x0 = _infeasible_problem()
        solver = _make_slsqp(
            rtol=1e-6,
            atol=1e-6,
            eq_constraint_fn=eq,
            n_eq_constraints=2,
            max_steps=50,
            restoration_enabled=False,
        )
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        assert sol.stats["slsqp_result"] == RESULTS.infeasible
        diag = get_diagnostics(sol.state)
        assert not bool(diag.restoration_triggered)

    def test_entry_step_does_not_reset_lbfgs(self):
        """Window-separation guard: on the step where ``ω`` flips 1->0 the
        L-BFGS history must be carried through unchanged (no append, no
        reset) -- entry and reset can never co-occur."""
        objective, eq, x0 = _infeasible_problem()
        solver = _make_slsqp(
            rtol=1e-6,
            atol=1e-6,
            eq_constraint_fn=eq,
            n_eq_constraints=2,
            max_steps=50,
            restoration_enabled=True,
        )
        state = solver.init(objective, x0, None, {}, None, None, frozenset())
        y = x0
        prev_hist = state.lbfgs_history
        prev_omega = float(state.omega)

        entry_seen = False
        for _ in range(15):
            y, state, _ = solver.step(objective, y, None, {}, state, frozenset())
            omega = float(state.omega)
            if prev_omega == 1.0 and omega == 0.0:
                # Entry step: the post-step history must equal the history
                # the step was handed (frozen across the entry).
                entry_seen = True
                assert int(state.lbfgs_history.count) == int(prev_hist.count)
                np.testing.assert_array_equal(
                    np.asarray(state.lbfgs_history.diagonal),
                    np.asarray(prev_hist.diagonal),
                )
                np.testing.assert_array_equal(
                    np.asarray(state.lbfgs_history.s_history),
                    np.asarray(prev_hist.s_history),
                )
                break
            prev_hist = state.lbfgs_history
            prev_omega = omega

        assert entry_seen, "restoration never entered; cannot check entry window"
