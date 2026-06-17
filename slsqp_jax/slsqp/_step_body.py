"""Step / terminate bodies for :class:`slsqp_jax.slsqp.solver.SLSQP`.

These two functions are kept in their own module purely as a code-
organisation device: they are large enough that inlining them on the
class would dominate the file, but they are not separately re-usable
(both close over the full ``self``).  They are imported as instance
methods on :class:`SLSQP` via the unusual ``from ... import _step_impl
as _step_impl`` pattern in ``slsqp/solver.py`` to keep the symbol
binding explicit.

Algorithmic logic is preserved verbatim from the legacy
``slsqp_jax/solver.py`` ``step`` / ``terminate`` methods so that
behaviour matches ``main`` byte-for-byte; the refactor only:

* swaps the ``self._build_*`` helper calls for the new free-function
  helpers in ``slsqp_jax.slsqp.{bounds,hvp,preconditioner}``,
* replaces the inlined termination cascade with calls to
  :func:`slsqp_jax.slsqp.termination.classify_outcome` /
  :func:`coarse_outcome` (single source of truth), and
* delegates the box-bound active-set loop to
  :func:`slsqp_jax.qp.bound_fixing.run_bound_fixing` (now reachable
  through ``self._solve_qp_subproblem``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from slsqp_jax.hessian import (
    compute_lagrangian_gradient,
    lbfgs_append,
    lbfgs_curvature_diagnostics,
    lbfgs_identity_reset,
    lbfgs_soft_reset,
)
from slsqp_jax.merit import (
    backtracking_line_search,
    compute_merit,
    update_penalty_parameter,
)
from slsqp_jax.slsqp.hvp import build_exact_lagrangian_hvp
from slsqp_jax.slsqp.multipliers import (
    compute_at_lower_mask,
    compute_at_upper_mask,
    recover_ls_multipliers_at_iterate,
)
from slsqp_jax.slsqp.termination import (
    TerminationFlags,
    coarse_outcome,
    compute_mu_max,
)
from slsqp_jax.state import SLSQPDiagnostics, SLSQPState
from slsqp_jax.types import Vector


def _step_impl(
    self,
    fn: Callable,
    y: Vector,
    args: Any,
    options: dict[str, Any],
    state: SLSQPState,
    tags: frozenset[object],
) -> tuple[Vector, SLSQPState, Any]:
    y = self._clip_to_bounds(y)
    hvp_fn = self._build_lagrangian_hvp(fn, y, args, state)
    qp_result = self._solve_qp_subproblem(state, hvp_fn, y)

    # Fallback direction when the QP does not produce a usable step.
    #
    # Normal mode (ω = 1): projected steepest descent ``P(-∇f)`` onto
    # ``null(J_eq)``.  Restoration mode (ω = 0): steepest descent on the
    # constraint-violation measure ``v(x)``, i.e. ``-∇v``, left
    # *unprojected* because the goal is to move out of the (infeasible)
    # constraint manifold to reduce ``‖c_eq‖``.  ``∇v`` uses the same
    # Jacobian contractions as the L1 merit directional derivative:
    #   ∇v = J_eqᵀ sign(c_eq) - J_ineq_generalᵀ [c_ineq_general < 0].
    neg_grad = -state.grad
    if self.n_eq_constraints > 0:
        J = state.eq_jac
        JJT = J @ J.T
        m_eq = self.n_eq_constraints
        JJT_reg = JJT + 1e-10 * jnp.eye(m_eq)
        Jv = J @ neg_grad
        w = jnp.linalg.solve(JJT_reg, Jv)
        normal_fallback = neg_grad - J.T @ w
    else:
        normal_fallback = neg_grad

    m_ineq_general_fb = self.n_ineq_constraints
    g_feas = jnp.zeros_like(state.grad)
    if self.n_eq_constraints > 0:
        g_feas = g_feas + state.eq_jac.T @ jnp.sign(state.eq_val)
    if m_ineq_general_fb > 0:
        ineq_jac_general_fb = state.ineq_jac[:m_ineq_general_fb]
        viol_mask_fb = (state.ineq_val[:m_ineq_general_fb] < 0.0).astype(
            state.grad.dtype
        )
        g_feas = g_feas - ineq_jac_general_fb.T @ viol_mask_fb
    restoration_fallback = -g_feas
    fallback_direction = jnp.where(
        state.restoration, restoration_fallback, normal_fallback
    )

    direction = jnp.where(qp_result.converged, qp_result.direction, fallback_direction)
    zero_direction = jnp.linalg.norm(direction) < 1e-30
    fallback_source_norm = jnp.where(
        state.restoration,
        jnp.linalg.norm(restoration_fallback),
        jnp.linalg.norm(state.grad),
    )
    grad_nonzero = fallback_source_norm > self.atol
    direction = jnp.where(
        zero_direction & grad_nonzero & ~qp_result.converged,
        fallback_direction,
        direction,
    )
    direction_nonfinite = ~jnp.isfinite(qp_result.direction).all()
    n_vars = state.grad.shape[0]
    direction = jnp.reshape(direction, (n_vars,))
    d_norm = jnp.reshape(jnp.linalg.norm(direction), ())
    # In restoration mode a near-zero feasibility direction is the
    # ``∇v ≈ 0`` infeasible-stationary signal, so the zero-step detector
    # also fires when the QP itself did not "converge" (the linearised
    # constraints are typically inconsistent at an infeasible point).
    is_zero_step_pre = jnp.reshape(
        (d_norm < self.atol) & (qp_result.converged | state.restoration), ()
    )

    # Han-Powell penalty update: drop wrong-sign reduced-gradient noise
    # from the QP-side bound multipliers before letting them ratchet
    # ``rho``.  ``update_penalty_parameter`` reads ``max(abs(.))``, so a
    # large *negative* reduced-gradient value at a "barely active" bound
    # would otherwise inflate ``rho`` permanently (the rule is monotone
    # non-decreasing).  Clamping to dual-feasible magnitudes here mirrors
    # the post-step LS bound-multiplier recovery in
    # :func:`slsqp_jax.slsqp.bounds.recover_bound_multipliers` and only
    # affects the penalty calculation.  The QP step direction,
    # ``state.multipliers_ineq_qp``, ``qp_result.multipliers_ineq``, the
    # LS multiplier recovery, the LPEC-A predictor, and the QP active-set
    # warm-start all keep the raw signed multipliers.
    n_lower = self._n_lower_bounds
    n_upper = self._n_upper_bounds
    m_ineq_general = self.n_ineq_constraints
    if (n_lower + n_upper) > 0:
        mult_ineq_for_penalty = qp_result.multipliers_ineq.at[m_ineq_general:].set(
            jnp.maximum(qp_result.multipliers_ineq[m_ineq_general:], 0.0)
        )
    else:
        mult_ineq_for_penalty = qp_result.multipliers_ineq

    new_penalty = update_penalty_parameter(
        state.merit_penalty,
        qp_result.multipliers_eq,
        mult_ineq_for_penalty,
    )
    # Freeze the Han-Powell penalty during restoration: with ω = 0 the
    # merit is ρ·v and the QP/feasibility multipliers are not NLP KKT
    # multipliers, so they must not ratchet ρ.
    merit_penalty = jnp.where(
        qp_result.converged & ~state.restoration, new_penalty, state.merit_penalty
    )

    ls_result = backtracking_line_search(
        fn=fn,
        eq_constraint_fn=self.eq_constraint_fn,
        ineq_constraint_fn=self.ineq_constraint_fn,
        x=y,
        direction=direction,
        args=args,
        f_val=state.f_val,
        eq_val=state.eq_val,
        ineq_val=state.ineq_val,
        penalty=merit_penalty,
        grad=state.grad,
        c1=self.armijo_c1,
        max_iter=self.line_search_max_steps,
        bounds=self.bounds,
        lower_bound_mask=self._lower_bound_mask,
        upper_bound_mask=self._upper_bound_mask,
        eq_jac=state.eq_jac if self.n_eq_constraints > 0 else None,
        ineq_jac=state.ineq_jac[: self.n_ineq_constraints]
        if self.n_ineq_constraints > 0
        else None,
        obj_weight=state.omega,
    )

    alpha = ls_result.alpha
    y_new = self._clip_to_bounds(y + alpha * direction)
    f_val_new = ls_result.f_val
    eq_val_new = ls_result.eq_val
    ineq_val_new = ls_result.ineq_val

    _, aux = fn(y_new, args)

    # ------------------------------------------------------------------
    # Feasibility-restoration bookkeeping (Curtis-Johnson-Robinson-Wächter
    # 2014).  Decided *here*, before the L-BFGS update, so the L-BFGS
    # append/reset can be frozen on the entry step and throughout
    # restoration (objective curvature preserved as a metric).  Entry is
    # gated on a dedicated ``infeasible_stall_count`` that is *disjoint*
    # from the QP/LS failure counters that drive the L-BFGS reset chain,
    # guaranteeing entry never coincides with a reset.
    # ------------------------------------------------------------------
    m_eq_feas = self.n_eq_constraints
    m_ineq_total_feas = (
        self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
    )
    eq_violation_new = jnp.max(jnp.abs(eq_val_new)) if m_eq_feas > 0 else jnp.array(0.0)
    ineq_violation_new = (
        jnp.max(jnp.maximum(0.0, -ineq_val_new))
        if m_ineq_total_feas > 0
        else jnp.array(0.0)
    )
    max_violation_new = jnp.maximum(eq_violation_new, ineq_violation_new)
    primal_feasible_new = jnp.reshape(max_violation_new <= self.atol, ())
    restoration_exit_feasible = jnp.reshape(
        max_violation_new <= self.restoration_exit_tol_factor * self.atol, ()
    )

    # Violation-progress test (drives both restoration entry and the
    # slow-crawl early termination).  ``best_violation`` is the running
    # minimum max-norm violation over the current infeasible episode
    # (seeded ``+inf`` so the first infeasible step always counts as
    # progress).  A step is *meaningful progress* iff it shrinks the
    # violation by at least ``stall_rtol`` relative to that best.  Steps
    # below the threshold accumulate the stall counters even when the QP
    # nominally "converged" and the line search "succeeded" with a small
    # nonzero direction -- the failure mode the exact zero-step detector
    # misses.
    v_new = max_violation_new
    v_improved_enough = jnp.reshape(
        v_new < state.best_violation * (1.0 - self.restoration_stall_rtol), ()
    )
    best_violation_new = jnp.where(
        primal_feasible_new,
        jnp.asarray(jnp.inf),
        jnp.minimum(state.best_violation, v_new),
    )

    # Failure classification.  *Surprising* (feasible) QP / line-search
    # failures feed the L-BFGS reset chain; their infeasibility-driven
    # counterparts are split off into the dedicated
    # ``infeasible_stall_count``.  The ``primal_feasible_new`` factor makes
    # the two disjoint, so an infeasibility stall can never trip an L-BFGS
    # reset (or coincide with restoration entry).
    qp_unconverged = ~qp_result.converged
    ls_failed = ~ls_result.success
    qp_real_failure = jnp.reshape(
        qp_unconverged & ~qp_result.reached_max_iter & primal_feasible_new, ()
    )
    ls_real_failure = jnp.reshape(ls_failed & primal_feasible_new, ())
    # Infeasibility stall: while infeasible, the step made no usable
    # progress.  Three symptoms count: the QP failed (inconsistent
    # linearisation), the Han-Powell merit could not find a descent step,
    # or the QP "converged" to a near-zero direction (``∇v ≈ 0`` at an
    # infeasible point — the most common case, since at a locally
    # infeasible iterate the least-violation QP direction collapses to
    # ``d ≈ 0`` and the line search trivially accepts it).  Without the
    # zero-step clause restoration would never arm on a cleanly
    # locally-infeasible problem and the run would silently exhaust
    # ``max_steps`` (the ``restoration_arming`` guard suppresses the
    # generic stagnation / infeasible paths while restoration is eligible).
    # ``~v_improved_enough`` additionally covers the *slow-crawl* case: a
    # converged QP + successful line search whose nonzero step shrinks the
    # violation by a negligible amount.  Without it such a crawl would
    # never accumulate the entry counter and would run to ``max_steps``.
    infeasible_stall = jnp.reshape(
        ~primal_feasible_new
        & (qp_unconverged | ls_failed | is_zero_step_pre | ~v_improved_enough),
        (),
    )
    new_consecutive_qp_failures = jnp.where(
        qp_real_failure,
        state.consecutive_qp_failures + 1,
        jnp.array(0),
    )
    new_consecutive_ls_failures = jnp.where(
        ls_real_failure,
        state.consecutive_ls_failures + 1,
        jnp.array(0),
    )
    new_infeasible_stall_count = jnp.where(
        infeasible_stall,
        state.infeasible_stall_count + 1,
        jnp.array(0),
    )

    # Restoration entry / exit decisions (effective for the *next* step).
    cooldown_clear = state.restoration_cooldown == 0
    under_entry_cap = state.restoration_entries < self.restoration_max_entries
    entering_restoration = jnp.reshape(
        jnp.asarray(self.enable_restoration)
        & ~state.restoration
        & ~primal_feasible_new
        & (new_infeasible_stall_count >= self.restoration_patience)
        & cooldown_clear
        & under_entry_cap,
        (),
    )
    # "Restoration is eligible to arm" on this infeasible iterate.  Used to
    # suppress the divergence-rollback and merit-stagnation guardrails
    # while restoration accumulates its entry patience, so those generic
    # failure paths cannot terminate the run before restoration takes
    # over (and so the more informative ``infeasible_stationary`` outcome
    # wins over the generic ``infeasible`` override).
    restoration_arming = jnp.reshape(
        jnp.asarray(self.enable_restoration)
        & ~state.restoration
        & ~primal_feasible_new
        & cooldown_clear
        & under_entry_cap,
        (),
    )
    exiting_restoration = jnp.reshape(state.restoration & restoration_exit_feasible, ())
    omega_new = jnp.where(
        entering_restoration,
        jnp.array(0.0),
        jnp.where(exiting_restoration, jnp.array(1.0), state.omega),
    )
    restoration_new = jnp.reshape(
        jnp.where(
            entering_restoration,
            jnp.array(True),
            jnp.where(exiting_restoration, jnp.array(False), state.restoration),
        ),
        (),
    )
    omega_changed = jnp.reshape(omega_new != state.omega, ())

    infeasible_stall_count_new = jnp.where(
        entering_restoration | state.restoration,
        jnp.array(0),
        new_infeasible_stall_count,
    )
    restoration_entries_new = jnp.where(
        entering_restoration,
        state.restoration_entries + 1,
        state.restoration_entries,
    )
    restoration_cooldown_new = jnp.where(
        exiting_restoration,
        jnp.asarray(self.restoration_cooldown),
        jnp.where(
            restoration_new,
            state.restoration_cooldown,
            jnp.maximum(state.restoration_cooldown - 1, 0),
        ),
    )
    # Restoration slow-crawl stall: count consecutive restoration steps
    # (``state.restoration`` is the *pre-step* mode flag, so the counter
    # naturally starts at 0 the step after entry, giving restoration a
    # full ``stall_patience`` window) that stay infeasible without a
    # meaningful violation decrease.  When it saturates the run is
    # terminated at the minimum-violation point via
    # ``infeasible_stationary`` (see below) -- the exact-zero-step
    # detector alone never fires on a crawl.
    restoration_stall_count_new = jnp.where(
        state.restoration & ~primal_feasible_new & ~v_improved_enough,
        state.restoration_stall_count + 1,
        jnp.array(0),
    )
    restoration_stalled = jnp.reshape(
        restoration_stall_count_new >= self.restoration_stall_patience, ()
    )

    # Freeze L-BFGS (no append, no reset) on the entry step and while in
    # restoration so objective curvature is preserved across the switch.
    should_update_lbfgs = jnp.reshape(~(state.restoration | entering_restoration), ())

    grad_new = self._grad_impl(fn, y_new, args)
    eq_jac_new = self._eq_jac_impl(y_new, args)
    ineq_jac_general_new = self._ineq_jac_impl(y_new, args)
    ineq_jac_new = jnp.concatenate([ineq_jac_general_new, state.bound_jac], axis=0)

    m_ineq_general_static = self.n_ineq_constraints
    m_bounds_static = self._n_lower_bounds + self._n_upper_bounds
    n_lower_static = self._n_lower_bounds
    n_upper_static = self._n_upper_bounds

    # Free-mask at x_{k+1}: the LS multiplier recovery is restricted to
    # the free subspace so the at-bound gradient components do not leak
    # into the equality / general-inequality multipliers (those
    # components belong to the subsequent bound-multiplier recovery).
    if m_bounds_static > 0:
        at_lower_full = compute_at_lower_mask(y_new, self.bounds, self._lower_indices)
        at_upper_full = compute_at_upper_mask(y_new, self.bounds, self._upper_indices)
        free_mask = ~(at_lower_full | at_upper_full)
    else:
        free_mask = None

    # Hessian-free LS multiplier recovery at x_{k+1}.  Independent of
    # B / d / alpha; replaces the alpha-blended QP multipliers as the
    # multiplier vector consumed by the L-BFGS secant pair and the
    # convergence-test Lagrangian.  The QP-recovered multipliers are
    # still surfaced separately on ``state.multipliers_*_qp`` for
    # Han-Powell, LPEC-A and the QP active-set warm-start.
    ineq_val_general_new = (
        ineq_val_new[:m_ineq_general_static]
        if m_ineq_general_static > 0
        else jnp.zeros((0,), dtype=ineq_val_new.dtype)
    )
    ineq_jac_general_active = (
        ineq_jac_new[:m_ineq_general_static]
        if m_ineq_general_static > 0
        else jnp.zeros((0, grad_new.shape[0]), dtype=grad_new.dtype)
    )
    ls_mult_eq, ls_mult_ineq_general = recover_ls_multipliers_at_iterate(
        grad_new=grad_new,
        eq_jac_new=eq_jac_new,
        ineq_jac_general_new=ineq_jac_general_active,
        ineq_val_general_new=ineq_val_general_new,
        free_mask=free_mask,
        active_tol=self.atol,
    )

    # Bound multipliers — re-use the existing post-step recovery, fed
    # the LS equality and general-inequality multipliers so the whole
    # stationarity-side multiplier vector is internally consistent.
    if m_bounds_static > 0:
        mu_lower_corr, mu_upper_corr = self._recover_bound_multipliers(
            y_new=y_new,
            grad_new=grad_new,
            eq_jac_new=eq_jac_new,
            ineq_jac_new=ineq_jac_new,
            mult_eq=ls_mult_eq,
            mult_ineq_general=ls_mult_ineq_general,
        )
    else:
        mu_lower_corr = jnp.zeros((0,), dtype=grad_new.dtype)
        mu_upper_corr = jnp.zeros((0,), dtype=grad_new.dtype)

    # Stitch the full inequality LS multiplier (general + lower-bound +
    # upper-bound blocks) for downstream consumption.
    ls_mult_ineq_full = jnp.concatenate(
        [ls_mult_ineq_general, mu_lower_corr, mu_upper_corr], axis=0
    )

    s = y_new - y
    grad_lagrangian_new = compute_lagrangian_gradient(
        grad_new,
        eq_jac_new,
        ineq_jac_new,
        ls_mult_eq,
        ls_mult_ineq_full,
    )

    if self._obj_hvp_impl is not None:
        exact_hvp_fn = build_exact_lagrangian_hvp(
            fn=fn,
            y=y,
            args=args,
            multipliers_eq=ls_mult_eq,
            multipliers_ineq=ls_mult_ineq_full,
            obj_hvp_impl=self._obj_hvp_impl,
            eq_hvp_contrib_impl=self._eq_hvp_contrib_impl,
            ineq_hvp_contrib_impl=self._ineq_hvp_contrib_impl,
            n_ineq_general=self.n_ineq_constraints,
        )
        y_for_lbfgs = exact_hvp_fn(s)
    else:
        grad_lagrangian_old = compute_lagrangian_gradient(
            state.grad,
            state.eq_jac,
            state.ineq_jac,
            ls_mult_eq,
            ls_mult_ineq_full,
        )
        y_for_lbfgs = grad_lagrangian_new - grad_lagrangian_old

    computed_history = lbfgs_append(
        state.lbfgs_history,
        s,
        y_for_lbfgs,
        damping_threshold=self.damping_threshold,
        diag_floor=self.lbfgs_diag_floor,
        diag_ceil=self.lbfgs_diag_ceil,
    )

    kappa_est = computed_history.eig_upper / jnp.maximum(
        computed_history.eig_lower, 1e-30
    )
    computed_history = jax.lax.cond(
        (computed_history.count > 1) & (kappa_est > 1e6),
        lbfgs_soft_reset,
        lambda h: h,
        computed_history,
    )

    # ``qp_real_failure`` / ``new_consecutive_qp_failures`` /
    # ``ls_failed`` / ``new_consecutive_ls_failures`` are computed in the
    # restoration-bookkeeping block above; infeasibility-driven QP stalls
    # are routed away from ``qp_real_failure`` there so they never trip
    # these resets.
    computed_history = jax.lax.cond(
        qp_real_failure & (new_consecutive_qp_failures == 1),
        lbfgs_soft_reset,
        lambda h: h,
        computed_history,
    )
    computed_history = jax.lax.cond(
        new_consecutive_qp_failures >= self.qp_failure_patience,
        lbfgs_identity_reset,
        lambda h: h,
        computed_history,
    )
    computed_history = jax.lax.cond(
        ls_failed & (new_consecutive_ls_failures == 1),
        lbfgs_soft_reset,
        lambda h: h,
        computed_history,
    )
    computed_history = jax.lax.cond(
        new_consecutive_ls_failures >= self.ls_failure_patience,
        lbfgs_identity_reset,
        lambda h: h,
        computed_history,
    )

    # Freeze L-BFGS on the restoration entry step and throughout
    # restoration: reuse the existing history unchanged so objective
    # curvature is preserved across the ω switch and restoration entry
    # never coincides with a curvature reset.
    new_lbfgs_history = jax.lax.cond(
        should_update_lbfgs,
        lambda computed, _old: computed,
        lambda _computed, old: old,
        computed_history,
        state.lbfgs_history,
    )

    is_zero_step_post = jnp.reshape(
        (alpha * d_norm < self.atol)
        & (qp_result.converged | state.restoration)
        & ls_result.success,
        (),
    )
    is_zero_step = jnp.reshape(is_zero_step_pre | is_zero_step_post, ())
    new_consecutive_zero_steps = jnp.reshape(
        jnp.where(is_zero_step, state.consecutive_zero_steps + 1, jnp.array(0)),
        (),
    )
    new_qp_optimal = jnp.reshape(
        new_consecutive_zero_steps >= self.zero_step_patience, ()
    )

    ls_fatal = new_consecutive_ls_failures >= 2 * self.ls_failure_patience
    qp_fatal = new_consecutive_qp_failures >= 2 * self.qp_failure_patience

    merit_new = compute_merit(
        f_val_new, eq_val_new, ineq_val_new, merit_penalty, state.omega
    )
    merit_threshold = self.stagnation_tol * jnp.maximum(jnp.abs(state.best_merit), 1.0)
    improved = merit_new < state.best_merit - merit_threshold
    new_best_merit = jnp.where(improved, merit_new, state.best_merit)
    new_best_x = jnp.where(improved, y_new, state.best_x)
    new_steps_without = jnp.where(
        improved, jnp.array(0), state.steps_without_improvement + 1
    )
    # When the objective weight ω flips (restoration entry/exit) the merit
    # changes units (``f + ρv`` <-> ``ρv``).  Re-seed the best-merit
    # tracking with the merit evaluated in the *new* units so the next
    # step's comparison is well-posed, and suppress this step's
    # stagnation / divergence bookkeeping (the cross-unit delta is
    # meaningless).
    merit_seed_next_units = compute_merit(
        f_val_new, eq_val_new, ineq_val_new, merit_penalty, omega_new
    )
    new_best_merit = jnp.where(omega_changed, merit_seed_next_units, new_best_merit)
    new_best_x = jnp.where(omega_changed, y_new, new_best_x)
    new_steps_without = jnp.where(omega_changed, jnp.array(0), new_steps_without)
    patience = self._stagnation_window
    # Suppress merit-stagnation while restoration is armed or active: the
    # feasibility fallback (or its ``infeasible_stationary`` terminal)
    # owns the infeasible-stall outcome.
    merit_stagnation = (
        (state.step_count >= patience)
        & (new_steps_without >= patience)
        & ~restoration_arming
        & ~state.restoration
    )

    blowup_threshold = self.divergence_factor * jnp.maximum(
        jnp.abs(new_best_merit), 1.0
    )
    merit_finite = jnp.isfinite(merit_new)
    # Suppress the divergence rollback on the ω-switch step and while
    # restoration is armed or active (the penalty ratchet on an infeasible
    # problem would otherwise trip it before restoration engages).
    blowup_now = (
        (((merit_new - new_best_merit) > blowup_threshold) | ~merit_finite)
        & ~omega_changed
        & ~restoration_arming
        & ~state.restoration
    )
    new_blowup_count = jnp.where(blowup_now, state.blowup_count + 1, jnp.array(0))
    diverging_now = jnp.reshape(new_blowup_count >= self.divergence_patience, ())
    # On a restoration slow-crawl stall (terminating below as
    # ``infeasible_stationary``) return the episode minimum-violation
    # iterate ``best_x`` rather than the last crawl point.
    y_returned = jnp.where(
        diverging_now | (restoration_stalled & ~primal_feasible_new),
        new_best_x,
        y_new,
    )

    prev_diag = state.diagnostics
    lbfgs_sty, lbfgs_relcurv, lbfgs_skipped = lbfgs_curvature_diagnostics(
        s, y_for_lbfgs
    )
    if self.n_eq_constraints > 0:
        JJT_reg = state.eq_jac @ state.eq_jac.T + 1e-12 * jnp.eye(self.n_eq_constraints)
        L_eq = jnp.linalg.cholesky(JJT_reg)
        diag_L = jnp.abs(jnp.diag(L_eq))
        eq_sv_est = jnp.where(
            jnp.all(jnp.isfinite(diag_L)),
            jnp.min(diag_L),
            jnp.inf,
        )
    else:
        eq_sv_est = jnp.inf

    merit_regression = ls_result.success & (merit_new > state.best_merit)
    n_active_ineq_now = jnp.sum(qp_result.active_set.astype(jnp.int32))

    new_diagnostics = SLSQPDiagnostics(
        n_qp_inner_failures=prev_diag.n_qp_inner_failures
        + jnp.where(~qp_result.converged, 1, 0),
        n_ls_failures=prev_diag.n_ls_failures + jnp.where(ls_failed, 1, 0),
        n_lbfgs_skips=prev_diag.n_lbfgs_skips + jnp.where(lbfgs_skipped, 1, 0),
        n_nan_directions=prev_diag.n_nan_directions
        + jnp.where(direction_nonfinite, 1, 0),
        max_gamma=jnp.maximum(prev_diag.max_gamma, new_lbfgs_history.gamma),
        min_diag=jnp.minimum(prev_diag.min_diag, jnp.min(new_lbfgs_history.diagonal)),
        max_diag=jnp.maximum(prev_diag.max_diag, jnp.max(new_lbfgs_history.diagonal)),
        eq_jac_min_sv_est=jnp.minimum(prev_diag.eq_jac_min_sv_est, eq_sv_est),
        ls_alpha_min=jnp.minimum(prev_diag.ls_alpha_min, alpha),
        tail_ls_failures=new_consecutive_ls_failures,
        n_bound_fix_solves=prev_diag.n_bound_fix_solves + qp_result.bound_fix_solves,
        max_bound_fixed=jnp.maximum(prev_diag.max_bound_fixed, qp_result.n_bound_fixed),
        max_active_ineq=jnp.maximum(prev_diag.max_active_ineq, n_active_ineq_now),
        n_merit_regressions=prev_diag.n_merit_regressions
        + jnp.where(merit_regression, 1, 0),
        n_qp_budget_exhausted=prev_diag.n_qp_budget_exhausted
        + jnp.where(qp_result.reached_max_iter, 1, 0),
        n_qp_ping_pong=prev_diag.n_qp_ping_pong
        + jnp.where(qp_result.ping_ponged, 1, 0),
        max_qp_iterations=jnp.maximum(
            prev_diag.max_qp_iterations, qp_result.iterations
        ),
        max_qp_active_size=jnp.maximum(prev_diag.max_qp_active_size, n_active_ineq_now),
        n_lpeca_bypassed=prev_diag.n_lpeca_bypassed
        + jnp.where(qp_result.lpeca_bypassed, 1, 0),
        n_lpeca_capped=prev_diag.n_lpeca_capped
        + jnp.where(qp_result.lpeca_capped, 1, 0),
        n_lpeca_bounds_prefixed=prev_diag.n_lpeca_bounds_prefixed
        + qp_result.n_lpeca_bounds_prefixed,
        n_proj_refinements=prev_diag.n_proj_refinements + qp_result.n_proj_refinements,
        max_proj_residual=jnp.maximum(
            prev_diag.max_proj_residual,
            qp_result.proj_residual.astype(prev_diag.max_proj_residual.dtype),
        ),
        n_divergence_blowups=prev_diag.n_divergence_blowups
        + jnp.where(blowup_now, 1, 0),
        divergence_triggered=prev_diag.divergence_triggered | diverging_now,
        min_projected_grad_norm=jnp.minimum(
            prev_diag.min_projected_grad_norm,
            qp_result.projected_grad_norm.astype(
                prev_diag.min_projected_grad_norm.dtype
            ),
        ),
        n_steps_inexact_below_classical=prev_diag.n_steps_inexact_below_classical
        + jnp.where(
            qp_result.projected_grad_norm.astype(jnp.float64)
            < jnp.linalg.norm(grad_lagrangian_new).astype(jnp.float64),
            1,
            0,
        ),
        n_restoration_entries=prev_diag.n_restoration_entries
        + jnp.where(entering_restoration, 1, 0),
        restoration_triggered=prev_diag.restoration_triggered | entering_restoration,
        n_restoration_steps=prev_diag.n_restoration_steps
        + jnp.where(state.restoration, 1, 0),
        min_violation_in_restoration=jnp.minimum(
            prev_diag.min_violation_in_restoration,
            jnp.where(
                state.restoration | entering_restoration,
                max_violation_new.astype(prev_diag.min_violation_in_restoration.dtype),
                prev_diag.min_violation_in_restoration,
            ),
        ),
    )

    # QP-side state writeback: keep the QP-recovered general-inequality
    # multipliers (Han-Powell, LPEC-A and the next QP's warm-start
    # consume them) but splice the LS bound recovery into the bound
    # block — bound rows have no QP-side recovery (they are excluded
    # from the QP's `(A_ineq, b_ineq)` pair on purpose) so the LS
    # bound block is the only post-step value either side has.
    if m_bounds_static > 0:
        multipliers_ineq_qp_for_state = qp_result.multipliers_ineq
        if n_lower_static > 0:
            multipliers_ineq_qp_for_state = multipliers_ineq_qp_for_state.at[
                m_ineq_general_static : m_ineq_general_static + n_lower_static
            ].set(mu_lower_corr)
        if n_upper_static > 0:
            multipliers_ineq_qp_for_state = multipliers_ineq_qp_for_state.at[
                m_ineq_general_static + n_lower_static :
            ].set(mu_upper_corr)
    else:
        multipliers_ineq_qp_for_state = qp_result.multipliers_ineq

    # Granular termination-code classification, mirrored at the new
    # iterate so ``state.termination_code`` agrees with the value
    # ``terminate()`` will eventually settle on.  Uses the LS
    # multipliers so the numerator (``grad_lagrangian_new`` is
    # LS-based) and the filterSQP μ_max denominator share the same
    # multiplier vector.  ``primal_feasible_new`` was computed in the
    # restoration-bookkeeping block above and is reused here.
    m_ineq_general_for_mu = self.n_ineq_constraints
    # filterSQP normalisation denominator (eq. 5 of the manual): the
    # largest single contributor to ``∇_x L = ∇f − Jᵀλ − νᵀI_b``.
    # Replaces the legacy ``|L|``-based denominator so the test is
    # invariant to absolute objective magnitude and tracks
    # multiplier-magnitude blow-up under near-rank-deficient
    # active sets.  Both ``classical`` and ``inexact`` branches share
    # the same denominator: even though the projected-gradient
    # numerator has had the multiplier terms algebraically projected
    # out, the reference scale "ε relative to the largest
    # contributor" is the consistent meaning of ``rtol`` across the
    # two paths.
    mu_max_new = compute_mu_max(
        grad_f=grad_new,
        eq_jac=eq_jac_new,
        ineq_jac_general=ineq_jac_new[:m_ineq_general_for_mu],
        mult_eq=ls_mult_eq,
        mult_ineq_general=ls_mult_ineq_full[:m_ineq_general_for_mu],
        mult_bound=ls_mult_ineq_full[m_ineq_general_for_mu:],
    )
    grad_norm_new = jnp.linalg.norm(grad_lagrangian_new)
    rtol_target_new = self.rtol * jnp.maximum(mu_max_new, 1.0)
    classical_stationarity_new = grad_norm_new <= rtol_target_new
    inexact_stationarity_new = qp_result.projected_grad_norm <= rtol_target_new
    stationarity_new = classical_stationarity_new | (
        jnp.asarray(self.use_inexact_stationarity) & inexact_stationarity_new
    )
    new_step_count = state.step_count + 1
    has_min_steps_new = new_step_count >= self.min_steps
    max_iters_reached_new = new_step_count >= self.max_steps
    classical_converged_new = stationarity_new & primal_feasible_new & has_min_steps_new
    qp_kkt_success_new = (
        new_qp_optimal
        & primal_feasible_new
        & ls_result.success
        & (alpha >= 1.0 - 1e-6)
        & has_min_steps_new
    )
    converged_new = jnp.reshape(classical_converged_new | qp_kkt_success_new, ())
    nonfinite_new = ~jnp.all(jnp.isfinite(y_returned))

    # Infeasible stationary point (CJRW 2014): while in restoration the
    # feasibility direction has either collapsed exactly (``new_qp_optimal``
    # via the relaxed zero-step detector) or crawled without meaningful
    # progress for ``stall_patience`` steps (``restoration_stalled``), yet
    # the iterate remains infeasible.  This is the informative "converged
    # to a minimum-violation infeasible stationary point" outcome.
    infeasible_stationary_new = jnp.reshape(
        restoration_new
        & (new_qp_optimal | restoration_stalled)
        & ~primal_feasible_new
        & has_min_steps_new,
        (),
    )

    flags = TerminationFlags(
        converged=converged_new,
        nonfinite=jnp.reshape(nonfinite_new, ()),
        diverging=diverging_now,
        ls_fatal=jnp.reshape(ls_fatal, ()),
        qp_fatal=jnp.reshape(qp_fatal, ()),
        merit_stagnation=jnp.reshape(merit_stagnation, ()),
        max_iters_reached=jnp.reshape(max_iters_reached_new, ()),
        primal_feasible=primal_feasible_new,
        infeasible_stationary=infeasible_stationary_new,
    )
    from slsqp_jax.slsqp.termination import classify_outcome

    termination_code = classify_outcome(flags)

    new_state = SLSQPState(
        step_count=state.step_count + 1,
        f_val=f_val_new,
        grad=grad_new,
        eq_val=eq_val_new,
        ineq_val=ineq_val_new,
        eq_jac=eq_jac_new,
        ineq_jac=ineq_jac_new,
        lbfgs_history=new_lbfgs_history,
        multipliers_eq_qp=qp_result.multipliers_eq,
        multipliers_ineq_qp=multipliers_ineq_qp_for_state,
        multipliers_eq_ls=ls_mult_eq,
        multipliers_ineq_ls=ls_mult_ineq_full,
        kkt_residual_grad=grad_lagrangian_new,
        grad_lagrangian=grad_lagrangian_new,
        merit_penalty=merit_penalty,
        bound_jac=state.bound_jac,
        qp_iterations=state.qp_iterations + qp_result.iterations,
        qp_converged=qp_result.converged,
        prev_active_set=qp_result.active_set,
        consecutive_qp_failures=new_consecutive_qp_failures,
        consecutive_ls_failures=new_consecutive_ls_failures,
        consecutive_zero_steps=new_consecutive_zero_steps,
        qp_optimal=new_qp_optimal,
        best_merit=new_best_merit,
        steps_without_improvement=new_steps_without,
        stagnation=merit_stagnation,
        last_alpha=alpha,
        last_projected_grad_norm=qp_result.projected_grad_norm,
        ls_success=ls_result.success,
        ls_fatal=ls_fatal,
        qp_fatal=qp_fatal,
        termination_code=termination_code,
        best_x=new_best_x,
        blowup_count=new_blowup_count,
        diverging=diverging_now,
        omega=omega_new,
        restoration=restoration_new,
        infeasible_stall_count=infeasible_stall_count_new,
        restoration_cooldown=restoration_cooldown_new,
        restoration_entries=restoration_entries_new,
        best_violation=best_violation_new,
        restoration_stall_count=restoration_stall_count_new,
        diagnostics=new_diagnostics,
    )

    # Verbose output
    m_eq = self.n_eq_constraints
    m_ineq_total = self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
    eq_viol = jnp.max(jnp.abs(eq_val_new)) if m_eq > 0 else jnp.array(0.0)
    ineq_viol = (
        jnp.max(jnp.maximum(0.0, -ineq_val_new)) if m_ineq_total > 0 else jnp.array(0.0)
    )
    c_viol = jnp.maximum(eq_viol, ineq_viol)
    kkt = jnp.linalg.norm(grad_lagrangian_new)
    lagrangian_val_est = (
        f_val_new
        - jnp.dot(ls_mult_eq, eq_val_new)
        - jnp.dot(ls_mult_ineq_full, ineq_val_new)
    )
    rel_kkt = kkt / jnp.maximum(jnp.abs(lagrangian_val_est), 1.0)
    dir_norm = jnp.linalg.norm(direction)
    grad_norm = jnp.linalg.norm(grad_new)
    n_active = jnp.sum(qp_result.active_set.astype(jnp.int32))
    diag_cond = jnp.max(new_lbfgs_history.diagonal) / jnp.maximum(
        jnp.min(new_lbfgs_history.diagonal), 1e-30
    )
    merit_delta = merit_new - state.best_merit
    qp_cyc = (qp_result.ping_ponged.astype(jnp.int32)) | (
        qp_result.reached_max_iter.astype(jnp.int32) << 1
    )
    self.verbose(
        num_steps=("Step", new_state.step_count),  # ty: ignore[unresolved-attribute]
        objective=("f", f_val_new, ".6e"),
        constraint_violation=("|c|", c_viol, ".3e"),
        kkt_residual=("|∇L|", kkt, ".3e"),
        kkt_relative=("|∇L|/|L|", rel_kkt, ".3e"),
        kkt_scale=("μ_max", mu_max_new, ".3e"),
        proj_grad_norm=("|W̃g|", qp_result.projected_grad_norm, ".3e"),
        grad_norm=("|∇f|", grad_norm, ".3e"),
        step_size=("α", alpha, ".3e"),
        direction_norm=("|d|", dir_norm, ".3e"),
        merit=("merit", merit_new, ".6e"),
        merit_delta=("Δmerit", merit_delta, "+.2e"),
        stag_count=("stag#", new_steps_without),
        stagnation=("stag", merit_stagnation),
        restoration_mode=("R", restoration_new),
        penalty=("ρ", merit_penalty, ".3e"),
        lbfgs_gamma=("γ", new_lbfgs_history.gamma, ".3e"),
        lbfgs_diag_cond=("κ_B", diag_cond, ".1e"),
        lbfgs_skipped=("skip", lbfgs_skipped),
        lbfgs_sty=("s·y", lbfgs_sty, ".3e"),
        lbfgs_relcurv=("rel_curv", lbfgs_relcurv, ".3e"),
        qp_iters=("QPiter", qp_result.iterations),
        qp_cyc=("QPcyc", qp_cyc),
        qp_converged=("QP ok", qp_result.converged),
        n_active=("#act", n_active),
        n_bound_fixed=("#fix", qp_result.n_bound_fixed),
        bound_fix_solves=("fix#", qp_result.bound_fix_solves),
        ls_steps=("LS it", ls_result.n_evals),
        ls_success=("LS ok", ls_result.success),
        ls_tail=("LS tail", new_consecutive_ls_failures),
        blowup_count=("blowup#", new_blowup_count),
    )

    return y_returned, new_state, aux  # ty: ignore[invalid-return-type]


def _terminate_impl(
    self,
    fn: Callable,
    y: Vector,
    args: Any,
    options: dict[str, Any],
    state: SLSQPState,
    tags: frozenset[object],
) -> tuple[Bool[Array, ""], Any]:
    m_eq = self.n_eq_constraints
    m_ineq_general = self.n_ineq_constraints
    m_ineq_total = self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
    grad_lagrangian = state.grad_lagrangian

    # filterSQP normalisation denominator (eq. 5 of the manual).
    # Mirrors the ``_step_impl`` computation so ``terminate`` and the
    # per-step ``state.termination_code`` cannot disagree.
    mu_max = compute_mu_max(
        grad_f=state.grad,
        eq_jac=state.eq_jac,
        ineq_jac_general=state.ineq_jac[:m_ineq_general],
        mult_eq=state.multipliers_eq_ls,
        mult_ineq_general=state.multipliers_ineq_ls[:m_ineq_general],
        mult_bound=state.multipliers_ineq_ls[m_ineq_general:],
    )
    grad_norm = jnp.linalg.norm(grad_lagrangian)
    rtol_target = self.rtol * jnp.maximum(mu_max, 1.0)
    classical_stationarity = grad_norm <= rtol_target
    inexact_stationarity = state.last_projected_grad_norm <= rtol_target
    stationarity = classical_stationarity | (
        jnp.asarray(self.use_inexact_stationarity) & inexact_stationarity
    )

    eq_feasible = jnp.array(True)
    if m_eq > 0:
        eq_feasible = jnp.max(jnp.abs(state.eq_val)) <= self.atol
    ineq_feasible = jnp.array(True)
    if m_ineq_total > 0:
        ineq_feasible = jnp.max(jnp.maximum(0.0, -state.ineq_val)) <= self.atol
    primal_feasible = eq_feasible & ineq_feasible

    max_iters_reached = state.step_count >= self.max_steps
    nonfinite_iter = ~jnp.all(jnp.isfinite(y))

    has_min_steps = state.step_count >= self.min_steps
    classical_converged = stationarity & primal_feasible & has_min_steps
    qp_kkt_success = (
        state.qp_optimal
        & primal_feasible
        & state.ls_success
        & (state.last_alpha >= 1.0 - 1e-6)
        & has_min_steps
    )
    converged = classical_converged | qp_kkt_success

    # Infeasible stationary point (CJRW 2014): recomputed from state so
    # ``terminate`` and ``step``'s ``state.termination_code`` agree.  The
    # slow-crawl stall (``restoration_stall_count``) is OR'd with the
    # exact-zero-step ``qp_optimal`` path, mirroring ``_step_impl``.
    restoration_stalled = jnp.reshape(
        state.restoration
        & (state.restoration_stall_count >= self.restoration_stall_patience),
        (),
    )
    infeasible_stationary = jnp.reshape(
        state.restoration
        & (state.qp_optimal | restoration_stalled)
        & ~primal_feasible
        & has_min_steps,
        (),
    )

    flags = TerminationFlags(
        converged=jnp.reshape(converged, ()),
        nonfinite=jnp.reshape(nonfinite_iter, ()),
        diverging=jnp.reshape(state.diverging, ()),
        ls_fatal=jnp.reshape(state.ls_fatal, ()),
        qp_fatal=jnp.reshape(state.qp_fatal, ()),
        merit_stagnation=jnp.reshape(state.stagnation, ()),
        max_iters_reached=jnp.reshape(max_iters_reached, ()),
        primal_feasible=jnp.reshape(primal_feasible, ()),
        infeasible_stationary=infeasible_stationary,
    )
    return coarse_outcome(flags)


__all__ = ["_step_impl", "_terminate_impl"]
