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
from slsqp_jax.slsqp.termination import TerminationFlags, coarse_outcome
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

    # Projected steepest-descent fallback direction.
    neg_grad = -state.grad
    if self.n_eq_constraints > 0:
        J = state.eq_jac
        JJT = J @ J.T
        m_eq = self.n_eq_constraints
        JJT_reg = JJT + 1e-10 * jnp.eye(m_eq)
        Jv = J @ neg_grad
        w = jnp.linalg.solve(JJT_reg, Jv)
        fallback_direction = neg_grad - J.T @ w
    else:
        fallback_direction = neg_grad

    direction = jnp.where(qp_result.converged, qp_result.direction, fallback_direction)
    zero_direction = jnp.linalg.norm(direction) < 1e-30
    grad_nonzero = jnp.linalg.norm(state.grad) > self.atol
    direction = jnp.where(
        zero_direction & grad_nonzero & ~qp_result.converged,
        fallback_direction,
        direction,
    )
    direction_nonfinite = ~jnp.isfinite(qp_result.direction).all()
    n_vars = state.grad.shape[0]
    direction = jnp.reshape(direction, (n_vars,))
    d_norm = jnp.reshape(jnp.linalg.norm(direction), ())
    is_zero_step_pre = jnp.reshape((d_norm < self.atol) & qp_result.converged, ())

    new_penalty = update_penalty_parameter(
        state.merit_penalty,
        qp_result.multipliers_eq,
        qp_result.multipliers_ineq,
    )
    merit_penalty = jnp.where(qp_result.converged, new_penalty, state.merit_penalty)

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
    )

    alpha = ls_result.alpha
    y_new = self._clip_to_bounds(y + alpha * direction)
    f_val_new = ls_result.f_val
    eq_val_new = ls_result.eq_val
    ineq_val_new = ls_result.ineq_val

    _, aux = fn(y_new, args)

    grad_new = self._grad_impl(fn, y_new, args)
    eq_jac_new = self._eq_jac_impl(y_new, args)
    ineq_jac_general_new = self._ineq_jac_impl(y_new, args)
    ineq_jac_new = jnp.concatenate([ineq_jac_general_new, state.bound_jac], axis=0)

    blended_mult_eq = state.multipliers_eq + alpha * (
        qp_result.multipliers_eq - state.multipliers_eq
    )
    blended_mult_ineq = state.multipliers_ineq + alpha * (
        qp_result.multipliers_ineq - state.multipliers_ineq
    )

    m_ineq_general_static = self.n_ineq_constraints
    m_bounds_static = self._n_lower_bounds + self._n_upper_bounds
    n_lower_static = self._n_lower_bounds
    if m_bounds_static > 0:
        mult_ineq_general_blended = (
            blended_mult_ineq[:m_ineq_general_static]
            if m_ineq_general_static > 0
            else jnp.zeros((0,), dtype=blended_mult_ineq.dtype)
        )
        mu_lower_corr, mu_upper_corr = self._recover_bound_multipliers(
            y_new=y_new,
            grad_new=grad_new,
            eq_jac_new=eq_jac_new,
            ineq_jac_new=ineq_jac_new,
            mult_eq=blended_mult_eq,
            mult_ineq_general=mult_ineq_general_blended,
        )
        if n_lower_static > 0:
            blended_mult_ineq = blended_mult_ineq.at[
                m_ineq_general_static : m_ineq_general_static + n_lower_static
            ].set(mu_lower_corr)
        if self._n_upper_bounds > 0:
            blended_mult_ineq = blended_mult_ineq.at[
                m_ineq_general_static + n_lower_static :
            ].set(mu_upper_corr)
    else:
        mu_lower_corr = jnp.zeros((0,), dtype=blended_mult_ineq.dtype)
        mu_upper_corr = jnp.zeros((0,), dtype=blended_mult_ineq.dtype)

    s = y_new - y
    grad_lagrangian_new = compute_lagrangian_gradient(
        grad_new,
        eq_jac_new,
        ineq_jac_new,
        blended_mult_eq,
        blended_mult_ineq,
    )

    if self._obj_hvp_impl is not None:
        exact_hvp_fn = build_exact_lagrangian_hvp(
            fn=fn,
            y=y,
            args=args,
            multipliers_eq=blended_mult_eq,
            multipliers_ineq=blended_mult_ineq,
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
            blended_mult_eq,
            blended_mult_ineq,
        )
        y_for_lbfgs = grad_lagrangian_new - grad_lagrangian_old

    new_lbfgs_history = lbfgs_append(
        state.lbfgs_history,
        s,
        y_for_lbfgs,
        damping_threshold=self.damping_threshold,
        diag_floor=self.lbfgs_diag_floor,
        diag_ceil=self.lbfgs_diag_ceil,
    )

    kappa_est = new_lbfgs_history.eig_upper / jnp.maximum(
        new_lbfgs_history.eig_lower, 1e-30
    )
    new_lbfgs_history = jax.lax.cond(
        (new_lbfgs_history.count > 1) & (kappa_est > 1e6),
        lbfgs_soft_reset,
        lambda h: h,
        new_lbfgs_history,
    )

    qp_real_failure = ~qp_result.converged & ~qp_result.reached_max_iter
    new_consecutive_qp_failures = jnp.where(
        qp_real_failure,
        state.consecutive_qp_failures + 1,
        jnp.array(0),
    )
    new_lbfgs_history = jax.lax.cond(
        qp_real_failure & (new_consecutive_qp_failures == 1),
        lbfgs_soft_reset,
        lambda h: h,
        new_lbfgs_history,
    )
    new_lbfgs_history = jax.lax.cond(
        new_consecutive_qp_failures >= self.qp_failure_patience,
        lbfgs_identity_reset,
        lambda h: h,
        new_lbfgs_history,
    )

    ls_failed = ~ls_result.success
    new_consecutive_ls_failures = jnp.where(
        ls_failed,
        state.consecutive_ls_failures + 1,
        jnp.array(0),
    )
    new_lbfgs_history = jax.lax.cond(
        ls_failed & (new_consecutive_ls_failures == 1),
        lbfgs_soft_reset,
        lambda h: h,
        new_lbfgs_history,
    )
    new_lbfgs_history = jax.lax.cond(
        new_consecutive_ls_failures >= self.ls_failure_patience,
        lbfgs_identity_reset,
        lambda h: h,
        new_lbfgs_history,
    )

    is_zero_step_post = jnp.reshape(
        (alpha * d_norm < self.atol) & qp_result.converged & ls_result.success,
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

    merit_new = compute_merit(f_val_new, eq_val_new, ineq_val_new, merit_penalty)
    merit_threshold = self.stagnation_tol * jnp.maximum(jnp.abs(state.best_merit), 1.0)
    improved = merit_new < state.best_merit - merit_threshold
    new_best_merit = jnp.where(improved, merit_new, state.best_merit)
    new_best_x = jnp.where(improved, y_new, state.best_x)
    new_steps_without = jnp.where(
        improved, jnp.array(0), state.steps_without_improvement + 1
    )
    patience = self._stagnation_window
    merit_stagnation = (state.step_count >= patience) & (new_steps_without >= patience)

    blowup_threshold = self.divergence_factor * jnp.maximum(
        jnp.abs(new_best_merit), 1.0
    )
    merit_finite = jnp.isfinite(merit_new)
    blowup_now = ((merit_new - new_best_merit) > blowup_threshold) | ~merit_finite
    new_blowup_count = jnp.where(blowup_now, state.blowup_count + 1, jnp.array(0))
    diverging_now = jnp.reshape(new_blowup_count >= self.divergence_patience, ())
    y_returned = jnp.where(diverging_now, new_best_x, y_new)

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
    )

    if m_bounds_static > 0:
        multipliers_ineq_for_state = qp_result.multipliers_ineq
        if n_lower_static > 0:
            multipliers_ineq_for_state = multipliers_ineq_for_state.at[
                m_ineq_general_static : m_ineq_general_static + n_lower_static
            ].set(mu_lower_corr)
        if self._n_upper_bounds > 0:
            multipliers_ineq_for_state = multipliers_ineq_for_state.at[
                m_ineq_general_static + n_lower_static :
            ].set(mu_upper_corr)
    else:
        multipliers_ineq_for_state = qp_result.multipliers_ineq

    # Granular termination-code classification, mirrored at the new
    # iterate so ``state.termination_code`` agrees with the value
    # ``terminate()`` will eventually settle on.
    m_eq_static = self.n_eq_constraints
    m_ineq_total_static = (
        self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
    )
    eq_feasible_new = (
        jnp.max(jnp.abs(eq_val_new)) <= self.atol
        if m_eq_static > 0
        else jnp.array(True)
    )
    ineq_feasible_new = (
        jnp.max(jnp.maximum(0.0, -ineq_val_new)) <= self.atol
        if m_ineq_total_static > 0
        else jnp.array(True)
    )
    primal_feasible_new = jnp.reshape(eq_feasible_new & ineq_feasible_new, ())
    lagrangian_val_new = f_val_new
    if m_eq_static > 0:
        lagrangian_val_new = lagrangian_val_new - jnp.dot(
            qp_result.multipliers_eq, eq_val_new
        )
    if m_ineq_total_static > 0:
        lagrangian_val_new = lagrangian_val_new - jnp.dot(
            multipliers_ineq_for_state, ineq_val_new
        )
    grad_norm_new = jnp.linalg.norm(grad_lagrangian_new)
    rtol_target_new = self.rtol * jnp.maximum(jnp.abs(lagrangian_val_new), 1.0)
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

    flags = TerminationFlags(
        converged=converged_new,
        nonfinite=jnp.reshape(nonfinite_new, ()),
        diverging=diverging_now,
        ls_fatal=jnp.reshape(ls_fatal, ()),
        qp_fatal=jnp.reshape(qp_fatal, ()),
        merit_stagnation=jnp.reshape(merit_stagnation, ()),
        max_iters_reached=jnp.reshape(max_iters_reached_new, ()),
        primal_feasible=primal_feasible_new,
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
        multipliers_eq=qp_result.multipliers_eq,
        multipliers_ineq=multipliers_ineq_for_state,
        prev_grad_lagrangian=grad_lagrangian_new,
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
        - jnp.dot(qp_result.multipliers_eq, eq_val_new)
        - jnp.dot(qp_result.multipliers_ineq, ineq_val_new)
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
        proj_grad_norm=("|W̃g|", qp_result.projected_grad_norm, ".3e"),
        grad_norm=("|∇f|", grad_norm, ".3e"),
        step_size=("α", alpha, ".3e"),
        direction_norm=("|d|", dir_norm, ".3e"),
        merit=("merit", merit_new, ".6e"),
        merit_delta=("Δmerit", merit_delta, "+.2e"),
        stag_count=("stag#", new_steps_without),
        stagnation=("stag", merit_stagnation),
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
    m_ineq_total = self.n_ineq_constraints + self._n_lower_bounds + self._n_upper_bounds
    grad_lagrangian = state.grad_lagrangian

    lagrangian_val = state.f_val
    if m_eq > 0:
        lagrangian_val = lagrangian_val - state.multipliers_eq @ state.eq_val
    if m_ineq_total > 0:
        lagrangian_val = lagrangian_val - state.multipliers_ineq @ state.ineq_val

    grad_norm = jnp.linalg.norm(grad_lagrangian)
    rtol_target = self.rtol * jnp.maximum(jnp.abs(lagrangian_val), 1.0)
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

    flags = TerminationFlags(
        converged=jnp.reshape(converged, ()),
        nonfinite=jnp.reshape(nonfinite_iter, ()),
        diverging=jnp.reshape(state.diverging, ()),
        ls_fatal=jnp.reshape(state.ls_fatal, ()),
        qp_fatal=jnp.reshape(state.qp_fatal, ()),
        merit_stagnation=jnp.reshape(state.stagnation, ()),
        max_iters_reached=jnp.reshape(max_iters_reached, ()),
        primal_feasible=jnp.reshape(primal_feasible, ()),
    )
    return coarse_outcome(flags)


__all__ = ["_step_impl", "_terminate_impl"]
