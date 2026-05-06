"""Box-bound active-set loop run on top of the QP solution.

Previously inlined inside ``SLSQP._solve_qp_subproblem``, this module
hosts the iterative bound-fixing post-pass: after the main QP solve
returns a direction in the unconstrained-on-bounds space, fix the
variables that violate their box bounds, re-solve the inner equality-
constrained QP in the reduced free subspace, check for new violations
and wrong-sign bound multipliers, repeat (up to 5 passes).

The loop also recovers the bound multipliers from the reduced gradient
``Bd + g − A^T λ`` at the final clipped direction and packages
everything into the outer-facing :class:`QPResult`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.state import InnerSolveResult, QPResult, QPSolverResult
from slsqp_jax.types import Scalar, Vector


def run_bound_fixing(
    qp_result: QPSolverResult,
    *,
    inner_solver: AbstractInnerSolver,
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq_general: Float[Array, "m_gen n"],
    b_ineq_general: Float[Array, " m_gen"],
    n_eq_constraints: int,
    m_ineq_general: int,
    bounds: Float[Array, "n 2"] | None,
    y: Vector,
    n_lower_bounds: int,
    n_upper_bounds: int,
    lower_indices: tuple[int, ...] | None,
    upper_indices: tuple[int, ...] | None,
    precond_fn: Callable[[Vector], Vector] | None,
    adaptive_tol: Scalar | float | None,
    lpeca_bound_lower: Bool[Array, " n"],
    lpeca_bound_upper: Bool[Array, " n"],
    lpeca_bypassed: Bool[Array, ""],
    lpeca_capped: Bool[Array, ""],
    lpeca_bounds_prefixed_count: Array,
) -> QPResult:
    """Run the bound-fixing post-pass on top of a QP solution.

    Args:
        qp_result: Result of the main QP solve (no bound block).
        inner_solver: The inner equality-constrained QP solver.  Used
            for the reduced-space re-solves in each bound-fixing pass.
        hvp_fn: Lagrangian HVP.
        g: Objective gradient.
        A_eq, b_eq: Equality constraint matrix / RHS.
        A_ineq_general, b_ineq_general: General-inequality matrix / RHS
            (bounds are NOT included; they are handled here).
        n_eq_constraints: Number of equality constraints.
        m_ineq_general: Number of general inequality constraints
            (excluding bounds).
        bounds: ``(n, 2)`` array of ``[lower, upper]`` per variable, or
            ``None`` (in which case this function should not be called).
        y: Current iterate (for converting bounds to direction-space
            limits ``d_lower = lb - y``, ``d_upper = ub - y``).
        n_lower_bounds, n_upper_bounds: Counts of finite lower / upper
            bounds.
        lower_indices, upper_indices: Variable indices with finite
            lower / upper bounds (precomputed at NLP construction).
        precond_fn: Preconditioner forwarded to the inner solver.
        adaptive_tol: Eisenstat-Walker adaptive CG tolerance, or
            ``None``.
        lpeca_bound_lower, lpeca_bound_upper: LPEC-A predicted active
            bound masks (length ``n``).  All-False when LPEC-A is
            disabled / bypassed / not predicting bounds.
        lpeca_bypassed, lpeca_capped: LPEC-A status flags forwarded to
            the outer ``QPResult``.
        lpeca_bounds_prefixed_count: Number of variables warm-started
            from LPEC-A bound predictions (diagnostic).

    Returns:
        Outer-facing ``QPResult`` with bound multipliers recovered and
        all bound-related diagnostics populated.
    """
    assert bounds is not None
    assert lower_indices is not None
    assert upper_indices is not None

    direction = qp_result.d
    n_vars = g.shape[0]

    d_lower = bounds[:, 0] - y
    d_upper = bounds[:, 1] - y
    finite_lower = jnp.isfinite(d_lower)
    finite_upper = jnp.isfinite(d_upper)

    A_combined = jnp.concatenate([A_eq, A_ineq_general], axis=0)
    b_combined = jnp.concatenate([b_eq, b_ineq_general], axis=0)
    eq_active = jnp.ones(n_eq_constraints, dtype=bool)
    combined_active = jnp.concatenate([eq_active, qp_result.active_set])

    # LPEC-A bound warm-start (all-False when disabled / bypassed).
    has_lpeca_bound_prefix = jnp.any(lpeca_bound_lower | lpeca_bound_upper)
    free_mask = ~(lpeca_bound_lower | lpeca_bound_upper)
    d_fixed = jnp.where(
        lpeca_bound_lower,
        d_lower,
        jnp.where(lpeca_bound_upper, d_upper, jnp.zeros(n_vars)),
    )
    mult_combined = jnp.zeros(A_combined.shape[0])

    bound_fix_solves = jnp.array(0)
    proj_residual_accum = qp_result.proj_residual
    n_proj_refinements_accum = qp_result.n_proj_refinements
    projected_grad_norm_accum = qp_result.projected_grad_norm

    bound_fix_tol = 1e-12
    for _bound_pass in range(5):
        # --- Add step: fix free variables that violate bounds ---
        add_lower = (direction <= d_lower + bound_fix_tol) & finite_lower & free_mask
        add_upper = (direction >= d_upper - bound_fix_tol) & finite_upper & free_mask
        add_set = add_lower | add_upper

        # --- Drop step: release fixed variables with wrong-sign bound
        # multipliers.  A lower-bound multiplier should be >= 0 (pushing
        # the variable up); if negative, the variable wants to move away
        # from the bound and should be freed.  Similarly for upper bounds.
        Bd_cur = hvp_fn(direction)
        grad_qp_cur = Bd_cur + g
        cf = jnp.zeros_like(g)
        if n_eq_constraints > 0:
            cf = cf + A_eq.T @ mult_combined[:n_eq_constraints]
        if m_ineq_general > 0:
            cf = cf + A_ineq_general.T @ mult_combined[n_eq_constraints:]
        reduced_grad_cur = grad_qp_cur - cf

        at_lower_cur = ~free_mask & (d_fixed <= d_lower + bound_fix_tol)
        at_upper_cur = ~free_mask & (d_fixed >= d_upper - bound_fix_tol)
        drop_lower = at_lower_cur & (reduced_grad_cur < -bound_fix_tol)
        drop_upper = at_upper_cur & (-reduced_grad_cur < -bound_fix_tol)
        drop_set = drop_lower | drop_upper

        any_change = jnp.any(add_set | drop_set)

        # On the first pass, force the reduced-space solve when LPEC-A
        # pre-fixed any bounds, even if no add/drop change is needed.
        force_initial_solve = jnp.array(_bound_pass == 0) & has_lpeca_bound_prefix

        new_free_mask = (free_mask & ~add_set) | drop_set
        new_d_fixed = jnp.where(
            add_lower,
            d_lower,
            jnp.where(add_upper, d_upper, d_fixed),
        )
        new_d_fixed = jnp.where(drop_set, 0.0, new_d_fixed)

        free_mask = jnp.where(any_change, new_free_mask, free_mask)
        d_fixed = jnp.where(any_change, new_d_fixed, d_fixed)

        any_fixed = ~jnp.all(free_mask)
        needs_solve = (any_change | force_initial_solve) & any_fixed

        def _do_solve(_=None):
            return inner_solver.solve(
                hvp_fn,
                g,
                A_combined,
                b_combined,
                combined_active,
                precond_fn=precond_fn,
                free_mask=free_mask,
                d_fixed=d_fixed,
                adaptive_tol=adaptive_tol,
            )

        def _skip_solve(_=None):
            # PyTree shape parity with ``_do_solve`` is required by
            # ``jax.lax.cond``.
            return InnerSolveResult(
                d=direction,
                multipliers=mult_combined,
                converged=jnp.array(True),
                proj_residual=jnp.asarray(0.0, dtype=direction.dtype),
                n_proj_refinements=jnp.asarray(0),
                projected_grad_norm=jnp.asarray(jnp.inf, dtype=direction.dtype),
            )

        bound_result = jax.lax.cond(needs_solve, _do_solve, _skip_solve, operand=None)
        d_new = bound_result.d
        mult_new = bound_result.multipliers

        bound_fix_solves = bound_fix_solves + jnp.where(needs_solve, 1, 0)

        use_new = needs_solve
        direction = jnp.where(use_new, d_new, direction)
        mult_combined = jnp.where(use_new, mult_new, mult_combined)

        proj_residual_accum = jnp.where(
            use_new,
            bound_result.proj_residual.astype(proj_residual_accum.dtype),
            proj_residual_accum,
        )
        n_proj_refinements_accum = (
            n_proj_refinements_accum + bound_result.n_proj_refinements
        )
        projected_grad_norm_accum = jnp.where(
            use_new,
            bound_result.projected_grad_norm.astype(projected_grad_norm_accum.dtype),
            projected_grad_norm_accum,
        )

    # Final bound-active identification from the converged direction.
    at_lower_full = (direction <= d_lower + bound_fix_tol) & finite_lower
    at_upper_full = (direction >= d_upper - bound_fix_tol) & finite_upper
    any_bound_active = jnp.any(at_lower_full | at_upper_full)

    mult_eq_final = jnp.where(
        any_bound_active,
        mult_combined[:n_eq_constraints],
        qp_result.multipliers_eq,
    )
    mult_gen_final = (
        jnp.where(
            any_bound_active,
            mult_combined[n_eq_constraints:],
            qp_result.multipliers_ineq,
        )
        if m_ineq_general > 0
        else qp_result.multipliers_ineq
    )

    # Recover bound multipliers from the reduced gradient.
    lower_idx = np.array(lower_indices)
    upper_idx = np.array(upper_indices)

    Bd = hvp_fn(direction)
    grad_qp = Bd + g
    constraint_force = jnp.zeros_like(g)
    if n_eq_constraints > 0:
        constraint_force = constraint_force + A_eq.T @ mult_eq_final
    if m_ineq_general > 0:
        constraint_force = constraint_force + A_ineq_general.T @ mult_gen_final
    reduced_grad = grad_qp - constraint_force

    at_lower = (
        at_lower_full[lower_idx] if len(lower_idx) > 0 else jnp.zeros((0,), dtype=bool)
    )
    at_upper = (
        at_upper_full[upper_idx] if len(upper_idx) > 0 else jnp.zeros((0,), dtype=bool)
    )

    bound_mult_lower = (
        jnp.where(at_lower, reduced_grad[lower_idx], 0.0)
        if len(lower_idx) > 0
        else jnp.zeros((0,))
    )
    bound_mult_upper = (
        jnp.where(at_upper, -reduced_grad[upper_idx], 0.0)
        if len(upper_idx) > 0
        else jnp.zeros((0,))
    )

    multipliers_eq = mult_eq_final
    multipliers_ineq = jnp.concatenate(
        [mult_gen_final, bound_mult_lower, bound_mult_upper]
    )
    active_set = jnp.concatenate([qp_result.active_set, at_lower, at_upper])
    n_bound_fixed = jnp.sum((at_lower_full | at_upper_full).astype(jnp.int32))

    return QPResult(  # ty: ignore[invalid-return-type]
        direction=direction,
        multipliers_eq=multipliers_eq,
        multipliers_ineq=multipliers_ineq,
        active_set=active_set,
        converged=qp_result.converged,
        iterations=qp_result.iterations,
        bound_fix_solves=bound_fix_solves,
        n_bound_fixed=n_bound_fixed,
        ping_ponged=qp_result.ping_ponged,
        reached_max_iter=qp_result.reached_max_iter,
        lpeca_bypassed=lpeca_bypassed,
        lpeca_capped=lpeca_capped,
        n_lpeca_bounds_prefixed=lpeca_bounds_prefixed_count,
        proj_residual=proj_residual_accum,
        n_proj_refinements=n_proj_refinements_accum,
        projected_grad_norm=projected_grad_norm_accum,
    )


def package_qp_result_no_bounds(qp_result: QPSolverResult) -> QPResult:
    """Wrap a :class:`QPSolverResult` as the outer :class:`QPResult` when
    no bound constraints are present.

    Mirrors the ``else`` branch of the legacy ``_solve_qp_subproblem``
    bound-fixing block: copy through multipliers and active set, set the
    bound-related diagnostics to zero / False, and copy the projection
    diagnostics from the inner result.
    """
    return QPResult(  # ty: ignore[invalid-return-type]
        direction=qp_result.d,
        multipliers_eq=qp_result.multipliers_eq,
        multipliers_ineq=qp_result.multipliers_ineq,
        active_set=qp_result.active_set,
        converged=qp_result.converged,
        iterations=qp_result.iterations,
        bound_fix_solves=jnp.array(0),
        n_bound_fixed=jnp.array(0),
        ping_ponged=qp_result.ping_ponged,
        reached_max_iter=qp_result.reached_max_iter,
        lpeca_bypassed=jnp.array(False),
        lpeca_capped=jnp.array(False),
        n_lpeca_bounds_prefixed=jnp.array(0),
        proj_residual=qp_result.proj_residual,
        n_proj_refinements=qp_result.n_proj_refinements,
        projected_grad_norm=qp_result.projected_grad_norm,
    )


__all__ = ["package_qp_result_no_bounds", "run_bound_fixing"]
