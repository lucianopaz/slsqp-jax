"""QP Subproblem Solver for SLSQP.

This module implements a QP solver for the Quadratic Programming
subproblem that arises at each SLSQP iteration.

The QP subproblem has the form:
    minimize    (1/2) d^T H d + g^T d
    subject to  A_eq d = b_eq
                A_ineq d >= b_ineq

The solver uses a pluggable inner solver (see ``inner_solver.py``) for
the equality-constrained QP subproblem, wrapped in an **active-set**
method for inequality constraints. The Hessian is accessed only through
a Hessian-vector product (HVP) function, enabling matrix-free operation
for large-scale problems (n > 5000).

For inequality constraints A d >= b, the Lagrangian is:
    L(d, lambda) = (1/2) d^T H d + g^T d - lambda^T (A d - b)

with lambda >= 0 for active constraints.

**Anti-cycling.**  The active-set loop uses the EXPAND procedure
(Gill, Murray, Saunders & Wright, *Math. Programming* 45, 1989) to
prevent cycling caused by degenerate constraints.  A working
feasibility tolerance ``delta_k = tol + k * tau`` increases
monotonically at each active-set iteration, ensuring strict progress
and preventing the same constraint from being repeatedly activated
and deactivated.

**Proximal stabilization (sSQP).**  When equality constraints are
present, they are absorbed into the QP objective through an
augmented-Lagrangian penalty with adaptive parameter ``mu``, following
Hager (*Comp. Optim. Appl.*, 1999) and Wright (*Math. Oper. Res.*,
2002, eq 6.6).  This regularizes the dual solution and prevents QP
infeasibility at degenerate vertices.
"""

from collections.abc import Callable
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from slsqp_jax.inner_solver import (
    AbstractInnerSolver,
    ProjectedCGCholesky,
    solve_unconstrained_cg,
)
from slsqp_jax.types import Scalar, Vector


class QPState(eqx.Module):
    """State for the Active Set QP solver.

    ``any_inner_failure`` accumulates whether any inner-solver call
    during the active-set loop reported non-convergence or produced a
    non-finite direction.  The final ``QPResult.converged`` combines
    this with the active-set completion check.

    ``last_add_idx`` / ``last_drop_idx`` / ``ping_pong_count`` /
    ``ping_ponged`` implement an explicit anti-cycling guard on top of
    EXPAND: when the active-set loop alternately adds and drops the
    *same* constraint several times in a row (typical signature of
    multiplier-recovery noise on a degenerate vertex) the loop is
    short-circuited as ``converged=True`` with the sticky
    ``ping_ponged`` flag set so the outer solver can surface it
    through the diagnostic counters.  Indices are initialised to
    ``-1`` to mean "no add/drop yet".
    """

    d: Vector
    active_set: Bool[Array, " m_ineq"]
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]
    any_inner_failure: Bool[Array, ""]
    last_add_idx: Int[Array, ""]
    last_drop_idx: Int[Array, ""]
    ping_pong_count: Int[Array, ""]
    ping_ponged: Bool[Array, ""]


def _inner_ok(result) -> Bool[Array, ""]:
    """Did the inner solve return a usable (finite) direction?

    NOTE: we deliberately do *not* require ``result.converged`` here.
    For an outer SQP solver an imprecise inner CG/MINRES direction is
    usually still a productive descent direction -- failing the QP just
    because CG ran out of iterations would trigger an L-BFGS reset and
    a projected-gradient fallback at every outer step, which is far
    worse than taking a slightly stale Newton direction.  We flag only
    unusable directions (NaN/Inf), and track CG non-convergence via the
    diagnostic counters in ``SLSQPState`` instead.
    """
    return jnp.isfinite(result.d).all()


class QPResult(NamedTuple):
    """Result from the QP solver.

    The Bundle 1 diagnostic fields ``ping_ponged`` / ``reached_max_iter``
    / ``final_working_tol`` are surfaced from the active-set loop so
    the outer solver can track *why* the QP stopped (clean convergence
    versus cycling versus iteration-budget exhaustion).  They default
    to ``False`` / ``False`` / ``0.0`` for the trivial QP paths that
    do not run an active-set loop.
    """

    d: Vector
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    active_set: Bool[Array, " m_ineq"]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]
    ping_ponged: Bool[Array, ""]
    reached_max_iter: Bool[Array, ""]
    final_working_tol: Scalar


def _solve_qp_proximal(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    m_eq: int,
    m_ineq: int,
    max_iter: int,
    max_cg_iter: int,
    tol: Scalar | float,
    expand_factor: float,
    initial_active_set: Bool[Array, " m_ineq"] | None,
    kkt_residual: Scalar | float,
    proximal_mu: Scalar | float,
    prev_multipliers_eq: Float[Array, " m_eq"] | None,
    inner_solver: AbstractInnerSolver,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
    predicted_active_set: Bool[Array, " m_ineq"] | None = None,
    use_expand: bool = True,
    mult_drop_floor: float = 1e-6,
    ping_pong_threshold: int = 2**31 - 1,
) -> QPResult:
    """Solve the QP using the stabilized SQP (sSQP) formulation.

    Equality constraints are absorbed into the objective via an
    augmented-Lagrangian penalty with weight ``1/mu``.
    The active-set loop operates on inequality constraints only.

    The stabilized objective is::

        (1/2) d^T B_tilde d + g_tilde^T d

    where ``B_tilde(v) = H v + (1/mu) A_eq^T (A_eq v)`` and
    ``g_tilde = g - (1/mu) A_eq^T b_eq - A_eq^T lambda_k``.

    Equality multipliers are recovered from the penalty optimality
    condition: ``lambda = lambda_k - (1/mu)(A_eq d - b_eq)``.
    """
    inv_mu = 1.0 / jnp.maximum(proximal_mu, 1e-10)
    prev_mult_eq = (
        prev_multipliers_eq if prev_multipliers_eq is not None else jnp.zeros((m_eq,))
    )
    inner_cg_tol: Scalar | float = cg_tol if cg_tol is not None else tol

    def stabilized_hvp(v: Vector) -> Vector:
        return hvp_fn(v) + inv_mu * (A_eq.T @ (A_eq @ v))

    g_mod = g - inv_mu * (A_eq.T @ b_eq) - A_eq.T @ prev_mult_eq

    def _recover_mult_eq(d: Vector) -> Float[Array, " m_eq"]:
        return prev_mult_eq - inv_mu * (A_eq @ d - b_eq)

    # Sub-case: no inequality constraints — just unconstrained CG.
    if m_ineq == 0:
        d, _cg_converged = solve_unconstrained_cg(
            stabilized_hvp,
            g_mod,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )
        finite_d = jnp.isfinite(d).all()
        return QPResult(
            d=d,
            multipliers_eq=_recover_mult_eq(d),
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=finite_d,
            iterations=jnp.array(1),
            ping_ponged=jnp.array(False),
            reached_max_iter=jnp.array(False),
            final_working_tol=jnp.asarray(0.0, dtype=jnp.float64),
        )

    # Sub-case: inequalities present — active-set loop on A_ineq only
    kkt_res = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_res, 1.0) * tol
    _adaptive_tol: Scalar | float | None = cg_tol

    # Initial unconstrained solve (equalities absorbed into objective)
    d_init, _d_init_converged = solve_unconstrained_cg(
        stabilized_hvp,
        g_mod,
        max_cg_iter,
        inner_cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
    )
    init_inner_failure = ~jnp.isfinite(d_init).all()

    # Determine starting active set
    residuals_init = A_ineq @ d_init - b_ineq
    if predicted_active_set is not None:
        init_active = predicted_active_set | (residuals_init < -base_tol)
    elif initial_active_set is not None:
        init_active = initial_active_set | (residuals_init < -base_tol)
    else:
        init_active = residuals_init < -base_tol
    init_converged = ~jnp.any(init_active)

    init_state = QPState(
        d=d_init,
        active_set=init_active,
        multipliers_eq=_recover_mult_eq(d_init),
        multipliers_ineq=jnp.zeros((m_ineq,)),
        iteration=jnp.array(0),
        converged=init_converged,
        any_inner_failure=init_inner_failure,
        last_add_idx=jnp.array(-1),
        last_drop_idx=jnp.array(-1),
        ping_pong_count=jnp.array(0),
        ping_ponged=jnp.array(False),
    )

    def cond_fn(state: QPState) -> Bool[Array, ""]:
        return ~state.converged & (state.iteration < max_iter)

    # EXPAND anti-cycling: ``working_tol = base_tol + k * tau`` with
    # ``tau = base_tol * expand_factor / max_iter``.  The tolerance
    # grows from ``base_tol`` at k=0 to ``base_tol * (1 + expand_factor)``
    # at k=max_iter, which widens both the constraint-violation and
    # multiplier-drop tests enough to break ties at degenerate vertices
    # while keeping the QP's working-set decisions coherent with the
    # outer SLSQP convergence tolerance.  Setting ``expand_factor = 0``
    # disables the ramp entirely.
    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)
    effective_tau = tau if use_expand else 0.0

    # Floor for the negative-multiplier drop test.  Multiplier recovery
    # via ``(AAᵀ + εI)^{-1}`` typically incurs O(eps · cond(AAᵀ))
    # error; with the default 1e-6 floor, a freshly-added constraint
    # with true multiplier ~0 will not be dropped just because CG
    # recovered a slightly negative value, which is the canonical
    # add/drop ping-pong source.
    drop_floor = jnp.asarray(mult_drop_floor, dtype=jnp.float64)

    def body_fn(state: QPState) -> QPState:
        working_tol = base_tol + state.iteration * effective_tau

        # Solve with current active set — inequalities only
        result = inner_solver.solve(
            stabilized_hvp,
            g_mod,
            A_ineq,
            b_ineq,
            state.active_set,
            precond_fn=precond_fn,
            adaptive_tol=_adaptive_tol,
        )
        d_new = result.d
        mult_ineq_new = result.multipliers
        inner_failed = ~_inner_ok(result)
        new_any_inner_failure = state.any_inner_failure | inner_failed

        mult_eq_new = _recover_mult_eq(d_new)

        # Check feasibility with expanding tolerance
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -working_tol) & ~state.active_set
        any_violated = jnp.any(violated)

        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        # Noise-aware drop: ignore negative multipliers smaller in
        # magnitude than ``drop_floor``; these are the residual
        # multiplier-recovery noise that triggers ping-pong.
        drop_tol = jnp.maximum(working_tol, drop_floor)
        negative_mult = (mult_ineq_new < -drop_tol) & state.active_set
        any_negative = jnp.any(negative_mult)

        mult_scores = jnp.where(state.active_set, mult_ineq_new, jnp.inf)
        most_negative_idx = jnp.argmin(mult_scores)

        def add_constraint():
            # Ping-pong detector: if we are about to add the constraint
            # we just dropped, count it.  When the run hits the
            # threshold, freeze the loop (do NOT apply the add) and
            # mark converged with the sticky ``ping_ponged`` flag so
            # the outer solver knows.
            is_pp = (state.last_drop_idx >= 0) & (
                most_violated_idx == state.last_drop_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_violated_idx].set(True),
            )
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=jnp.where(
                    triggered, state.last_add_idx, most_violated_idx
                ),
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
            )

        def drop_constraint():
            is_pp = (state.last_add_idx >= 0) & (
                most_negative_idx == state.last_add_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_negative_idx].set(False),
            )
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=jnp.where(
                    triggered, state.last_drop_idx, most_negative_idx
                ),
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
            )

        def mark_converged():
            return QPState(
                d=d_new,
                active_set=state.active_set,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(True),
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=state.ping_pong_count,
                ping_ponged=state.ping_ponged,
            )

        return jax.lax.cond(
            any_violated,
            add_constraint,
            lambda: jax.lax.cond(any_negative, drop_constraint, mark_converged),
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # The EXPAND procedure's growing tolerance can cause mark_converged()
    # to fire on the last iteration under relaxed tolerances.  Override
    # convergence when the iteration limit was actually reached.  A
    # genuine ping-pong short-circuit also counts as convergence (the
    # loop did not exhaust its budget).  any_inner_failure is kept
    # on QPState for diagnostics only — treating a merely-imprecise
    # inner CG/MINRES iterate as a QP failure would trigger the outer
    # steepest-descent fallback on every truncated solve and regress
    # production paths that rely on those directions for progress.
    reached_max_iter = final_state.iteration >= max_iter
    final_converged = final_state.converged & ~reached_max_iter
    final_working_tol = base_tol + final_state.iteration * effective_tau

    return QPResult(
        d=final_state.d,
        multipliers_eq=final_state.multipliers_eq,
        multipliers_ineq=final_state.multipliers_ineq,
        active_set=final_state.active_set,
        converged=final_converged,
        iterations=final_state.iteration,
        ping_ponged=final_state.ping_ponged,
        reached_max_iter=reached_max_iter,
        final_working_tol=jnp.asarray(final_working_tol, dtype=jnp.float64),
    )


def _solve_qp_direct(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    m_eq: int,
    m_ineq: int,
    max_iter: int,
    tol: Scalar | float,
    expand_factor: float,
    initial_active_set: Bool[Array, " m_ineq"] | None,
    kkt_residual: Scalar | float,
    inner_solver: AbstractInnerSolver,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
    predicted_active_set: Bool[Array, " m_ineq"] | None = None,
    use_expand: bool = True,
    mult_drop_floor: float = 1e-6,
    ping_pong_threshold: int = 2**31 - 1,
) -> QPResult:
    """Solve the QP with equality constraints enforced via direct projection.

    Unlike ``_solve_qp_proximal``, equality constraints are not absorbed
    into the objective via an augmented-Lagrangian penalty.  Instead,
    they are enforced exactly through the null-space projector in the
    inner solver.  This avoids the ill-conditioning from the
    ``(1/mu) A_eq^T A_eq`` proximal term.

    A combined constraint matrix ``[A_eq; A_ineq]`` is formed.  Equality
    rows are permanently active; the active-set loop only adds/drops
    inequality rows.
    """
    A_combined = jnp.concatenate([A_eq, A_ineq], axis=0)
    b_combined = jnp.concatenate([b_eq, b_ineq], axis=0)
    eq_active = jnp.ones(m_eq, dtype=bool)

    _adaptive_tol: Scalar | float | None = cg_tol

    if m_ineq == 0:
        # Equality-only: single projected CG/MINRES solve, no active-set
        # loop.  Forward the inner solver's convergence flag so outer
        # logic (penalty gating, steepest-descent fallback) can react to
        # inner-solver stagnation instead of trusting a spurious success.
        combined_active = eq_active
        result = inner_solver.solve(
            hvp_fn,
            g,
            A_combined,
            b_combined,
            combined_active,
            precond_fn=precond_fn,
            adaptive_tol=_adaptive_tol,
        )
        return QPResult(
            d=result.d,
            multipliers_eq=result.multipliers[:m_eq],
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=_inner_ok(result),
            iterations=jnp.array(1),
            ping_ponged=jnp.array(False),
            reached_max_iter=jnp.array(False),
            final_working_tol=jnp.asarray(0.0, dtype=jnp.float64),
        )

    # Equality + inequality: active-set loop on the inequality portion.
    kkt_res = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_res, 1.0) * tol

    # Initial solve with equalities only (inequalities inactive).
    init_result = inner_solver.solve(
        hvp_fn,
        g,
        A_combined,
        b_combined,
        jnp.concatenate([eq_active, jnp.zeros(m_ineq, dtype=bool)]),
        precond_fn=precond_fn,
        adaptive_tol=_adaptive_tol,
    )
    d_init = init_result.d
    mult_init = init_result.multipliers
    init_inner_failure = ~_inner_ok(init_result)

    residuals_init = A_ineq @ d_init - b_ineq
    if predicted_active_set is not None:
        init_ineq_active = predicted_active_set | (residuals_init < -base_tol)
    elif initial_active_set is not None:
        init_ineq_active = initial_active_set | (residuals_init < -base_tol)
    else:
        init_ineq_active = residuals_init < -base_tol  # pragma: no cover
    init_converged = ~jnp.any(init_ineq_active)

    init_state = QPState(
        d=d_init,
        active_set=init_ineq_active,
        multipliers_eq=mult_init[:m_eq],
        multipliers_ineq=jnp.zeros((m_ineq,)),
        iteration=jnp.array(0),
        converged=init_converged,
        any_inner_failure=init_inner_failure,
        last_add_idx=jnp.array(-1),
        last_drop_idx=jnp.array(-1),
        ping_pong_count=jnp.array(0),
        ping_ponged=jnp.array(False),
    )

    def cond_fn(state: QPState) -> Bool[Array, ""]:
        return ~state.converged & (state.iteration < max_iter)

    # EXPAND ramp (see ``_solve_qp_proximal``).
    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)
    effective_tau = tau if use_expand else 0.0
    drop_floor = jnp.asarray(mult_drop_floor, dtype=jnp.float64)

    def body_fn(state: QPState) -> QPState:
        working_tol = base_tol + state.iteration * effective_tau

        combined_active = jnp.concatenate([eq_active, state.active_set])
        result = inner_solver.solve(
            hvp_fn,
            g,
            A_combined,
            b_combined,
            combined_active,
            precond_fn=precond_fn,
            adaptive_tol=_adaptive_tol,
        )
        d_new = result.d
        mult_all = result.multipliers
        inner_failed = ~_inner_ok(result)
        new_any_inner_failure = state.any_inner_failure | inner_failed

        mult_eq_new = mult_all[:m_eq]
        mult_ineq_new = mult_all[m_eq:]

        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -working_tol) & ~state.active_set
        any_violated = jnp.any(violated)

        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        drop_tol = jnp.maximum(working_tol, drop_floor)
        negative_mult = (mult_ineq_new < -drop_tol) & state.active_set
        any_negative = jnp.any(negative_mult)

        mult_scores = jnp.where(state.active_set, mult_ineq_new, jnp.inf)
        most_negative_idx = jnp.argmin(mult_scores)

        def add_constraint():
            is_pp = (state.last_drop_idx >= 0) & (
                most_violated_idx == state.last_drop_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_violated_idx].set(True),
            )
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=jnp.where(
                    triggered, state.last_add_idx, most_violated_idx
                ),
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
            )

        def drop_constraint():
            is_pp = (state.last_add_idx >= 0) & (
                most_negative_idx == state.last_add_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_negative_idx].set(False),
            )
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=jnp.where(
                    triggered, state.last_drop_idx, most_negative_idx
                ),
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
            )

        def mark_converged():
            return QPState(
                d=d_new,
                active_set=state.active_set,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(True),
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=state.ping_pong_count,
                ping_ponged=state.ping_ponged,
            )

        return jax.lax.cond(
            any_violated,
            add_constraint,
            lambda: jax.lax.cond(any_negative, drop_constraint, mark_converged),
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    reached_max_iter = final_state.iteration >= max_iter
    final_converged = final_state.converged & ~reached_max_iter
    final_working_tol = base_tol + final_state.iteration * effective_tau

    return QPResult(
        d=final_state.d,
        multipliers_eq=final_state.multipliers_eq,
        multipliers_ineq=final_state.multipliers_ineq,
        active_set=final_state.active_set,
        converged=final_converged,
        iterations=final_state.iteration,
        ping_ponged=final_state.ping_ponged,
        reached_max_iter=reached_max_iter,
        final_working_tol=jnp.asarray(final_working_tol, dtype=jnp.float64),
    )


@jaxtyped(typechecker=beartype)
def solve_qp(
    hvp_fn: Callable,
    g: Vector,
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    max_iter: int = 100,
    max_cg_iter: int = 50,
    tol: float = 1e-8,
    expand_factor: float = 1.0,
    initial_active_set: Bool[Array, " m_ineq"] | None = None,
    kkt_residual: Scalar | float = 0.0,
    proximal_mu: Scalar | float = 0.0,
    prev_multipliers_eq: Float[Array, " m_eq"] | None = None,
    precond_fn: Callable | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
    use_proximal: bool = True,
    predicted_active_set: Bool[Array, " m_ineq"] | None = None,
    active_set_method: str = "expand",
    use_constraint_preconditioner: bool = False,
    inner_solver: AbstractInnerSolver | None = None,
    mult_drop_floor: float = 1e-6,
    ping_pong_threshold: int = 2**31 - 1,
) -> QPResult:
    """Solve a QP with equality and inequality constraints.

    Solves::

        minimize    (1/2) d^T H d + g^T d
        subject to  A_eq d = b_eq
                    A_ineq d >= b_ineq

    where H is provided implicitly via ``hvp_fn(v) = H @ v``.

    Uses a primal active-set method: at each iteration, active inequality
    constraints are treated as equalities, and the resulting
    equality-constrained QP is solved using the provided ``inner_solver``
    (defaulting to projected CG with Cholesky projection).
    Constraints are added/removed from the active set based on
    feasibility violations and multiplier signs until optimality is reached.

    To prevent cycling due to degenerate constraints, the EXPAND
    procedure is used: the feasibility tolerance increases by a small
    increment ``tau = tol * expand_factor / max_iter`` at every
    active-set iteration.  Set *expand_factor* to 0 to disable.

    When there are equality constraints, the solver uses the **stabilized
    SQP (sSQP)** formulation (Hager, 1999; Wright, 2002).  Equality
    constraints are absorbed into the objective via an augmented-Lagrangian
    penalty::

        minimize  (1/2) d^T B_tilde d + g_tilde^T d
        subject to  A_ineq d >= b_ineq

    where ``B_tilde(v) = H v + (1/mu) A_eq^T (A_eq v)`` and
    ``g_tilde = g - (1/mu) A_eq^T b_eq - A_eq^T lambda_k``.
    Equality multipliers are recovered as
    ``lambda = lambda_k - (1/mu)(A_eq d - b_eq)``.

    Args:
        hvp_fn: Hessian-vector product function v -> H @ v.
        g: Linear term of the objective (gradient).
        A_eq: Equality constraint matrix (m_eq x n).
        b_eq: Equality constraint RHS (m_eq,).
        A_ineq: Inequality constraint matrix (m_ineq x n).
        b_ineq: Inequality constraint RHS (m_ineq,).
        max_iter: Maximum active-set iterations.
        max_cg_iter: Maximum CG iterations per active-set step.  Used
            to construct a default ``inner_solver`` when none is provided.
        tol: Feasibility and optimality tolerance.
        expand_factor: Controls the EXPAND tolerance growth rate.
            The per-iteration increment is ``tol * expand_factor / max_iter``.
            Default 1.0 doubles the tolerance over the full iteration budget.
            Set to 0.0 to disable expansion.
        initial_active_set: Optional warm-start active set from a previous
            QP solve.  When provided, the active-set loop starts from this
            set instead of a cold-start violation check, promoting multiplier
            stability across outer SLSQP iterations (Wright, SIAM J. Optim.,
            2002, Section 8).
        kkt_residual: Norm of the KKT residual from the outer solver.
            When nonzero, the EXPAND base tolerance is widened
            proportionally so that the QP tolerates larger violations
            far from optimality and tightens automatically as convergence
            proceeds.
        proximal_mu: Adaptive proximal parameter for the sSQP formulation
            (Wright, 2002, eq 6.6).  Equality constraints are absorbed
            into the objective with penalty weight ``1/mu``.  Larger
            values mean more relaxation.  Computed adaptively as
            ``mu = max(kkt_residual^tau, mu_min)`` by the outer solver.
        prev_multipliers_eq: Equality multipliers from the previous outer
            iteration, used as the proximal center.
            When ``None``, defaults to zeros.
        precond_fn: Optional preconditioner function v -> M @ v where
            M approximates H^{-1}.  When provided, the inner CG solver
            uses preconditioned CG (PCG), which dramatically improves
            convergence on ill-conditioned subproblems.  Typically
            the L-BFGS inverse Hessian (two-loop recursion) is used.
        cg_tol: Optional CG convergence tolerance that overrides ``tol``
            for the inner CG solver only.  When ``None`` (default), the
            CG solver uses ``tol``.  This allows the CG tolerance to be
            adapted (e.g. Eisenstat-Walker) independently of the
            feasibility tolerance used by the active-set method.
        cg_regularization: Minimum eigenvalue threshold ``delta^2`` for the
            CG curvature guard.  CG declares "bad curvature" when
            ``p^T B p / ||p||^2 < delta^2``, preventing premature termination
            when the Hessian has small but positive eigenvalues.  Based on
            SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).  Default
            ``1e-6`` (delta ~ 1e-3).  Set to ``0.0`` to disable.
        use_proximal: When True (default), equality constraints are handled
            via the sSQP proximal path (augmented Lagrangian penalty).
            When False, equality constraints are enforced via direct
            null-space projection, avoiding the ill-conditioning introduced
            by the ``(1/mu) A_eq^T A_eq`` term.
        predicted_active_set: Optional LPEC-A predicted active set from
            the outer NLP solver.  Used when ``active_set_method`` is
            ``"lpeca_init"`` or ``"lpeca"`` to warm-start the active-set
            loop with a better initial estimate.  Merged with violated
            constraints (``residuals < -base_tol``) for the initial set.
        active_set_method: Controls how the active set is initialized and
            how tolerance grows during the active-set loop.  One of:

            - ``"expand"`` (default): Standard EXPAND anti-cycling
              procedure.  Uses ``initial_active_set`` for warm-starting
              and monotonically increasing tolerance.
            - ``"lpeca_init"``: Uses ``predicted_active_set`` from
              LPEC-A for initialization, then runs the EXPAND loop
              normally.  Provides a better starting point while
              retaining anti-cycling guarantees.
            - ``"lpeca"``: Uses ``predicted_active_set`` for
              initialization and runs with a fixed tolerance (no
              EXPAND growth).  Relies on LPEC-A accuracy for
              anti-cycling; ``max_iter`` provides a hard stop.
        use_constraint_preconditioner: When ``True`` and a preconditioner
            is provided, use the Gould-Hribar-Nocedal constraint
            preconditioner.  Only used when constructing a default
            ``inner_solver``.
        inner_solver: Pluggable strategy for the inner
            equality-constrained QP solve.  When ``None`` (default),
            a ``ProjectedCGCholesky`` is constructed from the
            ``max_cg_iter``, ``cg_regularization``, and
            ``use_constraint_preconditioner`` arguments.
        mult_drop_floor: Minimum magnitude (lower bound) on the
            negative-multiplier drop test.  An active inequality is
            only dropped when its recovered multiplier is *more
            negative* than ``-max(working_tol, mult_drop_floor)``.
            This prevents multiplier-recovery noise from the inner
            CG/MINRES solve from triggering spurious add/drop cycles
            on ill-conditioned ``AAᵀ``.  Default ``1e-6`` matches the
            typical iterative-refinement floor of the Cholesky
            projection.
        ping_pong_threshold: Number of consecutive add/drop reversals
            on the *same* index that triggers an explicit ping-pong
            short-circuit.  When the threshold is hit, the active-set
            loop returns its current direction with
            ``QPResult.ping_ponged=True`` and ``converged=True`` (the
            QP did not exhaust its budget; it stopped on a stable
            cycle).  Default 3.

    Returns:
        QPResult containing the solution, multipliers, active set, and
        convergence info.
    """
    if active_set_method not in ("expand", "lpeca_init", "lpeca"):
        raise ValueError(
            f"active_set_method must be 'expand', 'lpeca_init', or 'lpeca', "
            f"got {active_set_method!r}"
        )

    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]
    m_total = m_eq + m_ineq

    # LPEC-A modes: determine whether to use EXPAND and the effective
    # predicted active set for initialization.
    use_expand = active_set_method != "lpeca"
    effective_predicted = (
        predicted_active_set if active_set_method in ("lpeca_init", "lpeca") else None
    )

    # Resolve CG tolerance: use cg_tol if provided, else fall back to tol.
    inner_cg_tol: Scalar | float = cg_tol if cg_tol is not None else tol

    # Construct default inner solver when none is provided.
    if inner_solver is None:
        inner_solver = cast(
            AbstractInnerSolver,
            ProjectedCGCholesky(
                max_cg_iter=max_cg_iter,
                cg_tol=inner_cg_tol,
                cg_regularization=cg_regularization,
                use_constraint_preconditioner=use_constraint_preconditioner,
            ),
        )

    # Case 1: No constraints at all — truncated CG is always valid.
    if m_total == 0:
        d, _cg_converged = solve_unconstrained_cg(
            hvp_fn,
            g,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )
        finite_d = jnp.isfinite(d).all()
        return QPResult(
            d=d,
            multipliers_eq=jnp.zeros((0,)),
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=finite_d,
            iterations=jnp.array(1),
            ping_ponged=jnp.array(False),
            reached_max_iter=jnp.array(False),
            final_working_tol=jnp.asarray(0.0, dtype=jnp.float64),
        )

    # ---- Proximal stabilized path (sSQP) ----
    # Used when equality constraints are present and proximal is enabled.
    if m_eq > 0 and use_proximal:
        return _solve_qp_proximal(
            hvp_fn=hvp_fn,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            m_eq=m_eq,
            m_ineq=m_ineq,
            max_iter=max_iter,
            max_cg_iter=max_cg_iter,
            tol=tol,
            expand_factor=expand_factor,
            initial_active_set=initial_active_set,
            kkt_residual=kkt_residual,
            proximal_mu=proximal_mu,
            prev_multipliers_eq=prev_multipliers_eq,
            inner_solver=inner_solver,
            precond_fn=precond_fn,
            cg_tol=inner_cg_tol,
            cg_regularization=cg_regularization,
            predicted_active_set=effective_predicted,
            use_expand=use_expand,
            mult_drop_floor=mult_drop_floor,
            ping_pong_threshold=ping_pong_threshold,
        )

    # ---- Direct projection path (no proximal) ----
    # Equality constraints enforced via null-space projection.
    if m_eq > 0 and not use_proximal:
        return _solve_qp_direct(
            hvp_fn=hvp_fn,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            m_eq=m_eq,
            m_ineq=m_ineq,
            max_iter=max_iter,
            tol=tol,
            expand_factor=expand_factor,
            initial_active_set=initial_active_set,
            kkt_residual=kkt_residual,
            inner_solver=inner_solver,
            precond_fn=precond_fn,
            cg_tol=inner_cg_tol,
            cg_regularization=cg_regularization,
            predicted_active_set=effective_predicted,
            use_expand=use_expand,
            mult_drop_floor=mult_drop_floor,
            ping_pong_threshold=ping_pong_threshold,
        )

    # ---- Inequality-only path (m_eq == 0 guaranteed here) ----

    # Unconstrained initial solve
    d_init, _d_init_converged = solve_unconstrained_cg(
        hvp_fn,
        g,
        max_cg_iter,
        inner_cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
    )
    init_inner_failure = ~jnp.isfinite(d_init).all()

    kkt_residual = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_residual, 1.0) * tol

    # Determine starting active set: LPEC-A predicted, warm-start, or cold-start
    residuals_init = A_ineq @ d_init - b_ineq
    if effective_predicted is not None:
        init_active = effective_predicted | (residuals_init < -base_tol)
    elif initial_active_set is not None:
        init_active = initial_active_set | (residuals_init < -base_tol)
    else:
        init_active = residuals_init < -base_tol
    init_converged = ~jnp.any(init_active)

    init_state = QPState(
        d=d_init,
        active_set=init_active,
        multipliers_eq=jnp.zeros((0,)),
        multipliers_ineq=jnp.zeros((m_ineq,)),
        iteration=jnp.array(0),
        converged=init_converged,
        any_inner_failure=init_inner_failure,
        last_add_idx=jnp.array(-1),
        last_drop_idx=jnp.array(-1),
        ping_pong_count=jnp.array(0),
        ping_ponged=jnp.array(False),
    )

    def cond_fn(state: QPState) -> Bool[Array, ""]:
        return ~state.converged & (state.iteration < max_iter)

    # EXPAND ramp (see ``_solve_qp_proximal``).
    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)
    effective_tau = tau if use_expand else 0.0
    drop_floor = jnp.asarray(mult_drop_floor, dtype=jnp.float64)

    def body_fn(state: QPState) -> QPState:
        working_tol = base_tol + state.iteration * effective_tau

        # Solve with current active set using projected CG (ineq only)
        result = inner_solver.solve(
            hvp_fn,
            g,
            A_ineq,
            b_ineq,
            state.active_set,
            precond_fn=precond_fn,
            adaptive_tol=cg_tol,
        )
        d_new = result.d
        mult_ineq_new = result.multipliers
        inner_failed = ~_inner_ok(result)
        new_any_inner_failure = state.any_inner_failure | inner_failed

        # Check feasibility with expanding tolerance (stricter activation)
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -working_tol) & ~state.active_set
        any_violated = jnp.any(violated)

        # Find the most violated inactive constraint
        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        # Noise-aware drop: ignore negative multipliers smaller in
        # magnitude than ``drop_floor`` to suppress add/drop ping-pong
        # driven by multiplier-recovery noise.
        drop_tol = jnp.maximum(working_tol, drop_floor)
        negative_mult = (mult_ineq_new < -drop_tol) & state.active_set
        any_negative = jnp.any(negative_mult)

        mult_scores = jnp.where(state.active_set, mult_ineq_new, jnp.inf)
        most_negative_idx = jnp.argmin(mult_scores)

        empty_eq = jnp.zeros((0,))

        def add_constraint():
            is_pp = (state.last_drop_idx >= 0) & (
                most_violated_idx == state.last_drop_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_violated_idx].set(True),
            )
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=empty_eq,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=jnp.where(
                    triggered, state.last_add_idx, most_violated_idx
                ),
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
            )

        def drop_constraint():
            is_pp = (state.last_add_idx >= 0) & (
                most_negative_idx == state.last_add_idx
            )
            new_pp_count = jnp.where(is_pp, state.ping_pong_count + 1, 0)
            triggered = new_pp_count >= ping_pong_threshold
            new_active = jnp.where(
                triggered,
                state.active_set,
                state.active_set.at[most_negative_idx].set(False),
            )
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=empty_eq,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=triggered,
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=jnp.where(
                    triggered, state.last_drop_idx, most_negative_idx
                ),
                ping_pong_count=jnp.where(
                    triggered, state.ping_pong_count, new_pp_count
                ),
                ping_ponged=state.ping_ponged | triggered,
            )

        def mark_converged():
            return QPState(
                d=d_new,
                active_set=state.active_set,
                multipliers_eq=empty_eq,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(True),
                any_inner_failure=new_any_inner_failure,
                last_add_idx=state.last_add_idx,
                last_drop_idx=state.last_drop_idx,
                ping_pong_count=state.ping_pong_count,
                ping_ponged=state.ping_ponged,
            )

        return jax.lax.cond(
            any_violated,
            add_constraint,
            lambda: jax.lax.cond(any_negative, drop_constraint, mark_converged),
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    reached_max_iter = final_state.iteration >= max_iter
    final_converged = final_state.converged & ~reached_max_iter
    final_working_tol = base_tol + final_state.iteration * effective_tau

    return QPResult(
        d=final_state.d,
        multipliers_eq=final_state.multipliers_eq,
        multipliers_ineq=final_state.multipliers_ineq,
        active_set=final_state.active_set,
        converged=final_converged,
        iterations=final_state.iteration,
        ping_ponged=final_state.ping_ponged,
        reached_max_iter=reached_max_iter,
        final_working_tol=jnp.asarray(final_working_tol, dtype=jnp.float64),
    )
