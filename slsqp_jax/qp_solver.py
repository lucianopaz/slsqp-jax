"""QP Subproblem Solver for SLSQP.

This module implements a QP solver for the Quadratic Programming
subproblem that arises at each SLSQP iteration.

The QP subproblem has the form:
    minimize    (1/2) d^T H d + g^T d
    subject to  A_eq d = b_eq
                A_ineq d >= b_ineq

The solver uses a **projected conjugate gradient** method for the inner
equality-constrained QP solve, wrapped in an **active-set** method for
inequality constraints. The Hessian is accessed only through a
Hessian-vector product (HVP) function, enabling matrix-free operation
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

**Proximal stabilization (sSQP).**  When enabled via ``proximal_sigma``,
equality constraints are absorbed into the QP objective through an
augmented-Lagrangian penalty, following Hager (*Comp. Optim. Appl.*,
1999) and Wright (*Math. Oper. Res.*, 2002).  This regularizes the
dual solution and prevents QP infeasibility at degenerate vertices.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from slsqp_jax.types import Scalar, Vector


class QPState(eqx.Module):
    """State for the Active Set QP solver."""

    d: Vector
    active_set: Bool[Array, " m_ineq"]
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


class QPResult(NamedTuple):
    """Result from the QP solver."""

    d: Vector
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    active_set: Bool[Array, " m_ineq"]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]


class _CGState(NamedTuple):
    """Internal state for the (preconditioned) conjugate gradient solver.

    When a preconditioner M is used, ``rz`` stores r^T z (where z = M r)
    instead of r^T r, and ``p`` is built from z rather than r.
    """

    d: Vector
    r: Vector
    p: Vector
    rz: Scalar  # r^T z (preconditioned) or r^T r (unpreconditioned)
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def _solve_unconstrained_cg(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
) -> Vector:
    """Solve the unconstrained QP: min (1/2) d^T B d + g^T d.

    Uses (preconditioned) conjugate gradient to solve B d = -g without
    forming B.  When *precond_fn* is provided, the standard PCG algorithm
    is used: z = M r, beta = r_new^T z_new / r_old^T z_old, and p is
    built from z (Nocedal & Wright, Algorithm 5.3).

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        g: Linear term (gradient).
        max_cg_iter: Maximum CG iterations.
        cg_tol: Convergence tolerance on residual norm.
        precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
        cg_regularization: Minimum eigenvalue threshold for the curvature
            guard.  CG declares "bad curvature" when the effective
            eigenvalue ``p^T B p / ||p||^2`` falls below this value.
            Based on SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).

    Returns:
        Solution vector d.
    """
    n = g.shape[0]
    r0 = -g
    r0_norm_sq = jnp.dot(r0, r0)

    if precond_fn is not None:
        z0 = precond_fn(r0)
        rz0_raw = jnp.dot(r0, z0)
        # Fall back to identity if preconditioner is not SPD for this residual
        z0 = jnp.where(rz0_raw > 0, z0, r0)
        rz0 = jnp.where(rz0_raw > 0, rz0_raw, r0_norm_sq)
        p0 = z0
    else:
        rz0 = r0_norm_sq
        p0 = r0

    init_cg = _CGState(
        d=jnp.zeros(n),
        r=r0,
        p=p0,
        rz=rz0,
        iteration=jnp.array(0),
        converged=r0_norm_sq < cg_tol**2,
    )

    def cg_step(i, state):
        def do_step(state):
            Bp = hvp_fn(state.p)
            pBp = jnp.dot(state.p, Bp)

            # SNOPT-style curvature guard: declare "bad curvature" when the
            # effective eigenvalue p^T B p / ||p||^2 falls below delta^2.
            # This is scale-invariant (no false stops when ||p|| is small)
            # while still catching numerical noise from the null-space
            # projector.  Alpha and the residual use the true curvature so
            # the CG recurrence is exact and the solution is unbiased.
            # Based on SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).
            pp = jnp.dot(state.p, state.p)
            has_bad_curvature = pBp <= cg_regularization * pp

            alpha = jnp.where(
                has_bad_curvature,
                jnp.array(0.0),
                state.rz / jnp.maximum(pBp, 1e-30),
            )

            d_new = state.d + alpha * state.p
            r_new = state.r - alpha * Bp
            r_new_norm_sq = jnp.dot(r_new, r_new)

            if precond_fn is not None:
                z_new_raw = precond_fn(r_new)
                rz_raw = jnp.dot(r_new, z_new_raw)
                z_new = jnp.where(rz_raw > 0, z_new_raw, r_new)
                rz_new = jnp.where(rz_raw > 0, rz_raw, r_new_norm_sq)
            else:
                z_new = r_new
                rz_new = r_new_norm_sq

            beta = rz_new / jnp.maximum(state.rz, 1e-30)
            p_new = z_new + beta * state.p

            converged = (r_new_norm_sq < cg_tol**2) | has_bad_curvature

            return jax.lax.cond(
                has_bad_curvature,
                lambda: _CGState(
                    d=state.d,
                    r=state.r,
                    p=state.p,
                    rz=state.rz,
                    iteration=state.iteration + 1,
                    converged=jnp.array(True),
                ),
                lambda: _CGState(
                    d=d_new,
                    r=r_new,
                    p=p_new,
                    rz=rz_new,
                    iteration=state.iteration + 1,
                    converged=converged,
                ),
            )

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)
    return final_cg.d


def _solve_projected_cg(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
) -> tuple[Vector, Float[Array, " m"]]:
    """Solve equality-constrained QP using projected (preconditioned) CG.

    Solves:
        minimize    (1/2) d^T B d + g^T d
        subject to  A[active] d = b[active]

    where B is given implicitly via hvp_fn(v) = B @ v.

    The method:
    1. Computes a particular solution d_p satisfying A d_p = b for
       active constraints.
    2. Defines the projection P(v) = v - A^T (A A^T)^{-1} A v onto
       the null space of A (for active constraints only).
    3. Runs CG on the reduced problem in the null space of A.
    4. Recovers Lagrange multipliers from the KKT conditions.

    When *precond_fn* is provided, the projected PCG algorithm is used:
    z = P(M(P(r))) where M is the preconditioner (Nocedal & Wright,
    Chapter 16).  The extra projection ensures z stays in the null space.

    Complexity per CG iteration: O(kn) for HVP + O(mn) for projection,
    where k is L-BFGS memory and m is the number of constraints.

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        g: Linear term (gradient of objective).
        A: Combined constraint matrix (m x n), rows are constraint normals.
        b: Combined RHS vector (m,).
        active_mask: Boolean mask (m,) indicating which constraints are active.
        max_cg_iter: Maximum CG iterations.
        cg_tol: CG convergence tolerance.
        precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
        cg_regularization: Minimum eigenvalue threshold for the curvature
            guard.  CG declares "bad curvature" when the projected
            eigenvalue falls below this value.  Based on SNOPT Section 4.5.

    Returns:
        Tuple of (d, multipliers) where d is the solution and multipliers
        is a vector of Lagrange multipliers for all m constraints (0 for
        inactive constraints).
    """
    m = A.shape[0]

    A_masked = jnp.where(active_mask[:, None], A, 0.0)
    b_masked = jnp.where(active_mask, b, 0.0)

    reg_diag = jnp.where(active_mask, 0.0, 1.0)
    AAt = A_masked @ A_masked.T + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)

    def solve_AAt(rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        return jnp.linalg.solve(AAt, rhs)

    d_p = A_masked.T @ solve_AAt(b_masked)

    def project(v: Vector) -> Vector:
        return v - A_masked.T @ solve_AAt(A_masked @ v)

    Bd_p = hvp_fn(d_p)
    r0 = project(-(g + Bd_p))
    r0_norm_sq = jnp.dot(r0, r0)

    if precond_fn is not None:
        z0 = project(precond_fn(r0))
        rz0_raw = jnp.dot(r0, z0)
        z0 = jnp.where(rz0_raw > 0, z0, r0)
        rz0 = jnp.where(rz0_raw > 0, rz0_raw, r0_norm_sq)
        p0 = z0
    else:
        rz0 = r0_norm_sq
        p0 = r0

    init_cg = _CGState(
        d=d_p,
        r=r0,
        p=p0,
        rz=rz0,
        iteration=jnp.array(0),
        converged=r0_norm_sq < cg_tol**2,
    )

    def cg_step(i, state):
        def do_step(state):
            Bp = hvp_fn(state.p)
            PBp = project(Bp)
            pPBp = jnp.dot(state.p, PBp)

            # SNOPT-style curvature guard (see unconstrained CG for detail).
            # For projected CG, p is in the null space, so this checks the
            # effective eigenvalue of the reduced Hessian Z^T B Z along p.
            pp = jnp.dot(state.p, state.p)
            has_bad_curvature = pPBp <= cg_regularization * pp

            alpha = jnp.where(
                has_bad_curvature,
                jnp.array(0.0),
                state.rz / jnp.maximum(pPBp, 1e-30),
            )

            d_new = state.d + alpha * state.p
            r_new = state.r - alpha * PBp
            r_new_norm_sq = jnp.dot(r_new, r_new)

            if precond_fn is not None:
                z_new_raw = project(precond_fn(r_new))
                rz_raw = jnp.dot(r_new, z_new_raw)
                z_new = jnp.where(rz_raw > 0, z_new_raw, r_new)
                rz_new = jnp.where(rz_raw > 0, rz_raw, r_new_norm_sq)
            else:
                z_new = r_new
                rz_new = r_new_norm_sq

            beta = rz_new / jnp.maximum(state.rz, 1e-30)
            p_new = z_new + beta * state.p

            converged = (r_new_norm_sq < cg_tol**2) | has_bad_curvature

            return jax.lax.cond(
                has_bad_curvature,
                lambda: _CGState(
                    d=state.d,
                    r=state.r,
                    p=state.p,
                    rz=state.rz,
                    iteration=state.iteration + 1,
                    converged=jnp.array(True),
                ),
                lambda: _CGState(
                    d=d_new,
                    r=r_new,
                    p=p_new,
                    rz=rz_new,
                    iteration=state.iteration + 1,
                    converged=converged,
                ),
            )

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)

    Bd = hvp_fn(final_cg.d)
    kkt_residual = A_masked @ (Bd + g)
    multipliers = solve_AAt(kkt_residual)
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    return final_cg.d, multipliers


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
    proximal_sigma: float,
    prev_multipliers_eq: Float[Array, " m_eq"] | None,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
) -> QPResult:
    """Solve the QP using the stabilized SQP (sSQP) formulation.

    Equality constraints are absorbed into the objective via an
    augmented-Lagrangian penalty with weight ``1/proximal_sigma``.
    The active-set loop operates on inequality constraints only.

    The stabilized objective is::

        (1/2) d^T B_tilde d + g_tilde^T d

    where ``B_tilde(v) = H v + (1/sigma) A_eq^T (A_eq v)`` and
    ``g_tilde = g - (1/sigma) A_eq^T b_eq - A_eq^T lambda_k``.

    Equality multipliers are recovered from the penalty optimality
    condition: ``lambda = lambda_k - (1/sigma)(A_eq d - b_eq)``.
    """
    inv_sigma = 1.0 / proximal_sigma
    prev_mult_eq = (
        prev_multipliers_eq if prev_multipliers_eq is not None else jnp.zeros((m_eq,))
    )
    inner_cg_tol: Scalar | float = cg_tol if cg_tol is not None else tol

    def stabilized_hvp(v: Vector) -> Vector:
        return hvp_fn(v) + inv_sigma * (A_eq.T @ (A_eq @ v))

    g_mod = g - inv_sigma * (A_eq.T @ b_eq) - A_eq.T @ prev_mult_eq

    def _recover_mult_eq(d: Vector) -> Float[Array, " m_eq"]:
        return prev_mult_eq - inv_sigma * (A_eq @ d - b_eq)

    # Sub-case: no inequality constraints — just unconstrained CG
    if m_ineq == 0:
        d = _solve_unconstrained_cg(
            stabilized_hvp,
            g_mod,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )
        return QPResult(
            d=d,
            multipliers_eq=_recover_mult_eq(d),
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=jnp.array(True),
            iterations=jnp.array(1),
        )

    # Sub-case: inequalities present — active-set loop on A_ineq only
    kkt_res = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_res, 1.0) * tol

    # Initial unconstrained solve (equalities absorbed into objective)
    d_init = _solve_unconstrained_cg(
        stabilized_hvp,
        g_mod,
        max_cg_iter,
        inner_cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
    )

    # Determine starting active set
    residuals_init = A_ineq @ d_init - b_ineq
    if initial_active_set is not None:
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
    )

    def cond_fn(state: QPState) -> Bool[Array, ""]:
        return ~state.converged & (state.iteration < max_iter)

    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)

    def body_fn(state: QPState) -> QPState:
        working_tol = base_tol + state.iteration * tau

        # Solve with current active set — inequalities only
        d_new, mult_ineq_new = _solve_projected_cg(
            stabilized_hvp,
            g_mod,
            A_ineq,
            b_ineq,
            state.active_set,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )

        mult_eq_new = _recover_mult_eq(d_new)

        # Check feasibility with expanding tolerance
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -working_tol) & ~state.active_set
        any_violated = jnp.any(violated)

        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        negative_mult = (mult_ineq_new < -working_tol) & state.active_set
        any_negative = jnp.any(negative_mult)

        mult_scores = jnp.where(state.active_set, mult_ineq_new, jnp.inf)
        most_negative_idx = jnp.argmin(mult_scores)

        def add_constraint():
            new_active = state.active_set.at[most_violated_idx].set(True)
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(False),
            )

        def drop_constraint():
            new_active = state.active_set.at[most_negative_idx].set(False)
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(False),
            )

        def mark_converged():
            return QPState(
                d=d_new,
                active_set=state.active_set,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(True),
            )

        return jax.lax.cond(
            any_violated,
            add_constraint,
            lambda: jax.lax.cond(any_negative, drop_constraint, mark_converged),
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return QPResult(
        d=final_state.d,
        multipliers_eq=final_state.multipliers_eq,
        multipliers_ineq=final_state.multipliers_ineq,
        active_set=final_state.active_set,
        converged=final_state.converged,
        iterations=final_state.iteration,
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
    proximal_sigma: float = 0.0,
    prev_multipliers_eq: Float[Array, " m_eq"] | None = None,
    precond_fn: Callable | None = None,
    cg_tol: Scalar | float | None = None,
    cg_regularization: float = 1e-6,
) -> QPResult:
    """Solve a QP with equality and inequality constraints.

    Solves::

        minimize    (1/2) d^T H d + g^T d
        subject to  A_eq d = b_eq
                    A_ineq d >= b_ineq

    where H is provided implicitly via ``hvp_fn(v) = H @ v``.

    Uses a primal active-set method: at each iteration, active inequality
    constraints are treated as equalities, and the resulting
    equality-constrained QP is solved using projected conjugate gradient.
    Constraints are added/removed from the active set based on
    feasibility violations and multiplier signs until optimality is reached.

    To prevent cycling due to degenerate constraints, the EXPAND
    procedure is used: the feasibility tolerance increases by a small
    increment ``tau = tol * expand_factor / max_iter`` at every
    active-set iteration.  Set *expand_factor* to 0 to disable.

    When ``proximal_sigma > 0`` and there are equality constraints, the
    solver uses the **stabilized SQP (sSQP)** formulation (Hager, 1999;
    Wright, 2002).  Equality constraints are absorbed into the objective
    via an augmented-Lagrangian penalty::

        minimize  (1/2) d^T B_tilde d + g_tilde^T d
        subject to  A_ineq d >= b_ineq

    where ``B_tilde(v) = H v + (1/sigma) A_eq^T (A_eq v)`` and
    ``g_tilde = g - (1/sigma) A_eq^T b_eq - A_eq^T lambda_k``.
    Equality multipliers are recovered as
    ``lambda = lambda_k - (1/sigma)(A_eq d - b_eq)``.

    Args:
        hvp_fn: Hessian-vector product function v -> H @ v.
        g: Linear term of the objective (gradient).
        A_eq: Equality constraint matrix (m_eq x n).
        b_eq: Equality constraint RHS (m_eq,).
        A_ineq: Inequality constraint matrix (m_ineq x n).
        b_ineq: Inequality constraint RHS (m_ineq,).
        max_iter: Maximum active-set iterations.
        max_cg_iter: Maximum CG iterations per active-set step.
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
        proximal_sigma: Stabilization parameter for the sSQP formulation.
            When positive, equality constraints are absorbed into the
            objective with penalty weight ``1/sigma``.  Larger values
            mean more relaxation.  Recommended range: ``[1e-4, 1e-1]``.
            Default 0.0 disables stabilization (standard QP).
        prev_multipliers_eq: Equality multipliers from the previous outer
            iteration, used as the proximal center when ``proximal_sigma > 0``.
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

    Returns:
        QPResult containing the solution, multipliers, active set, and
        convergence info.
    """
    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]
    m_total = m_eq + m_ineq

    # Resolve CG tolerance: use cg_tol if provided, else fall back to tol.
    inner_cg_tol: Scalar | float = cg_tol if cg_tol is not None else tol

    # Case 1: No constraints at all
    if m_total == 0:
        d = _solve_unconstrained_cg(
            hvp_fn,
            g,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )
        return QPResult(
            d=d,
            multipliers_eq=jnp.zeros((0,)),
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=jnp.array(True),
            iterations=jnp.array(1),
        )

    # ---- Proximal stabilized path (sSQP) ----
    if proximal_sigma > 0 and m_eq > 0:
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
            proximal_sigma=proximal_sigma,
            prev_multipliers_eq=prev_multipliers_eq,
            precond_fn=precond_fn,
            cg_tol=inner_cg_tol,
            cg_regularization=cg_regularization,
        )

    # ---- Standard path ----

    # Case 2: Only equality constraints (no active-set loop needed)
    if m_ineq == 0:
        active_mask = jnp.ones(m_eq, dtype=bool)
        d, mult_eq = _solve_projected_cg(
            hvp_fn,
            g,
            A_eq,
            b_eq,
            active_mask,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )
        return QPResult(
            d=d,
            multipliers_eq=mult_eq,
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=jnp.array(True),
            iterations=jnp.array(1),
        )

    # Case 3: Has inequality constraints -> active-set method
    A_combined = jnp.concatenate([A_eq, A_ineq], axis=0)
    b_combined = jnp.concatenate([b_eq, b_ineq])

    kkt_residual = jnp.asarray(kkt_residual, dtype=jnp.float64)
    base_tol = tol + jnp.minimum(kkt_residual, 1.0) * tol

    eq_only_mask = jnp.concatenate(
        [jnp.ones(m_eq, dtype=bool), jnp.zeros(m_ineq, dtype=bool)]
    )
    d_init, mult_init = _solve_projected_cg(
        hvp_fn,
        g,
        A_combined,
        b_combined,
        eq_only_mask,
        max_cg_iter,
        inner_cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
    )

    # Determine starting active set: warm-start or cold-start
    residuals_init = A_ineq @ d_init - b_ineq
    if initial_active_set is not None:
        init_active = initial_active_set | (residuals_init < -base_tol)
    else:
        init_active = residuals_init < -base_tol
    init_converged = ~jnp.any(init_active)

    mult_eq_init = mult_init[:m_eq] if m_eq > 0 else jnp.zeros((0,))

    init_state = QPState(
        d=d_init,
        active_set=init_active,
        multipliers_eq=mult_eq_init,
        multipliers_ineq=jnp.zeros((m_ineq,)),
        iteration=jnp.array(0),
        converged=init_converged,
    )

    def cond_fn(state: QPState) -> Bool[Array, ""]:
        return ~state.converged & (state.iteration < max_iter)

    # EXPAND anti-cycling: per-iteration tolerance increment
    tau = base_tol * expand_factor / jnp.maximum(max_iter, 1)

    def body_fn(state: QPState) -> QPState:
        # EXPAND: working tolerance grows monotonically each iteration
        working_tol = base_tol + state.iteration * tau

        # Build active mask for the combined constraint matrix
        combined_mask = jnp.concatenate([jnp.ones(m_eq, dtype=bool), state.active_set])

        # Solve with current active set using projected CG
        d_new, mult_all = _solve_projected_cg(
            hvp_fn,
            g,
            A_combined,
            b_combined,
            combined_mask,
            max_cg_iter,
            inner_cg_tol,
            precond_fn=precond_fn,
            cg_regularization=cg_regularization,
        )

        mult_eq_new = mult_all[:m_eq] if m_eq > 0 else jnp.zeros((0,))
        mult_ineq_new = mult_all[m_eq:]

        # Check feasibility with expanding tolerance (stricter activation)
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -working_tol) & ~state.active_set
        any_violated = jnp.any(violated)

        # Find the most violated inactive constraint
        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        # Check multiplier signs with expanding tolerance (stricter deactivation)
        negative_mult = (mult_ineq_new < -working_tol) & state.active_set
        any_negative = jnp.any(negative_mult)

        mult_scores = jnp.where(state.active_set, mult_ineq_new, jnp.inf)
        most_negative_idx = jnp.argmin(mult_scores)

        def add_constraint():
            new_active = state.active_set.at[most_violated_idx].set(True)
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(False),
            )

        def drop_constraint():
            new_active = state.active_set.at[most_negative_idx].set(False)
            return QPState(
                d=d_new,
                active_set=new_active,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(False),
            )

        def mark_converged():
            return QPState(
                d=d_new,
                active_set=state.active_set,
                multipliers_eq=mult_eq_new,
                multipliers_ineq=mult_ineq_new,
                iteration=state.iteration + 1,
                converged=jnp.array(True),
            )

        return jax.lax.cond(
            any_violated,
            add_constraint,
            lambda: jax.lax.cond(any_negative, drop_constraint, mark_converged),
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return QPResult(
        d=final_state.d,
        multipliers_eq=final_state.multipliers_eq,
        multipliers_ineq=final_state.multipliers_ineq,
        active_set=final_state.active_set,
        converged=final_state.converged,
        iterations=final_state.iteration,
    )
