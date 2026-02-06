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
"""

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped


class QPState(eqx.Module):
    """State for the Active Set QP solver."""

    d: Float[Array, " n"]
    active_set: Bool[Array, " m_ineq"]
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


class QPResult(NamedTuple):
    """Result from the QP solver."""

    d: Float[Array, " n"]
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]


class _CGState(NamedTuple):
    """Internal state for the projected conjugate gradient solver."""

    d: Float[Array, " n"]
    r: Float[Array, " n"]
    p: Float[Array, " n"]
    r_norm_sq: Float[Array, ""]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def _solve_unconstrained_cg(
    hvp_fn: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    g: Float[Array, " n"],
    max_cg_iter: int,
    cg_tol: float,
) -> Float[Array, " n"]:
    """Solve the unconstrained QP: min (1/2) d^T B d + g^T d.

    Uses conjugate gradient to solve B d = -g without forming B.

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        g: Linear term (gradient).
        max_cg_iter: Maximum CG iterations.
        cg_tol: Convergence tolerance on residual norm.

    Returns:
        Solution vector d.
    """
    n = g.shape[0]
    r0 = -g
    r0_norm_sq = jnp.dot(r0, r0)

    init_cg = _CGState(
        d=jnp.zeros(n),
        r=r0,
        p=r0,
        r_norm_sq=r0_norm_sq,
        iteration=jnp.array(0),
        converged=r0_norm_sq < cg_tol**2,
    )

    def cg_step(i, state):
        def do_step(state):
            Bp = hvp_fn(state.p)
            pBp = jnp.dot(state.p, Bp)
            safe_pBp = jnp.maximum(pBp, 1e-12)
            alpha = state.r_norm_sq / safe_pBp
            # Clip alpha to prevent numerical instability
            alpha = jnp.clip(alpha, 0.0, 1e10)

            d_new = state.d + alpha * state.p
            r_new = state.r - alpha * Bp
            r_new_norm_sq = jnp.dot(r_new, r_new)

            beta = r_new_norm_sq / jnp.maximum(state.r_norm_sq, 1e-30)
            beta = jnp.clip(beta, 0.0, 1e10)
            p_new = r_new + beta * state.p

            # Check for NaN and stop if detected
            has_nan = jnp.any(jnp.isnan(d_new)) | jnp.any(jnp.isnan(r_new))
            converged = (r_new_norm_sq < cg_tol**2) | (pBp <= 1e-12) | has_nan

            # If NaN detected, keep the previous valid state
            d_new = jnp.where(has_nan, state.d, d_new)
            r_new = jnp.where(has_nan, state.r, r_new)
            p_new = jnp.where(has_nan, state.p, p_new)
            r_new_norm_sq = jnp.where(has_nan, state.r_norm_sq, r_new_norm_sq)

            return _CGState(
                d=d_new,
                r=r_new,
                p=p_new,
                r_norm_sq=r_new_norm_sq,
                iteration=state.iteration + 1,
                converged=converged,
            )

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)
    return final_cg.d


def _solve_projected_cg(
    hvp_fn: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    g: Float[Array, " n"],
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_cg_iter: int,
    cg_tol: float,
) -> tuple[Float[Array, " n"], Float[Array, " m"]]:
    """Solve equality-constrained QP using projected conjugate gradient.

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

    The projection involves a small m x m system (A A^T) which is
    solved directly. Inactive constraint rows are zeroed and regularized
    so that the system remains non-singular.

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

    Returns:
        Tuple of (d, multipliers) where d is the solution and multipliers
        is a vector of Lagrange multipliers for all m constraints (0 for
        inactive constraints).
    """
    m = A.shape[0]

    # Mask inactive constraints: zero out their rows
    A_masked = jnp.where(active_mask[:, None], A, 0.0)
    b_masked = jnp.where(active_mask, b, 0.0)

    # Build regularized A A^T (m x m, small)
    # Active rows contribute normally; inactive rows get diagonal 1.0
    # to keep the system non-singular (their multiplier will be 0).
    reg_diag = jnp.where(active_mask, 0.0, 1.0)
    AAt = A_masked @ A_masked.T + jnp.diag(reg_diag) + 1e-10 * jnp.eye(m)

    def solve_AAt(rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        """Solve AAt @ x = rhs using lstsq for numerical stability."""
        result, _, _, _ = jnp.linalg.lstsq(AAt, rhs, rcond=1e-10)
        # Guard against NaN
        result = jnp.where(jnp.any(jnp.isnan(result)), jnp.zeros_like(result), result)
        return result

    # Particular solution satisfying A d_p = b (for active constraints)
    d_p = A_masked.T @ solve_AAt(b_masked)

    # Projection onto null space of active constraints:
    # P(v) = v - A^T (A A^T)^{-1} A v
    def project(v: Float[Array, " n"]) -> Float[Array, " n"]:
        return v - A_masked.T @ solve_AAt(A_masked @ v)

    # Initial residual for CG in the null space
    Bd_p = hvp_fn(d_p)
    r0 = project(-(g + Bd_p))
    r0_norm_sq = jnp.dot(r0, r0)

    init_cg = _CGState(
        d=d_p,
        r=r0,
        p=r0,
        r_norm_sq=r0_norm_sq,
        iteration=jnp.array(0),
        converged=r0_norm_sq < cg_tol**2,
    )

    def cg_step(i, state):
        def do_step(state):
            Bp = hvp_fn(state.p)
            PBp = project(Bp)
            pPBp = jnp.dot(state.p, PBp)

            # Guard against negative curvature (stop CG if detected)
            safe_pPBp = jnp.maximum(pPBp, 1e-12)
            alpha = state.r_norm_sq / safe_pPBp
            # Clip alpha to prevent numerical instability
            alpha = jnp.clip(alpha, 0.0, 1e10)

            d_new = state.d + alpha * state.p
            r_new = state.r - alpha * PBp
            r_new_norm_sq = jnp.dot(r_new, r_new)

            beta = r_new_norm_sq / jnp.maximum(state.r_norm_sq, 1e-30)
            beta = jnp.clip(beta, 0.0, 1e10)
            p_new = r_new + beta * state.p

            # Check for NaN and stop if detected
            has_nan = jnp.any(jnp.isnan(d_new)) | jnp.any(jnp.isnan(r_new))
            converged = (r_new_norm_sq < cg_tol**2) | (pPBp <= 1e-12) | has_nan

            # If NaN detected, keep the previous valid state
            d_new = jnp.where(has_nan, state.d, d_new)
            r_new = jnp.where(has_nan, state.r, r_new)
            p_new = jnp.where(has_nan, state.p, p_new)
            r_new_norm_sq = jnp.where(has_nan, state.r_norm_sq, r_new_norm_sq)

            return _CGState(
                d=d_new,
                r=r_new,
                p=p_new,
                r_norm_sq=r_new_norm_sq,
                iteration=state.iteration + 1,
                converged=converged,
            )

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)

    # Guard against NaN in the solution
    d_final = jnp.where(
        jnp.any(jnp.isnan(final_cg.d)),
        d_p,  # Fall back to particular solution if CG produced NaN
        final_cg.d,
    )

    # Recover multipliers from KKT stationarity:
    # B d + g - A^T lambda = 0  =>  lambda = (A A^T)^{-1} A (B d + g)
    Bd = hvp_fn(d_final)
    kkt_residual = A_masked @ (Bd + g)
    multipliers = solve_AAt(kkt_residual)
    # Zero out multipliers for inactive constraints
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    return d_final, multipliers


@jaxtyped(typechecker=beartype)
def solve_qp(
    hvp_fn: Callable,
    g: Float[Array, " n"],
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    max_iter: int = 100,
    max_cg_iter: int = 50,
    tol: float = 1e-8,
) -> QPResult:
    """Solve a QP with equality and inequality constraints.

    Solves:
        minimize    (1/2) d^T H d + g^T d
        subject to  A_eq d = b_eq
                    A_ineq d >= b_ineq

    where H is provided implicitly via hvp_fn(v) = H @ v.

    Uses a primal active-set method: at each iteration, active inequality
    constraints are treated as equalities, and the resulting
    equality-constrained QP is solved using projected conjugate gradient.
    Constraints are added/removed from the active set based on
    feasibility violations and multiplier signs until optimality is reached.

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

    Returns:
        QPResult containing the solution, multipliers, and convergence info.
    """
    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]
    m_total = m_eq + m_ineq

    # Case 1: No constraints at all
    if m_total == 0:
        d = _solve_unconstrained_cg(hvp_fn, g, max_cg_iter, tol)
        return QPResult(
            d=d,
            multipliers_eq=jnp.zeros((0,)),
            multipliers_ineq=jnp.zeros((0,)),
            converged=jnp.array(True),
            iterations=jnp.array(1),
        )

    # Case 2: Only equality constraints (no active-set loop needed)
    if m_ineq == 0:
        active_mask = jnp.ones(m_eq, dtype=bool)
        d, mult_eq = _solve_projected_cg(
            hvp_fn, g, A_eq, b_eq, active_mask, max_cg_iter, tol
        )
        return QPResult(
            d=d,
            multipliers_eq=mult_eq,
            multipliers_ineq=jnp.zeros((0,)),
            converged=jnp.array(True),
            iterations=jnp.array(1),
        )

    # Case 3: Has inequality constraints -> active-set method
    # Build combined constraint matrix: [A_eq; A_ineq]
    A_combined = jnp.concatenate([A_eq, A_ineq], axis=0)
    b_combined = jnp.concatenate([b_eq, b_ineq])

    # Step 1: Initial solve with equality constraints only
    eq_only_mask = jnp.concatenate(
        [jnp.ones(m_eq, dtype=bool), jnp.zeros(m_ineq, dtype=bool)]
    )
    d_init, mult_init = _solve_projected_cg(
        hvp_fn, g, A_combined, b_combined, eq_only_mask, max_cg_iter, tol
    )

    # Check which inequality constraints are violated by the initial solution
    residuals_init = A_ineq @ d_init - b_ineq
    init_active = residuals_init < -tol
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

    def body_fn(state: QPState) -> QPState:
        # Build active mask for the combined constraint matrix
        combined_mask = jnp.concatenate([jnp.ones(m_eq, dtype=bool), state.active_set])

        # Solve with current active set using projected CG
        d_new, mult_all = _solve_projected_cg(
            hvp_fn, g, A_combined, b_combined, combined_mask, max_cg_iter, tol
        )

        mult_eq_new = mult_all[:m_eq] if m_eq > 0 else jnp.zeros((0,))
        mult_ineq_new = mult_all[m_eq:]

        # Check feasibility of solution for inactive inequality constraints
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -tol) & ~state.active_set
        any_violated = jnp.any(violated)

        # Find the most violated inactive constraint
        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)

        # Check multiplier signs of active inequality constraints
        # For A d >= b, optimal multipliers should be non-negative
        negative_mult = (mult_ineq_new < -tol) & state.active_set
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
        converged=final_state.converged,
        iterations=final_state.iteration,
    )
