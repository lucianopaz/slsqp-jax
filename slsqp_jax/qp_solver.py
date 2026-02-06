"""QP Subproblem Solver for SLSQP.

This module implements a QP solver for the Quadratic Programming
subproblem that arises at each SLSQP iteration.

The QP subproblem has the form:
    minimize    (1/2) d^T H d + g^T d
    subject to  A_eq d = b_eq
                A_ineq d >= b_ineq

For inequality constraints A d >= b, the Lagrangian is:
    L(d, λ) = (1/2) d^T H d + g^T d - λ^T (A d - b)
    
with λ >= 0 for active constraints.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from beartype import beartype


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


@jaxtyped(typechecker=beartype)
def solve_equality_qp(
    H: Float[Array, "n n"],
    g: Float[Array, " n"],
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
) -> tuple[Float[Array, " n"], Float[Array, " m"]]:
    """Solve equality-constrained QP using the KKT system.
    
    Solves:
        minimize    (1/2) d^T H d + g^T d
        subject to  A d = b
    
    KKT conditions:
        H d - A^T λ = -g   (stationarity, with sign for equality constraints)
        A d = b            (primal feasibility)
    
    Note: For equality constraints A d = b, the multiplier λ can be any sign.
    """
    n = H.shape[0]
    m = A.shape[0]
    eps = 1e-12
    H_reg = H + eps * jnp.eye(n)
    
    if m == 0:
        d = jnp.linalg.solve(H_reg, -g)
        return d, jnp.zeros((0,))
    
    # KKT system:
    # [H   -A^T] [d]   [-g]
    # [A    0  ] [λ] = [b ]
    # 
    # Solving: H d - A^T λ = -g, A d = b
    
    KKT = jnp.zeros((n + m, n + m))
    KKT = KKT.at[:n, :n].set(H_reg)
    KKT = KKT.at[:n, n:].set(-A.T)  # Note: -A^T
    KKT = KKT.at[n:, :n].set(A)
    KKT = KKT.at[n:, n:].set(eps * jnp.eye(m))  # Small regularization
    
    rhs = jnp.concatenate([-g, b])
    solution = jnp.linalg.solve(KKT, rhs)
    
    return solution[:n], solution[n:]


def _solve_kkt_with_active_set(
    H: Float[Array, "n n"],
    g: Float[Array, " n"],
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    active_mask: Bool[Array, " m_ineq"],
) -> tuple[Float[Array, " n"], Float[Array, " m_eq"], Float[Array, " m_ineq"]]:
    """Solve the KKT system treating active inequalities as equalities.
    
    For inequality constraints A d >= b:
        L(d, λ) = (1/2) d^T H d + g^T d - λ^T (A d - b)
    
    KKT conditions for active constraints:
        H d - A_eq^T λ_eq - A_ineq^T λ_ineq = -g
        A_eq d = b_eq
        A_ineq[active] d = b_ineq[active]
        λ_ineq[inactive] = 0
        λ_ineq[active] >= 0  (checked separately)
    """
    n = H.shape[0]
    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]
    eps = 1e-12
    
    m_total = m_eq + m_ineq
    H_reg = H + eps * jnp.eye(n)
    
    if m_total == 0:
        d = jnp.linalg.solve(H_reg, -g)
        return d, jnp.zeros((0,)), jnp.zeros((0,))
    
    # Build KKT matrix
    kkt_size = n + m_total
    KKT = jnp.zeros((kkt_size, kkt_size))
    
    # Top-left: H
    KKT = KKT.at[:n, :n].set(H_reg)
    
    # Top-right: -[A_eq^T, A_ineq^T]  (note negative sign!)
    if m_eq > 0:
        KKT = KKT.at[:n, n:n+m_eq].set(-A_eq.T)
    if m_ineq > 0:
        KKT = KKT.at[:n, n+m_eq:].set(-A_ineq.T)
    
    # Equality constraint rows
    if m_eq > 0:
        KKT = KKT.at[n:n+m_eq, :n].set(A_eq)
        KKT = KKT.at[n:n+m_eq, n:n+m_eq].set(eps * jnp.eye(m_eq))
    
    # Inequality constraint rows: 
    # Active: enforce A_ineq[i] d = b_ineq[i]
    # Inactive: set λ_ineq[i] = 0
    if m_ineq > 0:
        A_ineq_active = jnp.where(active_mask[:, None], A_ineq, 0.0)
        KKT = KKT.at[n+m_eq:, :n].set(A_ineq_active)
        
        # Diagonal: small eps for active, 1.0 for inactive (to enforce λ=0)
        diag_ineq = jnp.where(active_mask, eps, 1.0)
        KKT = KKT.at[n+m_eq:, n+m_eq:].set(jnp.diag(diag_ineq))
    
    # Build RHS
    if m_eq > 0 and m_ineq > 0:
        b_ineq_rhs = jnp.where(active_mask, b_ineq, 0.0)
        rhs = jnp.concatenate([-g, b_eq, b_ineq_rhs])
    elif m_eq > 0:
        rhs = jnp.concatenate([-g, b_eq])
    elif m_ineq > 0:
        b_ineq_rhs = jnp.where(active_mask, b_ineq, 0.0)
        rhs = jnp.concatenate([-g, b_ineq_rhs])
    else:
        rhs = -g
    
    # Solve KKT system
    solution = jnp.linalg.solve(KKT, rhs)
    
    d = solution[:n]
    mult_eq = solution[n:n+m_eq] if m_eq > 0 else jnp.zeros((0,))
    mult_ineq = solution[n+m_eq:] if m_ineq > 0 else jnp.zeros((0,))
    
    return d, mult_eq, mult_ineq


@jaxtyped(typechecker=beartype)
def solve_qp(
    H: Float[Array, "n n"],
    g: Float[Array, " n"],
    A_eq: Float[Array, "m_eq n"],
    b_eq: Float[Array, " m_eq"],
    A_ineq: Float[Array, "m_ineq n"],
    b_ineq: Float[Array, " m_ineq"],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> QPResult:
    """Solve a QP with equality and inequality constraints.
    
    Solves:
        minimize    (1/2) d^T H d + g^T d
        subject to  A_eq d = b_eq
                    A_ineq d >= b_ineq
    
    Uses a primal active-set method.
    
    For a constraint A d >= b to be optimal when active, its multiplier
    should be non-negative (λ >= 0).
    """
    n = H.shape[0]
    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]
    
    # Handle case with no inequality constraints
    if m_ineq == 0:
        d, mult_eq = solve_equality_qp(H, g, A_eq, b_eq)
        return QPResult(
            d=d,
            multipliers_eq=mult_eq,
            multipliers_ineq=jnp.zeros((0,)),
            converged=jnp.array(True),
            iterations=jnp.array(1),
        )
    
    # Solve equality-only problem first
    d_init, mult_eq_init = solve_equality_qp(H, g, A_eq, b_eq)
    
    # Check which inequality constraints are violated
    residuals_init = A_ineq @ d_init - b_ineq
    init_active = residuals_init < -tol
    init_converged = ~jnp.any(init_active)
    
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
        # Solve with current active set
        d_new, mult_eq_new, mult_ineq_new = _solve_kkt_with_active_set(
            H, g, A_eq, b_eq, A_ineq, b_ineq, state.active_set
        )
        
        # Check feasibility of solution for inactive constraints
        residuals = A_ineq @ d_new - b_ineq
        violated = (residuals < -tol) & ~state.active_set
        any_violated = jnp.any(violated)
        
        # Find most violated inactive constraint
        violation_scores = jnp.where(violated, -residuals, -jnp.inf)
        most_violated_idx = jnp.argmax(violation_scores)
        
        # Check multipliers of active constraints
        # For A d >= b constraints, optimal multipliers should be >= 0
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
