"""Thin :func:`solve_qp` router dispatching to the three QP strategies.

The legacy ``solve_qp`` did three things at once: routed to the proximal
or direct equality strategy, inlined the inequality-only strategy, and
constructed a default ``ProjectedCGCholesky`` when none was provided.
This module reduces it to the routing table and the default-inner-solver
construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, jaxtyped

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.cholesky import ProjectedCGCholesky
from slsqp_jax.inner.krylov import solve_unconstrained_cg
from slsqp_jax.qp.direct import solve_qp_direct
from slsqp_jax.qp.inequality import solve_qp_inequality
from slsqp_jax.qp.proximal import solve_qp_proximal
from slsqp_jax.state import QPSolverResult
from slsqp_jax.types import Scalar, Vector


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
) -> QPSolverResult:
    """Solve a QP with equality and inequality constraints.

    Solves::

        minimize    (1/2) d^T H d + g^T d
        subject to  A_eq d = b_eq
                    A_ineq d >= b_ineq

    where H is provided implicitly via ``hvp_fn(v) = H @ v``.

    The QP active-set loop dispatches to one of three strategies:

    * **Proximal sSQP** (``m_eq > 0`` and ``use_proximal=True``):
      equality constraints absorbed into the objective via
      augmented-Lagrangian penalty.  See :mod:`slsqp_jax.qp.proximal`.
    * **Direct projection** (``m_eq > 0`` and ``use_proximal=False``):
      equality constraints enforced via null-space projection in the
      inner solver.  See :mod:`slsqp_jax.qp.direct`.
    * **Inequality-only** (``m_eq == 0``): no equality block; just an
      active-set loop on inequality constraints.  See
      :mod:`slsqp_jax.qp.inequality`.

    All three share the same EXPAND / ping-pong active-set loop body
    (see :mod:`slsqp_jax.qp.active_set`).

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
        expand_factor: EXPAND tolerance growth rate.
        initial_active_set: Optional warm-start active set from a
            previous QP solve.
        kkt_residual: Norm of the KKT residual from the outer solver.
        proximal_mu: Adaptive proximal parameter for sSQP.
        prev_multipliers_eq: Equality multipliers from the previous
            outer iteration (proximal centre).
        precond_fn: Optional preconditioner v -> M @ v.
        cg_tol: Optional CG convergence tolerance overriding ``tol``
            for the inner solver only.
        cg_regularization: Curvature-guard threshold for CG.
        use_proximal: When True, equality constraints go through the
            sSQP proximal path.  When False, direct projection.
        predicted_active_set: Optional LPEC-A predicted active set
            for warm-start.
        active_set_method: ``"expand"``, ``"lpeca_init"``, or ``"lpeca"``.
        use_constraint_preconditioner: Used only when constructing a
            default ``inner_solver``.
        inner_solver: Pluggable strategy for the inner equality-
            constrained QP solve.  Defaults to ``ProjectedCGCholesky``.
        mult_drop_floor: Floor on the negative-multiplier drop test.
        ping_pong_threshold: Threshold for the explicit ping-pong
            short-circuit.  Defaults to ``2**31 - 1`` (effectively
            disabled).

    Returns:
        ``QPSolverResult`` containing the solution, multipliers, active
        set, and convergence info.
    """
    if active_set_method not in ("expand", "lpeca_init", "lpeca"):
        raise ValueError(
            f"active_set_method must be 'expand', 'lpeca_init', or 'lpeca', "
            f"got {active_set_method!r}"
        )

    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]
    m_total = m_eq + m_ineq

    use_expand = active_set_method != "lpeca"
    effective_predicted = (
        predicted_active_set if active_set_method in ("lpeca_init", "lpeca") else None
    )

    inner_cg_tol: Scalar | float = cg_tol if cg_tol is not None else tol

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
        return QPSolverResult(
            d=d,
            multipliers_eq=jnp.zeros((0,)),
            multipliers_ineq=jnp.zeros((0,)),
            active_set=jnp.zeros((0,), dtype=bool),
            converged=finite_d,
            iterations=jnp.array(1),
            ping_ponged=jnp.array(False),
            reached_max_iter=jnp.array(False),
            final_working_tol=jnp.asarray(0.0, dtype=jnp.float64),
            proj_residual=jnp.asarray(0.0, dtype=jnp.float64),
            n_proj_refinements=jnp.asarray(0),
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=jnp.float64),
        )

    # Case 2: equality + (any) ineq, sSQP enabled.
    if m_eq > 0 and use_proximal:
        return solve_qp_proximal(
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

    # Case 3: equality + (any) ineq, sSQP disabled — direct projection.
    if m_eq > 0 and not use_proximal:
        return solve_qp_direct(
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

    # Case 4: inequality only.
    return solve_qp_inequality(
        hvp_fn=hvp_fn,
        g=g,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        m_ineq=m_ineq,
        max_iter=max_iter,
        max_cg_iter=max_cg_iter,
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


__all__ = ["solve_qp"]
