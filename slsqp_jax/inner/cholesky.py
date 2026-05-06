"""Projected CG with Cholesky-based null-space projection."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.masking import make_active_subproblem
from slsqp_jax.inner.projected_cg import run_projected_pcg
from slsqp_jax.state import InnerSolveResult, ProjectionContext
from slsqp_jax.types import Scalar, Vector


def _make_cholesky_projection_ctx(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
) -> ProjectionContext:
    """Build a ``ProjectionContext`` backed by a regularised Cholesky
    factorisation of ``A_work A_workᵀ``.

    Mirrors the projector-construction prefix of
    ``ProjectedCGCholesky.solve``: masks ``A`` and ``b`` to the active
    rows, applies bound-fixing, factorises ``AAᵀ + 1e-8·I``, and packages
    the resulting projector, particular solution and multiplier-recovery
    closure (with one round of iterative refinement) into a
    ``ProjectionContext`` for reuse.
    """
    sub = make_active_subproblem(
        hvp_fn=hvp_fn,
        g=g,
        A=A,
        b=b,
        active_mask=active_mask,
        free_mask=free_mask,
        d_fixed=d_fixed,
    )
    m = A.shape[0]

    reg_diag = jnp.where(active_mask, 0.0, 1.0)
    AAt = sub.A_work @ sub.A_work.T + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
    AAt_chol = jnp.linalg.cholesky(AAt)

    def solve_AAt(rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        return jax.scipy.linalg.cho_solve((AAt_chol, True), rhs)

    d_p_free = sub.A_work.T @ solve_AAt(sub.b_work)
    d_p = d_p_free + sub.d_fixed if sub.has_fixed else d_p_free

    def project(v: Vector) -> Vector:
        v_work = sub.free_mask * v if sub.has_fixed else v
        return v_work - sub.A_work.T @ solve_AAt(sub.A_work @ v_work)

    def recover_multipliers(Bd_plus_g: Vector) -> Float[Array, " m"]:
        # KKT recovery with one step of iterative refinement to absorb
        # the O(eps · cond(AAt)) error introduced by the 1e-8 ridge.
        kkt_rhs = sub.A_work @ Bd_plus_g
        mult = solve_AAt(kkt_rhs)
        mult = jnp.where(active_mask, mult, 0.0)
        grad_L_qp = Bd_plus_g - sub.A_work.T @ mult
        delta = solve_AAt(sub.A_work @ grad_L_qp)
        mult = mult + delta
        mult = jnp.where(active_mask, mult, 0.0)
        return mult

    return ProjectionContext(
        project=project,
        d_p=d_p,
        recover_multipliers=recover_multipliers,
        hvp_work=sub.hvp_work,
        g_eff=sub.g_eff,
        A_work=sub.A_work,
        free_mask=sub.free_mask,
        d_fixed=sub.d_fixed,
        has_fixed=sub.has_fixed,
        converged=jnp.asarray(True),
    )


def _solve_projected_cg_cholesky(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
    use_constraint_preconditioner: bool = False,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Solve equality-constrained QP using projected (preconditioned) CG.

    Implementation backing :class:`ProjectedCGCholesky`.  See the class
    docstring for the algorithmic description.

    When ``use_constraint_preconditioner`` is ``True`` and a
    preconditioner is provided, the constraint preconditioner
    (Gould, Hribar & Nocedal, 2001) is wrapped in front of the shared
    PCG driver: ``z = M r - M A^T (A M A^T)^{-1} A M r``.
    """
    m = A.shape[0]

    ctx = _make_cholesky_projection_ctx(
        hvp_fn=hvp_fn,
        g=g,
        A=A,
        b=b,
        active_mask=active_mask,
        free_mask=free_mask,
        d_fixed=d_fixed,
    )

    if precond_fn is not None and use_constraint_preconditioner:
        _raw_precond = precond_fn
        reg_diag = jnp.where(active_mask, 0.0, 1.0)
        M_AT = jax.vmap(_raw_precond)(ctx.A_work).T  # (n, m)
        A_M_AT = ctx.A_work @ M_AT + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_M_AT_chol = jnp.linalg.cholesky(A_M_AT)

        def _solve_AMAT(rhs: Float[Array, " m"]) -> Float[Array, " m"]:
            return jax.scipy.linalg.cho_solve((A_M_AT_chol, True), rhs)

        def _constraint_precond(r: Vector) -> Vector:
            Mr = _raw_precond(r)
            w = _solve_AMAT(ctx.A_work @ Mr)
            return Mr - M_AT @ w

        effective_precond: Callable[[Vector], Vector] | None = _constraint_precond
    else:
        effective_precond = precond_fn

    return run_projected_pcg(
        ctx=ctx,
        hvp_fn=hvp_fn,
        g=g,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        effective_precond=effective_precond,
        cg_regularization=cg_regularization,
    )


class ProjectedCGCholesky(AbstractInnerSolver):
    """Projected CG with Cholesky-based null-space projection.

    This is the original implementation: Cholesky-factor ``A A^T`` (with
    regularization), use it for the null-space projector and particular
    solution, run CG in the null space, and recover multipliers via
    iterative refinement.

    When ``use_constraint_preconditioner`` is ``True`` and a
    preconditioner is provided, the constraint preconditioner
    (Gould, Hribar & Nocedal, 2001) is used instead of the naive
    ``P(M(r))``.
    """

    max_cg_iter: int
    cg_tol: Scalar | float
    cg_regularization: float = 1e-6
    use_constraint_preconditioner: bool = False

    def solve(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
        adaptive_tol: Scalar | float | None = None,
    ) -> InnerSolveResult:
        effective_tol = adaptive_tol if adaptive_tol is not None else self.cg_tol
        d, multipliers, converged = _solve_projected_cg_cholesky(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_cg_iter=self.max_cg_iter,
            cg_tol=effective_tol,
            precond_fn=precond_fn,
            cg_regularization=self.cg_regularization,
            free_mask=free_mask,
            d_fixed=d_fixed,
            use_constraint_preconditioner=self.use_constraint_preconditioner,
        )
        # Null-space CG enforces ``A d = b`` structurally; the residual
        # is at floating-point floor by construction.  ``inf`` projected
        # gradient norm so the inexact-stationarity test cannot trip on
        # a non-HR inner solver.
        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            proj_residual=jnp.asarray(0.0, dtype=d.dtype),
            n_proj_refinements=jnp.asarray(0),
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=d.dtype),
        )

    def build_projection_context(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> ProjectionContext:
        return _make_cholesky_projection_ctx(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )


__all__ = ["ProjectedCGCholesky"]
