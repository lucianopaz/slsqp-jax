"""Projected CG with CRAIG-based iterative null-space projection."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.krylov import _CRAIG_TOL_ABS, craig_solve, solve_unconstrained_cg
from slsqp_jax.inner.masking import make_active_subproblem
from slsqp_jax.inner.projected_cg import run_projected_pcg
from slsqp_jax.state import InnerSolveResult, ProjectionContext
from slsqp_jax.types import Scalar, Vector


def _make_craig_projection_ctx(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
    craig_tol: float = 1e-10,
    craig_max_iter: int = 200,
    mult_recovery_tol: float = 1e-12,
    mult_recovery_max_iter: int = 200,
) -> ProjectionContext:
    """Build a ``ProjectionContext`` backed by CRAIG (Golub-Kahan
    bidiagonalisation) for the null-space projection.

    The ``converged`` flag carries the CRAIG breakdown / convergence
    status of the *particular solution* solve (``A_work d_p = b_work``);
    per-projector-call CRAIG convergence flags inside the CG loop are
    not threaded through.  The multiplier-recovery closure runs CG on
    the normal equations ``A A^T λ = -A (B d + g)``.
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

    d_p_free, d_p_craig_conv = craig_solve(
        sub.A_work, sub.b_work, tol=craig_tol, max_iter=craig_max_iter
    )
    d_p_free_finite = jnp.isfinite(d_p_free).all()
    d_p_free = jnp.where(d_p_free_finite, d_p_free, jnp.zeros_like(d_p_free))
    d_p = d_p_free + sub.d_fixed if sub.has_fixed else d_p_free

    def project(v: Vector) -> Vector:
        v_work = sub.free_mask * v if sub.has_fixed else v
        x_proj, _ = craig_solve(
            sub.A_work,
            sub.A_work @ v_work,
            tol=craig_tol,
            max_iter=craig_max_iter,
        )
        x_proj = jnp.where(jnp.isfinite(x_proj).all(), x_proj, jnp.zeros_like(x_proj))
        return v_work - x_proj

    reg_diag = jnp.where(active_mask, 0.0, 1.0)

    def normal_hvp(v: Float[Array, " m"]) -> Float[Array, " m"]:
        return sub.A_work @ (sub.A_work.T @ v) + reg_diag * v

    def recover_multipliers(Bd_plus_g: Vector) -> Float[Array, " m"]:
        kkt_rhs = sub.A_work @ Bd_plus_g
        # The absolute floor _CRAIG_TOL_ABS is reused so the projector
        # and the multiplier recovery stop at the same noise floor.
        mult, _ = solve_unconstrained_cg(
            normal_hvp,
            -kkt_rhs,
            mult_recovery_max_iter,
            mult_recovery_tol,
            cg_atol=_CRAIG_TOL_ABS,
        )
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
        converged=d_p_craig_conv,
    )


def _solve_projected_cg_craig(
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
    craig_tol: float = 1e-10,
    craig_max_iter: int = 200,
    mult_recovery_tol: float = 1e-12,
    mult_recovery_max_iter: int = 200,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Projected CG using CRAIG for the null-space projection."""
    ctx = _make_craig_projection_ctx(
        hvp_fn=hvp_fn,
        g=g,
        A=A,
        b=b,
        active_mask=active_mask,
        free_mask=free_mask,
        d_fixed=d_fixed,
        craig_tol=craig_tol,
        craig_max_iter=craig_max_iter,
        mult_recovery_tol=mult_recovery_tol,
        mult_recovery_max_iter=mult_recovery_max_iter,
    )

    if precond_fn is not None and use_constraint_preconditioner:
        _raw_precond = precond_fn

        def _constraint_precond(r: Vector) -> Vector:
            Mr = _raw_precond(r)
            AMr = ctx.A_work @ Mr

            # CG on normal equations A M A^T w = A M r:
            def amat_hvp(v: Float[Array, " m"]) -> Float[Array, " m"]:
                return ctx.A_work @ _raw_precond(ctx.A_work.T @ v)

            reg_diag = jnp.where(active_mask, 0.0, 1.0)

            def amat_hvp_reg(v: Float[Array, " m"]) -> Float[Array, " m"]:
                return amat_hvp(v) + reg_diag * v

            w, _ = solve_unconstrained_cg(
                amat_hvp_reg,
                -AMr,
                mult_recovery_max_iter,
                mult_recovery_tol,
                cg_atol=_CRAIG_TOL_ABS,
            )
            return Mr - _raw_precond(ctx.A_work.T @ w)

        effective_precond: Callable[[Vector], Vector] | None = _constraint_precond
    else:
        effective_precond = precond_fn

    d, multipliers, cg_converged = run_projected_pcg(
        ctx=ctx,
        hvp_fn=hvp_fn,
        g=g,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        effective_precond=effective_precond,
        cg_regularization=cg_regularization,
    )

    finite_d = jnp.isfinite(d).all()
    finite_mult = jnp.isfinite(multipliers).all()
    converged = cg_converged & ctx.converged & finite_d & finite_mult

    return d, multipliers, converged


class ProjectedCGCraig(AbstractInnerSolver):
    """Projected CG with CRAIG-based iterative null-space projection.

    Replaces the Cholesky factorization of ``A A^T`` with iterative
    CRAIG solves (Golub-Kahan bidiagonalization).  This eliminates the
    ``O(m^3)`` factorization cost and the ``1e-8`` diagonal
    regularization, at the cost of an iterative solve per projection.

    For multiplier recovery (done once after the CG loop), CG on the
    normal equations ``A A^T y = rhs`` is used, reusing the existing
    ``solve_unconstrained_cg`` infrastructure.
    """

    max_cg_iter: int
    cg_tol: Scalar | float
    cg_regularization: float = 1e-6
    use_constraint_preconditioner: bool = False
    craig_tol: float = 1e-10
    craig_max_iter: int = 200
    # Multiplier recovery uses CG on the normal equations.  Its tolerance
    # is kept tight (and independent of craig_tol) so the Lagrangian
    # residual is not polluted by imprecise multipliers, without forcing
    # the inner CRAIG projections to the same accuracy.
    mult_recovery_tol: float = 1e-12
    mult_recovery_max_iter: int = 200

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
        d, multipliers, converged = _solve_projected_cg_craig(
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
            craig_tol=self.craig_tol,
            craig_max_iter=self.craig_max_iter,
            mult_recovery_tol=self.mult_recovery_tol,
            mult_recovery_max_iter=self.mult_recovery_max_iter,
        )
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
        return _make_craig_projection_ctx(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            free_mask=free_mask,
            d_fixed=d_fixed,
            craig_tol=self.craig_tol,
            craig_max_iter=self.craig_max_iter,
            mult_recovery_tol=self.mult_recovery_tol,
            mult_recovery_max_iter=self.mult_recovery_max_iter,
        )


__all__ = ["ProjectedCGCraig"]
