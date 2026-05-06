"""Preconditioned MINRES-QLP on the full saddle-point KKT system."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.krylov import pminres_qlp_solve
from slsqp_jax.inner.masking import make_active_subproblem
from slsqp_jax.state import InnerSolveResult
from slsqp_jax.types import Scalar, Vector
from slsqp_jax.utils import to_scalar


def _solve_kkt_minres_qlp(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_iter: int,
    tol: float | Scalar,
    precond_fn: Callable[[Vector], Vector] | None = None,
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
    proj_refine_max_iter: int = 3,
    proj_refine_rtol: float = 1e-10,
    proj_refine_atol: float = 1e-14,
) -> tuple[
    Vector,
    Float[Array, " m"],
    Bool[Array, ""],
    Scalar,
    Int[Array, ""],
]:
    """Solve equality-constrained QP via PMINRES-QLP on the full KKT system.

    The KKT system::

        [B    A^T] [d]       [-g]
        [A    0  ] [lambda] = [b ]

    is symmetric indefinite.  PMINRES-QLP solves it directly, producing
    both d and the Lagrange multipliers lambda.

    Uses a block-diagonal SPD preconditioner ``M^{-1} = diag(B_diag^{-1},
    S^{-1})`` where ``B_diag^{-1}`` is the user-supplied preconditioner
    (typically L-BFGS inverse Hessian diagonal) and ``S = A B_diag^{-1}
    A^T`` is the Schur complement.  This satisfies the SPD requirement
    from Choi (2006, Section 3.4).
    """
    n = g.shape[0]
    m = A.shape[0]
    tol = to_scalar(tol)

    sub = make_active_subproblem(
        hvp_fn=hvp_fn,
        g=g,
        A=A,
        b=b,
        active_mask=active_mask,
        free_mask=free_mask,
        d_fixed=d_fixed,
    )
    A_work = sub.A_work
    b_work = sub.b_work
    hvp_work = sub.hvp_work
    g_eff = sub.g_eff
    _free = sub.free_mask
    _dfixed = sub.d_fixed
    has_fixed = sub.has_fixed

    # KKT operator on (n+m)-dimensional vectors [d; lambda]
    def kkt_matvec(z: Vector) -> Vector:
        d_part = z[:n]
        lam_part = z[n:]
        top = hvp_work(d_part) + A_work.T @ lam_part
        bot = A_work @ d_part
        return jnp.concatenate([top, bot])

    kkt_rhs = jnp.concatenate([-g_eff, b_work])

    # Inactive constraint rows are zeroed in A_work / b_work; the
    # range-space and Schur factorisations need a "1" on those diagonal
    # positions to stay invertible without coupling into the active
    # block.  Hoisted out of the preconditioner branch so the no-precond
    # path can also reuse it for the posterior projection.
    reg_diag = jnp.where(active_mask, 0.0, 1.0)

    if precond_fn is not None:
        _raw_precond = precond_fn

        # When free_mask is active, mask the primal block so L-BFGS
        # cross-coupling does not leak non-zero values into the
        # zero-row/column dimensions.
        if has_fixed:
            _free_f = _free.astype(g.dtype)

            def _primal_precond(v: Vector) -> Vector:
                return _free_f * _raw_precond(_free_f * v)
        else:
            _primal_precond = _raw_precond

        # Schur complement S = A M^{-1} A^T  (m x m, SPD)
        M_AT = jax.vmap(_primal_precond)(A_work).T  # (n, m)
        A_M_AT = A_work @ M_AT + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_M_AT_chol = jnp.linalg.cholesky(A_M_AT)

        def _solve_schur(rhs_s: Float[Array, " m"]) -> Float[Array, " m"]:
            return jax.scipy.linalg.cho_solve((A_M_AT_chol, True), rhs_s)

        def kkt_precond(z: Vector) -> Vector:
            r1 = z[:n]
            r2 = z[n:]
            v1 = _primal_precond(r1)
            v2 = _solve_schur(r2)
            return jnp.concatenate([v1, v2])

        solution, converged = pminres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter, precond=kkt_precond
        )

        # M-metric range-space projection: minimise ||δd||_{M^{-1}} s.t.
        # A_work (d - δd) = b_work.
        def _project_step(d_in: Vector) -> tuple[Vector, Scalar]:
            r_dual = jnp.where(active_mask, A_work @ d_in - b_work, 0.0)
            r_norm = jnp.linalg.norm(r_dual)
            delta_lambda = _solve_schur(r_dual)
            delta_d = _primal_precond(A_work.T @ delta_lambda)
            return d_in - delta_d, r_norm
    else:
        solution, converged = pminres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter
        )

        # Build a small dedicated m x m Cholesky just for the posterior
        # 2-norm projection.
        A_AT = A_work @ A_work.T + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_AT_chol = jnp.linalg.cholesky(A_AT)

        def _project_step(d_in: Vector) -> tuple[Vector, Scalar]:
            r_dual = jnp.where(active_mask, A_work @ d_in - b_work, 0.0)
            r_norm = jnp.linalg.norm(r_dual)
            delta_lambda = jax.scipy.linalg.cho_solve((A_AT_chol, True), r_dual)
            return d_in - A_work.T @ delta_lambda, r_norm

    # Iterative refinement of the projection (HR 2014, Algorithm 4.18,
    # step 1(a)).  Each round squares the relative feasibility error.
    b_norm_floor = jnp.linalg.norm(b_work) + jnp.asarray(1.0, dtype=b_work.dtype)
    proj_atol = jnp.asarray(proj_refine_atol, dtype=b_work.dtype)
    proj_rtol = jnp.asarray(proj_refine_rtol, dtype=b_work.dtype)
    refine_target = proj_atol + proj_rtol * b_norm_floor

    d_proj, residual_pre = _project_step(solution[:n])
    n_refinements = jnp.asarray(0)

    def _refine_body(carry, _):
        d_cur, _r_prev, done_prev, n_done = carry
        d_next, r_next = _project_step(d_cur)
        d_out = jnp.where(done_prev, d_cur, d_next)
        r_out = jnp.where(done_prev, _r_prev, r_next)
        n_out = jnp.where(done_prev, n_done, n_done + 1)
        done_next = done_prev | (r_out <= refine_target)
        return (d_out, r_out, done_next, n_out), r_out

    if proj_refine_max_iter > 0:
        residual_init = jnp.linalg.norm(
            jnp.where(active_mask, A_work @ d_proj - b_work, 0.0)
        )
        done_init = residual_init <= refine_target
        (d_proj, residual_post, _done_final, n_refinements), _ = jax.lax.scan(
            _refine_body,
            (d_proj, residual_init, done_init, n_refinements),
            None,
            length=proj_refine_max_iter,
        )
    else:
        residual_post = jnp.linalg.norm(
            jnp.where(active_mask, A_work @ d_proj - b_work, 0.0)
        )

    del residual_pre

    d = d_proj
    if has_fixed:
        # Force the direction to respect the fixed mask.
        d = _free * d + _dfixed
    multipliers = -solution[n:]
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    finite = jnp.isfinite(d).all() & jnp.isfinite(multipliers).all()
    return d, multipliers, converged & finite, residual_post, n_refinements


class MinresQLPSolver(AbstractInnerSolver):
    """Preconditioned MINRES-QLP on the full saddle-point KKT system.

    Solves the KKT system directly::

        [B    A^T] [d]       [-g]
        [A    0  ] [lambda] = [b ]

    using PMINRES-QLP (Choi, Paige & Saunders, SISC 2011, Table 3.5)
    with a block-diagonal SPD preconditioner::

        M = [B_diag^{-1}    0      ]
            [0              S^{-1} ]

    where ``B_diag = diag(B_0)`` (L-BFGS diagonal) and
    ``S = A B_diag^{-1} A^T`` is the Schur complement.

    After PMINRES-QLP returns the iterate ``d``, an M-metric range-space
    projection drives ``A d = b`` on the active rows.  The single shot
    is followed by up to ``proj_refine_max_iter`` rounds of iterative
    refinement, each costing one matvec + one Schur back-solve (no
    refactorisation).  Refinement squares the relative feasibility
    error per round.  See HR (2014, Algorithm 4.18 step 1(a)) for the
    motivation.
    """

    max_iter: int = 200
    tol: float = 1e-10
    max_cg_iter: int = 50
    # Iterative refinement of the M-metric projection.  See class docstring.
    proj_refine_max_iter: int = 3
    proj_refine_rtol: float = 1e-10
    proj_refine_atol: float = 1e-14

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
        # MINRES-QLP solves the full KKT system where constraints A d = b
        # are part of the linear system. Loosening the tolerance would
        # degrade constraint satisfaction (unlike null-space CG where
        # constraints are enforced by the projector). Keep self.tol.
        d, multipliers, converged, proj_residual, n_refinements = _solve_kkt_minres_qlp(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_iter=self.max_iter,
            tol=self.tol,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
            proj_refine_max_iter=self.proj_refine_max_iter,
            proj_refine_rtol=self.proj_refine_rtol,
            proj_refine_atol=self.proj_refine_atol,
        )
        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            proj_residual=proj_residual,
            n_proj_refinements=n_refinements,
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=d.dtype),
        )


__all__ = ["MinresQLPSolver"]
