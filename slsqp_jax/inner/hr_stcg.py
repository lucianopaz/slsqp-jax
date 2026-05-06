"""Heinkenschloss-Ridzal (2014) Algorithm 4.5 — STCG with inexact projections."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.state import InnerSolveResult, ProjectionContext
from slsqp_jax.types import Scalar, Vector
from slsqp_jax.utils import to_scalar

# Hardcoded absolute floor for the HR-STCG inner-convergence test.  The
# Step 1(a) test ``||z̃|| <= cg_tol * ||r̃_0||`` becomes
# ``||z̃|| <= max(_HRSTCG_TOL_ABS, cg_tol * ||r̃_0||)`` so that when
# ``||r̃_0||`` itself is at or below machine epsilon the iteration does
# not chase a target below ``eps`` and return a spurious
# ``converged=False`` flag.
_HRSTCG_TOL_ABS = 1e-14


class _HRSTCGState(NamedTuple):
    """Internal state for the HR-inexact-STCG iteration.

    See the legacy ``inner_solver._HRSTCGState`` docstring for the full
    field-by-field description; the layout is preserved verbatim.
    """

    t: Vector
    r: Vector
    z: Vector
    p: Vector
    rz: Scalar
    proj_grad_norm: Scalar
    P: Float[Array, "imax n"]
    HP: Float[Array, "imax n"]
    pHp_diag: Float[Array, " imax"]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def _hr_stcg(
    hvp_work: Callable[[Vector], Vector],
    g_eff: Vector,
    project: Callable[[Vector], Vector],
    cg_tol: Scalar | float,
    cg_regularization: float,
    max_cg_iter: int,
) -> tuple[Vector, Scalar, Bool[Array, ""]]:
    """Run HR (2014) Algorithm 4.5 — STCG with inexact null-space projections.

    See the docstring of :class:`HRInexactSTCG` for the algorithmic
    description and rationale.
    """
    n = g_eff.shape[0]
    cg_tol = to_scalar(cg_tol)

    Wg = project(g_eff)
    r0 = -Wg
    z0 = r0
    proj_grad_norm = jnp.sqrt(jnp.maximum(jnp.dot(r0, r0), 0.0))
    rz0 = jnp.dot(r0, z0)
    p0 = z0

    P = jnp.zeros((max_cg_iter, n), dtype=g_eff.dtype)
    HP = jnp.zeros((max_cg_iter, n), dtype=g_eff.dtype)
    pHp_diag = jnp.zeros(max_cg_iter, dtype=g_eff.dtype)

    init_state = _HRSTCGState(
        t=jnp.zeros(n, dtype=g_eff.dtype),
        r=r0,
        z=z0,
        p=p0,
        rz=rz0,
        proj_grad_norm=proj_grad_norm,
        P=P,
        HP=HP,
        pHp_diag=pHp_diag,
        iteration=jnp.array(0),
        # Pre-converge if the projected gradient is already at the
        # absolute floor.
        converged=jnp.reshape(proj_grad_norm <= _HRSTCG_TOL_ABS, ()),
    )

    indices = jnp.arange(max_cg_iter)

    def step_fn(_i: int, state: _HRSTCGState) -> _HRSTCGState:
        def do_step(state: _HRSTCGState) -> _HRSTCGState:
            i = state.iteration
            Hp = hvp_work(state.p)
            pHp = jnp.dot(state.p, Hp)
            pp = jnp.dot(state.p, state.p)

            # SNOPT-style scale-invariant curvature guard plus an
            # absolute floor anchored to the initial projected gradient.
            abs_floor = cg_regularization * state.proj_grad_norm * state.proj_grad_norm
            bad_curvature = pHp <= jnp.maximum(cg_regularization * pp, abs_floor)
            rp = jnp.dot(state.r, state.p)
            stagnation = jnp.abs(rp) < 1e-30
            short_circuit = bad_curvature | stagnation

            pHp_safe = jnp.maximum(pHp, 1e-30)
            alpha = jnp.where(short_circuit, jnp.array(0.0), state.rz / pHp_safe)

            t_new = state.t + alpha * state.p
            # HR Remark 4.6.i — modified residual recurrence.
            r_new = state.r - alpha * Hp
            z_new = project(r_new)

            P_buf = state.P.at[i].set(state.p)
            HP_buf = state.HP.at[i].set(Hp)
            pHp_buf = state.pHp_diag.at[i].set(pHp)

            # Full reorthogonalisation: enforce H-conjugacy against
            # every stored p_j.
            mask_j = indices <= i  # include the just-stored p_i
            Hz_dots = HP_buf @ z_new  # (max_cg_iter,)
            pHp_diag_safe = jnp.where(jnp.abs(pHp_buf) > 1e-30, pHp_buf, 1e-30)
            coeffs = jnp.where(mask_j, Hz_dots / pHp_diag_safe, 0.0)
            p_new = z_new - coeffs @ P_buf

            rz_new = jnp.dot(r_new, z_new)
            z_norm = jnp.sqrt(jnp.maximum(jnp.dot(z_new, z_new), 0.0))
            tol_target = jnp.maximum(_HRSTCG_TOL_ABS, cg_tol * state.proj_grad_norm)
            converged_new = (z_norm <= tol_target) | short_circuit

            return _HRSTCGState(
                t=jnp.where(short_circuit, state.t, t_new),
                r=jnp.where(short_circuit, state.r, r_new),
                z=jnp.where(short_circuit, state.z, z_new),
                p=jnp.where(short_circuit, state.p, p_new),
                rz=jnp.where(short_circuit, state.rz, rz_new),
                proj_grad_norm=state.proj_grad_norm,
                P=P_buf,
                HP=HP_buf,
                pHp_diag=pHp_buf,
                iteration=state.iteration + 1,
                converged=converged_new,
            )

        converged_pred = jnp.reshape(state.converged, ())
        return jax.lax.cond(converged_pred, lambda s: s, do_step, state)

    final = jax.lax.fori_loop(0, max_cg_iter, step_fn, init_state)
    return final.t, final.proj_grad_norm, final.converged


class HRInexactSTCG(AbstractInnerSolver):
    """Heinkenschloss-Ridzal (2014) Algorithm 4.5 — STCG with inexact
    null-space projections.

    Composes an existing null-space inner solver
    (``ProjectedCGCholesky`` or ``ProjectedCGCraig``) to obtain its
    projector ``W̃_k``, particular solution ``d_p`` and
    multiplier-recovery closure, then runs a *separate* CG iteration
    on top whose three textbook three-term-recurrence cancellations are
    replaced by full H-conjugacy reorthogonalisation against every
    previous search direction.

    See ``AGENTS.md`` ("Pluggable Inner QP Solvers" → ``HRInexactSTCG``)
    for the full algorithmic discussion and references.

    Attributes:
        inner: Composed null-space inner solver supplying the
            projector and multiplier-recovery infrastructure.  Must
            implement ``build_projection_context``; the saddle-point
            ``MinresQLPSolver`` does not and will raise on the first
            ``solve`` call.
        max_cg_iter: Static upper bound on the number of inner CG
            iterations.  Determines the size of the reorth buffers.
        cg_tol: Relative convergence tolerance for the projected
            residual ``‖z̃_i‖ ≤ tol · ‖r̃_0‖``.
        cg_regularization: Curvature-guard threshold ``δ²`` used by
            the SNOPT-style scale-invariant short-circuit
            ``⟨p̃, H p̃⟩ ≤ δ² ‖p̃‖²``.  Defaults to ``1e-6``; set to
            ``0.0`` to disable.
    """

    inner: AbstractInnerSolver
    max_cg_iter: int
    cg_tol: Scalar | float
    cg_regularization: float = 1e-6

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
        # Delegate to the composed projector; HRInexactSTCG itself does
        # not provide an additional projector layer.
        return self.inner.build_projection_context(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )

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
        # ``precond_fn`` is accepted for interface compatibility but
        # silently dropped: HR Algorithm 4.5 uses no inner
        # preconditioner.
        ctx = self.inner.build_projection_context(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )
        effective_tol = adaptive_tol if adaptive_tol is not None else self.cg_tol

        # HR-STCG iterates a null-space step ``t̃`` starting from
        # ``d_p``; the effective gradient handed to the CG iteration is
        # the gradient of the quadratic at ``d_p``: ``g_k = g_eff + B d_p``.
        Bd_p = ctx.hvp_work(ctx.d_p)
        g_at_dp = ctx.g_eff + Bd_p

        t_tilde, proj_grad_norm, cg_converged = _hr_stcg(
            hvp_work=ctx.hvp_work,
            g_eff=g_at_dp,
            project=ctx.project,
            cg_tol=effective_tol,
            cg_regularization=self.cg_regularization,
            max_cg_iter=self.max_cg_iter,
        )

        d = ctx.d_p + t_tilde

        Bd = hvp_fn(d)
        multipliers = ctx.recover_multipliers(Bd + g)

        finite_d = jnp.isfinite(d).all()
        finite_mult = jnp.isfinite(multipliers).all()
        converged = cg_converged & ctx.converged & finite_d & finite_mult

        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            # Null-space projector enforces ``A d = b`` structurally.
            proj_residual=jnp.asarray(0.0, dtype=d.dtype),
            n_proj_refinements=jnp.asarray(0),
            projected_grad_norm=proj_grad_norm.astype(d.dtype),
        )


__all__ = ["HRInexactSTCG"]
