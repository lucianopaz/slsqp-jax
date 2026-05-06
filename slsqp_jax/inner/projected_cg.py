"""Shared projected-PCG driver for Cholesky and CRAIG inner solvers.

Both ``ProjectedCGCholesky`` and ``ProjectedCGCraig`` historically held a
copy of the same projected-PCG outer loop: build ``r0 = project(-(g_eff
+ B d_p))``, prime ``z0 = project(precond(r0))``, run :func:`build_cg_step`
inside :func:`jax.lax.fori_loop`, then recover multipliers from the full
HVP via the projection context.  The only difference between the two was
the *constraint preconditioner* construction, which each solver supplies
externally.

This module hosts the shared driver :func:`run_projected_pcg`.  Each
concrete solver builds its own ``ProjectionContext`` and (optionally) its
own constraint-preconditioner factory and delegates the rest here.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.inner.krylov import _CGState, build_cg_step
from slsqp_jax.state import ProjectionContext
from slsqp_jax.types import Scalar, Vector
from slsqp_jax.utils import to_scalar


def run_projected_pcg(
    ctx: ProjectionContext,
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    max_cg_iter: int,
    cg_tol: Scalar | float,
    effective_precond: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Run projected (preconditioned) CG against an existing context.

    Performs the body of ``_solve_projected_cg_*`` shared by both the
    Cholesky and CRAIG solvers:

    1. Initialise ``r0 = project(-(g_eff + B d_p))``.
    2. Optionally precondition ``z0 = project(precond(r0))`` (positive
       definiteness check; fall back to ``z0 = r0`` on rz0 <= 0).
    3. Run :func:`build_cg_step` inside ``jax.lax.fori_loop``.
    4. Recover Lagrange multipliers from the full unmasked HVP via
       ``ctx.recover_multipliers(B d + g)``.

    Args:
        ctx: Projection context built by the caller (Cholesky, CRAIG,
            or a composed strategy).  Provides ``A_work``, ``project``,
            ``hvp_work``, ``g_eff``, ``d_p``.
        hvp_fn: The full *unmasked* HVP, used only for multiplier
            recovery.  ``ctx.hvp_work`` is used inside the CG loop.
        g: Full unmasked gradient (used for multiplier recovery).
        max_cg_iter: CG iteration budget.
        cg_tol: CG residual-norm tolerance.
        effective_precond: Already-wrapped preconditioner (e.g. from a
            constraint preconditioner factory) or ``None``.
        cg_regularization: Curvature-guard threshold.

    Returns:
        ``(d, multipliers, cg_converged)``.
    """
    cg_tol = to_scalar(cg_tol)

    Bd_p = ctx.hvp_work(ctx.d_p)
    r0 = ctx.project(-(ctx.g_eff + Bd_p))
    r0_norm_sq = jnp.dot(r0, r0)

    if effective_precond is not None:
        z0 = ctx.project(effective_precond(r0))
        rz0_raw = jnp.dot(r0, z0)
        z0 = jnp.where(rz0_raw > 0, z0, r0)
        rz0 = jnp.where(rz0_raw > 0, rz0_raw, r0_norm_sq)
        p0 = z0
    else:
        rz0 = r0_norm_sq
        p0 = r0

    init_cg = _CGState(
        d=ctx.d_p,
        r=r0,
        p=p0,
        rz=rz0,
        iteration=jnp.array(0),
        converged=jnp.reshape(r0_norm_sq < cg_tol**2, ()),
    )

    cg_step = build_cg_step(
        hvp_fn=ctx.hvp_work,
        cg_tol=cg_tol,
        precond_fn=effective_precond,
        project=ctx.project,
        cg_regularization=cg_regularization,
    )

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)

    Bd = hvp_fn(final_cg.d)
    multipliers = ctx.recover_multipliers(Bd + g)

    return final_cg.d, multipliers, final_cg.converged


__all__ = ["run_projected_pcg"]
