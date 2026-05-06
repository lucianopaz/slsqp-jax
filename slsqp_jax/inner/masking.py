"""Shared active-row masking + bound-fix helper for inner solvers.

The three projector-based inner solvers (``ProjectedCGCholesky``,
``ProjectedCGCraig``, ``MinresQLPSolver``) all start from the same
five-line preamble: mask ``A`` and ``b`` to the active rows, optionally
project away the bound-fixed columns, and build a working HVP and
effective gradient that hide the fixed coordinates from the iteration.

Before this module that preamble was copy-pasted in three places.  This
module hosts the single shared implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from slsqp_jax.types import Vector


class ActiveSubproblem(NamedTuple):
    """Data carried by every projector-based inner solver after masking.

    Attributes:
        A_work: ``A`` with inactive rows zeroed and (when bound-fixing
            is in effect) fixed columns zeroed.
        b_work: ``b`` with inactive entries zeroed and the fixed-column
            contribution ``A_masked @ d_fixed`` subtracted.
        free_mask: Boolean mask of free variables (``ones(n)`` when no
            bound fixing).
        d_fixed: Fixed-variable values on bound-active coordinates
            (zeros elsewhere; zeros everywhere when no bound fixing).
        has_fixed: ``True`` iff any coordinate is bound-fixed.
        hvp_work: Working-subspace HVP.  Equals ``hvp_fn`` when no
            bound-fixing; otherwise ``v -> _free * hvp_fn(_free * v)``
            so the iteration only sees the free coordinates.
        g_eff: Effective gradient.  Equals ``g`` when no bound-fixing;
            otherwise ``_free * (g + hvp_fn(d_fixed))`` to absorb the
            fixed-column cross-coupling into the linear term.
    """

    A_work: Float[Array, "m n"]
    b_work: Float[Array, " m"]
    free_mask: Bool[Array, " n"]
    d_fixed: Vector
    has_fixed: bool
    hvp_work: Callable[[Vector], Vector]
    g_eff: Vector


def make_active_subproblem(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
) -> ActiveSubproblem:
    """Build the masked subproblem consumed by every projector-based solver.

    Implements the shared preamble:

    1. ``A_masked = A`` with inactive rows zeroed.
    2. ``b_masked = b`` with inactive entries zeroed.
    3. If bound-fixing is in effect (``free_mask`` and ``d_fixed`` both
       provided), zero the fixed columns of ``A_masked`` and absorb the
       fixed-column contribution into ``b_work``.
    4. Build ``hvp_work`` and ``g_eff`` so the iteration only sees the
       free coordinates.

    See :class:`ActiveSubproblem` for the field semantics.
    """
    n = A.shape[1]
    has_fixed = free_mask is not None and d_fixed is not None

    A_masked = jnp.where(active_mask[:, None], A, 0.0)
    b_masked = jnp.where(active_mask, b, 0.0)

    if has_fixed and free_mask is not None and d_fixed is not None:
        A_work = A_masked * free_mask[None, :]
        b_work = b_masked - A_masked @ d_fixed
    else:
        A_work = A_masked
        b_work = b_masked

    _free: Bool[Array, " n"] = (
        free_mask if free_mask is not None else jnp.ones(n, dtype=bool)
    )
    _dfixed: Vector = d_fixed if d_fixed is not None else jnp.zeros(n)

    if has_fixed:

        def hvp_work(v: Vector) -> Vector:
            return _free * hvp_fn(_free * v)

        g_eff = _free * (g + hvp_fn(_dfixed))
    else:
        hvp_work = hvp_fn
        g_eff = g

    return ActiveSubproblem(
        A_work=A_work,
        b_work=b_work,
        free_mask=_free,
        d_fixed=_dfixed,
        has_fixed=bool(has_fixed),
        hvp_work=hvp_work,
        g_eff=g_eff,
    )


__all__ = ["ActiveSubproblem", "make_active_subproblem"]
