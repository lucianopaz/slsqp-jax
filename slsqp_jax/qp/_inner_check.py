"""Shared utility for checking inner-solver result usability.

The QP layer flags an inner solve as a hard failure only when the
direction is non-finite — see the long comment on :func:`inner_ok` for
the rationale.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool


def inner_ok(result) -> Bool[Array, ""]:
    """Did the inner solve return a usable (finite) direction?

    NOTE: we deliberately do *not* require ``result.converged`` here.
    For an outer SQP solver an imprecise inner CG/MINRES direction is
    usually still a productive descent direction — failing the QP just
    because CG ran out of iterations would trigger an L-BFGS reset and
    a projected-gradient fallback at every outer step, which is far
    worse than taking a slightly stale Newton direction.  We flag only
    unusable directions (NaN/Inf), and track CG non-convergence via the
    diagnostic counters in ``SLSQPState`` instead.
    """
    return jnp.isfinite(result.d).all()


__all__ = ["inner_ok"]
