"""Single source of truth for SLSQP termination classification.

Both ``SLSQP.step`` and ``SLSQP.terminate`` need to decide whether
the run should stop and *why*.  Historically these were two
near-identical bodies (the ``step`` mirror existed because Optimistix's
``iterate_solve`` driver requires the coarse ``optx.RESULTS`` enum
while we want to expose the granular :class:`slsqp_jax.RESULTS`).
This module collapses both into a pair of small pure helpers that
share their inputs by structure.

* :func:`classify_outcome` — given primal-feasibility / stationarity /
  failure flags, return the granular :class:`slsqp_jax.RESULTS` code.
  Used by ``step`` to populate ``state.termination_code``.
* :func:`coarse_outcome` — collapse the same flags onto the
  ``optx.RESULTS`` enum that the Optimistix driver requires.  Used
  by ``terminate``.

Both are thin pattern-matchers over the same flag set; refactoring
either one in the future should preserve their structural symmetry.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool

from slsqp_jax.results import RESULTS


class TerminationFlags(NamedTuple):
    """Flag bundle classify_outcome / coarse_outcome both consume."""

    converged: Bool[Array, ""]
    nonfinite: Bool[Array, ""]
    diverging: Bool[Array, ""]
    ls_fatal: Bool[Array, ""]
    qp_fatal: Bool[Array, ""]
    merit_stagnation: Bool[Array, ""]
    max_iters_reached: Bool[Array, ""]
    primal_feasible: Bool[Array, ""]


def classify_outcome(flags: TerminationFlags) -> RESULTS:
    """Granular :class:`slsqp_jax.RESULTS` classification.

    Priority order (highest wins):

    1. ``successful``
    2. ``nonfinite``
    3. ``infeasible`` (override on any non-success termination at
       an infeasible iterate)
    4. ``iterate_blowup`` (best-iterate divergence rollback)
    5. ``line_search_failure``
    6. ``qp_subproblem_failure``
    7. ``merit_stagnation``
    8. ``nonlinear_max_steps_reached``
    """
    code = RESULTS.successful
    code = RESULTS.where(
        jnp.reshape(flags.max_iters_reached, ()),
        RESULTS.nonlinear_max_steps_reached,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.merit_stagnation, ()),
        RESULTS.merit_stagnation,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.qp_fatal, ()),
        RESULTS.qp_subproblem_failure,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.ls_fatal, ()),
        RESULTS.line_search_failure,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.diverging, ()),
        RESULTS.iterate_blowup,
        code,
    )
    non_success_done = (
        flags.merit_stagnation | flags.qp_fatal | flags.ls_fatal | flags.diverging
    )
    code = RESULTS.where(
        jnp.reshape(non_success_done & ~flags.primal_feasible, ()),
        RESULTS.infeasible,
        code,
    )
    code = RESULTS.where(
        jnp.reshape(flags.nonfinite, ()),
        RESULTS.nonfinite,
        code,
    )
    code = RESULTS.where(flags.converged, RESULTS.successful, code)
    return code


def coarse_outcome(flags: TerminationFlags) -> tuple[Bool[Array, ""], optx.RESULTS]:
    """Coarse ``(done, result)`` for the Optimistix driver.

    The driver requires the parent ``optx.RESULTS`` enum class
    (subclass instances would crash ``RESULTS.where``), so we only
    expose the four codes it knows about: ``successful``,
    ``nonfinite``, ``nonlinear_divergence`` (catch-all for stagnation /
    LS / QP / blowup), and ``nonlinear_max_steps_reached``.
    """
    done = (
        flags.converged
        | flags.max_iters_reached
        | flags.merit_stagnation
        | flags.ls_fatal
        | flags.qp_fatal
        | flags.diverging
        | flags.nonfinite
    )
    non_success_done = (
        flags.merit_stagnation | flags.ls_fatal | flags.qp_fatal | flags.diverging
    )
    result = jax.lax.cond(
        flags.converged,
        lambda: optx.RESULTS.successful,
        lambda: jax.lax.cond(
            flags.nonfinite,
            lambda: optx.RESULTS.nonfinite,
            lambda: jax.lax.cond(
                non_success_done,
                lambda: optx.RESULTS.nonlinear_divergence,
                lambda: jax.lax.cond(
                    flags.max_iters_reached,
                    lambda: optx.RESULTS.nonlinear_max_steps_reached,
                    lambda: optx.RESULTS.successful,
                ),
            ),
        ),
    )
    return done, result


__all__ = ["TerminationFlags", "classify_outcome", "coarse_outcome"]
