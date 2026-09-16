"""Termination metrics, flags, and the shared outcome classifier.

Termination is split into three stages so that only the *first* is
algorithm-specific:

1. ``termination_metrics(ctx)`` measures whatever scalars the algorithm's
   convergence test needs, in a :class:`TerminationMetrics` subclass of its
   own choosing;
2. ``termination_flags(ctx, metrics)`` reduces those measurements to the
   universal :class:`TerminationFlags` booleans;
3. :func:`classify_termination` maps the flags to the ``(done, result)`` pair
   that :meth:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser.terminate`
   returns.

Only stage 1 has a per-algorithm type, tracked by the
:data:`TerminationMetricsType` type variable on
:class:`~slsqp_jax.sqpdax.minimiser.base.AbstractConstrainedMinimiser` so a
minimiser cannot be wired to another algorithm's metrics without a static
type error. Stages 2 and 3 are shared, which is what keeps the meaning of an
:class:`optimistix.RESULTS` code the same across every minimiser.
"""

from typing import TypeVar

import jax
import optimistix as optx
from equinox import Module
from jax import Array
from jaxtyping import Bool

__all__ = [
    "TerminationMetrics",
    "TerminationMetricsType",
    "TerminationFlags",
    "classify_termination",
]


class TerminationMetrics(Module):
    """Base schema for the measurements a termination test consumes.

    Only the two algorithm-independent quantities live here. Concrete
    minimisers subclass this with the residuals and tolerances their own
    convergence test needs — an active-set line search tracks stationarity
    and feasibility, a trust-region interior-point method tracks the
    Nocedal & Wright barrier KKT error — rather than sharing one schema
    that fits neither.

    Attributes
    ----------
    nonfinite
        ``True`` when any tracked quantity at the current iterate is NaN or
        ``±Inf``. No further iteration can recover from this, so termination
        exits immediately.
    subproblem_result
        Status left behind by the last subproblem solve, promoted to
        :class:`optimistix.RESULTS`. Termination escalates on it because the
        outer loop did not observe the failure itself.
    """

    nonfinite: Bool[Array, ""]
    subproblem_result: optx.RESULTS


TerminationMetricsType = TypeVar("TerminationMetricsType", bound=TerminationMetrics)
"""Per-algorithm :class:`TerminationMetrics` subclass carried by a minimiser."""


class TerminationFlags(Module):
    """Algorithm-independent termination decision, before classification.

    Every minimiser reduces its own metrics to these booleans, which is the
    point at which the algorithms stop differing: :func:`classify_termination`
    consumes only this type, so the mapping from "what happened" to an
    :class:`optimistix.RESULTS` code is defined once.

    Attributes
    ----------
    converged
        ``True`` when the algorithm's convergence test is satisfied.
    nonfinite
        ``True`` when a non-finite quantity was detected (see
        :attr:`TerminationMetrics.nonfinite`).
    fatal
        ``True`` when an unrecoverable failure fired, e.g. a subproblem solve
        that did not return :attr:`optimistix.RESULTS.successful`.
    subproblem_result
        Status code reported when :attr:`fatal` fires and :attr:`nonfinite`
        does not. Forwarded from the metrics so a singular KKT solve surfaces
        as :attr:`optimistix.RESULTS.singular` rather than being flattened
        into a generic divergence code.
    """

    converged: Bool[Array, ""]
    nonfinite: Bool[Array, ""]
    fatal: Bool[Array, ""]
    subproblem_result: optx.RESULTS


def classify_termination(
    flags: TerminationFlags,
) -> tuple[Bool[Array, ""], optx.RESULTS]:
    """Map termination flags to the ``(done, result)`` pair the driver expects.

    Precedence is ``nonfinite`` > ``converged`` > ``fatal``: a non-finite
    iterate is reported even if the convergence test happens to pass on it,
    and convergence outranks a failed subproblem because an iterate that meets
    the tolerances is a usable answer regardless of how the last solve went.

    ``result`` stays :attr:`optimistix.RESULTS.successful` while the loop is
    still running; the driver maps ``done == False`` at the step budget to
    :attr:`optimistix.RESULTS.nonlinear_max_steps_reached`.

    Parameters
    ----------
    flags
        Reduced termination decision from a minimiser's ``termination_flags``.

    Returns
    -------
    done
        ``True`` when the outer loop should stop.
    result
        Status code for :class:`optimistix.Solution`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import optimistix as optx
    >>> from slsqp_jax.sqpdax.minimiser.termination import (
    ...     TerminationFlags,
    ...     classify_termination,
    ... )
    >>> def flags(converged, nonfinite, fatal):
    ...     return TerminationFlags(
    ...         converged=jnp.asarray(converged),
    ...         nonfinite=jnp.asarray(nonfinite),
    ...         fatal=jnp.asarray(fatal),
    ...         subproblem_result=optx.RESULTS.singular,
    ...     )
    >>> done, result = classify_termination(flags(True, False, False))
    >>> bool(done), bool(result == optx.RESULTS.successful)
    (True, True)
    >>> done, result = classify_termination(flags(False, False, True))
    >>> bool(done), bool(result == optx.RESULTS.singular)
    (True, True)
    >>> done, result = classify_termination(flags(False, False, False))
    >>> bool(done), bool(result == optx.RESULTS.successful)
    (False, True)
    """
    done = flags.nonfinite | flags.converged | flags.fatal
    return (
        done,
        jax.lax.cond(
            done,
            lambda: jax.lax.cond(
                flags.nonfinite,
                lambda: optx.RESULTS.nonfinite,
                lambda: jax.lax.cond(
                    flags.converged,
                    lambda: optx.RESULTS.successful,
                    lambda: jax.lax.cond(
                        flags.fatal,
                        lambda: flags.subproblem_result,
                        lambda: optx.RESULTS.successful,
                    ),
                ),
            ),
            lambda: optx.RESULTS.successful,
        ),
    )
