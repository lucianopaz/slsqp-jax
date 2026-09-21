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
type error. Stages 2 and 3 are shared while the result adapter supplies the
concrete algorithm-native enumeration.
"""

from enum import IntEnum
from typing import Generic, TypeVar

from equinox import Module
from jax import Array
from jax import numpy as jnp
from jaxtyping import Bool, Float

from ..lagrangian import EvaluatedLagrangian
from ..results import ResultAdapter, ResultType
from ..types import Scalar

__all__ = [
    "TerminationMetrics",
    "TerminationMetricsType",
    "TerminationFlags",
    "TerminationPriority",
    "compute_mu_max",
    "classify_termination",
]


class TerminationMetrics(Module, Generic[ResultType]):
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
    fatal_result
        Algorithm-native result selected if a fatal diagnostic fires.
    """

    nonfinite: Bool[Array, ""]
    fatal_result: ResultType


TerminationMetricsType = TypeVar("TerminationMetricsType", bound=TerminationMetrics)
"""Per-algorithm :class:`TerminationMetrics` subclass carried by a minimiser."""


class TerminationFlags(Module, Generic[ResultType]):
    """Algorithm-independent termination decision, before classification.

    Every minimiser reduces its own metrics to these booleans, which is the
    point at which the algorithms stop differing: :func:`classify_termination`
    consumes only this type and a result adapter.

    Attributes
    ----------
    converged
        ``True`` when the algorithm's convergence test is satisfied.
    nonfinite
        ``True`` when a non-finite quantity was detected (see
        :attr:`TerminationMetrics.nonfinite`).
    fatal
        ``True`` when an unrecoverable algorithm-specific failure fired.
    fatal_result
        Fine-grained native status reported when :attr:`fatal` wins.
    """

    converged: Bool[Array, ""]
    nonfinite: Bool[Array, ""]
    fatal: Bool[Array, ""]
    fatal_result: ResultType


class TerminationPriority(IntEnum):
    """Default precedence for simultaneously active termination candidates."""

    running = 0
    fatal = 10
    converged = 20
    nonfinite = 30


def compute_mu_max(lagrangian: EvaluatedLagrangian) -> Scalar:
    """Compute filterSQP's scale from gradient, Jacobian rows, and multipliers.

    Parameters
    ----------
    lagrangian
        Evaluated NLP Lagrangian at the candidate solution.

    Returns
    -------
    Scalar
        Largest objective, general-constraint, or bound contribution.
    """
    dual = lagrangian.dual
    terms: list[Float[Array, " n_terms"]] = [
        jnp.reshape(jnp.linalg.norm(lagrangian.grad_val), (1,)),
        jnp.linalg.norm(lagrangian.eq_fn_jac_val, axis=1)
        * jnp.abs(dual.eq_multipliers),
        jnp.linalg.norm(lagrangian.ineq_fn_jac_val, axis=1)
        * jnp.abs(dual.ineq_multipliers),
        jnp.where(lagrangian.null_lb, 0.0, jnp.abs(dual.lb_multipliers)),
        jnp.where(lagrangian.null_ub, 0.0, jnp.abs(dual.ub_multipliers)),
    ]
    return jnp.reshape(jnp.max(jnp.concatenate(terms)), ())


def classify_termination(
    flags: TerminationFlags[ResultType],
    adapter: ResultAdapter[ResultType],
) -> tuple[Bool[Array, ""], ResultType]:
    """Map termination flags to the ``(done, result)`` pair the driver expects.

    Precedence is ``nonfinite`` > ``converged`` > ``fatal``: a non-finite
    iterate is reported even if the convergence test happens to pass on it,
    and convergence outranks a failed subproblem because an iterate that meets
    the tolerances is a usable answer regardless of how the last solve went.

    ``result`` is the algorithm's native ``running`` member while the loop is
    active. The owned driver maps step-budget exhaustion to the corresponding
    native ``max_steps_reached`` member.

    Parameters
    ----------
    flags
        Reduced termination decision from a minimiser's ``termination_flags``.
    adapter
        Construction and conversion policy for the concrete result enumeration.

    Returns
    -------
    done
        ``True`` when the outer loop should stop.
    result
        Fine-grained algorithm-native status code.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.minimiser.active_set_linesearch import (
    ...     ACTIVE_SET_LINE_SEARCH_RESULTS as RESULTS,
    ...     ActiveSetLineSearchResultAdapter,
    ... )
    >>> from slsqp_jax.sqpdax.minimiser.termination import (
    ...     TerminationFlags,
    ...     classify_termination,
    ... )
    >>> def flags(converged, nonfinite, fatal):
    ...     return TerminationFlags(
    ...         converged=jnp.asarray(converged),
    ...         nonfinite=jnp.asarray(nonfinite),
    ...         fatal=jnp.asarray(fatal),
    ...         fatal_result=RESULTS.qp_subproblem_failure,
    ...     )
    >>> adapter = ActiveSetLineSearchResultAdapter()
    >>> done, result = classify_termination(flags(True, False, False), adapter)
    >>> bool(done), bool(result == RESULTS.successful)
    (True, True)
    >>> done, result = classify_termination(flags(False, False, True), adapter)
    >>> bool(done), bool(result == RESULTS.qp_subproblem_failure)
    (True, True)
    >>> done, result = classify_termination(flags(False, False, False), adapter)
    >>> bool(done), bool(result == RESULTS.running)
    (False, True)
    """
    done = flags.nonfinite | flags.converged | flags.fatal
    result = adapter.running
    candidates = (
        (TerminationPriority.fatal, flags.fatal, flags.fatal_result),
        (TerminationPriority.converged, flags.converged, adapter.successful),
        (TerminationPriority.nonfinite, flags.nonfinite, adapter.nonfinite),
    )
    for _, active, candidate in sorted(candidates, key=lambda item: item[0]):
        result = adapter.result_type.where(active, candidate, result)
    return done, result
