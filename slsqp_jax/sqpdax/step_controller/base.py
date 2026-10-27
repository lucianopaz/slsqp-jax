"""Abstract step-controller interface and per-step result carrier."""

from abc import abstractmethod
from typing import ClassVar, Generic, cast

import jax
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ..logging import Logger
from ..merit import Merit
from ..primal import PrimalType
from ..subproblem.solver import SubProblemSolverStateType
from .correction import CorrectionResult, StepCorrection

__all__ = [
    "ROUNDING_LEVEL_FACTOR",
    "StepResult",
    "StepController",
    "MeritStepController",
    "at_rounding_level",
]

ROUNDING_LEVEL_FACTOR: float = 100.0
"""Multiple of the machine epsilon below which a predicted decrease is noise."""


def at_rounding_level(predicted: Scalar, merit0: Scalar) -> Bool[Array, ""]:
    """Whether a predicted merit decrease is below the merit's rounding level.

    A step controller compares the merit at the trial point with the merit at
    the current point; once the predicted decrease is a small multiple of the
    machine epsilon times ``max(1, |merit0|)``, the difference of the two
    merit values is dominated by round-off and the acceptance test carries no
    information about the model. Such rejections must not be counted as model
    or line-search failures by the outer loop.

    Parameters
    ----------
    predicted
        Predicted decrease of the merit for the full step (first-order slope
        for a line search, model reduction for a trust region).
    merit0
        Merit value at the current iterate.

    Returns
    -------
    Bool[Array, ""]
        ``|predicted| <= ROUNDING_LEVEL_FACTOR * eps * max(1, |merit0|)``;
        ``False`` for a non-finite ``predicted``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.step_controller import at_rounding_level
    >>> bool(at_rounding_level(jnp.asarray(1e-20), jnp.asarray(2.0)))
    True
    >>> bool(at_rounding_level(jnp.asarray(1e-3), jnp.asarray(2.0)))
    False
    """
    predicted = jnp.asarray(predicted)
    eps = jnp.asarray(jnp.finfo(predicted.dtype).eps, predicted.dtype)
    scale = jnp.maximum(1.0, jnp.abs(merit0))
    return jnp.isfinite(predicted) & (
        jnp.abs(predicted) <= ROUNDING_LEVEL_FACTOR * eps * scale
    )


class StepResult(Module, Generic[PrimalType, SubProblemSolverStateType]):
    """Outcome of a single controlled step.

    ``x`` is the iterate to move to (equal to the incoming ``x0`` when the step
    was rejected). ``accepted`` records whether the controller took the step.
    ``merit_val`` is the controller's primary acceptance score at ``x``
    (the merit for :class:`MeritStepController` subclasses; the barrier
    value for a funnel controller). ``solver_state`` is the (possibly
    updated) subproblem-solver state: the trust-region controller writes
    the new radius here; the line search passes it through unchanged.

    Attributes
    ----------
    x
        Accepted (or retained) primal iterate.
    accepted
        ``True`` when the controller took the proposed step.
    merit_val
        Primary acceptance score at ``x``.
    solver_state
        Refreshed (or threaded) subproblem-solver carry, or ``None``.
    accepted_by_fallback
        ``True`` only when a line search accepted through its weaker
        fallback after the primary sufficient-decrease condition failed.
        Non-line-search controllers leave this ``False``.
    rounding_level
        ``True`` when the controller's predicted decrease for the proposed
        step was below the rounding level of the merit (see
        :func:`at_rounding_level`), so ``accepted`` carries no information
        about the quality of the model. Outer loops must not count such a
        rejection towards their failure streaks.
    corrected
        ``True`` when the accepted point is the second-order corrected trial
        ``x0 + d + d̂`` rather than a point on the ray ``x0 + α d`` (see
        :attr:`StepController.correction`).
    """

    x: PrimalType
    accepted: Bool[Array, ""]
    merit_val: Scalar
    solver_state: SubProblemSolverStateType | None = None
    step_size: Scalar = field(default_factory=lambda: jnp.asarray(1.0))
    proposed_step_norm: Scalar = field(default_factory=lambda: jnp.asarray(0.0))
    accepted_by_fallback: Bool[Array, ""] = field(
        default_factory=lambda: jnp.asarray(False)
    )
    rounding_level: Bool[Array, ""] = field(default_factory=lambda: jnp.asarray(False))
    corrected: Bool[Array, ""] = field(default_factory=lambda: jnp.asarray(False))


class StepController(Module, Generic[PrimalType, SubProblemSolverStateType]):
    """Executes a step given a proposed direction.

    The analogue of optimistix's per-iteration ``step``: given the current iterate
    ``x0`` and a ``direction`` produced by a ``SubProblemSolver``, decide the actual
    move.  Merit-driven controllers inherit
    :class:`MeritStepController`; funnel-style controllers score the
    barrier and constraint violation independently.

    This is not a "globalization" abstraction -- these classes *execute* the step.
    Swapping a ``StepController`` is not free: each concrete controller is matched
    to the subproblem solvers whose state it consumes (the trust-region manager
    reads the ``predicted_reduction`` / ``radius`` that only a trust-region solver
    produces).  The shared base buys a uniform outer loop and isolated tests, not
    arbitrary interchange.

    Attributes
    ----------
    logger
        :class:`~slsqp_jax.sqpdax.logging.logger.Logger` for the
        controller's own records (trial steps, acceptance / rejection).
        Minimisers pass their ``minimiser.step_controller`` child; the
        default never emits.
    correction
        Optional bound
        :class:`~slsqp_jax.sqpdax.step_controller.correction.StepCorrection`
        tried on a rejected full step before the controller shortens it.
        Each controller decides when to try it and how to accept the
        corrected point; controllers that have no such stage declare
        :attr:`supports_correction` ``False`` and refuse a correction.
    supports_correction
        Class-level flag: whether :meth:`step` implements a correction stage.
    """

    supports_correction: ClassVar[bool] = False

    logger: Logger = field(static=True, default_factory=Logger.disabled, kw_only=True)
    correction: StepCorrection | None = field(default=None, kw_only=True)

    def __check_init__(self) -> None:
        """Refuse a correction on a controller without a correction stage."""
        if self.correction is not None and not self.supports_correction:
            raise ValueError(
                f"{type(self).__name__} has no second-order correction stage; "
                "pass correction=None"
            )

    def _corrected_point(
        self, x0: PrimalType, direction: PrimalType
    ) -> tuple[PrimalType, CorrectionResult[PrimalType]]:
        """Second-order corrected trial point ``x0 + direction + d̂``.

        Parameters
        ----------
        x0
            Current iterate.
        direction
            Full proposed step.

        Returns
        -------
        x
            Corrected trial point; ``x0 + direction`` when the correction is
            invalid (its step is then zero).
        result
            The :class:`~slsqp_jax.sqpdax.step_controller.correction.CorrectionResult`.

        Raises
        ------
        ValueError
            If the controller carries no :attr:`correction`.
        """
        if self.correction is None:
            raise ValueError(f"{type(self).__name__} carries no correction")
        result = self.correction(x0, direction)
        x = jax.tree.map(lambda a, b, c: a + b + c, x0, direction, result.step)
        return cast(PrimalType, x), result

    @abstractmethod
    def step(
        self,
        x0: PrimalType,
        direction: PrimalType,
        solver_state: SubProblemSolverStateType | None = None,
    ) -> StepResult[PrimalType, SubProblemSolverStateType]:
        """Take (or reject) a step along ``direction`` from ``x0``.

        Parameters
        ----------
        x0
            Current primal iterate.
        direction
            Proposed primal step (full step for line search; composite
            trust-region step for the radius manager).
        solver_state
            Optional subproblem-solver carry. Concrete controllers may
            require a specific subclass (e.g. trust-region radius /
            predicted reduction) or pass the value through unchanged.

        Returns
        -------
        StepResult
            Accepted iterate, acceptance flag, merit at that iterate, and
            the (possibly updated) solver carry.
        """
        ...


class MeritStepController(
    StepController[PrimalType, SubProblemSolverStateType],
    Generic[PrimalType, SubProblemSolverStateType],
):
    """Step controller that scores candidates with a single :class:`Merit`.

    :class:`~slsqp_jax.sqpdax.step_controller.line_search.LineSearch` and
    :class:`~slsqp_jax.sqpdax.step_controller.trust_region_radius.TrustRegionManager`
    inherit this so the merit field stays required for those algorithms
    without forcing every controller to be merit-driven.

    Attributes
    ----------
    merit
        Merit used to score candidate iterates.
    """

    merit: Merit
