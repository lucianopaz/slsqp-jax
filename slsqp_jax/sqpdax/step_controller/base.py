"""Abstract step-controller interface and per-step result carrier."""

from abc import abstractmethod
from typing import Generic

from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ..logging import Logger
from ..merit import Merit
from ..primal import PrimalType
from ..subproblem.solver import SubProblemSolverStateType

__all__ = [
    "StepResult",
    "StepController",
    "MeritStepController",
]


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
    """

    logger: Logger = field(static=True, default_factory=Logger.disabled, kw_only=True)

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
