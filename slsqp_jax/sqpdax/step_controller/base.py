"""Abstract step-controller interface and per-step result carrier."""

from abc import abstractmethod
from typing import Generic

from equinox import Module
from jaxtyping import Array, Bool, Scalar

from ..merit import Merit
from ..primal import PrimalType
from ..subproblem.solver import SubProblemSolverStateType

__all__ = [
    "StepResult",
    "StepController",
]


class StepResult(Module, Generic[PrimalType, SubProblemSolverStateType]):
    """Outcome of a single controlled step.

    ``x`` is the iterate to move to (equal to the incoming ``x0`` when the step
    was rejected).  ``accepted`` records whether the controller took the step.
    ``merit_val`` is the merit at ``x`` (for the outer loop's best-merit / progress
    tracking).  ``solver_state`` is the (possibly updated) subproblem-solver state:
    the trust-region controller writes the new radius here; the line search passes
    it through unchanged.

    Attributes
    ----------
    x
        Accepted (or retained) primal iterate.
    accepted
        ``True`` when the controller took the proposed step.
    merit_val
        Merit value at ``x``.
    solver_state
        Refreshed (or threaded) subproblem-solver carry, or ``None``.
    """

    x: PrimalType
    accepted: Bool[Array, ""]
    merit_val: Scalar
    solver_state: SubProblemSolverStateType | None = None


class StepController(Module, Generic[PrimalType, SubProblemSolverStateType]):
    """Executes a step given a proposed direction.

    The analogue of optimistix's per-iteration ``step``: given the current iterate
    ``x0`` and a ``direction`` produced by a ``SubProblemSolver``, decide the actual
    move.  :class:`~slsqp_jax.sqpdax.step_controller.line_search.LineSearch`
    selects the longest step length along ``direction`` that achieves the desired
    merit reduction;
    :class:`~slsqp_jax.sqpdax.step_controller.trust_region_radius.TrustRegionManager`
    accepts or rejects the step from the actual/predicted-reduction ratio and
    updates the radius.

    This is not a "globalization" abstraction -- these classes *execute* the step.
    Swapping a ``StepController`` is not free: each concrete controller is matched
    to the subproblem solvers whose state it consumes (the trust-region manager
    reads the ``predicted_reduction`` / ``radius`` that only a trust-region solver
    produces).  The shared base buys a uniform outer loop and isolated tests, not
    arbitrary interchange.

    Attributes
    ----------
    merit
        Merit used to score candidate iterates.
    """

    merit: Merit

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
