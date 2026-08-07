"""Step controllers that turn a proposed direction into an accepted iterate.

A :class:`~slsqp_jax.sqpdax.step_controller.base.StepController` is the
outer-loop counterpart of a
:class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolver`: the
solver proposes a direction, the controller decides how (or whether) to
take it and returns a :class:`~slsqp_jax.sqpdax.step_controller.base.StepResult`.
"""

from . import base, line_search, trust_region_radius
from .base import StepController, StepResult
from .line_search import ArmijoLineSearch, LineSearch, LineSearchState
from .trust_region_radius import TrustRegionManager

__all__ = [
    "base",
    "line_search",
    "trust_region_radius",
    "StepResult",
    "StepController",
    "LineSearchState",
    "LineSearch",
    "ArmijoLineSearch",
    "TrustRegionManager",
]
