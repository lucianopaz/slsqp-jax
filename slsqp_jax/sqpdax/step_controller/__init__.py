"""Step controllers that turn a proposed direction into an accepted iterate.

A :class:`~slsqp_jax.sqpdax.step_controller.base.StepController` is the
outer-loop counterpart of a
:class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolver`: the
solver proposes a direction, the controller decides how (or whether) to
take it and returns a :class:`~slsqp_jax.sqpdax.step_controller.base.StepResult`.
A :class:`~slsqp_jax.sqpdax.step_controller.correction.StepCorrection`
supplies the second-order correction a controller may try before it shortens
a rejected full step.
"""

from . import base, correction, line_search, trust_funnel, trust_region_radius
from .base import (
    ROUNDING_LEVEL_FACTOR,
    MeritStepController,
    StepController,
    StepResult,
    at_rounding_level,
)
from .correction import CorrectionResult, LeastNormCorrection, StepCorrection
from .line_search import ArmijoLineSearch, LineSearch, LineSearchState
from .trust_funnel import TrustFunnelManager
from .trust_region_radius import TrustRegionManager

__all__ = [
    "base",
    "correction",
    "line_search",
    "trust_funnel",
    "trust_region_radius",
    "ROUNDING_LEVEL_FACTOR",
    "at_rounding_level",
    "StepResult",
    "StepController",
    "MeritStepController",
    "CorrectionResult",
    "StepCorrection",
    "LeastNormCorrection",
    "LineSearchState",
    "LineSearch",
    "ArmijoLineSearch",
    "TrustFunnelManager",
    "TrustRegionManager",
]
