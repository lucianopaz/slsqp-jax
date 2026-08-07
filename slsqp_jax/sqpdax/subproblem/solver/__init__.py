"""Matrix-free QP / trust-region subproblem solvers.

Public leaf solvers:

* :class:`~slsqp_jax.sqpdax.subproblem.solver.projected_cg.ProjectedCGSubProblemSolver`
  — null-space projected CG for a fixed active set.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolver`
  — primal-dual active-set loop around an inner KKT solver.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.gradient_projection.GradientProjection`
  — bound-constrained Cauchy / subspace step (N&W §16.7).
* :class:`~slsqp_jax.sqpdax.subproblem.solver.dogleg.DogLegSolver`
  — Powell dogleg normal (feasibility) step.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.steihaug_toint_cg.SteihaugTointCGTangentialStepSolver`
  — Steihaug–Toint tangential step on a scaled barrier QP.
"""

from . import (
    active_set_loop,
    base,
    dogleg,
    gradient_projection,
    projected_cg,
    steihaug_toint_cg,
)
from .active_set_loop import ActiveSetQPSolver, ActiveSetQPSolverState
from .base import (
    RESULTS,
    SubproblemContext,
    SubProblemSolver,
    SubProblemSolverState,
    SubProblemSolverStateType,
)
from .dogleg import DogLegSolver, DogLegSolverState
from .gradient_projection import GradientProjection, GradientProjectionState
from .projected_cg import ProjectedCGState, ProjectedCGSubProblemSolver
from .steihaug_toint_cg import (
    SteihaugTointCGTangentialStepSolver,
    SteihaugTointCGTangentialStepSolverState,
)

__all__ = [
    "active_set_loop",
    "base",
    "dogleg",
    "gradient_projection",
    "projected_cg",
    "steihaug_toint_cg",
    "RESULTS",
    "SubProblemSolverState",
    "SubProblemSolverStateType",
    "SubProblemSolver",
    "SubproblemContext",
    "ActiveSetQPSolverState",
    "ActiveSetQPSolver",
    "DogLegSolverState",
    "DogLegSolver",
    "GradientProjectionState",
    "GradientProjection",
    "ProjectedCGState",
    "ProjectedCGSubProblemSolver",
    "SteihaugTointCGTangentialStepSolverState",
    "SteihaugTointCGTangentialStepSolver",
]
