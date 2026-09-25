"""Local KKT subproblems consumed by SQP step solvers.

The abstract :class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem` surface is
a matrix-free saddle system that every QP / trust-region solver in this
package can drive. Concrete subclasses attach method-specific geometry:

* :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem` —
  equality / inequality / bound QP restricted to a working set.
* :class:`~slsqp_jax.sqpdax.subproblem.proximal.ProximalActiveSetSubProblem`
  — working-set QP with equalities eliminated through a proximal
  (stabilised-SQP) term.
* :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`
  — primal-dual interior-point Newton system in scaled slack coordinates.
"""

from . import active_set, base, proximal, scaled_barrier, solver
from .active_set import ActiveSetSubProblem
from .base import SubProblem
from .proximal import ProximalActiveSetSubProblem
from .scaled_barrier import ScaledBarrierSubProblem
from .solver import (
    ACTIVE_SET_QP_RESULTS,
    RESULTS,
    ActiveSetQPSolver,
    ActiveSetQPSolverState,
    DogLegSolver,
    DogLegSolverState,
    GradientProjection,
    GradientProjectionState,
    ProjectedCGState,
    ProjectedCGSubProblemSolver,
    ProximalActiveSetQPSolver,
    ProximalActiveSetQPSolverState,
    SteihaugTointCGTangentialStepSolver,
    SteihaugTointCGTangentialStepSolverState,
    SubproblemContext,
    SubProblemSolver,
    SubProblemSolverState,
    SubProblemSolverStateType,
    TrustRegionInteriorPointSolver,
    TrustRegionSolverState,
)

__all__ = [
    "active_set",
    "base",
    "proximal",
    "scaled_barrier",
    "solver",
    "ActiveSetSubProblem",
    "ProximalActiveSetSubProblem",
    "SubProblem",
    "ScaledBarrierSubProblem",
    "RESULTS",
    "SubProblemSolverState",
    "SubProblemSolverStateType",
    "SubProblemSolver",
    "SubproblemContext",
    "ACTIVE_SET_QP_RESULTS",
    "ActiveSetQPSolverState",
    "ActiveSetQPSolver",
    "DogLegSolverState",
    "DogLegSolver",
    "GradientProjectionState",
    "GradientProjection",
    "ProjectedCGState",
    "ProjectedCGSubProblemSolver",
    "ProximalActiveSetQPSolverState",
    "ProximalActiveSetQPSolver",
    "SteihaugTointCGTangentialStepSolverState",
    "SteihaugTointCGTangentialStepSolver",
    "TrustRegionSolverState",
    "TrustRegionInteriorPointSolver",
]
