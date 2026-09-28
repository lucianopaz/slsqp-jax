"""Matrix-free QP / trust-region subproblem solvers.

Public leaf solvers:

* :class:`~slsqp_jax.sqpdax.subproblem.solver.projected_cg.ProjectedCGSubProblemSolver`
  — null-space projected CG for a fixed active set.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.minres_qlp.MinresQLPSubProblemSolver`
  — preconditioned MINRES-QLP on the full saddle-point operator of a fixed
  active set.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.active_set_loop.ActiveSetQPSolver`
  — primal-dual active-set loop around an inner KKT solver.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.proximal_active_set_loop.ProximalActiveSetQPSolver`
  — stabilised-SQP (proximal equality) variant of the active-set loop.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.gradient_projection.GradientProjection`
  — bound-constrained Cauchy / subspace step (N&W §16.7).
* :class:`~slsqp_jax.sqpdax.subproblem.solver.dogleg.DogLegSolver`
  — Powell dogleg normal (feasibility) step.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.steihaug_toint_cg.SteihaugTointCGTangentialStepSolver`
  — Steihaug–Toint tangential step on a scaled barrier QP.
* :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_region.TrustRegionInteriorPointSolver`
  — composite-step trust-region interior-point orchestrator (N&W §19.5).

Shared infrastructure:

* :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.Projector` /
  :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.ProjectionContext`
  — null-space projector, particular solution and range-space solves for a
  working set (pluggable backend for the null-space solvers:
  :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.SVDProjector` direct,
  :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.CraigProjector`
  matrix-free).
* :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.MultiplierRecovery`
  — strategies returning a subproblem's dual for a primal step
  (KKT-consistent or Hessian-free least squares, optional safeguards).
"""

from . import (
    active_set_loop,
    base,
    dogleg,
    gradient_projection,
    minres_qlp,
    multiplier_recovery,
    projected_cg,
    projector,
    proximal_active_set_loop,
    steihaug_toint_cg,
    trust_region,
    working_set_policy,
)
from .active_set_loop import (
    ACTIVE_SET_QP_RESULTS,
    ActiveSetQPSolver,
    ActiveSetQPSolverState,
    ActiveSetStateType,
    KKTSolverStateType,
)
from .base import (
    KKT_SOLVER_RESULTS,
    RESULTS,
    KKTSolverState,
    SubproblemContext,
    SubProblemSolver,
    SubProblemSolverState,
    SubProblemSolverStateType,
)
from .dogleg import DogLegSolver, DogLegSolverState
from .gradient_projection import GradientProjection, GradientProjectionState
from .minres_qlp import MinresQLPState, MinresQLPSubProblemSolver
from .multiplier_recovery import (
    BarrierSafeguard,
    ClampSafeguard,
    KKTMultiplierRecovery,
    LeastSquaresMultiplierRecovery,
    MultiplierRecovery,
    Safeguard,
)
from .projected_cg import ProjectedCGState, ProjectedCGSubProblemSolver
from .projector import (
    CraigProjectionContext,
    CraigProjector,
    ProjectionContext,
    Projector,
    SVDProjectionContext,
    SVDProjector,
)
from .proximal_active_set_loop import (
    ProximalActiveSetQPSolver,
    ProximalActiveSetQPSolverState,
)
from .steihaug_toint_cg import (
    SteihaugTointCGTangentialStepSolver,
    SteihaugTointCGTangentialStepSolverState,
)
from .trust_region import (
    TrustRegionInteriorPointSolver,
    TrustRegionSolverState,
    TrustRegionStateType,
)
from .working_set_policy import (
    SingleExchangeWorkingSetPolicy,
    ThresholdWorkingSetPolicy,
    WorkingSetPolicy,
    WorkingSetPolicyState,
)

__all__ = [
    "active_set_loop",
    "base",
    "dogleg",
    "gradient_projection",
    "minres_qlp",
    "multiplier_recovery",
    "projected_cg",
    "projector",
    "proximal_active_set_loop",
    "steihaug_toint_cg",
    "trust_region",
    "working_set_policy",
    "RESULTS",
    "KKT_SOLVER_RESULTS",
    "KKTSolverState",
    "SubProblemSolverState",
    "SubProblemSolverStateType",
    "SubProblemSolver",
    "SubproblemContext",
    "ACTIVE_SET_QP_RESULTS",
    "ActiveSetQPSolverState",
    "ActiveSetStateType",
    "KKTSolverStateType",
    "ActiveSetQPSolver",
    "DogLegSolverState",
    "DogLegSolver",
    "GradientProjectionState",
    "GradientProjection",
    "ProjectedCGState",
    "ProjectedCGSubProblemSolver",
    "MinresQLPState",
    "MinresQLPSubProblemSolver",
    "ProjectionContext",
    "Projector",
    "SVDProjectionContext",
    "SVDProjector",
    "CraigProjectionContext",
    "CraigProjector",
    "Safeguard",
    "ClampSafeguard",
    "BarrierSafeguard",
    "MultiplierRecovery",
    "KKTMultiplierRecovery",
    "LeastSquaresMultiplierRecovery",
    "ProximalActiveSetQPSolverState",
    "ProximalActiveSetQPSolver",
    "SteihaugTointCGTangentialStepSolverState",
    "SteihaugTointCGTangentialStepSolver",
    "TrustRegionSolverState",
    "TrustRegionStateType",
    "TrustRegionInteriorPointSolver",
    "WorkingSetPolicyState",
    "WorkingSetPolicy",
    "SingleExchangeWorkingSetPolicy",
    "ThresholdWorkingSetPolicy",
]
