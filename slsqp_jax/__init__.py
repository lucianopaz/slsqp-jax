"""SLSQP-JAX: Sequential Least Squares Programming in pure JAX.

This package provides a GPU-compatible implementation of the SLSQP optimization
algorithm using JAX and the Optimistix framework. Designed for large-scale
problems (n > 5000) using matrix-free L-BFGS Hessian approximation and
projected conjugate gradient for the QP subproblem.
"""

from slsqp_jax.compat import (
    ParsedConstraints,
    minimize_like_scipy,
    parse_constraints,
)
from slsqp_jax.hessian import (
    LBFGSHistory,
    compute_lagrangian_gradient,
    compute_partial_lagrangian_gradient,
    estimate_hessian_diagonal,
    lbfgs_append,
    lbfgs_compute_diagonal,
    lbfgs_curvature_diagnostics,
    lbfgs_estimate_condition,
    lbfgs_hvp,
    lbfgs_identity_reset,
    lbfgs_init,
    lbfgs_inverse_hvp,
    lbfgs_reset,
    lbfgs_should_skip,
    lbfgs_soft_reset,
)
from slsqp_jax.inner_solver import (
    AbstractInnerSolver,
    HRInexactSTCG,
    InnerSolveResult,
    MinresQLPSolver,
    ProjectedCGCholesky,
    ProjectedCGCraig,
    ProjectionContext,
)
from slsqp_jax.lpeca import (
    LPECAResult,
    compute_lpeca_active_set,
    compute_rho_bar,
    identify_active_set_lpeca,
)
from slsqp_jax.merit import (
    LineSearchResult,
    backtracking_line_search,
    compute_merit,
    update_penalty_parameter,
)
from slsqp_jax.qp_solver import solve_qp
from slsqp_jax.results import RESULTS, is_successful
from slsqp_jax.solver import (
    SLSQP,
    STAGNATION_MESSAGE,
    QPResult,
    SLSQPDiagnostics,
    SLSQPState,
    get_diagnostics,
)
from slsqp_jax.types import (
    ConstraintFn,
    ConstraintHVPFn,
    GradFn,
    HVPFn,
    JacobianFn,
)

__all__ = [
    # Main solver
    "SLSQP",
    "STAGNATION_MESSAGE",
    "SLSQPState",
    "SLSQPDiagnostics",
    "get_diagnostics",
    "QPResult",
    # Termination codes
    "RESULTS",
    "is_successful",
    # Types
    "ConstraintFn",
    "GradFn",
    "JacobianFn",
    "HVPFn",
    "ConstraintHVPFn",
    # SciPy compatibility
    "ParsedConstraints",
    "parse_constraints",
    "minimize_like_scipy",
    # Inner solver strategies
    "AbstractInnerSolver",
    "HRInexactSTCG",
    "InnerSolveResult",
    "MinresQLPSolver",
    "ProjectedCGCholesky",
    "ProjectedCGCraig",
    "ProjectionContext",
    # QP solver
    "solve_qp",
    # L-BFGS
    "LBFGSHistory",
    "lbfgs_init",
    "lbfgs_hvp",
    "lbfgs_inverse_hvp",
    "lbfgs_append",
    "lbfgs_compute_diagonal",
    "lbfgs_estimate_condition",
    "lbfgs_reset",
    "lbfgs_should_skip",
    "lbfgs_curvature_diagnostics",
    "lbfgs_soft_reset",
    "lbfgs_identity_reset",
    "estimate_hessian_diagonal",
    # Merit function
    "LineSearchResult",
    "compute_merit",
    "backtracking_line_search",
    "update_penalty_parameter",
    # Hessian utilities
    "compute_lagrangian_gradient",
    "compute_partial_lagrangian_gradient",
    # LPEC-A active set identification
    "compute_rho_bar",
    "identify_active_set_lpeca",
    "compute_lpeca_active_set",
    "LPECAResult",
]
