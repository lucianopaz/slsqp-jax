"""SLSQP-JAX: Sequential Least Squares Programming in pure JAX.

This package provides a GPU-compatible implementation of the SLSQP optimization
algorithm using JAX and the Optimistix framework. Designed for large-scale
problems (n > 5000) using matrix-free L-BFGS Hessian approximation and
projected conjugate gradient for the QP subproblem.
"""

from slsqp_jax.hessian import (
    LBFGSHistory,
    compute_lagrangian_gradient,
    lbfgs_append,
    lbfgs_hvp,
    lbfgs_init,
)
from slsqp_jax.merit import backtracking_line_search, compute_merit
from slsqp_jax.qp_solver import solve_qp
from slsqp_jax.solver import SLSQP, QPResult, SLSQPState
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
    "SLSQPState",
    "QPResult",
    # Types
    "ConstraintFn",
    "GradFn",
    "JacobianFn",
    "HVPFn",
    "ConstraintHVPFn",
    # QP solver
    "solve_qp",
    # L-BFGS
    "LBFGSHistory",
    "lbfgs_init",
    "lbfgs_hvp",
    "lbfgs_append",
    # Merit function
    "compute_merit",
    "backtracking_line_search",
    # Hessian utilities
    "compute_lagrangian_gradient",
]
