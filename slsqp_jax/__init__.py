"""SLSQP-JAX: Sequential Least Squares Programming in pure JAX.

This package provides a GPU-compatible implementation of the SLSQP optimization
algorithm using JAX and the Optimistix framework.
"""

from slsqp_jax.solver import SLSQP, SLSQPState, QPResult
from slsqp_jax.types import ConstraintFn
from slsqp_jax.qp_solver import solve_qp, solve_equality_qp
from slsqp_jax.merit import compute_merit, backtracking_line_search
from slsqp_jax.hessian import damped_bfgs_update, compute_lagrangian_gradient

__all__ = [
    # Main solver
    "SLSQP",
    "SLSQPState",
    "QPResult",
    # Types
    "ConstraintFn",
    # QP solver
    "solve_qp",
    "solve_equality_qp",
    # Merit function
    "compute_merit",
    "backtracking_line_search",
    # Hessian updates
    "damped_bfgs_update",
    "compute_lagrangian_gradient",
]
