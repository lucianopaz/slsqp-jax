"""Type definitions for SLSQP-JAX.

This module contains type aliases and custom types used throughout the package.
All types use jaxtyping for runtime type checking with beartype.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from jaxtyping import Array, Float

# Type aliases for common array shapes
Scalar = Float[Array, ""]
Vector = Float[Array, " n"]

# Constraint function type: takes parameters and args, returns constraint values
# For equality constraints: c_eq(x) = 0
# For inequality constraints: c_ineq(x) >= 0
ConstraintFn = Callable[[Vector, Any], Float[Array, " m"]]

# Objective function type
ObjectiveFn = Callable[[Vector, Any], Scalar]

# Gradient function type: takes parameters and args, returns gradient of objective
# grad_fn(x, args) -> ∇f(x)
GradFn = Callable[[Vector, Any], Vector]

# Jacobian function type: takes parameters and args, returns Jacobian matrix
# jac_fn(x, args) -> J(x) where J[i, j] = dc_i/dx_j
JacobianFn = Callable[[Vector, Any], Float[Array, "m n"]]

# Hessian-vector product function type for scalar-valued functions (objective)
# hvp_fn(x, v, args) -> H_f(x) @ v
HVPFn = Callable[[Vector, Vector, Any], Vector]

# Hessian-vector product function type for vector-valued functions (constraints)
# For m constraints, returns an (m, n) array where row i is H_{c_i}(x) @ v
# constraint_hvp_fn(x, v, args) -> stack of H_{c_i}(x) @ v
ConstraintHVPFn = Callable[[Vector, Vector, Any], Float[Array, "m n"]]

# Auxiliary output type (generic)
Aux = TypeVar("Aux")


# Result codes for solver termination
class SolverResult:
    """Constants for solver termination status."""

    SUCCESS = 0
    MAX_ITERATIONS = 1
    LINE_SEARCH_FAILED = 2
    QP_FAILED = 3
    INFEASIBLE = 4
