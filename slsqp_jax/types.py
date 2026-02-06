"""Type definitions for SLSQP-JAX.

This module contains type aliases and custom types used throughout the package.
All types use jaxtyping for runtime type checking with beartype.
"""

from typing import Any, Callable, TypeVar

from jaxtyping import Array, Float

# Type aliases for common array shapes
Scalar = Float[Array, ""]
Vector = Float[Array, " n"]
Matrix = Float[Array, "n n"]

# Constraint function type: takes parameters and args, returns constraint values
# For equality constraints: c_eq(x) = 0
# For inequality constraints: c_ineq(x) >= 0
ConstraintFn = Callable[[Float[Array, " n"], Any], Float[Array, " m"]]

# Objective function type
ObjectiveFn = Callable[[Float[Array, " n"], Any], Float[Array, ""]]

# Gradient function type: takes parameters and args, returns gradient of objective
# grad_fn(x, args) -> ∇f(x)
GradFn = Callable[[Float[Array, " n"], Any], Float[Array, " n"]]

# Jacobian function type: takes parameters and args, returns Jacobian matrix
# jac_fn(x, args) -> J(x) where J[i, j] = dc_i/dx_j
JacobianFn = Callable[[Float[Array, " n"], Any], Float[Array, "m n"]]

# Hessian-vector product function type for scalar-valued functions (objective)
# hvp_fn(x, v, args) -> H_f(x) @ v
HVPFn = Callable[[Float[Array, " n"], Float[Array, " n"], Any], Float[Array, " n"]]

# Hessian-vector product function type for vector-valued functions (constraints)
# For m constraints, returns an (m, n) array where row i is H_{c_i}(x) @ v
# constraint_hvp_fn(x, v, args) -> stack of H_{c_i}(x) @ v
ConstraintHVPFn = Callable[
    [Float[Array, " n"], Float[Array, " n"], Any], Float[Array, "m n"]
]

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
