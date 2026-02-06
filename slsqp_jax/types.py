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
