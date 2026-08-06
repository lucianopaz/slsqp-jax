"""Matrix-free preconditioners and lineax operator adapters.

See :mod:`slsqp_jax.sqpdax.preconditioner.base` for the
:class:`~slsqp_jax.sqpdax.preconditioner.base.Preconditioner` hierarchy and
:mod:`slsqp_jax.sqpdax.preconditioner.lineax_compat` for
:class:`~slsqp_jax.sqpdax.preconditioner.lineax_compat.GenericLinearOperator`.
"""

from . import base, lineax_compat, utils
from .base import (
    GenericPreconditioner,
    IdentityPreconditioner,
    MatrixPreconditioner,
    Preconditioner,
)
from .utils import linear_adjoint, preconditioner_from_secant

__all__ = [
    "base",
    "lineax_compat",
    "utils",
    "Preconditioner",
    "IdentityPreconditioner",
    "MatrixPreconditioner",
    "GenericPreconditioner",
    "linear_adjoint",
    "preconditioner_from_secant",
]
