"""Matrix-free preconditioners and lineax operator adapters.

See :mod:`slsqp_jax.sqpdax.preconditioner.base` for the
:class:`~slsqp_jax.sqpdax.preconditioner.base.Preconditioner` hierarchy,
:mod:`slsqp_jax.sqpdax.preconditioner.strategy` for the per-step
:class:`~slsqp_jax.sqpdax.preconditioner.strategy.PreconditionerStrategy`
policies, and :mod:`slsqp_jax.sqpdax.preconditioner.lineax_compat` for
:class:`~slsqp_jax.sqpdax.preconditioner.lineax_compat.GenericLinearOperator`.
"""

from . import base, lineax_compat, strategy, utils
from .base import (
    DiagonalPreconditioner,
    GenericPreconditioner,
    IdentityPreconditioner,
    MatrixPreconditioner,
    Preconditioner,
)
from .strategy import (
    NoPreconditioner,
    PreconditionerContext,
    PreconditionerStrategy,
    SecantPreconditioner,
    StochasticDiagonalPreconditioner,
)
from .utils import (
    linear_adjoint,
    preconditioner_from_secant,
    stochastic_diagonal,
    woodbury_preconditioner,
)

__all__ = [
    "base",
    "lineax_compat",
    "strategy",
    "utils",
    "Preconditioner",
    "IdentityPreconditioner",
    "MatrixPreconditioner",
    "DiagonalPreconditioner",
    "GenericPreconditioner",
    "PreconditionerContext",
    "PreconditionerStrategy",
    "NoPreconditioner",
    "SecantPreconditioner",
    "StochasticDiagonalPreconditioner",
    "linear_adjoint",
    "preconditioner_from_secant",
    "stochastic_diagonal",
    "woodbury_preconditioner",
]
