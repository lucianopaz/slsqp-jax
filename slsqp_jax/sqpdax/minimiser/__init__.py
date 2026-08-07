"""Outer constrained-optimisation drivers built on subproblem solvers.

A :class:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser` owns both
configuration and running state. Concrete algorithms specialise the
``_``-prefixed hooks (subproblem construction, step controller, feasibility
measure, …) while sharing the init / step / terminate / postprocess
driver. :func:`~slsqp_jax.sqpdax.minimiser.interface.minimise` is the
owned loop; :func:`~slsqp_jax.sqpdax.minimiser.optimistix_compat.as_optimistix_minimiser`
exposes the same solver to ``optimistix.minimise``.
"""

from . import (
    active_set_linesearch,
    base,
    interface,
    optimistix_compat,
    trust_region_interior_point,
    utils,
)
from .active_set_linesearch import ActiveSetLineSearchMinimiser
from .base import AbstractConstrainedMinimiser, CommonMinimiser, OptimisationContext
from .interface import minimise
from .optimistix_compat import OptimistixMinimiser, as_optimistix_minimiser
from .trust_region_interior_point import TrustRegionInteriorPointMinimiser
from .utils import minimiser_option_keys, solver_option_keys

__all__ = [
    "base",
    "interface",
    "optimistix_compat",
    "utils",
    "active_set_linesearch",
    "trust_region_interior_point",
    "OptimisationContext",
    "AbstractConstrainedMinimiser",
    "CommonMinimiser",
    "ActiveSetLineSearchMinimiser",
    "TrustRegionInteriorPointMinimiser",
    "minimise",
    "OptimistixMinimiser",
    "as_optimistix_minimiser",
    "minimiser_option_keys",
    "solver_option_keys",
]
