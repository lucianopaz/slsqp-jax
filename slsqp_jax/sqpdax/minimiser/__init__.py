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
    diagnostics,
    interface,
    interior_point,
    optimistix_compat,
    proximal_active_set_linesearch,
    trust_funnel_interior_point,
    trust_region_interior_point,
    utils,
)
from .active_set_linesearch import (
    ACTIVE_SET_LINE_SEARCH_RESULTS,
    ActiveSetLineSearchMinimiser,
    ActiveSetLineSearchResultAdapter,
)
from .base import AbstractConstrainedMinimiser, CommonMinimiser, OptimisationContext
from .diagnostics import ActiveSetLineSearchDiagnostics, FunnelDiagnostics
from .interface import minimise
from .interior_point import InteriorPointMinimiser
from .optimistix_compat import OptimistixMinimiser, as_optimistix_minimiser
from .proximal_active_set_linesearch import ProximalActiveSetLineSearchMinimiser
from .trust_funnel_interior_point import (
    TRUST_FUNNEL_INTERIOR_POINT_RESULTS,
    TrustFunnelInteriorPointMinimiser,
    TrustFunnelInteriorPointResultAdapter,
    TrustFunnelTerminationMetrics,
)
from .trust_region_interior_point import (
    TRUST_REGION_INTERIOR_POINT_RESULTS,
    TrustRegionInteriorPointMinimiser,
    TrustRegionInteriorPointResultAdapter,
)
from .utils import minimiser_option_keys, solver_option_keys

__all__ = [
    "base",
    "diagnostics",
    "interface",
    "interior_point",
    "optimistix_compat",
    "utils",
    "active_set_linesearch",
    "proximal_active_set_linesearch",
    "trust_region_interior_point",
    "trust_funnel_interior_point",
    "OptimisationContext",
    "AbstractConstrainedMinimiser",
    "CommonMinimiser",
    "InteriorPointMinimiser",
    "ActiveSetLineSearchMinimiser",
    "ProximalActiveSetLineSearchMinimiser",
    "TrustRegionInteriorPointMinimiser",
    "TrustFunnelInteriorPointMinimiser",
    "minimise",
    "OptimistixMinimiser",
    "as_optimistix_minimiser",
    "ACTIVE_SET_LINE_SEARCH_RESULTS",
    "TRUST_REGION_INTERIOR_POINT_RESULTS",
    "TRUST_FUNNEL_INTERIOR_POINT_RESULTS",
    "ActiveSetLineSearchResultAdapter",
    "TrustRegionInteriorPointResultAdapter",
    "TrustFunnelInteriorPointResultAdapter",
    "TrustFunnelTerminationMetrics",
    "ActiveSetLineSearchDiagnostics",
    "FunnelDiagnostics",
    "minimiser_option_keys",
    "solver_option_keys",
]
