"""Minimiser-specific diagnostic carries.

Each module here holds the run-level counters, maxima and consistency flags
that one concrete :mod:`~slsqp_jax.sqpdax.minimiser` carries through the
traced loop purely for reporting. They never feed back into the iteration;
the minimisers surface them through the per-step diagnostic records and the
final ``stats`` mapping.

- :mod:`~slsqp_jax.sqpdax.minimiser.diagnostics.active_set_linesearch` --
  :class:`ActiveSetLineSearchDiagnostics`: Armijo / fallback acceptance and
  LPEC-A predictor counters of the active-set line-search minimiser.
"""

from . import active_set_linesearch
from .active_set_linesearch import ActiveSetLineSearchDiagnostics

__all__ = [
    "ActiveSetLineSearchDiagnostics",
    "active_set_linesearch",
]
