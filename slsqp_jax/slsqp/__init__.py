"""SLSQP outer-loop subpackage.

Splits the legacy monolithic ``slsqp_jax/solver.py`` into focused modules:

* :mod:`slsqp_jax.slsqp.verbose` — verbose printer callables.
* :mod:`slsqp_jax.slsqp.derivatives` — gradient / Jacobian / HVP
  closure factories used by ``__check_init__``.
* :mod:`slsqp_jax.slsqp.bounds` — NLP-level bound machinery
  (clipping, bound Jacobian, bound-multiplier recovery).
* :mod:`slsqp_jax.slsqp.preconditioner` — L-BFGS / diagonal
  preconditioner factories with Woodbury proximal correction.
* :mod:`slsqp_jax.slsqp.hvp` — Lagrangian HVP factories
  (L-BFGS-frozen / Newton-CG exact).
* :mod:`slsqp_jax.slsqp.termination` — single source of truth for
  the termination classification used by ``step`` and ``terminate``.
* :mod:`slsqp_jax.slsqp.solver` — the :class:`SLSQP` outer-loop
  class itself.
"""

from slsqp_jax.slsqp.solver import SLSQP

__all__ = ["SLSQP"]
