"""QP-subproblem solver subpackage.

Splits the legacy ``slsqp_jax/qp_solver.py`` into:

* :mod:`slsqp_jax.qp.active_set` — the single shared active-set loop
  (add / drop / EXPAND / ping-pong) used by all three solver paths.
* :mod:`slsqp_jax.qp.proximal` — sSQP (augmented-Lagrangian) path.
* :mod:`slsqp_jax.qp.direct` — null-space-projection equality path.
* :mod:`slsqp_jax.qp.inequality` — inequality-only path.
* :mod:`slsqp_jax.qp.bound_fixing` — the box-bound active-set loop
  previously inlined in ``SLSQP._solve_qp_subproblem``.
* :mod:`slsqp_jax.qp.api` — the thin :func:`solve_qp` router.
"""

from slsqp_jax.qp.api import solve_qp
from slsqp_jax.qp.bound_fixing import run_bound_fixing
from slsqp_jax.state import QPSolverResult

__all__ = ["QPSolverResult", "run_bound_fixing", "solve_qp"]
