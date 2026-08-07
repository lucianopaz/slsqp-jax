"""Local KKT subproblems consumed by SQP step solvers.

The abstract :class:`~slsqp_jax.sqpdax.subproblem.base.SubProblem` surface is
a matrix-free saddle system that every QP / trust-region solver in this
package can drive. Concrete subclasses attach method-specific geometry:

* :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem` —
  equality / inequality / bound QP restricted to a working set.
* :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`
  — primal-dual interior-point Newton system in scaled slack coordinates.
"""

from . import active_set, base, scaled_barrier
from .active_set import ActiveSetSubProblem
from .base import SubProblem
from .scaled_barrier import ScaledBarrierSubProblem

__all__ = [
    "active_set",
    "base",
    "scaled_barrier",
    "ActiveSetSubProblem",
    "SubProblem",
    "ScaledBarrierSubProblem",
]
