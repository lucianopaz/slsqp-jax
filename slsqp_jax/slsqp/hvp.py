"""Lagrangian Hessian-vector product factories.

The QP subproblem needs ``v -> B v`` where ``B`` is some
positive-definite approximation to the Lagrangian Hessian.  Two
strategies are supported:

* **Frozen L-BFGS** (default): the L-BFGS history at the *start* of
  the SLSQP step is treated as constant for the duration of the inner
  CG loop.  ``v -> B_k v`` is implemented via the compact
  representation in :func:`slsqp_jax.hessian.lbfgs_hvp`.
* **Newton-CG** (``QPConfig.use_exact_hvp_in_qp = True``): the exact
  Lagrangian HVP at the current iterate is used directly inside the
  CG loop.  Each CG step costs one forward-over-reverse AD pass; on
  ill-conditioned problems this dramatically accelerates convergence.

The L-BFGS history is still updated regardless of which mode is
active, because (a) it is the canonical source for the preconditioner
and (b) it is the fallback in case a future revision skips an exact
HVP evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from jaxtyping import Array, Float

from slsqp_jax.hessian import LBFGSHistory, lbfgs_hvp
from slsqp_jax.types import Vector


def build_lbfgs_lagrangian_hvp(
    lbfgs_history: LBFGSHistory,
) -> Callable[[Vector], Vector]:
    """Closure over the *frozen* L-BFGS history."""

    def hvp(v: Vector) -> Vector:
        return lbfgs_hvp(lbfgs_history, v)

    return hvp


def build_exact_lagrangian_hvp(
    *,
    fn: Callable,
    y: Vector,
    args: Any,
    multipliers_eq: Float[Array, " m_eq"],
    multipliers_ineq: Float[Array, " m_ineq"],
    obj_hvp_impl: Callable | None,
    eq_hvp_contrib_impl: Callable,
    ineq_hvp_contrib_impl: Callable,
    n_ineq_general: int,
) -> Callable[[Vector], Vector]:
    """Build the exact Lagrangian HVP at the current iterate.

    ``H_L v = H_f v − Σ λ_i H_{c_eq_i} v − Σ μ_j H_{c_ineq_j} v``.
    The dispatch to user-supplied vs AD-computed contribution
    callables is resolved upstream in
    :func:`slsqp_jax.slsqp.derivatives.make_derivative_closures`.
    """
    obj_hvp = cast(Callable, obj_hvp_impl)
    eq_hvp = eq_hvp_contrib_impl
    ineq_hvp = ineq_hvp_contrib_impl

    def lagrangian_hvp(v: Vector) -> Vector:
        obj_val = obj_hvp(fn, y, v, args)
        eq_val = eq_hvp(y, v, args, multipliers_eq)
        ineq_val = ineq_hvp(y, v, args, multipliers_ineq[:n_ineq_general])
        return obj_val - eq_val - ineq_val

    return lagrangian_hvp


__all__ = [
    "build_exact_lagrangian_hvp",
    "build_lbfgs_lagrangian_hvp",
]
