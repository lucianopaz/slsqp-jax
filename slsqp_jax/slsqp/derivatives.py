"""Gradient / Jacobian / HVP closure factories.

Centralises the dispatch between user-supplied derivative callables
and the AD fallbacks so the SLSQP class can remain agnostic to which
side provided each derivative.

Two helpers are public:

* :func:`build_jacobian_impl` — returns a closure ``(y, args) -> J``
  for a single constraint slot, parameterised by the user-supplied
  ``user_jac`` (or ``None``) and the user-supplied constraint function.
* :func:`build_hvp_contrib_impl` — returns a closure
  ``(y, v, args, multipliers) -> Σ μ_i H_{c_i} v`` for a single
  constraint slot, parameterised analogously.

The single-constraint-slot helpers eliminate the EQ/INEQ duplication
in the legacy ``__check_init__``: each side calls the same factory
with its own slot's user-supplied callables and ``m_constraints``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from slsqp_jax.utils import args_closure


def build_grad_impl(user_grad_fn: Callable | None) -> Callable:
    """Closure ``(fn, y, args) -> ∇f(y)`` dispatching to user / AD."""
    if user_grad_fn is not None:
        ufn = user_grad_fn

        def grad_impl(fn, y, args):
            return ufn(y, args)

        return grad_impl

    def grad_impl(fn, y, args):
        return jax.grad(lambda x: fn(x, args)[0])(y)

    return grad_impl


def build_jacobian_impl(
    *,
    user_jac: Callable | None,
    constraint_fn: Callable | None,
    n_constraints: int,
) -> Callable:
    """Closure ``(y, args) -> J(y)`` for one constraint slot.

    Returns a zero-Jacobian closure when no constraint function is
    supplied or the slot is empty (``n_constraints == 0``).
    """
    if constraint_fn is not None and n_constraints > 0:
        if user_jac is not None:
            ujac = user_jac

            def jac_impl(y, args):
                return ujac(y, args)

            return jac_impl

        cfn = constraint_fn

        def jac_impl(y, args):
            return jax.jacrev(args_closure(cfn, args))(y)

        return jac_impl

    def jac_impl(y, _args):
        return jnp.zeros((n_constraints, y.shape[0]))

    return jac_impl


def build_hvp_contrib_impl(
    *,
    user_hvp: Callable | None,
    constraint_fn: Callable | None,
    n_constraints: int,
) -> Callable:
    """Closure ``(y, v, args, μ) -> Σ μ_i H_{c_i}(y) v`` for one slot.

    Mirrors :func:`build_jacobian_impl`: dispatches between
    user-supplied ``user_hvp``, AD via ``jvp(grad(weighted))``, or a
    zero-vector fallback.
    """
    if constraint_fn is not None and n_constraints > 0:
        if user_hvp is not None:
            uhvp = user_hvp

            def hvp_contrib(y, v, args, multipliers):
                return multipliers @ uhvp(y, v, args)

            return hvp_contrib

        cfn = constraint_fn

        def hvp_contrib(y, v, args, multipliers):
            def weighted(x):
                return jnp.dot(multipliers, cfn(x, args))

            _, contrib = jax.jvp(jax.grad(weighted), (y,), (v,))
            return contrib

        return hvp_contrib

    def hvp_contrib(_y, v, _args, _multipliers):
        return jnp.zeros_like(v)

    return hvp_contrib


def build_obj_hvp_impl(
    *,
    user_obj_hvp: Callable | None,
    use_exact_hvp_in_qp: bool,
) -> Callable | None:
    """Optional ``(fn, y, v, args) -> H_f v`` closure.

    Returns ``None`` when no exact HVP source is available *and*
    Newton-CG is disabled, mirroring the legacy
    ``self._obj_hvp_impl is None`` sentinel used in ``step()`` to
    decide whether to probe the Hessian for the L-BFGS secant pair.
    """
    if user_obj_hvp is not None:
        uhvp = user_obj_hvp

        def obj_hvp_impl(_fn, y, v, args):
            return uhvp(y, v, args)

        return obj_hvp_impl

    if use_exact_hvp_in_qp:

        def obj_hvp_impl(fn, y, v, args):
            _, hvp_val = jax.jvp(jax.grad(lambda x: fn(x, args)[0]), (y,), (v,))
            return hvp_val

        return obj_hvp_impl

    return None


__all__ = [
    "build_grad_impl",
    "build_hvp_contrib_impl",
    "build_jacobian_impl",
    "build_obj_hvp_impl",
]
