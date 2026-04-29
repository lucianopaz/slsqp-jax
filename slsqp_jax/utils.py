from collections.abc import Callable
from typing import TypeVar, Union

import jax
import jax.numpy as jnp

T = TypeVar("T")


def args_closure(
    fn: Callable[[jax.Array, T], jax.Array], args: T
) -> Callable[[jax.Array], jax.Array]:
    def wrapped(x: jax.Array) -> jax.Array:
        return fn(x, args)

    return wrapped


def to_scalar(x: Union[jax.Array, int, float, bool]) -> jax.Array:
    """Coerce a (possibly non-0-d, size-1) array to a true 0-d scalar.

    This guards the SLSQP internals against user objective functions that
    return e.g. shape ``(1,)`` instead of a true 0-d scalar.  When the input
    already has size 1 (any shape), it is reshaped to ``()``.  Any other
    shape will fail at trace time with a clear shape error from JAX, which
    is the desired behaviour: the objective is contractually scalar-valued.

    Using this at the boundaries (init, line search) prevents a ``(1,)``
    shape from propagating into ``f_val`` / ``lagrangian_val`` and turning
    the boolean ``done`` predicate fed to ``jax.lax.cond`` in
    ``terminate`` into a non-scalar (which raises ``TypeError: Pred must
    be a scalar`` deep inside JAX rather than at the call site).
    """
    return jnp.asarray(x).reshape(())
