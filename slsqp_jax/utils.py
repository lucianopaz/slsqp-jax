from typing import Callable, TypeVar

import jax

T = TypeVar("T")


def args_closure(
    fn: Callable[[jax.Array, T], jax.Array], args: T
) -> Callable[[jax.Array], jax.Array]:
    def wrapped(x: jax.Array) -> jax.Array:
        return fn(x, args)

    return wrapped
