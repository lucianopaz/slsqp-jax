"""Fixtures shared by :mod:`slsqp_jax.sqpdax.preconditioner` tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from slsqp_jax.sqpdax.preconditioner import GenericPreconditioner, MatrixPreconditioner


def make_spd_matrix(n: int = 3) -> Array:
    """Build a small dense SPD matrix for invertible preconditioner tests."""
    a = jnp.arange(1.0, n * n + 1.0).reshape(n, n)
    return a @ a.T + n * jnp.eye(n)


def make_rectangular_matrix(out: int = 3, inn: int = 2) -> Array:
    """Full-column-rank rectangular matrix of shape ``(out, inn)``."""
    return jnp.arange(1.0, out * inn + 1.0).reshape(out, inn)


def make_matrix_preconditioner(n: int = 3) -> MatrixPreconditioner:
    """Square :class:`MatrixPreconditioner` from :func:`make_spd_matrix`."""
    return MatrixPreconditioner(make_spd_matrix(n))


def make_generic_from_matrix(
    matrix: Array,
    *,
    with_invert: bool = False,
) -> GenericPreconditioner:
    """:class:`GenericPreconditioner` whose maps match a dense ``matrix``."""
    in_struct = jax.ShapeDtypeStruct(shape=(matrix.shape[1],), dtype=matrix.dtype)
    out_struct = jax.ShapeDtypeStruct(shape=(matrix.shape[0],), dtype=matrix.dtype)

    def pushforward(x: Array) -> Array:
        return matrix @ x

    def pullback(x: Array) -> Array:
        return matrix.T @ x

    if with_invert:
        return GenericPreconditioner(
            in_struct,
            out_struct,
            pushforward,
            pullback,
            invert=lambda y: jnp.linalg.solve(matrix, y),
            invert_transpose=lambda y: jnp.linalg.solve(matrix.T, y),
        )
    return GenericPreconditioner(in_struct, out_struct, pushforward, pullback)
