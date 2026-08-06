"""Unit tests for :mod:`slsqp_jax.sqpdax.preconditioner.utils`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from lineax import FunctionLinearOperator, symmetric_tag

from slsqp_jax.sqpdax.preconditioner.utils import (
    linear_adjoint,
    preconditioner_from_secant,
)
from slsqp_jax.sqpdax.secant import LBFGS


def _lbfgs_with_pairs(n: int, pairs: list[tuple], *, memory: int = 4) -> LBFGS:
    """Append ``(s, y)`` pairs to a fresh :class:`LBFGS` (local helper)."""
    hist = LBFGS(n=n, memory=memory)
    for s, y in pairs:
        hist = hist.append(jnp.asarray(s), jnp.asarray(y))
    return hist


@pytest.mark.parametrize(
    "matrix_factory",
    [
        lambda: jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        lambda: jnp.array([[2.0, 0.5], [-1.0, 3.0]]),
    ],
    ids=["rectangular", "square-nonsym"],
)
def test_linear_adjoint_matches_matrix_transpose(matrix_factory):
    """Without symmetry tags, adjoint matches ``Mᵀ`` via ``jax.linear_transpose``."""
    matrix = matrix_factory()
    in_struct = jax.ShapeDtypeStruct((matrix.shape[1],), matrix.dtype)
    push = lambda x: matrix @ x  # noqa: E731
    pull = linear_adjoint(push, in_struct)

    y = jnp.linspace(-1.0, 1.0, matrix.shape[0])
    x = jnp.linspace(0.25, 1.0, matrix.shape[1])
    assert jnp.allclose(pull(y), matrix.T @ y)
    assert jnp.allclose(jnp.dot(push(x), y), jnp.dot(x, pull(y)))
    # Same object lineax would build.
    assert jnp.allclose(pull(y), FunctionLinearOperator(push, in_struct).T.mv(y))


def test_linear_adjoint_symmetric_tag_returns_forward():
    """With ``symmetric_tag``, the adjoint is the forward map (lineax short-circuit)."""
    matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    in_struct = jax.ShapeDtypeStruct((2,), matrix.dtype)
    push = lambda x: matrix @ x  # noqa: E731
    pull = linear_adjoint(push, in_struct, tags=symmetric_tag)
    v = jnp.array([0.5, -1.0])
    assert jnp.allclose(pull(v), push(v))


def test_linear_adjoint_of_inverse_is_inverse_transpose():
    """``linear_adjoint(M⁻¹)`` equals ``(Mᵀ)⁻¹`` for nonsingular ``M``."""
    matrix = jnp.array([[2.0, 1.0], [0.5, 3.0]])
    out_struct = jax.ShapeDtypeStruct((2,), matrix.dtype)

    def invert(y: Array) -> Array:
        return jnp.linalg.solve(matrix, y)

    invert_transpose = linear_adjoint(invert, out_struct)
    x = jnp.array([1.0, -2.0])
    assert jnp.allclose(invert_transpose(x), jnp.linalg.solve(matrix.T, x))


@pytest.mark.parametrize("inverse_as_forward", [True, False])
@pytest.mark.parametrize("with_pairs", [False, True], ids=["empty", "with-pairs"])
def test_preconditioner_from_secant_maps(inverse_as_forward: bool, with_pairs: bool):
    """Forward / inverse wrap the secant; symmetric adjoints match the primal maps."""
    n = 4
    if with_pairs:
        secant = _lbfgs_with_pairs(
            n,
            [
                (jnp.array([0.5, -0.2, 0.1, 0.3]), jnp.array([0.4, -0.1, 0.2, 0.25])),
                (
                    jnp.array([-0.1, 0.3, -0.2, 0.1]),
                    jnp.array([-0.05, 0.2, -0.15, 0.08]),
                ),
            ],
            memory=4,
        )
    else:
        secant = LBFGS(n=n, memory=4)

    prec = preconditioner_from_secant(
        secant,
        n=n,
        dtype=secant.diagonal.dtype,
        inverse_as_forward=inverse_as_forward,
    )
    v = jnp.linspace(-1.0, 1.0, n).astype(secant.diagonal.dtype)

    expected_push = secant.inverse_hvp if inverse_as_forward else secant.hvp
    expected_inv = secant.hvp if inverse_as_forward else secant.inverse_hvp
    assert jnp.allclose(prec.pushforward(v), expected_push(v))
    assert jnp.allclose(prec.invert(v), expected_inv(v))

    # Symmetric tag → adjoint equals the forward / inverse map.
    assert jnp.allclose(prec.pullback(v), prec.pushforward(v))
    assert jnp.allclose(prec.invert_transpose(v), prec.invert(v))
    assert jnp.allclose(prec.invert(prec.pushforward(v)), v, atol=1e-5)


def test_preconditioner_from_secant_jittable():
    """Secant-backed maps remain usable under :func:`equinox.filter_jit`."""
    secant = LBFGS(n=3, memory=2)
    prec = preconditioner_from_secant(secant, n=3, dtype=secant.diagonal.dtype)
    v = jnp.asarray([1.0, -0.5, 0.25], dtype=secant.diagonal.dtype)
    assert jnp.allclose(eqx.filter_jit(prec.pushforward)(v), prec.pushforward(v))
    assert jnp.allclose(eqx.filter_jit(prec.pullback)(v), prec.pullback(v))
    assert jnp.allclose(eqx.filter_jit(prec.invert)(v), prec.invert(v))
    assert jnp.allclose(
        eqx.filter_jit(prec.invert_transpose)(v), prec.invert_transpose(v)
    )
