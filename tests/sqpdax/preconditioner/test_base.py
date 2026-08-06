"""Unit tests for :mod:`slsqp_jax.sqpdax.preconditioner.base`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from jax import Array
from lineax._operator import IdentityLinearOperator, MatrixLinearOperator

from slsqp_jax.sqpdax.preconditioner import (
    GenericPreconditioner,
    IdentityPreconditioner,
    MatrixPreconditioner,
    Preconditioner,
)
from slsqp_jax.sqpdax.preconditioner.base import (
    generic_invert,
    generic_invert_transpose,
)
from slsqp_jax.sqpdax.preconditioner.lineax_compat import GenericLinearOperator

from .conftest import (
    make_generic_from_matrix,
    make_matrix_preconditioner,
    make_rectangular_matrix,
    make_spd_matrix,
)


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("identity", IdentityPreconditioner),
        ("matrix", MatrixPreconditioner),
        ("generic", GenericPreconditioner),
    ],
)
def test_preconditioner_kind_registry(kind: str, cls: type):
    """Concrete preconditioners register on the shared kind registry."""
    assert Preconditioner._registry[kind] is cls


class _ForwardOnlyPreconditioner(Preconditioner):
    """Minimal concrete subclass that relies on base invert / operators."""

    matrix: Array

    def __init__(self, matrix: Array):
        self.matrix = matrix
        self.input_structure = jax.ShapeDtypeStruct(
            shape=(matrix.shape[1],), dtype=matrix.dtype
        )
        self.output_structure = jax.ShapeDtypeStruct(
            shape=(matrix.shape[0],), dtype=matrix.dtype
        )

    def pushforward(self, x: Array) -> Array:
        return self.matrix @ x

    def pullback(self, x: Array) -> Array:
        return self.matrix.T @ x


def test_base_preconditioner_default_invert_and_operators():
    """Base ``invert`` / lineax wrappers work when subclasses only define ``M``."""
    matrix = make_spd_matrix(3)
    prec = _ForwardOnlyPreconditioner(matrix)
    x = jnp.array([0.4, -0.2, 1.0])
    y = matrix @ x

    assert isinstance(prec.as_linear_operator(), GenericLinearOperator)
    assert jnp.allclose(prec.as_linear_operator().as_matrix(), matrix, atol=1e-5)
    assert jnp.allclose(prec.invert(y), x, atol=1e-5)
    assert jnp.allclose(
        prec.invert_transpose(x), jnp.linalg.solve(matrix.T, x), atol=1e-5
    )
    inv_op = prec.as_inverse_linear_operator()
    assert jnp.allclose(inv_op.mv(y), x, atol=1e-5)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_identity_preconditioner_maps(n: int):
    """Identity maps leave vectors unchanged and invert is the identity."""
    # Build at test time so the dtype matches JAX's current default float
    # (float64 when another suite has enabled x64).
    proto = jnp.arange(n) + 1.0
    prec = IdentityPreconditioner(proto)
    x = jnp.linspace(-1.0, 1.0, n)

    assert prec.input_structure.shape == (n,)
    assert prec.output_structure.shape == (n,)
    assert prec.input_structure.dtype == x.dtype
    assert jnp.allclose(prec.pushforward(x), x)
    assert jnp.allclose(prec.pullback(x), x)
    assert jnp.allclose(prec.invert(x), x)
    assert jnp.allclose(prec.invert_transpose(x), x)
    assert isinstance(prec.as_linear_operator(), IdentityLinearOperator)
    assert isinstance(prec.as_inverse_linear_operator(), IdentityLinearOperator)
    assert jnp.allclose(prec.as_linear_operator().mv(x), x)
    assert jnp.allclose(prec.as_inverse_linear_operator().mv(x), x)


@pytest.mark.parametrize(
    "matrix_factory",
    [
        lambda: make_spd_matrix(2),
        lambda: make_spd_matrix(3),
        lambda: make_rectangular_matrix(4, 2),
    ],
    ids=["spd-2", "spd-3", "rect-4x2"],
)
def test_matrix_preconditioner_forward_and_inverse(matrix_factory):
    """Dense ``M`` matches matvecs; invert recovers least-squares solutions."""
    matrix = matrix_factory()
    prec = MatrixPreconditioner(matrix)
    x_in = jnp.linspace(0.5, 1.5, matrix.shape[1])
    x_out = jnp.linspace(-1.0, 2.0, matrix.shape[0])

    assert prec.input_structure.shape == (matrix.shape[1],)
    assert prec.output_structure.shape == (matrix.shape[0],)
    assert jnp.allclose(prec.pushforward(x_in), matrix @ x_in)
    assert jnp.allclose(prec.pullback(x_out), matrix.T @ x_out)
    assert jnp.allclose(prec.invert(x_out), jnp.linalg.lstsq(matrix, x_out)[0])
    assert jnp.allclose(
        prec.invert_transpose(x_in), jnp.linalg.lstsq(matrix.T, x_in)[0]
    )
    assert isinstance(prec.as_linear_operator(), MatrixLinearOperator)
    assert jnp.allclose(prec.as_linear_operator().as_matrix(), matrix)


def test_matrix_preconditioner_square_roundtrip():
    """For SPD ``M``, ``invert(pushforward(x))`` recovers ``x``."""
    matrix = make_spd_matrix(3)
    prec = MatrixPreconditioner(matrix)
    x = jnp.array([0.2, -0.5, 1.1])
    assert jnp.allclose(prec.invert(prec.pushforward(x)), x, atol=1e-5)
    assert jnp.allclose(prec.invert_transpose(prec.pullback(x)), x, atol=1e-5)


@pytest.mark.parametrize(
    "with_invert", [False, True], ids=["lineax-invert", "direct-invert"]
)
def test_generic_preconditioner_matches_matrix(with_invert: bool):
    """Generic maps match a dense matrix; optional direct invert is used."""
    matrix = make_spd_matrix(3)
    prec = make_generic_from_matrix(matrix, with_invert=with_invert)
    x = jnp.array([1.0, -0.5, 0.25])
    y = matrix @ x

    assert jnp.allclose(prec.pushforward(x), y)
    assert jnp.allclose(prec.pullback(y), matrix.T @ y)
    assert jnp.allclose(prec.invert(y), x, atol=1e-5)
    assert jnp.allclose(
        prec.invert_transpose(x), jnp.linalg.solve(matrix.T, x), atol=1e-5
    )
    assert isinstance(prec.as_linear_operator(), GenericLinearOperator)
    assert jnp.allclose(prec.as_linear_operator().as_matrix(), matrix, atol=1e-5)


def test_generic_invert_helpers_match_solve():
    """Module-level :func:`generic_invert` helpers solve the forward operator."""
    prec = make_matrix_preconditioner(3)
    # Use a GenericPreconditioner without direct invert so helpers are exercised
    # through the base path (MatrixPreconditioner overrides invert).
    generic = make_generic_from_matrix(prec.matrix, with_invert=False)
    x = jnp.array([0.3, -1.0, 2.0])
    y = prec.matrix @ x

    assert jnp.allclose(generic_invert(generic, y), x, atol=1e-5)
    assert jnp.allclose(
        generic_invert_transpose(generic, x),
        jnp.linalg.solve(prec.matrix.T, x),
        atol=1e-5,
    )


def test_as_inverse_linear_operator_applies_inverse():
    """``as_inverse_linear_operator().mv`` applies ``M⁻¹``."""
    matrix = make_spd_matrix(3)
    prec = MatrixPreconditioner(matrix)
    y = jnp.array([1.0, 2.0, 3.0])
    inv_op = prec.as_inverse_linear_operator()
    assert jnp.allclose(inv_op.mv(y), prec.invert(y), atol=1e-5)
    assert jnp.allclose(inv_op.T.mv(y), prec.invert_transpose(y), atol=1e-5)


@pytest.mark.parametrize(
    "prec_factory",
    [
        lambda: IdentityPreconditioner(jnp.zeros(3)),
        lambda: make_matrix_preconditioner(3),
        lambda: make_generic_from_matrix(make_spd_matrix(3), with_invert=True),
    ],
    ids=["identity", "matrix", "generic"],
)
def test_preconditioner_jittable(prec_factory):
    """Pushforward / invert remain usable under :func:`equinox.filter_jit`."""
    prec = prec_factory()
    x_in = jnp.linspace(0.1, 1.0, prec.input_structure.shape[0])
    x_out = jnp.linspace(-0.5, 0.5, prec.output_structure.shape[0])
    assert jnp.allclose(eqx.filter_jit(prec.pushforward)(x_in), prec.pushforward(x_in))
    assert jnp.allclose(
        eqx.filter_jit(prec.invert)(x_out), prec.invert(x_out), atol=1e-5
    )


def test_lineax_solve_with_inverse_preconditioner():
    """CG solve of an SPD system accepts ``as_inverse_linear_operator``."""
    a = make_spd_matrix(3)
    b = jnp.array([1.0, 0.5, -0.25])
    prec = IdentityPreconditioner(b)
    sol = lx.linear_solve(
        lx.MatrixLinearOperator(
            a, tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
        ),
        b,
        solver=lx.CG(rtol=1e-8, atol=1e-8),
        options={"preconditioner": prec.as_inverse_linear_operator()},
    ).value
    assert jnp.allclose(sol, jnp.linalg.solve(a, b), atol=1e-5)

    # Jacobi ``M = diag(A)``: inverse operator matches the explicit diagonal inverse.
    jac = MatrixPreconditioner(jnp.diag(jnp.diag(a)))
    d_inv = jnp.diag(1.0 / jnp.diag(a))
    assert jnp.allclose(jac.as_inverse_linear_operator().mv(b), d_inv @ b)
