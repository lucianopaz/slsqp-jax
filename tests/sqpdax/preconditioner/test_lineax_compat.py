"""Unit tests for :mod:`slsqp_jax.sqpdax.preconditioner.lineax_compat`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from jax import Array
from lineax import (
    conj,
    diagonal,
    diagonal_tag,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    lower_triangular_tag,
    materialise,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
    upper_triangular_tag,
)
from lineax._operator import MatrixLinearOperator

from slsqp_jax.sqpdax.preconditioner.lineax_compat import GenericLinearOperator

from .conftest import make_rectangular_matrix, make_spd_matrix


def _operator_from_matrix(
    matrix: Array, tags: object | frozenset = frozenset()
) -> GenericLinearOperator:
    in_struct = jax.ShapeDtypeStruct(shape=(matrix.shape[1],), dtype=matrix.dtype)
    out_struct = jax.ShapeDtypeStruct(shape=(matrix.shape[0],), dtype=matrix.dtype)
    return GenericLinearOperator(
        pushforward=lambda x: matrix @ x,
        pullback=lambda y: matrix.T @ y,
        input_structure=in_struct,
        output_structure=out_struct,
        tags=tags,
    )


@pytest.mark.parametrize(
    "matrix_factory",
    [lambda: make_spd_matrix(2), lambda: make_rectangular_matrix(3, 2)],
    ids=["square", "rectangular"],
)
def test_generic_linear_operator_mv_and_matrix(matrix_factory):
    """``mv`` / ``as_matrix`` / structures match the dense matrix."""
    matrix = matrix_factory()
    op = _operator_from_matrix(matrix)
    x = jnp.linspace(0.25, 1.0, matrix.shape[1])
    y = jnp.linspace(-1.0, 0.5, matrix.shape[0])

    assert op.in_structure().shape == (matrix.shape[1],)
    assert op.out_structure().shape == (matrix.shape[0],)
    assert jnp.allclose(op.mv(x), matrix @ x)
    assert jnp.allclose(op.as_matrix(), matrix)
    assert jnp.allclose(op.T.mv(y), matrix.T @ y)
    assert jnp.allclose(op.T.as_matrix(), matrix.T)


def test_generic_linear_operator_transpose_swaps_structures():
    """Transpose swaps in/out structures and pushforward / pullback."""
    matrix = make_rectangular_matrix(3, 2)
    op = _operator_from_matrix(matrix)
    op_t = op.T
    assert op_t.input_structure.shape == matrix.shape[:1]
    assert op_t.output_structure.shape == (matrix.shape[1],)
    x = jnp.ones(matrix.shape[0])
    assert jnp.allclose(op_t.mv(x), matrix.T @ x)


@pytest.mark.parametrize(
    ("tags", "checks"),
    [
        (frozenset(), {"symmetric": False, "diagonal": False}),
        (symmetric_tag, {"symmetric": True, "diagonal": False}),
        (diagonal_tag, {"symmetric": True, "diagonal": True}),
        (positive_semidefinite_tag, {"symmetric": True, "psd": True}),
        (negative_semidefinite_tag, {"symmetric": True, "nsd": True}),
        (tridiagonal_tag, {"tridiagonal": True}),
        (lower_triangular_tag, {"lower": True}),
        (upper_triangular_tag, {"upper": True}),
        (
            frozenset({symmetric_tag, diagonal_tag}),
            {"symmetric": True, "diagonal": True},
        ),
    ],
    ids=[
        "none",
        "symmetric",
        "diagonal",
        "psd",
        "nsd",
        "tridiagonal",
        "lower",
        "upper",
        "symmetric+diagonal",
    ],
)
def test_generic_linear_operator_tag_predicates(tags: object, checks: dict[str, bool]):
    """Singledispatch predicates read property tags off the operator."""
    op = _operator_from_matrix(make_spd_matrix(2), tags=tags)
    assert is_symmetric(op) is checks.get("symmetric", False)
    assert is_diagonal(op) is checks.get("diagonal", False)
    assert is_tridiagonal(op) is checks.get("tridiagonal", False)
    assert is_lower_triangular(op) is checks.get("lower", False)
    assert is_upper_triangular(op) is checks.get("upper", False)
    assert is_positive_semidefinite(op) is checks.get("psd", False)
    assert is_negative_semidefinite(op) is checks.get("nsd", False)


def test_generic_linear_operator_single_tag_wrapped_as_frozenset():
    """A lone non-iterable tag is stored as a one-element frozenset."""
    op = _operator_from_matrix(make_spd_matrix(2), tags=diagonal_tag)
    assert op.tags == frozenset({diagonal_tag})
    assert is_diagonal(op)


def test_linearise_materialise_diagonal():
    """``linearise`` is identity; ``materialise`` / ``diagonal`` match the matrix."""
    matrix = make_spd_matrix(3)
    op = _operator_from_matrix(matrix, tags=symmetric_tag)
    assert linearise(op) is op

    mat_op = materialise(op)
    assert isinstance(mat_op, MatrixLinearOperator)
    assert jnp.allclose(mat_op.as_matrix(), matrix)
    assert jnp.allclose(diagonal(op), jnp.diag(matrix))


def test_conj_preserves_real_action():
    """``conj`` folds conjugation through the maps without changing real ``mv``."""
    matrix = make_spd_matrix(2)
    op = _operator_from_matrix(matrix)
    conj_op = conj(op)
    x = jnp.array([1.0, -2.0])
    assert jnp.allclose(conj_op.mv(x), op.mv(x))


def test_generic_linear_operator_jittable():
    """``mv`` and ``as_matrix`` work under :func:`equinox.filter_jit`."""
    matrix = make_spd_matrix(3)
    op = _operator_from_matrix(matrix)
    x = jnp.arange(3.0)
    assert jnp.allclose(eqx.filter_jit(op.mv)(x), matrix @ x)
    assert jnp.allclose(eqx.filter_jit(op.as_matrix)(), matrix)


def test_auto_linear_solver_uses_materialised_operator():
    """Tagged operator inverts via :class:`~lineax.AutoLinearSolver` materialisation."""
    matrix = make_spd_matrix(3)
    op = _operator_from_matrix(
        matrix, tags=frozenset({symmetric_tag, positive_semidefinite_tag})
    )
    b = jnp.array([1.0, 2.0, 3.0])
    sol = lx.linear_solve(op, b, solver=lx.AutoLinearSolver(well_posed=None)).value
    assert jnp.allclose(sol, jnp.linalg.solve(matrix, b), atol=1e-5)
