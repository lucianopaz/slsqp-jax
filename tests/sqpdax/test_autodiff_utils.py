"""Unit tests for :mod:`slsqp_jax.sqpdax.autodiff_utils`."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float

from slsqp_jax.sqpdax.autodiff_utils import (
    _contract_jac_tangent,
    autodiff_wrapper,
    fn_proxy_autodiff,
)
from slsqp_jax.sqpdax.primal import Primal


def _scalar_fn(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return a * b * jnp.sum(x**3)


def _scalar_jac(x: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0) -> Array:
    return 3 * a * b * x**2


def _scalar_hvp(
    x: Array, v: Array, a: Array = jnp.asarray(1.0), *, b: float = 1.0
) -> Array:
    return 6 * a * b * x * v


def _vec_fn(x: Array) -> Array:
    return jnp.stack([x[0] ** 2 + x[1], x[0] * x[1] ** 2])


def _vec_jac(x: Array) -> Array:
    return jnp.array(
        [
            [2 * x[0], 1.0],
            [x[1] ** 2, 2 * x[0] * x[1]],
        ]
    )


def _vec_hvp(x: Array, v: Array) -> Array:
    return jnp.array(
        [
            [2 * v[0], 0.0],
            [2 * x[1] * v[1], 2 * v[0] * x[1] + 2 * x[0] * v[1]],
        ]
    )


def _primal_fn(p: Primal) -> Array:
    return jnp.sum(p.x**3)


def _primal_jac(p: Primal) -> Primal:
    return Primal(x=3 * p.x**2)


def _primal_hvp(p: Primal, t: Primal) -> Primal:
    return Primal(x=6 * p.x * t.x)


@pytest.mark.parametrize(
    ("fn", "fn_jac", "fn_hvp", "x", "v", "call_args", "call_kwargs"),
    [
        (
            _scalar_fn,
            _scalar_jac,
            _scalar_hvp,
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([0.1, -0.2, 0.3]),
            (jnp.asarray(2.0),),
            {"b": 3.0},
        ),
        (
            _vec_fn,
            _vec_jac,
            _vec_hvp,
            jnp.array([1.0, 2.0]),
            jnp.array([0.5, -0.3]),
            (),
            {},
        ),
        (
            _primal_fn,
            _primal_jac,
            _primal_hvp,
            Primal(x=jnp.array([1.0, 2.0])),
            Primal(x=jnp.array([0.5, -0.25])),
            (),
            {},
        ),
    ],
)
@pytest.mark.parametrize("with_hvp", [True, False])
def test_fn_proxy_matches_supplied_derivatives(
    fn: Callable,
    fn_jac: Callable,
    fn_hvp: Callable,
    x: Array | Primal,
    v: Array | Primal,
    call_args: tuple,
    call_kwargs: dict,
    with_hvp: bool,
):
    """Primal, JVP, VJP, and optional HVP match the supplied callables."""
    hvp_in = fn_hvp if with_hvp else None
    wrapped, wrapped_jac, hvp_out = fn_proxy_autodiff(fn, fn_jac, hvp_in)
    assert hvp_out is hvp_in

    bound = lambda z: wrapped(z, *call_args, **call_kwargs)  # noqa: E731
    bound_jac = lambda z: wrapped_jac(z, *call_args, **call_kwargs)  # noqa: E731

    assert jnp.allclose(bound(x), fn(x, *call_args, **call_kwargs))

    expected_jac = fn_jac(x, *call_args, **call_kwargs)
    if isinstance(x, Primal):
        assert jnp.allclose(bound_jac(x).x, expected_jac.x)
        got_grad = eqx.filter_grad(bound)(x)
        assert jnp.allclose(got_grad.x, expected_jac.x)
        assert jnp.allclose(jax.jvp(bound, (x,), (v,))[1], jnp.dot(expected_jac.x, v.x))
    else:
        assert jnp.allclose(bound_jac(x), expected_jac)
        assert jnp.allclose(jax.jacfwd(bound)(x), expected_jac)
        assert jnp.allclose(jax.jacrev(bound)(x), expected_jac)
        assert jnp.allclose(
            jax.jvp(bound, (x,), (v,))[1],
            jnp.tensordot(expected_jac, v, axes=1),
        )

    if with_hvp:
        expected_hvp = fn_hvp(x, v, *call_args, **call_kwargs)
        got_from_jac = jax.jvp(bound_jac, (x,), (v,))[1]
        if isinstance(x, Primal):
            assert jnp.allclose(got_from_jac.x, expected_hvp.x)
            got = jax.jvp(eqx.filter_grad(bound), (x,), (v,))[1]
            assert jnp.allclose(got.x, expected_hvp.x)
        else:
            assert jnp.allclose(got_from_jac, expected_hvp)
            diff = (
                eqx.filter_grad(bound)
                if jnp.shape(bound(x)) == ()
                else jax.jacrev(bound)
            )
            got = jax.jvp(diff, (x,), (v,))[1]
            assert jnp.allclose(got, expected_hvp)
    else:
        assert wrapped_jac is fn_jac


def test_fn_proxy_ignores_arg_tangents():
    """Custom rules differentiate only ``x``; ``*args`` tangents are zero."""
    wrapped, wrapped_jac, _ = fn_proxy_autodiff(_scalar_fn, _scalar_jac, _scalar_hvp)
    x = jnp.array([1.0, 2.0])
    a = jnp.asarray(2.0)

    # Symbolic-zero tangent on ``x`` (covers ``tx is None`` / ``tangent is None``).
    assert jnp.allclose(jax.grad(lambda a: wrapped(x, a))(a), 0.0)
    assert jnp.allclose(jax.grad(lambda a: jnp.sum(wrapped_jac(x, a)))(a), 0.0)


def test_fn_proxy_jittable():
    """Wrapped function and its gradient remain JIT-compatible."""
    wrapped, wrapped_jac, hvp = fn_proxy_autodiff(_scalar_fn, _scalar_jac, _scalar_hvp)
    x = jnp.array([1.0, 2.0])
    a = jnp.asarray(2.0)

    assert hvp is _scalar_hvp
    assert jnp.allclose(jax.jit(wrapped)(x, a, b=3.0), _scalar_fn(x, a, b=3.0))
    assert jnp.allclose(jax.jit(wrapped_jac)(x, a, b=3.0), _scalar_jac(x, a, b=3.0))
    assert jnp.allclose(
        eqx.filter_jit(eqx.filter_grad(wrapped))(x, a, b=3.0),
        _scalar_jac(x, a, b=3.0),
    )


def test_fn_proxy_primal_static_fields_preserved():
    """Static module fields survive custom JVP / filter_grad."""

    class TaggedPrimal(eqx.Module):
        x: Float[Array, " n"]
        name: str = eqx.field(static=True)

    def fn(p: TaggedPrimal) -> Array:
        return jnp.sum(p.x**2)

    def fn_jac(p: TaggedPrimal) -> TaggedPrimal:
        return TaggedPrimal(x=2 * p.x, name=p.name)

    wrapped, wrapped_jac, hvp = fn_proxy_autodiff(fn, fn_jac)
    assert hvp is None
    assert wrapped_jac is fn_jac
    p = TaggedPrimal(x=jnp.array([1.0, -2.0]), name="decision")
    grad = eqx.filter_grad(wrapped)(p)
    assert grad.name == "decision"
    assert jnp.allclose(grad.x, 2 * p.x)


def test_contract_jac_tangent_rejects_empty_jac():
    """Empty Jacobian pytrees raise a clear error."""
    with pytest.raises(ValueError, match="at least one array leaf"):
        _contract_jac_tangent((), jnp.zeros(2), jnp.zeros(2))


@pytest.mark.parametrize(
    ("fn", "expected_jac", "x", "v", "expected_hvp"),
    [
        (
            _scalar_fn,
            lambda x: _scalar_jac(x),
            jnp.array([1.0, 2.0]),
            jnp.array([0.5, -0.25]),
            lambda x, v: _scalar_hvp(x, v),
        ),
        (
            _vec_fn,
            _vec_jac,
            jnp.array([1.0, 2.0]),
            jnp.array([0.5, -0.3]),
            _vec_hvp,
        ),
        (
            _primal_fn,
            lambda p: _primal_jac(p).x,
            Primal(x=jnp.array([1.0, 2.0])),
            Primal(x=jnp.array([0.5, -0.25])),
            lambda p, t: _primal_hvp(p, t).x,
        ),
    ],
)
@pytest.mark.parametrize(
    ("force_hvp", "hvp_placeholder"),
    [
        (False, None),
        (True, None),
        (False, _scalar_hvp),  # non-None placeholder also forces a JAX HVP
    ],
    ids=["default-no-hvp", "force-hvp", "placeholder-forces-hvp"],
)
def test_autodiff_wrapper_jax_mode(
    fn: Callable,
    expected_jac: Callable,
    x: Array | Primal,
    v: Array | Primal,
    expected_hvp: Callable,
    force_hvp: bool,
    hvp_placeholder: Callable | None,
):
    """``autodiff_mode='jax'`` builds jac; HVP only when forced or placeholder."""
    # The placeholder path is only meaningful for array-valued callables.
    if hvp_placeholder is not None and isinstance(x, Primal):
        pytest.skip("placeholder HVP path covered on array callables")

    wrapped_fn, grad, hvp = autodiff_wrapper(
        fn,
        hvp=hvp_placeholder,
        autodiff_mode="jax",
        force_hvp_in_jax_mode=force_hvp,
    )
    assert wrapped_fn is fn

    got_grad = grad(x)
    exp_jac = expected_jac(x)
    if isinstance(x, Primal):
        assert jnp.allclose(got_grad.x, exp_jac)
    else:
        assert jnp.allclose(got_grad, exp_jac)

    expect_hvp = force_hvp or hvp_placeholder is not None
    if not expect_hvp:
        assert hvp is None
        return

    assert hvp is not None
    got_hvp = hvp(x, v)
    exp_hvp = expected_hvp(x, v)
    if isinstance(x, Primal):
        assert jnp.allclose(got_hvp.x, exp_hvp)
    else:
        assert jnp.allclose(got_hvp, exp_hvp)


@pytest.mark.parametrize("with_hvp", [True, False])
def test_autodiff_wrapper_custom_mode(with_hvp: bool):
    """``autodiff_mode='custom'`` delegates to :func:`fn_proxy_autodiff`."""
    hvp_in = _scalar_hvp if with_hvp else None
    fn, grad, hvp = autodiff_wrapper(
        _scalar_fn, _scalar_jac, hvp_in, autodiff_mode="custom"
    )
    x = jnp.array([1.0, 2.0])
    assert hvp is hvp_in
    assert jnp.allclose(fn(x), _scalar_fn(x))
    assert jnp.allclose(eqx.filter_grad(fn)(x), _scalar_jac(x))
    if with_hvp:
        v = jnp.array([0.1, -0.2])
        assert jnp.allclose(
            jax.jvp(eqx.filter_grad(fn), (x,), (v,))[1],
            _scalar_hvp(x, v),
        )


def test_autodiff_wrapper_none_mode():
    """``autodiff_mode='none'`` returns the inputs unchanged."""
    fn, grad, hvp = autodiff_wrapper(
        _scalar_fn, _scalar_jac, _scalar_hvp, autodiff_mode="none"
    )
    assert fn is _scalar_fn
    assert grad is _scalar_jac
    assert hvp is _scalar_hvp


@pytest.mark.parametrize(
    ("mode", "grad", "match"),
    [
        ("custom", None, "grad must be provided when autodiff_mode is 'custom'"),
        ("none", None, "grad must be provided when autodiff_mode is 'none'"),
        ("bogus", _scalar_jac, "Invalid autodiff_mode"),
    ],
)
def test_autodiff_wrapper_errors(mode: str, grad: Callable | None, match: str):
    """Invalid modes / missing jacobians raise ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        autodiff_wrapper(_scalar_fn, grad, autodiff_mode=mode)  # ty: ignore[arg-type]
