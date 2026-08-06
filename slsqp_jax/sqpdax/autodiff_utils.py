"""Custom AD wrappers that inject user-supplied Jacobians and HVPs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from .primal import Primal

__all__ = [
    "fn_proxy_autodiff",
    "autodiff_wrapper",
    "ad_proxy_from_constants",
]

FnCallable = TypeVar("FnCallable", bound=Callable[..., Any])
GradCallable = TypeVar("GradCallable", bound=Callable[..., Any])
HVPCallable = TypeVar("HVPCallable", bound=Callable[..., Any])


def _contract_jac_tangent(jac: Any, x: Any, tangent: Any) -> Array:
    """Form the JVP ``⟨jac, tangent⟩`` for array- or pytree-valued ``x``.

    ``jac`` must have the same pytree structure as ``x``. Each leaf ``j`` of
    ``jac`` has shape ``out_shape + leaf_shape``, where ``leaf_shape`` is the
    shape of the corresponding leaf of ``x``. Contracting every leaf with its
    tangent and summing yields the JVP with shape ``out_shape``.

    Parameters
    ----------
    jac
        Jacobian pytree matching ``x``.
    x
        Primal input pytree (arrays or Equinox modules).
    tangent
        Tangent pytree matching ``x``, or ``None`` for a symbolic zero.

    Returns
    -------
    Array
        The Jacobian-vector product.

    Raises
    ------
    ValueError
        If ``jac`` has no array leaves.
    """
    x_leaves = jtu.tree_leaves(x)
    jac_leaves = jtu.tree_leaves(jac)
    if not jac_leaves:
        msg = "fn_jac must return at least one array leaf"
        raise ValueError(msg)

    if tangent is None:
        j0, x0 = jac_leaves[0], x_leaves[0]
        return jnp.zeros(j0.shape[: j0.ndim - x0.ndim], dtype=j0.dtype)

    total: Array | None = None
    for j, x_leaf, t in zip(
        jac_leaves, x_leaves, jtu.tree_leaves(tangent), strict=True
    ):
        contrib = (
            jnp.zeros(j.shape[: j.ndim - x_leaf.ndim], dtype=j.dtype)
            if t is None
            else jnp.tensordot(j, t, axes=x_leaf.ndim)
        )
        total = contrib if total is None else total + contrib
    assert total is not None
    return total


@overload
def fn_proxy_autodiff(
    fn: FnCallable,
    fn_jac: GradCallable,
    fn_hvp: None = None,
) -> tuple[FnCallable, GradCallable, None]: ...


@overload
def fn_proxy_autodiff(
    fn: FnCallable,
    fn_jac: GradCallable,
    fn_hvp: HVPCallable,
) -> tuple[FnCallable, GradCallable, HVPCallable]: ...


def fn_proxy_autodiff(
    fn: FnCallable,
    fn_jac: GradCallable,
    fn_hvp: HVPCallable | None = None,
) -> tuple[FnCallable, GradCallable, HVPCallable | None]:
    """Wrap ``fn`` / ``fn_jac`` so JAX AD uses the supplied derivatives.

    Returns ``(wrapped_fn, wrapped_jac, fn_hvp)`` suitable for unpacking into
    :func:`~slsqp_jax.sqpdax.problem.builder.build_problem` (as ``fn`` /
    ``grad`` / ``hvp``, or the analogous constraint triple). ``fn_hvp`` is
    returned unchanged.

    The leading argument ``x`` may be a plain array or an Equinox module
    (e.g. :class:`~slsqp_jax.sqpdax.primal.Primal`) with static fields;
    differentiation uses :func:`equinox.filter_custom_jvp`, which partitions
    differentiable floating-point leaves from static / nondifferentiable
    structure.

    First-order forward- and reverse-mode derivatives of ``wrapped_fn`` w.r.t.
    ``x`` are taken from ``wrapped_jac`` (reverse mode via transposition of the
    custom JVP). When ``fn_hvp`` is not ``None``, ``wrapped_jac`` itself has a
    custom JVP so that second-order AD of ``wrapped_fn`` (e.g.
    ``jax.jvp(eqx.filter_grad(wrapped_fn), ...)``) evaluates ``fn_hvp``. When
    ``fn_hvp`` is ``None``, ``wrapped_jac`` is ``fn_jac`` unchanged.

    Only the leading argument ``x`` is treated as differentiable. Keyword
    arguments are always nondifferentiable (Equinox convention), and tangents
    of ``*args`` are ignored by the custom rules.

    Parameters
    ----------
    fn
        Primal function ``fn(x, *args, **kwargs) -> y``.
    fn_jac
        Jacobian of ``fn`` w.r.t. ``x``, with the **same pytree structure**
        as ``x``. Each leaf has shape ``out_shape + leaf_shape`` so that
        contracting trailing axes with the tangent of ``x`` yields the JVP.
        For a scalar objective and array ``x`` of shape ``(n,)`` this is the
        gradient of shape ``(n,)``; for a length-``m`` constraint it is an
        ``(m, n)`` matrix. For modular ``x``, return a module of matching
        type (e.g. ``Primal(x=grad)``).
    fn_hvp
        Optional directional derivative of ``fn_jac``,
        ``fn_hvp(x, tangent, *args, **kwargs)``, with the same pytree
        structure / shapes as ``fn_jac(x, ...)``. ``tangent`` matches ``x``.
        Returned as-is in the output triple.

    Returns
    -------
    wrapped_fn
        ``fn`` with a custom JVP that uses ``wrapped_jac``.
    wrapped_jac
        ``fn_jac``, with a custom JVP from ``fn_hvp`` when the latter is not
        ``None``; otherwise ``fn_jac`` itself.
    fn_hvp
        The input ``fn_hvp``, unmodified.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.autodiff_utils import fn_proxy_autodiff
    >>> def f(x):
    ...     return jnp.sum(x**3)
    >>> def f_jac(x):
    ...     return 3 * x**2
    >>> def f_hvp(x, v):
    ...     return 6 * x * v
    >>> wrapped, wrapped_jac, hvp = fn_proxy_autodiff(f, f_jac, f_hvp)
    >>> x = jnp.array([1.0, 2.0])
    >>> float(wrapped(x))
    9.0
    >>> eqx.filter_grad(wrapped)(x).tolist()
    [3.0, 12.0]
    >>> wrapped_jac(x).tolist()
    [3.0, 12.0]
    >>> hvp is f_hvp
    True
    >>> v = jnp.array([0.5, -0.25])
    >>> jax.jvp(eqx.filter_grad(wrapped), (x,), (v,))[1].tolist()
    [3.0, -3.0]
    """
    if fn_hvp is not None:
        hvp = fn_hvp

        @eqx.filter_custom_jvp
        def wrapped_jac(x: Any, *args: Any, **kwargs: Any) -> Any:
            return fn_jac(x, *args, **kwargs)

        @wrapped_jac.def_jvp
        def _jac_jvp(
            primals: tuple[Any, ...], tangents: tuple[Any, ...], **kwargs: Any
        ) -> tuple[Any, Any]:
            x, *args = primals
            tx, *_ = tangents
            jac = wrapped_jac(x, *args, **kwargs)
            if tx is None:
                return jac, jtu.tree_map(jnp.zeros_like, jac)
            return jac, hvp(x, tx, *args, **kwargs)
    else:
        wrapped_jac = fn_jac

    @eqx.filter_custom_jvp
    def wrapped_fn(x: Any, *args: Any, **kwargs: Any) -> Array:
        return fn(x, *args, **kwargs)

    @wrapped_fn.def_jvp
    def _fn_jvp(
        primals: tuple[Any, ...], tangents: tuple[Any, ...], **kwargs: Any
    ) -> tuple[Array, Array]:
        x, *args = primals
        tx, *_ = tangents
        jac = wrapped_jac(x, *args, **kwargs)
        return wrapped_fn(x, *args, **kwargs), _contract_jac_tangent(jac, x, tx)

    return wrapped_fn, wrapped_jac, fn_hvp  # ty: ignore[invalid-return-type]


def autodiff_wrapper(
    fn: FnCallable,
    grad: GradCallable | None = None,
    hvp: HVPCallable | None = None,
    autodiff_mode: Literal["jax", "custom", "none"] = "custom",
    force_hvp_in_jax_mode: bool = False,
) -> tuple[FnCallable, GradCallable, HVPCallable | None]:
    """Resolve ``(fn, grad, hvp)`` according to ``autodiff_mode``.

    Intended as the single dispatch point used by
    :func:`~slsqp_jax.sqpdax.problem.builder.build_problem` for the objective
    and each constraint triple.

    Parameters
    ----------
    fn
        Primal callable ``fn(x, *args, **kwargs)``.
    grad
        User Jacobian / gradient of ``fn`` w.r.t. ``x``. Required when
        ``autodiff_mode`` is ``"custom"`` or ``"none"``.
    hvp
        Optional directional derivative of ``grad``,
        ``hvp(x, tangent, *args, **kwargs)``.
    autodiff_mode
        Derivative source:

        * ``"jax"`` — ignore ``grad`` / ``hvp`` and build the Jacobian from
          ``fn`` via :func:`equinox.filter_jacrev` (works for scalar and
          vector-valued ``fn``). The HVP is built from that Jacobian via
          :func:`equinox.filter_jvp` only when requested (see
          ``force_hvp_in_jax_mode``).
        * ``"custom"`` — wrap with :func:`fn_proxy_autodiff` so AD of ``fn``
          uses ``grad`` (and ``hvp`` when given).
        * ``"none"`` — return the three callables unchanged (no AD wiring).
    force_hvp_in_jax_mode
        Only used when ``autodiff_mode="jax"``. When ``False`` (default) and
        ``hvp`` is ``None``, the returned HVP is ``None`` so callers that do
        not need second-order information avoid building it. When ``True``,
        or when a non-``None`` ``hvp`` placeholder is passed, an HVP is
        constructed by differentiating the JAX-built Jacobian. The user
        ``hvp`` callable itself is never invoked in ``"jax"`` mode.

    Returns
    -------
    fn
        Possibly wrapped primal.
    grad
        Jacobian / gradient callable (never ``None``).
    hvp
        HVP callable, or ``None`` when not available.

    Raises
    ------
    ValueError
        If ``autodiff_mode`` is ``"custom"`` or ``"none"`` and ``grad`` is
        ``None``, or if ``autodiff_mode`` is not one of the allowed values.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.autodiff_utils import autodiff_wrapper
    >>> def f(x):
    ...     return jnp.sum(x**2)
    >>> fn, grad, hvp = autodiff_wrapper(
    ...     f, autodiff_mode="jax", force_hvp_in_jax_mode=True
    ... )
    >>> x = jnp.array([1.0, 2.0])
    >>> grad(x).tolist()
    [2.0, 4.0]
    >>> hvp(x, jnp.array([0.5, 0.0])).tolist()
    [1.0, 0.0]
    """
    if autodiff_mode == "jax":

        def _grad(x: Any, *args: Any, **kwargs: Any) -> Any:
            return eqx.filter_jacrev(lambda z: fn(z, *args, **kwargs))(x)

        if hvp is None and not force_hvp_in_jax_mode:
            resolved_hvp: HVPCallable | None = None
        else:

            def _hvp(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Any:
                return eqx.filter_jvp(
                    lambda z: _grad(z, *args, **kwargs),
                    (x,),
                    (tangent,),
                )[1]

            resolved_hvp = _hvp

        return cast(
            tuple[FnCallable, GradCallable, HVPCallable | None],
            (fn, _grad, resolved_hvp),
        )

    if autodiff_mode == "custom":
        if grad is None:
            msg = "grad must be provided when autodiff_mode is 'custom'"
            raise ValueError(msg)
        return fn_proxy_autodiff(fn, grad, hvp)

    if autodiff_mode == "none":
        if grad is None:
            msg = "grad must be provided when autodiff_mode is 'none'"
            raise ValueError(msg)
        return fn, grad, hvp

    msg = f"Invalid autodiff_mode: {autodiff_mode!r}"
    raise ValueError(msg)


def ad_proxy_from_constants(
    fn_val: Array,
    fn_jac: Array,
) -> Callable[[Primal], Array]:
    """Build a differentiable surrogate from a cached value and Jacobian.

    NLP callables (``problem.fn``, constraint residuals, …) need not be
    reverse-mode differentiable, but an :class:`~slsqp_jax.sqpdax.problem.basic.EvaluatedProblem`
    already stores their value and Jacobian at the current iterate. This
    returns ``fn(arg: Primal) -> Array`` whose primal value is ``fn_val`` and
    whose derivative w.r.t. ``arg.x`` is exactly ``fn_jac``, so AD of a
    consumer (e.g. a merit function) picks up the supplied Jacobian without
    differentiating the underlying residual.

    Unlike :func:`fn_proxy_autodiff`, this is a stop-gradient first-order
    surrogate rather than a custom JVP: ``fn_val`` / ``fn_jac`` are often
    tracers of the computation being differentiated, and
    ``jax.custom_vjp`` / ``filter_custom_jvp`` cannot close over such
    tracers. Stopping the gradient through the cached arrays also severs
    reverse mode through a possibly non-differentiable primal evaluation.

    Parameters
    ----------
    fn_val
        Cached function value at the evaluation point. Scalar for an
        objective; shape ``(m,)`` for an ``m``-constraint residual.
    fn_jac
        Cached Jacobian w.r.t. the decision vector ``x``. Shape ``(n,)``
        for a scalar objective, or ``(m, n)`` for a vector residual, so
        that ``tensordot(fn_jac, dx, axes=1)`` yields the JVP.

    Returns
    -------
    Callable[[Primal], Array]
        Surrogate ``fn(arg)`` with value ``fn_val`` and
        ``∂fn/∂arg.x = fn_jac``.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.autodiff_utils import ad_proxy_from_constants
    >>> from slsqp_jax.sqpdax.primal import Primal
    >>> fn_val = jnp.asarray(9.0)
    >>> fn_jac = jnp.array([3.0, 12.0])
    >>> proxy = ad_proxy_from_constants(fn_val, fn_jac)
    >>> p = Primal(x=jnp.array([1.0, 2.0]))
    >>> float(proxy(p))
    9.0
    >>> eqx.filter_grad(proxy)(p).x.tolist()
    [3.0, 12.0]
    """
    value = jax.lax.stop_gradient(fn_val)
    jac = jax.lax.stop_gradient(fn_jac)

    def fn(arg: Primal) -> Array:
        # arg.x - stop_gradient(arg.x) is numerically zero -- so the value stays
        # exactly `value` -- but carries a unit tangent, injecting `jac` as the
        # gradient. tensordot contracts `jac`'s trailing (n,) axis with `dx`:
        # scalar objective -> jac (n,); vector constraint -> jac (m, n) -> (m,).
        dx = arg.x - jax.lax.stop_gradient(arg.x)
        return value + jnp.tensordot(jac, dx, axes=1)

    return fn
