"""SciPy compatibility layer for SLSQP-JAX.

Provides utilities to convert SciPy-style constraint specifications
(dicts, LinearConstraint, NonlinearConstraint) into the function/Jacobian/HVP
signatures expected by the SLSQP solver, and a convenience
``minimize_like_scipy`` entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, Float
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from slsqp_jax.solver import SLSQP
from slsqp_jax.types import (
    ConstraintFn,
    ConstraintHVPFn,
    GradFn,
    HVPFn,
    JacobianFn,
)

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class _CachedEvaluator:
    """Identity-based cache that deduplicates calls sharing the same ``x``.

    Within a single SLSQP iteration the solver passes the *same* Python
    object ``y`` to both ``eq_constraint_fn`` and ``ineq_constraint_fn``.
    When a single underlying constraint source contributes to both groups,
    wrapping it in a ``_CachedEvaluator`` avoids evaluating the source
    function twice.

    Only one entry is stored (the most recent).  Across iterations ``x``
    is a new object so the cache auto-invalidates.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._cache_key: int | None = None
        self._cache_val: Any = None

    def __call__(self, x: Any, args: Any) -> Any:
        key = id(x)
        if key != self._cache_key:
            self._cache_val = self._fn(x, args)
            self._cache_key = key
        return self._cache_val


class _CachedEvaluator2:
    """Like ``_CachedEvaluator`` but keyed on two positional args (x, v)."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._cache_key: tuple[int, int] | None = None
        self._cache_val: Any = None

    def __call__(self, x: Any, v: Any, args: Any) -> Any:
        key = (id(x), id(v))
        if key != self._cache_key:
            self._cache_val = self._fn(x, v, args)
            self._cache_key = key
        return self._cache_val


# ---------------------------------------------------------------------------
# ParsedConstraints dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedConstraints:
    """Result of converting SciPy-style constraints for use with SLSQP.

    All fields map directly to the corresponding ``SLSQP`` constructor
    arguments.
    """

    eq_constraint_fn: Optional[ConstraintFn] = None
    ineq_constraint_fn: Optional[ConstraintFn] = None
    n_eq_constraints: int = 0
    n_ineq_constraints: int = 0
    eq_jac_fn: Optional[JacobianFn] = None
    ineq_jac_fn: Optional[JacobianFn] = None
    eq_hvp_fn: Optional[ConstraintHVPFn] = None
    ineq_hvp_fn: Optional[ConstraintHVPFn] = None


# ---------------------------------------------------------------------------
# Internal helpers to decompose a single constraint source
# ---------------------------------------------------------------------------


@dataclass
class _ConstraintParts:
    """Intermediate representation of a single SciPy constraint source.

    ``eq_fns`` and ``ineq_fns`` are lists of callables
    ``(x, args) -> Float[Array, " m_i"]``.  Similarly for Jacobians
    (``(x, args) -> Float[Array, "m_i n"]``) and HVPs
    (``(x, v, args) -> Float[Array, "m_i n"]``).
    """

    eq_fns: list[ConstraintFn] = field(default_factory=list)
    ineq_fns: list[ConstraintFn] = field(default_factory=list)
    n_eq: int = 0
    n_ineq: int = 0
    eq_jac_fns: list[JacobianFn | None] = field(default_factory=list)
    ineq_jac_fns: list[JacobianFn | None] = field(default_factory=list)
    eq_hvp_fns: list[ConstraintHVPFn | None] = field(default_factory=list)
    ineq_hvp_fns: list[ConstraintHVPFn | None] = field(default_factory=list)


def _parse_dict_constraint(con: dict, x0: Array) -> _ConstraintParts:
    """Parse a single SciPy dict constraint."""
    ctype = con["type"]
    raw_fun = con["fun"]
    raw_jac = con.get("jac", None)
    extra_args = con.get("args", ())

    def wrapped_fn(x: Any, args: Any) -> Float[Array, " m"]:
        val = raw_fun(x, *extra_args)
        return jnp.atleast_1d(jnp.asarray(val))

    jac_fn: JacobianFn | None = None
    if callable(raw_jac):

        def jac_fn(x: Any, args: Any) -> Float[Array, "m n"]:
            val = raw_jac(x, *extra_args)
            return jnp.atleast_2d(jnp.asarray(val))

    size = int(jnp.atleast_1d(jnp.asarray(raw_fun(x0, *extra_args))).shape[0])

    parts = _ConstraintParts()
    if ctype == "eq":
        parts.eq_fns.append(wrapped_fn)
        parts.n_eq = size
        parts.eq_jac_fns.append(jac_fn)
        parts.eq_hvp_fns.append(None)
    elif ctype == "ineq":
        parts.ineq_fns.append(wrapped_fn)
        parts.n_ineq = size
        parts.ineq_jac_fns.append(jac_fn)
        parts.ineq_hvp_fns.append(None)
    else:
        raise ValueError(f"Unknown constraint type '{ctype}'; expected 'eq' or 'ineq'")
    return parts


def _parse_linear_constraint(con: LinearConstraint) -> _ConstraintParts:
    """Parse a ``scipy.optimize.LinearConstraint``."""
    A = jnp.asarray(np.atleast_2d(np.asarray(con.A, dtype=float)))
    m = A.shape[0]
    lb = jnp.broadcast_to(jnp.asarray(np.asarray(con.lb, dtype=float)), (m,))
    ub = jnp.broadcast_to(jnp.asarray(np.asarray(con.ub, dtype=float)), (m,))

    # We need to decide at Python time which indices are eq / ineq, so use
    # concrete NumPy values.
    lb_np = np.asarray(lb)
    ub_np = np.asarray(ub)
    eq_mask = lb_np == ub_np
    has_lower = np.isfinite(lb_np) & ~eq_mask
    has_upper = np.isfinite(ub_np) & ~eq_mask

    parts = _ConstraintParts()

    # Equality parts (lb == ub)
    eq_indices = np.where(eq_mask)[0]
    if len(eq_indices) > 0:
        A_eq = A[eq_indices]
        lb_eq = lb[eq_indices]

        def eq_fn(x: Any, args: Any) -> Float[Array, " k"]:
            return A_eq @ x - lb_eq

        def eq_jac(x: Any, args: Any) -> Float[Array, "k n"]:
            return A_eq

        n_eq = len(eq_indices)

        def eq_hvp(x: Any, v: Any, args: Any) -> Float[Array, "k n"]:
            return jnp.zeros((n_eq, x.shape[0]))

        parts.eq_fns.append(eq_fn)
        parts.n_eq = n_eq
        parts.eq_jac_fns.append(eq_jac)
        parts.eq_hvp_fns.append(eq_hvp)

    # Inequality parts: lower bound  (A @ x - lb >= 0)
    lower_indices = np.where(has_lower)[0]
    upper_indices = np.where(has_upper)[0]
    n_lower = len(lower_indices)
    n_upper = len(upper_indices)
    n_ineq = n_lower + n_upper

    if n_ineq > 0:
        A_lower = A[lower_indices] if n_lower > 0 else jnp.zeros((0, A.shape[1]))
        lb_lower = lb[lower_indices] if n_lower > 0 else jnp.zeros((0,))
        A_upper = A[upper_indices] if n_upper > 0 else jnp.zeros((0, A.shape[1]))
        ub_upper = ub[upper_indices] if n_upper > 0 else jnp.zeros((0,))

        def ineq_fn(x: Any, args: Any) -> Float[Array, " k"]:
            lower_vals = A_lower @ x - lb_lower
            upper_vals = ub_upper - A_upper @ x
            return jnp.concatenate([lower_vals, upper_vals])

        ineq_jac_matrix = jnp.concatenate([A_lower, -A_upper], axis=0)

        def ineq_jac(x: Any, args: Any) -> Float[Array, "k n"]:
            return ineq_jac_matrix

        def ineq_hvp(x: Any, v: Any, args: Any) -> Float[Array, "k n"]:
            return jnp.zeros((n_ineq, x.shape[0]))

        parts.ineq_fns.append(ineq_fn)
        parts.n_ineq = n_ineq
        parts.ineq_jac_fns.append(ineq_jac)
        parts.ineq_hvp_fns.append(ineq_hvp)

    return parts


def _parse_nonlinear_constraint(con: NonlinearConstraint) -> _ConstraintParts:
    """Parse a ``scipy.optimize.NonlinearConstraint``."""
    raw_fun = con.fun
    lb = np.atleast_1d(np.asarray(con.lb, dtype=float))
    ub = np.atleast_1d(np.asarray(con.ub, dtype=float))
    m = max(lb.shape[0], ub.shape[0])
    lb = np.broadcast_to(lb, (m,))
    ub = np.broadcast_to(ub, (m,))

    lb_jnp = jnp.asarray(lb)
    ub_jnp = jnp.asarray(ub)

    eq_mask = lb == ub
    has_lower = np.isfinite(lb) & ~eq_mask
    has_upper = np.isfinite(ub) & ~eq_mask

    eq_indices = np.where(eq_mask)[0]
    lower_indices = np.where(has_lower)[0]
    upper_indices = np.where(has_upper)[0]

    needs_eq = len(eq_indices) > 0
    needs_ineq = len(lower_indices) + len(upper_indices) > 0

    # If a single source feeds both eq and ineq, share via cache
    if needs_eq and needs_ineq:

        def fun_closure(x, args):
            return jnp.atleast_1d(jnp.asarray(raw_fun(x)))

        cached_fn = _CachedEvaluator(fun_closure)
    else:
        cached_fn = None

    # Jacobian / HVP availability
    raw_jac = getattr(con, "jac", None)
    has_jac = callable(raw_jac)
    raw_hess = getattr(con, "hess", None)
    has_hess = callable(raw_hess)

    # Cached Jacobian evaluator (if callable)
    if raw_jac is not None and has_jac and needs_eq and needs_ineq:

        def jac_closure(x, args):
            return jnp.atleast_2d(jnp.asarray(raw_jac(x)))

        cached_jac = _CachedEvaluator(jac_closure)
    else:
        cached_jac = None

    # Cached HVP evaluator: compute all m per-component HVPs once,
    # then let eq_hvp_fn / ineq_hvp_fn select their rows.
    if raw_hess is not None and has_hess and needs_eq and needs_ineq:

        def _all_component_hvps(x: Any, v: Any, args: Any) -> Float[Array, "m n"]:
            rows = []
            for i in range(m):
                e_i = jnp.zeros((m,)).at[i].set(1.0)
                H_i = jnp.asarray(raw_hess(x, e_i))
                rows.append(H_i @ v)
            return jnp.stack(rows)

        cached_hvp = _CachedEvaluator2(_all_component_hvps)
    else:
        cached_hvp = None

    parts = _ConstraintParts()

    # --- Equality portion ---
    if needs_eq:
        lb_eq = lb_jnp[eq_indices]

        if cached_fn is not None:

            def eq_fn(x: Any, args: Any) -> Float[Array, " k"]:
                return cached_fn(x, args)[eq_indices] - lb_eq
        else:

            def eq_fn(x: Any, args: Any) -> Float[Array, " k"]:
                return jnp.atleast_1d(jnp.asarray(raw_fun(x)))[eq_indices] - lb_eq

        parts.eq_fns.append(eq_fn)
        parts.n_eq = len(eq_indices)

        # Jacobian
        eq_jac_fn: JacobianFn | None = None
        if raw_jac is not None and has_jac:
            if cached_jac is not None:

                def eq_jac_fn(x: Any, args: Any) -> Float[Array, "k n"]:
                    return cached_jac(x, args)[eq_indices]
            else:

                def eq_jac_fn(x: Any, args: Any) -> Float[Array, "k n"]:
                    return jnp.atleast_2d(jnp.asarray(raw_jac(x)))[eq_indices]

        parts.eq_jac_fns.append(eq_jac_fn)

        # HVP
        eq_hvp_fn: ConstraintHVPFn | None = None
        if raw_hess is not None and has_hess:
            if cached_hvp is not None:

                def eq_hvp_fn(x: Any, v: Any, args: Any) -> Float[Array, "k n"]:
                    return cached_hvp(x, v, args)[eq_indices]
            else:
                n_eq = len(eq_indices)

                def eq_hvp_fn(x: Any, v: Any, args: Any) -> Float[Array, "k n"]:
                    rows = []
                    for i in range(n_eq):
                        e_i = jnp.zeros((m,)).at[eq_indices[i]].set(1.0)
                        H_i = jnp.asarray(raw_hess(x, e_i))
                        rows.append(H_i @ v)
                    return jnp.stack(rows)

        parts.eq_hvp_fns.append(eq_hvp_fn)

    # --- Inequality portion ---
    if needs_ineq:
        n_lower = len(lower_indices)
        n_upper = len(upper_indices)
        n_ineq = n_lower + n_upper
        lb_lower = lb_jnp[lower_indices] if n_lower > 0 else jnp.zeros((0,))
        ub_upper = ub_jnp[upper_indices] if n_upper > 0 else jnp.zeros((0,))

        if cached_fn is not None:

            def ineq_fn(x: Any, args: Any) -> Float[Array, " k"]:
                vals = cached_fn(x, args)
                lower_part = (
                    vals[lower_indices] - lb_lower if n_lower > 0 else jnp.zeros((0,))
                )
                upper_part = (
                    ub_upper - vals[upper_indices] if n_upper > 0 else jnp.zeros((0,))
                )
                return jnp.concatenate([lower_part, upper_part])
        else:

            def ineq_fn(x: Any, args: Any) -> Float[Array, " k"]:
                vals = jnp.atleast_1d(jnp.asarray(raw_fun(x)))
                lower_part = (
                    vals[lower_indices] - lb_lower if n_lower > 0 else jnp.zeros((0,))
                )
                upper_part = (
                    ub_upper - vals[upper_indices] if n_upper > 0 else jnp.zeros((0,))
                )
                return jnp.concatenate([lower_part, upper_part])

        parts.ineq_fns.append(ineq_fn)
        parts.n_ineq = n_ineq

        # Jacobian
        ineq_jac_fn: JacobianFn | None = None
        if raw_jac is not None and has_jac:
            if cached_jac is not None:

                def ineq_jac_fn(x: Any, args: Any) -> Float[Array, "k n"]:
                    full_jac = cached_jac(x, args)
                    lower_jac = (
                        full_jac[lower_indices]
                        if n_lower > 0
                        else jnp.zeros((0, x.shape[0]))
                    )
                    upper_jac = (
                        -full_jac[upper_indices]
                        if n_upper > 0
                        else jnp.zeros((0, x.shape[0]))
                    )
                    return jnp.concatenate([lower_jac, upper_jac], axis=0)
            else:

                def ineq_jac_fn(x: Any, args: Any) -> Float[Array, "k n"]:
                    full_jac = jnp.atleast_2d(jnp.asarray(raw_jac(x)))
                    lower_jac = (
                        full_jac[lower_indices]
                        if n_lower > 0
                        else jnp.zeros((0, x.shape[0]))
                    )
                    upper_jac = (
                        -full_jac[upper_indices]
                        if n_upper > 0
                        else jnp.zeros((0, x.shape[0]))
                    )
                    return jnp.concatenate([lower_jac, upper_jac], axis=0)

        parts.ineq_jac_fns.append(ineq_jac_fn)

        # HVP
        ineq_hvp_fn: ConstraintHVPFn | None = None
        if raw_hess is not None and has_hess:
            if cached_hvp is not None:

                def ineq_hvp_fn(x: Any, v: Any, args: Any) -> Float[Array, "k n"]:
                    all_hvps = cached_hvp(x, v, args)
                    lower_rows = (
                        all_hvps[lower_indices]
                        if n_lower > 0
                        else jnp.zeros((0, x.shape[0]))
                    )
                    upper_rows = (
                        -all_hvps[upper_indices]
                        if n_upper > 0
                        else jnp.zeros((0, x.shape[0]))
                    )
                    return jnp.concatenate([lower_rows, upper_rows])
            else:

                def ineq_hvp_fn(x: Any, v: Any, args: Any) -> Float[Array, "k n"]:
                    rows = []
                    for idx in lower_indices:
                        e_i = jnp.zeros((m,)).at[idx].set(1.0)
                        H_i = jnp.asarray(raw_hess(x, e_i))
                        rows.append(H_i @ v)
                    for idx in upper_indices:
                        e_i = jnp.zeros((m,)).at[idx].set(1.0)
                        H_i = jnp.asarray(raw_hess(x, e_i))
                        rows.append(-(H_i @ v))
                    return jnp.stack(rows) if rows else jnp.zeros((0, x.shape[0]))

        parts.ineq_hvp_fns.append(ineq_hvp_fn)

    return parts


# ---------------------------------------------------------------------------
# Combining helpers
# ---------------------------------------------------------------------------


def _combine_fns(
    fns: list[ConstraintFn],
) -> ConstraintFn:
    """Concatenate outputs of multiple constraint functions."""
    if len(fns) == 1:
        return fns[0]

    def combined(x: Any, args: Any) -> Float[Array, " m"]:
        return jnp.concatenate([f(x, args) for f in fns])

    return combined


def _combine_jac_fns(
    jac_fns: list[JacobianFn | None],
) -> JacobianFn | None:
    """Vertically stack Jacobians.  Returns ``None`` if any entry is None."""
    if any(j is None for j in jac_fns):
        return None
    fns = [j for j in jac_fns if j is not None]  # for type narrowing
    if len(fns) == 1:
        return fns[0]

    def combined(x: Any, args: Any) -> Float[Array, "m n"]:
        return jnp.concatenate([f(x, args) for f in fns], axis=0)

    return combined


def _combine_hvp_fns(
    hvp_fns: list[ConstraintHVPFn | None],
) -> ConstraintHVPFn | None:
    """Vertically stack per-constraint HVP outputs.  ``None`` if any is None."""
    if any(h is None for h in hvp_fns):
        return None
    fns = [h for h in hvp_fns if h is not None]
    if len(fns) == 1:
        return fns[0]

    def combined(x: Any, v: Any, args: Any) -> Float[Array, "m n"]:
        return jnp.concatenate([f(x, v, args) for f in fns], axis=0)

    return combined


# ---------------------------------------------------------------------------
# parse_constraints  (public API)
# ---------------------------------------------------------------------------


def parse_constraints(
    constraints: dict | list | LinearConstraint | NonlinearConstraint | tuple,
    x0: Array,
) -> ParsedConstraints:
    """Convert SciPy-style constraints into SLSQP-JAX constraint functions.

    Parameters
    ----------
    constraints
        Any form accepted by ``scipy.optimize.minimize``:
        a dict, list of dicts, ``LinearConstraint``, ``NonlinearConstraint``,
        or a list/tuple mixing those types.  An empty tuple/list means
        "no constraints".
    x0
        Initial guess -- used to evaluate dict constraint functions once to
        determine their output size.

    Returns
    -------
    ParsedConstraints
        Dataclass whose fields map to ``SLSQP`` constructor arguments.
    """
    # Normalise to list
    if isinstance(constraints, dict):
        constraint_list: list = [constraints]
    elif isinstance(constraints, (LinearConstraint, NonlinearConstraint)):
        constraint_list = [constraints]
    elif isinstance(constraints, (list, tuple)):
        constraint_list = list(constraints)
    else:
        raise TypeError(  # pragma: no cover
            f"Unsupported constraints type {type(constraints)}. "
            "Expected a dict, list, LinearConstraint, or NonlinearConstraint."
        )

    if len(constraint_list) == 0:
        return ParsedConstraints()

    # Collect parts from each source
    all_parts: list[_ConstraintParts] = []
    for con in constraint_list:
        if isinstance(con, dict):
            all_parts.append(_parse_dict_constraint(con, x0))
        elif isinstance(con, LinearConstraint):
            all_parts.append(_parse_linear_constraint(con))
        elif isinstance(con, NonlinearConstraint):
            all_parts.append(_parse_nonlinear_constraint(con))
        else:
            raise TypeError(
                f"Unsupported constraint object type: {type(con)}"
            )  # pragma: no cover

    # Merge all parts
    eq_fns: list[ConstraintFn] = []
    ineq_fns: list[ConstraintFn] = []
    n_eq = 0
    n_ineq = 0
    eq_jac_fns: list[JacobianFn | None] = []
    ineq_jac_fns: list[JacobianFn | None] = []
    eq_hvp_fns: list[ConstraintHVPFn | None] = []
    ineq_hvp_fns: list[ConstraintHVPFn | None] = []

    for p in all_parts:
        eq_fns.extend(p.eq_fns)
        ineq_fns.extend(p.ineq_fns)
        n_eq += p.n_eq
        n_ineq += p.n_ineq
        eq_jac_fns.extend(p.eq_jac_fns)
        ineq_jac_fns.extend(p.ineq_jac_fns)
        eq_hvp_fns.extend(p.eq_hvp_fns)
        ineq_hvp_fns.extend(p.ineq_hvp_fns)

    result = ParsedConstraints(n_eq_constraints=n_eq, n_ineq_constraints=n_ineq)

    if eq_fns:
        result.eq_constraint_fn = _combine_fns(eq_fns)
        result.eq_jac_fn = _combine_jac_fns(eq_jac_fns)
        result.eq_hvp_fn = _combine_hvp_fns(eq_hvp_fns)

    if ineq_fns:
        result.ineq_constraint_fn = _combine_fns(ineq_fns)
        result.ineq_jac_fn = _combine_jac_fns(ineq_jac_fns)
        result.ineq_hvp_fn = _combine_hvp_fns(ineq_hvp_fns)

    return result


# ---------------------------------------------------------------------------
# Bounds conversion
# ---------------------------------------------------------------------------


def _convert_bounds(
    bounds: Bounds | list | tuple | None,
    n: int,
) -> Optional[Float[Array, "n 2"]]:
    """Convert SciPy-style bounds to the ``(n, 2)`` array used by SLSQP.

    Parameters
    ----------
    bounds
        ``None``, a ``scipy.optimize.Bounds`` instance, or a sequence of
        ``(min, max)`` pairs (with ``None`` meaning unbounded).
    n
        Number of variables (for validation).

    Returns
    -------
    jax array of shape ``(n, 2)`` or ``None``.
    """
    if bounds is None:
        return None

    if isinstance(bounds, Bounds):
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        lb = np.broadcast_to(lb, (n,))
        ub = np.broadcast_to(ub, (n,))
        arr = np.stack([lb, ub], axis=1)
        # Replace np.nan with inf (Bounds uses nan for no-bound sometimes)
        arr[np.isnan(arr[:, 0]), 0] = -np.inf
        arr[np.isnan(arr[:, 1]), 1] = np.inf
        return jnp.asarray(arr)

    # Sequence of (min, max) pairs
    bounds_list = list(bounds)
    if len(bounds_list) != n:
        raise ValueError(
            f"bounds has {len(bounds_list)} entries but x0 has {n} elements"
        )
    arr = np.full((n, 2), [-np.inf, np.inf])
    for i, (lo, hi) in enumerate(bounds_list):
        if lo is not None:
            arr[i, 0] = float(lo)
        if hi is not None:
            arr[i, 1] = float(hi)
    return jnp.asarray(arr)


# ---------------------------------------------------------------------------
# minimize_like_scipy  (public API)
# ---------------------------------------------------------------------------


def minimize_like_scipy(
    fun: Callable,
    x0: Any,
    args: tuple = (),
    *,
    jac: Callable | bool | None = None,
    hessp: Callable | None = None,
    bounds: Bounds | list | tuple | None = None,
    constraints: dict | list | LinearConstraint | NonlinearConstraint | tuple = (),
    options: dict[str, Any] | None = None,
    has_aux: bool = False,
    throw: bool = False,
    verbose: bool | Callable[..., None] = False,
) -> optx.Solution:
    """Minimise a function using SLSQP with a SciPy-like interface.

    This is a convenience wrapper that accepts SciPy-style arguments,
    converts them for the SLSQP solver, and delegates to
    ``optimistix.minimise``.

    Parameters
    ----------
    fun
        Objective function.  Signature ``(x, *args) -> scalar`` or, when
        *has_aux* is ``True``, ``(x, *args) -> (scalar, aux)``.
    x0
        Initial guess (array-like).
    args
        Extra positional arguments forwarded to *fun* (unpacked).
    jac
        Gradient of *fun*.  A callable ``(x, *args) -> array`` or ``True``
        to indicate that *fun* returns ``(f, g)`` (or ``((f, g), aux)``
        when *has_aux* is set).
    hessp
        Hessian-vector product ``(x, p, *args) -> array``.
    bounds
        Variable bounds -- ``None``, ``Bounds``, or sequence of
        ``(min, max)`` pairs.
    constraints
        SciPy-style constraints (dict / list-of-dicts /
        ``LinearConstraint`` / ``NonlinearConstraint``).
    options
        Solver options dict.  The following keys are popped with
        the listed defaults (which match the ``SLSQP`` constructor
        defaults):

        * ``rtol`` (``1e-6``) -- relative tolerance for stationarity.
        * ``atol`` (``1e-6``) -- absolute tolerance for stationarity
          and feasibility.
        * ``max_steps`` or ``maxiter`` (``100``) -- maximum outer
          iterations.
        * ``min_steps`` (``1``) -- minimum iterations before
          convergence is allowed.
        * ``lbfgs_memory`` (``10``) -- number of L-BFGS pairs.
        * ``line_search_max_steps`` (``20``) -- backtracking steps.
        * ``armijo_c1`` (``1e-4``) -- Armijo sufficient decrease.
        * ``qp_max_iter`` (``100``) -- active-set iteration budget.
        * ``qp_max_cg_iter`` (``50``) -- CG iterations per QP step.

        Any remaining keys are forwarded as ``**kwargs`` to the
        ``SLSQP`` constructor, so any ``SLSQP`` attribute can be
        set here (e.g. ``proximal_tau``, ``proximal_mu_min``,
        ``proximal_mu_max``, ``use_preconditioner``, ``adaptive_cg_tol``,
        ``cg_regularization``, ``stagnation_tol``).
    has_aux
        If ``True``, *fun* returns ``(value, aux)``.
    throw
        Whether to raise on solver failure.
    verbose
        Passed to the ``SLSQP`` constructor.  ``False`` (default) for
        silent, ``True`` to print all diagnostics, or a custom callable.

    Returns
    -------
    optimistix.Solution
    """
    opts = dict(options) if options is not None else {}
    x0 = jnp.asarray(x0, dtype=float)
    n = x0.shape[0]

    # --- Parse constraints ---
    parsed = parse_constraints(constraints, x0)

    # --- Convert bounds ---
    jax_bounds = _convert_bounds(bounds, n)

    # --- Wrap objective ---
    obj_grad_fn: GradFn | None = None

    if jac is True:
        # fun returns (f, g) or ((f, g), aux)
        if has_aux:  # pragma: no cover

            def wrapped_fn(x: Any, packed_args: Any) -> tuple:
                (f, g), aux = fun(x, *packed_args)
                return jnp.asarray(f), aux
        else:

            def wrapped_fn(x: Any, packed_args: Any) -> tuple:
                f, g = fun(x, *packed_args)
                return jnp.asarray(f), None

        # Extract gradient
        if has_aux:  # pragma: no cover

            def obj_grad_fn(x: Any, packed_args: Any) -> Any:
                (f, g), _aux = fun(x, *packed_args)
                return jnp.asarray(g)
        else:

            def obj_grad_fn(x: Any, packed_args: Any) -> Any:
                _f, g = fun(x, *packed_args)
                return jnp.asarray(g)
    else:
        if has_aux:

            def wrapped_fn(x: Any, packed_args: Any) -> tuple:
                f, aux = fun(x, *packed_args)
                return jnp.asarray(f), aux
        else:

            def wrapped_fn(x: Any, packed_args: Any) -> tuple:
                return jnp.asarray(fun(x, *packed_args)), None

    # --- Wrap jac (if callable) ---
    if callable(jac):
        user_jac = jac

        def obj_grad_fn(x: Any, packed_args: Any) -> Any:
            return jnp.asarray(user_jac(x, *packed_args))

    # --- Wrap hessp ---
    obj_hvp_fn: HVPFn | None = None
    if callable(hessp):
        user_hessp = hessp

        def obj_hvp_fn(x: Any, v: Any, packed_args: Any) -> Any:
            return jnp.asarray(user_hessp(x, v, *packed_args))

    # --- Build solver ---
    max_steps = opts.pop("max_steps", opts.pop("maxiter", 100))
    solver = SLSQP(
        rtol=opts.pop("rtol", 1e-6),
        atol=opts.pop("atol", 1e-6),
        max_steps=max_steps,
        min_steps=opts.pop("min_steps", 1),
        eq_constraint_fn=parsed.eq_constraint_fn,
        ineq_constraint_fn=parsed.ineq_constraint_fn,
        n_eq_constraints=parsed.n_eq_constraints,
        n_ineq_constraints=parsed.n_ineq_constraints,
        bounds=jax_bounds,
        obj_grad_fn=obj_grad_fn,
        eq_jac_fn=parsed.eq_jac_fn,
        ineq_jac_fn=parsed.ineq_jac_fn,
        obj_hvp_fn=obj_hvp_fn,
        eq_hvp_fn=parsed.eq_hvp_fn,
        ineq_hvp_fn=parsed.ineq_hvp_fn,
        lbfgs_memory=opts.pop("lbfgs_memory", 10),
        line_search_max_steps=opts.pop("line_search_max_steps", 20),
        armijo_c1=opts.pop("armijo_c1", 1e-4),
        qp_max_iter=opts.pop("qp_max_iter", 100),
        qp_max_cg_iter=opts.pop("qp_max_cg_iter", 50),
        verbose=verbose,  # type: ignore[arg-type]  # verbose is resolved in __check_init__
        **opts,
    )

    sol = optx.minimise(
        wrapped_fn,
        solver,
        x0,
        args=args,
        has_aux=True,
        max_steps=max_steps,
        throw=throw,
    )
    return sol
