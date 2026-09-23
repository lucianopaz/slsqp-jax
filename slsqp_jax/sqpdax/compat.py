"""SciPy-compatible input conversion and convenience entry point for sqpdax.

The functions in this module translate SciPy's public optimisation inputs to
the native sqpdax :class:`~slsqp_jax.sqpdax.problem.Problem` convention.
SciPy dictionary inequalities use ``c(x) >= 0``; sqpdax uses ``h(x) <= 0``,
so inequality values and all of their derivatives are negated here.

Non-standard ``NonlinearConstraint.hessp`` extension
----------------------------------------------------
SciPy does not define ``NonlinearConstraint.hessp``. For parity with the
legacy SLSQP-JAX compatibility API, a user-attached callable is accepted with
signature ``hessp(x, p) -> (m, n)``. It takes precedence over SciPy's
``hess(x, v)`` and returns one component Hessian-vector product per row.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
)

from .autodiff_utils import autodiff_wrapper
from .caching import _CachedEvaluator, _CachedEvaluator2
from .minimiser import (
    AbstractConstrainedMinimiser,
    ActiveSetLineSearchMinimiser,
    minimise,
)
from .problem import Problem, ProblemProtocol, build_problem

__all__ = [
    "ParsedConstraints",
    "parse_constraints",
    "build_problem_from_scipy_specification",
    "minimise_and_return_scipy_result",
    "minimize_like_scipy",
]

ConstraintSpec = (
    dict[str, Any]
    | LinearConstraint
    | NonlinearConstraint
    | list[Any]
    | tuple[Any, ...]
)


def _validate_hessp_signature(fn: Callable[..., Any]) -> None:
    """Validate the legacy two-argument nonlinear-constraint HVP contract."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.default is inspect.Parameter.empty
    ]
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if not has_varargs and len(positional) != 2:
        raise TypeError(
            "NonlinearConstraint.hessp must accept exactly two positional "
            "arguments (x, p); got a callable with "
            f"{len(positional)} required positional parameters. Expected "
            "hessp(x, p) -> Array of shape (m, n)."
        )


@dataclass
class ParsedConstraints:
    """SciPy constraints converted to sqpdax callable blocks."""

    eq_fn: Callable[..., Any] | None = None
    ineq_fn: Callable[..., Any] | None = None
    meq: int = 0
    mineq: int = 0
    eq_fn_jac: Callable[..., Any] | None = None
    ineq_fn_jac: Callable[..., Any] | None = None
    eq_fn_hvp: Callable[..., Any] | None = None
    ineq_fn_hvp: Callable[..., Any] | None = None


@dataclass
class _ConstraintParts:
    eq_fns: list[Callable[..., Any]] = field(default_factory=list)
    ineq_fns: list[Callable[..., Any]] = field(default_factory=list)
    meq: int = 0
    mineq: int = 0
    eq_jacs: list[Callable[..., Any] | None] = field(default_factory=list)
    ineq_jacs: list[Callable[..., Any] | None] = field(default_factory=list)
    eq_hvps: list[Callable[..., Any] | None] = field(default_factory=list)
    ineq_hvps: list[Callable[..., Any] | None] = field(default_factory=list)


def _validate_endpoint_arrays(lb: np.ndarray, ub: np.ndarray, *, name: str) -> None:
    """Reject invalid bounds before any values enter JAX-compiled code."""
    invalid = {
        "NaN endpoint": np.isnan(lb) | np.isnan(ub),
        "lower endpoint greater than upper endpoint": lb > ub,
        "+inf lower endpoint": np.isposinf(lb),
        "-inf upper endpoint": np.isneginf(ub),
    }
    for reason, mask in invalid.items():
        indices = np.flatnonzero(mask)
        if indices.size:
            index = int(indices[0])
            raise ValueError(
                f"{name} has {reason} at index {index}: ({lb[index]!r}, {ub[index]!r})"
            )


def _broadcast_constraint_bounds(
    lb: Any, ub: Any, m: int, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    try:
        lower = np.broadcast_to(np.asarray(lb, dtype=float), (m,))
        upper = np.broadcast_to(np.asarray(ub, dtype=float), (m,))
    except ValueError as error:
        raise ValueError(f"{name} bounds cannot be broadcast to {m} rows") from error
    _validate_endpoint_arrays(lower, upper, name=name)
    return lower, upper


def _parse_dict_constraint(con: Mapping[str, Any], x0: Array) -> _ConstraintParts:
    try:
        constraint_type = con["type"]
        raw_fn = con["fun"]
    except KeyError as error:
        raise ValueError(
            f"constraint dictionary is missing {error.args[0]!r}"
        ) from error
    raw_jac = con.get("jac")
    extra_args = con.get("args", ())

    def fn(x: Any, *args: Any, **kwargs: Any) -> Array:
        del args, kwargs
        return jnp.atleast_1d(jnp.asarray(raw_fn(x, *extra_args)))

    jac = None
    if callable(raw_jac):

        def jac(x: Any, *args: Any, **kwargs: Any) -> Array:
            del args, kwargs
            return jnp.atleast_2d(jnp.asarray(raw_jac(x, *extra_args)))

    size = int(jnp.atleast_1d(jnp.asarray(raw_fn(x0, *extra_args))).shape[0])
    parts = _ConstraintParts()
    if constraint_type == "eq":
        parts.eq_fns.append(fn)
        parts.eq_jacs.append(jac)
        parts.eq_hvps.append(None)
        parts.meq = size
    elif constraint_type == "ineq":

        def ineq_fn(x: Any, *args: Any, **kwargs: Any) -> Array:
            return -fn(x, *args, **kwargs)

        ineq_jac = None
        if jac is not None:

            def ineq_jac(x: Any, *args: Any, **kwargs: Any) -> Array:
                return -jac(x, *args, **kwargs)

        parts.ineq_fns.append(ineq_fn)
        parts.ineq_jacs.append(ineq_jac)
        parts.ineq_hvps.append(None)
        parts.mineq = size
    else:
        raise ValueError(
            f"Unknown constraint type {constraint_type!r}; expected 'eq' or 'ineq'"
        )
    return parts


def _parse_linear_constraint(con: LinearConstraint) -> _ConstraintParts:
    matrix = jnp.asarray(np.atleast_2d(np.asarray(con.A, dtype=float)))
    m = matrix.shape[0]
    lb, ub = _broadcast_constraint_bounds(con.lb, con.ub, m, name="LinearConstraint")
    lb_array, ub_array = jnp.asarray(lb), jnp.asarray(ub)
    eq_mask = lb == ub
    lower_indices = np.flatnonzero(np.isfinite(lb) & ~eq_mask)
    upper_indices = np.flatnonzero(np.isfinite(ub) & ~eq_mask)
    eq_indices = np.flatnonzero(eq_mask)
    parts = _ConstraintParts()

    if eq_indices.size:
        eq_matrix = matrix[eq_indices]
        eq_lb = lb_array[eq_indices]

        def eq_fn(x: Any, *args: Any, **kwargs: Any) -> Array:
            del args, kwargs
            return eq_matrix @ x - eq_lb

        def eq_jac(x: Any, *args: Any, **kwargs: Any) -> Array:
            del x, args, kwargs
            return eq_matrix

        def eq_hvp(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Array:
            del tangent, args, kwargs
            return jnp.zeros((eq_indices.size, x.shape[0]), dtype=x.dtype)

        parts.eq_fns.append(eq_fn)
        parts.eq_jacs.append(eq_jac)
        parts.eq_hvps.append(eq_hvp)
        parts.meq = int(eq_indices.size)

    if lower_indices.size or upper_indices.size:
        lower_matrix = matrix[lower_indices]
        upper_matrix = matrix[upper_indices]
        lower_bounds = lb_array[lower_indices]
        upper_bounds = ub_array[upper_indices]
        ineq_matrix = jnp.concatenate([-lower_matrix, upper_matrix], axis=0)

        def ineq_fn(x: Any, *args: Any, **kwargs: Any) -> Array:
            del args, kwargs
            return jnp.concatenate(
                [lower_bounds - lower_matrix @ x, upper_matrix @ x - upper_bounds]
            )

        def ineq_jac(x: Any, *args: Any, **kwargs: Any) -> Array:
            del x, args, kwargs
            return ineq_matrix

        def ineq_hvp(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Array:
            del tangent, args, kwargs
            return jnp.zeros(
                (lower_indices.size + upper_indices.size, x.shape[0]), dtype=x.dtype
            )

        parts.ineq_fns.append(ineq_fn)
        parts.ineq_jacs.append(ineq_jac)
        parts.ineq_hvps.append(ineq_hvp)
        parts.mineq = int(lower_indices.size + upper_indices.size)
    return parts


def _parse_nonlinear_constraint(
    con: NonlinearConstraint, x0: Array
) -> _ConstraintParts:
    raw_fn = con.fun
    initial_value = jnp.atleast_1d(jnp.asarray(raw_fn(x0)))
    m = int(initial_value.shape[0])
    lb, ub = _broadcast_constraint_bounds(con.lb, con.ub, m, name="NonlinearConstraint")
    lb_array, ub_array = jnp.asarray(lb), jnp.asarray(ub)
    eq_mask = lb == ub
    eq_indices = np.flatnonzero(eq_mask)
    lower_indices = np.flatnonzero(np.isfinite(lb) & ~eq_mask)
    upper_indices = np.flatnonzero(np.isfinite(ub) & ~eq_mask)
    needs_eq = bool(eq_indices.size)
    needs_ineq = bool(lower_indices.size or upper_indices.size)

    def all_values(x: Any, *args: Any, **kwargs: Any) -> Array:
        del args, kwargs
        return jnp.atleast_1d(jnp.asarray(raw_fn(x)))

    values = _CachedEvaluator(all_values) if needs_eq and needs_ineq else all_values

    raw_jac = getattr(con, "jac", None)
    jac_callable = cast(Callable[..., Any], raw_jac) if callable(raw_jac) else None

    def all_jacobians(x: Any, *args: Any, **kwargs: Any) -> Array:
        del args, kwargs
        assert jac_callable is not None
        return jnp.atleast_2d(jnp.asarray(jac_callable(x)))

    jacobians: Callable[..., Any] | None = None
    if jac_callable is not None:
        jacobians = (
            _CachedEvaluator(all_jacobians)
            if needs_eq and needs_ineq
            else all_jacobians
        )

    raw_hessp = getattr(con, "hessp", None)
    hessp = raw_hessp if callable(raw_hessp) else None
    if hessp is not None:
        _validate_hessp_signature(hessp)
    raw_hess = getattr(con, "hess", None)
    hess = raw_hess if callable(raw_hess) else None

    all_hvps: Callable[..., Any] | None = None
    if hessp is not None:

        def evaluate_hvps(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Array:
            del args, kwargs
            return jnp.atleast_2d(jnp.asarray(hessp(x, tangent)))

        all_hvps = evaluate_hvps
    elif hess is not None:

        def evaluate_hvps(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Array:
            del args, kwargs
            rows = []
            for index in range(m):
                weight = jnp.zeros((m,)).at[index].set(1.0)
                rows.append(jnp.asarray(hess(x, weight)) @ tangent)
            return jnp.stack(rows)

        all_hvps = evaluate_hvps
    if all_hvps is not None and needs_eq and needs_ineq:
        all_hvps = _CachedEvaluator2(all_hvps)

    parts = _ConstraintParts()
    if needs_eq:
        eq_lb = lb_array[eq_indices]

        def eq_fn(x: Any, *args: Any, **kwargs: Any) -> Array:
            return values(x, *args, **kwargs)[eq_indices] - eq_lb

        eq_jac = None
        if jacobians is not None:

            def eq_jac(x: Any, *args: Any, **kwargs: Any) -> Array:
                return jacobians(x, *args, **kwargs)[eq_indices]

        eq_hvp = None
        if all_hvps is not None:

            def eq_hvp(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Array:
                return all_hvps(x, tangent, *args, **kwargs)[eq_indices]

        parts.eq_fns.append(eq_fn)
        parts.eq_jacs.append(eq_jac)
        parts.eq_hvps.append(eq_hvp)
        parts.meq = int(eq_indices.size)

    if needs_ineq:
        lower_bounds = lb_array[lower_indices]
        upper_bounds = ub_array[upper_indices]

        def ineq_fn(x: Any, *args: Any, **kwargs: Any) -> Array:
            value = values(x, *args, **kwargs)
            return jnp.concatenate(
                [
                    lower_bounds - value[lower_indices],
                    value[upper_indices] - upper_bounds,
                ]
            )

        ineq_jac = None
        if jacobians is not None:

            def ineq_jac(x: Any, *args: Any, **kwargs: Any) -> Array:
                jacobian = jacobians(x, *args, **kwargs)
                return jnp.concatenate(
                    [-jacobian[lower_indices], jacobian[upper_indices]], axis=0
                )

        ineq_hvp = None
        if all_hvps is not None:

            def ineq_hvp(x: Any, tangent: Any, *args: Any, **kwargs: Any) -> Array:
                rows = all_hvps(x, tangent, *args, **kwargs)
                return jnp.concatenate(
                    [-rows[lower_indices], rows[upper_indices]], axis=0
                )

        parts.ineq_fns.append(ineq_fn)
        parts.ineq_jacs.append(ineq_jac)
        parts.ineq_hvps.append(ineq_hvp)
        parts.mineq = int(lower_indices.size + upper_indices.size)
    return parts


def _combine_values(
    functions: list[Callable[..., Any]],
) -> Callable[..., Any] | None:
    if not functions:
        return None
    if len(functions) == 1:
        return functions[0]

    def combined(x: Any, *args: Any, **kwargs: Any) -> Array:
        return jnp.concatenate([function(x, *args, **kwargs) for function in functions])

    return combined


def _combine_derivatives(
    functions: list[Callable[..., Any] | None],
) -> Callable[..., Any] | None:
    if not functions or any(function is None for function in functions):
        return None
    available = [function for function in functions if function is not None]
    if len(available) == 1:
        return available[0]

    def combined(x: Any, *args: Any, **kwargs: Any) -> Array:
        return jnp.concatenate(
            [function(x, *args, **kwargs) for function in available], axis=0
        )

    return combined


def parse_constraints(
    constraints: ConstraintSpec,
    x0: Array,
) -> ParsedConstraints:
    """Convert SciPy constraint objects to sqpdax's ``g=0`` / ``h<=0`` form."""
    if isinstance(constraints, (dict, LinearConstraint, NonlinearConstraint)):
        constraint_list = [constraints]
    elif isinstance(constraints, (list, tuple)):
        constraint_list = list(constraints)
    else:
        raise TypeError(
            f"Unsupported constraints type {type(constraints)}. Expected a dict, "
            "list, tuple, LinearConstraint, or NonlinearConstraint."
        )

    parts: list[_ConstraintParts] = []
    for constraint in constraint_list:
        if isinstance(constraint, dict):
            parts.append(_parse_dict_constraint(constraint, x0))
        elif isinstance(constraint, LinearConstraint):
            parts.append(_parse_linear_constraint(constraint))
        elif isinstance(constraint, NonlinearConstraint):
            parts.append(_parse_nonlinear_constraint(constraint, x0))
        else:
            raise TypeError(f"Unsupported constraint object type: {type(constraint)}")

    return ParsedConstraints(
        eq_fn=_combine_values([fn for part in parts for fn in part.eq_fns]),
        ineq_fn=_combine_values([fn for part in parts for fn in part.ineq_fns]),
        meq=sum(part.meq for part in parts),
        mineq=sum(part.mineq for part in parts),
        eq_fn_jac=_combine_derivatives([fn for part in parts for fn in part.eq_jacs]),
        ineq_fn_jac=_combine_derivatives(
            [fn for part in parts for fn in part.ineq_jacs]
        ),
        eq_fn_hvp=_combine_derivatives([fn for part in parts for fn in part.eq_hvps]),
        ineq_fn_hvp=_combine_derivatives(
            [fn for part in parts for fn in part.ineq_hvps]
        ),
    )


def _convert_bounds(
    bounds: Bounds | Sequence[tuple[Any, Any]] | None, n: int
) -> tuple[Array | None, Array | None]:
    """Convert and validate SciPy variable bounds outside compiled code."""
    if bounds is None:
        return None, None
    if isinstance(bounds, Bounds):
        try:
            lb = np.broadcast_to(np.asarray(bounds.lb, dtype=float), (n,))
            ub = np.broadcast_to(np.asarray(bounds.ub, dtype=float), (n,))
        except ValueError as error:
            raise ValueError(f"bounds cannot be broadcast to {n} variables") from error
    else:
        pairs = list(bounds)
        if len(pairs) != n:
            raise ValueError(f"bounds has {len(pairs)} entries but x0 has {n} elements")
        lb = np.empty((n,), dtype=float)
        ub = np.empty((n,), dtype=float)
        for index, pair in enumerate(pairs):
            try:
                lower, upper = pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"bounds entry {index} must be a (lower, upper) pair"
                ) from error
            lb[index] = -np.inf if lower is None else float(lower)
            ub[index] = np.inf if upper is None else float(upper)
    _validate_endpoint_arrays(lb, ub, name="bounds")
    return jnp.asarray(lb), jnp.asarray(ub)


def _prepare_options(
    options: Mapping[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    native_options = dict(options or {})
    max_steps = int(native_options.pop("maxiter", 100))
    return max_steps, native_options


def build_problem_from_scipy_specification(
    fun: Callable[..., Any],
    x0: Any,
    *,
    jac: Callable[..., Any] | bool | None = None,
    hessp: Callable[..., Any] | None = None,
    bounds: Bounds | Sequence[tuple[Any, Any]] | None = None,
    constraints: ConstraintSpec = (),
    has_aux: bool = False,
) -> Problem:
    """Build a native sqpdax problem from SciPy-style specifications.

    Parameters
    ----------
    fun
        Objective callable following SciPy's ``fun(x, *args)`` convention.
    x0
        Initial point, used for dimensions and constraint row discovery.
    jac
        Callable objective gradient, ``True`` when ``fun`` returns its
        gradient, or ``None`` for JAX autodiff.
    hessp
        Optional objective Hessian-vector product.
    bounds
        SciPy ``Bounds`` or a sequence of ``(lower, upper)`` pairs.
    constraints
        SciPy dictionary, linear, or nonlinear constraint specifications.
    has_aux
        Whether the objective also returns auxiliary data.

    Returns
    -------
    Problem
        Native sqpdax problem whose callables retain the SciPy ``*args``
        calling convention.
    """
    x0_array = jnp.asarray(x0, dtype=float)
    if x0_array.ndim != 1:
        raise ValueError(f"x0 must be one-dimensional; got shape {x0_array.shape}")
    parsed = parse_constraints(constraints, x0_array)
    lb, ub = _convert_bounds(bounds, x0_array.shape[0])

    wrapped_hessp: Callable[..., Any] | None = None
    if callable(hessp):
        user_hessp = hessp

        def user_objective_hvp(
            x: Any, tangent: Any, *problem_args: Any, **kwargs: Any
        ) -> Array:
            del kwargs
            return jnp.asarray(user_hessp(x, tangent, *problem_args))

        wrapped_hessp = user_objective_hvp

    if jac is True:
        if has_aux:

            def objective_with_jac_and_aux(
                x: Any, *problem_args: Any, **kwargs: Any
            ) -> tuple[Array, Any]:
                del kwargs
                (value, _gradient), aux = fun(x, *problem_args)
                return jnp.asarray(value), aux

            def gradient_with_aux(x: Any, *problem_args: Any, **kwargs: Any) -> Array:
                del kwargs
                (_value, value), _aux = fun(x, *problem_args)
                return jnp.asarray(value)

            raw_objective = objective_with_jac_and_aux
            raw_gradient = gradient_with_aux
        else:

            def objective_with_jac(x: Any, *problem_args: Any, **kwargs: Any) -> Array:
                del kwargs
                value, _gradient = fun(x, *problem_args)
                return jnp.asarray(value)

            def gradient_from_objective(
                x: Any, *problem_args: Any, **kwargs: Any
            ) -> Array:
                del kwargs
                _value, value = fun(x, *problem_args)
                return jnp.asarray(value)

            raw_objective = objective_with_jac
            raw_gradient = gradient_from_objective

        objective_fn, gradient_fn, objective_hvp = autodiff_wrapper(
            raw_objective,
            raw_gradient,
            wrapped_hessp,
            "custom",
            has_aux=has_aux,
        )
    else:
        if has_aux:

            def objective_with_aux(
                x: Any, *problem_args: Any, **kwargs: Any
            ) -> tuple[Array, Any]:
                del kwargs
                value, aux = fun(x, *problem_args)
                return jnp.asarray(value), aux

            raw_objective = objective_with_aux
        else:

            def scalar_objective(x: Any, *problem_args: Any, **kwargs: Any) -> Array:
                del kwargs
                return jnp.asarray(fun(x, *problem_args))

            raw_objective = scalar_objective

        user_gradient: Callable[..., Any] | None = None
        if callable(jac):

            def callable_gradient(x: Any, *problem_args: Any, **kwargs: Any) -> Array:
                del kwargs
                return jnp.asarray(jac(x, *problem_args))

            user_gradient = callable_gradient

        objective_fn, gradient_fn, objective_hvp = autodiff_wrapper(
            raw_objective,
            user_gradient,
            wrapped_hessp,
            "custom" if user_gradient is not None else "jax",
            has_aux=has_aux,
        )
    if wrapped_hessp is not None:
        objective_hvp = wrapped_hessp

    def resolve_constraint_block(
        fn: Callable[..., Any] | None,
        block_jac: Callable[..., Any] | None,
        block_hvp: Callable[..., Any] | None,
    ) -> tuple[
        Callable[..., Any] | None, Callable[..., Any] | None, Callable[..., Any] | None
    ]:
        if fn is None:
            return None, None, None
        resolved_fn, resolved_jac, resolved_hvp = autodiff_wrapper(
            fn,
            block_jac,
            block_hvp,
            "custom" if block_jac is not None else "jax",
        )
        # JAX mode deliberately ignores a supplied HVP callable. The SciPy
        # compatibility contract gives an explicit constraint HVP precedence.
        if block_hvp is not None:
            resolved_hvp = block_hvp
        return resolved_fn, resolved_jac, resolved_hvp

    eq_fn, eq_jac, eq_hvp = resolve_constraint_block(
        parsed.eq_fn, parsed.eq_fn_jac, parsed.eq_fn_hvp
    )
    ineq_fn, ineq_jac, ineq_hvp = resolve_constraint_block(
        parsed.ineq_fn, parsed.ineq_fn_jac, parsed.ineq_fn_hvp
    )
    return build_problem(
        n=x0_array.shape[0],
        meq=parsed.meq,
        mineq=parsed.mineq,
        fn=objective_fn,
        grad=gradient_fn,
        hvp=objective_hvp,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        eq_fn_jac=eq_jac,
        ineq_fn_jac=ineq_jac,
        eq_fn_hvp=eq_hvp,
        ineq_fn_hvp=ineq_hvp,
        lb=lb,
        ub=ub,
        autodiff_mode="none",
        force_hvp_in_jax_mode=False,
        has_aux=has_aux,
    )


def minimise_and_return_scipy_result(
    problem: ProblemProtocol[Any],
    x0: Any,
    args: tuple[Any, ...] = (),
    *,
    options: Mapping[str, Any] | None = None,
    throw: bool = False,
    solver: AbstractConstrainedMinimiser[Any, Any, Any, Any, Any] | None = None,
) -> OptimizeResult:
    """Run a native sqpdax problem and return a SciPy result.

    Parameters
    ----------
    problem
        Native problem to solve.
    x0
        Initial decision vector.
    args
        Positional arguments forwarded to the problem callables.
    options
        Solver options. ``maxiter`` controls the outer iteration budget;
        remaining values are forwarded unchanged to :func:`minimise`.
    throw
        Whether native solve failures should raise.
    solver
        Configured sqpdax minimiser. Defaults to the active-set line-search
        minimiser.

    Returns
    -------
    scipy.optimize.OptimizeResult
        SciPy result containing ``x``, ``fun``, ``jac``, ``nit``, ``success``,
        ``status``, and ``message`` plus sqpdax-specific diagnostics.
    """
    x0_array = jnp.asarray(x0, dtype=float)
    if x0_array.ndim != 1:
        raise ValueError(f"x0 must be one-dimensional; got shape {x0_array.shape}")
    chosen_solver = cast(
        AbstractConstrainedMinimiser[Any, Any, Any, Any, Any],
        solver if solver is not None else ActiveSetLineSearchMinimiser(),
    )
    max_steps, native_options = _prepare_options(options)
    solution = minimise(
        problem,
        chosen_solver,
        x0_array,
        max_steps=max_steps,
        throw=throw,
        options=native_options,
        problem_args=args,
    )
    gradient = problem.grad(solution.value, *args)
    success = bool(solution.state.result_adapter.is_successful(solution.result))
    max_steps_reached = bool(
        solution.result == solution.state.result_adapter.max_steps_reached
    )
    status = 0 if success else 1 if max_steps_reached else 2
    message = solution.state.result_adapter.result_type[solution.result]
    return OptimizeResult(
        x=np.asarray(solution.value),
        fun=float(np.asarray(solution.stats["final_objective"])),
        jac=np.asarray(gradient),
        nit=int(np.asarray(solution.stats["num_steps"])),
        success=success,
        status=status,
        message=message,
        aux=solution.aux,
        sqpdax_result=solution.result,
        stats=solution.stats,
    )


def minimize_like_scipy(
    fun: Callable[..., Any],
    x0: Any,
    args: tuple[Any, ...] = (),
    *,
    jac: Callable[..., Any] | bool | None = None,
    hessp: Callable[..., Any] | None = None,
    bounds: Bounds | Sequence[tuple[Any, Any]] | None = None,
    constraints: ConstraintSpec = (),
    options: Mapping[str, Any] | None = None,
    has_aux: bool = False,
    throw: bool = False,
    solver: AbstractConstrainedMinimiser[Any, Any, Any, Any, Any] | None = None,
) -> OptimizeResult:
    """Build and solve a SciPy-style specification with sqpdax.

    Parameters
    ----------
    fun, x0, args, jac, hessp, bounds, constraints, options
        Match the corresponding SciPy minimisation inputs.
    has_aux
        Whether ``fun`` returns auxiliary data.
    throw
        Whether native solve failures should raise.
    solver
        Optional configured sqpdax minimiser.

    Returns
    -------
    scipy.optimize.OptimizeResult
        SciPy-native optimisation result.
    """
    problem = build_problem_from_scipy_specification(
        fun,
        x0,
        jac=jac,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        has_aux=has_aux,
    )
    return minimise_and_return_scipy_result(
        problem,
        x0,
        args,
        options=options,
        throw=throw,
        solver=solver,
    )
