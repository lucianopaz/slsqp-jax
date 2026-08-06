"""Factory for assembling a :class:`~slsqp_jax.sqpdax.problem.basic.Problem`."""

from typing import Literal

from jax import numpy as jnp

from ..autodiff_utils import autodiff_wrapper
from ..types import (
    EqConstraintFn,
    EqConstraintHVPFn,
    EqConstraintJacFn,
    IneqConstraintFn,
    IneqConstraintHVPFn,
    IneqConstraintJacFn,
    Matrix_meqn,
    Matrix_mineqn,
    ObjectiveFn,
    ObjectiveGradFn,
    ObjectiveHVPFn,
    Vector_meq,
    Vector_mineq,
    Vector_n,
)
from .basic import Problem


def build_problem(
    n: int,
    meq: int | None,
    mineq: int | None,
    fn: ObjectiveFn,
    grad: ObjectiveGradFn,
    hvp: ObjectiveHVPFn | None,
    eq_fn: EqConstraintFn | None,  # g(x) = 0
    ineq_fn: IneqConstraintFn | None,  # h(x) <= 0
    eq_fn_jac: EqConstraintJacFn | None,
    ineq_fn_jac: IneqConstraintJacFn | None,
    eq_fn_hvp: EqConstraintHVPFn | None,
    ineq_fn_hvp: IneqConstraintHVPFn | None,
    lb: Vector_n | None,
    ub: Vector_n | None,
    autodiff_mode: Literal["jax", "custom", "none"] = "custom",
    force_hvp_in_jax_mode: bool = False,
) -> Problem:
    """Assemble an NLP :class:`~slsqp_jax.sqpdax.problem.basic.Problem`.

    Wraps the objective and optional equality / inequality callables through
    :func:`~slsqp_jax.sqpdax.autodiff_utils.autodiff_wrapper`, fills in empty
    constraint stubs when a constraint family is omitted, and materialises
    infinite box bounds (with matching ``null_lb`` / ``null_ub`` masks) when
    ``lb`` or ``ub`` is ``None``.

    Equality constraints are written ``g(x) = 0``; inequalities as
    ``h(x) <= 0``. Missing constraint families become zero-length vector /
    Jacobian / HVP callables so downstream code can treat every problem as
    having all three blocks.

    Parameters
    ----------
    n
        Number of decision variables.
    meq
        Number of equality constraints. Required (must not be ``None``) when
        ``eq_fn`` is given; ignored and forced to ``0`` when ``eq_fn`` is
        ``None``.
    mineq
        Number of inequality constraints. Required when ``ineq_fn`` is given;
        forced to ``0`` when ``ineq_fn`` is ``None``.
    fn
        Scalar objective ``fn(x, *args, **kwargs)``.
    grad
        Gradient of ``fn`` w.r.t. ``x``. Required when ``autodiff_mode`` is
        ``"custom"`` or ``"none"``; ignored when ``autodiff_mode="jax"``.
    hvp
        Optional Hessian-vector product of ``fn``.
    eq_fn
        Equality residual ``g(x)`` of length ``meq``, or ``None``.
    ineq_fn
        Inequality residual ``h(x)`` of length ``mineq``, or ``None``.
    eq_fn_jac
        Jacobian of ``eq_fn`` with shape ``(meq, n)``. Required under
        ``"custom"`` / ``"none"`` when ``eq_fn`` is given.
    ineq_fn_jac
        Jacobian of ``ineq_fn`` with shape ``(mineq, n)``. Required under
        ``"custom"`` / ``"none"`` when ``ineq_fn`` is given.
    eq_fn_hvp
        Optional directional derivative of ``eq_fn_jac``.
    ineq_fn_hvp
        Optional directional derivative of ``ineq_fn_jac``.
    lb
        Lower bounds of length ``n``, or ``None`` for all ``-inf``.
    ub
        Upper bounds of length ``n``, or ``None`` for all ``+inf``.
    autodiff_mode
        Derivative source forwarded to
        :func:`~slsqp_jax.sqpdax.autodiff_utils.autodiff_wrapper` for the
        objective and each provided constraint family:
        ``"jax"``, ``"custom"`` (default), or ``"none"``.
    force_hvp_in_jax_mode
        Whether to force the use of the Hessian-vector product in JAX mode.
        Forwarded to :func:`~slsqp_jax.sqpdax.autodiff_utils.autodiff_wrapper`.

    Returns
    -------
    Problem
        Equinox module holding the resolved callables, bounds, null-bound
        masks, and dimensions ``(n, meq, mineq)``.

    Raises
    ------
    ValueError
        If autodiff setup fails for the objective or a constraint family
        (message prefixed with the failing block), or if ``autodiff_mode``
        is invalid / missing Jacobians as documented by
        :func:`~slsqp_jax.sqpdax.autodiff_utils.autodiff_wrapper`.
    AssertionError
        If ``eq_fn`` is given but ``meq`` is ``None``, or ``ineq_fn`` is
        given but ``mineq`` is ``None``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.problem.builder import build_problem
    >>> def f(x):
    ...     return jnp.sum(x**2)
    >>> def f_grad(x):
    ...     return 2 * x
    >>> problem = build_problem(
    ...     n=2,
    ...     meq=None,
    ...     mineq=None,
    ...     fn=f,
    ...     grad=f_grad,
    ...     hvp=None,
    ...     eq_fn=None,
    ...     ineq_fn=None,
    ...     eq_fn_jac=None,
    ...     ineq_fn_jac=None,
    ...     eq_fn_hvp=None,
    ...     ineq_fn_hvp=None,
    ...     lb=None,
    ...     ub=None,
    ...     autodiff_mode="none",
    ... )
    >>> problem.n, problem.meq, problem.mineq
    (2, 0, 0)
    >>> float(problem.fn(jnp.array([1.0, 2.0])))
    5.0
    >>> problem.eq_fn(jnp.array([1.0, 2.0])).shape
    (0,)
    """
    fn, grad, hvp = autodiff_wrapper(
        fn, grad, hvp, autodiff_mode, force_hvp_in_jax_mode
    )
    if eq_fn is not None:
        try:
            eq_fn, eq_fn_jac, eq_fn_hvp = autodiff_wrapper(
                eq_fn, eq_fn_jac, eq_fn_hvp, autodiff_mode
            )
        except ValueError as e:
            raise ValueError("Failed to autodiff equality constraint: " + str(e)) from e
        assert meq is not None, (
            "When equality constraints are given, you must also specify the "
            "number of equality constraint functions, meq"
        )
    else:
        meq = 0

        def eq_fn(x: Vector_n, *args, **kwargs) -> Vector_meq:
            return jnp.zeros(shape=(0,), dtype=x.dtype)

        def eq_fn_jac(x: Vector_n, *args, **kwargs) -> Matrix_meqn:
            return jnp.zeros(shape=(0, n), dtype=x.dtype)

        def eq_fn_hvp(x: Vector_n, p: Vector_n, *args, **kwargs) -> Matrix_meqn:
            return jnp.zeros(shape=(0, n), dtype=x.dtype)

    if ineq_fn is not None:
        try:
            ineq_fn, ineq_fn_jac, ineq_fn_hvp = autodiff_wrapper(
                ineq_fn, ineq_fn_jac, ineq_fn_hvp, autodiff_mode
            )
        except ValueError as e:
            raise ValueError(
                "Failed to autodiff inequality constraint: " + str(e)
            ) from e
        assert mineq is not None, (
            "When inequality constraints are given, you must also specify the "
            "number of inequality constraint functions, mineq"
        )
    else:
        mineq = 0

        def ineq_fn(x: Vector_n, *args, **kwargs) -> Vector_mineq:
            return jnp.zeros(shape=(0,), dtype=x.dtype)

        def ineq_fn_jac(x: Vector_n, *args, **kwargs) -> Matrix_mineqn:
            return jnp.zeros(shape=(0, n), dtype=x.dtype)

        def ineq_fn_hvp(x: Vector_n, p: Vector_n, *args, **kwargs) -> Matrix_mineqn:
            return jnp.zeros(shape=(0, n), dtype=x.dtype)

    _lb = lb if lb is not None else jnp.full(shape=(n,), fill_value=-jnp.inf)
    _ub = ub if ub is not None else jnp.full(shape=(n,), fill_value=jnp.inf)
    null_lb = jnp.isinf(_lb) & (_lb < 0)
    null_ub = jnp.isinf(_ub) & (_ub > 0)
    return Problem(  # ty: ignore[invalid-return-type]
        fn=fn,
        grad=grad,
        hvp=hvp,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        eq_fn_jac=eq_fn_jac,
        ineq_fn_jac=ineq_fn_jac,
        eq_fn_hvp=eq_fn_hvp,
        ineq_fn_hvp=ineq_fn_hvp,
        lb=_lb,
        ub=_ub,
        null_lb=null_lb,
        null_ub=null_ub,
        n=n,
        meq=meq,
        mineq=mineq,
    )
