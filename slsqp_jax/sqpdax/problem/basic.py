"""NLP problem containers: unevaluated callables and pointwise evaluations."""

from typing import Callable, Generic, Protocol, cast, runtime_checkable

from equinox import Module, field
from jaxtyping import Array, Bool

from ..primal import PrimalType
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
    Scalar,
    Vector_meq,
    Vector_mineq,
    Vector_n,
)

__all__ = [
    "EvaluatedProblem",
    "ProblemProtocol",
    "Problem",
]


class EvaluatedProblem(Module, Generic[PrimalType]):
    """Objective and constraint values at a fixed primal point.

    Produced by :meth:`Problem.__call__`. First-order quantities are always
    stored as arrays; second-order directional derivatives are optional
    closures ``q |-> HVP(x, q)`` bound to the evaluation point (``None`` when
    the parent problem lacks exact curvature).

    Attributes
    ----------
    ref
        Primal at which the problem was evaluated.
    fn_val
        Objective value ``f(x)``.
    grad_val
        Objective gradient ``∇f(x)``.
    fn_qvp
        Optional map ``p ↦ ∇²f(x) p``, or ``None``.
    eq_fn_val
        Equality residual ``g(x)``.
    eq_fn_jac_val
        Equality Jacobian ``∇g(x)`` with shape ``(meq, n)``.
    eq_fn_qvp
        Optional map ``p ↦ D(∇g)(x)[p]``, or ``None``.
    ineq_fn_val
        Inequality residual ``h(x)``.
    ineq_fn_jac_val
        Inequality Jacobian ``∇h(x)`` with shape ``(mineq, n)``.
    ineq_fn_qvp
        Optional map ``p ↦ D(∇h)(x)[p]``, or ``None``.
    lb
        Lower bounds copied from the parent problem.
    ub
        Upper bounds copied from the parent problem.
    null_lb
        Mask of inactive (``-inf``) lower bounds.
    null_ub
        Mask of inactive (``+inf``) upper bounds.
    """

    ref: PrimalType
    fn_val: Scalar
    grad_val: Vector_n
    fn_qvp: Callable[[Vector_n], Vector_n] | None = field(static=True)
    eq_fn_val: Vector_meq
    eq_fn_jac_val: Matrix_meqn
    eq_fn_qvp: Callable[[Vector_n], Matrix_meqn] | None = field(static=True)
    ineq_fn_val: Vector_mineq
    ineq_fn_jac_val: Matrix_mineqn
    ineq_fn_qvp: Callable[[Vector_n], Matrix_mineqn] | None = field(static=True)
    lb: Vector_n
    ub: Vector_n
    null_lb: Bool[Array, " n"]
    null_ub: Bool[Array, " n"]

    @property
    def n(self) -> int:
        """Number of decision variables."""
        return self.ref.x.shape[-1]

    @property
    def meq(self) -> int:
        """Number of equality constraints."""
        return self.eq_fn_val.shape[-1]

    @property
    def mineq(self) -> int:
        """Number of inequality constraints."""
        return self.ineq_fn_val.shape[-1]

    @property
    def has_exact_curvature(self) -> bool:
        """Whether all three directional-curvature maps are available."""
        return (
            self.fn_qvp is not None
            and self.eq_fn_qvp is not None
            and self.ineq_fn_qvp is not None
        )


@runtime_checkable
class ProblemProtocol(Protocol, Generic[PrimalType]):
    """Structural interface shared by NLP problem containers.

    Downstream solvers depend on these fields and on
    ``__call__(x, *args, **kwargs) -> EvaluatedProblem`` rather than on a
    concrete class, so alternate problem representations can satisfy the
    protocol without subclassing :class:`Problem`.
    """

    fn: ObjectiveFn
    grad: ObjectiveGradFn
    hvp: ObjectiveHVPFn | None
    eq_fn: EqConstraintFn
    eq_fn_jac: EqConstraintJacFn
    eq_fn_hvp: EqConstraintHVPFn | None
    ineq_fn: IneqConstraintFn
    ineq_fn_jac: IneqConstraintJacFn
    ineq_fn_hvp: IneqConstraintHVPFn | None
    lb: Vector_n
    ub: Vector_n
    null_lb: Bool[Array, " n"]
    null_ub: Bool[Array, " n"]
    n: int
    meq: int
    mineq: int
    has_exact_curvature: bool

    def __call__(
        self, x: PrimalType, *args, **kwargs
    ) -> EvaluatedProblem[PrimalType]: ...

    @property
    def has_exact_curvature(self) -> bool: ...


class Problem(Module):
    """Unevaluated NLP: objective, constraints, bounds, and derivatives.

    Equality constraints are ``g(x) = 0``; inequalities are ``h(x) <= 0``.
    Prefer :func:`~slsqp_jax.sqpdax.problem.builder.build_problem` to
    construct instances with autodiff wiring and empty-constraint stubs.

    Attributes
    ----------
    fn
        Scalar objective ``f(x, *args, **kwargs)``.
    grad
        Objective gradient w.r.t. ``x``.
    hvp
        Optional objective Hessian-vector product, or ``None``.
    eq_fn
        Equality residual ``g(x)``.
    ineq_fn
        Inequality residual ``h(x)``.
    eq_fn_jac
        Equality Jacobian with shape ``(meq, n)``.
    ineq_fn_jac
        Inequality Jacobian with shape ``(mineq, n)``.
    eq_fn_hvp
        Optional directional derivative of ``eq_fn_jac``, or ``None``.
    ineq_fn_hvp
        Optional directional derivative of ``ineq_fn_jac``, or ``None``.
    lb
        Lower bounds of length ``n``.
    ub
        Upper bounds of length ``n``.
    null_lb
        Mask of inactive (``-inf``) lower bounds.
    null_ub
        Mask of inactive (``+inf``) upper bounds.
    n
        Number of decision variables.
    meq
        Number of equality constraints.
    mineq
        Number of inequality constraints.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.primal import Primal
    >>> from slsqp_jax.sqpdax.problem.basic import Problem
    >>> def f(x):
    ...     return jnp.sum(x**2)
    >>> def g(x):
    ...     return jnp.zeros((0,), dtype=x.dtype)
    >>> problem = Problem(
    ...     fn=f,
    ...     grad=lambda x: 2 * x,
    ...     hvp=None,
    ...     eq_fn=g,
    ...     ineq_fn=g,
    ...     eq_fn_jac=lambda x: jnp.zeros((0, x.shape[-1]), dtype=x.dtype),
    ...     ineq_fn_jac=lambda x: jnp.zeros((0, x.shape[-1]), dtype=x.dtype),
    ...     eq_fn_hvp=None,
    ...     ineq_fn_hvp=None,
    ...     lb=jnp.full((2,), -jnp.inf),
    ...     ub=jnp.full((2,), jnp.inf),
    ...     null_lb=jnp.array([True, True]),
    ...     null_ub=jnp.array([True, True]),
    ...     n=2,
    ...     meq=0,
    ...     mineq=0,
    ... )
    >>> ev = problem(Primal(x=jnp.array([1.0, 2.0])))
    >>> float(ev.fn_val), ev.has_exact_curvature
    (5.0, False)
    """

    fn: ObjectiveFn = field(static=True)
    grad: ObjectiveGradFn = field(static=True)
    hvp: ObjectiveHVPFn | None = field(static=True)
    eq_fn: EqConstraintFn = field(static=True)  # g(x) = 0
    ineq_fn: IneqConstraintFn = field(static=True)  # h(x) <= 0
    eq_fn_jac: EqConstraintJacFn = field(static=True)
    ineq_fn_jac: IneqConstraintJacFn = field(static=True)
    eq_fn_hvp: EqConstraintHVPFn | None = field(static=True)
    ineq_fn_hvp: IneqConstraintHVPFn | None = field(static=True)
    lb: Vector_n
    ub: Vector_n
    null_lb: Bool[Array, " n"]
    null_ub: Bool[Array, " n"]
    n: int
    meq: int
    mineq: int

    def __call__(self, x: PrimalType, *args, **kwargs) -> EvaluatedProblem[PrimalType]:
        """Evaluate the NLP at ``x``, forwarding ``*args`` / ``**kwargs``.

        When :attr:`has_exact_curvature` is true, the returned
        :class:`EvaluatedProblem` carries closures that apply each HVP at
        the fixed primal ``x.x``. Otherwise the three ``*_qvp`` fields are
        ``None``.

        Parameters
        ----------
        x
            Primal decision point.
        *args, **kwargs
            Extra arguments forwarded to every objective / constraint
            callable.

        Returns
        -------
        EvaluatedProblem
            Pointwise values, Jacobians, bounds, and optional QVPs.
        """
        fn_val = self.fn(x.x, *args, **kwargs)
        grad_val = self.grad(x.x, *args, **kwargs)

        fn_qvp = None
        if self.has_exact_curvature:
            _hvp = cast(ObjectiveHVPFn, self.hvp)

            def fn_qvp(p: Vector_n) -> Vector_n:
                return _hvp(x.x, p, *args, **kwargs)

        eq_fn_val = self.eq_fn(x.x, *args, **kwargs)
        eq_fn_jac_val = self.eq_fn_jac(x.x, *args, **kwargs)
        eq_fn_qvp = None
        if self.has_exact_curvature:
            _eq_fn_hvp = cast(EqConstraintHVPFn, self.eq_fn_hvp)

            def eq_fn_qvp(p: Vector_n) -> Matrix_meqn:
                return _eq_fn_hvp(x.x, p, *args, **kwargs)

        ineq_fn_val = self.ineq_fn(x.x, *args, **kwargs)
        ineq_fn_jac_val = self.ineq_fn_jac(x.x, *args, **kwargs)
        ineq_fn_qvp = None
        if self.has_exact_curvature:
            _ineq_fn_hvp = cast(IneqConstraintHVPFn, self.ineq_fn_hvp)

            def ineq_fn_qvp(p: Vector_n) -> Matrix_mineqn:
                return _ineq_fn_hvp(x.x, p, *args, **kwargs)

        return cast(
            EvaluatedProblem[PrimalType],
            EvaluatedProblem(
                ref=x,
                fn_val=fn_val,
                grad_val=grad_val,
                fn_qvp=fn_qvp,
                eq_fn_val=eq_fn_val,
                eq_fn_jac_val=eq_fn_jac_val,
                eq_fn_qvp=eq_fn_qvp,
                ineq_fn_val=ineq_fn_val,
                ineq_fn_jac_val=ineq_fn_jac_val,
                ineq_fn_qvp=ineq_fn_qvp,
                lb=self.lb,
                ub=self.ub,
                null_lb=self.null_lb,
                null_ub=self.null_ub,
            ),
        )

    @property
    def has_exact_curvature(self) -> bool:
        """Whether objective and both constraint HVPs are present."""
        return (
            self.hvp is not None
            and self.eq_fn_hvp is not None
            and self.ineq_fn_hvp is not None
        )
