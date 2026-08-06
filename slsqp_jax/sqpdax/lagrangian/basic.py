"""Unevaluated Lagrangian wrappers over an NLP problem."""

from typing import Generic, cast

from equinox import Module, tree_at
from jaxtyping import Array, Float

from ..dual import Dual
from ..primal import PrimalType
from ..problem.basic import ProblemProtocol
from ..secant import Secant
from ..types import Scalar, Vector_n
from .evaluated import EvaluatedLagrangian, EvaluatedLagrangianType

__all__ = ["Lagrangian"]


class Lagrangian(Module, Generic[PrimalType, EvaluatedLagrangianType]):
    """Unevaluated Lagrangian ``L(x, λ)`` built from a problem and optional secant.

    Calling an instance evaluates the underlying
    :class:`~slsqp_jax.sqpdax.problem.basic.ProblemProtocol` at the primal
    and returns an :class:`~slsqp_jax.sqpdax.lagrangian.evaluated.EvaluatedLagrangian`
    (or a specialized subclass) with cached values, gradients, and KKT
    products. Methods such as :meth:`value` and :meth:`kkt_mvp` are thin
    facades that evaluate and then delegate.

    The type parameters ``PrimalType`` and ``EvaluatedLagrangianType`` pair
    the primal flavour with its evaluated Lagrangian. The default
    implementation constructs a plain
    :class:`~slsqp_jax.sqpdax.lagrangian.evaluated.EvaluatedLagrangian`;
    subclasses (e.g. interior-point) override :meth:`__call__`.

    Attributes
    ----------
    problem
        NLP problem providing objective, constraints, and bounds.
    secant
        Optional matrix-free Hessian approximation used when the problem
        lacks exact curvature. Required when
        ``problem.has_exact_curvature`` is false.
    """

    problem: ProblemProtocol[PrimalType]
    secant: Secant | None

    def __init__(
        self, problem: ProblemProtocol[PrimalType], secant: Secant | None = None
    ):
        """Attach ``problem`` and an optional secant approximation.

        Parameters
        ----------
        problem
            NLP problem to wrap.
        secant
            Secant Hessian, or ``None`` when the problem supplies exact HVPs.

        Raises
        ------
        TypeError
            If ``secant is None`` and ``problem.has_exact_curvature`` is false.
        """
        if secant is None and not problem.has_exact_curvature:
            raise TypeError("secant is required for problems without exact curvature")
        self.problem = problem
        self.secant = secant

    def objective_fn(self, x: PrimalType, *args, **kwargs) -> Scalar:
        """Objective value ``f(x)`` via the wrapped problem.

        Parameters
        ----------
        x
            Primal point.
        *args, **kwargs
            Forwarded to the problem callables.

        Returns
        -------
        Scalar
            Objective value.
        """
        return self.problem.fn(x.x, *args, **kwargs)

    def objective_grad(self, x: PrimalType, *args, **kwargs) -> Vector_n:
        """Objective gradient ``∇f(x)``.

        Parameters
        ----------
        x
            Primal point.
        *args, **kwargs
            Forwarded to the problem callables.

        Returns
        -------
        Vector_n
            Objective gradient.
        """
        return self.problem.grad(x.x, *args, **kwargs)

    def __call__(
        self, x: PrimalType, d: Dual, *args, **kwargs
    ) -> EvaluatedLagrangianType:
        """Evaluate the Lagrangian at ``(x, d)``.

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to the problem evaluation.

        Returns
        -------
        EvaluatedLagrangianType
            Cached Lagrangian at the reference point.
        """
        # Default packs a plain EvaluatedLagrangian; specialized subclasses
        # (e.g. interior-point) override ``__call__`` to build their evaluated type.
        return cast(
            EvaluatedLagrangianType,
            EvaluatedLagrangian(
                evaluated=self.problem(x, *args, **kwargs),
                secant=self.secant,
                dual=d,
            ),
        )

    def value(self, x: PrimalType, d: Dual, *args, **kwargs) -> Scalar:
        """Scalar Lagrangian value ``L(x, λ)``.

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Scalar
            ``L(x, λ)``.
        """
        return self(x, d, *args, **kwargs).value

    def x_grad(self, x: PrimalType, d: Dual, *args, **kwargs) -> Vector_n:
        """Partial gradient ``∇_x L(x, λ)`` as a flat vector.

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Vector_n
            Decision-variable block of the Lagrangian gradient.
        """
        return self(x, d, *args, **kwargs).x_grad

    def primal_grad(self, x: PrimalType, d: Dual, *args, **kwargs) -> PrimalType:
        """Primal-block Lagrangian gradient packed as a :class:`~slsqp_jax.sqpdax.primal.Primal`.

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        PrimalType
            Primal-shaped gradient.
        """
        return cast(PrimalType, self(x, d, *args, **kwargs).primal_grad)

    def dual_grad(self, x: PrimalType, d: Dual, *args, **kwargs) -> Dual:
        """Dual-block residual ``∇_λ L`` (constraint / bound violations).

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Dual
            Dual-shaped residual.
        """
        return self(x, d, *args, **kwargs).dual_grad

    def nonbound_constraint_jac(
        self, x: PrimalType, d: Dual, *args, **kwargs
    ) -> Float[Array, " meq+mineq n"]:
        """Stacked equality / inequality Jacobian at ``x``.

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers (unused by the base Jacobian; accepted for API
            uniformity).
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        jax.Array
            Array of shape ``(meq + mineq, n)``.
        """
        return self(x, d, *args, **kwargs).nonbound_constraint_jac

    def grad(self, x: PrimalType, d: Dual, *args, **kwargs) -> tuple[PrimalType, Dual]:
        """Full primal-dual Lagrangian gradient ``(∇_x L, ∇_λ L)``.

        Parameters
        ----------
        x
            Primal point.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        tuple[PrimalType, Dual]
            Primal and dual gradient blocks.
        """
        return cast(tuple[PrimalType, Dual], self(x, d, *args, **kwargs).grad)

    def primal_hvp(
        self, primal: PrimalType, d: Dual, tangent: PrimalType, *args, **kwargs
    ) -> PrimalType:
        """Lagrangian Hessian-vector product in the primal block.

        Parameters
        ----------
        primal
            Primal evaluation point.
        d
            Dual multipliers (enter the Lagrangian Hessian via constraint
            curvature weighted by multipliers).
        tangent
            Primal-shaped tangent.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        PrimalType
            ``∇²_{xx} L[tangent]``.
        """
        return cast(PrimalType, self(primal, d, *args, **kwargs).primal_hvp(tangent))

    def kkt_mvp_primal(
        self,
        x: PrimalType,
        d: Dual,
        tangent: tuple[PrimalType, Dual],
        *args,
        **kwargs,
    ) -> PrimalType:
        """Primal diagonal block of the KKT matrix-vector product.

        Parameters
        ----------
        x
            Primal evaluation point.
        d
            Dual multipliers at the linearisation.
        tangent
            ``(dx, dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        PrimalType
            Primal-block contribution.
        """
        return cast(PrimalType, self(x, d, *args, **kwargs).kkt_mvp_primal(tangent))

    def kkt_mvp_upper_offdiag(
        self,
        x: PrimalType,
        d: Dual,
        tangent: tuple[PrimalType, Dual],
        *args,
        **kwargs,
    ) -> PrimalType:
        """Upper off-diagonal (constraint Jacobianᵀ) KKT block times ``dλ``.

        Parameters
        ----------
        x
            Primal evaluation point.
        d
            Dual multipliers at the linearisation.
        tangent
            ``(dx, dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        PrimalType
            Contribution of the transposed constraint Jacobian.
        """
        return cast(
            PrimalType, self(x, d, *args, **kwargs).kkt_mvp_upper_offdiag(tangent)
        )

    def kkt_mvp_lower_offdiag(
        self,
        x: PrimalType,
        d: Dual,
        tangent: tuple[PrimalType, Dual],
        *args,
        **kwargs,
    ) -> Dual:
        """Lower off-diagonal (constraint Jacobian) KKT block times ``dx``.

        Parameters
        ----------
        x
            Primal evaluation point.
        d
            Dual multipliers at the linearisation.
        tangent
            ``(dx, dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Dual
            Dual-block contribution.
        """
        return self(x, d, *args, **kwargs).kkt_mvp_lower_offdiag(tangent)

    def kkt_mvp_dual(
        self,
        x: PrimalType,
        d: Dual,
        tangent: tuple[PrimalType, Dual],
        *args,
        **kwargs,
    ) -> Dual:
        """Dual diagonal KKT block (zero for the standard SQP Lagrangian).

        Parameters
        ----------
        x
            Primal evaluation point.
        d
            Dual multipliers at the linearisation.
        tangent
            ``(dx, dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Dual
            Dual-dual contribution (zeros in the base class).
        """
        return self(x, d, *args, **kwargs).kkt_mvp_dual(tangent)

    def kkt_mvp(
        self,
        x: PrimalType,
        d: Dual,
        tangent: tuple[PrimalType, Dual],
        *args,
        **kwargs,
    ) -> tuple[PrimalType, Dual]:
        """Full KKT saddle-point matrix-vector product.

        Parameters
        ----------
        x
            Primal evaluation point.
        d
            Dual multipliers at the linearisation.
        tangent
            ``(dx, dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        tuple[PrimalType, Dual]
            ``(primal_row, dual_row)`` of the KKT product.
        """
        return cast(
            tuple[PrimalType, Dual], self(x, d, *args, **kwargs).kkt_mvp(tangent)
        )

    def curvature_estimate(
        self, x: PrimalType, prev: EvaluatedLagrangianType, *args, **kwargs
    ) -> Vector_n:
        """Secant pair ``y = ∇_x L(x_new, λ) - ∇_x L(x_prev, λ)`` at shared ``λ``.

        Rebuilds the Lagrangian at ``x`` using ``prev.dual`` so the bound
        Jacobian contribution cancels (Nocedal & Wright §18.3).

        Parameters
        ----------
        x
            New primal point (only ``x.x`` is used; other primal fields are
            taken from ``prev.ref`` via :func:`equinox.tree_at`).
        prev
            Evaluated Lagrangian at the previous iterate.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Vector_n
            Curvature vector ``y`` for a secant update.
        """
        x_new = cast(PrimalType, tree_at(lambda p: p.x, prev.ref, x.x))
        return self(x_new, prev.dual, *args, **kwargs).curvature_estimate(prev)
