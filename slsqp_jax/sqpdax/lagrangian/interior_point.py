"""Interior-point Lagrangian with a barrier on slack variables."""

from typing import cast

from ..barrier.base import Barrier
from ..dual import Dual
from ..primal import InteriorPointPrimal
from ..problem.basic import ProblemProtocol
from ..secant import Secant
from ..types import Scalar, Vector_n
from .basic import Lagrangian
from .evaluated import InteriorPointEvaluatedLagrangian

__all__ = ["InteriorPointLagrangian"]


class InteriorPointLagrangian(
    Lagrangian[InteriorPointPrimal, InteriorPointEvaluatedLagrangian]
):
    """Lagrangian for primal / primal-dual interior-point SQP.

    Evaluates to
    :class:`~slsqp_jax.sqpdax.lagrangian.evaluated.InteriorPointEvaluatedLagrangian`,
    which folds a :class:`~slsqp_jax.sqpdax.barrier.Barrier` evaluated at the
    current slacks into the objective and KKT operator. When
    ``primal_dual=True`` the barrier must be a
    :class:`~slsqp_jax.sqpdax.barrier.LogBarrier` (enforced at evaluation
    time).

    Attributes
    ----------
    problem
        NLP problem; ``__call__`` must accept an
        :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal`.
    secant
        Optional secant Hessian for the decision-variable block.
    barrier
        Unevaluated barrier (weight / null masks) applied to slacks.
    dual_kkt_regularization
        Nonnegative dual-dual regularization added to equality rows of the
        KKT operator (see Nocedal & Wright §19.3).
    primal_dual
        If ``True``, use the primal-dual slack-slack block
        (``Λ S^{-1}`` for the log barrier) instead of the pure primal
        barrier Hessian.
    """

    problem: ProblemProtocol[InteriorPointPrimal]
    secant: Secant | None
    barrier: Barrier
    dual_kkt_regularization: float = 0.0
    primal_dual: bool = False

    def __init__(
        self,
        problem: ProblemProtocol[InteriorPointPrimal],
        secant: Secant | None,
        barrier: Barrier,
        dual_kkt_regularization: float = 0.0,
        primal_dual: bool = False,
    ):
        """Attach problem, optional secant, and barrier.

        Parameters
        ----------
        problem
            NLP problem evaluated at interior-point primals.
        secant
            Secant Hessian, or ``None`` when the problem has exact curvature.
        barrier
            Barrier applied to inequality / bound slacks.
        dual_kkt_regularization
            Dual-dual regularization strength (must be nonnegative).
        primal_dual
            Select the primal-dual KKT slack block when ``True``.

        Raises
        ------
        TypeError
            If ``secant is None`` and the problem lacks exact curvature.
        ValueError
            If ``dual_kkt_regularization < 0``.
        """
        if secant is None and not problem.has_exact_curvature:
            raise TypeError("secant is required for problems without exact curvature")
        if dual_kkt_regularization < 0:
            raise ValueError("dual_kkt_regularization must be non-negative")
        self.problem = problem
        self.secant = secant
        self.barrier = barrier
        self.dual_kkt_regularization = dual_kkt_regularization
        self.primal_dual = primal_dual

    def __call__(
        self, x: InteriorPointPrimal, d: Dual, *args, **kwargs
    ) -> InteriorPointEvaluatedLagrangian:
        """Evaluate the barrier-augmented Lagrangian at ``(x, d)``.

        Parameters
        ----------
        x
            Interior-point primal (decision variables + slacks).
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to the problem evaluation.

        Returns
        -------
        InteriorPointEvaluatedLagrangian
            Cached IP Lagrangian at the reference point.
        """
        return cast(
            InteriorPointEvaluatedLagrangian,
            InteriorPointEvaluatedLagrangian(
                evaluated=self.problem(x, *args, **kwargs),
                secant=self.secant,
                dual=d,
                barrier=self.barrier(x.slack),
                dual_kkt_regularization=self.dual_kkt_regularization,
                primal_dual=self.primal_dual,
            ),
        )

    def objective_fn(self, x: InteriorPointPrimal, *args, **kwargs) -> Scalar:
        """Objective value ``f(x)`` (barrier excluded).

        Parameters
        ----------
        x
            Interior-point primal.
        *args, **kwargs
            Forwarded to the problem callables.

        Returns
        -------
        Scalar
            Objective value at ``x.x``.
        """
        return self.problem.fn(x.x, *args, **kwargs)

    def objective_grad(self, x: InteriorPointPrimal, *args, **kwargs) -> Vector_n:
        """Objective gradient ``∇f(x)``.

        Parameters
        ----------
        x
            Interior-point primal.
        *args, **kwargs
            Forwarded to the problem callables.

        Returns
        -------
        Vector_n
            Objective gradient.
        """
        return self.problem.grad(x.x, *args, **kwargs)

    def value(self, x: InteriorPointPrimal, d: Dual, *args, **kwargs) -> Scalar:
        """Barrier-augmented Lagrangian value.

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Scalar
            ``L(x, s, λ) + B(s)``.
        """
        return self(x, d, *args, **kwargs).value

    def x_grad(self, x: InteriorPointPrimal, d: Dual, *args, **kwargs) -> Vector_n:
        """Decision-variable block of ``∇L``.

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Vector_n
            ``∇_x L`` (barrier does not depend on ``x`` directly).
        """
        return self(x, d, *args, **kwargs).x_grad

    def primal_grad(
        self, x: InteriorPointPrimal, d: Dual, *args, **kwargs
    ) -> InteriorPointPrimal:
        """Full primal gradient including the slack block.

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        InteriorPointPrimal
            ``(∇_x L, ∇_s L)``.
        """
        return self(x, d, *args, **kwargs).primal_grad

    def dual_grad(self, x: InteriorPointPrimal, d: Dual, *args, **kwargs) -> Dual:
        """Dual residual with slack-augmented bound / inequality equalities.

        Parameters
        ----------
        x
            Interior-point primal.
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

    def grad(
        self, x: InteriorPointPrimal, d: Dual, *args, **kwargs
    ) -> tuple[InteriorPointPrimal, Dual]:
        """Full primal-dual gradient of the IP Lagrangian.

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        tuple[InteriorPointPrimal, Dual]
            Primal and dual gradient blocks.
        """
        return self(x, d, *args, **kwargs).grad

    def kkt_mvp_primal(
        self,
        x: InteriorPointPrimal,
        d: Dual,
        tangent: tuple[InteriorPointPrimal, Dual],
        *args,
        **kwargs,
    ) -> InteriorPointPrimal:
        """Primal diagonal KKT block (``x`` and slack Hessians).

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        tangent
            ``(d(x,s), dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        InteriorPointPrimal
            Primal-block KKT contribution.
        """
        return self(x, d, *args, **kwargs).kkt_mvp_primal(tangent)

    def kkt_mvp_upper_offdiag(
        self,
        x: InteriorPointPrimal,
        d: Dual,
        tangent: tuple[InteriorPointPrimal, Dual],
        *args,
        **kwargs,
    ) -> InteriorPointPrimal:
        """Upper off-diagonal KKT block for the IP system.

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        tangent
            ``(d(x,s), dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        InteriorPointPrimal
            Upper off-diagonal contribution.
        """
        return self(x, d, *args, **kwargs).kkt_mvp_upper_offdiag(tangent)

    def kkt_mvp_lower_offdiag(
        self,
        x: InteriorPointPrimal,
        d: Dual,
        tangent: tuple[InteriorPointPrimal, Dual],
        *args,
        **kwargs,
    ) -> Dual:
        """Lower off-diagonal KKT block for the IP system.

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        tangent
            ``(d(x,s), dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Dual
            Lower off-diagonal contribution.
        """
        return self(x, d, *args, **kwargs).kkt_mvp_lower_offdiag(tangent)

    def kkt_mvp(
        self,
        x: InteriorPointPrimal,
        d: Dual,
        tangent: tuple[InteriorPointPrimal, Dual],
        *args,
        **kwargs,
    ) -> tuple[InteriorPointPrimal, Dual]:
        """Full IP KKT matrix-vector product (including dual regularization).

        Parameters
        ----------
        x
            Interior-point primal.
        d
            Dual multipliers.
        tangent
            ``(d(x,s), dλ)`` tangent pair.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        tuple[InteriorPointPrimal, Dual]
            Full KKT product.
        """
        return self(x, d, *args, **kwargs).kkt_mvp(tangent)

    def curvature_estimate(
        self,
        x: InteriorPointPrimal,
        prev: InteriorPointEvaluatedLagrangian,
        *args,
        **kwargs,
    ) -> Vector_n:
        """Secant curvature in ``x`` at the shared multiplier ``prev.dual``.

        Parameters
        ----------
        x
            New interior-point primal (decision ``x.x`` is used for the
            finite-difference pair).
        prev
            Evaluated IP Lagrangian at the previous iterate.
        *args, **kwargs
            Forwarded to :meth:`__call__`.

        Returns
        -------
        Vector_n
            Curvature vector for a secant update on the ``x``-block.
        """
        return self(x, prev.dual, *args, **kwargs).curvature_estimate(prev)
