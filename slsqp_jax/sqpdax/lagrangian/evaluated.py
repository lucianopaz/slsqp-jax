"""Cached Lagrangian values, gradients, and KKT operators at a reference point."""

from typing import Any, Generic, TypeVar, cast

import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float

from ..barrier import EvaluatedBarrier, LogBarrier
from ..dual import Dual
from ..primal import InteriorPointPrimal, Primal, PrimalType, Slack
from ..problem.basic import EvaluatedProblem
from ..secant import Secant
from ..types import (
    Matrix_meqn,
    Matrix_mineqn,
    Scalar,
    Vector_meq,
    Vector_mineq,
    Vector_n,
)

__all__ = [
    "EvaluatedLagrangian",
    "EvaluatedLagrangianType",
    "InteriorPointEvaluatedLagrangian",
]


class EvaluatedLagrangian(Module, Generic[PrimalType]):
    """Lagrangian built from values already cached at a reference point.

    This is the cached counterpart to
    :class:`~slsqp_jax.sqpdax.lagrangian.basic.Lagrangian`: use it when the
    algorithm has already evaluated the nonlinear problem at ``evaluated.ref``
    and wants Lagrangian values, gradients, or KKT products without calling
    the original problem functions again.

    Attributes
    ----------
    evaluated
        Pointwise NLP evaluation at the primal reference.
    secant
        Optional matrix-free Hessian used in place of exact QVPs.
    dual
        Multipliers ``λ`` at which the Lagrangian is formed.
    """

    evaluated: EvaluatedProblem[PrimalType]
    secant: Secant | None
    dual: Dual

    @property
    def has_exact_curvature(self) -> bool:
        """Whether exact QVPs are available and no secant is attached."""
        return self.evaluated.has_exact_curvature and self.secant is None

    @property
    def is_kkt_dual_regularized(self) -> bool:
        """Whether the dual-dual KKT block is regularized (always ``False`` here)."""
        return False

    @property
    def ref(self) -> PrimalType:
        """Primal reference point."""
        return self.evaluated.ref

    @property
    def x_ref(self) -> Vector_n:
        """Decision vector ``x`` at the reference point."""
        return self.ref.x

    @property
    def n(self) -> int:
        """Number of decision variables."""
        return self.x_ref.shape[0]

    @property
    def meq(self) -> int:
        """Number of equality constraints."""
        return self.evaluated.meq

    @property
    def mineq(self) -> int:
        """Number of inequality constraints."""
        return self.evaluated.mineq

    @property
    def fn_val(self) -> Scalar:
        """Objective value at the reference."""
        return self.evaluated.fn_val

    @property
    def grad_val(self) -> Vector_n:
        """Objective gradient at the reference."""
        return self.evaluated.grad_val

    @property
    def eq_fn_val(self) -> Vector_meq:
        """Equality residual at the reference."""
        return self.evaluated.eq_fn_val

    @property
    def eq_fn_jac_val(self) -> Matrix_meqn:
        """Equality Jacobian at the reference."""
        return self.evaluated.eq_fn_jac_val

    @property
    def ineq_fn_val(self) -> Vector_mineq:
        """Inequality residual at the reference."""
        return self.evaluated.ineq_fn_val

    @property
    def ineq_fn_jac_val(self) -> Matrix_mineqn:
        """Inequality Jacobian at the reference."""
        return self.evaluated.ineq_fn_jac_val

    @property
    def lb(self) -> Vector_n:
        """Lower bounds."""
        return self.evaluated.lb

    @property
    def ub(self) -> Vector_n:
        """Upper bounds."""
        return self.evaluated.ub

    @property
    def null_lb(self) -> Bool[Array, " n"]:
        """Mask of inactive lower bounds."""
        return self.evaluated.null_lb

    @property
    def null_ub(self) -> Bool[Array, " n"]:
        """Mask of inactive upper bounds."""
        return self.evaluated.null_ub

    @property
    def eq_multipliers(self) -> Vector_meq:
        """Equality multipliers from :attr:`dual`."""
        return self.dual.eq_multipliers

    @property
    def ineq_multipliers(self) -> Vector_mineq:
        """Inequality multipliers from :attr:`dual`."""
        return self.dual.ineq_multipliers

    @property
    def lb_multipliers(self) -> Vector_n:
        """Lower-bound multipliers from :attr:`dual`."""
        return self.dual.lb_multipliers

    @property
    def ub_multipliers(self) -> Vector_n:
        """Upper-bound multipliers from :attr:`dual`."""
        return self.dual.ub_multipliers

    @property
    def value(self) -> Scalar:
        """Scalar Lagrangian ``L(x, λ)`` at the reference point."""
        x = self.ref.x
        return (
            self.fn_val
            + self.eq_multipliers @ self.eq_fn_val
            + self.ineq_multipliers @ self.ineq_fn_val
            + self.lb_multipliers
            @ jnp.where(
                self.null_lb,
                jnp.zeros_like(self.lb),
                self.lb - x,
            )
            + self.ub_multipliers
            @ jnp.where(
                self.null_ub,
                jnp.zeros_like(self.ub),
                x - self.ub,
            )
        )

    @property
    def x_grad(self) -> Vector_n:
        """Partial gradient ``∇_x L`` as a flat decision vector."""
        return (
            self.grad_val
            + self.eq_multipliers @ self.eq_fn_jac_val
            + self.ineq_multipliers @ self.ineq_fn_jac_val
            + jnp.where(
                self.null_lb,
                jnp.zeros_like(self.lb),
                -self.lb_multipliers,
            )
            + jnp.where(
                self.null_ub,
                jnp.zeros_like(self.ub),
                self.ub_multipliers,
            )
        )

    @property
    def primal_grad(self) -> PrimalType:
        """Primal-block Lagrangian gradient packed as ``PrimalType``.

        The base implementation packs only ``x``; subclasses with richer
        primals (e.g. interior-point slacks) must override.
        """
        return cast(PrimalType, Primal(x=self.x_grad))

    @property
    def dual_grad(self) -> Dual:
        """Dual residual ``∇_λ L`` (constraint and bound violations)."""
        x = self.ref.x
        return cast(
            Dual,
            Dual(
                eq_multipliers=self.eq_fn_val,
                ineq_multipliers=self.ineq_fn_val,
                lb_multipliers=jnp.where(
                    self.null_lb,
                    jnp.zeros_like(self.lb),
                    self.lb - x,
                ),
                ub_multipliers=jnp.where(
                    self.null_ub,
                    jnp.zeros_like(self.ub),
                    x - self.ub,
                ),
            ),
        )

    @property
    def nonbound_constraint_jac(self) -> Float[Array, " meq+mineq n"]:
        """Stacked equality and inequality Jacobians."""
        return jnp.concatenate(
            [self.eq_fn_jac_val, self.ineq_fn_jac_val],
            axis=0,
        )

    def hvp(self, p: Vector_n) -> Vector_n:
        """Lagrangian Hessian-vector product in the decision variables.

        Uses :attr:`secant` when present; otherwise the exact QVPs from
        :attr:`evaluated` weighted by the current multipliers.

        Parameters
        ----------
        p
            Decision-variable tangent of length ``n``.

        Returns
        -------
        Vector_n
            ``∇²_{xx} L[p]``.

        Raises
        ------
        TypeError
            If ``secant is None`` and any QVP is missing.
        """
        if self.secant is not None:
            return self.secant.hvp(p)
        fn_qvp = self.evaluated.fn_qvp
        eq_fn_qvp = self.evaluated.eq_fn_qvp
        ineq_fn_qvp = self.evaluated.ineq_fn_qvp
        if fn_qvp is None or eq_fn_qvp is None or ineq_fn_qvp is None:
            raise TypeError(
                "EvaluatedLagrangian.hvp requires exact curvature (all QVPs) "
                "when no secant approximation is attached"
            )
        return (
            fn_qvp(p)
            + self.eq_multipliers @ eq_fn_qvp(p)
            + self.ineq_multipliers @ ineq_fn_qvp(p)
        )

    def primal_hvp(self, tangent: PrimalType) -> PrimalType:
        """Primal-packed Lagrangian HVP.

        Parameters
        ----------
        tangent
            Primal-shaped tangent.

        Returns
        -------
        PrimalType
            ``∇²_{xx} L`` applied to ``tangent.x``, packed as a primal.
        """
        return cast(PrimalType, Primal(self.hvp(tangent.x)))

    @property
    def grad(self) -> tuple[PrimalType, Dual]:
        """Full primal-dual gradient ``(∇_x L, ∇_λ L)``."""
        return (self.primal_grad, self.dual_grad)

    def kkt_mvp_primal(self, tangent: tuple[PrimalType, Dual]) -> PrimalType:
        """Primal diagonal block of the KKT MVP.

        Parameters
        ----------
        tangent
            ``(dx, dλ)`` pair.

        Returns
        -------
        PrimalType
            Hessian contribution from ``dx``.
        """
        return cast(PrimalType, Primal(self.hvp(tangent[0].x)))

    def kkt_mvp_upper_offdiag(self, tangent: tuple[PrimalType, Dual]) -> PrimalType:
        """Upper off-diagonal KKT block (``Jᵀ dλ`` plus bound signs).

        Parameters
        ----------
        tangent
            ``(dx, dλ)`` pair.

        Returns
        -------
        PrimalType
            Contribution of the transposed constraint Jacobian.
        """
        nonbound_jac = self.nonbound_constraint_jac
        lower_jac_diag = jnp.where(
            self.null_lb,
            jnp.zeros_like(self.lb),
            -jnp.ones_like(self.lb),
        )
        upper_jac_diag = jnp.where(
            self.null_ub,
            jnp.zeros_like(self.ub),
            jnp.ones_like(self.ub),
        )
        return cast(
            PrimalType,
            Primal(
                x=(
                    jnp.concatenate(
                        [tangent[1].eq_multipliers, tangent[1].ineq_multipliers]
                    )
                    @ nonbound_jac
                    + tangent[1].lb_multipliers * lower_jac_diag
                    + tangent[1].ub_multipliers * upper_jac_diag
                )
            ),
        )

    def kkt_mvp_lower_offdiag(self, tangent: tuple[PrimalType, Dual]) -> Dual:
        """Lower off-diagonal KKT block (``J dx`` plus bound signs).

        Parameters
        ----------
        tangent
            ``(dx, dλ)`` pair.

        Returns
        -------
        Dual
            Dual-block contribution from ``dx``.
        """
        dx = tangent[0].x
        lower_jac_diag = jnp.where(
            self.null_lb,
            jnp.zeros_like(self.lb),
            -jnp.ones_like(self.lb),
        )
        upper_jac_diag = jnp.where(
            self.null_ub,
            jnp.zeros_like(self.ub),
            jnp.ones_like(self.ub),
        )
        return cast(
            Dual,
            Dual(
                eq_multipliers=self.eq_fn_jac_val @ dx,
                ineq_multipliers=self.ineq_fn_jac_val @ dx,
                lb_multipliers=lower_jac_diag * dx,
                ub_multipliers=upper_jac_diag * dx,
            ),
        )

    def kkt_mvp_dual(self, tangent: tuple[PrimalType, Dual]) -> Dual:
        """Dual-dual KKT block (zeros for the standard SQP Lagrangian).

        Parameters
        ----------
        tangent
            ``(dx, dλ)`` pair.

        Returns
        -------
        Dual
            Zero dual shaped like ``tangent[1]``.
        """
        return jax.tree.map(jnp.zeros_like, tangent[1])

    def kkt_mvp(self, tangent: tuple[PrimalType, Dual]) -> tuple[PrimalType, Dual]:
        """Full KKT saddle-point matrix-vector product.

        Parameters
        ----------
        tangent
            ``(dx, dλ)`` pair.

        Returns
        -------
        tuple[PrimalType, Dual]
            ``(primal_row, dual_row)``.
        """
        primal_row = jax.tree.map(
            jnp.add,
            self.kkt_mvp_primal(tangent),
            self.kkt_mvp_upper_offdiag(tangent),
        )
        return (primal_row, self.kkt_mvp_lower_offdiag(tangent))

    def curvature_estimate(self, prev: "EvaluatedLagrangian[PrimalType]") -> Vector_n:
        """Secant curvature ``y = ∇_x L(x_new, λ) − ∇_x L(x_prev, λ)``.

        Uses a shared multiplier ``λ = prev.dual``. The bound block has a
        constant ``±1`` Jacobian, so with a shared ``λ`` it cancels and does
        not contribute to the pair (Nocedal & Wright §18.3).

        Parameters
        ----------
        prev
            Evaluated Lagrangian at the previous iterate (provides ``λ`` and
            the previous first-order quantities).

        Returns
        -------
        Vector_n
            Curvature vector for a secant update.
        """
        lam = prev.dual
        return (
            (self.grad_val - prev.grad_val)
            + lam.eq_multipliers @ (self.eq_fn_jac_val - prev.eq_fn_jac_val)
            + lam.ineq_multipliers @ (self.ineq_fn_jac_val - prev.ineq_fn_jac_val)
        )


EvaluatedLagrangianType = TypeVar(
    "EvaluatedLagrangianType", bound=EvaluatedLagrangian[Any]
)


class InteriorPointEvaluatedLagrangian(EvaluatedLagrangian[InteriorPointPrimal]):
    """Barrier-augmented Lagrangian at an interior-point reference.

    Combines the NLP Lagrangian with an
    :class:`~slsqp_jax.sqpdax.barrier.EvaluatedBarrier` on the slacks. With a
    :class:`~slsqp_jax.sqpdax.barrier.LogBarrier`, both the primal
    (Nocedal & Wright §19.14) and primal-dual (§19.13) KKT forms are
    available: the distinction appears in
    :meth:`kkt_mvp_slack_slack`, which uses either the barrier Hessian or
    the ``Λ S^{-1}`` product. Setting ``primal_dual=True`` with a non-log
    barrier raises ``TypeError``.

    Attributes
    ----------
    evaluated
        Pointwise NLP evaluation; ``ref`` must be an
        :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal`.
    secant
        Optional secant for the decision-variable Hessian block.
    dual
        Multipliers at the reference.
    barrier
        Barrier evaluation at the current slacks.
    dual_kkt_regularization
        Nonnegative dual-dual regularization on equality rows.
    primal_dual
        Select the primal-dual slack-slack KKT block when ``True``.
    """

    evaluated: EvaluatedProblem[InteriorPointPrimal]
    secant: Secant | None
    dual: Dual
    barrier: EvaluatedBarrier
    dual_kkt_regularization: float = 0.0
    primal_dual: bool = False

    def __init__(
        self,
        evaluated: EvaluatedProblem[InteriorPointPrimal],
        secant: Secant | None,
        dual: Dual,
        barrier: EvaluatedBarrier,
        dual_kkt_regularization: float = 0.0,
        primal_dual: bool = False,
    ):
        """Validate and store the IP Lagrangian ingredients.

        Parameters
        ----------
        evaluated
            NLP evaluation at an interior-point primal.
        secant
            Optional secant Hessian.
        dual
            Multipliers.
        barrier
            Barrier evaluation at the reference slacks.
        dual_kkt_regularization
            Dual-dual regularization strength.
        primal_dual
            Use the primal-dual slack block when ``True``.

        Raises
        ------
        TypeError
            If ``evaluated.ref`` is not an
            :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal`, or if
            ``primal_dual`` is set with a non-log barrier.
        """
        if not isinstance(evaluated.ref, InteriorPointPrimal):
            raise TypeError(
                "evaluated.ref must be an InteriorPointPrimal. "
                f"Got {type(evaluated.ref)} instead."
            )
        if primal_dual and not isinstance(barrier.original, LogBarrier):
            raise TypeError(
                "PrimalDualInteriorPointEvaluatedLagrangian requires the barrier to evaluated from "
                f"a LogBarrier. Got {type(barrier.original)} instead."
            )
        self.evaluated = evaluated
        self.secant = secant
        self.dual = dual
        self.barrier = barrier
        self.dual_kkt_regularization = dual_kkt_regularization
        self.primal_dual = primal_dual

    @property
    def is_kkt_dual_regularized(self) -> bool:
        """Whether ``dual_kkt_regularization`` is strictly positive."""
        return self.dual_kkt_regularization > 0

    @property
    def x_ref(self) -> Vector_n:
        """Decision vector at the reference."""
        return self.ref.x

    @property
    def ref(self) -> InteriorPointPrimal:
        """Interior-point primal reference."""
        return self.evaluated.ref

    @property
    def slack(self) -> Slack:
        """Slack block of the reference primal."""
        return self.ref.slack

    @property
    def ineq_fn_val(self) -> Float[Array, " mineq"]:
        """Slack-augmented inequality residual ``h(x) + s``."""
        return self.evaluated.ineq_fn_val + self.slack.s

    @property
    def ineq_fn_jac_val(self) -> Float[Array, " mineq n"]:
        """Inequality Jacobian (unchanged by slacks)."""
        return self.evaluated.ineq_fn_jac_val

    @property
    def value(self) -> Scalar:
        """Barrier-augmented Lagrangian value."""
        x = self.x_ref
        return (
            self.fn_val
            + self.eq_multipliers @ self.eq_fn_val
            + self.ineq_multipliers @ self.ineq_fn_val
            + self.lb_multipliers
            @ jnp.where(
                self.null_lb,
                jnp.zeros_like(self.lb),
                self.lb - x + self.slack.s_lb,
            )
            + self.ub_multipliers
            @ jnp.where(
                self.null_ub,
                jnp.zeros_like(self.ub),
                x - self.ub + self.slack.s_ub,
            )
            + self.barrier.fn_val
        )

    @property
    def x_grad(self) -> Vector_n:
        """Decision-variable block of ``∇L`` (delegates to the base formula)."""
        return super().x_grad

    @property
    def slack_grad(self) -> Slack:
        """Slack-block stationarity residual (multipliers + barrier gradient)."""
        return jax.tree.map(
            lambda x, y: x + y,
            Slack(
                s=self.ineq_multipliers,
                s_lb=jnp.where(
                    self.null_lb,
                    jnp.zeros_like(self.slack.s_lb),
                    self.dual.lb_multipliers,
                ),
                s_ub=jnp.where(
                    self.null_ub,
                    jnp.zeros_like(self.slack.s_ub),
                    self.dual.ub_multipliers,
                ),
            ),
            self.barrier.grad_val,
        )

    @property
    def primal_grad(self) -> InteriorPointPrimal:
        """Full primal gradient ``(∇_x L, ∇_s L)``."""
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(x=self.x_grad, slack=self.slack_grad),
        )

    @property
    def dual_grad(self) -> Dual:
        """Dual residual with slack-augmented inequalities and bounds."""
        x = self.ref.x
        return cast(
            Dual,
            Dual(
                eq_multipliers=self.eq_fn_val,
                ineq_multipliers=self.ineq_fn_val,
                lb_multipliers=jnp.where(
                    self.null_lb,
                    jnp.zeros_like(self.lb),
                    self.lb - x + self.slack.s_lb,
                ),
                ub_multipliers=jnp.where(
                    self.null_ub,
                    jnp.zeros_like(self.ub),
                    x - self.ub + self.slack.s_ub,
                ),
            ),
        )

    @property
    def grad(self) -> tuple[InteriorPointPrimal, Dual]:
        """Full primal-dual gradient of the IP Lagrangian."""
        return (self.primal_grad, self.dual_grad)

    def kkt_mvp_x_x(self, tangent: tuple[InteriorPointPrimal, Dual]) -> Vector_n:
        """``x``-``x`` Hessian block of the IP KKT operator."""
        return self.hvp(tangent[0].x)

    def kkt_mvp_x_slack_upper_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> Vector_n:
        """``x``-slack coupling (zero for a separable barrier)."""
        return jax.tree.map(
            lambda x: jnp.zeros_like(x),
            tangent[0].x,
        )

    def kkt_mvp_x_dual_upper_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> Vector_n:
        """``x``-dual upper off-diagonal (constraint Jacobianᵀ)."""
        nonbound_jac = self.nonbound_constraint_jac
        lower_jac_diag = jnp.where(
            self.null_lb,
            jnp.zeros_like(self.lb),
            -jnp.ones_like(self.lb),
        )
        upper_jac_diag = jnp.where(
            self.null_ub,
            jnp.zeros_like(self.ub),
            jnp.ones_like(self.ub),
        )
        return (
            jnp.concatenate([tangent[1].eq_multipliers, tangent[1].ineq_multipliers])
            @ nonbound_jac
            + tangent[1].lb_multipliers * lower_jac_diag
            + tangent[1].ub_multipliers * upper_jac_diag
        )

    def kkt_mvp_slack_x_lower_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> Slack:
        """Slack-``x`` coupling (zero for a separable barrier)."""
        return jax.tree.map(
            lambda x: jnp.zeros_like(x),
            tangent[0].slack,
        )

    def kkt_mvp_slack_slack(self, tangent: tuple[InteriorPointPrimal, Dual]) -> Slack:
        """Slack-slack Hessian: barrier HVP or primal-dual ``Λ S^{-1}`` product."""
        if self.primal_dual:
            return jax.tree.map(
                lambda x, y, z: x / y * z,
                Slack(
                    s=self.ineq_multipliers,
                    s_lb=self.lb_multipliers,
                    s_ub=self.ub_multipliers,
                ),
                self.slack,
                tangent[0].slack,
            )
        else:
            return self.barrier.hvp(tangent[0].slack)

    def kkt_mvp_slack_dual_upper_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> Slack:
        """Slack-dual coupling (identity on active slack blocks)."""
        tangent_dual = tangent[1]
        return cast(
            Slack,
            Slack(
                s=tangent_dual.ineq_multipliers,
                s_lb=jnp.where(self.null_lb, 0.0, tangent_dual.lb_multipliers),
                s_ub=jnp.where(self.null_ub, 0.0, tangent_dual.ub_multipliers),
            ),
        )

    def kkt_mvp_dual_x_lower_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> Dual:
        """Dual-``x`` lower off-diagonal (constraint Jacobian)."""
        dx = tangent[0].x
        lower_jac_diag = jnp.where(
            self.null_lb,
            jnp.zeros_like(self.lb),
            -jnp.ones_like(self.lb),
        )
        upper_jac_diag = jnp.where(
            self.null_ub,
            jnp.zeros_like(self.ub),
            jnp.ones_like(self.ub),
        )
        return cast(
            Dual,
            Dual(
                eq_multipliers=self.eq_fn_jac_val @ dx,
                ineq_multipliers=self.ineq_fn_jac_val @ dx,
                lb_multipliers=lower_jac_diag * dx,
                ub_multipliers=upper_jac_diag * dx,
            ),
        )

    def kkt_mvp_dual_slack_lower_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> Dual:
        """Dual-slack lower off-diagonal (identity on active slack blocks)."""
        tangent_slack = tangent[0].slack
        return cast(
            Dual,
            Dual(
                eq_multipliers=jnp.zeros_like(tangent[1].eq_multipliers),
                ineq_multipliers=tangent_slack.s,
                lb_multipliers=jnp.where(self.null_lb, 0.0, tangent_slack.s_lb),
                ub_multipliers=jnp.where(self.null_ub, 0.0, tangent_slack.s_ub),
            ),
        )

    def kkt_mvp_dual_dual(self, tangent: tuple[InteriorPointPrimal, Dual]) -> Dual:
        """Dual-dual block (``-δ I`` on equalities when regularized)."""
        return cast(
            Dual,
            Dual(
                eq_multipliers=-self.dual_kkt_regularization
                * tangent[1].eq_multipliers,
                ineq_multipliers=jnp.zeros_like(tangent[1].ineq_multipliers),
                lb_multipliers=jnp.zeros_like(tangent[1].lb_multipliers),
                ub_multipliers=jnp.zeros_like(tangent[1].ub_multipliers),
            ),
        )

    def kkt_mvp_primal(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> InteriorPointPrimal:
        """Primal diagonal IP KKT block (``x`` and slack Hessians)."""
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=jax.tree.map(
                    lambda x, y: x + y,
                    self.kkt_mvp_x_x(tangent),
                    self.kkt_mvp_x_slack_upper_offdiag(tangent),
                ),
                slack=jax.tree.map(
                    lambda x, y: x + y,
                    self.kkt_mvp_slack_x_lower_offdiag(tangent),
                    self.kkt_mvp_slack_slack(tangent),
                ),
            ),
        )

    def kkt_mvp_upper_offdiag(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> InteriorPointPrimal:
        """Upper off-diagonal IP KKT block."""
        return cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=self.kkt_mvp_x_dual_upper_offdiag(tangent),
                slack=self.kkt_mvp_slack_dual_upper_offdiag(tangent),
            ),
        )

    def kkt_mvp_lower_offdiag(self, tangent: tuple[InteriorPointPrimal, Dual]) -> Dual:
        """Lower off-diagonal IP KKT block (``x`` and slack couplings)."""
        return jax.tree.map(
            lambda x, y: x + y,
            self.kkt_mvp_dual_x_lower_offdiag(tangent),
            self.kkt_mvp_dual_slack_lower_offdiag(tangent),
        )

    def kkt_mvp(
        self, tangent: tuple[InteriorPointPrimal, Dual]
    ) -> tuple[InteriorPointPrimal, Dual]:
        """Full IP KKT matrix-vector product including dual regularization."""
        primal_row = jax.tree.map(
            jnp.add,
            self.kkt_mvp_primal(tangent),
            self.kkt_mvp_upper_offdiag(tangent),
        )
        dual_row = jax.tree.map(
            lambda x, y: x + y,
            self.kkt_mvp_lower_offdiag(tangent),
            self.kkt_mvp_dual_dual(tangent),
        )
        return (primal_row, dual_row)
