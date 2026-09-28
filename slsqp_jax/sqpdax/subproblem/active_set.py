"""Active-set QP subproblem restricted to a working set."""

from typing import Self, cast

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float

from ..active_set import ActiveSet
from ..dual import Dual
from ..lagrangian import EvaluatedLagrangian
from ..primal import Primal
from .base import SubProblem

__all__ = [
    "ActiveSetSubProblem",
]


class ActiveSetSubProblem(SubProblem[Primal]):
    """Equality / inequality / bound QP on a fixed working set.

    Stores the full Lagrangian at the reference point and a masked copy
    ``L_k``. The constraint blocks of the KKT system are evaluated on
    ``L_k``; the Hessian block comes from the unmasked Lagrangian so the QP
    model ``½ dᵀ ∇²ₓₓL(x_k, λ_k) d + ∇f_kᵀ d`` (Nocedal & Wright eq. 18.11a)
    is the same for every working set of one outer step.

    Attributes
    ----------
    lagrangian
        Unmasked Lagrangian at the reference point.
    L_k
        Working-set-restricted Lagrangian
        (``active_set.mask_lagrangian(lagrangian)``).
    active_set
        Boolean masks for active inequalities and bounds.

    Notes
    -----
    Inactive inequality and bound rows are zeroed via
    :class:`~slsqp_jax.sqpdax.active_set.ActiveSet`, so inactive multipliers
    and Jacobian rows do not enter the saddle system.

    **Dual convention.** This subproblem encodes the SQP view (Nocedal &
    Wright eq. 18.9 / 18.11): :meth:`primal_grad` is the *objective*
    gradient ``∇f_k``, so the dual block of a KKT solution is the multiplier
    ``λ_{k+1}`` itself and
    :attr:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.is_kkt_dual_increment`
    is ``False``. The multipliers stored on :attr:`lagrangian` only enter
    the Hessian.
    """

    L_k: EvaluatedLagrangian[Primal]
    active_set: ActiveSet

    def __init__(self, lagrangian: EvaluatedLagrangian[Primal], active_set: ActiveSet):
        """Build a working-set QP from a Lagrangian and an active set.

        Parameters
        ----------
        lagrangian
            Cached Lagrangian at the current iterate.
        active_set
            Working-set masks applied to form :attr:`L_k`.
        """
        self.lagrangian = lagrangian
        self.L_k = active_set.mask_lagrangian(lagrangian)
        self.active_set = active_set

    def with_active_set(self, active_set: ActiveSet) -> Self:
        """Rebuild the subproblem on the same Lagrangian with a new working set.

        Parameters
        ----------
        active_set
            Working-set masks for the rebuilt subproblem.

        Returns
        -------
        Self
            New instance of the same class sharing :attr:`lagrangian`.

        Notes
        -----
        The active-set loop refreshes the working set every iteration; this
        hook lets subclasses carrying extra parameters (e.g. proximal
        stabilisation) survive the refresh by overriding it.
        """
        return type(self)(self.lagrangian, active_set)

    @property
    def x_k(self) -> Primal:
        """Reference primal ``x_k`` from the unmasked Lagrangian."""
        return self.lagrangian.ref

    @property
    def d_k(self) -> Dual:
        """Reference dual multipliers from the unmasked Lagrangian."""
        return self.lagrangian.dual

    @property
    def n(self) -> int:
        """Number of decision variables."""
        return self.x_k.n

    @property
    def meq(self) -> int:
        """Number of equality constraints."""
        return self.d_k.meq

    @property
    def mineq(self) -> int:
        """Number of inequality constraints (including inactive)."""
        return self.d_k.mineq

    @property
    def m(self) -> int:
        """Number of equalities plus currently active inequalities / bounds.

        Notes
        -----
        Bound and inequality contributions are JAX reductions, so the runtime
        value is a 0-d integer array even though the annotation is ``int``.
        """
        return (
            self.meq  # ty: ignore[invalid-return-type]
            + self.active_set.active_inequalities.astype(jnp.int32).sum()
            + self.active_set.active_lb.astype(jnp.int32).sum()
            + self.active_set.active_ub.astype(jnp.int32).sum()
        )

    def primal_grad(self) -> Primal:
        """Objective gradient ``∇f_k`` (SQP-view right-hand side, eq. 18.9).

        Notes
        -----
        Deliberately *not* the Lagrangian gradient: the multiplier terms
        ``Aᵀλ`` enter through the dual block of the KKT system, so the
        solved dual is the full ``λ_{k+1}``.
        """
        return cast(Primal, Primal(x=self.lagrangian.grad_val))

    def dual_grad(self) -> Dual:
        """Dual residual of the masked Lagrangian ``L_k``."""
        return self.L_k.dual_grad

    def kkt_mvp_primal(self, step: tuple[Primal, Dual]) -> Primal:
        """Lagrangian Hessian product ``∇²ₓₓL(x_k, λ_k) d`` (unmasked multipliers).

        Notes
        -----
        Uses :attr:`lagrangian` rather than :attr:`L_k` so the curvature of
        constraints that are inactive in the current working set (but carry
        a multiplier at ``x_k``) is kept, and the QP Hessian does not change
        as the working set is refreshed. With a secant the two coincide.
        """
        return self.lagrangian.kkt_mvp_primal(step)

    def kkt_mvp_upper_offdiag(self, step: tuple[Primal, Dual]) -> Primal:
        """Primal-dual upper off-diagonal on the working-set Lagrangian."""
        return self.L_k.kkt_mvp_upper_offdiag(step)

    def kkt_mvp_lower_offdiag(self, step: tuple[Primal, Dual]) -> Dual:
        """Dual-primal lower off-diagonal on the working-set Lagrangian."""
        return self.L_k.kkt_mvp_lower_offdiag(step)

    def kkt_mvp_dual(self, step: tuple[Primal, Dual]) -> Dual:
        """Dual-dual KKT product on the working-set Lagrangian."""
        return self.L_k.kkt_mvp_dual(step)

    def kkt_mvp(self, step: tuple[Primal, Dual]) -> tuple[Primal, Dual]:
        """Full KKT product ``K z`` assembled from this subproblem's blocks.

        Parameters
        ----------
        step
            Primal-dual tangent.

        Returns
        -------
        tuple of Primal and Dual
            :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.kkt_operator`
            applied to ``step`` (unmasked Hessian, working-set constraints).
        """
        return self.kkt_operator(step)

    def free_subspace(self) -> tuple[Bool[Array, " n"], Float[Array, " n"]]:
        """Pin the variables on an active bound to their bound.

        Returns
        -------
        free_mask
            ``False`` on coordinates fixed by an active lower or upper bound.
        fixed_values
            ``lb - x_k`` (resp. ``ub - x_k``) on those coordinates, zero
            elsewhere. A lower bound wins when both are active (``lb == ub``).
        """
        bounds = self.L_k.dual_grad
        fix_lb = self.active_set.active_lb
        fix_ub = self.active_set.active_ub & ~fix_lb
        zeros = jnp.zeros_like(bounds.lb_multipliers)
        # ``dual_grad`` bound rows are ``lb - x`` / ``x - ub`` on active rows.
        fixed = jnp.where(
            fix_lb,
            bounds.lb_multipliers,
            jnp.where(fix_ub, -bounds.ub_multipliers, zeros),
        )
        return ~(fix_lb | fix_ub), fixed

    def residual(self, step: tuple[Primal, Dual]) -> tuple[Primal, Dual]:
        """KKT residual ``K z - rhs`` using :meth:`kkt_mvp` and :meth:`kkt_rhs`."""
        return jax.tree.map(lambda x, y: x - y, self.kkt_mvp(step), self.kkt_rhs())

    def nonbound_constraint_jac(self) -> Float[Array, " meq+mineq n"]:
        """Stacked equality / inequality Jacobians from the masked Lagrangian."""
        return self.L_k.nonbound_constraint_jac

    def active_constraint_rows(self) -> Bool[Array, " meq+mineq"]:
        """Equalities plus the inequalities in the working set."""
        return self.active_set.active_gen
