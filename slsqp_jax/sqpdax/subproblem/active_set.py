"""Active-set QP subproblem restricted to a working set."""

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

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
    ``L_k``. All KKT products are evaluated on ``L_k``.

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
        """Primal gradient of the masked Lagrangian ``L_k``."""
        return self.L_k.primal_grad

    def dual_grad(self) -> Dual:
        """Dual residual of the masked Lagrangian ``L_k``."""
        return self.L_k.dual_grad

    def kkt_mvp_primal(self, step: tuple[Primal, Dual]) -> Primal:
        """Primal-primal KKT product on the working-set Lagrangian."""
        return self.L_k.kkt_mvp_primal(step)

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
        """Full KKT product ``K z`` on the working-set Lagrangian.

        Parameters
        ----------
        step
            Primal-dual tangent.

        Returns
        -------
        tuple of Primal and Dual
            Assembled KKT action from :attr:`L_k`.
        """
        return self.L_k.kkt_mvp(step)

    def residual(self, step: tuple[Primal, Dual]) -> tuple[Primal, Dual]:
        """KKT residual ``K z - rhs`` using :meth:`kkt_mvp` and :meth:`kkt_rhs`."""
        return jax.tree.map(lambda x, y: x - y, self.kkt_mvp(step), self.kkt_rhs())

    def nonbound_constraint_jac(self) -> Float[Array, " meq+mineq n"]:
        """Stacked equality / inequality Jacobians from the masked Lagrangian."""
        return self.L_k.nonbound_constraint_jac
