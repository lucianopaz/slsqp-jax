"""Barrier-parameter (``μ``) update policies for interior-point SQP."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import equinox as eqx
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array

from ..registry import KindRegistryMixin
from ..types import Scalar
from .base import Barrier

if TYPE_CHECKING:
    from ..lagrangian import InteriorPointEvaluatedLagrangian

__all__ = [
    "BarrierUpdate",
    "MonotoneBarrierUpdate",
    "AdaptiveBarrierUpdate",
]


def _inf_norm(v: Array) -> Scalar:
    """``||v||_∞`` that is well-defined (``== 0``) for empty vectors.

    Parameters
    ----------
    v
        One-dimensional array (may have length zero).

    Returns
    -------
    Scalar
        Maximum absolute entry, or ``0`` when ``v`` is empty.
    """
    return jnp.max(jnp.abs(v), initial=0.0)


class BarrierUpdate(KindRegistryMixin, Module):
    """Interior-point barrier-parameter (``μ``) update policy.

    ``update`` returns a *new* :class:`~slsqp_jax.sqpdax.barrier.base.Barrier`
    with an updated ``weight`` (``μ``), driven by the interior-point
    complementarity / KKT state of the current iterate rather than by the
    merit (which belongs to the step controller). This mirrors the
    :class:`~slsqp_jax.sqpdax.secant.base.Secant` "return a new instance"
    contract and keeps the ``μ`` schedule pluggable (monotone
    Fiacco–McCormick vs adaptive) independently of the barrier *shape*
    (log vs quadratic).

    The barrier ``weight`` is the single source of truth for ``μ``; it is
    read by both the interior-point Lagrangian and the merit, so the outer
    loop must re-thread the returned :class:`Barrier` into both after an
    update.

    Concrete members register themselves under a ``kind``
    :class:`~typing.ClassVar` via
    :class:`~slsqp_jax.sqpdax.registry.KindRegistryMixin` and can be built
    with :meth:`from_spec`.
    """

    _registry: ClassVar[dict] = {}

    @abstractmethod
    def update(
        self, barrier: Barrier, lag: InteriorPointEvaluatedLagrangian
    ) -> Barrier:
        """Return a copy of ``barrier`` with an updated weight.

        Parameters
        ----------
        barrier
            Current unevaluated barrier (source of ``μ = barrier.weight``).
        lag
            Interior-point Lagrangian evaluation at the current iterate.

        Returns
        -------
        Barrier
            Barrier whose ``weight`` may have been reduced (or left
            unchanged). Other fields are preserved.
        """
        ...  # pragma: no cover

    def complementarity(self, lag: InteriorPointEvaluatedLagrangian) -> Scalar:
        """Average complementarity ``sᵀ z / m`` over non-null slack/mult pairs.

        Parameters
        ----------
        lag
            Interior-point Lagrangian evaluation providing slacks, duals,
            and null-bound masks.

        Returns
        -------
        Scalar
            ``(∑ sᵢ zᵢ + ∑_{active} s_lb z_lb + ∑_{active} s_ub z_ub) / m``,
            where ``m`` is the number of active slack/multiplier pairs
            (at least ``1`` in the denominator).
        """
        slack = lag.slack
        dual = lag.dual
        comp_ineq = jnp.sum(slack.s * dual.ineq_multipliers)
        comp_lb = jnp.sum(jnp.where(lag.null_lb, 0.0, slack.s_lb * dual.lb_multipliers))
        comp_ub = jnp.sum(jnp.where(lag.null_ub, 0.0, slack.s_ub * dual.ub_multipliers))
        m = slack.s.shape[-1] + jnp.sum(~lag.null_lb) + jnp.sum(~lag.null_ub)
        return (comp_ineq + comp_lb + comp_ub) / jnp.maximum(m, 1)

    def optimality_residual(
        self, lag: InteriorPointEvaluatedLagrangian, mu: Scalar
    ) -> Scalar:
        """KKT error ``E(x, s; μ)`` of the perturbed system (N&W eq. 19.8–19.9).

        The max of the inf-norms of (i) the dual stationarity residual
        ``∇_x L``, (ii) the primal-feasibility residual (constraint values
        stored on ``dual_grad``), and (iii) the perturbed complementarity
        ``s ⊙ z − μ`` over the non-null pairs. Unscaled: the N&W ``s_d`` /
        ``s_c`` scaling factors are a documented extension.

        Parameters
        ----------
        lag
            Interior-point Lagrangian evaluation at the current iterate.
        mu
            Barrier parameter used in the complementarity residual.

        Returns
        -------
        Scalar
            ``max(‖∇_x L‖_∞, ‖c‖_∞, ‖s ⊙ z − μ‖_∞)``.
        """
        stationarity = _inf_norm(lag.x_grad)

        feas = lag.dual_grad
        feasibility = jnp.max(
            jnp.stack(
                [
                    _inf_norm(feas.eq_multipliers),
                    _inf_norm(feas.ineq_multipliers),
                    _inf_norm(feas.lb_multipliers),
                    _inf_norm(feas.ub_multipliers),
                ]
            )
        )

        slack = lag.slack
        dual = lag.dual
        comp_ineq = slack.s * dual.ineq_multipliers - mu
        comp_lb = jnp.where(lag.null_lb, 0.0, slack.s_lb * dual.lb_multipliers - mu)
        comp_ub = jnp.where(lag.null_ub, 0.0, slack.s_ub * dual.ub_multipliers - mu)
        complementarity = jnp.max(
            jnp.stack([_inf_norm(comp_ineq), _inf_norm(comp_lb), _inf_norm(comp_ub)])
        )

        return jnp.max(jnp.stack([stationarity, feasibility, complementarity]))


class MonotoneBarrierUpdate(BarrierUpdate):
    """Fiacco–McCormick monotone reduction (N&W Algorithm 19.6).

    Reduces ``μ`` by ``sigma`` once the current barrier subproblem is solved
    to the tolerance ``E(x, s; μ) ≤ kappa_eps * μ``; otherwise ``μ`` is held
    so the current subproblem can be solved further. Floored at ``mu_min``.

    Attributes
    ----------
    sigma
        Fractional reduction applied when the subproblem is accepted
        (``0 < sigma < 1``).
    kappa_eps
        Tolerance factor: accept the subproblem when
        ``E(x, s; μ) ≤ kappa_eps * μ``.
    mu_min
        Lower floor on the barrier parameter.
    """

    kind: ClassVar[str] = "monotone"

    sigma: float = eqx.field(default=0.2)
    kappa_eps: float = eqx.field(default=10.0)
    mu_min: float = eqx.field(default=1e-11)

    def update(
        self, barrier: Barrier, lag: InteriorPointEvaluatedLagrangian
    ) -> Barrier:
        """Reduce ``μ`` by ``sigma`` when the barrier subproblem is solved.

        Parameters
        ----------
        barrier
            Current barrier; ``weight`` is the current ``μ``.
        lag
            Interior-point Lagrangian evaluation used to form ``E(x, s; μ)``.

        Returns
        -------
        Barrier
            Copy with ``weight = max(sigma * μ, mu_min)`` if
            ``E ≤ kappa_eps * μ``, otherwise the same ``weight``.
        """
        mu = barrier.weight
        e_mu = self.optimality_residual(lag, mu)
        subproblem_solved = e_mu <= self.kappa_eps * mu
        new_mu = jnp.where(
            subproblem_solved,
            jnp.maximum(self.sigma * mu, self.mu_min),
            mu,
        )
        return eqx.tree_at(lambda b: b.weight, barrier, new_mu)


class AdaptiveBarrierUpdate(BarrierUpdate):
    """Adaptive (Mehrotra-style) update: ``μ ← σ (sᵀ z / m)``.

    Sets the next barrier parameter from a centering fraction of the current
    complementarity, so ``μ`` tracks the duality gap without waiting for the
    barrier subproblem to be solved to tolerance. Floored at ``mu_min``.

    Attributes
    ----------
    sigma
        Centering fraction applied to average complementarity.
    mu_min
        Lower floor on the barrier parameter.
    """

    kind: ClassVar[str] = "adaptive"

    sigma: float = eqx.field(default=0.2)
    mu_min: float = eqx.field(default=1e-11)

    def update(
        self, barrier: Barrier, lag: InteriorPointEvaluatedLagrangian
    ) -> Barrier:
        """Set ``μ`` from a fraction of current average complementarity.

        Parameters
        ----------
        barrier
            Current barrier (only structural fields are reused; ``weight``
            is replaced).
        lag
            Interior-point Lagrangian evaluation providing complementarity.

        Returns
        -------
        Barrier
            Copy with ``weight = max(sigma * (sᵀ z / m), mu_min)``.
        """
        comp = self.complementarity(lag)
        new_mu = jnp.maximum(self.sigma * comp, self.mu_min)
        return eqx.tree_at(lambda b: b.weight, barrier, new_mu)
