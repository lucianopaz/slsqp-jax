"""Barrier-parameter (``μ``) update policies for interior-point SQP."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

import equinox as eqx
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool

from ..registry import KindRegistryMixin
from ..types import Scalar
from .base import Barrier

if TYPE_CHECKING:
    from ..lagrangian import InteriorPointEvaluatedLagrangian

__all__ = [
    "BarrierUpdate",
    "MonotoneBarrierUpdate",
    "AdaptiveBarrierUpdate",
    "FunnelBarrierUpdate",
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
    ) -> tuple[Barrier, Bool[Array, ""]]:
        """Return a copy of ``barrier`` with an updated weight.

        Parameters
        ----------
        barrier
            Current unevaluated barrier (source of ``μ = barrier.weight``).
        lag
            Interior-point Lagrangian evaluation at the current iterate.

        Returns
        -------
        barrier
            Barrier whose ``weight`` may have been reduced (or left
            unchanged). Other fields are preserved.
        updated
            Whether the policy reduced ``μ`` on this call. For schedules that
            wait on a barrier-subproblem tolerance this doubles as the inner
            ``E(x, s; μ) ≤ ε_μ`` test of N&W Algorithm 19.4, which the
            interior-point minimiser reads as part of its convergence check.
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
    ) -> tuple[Barrier, Bool[Array, ""]]:
        """Reduce ``μ`` by ``sigma`` when the barrier subproblem is solved.

        Parameters
        ----------
        barrier
            Current barrier; ``weight`` is the current ``μ``.
        lag
            Interior-point Lagrangian evaluation used to form ``E(x, s; μ)``.

        Returns
        -------
        barrier
            Copy with ``weight = max(sigma * μ, mu_min)`` if
            ``E ≤ kappa_eps * μ``, otherwise the same ``weight``.
        updated
            Whether ``E(x, s; μ) ≤ kappa_eps * μ``, i.e. whether the current
            barrier subproblem counts as solved. Because the tolerance is
            proportional to ``μ``, it tightens automatically as ``μ`` shrinks.
        """
        mu = barrier.weight
        e_mu = self.optimality_residual(lag, mu)
        subproblem_solved = e_mu <= self.kappa_eps * mu
        new_mu = jnp.where(
            subproblem_solved,
            jnp.maximum(self.sigma * mu, self.mu_min),
            mu,
        )
        return eqx.tree_at(lambda b: b.weight, barrier, new_mu), subproblem_solved


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
    ) -> tuple[Barrier, Bool[Array, ""]]:
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
        barrier
            Copy with ``weight = max(sigma * (sᵀ z / m), mu_min)``.
        updated
            Always ``True``: this schedule has no inner loop to wait on, so
            ``μ`` moves on every call.
        """
        comp = self.complementarity(lag)
        new_mu = jnp.maximum(self.sigma * comp, self.mu_min)
        return eqx.tree_at(lambda b: b.weight, barrier, new_mu), jnp.ones(
            (), dtype=bool
        )


class FunnelBarrierUpdate(BarrierUpdate):
    """Trust-funnel outer loop (CGRT 2017, Algorithm 3 and Table 1).

    The barrier subproblem ``BSP(μ)`` counts as solved when the scaled
    stationarity and the constraint violation meet the ``μ``-dependent
    tolerances of (3.15a),

    ```
    πᶠ ≤ ε_π(μ) = ζ₁ μ^α    and    v ≤ ε_v(μ) = ζ₂ μ^β,            (5.3)
    ```

    in which case ``μ ← max{γ_μ μ, μ_min}`` (Algorithm 3, Step 8). The other
    ``μ``-dependent constants of Table 1 are exposed as schedules satisfying
    the limits (5.1)–(5.2):

    ```
    κ_fbn(μ) = min{κ_fb_max, κ_fb_scale μ^{κ_fb_power}}  → 0
    κ_fbt(μ) = κ_fbn(μ)
    κ_y(μ)   = κ_y_scale / μ                            → ∞
    κ_D(μ)   = κ_D_scale / μ                            → ∞
    ```

    The paper fixes only the limits (and the form (5.3)); the power-law
    choices above are conventional. ``κ_fbn``/``κ_fbt`` are the
    fraction-to-boundary constants of
    :class:`~slsqp_jax.sqpdax.subproblem.funnel_barrier.FunnelBarrierSubProblem`,
    ``κ_y`` the multiplier cap (3.10) of
    :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.BarrierSafeguard`
    and ``κ_D`` the cap on the primal slack curvature ``μ S⁻²`` in (5.4).

    Attributes
    ----------
    gamma_mu
        Reduction factor ``γ_μ ∈ (0, 1)`` applied when the subproblem is
        solved.
    mu_min
        Lower floor on the barrier parameter.
    zeta1, alpha
        Stationarity tolerance ``ε_π(μ) = ζ₁ μ^α`` with ``ζ₁ ∈ (0, 1)``,
        ``α ≥ 1``.
    zeta2, beta
        Violation tolerance ``ε_v(μ) = ζ₂ μ^β`` with ``ζ₂, β > 0``.
    kappa_fb_max, kappa_fb_scale, kappa_fb_power
        Fraction-to-boundary schedule ``κ_fb(μ) = min{max, scale μ^power}``
        shared by ``κ_fbn`` and ``κ_fbt``; ``max ∈ (0, 1)``, ``scale > 0``,
        ``power > 0``.
    kappa_y_scale, kappa_D_scale
        Scales of the diverging caps ``κ_y(μ)``, ``κ_D(μ)``; both ``> 0``.
    """

    kind: ClassVar[str] = "funnel"

    gamma_mu: float = eqx.field(default=0.2)
    mu_min: float = eqx.field(default=1e-11)
    zeta1: float = eqx.field(default=0.5)
    alpha: float = eqx.field(default=1.0)
    zeta2: float = eqx.field(default=1.0)
    beta: float = eqx.field(default=1.0)
    kappa_fb_max: float = eqx.field(default=0.1)
    kappa_fb_scale: float = eqx.field(default=1.0)
    kappa_fb_power: float = eqx.field(default=1.0)
    kappa_y_scale: float = eqx.field(default=1e2)
    kappa_D_scale: float = eqx.field(default=1e2)

    def __check_init__(self):
        def require(ok: bool, msg: str):
            if not ok:
                raise ValueError(msg)

        require(
            0.0 < self.gamma_mu < 1.0,
            f"gamma_mu must be in (0, 1); got {self.gamma_mu}",
        )
        require(self.mu_min > 0.0, f"mu_min must be > 0; got {self.mu_min}")
        require(0.0 < self.zeta1 < 1.0, f"zeta1 must be in (0, 1); got {self.zeta1}")
        require(self.alpha >= 1.0, f"alpha must be >= 1; got {self.alpha}")
        require(self.zeta2 > 0.0, f"zeta2 must be > 0; got {self.zeta2}")
        require(self.beta > 0.0, f"beta must be > 0; got {self.beta}")
        require(
            0.0 < self.kappa_fb_max < 1.0,
            f"kappa_fb_max must be in (0, 1); got {self.kappa_fb_max}",
        )
        require(
            self.kappa_fb_scale > 0.0,
            f"kappa_fb_scale must be > 0; got {self.kappa_fb_scale}",
        )
        require(
            self.kappa_fb_power > 0.0,
            f"kappa_fb_power must be > 0; got {self.kappa_fb_power}",
        )
        require(
            self.kappa_y_scale > 0.0,
            f"kappa_y_scale must be > 0; got {self.kappa_y_scale}",
        )
        require(
            self.kappa_D_scale > 0.0,
            f"kappa_D_scale must be > 0; got {self.kappa_D_scale}",
        )

    # --- Table 1 schedules -----------------------------------------------------

    def eps_pi(self, mu: Scalar) -> Scalar:
        """Stationarity tolerance ``ε_π(μ) = ζ₁ μ^α`` of (3.15a)/(5.3)."""
        return self.zeta1 * jnp.asarray(mu) ** self.alpha

    def eps_v(self, mu: Scalar) -> Scalar:
        """Violation tolerance ``ε_v(μ) = ζ₂ μ^β`` of (3.15a)/(5.3)."""
        return self.zeta2 * jnp.asarray(mu) ** self.beta

    def kappa_fbn(self, mu: Scalar) -> Scalar:
        """Normal-step fraction-to-boundary constant ``κ_fbn(μ)`` of (2.2)/(3.5)."""
        return jnp.minimum(
            self.kappa_fb_max,
            self.kappa_fb_scale * jnp.asarray(mu) ** self.kappa_fb_power,
        )

    def kappa_fbt(self, mu: Scalar) -> Scalar:
        """Tangential-step fraction-to-boundary constant ``κ_fbt(μ)`` of (3.19b)/(3.23b)."""
        return self.kappa_fbn(mu)

    def kappa_y(self, mu: Scalar) -> Scalar:
        """Multiplier norm cap ``κ_y(μ)`` of (3.10)."""
        return self.kappa_y_scale / jnp.asarray(mu)

    def kappa_D(self, mu: Scalar) -> Scalar:
        """Cap ``κ_D(μ)`` on the primal slack curvature ``μ S⁻²`` of (3.11)/(5.4)."""
        return self.kappa_D_scale / jnp.asarray(mu)

    # --- outer-loop test ---------------------------------------------------------

    def solved(self, mu: Scalar, pi_f: Scalar, v: Scalar) -> Bool[Array, ""]:
        """Whether ``(πᶠ, v)`` meet the (3.15a) tolerances at barrier weight ``μ``."""
        return (pi_f <= self.eps_pi(mu)) & (v <= self.eps_v(mu))

    def update(
        self,
        barrier: Barrier,
        lag: InteriorPointEvaluatedLagrangian,
        *,
        pi_f: Scalar | None = None,
        v: Scalar | None = None,
    ) -> tuple[Barrier, Bool[Array, ""]]:
        """Reduce ``μ`` by ``γ_μ`` once the barrier subproblem meets (3.15a).

        Parameters
        ----------
        barrier
            Current barrier; ``weight`` is the current ``μ``.
        lag
            Interior-point Lagrangian evaluation at the committed iterate.
        pi_f
            Scaled stationarity ``πᶠ`` at the iterate (Definition 1.2,
            ``n = 0``). When omitted it is computed from ``lag`` through a
            :class:`~slsqp_jax.sqpdax.subproblem.funnel_barrier.FunnelBarrierSubProblem`
            with ``lag.dual`` as the multiplier estimate.
        v
            Constraint violation ``‖c(x, s)‖₂``. Defaults to the norm of
            ``lag.dual_grad``.

        Returns
        -------
        barrier
            Copy with ``weight = max(γ_μ μ, μ_min)`` when solved, else unchanged.
        solved
            Whether ``πᶠ ≤ ε_π(μ)`` and ``v ≤ ε_v(μ)`` held.
        """
        if pi_f is None or v is None:
            # Local import: the subproblem package depends on this module.
            from ..subproblem.funnel_barrier import FunnelBarrierSubProblem

            sub = cast(FunnelBarrierSubProblem, FunnelBarrierSubProblem(lag))
            if pi_f is None:
                pi_f = sub.pi_f(sub._zero_primal(), lag.dual)
            if v is None:
                v = sub.violation()
        mu = barrier.weight
        is_solved = self.solved(mu, pi_f, v)
        new_mu = jnp.where(is_solved, jnp.maximum(self.gamma_mu * mu, self.mu_min), mu)
        return eqx.tree_at(lambda b: b.weight, barrier, new_mu), is_solved
