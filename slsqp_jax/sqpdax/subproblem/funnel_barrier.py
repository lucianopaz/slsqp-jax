"""Scaled barrier subproblem with the trust-funnel models and criticality measures."""

from typing import cast

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..dual import Dual
from ..lagrangian import InteriorPointEvaluatedLagrangian
from ..primal import InteriorPointPrimal
from ..types import Scalar
from .scaled_barrier import ScaledBarrierSubProblem

__all__ = [
    "FunnelBarrierSubProblem",
]


class FunnelBarrierSubProblem(ScaledBarrierSubProblem):
    """Barrier subproblem models of the interior-point trust-funnel method.

    Extends :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`
    with the objects Curtis, Gould, Robinson & Toint (2017, *Math. Prog.*
    *Comp.*) build their normal / tangential subproblems from, all expressed
    in the scaled step ``w = P⁻¹ d`` with ``P = diag(I, S)``:

    * the barrier model ``m_f(w) = ĝᵀw + ½ wᵀ Ĥ w`` with ``ĝ = P ∇f(x, s)``
      and ``Ĥ = P G P``, ``G = diag(∇²ₓₓL(x, y), D)``;
    * the infeasibility model ``m_v(w) = ‖ĉ + Â w‖₂`` with ``ĉ = c(x, s)``
      and ``Â = J(x, s) P``;
    * the v-criticality measures ``πᵛ = ‖Âᵀ ĉ‖₂`` and ``χᵛ = πᵛ / v``
      (eq. 3.1), and the f-criticality measures ``πᶠ = ‖r̂‖₂`` and
      ``χᶠ = (ĝ + Ĥ w_n)ᵀ r̂ / πᶠ`` built from the scaled residual
      ``r̂(w_n, y) = ĝ + Ĥ w_n + Âᵀ y`` (eqs. 3.13–3.14);
    * the fraction-to-boundary boxes of the normal step (eq. 3.1 of the
      paper's Sect. 2, constant ``κ_fbn``) and of the tangential step
      relative to ``s + nˢ`` (eq. 3.17, constant ``κ_fbt``).

    Attributes
    ----------
    lagrangian
        Interior-point Lagrangian at the reference point. Either slack
        curvature is accepted: ``primal_dual=True`` gives the paper's
        ``D = Y S⁻¹`` choice, ``primal_dual=False`` the primal barrier
        Hessian (``μ S⁻²`` for the log barrier).
    tau
        Fraction-to-boundary parameter of :meth:`primal_box`, fixed to
        ``1 - kappa_fbn`` so the inherited box is the normal-step box.
    kappa_fbn
        Normal-step fraction-to-boundary constant ``κ_fbn ∈ (0, 1)``:
        ``s + nˢ ≥ κ_fbn s``.
    kappa_fbt
        Tangential-step fraction-to-boundary constant ``κ_fbt ∈ (0, 1)``:
        ``s + nˢ + tˢ ≥ κ_fbt (s + nˢ)``.

    Notes
    -----
    The paper writes the Cauchy residual in native coordinates as
    ``r_k = P²(∇m_f(n) + Jᵀy)`` and measures it with ``‖P⁻¹ r_k‖``; the
    scaled vector returned by :meth:`r` is ``r̂ = P⁻¹ r_k``, so that
    ``πᶠ = ‖r̂‖`` and ``−r̂`` is the steepest-descent direction of ``m_f`` in
    ``w``. Null bound slacks are dead coordinates: every scaled operator
    masks them to zero, so they contribute nothing to any measure.
    """

    kappa_fbn: float | Scalar = 0.1
    kappa_fbt: float | Scalar = 0.1
    tau: float | Scalar = 0.9

    def __init__(
        self,
        lagrangian: InteriorPointEvaluatedLagrangian,
        kappa_fbn: float | Scalar = 0.1,
        kappa_fbt: float | Scalar = 0.1,
    ):
        """Attach an interior-point Lagrangian and the fraction-to-boundary constants.

        Parameters
        ----------
        lagrangian
            Cached interior-point Lagrangian (primal-dual or primal).
        kappa_fbn
            Normal-step fraction-to-boundary constant in ``(0, 1)``.
        kappa_fbt
            Tangential-step fraction-to-boundary constant in ``(0, 1)``.

        Raises
        ------
        ValueError
            If either constant is a Python number outside ``(0, 1)``. Array
            values (e.g. the ``μ``-dependent schedules of
            :class:`~slsqp_jax.sqpdax.barrier.update.FunnelBarrierUpdate`
            traced inside the outer loop) are accepted unchecked.
        """
        for name, value in (("kappa_fbn", kappa_fbn), ("kappa_fbt", kappa_fbt)):
            if isinstance(value, (int, float)) and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1); got {value}")
        self.lagrangian = lagrangian
        self.kappa_fbn = kappa_fbn
        self.kappa_fbt = kappa_fbt
        self.tau = 1.0 - kappa_fbn
        self.schur_cache = None

    # ------------------------------------------------------------------
    # models (``hess_mvp`` / ``jac_mvp`` / ``jac_t_mvp`` are inherited)
    # ------------------------------------------------------------------
    def model_f_grad(self, w_n: InteriorPointPrimal) -> InteriorPointPrimal:
        """Scaled gradient of the barrier model at ``w_n``: ``ĝ + Ĥ w_n``.

        Parameters
        ----------
        w_n
            Scaled normal step (zero when no normal step was computed).

        Returns
        -------
        InteriorPointPrimal
            ``P ∇m_f(n)`` in scaled primal layout.
        """
        return cast(
            InteriorPointPrimal,
            jax.tree.map(jnp.add, self.primal_grad(), self.hess_mvp(w_n)),
        )

    def model_f(self, w: InteriorPointPrimal) -> Scalar:
        """Barrier model change ``m_f(w) − m_f(0) = ĝᵀw + ½ wᵀ Ĥ w``.

        Parameters
        ----------
        w
            Scaled primal step (``x`` and slack blocks both enter).

        Returns
        -------
        Scalar
            Predicted change of the barrier function ``f(x, s)``.

        Notes
        -----
        Unlike the base :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.model_value`,
        the slack block contributes through the barrier gradient ``−μ e``
        and the slack curvature ``S D S``; both are part of the paper's
        ``m_f`` and are needed for ``Δm_f`` bookkeeping.
        """
        flat = w.flatten()
        return jnp.inner(self.primal_grad().flatten(), flat) + 0.5 * jnp.inner(
            flat, self.hess_mvp(w).flatten()
        )

    def model_value(self, step: tuple[InteriorPointPrimal, Dual]) -> Scalar:
        """Full-primal barrier model; alias of :meth:`model_f` on ``step[0]``.

        Parameters
        ----------
        step
            Primal-dual step; only the primal block enters.

        Returns
        -------
        Scalar
            ``m_f(step[0])``.
        """
        return self.model_f(step[0])

    def model_v(self, w: InteriorPointPrimal) -> Scalar:
        """Linearised infeasibility model ``m_v(w) = ‖ĉ + Â w‖₂``.

        Parameters
        ----------
        w
            Scaled primal step.

        Returns
        -------
        Scalar
            Predicted constraint violation after the step; ``m_v(0) = v``.

        Notes
        -----
        Deliberately ignores the dual-dual regularisation that
        :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.linearized_infeasibility`
        folds in through :meth:`residual`: the funnel bookkeeping must see
        the plain linearisation ``c + J d``.
        """
        resid = jax.tree.map(jnp.add, self.dual_grad(), self.jac_mvp(w))
        return jnp.linalg.norm(resid.flatten())

    # ------------------------------------------------------------------
    # criticality measures
    # ------------------------------------------------------------------
    def violation(self) -> Scalar:
        """Constraint violation ``v = ‖c(x, s)‖₂`` at the reference point."""
        return jnp.linalg.norm(self.dual_grad().flatten())

    def pi_v(self) -> Scalar:
        """v-criticality ``πᵛ = ‖Âᵀ ĉ‖₂ = ‖P J(x, s)ᵀ c(x, s)‖₂`` (eq. 3.1a)."""
        return jnp.linalg.norm(self.jac_t_mvp(self.dual_grad()).flatten())

    def chi_v(self) -> Scalar:
        """Relative v-criticality ``χᵛ = πᵛ / v`` (``0`` when ``v = 0``; eq. 3.1b)."""
        v = self.violation()
        safe_v = jnp.where(v > 0.0, v, 1.0)
        return jnp.where(v > 0.0, self.pi_v() / safe_v, 0.0)

    def r(self, w_n: InteriorPointPrimal, y: Dual) -> InteriorPointPrimal:
        """Scaled stationarity residual ``r̂ = ĝ + Ĥ w_n + Âᵀ y``.

        Parameters
        ----------
        w_n
            Scaled normal step.
        y
            Multiplier estimate.

        Returns
        -------
        InteriorPointPrimal
            ``P (∇m_f(n) + J(x, s)ᵀ y)``; equals ``P⁻¹ r_k`` of eq. 3.13.
        """
        return cast(
            InteriorPointPrimal,
            jax.tree.map(jnp.add, self.model_f_grad(w_n), self.jac_t_mvp(y)),
        )

    def pi_f(self, w_n: InteriorPointPrimal, y: Dual) -> Scalar:
        """f-criticality ``πᶠ = ‖r̂(w_n, y)‖₂`` (eq. 3.14a)."""
        return jnp.linalg.norm(self.r(w_n, y).flatten())

    def chi_f(self, w_n: InteriorPointPrimal, y: Dual) -> Scalar:
        """Cauchy-angle measure ``χᶠ = (ĝ + Ĥ w_n)ᵀ r̂ / πᶠ`` (eq. 3.14b).

        Parameters
        ----------
        w_n
            Scaled normal step.
        y
            Multiplier estimate.

        Returns
        -------
        Scalar
            ``∇m_f(n)ᵀ r_k / πᶠ`` in the paper's notation; ``0`` when
            ``πᶠ = 0``.
        """
        r = self.r(w_n, y).flatten()
        pi_f = jnp.linalg.norm(r)
        safe = jnp.where(pi_f > 0.0, pi_f, 1.0)
        num = jnp.inner(self.model_f_grad(w_n).flatten(), r)
        return jnp.where(pi_f > 0.0, num / safe, 0.0)

    # ------------------------------------------------------------------
    # boxes
    # ------------------------------------------------------------------
    def tangential_box(
        self, w_n: InteriorPointPrimal
    ) -> tuple[Float[Array, " n_p"], Float[Array, " n_p"]]:
        """Fraction-to-boundary box of the scaled tangential step (eq. 3.17).

        Parameters
        ----------
        w_n
            Scaled normal step the tangential step is added to.

        Returns
        -------
        lower, upper
            Flat bound vectors in
            :meth:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal.flatten`
            order.

        Notes
        -----
        The paper requires ``s + nˢ + tˢ ≥ κ_fbt (s + nˢ)``. Dividing by
        ``s`` gives the lower face ``−(1 − κ_fbt)(1 + w_n,s)`` on every live
        slack coordinate; ``x`` is free and null bound slacks keep ``−inf``.
        """
        lag = self.lagrangian
        n, mineq = lag.n, lag.mineq
        dtype = lag.ref.x.dtype
        inf = jnp.asarray(jnp.inf, dtype)
        flat_n = w_n.flatten()
        face = -(1.0 - self.kappa_fbt) * (1.0 + flat_n[n:])
        lo_x = jnp.full((n,), -inf, dtype)
        lo_s = face[:mineq]
        lo_lb = jnp.where(lag.null_lb, -inf, face[mineq : mineq + n])
        lo_ub = jnp.where(lag.null_ub, -inf, face[mineq + n :])
        lo = jnp.concatenate([lo_x, lo_s, lo_lb, lo_ub])
        hi = jnp.full_like(lo, inf)
        return lo, hi
