"""Projected Steihaug–Toint CG tangential step for the trust-funnel barrier subproblem."""

from typing import Self, cast

import jax
from equinox import tree_at
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Scalar

from ...dual import Dual
from ...linalg import (
    boundary_step_length,
    box_fraction,
    box_ray_length,
    null_space_projector,
    steihaug_cg,
)
from ...primal import InteriorPointPrimal
from ..funnel_barrier import FunnelBarrierSubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState

__all__ = [
    "FunnelTangentialStepState",
    "FunnelTangentialStepSolver",
]


class FunnelTangentialStepState(SubProblemSolverState):
    """Carry of a trust-funnel tangential-step solve.

    Attributes
    ----------
    radius
        Composite radius on ``‖w_n + t‖₂`` used for this solve:
        ``min{κ_vf δᵛ, δᶠ}`` for a relaxed step (3.19c) or
        ``min{κ_vf δᵛ, δᶠ, κ_v v_max}`` for a very relaxed one (3.23c).
    n_cg_iter
        Cumulative projected-CG iterations.
    on_boundary
        ``True`` when ``‖w_n + t‖`` saturates ``radius``.
    dm_f_t
        Model decrease ``Δm_f,t = m_f(w_n) − m_f(w_n + t)`` of the returned
        step.
    cauchy_decrease
        Decrease ``m_f(w_n) − m_f(w_n + t_C)`` of the Cauchy point
        (3.17)–(3.18); the returned step never does worse (3.19a)/(3.23a).
    step_norm
        ``‖t‖₂ = ‖P⁻¹ t‖₂`` (used by the (3.20) test).
    total_norm
        ``‖w_n + t‖₂`` (the quantity bounded by ``radius``).
    model_v_after
        ``m_v(w_n + t)``, needed by the (3.19d)/(3.23d) checks.
    ftb_truncated
        ``True`` when the fraction-to-boundary rule shortened the CG step.
    """

    radius: Scalar
    n_cg_iter: int
    on_boundary: Bool[Array, ""]
    dm_f_t: Scalar
    cauchy_decrease: Scalar
    step_norm: Scalar
    total_norm: Scalar
    model_v_after: Scalar
    ftb_truncated: Bool[Array, ""]

    @classmethod
    def cold(cls, radius: Scalar | float, dtype: DTypeLike = float) -> Self:
        """Zero-iteration carry with the given composite radius.

        Parameters
        ----------
        radius
            Trust-region radius on ``‖w_n + t‖``.
        dtype
            Floating dtype of the scalar fields.

        Returns
        -------
        Self
            Cold state with zero counters / decreases and ``success=False``.
        """
        zero = jnp.asarray(0.0, dtype)
        return cls(
            n_iter=jnp.asarray(0, jnp.int32),
            success=jnp.asarray(False),
            status=RESULTS.successful,
            radius=jnp.asarray(radius, dtype),
            n_cg_iter=jnp.asarray(0, jnp.int32),
            on_boundary=jnp.asarray(False),
            dm_f_t=zero,
            cauchy_decrease=zero,
            step_norm=zero,
            total_norm=zero,
            model_v_after=zero,
            ftb_truncated=jnp.asarray(False),
        )


class FunnelTangentialStepSolver(
    SubProblemSolver[
        InteriorPointPrimal, FunnelBarrierSubProblem, FunnelTangentialStepState
    ]
):
    """Inexact tangential step of Curtis, Gould, Robinson & Toint (2017), Sect. 3.2.

    Given the scaled normal step ``w_n`` and a multiplier estimate ``y`` in
    ``x0``, approximately solves

    ```
    min_t  m_f(w_n + t)   s.t.  Â t = 0,  ‖w_n + t‖₂ ≤ radius,
                                s + nˢ + tˢ ≥ κ_fbt (s + nˢ)
    ```

    with the scaled barrier model ``m_f`` and constraint operator ``Â`` taken
    matrix-free from a
    :class:`~slsqp_jax.sqpdax.subproblem.funnel_barrier.FunnelBarrierSubProblem`.

    * **Cauchy point** (3.17)–(3.18) / (3.21)–(3.22): ``t_C = −α_C r̂`` with
      ``r̂ = ĝ + Ĥ w_n + Âᵀ y`` the scaled residual (3.13) and ``α_C`` the
      exact 1-D minimiser of ``m_f(w_n − α r̂)`` clipped by the ball on
      ``‖w_n − α r̂‖`` and the tangential fraction-to-boundary box
      :meth:`~slsqp_jax.sqpdax.subproblem.funnel_barrier.FunnelBarrierSubProblem.tangential_box`.
    * **Projected CG** with Steihaug–Toint termination
      (:func:`~slsqp_jax.sqpdax.linalg.steihaug_cg`), warm-started at
      ``w_n``: residuals are projected exactly onto ``null(Â)`` by a
      matrix-free inner CG (:func:`~slsqp_jax.sqpdax.linalg.null_space_projector`),
      so ``Â t = 0`` and ``m_v(w_n + t) = m_v(w_n)`` by construction, which
      makes (3.19d) and (3.23d) inherit from the normal step. The iteration
      stops at the ball, on negative curvature, or on convergence.
    * **Fraction to boundary**: the CG step is backtracked along its ray onto
      the box (ray scaling keeps ``null(Â)`` and, by convexity, the ball).
    * **Cauchy fallback** (3.19a)/(3.23a): the Cauchy point is returned
      instead whenever the backtracked CG step decreases ``m_f`` by less.

    The dual block of the returned step is zero and the step is in *scaled*
    coordinates; the orchestrator adds it to ``w_n`` and unscales.

    Attributes
    ----------
    solver_state_class
        :class:`FunnelTangentialStepState`.
    max_iter
        Maximum projected-CG iterations.
    tol
        Relative tolerance on the projected residual,
        ``‖proj(ĝ + Ĥ w)‖ ≤ tol · ‖proj(ĝ + Ĥ w_n)‖``, floored at the
        roundoff level ``√eps ‖ĝ + Ĥ w_n‖`` of the working dtype.
    cg_regularization
        Scale-invariant floor for the negative-curvature test
        ``pᵀ Ĥ p ≤ cg_regularization ‖p‖²``.
    proj_cg_max_iter, proj_cg_tol, proj_reg
        Controls of the inner normal-equation CG behind the projector.
    """

    solver_state_class: type[FunnelTangentialStepState] = FunnelTangentialStepState

    max_iter: int = 100
    tol: float = 1e-6
    cg_regularization: float = 1e-10
    proj_cg_max_iter: int = 100
    proj_cg_tol: float = 1e-10
    proj_reg: float = 0.0

    def solve(
        self,
        subproblem: FunnelBarrierSubProblem,
        x0: tuple[InteriorPointPrimal, Dual],
        initial_state: FunnelTangentialStepState,
    ) -> tuple[tuple[InteriorPointPrimal, Dual], FunnelTangentialStepState]:
        """Compute the scaled tangential step on top of ``w_n``.

        Parameters
        ----------
        subproblem
            Funnel barrier subproblem at the current iterate.
        x0
            ``(w_n, y)``: scaled normal step and multiplier estimate defining
            the Cauchy direction ``−r̂(w_n, y)``.
        initial_state
            Carries the composite ``radius`` on ``‖w_n + t‖``.

        Returns
        -------
        step
            ``(scaled_tangential_step, zero_dual)``.
        state
            Refreshed :class:`FunnelTangentialStepState`.

        Raises
        ------
        TypeError
            If ``subproblem`` is not a ``FunnelBarrierSubProblem``.
        """
        if not isinstance(subproblem, FunnelBarrierSubProblem):
            raise TypeError(
                "subproblem must be a FunnelBarrierSubProblem. Got "
                f"{type(subproblem)} instead."
            )
        lag = subproblem.lagrangian
        n, meq, mineq = lag.n, lag.meq, lag.mineq
        dtype = lag.ref.x.dtype
        radius = jnp.asarray(initial_state.radius, dtype)
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
        w_n_tree, y = x0
        w_n = w_n_tree.flatten()

        def to_primal(w: Float[Array, " w"]) -> InteriorPointPrimal:
            return InteriorPointPrimal.from_flat(w, n, mineq)

        def to_dual(z: Float[Array, " m"]) -> Dual:
            return Dual.from_flat(z, n, mineq, meq)

        def H(w: Float[Array, " w"]) -> Float[Array, " w"]:
            return subproblem.hess_mvp(to_primal(w)).flatten()

        def A(w: Float[Array, " w"]) -> Float[Array, " m"]:
            return subproblem.jac_mvp(to_primal(w)).flatten()

        def At(z: Float[Array, " m"]) -> Float[Array, " w"]:
            return subproblem.jac_t_mvp(to_dual(z)).flatten()

        def model_f(w: Float[Array, " w"]) -> Scalar:
            return subproblem.model_f(to_primal(w))

        g_hat = subproblem.primal_grad().flatten()
        g_n = g_hat + H(w_n)  # ĝ + Ĥ w_n = P ∇m_f(n)
        m_f_n = model_f(w_n)
        lo, _hi = subproblem.tangential_box(w_n_tree)
        # ``w_n`` must lie inside the ball for any tangential step to exist;
        # the (3.12) gate guarantees it, but guard against roundoff anyway.
        w_n_norm = jnp.linalg.norm(w_n)
        room = w_n_norm < radius

        # --- Cauchy point along −r̂ (3.17)–(3.18) --------------------------------
        r_hat = g_n + At(y.flatten())
        pi_f = jnp.linalg.norm(r_hat)
        slope = jnp.dot(g_n, r_hat)  # = χᶠ πᶠ; must be > 0 for descent along −r̂
        curv = jnp.dot(r_hat, H(r_hat))
        alpha_star = jnp.where(
            curv > tiny, slope / jnp.maximum(curv, tiny), jnp.asarray(jnp.inf, dtype)
        )
        alpha_ball = boundary_step_length(w_n, -r_hat, radius)
        alpha_box = box_ray_length(-r_hat, lo)
        alpha_c = jnp.minimum(alpha_star, jnp.minimum(alpha_ball, alpha_box))
        alpha_c = jnp.where((slope > 0.0) & (pi_f > tiny) & room, alpha_c, 0.0)
        t_c = -alpha_c * r_hat
        cauchy_decrease = m_f_n - model_f(w_n + t_c)

        # --- projected CG with Steihaug–Toint termination from w_n --------------
        proj = null_space_projector(
            A,
            At,
            reg=self.proj_reg,
            tol=self.proj_cg_tol,
            max_iter=self.proj_cg_max_iter,
        )
        r0 = proj(-g_n)
        rz0 = jnp.dot(r0, r0)
        # Convergence threshold: relative to the initial projected residual, with
        # an absolute floor at the projection noise level ``√eps ‖ĝ + Ĥ w_n‖`` so
        # that CG stops once the residual is pure roundoff (a 1-D null space is
        # solved in one step, but its residual never reaches ``tol`` in float32).
        noise = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype)) * jnp.linalg.norm(
            g_n
        )
        rel_tol_sq = jnp.maximum(jnp.asarray(self.tol, dtype) ** 2 * rz0, noise**2)
        cg = steihaug_cg(
            H,
            r0,
            w_n,
            radius,
            tol_sq=rel_tol_sq,
            max_iter=self.max_iter,
            residual=lambda w: proj(-(g_hat + H(w))),
            curvature_floor=self.cg_regularization,
            done=~room,
        )
        n_cg = cg.n_iter
        t_cg = cg.w - w_n

        # --- fraction-to-boundary backtrack along the ray, Cauchy fallback ------
        beta_ftb = box_fraction(t_cg, lo)
        t_bt = beta_ftb * t_cg
        dm_cg = m_f_n - model_f(w_n + t_bt)
        use_cauchy = dm_cg < cauchy_decrease
        t = jnp.where(use_cauchy, t_c, t_bt)
        dm_f_t = jnp.where(use_cauchy, cauchy_decrease, dm_cg)
        step_norm = jnp.linalg.norm(t)
        total_norm = jnp.linalg.norm(w_n + t)
        on_boundary = total_norm >= radius * (1.0 - 1e-6)
        ftb_truncated = beta_ftb < 1.0
        model_v_after = subproblem.model_v(to_primal(w_n + t))
        finite = jnp.all(jnp.isfinite(t)) & jnp.isfinite(dm_f_t)
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)

        self.logger.debug(
            "tangential step: radius={radius:.3e} pi_f={pi_f:.3e} cg_iters={cg_iters} "
            "dm_f={dm:.3e} cauchy={cauchy:.3e} ftb_beta={beta:.3f} "
            "cauchy_fallback={fallback} on_boundary={on_boundary} m_v={m_v:.3e}",
            radius=radius,
            pi_f=pi_f,
            cg_iters=n_cg,
            dm=dm_f_t,
            cauchy=cauchy_decrease,
            beta=beta_ftb,
            fallback=use_cauchy,
            on_boundary=on_boundary,
            m_v=model_v_after,
        )
        self.logger.warning(
            "tangential step is non-finite (status={status})",
            when=~finite,
            status=status,
        )

        step = (to_primal(t), cast(Dual, jax.tree.map(jnp.zeros_like, lag.dual)))
        new_state = cast(
            FunnelTangentialStepState,
            tree_at(
                lambda st: (
                    st.n_iter,
                    st.success,
                    st.status,
                    st.radius,
                    st.n_cg_iter,
                    st.on_boundary,
                    st.dm_f_t,
                    st.cauchy_decrease,
                    st.step_norm,
                    st.total_norm,
                    st.model_v_after,
                    st.ftb_truncated,
                ),
                initial_state,
                (
                    initial_state.n_iter + 1,
                    finite,
                    status,
                    radius,
                    initial_state.n_cg_iter + n_cg,
                    on_boundary,
                    dm_f_t,
                    cauchy_decrease,
                    step_norm,
                    total_norm,
                    model_v_after,
                    ftb_truncated,
                ),
            ),
        )
        return step, new_state
