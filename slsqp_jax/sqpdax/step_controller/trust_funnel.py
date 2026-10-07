"""Trust-funnel step acceptance and radius / funnel updates (CGRT 2017, §3.3)."""

from typing import cast

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Scalar

from ..merit import ConstraintViolation, Merit
from ..primal import InteriorPointPrimal
from ..subproblem.solver.trust_funnel import IterationType, TrustFunnelSolverState
from .base import StepController, StepResult

__all__ = ["TrustFunnelManager"]


class TrustFunnelManager(StepController[InteriorPointPrimal, TrustFunnelSolverState]):
    """Step controller for the interior-point trust-funnel method.

    Implements Steps 38–50 of Algorithm 2 in Curtis, Gould, Robinson and
    Toint (2017): the trial step ``d = n + t`` proposed by
    :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_funnel.TrustFunnelSolver`
    is classified as a **y-**, **f-** or **v-iteration**, scored with the
    actual-to-predicted ratio of the barrier function (``ρᶠ``, eq. 2.12) or
    of the constraint violation (``ρᵛ``, eq. 2.13), and the two trust-region
    radii and the funnel radius are updated by (3.24)–(3.37).

    Classification (Definitions 2.1–2.3), with ``v⁺ = v(x⁺, s⁺)``:

    ```
    Y :  d = 0
    F :  t ≠ 0  ∧  Δm_f,d ≥ κ_δ Δm_f,t (2.10)  ∧  v⁺ ≤ v_max (2.11)
    V :  otherwise
    ```

    Updates (``δ`` denotes the radius being updated, ``grow_factor = γ₃``):

    ```
    Y                     : nothing changes (3.24)
    F, ρᶠ ≥ η1  (k ∈ S_f) : accept; δᶠ ← min{γ₃ δᶠ, δ_max} if ρᶠ ≥ η2 else δᶠ;
                            δᵛ, v_max unchanged; S_f-flag ← true (3.25)–(3.28), (3.30)
    F, ρᶠ < η1            : reject; δᶠ ← γ₁ δᶠ if ρᶠ < 0 else γ₂ δᶠ (3.29)
    V, ρᵛ ≥ η1 ∧ (2.15)   : accept; δᵛ ← min{γ₃ δᵛ, δ_max} if ρᵛ ≥ η2 else δᵛ;
      (k ∈ S_v)             v_max ← max{κ_t1 v_max, v⁺ + κ_t2 (v − v⁺)} (3.32)–(3.35)
    V, otherwise          : reject; δᵛ ← γ₁ δᵛ if ρᵛ < 0 else γ₂ δᵛ; v_max unchanged (3.36)
    ```

    where (2.15) is ``n ≠ 0 ∧ Δm_v,d ≥ κ_cd Δm_v,n``. The paper prescribes
    intervals (``[δ, ∞)`` on success, ``[γ₁ δ, γ₂ δ]`` on failure); the
    choices above are conventional points inside them. On both failure
    branches the iterate is left unchanged (``x == x0``), so the outer loop
    re-solves at the shrunk radius on its next iteration.

    The (2.10) and (2.15) model tests and the ``Δm`` values are read from the
    incoming
    :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_funnel.TrustFunnelSolverState`
    (``objective_decrease_ok``, ``contraction_ok``, ``normal_norm``,
    ``dm_f_n + dm_f_t``, ``dm_v_d``) so that the constants ``κ_δ``, ``κ_cd``
    live in one place, the solver. The controller only adds the actual
    function values ``f(x⁺, s⁺)`` and ``v(x⁺, s⁺)``.

    The slack reset (3.26)/(3.33) is **not** applied here: it belongs to the
    minimiser's post-iterate phase, which rewrites the committed iterate.

    Attributes
    ----------
    barrier_merit
        :class:`~slsqp_jax.sqpdax.merit.Merit` evaluating the barrier
        function ``f(x, s) = f(x) − μ Σ log s`` (a
        :class:`~slsqp_jax.sqpdax.merit.NormMerit` with a barrier and
        ``feasibility_weight = 0``).
    violation
        :class:`~slsqp_jax.sqpdax.merit.ConstraintViolation` evaluating
        ``v(x, s) = ‖c(x, s)‖₂`` in the norm of the subproblem's ``m_v``.
    eta1, eta2
        Acceptance / growth thresholds, ``0 < η1 ≤ η2 < 1``.
    gamma1, gamma2
        Shrink factors on failure, ``0 < γ₁ ≤ γ₂ < 1``.
    grow_factor
        Growth factor ``γ₃ > 1`` applied when ``ρ ≥ η2``.
    max_radius
        Cap on either radius after growth.
    kappa_t1, kappa_t2
        Funnel contraction constants of (3.35), both in ``(0, 1)``.
    """

    barrier_merit: Merit
    violation: ConstraintViolation
    eta1: float = eqx.field(static=True, default=0.1)
    eta2: float = eqx.field(static=True, default=0.75)
    gamma1: float = eqx.field(static=True, default=0.25)
    gamma2: float = eqx.field(static=True, default=0.5)
    grow_factor: float = eqx.field(static=True, default=2.0)
    max_radius: float = eqx.field(static=True, default=1e10)
    kappa_t1: float = eqx.field(static=True, default=0.5)
    kappa_t2: float = eqx.field(static=True, default=0.5)

    def __check_init__(self):
        def require(ok: bool, msg: str):
            if not ok:
                raise ValueError(msg)

        require(
            0.0 < self.eta1 <= self.eta2 < 1.0,
            f"need 0 < eta1 <= eta2 < 1; got eta1={self.eta1}, eta2={self.eta2}",
        )
        require(
            0.0 < self.gamma1 <= self.gamma2 < 1.0,
            "need 0 < gamma1 <= gamma2 < 1; got "
            f"gamma1={self.gamma1}, gamma2={self.gamma2}",
        )
        require(
            self.grow_factor > 1.0, f"grow_factor must be > 1; got {self.grow_factor}"
        )
        require(self.max_radius > 0.0, f"max_radius must be > 0; got {self.max_radius}")
        require(
            0.0 < self.kappa_t1 < 1.0,
            f"kappa_t1 must be in (0, 1); got {self.kappa_t1}",
        )
        require(
            0.0 < self.kappa_t2 < 1.0,
            f"kappa_t2 must be in (0, 1); got {self.kappa_t2}",
        )

    def step(
        self,
        x0: InteriorPointPrimal,
        direction: InteriorPointPrimal,
        solver_state: TrustFunnelSolverState | None = None,
    ) -> StepResult[InteriorPointPrimal, TrustFunnelSolverState]:
        """Classify the iteration, accept or reject ``direction``, update radii.

        Parameters
        ----------
        x0
            Current iterate ``(x_k, s_k)``.
        direction
            Native-scale trial step ``d_k = n_k + t_k`` from the funnel
            solver (zero on y-iterations).
        solver_state
            The
            :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_funnel.TrustFunnelSolverState`
            returned by the solver for this step.

        Returns
        -------
        StepResult
            ``x`` is ``x0 + direction`` when accepted and ``x0`` otherwise;
            ``merit_val`` is the barrier value ``f(x, s)`` at the returned
            iterate; ``solver_state`` carries the updated ``radius_v``,
            ``radius_f``, ``v_max``, ``sf_flag``, the final
            ``iteration_type`` and ``rho`` (``ρᶠ`` on f-iterations, ``ρᵛ``
            on v-iterations, ``1`` on y-iterations), plus the diagnostics
            ``v_trial``, ``model_error_f`` / ``model_error_v``
            (``|f(x⁺) − m_f(d)|``, ``|v(x⁺) − m_v(d)|``; zero on
            y-iterations), ``demoted`` ((2.10) held but (2.11) failed) and
            ``funnel_violated`` (accepted with ``v⁺ > v_max``).

        Raises
        ------
        TypeError
            If ``solver_state`` is not a ``TrustFunnelSolverState``.
        """
        if not isinstance(solver_state, TrustFunnelSolverState):
            raise TypeError(
                "TrustFunnelManager.step requires a TrustFunnelSolverState "
                f"(carrying the funnel radii and Δm values). Got "
                f"{type(solver_state)} instead."
            )
        st = solver_state
        dtype = st.radius_v.dtype
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
        neg_inf = jnp.asarray(-jnp.inf, dtype)

        # --- actual values at the current and trial points ---------------------
        f0 = self.barrier_merit(x0)
        v0 = self.violation(x0)
        x_trial = jax.tree.map(jnp.add, x0, direction)
        f_trial = self.barrier_merit(x_trial)
        v_trial = self.violation(x_trial)

        # --- classification (Definitions 2.1–2.3) -------------------------------
        d_zero = jnp.all(direction.flatten() == 0.0)
        is_y = d_zero
        is_f = (
            ~is_y
            & (st.tangential_norm > 0.0)
            & st.objective_decrease_ok
            & (v_trial <= st.v_max)
        )
        is_v = ~is_y & ~is_f

        # --- ratios (2.12), (2.13) --------------------------------------------
        def ratio(actual: Scalar, predicted: Scalar) -> Scalar:
            good = predicted > tiny
            return jnp.where(
                good, actual / jnp.where(good, predicted, 1.0), neg_inf
            ).astype(dtype)

        dm_f_d = st.dm_f_n + st.dm_f_t
        rho_f = ratio(f0 - f_trial, dm_f_d)
        rho_v = ratio(v0 - v_trial, st.dm_v_d)
        rho = jnp.where(is_f, rho_f, jnp.where(is_v, rho_v, jnp.ones((), dtype)))

        # --- acceptance ----------------------------------------------------------
        success_f = is_f & (rho_f >= self.eta1)
        contraction = (st.normal_norm > 0.0) & st.contraction_ok  # (2.15)
        success_v = is_v & (rho_v >= self.eta1) & contraction
        accepted = success_f | success_v

        # --- radius updates --------------------------------------------------------
        def grown(radius: Scalar) -> Scalar:
            return jnp.minimum(self.grow_factor * radius, self.max_radius)

        def shrunk(radius: Scalar, rho_value: Scalar) -> Scalar:
            factor = jnp.where(rho_value < 0.0, self.gamma1, self.gamma2)
            return factor * radius

        radius_f = jnp.where(
            is_f,
            jnp.where(
                success_f,
                jnp.where(rho_f >= self.eta2, grown(st.radius_f), st.radius_f),
                shrunk(st.radius_f, rho_f),
            ),
            st.radius_f,
        )
        radius_v = jnp.where(
            is_v,
            jnp.where(
                success_v,
                jnp.where(rho_v >= self.eta2, grown(st.radius_v), st.radius_v),
                shrunk(st.radius_v, rho_v),
            ),
            st.radius_v,
        )
        # Funnel contraction (3.35) on successful v-iterations only.
        v_max = jnp.where(
            success_v,
            jnp.maximum(
                self.kappa_t1 * st.v_max, v_trial + self.kappa_t2 * (v0 - v_trial)
            ),
            st.v_max,
        )
        sf_flag = st.sf_flag | success_f

        iteration_type = IterationType.where(
            is_y,
            IterationType.y_iteration,
            IterationType.where(
                is_f, IterationType.f_iteration, IterationType.v_iteration
            ),
        )

        x_new = jax.tree.map(lambda a, b: jnp.where(accepted, b, a), x0, x_trial)
        f_new = jnp.where(accepted, f_trial, f0)

        # --- diagnostics: model errors (Lemma 4.5), demotion, funnel check -----
        model_error_f = jnp.where(is_y, 0.0, jnp.abs((f0 - f_trial) - dm_f_d))
        model_error_v = jnp.where(is_y, 0.0, jnp.abs((v0 - v_trial) - st.dm_v_d))
        demoted = (
            ~is_y
            & (st.tangential_norm > 0.0)
            & st.objective_decrease_ok
            & (v_trial > st.v_max)
        )
        funnel_violated = accepted & (v_trial > v_max)

        self.logger.info(
            "funnel {kind}: rho={rho:.3e} accepted={accepted} f {f0:.6e} -> "
            "{f_new:.6e} v {v0:.3e} -> {v_new:.3e} radius_f {rf0:.3e} -> {rf:.3e} "
            "radius_v {rv0:.3e} -> {rv:.3e} v_max {vmax0:.3e} -> {vmax:.3e}",
            kind=iteration_type,
            rho=rho,
            accepted=accepted,
            f0=f0,
            f_new=f_new,
            v0=v0,
            v_new=jnp.where(accepted, v_trial, v0),
            rf0=st.radius_f,
            rf=radius_f,
            rv0=st.radius_v,
            rv=radius_v,
            vmax0=st.v_max,
            vmax=v_max,
        )
        self.logger.warning(
            "funnel step rejected: rho={rho:.3e} (predicted decrease "
            "dm_f_d={dm_f_d:.3e}, dm_v_d={dm_v_d:.3e})",
            when=~accepted & ~is_y,
            rho=rho,
            dm_f_d=dm_f_d,
            dm_v_d=st.dm_v_d,
        )
        self.logger.warning(
            "funnel invariant violated: accepted iterate has v={v:.3e} > v_max={vmax:.3e}",
            when=funnel_violated,
            v=v_trial,
            vmax=v_max,
        )

        new_state = cast(
            TrustFunnelSolverState,
            eqx.tree_at(
                lambda s: (
                    s.radius_v,
                    s.radius_f,
                    s.v_max,
                    s.sf_flag,
                    s.rho,
                    s.iteration_type,
                    s.v_trial,
                    s.model_error_f,
                    s.model_error_v,
                    s.funnel_violated,
                    s.demoted,
                ),
                st,
                (
                    radius_v,
                    radius_f,
                    v_max,
                    sf_flag,
                    rho,
                    iteration_type,
                    v_trial.astype(dtype),
                    model_error_f.astype(dtype),
                    model_error_v.astype(dtype),
                    funnel_violated,
                    demoted,
                ),
            ),
        )
        return cast(
            StepResult[InteriorPointPrimal, TrustFunnelSolverState],
            StepResult(
                x=x_new,
                accepted=accepted,
                merit_val=f_new,
                solver_state=new_state,
                step_size=jnp.where(accepted, 1.0, 0.0).astype(dtype),
                proposed_step_norm=jnp.linalg.norm(direction.flatten()),
            ),
        )
