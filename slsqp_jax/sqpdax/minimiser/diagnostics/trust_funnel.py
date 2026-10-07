"""Running diagnostics of the trust-funnel loop derived from its convergence theory."""

from __future__ import annotations

from typing import Self, cast

import equinox as eqx
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Int

from ...subproblem.solver import (
    IterationType,
    MultiplierCase,
    TrustFunnelSolver,
    TrustFunnelSolverState,
)
from ...types import Scalar

__all__ = ["FunnelDiagnostics"]


def _kappa_v_threshold(kappa_c: Scalar, solver: TrustFunnelSolver) -> Scalar:
    """``(1 − κ_tt) / (κ_C κ_v)``, ``+inf`` while ``κ_C`` is still zero."""
    tiny = jnp.asarray(jnp.finfo(kappa_c.dtype).tiny, kappa_c.dtype)
    return jnp.where(
        kappa_c > tiny,
        (1.0 - solver.kappa_tt) / (jnp.maximum(kappa_c, tiny) * solver.kappa_v),
        jnp.asarray(jnp.inf, kappa_c.dtype),
    )


class FunnelDiagnostics(eqx.Module):
    """Counters, running maxima and consistency flags of Algorithm 2.

    Carried by :class:`~slsqp_jax.sqpdax.minimiser.trust_funnel_interior_point.TrustFunnelInteriorPointMinimiser`
    and refreshed once per outer step from the committed
    :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_funnel.TrustFunnelSolverState`.
    Every field is an array so the module travels through the traced loop;
    the per-step flags describe the step just taken, the sticky flags and
    the counters accumulate over the run (the counters are *not* reset when
    ``μ`` is reduced, so that rates over the whole run can be formed).

    The quantities map onto the convergence analysis of Curtis, Gould,
    Robinson & Toint (2017) as follows.

    **Invariants (2.1), Lemmas 3.4 / 3.7.** ``slack_positive`` (``s > 0``),
    ``residual_nonnegative`` (``c(x, s) ≥ 0`` after the slack reset),
    ``in_funnel`` (``v ≤ v_max``) and ``v_max_monotone`` (``v_max`` never
    grows within one barrier subproblem). ``invariant_violated`` is the
    sticky disjunction of their failures and of the controller's
    ``funnel_violated``.

    **Lemma 3.3 / (3.16).** A y-iteration that does not terminate the
    barrier subproblem is *healthy* only when it is a (3.15b) iteration
    whose ``πᶠ`` contracted geometrically, ``πᶠ_k ≤ κ_ω πᶠ_{k−1}``: with the
    normal gate (3.2) closed, (3.15b) and the forcing compatibility (3.16)
    imply exactly that. Any other y-iteration (none of (3.15a)–(3.15c) held,
    or a tangential step was requested and came back empty, or ``πᶠ`` did
    not contract) points at an inexact multiplier estimate (Assumption 3.1)
    and increments ``multiplier_failure_streak`` — unless it is explained
    by a collapsed radius (below), which is a model problem rather than a
    multiplier one.

    **Cauchy conditions (3.6), (3.19a)/(3.23a).** ``cauchy_violated`` is the
    sticky negation of the solver's ``cauchy_ok``.

    **Lemma 4.5 / 4.6.** ``kappa_g`` and ``kappa_c`` are the running maxima
    of ``|f(x⁺) − m_f(d)| / ‖w‖²`` and ``|v(x⁺) − m_v(d)| / ‖w‖²`` over the
    non-trivial steps (``‖w‖`` is the scaled step norm,
    ``‖w_n‖² + ‖t‖²``). They estimate the model-error constants
    ``κ_G, κ_C``; a blow-up signals iterates leaving the compact set of
    Assumption 4.1, collapsing slacks or a poor curvature model. From
    ``κ_C`` Lemma 4.6 gives the radius threshold
    ``κ_V = (1 − κ_tt) / (κ_C κ_v)`` below which every v-iteration must lie
    in ``D``; ``outside_d_small_radius`` flags the exceptions and
    ``n_outside_d_small_radius`` counts them.

    **Lemmas 4.8–4.10.** ``criticality_collapse`` marks a step whose
    governing radius collapsed below ``radius_floor · max{1, ‖x‖}`` while
    the matching criticality measure was still above its tolerance
    (``δᶠ`` with ``πᶠ ≥ ε_π`` on f-iterations and on y-iterations whose
    tangential step came back empty, ``δᵛ`` with ``v ≥ ε_v`` and
    ``χᵛ ≥ χ_tol`` on v-iterations) — impossible with an accurate model, so
    the streak feeds the ``model`` secant-reset channel.

    **Lemmas 4.18 / 4.22, Assumptions 4.2 / 4.3.** ``normal_ratio_max`` is
    the running maximum of ``‖P⁻¹n‖ / πᵛ``, bounded by ``2 / σ_min(Â)²``:
    growth signals a degenerating constraint Jacobian without an SVD.

    **Theorem 4.30 / Lemma 4.27.** Iteration counters ``n_y, n_f, n_v``,
    the successful sets ``n_sf = |S_f|``, ``n_sv = |S_v|``, the tangential
    resets ``n_t0 = |T₀|`` and the rejection reasons: (2.11) demotions
    ``n_demoted``, ``ρᶠ < η₁`` on f-iterations ``n_rejected_f``, ``ρᵛ < η₁``
    and (2.15) failures on v-iterations ``n_rejected_v_rho`` /
    ``n_rejected_v_contraction``. ``n_normal_skipped`` and
    ``n_multiplier_skipped`` count the closed gates (3.2) and (3.12),
    ``n_tangential_rejected`` the (3.19d)/(3.23d) discards, ``n_ftb_normal``
    / ``n_ftb_tangential`` the fraction-to-boundary truncations.

    **Multipliers (3.10), (5.4).** ``multiplier_bound_exceeded`` /
    ``n_multiplier_bound`` record ``‖y‖ > κ_y(μ)`` and ``n_d_cap_hits`` the
    cumulative number of slack-curvature entries above ``κ_D(μ)``.

    **Outer loop.** ``n_mu_updates`` and ``n_slack_resets``.
    """

    # --- iteration-type counters -----------------------------------------
    n_y: Int[Array, ""]
    n_f: Int[Array, ""]
    n_v: Int[Array, ""]
    n_sf: Int[Array, ""]
    n_sv: Int[Array, ""]
    n_t0: Int[Array, ""]
    n_demoted: Int[Array, ""]
    n_rejected_f: Int[Array, ""]
    n_rejected_v_rho: Int[Array, ""]
    n_rejected_v_contraction: Int[Array, ""]
    n_normal_skipped: Int[Array, ""]
    n_multiplier_skipped: Int[Array, ""]
    n_tangential_rejected: Int[Array, ""]
    n_ftb_normal: Int[Array, ""]
    n_ftb_tangential: Int[Array, ""]
    n_multiplier_bound: Int[Array, ""]
    n_d_cap_hits: Int[Array, ""]
    n_outside_d_small_radius: Int[Array, ""]
    n_slack_resets: Int[Array, ""]
    n_mu_updates: Int[Array, ""]
    # --- running maxima ------------------------------------------------------
    kappa_g: Scalar
    kappa_c: Scalar
    normal_ratio_max: Scalar
    # --- streaks and memory --------------------------------------------------
    pi_f_last: Scalar
    v_max_prev: Scalar
    multiplier_failure_streak: Int[Array, ""]
    criticality_collapse_streak: Int[Array, ""]
    # --- per-step flags -------------------------------------------------------
    slack_positive: Bool[Array, ""]
    residual_nonnegative: Bool[Array, ""]
    in_funnel: Bool[Array, ""]
    v_max_monotone: Bool[Array, ""]
    unhealthy_y_iteration: Bool[Array, ""]
    criticality_collapse: Bool[Array, ""]
    outside_d_small_radius: Bool[Array, ""]
    multiplier_bound_exceeded: Bool[Array, ""]
    # --- sticky flags ---------------------------------------------------------
    invariant_violated: Bool[Array, ""]
    cauchy_violated: Bool[Array, ""]

    @classmethod
    def zero(cls, v_max: Scalar | float, *, dtype: DTypeLike = float) -> Self:
        """Fresh carry for a run starting with funnel radius ``v_max``.

        Parameters
        ----------
        v_max
            Initial funnel radius ``v_max₀`` (seeds ``v_max_prev``).
        dtype
            Floating dtype of the scalar fields.

        Returns
        -------
        Self
            All counters zero, maxima zero, flags healthy.
        """
        zero_i = jnp.asarray(0, jnp.int32)
        zero_f = jnp.asarray(0.0, dtype)
        true = jnp.asarray(True)
        false = jnp.asarray(False)
        return cls(
            n_y=zero_i,
            n_f=zero_i,
            n_v=zero_i,
            n_sf=zero_i,
            n_sv=zero_i,
            n_t0=zero_i,
            n_demoted=zero_i,
            n_rejected_f=zero_i,
            n_rejected_v_rho=zero_i,
            n_rejected_v_contraction=zero_i,
            n_normal_skipped=zero_i,
            n_multiplier_skipped=zero_i,
            n_tangential_rejected=zero_i,
            n_ftb_normal=zero_i,
            n_ftb_tangential=zero_i,
            n_multiplier_bound=zero_i,
            n_d_cap_hits=zero_i,
            n_outside_d_small_radius=zero_i,
            n_slack_resets=zero_i,
            n_mu_updates=zero_i,
            kappa_g=zero_f,
            kappa_c=zero_f,
            normal_ratio_max=zero_f,
            pi_f_last=zero_f,
            v_max_prev=jnp.asarray(v_max, dtype),
            multiplier_failure_streak=zero_i,
            criticality_collapse_streak=zero_i,
            slack_positive=true,
            residual_nonnegative=true,
            in_funnel=true,
            v_max_monotone=true,
            unhealthy_y_iteration=false,
            criticality_collapse=false,
            outside_d_small_radius=false,
            multiplier_bound_exceeded=false,
            invariant_violated=false,
            cauchy_violated=false,
        )

    def kappa_v_threshold(self, solver: TrustFunnelSolver) -> Scalar:
        """Radius threshold ``κ_V = (1 − κ_tt) / (κ_C κ_v)`` of Lemma 4.6.

        Parameters
        ----------
        solver
            Funnel solver providing ``κ_tt`` and ``κ_v``.

        Returns
        -------
        Scalar
            ``κ_V`` from the current ``kappa_c`` estimate; ``+inf`` while no
            model error has been observed.
        """
        return _kappa_v_threshold(self.kappa_c, solver)

    def update(
        self,
        state: TrustFunnelSolverState,
        solver: TrustFunnelSolver,
        *,
        accepted: Bool[Array, ""],
        solved: Bool[Array, ""],
        reset: Bool[Array, ""],
        slack_positive: Bool[Array, ""],
        residual_min: Scalar,
        violation: Scalar,
        v_max_next: Scalar,
        x_norm: Scalar,
        multiplier_norm: Scalar,
        kappa_y: Scalar,
        d_cap_hits: Int[Array, ""],
        radius_floor: float,
        chi_tol: float,
        tol: float,
    ) -> Self:
        """Fold one committed step into the carry.

        Parameters
        ----------
        state
            Solver state after the controller wrote back its decision
            (radii, ``v_max``, ``rho``, ``iteration_type``, ``v_trial``,
            model errors) and *before* any ``μ``-reduction re-seeding.
        solver
            The :class:`~slsqp_jax.sqpdax.subproblem.solver.trust_funnel.TrustFunnelSolver`
            of the step, read for ``κ_ω``, ``κ_tt`` and ``κ_v``.
        accepted
            Whether the primal step was accepted.
        solved
            Whether the barrier subproblem test (3.15a) passed at the
            committed iterate (``μ`` is about to be reduced).
        reset
            Whether the slack reset moved any slack.
        slack_positive
            ``s > 0`` on every live slack of the committed iterate.
        residual_min
            ``min_i c_i(x, s)`` at the committed (reset) iterate.
        violation
            ``v = ‖c(x, s)‖₂`` at the committed iterate.
        v_max_next
            Funnel radius carried into the next step (re-seeded when
            ``solved``); stored as ``v_max_prev``.
        x_norm
            ``‖x‖`` of the committed iterate (radius-collapse scale).
        multiplier_norm
            ``‖y‖₂`` of the committed multipliers.
        kappa_y
            Multiplier cap ``κ_y(μ)`` of (3.10).
        d_cap_hits
            Number of slack-curvature entries above ``κ_D(μ)`` (5.4).
        radius_floor
            Relative floor defining a collapsed radius.
        chi_tol
            Threshold on ``χᵛ`` below which the iterate counts as a
            stationary point of the violation.
        tol
            Relative tolerance of the invariant checks.

        Returns
        -------
        Self
            Updated carry.
        """
        dtype = self.kappa_g.dtype
        one = jnp.asarray(1, jnp.int32)
        zero = jnp.asarray(0, jnp.int32)

        def count(flag: Bool[Array, ""]) -> Int[Array, ""]:
            return jnp.where(flag, one, zero)

        kind = state.iteration_type
        is_y = kind == IterationType.y_iteration
        is_f = kind == IterationType.f_iteration
        is_v = kind == IterationType.v_iteration
        success_f = is_f & accepted
        success_v = is_v & accepted
        contraction = (state.normal_norm > 0.0) & state.contraction_ok
        rejected_v_contraction = is_v & ~accepted & ~contraction
        rejected_v_rho = is_v & ~accepted & contraction

        # --- invariants ------------------------------------------------------
        scale_v = jnp.maximum(1.0, state.v_max)
        residual_nonnegative = residual_min >= -tol * (1.0 + violation)
        in_funnel = violation <= state.v_max + tol * scale_v
        v_max_monotone = state.v_max <= self.v_max_prev * (1.0 + tol)
        invariant_now = (
            ~slack_positive
            | ~residual_nonnegative
            | ~in_funnel
            | ~v_max_monotone
            | state.funnel_violated
        )

        # --- Lemmas 4.8–4.10: radius collapse with criticality bounded away ---
        # A y-iteration whose tangential step came back empty from a collapsed
        # radius is the limit of the f-iteration case.
        floor = radius_floor * jnp.maximum(1.0, x_norm)
        collapse_f = (
            (is_f | (is_y & state.tangential_computed))
            & (state.radius_f < floor)
            & (state.pi_f >= state.eps_pi)
        )
        collapse_v = (
            is_v
            & (state.radius_v < floor)
            & (state.violation >= state.eps_v)
            & (state.chi_v >= chi_tol)
        )
        collapse = collapse_f | collapse_v
        collapse_streak = jnp.where(
            collapse, self.criticality_collapse_streak + one, zero
        )

        # --- Lemma 3.3 / (3.16) health of y-iterations -------------------------
        healthy_y = (state.multiplier_case == MultiplierCase.skip_tangential) & (
            state.pi_f <= solver.kappa_omega * self.pi_f_last
        )
        unhealthy_y = is_y & ~state.kkt_satisfied & ~solved & ~healthy_y & ~collapse
        multiplier_streak = jnp.where(
            unhealthy_y, self.multiplier_failure_streak + one, zero
        )

        # --- Lemma 4.5 model-error constants ----------------------------------
        w_norm_sq = state.normal_norm**2 + state.tangential_norm**2
        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype)
        informative = ~is_y & (w_norm_sq > eps)
        safe_w = jnp.where(informative, w_norm_sq, 1.0)
        kappa_g = jnp.where(
            informative,
            jnp.maximum(self.kappa_g, state.model_error_f / safe_w),
            self.kappa_g,
        )
        kappa_c = jnp.where(
            informative,
            jnp.maximum(self.kappa_c, state.model_error_v / safe_w),
            self.kappa_c,
        )

        # --- Lemma 4.6: small radius ⇒ v-iterations lie in D -------------------
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
        kappa_v_threshold = _kappa_v_threshold(kappa_c, solver)
        radius_very_relaxed = jnp.minimum(
            state.radius_t, solver.kappa_v * self.v_max_prev
        )
        outside_d = (
            is_v
            & jnp.isfinite(kappa_v_threshold)
            & (radius_very_relaxed <= kappa_v_threshold)
            & ~state.in_d
        )

        # --- Lemma 4.18 / 4.22 degeneracy proxy ----------------------------------
        normal_ratio = jnp.where(
            state.normal_computed & (state.pi_v > tiny),
            state.normal_norm / jnp.maximum(state.pi_v, tiny),
            0.0,
        )

        bound_exceeded = multiplier_norm > kappa_y

        return cast(
            Self,
            FunnelDiagnostics(
                n_y=self.n_y + count(is_y),
                n_f=self.n_f + count(is_f),
                n_v=self.n_v + count(is_v),
                n_sf=self.n_sf + count(success_f),
                n_sv=self.n_sv + count(success_v),
                n_t0=self.n_t0 + count(state.tangential_reset),
                n_demoted=self.n_demoted + count(state.demoted),
                n_rejected_f=self.n_rejected_f + count(is_f & ~accepted),
                n_rejected_v_rho=self.n_rejected_v_rho + count(rejected_v_rho),
                n_rejected_v_contraction=self.n_rejected_v_contraction
                + count(rejected_v_contraction),
                n_normal_skipped=self.n_normal_skipped + count(~state.gate_normal),
                n_multiplier_skipped=self.n_multiplier_skipped
                + count(~state.gate_multiplier),
                n_tangential_rejected=self.n_tangential_rejected
                + count(state.tangential_rejected),
                n_ftb_normal=self.n_ftb_normal + count(state.ftb_normal),
                n_ftb_tangential=self.n_ftb_tangential + count(state.ftb_tangential),
                n_multiplier_bound=self.n_multiplier_bound + count(bound_exceeded),
                n_d_cap_hits=self.n_d_cap_hits + d_cap_hits.astype(jnp.int32),
                n_outside_d_small_radius=self.n_outside_d_small_radius
                + count(outside_d),
                n_slack_resets=self.n_slack_resets + count(reset),
                n_mu_updates=self.n_mu_updates + count(solved),
                kappa_g=kappa_g.astype(dtype),
                kappa_c=kappa_c.astype(dtype),
                normal_ratio_max=jnp.maximum(
                    self.normal_ratio_max, normal_ratio
                ).astype(dtype),
                pi_f_last=state.pi_f.astype(dtype),
                v_max_prev=jnp.asarray(v_max_next, dtype),
                multiplier_failure_streak=multiplier_streak.astype(jnp.int32),
                criticality_collapse_streak=collapse_streak.astype(jnp.int32),
                slack_positive=slack_positive,
                residual_nonnegative=residual_nonnegative,
                in_funnel=in_funnel,
                v_max_monotone=v_max_monotone,
                unhealthy_y_iteration=unhealthy_y,
                criticality_collapse=collapse,
                outside_d_small_radius=outside_d,
                multiplier_bound_exceeded=bound_exceeded,
                invariant_violated=self.invariant_violated | invariant_now,
                cauchy_violated=self.cauchy_violated | ~state.cauchy_ok,
            ),
        )
