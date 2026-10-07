"""Trust-funnel composite-step orchestrator (Curtis, Gould, Robinson & Toint 2017, Alg. 2)."""

from typing import Literal, Self, cast

import jax
from equinox import Enumeration, field, tree_at
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Scalar

from ...dual import Dual
from ...linalg import (
    ResolvedNormalEquationsStrategy,
    SchurNormalEquations,
    power_iteration_norm,
    spectral_norm_estimate,
    strategy_code,
)
from ...primal import InteriorPointPrimal
from ..funnel_barrier import FunnelBarrierSubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState
from .funnel_multipliers import (
    ForcingFunction,
    MultiplierCase,
    classify_multiplier_case,
    default_omega_n,
    default_omega_t,
    satisfies_forcing_condition,
)
from .funnel_tangential_step import (
    FunnelTangentialStepSolver,
    FunnelTangentialStepState,
)
from .multiplier_recovery import KKTMultiplierRecovery, MultiplierRecovery
from .scaled_normal_step import ScaledNormalStepSolver, ScaledNormalStepState

__all__ = [
    "IterationType",
    "TrustFunnelSolverState",
    "TrustFunnelSolver",
]


class IterationType(Enumeration):
    """Trust-funnel iteration classes (Definitions 2.1–2.3)."""

    y_iteration = "y-iteration: d = 0, only the multipliers change."
    f_iteration = (
        "f-iteration: t ≠ 0 with (2.10); the controller still requires (2.11)."
    )
    v_iteration = "v-iteration: d ≠ 0 but not an f-iteration."


class TrustFunnelSolverState(SubProblemSolverState):
    """Carry threaded across trust-funnel solves.

    The first group is owned by the outer loop (controller / minimiser) and
    read by the solver; the second group is written by the solver for the
    controller and the diagnostics.

    Attributes
    ----------
    radius_v
        Normal trust-region radius ``δᵛ``. The solver may enlarge it through
        the reset (3.31) when :attr:`sf_flag` is set.
    radius_f
        Tangential trust-region radius ``δᶠ``.
    v_max
        Funnel radius ``v_max``.
    eps_pi, eps_v
        Tolerances ``ε_π``, ``ε_v`` of the current barrier subproblem used in
        (3.15a).
    pi_f_prev
        ``πᶠ_{k−1}`` for the normal gate (3.2); refreshed to ``πᶠ_k`` by every
        solve.
    sf_flag
        ``True`` after a successful f-iteration until the next normal step
        is computed (Steps 11–13, 43).
    rho
        Actual / predicted ratio written back by the controller; untouched
        by the solver.
    iteration_type
        Provisional classification: ``y`` when ``d = 0``, ``f`` when
        ``t ≠ 0`` and (2.10) holds, else ``v``. The controller demotes ``f``
        to ``v`` when (2.11) fails.
    radius_t
        Tangential radius ``δᵗ`` actually used, (3.38).
    violation, pi_v, chi_v
        ``v_k``, ``πᵛ_k``, ``χᵛ_k`` (3.1).
    pi_f, chi_f
        ``πᶠ_k``, ``χᶠ_k`` (3.14) at the returned multipliers.
    multiplier_case
        Branch of (3.15) that held (``tangential`` when neither (3.15a) nor
        (3.15b) did).
    multiplier_acceptable
        Any of (3.15a)–(3.15c) held for the returned multipliers.
    kkt_satisfied
        (3.15a) held: the returned step is zero and the minimiser should
        terminate (Steps 21 / 35).
    infeasible_stationary
        ``v > 0`` with ``πᵛ = 0`` (Steps 8–9).
    normal_computed, tangential_computed
        ``k ∈ N`` and ``k ∈ T``.
    tangential_reset
        ``k ∈ T₀``: the tangential step satisfied (3.19) and (3.20) but not
        (2.10) and was reset to zero (Steps 30–31).
    tangential_rejected
        The computed tangential step violated (3.19d) / (3.23d) and was set
        to zero (only possible when the multiplier estimate is inexact).
    in_td, in_d
        ``k ∈ T_D`` and ``k ∈ D``.
    dm_f_n, dm_f_t, dm_v_n, dm_v_d
        Model decreases ``Δm_f,n``, ``Δm_f,t``, ``Δm_v,n``, ``Δm_v,d``
        (2.9), (2.14).
    objective_decrease_ok
        (2.10) ``Δm_f,d ≥ κ_δ Δm_f,t`` for the returned step.
    contraction_ok
        Second condition of (2.15) ``Δm_v,d ≥ κ_cd Δm_v,n``.
    normal_norm, tangential_norm
        ``‖P⁻¹ n‖``, ``‖P⁻¹ t‖``.
    cauchy_ratio_v, cauchy_ratio_f
        ``Δm_v,n / (m_v(0) − m_v(n_C))`` and ``Δm_f,t / (m_f(n) − m_f(n + t_C))``
        (``1`` when the Cauchy decrease is zero); both must be ``≥ 1``.
    cauchy_ok
        Both Cauchy ratios are at least ``1 − cauchy_tol`` ((3.6) and
        (3.19a)/(3.23a) hold for the returned sub-steps).
    gate_normal, gate_multiplier
        The normal gate (3.2) and the multiplier gate (3.12) held at this
        iteration (``normal_computed`` additionally requires ``πᵛ > 0``).
    very_relaxed
        The tangential step used the very relaxed radius (3.22).
    ftb_normal, ftb_tangential
        The corresponding sub-step was cut by its fraction-to-boundary rule.
    n_cg_iter
        Cumulative inner iterations of the normal and tangential solvers.
    v_trial
        ``v(x⁺, s⁺)`` at the trial point, written by the controller.
    model_error_f, model_error_v
        ``|f(x⁺) − m_f(d)|`` and ``|v(x⁺) − m_v(d)|`` at the trial point,
        written by the controller (zero on y-iterations).
    funnel_violated
        The controller accepted an iterate with ``v⁺ > v_max`` — impossible
        for an exact implementation of (2.11)/(3.35).
    demoted
        A step with ``t ≠ 0`` and (2.10) was classified as a v-iteration
        because (2.11) ``v⁺ ≤ v_max`` failed (written by the controller).
    """

    radius_v: Scalar
    radius_f: Scalar
    v_max: Scalar
    eps_pi: Scalar
    eps_v: Scalar
    pi_f_prev: Scalar
    sf_flag: Bool[Array, ""]
    rho: Scalar
    iteration_type: IterationType
    radius_t: Scalar
    violation: Scalar
    pi_v: Scalar
    chi_v: Scalar
    pi_f: Scalar
    chi_f: Scalar
    multiplier_case: MultiplierCase
    multiplier_acceptable: Bool[Array, ""]
    kkt_satisfied: Bool[Array, ""]
    infeasible_stationary: Bool[Array, ""]
    normal_computed: Bool[Array, ""]
    tangential_computed: Bool[Array, ""]
    tangential_reset: Bool[Array, ""]
    tangential_rejected: Bool[Array, ""]
    in_td: Bool[Array, ""]
    in_d: Bool[Array, ""]
    dm_f_n: Scalar
    dm_f_t: Scalar
    dm_v_n: Scalar
    dm_v_d: Scalar
    objective_decrease_ok: Bool[Array, ""]
    contraction_ok: Bool[Array, ""]
    normal_norm: Scalar
    tangential_norm: Scalar
    cauchy_ratio_v: Scalar
    cauchy_ratio_f: Scalar
    cauchy_ok: Bool[Array, ""]
    gate_normal: Bool[Array, ""]
    gate_multiplier: Bool[Array, ""]
    very_relaxed: Bool[Array, ""]
    ftb_normal: Bool[Array, ""]
    ftb_tangential: Bool[Array, ""]
    n_cg_iter: int
    v_trial: Scalar
    model_error_f: Scalar
    model_error_v: Scalar
    funnel_violated: Bool[Array, ""]
    demoted: Bool[Array, ""]

    @classmethod
    def cold(
        cls,
        radius_v: Scalar | float,
        radius_f: Scalar | float,
        v_max: Scalar | float,
        *,
        eps_pi: Scalar | float = 1e-6,
        eps_v: Scalar | float = 1e-6,
        dtype: DTypeLike = float,
    ) -> Self:
        """Cold carry for iteration ``k = 0`` (``πᶠ_{−1} = 0``, flag cleared).

        Parameters
        ----------
        radius_v, radius_f
            Initial radii ``δᵛ₀``, ``δᶠ₀``.
        v_max
            Initial funnel radius ``v_max₀``.
        eps_pi, eps_v
            Tolerances of the barrier subproblem.
        dtype
            Floating dtype of the scalar fields.

        Returns
        -------
        Self
            Cold state with zero counters / decreases and ``success=False``.
        """
        zero = jnp.asarray(0.0, dtype)
        false = jnp.asarray(False)
        return cls(
            n_iter=jnp.asarray(0, jnp.int32),
            success=false,
            status=RESULTS.successful,
            radius_v=jnp.asarray(radius_v, dtype),
            radius_f=jnp.asarray(radius_f, dtype),
            v_max=jnp.asarray(v_max, dtype),
            eps_pi=jnp.asarray(eps_pi, dtype),
            eps_v=jnp.asarray(eps_v, dtype),
            pi_f_prev=zero,
            sf_flag=false,
            rho=jnp.asarray(1.0, dtype),
            iteration_type=IterationType.y_iteration,
            radius_t=jnp.asarray(1.0, dtype),
            violation=zero,
            pi_v=zero,
            chi_v=zero,
            pi_f=zero,
            chi_f=zero,
            multiplier_case=MultiplierCase.tangential,
            multiplier_acceptable=false,
            kkt_satisfied=false,
            infeasible_stationary=false,
            normal_computed=false,
            tangential_computed=false,
            tangential_reset=false,
            tangential_rejected=false,
            in_td=false,
            in_d=false,
            dm_f_n=zero,
            dm_f_t=zero,
            dm_v_n=zero,
            dm_v_d=zero,
            objective_decrease_ok=false,
            contraction_ok=false,
            normal_norm=zero,
            tangential_norm=zero,
            cauchy_ratio_v=jnp.asarray(1.0, dtype),
            cauchy_ratio_f=jnp.asarray(1.0, dtype),
            cauchy_ok=jnp.asarray(True),
            gate_normal=false,
            gate_multiplier=false,
            very_relaxed=false,
            ftb_normal=false,
            ftb_tangential=false,
            n_cg_iter=jnp.asarray(0, jnp.int32),
            v_trial=zero,
            model_error_f=zero,
            model_error_v=zero,
            funnel_violated=false,
            demoted=false,
        )


class TrustFunnelSolver(
    SubProblemSolver[
        InteriorPointPrimal, FunnelBarrierSubProblem, TrustFunnelSolverState
    ]
):
    """Steps 7–37 of Algorithm 2: normal step, multipliers, tangential step.

    One call computes the trial step ``d_k = n_k + t_k`` and the multiplier
    estimate ``y_k`` for the current
    :class:`~slsqp_jax.sqpdax.subproblem.funnel_barrier.FunnelBarrierSubProblem`,
    whose Lagrangian is assumed to be evaluated at ``(x_k, s_k, yᴮ_k)`` with
    ``yᴮ_k`` the positive, bounded multipliers of (3.10) (the caller applies
    the barrier safeguard when building the subproblem; the multipliers
    returned here are the raw least-squares estimate of (2.7)). Step
    acceptance, the radius / funnel updates (Steps 38–50) and termination
    belong to the step controller and the minimiser.

    The flow, with every branch a ``lax.cond`` so that skipped solves cost
    nothing:

    1. ``v, πᵛ, χᵛ`` (3.1); ``v > 0 ∧ πᵛ = 0`` is reported as
       :attr:`~TrustFunnelSolverState.infeasible_stationary`.
    2. Normal gate (3.2) (optionally also ``πᵛ > 0``); when the gate opens
       and :attr:`~TrustFunnelSolverState.sf_flag` is set, ``δᵛ`` is reset
       by (3.31) using ``‖P⁻¹ n*‖`` of (3.8); then the normal solver runs.
    3. Gate (3.12). If it holds, the multipliers are recovered and
       classified by (3.15); otherwise ``y_k ← y_{k−1}``, ``t_k ← 0``, and
       ``πᶠ, χᶠ`` are still evaluated so (3.15a) can terminate.
    4. Tangential step with radius (3.38): the relaxed radius
       ``min{κ_vf δᵛ, δᶠ}`` when ``n_k ≠ 0``; when ``n_k = 0`` the policy
       :attr:`zero_normal_tangential` selects the relaxed (3.19) or the very
       relaxed (3.23) step. The step is kept only if (3.19d) / (3.23d)
       holds — automatic for the projected CG step, possible to fail for the
       Cauchy fallback with inexact multipliers.
    5. (3.20) ∧ ¬(2.10) reset (``k ∈ T₀``), ``D`` membership, ``Δm``
       bookkeeping, provisional Y / F / V classification, native-scale step.

    Attributes
    ----------
    solver_state_class
        :class:`TrustFunnelSolverState`.
    kappa_vf
        ``κ_vf > 0`` in ``min{κ_vf δᵛ, δᶠ}``.
    kappa_B
        ``κ_B ∈ (0, 1)`` of the gate (3.12).
    kappa_vv
        ``κ_vv ∈ (0, 1)`` of the normal gate (3.2).
    kappa_tt
        ``κ_tt ∈ (κ_vv, 1)`` of (3.23d).
    kappa_tg
        ``κ_tg ∈ (0, 1)`` of (3.19d).
    kappa_cd
        ``κ_cd ∈ (0, 1 − κ_tg]`` of (2.15).
    kappa_delta
        ``κ_δ ∈ (0, 1)`` of (2.10).
    kappa_tn
        ``κ_tn > 1`` of (3.20).
    kappa_v
        ``κ_v > 1`` of the very relaxed radius (3.22).
    kappa_n
        ``κ_n > 0`` of the radius reset (3.31).
    kappa_chi
        ``κ_χ ∈ (0, 1)`` of (3.15c).
    kappa_omega
        ``κ_ω ∈ (0, 1)`` of the forcing compatibility (3.16), checked at
        construction on a grid.
    omega_n, omega_t
        Forcing functions of (3.2) and (3.15b).
    always_normal
        Also compute a normal step whenever ``πᵛ > 0`` (the optional
        extension of Step 10).
    zero_normal_tangential
        ``"very_relaxed"`` (default, the paper's recommendation) or
        ``"relaxed"``: which tangential step to compute when ``n_k = 0``.
    multiplier_recovery
        Strategy for (2.7); default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.KKTMultiplierRecovery`
        without safeguard.
    normal_solver, tangential_solver
        The two sub-solvers.
    cauchy_tol
        Relative slack on the Cauchy ratios: ``cauchy_ok`` requires
        ``Δm ≥ (1 − cauchy_tol) × (Cauchy decrease)``.
    norm_estimate_iters
        Power iterations spent on the ``‖Â‖₂`` / ``‖Ĥ‖₂`` estimates of the
        ``funnel_step`` diagnostic record (only traced when the diagnostics
        channel is on).

    Besides the text log, an enabled diagnostics channel receives one
    ``"funnel_step"`` record per solve carrying the criticality measures,
    gates, radii, both sub-solver states, the scaled sub-steps, the
    multipliers, the Lemma 3.5 / 3.9 Cauchy lower bounds evaluated with
    power-iteration estimates of ``‖Â‖₂`` and ``‖Ĥ‖₂`` (see
    :meth:`cauchy_lower_bounds`), and the resolved normal-equations
    strategy (``normal_equations_strategy``: ``0`` generic, ``1`` schur,
    ``2`` matrix-free) with the rank of the explicit Schur complement
    (``normal_equations_rank``, ``-1`` when no explicit factor is built).

    The ``(Â Âᵀ)⁺`` solves of the multiplier recovery and of the tangential
    projector follow the ``normal_equations`` strategy of the respective
    component; when either resolves to ``"schur"`` the factor is built once
    per solve here and attached to the subproblem
    (:meth:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem.with_schur_normal_equations`)
    so both share it.

    Notes
    -----
    The paper states only admissible intervals for the constants; the
    defaults here are conventional choices inside them. In particular
    ``κ_vf κ_B = 1.2 > 1`` makes the gate (3.12) read
    ``‖P⁻¹n‖ ≤ min{1.2 δᵛ, κ_B δᶠ}``: a normal step that saturates ``δᵛ``
    still admits a tangential step (robustly, not at float equality), and
    only the barrier radius ``δᶠ`` can veto it (the N&W §19.5 rule that the
    normal step may use at most a fraction ``κ_B`` of the tangential radius).
    """

    solver_state_class: type[TrustFunnelSolverState] = TrustFunnelSolverState

    kappa_vf: float = field(static=True, default=1.5)
    kappa_B: float = field(static=True, default=0.8)
    kappa_vv: float = field(static=True, default=0.5)
    kappa_tt: float = field(static=True, default=0.9)
    kappa_tg: float = field(static=True, default=0.1)
    kappa_cd: float = field(static=True, default=0.1)
    kappa_delta: float = field(static=True, default=0.1)
    kappa_tn: float = field(static=True, default=2.0)
    kappa_v: float = field(static=True, default=2.0)
    kappa_n: float = field(static=True, default=1.0)
    kappa_chi: float = field(static=True, default=0.1)
    kappa_omega: float = field(static=True, default=0.5)
    omega_n: ForcingFunction = field(default_factory=default_omega_n)
    omega_t: ForcingFunction = field(default_factory=default_omega_t)
    always_normal: bool = field(static=True, default=False)
    zero_normal_tangential: Literal["relaxed", "very_relaxed"] = field(
        static=True, default="very_relaxed"
    )
    multiplier_recovery: MultiplierRecovery = field(
        default_factory=KKTMultiplierRecovery
    )
    normal_solver: ScaledNormalStepSolver = field(
        default_factory=ScaledNormalStepSolver
    )
    tangential_solver: FunnelTangentialStepSolver = field(
        default_factory=FunnelTangentialStepSolver
    )
    cauchy_tol: float = field(static=True, default=1e-6)
    norm_estimate_iters: int = field(static=True, default=10)

    def __check_init__(self):
        def require(ok: bool, msg: str):
            if not ok:
                raise ValueError(msg)

        require(
            self.cauchy_tol >= 0.0, f"cauchy_tol must be >= 0; got {self.cauchy_tol}"
        )
        require(
            self.norm_estimate_iters >= 1,
            f"norm_estimate_iters must be >= 1; got {self.norm_estimate_iters}",
        )
        require(self.kappa_vf > 0.0, f"kappa_vf must be > 0; got {self.kappa_vf}")
        require(
            0.0 < self.kappa_B < 1.0, f"kappa_B must lie in (0, 1); got {self.kappa_B}"
        )
        require(
            0.0 < self.kappa_vv < 1.0,
            f"kappa_vv must lie in (0, 1); got {self.kappa_vv}",
        )
        require(
            self.kappa_vv < self.kappa_tt < 1.0,
            f"kappa_tt must lie in (kappa_vv, 1); got {self.kappa_tt}",
        )
        require(
            0.0 < self.kappa_tg < 1.0,
            f"kappa_tg must lie in (0, 1); got {self.kappa_tg}",
        )
        require(
            0.0 < self.kappa_cd <= 1.0 - self.kappa_tg,
            f"kappa_cd must lie in (0, 1 - kappa_tg]; got {self.kappa_cd}",
        )
        require(
            0.0 < self.kappa_delta < 1.0,
            f"kappa_delta must lie in (0, 1); got {self.kappa_delta}",
        )
        require(self.kappa_tn > 1.0, f"kappa_tn must be > 1; got {self.kappa_tn}")
        require(self.kappa_v > 1.0, f"kappa_v must be > 1; got {self.kappa_v}")
        require(self.kappa_n > 0.0, f"kappa_n must be > 0; got {self.kappa_n}")
        require(
            0.0 < self.kappa_chi < 1.0,
            f"kappa_chi must lie in (0, 1); got {self.kappa_chi}",
        )
        require(
            0.0 < self.kappa_omega < 1.0,
            f"kappa_omega must lie in (0, 1); got {self.kappa_omega}",
        )
        require(
            self.zero_normal_tangential in ("relaxed", "very_relaxed"),
            "zero_normal_tangential must be 'relaxed' or 'very_relaxed'; got "
            f"{self.zero_normal_tangential!r}",
        )
        # The solver is rebuilt inside the minimiser's traced loop body; the
        # forcing-function check only involves static constants, so evaluate
        # it eagerly rather than letting omnistaging turn it into a tracer.
        with jax.ensure_compile_time_eval():
            taus = jnp.array([1e-8, 1e-4, 1.0, 1e4])
            forcing_ok = bool(
                satisfies_forcing_condition(
                    self.omega_n, self.omega_t, self.kappa_omega, taus
                )
            )
        require(
            forcing_ok,
            "omega_n / omega_t violate (3.16): omega_t(omega_n(tau)) > kappa_omega tau",
        )

    # ------------------------------------------------------------------
    def solve(
        self,
        subproblem: FunnelBarrierSubProblem,
        x0: tuple[InteriorPointPrimal, Dual],
        initial_state: TrustFunnelSolverState,
    ) -> tuple[tuple[InteriorPointPrimal, Dual], TrustFunnelSolverState]:
        """Compute the trial step ``d_k`` and multipliers ``y_k``.

        Parameters
        ----------
        subproblem
            Funnel barrier subproblem at ``(x_k, s_k)`` built with ``yᴮ_k``;
            its ``lagrangian.dual`` is taken as ``y_{k−1}``.
        x0
            Unused warm start (both sub-solvers start from zero); kept for
            the :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolver`
            interface.
        initial_state
            Carry with the radii, ``v_max``, tolerances, ``πᶠ_{k−1}`` and
            the ``S_f`` flag.

        Returns
        -------
        step
            Native-scale primal step ``d = (n_x + t_x, S (w_n,s + t_s))`` and
            the multipliers ``y_k`` (``y_{k−1}`` when (3.12) failed). Zero
            primal step when (3.15a) holds.
        state
            Refreshed :class:`TrustFunnelSolverState`.

        Raises
        ------
        TypeError
            If ``subproblem`` is not a ``FunnelBarrierSubProblem``.
        """
        del x0
        if not isinstance(subproblem, FunnelBarrierSubProblem):
            raise TypeError(
                "subproblem must be a FunnelBarrierSubProblem. Got "
                f"{type(subproblem)} instead."
            )
        st = initial_state
        lag = subproblem.lagrangian
        n, mineq = lag.n, lag.mineq
        dtype = lag.ref.x.dtype
        tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
        # Share one Schur factor of Â Âᵀ between the multiplier recovery and the
        # tangential projector when either resolves to the explicit path; it
        # is built once here, outside the ``lax.cond`` branches below.
        subproblem, ne_strategy, ne_rank = self._attach_normal_equations(subproblem)
        zero_w = subproblem._zero_primal()
        zero_dual = subproblem._zero_dual()
        y_prev = lag.dual

        def to_primal(w: Array) -> InteriorPointPrimal:
            return InteriorPointPrimal.from_flat(w, n, mineq)

        # --- Step 7–9: criticality at the iterate ------------------------------
        v = subproblem.violation()
        pi_v = subproblem.pi_v()
        chi_v = subproblem.chi_v()
        infeasible_stationary = (v > 0.0) & (pi_v <= 0.0)

        # --- Step 10: normal gate (3.2) ----------------------------------------
        gate_normal = (pi_v > self.omega_n(st.pi_f_prev)) | (
            v >= self.kappa_vv * st.v_max
        )
        if self.always_normal:
            gate_normal = gate_normal | (pi_v > 0.0)
        compute_normal = gate_normal & (pi_v > 0.0)

        # --- Steps 11–13: radius reset (3.31) after a successful f-iteration ---
        n_star_norm = self._unconstrained_cauchy_norm(subproblem, tiny)
        reset_radius = compute_normal & st.sf_flag
        radius_v = jnp.where(
            reset_radius,
            jnp.maximum(st.radius_v, self.kappa_n * n_star_norm),
            st.radius_v,
        )
        sf_flag = st.sf_flag & ~compute_normal

        # --- Step 14–16: normal step -------------------------------------------
        normal_cold = ScaledNormalStepState.cold(radius_v, dtype)

        def run_normal(_):
            return self.normal_solver.solve(
                subproblem, (zero_w, zero_dual), normal_cold
            )

        def skip_normal(_):
            return (zero_w, zero_dual), normal_cold

        (w_n, _), normal_state = jax.lax.cond(
            compute_normal, run_normal, skip_normal, None
        )
        w_n_flat = w_n.flatten()
        normal_norm = jnp.linalg.norm(w_n_flat)
        m_v_n = subproblem.model_v(w_n)
        dm_v_n = v - m_v_n
        dm_f_n = -subproblem.model_f(w_n)
        cauchy_v = normal_state.cauchy_decrease

        # --- Step 18 / 32–33: gate (3.12) and multipliers ----------------------
        radius_td = jnp.minimum(self.kappa_vf * radius_v, st.radius_f)
        gate_mult = normal_norm <= self.kappa_B * radius_td

        def recover(_):
            return self.multiplier_recovery.recover(subproblem, None, w_n)

        y = jax.lax.cond(gate_mult, recover, lambda _: y_prev, None)
        pi_f = subproblem.pi_f(w_n, y)
        chi_f = subproblem.chi_f(w_n, y)
        classification = classify_multiplier_case(
            pi_f,
            chi_f,
            pi_v,
            v,
            eps_pi=st.eps_pi,
            eps_v=st.eps_v,
            kappa_chi=self.kappa_chi,
            omega_t=self.omega_t,
        )
        kkt = classification.kkt_satisfied

        # --- Steps 24–28: tangential step with radius (3.38) --------------------
        compute_tangential = gate_mult & ~kkt & ~classification.infeasibility_dominates
        if self.zero_normal_tangential == "very_relaxed":
            very_relaxed = compute_tangential & ~compute_normal
        else:
            very_relaxed = jnp.asarray(False)
        radius_t = jnp.where(
            very_relaxed, jnp.minimum(radius_td, self.kappa_v * st.v_max), radius_td
        )
        tang_cold = FunnelTangentialStepState.cold(radius_t, dtype)

        def run_tangential(_):
            return self.tangential_solver.solve(subproblem, (w_n, y), tang_cold)

        def skip_tangential(_):
            return (zero_w, zero_dual), tang_cold

        (t, _), tang_state = jax.lax.cond(
            compute_tangential, run_tangential, skip_tangential, None
        )
        t_flat = t.flatten()
        m_v_nt = jnp.where(compute_tangential, tang_state.model_v_after, m_v_n)
        # Roundoff slack: the projected step has m_v(n + t) = m_v(n) up to eps.
        slack = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype)) * (1.0 + v)
        cond_319d = m_v_nt <= self.kappa_tg * v + (1.0 - self.kappa_tg) * m_v_n + slack
        cond_323d = m_v_nt <= self.kappa_tt * st.v_max + slack
        accepted = jnp.where(very_relaxed, cond_323d, cond_319d)
        tangential_rejected = compute_tangential & ~accepted
        keep = compute_tangential & accepted
        m_v_d = jnp.where(keep, m_v_nt, m_v_n)
        t_flat = jnp.where(keep, t_flat, 0.0)
        dm_f_t = jnp.where(keep, tang_state.dm_f_t, 0.0)
        cauchy_f = jnp.where(keep, tang_state.cauchy_decrease, 0.0)
        in_td = keep & ~very_relaxed

        # --- Steps 29–31: (3.20) ∧ ¬(2.10) reset, k ∈ T₀ -----------------------
        tangential_norm = jnp.linalg.norm(t_flat)
        decrease_ok = dm_f_n + dm_f_t >= self.kappa_delta * dm_f_t
        large_t = tangential_norm > self.kappa_tn * normal_norm
        tangential_reset = in_td & large_t & ~decrease_ok
        t_flat = jnp.where(tangential_reset, 0.0, t_flat)
        dm_f_t = jnp.where(tangential_reset, 0.0, dm_f_t)
        cauchy_f = jnp.where(tangential_reset, 0.0, cauchy_f)
        tangential_norm = jnp.where(tangential_reset, 0.0, tangential_norm)
        m_v_d = jnp.where(tangential_reset, m_v_n, m_v_d)

        # --- Step 21 / 35: (3.15a) terminates with a zero step -----------------
        zero_step = kkt
        w_n_flat = jnp.where(zero_step, 0.0, w_n_flat)
        t_flat = jnp.where(zero_step, 0.0, t_flat)
        normal_norm = jnp.where(zero_step, 0.0, normal_norm)
        tangential_norm = jnp.where(zero_step, 0.0, tangential_norm)
        normal_computed = compute_normal & ~zero_step
        dm_v_n = jnp.where(zero_step, 0.0, dm_v_n)
        dm_f_n = jnp.where(zero_step, 0.0, dm_f_n)
        dm_f_t = jnp.where(zero_step, 0.0, dm_f_t)
        cauchy_v = jnp.where(zero_step, 0.0, cauchy_v)
        cauchy_f = jnp.where(zero_step, 0.0, cauchy_f)
        m_v_n = jnp.where(zero_step, v, m_v_n)
        m_v_d = jnp.where(zero_step, v, m_v_d)

        # --- Steps 36–37: D membership, Δm bookkeeping, trial step -------------
        in_d = m_v_d <= self.kappa_tg * v + (1.0 - self.kappa_tg) * m_v_n + slack
        dm_v_d = v - m_v_d
        dm_f_d = dm_f_n + dm_f_t
        decrease_ok = dm_f_d >= self.kappa_delta * dm_f_t
        contraction_ok = dm_v_d >= self.kappa_cd * dm_v_n
        w_flat = w_n_flat + t_flat
        d_zero = jnp.all(w_flat == 0.0)
        iteration_type = IterationType.where(
            d_zero,
            IterationType.y_iteration,
            IterationType.where(
                (tangential_norm > 0.0) & decrease_ok,
                IterationType.f_iteration,
                IterationType.v_iteration,
            ),
        )
        w = to_primal(w_flat)
        step_primal = cast(
            InteriorPointPrimal,
            InteriorPointPrimal(x=w.x, slack=subproblem._slack_to_orig_scale(w.slack)),
        )

        def ratio(dm: Scalar, cauchy: Scalar) -> Scalar:
            return jnp.where(cauchy > 0.0, dm / jnp.maximum(cauchy, tiny), 1.0)

        cauchy_ratio_v = ratio(dm_v_n, cauchy_v)
        cauchy_ratio_f = ratio(dm_f_t, cauchy_f)
        cauchy_ok = (cauchy_ratio_v >= 1.0 - self.cauchy_tol) & (
            cauchy_ratio_f >= 1.0 - self.cauchy_tol
        )
        ftb_normal = normal_computed & normal_state.ftb_truncated
        ftb_tangential = keep & ~zero_step & tang_state.ftb_truncated
        n_cg = normal_state.n_cg_iter + tang_state.n_cg_iter
        finite = (
            jnp.all(jnp.isfinite(step_primal.flatten()))
            & jnp.all(jnp.isfinite(y.flatten()))
            & jnp.isfinite(dm_f_d)
            & jnp.isfinite(dm_v_d)
        )
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)

        self.logger.debug(
            "funnel step: v={v:.3e} pi_v={pi_v:.3e} pi_f={pi_f:.3e} chi_f={chi_f:.3e} "
            "radius_v={radius_v:.3e} radius_f={radius_f:.3e} radius_t={radius_t:.3e} "
            "normal={normal} tangential={tangential} |n|={nn:.3e} |t|={tn:.3e} "
            "dm_f_n={dm_f_n:.3e} dm_f_t={dm_f_t:.3e} dm_v_n={dm_v_n:.3e} "
            "dm_v_d={dm_v_d:.3e} reset_t={reset_t} in_D={in_d} cg_iters={cg_iters}",
            v=v,
            pi_v=pi_v,
            pi_f=pi_f,
            chi_f=chi_f,
            radius_v=radius_v,
            radius_f=st.radius_f,
            radius_t=radius_t,
            normal=normal_computed,
            tangential=compute_tangential,
            nn=normal_norm,
            tn=tangential_norm,
            dm_f_n=dm_f_n,
            dm_f_t=dm_f_t,
            dm_v_n=dm_v_n,
            dm_v_d=dm_v_d,
            reset_t=tangential_reset,
            in_d=in_d,
            cg_iters=n_cg,
        )
        self.logger.warning(
            "multiplier estimate satisfies none of (3.15a)-(3.15c): "
            "pi_f={pi_f:.3e} chi_f={chi_f:.3e} pi_v={pi_v:.3e}",
            when=gate_mult & ~classification.acceptable,
            pi_f=pi_f,
            chi_f=chi_f,
            pi_v=pi_v,
        )
        self.logger.warning(
            "tangential step violated (3.19d)/(3.23d) and was discarded: "
            "m_v(n+t)={m_v_d:.3e} m_v(n)={m_v_n:.3e} v_max={v_max:.3e}",
            when=tangential_rejected,
            m_v_d=m_v_nt,
            m_v_n=m_v_n,
            v_max=st.v_max,
        )
        self.logger.warning(
            "funnel step is non-finite (status={status})",
            when=~finite,
            status=status,
        )
        self.logger.warning(
            "sub-step below its Cauchy decrease: cauchy_ratio_v={rv:.6f} "
            "cauchy_ratio_f={rf:.6f} (tolerance {tol:.1e})",
            when=~cauchy_ok,
            rv=cauchy_ratio_v,
            rf=cauchy_ratio_f,
            tol=jnp.asarray(self.cauchy_tol),
        )

        def funnel_step_payload() -> dict:
            a_norm, h_norm = self.operator_norms(subproblem)
            bound_v, bound_f = self.cauchy_lower_bounds(
                subproblem,
                a_norm=a_norm,
                h_norm=h_norm,
                radius_v=radius_v,
                radius_t=radius_t,
                pi_f=pi_f,
            )
            nan = jnp.asarray(jnp.nan, dtype)
            bound_ratio_v = jnp.where(
                normal_computed & (bound_v > 0.0),
                cauchy_v / jnp.maximum(bound_v, tiny),
                nan,
            )
            lemma_3_9_applies = keep & ~zero_step & (chi_f >= self.kappa_chi * pi_f)
            bound_ratio_f = jnp.where(
                lemma_3_9_applies & (bound_f > 0.0),
                cauchy_f / jnp.maximum(bound_f, tiny),
                nan,
            )
            return {
                "violation": v,
                "pi_v": pi_v,
                "chi_v": chi_v,
                "pi_f": pi_f,
                "chi_f": chi_f,
                "pi_f_prev": st.pi_f_prev,
                "radius_v": radius_v,
                "radius_f": st.radius_f,
                "radius_t": radius_t,
                "v_max": st.v_max,
                "eps_pi": st.eps_pi,
                "eps_v": st.eps_v,
                "gate_normal": gate_normal,
                "gate_multiplier": gate_mult,
                "normal_computed": normal_computed,
                "tangential_computed": compute_tangential,
                "very_relaxed": very_relaxed,
                "tangential_rejected": tangential_rejected,
                "tangential_reset": tangential_reset,
                "in_td": in_td,
                "in_d": in_d,
                "multiplier_case": classification.case,
                "multiplier_acceptable": classification.acceptable,
                "kkt_satisfied": kkt,
                "infeasible_stationary": infeasible_stationary,
                "iteration_type": iteration_type,
                "dm_f_n": dm_f_n,
                "dm_f_t": dm_f_t,
                "dm_v_n": dm_v_n,
                "dm_v_d": dm_v_d,
                "normal_norm": normal_norm,
                "tangential_norm": tangential_norm,
                "normal_ratio": jnp.where(
                    pi_v > 0.0, normal_norm / jnp.maximum(pi_v, tiny), nan
                ),
                "cauchy_decrease_v": cauchy_v,
                "cauchy_decrease_f": cauchy_f,
                "cauchy_ratio_v": cauchy_ratio_v,
                "cauchy_ratio_f": cauchy_ratio_f,
                "cauchy_ok": cauchy_ok,
                "a_norm": a_norm,
                "h_norm": h_norm,
                "cauchy_bound_v": bound_v,
                "cauchy_bound_f": bound_f,
                "cauchy_bound_ratio_v": bound_ratio_v,
                "cauchy_bound_ratio_f": bound_ratio_f,
                "ftb_normal": ftb_normal,
                "ftb_tangential": ftb_tangential,
                "normal_step": to_primal(w_n_flat),
                "tangential_step": to_primal(t_flat),
                "step": step_primal,
                "multipliers": y,
                "normal_state": normal_state,
                "tangential_state": tang_state,
                "cg_iters": n_cg,
                "finite": finite,
                "normal_equations_strategy": strategy_code(ne_strategy),
                "normal_equations_rank": ne_rank,
            }

        self.logger.diagnostic("funnel_step", funnel_step_payload)

        new_state = cast(
            TrustFunnelSolverState,
            tree_at(
                lambda s: (
                    s.n_iter,
                    s.success,
                    s.status,
                    s.radius_v,
                    s.pi_f_prev,
                    s.sf_flag,
                    s.iteration_type,
                    s.radius_t,
                    s.violation,
                    s.pi_v,
                    s.chi_v,
                    s.pi_f,
                    s.chi_f,
                    s.multiplier_case,
                    s.multiplier_acceptable,
                    s.kkt_satisfied,
                    s.infeasible_stationary,
                    s.normal_computed,
                    s.tangential_computed,
                    s.tangential_reset,
                    s.tangential_rejected,
                    s.in_td,
                    s.in_d,
                    s.dm_f_n,
                    s.dm_f_t,
                    s.dm_v_n,
                    s.dm_v_d,
                    s.objective_decrease_ok,
                    s.contraction_ok,
                    s.normal_norm,
                    s.tangential_norm,
                    s.cauchy_ratio_v,
                    s.cauchy_ratio_f,
                    s.cauchy_ok,
                    s.gate_normal,
                    s.gate_multiplier,
                    s.very_relaxed,
                    s.ftb_normal,
                    s.ftb_tangential,
                    s.n_cg_iter,
                    s.v_trial,
                    s.model_error_f,
                    s.model_error_v,
                    s.funnel_violated,
                    s.demoted,
                ),
                st,
                (
                    st.n_iter + 1,
                    finite,
                    status,
                    radius_v,
                    pi_f,
                    sf_flag,
                    iteration_type,
                    radius_t,
                    v,
                    pi_v,
                    chi_v,
                    pi_f,
                    chi_f,
                    classification.case,
                    classification.acceptable,
                    kkt,
                    infeasible_stationary,
                    normal_computed,
                    compute_tangential,
                    tangential_reset,
                    tangential_rejected,
                    in_td,
                    in_d,
                    dm_f_n,
                    dm_f_t,
                    dm_v_n,
                    dm_v_d,
                    decrease_ok,
                    contraction_ok,
                    normal_norm,
                    tangential_norm,
                    cauchy_ratio_v,
                    cauchy_ratio_f,
                    cauchy_ok,
                    gate_normal,
                    gate_mult,
                    very_relaxed,
                    ftb_normal,
                    ftb_tangential,
                    st.n_cg_iter + n_cg,
                    v,
                    jnp.zeros((), dtype),
                    jnp.zeros((), dtype),
                    jnp.asarray(False),
                    jnp.asarray(False),
                ),
            ),
        )
        return (step_primal, y), new_state

    # ------------------------------------------------------------------
    # normal-equations strategy
    # ------------------------------------------------------------------
    def _attach_normal_equations(
        self, subproblem: FunnelBarrierSubProblem
    ) -> tuple[FunnelBarrierSubProblem, ResolvedNormalEquationsStrategy, Array]:
        """Attach the shared Schur factor of ``Â Âᵀ`` when the explicit path is used.

        Returns the (possibly updated) subproblem, the strategy resolved by
        the tangential solver and the rank of the Schur complement (``-1``
        when no explicit factor is built).
        """
        strategy = self.tangential_solver.resolve_normal_equations(subproblem)
        rec_strategy = self.multiplier_recovery.resolve_normal_equations(subproblem)
        rank = jnp.asarray(-1, jnp.int32)
        if strategy == "schur" or rec_strategy == "schur":
            rcond = (
                self.tangential_solver.rcond
                if strategy == "schur"
                else self.multiplier_recovery.rcond
            )
            subproblem = subproblem.with_schur_normal_equations(rcond=rcond)
            rank = cast(SchurNormalEquations, subproblem.schur_cache).rank
        return subproblem, strategy, rank

    # ------------------------------------------------------------------
    # diagnostics helpers
    # ------------------------------------------------------------------
    def operator_norms(
        self, subproblem: FunnelBarrierSubProblem
    ) -> tuple[Scalar, Scalar]:
        """Power-iteration estimates of ``‖Â‖₂`` and ``‖Ĥ‖₂``.

        Both are lower bounds (see
        :func:`~slsqp_jax.sqpdax.linalg.operator_norm.power_iteration_norm`)
        after :attr:`norm_estimate_iters` iterations from a vector of ones.

        Parameters
        ----------
        subproblem
            Funnel subproblem providing ``jac_mvp`` / ``jac_t_mvp`` and
            ``hess_mvp``.

        Returns
        -------
        a_norm
            Estimate of ``‖Â‖₂``.
        h_norm
            Estimate of ``‖Ĥ‖₂`` (dominant eigenvalue magnitude).
        """
        lag = subproblem.lagrangian
        n, mineq, meq = lag.n, lag.mineq, lag.meq
        ones = jnp.ones_like(lag.ref.flatten())

        def to_primal(w: Array) -> InteriorPointPrimal:
            return InteriorPointPrimal.from_flat(w, n, mineq)

        a_norm = spectral_norm_estimate(
            lambda w: subproblem.jac_mvp(to_primal(w)).flatten(),
            lambda yv: subproblem.jac_t_mvp(
                Dual.from_flat(yv, n, mineq, meq)
            ).flatten(),
            ones,
            n_iter=self.norm_estimate_iters,
        )
        h_norm = power_iteration_norm(
            lambda w: subproblem.hess_mvp(to_primal(w)).flatten(),
            ones,
            n_iter=self.norm_estimate_iters,
        )
        return a_norm, h_norm

    def cauchy_lower_bounds(
        self,
        subproblem: FunnelBarrierSubProblem,
        *,
        a_norm: Scalar,
        h_norm: Scalar,
        radius_v: Scalar,
        radius_t: Scalar,
        pi_f: Scalar,
    ) -> tuple[Scalar, Scalar]:
        """Cauchy decrease lower bounds of Lemmas 3.5 and 3.9.

        ```
        m_v(0) − m_v(n_C) ≥ χᵛ min{πᵛ, δᵛ, 1 − κ_fbn} / (1 + ‖Â‖²)
        m_f(n) − m_f(n + t_C) ≥ κ_ct πᶠ min{πᶠ, (1 − κ_B) δᵗ, (1 − κ_fbt) κ_fbn}
        κ_ct = κ_χ² / (2 (1 + ‖Ĥ‖))
        ```

        The second bound assumes ``χᶠ ≥ κ_χ πᶠ`` (the tangential case of
        (3.15)). With norm *estimates from below* the bounds are
        overestimated, so a ratio slightly under ``1`` is not by itself a
        violation.

        Parameters
        ----------
        subproblem
            Funnel subproblem (for ``πᵛ``, ``χᵛ``, ``κ_fbn``, ``κ_fbt``).
        a_norm, h_norm
            Estimates of ``‖Â‖₂`` and ``‖Ĥ‖₂``.
        radius_v
            Normal radius ``δᵛ`` used for the normal step.
        radius_t
            Tangential radius ``δᵗ`` of (3.38).
        pi_f
            ``πᶠ`` at the returned multipliers.

        Returns
        -------
        bound_v
            Lemma 3.5 bound.
        bound_f
            Lemma 3.9 bound.
        """
        pi_v = subproblem.pi_v()
        chi_v = subproblem.chi_v()
        kappa_fbn = jnp.asarray(subproblem.kappa_fbn)
        kappa_fbt = jnp.asarray(subproblem.kappa_fbt)
        bound_v = (
            chi_v
            * jnp.minimum(pi_v, jnp.minimum(radius_v, 1.0 - kappa_fbn))
            / (1.0 + a_norm**2)
        )
        kappa_ct = self.kappa_chi**2 / (2.0 * (1.0 + h_norm))
        bound_f = (
            kappa_ct
            * pi_f
            * jnp.minimum(
                pi_f,
                jnp.minimum(
                    (1.0 - self.kappa_B) * radius_t, (1.0 - kappa_fbt) * kappa_fbn
                ),
            )
        )
        return bound_v, bound_f

    @staticmethod
    def _unconstrained_cauchy_norm(
        subproblem: FunnelBarrierSubProblem, tiny: Scalar
    ) -> Scalar:
        """``‖P⁻¹ n*‖ = α* πᵛ`` with ``α* = πᵛ² / ‖Â Âᵀ ĉ‖²`` (eq. 3.8)."""
        c_hat = subproblem.dual_grad()
        d = subproblem.jac_t_mvp(c_hat).flatten()
        pi_v_sq = jnp.dot(d, d)
        ad = subproblem.jac_mvp(
            InteriorPointPrimal.from_flat(
                d, subproblem.lagrangian.n, subproblem.lagrangian.mineq
            )
        ).flatten()
        curv = jnp.dot(ad, ad)
        alpha_star = jnp.where(curv > tiny, pi_v_sq / jnp.maximum(curv, tiny), 0.0)
        return alpha_star * jnp.sqrt(pi_v_sq)
