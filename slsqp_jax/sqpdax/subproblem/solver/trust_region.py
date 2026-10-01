"""Composite-step trust-region interior-point subproblem solver (N&W §19.5)."""

from typing import Generic, cast

import jax
from equinox import field, tree_at
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar
from typing_extensions import TypeVar

from ...dual import Dual
from ...primal import InteriorPointPrimal, Slack
from ..scaled_barrier import ScaledBarrierSubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState
from .dogleg import DogLegSolver, DogLegSolverState
from .gradient_projection import GradientProjection
from .multiplier_recovery import (
    BarrierSafeguard,
    LeastSquaresMultiplierRecovery,
    MultiplierRecovery,
)
from .steihaug_toint_cg import (
    SteihaugTointCGTangentialStepSolver,
    SteihaugTointCGTangentialStepSolverState,
)

__all__ = [
    "TrustRegionSolverState",
    "TrustRegionStateType",
    "TrustRegionInteriorPointSolver",
]


def _all_finite(tree: object) -> Bool[Array, ""]:
    """``True`` when every leaf of ``tree`` is finite."""
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)])
    )


class TrustRegionSolverState(SubProblemSolverState):
    """Carry threaded across trust-region composite-step solves.

    ``radius`` is the incoming trust-region radius (the outer loop updates it
    from the actual / predicted-reduction ratio, N&W eq. 19.39).
    ``predicted_reduction`` (eq. 19.41) and ``merit_penalty`` (``ν``, eq. 19.42)
    are the quantities the outer loop needs to form ``pred`` / choose ``ν`` and
    to test step acceptance.

    Attributes
    ----------
    radius
        Trust-region radius used for this solve (scaled ``w``-space).
    predicted_reduction
        Model predicted reduction ``pred = -q(w) + ν (m(0) - m(w))``.
    merit_penalty
        Updated merit penalty ``ν`` (eq. 19.42).
    n_cg_iter
        Cumulative tangential projected-CG iterations.
    on_boundary
        ``True`` when the tangential step saturates the trust-region radius.
    rho
        Actual / predicted reduction ratio of the last controlled step, as
        written back by
        :class:`~slsqp_jax.sqpdax.step_controller.trust_region_radius.TrustRegionManager`;
        ``-inf`` when the model predicted no decrease. Seeded at ``1`` before
        the first step and untouched by the subproblem solver itself.
    """

    radius: Scalar
    predicted_reduction: Scalar
    merit_penalty: Scalar
    n_cg_iter: int
    on_boundary: Bool[Array, ""]
    rho: Scalar = field(default_factory=lambda: jnp.asarray(1.0))


# Solver carry; defaulted so bare ``TrustRegionInteriorPointSolver`` keeps
# meaning ``TrustRegionInteriorPointSolver[TrustRegionSolverState]`` while
# subclasses can bind a richer carry and specialise ``solve`` without casts.
TrustRegionStateType = TypeVar(
    "TrustRegionStateType",
    bound=TrustRegionSolverState,
    default=TrustRegionSolverState,
)


class TrustRegionInteriorPointSolver(
    SubProblemSolver[
        InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionStateType
    ],
    Generic[TrustRegionStateType],
):
    """Composite-step trust-region interior-point solver (N&W Algorithms 19.3 / 19.4).

    Orchestrates a normal (feasibility) step and a tangential (optimality)
    step on a
    :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`,
    recovers the multipliers through the pluggable :attr:`multiplier_recovery`
    (by default least squares, eq. 19.37, with the positivity safeguard,
    eq. 19.38), and reports the predicted reduction (eq. 19.41) and merit
    penalty ``ν`` (eq. 19.42). The radius update and the
    actual / predicted step-acceptance test (eq. 19.39) belong to the outer
    loop.

    Active variable-bound faces are identified once via
    :class:`~slsqp_jax.sqpdax.subproblem.solver.gradient_projection.GradientProjection`
    (N&W ``A(x^c)``) and threaded into both sub-solvers so they freeze the
    same set.

    Attributes
    ----------
    solver_state_class
        :class:`TrustRegionSolverState`.
    zeta
        Normal-step radius fraction (eq. 19.34b); the dogleg runs at
        ``zeta * radius``.
    tau
        Fraction-to-boundary parameter forwarded to the tangential solver.
    penalty_rho
        ``ρ`` in the merit-penalty update (eq. 19.42).
    penalty_margin
        Additive margin when raising ``ν``.
    multiplier_recovery
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.MultiplierRecovery`
        producing the dual block (default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.LeastSquaresMultiplierRecovery`
        with a
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.BarrierSafeguard`,
        run matrix-free on ``Âᵀ`` since the primal carries slacks). Its
        ``rtol`` / ``atol`` / ``max_steps`` are the LSMR tolerances.
    normal_solver
        Feasibility-step solver (default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.dogleg.DogLegSolver`).
    tangential_solver
        Optimality-step solver (default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.steihaug_toint_cg.SteihaugTointCGTangentialStepSolver`).
    gradient_projection
        Bound-face identifier run once per outer solve.
    """

    solver_state_class: type[TrustRegionSolverState] = TrustRegionSolverState

    zeta: float = 0.8  # normal-step radius fraction (eq. 19.34b)
    tau: float = 0.995  # fraction-to-boundary parameter (eq. 19.31e)
    penalty_rho: float = 0.3  # rho in eq. 19.42
    penalty_margin: float = 1e-4
    # Multiplier recovery (eq. 19.37 + 19.38 by default); matrix-free on Ahat^T.
    multiplier_recovery: MultiplierRecovery = field(
        default_factory=lambda: LeastSquaresMultiplierRecovery(
            safeguard=BarrierSafeguard()
        )
    )
    normal_solver: SubProblemSolver = field(default_factory=DogLegSolver)
    tangential_solver: SubProblemSolver = field(
        default_factory=SteihaugTointCGTangentialStepSolver
    )
    gradient_projection: GradientProjection = field(default_factory=GradientProjection)

    def solve(
        self,
        subproblem: ScaledBarrierSubProblem,
        x0: tuple[InteriorPointPrimal, Dual],
        initial_state: TrustRegionStateType,
    ) -> tuple[tuple[InteriorPointPrimal, Dual], TrustRegionStateType]:
        """Compute the composite trust-region step at ``initial_state.radius``.

        Parameters
        ----------
        subproblem
            Scaled barrier QP. Must be a
            :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`.
        x0
            Warm-start ``(InteriorPointPrimal, Dual)`` passed to the normal
            solver (typically a zero step).
        initial_state
            Must carry ``radius`` and ``merit_penalty``; ``n_cg_iter`` is
            accumulated from the tangential solve.

        Returns
        -------
        step
            Native-scale ``(p_x, p_s)`` primal step and recovered multipliers.
        state
            ``initial_state`` with ``predicted_reduction``, ``merit_penalty``,
            boundary / success flags, and CG count refreshed (same type as
            the input).

        Raises
        ------
        TypeError
            If ``subproblem`` is not a ``ScaledBarrierSubProblem``.
        ValueError
            If ``subproblem.is_kkt_dual_regularized`` is true. The
            composite-step (Byrd-Omojokun) decomposition solves the
            unregularised system ``Â p = -ĉ`` and never applies the dual-dual
            block, so a regularised subproblem would report a ``residual``
            that disagrees with the computed step.
        """
        if not isinstance(subproblem, ScaledBarrierSubProblem):
            raise TypeError(
                "subproblem must be a ScaledBarrierSubProblem. Got "
                f"{type(subproblem)} instead."
            )
        if subproblem.is_kkt_dual_regularized:
            raise ValueError(
                "TrustRegionInteriorPointSolver solves the unregularised KKT system "
                "and ignores the dual-dual block; set dual_kkt_regularization=0 on "
                "the InteriorPointLagrangian (the block is consumed only by "
                "full-space KKT solvers)."
            )
        lag = subproblem.lagrangian
        n, mineq = lag.n, lag.mineq
        radius = initial_state.radius
        dtype = x0[0].flatten().dtype
        zero_i = jnp.zeros((), jnp.int32)
        false_ = jnp.asarray(False)

        # --- scaled operators from the SubProblem interface (same construction as
        # the tangential solver): ghat / chat are the scaled objective gradient and
        # constraint residual, apply_H the scaled Hessian block, and A / At the
        # scaled constraint Jacobian Ahat.  Everything is applied matrix-free: the
        # dense Ahat (O(n^2) for the barrier system) is never assembled -- neither
        # for the multipliers nor the predicted reduction. ---
        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, lag.dual))
        ghat = subproblem.primal_grad().flatten()
        chat = subproblem.dual_grad().flatten()

        def apply_H(w: Float[Array, " w"]) -> Float[Array, " w"]:
            step = (InteriorPointPrimal.from_flat(w, n, mineq), zero_dual)
            return subproblem.kkt_mvp_primal(step).flatten()

        def A(v: Float[Array, " w"]) -> Float[Array, " m_total"]:  # Ahat v
            return subproblem.kkt_mvp_lower_offdiag(
                (InteriorPointPrimal.from_flat(v, n, mineq), zero_dual)
            ).flatten()

        # --- composite step: normal (feasibility) then tangential (optimality) ---
        # The normal solver runs at the reduced radius zeta * radius (eq. 19.34b).
        # DogLegSolver returns an x-space feasibility step for the general (eq +
        # ineq) constraints; it is lifted into the scaled (x, s_tilde) space with
        # zero slack components to warm-start the tangential CG.
        #
        # The active variable-bound faces are identified once here (N&W A(x^c)) and
        # threaded into both sub-solvers so they freeze the same set and GP is not
        # re-run per sub-solve.
        active_bounds = self.gradient_projection.find_active_bounds(subproblem)
        normal_state = cast(
            DogLegSolverState,
            DogLegSolverState(
                n_iter=zero_i,
                n_cg_iter=zero_i,
                on_boundary=false_,
                success=false_,
                status=RESULTS.successful,
                radius=self.zeta * radius,
                active_bounds=active_bounds,
            ),
        )
        (normal_primal, _), normal_new_state = self.normal_solver.solve(
            subproblem, x0, normal_state
        )
        normal_finite = _all_finite(normal_primal)
        self.logger.diagnostic(
            "normal_step_failure",
            lambda: {
                "radius": radius,
                "normal_radius": self.zeta * radius,
                "normal_step": normal_primal,
                "normal_state": normal_new_state,
                "finite": normal_finite,
                "active_bounds": active_bounds,
            },
            when=~normal_new_state.success | ~normal_finite,
        )
        w_normal_primal = cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=normal_primal.x,
                slack=Slack(
                    s=jnp.zeros((mineq,), dtype),
                    s_lb=jnp.zeros((n,), dtype),
                    s_ub=jnp.zeros((n,), dtype),
                ),
            ),
        )
        tang_state = cast(
            SteihaugTointCGTangentialStepSolverState,
            SteihaugTointCGTangentialStepSolverState(
                n_iter=zero_i,
                n_cg_iter=zero_i,
                on_boundary=false_,
                success=false_,
                status=RESULTS.successful,
                radius=radius,
                active_bounds=active_bounds,
            ),
        )
        (tang_primal, _), tang_new_state = self.tangential_solver.solve(
            subproblem, (w_normal_primal, zero_dual), tang_state
        )
        w = tang_primal.flatten()  # full scaled step
        n_cg = tang_new_state.n_cg_iter
        on_bnd = tang_new_state.on_boundary
        tang_finite = jnp.all(jnp.isfinite(w))
        self.logger.diagnostic(
            "tangential_step_failure",
            lambda: {
                "radius": radius,
                "normal_step": w_normal_primal,
                "tangential_step": tang_primal,
                "tangential_state": tang_new_state,
                "finite": tang_finite,
                "active_bounds": active_bounds,
            },
            when=~tang_new_state.success | ~tang_finite,
        )

        # --- recover the native primal step p = (p_x, p_s = S p_s_tilde) ---
        step_primal = cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=tang_primal.x,
                slack=subproblem._slack_to_orig_scale(tang_primal.slack),
            ),
        )

        # --- multipliers (eq. 19.37 least squares + eq. 19.38 safeguard by
        # default).  The scaled-barrier primal carries slacks, so the recovery runs
        # its matrix-free LSMR path on ``Âᵀ`` (``projector=None``); the LS target
        # ``-ĝ`` is step-independent, hence the primal step is only a placeholder.
        step_dual = self.multiplier_recovery.recover(subproblem, None, step_primal)

        # --- predicted reduction (eq. 19.41) and penalty update (eq. 19.42) ---
        Hw = apply_H(w)
        obj_model = jnp.dot(ghat, w) + 0.5 * jnp.dot(w, Hw)
        m0 = jnp.linalg.norm(chat)  # m(0), eq. 19.41
        mp = jnp.linalg.norm(chat + A(w))  # m(p), matrix-free
        v_pred = m0 - mp
        nu_old = initial_state.merit_penalty
        # nu >= obj_model / ((1 - rho)(m0 - m(p))) when the step reduces infeasibility.
        need = jnp.where(
            v_pred > 1e-30,
            obj_model / ((1.0 - self.penalty_rho) * jnp.maximum(v_pred, 1e-30)),
            0.0,
        )
        nu_new = jnp.where(
            (v_pred > 1e-30) & (obj_model > 0),
            jnp.maximum(nu_old, need + self.penalty_margin),
            nu_old,
        )
        pred = -obj_model + nu_new * v_pred

        step = (step_primal, step_dual)
        finite = _all_finite(step)
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)
        self.logger.diagnostic(
            "recovery_failure",
            lambda: {
                "radius": radius,
                "step_primal": step_primal,
                "step_dual": step_dual,
                "primal_finite": _all_finite(step_primal),
                "dual_finite": _all_finite(step_dual),
            },
            when=~finite,
        )
        self.logger.diagnostic(
            "tr_step",
            lambda: {
                "radius": radius,
                "cg_iters": n_cg,
                "on_boundary": on_bnd,
                "obj_model": obj_model,
                "v_pred": v_pred,
                "pred": pred,
                "nu": nu_new,
                "step_primal": step_primal,
                "step_dual": step_dual,
                "normal_step": normal_primal,
                "status": status,
                "success": finite,
                "active_bounds": active_bounds,
            },
        )
        self.logger.debug(
            "trust-region step: radius={radius:.3e} cg_iters={cg_iters} "
            "on_boundary={on_boundary} pred={pred:.3e} nu={nu:.3e} "
            "status={status}",
            radius=radius,
            cg_iters=n_cg,
            on_boundary=on_bnd,
            pred=pred,
            nu=nu_new,
            status=status,
        )
        self.logger.warning(
            "trust-region step is non-finite (status={status})",
            when=~finite,
            status=status,
        )
        new_state = cast(
            TrustRegionStateType,
            tree_at(
                lambda state: (
                    state.n_iter,
                    state.success,
                    state.status,
                    state.radius,
                    state.predicted_reduction,
                    state.merit_penalty,
                    state.n_cg_iter,
                    state.on_boundary,
                ),
                initial_state,
                (
                    initial_state.n_iter + 1,
                    finite,
                    status,
                    radius,
                    pred,
                    nu_new,
                    initial_state.n_cg_iter + n_cg,
                    on_bnd,
                ),
            ),
        )
        return step, new_state
