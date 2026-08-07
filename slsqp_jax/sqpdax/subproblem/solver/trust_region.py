"""Composite-step trust-region interior-point subproblem solver (N&W §19.5)."""

from typing import cast

import jax
import lineax as lx
from equinox import field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar
from lineax import AbstractLinearOperator, AbstractLinearSolver

from ...dual import Dual
from ...primal import InteriorPointPrimal, Slack
from ..scaled_barrier import ScaledBarrierSubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState
from .dogleg import DogLegSolver, DogLegSolverState
from .gradient_projection import GradientProjection
from .steihaug_toint_cg import (
    SteihaugTointCGTangentialStepSolver,
    SteihaugTointCGTangentialStepSolverState,
)

__all__ = [
    "TrustRegionSolverState",
    "TrustRegionInteriorPointSolver",
]


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
    """

    radius: Scalar
    predicted_reduction: Scalar
    merit_penalty: Scalar
    n_cg_iter: int
    on_boundary: Bool[Array, ""]


class TrustRegionInteriorPointSolver(
    SubProblemSolver[
        InteriorPointPrimal, ScaledBarrierSubProblem, TrustRegionSolverState
    ]
):
    """Composite-step trust-region interior-point solver (N&W Algorithms 19.3 / 19.4).

    Orchestrates a normal (feasibility) step and a tangential (optimality)
    step on a
    :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`,
    recovers least-squares multipliers (eq. 19.37) with the positivity
    safeguard (eq. 19.38), and reports the predicted reduction (eq. 19.41) and
    merit penalty ``ν`` (eq. 19.42). The radius update and the
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
    mult_rtol, mult_atol, mult_max_steps
        Lineax LSMR tolerances for matrix-free multiplier recovery on
        ``Âᵀ``.
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
    # Matrix-free least-squares multiplier recovery (lineax LSMR on Ahat^T).
    mult_rtol: float = 1e-8
    mult_atol: float = 1e-8
    mult_max_steps: int | None = None
    normal_solver: SubProblemSolver = field(default_factory=DogLegSolver)
    tangential_solver: SubProblemSolver = field(
        default_factory=SteihaugTointCGTangentialStepSolver
    )
    gradient_projection: GradientProjection = field(default_factory=GradientProjection)

    def solve(
        self,
        subproblem: ScaledBarrierSubProblem,
        x0: tuple[InteriorPointPrimal, Dual],
        initial_state: TrustRegionSolverState,
    ) -> tuple[tuple[InteriorPointPrimal, Dual], TrustRegionSolverState]:
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
            Updated :class:`TrustRegionSolverState` with ``predicted_reduction``,
            ``merit_penalty``, boundary / success flags, and CG count.

        Raises
        ------
        TypeError
            If ``subproblem`` is not a ``ScaledBarrierSubProblem``.
        """
        if not isinstance(subproblem, ScaledBarrierSubProblem):
            raise TypeError(
                "subproblem must be a ScaledBarrierSubProblem. Got "
                f"{type(subproblem)} instead."
            )
        lag = subproblem.lagrangian
        n, meq, mineq = lag.n, lag.meq, lag.mineq
        m_total = meq + mineq + 2 * n
        mu = lag.barrier.weight
        null_lb, null_ub = lag.null_lb, lag.null_ub
        radius = initial_state.radius
        dtype = x0[0].flatten().dtype
        zero_i = jnp.zeros((), jnp.int32)
        false_ = jnp.asarray(False)

        # --- scaled operators from the SubProblem interface (same construction as
        # the tangential solver): ghat / chat are the scaled objective gradient and
        # constraint residual, apply_H the scaled Hessian block, and A / At the
        # scaled constraint Jacobian Ahat and its transpose.  Everything is applied
        # matrix-free: the dense Ahat (O(n^2) for the barrier system) is never
        # assembled -- neither for the multipliers nor the predicted reduction. ---
        w_dim = 3 * n + mineq
        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, lag.dual))
        zero_primal = InteriorPointPrimal.from_flat(
            jnp.zeros((w_dim,), dtype), n, mineq
        )
        ghat = subproblem.primal_grad().flatten()
        chat = subproblem.dual_grad().flatten()
        s, s_lb, s_ub = lag.slack.s, lag.slack.s_lb, lag.slack.s_ub

        def apply_H(w: Float[Array, " w"]) -> Float[Array, " w"]:
            step = (InteriorPointPrimal.from_flat(w, n, mineq), zero_dual)
            return subproblem.kkt_mvp_primal(step).flatten()

        def A(v: Float[Array, " w"]) -> Float[Array, " m_total"]:  # Ahat v
            return subproblem.kkt_mvp_lower_offdiag(
                (InteriorPointPrimal.from_flat(v, n, mineq), zero_dual)
            ).flatten()

        def At(lam: Float[Array, " m_total"]) -> Float[Array, " w"]:  # Ahat^T lam
            return subproblem.kkt_mvp_upper_offdiag(
                (zero_primal, Dual.from_flat(lam, n, mineq, meq))
            ).flatten()

        # --- least-squares multipliers via LSMR on the rectangular Ahat^T --------
        # ``min_lam ||Ahat^T lam + ghat||`` <=> solve ``Ahat^T lam = -ghat`` in the
        # least-squares sense.  lineax's LSMR works matrix-free on the ``At``
        # operator and returns the pseudoinverse solution for rank-deficient Ahat
        # (e.g. null-bound zero rows).
        At_op = cast(
            AbstractLinearOperator,
            lx.FunctionLinearOperator(At, jax.ShapeDtypeStruct((m_total,), dtype)),
        )
        lsmr = cast(
            AbstractLinearSolver,
            lx.LSMR(
                rtol=self.mult_rtol,
                atol=self.mult_atol,
                max_steps=self.mult_max_steps,
            ),
        )

        def least_squares_multipliers() -> Float[Array, " m_total"]:
            sol = lx.linear_solve(At_op, -ghat, lsmr, throw=False)
            return sol.value

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
        (normal_primal, _), _ = self.normal_solver.solve(subproblem, x0, normal_state)
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

        # --- recover the native primal step p = (p_x, p_s = S p_s_tilde) ---
        step_primal = cast(
            InteriorPointPrimal,
            InteriorPointPrimal(
                x=tang_primal.x,
                slack=subproblem._slack_to_orig_scale(tang_primal.slack),
            ),
        )

        # --- least-squares multipliers (eq. 19.37). Positivity safeguard (eq. 19.38):
        # z_i <- min(1e-3, mu / s_i) if z_i <= 0. Null bounds get z = 0.
        lam = least_squares_multipliers()
        y = lam[:meq]
        z_ineq = lam[meq : meq + mineq]
        z_lb = lam[meq + mineq : meq + mineq + n]
        z_ub = lam[meq + mineq + n :]
        z_ineq = jnp.where(z_ineq > 0, z_ineq, jnp.minimum(1e-3, mu / s))
        z_lb = jnp.where(
            null_lb,
            0.0,
            jnp.where(
                z_lb > 0,
                z_lb,
                jnp.minimum(1e-3, mu / jnp.where(null_lb, 1.0, s_lb)),
            ),
        )
        z_ub = jnp.where(
            null_ub,
            0.0,
            jnp.where(
                z_ub > 0,
                z_ub,
                jnp.minimum(1e-3, mu / jnp.where(null_ub, 1.0, s_ub)),
            ),
        )
        step_dual = cast(
            Dual,
            Dual(
                eq_multipliers=y,
                ineq_multipliers=z_ineq,
                lb_multipliers=z_lb,
                ub_multipliers=z_ub,
            ),
        )

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
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(step)])
        )
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)
        new_state = cast(
            TrustRegionSolverState,
            TrustRegionSolverState(
                n_iter=jnp.asarray(initial_state.n_iter, jnp.int32) + 1,
                success=finite,
                status=status,
                radius=radius,
                predicted_reduction=pred,
                merit_penalty=nu_new,
                n_cg_iter=jnp.asarray(initial_state.n_cg_iter, jnp.int32) + n_cg,
                on_boundary=on_bnd,
            ),
        )
        return step, new_state
