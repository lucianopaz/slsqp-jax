from typing import cast

import jax
from equinox import field, tree_at
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar

from ...dual import Dual
from ...linalg import box_fraction, null_space_projector, steihaug_cg
from ...primal import InteriorPointPrimal
from ..scaled_barrier import ScaledBarrierSubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState
from .gradient_projection import GradientProjection


class SteihaugTointCGTangentialStepSolverState(SubProblemSolverState):
    """Carry for a Steihaug–Toint tangential-step solve.

    Attributes
    ----------
    n_cg_iter
        Cumulative projected-CG iterations.
    on_boundary
        ``True`` when the accepted step saturates the trust-region radius
        (or stopped on negative curvature at the boundary).
    radius
        Trust-region radius used for this solve (in scaled ``w``-space).
    active_bounds
        Optional ``(active_lb, active_ub)`` masks of length ``n``. ``None``
        means the solver identifies them via
        :class:`~slsqp_jax.sqpdax.subproblem.solver.gradient_projection.GradientProjection`.
    """

    n_cg_iter: int
    on_boundary: Bool[Array, ""]
    radius: Scalar
    active_bounds: tuple[Bool[Array, " n"], Bool[Array, " n"]] | None = None


class SteihaugTointCGTangentialStepSolver(
    SubProblemSolver[
        InteriorPointPrimal,
        ScaledBarrierSubProblem,
        SteihaugTointCGTangentialStepSolverState,
    ]
):
    """Projected CG with Steihaug termination on the scaled barrier QP.

    Implements N&W Algorithm 16.2 + eq. 19.33 on a
    :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`:

    * ``kkt_mvp_primal`` supplies the scaled Hessian ``H``;
    * ``primal_grad`` the scaled barrier gradient ``ĝ``;
    * ``kkt_mvp_lower_offdiag`` / ``kkt_mvp_upper_offdiag`` the scaled
      constraint operator ``Â`` and its transpose (matrix-free; the dense
      ``Â`` is never assembled).

    The null-space projector ``P = I - Âᵀ(Â Âᵀ)⁺Â`` is realised with an
    inner matrix-free CG on the normal equations (Tikhonov-regularised by
    ``proj_reg``). CG iterates in ``null(Â)`` so a warm-started normal step
    keeps its linearised feasibility; Steihaug's rules stop at
    ``‖w‖ = radius`` or on negative curvature. The full step is finally
    backtracked once for the fraction-to-boundary rule (eq. 19.33e).
    Multipliers are left at zero for the orchestrator (eq. 19.37).

    Active-bound coordinates (and their bound slacks) supplied in
    ``initial_state.active_bounds`` are pinned at zero step. When the masks
    are ``None``, :class:`GradientProjection` identifies them.

    Attributes
    ----------
    solver_state_class
        :class:`SteihaugTointCGTangentialStepSolverState`.
    max_iter
        Maximum projected-CG iterations.
    tol
        Absolute projected-residual tolerance.
    cg_regularization
        Scale-invariant floor for the curvature check.
    tau
        Fraction-to-boundary parameter (eq. 19.33e).
    proj_cg_max_iter, proj_cg_tol, proj_reg
        Inner normal-equation CG controls.
    gradient_projection
        Bound-face identifier used when ``active_bounds`` is ``None``.
    """

    solver_state_class: type[SteihaugTointCGTangentialStepSolverState] = (
        SteihaugTointCGTangentialStepSolverState
    )

    max_iter: int = 100
    tol: float = 1e-8
    cg_regularization: float = 1e-10
    tau: float = 0.995
    # Inner CG for the matrix-free null-space projection (Ahat Ahatᵀ solve).
    proj_cg_max_iter: int = 100
    proj_cg_tol: float = 1e-10
    proj_reg: float = 0.0
    gradient_projection: GradientProjection = field(default_factory=GradientProjection)

    def solve(
        self,
        subproblem: ScaledBarrierSubProblem,
        x0: tuple[InteriorPointPrimal, Dual],
        initial_state: SteihaugTointCGTangentialStepSolverState,
    ) -> tuple[
        tuple[InteriorPointPrimal, Dual], SteihaugTointCGTangentialStepSolverState
    ]:
        """Compute the Steihaug–Toint tangential step in scaled coordinates.

        Parameters
        ----------
        subproblem
            Scaled barrier QP. Must be a
            :class:`~slsqp_jax.sqpdax.subproblem.scaled_barrier.ScaledBarrierSubProblem`.
        x0
            Warm-start ``(scaled_primal_step, dual)``, typically the normal
            step (or zero).
        initial_state
            Must carry ``radius``; optional ``active_bounds``.

        Returns
        -------
        step
            ``(scaled_primal_step, zero_dual)`` after FTB backtracking.
        state
            Updated carry with boundary / success flags and CG count.

        Raises
        ------
        TypeError
            If ``subproblem`` is not a ``ScaledBarrierSubProblem``.
        """
        radius = initial_state.radius
        if not isinstance(subproblem, ScaledBarrierSubProblem):
            raise TypeError(
                "subproblem must be a ScaledBarrierSubProblem. Got "
                f"{type(subproblem)} instead."
            )
        lag = subproblem.lagrangian
        n, meq, mineq = lag.n, lag.meq, lag.mineq
        w_dim = 3 * n + mineq
        dtype = x0[0].flatten().dtype

        # Active bounds: use the caller-supplied set, else identify it via GP.
        if initial_state.active_bounds is None:
            active_lb, active_ub = self.gradient_projection.find_active_bounds(
                subproblem
            )
        else:
            active_lb, active_ub = initial_state.active_bounds

        # --- flat <-> pytree helpers (scaled ordering [x | s | s_lb | s_ub]) ---
        def to_primal(w: Float[Array, " w"]) -> InteriorPointPrimal:
            return InteriorPointPrimal.from_flat(w, n, mineq)

        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, lag.dual))

        # --- matrix-free scaled operators from the SubProblem interface ---
        # ``H`` is the scaled Lagrangian-Hessian block (N&W eq. 19.33a) and
        # ``ghat`` the scaled barrier objective gradient; the dual tangent is zero
        # because the tangential step only exercises the primal (H) block.
        def H(w: Float[Array, " w"]) -> Float[Array, " w"]:
            return subproblem.kkt_mvp_primal((to_primal(w), zero_dual)).flatten()

        ghat = subproblem.primal_grad().flatten()

        # --- matrix-free scaled constraint operator Ahat and its transpose ---
        # ``A(v) = Ahat v`` via ``kkt_mvp_lower_offdiag`` and ``At(y) = Ahatᵀ y``
        # via ``kkt_mvp_upper_offdiag``.  These are provably equal to the dense
        # assembly (rows [eq|ineq|lb|ub], columns [x|s|s_lb|s_ub], slack columns
        # scaled by S) but never form the O(n^2) matrix.
        def to_dual(y: Float[Array, " m_total"]) -> Dual:
            return Dual.from_flat(y, n, mineq, meq)

        zero_primal = to_primal(jnp.zeros((w_dim,), dtype))

        def A(v: Float[Array, " w"]) -> Float[Array, " m_total"]:
            return subproblem.kkt_mvp_lower_offdiag((to_primal(v), zero_dual)).flatten()

        def At(y: Float[Array, " m_total"]) -> Float[Array, " w"]:
            return subproblem.kkt_mvp_upper_offdiag((zero_primal, to_dual(y))).flatten()

        # --- free-column mask over the scaled primal [x | s | s_lb | s_ub] ---
        # Freeze active-bound variables AND their bound slacks (reduced active set):
        # zeroing both columns collapses the active bound row to 0 = 0 so it drops
        # out of the projection.  Null bound-slacks are pinned to zero as well.
        free_x = ~(active_lb | active_ub)
        free_lb = (~active_lb) & (~lag.null_lb)
        free_ub = (~active_ub) & (~lag.null_ub)
        free_w = jnp.concatenate(
            [
                free_x.astype(dtype),
                jnp.ones((mineq,), dtype),
                free_lb.astype(dtype),
                free_ub.astype(dtype),
            ]
        )
        # Real-slack mask for the fraction-to-boundary backtrack (null bounds out);
        # slacks constrained by ``w_s >= -tau e`` (eq. 19.33e).
        col_mask = jnp.concatenate(
            [
                jnp.ones((n + mineq,), dtype),
                jnp.where(lag.null_lb, 0.0, 1.0),
                jnp.where(lag.null_ub, 0.0, 1.0),
            ]
        )

        # --- matrix-free projector P = I - A_freeᵀ (A_free A_freeᵀ)⁺ A_free ------
        # ``A_free(p) = A(free_w ⊙ p)``.  The (Tikhonov-regularised) free-restricted
        # normal equations are solved by an inner matrix-free CG; no dense Ahat /
        # SVD is ever formed.
        proj = null_space_projector(
            A,
            At,
            free_mask=free_w,
            reg=self.proj_reg,
            tol=self.proj_cg_tol,
            max_iter=self.proj_cg_max_iter,
        )

        tol_sq = jnp.asarray(self.tol, dtype) ** 2

        # --- Steihaug projected CG, warm-started from the normal step x0 ---
        # Projected steepest-descent residual g = -P(H w + ghat) (N&W eq. 16.28c-d),
        # recomputed from scratch each step for numerical robustness.  ``w0`` is
        # free-masked so the frozen (active-bound) coordinates stay at zero step.
        # Textbook Steihaug-Toint termination: stop only on a trust-region
        # boundary crossing / negative curvature or on convergence.  The
        # projected-residual 2-norm is naturally non-monotonic on ill-conditioned
        # reduced systems, so it must NOT be used as a no-progress guard --
        # doing so aborts CG on the first descent step and reverts the interior
        # Newton step to zero.  ``max_iter`` bounds the loop.
        w0 = free_w * x0[0].flatten()
        cg = steihaug_cg(
            H,
            proj(-(H(w0) + ghat)),
            w0,
            radius,
            tol_sq=tol_sq,
            max_iter=self.max_iter,
            residual=lambda w: proj(-(H(w) + ghat)),
            curvature_floor=self.cg_regularization,
        )
        w, n_cg, on_bnd = cg.w, cg.n_iter, cg.on_boundary

        # Backtrack the whole (normal + tangential) step for the fraction-to-
        # boundary rule (eq. 19.33e), then repackage as a ``(Primal, Dual)`` step.
        # Multipliers are left at zero: the trust-region orchestrator recovers them
        # by least squares (N&W eq. 19.37), as in ``DogLegSolver``.
        w = w * self._fraction_to_boundary_beta(w, col_mask, n, self.tau)
        step = (
            to_primal(w),
            cast(Dual, jax.tree.map(jnp.zeros_like, x0[1])),
        )

        finite = jnp.all(jnp.isfinite(w))
        status = RESULTS.where(finite, RESULTS.successful, RESULTS.singular)
        new_state = cast(
            SteihaugTointCGTangentialStepSolverState,
            tree_at(
                lambda state: (
                    state.n_iter,
                    state.n_cg_iter,
                    state.on_boundary,
                    state.success,
                    state.status,
                    state.radius,
                    state.active_bounds,
                ),
                initial_state,
                (
                    initial_state.n_iter + 1,
                    initial_state.n_cg_iter + n_cg,
                    on_bnd,
                    finite,
                    status,
                    radius,
                    (active_lb, active_ub),
                ),
            ),
        )
        return step, new_state

    @staticmethod
    def _fraction_to_boundary_beta(
        w: Float[Array, " w"], slack_mask: Float[Array, " w"], n: int, tau: float
    ) -> Scalar:
        """Largest ``β ∈ (0, 1]`` satisfying the fraction-to-boundary rule.

        Enforces ``w_s ≥ -τ e`` on the scaled slack block (N&W eq. 19.33e /
        19.34c). Only coordinates with ``slack_mask > 0`` (real, non-null
        slacks) participate.

        Parameters
        ----------
        w
            Scaled step ``(p_x, p_s̃)`` of length ``3 n + m_ineq``.
        slack_mask
            Mask of the same length; zeros on decision variables and null
            bound-slacks.
        n
            Number of decision variables (slack block starts at index ``n``).
        tau
            Fraction-to-boundary parameter in ``(0, 1]``.

        Returns
        -------
        Scalar
            Backtracking factor ``β``.
        """
        inf = jnp.asarray(jnp.inf, w.dtype)
        lo = (
            jnp.where(slack_mask > 0, -jnp.asarray(tau, w.dtype), -inf).at[:n].set(-inf)
        )
        return box_fraction(w, lo)
