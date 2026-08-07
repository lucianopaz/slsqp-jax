from typing import cast

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from ...dual import Dual
from ...preconditioner import IdentityPreconditioner, Preconditioner
from ...primal import Primal
from ...types import Vector_n
from ..active_set import ActiveSetSubProblem
from .base import RESULTS, SubProblemSolver, SubProblemSolverState


class ProjectedCGState(SubProblemSolverState):
    """Carry for a single projected-CG KKT solve.

    Inherits ``n_iter`` / ``success`` / ``status`` from
    :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolverState`.
    ``n_iter`` accumulates CG iterations across calls so an outer active-set
    loop can report total inner work.
    """


class ProjectedCGSubProblemSolver(
    SubProblemSolver[Primal, ActiveSetSubProblem, ProjectedCGState]
):
    """Null-space projected conjugate-gradient solver for the active-set KKT system.

    Solves ``kkt_mvp(step) = sol`` (equivalently ``K @ step = -∇L``), where ``K``
    is the KKT system:

    ```
    [ H_k   Aᵀ ] [ x ]   [ -(∇_x L)_k ]
    [ A      0 ] [ λ ] = [ -c_k       ]
    ```

    The solver only consumes the matrix-free ``kkt_mvp_*`` blocks plus the
    ``nonbound_constraint_jac`` matrix and the bound masks; it never touches
    the underlying Hessian (e.g. L-BFGS) directly.

    Algorithm details follow Nocedal & Wright (2006) algorithm 16.2 with these
    implementation notes:

    * **Active bounds fix variables.** An active lower (resp. upper) bound
      forces ``dx_i = lb_i - x_i`` (resp. ``ub_i - x_i``). Targets are read
      from the bound rows of ``sol``. Fixed variables drop out; the rest are
      *free*. Lower bounds win ties (``lb == ub``).
    * **Projector.** Null / range bases are never formed. An SVD of the
      active general Jacobian yields the Moore–Penrose solve of
      ``A Aᵀ x = b``, so ``P(v) = v - Aᵀ (A Aᵀ)⁺ A v``. Singular values
      below ``rcond · max(s)`` are dropped (rank-revealing; handles LICQ
      violations and zero rows from inactive inequalities).
    * **Preconditioning (optional).** ``preconditioner`` supplies ``M⁻¹``
      (SPD reduced-Hessian approximation, N&W eq. 16.26), upgrading the
      projector to the constraint preconditioner (eq. 16.33). ``None`` is
      the identity.
    * **CG** runs in that null space against the Lagrangian Hessian via
      ``kkt_mvp_primal``, warm-started from ``x0``.
    * **Multipliers** are recovered from the stationarity residual (general
      multipliers by a normal-equation solve with one refinement round;
      bound multipliers from the residual on fixed variables).

    Attributes
    ----------
    solver_state_class
        :class:`ProjectedCGState`.
    max_iter
        Maximum CG iterations.
    tol
        Absolute projected-residual tolerance.
    cg_regularization
        Scale-invariant floor for the curvature check ``pᵀ H p``.
    rcond
        Relative singular-value floor for the pseudoinverse. ``None`` uses
        ``eps * max(A_work.shape)`` (numpy ``pinv`` convention).
    preconditioner
        Optional SPD reduced-Hessian preconditioner ``M``.
    """

    solver_state_class: type[ProjectedCGState] = ProjectedCGState

    max_iter: int = 100
    tol: float = 1e-10
    cg_regularization: float = 1e-6
    # Relative singular-value floor for the pseudoinverse rank cut. ``None``
    # falls back to ``eps * max(A_work.shape)`` (numpy ``pinv`` convention).
    rcond: float | None = None
    # Preconditioner ``M`` for the reduced Hessian (N&W eq. 16.26).  ``invert``
    # supplies ``M⁻¹``.  ``None`` is the identity (unpreconditioned Algorithm
    # 16.2, ``H = I``) and recovers the plain projector below exactly.  A supplied
    # preconditioner turns the projector into the constraint preconditioner
    # (eq. 16.33).  Must be SPD for CG.
    preconditioner: Preconditioner | None = None

    def solve(
        self,
        subproblem: ActiveSetSubProblem,
        x0: tuple[Primal, Dual],
        initial_state: ProjectedCGState,
    ) -> tuple[tuple[Primal, Dual], ProjectedCGState]:
        """Solve the active-set KKT system by projected CG.

        Parameters
        ----------
        subproblem
            Working-set QP. Must be an
            :class:`~slsqp_jax.sqpdax.subproblem.active_set.ActiveSetSubProblem`.
        x0
            Warm-start ``(primal_step, dual)``. Only the free components of
            the primal step are used.
        initial_state
            Carry whose ``n_iter`` is accumulated into the returned state.

        Returns
        -------
        step
            Primal-dual KKT solution ``(dx, λ)``.
        state
            Updated :class:`ProjectedCGState` (success / status / CG count).

        Raises
        ------
        TypeError
            If ``subproblem`` is not an ``ActiveSetSubProblem``.
        """
        if not isinstance(subproblem, ActiveSetSubProblem):
            raise TypeError(
                "subproblem must be an ActiveSetSubProblem. Got "
                f"{type(subproblem)} instead."
            )
        # ``A`` is the masked Jacobian exposed alongside the masked operator, so
        # inactive inequality rows are already zeroed.
        # N&W Ch. 16 ``A``: the constraint matrix of the QP
        # ``min ½ dᵀG d + cᵀd  s.t.  A d = b`` (eq. 16.3).
        A = subproblem.L_k.nonbound_constraint_jac
        n = subproblem.lagrangian.n
        meq = subproblem.lagrangian.meq
        mineq = subproblem.lagrangian.mineq
        m_gen = meq + mineq
        dtype = A.dtype

        # --- unpack the right-hand side (sol = -∇L) ---
        _sol_primal, sol_dual = subproblem.kkt_rhs()
        sol_primal = _sol_primal.x  # -∇_x L; note ``g0 = -sol_primal`` is N&W ``c``
        b_gen = jnp.concatenate(
            [sol_dual.eq_multipliers, sol_dual.ineq_multipliers]
        )  # -c_general (eq block then ineq block); N&W ``b`` in ``A d = b``
        sol_lb = sol_dual.lb_multipliers  # x - lb on active lb rows, else 0
        sol_ub = sol_dual.ub_multipliers  # ub - x on active ub rows, else 0

        # --- active bounds fix variables (lower bound wins ties) ---
        active_set = subproblem.active_set
        active_lb = active_set.active_lb
        active_ub = active_set.active_ub
        fix_lb = active_lb
        fix_ub = active_ub & (~active_lb)
        free_f = (~(fix_lb | fix_ub)).astype(dtype)
        d_fixed = jnp.where(
            fix_lb, -sol_lb, jnp.where(fix_ub, sol_ub, jnp.zeros((n,), dtype))
        )

        # Equality rows are always active; inequality rows follow the active set.
        active_gen = active_set.active_gen

        # --- Lagrangian Hessian operator (matrix-free, HVP only) ---
        zero_dual = cast(
            Dual,
            Dual(
                eq_multipliers=jnp.zeros((meq,), dtype),
                ineq_multipliers=jnp.zeros((mineq,), dtype),
                lb_multipliers=jnp.zeros((n,), dtype),
                ub_multipliers=jnp.zeros((n,), dtype),
            ),
        )

        def H(v: Vector_n) -> Vector_n:
            # N&W ``G`` (QP / Lagrangian Hessian) applied matrix-free.
            # ``kkt_mvp_primal`` takes a single ``(Primal, Dual)`` tangent (and
            # ignores the dual block); the tuple must be passed as one argument,
            # not splatted.
            return subproblem.kkt_mvp_primal(
                (cast(Primal, Primal(v)), zero_dual)
            ).flatten()

        def hvp_work(v: Vector_n) -> Vector_n:
            # Reduced Hessian on the free subspace: N&W ``G`` restricted to the
            # free variables left after active-bound fixing.
            return free_f * H(free_f * v)

        # --- null-space projector for the (free) general constraints ---
        # ``A_work`` masks out the fixed columns; inactive inequality rows are
        # already zeroed upstream.  A thin SVD reduces it once and gives the
        # Moore-Penrose pseudoinverse of ``A Aᵀ`` for free: with
        # ``A_work = U diag(s) Vᵀ`` we have ``(A_work A_workᵀ)⁺ = U
        # diag(pinv(s²)) Uᵀ``.  The rank cut is taken on ``s`` at full precision
        # (not on the squared ``s²``), and exactly-zero rows fall out on their
        # own (``s_i = 0`` -> ``0``), so no inactive-row ridge is needed.
        A_work = A * free_f[None, :]  # drop fixed columns
        U, s, _ = jnp.linalg.svd(A_work, full_matrices=False)
        eps = jnp.finfo(dtype).eps
        rcond = self.rcond if self.rcond is not None else eps * max(A_work.shape)
        # ``initial=0`` keeps ``m_gen == 0`` (bound-only QPs) well-defined.
        keep = s > rcond * jnp.max(s, initial=jnp.asarray(0.0, dtype))
        # Guard the reciprocal so the masked-out (tiny / zero) singular values
        # never form a ``1/0`` intermediate that could poison later AD.
        inv_s2 = jnp.where(keep, 1.0 / jnp.square(jnp.where(keep, s, 1.0)), 0.0)

        def solve_AAt(rhs: Float[Array, " m_gen"]) -> Float[Array, " m_gen"]:
            # Applies ``(A Aᵀ)⁺`` — N&W's ``(A Aᵀ)⁻¹`` (eq. 16.31), generalised
            # to the pseudoinverse for rank-deficient ``A`` (§16.8).
            return U @ (inv_s2 * (U.T @ rhs))

        # --- (optional) preconditioner: constraint preconditioner, N&W eq. 16.33 ---
        # ``preconditioner`` supplies ``M⁻¹`` through ``invert``.  With ``M = I``
        # (the default) ``apply_Minv`` is the identity on the free subspace and
        # ``solve_pcaat`` is the plain ``(A Aᵀ)⁺`` above, so ``project`` collapses
        # *exactly* to the orthogonal projector ``P_I``.  With a real ``M`` the
        # projector becomes ``P = M⁻¹ - M⁻¹Aᵀ(A M⁻¹Aᵀ)⁻¹ A M⁻¹`` — Algorithm 16.2
        # in its preconditioned form (H = M).  Note ``solve_AAt`` (plain SVD) is
        # kept for the 2-norm multiplier recovery / feasibility correction below,
        # which are preconditioner-independent.
        pre = self.preconditioner
        if pre is None or isinstance(pre, IdentityPreconditioner):

            def apply_Minv(v: Vector_n) -> Vector_n:
                return free_f * v

            solve_pcaat = solve_AAt
        else:

            def apply_Minv(v: Vector_n) -> Vector_n:
                return free_f * pre.invert(free_f * v)

            # Form the tiny ``m_gen × m_gen`` matrix ``A M⁻¹ Aᵀ`` by applying
            # ``M⁻¹`` to each row of ``A_work`` (a column of ``A_workᵀ``).  Zero
            # rows (inactive inequalities) stay zero and are handled by the same
            # ``pinv`` rank cut as the plain path.
            AMi = jax.vmap(apply_Minv)(A_work)  # (m_gen, n): ``(M⁻¹ aᵢ)``
            AMAt = A_work @ AMi.T  # (m_gen, m_gen)
            AMAt_pinv = jnp.linalg.pinv(AMAt, rcond=rcond)

            def solve_pcaat(rhs: Float[Array, " m_gen"]) -> Float[Array, " m_gen"]:
                return AMAt_pinv @ rhs

        # N&W projector (eq. 16.30 for ``M = I``; eq. 16.33 otherwise): maps into
        # ``null(A)`` and, in the ``M``-inner product, onto ``span(Z)``.
        def project(v: Vector_n) -> Vector_n:
            mv = apply_Minv(v)
            return mv - apply_Minv(A_work.T @ solve_pcaat(A_work @ mv))

        # --- particular solution: A_work d_p_free = b_gen - A d_fixed ---
        # N&W ``b`` (adjusted for the fixed variables) in ``A d = b``.
        b_eff = jnp.where(active_gen, b_gen - A @ d_fixed, jnp.zeros((m_gen,), dtype))
        # N&W range-space / ``Y``-space particular solution: the minimum-``M``-norm
        # ``x = M⁻¹Aᵀ(A M⁻¹Aᵀ)⁻¹b`` (§16.3, the initial point satisfying
        # ``A x = b``; collapses to ``Aᵀ(AAᵀ)⁻¹b`` when ``M = I``).
        d_p = apply_Minv(A_work.T @ solve_pcaat(b_eff)) + d_fixed

        g0 = (
            -sol_primal
        )  # N&W ``c`` (objective gradient ∇_x L); K step = sol <=> H d + g0 + Aᵀλ = 0

        # --- projected CG, warm-started from x0's free component ---
        # The residual is recomputed from scratch each step as
        # ``r = project(-(H d + g0))`` rather than via the usual recurrence.
        # This is defensive: it costs one extra HVP per step but keeps the stop
        # test and the noise-floor guard honest against floating-point roundoff
        # that would otherwise accumulate in the ``r -= alpha * P H p``
        # recurrence over many iterations.
        # N&W Algorithm 16.2 "Choose an initial point x satisfying A x = b": the
        # QP iterate ``d`` starts at ``d0`` (range-space particular solution plus
        # a null-space warm start from ``x0``).
        d0 = d_p + project(x0[0].x - d_p)
        # ``r`` carries N&W's preconditioned residual ``g = P r`` with the sign of
        # the search direction ``d = -g`` folded in (``r = -P(G d + c)``).
        neg_grad0 = -(H(d0) + g0)
        r0 = project(neg_grad0)
        # N&W ``rᵀg`` = ``dot(raw_residual, preconditioned_residual)``.  With the
        # sign folded in this is ``dot(neg_grad, r)``; for ``M = I`` it equals
        # ``‖r‖²`` because ``P`` is then an orthogonal projector.
        rz0 = jnp.dot(neg_grad0, r0)
        tol_sq = jnp.asarray(self.tol, dtype) ** 2

        def cg_body(_i: int, carry):
            d, r, p, rz, converged, n_cg = carry

            def do(carry):
                d, r, p, rz, _, n_cg = carry
                # N&W denominator ``dᵀ G d`` (eq. 16.28a).  ``p`` already lies in
                # ``null(A)`` (``project`` maps there), so ``G p`` needs no extra
                # projection — and projecting it with the (non-orthogonal)
                # preconditioned ``P`` would be *wrong*.
                Bp = hvp_work(p)
                pBp = jnp.dot(p, Bp)
                pp = jnp.dot(p, p)
                # SNOPT-style scale-invariant curvature guard.
                bad = pBp <= self.cg_regularization * pp
                # N&W eq. 16.28a: α = rᵀg / dᵀG d.
                alpha = jnp.where(
                    bad, jnp.asarray(0.0, dtype), rz / jnp.maximum(pBp, 1e-30)
                )
                d_new = d + alpha * p  # N&W eq. 16.28b: x ← x + α d
                # N&W eq. 16.28c-d (r⁺ = r + αG d; g⁺ = P r⁺) recomputed from
                # scratch instead of via the recurrence (see note above).
                neg_grad_new = -(H(d_new) + g0)
                r_new = project(neg_grad_new)
                rz_new = jnp.dot(neg_grad_new, r_new)  # N&W ``(r⁺)ᵀg⁺``
                beta = rz_new / jnp.maximum(
                    rz, 1e-30
                )  # N&W eq. 16.28e: β = (r⁺)ᵀg⁺ / rᵀg
                p_new = r_new + beta * p  # N&W eq. 16.28f: d ← -g⁺ + β d
                # Freeze on bad curvature or once the (true) residual stops
                # decreasing.  Once CG reaches its floor the next step has a
                # meaningless curvature ``pᵀBp`` that produces a huge spurious
                # ``alpha``; keeping the *previous* iterate makes the loop
                # return the best iterate seen and never accepts that
                # corrupting step.
                freeze = bad | (rz_new >= rz)
                conv = (rz_new < tol_sq) | freeze
                return (
                    jnp.where(freeze, d, d_new),
                    jnp.where(freeze, r, r_new),
                    jnp.where(freeze, p, p_new),
                    jnp.where(freeze, rz, rz_new),
                    conv,
                    n_cg + 1,
                )

            return jax.lax.cond(jnp.reshape(converged, ()), lambda c: c, do, carry)

        # ``n_cg`` counts CG steps that actually ran (the ``do`` branch); once
        # converged/frozen the ``lax.cond`` short-circuits and stops counting.
        # ``converged_flag`` is True on tol success *or* a residual-floor /
        # negative-curvature freeze — that is the solver's termination signal.
        init = (
            d0,
            r0,
            r0,
            rz0,
            jnp.reshape(rz0 < tol_sq, ()),
            jnp.zeros((), jnp.int32),
        )
        dx, _, _, _, converged_flag, n_cg = jax.lax.fori_loop(
            0, self.max_iter, cg_body, init
        )

        # Defensively pull the iterate back onto the constraint with one cached
        # back-solve (range-space correction, zero on fixed variables).  The
        # projector is exact, so this only mops up floating-point drift in
        # ``A_work dx = b_eff`` accumulated over the CG iterations.
        dx = dx - A_work.T @ solve_AAt(A_work @ dx - b_eff)

        # --- multiplier recovery ---
        # N&W ``λ*`` (eq. 16.20) via the normal equations (eq. 16.31 form):
        # ``λ_g = (AAᵀ)⁻¹ A · resid`` with ``resid = -(G d + c)`` on free vars.
        Hdx = H(dx)
        resid_primal = sol_primal - Hdx  # want Aᵀ λ_g ≈ this on the free variables
        lam_g = jnp.where(active_gen, solve_AAt(A_work @ (free_f * resid_primal)), 0.0)
        # one round of iterative refinement to clean up conditioning roundoff
        # (N&W §16.3 closing remark: iterative refinement recommended here).
        r_ref = free_f * (resid_primal - A.T @ lam_g)
        lam_g = jnp.where(active_gen, lam_g + solve_AAt(A_work @ r_ref), 0.0)

        # Bound multipliers close the primal stationarity residual on fixed vars
        # (the active-bound components of N&W eq. 16.37a — the KKT stationarity).
        resid_full = Hdx + A.T @ lam_g - sol_primal
        lam_lb = jnp.where(fix_lb, resid_full, jnp.zeros((n,), dtype))
        lam_ub = jnp.where(fix_ub, -resid_full, jnp.zeros((n,), dtype))

        step = (
            cast(Primal, Primal(dx)),
            cast(
                Dual,
                Dual(
                    eq_multipliers=lam_g[:meq],
                    ineq_multipliers=lam_g[meq:],
                    lb_multipliers=lam_lb,
                    ub_multipliers=lam_ub,
                ),
            ),
        )

        # --- carry the solver state: accumulate CG count, classify this solve ---
        # ``success`` follows the CG termination flag (tol hit *or* residual-floor
        # / negative-curvature freeze). Requiring ``final_rz < tol²`` alone would
        # reject exact float64 solves whose projected residual bottoms out around
        # a few units in the last place (~1e-16) while ``tol=1e-10`` demands
        # ``1e-20``. A non-finite step is ``singular``; exhausting ``max_iter``
        # without termination is ``max_steps_reached``.
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(step)])
        )
        success = finite & converged_flag
        status = RESULTS.where(
            finite,
            RESULTS.where(
                converged_flag, RESULTS.successful, RESULTS.max_steps_reached
            ),
            RESULTS.singular,
        )
        new_state = cast(
            ProjectedCGState,
            ProjectedCGState(
                n_iter=initial_state.n_iter + n_cg,
                success=success,
                status=status,
            ),
        )
        return step, new_state
