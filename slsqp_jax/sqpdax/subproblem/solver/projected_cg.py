from typing import cast

import jax
from equinox import field, tree_at
from jax import numpy as jnp

from ...dual import Dual
from ...preconditioner import Preconditioner
from ...primal import Primal
from ...types import Vector_n
from ..active_set import ActiveSetSubProblem
from .base import (
    KKT_SOLVER_RESULTS,
    RESULTS,
    KKTSolverState,
    SubProblemSolver,
    install_default_preconditioner,
)
from .multiplier_recovery import KKTMultiplierRecovery, MultiplierRecovery
from .projector import Projector, SVDProjector


class ProjectedCGState(KKTSolverState):
    """Carry for a single projected-CG KKT solve.

    Inherits the standardised fields of
    :class:`~slsqp_jax.sqpdax.subproblem.solver.base.KKTSolverState`.
    ``n_iter`` accumulates CG iterations across calls so an outer active-set
    loop can report total inner work; ``projected_grad_norm`` is the
    ``M``-norm of the projected residual the CG stopped at and
    ``feasibility_residual`` the working-constraint residual of the returned
    (range-space corrected) step.
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
    * **Projector.** Null / range bases are never formed. The pluggable
      ``projector`` supplies the range-space solve ``(A Aᵀ)⁺`` so that
      ``P(v) = v - Aᵀ (A Aᵀ)⁺ A v``; the default SVD backend drops singular
      values below ``rcond · max(s)`` (rank-revealing; handles LICQ
      violations and zero rows from inactive inequalities).
    * **Preconditioning (optional).** ``preconditioner`` supplies ``M⁻¹``
      (SPD reduced-Hessian approximation, N&W eq. 16.26), upgrading the
      projector to the constraint preconditioner (eq. 16.33). ``None`` is
      the identity.
    * **CG** runs in that null space against the Lagrangian Hessian via
      ``kkt_mvp_primal``, warm-started from ``x0``.
    * **Multipliers** are recovered by :attr:`multiplier_recovery` from the
      stationarity residual (default: general multipliers by the projector's
      normal-equation solve with one refinement round; bound multipliers
      from the residual on fixed variables).

    Attributes
    ----------
    solver_state_class
        :class:`ProjectedCGState`.
    max_iter
        Maximum CG iterations.
    tol
        Absolute projected-residual tolerance.
    rtol
        Relative projected-residual tolerance: CG also stops once
        ``‖r‖_M ≤ rtol · ‖r₀‖_M``, i.e. the reduced system is solved to
        working precision relative to its initial residual.
    cg_regularization
        Scale-invariant floor for the curvature check ``pᵀ H p``.
    preconditioner
        Optional SPD reduced-Hessian preconditioner ``M``.
    projector
        :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.Projector`
        backend building the per-working-set
        :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.ProjectionContext`
        (default :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.SVDProjector`;
        its ``rcond`` is configured through it).
    multiplier_recovery
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.MultiplierRecovery`
        producing the dual block from the CG step (default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.KKTMultiplierRecovery`,
        which keeps the working-set sign tests KKT-consistent).
    """

    solver_state_class: type[ProjectedCGState] = ProjectedCGState

    max_iter: int = 100
    tol: float = 1e-10
    rtol: float = 1e-8
    cg_regularization: float = 1e-6
    # Preconditioner ``M`` for the reduced Hessian (N&W eq. 16.26).  ``invert``
    # supplies ``M⁻¹``.  ``None`` is the identity (unpreconditioned Algorithm
    # 16.2, ``H = I``) and recovers the plain projector exactly.  A supplied
    # preconditioner turns the projector into the constraint preconditioner
    # (eq. 16.33).  Must be SPD for CG.
    preconditioner: Preconditioner | None = None
    # Range-space backend building the ``ProjectionContext`` per working set.
    projector: Projector = field(default_factory=SVDProjector)
    multiplier_recovery: MultiplierRecovery = field(
        default_factory=KKTMultiplierRecovery
    )

    def accepts_preconditioner(self) -> bool:
        """``True``: :attr:`preconditioner` drives the constraint preconditioner."""
        return True

    def with_default_preconditioner(
        self, preconditioner: Preconditioner | None
    ) -> "ProjectedCGSubProblemSolver":
        """Install ``preconditioner`` when :attr:`preconditioner` is ``None``."""
        return install_default_preconditioner(self, preconditioner)

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
        # N&W Ch. 16 ``A``: the constraint matrix of the QP
        # ``min ½ dᵀG d + cᵀd  s.t.  A d = b`` (eq. 16.3), read through the
        # projection context (inactive rows and fixed columns zeroed).
        n = subproblem.lagrangian.n
        meq = subproblem.lagrangian.meq
        mineq = subproblem.lagrangian.mineq
        dtype = subproblem.lagrangian.ref.x.dtype

        # --- unpack the right-hand side (sol = -∇f, -c) ---
        _sol_primal, sol_dual = subproblem.kkt_rhs()
        sol_primal = _sol_primal.x  # -∇f; note ``g0 = -sol_primal`` is N&W ``c``
        b_gen = jnp.concatenate(
            [sol_dual.eq_multipliers, sol_dual.ineq_multipliers]
        )  # -c_general (eq block then ineq block); N&W ``b`` in ``A d = b``

        # --- working-set geometry and range-space solves (N&W §16.3) ---
        # Active bounds fix variables (lower bound wins ties); the projector
        # acts on the free columns of the active general rows.
        ctx = self.projector.build(subproblem, self.preconditioner)
        free_f = ctx.free_f

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

        project = ctx.project

        # --- particular solution: A_work d_p_free = b_gen - A d_fixed ---
        # N&W range-space / ``Y``-space particular solution: the minimum-``M``-norm
        # ``x = M⁻¹Aᵀ(A M⁻¹Aᵀ)⁻¹b`` (§16.3, the initial point satisfying
        # ``A x = b``; collapses to ``Aᵀ(AAᵀ)⁻¹b`` when ``M = I``).
        d_p = ctx.particular_solution(b_gen)

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
        r0, w0 = ctx.project_pair(neg_grad0)
        # N&W ``rᵀg`` = ``dot(raw_residual, preconditioned_residual)``.  With the
        # sign folded in this is ``dot(neg_grad, r)``; for ``M = I`` it equals
        # ``‖r‖²`` because ``P`` is then an orthogonal projector.
        rz0 = jnp.dot(neg_grad0, r0)
        # Stop test on the squared ``M``-norm of the projected residual:
        # absolute ``tol²`` or relative ``rtol² · rz0``, whichever is looser.
        tol_sq = jnp.maximum(
            jnp.asarray(self.tol, dtype) ** 2,
            jnp.asarray(self.rtol, dtype) ** 2 * jnp.maximum(rz0, 0.0),
        )

        # Round-off floor detector.  With ``P = M̃⁻¹ Q`` (``Q`` idempotent,
        # ``P`` symmetric; see ``ProjectionContext.project_pair``) exact
        # arithmetic gives ``rz = neg_gradᵀ P neg_grad = (Q neg_grad)ᵀ r``.
        # Once CG has solved the reduced system to working precision the
        # recomputed ``r`` is round-off of size ``eps · ‖G d + c‖`` — *not*
        # in ``null(A)`` — and ``rz`` is then dominated by the contamination
        # ``‖neg_grad‖ · ‖r_noise‖`` while ``(Q neg_grad)ᵀ r`` stays at
        # ``‖noise‖²``: the two disagree by orders of magnitude.  Continuing
        # would step along the noise direction with ``alpha = rz / pᵀBp`` of
        # two meaningless quantities and drag the iterate out of the
        # constraint subspace while the model keeps "decreasing" (the decrease
        # is bought with infeasibility).  Neither ``tol`` nor ``rtol`` sees
        # this floor reliably because it scales with ``‖c‖``, not with the
        # (possibly much smaller) projected residual.  The check must use the
        # projector's own masked metric: ``rᵀ M r`` with the full ``M`` is
        # wrong as soon as active bounds fix variables under a non-diagonal
        # preconditioner (the inverse of a principal block of ``M⁻¹`` is not
        # the principal block of ``M``), and fired spuriously on every bound-
        # constrained PASLS iterate.
        def at_floor(rz_val: jax.Array, r_val: Vector_n, w_val: Vector_n) -> jax.Array:
            wr = jnp.dot(w_val, r_val)
            return (rz_val > 4.0 * wr) | (rz_val < 0.25 * wr)

        def model(d: Vector_n, neg_grad: Vector_n) -> jax.Array:
            # QP model ``q(d) = ½ dᵀ G d + cᵀ d`` evaluated from the already
            # available ``neg_grad = -(G d + c)``:
            # ``q = ½ dᵀ(G d + c) + ½ cᵀ d = -½ dᵀ neg_grad + ½ g0ᵀ d``.
            return 0.5 * (jnp.dot(g0, d) - jnp.dot(d, neg_grad))

        q0 = model(d0, neg_grad0)

        def cg_body(_i: int, carry):
            d, r, p, rz, q, converged, n_cg = carry

            def do(carry):
                d, r, p, rz, q, _, n_cg = carry
                # N&W denominator ``dᵀ G d`` (eq. 16.28a).  ``p`` already lies in
                # ``null(A)`` (``project`` maps there), so ``G p`` needs no extra
                # projection — and projecting it with the (non-orthogonal)
                # preconditioned ``P`` would be *wrong*.
                Bp = hvp_work(p)
                pBp = jnp.dot(p, Bp)
                pp = jnp.dot(p, p)
                # SNOPT-style scale-invariant curvature guard.
                bad = pBp <= self.cg_regularization * pp
                # Negative / vanishing curvature on the very first CG step
                # (``d`` is still the particular solution, zero on an interior
                # iterate): returning ``d`` unchanged would hand the outer
                # loop a zero step although ``p = -P ∇q`` is a descent
                # direction.  Use the curvature with its sign flipped
                # (``|G|`` modification of modified-Newton methods): the step
                # that would be Newton's on the convexified model.  Later
                # negative-curvature encounters keep the progress made so far
                # (Steihaug-style stop).
                first = n_cg == 0
                alpha_pd = rz / jnp.maximum(pBp, 1e-30)
                alpha_neg = rz / jnp.maximum(
                    jnp.abs(pBp), self.cg_regularization * jnp.maximum(pp, 1e-30)
                )
                take_neg = bad & first
                # N&W eq. 16.28a: α = rᵀg / dᵀG d.
                alpha = jnp.where(
                    bad,
                    jnp.where(take_neg, alpha_neg, jnp.asarray(0.0, dtype)),
                    alpha_pd,
                )
                d_new = d + alpha * p  # N&W eq. 16.28b: x ← x + α d
                # N&W eq. 16.28c-d (r⁺ = r + αG d; g⁺ = P r⁺) recomputed from
                # scratch instead of via the recurrence (see note above).
                neg_grad_new = -(H(d_new) + g0)
                r_new, w_new = ctx.project_pair(neg_grad_new)
                rz_new = jnp.dot(neg_grad_new, r_new)  # N&W ``(r⁺)ᵀg⁺``
                beta = rz_new / jnp.maximum(
                    rz, 1e-30
                )  # N&W eq. 16.28e: β = (r⁺)ᵀg⁺ / rᵀg
                p_new = r_new + beta * p  # N&W eq. 16.28f: d ← -g⁺ + β d
                q_new = model(d_new, neg_grad_new)
                # Freeze on bad curvature or once the QP model stops
                # decreasing.  In exact arithmetic every CG step with positive
                # curvature lowers ``q`` by ``½ rz² / pᵀBp > 0`` while the
                # (preconditioned) residual norm ``rz`` is *not* monotone, so
                # the model — not ``rz`` — is the quantity that detects the
                # round-off floor.  Freezing on ``rz_new >= rz`` stopped the
                # solve after one or two iterations on well-posed systems and
                # returned a near-zero step labelled ``residual_floor``.  Once
                # CG reaches its floor the next step has a meaningless
                # curvature ``pᵀBp`` that produces a huge spurious ``alpha``;
                # keeping the *previous* iterate makes the loop return the
                # best iterate seen and never accepts that corrupting step.
                freeze = (~take_neg & (bad | (q_new >= q))) | ~jnp.isfinite(q_new)
                # The step just taken is sound; a residual at the round-off
                # floor means the reduced system is solved: keep ``d_new`` and
                # stop before the next (noise) direction is used.
                conv = (
                    (rz_new < tol_sq)
                    | freeze
                    | take_neg
                    | at_floor(rz_new, r_new, w_new)
                )
                return (
                    jnp.where(freeze, d, d_new),
                    jnp.where(freeze, r, r_new),
                    jnp.where(freeze, p, p_new),
                    jnp.where(freeze, rz, rz_new),
                    jnp.where(freeze, q, q_new),
                    conv,
                    n_cg + 1,
                )

            return jax.lax.cond(jnp.reshape(converged, ()), lambda c: c, do, carry)

        # ``n_cg`` counts CG steps that actually ran (the ``do`` branch); once
        # converged/frozen the ``lax.cond`` short-circuits and stops counting.
        # ``converged_flag`` is True on tol success *or* a model-floor /
        # negative-curvature freeze — that is the solver's termination signal.
        init = (
            d0,
            r0,
            r0,
            rz0,
            q0,
            jnp.reshape((rz0 < tol_sq) | at_floor(rz0, r0, w0), ()),
            jnp.zeros((), jnp.int32),
        )
        dx, _, _, rz_f, _, converged_flag, n_cg = jax.lax.fori_loop(
            0, self.max_iter, cg_body, init
        )
        # ``rz`` is a squared ``M``-norm; roundoff can drive it slightly
        # negative once CG hits its floor, so clamp before the tolerance test.
        rz_f = jnp.maximum(rz_f, 0.0)
        hit_tol = rz_f < tol_sq

        # Defensively pull the iterate back onto the constraint with one cached
        # back-solve (range-space correction, zero on fixed variables).  The
        # projector is exact, so this only mops up floating-point drift in
        # ``A_work dx = b_eff`` accumulated over the CG iterations.
        dx = ctx.feasibility_correction(dx, b_gen)

        # --- multiplier recovery ---
        # N&W ``λ*`` (eq. 16.20) via the projector's range-space solve; the
        # strategy also closes the bound block on the fixed variables (the
        # active-bound components of N&W eq. 16.37a).
        primal_dx = cast(Primal, Primal(dx))
        step = (
            primal_dx,
            self.multiplier_recovery.recover(subproblem, ctx, primal_dx),
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
        success = finite & converged_flag & ctx.converged
        status = RESULTS.where(
            finite,
            RESULTS.where(
                converged_flag & ctx.converged,
                RESULTS.successful,
                RESULTS.max_steps_reached,
            ),
            RESULTS.singular,
        )
        reason = KKT_SOLVER_RESULTS.where(
            finite,
            KKT_SOLVER_RESULTS.where(
                ctx.converged,
                KKT_SOLVER_RESULTS.where(
                    hit_tol,
                    KKT_SOLVER_RESULTS.converged,
                    KKT_SOLVER_RESULTS.where(
                        converged_flag,
                        KKT_SOLVER_RESULTS.residual_floor,
                        KKT_SOLVER_RESULTS.max_iter_reached,
                    ),
                ),
                KKT_SOLVER_RESULTS.projector_failure,
            ),
            KKT_SOLVER_RESULTS.nonfinite,
        )
        res_dtype = initial_state.feasibility_residual.dtype
        new_state = cast(
            ProjectedCGState,
            tree_at(
                lambda state: (
                    state.n_iter,
                    state.success,
                    state.status,
                    state.feasibility_residual,
                    state.n_refinements,
                    state.projected_grad_norm,
                    state.reason,
                    state.nonfinite,
                ),
                initial_state,
                (
                    initial_state.n_iter + n_cg + ctx.n_iter,
                    success,
                    status,
                    ctx.feasibility_residual(dx, b_gen).astype(res_dtype),
                    jnp.asarray(0, jnp.int32),
                    jnp.sqrt(rz_f).astype(res_dtype),
                    reason,
                    jnp.logical_not(finite),
                ),
            ),
        )
        return step, new_state
