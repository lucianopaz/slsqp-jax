"""Preconditioned MINRES-QLP on the full saddle-point KKT operator.

Choi, Paige & Saunders, *MINRES-QLP: a Krylov subspace method for indefinite
or singular symmetric systems*, SIAM J. Sci. Comput. 33(4), 2011 (Table 3.5).
"""

from typing import Callable, NamedTuple, cast

import jax
from equinox import field, tree_at
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, Float, Int

from ...dual import Dual
from ...preconditioner import Preconditioner
from ...primal import PrimalType
from ...types import Scalar
from ..base import SubProblem
from .base import KKT_SOLVER_RESULTS, RESULTS, KKTSolverState, SubProblemSolver
from .multiplier_recovery import KKTMultiplierRecovery, MultiplierRecovery
from .projector import ProjectionContext, Projector, SVDProjector

__all__ = [
    "MinresQLPState",
    "MinresQLPSubProblemSolver",
]

Vector = Float[Array, " k"]


def _sym_ortho(a: Scalar, b: Scalar) -> tuple[Scalar, Scalar, Scalar]:
    """Stable symmetric Givens rotation ``(c, s, r)`` with ``r = √(a² + b²) ≥ 0``.

    Parameters
    ----------
    a, b
        Scalars to rotate; ``c = a / r`` and ``s = b / r``.

    Returns
    -------
    tuple of Scalar
        ``(c, s, r)``; ``(1, 0, 0)`` when both inputs vanish.

    Notes
    -----
    Choi (2006), Table 2.9 / ACM Algorithm 937. Written branch-free so it
    traces to a handful of ``where`` selects.
    """
    one = jnp.ones_like(a)
    zero = jnp.zeros_like(a)
    abs_a = jnp.abs(a)
    abs_b = jnp.abs(b)
    sign_a = jnp.where(a == 0, one, jnp.sign(a))
    sign_b = jnp.where(b == 0, one, jnp.sign(b))

    # |a| >= |b| branch (also covers b == 0 and a == b == 0).
    t_a = jnp.where(abs_a > 0, b / jnp.where(abs_a > 0, a, one), zero)
    scale_a = jnp.sqrt(1.0 + t_a * t_a)
    c_a = sign_a / scale_a
    s_a = c_a * t_a
    r_a = abs_a * scale_a

    # |b| > |a| branch (also covers a == 0, b != 0).
    t_b = jnp.where(abs_b > 0, a / jnp.where(abs_b > 0, b, one), zero)
    scale_b = jnp.sqrt(1.0 + t_b * t_b)
    s_b = sign_b / scale_b
    c_b = s_b * t_b
    r_b = abs_b * scale_b

    a_ge_b = abs_a >= abs_b
    return (
        jnp.where(a_ge_b, c_a, c_b),
        jnp.where(a_ge_b, s_a, s_b),
        jnp.where(a_ge_b, r_a, r_b),
    )


class _PMinresQLPCarry(NamedTuple):
    """Loop carry of :func:`_pminres_qlp_solve` (names follow the reference code)."""

    r1: Vector
    r2: Vector
    r3: Vector
    betal: Scalar
    betan: Scalar
    cs: Scalar
    sn: Scalar
    cr2: Scalar
    sr2: Scalar
    dltan: Scalar
    eplnn: Scalar
    gama: Scalar
    gamal: Scalar
    gamal2: Scalar
    eta: Scalar
    etal: Scalar
    etal2: Scalar
    vepln: Scalar
    veplnl: Scalar
    veplnl2: Scalar
    tau: Scalar
    taul: Scalar
    u: Scalar
    ul: Scalar
    ul2: Scalar
    ul3: Scalar
    w: Vector
    wl: Vector
    x: Vector
    xl2: Vector
    phi: Scalar
    xl2norm: Scalar
    Anorm: Scalar
    gmin: Scalar
    gminl: Scalar
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def _pminres_qlp_solve(
    matvec: Callable[[Vector], Vector],
    rhs: Vector,
    *,
    tol: float,
    max_iter: int,
    precond: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, Bool[Array, ""], Int[Array, ""]]:
    """Solve the symmetric (indefinite / singular) system ``A x = b``.

    Preconditioned MINRES-QLP, Choi, Paige & Saunders (2011) Table 3.5, run
    in QLP mode on every iteration. On singular consistent systems the
    iterate converges to the minimum-length solution.

    Parameters
    ----------
    matvec
        Symmetric operator ``v ↦ A v``.
    rhs
        Right-hand side.
    tol
        Relative residual tolerance ``‖r‖ / (‖A‖‖x‖ + ‖b‖_M)``.
    max_iter
        Maximum Lanczos iterations.
    precond
        Optional SPD preconditioner ``v ↦ M⁻¹ v``.

    Returns
    -------
    x
        Final iterate.
    converged
        ``True`` when the relative residual met ``tol`` and ``x`` is finite.
    n_iter
        Lanczos iterations actually run.
    """
    n = rhs.shape[0]
    dt = rhs.dtype
    tiny = jnp.asarray(1e-30, dt)
    tol_ = jnp.asarray(tol, dt)
    zeros = jnp.zeros((n,), dt)
    zero = jnp.zeros((), dt)

    def apply_precond(v: Vector) -> Vector:
        return v if precond is None else precond(v)

    r2 = rhs
    r3 = apply_precond(r2)
    beta1 = jnp.sqrt(jnp.maximum(jnp.dot(r2, r3), 0.0))
    beta1_safe = jnp.maximum(beta1, tiny)

    init = _PMinresQLPCarry(
        r1=zeros,
        r2=r2,
        r3=r3,
        betal=zero,
        betan=beta1,
        cs=-jnp.ones((), dt),
        sn=zero,
        cr2=-jnp.ones((), dt),
        sr2=zero,
        dltan=zero,
        eplnn=zero,
        gama=zero,
        gamal=zero,
        gamal2=zero,
        eta=zero,
        etal=zero,
        etal2=zero,
        vepln=zero,
        veplnl=zero,
        veplnl2=zero,
        tau=zero,
        taul=zero,
        u=zero,
        ul=zero,
        ul2=zero,
        ul3=zero,
        w=zeros,
        wl=zeros,
        x=zeros,
        xl2=zeros,
        phi=beta1,
        xl2norm=zero,
        Anorm=zero,
        gmin=zero,
        gminl=zero,
        iteration=jnp.zeros((), jnp.int32),
        converged=beta1 < tiny,
    )

    def safe(v: Scalar) -> Scalar:
        return jnp.where(jnp.abs(v) > tiny, v, tiny)

    def do_step(state: _PMinresQLPCarry) -> _PMinresQLPCarry:
        k = state.iteration + 1

        # --- Lanczos step ---
        betal = state.betal
        beta = state.betan
        v = state.r3 / jnp.maximum(beta, tiny)
        r3_new = matvec(v)
        r3_new = r3_new - jnp.where(
            k > 1, state.r1 * (beta / jnp.maximum(betal, tiny)), zeros
        )
        alfa = jnp.dot(r3_new, v)
        r3_new = r3_new - state.r2 * (alfa / jnp.maximum(beta, tiny))
        r1_new = state.r2
        r2_new = r3_new
        r3_new = apply_precond(r2_new)
        betan_new = jnp.sqrt(jnp.maximum(jnp.dot(r2_new, r3_new), 0.0))
        pnorm = jnp.sqrt(betal**2 + alfa**2 + betan_new**2)

        # --- previous left rotation Q_{k-1} ---
        dbar = state.dltan
        dlta = state.cs * dbar + state.sn * alfa
        gbar = state.sn * dbar - state.cs * alfa
        eplnn_new = state.sn * betan_new
        dltan_new = -state.cs * betan_new

        # --- current left rotation Q_k ---
        gamal2 = state.gamal
        gamal = state.gama
        cs_new, sn_new, gama_new = _sym_ortho(gbar, betan_new)
        taul2 = state.taul
        taul_new = state.tau
        tau_new = cs_new * state.phi
        phi_new = sn_new * state.phi

        # --- previous right rotation P_{k-2,k} (k > 2) ---
        veplnl2 = state.veplnl
        etal2 = state.etal
        dlta_k2 = jnp.where(k > 2, state.sr2 * state.vepln - state.cr2 * dlta, dlta)
        veplnl_new = jnp.where(
            k > 2, state.cr2 * state.vepln + state.sr2 * dlta, state.veplnl
        )
        etal_new = jnp.where(k > 2, state.eta, state.etal)
        eta_new = jnp.where(k > 2, state.sr2 * gama_new, zero)
        gama_k2 = jnp.where(k > 2, -state.cr2 * gama_new, gama_new)

        # --- current right rotation P_{k-1,k} (k > 1) ---
        cr1_raw, sr1_raw, gamal_raw = _sym_ortho(gamal, dlta_k2)
        cr1_new = jnp.where(k > 1, cr1_raw, -jnp.ones((), dt))
        sr1_new = jnp.where(k > 1, sr1_raw, zero)
        gamal_new = jnp.where(k > 1, gamal_raw, gamal)
        vepln_new = jnp.where(k > 1, sr1_new * gama_k2, zero)
        gama_final = jnp.where(k > 1, -cr1_new * gama_k2, gama_k2)

        # --- mu coefficients ---
        ul4 = state.ul3
        ul3_new = state.ul2
        ul2_new = jnp.where(
            k > 2,
            (taul2 - etal2 * ul4 - veplnl2 * ul3_new) / safe(gamal2),
            state.ul2,
        )
        ul_new = jnp.where(
            k > 1,
            (taul_new - etal_new * ul3_new - veplnl_new * ul2_new) / safe(gamal_new),
            state.ul,
        )
        u_new = jnp.where(
            jnp.abs(gama_final) > tiny,
            (tau_new - eta_new * ul2_new - vepln_new * ul_new) / safe(gama_final),
            zero,
        )
        xl2norm_new = jnp.sqrt(state.xl2norm**2 + ul2_new**2)

        # --- w vectors and solution (QLP mode) ---
        w_old = state.w
        wl_old = state.wl
        # k > 2
        w_g = wl_old * state.sr2 - v * state.cr2
        wl2_g = wl_old * state.cr2 + v * state.sr2
        v_tmp = w_old * cr1_new + w_g * sr1_new
        w_g = w_old * sr1_new - w_g * cr1_new
        wl_g = v_tmp
        # k == 2
        wl_2 = w_old * cr1_new + v * sr1_new
        w_2 = w_old * sr1_new - v * cr1_new
        # k == 1
        wl_1 = v * sr1_new
        w_1 = -v * cr1_new

        wl2_out = jnp.where(k > 2, wl2_g, wl_old)
        wl_out = jnp.where(k > 2, wl_g, jnp.where(k == 2, wl_2, wl_1))
        w_out = jnp.where(k > 2, w_g, jnp.where(k == 2, w_2, w_1))

        xl2_new = state.xl2 + wl2_out * ul2_new
        x_new = xl2_new + wl_out * ul_new + w_out * u_new

        # --- next right rotation P_{k-1,k+1} ---
        cr2_new, sr2_new, gamal_store = _sym_ortho(gamal_new, eplnn_new)

        # --- norms ---
        abs_gama = jnp.abs(gama_final)
        Anorm_new = jnp.maximum(
            jnp.maximum(state.Anorm, pnorm), jnp.maximum(gamal_new, abs_gama)
        )
        gminl_new = jnp.where(k == 1, gama_final, state.gmin)
        gmin_new = jnp.where(
            k == 1,
            gama_final,
            jnp.minimum(jnp.minimum(state.gminl, gamal_new), abs_gama),
        )

        # --- convergence ---
        xnorm = jnp.sqrt(xl2norm_new**2 + ul_new**2 + u_new**2)
        relres = jnp.abs(phi_new) / (Anorm_new * jnp.maximum(xnorm, tiny) + beta1_safe)
        lanczos_breakdown = betan_new < tiny * jnp.maximum(beta1_safe, 1.0)
        residual_small = jnp.abs(phi_new) < tol_ * beta1_safe
        converged = (relres < tol_) | (lanczos_breakdown & residual_small)
        stop_now = converged | lanczos_breakdown

        return _PMinresQLPCarry(
            r1=r1_new,
            r2=r2_new,
            r3=r3_new,
            betal=beta,
            betan=betan_new,
            cs=cs_new,
            sn=sn_new,
            cr2=cr2_new,
            sr2=sr2_new,
            dltan=dltan_new,
            eplnn=eplnn_new,
            gama=gama_final,
            gamal=gamal_store,
            gamal2=gamal_new,
            eta=eta_new,
            etal=etal_new,
            etal2=etal2,
            vepln=vepln_new,
            veplnl=veplnl_new,
            veplnl2=veplnl2,
            tau=tau_new,
            taul=taul_new,
            u=u_new,
            ul=ul_new,
            ul2=ul2_new,
            ul3=ul3_new,
            w=w_out,
            wl=wl_out,
            x=x_new,
            xl2=xl2_new,
            phi=phi_new,
            xl2norm=xl2norm_new,
            Anorm=Anorm_new,
            gmin=gmin_new,
            gminl=gminl_new,
            iteration=k,
            converged=stop_now,
        )

    def body(_i, state: _PMinresQLPCarry) -> _PMinresQLPCarry:
        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final = jax.lax.fori_loop(0, max_iter, body, init)
    final_relres = jnp.abs(final.phi) / (
        jnp.maximum(final.Anorm, tiny) * jnp.maximum(jnp.linalg.norm(final.x), tiny)
        + beta1_safe
    )
    success = (final_relres < tol_) & jnp.all(jnp.isfinite(final.x))
    return final.x, success, final.iteration


class MinresQLPState(KKTSolverState):
    """Carry for a single MINRES-QLP KKT solve.

    Inherits the standardised fields of
    :class:`~slsqp_jax.sqpdax.subproblem.solver.base.KKTSolverState`.
    ``n_iter`` accumulates Lanczos iterations (plus projector work);
    ``n_refinements`` counts the post-solve feasibility refinement rounds
    that ran; ``feasibility_residual`` is the working-constraint residual
    the projection bottomed out at and ``projected_grad_norm`` the
    stationarity residual ``‖free ⊙ (H d + g + Aᵀλ)‖`` of the returned step.
    """


class MinresQLPSubProblemSolver(
    SubProblemSolver[PrimalType, SubProblem[PrimalType], MinresQLPState]
):
    """Preconditioned MINRES-QLP on the saddle-point KKT operator of a working set.

    Solves the symmetric indefinite system

    ```
    [ H      Aᵀ ] [ d ]   [ -∇f ]
    [ A      D  ] [ λ ] = [  b  ]
    ```

    directly for the primal step and the general multipliers, where the
    blocks are read matrix-free through
    :meth:`~slsqp_jax.sqpdax.subproblem.base.SubProblem.kkt_operator` (the
    dual-dual block ``D`` is whatever the subproblem declares — zero for the
    active-set QP, ``-δI`` for a dual-regularised model). Active bounds fix
    variables exactly as in
    :class:`~slsqp_jax.sqpdax.subproblem.solver.projected_cg.ProjectedCGSubProblemSolver`:
    the working geometry (active rows, free columns, fixed block) comes from
    the pluggable :attr:`projector`, the fixed block is eliminated into the
    right-hand side and the reduced system is solved on the free variables
    and the active general rows.

    **Preconditioner.** Block-diagonal SPD (Choi 2006, §3.4)

    ```
    P⁻¹ = diag( M⁻¹ ,  (A_work M⁻¹ A_workᵀ)⁺ )
    ```

    with ``M⁻¹`` from :attr:`preconditioner` (identity when ``None``) and
    the Schur block from the projector's range-space solve. Fixed variables
    and inactive rows are preconditioned by the identity so ``P⁻¹`` stays
    positive definite on the whole reduced space.

    **Post-processing.** MINRES-QLP only satisfies ``A_work d = b_eff`` to
    its residual tolerance, so the iterate is projected onto the constraint
    along the ``M``-metric range space (one back-solve) and then refined for
    up to :attr:`proj_refine_max_iter` rounds while the residual exceeds
    ``proj_refine_atol + proj_refine_rtol · (‖b_eff‖ + 1)`` (Heinkenschloss &
    Ridzal 2014, Algorithm 4.18 step 1(a)). The multipliers are then
    re-recovered for the projected step through :attr:`multiplier_recovery`
    (the MINRES multipliers are discarded, so bound multipliers come out
    KKT-consistent too).

    **Diagnostics.** The residual the refinement bottomed out at is reported
    as ``feasibility_residual`` and the round count as ``n_refinements``. A
    residual still above the refinement target (also floored at a few units
    of roundoff in ``‖A_work‖‖d‖ + ‖b_eff‖``) yields
    ``reason = residual_floor`` with ``success = False`` and
    ``status = stagnation``: the step is infeasible for the working set and
    the outer active-set loop treats it as a KKT solver failure. An
    exhausted Lanczos budget on a feasible step is ``max_iter_reached`` /
    ``max_steps_reached`` (a usable partial step, not a failure).

    Attributes
    ----------
    solver_state_class
        :class:`MinresQLPState`.
    max_iter
        Maximum Lanczos iterations.
    tol
        MINRES-QLP relative residual tolerance.
    proj_refine_max_iter
        Maximum feasibility-refinement rounds after the initial projection.
    proj_refine_rtol, proj_refine_atol
        Refinement target ``atol + rtol · (‖b_eff‖ + 1)``.
    preconditioner
        Optional SPD ``M`` for the primal block (its ``invert`` supplies ``M⁻¹``).
    projector
        :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.Projector`
        supplying the working geometry, the Schur-block solve and the
        feasibility projection (default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.projector.SVDProjector`).
    multiplier_recovery
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.MultiplierRecovery`
        producing the dual block from the projected step (default
        :class:`~slsqp_jax.sqpdax.subproblem.solver.multiplier_recovery.KKTMultiplierRecovery`).
    """

    solver_state_class: type[MinresQLPState] = MinresQLPState

    max_iter: int = 200
    tol: float = 1e-10
    proj_refine_max_iter: int = field(static=True, default=3)
    proj_refine_rtol: float = 1e-10
    proj_refine_atol: float = 1e-14
    preconditioner: Preconditioner | None = None
    projector: Projector = field(default_factory=SVDProjector)
    multiplier_recovery: MultiplierRecovery = field(
        default_factory=KKTMultiplierRecovery
    )

    def solve(
        self,
        subproblem: SubProblem[PrimalType],
        x0: tuple[PrimalType, Dual],
        initial_state: MinresQLPState,
    ) -> tuple[tuple[PrimalType, Dual], MinresQLPState]:
        """Solve the working-set KKT system by preconditioned MINRES-QLP.

        Parameters
        ----------
        subproblem
            KKT model of the working set. Its primal must be the decision
            variables only (the projector's Jacobian columns).
        x0
            Warm-start ``(primal_step, dual)``; the free primal components and
            the active general multipliers seed the Krylov iteration.
        initial_state
            Carry whose ``n_iter`` is accumulated into the returned state.

        Returns
        -------
        step
            Primal-dual solution ``(d, λ)`` in the subproblem's dual convention.
        state
            Updated :class:`MinresQLPState`.

        Raises
        ------
        TypeError
            If the subproblem's primal has more coordinates than the
            constraint Jacobian has columns (e.g. a slack-augmented model).
        """
        ctx = self.projector.build(subproblem, self.preconditioner)
        rhs_primal, rhs_dual = subproblem.kkt_rhs()
        rhs_p, unravel_primal = ravel_pytree(rhs_primal)
        n = ctx.A.shape[1]
        if rhs_p.shape[0] != n:
            raise TypeError(
                "MinresQLPSubProblemSolver acts on the decision variables only; "
                f"the subproblem primal has {rhs_p.shape[0]} coordinates but the "
                f"constraint Jacobian {n} columns."
            )
        dtype = rhs_p.dtype
        m = ctx.A.shape[0]
        free_f = ctx.free_f
        active_f = ctx.active_rows.astype(dtype)
        zero_dual = cast(Dual, jax.tree.map(jnp.zeros_like, subproblem.lagrangian.dual))
        meq = zero_dual.eq_multipliers.shape[0]

        def general_dual(lam_g: Float[Array, " m"]) -> Dual:
            return cast(
                Dual,
                Dual(
                    eq_multipliers=lam_g[:meq],
                    ineq_multipliers=lam_g[meq:],
                    lb_multipliers=zero_dual.lb_multipliers,
                    ub_multipliers=zero_dual.ub_multipliers,
                ),
            )

        def kkt_blocks(
            d_flat: Float[Array, " n"], lam_g: Float[Array, " m"]
        ) -> tuple[Float[Array, " n"], Float[Array, " m"]]:
            kp, kd = subproblem.kkt_operator(
                (unravel_primal(d_flat), general_dual(lam_g))
            )
            return (
                ravel_pytree(kp)[0],
                jnp.concatenate([kd.eq_multipliers, kd.ineq_multipliers]),
            )

        # Reduced KKT operator on ``[d_free; λ_active]``: fixed variables and
        # inactive rows are masked out of both the input and the output.
        def matvec(z: Vector) -> Vector:
            top, bot = kkt_blocks(free_f * z[:n], active_f * z[n:])
            return jnp.concatenate([free_f * top, active_f * bot])

        # Right-hand side with the fixed block eliminated: the primal row
        # becomes ``free ⊙ (-∇f - H d_fixed)`` and the dual row ``b - A d_fixed``
        # on the active rows.
        b_gen = jnp.concatenate([rhs_dual.eq_multipliers, rhs_dual.ineq_multipliers])
        b_eff = ctx.effective_rhs(b_gen)
        h_fixed, _ = kkt_blocks(ctx.d_fixed, jnp.zeros((m,), dtype))
        rhs = jnp.concatenate([free_f * (rhs_p - h_fixed), b_eff])

        # Warm start: solve for the correction ``K δ = rhs - K z0``.
        z0 = jnp.concatenate(
            [
                free_f * ravel_pytree(x0[0])[0],
                active_f
                * jnp.concatenate([x0[1].eq_multipliers, x0[1].ineq_multipliers]),
            ]
        )
        r0 = rhs - matvec(z0)

        # Block-diagonal SPD preconditioner; identity on the eliminated
        # coordinates so the metric stays positive definite.
        def precond(z: Vector) -> Vector:
            zp, zd = z[:n], z[n:]
            v1 = ctx.apply_Minv(zp) + (1.0 - free_f) * zp
            v2 = ctx.solve_preconditioned_normal(active_f * zd) + (1.0 - active_f) * zd
            return jnp.concatenate([v1, v2])

        delta, minres_conv, n_minres = _pminres_qlp_solve(
            matvec, r0, tol=self.tol, max_iter=self.max_iter, precond=precond
        )
        z = z0 + delta
        d = free_f * z[:n] + ctx.d_fixed

        # --- M-metric feasibility projection + iterative refinement ---
        d, feas_res, n_ref, target = self._project_and_refine(ctx, d, b_gen, b_eff)

        # --- multipliers for the projected step ---
        primal_d = unravel_primal(d)
        dual = self.multiplier_recovery.recover(subproblem, ctx, primal_d)
        step = (primal_d, dual)
        res_primal, _ = subproblem.residual(step)
        projected_grad_norm = jnp.linalg.norm(free_f * ravel_pytree(res_primal)[0])

        # --- classify ---
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(step)])
        )
        feasible = feas_res <= target
        success = finite & minres_conv & feasible & ctx.converged
        status = RESULTS.where(
            finite,
            RESULTS.where(
                ctx.converged,
                RESULTS.where(
                    feasible,
                    RESULTS.where(
                        minres_conv, RESULTS.successful, RESULTS.max_steps_reached
                    ),
                    RESULTS.stagnation,
                ),
                RESULTS.max_steps_reached,
            ),
            RESULTS.singular,
        )
        reason = KKT_SOLVER_RESULTS.where(
            finite,
            KKT_SOLVER_RESULTS.where(
                ctx.converged,
                KKT_SOLVER_RESULTS.where(
                    feasible,
                    KKT_SOLVER_RESULTS.where(
                        minres_conv,
                        KKT_SOLVER_RESULTS.converged,
                        KKT_SOLVER_RESULTS.max_iter_reached,
                    ),
                    KKT_SOLVER_RESULTS.residual_floor,
                ),
                KKT_SOLVER_RESULTS.projector_failure,
            ),
            KKT_SOLVER_RESULTS.nonfinite,
        )
        res_dtype = initial_state.feasibility_residual.dtype
        new_state = cast(
            MinresQLPState,
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
                    initial_state.n_iter + n_minres + ctx.n_iter,
                    success,
                    status,
                    feas_res.astype(res_dtype),
                    n_ref,
                    projected_grad_norm.astype(res_dtype),
                    reason,
                    jnp.logical_not(finite),
                ),
            ),
        )
        return step, new_state

    def _project_and_refine(
        self,
        ctx: ProjectionContext,
        d: Float[Array, " n"],
        b_gen: Float[Array, " m"],
        b_eff: Float[Array, " m"],
    ) -> tuple[Float[Array, " n"], Scalar, Int[Array, ""], Scalar]:
        """Project ``d`` onto the working constraint and refine while it helps.

        Returns the projected step, its feasibility residual, the number of
        refinement rounds that ran and the residual target used.
        """
        dtype = d.dtype
        eps = jnp.finfo(dtype).eps
        d = ctx.preconditioned_feasibility_correction(d, b_gen)
        res = ctx.feasibility_residual(d, b_gen)
        b_norm_floor = jnp.linalg.norm(b_eff) + 1.0
        target = (
            jnp.asarray(self.proj_refine_atol, dtype)
            + jnp.asarray(self.proj_refine_rtol, dtype) * b_norm_floor
        )
        # Roundoff floor: ``A_work d - b_eff`` cannot be resolved below a few
        # units of ``eps · (‖A_work‖‖d‖ + ‖b_eff‖)``.
        roundoff = (
            8.0
            * eps
            * (jnp.linalg.norm(ctx.A_work) * jnp.linalg.norm(d) + b_norm_floor)
        )
        target = jnp.maximum(target, roundoff)

        def body(_i, carry):
            d_cur, res_cur, n_ref, done = carry

            def refine(c):
                d_cur, res_cur, n_ref, _ = c
                d_new = ctx.preconditioned_feasibility_correction(d_cur, b_gen)
                res_new = ctx.feasibility_residual(d_new, b_gen)
                # Keep the round only if it improved the residual; a round
                # that does not is the floor, so stop refining.
                better = res_new < res_cur
                d_out = jnp.where(better, d_new, d_cur)
                res_out = jnp.where(better, res_new, res_cur)
                return (d_out, res_out, n_ref + 1, ~better | (res_out <= target))

            return jax.lax.cond(done, lambda c: c, refine, carry)

        init = (d, res, jnp.zeros((), jnp.int32), res <= target)
        d, res, n_ref, _ = jax.lax.fori_loop(0, self.proj_refine_max_iter, body, init)
        return d, res, n_ref, target
