"""Krylov primitives shared by the inner solvers.

Hosts the building blocks used by every projector-based and saddle-point
inner solver:

* (Preconditioned, optionally projected) conjugate gradient via
  :func:`build_cg_step` and the unconstrained driver
  :func:`solve_unconstrained_cg`.
* CRAIG's method for ``min ||x|| s.t. A x = rhs`` via :func:`craig_solve`.
* Stable symmetric Givens rotation :func:`_sym_ortho` and the full
  Preconditioned MINRES-QLP iteration via :func:`pminres_qlp_solve`.

These were previously defined inline inside ``slsqp_jax/inner_solver.py``
mixed with the solver classes.  Splitting them out keeps each Krylov
recurrence in a single, navigable module and makes the high-level
solver classes easier to read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.types import Scalar, Vector
from slsqp_jax.utils import to_scalar

# ---------------------------------------------------------------------------
# (Preconditioned, projected) conjugate gradient
# ---------------------------------------------------------------------------


class _CGState(NamedTuple):
    """Internal state for the (preconditioned) conjugate gradient solver.

    When a preconditioner M is used, ``rz`` stores r^T z (where z = M r)
    instead of r^T r, and ``p`` is built from z rather than r.
    """

    d: Vector
    r: Vector
    p: Vector
    rz: Scalar  # r^T z (preconditioned) or r^T r (unpreconditioned)
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def build_cg_step(
    hvp_fn,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    project: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
    cg_atol: float | None = None,
):
    """Build a CG step function.

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        cg_tol: Convergence tolerance on residual norm (absolute when
            ``cg_atol`` is ``None``; otherwise the squared per-step test
            is ``r^T r < max(cg_atol**2, cg_tol**2)`` so the larger of
            the two acts as the effective floor).
        precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
        project: Optional projection function v -> P(v) where P is the
            projection onto the null space of A.
        cg_regularization: Minimum eigenvalue threshold for the curvature
            guard.
        cg_atol: Optional absolute residual-norm floor.  When provided,
            convergence is declared at ``r^T r < max(cg_atol**2,
            cg_tol**2)``.  Used by the multiplier-recovery CG inside
            ``ProjectedCGCraig`` so that a near-KKT iterate does not
            chase a relative target below ``eps``.  Defaults to ``None``
            (current behaviour: pure absolute test ``r^T r < cg_tol**2``).

    Returns:
        A CG step function.
    """
    if project is None:

        def project(v: Vector) -> Vector:
            return v

    # Coerce ``cg_tol`` to a true 0-d scalar.  See the long comment in
    # the legacy implementation for why this matters under JIT tracing.
    cg_tol = to_scalar(cg_tol)
    if cg_atol is not None:
        tol_sq: Scalar | float = jnp.maximum(cg_atol**2, cg_tol**2)
    else:
        tol_sq = cg_tol**2

    def cg_step(i, state):
        def do_step(state: _CGState) -> _CGState:
            Bp = hvp_fn(state.p)
            PBp = project(Bp)
            pPBp = jnp.dot(state.p, PBp)

            # SNOPT-style curvature guard (see unconstrained CG for detail).
            # For projected CG, p is in the null space, so this checks the
            # effective eigenvalue of the reduced Hessian Z^T B Z along p.
            pp = jnp.dot(state.p, state.p)
            has_bad_curvature = pPBp <= cg_regularization * pp

            alpha = jnp.where(
                has_bad_curvature,
                jnp.array(0.0),
                state.rz / jnp.maximum(pPBp, 1e-30),
            )

            d_new = state.d + alpha * state.p
            r_new = state.r - alpha * PBp
            r_new_norm_sq = jnp.dot(r_new, r_new)

            if precond_fn is not None:
                z_new_raw = project(precond_fn(r_new))
                rz_raw = jnp.dot(r_new, z_new_raw)
                z_new = jnp.where(rz_raw > 0, z_new_raw, r_new)
                rz_new = jnp.where(rz_raw > 0, rz_raw, r_new_norm_sq)
            else:
                z_new = r_new
                rz_new = r_new_norm_sq

            beta = rz_new / jnp.maximum(state.rz, 1e-30)
            p_new = z_new + beta * state.p

            converged = (r_new_norm_sq < tol_sq) | has_bad_curvature

            return jax.lax.cond(
                has_bad_curvature,
                lambda: _CGState(
                    d=state.d,
                    r=state.r,
                    p=state.p,
                    rz=state.rz,
                    iteration=state.iteration + 1,
                    converged=jnp.array(True),
                ),
                lambda: _CGState(
                    d=d_new,
                    r=r_new,
                    p=p_new,
                    rz=rz_new,
                    iteration=state.iteration + 1,
                    converged=converged,
                ),
            )

        # Defensive scalarisation of the predicate.
        converged_pred = jnp.reshape(state.converged, ())
        return jax.lax.cond(converged_pred, lambda s: s, do_step, state)

    return cg_step


def solve_unconstrained_cg(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
    cg_atol: float | None = None,
) -> tuple[Vector, Bool[Array, ""]]:
    """Solve the unconstrained QP: min (1/2) d^T B d + g^T d.

    Uses (preconditioned) conjugate gradient to solve B d = -g without
    forming B.  When *precond_fn* is provided, the standard PCG algorithm
    is used: z = M r, beta = r_new^T z_new / r_old^T z_old, and p is
    built from z (Nocedal & Wright, Algorithm 5.3).

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        g: Linear term (gradient).
        max_cg_iter: Maximum CG iterations.
        cg_tol: Convergence tolerance on residual norm.
        precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
        cg_regularization: Minimum eigenvalue threshold for the curvature
            guard.  CG declares "bad curvature" when the effective
            eigenvalue ``p^T B p / ||p||^2`` falls below this value.
            Based on SNOPT Section 4.5 (Gill, Murray & Saunders, 2005).
        cg_atol: Optional absolute residual-norm floor.  When provided,
            the per-step convergence test becomes
            ``r^T r < max(cg_atol**2, cg_tol**2)`` so an absolute floor
            kicks in whenever the user-supplied ``cg_tol`` would target
            a residual below the floor.

    Returns:
        Tuple of (d, converged) where d is the solution vector and
        converged indicates whether CG converged (residual below
        tolerance) as opposed to hitting bad curvature or exhausting
        the iteration budget.
    """
    # Sign convention: r = b - Ax = -g - Bd (descent residual).
    n = g.shape[0]
    cg_tol = to_scalar(cg_tol)
    if cg_atol is not None:
        init_tol_sq: Scalar | float = jnp.maximum(cg_atol**2, cg_tol**2)
    else:
        init_tol_sq = cg_tol**2
    r0 = -g
    r0_norm_sq = jnp.dot(r0, r0)

    if precond_fn is not None:
        z0 = precond_fn(r0)
        rz0_raw = jnp.dot(r0, z0)
        z0 = jnp.where(rz0_raw > 0, z0, r0)
        rz0 = jnp.where(rz0_raw > 0, rz0_raw, r0_norm_sq)
        p0 = z0
    else:
        rz0 = r0_norm_sq
        p0 = r0

    init_cg = _CGState(
        d=jnp.zeros(n),
        r=r0,
        p=p0,
        rz=rz0,
        iteration=jnp.array(0),
        converged=jnp.reshape(r0_norm_sq < init_tol_sq, ()),
    )

    cg_step = build_cg_step(
        hvp_fn=hvp_fn,
        cg_tol=cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
        cg_atol=cg_atol,
    )

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)
    return final_cg.d, final_cg.converged


# ---------------------------------------------------------------------------
# CRAIG (Golub-Kahan bidiagonalization)
# ---------------------------------------------------------------------------


class _CraigState(NamedTuple):
    """Internal state for the CRAIG iterative solver."""

    x: Vector  # primal solution (n,): x_k = A^T (A A^T)^{-1} rhs
    s: Scalar  # coefficient: s_k = (-1)^{k-1} prod(beta_i/alpha_i)
    u: Float[Array, " m"]  # left bidiag vector (m,)
    v: Vector  # right bidiag vector (n,)
    alpha: Scalar  # current alpha
    beta: Scalar  # current beta (beta_{k+1})
    residual: Scalar  # |beta_{k+1} * s_k|  (||A x_k - rhs||)
    converged: Bool[Array, ""]
    breakdown: Bool[Array, ""]
    iteration: Int[Array, ""]


_CRAIG_BREAKDOWN_TOL = 1e-14
# Hardcoded absolute residual floor for the CRAIG convergence test.  Together
# with the user-tunable relative ``tol``, the test becomes
# ``residual < max(_CRAIG_TOL_ABS, tol * ||rhs||)``.  Without this floor a
# near-KKT iterate with ``||rhs|| ~ 1e-13`` would target ``tol * 1e-13``
# (below ``eps``) and never converge, even though the actual residual is
# already at machine precision.
_CRAIG_TOL_ABS = 1e-12


def craig_solve(
    A: Float[Array, "m n"],
    rhs: Float[Array, " m"],
    tol: float | Scalar = 1e-10,
    max_iter: int = 100,
) -> tuple[Vector, Bool[Array, ""]]:
    """Solve ``min ||x|| s.t. A x = rhs`` via CRAIG's method.

    CRAIG's method (Paige & Saunders, 1982) uses Golub-Kahan
    bidiagonalization to solve the minimum-norm problem without forming
    ``A A^T``.  Only matrix-vector products ``A @ v`` and ``A.T @ u``
    are needed.

    Convergence test is hybrid absolute+relative:
    ``residual < max(_CRAIG_TOL_ABS=1e-12, tol * ||rhs||)``.  The
    absolute floor protects against the pathology where ``||rhs||``
    is itself near machine epsilon (e.g. when projecting at a near-KKT
    iterate where ``A v`` shrinks at the rate the SQP is converging),
    in which case the pure-relative target ``tol * ||rhs||`` would
    drop below ``eps`` and convergence could never fire.

    Args:
        A: Matrix (m x n).
        rhs: Right-hand side (m,).
        tol: Relative tolerance on ``||A x - rhs|| / ||rhs||``.  The
            effective convergence threshold also has a hardcoded
            absolute floor of ``1e-12``.
        max_iter: Maximum bidiagonalization steps.

    Returns:
        Tuple ``(x, converged)``.  ``converged`` is ``True`` only when
        the residual fell below the hybrid threshold; it is ``False``
        if CRAIG broke down (``alpha`` / ``beta`` below an absolute
        threshold, signalling rank deficiency or numerical collapse)
        or exhausted its iteration budget.  When ``converged`` is
        ``False`` the returned ``x`` is still the best iterate produced
        before the failure.
    """
    m, n = A.shape

    beta1 = jnp.linalg.norm(rhs)
    beta1_safe = jnp.maximum(beta1, 1e-30)
    u1 = rhs / beta1_safe

    Atu1 = A.T @ u1
    alpha1 = jnp.linalg.norm(Atu1)
    breakdown_init = alpha1 < _CRAIG_BREAKDOWN_TOL
    alpha1_safe = jnp.maximum(alpha1, 1e-30)
    v1 = Atu1 / alpha1_safe

    s1 = beta1 / alpha1_safe
    x1_raw = s1 * v1
    # Guard against alpha1 ≈ 0 with beta1 != 0; see legacy comment.
    x1 = jnp.where(breakdown_init, jnp.zeros_like(x1_raw), x1_raw)

    Av1 = A @ v1
    u_hat = Av1 - alpha1 * u1
    beta2 = jnp.linalg.norm(u_hat)
    beta2_safe = jnp.maximum(beta2, 1e-30)
    u2 = u_hat / beta2_safe

    # If beta1 is already zero, rhs is zero and x=0 is exact.
    trivially_converged = beta1 < tol * jnp.maximum(beta1_safe, 1.0)
    residual_init = jnp.abs(beta2 * s1)
    init_threshold = jnp.maximum(_CRAIG_TOL_ABS, tol * beta1_safe)
    init_converged = trivially_converged | (residual_init < init_threshold)
    init_breakdown = breakdown_init & ~trivially_converged

    init_state = _CraigState(
        x=x1,
        s=s1,
        u=u2,
        v=v1,
        alpha=alpha1,
        beta=beta2,
        residual=residual_init,
        converged=init_converged | init_breakdown,
        breakdown=init_breakdown,
        iteration=jnp.array(1),
    )

    def craig_step(i, state: _CraigState) -> _CraigState:
        def do_step(state: _CraigState) -> _CraigState:
            Atu = A.T @ state.u
            v_hat = Atu - state.beta * state.v
            v_hat = v_hat - jnp.dot(state.v, v_hat) * state.v
            alpha_new = jnp.linalg.norm(v_hat)
            alpha_breakdown = alpha_new < _CRAIG_BREAKDOWN_TOL
            alpha_safe = jnp.maximum(alpha_new, 1e-30)
            v_new = v_hat / alpha_safe

            s_new = -state.beta * state.s / alpha_safe

            # Stay at the last safe iterate when alpha breaks down.
            x_candidate = state.x + s_new * v_new
            x_new = jnp.where(alpha_breakdown, state.x, x_candidate)

            Av = A @ v_new
            u_hat = Av - alpha_new * state.u
            u_hat = u_hat - jnp.dot(state.u, u_hat) * state.u
            beta_new = jnp.linalg.norm(u_hat)
            beta_breakdown = beta_new < _CRAIG_BREAKDOWN_TOL
            beta_safe = jnp.maximum(beta_new, 1e-30)
            u_new = u_hat / beta_safe

            residual_new = jnp.abs(beta_new * s_new)
            step_threshold = jnp.maximum(_CRAIG_TOL_ABS, tol * beta1_safe)
            converged = residual_new < step_threshold

            broke = alpha_breakdown | beta_breakdown
            done = converged | broke

            return _CraigState(
                x=x_new,
                s=s_new,
                u=u_new,
                v=v_new,
                alpha=alpha_new,
                beta=beta_new,
                residual=residual_new,
                converged=done,
                breakdown=state.breakdown | (broke & ~converged),
                iteration=state.iteration + 1,
            )

        return jax.lax.cond(
            jnp.reshape(state.converged, ()), lambda s: s, do_step, state
        )

    final = jax.lax.fori_loop(0, max_iter, craig_step, init_state)
    final_threshold = jnp.maximum(_CRAIG_TOL_ABS, tol * beta1_safe)
    success = (final.residual < final_threshold) & ~final.breakdown
    return final.x, success


# ---------------------------------------------------------------------------
# Stable symmetric Givens rotation (used by MINRES-QLP)
# ---------------------------------------------------------------------------


def _sym_ortho(a: Scalar, b: Scalar) -> tuple[Scalar, Scalar, Scalar]:
    """Numerically stable symmetric Givens rotation (SymOrtho).

    Computes (c, s, r) such that r = sqrt(a^2 + b^2) >= 0, c = a/r,
    s = b/r.  Handles a=0, b=0, and |a|>|b| vs |b|>|a| separately
    to avoid overflow/underflow.

    Reference: Choi (2006), Table 2.9 / Algorithm 937 (TOMS 2014).
    """
    abs_a = jnp.abs(a)
    abs_b = jnp.abs(b)

    def _b_zero(_: None) -> tuple[Scalar, Scalar, Scalar]:
        c = jnp.where(a == 0.0, 1.0, jnp.sign(a))
        return c, jnp.array(0.0), abs_a

    def _a_zero(_: None) -> tuple[Scalar, Scalar, Scalar]:
        return jnp.array(0.0), jnp.sign(b), abs_b

    def _both_nonzero(_: None) -> tuple[Scalar, Scalar, Scalar]:
        def _a_ge_b(_: None) -> tuple[Scalar, Scalar, Scalar]:
            t = b / a
            r_local = abs_a * jnp.sqrt(1.0 + t * t)
            c_local = jnp.sign(a) / jnp.sqrt(1.0 + t * t)
            s_local = c_local * t
            return c_local, s_local, r_local

        def _b_gt_a(_: None) -> tuple[Scalar, Scalar, Scalar]:
            t = a / b
            r_local = abs_b * jnp.sqrt(1.0 + t * t)
            s_local = jnp.sign(b) / jnp.sqrt(1.0 + t * t)
            c_local = s_local * t
            return c_local, s_local, r_local

        return jax.lax.cond(abs_a >= abs_b, _a_ge_b, _b_gt_a, None)

    return jax.lax.cond(
        b == 0.0,
        _b_zero,
        lambda _: jax.lax.cond(a == 0.0, _a_zero, _both_nonzero, None),
        None,
    )


# ---------------------------------------------------------------------------
# Preconditioned MINRES-QLP (Choi, Paige & Saunders, SISC 2011)
# ---------------------------------------------------------------------------


class _PMinresQLPState(NamedTuple):
    """Internal state for the Preconditioned MINRES-QLP iteration.

    Follows the reference implementation by Choi, Paige & Saunders.
    Variables use the same names as the reference code for traceability.
    """

    # Lanczos vectors (raw, NOT normalized by beta)
    r1: Vector
    r2: Vector
    r3: Vector

    # Betas: previous (betal) and current (betan)
    betal: Scalar
    betan: Scalar

    # Left rotation (previous)
    cs: Scalar
    sn: Scalar

    # Right rotation P_{k-2,k}
    cr2: Scalar
    sr2: Scalar

    # QR/QLP intermediates
    dltan: Scalar
    eplnn: Scalar
    gama: Scalar
    gamal: Scalar
    gamal2: Scalar

    # Eta / vepln (for mu recurrence)
    eta: Scalar
    etal: Scalar
    etal2: Scalar
    vepln: Scalar
    veplnl: Scalar
    veplnl2: Scalar

    # Tau (for mu recurrence)
    tau: Scalar
    taul: Scalar

    # Mu / u coefficients
    u: Scalar
    ul: Scalar
    ul2: Scalar
    ul3: Scalar

    # w-vectors and solution
    w: Vector
    wl: Vector
    x: Vector
    xl2: Vector

    # Residual and norms
    phi: Scalar
    xl2norm: Scalar
    Anorm: Scalar
    gmin: Scalar
    gminl: Scalar

    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def pminres_qlp_solve(
    matvec: Callable[[Vector], Vector],
    rhs: Vector,
    tol: float | Scalar = 1e-10,
    max_iter: int = 200,
    precond: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, Bool[Array, ""]]:
    """Solve a symmetric (possibly indefinite/singular) system Ax = b.

    Implements the full Preconditioned MINRES-QLP algorithm (Table 3.5
    of Choi, Paige & Saunders, SIAM J. Sci. Comput. 33(4), 2011).

    All iterations use QLP mode (equivalent to TranCond=1 in the
    reference implementation).

    Args:
        matvec: Symmetric operator v -> A @ v.
        rhs: Right-hand side vector.
        tol: Convergence tolerance on relative residual.
        max_iter: Maximum Lanczos iterations.
        precond: Optional SPD preconditioner v -> M^{-1} @ v.

    Returns:
        Tuple of (x, converged).
    """
    n = rhs.shape[0]
    r2 = rhs
    if precond is None:
        r3 = r2
        beta1 = jnp.linalg.norm(r2)
    else:
        r3 = precond(r2)
        beta1 = jnp.sqrt(jnp.maximum(jnp.dot(r2, r3), 0.0))

    beta1_safe = jnp.maximum(beta1, 1e-30)
    zeros = jnp.zeros(n)

    init_state = _PMinresQLPState(
        r1=zeros,
        r2=r2,
        r3=r3,
        betal=jnp.array(0.0),
        betan=beta1,
        cs=jnp.array(-1.0),
        sn=jnp.array(0.0),
        cr2=jnp.array(-1.0),
        sr2=jnp.array(0.0),
        dltan=jnp.array(0.0),
        eplnn=jnp.array(0.0),
        gama=jnp.array(0.0),
        gamal=jnp.array(0.0),
        gamal2=jnp.array(0.0),
        eta=jnp.array(0.0),
        etal=jnp.array(0.0),
        etal2=jnp.array(0.0),
        vepln=jnp.array(0.0),
        veplnl=jnp.array(0.0),
        veplnl2=jnp.array(0.0),
        tau=jnp.array(0.0),
        taul=jnp.array(0.0),
        u=jnp.array(0.0),
        ul=jnp.array(0.0),
        ul2=jnp.array(0.0),
        ul3=jnp.array(0.0),
        w=zeros,
        wl=zeros,
        x=zeros,
        xl2=zeros,
        phi=beta1,
        xl2norm=jnp.array(0.0),
        Anorm=jnp.array(0.0),
        gmin=jnp.array(0.0),
        gminl=jnp.array(0.0),
        iteration=jnp.array(0),
        converged=beta1 < 1e-30,
    )

    def step_fn(_i: int, state: _PMinresQLPState) -> _PMinresQLPState:
        def do_step(state: _PMinresQLPState) -> _PMinresQLPState:
            k = state.iteration + 1  # 1-based iteration count

            # --- Lanczos step ---
            betal = state.betal
            beta = state.betan
            beta_safe = jnp.maximum(beta, 1e-30)
            betal_safe = jnp.maximum(betal, 1e-30)

            v = state.r3 / beta_safe
            r3_new = matvec(v)

            r3_new = r3_new - jnp.where(
                k > 1,
                state.r1 * (beta / betal_safe),
                zeros,
            )

            alfa = jnp.dot(r3_new, v)
            r3_new = r3_new - state.r2 * (alfa / beta_safe)

            r1_new = state.r2
            r2_new = r3_new

            if precond is None:
                betan_new = jnp.linalg.norm(r3_new)
            else:
                r3_new = precond(r2_new)
                betan_new = jnp.sqrt(jnp.maximum(jnp.dot(r2_new, r3_new), 0.0))

            pnorm = jnp.sqrt(betal**2 + alfa**2 + betan_new**2)

            # --- Previous left rotation Q_{k-1} ---
            dbar = state.dltan
            dlta = state.cs * dbar + state.sn * alfa
            gbar = state.sn * dbar - state.cs * alfa
            eplnn_new = state.sn * betan_new
            dltan_new = -state.cs * betan_new

            # --- Current left rotation Q_k ---
            gamal2 = state.gamal
            gamal = state.gama
            cs_new, sn_new, gama_new = _sym_ortho(gbar, betan_new)
            taul2 = state.taul
            taul_new = state.tau
            tau_new = cs_new * state.phi
            phi_new = sn_new * state.phi

            # --- Previous right rotation P_{k-2,k} (active when k > 2) ---
            veplnl2 = state.veplnl
            etal2 = state.etal
            etal_new = state.eta
            dlta_tmp = state.sr2 * state.vepln - state.cr2 * dlta
            veplnl_new = state.cr2 * state.vepln + state.sr2 * dlta
            dlta_k2 = jnp.where(k > 2, dlta_tmp, dlta)
            veplnl_new = jnp.where(k > 2, veplnl_new, state.veplnl)
            etal_new = jnp.where(k > 2, etal_new, state.etal)
            eta_new = jnp.where(k > 2, state.sr2 * gama_new, jnp.array(0.0))
            gama_k2 = jnp.where(k > 2, -state.cr2 * gama_new, gama_new)

            # --- Current right rotation P_{k-1,k} (active when k > 1) ---
            cr1_new, sr1_new, gamal_new = _sym_ortho(gamal, dlta_k2)
            cr1_new = jnp.where(k > 1, cr1_new, jnp.array(-1.0))
            sr1_new = jnp.where(k > 1, sr1_new, jnp.array(0.0))
            gamal_new = jnp.where(k > 1, gamal_new, gamal)
            vepln_new = jnp.where(k > 1, sr1_new * gama_k2, jnp.array(0.0))
            gama_final = jnp.where(k > 1, -cr1_new * gama_k2, gama_k2)

            # --- Update mu coefficients ---
            ul4 = state.ul3
            ul3_new = state.ul2

            gamal2_safe = jnp.where(jnp.abs(gamal2) > 1e-30, gamal2, 1e-30)
            ul2_new = jnp.where(
                k > 2,
                (taul2 - etal2 * ul4 - veplnl2 * ul3_new) / gamal2_safe,
                state.ul2,
            )

            gamal_safe = jnp.where(jnp.abs(gamal_new) > 1e-30, gamal_new, 1e-30)
            ul_new = jnp.where(
                k > 1,
                (taul_new - etal_new * ul3_new - veplnl_new * ul2_new) / gamal_safe,
                state.ul,
            )

            gama_safe = jnp.where(jnp.abs(gama_final) > 1e-30, gama_final, 1e-30)
            u_new = jnp.where(
                jnp.abs(gama_final) > 1e-30,
                (tau_new - eta_new * ul2_new - vepln_new * ul_new) / gama_safe,
                jnp.array(0.0),
            )

            xl2norm_new = jnp.sqrt(state.xl2norm**2 + ul2_new**2)

            # --- Update w-vectors and solution (QLP mode) ---
            w_old = state.w
            wl_old = state.wl

            # k > 2 path (general case)
            wl2_g = wl_old
            wl_g = w_old
            w_g = wl2_g * state.sr2 - v * state.cr2
            wl2_g = wl2_g * state.cr2 + v * state.sr2
            v_tmp = wl_g * cr1_new + w_g * sr1_new
            w_g = wl_g * sr1_new - w_g * cr1_new
            wl_g = v_tmp

            # k == 2 path
            wl2_2 = wl_old
            wl_2 = w_old * cr1_new + v * sr1_new
            w_2 = w_old * sr1_new - v * cr1_new

            # k == 1 path
            wl2_1 = wl_old
            wl_1 = v * sr1_new
            w_1 = -v * cr1_new

            wl2_out = jnp.where(k > 2, wl2_g, jnp.where(k == 2, wl2_2, wl2_1))
            wl_out = jnp.where(k > 2, wl_g, jnp.where(k == 2, wl_2, wl_1))
            w_out = jnp.where(k > 2, w_g, jnp.where(k == 2, w_2, w_1))

            xl2_new = state.xl2 + wl2_out * ul2_new
            x_new = xl2_new + wl_out * ul_new + w_out * u_new

            # --- Next right rotation P_{k-1,k+1} (for next iter) ---
            cr2_new, sr2_new, gamal_store = _sym_ortho(gamal_new, eplnn_new)

            # --- Update norms and condition estimate ---
            abs_gama = jnp.abs(gama_final)
            Anorm_new = jnp.maximum(state.Anorm, pnorm)
            Anorm_new = jnp.maximum(Anorm_new, gamal_new)
            Anorm_new = jnp.maximum(Anorm_new, abs_gama)

            gminl_new = jnp.where(k == 1, gama_final, state.gmin)
            gmin_new = jnp.where(
                k == 1,
                gama_final,
                jnp.minimum(
                    jnp.minimum(state.gminl, gamal_new),
                    abs_gama,
                ),
            )

            # --- Convergence check ---
            xnorm = jnp.sqrt(xl2norm_new**2 + ul_new**2 + u_new**2)
            relres = jnp.abs(phi_new) / (
                Anorm_new * jnp.maximum(xnorm, 1e-30) + beta1_safe
            )
            lanczos_breakdown = betan_new < 1e-30 * jnp.maximum(beta1_safe, 1.0)
            residual_small = jnp.abs(phi_new) < tol * beta1_safe
            converged = (relres < tol) | (lanczos_breakdown & residual_small)
            stop_now = converged | lanczos_breakdown

            return _PMinresQLPState(
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
                iteration=state.iteration + 1,
                converged=stop_now,
            )

        return jax.lax.cond(
            jnp.reshape(state.converged, ()), lambda s: s, do_step, state
        )

    final = jax.lax.fori_loop(0, max_iter, step_fn, init_state)
    final_relres = jnp.abs(final.phi) / (
        jnp.maximum(final.Anorm, 1e-30) * jnp.maximum(jnp.linalg.norm(final.x), 1e-30)
        + beta1_safe
    )
    success = (final_relres < tol) & jnp.isfinite(final.x).all()
    return final.x, success


__all__ = [
    "_CGState",
    "_CRAIG_BREAKDOWN_TOL",
    "_CRAIG_TOL_ABS",
    "_CraigState",
    "_PMinresQLPState",
    "_sym_ortho",
    "build_cg_step",
    "craig_solve",
    "pminres_qlp_solve",
    "solve_unconstrained_cg",
]
