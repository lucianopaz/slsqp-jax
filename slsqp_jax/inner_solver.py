"""Inner solvers for the equality-constrained QP subproblem.

This module provides pluggable strategies for solving the inner
equality-constrained QP that arises at each active-set iteration:

    minimize    (1/2) d^T B d + g^T d
    subject to  A[active] d = b[active]
                d[i] = d_fixed[i]  for i where free_mask[i] is False

Three strategies are available:

- ``ProjectedCGCholesky``: Projected conjugate gradient with Cholesky-based
  null-space projection.  This is the original implementation.
- ``ProjectedCGCraig`` (planned): Projected CG with CRAIG-based iterative
  projection, eliminating the Cholesky factorization.
- ``MinresQLP`` (planned): MINRES-QLP on the full KKT system, eliminating
  the need for explicit projection entirely.

All strategies implement the ``AbstractInnerSolver`` interface and return
an ``InnerSolveResult`` with the same shape and semantics.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.types import Scalar, Vector

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class InnerSolveResult(NamedTuple):
    """Result from an inner equality-constrained QP solve."""

    d: Vector
    multipliers: Float[Array, " m"]
    converged: Bool[Array, ""]


# ---------------------------------------------------------------------------
# CG internals (shared by CG-based strategies)
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
):
    """Build a CG step function.

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        cg_tol: Convergence tolerance on residual norm.
        precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
        project: Optional projection function v -> P(v) where P is the
            projection onto the null space of A.
        cg_regularization: Minimum eigenvalue threshold for the curvature
            guard.

    Returns:
        A CG step function.
    """
    if project is None:

        def project(v: Vector) -> Vector:
            return v

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

            converged = (r_new_norm_sq < cg_tol**2) | has_bad_curvature

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

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    return cg_step


def solve_unconstrained_cg(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
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

    Returns:
        Tuple of (d, converged) where d is the solution vector and
        converged indicates whether CG converged (residual below
        tolerance) as opposed to hitting bad curvature or exhausting
        the iteration budget.
    """
    # Sign convention: we define the residual as r = b - Ax = -g - Bd
    # (the "negative residual"), whereas Nocedal & Wright Algorithm 5.3 uses
    # r = Ax - b = Bd + g. So r_here = -r_NW throughout. The sign flip
    # propagates to z (via the preconditioner) but cancels in the scalar
    # products that define alpha, beta, and the search direction p, so those
    # quantities are identical to the textbook. See the summary table in
    # cg_sign_convention_analysis for the full derivation.
    n = g.shape[0]
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
        converged=r0_norm_sq < cg_tol**2,
    )

    cg_step = build_cg_step(
        hvp_fn=hvp_fn,
        cg_tol=cg_tol,
        precond_fn=precond_fn,
        cg_regularization=cg_regularization,
    )

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)
    return final_cg.d, final_cg.converged


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractInnerSolver(eqx.Module):
    """Strategy for solving the equality-constrained QP subproblem.

    Subclasses implement ``solve`` to compute the search direction ``d``
    and Lagrange multipliers for the active constraints.
    """

    @abc.abstractmethod
    def solve(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> InnerSolveResult:
        """Solve the equality-constrained QP subproblem.

        Solves::

            minimize    (1/2) d^T B d + g^T d
            subject to  A[active] d = b[active]
                        d[i] = d_fixed[i]  for i where free_mask[i] is False

        where B is given implicitly via ``hvp_fn(v) = B @ v``.

        Args:
            hvp_fn: Hessian-vector product function v -> B @ v.
            g: Linear term (gradient of objective).
            A: Combined constraint matrix (m x n).
            b: Combined RHS vector (m,).
            active_mask: Boolean mask (m,) indicating active constraints.
            precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
            free_mask: Optional boolean mask (n,).  When provided, only
                variables with ``free_mask[i] = True`` are optimized.
            d_fixed: Values for fixed variables (n,).  Required when
                ``free_mask`` is provided.

        Returns:
            ``InnerSolveResult`` with the direction, multipliers, and
            convergence flag.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Strategy 1: Projected CG with Cholesky-based null-space projection
# ---------------------------------------------------------------------------


class ProjectedCGCholesky(AbstractInnerSolver):
    """Projected CG with Cholesky-based null-space projection.

    This is the original implementation: Cholesky-factor ``A A^T`` (with
    regularization), use it for the null-space projector and particular
    solution, run CG in the null space, and recover multipliers via
    iterative refinement.

    When ``use_constraint_preconditioner`` is ``True`` and a
    preconditioner is provided, the constraint preconditioner
    (Gould, Hribar & Nocedal, 2001) is used instead of the naive
    ``P(M(r))``.
    """

    max_cg_iter: int
    cg_tol: Scalar | float
    cg_regularization: float = 1e-6
    use_constraint_preconditioner: bool = False

    def solve(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> InnerSolveResult:
        d, multipliers, converged = _solve_projected_cg_cholesky(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_cg_iter=self.max_cg_iter,
            cg_tol=self.cg_tol,
            precond_fn=precond_fn,
            cg_regularization=self.cg_regularization,
            free_mask=free_mask,
            d_fixed=d_fixed,
            use_constraint_preconditioner=self.use_constraint_preconditioner,
        )
        return InnerSolveResult(d=d, multipliers=multipliers, converged=converged)


# ---------------------------------------------------------------------------
# Implementation: projected CG with Cholesky projection
# ---------------------------------------------------------------------------


def _solve_projected_cg_cholesky(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
    use_constraint_preconditioner: bool = False,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Solve equality-constrained QP using projected (preconditioned) CG.

    Solves:
        minimize    (1/2) d^T B d + g^T d
        subject to  A[active] d = b[active]
                    d[i] = d_fixed[i]   for i where free_mask[i] is False

    where B is given implicitly via hvp_fn(v) = B @ v.

    The method:
    1. Computes a particular solution d_p satisfying A d_p = b for
       active constraints (with fixed variables at their values).
    2. Defines the projection P(v) = v - A^T (A A^T)^{-1} A v onto
       the null space of A (for active constraints only, in the
       free-variable subspace when ``free_mask`` is provided).
    3. Runs CG on the reduced problem in the null space of A.
    4. Recovers Lagrange multipliers from the KKT conditions.

    When ``free_mask`` and ``d_fixed`` are provided, bound-active
    variables are fixed at their values.  The CG operates only on
    free variables: the HVP is masked (zeroing fixed components),
    the projection uses only free columns of A, and the effective
    gradient accounts for the fixed-variable contribution.  This
    reduces the effective constraint matrix size from
    ``(m_eq + m_gen + m_bounds)`` to ``(m_eq + m_gen)``, dropping
    the projection cost from ``O(m_total^3)`` to ``O(m_core^3)``
    per CG step.

    When ``use_constraint_preconditioner`` is ``True`` and a
    preconditioner is provided, the constraint preconditioner
    (Gould, Hribar & Nocedal, 2001) is used instead of the naive
    ``P(M(r))``.  This solves the saddle-point system to produce
    ``z = Mr - M Aᵀ (A M Aᵀ)⁻¹ A M r``, preserving the
    ``M⁻¹``-inner product in null(A).  This is essential when the
    CG system matrix differs from ``M⁻¹`` (e.g. exact Hessian HVP
    with L-BFGS preconditioner), where the naive formulation
    destroys preconditioning quality.

    When ``use_constraint_preconditioner`` is ``False`` (default),
    the simpler projected preconditioner ``z = P(M(P(r)))`` is used.
    This works well when the CG system matrix is the L-BFGS
    approximation itself (so ``M B ≈ I``).

    Args:
        hvp_fn: Hessian-vector product function v -> B @ v.
        g: Linear term (gradient of objective).
        A: Combined constraint matrix (m x n), rows are constraint normals.
        b: Combined RHS vector (m,).
        active_mask: Boolean mask (m,) indicating which constraints are
            active.
        max_cg_iter: Maximum CG iterations.
        cg_tol: CG convergence tolerance.
        precond_fn: Optional preconditioner v -> M @ v where M ~ B^{-1}.
        cg_regularization: Minimum eigenvalue threshold for the curvature
            guard.  Based on SNOPT Section 4.5.
        free_mask: Optional boolean mask (n,). When provided, only
            variables with ``free_mask[i] = True`` are optimized;
            the rest are fixed at ``d_fixed[i]``.
        d_fixed: Values for fixed variables (n,).  Required when
            ``free_mask`` is provided.
        use_constraint_preconditioner: When ``True`` and a preconditioner
            is provided, use the Gould-Hribar-Nocedal constraint
            preconditioner instead of the naive ``P(M(r))``.

    Returns:
        Tuple of (d, multipliers, cg_converged) where d is the solution,
        multipliers is a vector of Lagrange multipliers for all m
        constraints (0 for inactive), and cg_converged indicates whether
        the inner CG solver converged (residual below tolerance) as
        opposed to hitting bad curvature or exhausting the iteration
        budget.
    """
    m = A.shape[0]
    has_fixed = free_mask is not None and d_fixed is not None

    A_masked = jnp.where(active_mask[:, None], A, 0.0)
    b_masked = jnp.where(active_mask, b, 0.0)

    if has_fixed and free_mask is not None and d_fixed is not None:
        A_work = A_masked * free_mask[None, :]
        b_work = b_masked - A_masked @ d_fixed
    else:
        A_work = A_masked
        b_work = b_masked

    reg_diag = jnp.where(active_mask, 0.0, 1.0)
    AAt = A_work @ A_work.T + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
    AAt_chol = jnp.linalg.cholesky(AAt)

    def solve_AAt(rhs: Float[Array, " m"]) -> Float[Array, " m"]:
        return jax.scipy.linalg.cho_solve((AAt_chol, True), rhs)

    _free: Bool[Array, " n"] = (
        free_mask if free_mask is not None else jnp.ones(A.shape[1], dtype=bool)
    )
    _dfixed: Vector = d_fixed if d_fixed is not None else jnp.zeros(A.shape[1])
    d_p_free = A_work.T @ solve_AAt(b_work)
    d_p = d_p_free + _dfixed if has_fixed else d_p_free

    def project(v: Vector) -> Vector:
        v_work = _free * v if has_fixed else v
        return v_work - A_work.T @ solve_AAt(A_work @ v_work)

    # Constraint preconditioner (Gould, Hribar & Nocedal, 2001).
    if precond_fn is not None and use_constraint_preconditioner:
        _raw_precond = precond_fn
        M_AT = jax.vmap(_raw_precond)(A_work).T  # (n, m)
        A_M_AT = A_work @ M_AT + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_M_AT_chol = jnp.linalg.cholesky(A_M_AT)

        def _solve_AMAT(rhs: Float[Array, " m"]) -> Float[Array, " m"]:
            return jax.scipy.linalg.cho_solve((A_M_AT_chol, True), rhs)

        def _constraint_precond(r: Vector) -> Vector:
            Mr = _raw_precond(r)
            w = _solve_AMAT(A_work @ Mr)
            return Mr - M_AT @ w

        effective_precond: Callable[[Vector], Vector] | None = _constraint_precond
    else:
        effective_precond = precond_fn

    if has_fixed:

        def hvp_work(v: Vector) -> Vector:
            return _free * hvp_fn(_free * v)

        g_eff = _free * (g + hvp_fn(_dfixed))
    else:
        hvp_work = hvp_fn
        g_eff = g

    Bd_p = hvp_work(d_p)
    r0 = project(-(g_eff + Bd_p))
    r0_norm_sq = jnp.dot(r0, r0)

    if effective_precond is not None:
        z0 = project(effective_precond(r0))
        rz0_raw = jnp.dot(r0, z0)
        z0 = jnp.where(rz0_raw > 0, z0, r0)
        rz0 = jnp.where(rz0_raw > 0, rz0_raw, r0_norm_sq)
        p0 = z0
    else:
        rz0 = r0_norm_sq
        p0 = r0

    init_cg = _CGState(
        d=d_p,
        r=r0,
        p=p0,
        rz=rz0,
        iteration=jnp.array(0),
        converged=r0_norm_sq < cg_tol**2,
    )

    cg_step = build_cg_step(
        hvp_fn=hvp_work,
        cg_tol=cg_tol,
        precond_fn=effective_precond,
        project=project,
        cg_regularization=cg_regularization,
    )

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)

    # Multiplier recovery using the full (unmasked) HVP
    Bd = hvp_fn(final_cg.d)
    kkt_residual = A_work @ (Bd + g)
    multipliers = solve_AAt(kkt_residual)
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    # Iterative refinement: the regularized Cholesky (AAt + eps*I) introduces
    # O(eps * cond(AAt)) error in the multipliers.  One refinement step squares
    # the relative error, e.g. from ~1e-5 to ~1e-10 for cond ~ 1e3.
    grad_L_qp = Bd + g - A_work.T @ multipliers
    refinement_rhs = A_work @ grad_L_qp
    delta_mult = solve_AAt(refinement_rhs)
    multipliers = multipliers + delta_mult
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    return final_cg.d, multipliers, final_cg.converged


# ---------------------------------------------------------------------------
# CRAIG solver (Golub-Kahan bidiagonalization)
# ---------------------------------------------------------------------------


class _CraigState(NamedTuple):
    """Internal state for the CRAIG iterative solver."""

    x: Vector  # primal solution (n,): x_k = A^T (A A^T)^{-1} rhs
    s: Scalar  # coefficient: s_k = (-1)^{k-1} prod(beta_i/alpha_i)
    u: Float[Array, " m"]  # left bidiag vector (m,)
    v: Vector  # right bidiag vector (n,)
    alpha: Scalar  # current alpha
    beta: Scalar  # current beta (beta_{k+1})
    converged: Bool[Array, ""]
    iteration: Int[Array, ""]


def craig_solve(
    A: Float[Array, "m n"],
    rhs: Float[Array, " m"],
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Vector:
    """Solve ``min ||x||  s.t.  A x = rhs`` via CRAIG's method.

    CRAIG's method (Paige & Saunders, 1982) uses Golub-Kahan
    bidiagonalization to solve the minimum-norm problem without forming
    ``A A^T``.  Only matrix-vector products ``A @ v`` and ``A.T @ u``
    are needed.

    The returned ``x`` satisfies ``x = A^T (A A^T)^{-1} rhs``, i.e.
    it is the minimum-norm solution of ``A x = rhs``.  For rank-deficient
    ``A``, the method naturally converges to the pseudoinverse solution.

    This is used as a drop-in replacement for the combined operation
    ``A.T @ cho_solve(chol(A A^T), rhs)`` in the null-space projection:

    - Particular solution: ``d_p = craig_solve(A, b)``
    - Projection: ``P(v) = v - craig_solve(A, A @ v)``

    Args:
        A: Matrix (m x n).
        rhs: Right-hand side (m,).
        tol: Convergence tolerance on ``||A x - rhs|| / ||rhs||``.
        max_iter: Maximum bidiagonalization steps.

    Returns:
        Minimum-norm solution x (n,).
    """
    m, n = A.shape

    # Initialize: beta_1 u_1 = rhs
    beta1 = jnp.linalg.norm(rhs)
    beta1_safe = jnp.maximum(beta1, 1e-30)
    u1 = rhs / beta1_safe

    # alpha_1 v_1 = A^T u_1
    Atu1 = A.T @ u1
    alpha1 = jnp.linalg.norm(Atu1)
    alpha1_safe = jnp.maximum(alpha1, 1e-30)
    v1 = Atu1 / alpha1_safe

    # First CRAIG iterate: x_1 = s_1 v_1 where s_1 = beta_1 / alpha_1
    s1 = beta1 / alpha1_safe
    x1 = s1 * v1

    # Advance bidiag: beta_2 u_2 = A v_1 - alpha_1 u_1
    Av1 = A @ v1
    u_hat = Av1 - alpha1 * u1
    beta2 = jnp.linalg.norm(u_hat)
    beta2_safe = jnp.maximum(beta2, 1e-30)
    u2 = u_hat / beta2_safe

    # Residual: ||A x_k - rhs|| = |beta_{k+1} * s_k|
    init_converged = (beta1 < tol) | (jnp.abs(beta2 * s1) < tol * beta1_safe)

    init_state = _CraigState(
        x=x1,
        s=s1,
        u=u2,
        v=v1,
        alpha=alpha1,
        beta=beta2,
        converged=init_converged,
        iteration=jnp.array(1),
    )

    def craig_step(i, state: _CraigState) -> _CraigState:
        def do_step(state: _CraigState) -> _CraigState:
            # Bidiag: alpha_{k+1} v_{k+1} = A^T u_{k+1} - beta_{k+1} v_k
            Atu = A.T @ state.u
            v_hat = Atu - state.beta * state.v
            alpha_new = jnp.linalg.norm(v_hat)
            alpha_safe = jnp.maximum(alpha_new, 1e-30)
            v_new = v_hat / alpha_safe

            # CRAIG coefficient: s_{k+1} = -beta_{k+1} * s_k / alpha_{k+1}
            s_new = -state.beta * state.s / alpha_safe

            # Update solution: x_{k+1} = x_k + s_{k+1} v_{k+1}
            x_new = state.x + s_new * v_new

            # Bidiag: beta_{k+2} u_{k+2} = A v_{k+1} - alpha_{k+1} u_{k+1}
            Av = A @ v_new
            u_hat = Av - alpha_new * state.u
            beta_new = jnp.linalg.norm(u_hat)
            beta_safe = jnp.maximum(beta_new, 1e-30)
            u_new = u_hat / beta_safe

            # Residual: ||A x_{k+1} - rhs|| = |beta_{k+2} * s_{k+1}|
            conv = jnp.abs(beta_new * s_new) < tol * beta1_safe

            return _CraigState(
                x=x_new,
                s=s_new,
                u=u_new,
                v=v_new,
                alpha=alpha_new,
                beta=beta_new,
                converged=conv,
                iteration=state.iteration + 1,
            )

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final = jax.lax.fori_loop(0, max_iter, craig_step, init_state)
    return final.x


# ---------------------------------------------------------------------------
# Strategy 2: Projected CG with CRAIG-based iterative projection
# ---------------------------------------------------------------------------


class ProjectedCGCraig(AbstractInnerSolver):
    """Projected CG with CRAIG-based iterative null-space projection.

    Replaces the Cholesky factorization of ``A A^T`` with iterative
    CRAIG solves (Golub-Kahan bidiagonalization).  This eliminates the
    ``O(m^3)`` factorization cost and the ``1e-8`` diagonal
    regularization, at the cost of an iterative solve per projection.

    For multiplier recovery (done once after the CG loop), CG on the
    normal equations ``A A^T y = rhs`` is used, reusing the existing
    ``solve_unconstrained_cg`` infrastructure.
    """

    max_cg_iter: int
    cg_tol: Scalar | float
    cg_regularization: float = 1e-6
    use_constraint_preconditioner: bool = False
    craig_tol: float = 1e-12
    craig_max_iter: int = 200

    def solve(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> InnerSolveResult:
        d, multipliers, converged = _solve_projected_cg_craig(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_cg_iter=self.max_cg_iter,
            cg_tol=self.cg_tol,
            precond_fn=precond_fn,
            cg_regularization=self.cg_regularization,
            free_mask=free_mask,
            d_fixed=d_fixed,
            use_constraint_preconditioner=self.use_constraint_preconditioner,
            craig_tol=self.craig_tol,
            craig_max_iter=self.craig_max_iter,
        )
        return InnerSolveResult(d=d, multipliers=multipliers, converged=converged)


# ---------------------------------------------------------------------------
# Implementation: projected CG with CRAIG projection
# ---------------------------------------------------------------------------


def _solve_projected_cg_craig(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_cg_iter: int,
    cg_tol: Scalar | float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    cg_regularization: float = 1e-6,
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
    use_constraint_preconditioner: bool = False,
    craig_tol: float = 1e-12,
    craig_max_iter: int = 200,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Projected CG using CRAIG for the null-space projection.

    Same interface and semantics as ``_solve_projected_cg_cholesky``
    but replaces the Cholesky factorization of ``A A^T`` with iterative
    CRAIG solves.  No regularization is needed.

    For multiplier recovery, CG on the normal equations is used since
    CRAIG naturally returns the primal (n-dim) rather than the dual
    (m-dim) solution.
    """
    has_fixed = free_mask is not None and d_fixed is not None

    A_masked = jnp.where(active_mask[:, None], A, 0.0)
    b_masked = jnp.where(active_mask, b, 0.0)

    if has_fixed and free_mask is not None and d_fixed is not None:
        A_work = A_masked * free_mask[None, :]
        b_work = b_masked - A_masked @ d_fixed
    else:
        A_work = A_masked
        b_work = b_masked

    _free: Bool[Array, " n"] = (
        free_mask if free_mask is not None else jnp.ones(A.shape[1], dtype=bool)
    )
    _dfixed: Vector = d_fixed if d_fixed is not None else jnp.zeros(A.shape[1])

    # Particular solution: d_p = A^T (A A^T)^{-1} b = craig_solve(A, b)
    d_p_free = craig_solve(A_work, b_work, tol=craig_tol, max_iter=craig_max_iter)
    d_p = d_p_free + _dfixed if has_fixed else d_p_free

    # Projection: P(v) = v - A^T (A A^T)^{-1} A v = v - craig_solve(A, Av)
    def project(v: Vector) -> Vector:
        v_work = _free * v if has_fixed else v
        return v_work - craig_solve(
            A_work, A_work @ v_work, tol=craig_tol, max_iter=craig_max_iter
        )

    # Constraint preconditioner (Gould, Hribar & Nocedal, 2001).
    # With CRAIG, the A M A^T system is also solved iteratively.
    if precond_fn is not None and use_constraint_preconditioner:
        _raw_precond = precond_fn

        def _constraint_precond(r: Vector) -> Vector:
            Mr = _raw_precond(r)
            # Solve A M A^T w = A M r iteratively.
            # Define operator B = A M^{1/2} (conceptually); then
            # B B^T w = A M r.  We use CRAIG on the composed operator.
            AMr = A_work @ Mr

            # CG on normal equations A M A^T w = A M r:
            # HVP: v -> A M A^T v = A_work @ (M(A_work^T @ v))
            def amat_hvp(v: Float[Array, " m"]) -> Float[Array, " m"]:
                return A_work @ _raw_precond(A_work.T @ v)

            reg_diag = jnp.where(active_mask, 0.0, 1.0)

            def amat_hvp_reg(v: Float[Array, " m"]) -> Float[Array, " m"]:
                return amat_hvp(v) + reg_diag * v

            w, _ = solve_unconstrained_cg(amat_hvp_reg, -AMr, craig_max_iter, craig_tol)
            return Mr - _raw_precond(A_work.T @ w)

        effective_precond: Callable[[Vector], Vector] | None = _constraint_precond
    else:
        effective_precond = precond_fn

    if has_fixed:

        def hvp_work(v: Vector) -> Vector:
            return _free * hvp_fn(_free * v)

        g_eff = _free * (g + hvp_fn(_dfixed))
    else:
        hvp_work = hvp_fn
        g_eff = g

    Bd_p = hvp_work(d_p)
    r0 = project(-(g_eff + Bd_p))
    r0_norm_sq = jnp.dot(r0, r0)

    if effective_precond is not None:
        z0 = project(effective_precond(r0))
        rz0_raw = jnp.dot(r0, z0)
        z0 = jnp.where(rz0_raw > 0, z0, r0)
        rz0 = jnp.where(rz0_raw > 0, rz0_raw, r0_norm_sq)
        p0 = z0
    else:
        rz0 = r0_norm_sq
        p0 = r0

    init_cg = _CGState(
        d=d_p,
        r=r0,
        p=p0,
        rz=rz0,
        iteration=jnp.array(0),
        converged=r0_norm_sq < cg_tol**2,
    )

    cg_step = build_cg_step(
        hvp_fn=hvp_work,
        cg_tol=cg_tol,
        precond_fn=effective_precond,
        project=project,
        cg_regularization=cg_regularization,
    )

    final_cg = jax.lax.fori_loop(0, max_cg_iter, cg_step, init_cg)

    # Multiplier recovery via CG on normal equations A A^T λ = A(Bd+g).
    # This avoids forming A A^T; each CG step uses A @ (A^T @ v).
    Bd = hvp_fn(final_cg.d)
    kkt_rhs = A_work @ (Bd + g)
    reg_diag = jnp.where(active_mask, 0.0, 1.0)

    def normal_hvp(v: Float[Array, " m"]) -> Float[Array, " m"]:
        return A_work @ (A_work.T @ v) + reg_diag * v

    multipliers, _ = solve_unconstrained_cg(
        normal_hvp, -kkt_rhs, craig_max_iter, craig_tol
    )
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    return final_cg.d, multipliers, final_cg.converged


# ---------------------------------------------------------------------------
# MINRES-QLP solver
# ---------------------------------------------------------------------------


class _MinresQLPState(NamedTuple):
    """Internal state for the MINRES-QLP iteration.

    Tracks the symmetric Lanczos process (v, z vectors), the QR
    factorization of the tridiagonal (Givens rotations), and the
    three-term solution update (w vectors).
    """

    v_prev: Vector  # Lanczos vector v_{k-1}
    v_curr: Vector  # Lanczos vector v_k
    z_prev: Vector  # preconditioned z_{k-1} (= v_{k-1} if no precond)
    z_curr: Vector  # preconditioned z_k (= v_k if no precond)

    beta: Scalar  # beta from Lanczos (0 at init, then beta_{k+1})

    # Two most recent Givens rotations for the QR factorization
    cs_prev: Scalar  # cs from rotation k-2 (init: 1.0 = identity)
    sn_prev: Scalar  # sn from rotation k-2 (init: 0.0 = identity)
    cs_curr: Scalar  # cs from rotation k-1 (init: 1.0 = identity)
    sn_curr: Scalar  # sn from rotation k-1 (init: 0.0 = identity)

    # Solution update vectors (three-term recurrence for w)
    w_prev: Vector  # w_{k-2}
    w_curr: Vector  # w_{k-1}

    x: Vector  # current solution
    phi: Scalar  # residual norm (|phi_k| = ||K x_k - rhs||)

    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def _givens_rotation(a: Scalar, b: Scalar) -> tuple[Scalar, Scalar, Scalar]:
    """Compute Givens rotation to zero out b in [a; b].

    Returns (cs, sn, r) where:
        [cs  sn] [a]   [r]
        [-sn cs] [b] = [0]
    """
    r = jnp.sqrt(a**2 + b**2)
    r_safe = jnp.maximum(r, 1e-30)
    cs = a / r_safe
    sn = b / r_safe
    return cs, sn, r


def minres_qlp_solve(
    matvec: Callable[[Vector], Vector],
    rhs: Vector,
    tol: float = 1e-10,
    max_iter: int = 200,
    precond: Callable[[Vector], Vector] | None = None,
) -> tuple[Vector, Bool[Array, ""]]:
    """Solve a symmetric (possibly indefinite) system Kx = rhs via MINRES.

    Uses the symmetric Lanczos process to build a tridiagonal T_k, then
    applies Givens rotations for residual minimization (Paige & Saunders,
    1975).  Handles indefinite systems without breakdown.

    For preconditioned MINRES, the Lanczos process operates on the
    preconditioned operator ``M^{-1} K`` in the ``M``-inner product.

    Args:
        matvec: Symmetric operator v -> K @ v.
        rhs: Right-hand side vector.
        tol: Convergence tolerance on ``||K x - rhs|| / ||rhs||``.
        max_iter: Maximum Lanczos iterations.
        precond: Optional SPD preconditioner v -> M^{-1} @ v.

    Returns:
        Tuple of (x, converged).
    """
    n = rhs.shape[0]
    rhs_norm = jnp.linalg.norm(rhs)
    rhs_norm_safe = jnp.maximum(rhs_norm, 1e-30)

    # Lanczos init: beta_1 v_1 = rhs
    if precond is not None:
        z0 = precond(rhs)
        beta1 = jnp.sqrt(jnp.abs(jnp.dot(rhs, z0)))
    else:
        z0 = rhs
        beta1 = jnp.linalg.norm(rhs)

    beta1_safe = jnp.maximum(beta1, 1e-30)
    v1 = rhs / beta1_safe
    z1 = z0 / beta1_safe

    zeros = jnp.zeros(n)

    init_state = _MinresQLPState(
        v_prev=zeros,
        v_curr=v1,
        z_prev=zeros,
        z_curr=z1,
        beta=jnp.array(0.0),  # no sub-diagonal for the 1st column
        cs_prev=jnp.array(1.0),  # identity rotation
        sn_prev=jnp.array(0.0),
        cs_curr=jnp.array(1.0),  # identity rotation
        sn_curr=jnp.array(0.0),
        w_prev=zeros,
        w_curr=zeros,
        x=zeros,
        phi=beta1,
        iteration=jnp.array(0),
        converged=beta1 < tol * rhs_norm_safe,
    )

    def minres_step(i, state: _MinresQLPState) -> _MinresQLPState:
        def do_step(state: _MinresQLPState) -> _MinresQLPState:
            # --- Lanczos step ---
            Kz = matvec(state.z_curr)  # K z_k (z_k = M^{-1} v_k)
            alpha = jnp.dot(state.v_curr, Kz)

            # v_next_raw = K z_k - alpha v_k - beta v_{k-1}
            v_next_raw = Kz - alpha * state.v_curr - state.beta * state.v_prev

            if precond is not None:
                z_next_raw = precond(v_next_raw)
                beta_next = jnp.sqrt(jnp.abs(jnp.dot(v_next_raw, z_next_raw)))
            else:
                z_next_raw = v_next_raw
                beta_next = jnp.linalg.norm(v_next_raw)

            beta_next_safe = jnp.maximum(beta_next, 1e-30)
            v_next = v_next_raw / beta_next_safe
            z_next = z_next_raw / beta_next_safe

            # --- QR factorization of the new tridiagonal column ---
            # Column k of the extended tridiagonal:
            #   [0, ..., 0, beta_k, alpha_k, beta_{k+1}]
            #
            # Apply G_{k-2} (rows k-2, k-1) to [0; beta_k]:
            eps_k = state.sn_prev * state.beta
            delta_hat = state.cs_prev * state.beta
            #
            # Apply G_{k-1} (rows k-1, k) to [delta_hat; alpha]:
            delta = state.cs_curr * delta_hat + state.sn_curr * alpha
            gamma_hat = -state.sn_curr * delta_hat + state.cs_curr * alpha
            #
            # New rotation G_k to zero out beta_{k+1}:
            cs_new, sn_new, gamma = _givens_rotation(gamma_hat, beta_next)

            # --- Residual and solution update ---
            # Apply G_k to the transformed RHS: [phi_{k-1}; 0]
            tau_k = cs_new * state.phi
            phi_new = -sn_new * state.phi

            # Solution update vector (three-term recurrence)
            gamma_safe = jnp.where(jnp.abs(gamma) > 1e-30, gamma, 1e-30)
            w_new = (
                state.z_curr - delta * state.w_curr - eps_k * state.w_prev
            ) / gamma_safe

            x_new = state.x + tau_k * w_new

            converged = jnp.abs(phi_new) < tol * rhs_norm_safe

            return _MinresQLPState(
                v_prev=state.v_curr,
                v_curr=v_next,
                z_prev=state.z_curr,
                z_curr=z_next,
                beta=beta_next,
                cs_prev=state.cs_curr,
                sn_prev=state.sn_curr,
                cs_curr=cs_new,
                sn_curr=sn_new,
                w_prev=state.w_curr,
                w_curr=w_new,
                x=x_new,
                phi=phi_new,
                iteration=state.iteration + 1,
                converged=converged,
            )

        return jax.lax.cond(state.converged, lambda s: s, do_step, state)

    final = jax.lax.fori_loop(0, max_iter, minres_step, init_state)
    return final.x, final.converged


# ---------------------------------------------------------------------------
# Strategy 3: MINRES-QLP on the full KKT system
# ---------------------------------------------------------------------------


class MinresQLPSolver(AbstractInnerSolver):
    """MINRES-QLP on the full saddle-point KKT system.

    Solves the KKT system directly::

        [B    A^T] [d]       [-g]
        [A    0  ] [lambda] = [b ]

    using MINRES-QLP with an optional constraint preconditioner.
    This eliminates the need for explicit null-space projection,
    particular solution computation, and multiplier recovery.

    The constraint preconditioner uses the Schur complement
    ``A G^{-1} A^T`` where ``G^{-1}`` is the user-supplied
    preconditioner (e.g. L-BFGS inverse Hessian).
    """

    max_iter: int = 200
    tol: float = 1e-10
    max_cg_iter: int = 50

    def solve(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> InnerSolveResult:
        d, multipliers, converged = _solve_kkt_minres_qlp(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_iter=self.max_iter,
            tol=self.tol,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )
        return InnerSolveResult(d=d, multipliers=multipliers, converged=converged)


def _solve_kkt_minres_qlp(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    max_iter: int,
    tol: float,
    precond_fn: Callable[[Vector], Vector] | None = None,
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Solve equality-constrained QP via MINRES-QLP on the full KKT system.

    The KKT system::

        [B    A^T] [d]       [-g]
        [A    0  ] [lambda] = [b ]

    is symmetric indefinite.  MINRES-QLP solves it directly, producing
    both d and the Lagrange multipliers lambda.
    """
    n = g.shape[0]
    m = A.shape[0]
    has_fixed = free_mask is not None and d_fixed is not None

    A_masked = jnp.where(active_mask[:, None], A, 0.0)
    b_masked = jnp.where(active_mask, b, 0.0)

    if has_fixed and free_mask is not None and d_fixed is not None:
        A_work = A_masked * free_mask[None, :]
        b_work = b_masked - A_masked @ d_fixed
    else:
        A_work = A_masked
        b_work = b_masked

    _free: Bool[Array, " n"] = (
        free_mask if free_mask is not None else jnp.ones(n, dtype=bool)
    )
    _dfixed: Vector = d_fixed if d_fixed is not None else jnp.zeros(n)

    if has_fixed:

        def hvp_work(v: Vector) -> Vector:
            return _free * hvp_fn(_free * v)

        g_eff = _free * (g + hvp_fn(_dfixed))
    else:
        hvp_work = hvp_fn
        g_eff = g

    # KKT operator on (n+m)-dimensional vectors [d; lambda]
    def kkt_matvec(z: Vector) -> Vector:
        d_part = z[:n]
        lam_part = z[n:]
        top = hvp_work(d_part) + A_work.T @ lam_part
        bot = A_work @ d_part
        return jnp.concatenate([top, bot])

    # RHS: [-g_eff; b_work]
    kkt_rhs = jnp.concatenate([-g_eff, b_work])

    # Constraint preconditioner (Schur complement based)
    if precond_fn is not None:
        _raw_precond = precond_fn
        reg_diag = jnp.where(active_mask, 0.0, 1.0)

        M_AT = jax.vmap(_raw_precond)(A_work).T  # (n, m)
        A_M_AT = A_work @ M_AT + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_M_AT_chol = jnp.linalg.cholesky(A_M_AT)

        def _solve_schur(rhs_s: Float[Array, " m"]) -> Float[Array, " m"]:
            return jax.scipy.linalg.cho_solve((A_M_AT_chol, True), rhs_s)

        def kkt_precond(z: Vector) -> Vector:
            r1 = z[:n]
            r2 = z[n:]
            Mr1 = _raw_precond(r1)
            schur_rhs = A_work @ Mr1 - r2
            w = _solve_schur(schur_rhs)
            v = Mr1 - M_AT @ w
            return jnp.concatenate([v, w])

        solution, converged = minres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter, precond=kkt_precond
        )
    else:
        solution, converged = minres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter
        )

    d = solution[:n]
    if has_fixed:
        d = d + _dfixed
    # KKT system uses L = ... + λ^T(Ad - b), but the active-set QP
    # solver uses L = ... - μ^T(Ad - b) with μ >= 0, so μ = -λ.
    multipliers = -solution[n:]
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    return d, multipliers, converged
