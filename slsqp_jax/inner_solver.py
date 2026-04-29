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
from slsqp_jax.utils import to_scalar

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

    # Coerce ``cg_tol`` to a true 0-d scalar.  Otherwise a ``(1,)``-shaped
    # tolerance (which can leak in via the adaptive Eisenstat-Walker path
    # if ``state.prev_grad_lagrangian`` ever carries an unexpected leading
    # axis) would broadcast every ``r_new_norm_sq < cg_tol ** 2`` comparison
    # below to shape ``(1,)``.  The boolean ``state.converged`` would then
    # latch onto that shape and the predicate fed to ``jax.lax.cond`` at
    # the bottom of this function would no longer be a scalar, raising
    # ``TypeError: Pred must be a scalar`` from deep inside JAX.
    cg_tol = to_scalar(cg_tol)

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

        # Defensive scalarisation of the predicate.  The CG state stores
        # ``converged: Bool[Array, ""]`` but a stale (1,)-shaped boolean
        # could otherwise propagate through ``fori_loop`` and trip the
        # scalar-predicate check inside ``cond``.
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
    cg_tol = to_scalar(cg_tol)
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
        converged=jnp.reshape(r0_norm_sq < cg_tol**2, ()),
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
        adaptive_tol: Scalar | float | None = None,
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
            adaptive_tol: Optional Eisenstat-Walker tolerance override.
                When provided, overrides the solver's default convergence
                tolerance for this call only.

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
        adaptive_tol: Scalar | float | None = None,
    ) -> InnerSolveResult:
        effective_tol = adaptive_tol if adaptive_tol is not None else self.cg_tol
        d, multipliers, converged = _solve_projected_cg_cholesky(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_cg_iter=self.max_cg_iter,
            cg_tol=effective_tol,
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
    cg_tol = to_scalar(cg_tol)

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
        converged=jnp.reshape(r0_norm_sq < cg_tol**2, ()),
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
    residual: Scalar  # |beta_{k+1} * s_k|  (||A x_k - rhs||)
    converged: Bool[Array, ""]
    breakdown: Bool[Array, ""]
    iteration: Int[Array, ""]


_CRAIG_BREAKDOWN_TOL = 1e-14


def craig_solve(
    A: Float[Array, "m n"],
    rhs: Float[Array, " m"],
    tol: float | Scalar = 1e-10,
    max_iter: int = 100,
) -> tuple[Vector, Bool[Array, ""]]:
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
        Tuple ``(x, converged)``.  ``converged`` is ``True`` only when the
        relative residual fell below ``tol``; it is ``False`` if CRAIG
        broke down (``alpha`` / ``beta`` below an absolute threshold,
        signalling rank deficiency or numerical collapse) or exhausted
        its iteration budget.  When ``converged`` is ``False`` the
        returned ``x`` is still the best iterate produced before the
        failure.
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
    # Guard against alpha1 ≈ 0 with beta1 != 0: ``s1`` gets amplified by
    # ``1 / 1e-30`` and the resulting ``x1`` is numerical garbage (often
    # catastrophically large).  When ``alpha1`` is below the breakdown
    # threshold the right answer is *no update from the CRAIG iterate*:
    # return ``x = 0`` so downstream consumers (projection, HVP) receive
    # a bounded vector instead of one that overflows.  We still signal
    # breakdown via ``init_breakdown`` so the outer solver knows CRAIG
    # did not actually solve the system.
    x1 = jnp.where(breakdown_init, jnp.zeros_like(x1_raw), x1_raw)

    Av1 = A @ v1
    u_hat = Av1 - alpha1 * u1
    beta2 = jnp.linalg.norm(u_hat)
    beta2_safe = jnp.maximum(beta2, 1e-30)
    u2 = u_hat / beta2_safe

    # If beta1 is already zero, rhs is zero and x=0 is exact.
    trivially_converged = beta1 < tol * jnp.maximum(beta1_safe, 1.0)
    residual_init = jnp.abs(beta2 * s1)
    init_converged = trivially_converged | (residual_init < tol * beta1_safe)
    # Breakdown: A^T rhs is (numerically) zero.  Only report a breakdown
    # when we would otherwise report non-convergence; a zero rhs is not
    # an error.
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
            # One step of re-orthogonalisation against v_k to keep the
            # short recurrence honest in the presence of floating-point
            # drift (partial reorthogonalisation).
            v_hat = v_hat - jnp.dot(state.v, v_hat) * state.v
            alpha_new = jnp.linalg.norm(v_hat)
            alpha_breakdown = alpha_new < _CRAIG_BREAKDOWN_TOL
            alpha_safe = jnp.maximum(alpha_new, 1e-30)
            v_new = v_hat / alpha_safe

            s_new = -state.beta * state.s / alpha_safe

            # Guard against ``alpha_breakdown``: if ``alpha_new`` is
            # below the breakdown threshold then ``s_new`` and ``v_new``
            # are both amplified by ``1 / 1e-30``.  The nominal update
            # ``state.x + s_new * v_new`` would therefore absorb the
            # ``1e60`` scaling and poison every downstream consumer
            # (including the HVP call inside the CG loop).  Stay at the
            # last safe iterate instead; the breakdown flag on the
            # returned state records the failure so the outer solver
            # can react.
            x_candidate = state.x + s_new * v_new
            x_new = jnp.where(alpha_breakdown, state.x, x_candidate)

            Av = A @ v_new
            u_hat = Av - alpha_new * state.u
            # Partial reorthogonalisation against u_{k+1} (state.u).
            u_hat = u_hat - jnp.dot(state.u, u_hat) * state.u
            beta_new = jnp.linalg.norm(u_hat)
            beta_breakdown = beta_new < _CRAIG_BREAKDOWN_TOL
            beta_safe = jnp.maximum(beta_new, 1e-30)
            u_new = u_hat / beta_safe

            residual_new = jnp.abs(beta_new * s_new)
            converged = residual_new < tol * beta1_safe

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
    # A "true" success requires the residual to be below tolerance at
    # termination and not having flagged a breakdown.
    success = (final.residual < tol * beta1_safe) & ~final.breakdown
    return final.x, success


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
    craig_tol: float = 1e-10
    craig_max_iter: int = 200
    # Multiplier recovery uses CG on the normal equations. Its tolerance
    # is kept tight (and independent of craig_tol) so the Lagrangian
    # residual is not polluted by imprecise multipliers, without forcing
    # the inner CRAIG projections to the same accuracy.
    mult_recovery_tol: float = 1e-12
    mult_recovery_max_iter: int = 200

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
        adaptive_tol: Scalar | float | None = None,
    ) -> InnerSolveResult:
        effective_tol = adaptive_tol if adaptive_tol is not None else self.cg_tol
        d, multipliers, converged = _solve_projected_cg_craig(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            max_cg_iter=self.max_cg_iter,
            cg_tol=effective_tol,
            precond_fn=precond_fn,
            cg_regularization=self.cg_regularization,
            free_mask=free_mask,
            d_fixed=d_fixed,
            use_constraint_preconditioner=self.use_constraint_preconditioner,
            craig_tol=self.craig_tol,
            craig_max_iter=self.craig_max_iter,
            mult_recovery_tol=self.mult_recovery_tol,
            mult_recovery_max_iter=self.mult_recovery_max_iter,
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
    craig_tol: float = 1e-10,
    craig_max_iter: int = 200,
    mult_recovery_tol: float = 1e-12,
    mult_recovery_max_iter: int = 200,
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
    cg_tol = to_scalar(cg_tol)

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

    # Particular solution: d_p = A^T (A A^T)^{-1} b = craig_solve(A, b).
    # Guard against a non-finite result (e.g. the rhs is non-zero but
    # the active Jacobian is numerically rank-deficient and CRAIG
    # breakdown-guarded back to ``x = 0``, or extreme magnitudes escaped
    # the breakdown detector): fall back to ``d_p = 0`` so the CG loop
    # at least starts from a finite iterate.
    d_p_free, d_p_craig_conv = craig_solve(
        A_work, b_work, tol=craig_tol, max_iter=craig_max_iter
    )
    d_p_free_finite = jnp.isfinite(d_p_free).all()
    d_p_free = jnp.where(d_p_free_finite, d_p_free, jnp.zeros_like(d_p_free))
    d_p = d_p_free + _dfixed if has_fixed else d_p_free

    # Projection: P(v) = v - A^T (A A^T)^{-1} A v = v - craig_solve(A, Av).
    # We deliberately do not propagate per-call CRAIG convergence from
    # inside the CG iteration (would require threading state into
    # ``_CGState``).  If CRAIG breaks down on the projection solve we
    # fall back to the identity on that component, which degrades the
    # projector to "no projection" for the problematic input but keeps
    # the CG iterates finite.  Without this guard, a breakdown-induced
    # non-finite ``x_proj`` would be subtracted from ``v_work`` and
    # poison every subsequent HVP/projection call.
    def project(v: Vector) -> Vector:
        v_work = _free * v if has_fixed else v
        x_proj, _ = craig_solve(
            A_work, A_work @ v_work, tol=craig_tol, max_iter=craig_max_iter
        )
        x_proj = jnp.where(jnp.isfinite(x_proj).all(), x_proj, jnp.zeros_like(x_proj))
        return v_work - x_proj

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

            w, _ = solve_unconstrained_cg(
                amat_hvp_reg, -AMr, mult_recovery_max_iter, mult_recovery_tol
            )
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
        converged=jnp.reshape(r0_norm_sq < cg_tol**2, ()),
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

    multipliers, mult_cg_converged = solve_unconstrained_cg(
        normal_hvp, -kkt_rhs, mult_recovery_max_iter, mult_recovery_tol
    )
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    finite_d = jnp.isfinite(final_cg.d).all()
    finite_mult = jnp.isfinite(multipliers).all()
    converged = (
        final_cg.converged & d_p_craig_conv & mult_cg_converged & finite_d & finite_mult
    )

    return final_cg.d, multipliers, converged


# ---------------------------------------------------------------------------
# MINRES-QLP solver  (Choi, Paige & Saunders, SIAM J. Sci. Comput. 2011)
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
            # state.betal is beta_{k-1} (zero at k=1); state.betan is beta_k.
            betal = state.betal
            beta = state.betan
            beta_safe = jnp.maximum(beta, 1e-30)
            betal_safe = jnp.maximum(betal, 1e-30)

            v = state.r3 / beta_safe
            r3_new = matvec(v)

            # Three-term subtraction (skip at k=1 via jnp.where)
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
            # k == 1: wl2=wl, wl=v*sr1, w=-v*cr1
            # k == 2: wl2=wl, wl=w*cr1+v*sr1, w=w*sr1-v*cr1
            # k > 2:  full P_{k-2,k} and P_{k-1,k} rotations
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
            # Lanczos breakdown: betan collapses either because the
            # system is solved (phi_new is small) or because of numerical
            # collapse.  In the former case we flag convergence; in the
            # latter we short-circuit to stop accumulating garbage.
            lanczos_breakdown = betan_new < 1e-30 * jnp.maximum(beta1_safe, 1.0)
            residual_small = jnp.abs(phi_new) < tol * beta1_safe
            converged = (relres < tol) | (lanczos_breakdown & residual_small)
            # If breakdown happened without the residual being small, we
            # stop the iteration by setting ``converged`` true so the
            # `jax.lax.cond` guard short-circuits subsequent calls; the
            # final success flag below will see relres >= tol and report
            # failure to the caller.
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
    # Report genuine success only when the last residual dropped below
    # tolerance; breakdown or budget exhaustion returns converged=False
    # so the QP layer triggers its fallback paths.
    final_relres = jnp.abs(final.phi) / (
        jnp.maximum(final.Anorm, 1e-30) * jnp.maximum(jnp.linalg.norm(final.x), 1e-30)
        + beta1_safe
    )
    success = (final_relres < tol) & jnp.isfinite(final.x).all()
    return final.x, success


# ---------------------------------------------------------------------------
# Strategy 3: MINRES-QLP on the full KKT system
# ---------------------------------------------------------------------------


class MinresQLPSolver(AbstractInnerSolver):
    """Preconditioned MINRES-QLP on the full saddle-point KKT system.

    Solves the KKT system directly::

        [B    A^T] [d]       [-g]
        [A    0  ] [lambda] = [b ]

    using PMINRES-QLP (Choi, Paige & Saunders, SISC 2011, Table 3.5)
    with a block-diagonal SPD preconditioner::

        M = [B_diag^{-1}    0      ]
            [0              S^{-1} ]

    where ``B_diag = diag(B_0)`` (L-BFGS diagonal) and
    ``S = A B_diag^{-1} A^T`` is the Schur complement.
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
        adaptive_tol: Scalar | float | None = None,
    ) -> InnerSolveResult:
        # MINRES-QLP solves the full KKT system where constraints A d = b
        # are part of the linear system. Loosening the tolerance would
        # degrade constraint satisfaction (unlike null-space CG where
        # constraints are enforced by the projector). Keep self.tol.
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
    tol: float | Scalar,
    precond_fn: Callable[[Vector], Vector] | None = None,
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
) -> tuple[Vector, Float[Array, " m"], Bool[Array, ""]]:
    """Solve equality-constrained QP via PMINRES-QLP on the full KKT system.

    The KKT system::

        [B    A^T] [d]       [-g]
        [A    0  ] [lambda] = [b ]

    is symmetric indefinite.  PMINRES-QLP solves it directly, producing
    both d and the Lagrange multipliers lambda.

    Uses a block-diagonal SPD preconditioner ``M^{-1} = diag(B_diag^{-1},
    S^{-1})`` where ``B_diag^{-1}`` is the user-supplied preconditioner
    (typically L-BFGS inverse Hessian diagonal) and ``S = A B_diag^{-1}
    A^T`` is the Schur complement.  This satisfies the SPD requirement
    from Choi (2006, Section 3.4).
    """
    n = g.shape[0]
    m = A.shape[0]
    has_fixed = free_mask is not None and d_fixed is not None
    tol = to_scalar(tol)

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

    # Block-diagonal SPD preconditioner (Section 3.4 of Choi 2006)
    if precond_fn is not None:
        _raw_precond = precond_fn
        reg_diag = jnp.where(active_mask, 0.0, 1.0)

        # When free_mask is active, the KKT system has zero rows/columns
        # at fixed-variable positions.  The L-BFGS inverse Hessian has
        # cross-coupling between all variables, so applying it unmasked
        # leaks non-zero values into those zero dimensions, contaminating
        # the preconditioned Lanczos vectors and degrading MINRES-QLP
        # convergence (particularly constraint satisfaction).  Masking
        # the primal block to the free subspace eliminates the leakage.
        if has_fixed:
            _free_f = _free.astype(g.dtype)

            def _primal_precond(v: Vector) -> Vector:
                return _free_f * _raw_precond(_free_f * v)
        else:
            _primal_precond = _raw_precond

        # Schur complement S = A M^{-1} A^T  (m x m, SPD)
        M_AT = jax.vmap(_primal_precond)(A_work).T  # (n, m)
        A_M_AT = A_work @ M_AT + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_M_AT_chol = jnp.linalg.cholesky(A_M_AT)

        def _solve_schur(rhs_s: Float[Array, " m"]) -> Float[Array, " m"]:
            return jax.scipy.linalg.cho_solve((A_M_AT_chol, True), rhs_s)

        def kkt_precond(z: Vector) -> Vector:
            r1 = z[:n]
            r2 = z[n:]
            v1 = _primal_precond(r1)
            v2 = _solve_schur(r2)
            return jnp.concatenate([v1, v2])

        solution, converged = pminres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter, precond=kkt_precond
        )
    else:
        solution, converged = pminres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter
        )

    d = solution[:n]
    if has_fixed:
        # Force the direction to respect the fixed mask.  The KKT matvec
        # and preconditioner branches already keep fixed positions at
        # zero; this final projection guards against floating-point
        # drift and preconditioner implementations that may leak small
        # cross-coupling through the free/fixed boundary.
        d = _free * d + _dfixed
    # KKT system uses L = ... + λ^T(Ad - b), but the active-set QP
    # solver uses L = ... - μ^T(Ad - b) with μ >= 0, so μ = -λ.
    multipliers = -solution[n:]
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    finite = jnp.isfinite(d).all() & jnp.isfinite(multipliers).all()
    return d, multipliers, converged & finite
