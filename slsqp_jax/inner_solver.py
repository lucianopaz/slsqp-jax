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
    """Result from an inner equality-constrained QP solve.

    Attributes:
        d: Search direction.
        multipliers: Lagrange multipliers (shape ``(m,)``; entries for
            inactive constraints are zero).
        converged: True when the inner Krylov / projection iteration
            satisfied its tolerance.
        proj_residual: Post-solve constraint residual ``||A d - b||``
            (Euclidean norm, restricted to active rows).  Always ``0`` for
            null-space solvers (CG / CRAIG) where feasibility is enforced
            structurally; non-zero for ``MinresQLPSolver`` where it
            reflects the floor of the M-metric range-space projection
            after iterative refinement.
        n_proj_refinements: Number of M-metric projection refinement
            rounds actually applied.  Always ``0`` for null-space
            solvers.  At most ``MinresQLPSolver.proj_refine_max_iter``.
    """

    d: Vector
    multipliers: Float[Array, " m"]
    converged: Bool[Array, ""]
    proj_residual: Scalar = jnp.asarray(0.0)
    n_proj_refinements: Int[Array, ""] = jnp.asarray(0)
    # Norm of the *projected* initial gradient ``W̃_k g`` that the inner
    # solver actually iterated against (HR 2014, Theorem 3.5).  This is
    # the noise-aware stationarity proxy: when the outer SQP enables
    # ``use_inexact_stationarity``, the run is allowed to converge once
    # this value drops below ``rtol * max(|L|, 1)``.  Defaults to ``inf``
    # so that solvers which do not produce this quantity (i.e. anything
    # other than ``HRInexactSTCG``) cannot accidentally satisfy a
    # ``< rtol`` test even if the user toggles the flag — the inexact
    # path silently degrades to "never converges this way".
    projected_grad_norm: Scalar = jnp.asarray(jnp.inf)


# ---------------------------------------------------------------------------
# Projection context (shared infrastructure for projector-based solvers)
# ---------------------------------------------------------------------------


class ProjectionContext(NamedTuple):
    """Reusable bundle of projector + particular-solution + multiplier-recovery
    closures for an active equality system ``A_active d = b_active``.

    Existing strategies (``ProjectedCGCholesky``, ``ProjectedCGCraig``) build
    these inline inside their ``solve`` methods.  Composed strategies (e.g.
    ``HRInexactSTCG``) call ``inner.build_projection_context(...)`` to reuse
    the projector and multiplier-recovery infrastructure of an underlying
    null-space solver while running their own CG loop on top.

    Attributes:
        project: Inexact null-space projector ``W̃_k(v)``.  Maps a vector
            in the (free) ambient space to its projection onto
            ``null(A_work)`` using whatever inner approximation the
            underlying strategy provides (Cholesky for
            ``ProjectedCGCholesky``, CRAIG for ``ProjectedCGCraig``).
        d_p: Particular solution.  ``A_work @ d_p == b_work`` to inner
            solver precision; ``d_p`` already incorporates ``d_fixed`` on
            the bound-fixed coordinates, so it lives in the full ``n``-
            dimensional space.
        recover_multipliers: Closure mapping ``(B d + g)`` to a length-``m``
            multiplier vector with zeros on inactive rows.  Encapsulates
            both the inversion of ``A_work A_workᵀ`` *and* the active-mask
            zeroing.  HR Algorithm 4.5 calls this once per outer step
            (after the modified-residual CG iteration converges).
        hvp_work: Working-subspace HVP.  Equal to ``hvp_fn`` when no bound
            fixing is in effect; otherwise ``v -> _free * hvp_fn(_free * v)``
            so the iteration only sees the free coordinates.
        g_eff: Effective gradient ``g + B @ d_p`` evaluated against
            ``hvp_work``.  HR's notation calls this ``g_k``; it is the
            input to the projected-residual recurrence.
        A_work: Already-masked working constraint matrix (active rows,
            free columns).  Surfaced primarily so callers can sanity-check
            the residual ``||A_work @ d - b_work||``; ``project`` and
            ``recover_multipliers`` already incorporate it.
        free_mask: Boolean mask of free variables (always present;
            equals ``ones(n)`` when no bound-fixing is in effect).
        d_fixed: Fixed-variable values on bound-active coordinates
            (zeros elsewhere; zeros everywhere when no bound-fixing).
        has_fixed: ``True`` iff any coordinate is bound-fixed.  Cheap
            indicator so consumers do not have to redo the mask check.
        converged: Convergence flag of the inner projector solve that
            built the context (always ``True`` for Cholesky; carries the
            CRAIG breakdown / convergence flag for the iterative
            projector).  Composed strategies AND this with their own
            convergence to surface inner-projector failures upstream.
    """

    project: Callable[[Vector], Vector]
    d_p: Vector
    recover_multipliers: Callable[[Vector], Float[Array, " m"]]
    hvp_work: Callable[[Vector], Vector]
    g_eff: Vector
    A_work: Float[Array, "m n"]
    free_mask: Bool[Array, " n"]
    d_fixed: Vector
    has_fixed: bool
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

    # Coerce ``cg_tol`` to a true 0-d scalar.  Otherwise a ``(1,)``-shaped
    # tolerance (which can leak in via the adaptive Eisenstat-Walker path
    # if ``state.prev_grad_lagrangian`` ever carries an unexpected leading
    # axis) would broadcast every ``r_new_norm_sq < cg_tol ** 2`` comparison
    # below to shape ``(1,)``.  The boolean ``state.converged`` would then
    # latch onto that shape and the predicate fed to ``jax.lax.cond`` at
    # the bottom of this function would no longer be a scalar, raising
    # ``TypeError: Pred must be a scalar`` from deep inside JAX.
    cg_tol = to_scalar(cg_tol)
    # ``tol_sq`` is the squared residual-norm threshold used by the
    # convergence test below.  When ``cg_atol`` is provided we take
    # the *larger* of ``cg_atol**2`` and ``cg_tol**2`` so that the
    # absolute floor only kicks in when ``cg_tol`` is itself tighter
    # than the floor (e.g. ``mult_recovery_tol = 1e-12`` chasing a
    # near-KKT residual already at machine precision).
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
            a residual below the floor.  Used by the multiplier-recovery
            CG inside ``ProjectedCGCraig`` (where ``cg_tol`` is set to
            ``mult_recovery_tol = 1e-12`` and the RHS can shrink to the
            same scale near KKT).  Defaults to ``None`` (pure absolute
            ``cg_tol`` test, current behaviour).

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
    # Mirror the hybrid convergence threshold logic from ``build_cg_step``
    # so the *initial* convergence test (``r0`` already at floor) honors
    # the same effective floor as subsequent CG iterations.
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

    def build_projection_context(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> ProjectionContext:
        """Build a reusable projector + multiplier-recovery context.

        Composed strategies (e.g. ``HRInexactSTCG``) call this on the
        underlying inner solver to obtain its null-space projector,
        particular solution and multiplier-recovery closure without
        running the projector's own CG loop.

        The default implementation raises ``NotImplementedError`` so
        full-KKT solvers (``MinresQLPSolver``) cleanly opt out — they
        have no separate projection step and therefore cannot supply
        the inexact-projector ``W̃_k`` that HR Algorithm 4.5 needs.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a projection context; "
            "it cannot be used as the inner projector for HRInexactSTCG."
        )


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
        # Null-space CG enforces ``A d = b`` structurally via the
        # particular solution + range-space projector; the residual
        # ``||A d - b||`` is at floating-point floor by construction,
        # so we report 0 (no refinement applies).
        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            proj_residual=jnp.asarray(0.0, dtype=d.dtype),
            n_proj_refinements=jnp.asarray(0),
            # ``inf`` so the opt-in inexact-stationarity test can never
            # be tripped accidentally on a non-HR inner solver.
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=d.dtype),
        )

    def build_projection_context(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> ProjectionContext:
        return _make_cholesky_projection_ctx(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )


# ---------------------------------------------------------------------------
# Implementation: projected CG with Cholesky projection
# ---------------------------------------------------------------------------


def _make_cholesky_projection_ctx(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
) -> ProjectionContext:
    """Build a ``ProjectionContext`` backed by a regularised Cholesky
    factorisation of ``A_work A_workᵀ``.

    Mirrors the projector-construction prefix of
    ``_solve_projected_cg_cholesky``: masks ``A`` and ``b`` to the
    active rows, applies bound-fixing, factorises ``AAᵀ + 1e-8·I``,
    and packages the resulting projector, particular solution and
    multiplier-recovery closure (with one round of iterative
    refinement) into a ``ProjectionContext`` for reuse.
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

    if has_fixed:

        def hvp_work(v: Vector) -> Vector:
            return _free * hvp_fn(_free * v)

        g_eff = _free * (g + hvp_fn(_dfixed))
    else:
        hvp_work = hvp_fn
        g_eff = g

    def recover_multipliers(Bd_plus_g: Vector) -> Float[Array, " m"]:
        # KKT recovery with one step of iterative refinement to absorb
        # the O(eps · cond(AAt)) error introduced by the 1e-8 ridge.
        kkt_rhs = A_work @ Bd_plus_g
        mult = solve_AAt(kkt_rhs)
        mult = jnp.where(active_mask, mult, 0.0)
        grad_L_qp = Bd_plus_g - A_work.T @ mult
        delta = solve_AAt(A_work @ grad_L_qp)
        mult = mult + delta
        mult = jnp.where(active_mask, mult, 0.0)
        return mult

    return ProjectionContext(
        project=project,
        d_p=d_p,
        recover_multipliers=recover_multipliers,
        hvp_work=hvp_work,
        g_eff=g_eff,
        A_work=A_work,
        free_mask=_free,
        d_fixed=_dfixed,
        has_fixed=bool(has_fixed),
        converged=jnp.asarray(True),
    )


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
    cg_tol = to_scalar(cg_tol)

    ctx = _make_cholesky_projection_ctx(
        hvp_fn=hvp_fn,
        g=g,
        A=A,
        b=b,
        active_mask=active_mask,
        free_mask=free_mask,
        d_fixed=d_fixed,
    )
    A_work = ctx.A_work
    project = ctx.project
    hvp_work = ctx.hvp_work
    g_eff = ctx.g_eff
    d_p = ctx.d_p

    # Constraint preconditioner (Gould, Hribar & Nocedal, 2001).
    if precond_fn is not None and use_constraint_preconditioner:
        _raw_precond = precond_fn
        reg_diag = jnp.where(active_mask, 0.0, 1.0)
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
    multipliers = ctx.recover_multipliers(Bd + g)

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
        Tuple ``(x, converged)``.  ``converged`` is ``True`` only when the
        residual fell below the hybrid threshold; it is ``False`` if CRAIG
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
    # Hybrid convergence threshold: max(absolute floor, relative on ||rhs||).
    init_threshold = jnp.maximum(_CRAIG_TOL_ABS, tol * beta1_safe)
    init_converged = trivially_converged | (residual_init < init_threshold)
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
            # Hybrid: absolute floor plus relative on ``||rhs||``.
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
    # A "true" success requires the residual to be below the hybrid
    # threshold at termination and not having flagged a breakdown.
    final_threshold = jnp.maximum(_CRAIG_TOL_ABS, tol * beta1_safe)
    success = (final.residual < final_threshold) & ~final.breakdown
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
        # See ``ProjectedCGCholesky.solve`` for why the projection
        # diagnostics are reported as zero on null-space solvers.
        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            proj_residual=jnp.asarray(0.0, dtype=d.dtype),
            n_proj_refinements=jnp.asarray(0),
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=d.dtype),
        )

    def build_projection_context(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> ProjectionContext:
        return _make_craig_projection_ctx(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            free_mask=free_mask,
            d_fixed=d_fixed,
            craig_tol=self.craig_tol,
            craig_max_iter=self.craig_max_iter,
            mult_recovery_tol=self.mult_recovery_tol,
            mult_recovery_max_iter=self.mult_recovery_max_iter,
        )


# ---------------------------------------------------------------------------
# Implementation: projected CG with CRAIG projection
# ---------------------------------------------------------------------------


def _make_craig_projection_ctx(
    hvp_fn: Callable[[Vector], Vector],
    g: Vector,
    A: Float[Array, "m n"],
    b: Float[Array, " m"],
    active_mask: Bool[Array, " m"],
    free_mask: Bool[Array, " n"] | None = None,
    d_fixed: Vector | None = None,
    craig_tol: float = 1e-10,
    craig_max_iter: int = 200,
    mult_recovery_tol: float = 1e-12,
    mult_recovery_max_iter: int = 200,
) -> ProjectionContext:
    """Build a ``ProjectionContext`` backed by CRAIG (Golub-Kahan
    bidiagonalisation) for the null-space projection.

    Mirrors the projector-construction prefix of
    ``_solve_projected_cg_craig``.  The ``converged`` flag carries the
    CRAIG breakdown / convergence status of the *particular solution*
    solve (``A_work d_p = b_work``); per-projector-call CRAIG
    convergence flags inside the CG loop are not threaded through.
    The multiplier-recovery closure runs CG on the normal equations
    ``A A^T λ = -A (B d + g)`` (matching ``_solve_projected_cg_craig``
    sign convention).
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

    d_p_free, d_p_craig_conv = craig_solve(
        A_work, b_work, tol=craig_tol, max_iter=craig_max_iter
    )
    d_p_free_finite = jnp.isfinite(d_p_free).all()
    d_p_free = jnp.where(d_p_free_finite, d_p_free, jnp.zeros_like(d_p_free))
    d_p = d_p_free + _dfixed if has_fixed else d_p_free

    def project(v: Vector) -> Vector:
        v_work = _free * v if has_fixed else v
        x_proj, _ = craig_solve(
            A_work, A_work @ v_work, tol=craig_tol, max_iter=craig_max_iter
        )
        x_proj = jnp.where(jnp.isfinite(x_proj).all(), x_proj, jnp.zeros_like(x_proj))
        return v_work - x_proj

    if has_fixed:

        def hvp_work(v: Vector) -> Vector:
            return _free * hvp_fn(_free * v)

        g_eff = _free * (g + hvp_fn(_dfixed))
    else:
        hvp_work = hvp_fn
        g_eff = g

    reg_diag = jnp.where(active_mask, 0.0, 1.0)

    def normal_hvp(v: Float[Array, " m"]) -> Float[Array, " m"]:
        return A_work @ (A_work.T @ v) + reg_diag * v

    def recover_multipliers(Bd_plus_g: Vector) -> Float[Array, " m"]:
        kkt_rhs = A_work @ Bd_plus_g
        # Pass the absolute floor so a near-KKT iterate (where ``kkt_rhs``
        # itself shrinks toward machine precision) does not chase a
        # ``mult_recovery_tol`` target below ``eps`` and stall.  The CRAIG
        # absolute floor _CRAIG_TOL_ABS is reused here so the projector
        # and the multiplier recovery stop at the same noise floor.
        mult, _ = solve_unconstrained_cg(
            normal_hvp,
            -kkt_rhs,
            mult_recovery_max_iter,
            mult_recovery_tol,
            cg_atol=_CRAIG_TOL_ABS,
        )
        mult = jnp.where(active_mask, mult, 0.0)
        return mult

    return ProjectionContext(
        project=project,
        d_p=d_p,
        recover_multipliers=recover_multipliers,
        hvp_work=hvp_work,
        g_eff=g_eff,
        A_work=A_work,
        free_mask=_free,
        d_fixed=_dfixed,
        has_fixed=bool(has_fixed),
        converged=d_p_craig_conv,
    )


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
    cg_tol = to_scalar(cg_tol)

    ctx = _make_craig_projection_ctx(
        hvp_fn=hvp_fn,
        g=g,
        A=A,
        b=b,
        active_mask=active_mask,
        free_mask=free_mask,
        d_fixed=d_fixed,
        craig_tol=craig_tol,
        craig_max_iter=craig_max_iter,
        mult_recovery_tol=mult_recovery_tol,
        mult_recovery_max_iter=mult_recovery_max_iter,
    )
    A_work = ctx.A_work
    project = ctx.project
    hvp_work = ctx.hvp_work
    g_eff = ctx.g_eff
    d_p = ctx.d_p
    d_p_craig_conv = ctx.converged

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
                amat_hvp_reg,
                -AMr,
                mult_recovery_max_iter,
                mult_recovery_tol,
                cg_atol=_CRAIG_TOL_ABS,
            )
            return Mr - _raw_precond(A_work.T @ w)

        effective_precond: Callable[[Vector], Vector] | None = _constraint_precond
    else:
        effective_precond = precond_fn

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

    Bd = hvp_fn(final_cg.d)
    multipliers = ctx.recover_multipliers(Bd + g)

    finite_d = jnp.isfinite(final_cg.d).all()
    finite_mult = jnp.isfinite(multipliers).all()
    converged = final_cg.converged & d_p_craig_conv & finite_d & finite_mult

    return final_cg.d, multipliers, converged


# ---------------------------------------------------------------------------
# Heinkenschloss-Ridzal (2014) Algorithm 4.5: Steihaug-Toint CG with
# inexact null-space projections (no trust radius / no normal-tangent
# split — the line-search SQP wrapper handles globalization).
# ---------------------------------------------------------------------------

# Hardcoded absolute floor for the HR-STCG inner-convergence test.  The
# Step 1(a) test ``||z̃|| <= cg_tol * ||r̃_0||`` becomes
# ``||z̃|| <= max(_HRSTCG_TOL_ABS, cg_tol * ||r̃_0||)`` so that when
# ``||r̃_0||`` itself is at or below machine epsilon the iteration does
# not chase a target below ``eps`` and return a spurious
# ``converged=False`` flag.  Set tighter than ``_CRAIG_TOL_ABS`` (1e-12)
# because the HR test is on the *projected* residual ``z̃`` (already
# noise-cleaned by the projector), whereas CRAIG's residual is the raw
# ``A x - rhs`` which carries one extra noise level.
_HRSTCG_TOL_ABS = 1e-14


class _HRSTCGState(NamedTuple):
    """Internal state for the HR-inexact-STCG iteration.

    Layout:

    - ``t``: current iterate, in the (free) ambient space.
    - ``r``: HR's "modified" residual ``r̃_i``.  At ``i=0`` this is
      ``-W̃ g_eff`` (descent-residual sign convention so the standard
      descent CG step direction ``+α p`` minimises the model).  At
      subsequent iterations ``r̃_{i+1} = r̃_i − α̃_i H p̃_i`` — the HR
      "modified" recurrence (Remark 4.6.i) which avoids re-projecting
      the residual at every step and is the technical move that gives
      the iteration a fixed-linear-operator interpretation under
      inexact ``W̃``.
    - ``z``: ``z̃_i = W̃(r̃_i)``.  At ``i=0`` equal to ``r`` (no
      inner preconditioner); HR projects only when computing ``z`` —
      not when updating ``r``.
    - ``p``: current search direction.
    - ``rz``: ``⟨r̃_i, z̃_i⟩`` (= ``⟨r̃_i, r̃_i⟩`` without preconditioner).
    - ``proj_grad_norm``: ``‖r̃_0‖`` cached for the relative
      convergence test ``‖z̃_i‖ ≤ tol · ‖r̃_0‖`` and surfaced as the
      noise-aware stationarity proxy.
    - ``P``, ``HP``, ``pHp_diag``: history buffers for full
      H-conjugacy reorthogonalisation.  ``P[j] = p̃_j``,
      ``HP[j] = H p̃_j``, ``pHp_diag[j] = ⟨p̃_j, H p̃_j⟩``.  Static
      shape ``(max_cg_iter, n)`` / ``(max_cg_iter,)``.
    - ``iteration``, ``converged``: standard CG bookkeeping.
    """

    t: Vector
    r: Vector
    z: Vector
    p: Vector
    rz: Scalar
    proj_grad_norm: Scalar
    P: Float[Array, "imax n"]
    HP: Float[Array, "imax n"]
    pHp_diag: Float[Array, " imax"]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]


def _hr_stcg(
    hvp_work: Callable[[Vector], Vector],
    g_eff: Vector,
    project: Callable[[Vector], Vector],
    cg_tol: Scalar | float,
    cg_regularization: float,
    max_cg_iter: int,
) -> tuple[Vector, Scalar, Bool[Array, ""]]:
    """Run HR (2014) Algorithm 4.5 — STCG with inexact null-space projections.

    Solves the equality-constrained reduced QP

        minimize  q(t̃) = g_eff^T t̃ + 1/2 t̃^T H t̃   over null(A_work)

    using projected conjugate gradient where the projector ``W̃_k`` is
    applied only when computing ``z̃_i = W̃(r̃_i)``.  The residual
    itself follows the modified recurrence ``r̃_{i+1} = r̃_i − α_i H p̃_i``
    (HR Remark 4.6.i), which absorbs the projector exactly once at
    initialisation and avoids the noise re-injection that the
    re-projected three-term recurrence in the textbook STCG induces
    under inexact ``W̃``.  Combined with full H-conjugacy
    reorthogonalisation of every ``p̃_i`` against all stored
    ``p̃_j``, the iteration becomes the unique CG sequence for
    *some* fixed linear operator ``W̃_k`` (HR Lemma 4.10), even though
    no such operator is ever formed.

    Sign convention.  We use the descent-residual convention
    ``r̃ = -∇q`` (so ``r̃_0 = -W̃ g_eff``); the resulting standard
    CG step ``t̃_{i+1} = t̃_i + α_i p̃_i`` with ``α_i =
    ⟨r̃_i, z̃_i⟩ / ⟨p̃_i, H p̃_i⟩`` minimises the model.  This is
    algebraically equivalent to HR's max-residual presentation in
    their Algorithm 4.5; the change keeps the descent-direction
    semantics of ``t̃_k`` consistent with how the outer SQP composes
    ``d = d_p + t̃_k``.

    Convergence test.  Step 1(a) uses
    ``‖z̃_i‖ ≤ max(_HRSTCG_TOL_ABS, tol_CG · ‖r̃_0‖)`` — a hybrid
    absolute+relative test against the initial *projected* gradient.
    The relative target is HR's noise-aware stationarity proxy: both
    numerator and denominator carry the same projector noise, so the
    test reaches the inner solver's precision floor and stops cleanly
    instead of grinding against it.  The absolute floor protects the
    edge case where ``‖r̃_0‖`` is itself at or below machine epsilon
    (e.g. the QP at a feasible KKT point with `g_eff` already in
    ``range(A_work^T)``); without the floor the relative target
    collapses below ``eps`` and the iteration would return a spurious
    ``converged=False`` flag even though the QP is at its KKT point.

    Curvature / stagnation guard.  We preserve the SNOPT-style
    scale-invariant guard ``⟨p̃, H p̃⟩ ≤ ε ‖p̃‖²`` for negative /
    near-zero curvature and a stagnation guard
    ``|⟨r̃_i, p̃_i⟩| < 1e-30``; either short-circuits the iteration
    with the last good iterate.

    Memory.  The full-reorth buffers ``P, HP`` are static-shape
    ``(max_cg_iter, n)``; total ``2 · imax · n · 8 B``
    (e.g. ``imax=50, n=50_000`` → 40 MB).  Reorth cost across all
    iterations is ``O(imax^2 · n)``.

    Args:
        hvp_work: Working-subspace HVP ``v -> H @ v`` (already masked
            to the free subspace when bound-fixing is in effect).
        g_eff: Effective gradient ``g + B d_p`` (HR's ``g_k``).
        project: Inexact null-space projector ``W̃_k`` (typically
            from ``ProjectionContext.project``).
        cg_tol: Relative convergence tolerance on ``‖z̃_i‖`` (Step 1(a)).
        cg_regularization: Curvature-guard threshold (``δ²``).  Set to
            zero to disable the guard.
        max_cg_iter: Static upper bound on the number of CG iterations.
            Determines the size of the reorth buffers.

    Returns:
        ``(t̃_k, ‖r̃_0‖, converged)`` — the descent step (in the free
        subspace), the initial *projected* gradient norm (which is
        the noise-aware stationarity proxy), and the convergence flag.
    """
    n = g_eff.shape[0]
    cg_tol = to_scalar(cg_tol)

    Wg = project(g_eff)
    r0 = -Wg
    z0 = r0
    proj_grad_norm = jnp.sqrt(jnp.maximum(jnp.dot(r0, r0), 0.0))
    rz0 = jnp.dot(r0, z0)
    p0 = z0

    P = jnp.zeros((max_cg_iter, n), dtype=g_eff.dtype)
    HP = jnp.zeros((max_cg_iter, n), dtype=g_eff.dtype)
    pHp_diag = jnp.zeros(max_cg_iter, dtype=g_eff.dtype)

    init_state = _HRSTCGState(
        t=jnp.zeros(n, dtype=g_eff.dtype),
        r=r0,
        z=z0,
        p=p0,
        rz=rz0,
        proj_grad_norm=proj_grad_norm,
        P=P,
        HP=HP,
        pHp_diag=pHp_diag,
        iteration=jnp.array(0),
        # Pre-converge if the projected gradient is already at the
        # absolute floor.  The previous form ``proj_grad_norm <= cg_tol *
        # max(proj_grad_norm, 1e-30)`` collapsed to ``proj_grad_norm
        # <= cg_tol * proj_grad_norm`` (i.e. essentially ``<= 0``) for
        # any ``cg_tol < 1`` and so never fired.  The hybrid form below
        # is the right pre-convergence check.
        converged=jnp.reshape(proj_grad_norm <= _HRSTCG_TOL_ABS, ()),
    )

    indices = jnp.arange(max_cg_iter)

    def step_fn(_i: int, state: _HRSTCGState) -> _HRSTCGState:
        def do_step(state: _HRSTCGState) -> _HRSTCGState:
            i = state.iteration
            Hp = hvp_work(state.p)
            pHp = jnp.dot(state.p, Hp)
            pp = jnp.dot(state.p, state.p)

            # Step 1(c)/(d): SNOPT-style scale-invariant curvature
            # guard plus an absolute floor anchored to the initial
            # projected gradient.  Without the absolute floor, when
            # ``p`` shrinks toward machine precision near convergence
            # both ``pHp`` and ``cg_regularization * pp`` decay
            # together; ``pHp`` can drop to ``O(eps²)`` while still
            # exceeding the relative threshold, leaving
            # ``alpha = rz / pHp`` to amplify the noisy direction by
            # ``O(1/eps²)`` and contaminate ``t``.  The absolute floor
            # ``cg_regularization · ‖r̃_0‖²`` cuts that noise
            # amplification — it is dimensionally consistent with
            # ``pHp`` (curvature × length²) and reflects the smallest
            # curvature the iteration could meaningfully resolve.
            # HR's stagnation check ``⟨r̃_i, p̃_i⟩ ≈ 0`` triggers a
            # separate exit.
            abs_floor = cg_regularization * state.proj_grad_norm * state.proj_grad_norm
            bad_curvature = pHp <= jnp.maximum(cg_regularization * pp, abs_floor)
            rp = jnp.dot(state.r, state.p)
            stagnation = jnp.abs(rp) < 1e-30
            short_circuit = bad_curvature | stagnation

            pHp_safe = jnp.maximum(pHp, 1e-30)
            alpha = jnp.where(short_circuit, jnp.array(0.0), state.rz / pHp_safe)

            t_new = state.t + alpha * state.p
            # HR Remark 4.6.i — modified residual recurrence.  Note: NOT
            # ``project(... )``; the projector is applied only when
            # computing ``z̃`` below.
            r_new = state.r - alpha * Hp
            z_new = project(r_new)

            # Update buffers with current p_i / Hp_i / pHp_i so the
            # full-reorth uses every direction including the freshly
            # consumed one.
            P_buf = state.P.at[i].set(state.p)
            HP_buf = state.HP.at[i].set(Hp)
            pHp_buf = state.pHp_diag.at[i].set(pHp)

            # Full reorthogonalisation: the new search direction p_{i+1}
            # is built from z_{i+1} so as to be H-conjugate to every
            # stored p_j.  Coeffs ``β_l = -⟨z_{i+1}, H p_l⟩/⟨p_l,
            # H p_l⟩`` enforce ``⟨p_{i+1}, H p_l⟩ = 0`` for l ≤ i.
            mask_j = indices <= i  # include the just-stored p_i
            Hz_dots = HP_buf @ z_new  # (max_cg_iter,)
            pHp_diag_safe = jnp.where(jnp.abs(pHp_buf) > 1e-30, pHp_buf, 1e-30)
            coeffs = jnp.where(mask_j, Hz_dots / pHp_diag_safe, 0.0)
            p_new = z_new - coeffs @ P_buf

            rz_new = jnp.dot(r_new, z_new)
            z_norm = jnp.sqrt(jnp.maximum(jnp.dot(z_new, z_new), 0.0))
            # Hybrid abs+rel convergence threshold: the absolute floor
            # ``_HRSTCG_TOL_ABS`` kicks in only when ``cg_tol *
            # ||r̃_0||`` would itself be below the floor (i.e. on a QP
            # whose initial projected gradient is already near machine
            # epsilon).  Otherwise the relative HR test dominates.
            tol_target = jnp.maximum(_HRSTCG_TOL_ABS, cg_tol * state.proj_grad_norm)
            converged_new = (z_norm <= tol_target) | short_circuit

            return _HRSTCGState(
                t=jnp.where(short_circuit, state.t, t_new),
                r=jnp.where(short_circuit, state.r, r_new),
                z=jnp.where(short_circuit, state.z, z_new),
                p=jnp.where(short_circuit, state.p, p_new),
                rz=jnp.where(short_circuit, state.rz, rz_new),
                proj_grad_norm=state.proj_grad_norm,
                P=P_buf,
                HP=HP_buf,
                pHp_diag=pHp_buf,
                iteration=state.iteration + 1,
                converged=converged_new,
            )

        # Defensive scalarisation of the predicate, mirroring
        # ``build_cg_step``.
        converged_pred = jnp.reshape(state.converged, ())
        return jax.lax.cond(converged_pred, lambda s: s, do_step, state)

    final = jax.lax.fori_loop(0, max_cg_iter, step_fn, init_state)
    return final.t, final.proj_grad_norm, final.converged


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

    After PMINRES-QLP returns the iterate ``d``, an M-metric range-space
    projection drives ``A d = b`` on the active rows.  The single shot
    is followed by up to ``proj_refine_max_iter`` rounds of iterative
    refinement, each costing one matvec + one Schur back-solve (no
    refactorisation).  Refinement squares the relative feasibility
    error per round, eliminating the residual floor ``O(eps · cond(A
    M Aᵀ) · ||r_dual||)`` from the ``1e-8·I`` Schur regularisation;
    on ill-conditioned problems this is the difference between the
    SQP step landing on the linearised feasible region and drifting
    off it.  See HR (2014, Algorithm 4.18 step 1(a)) for the
    motivation.
    """

    max_iter: int = 200
    tol: float = 1e-10
    max_cg_iter: int = 50
    # Iterative refinement of the M-metric projection.  See class docstring.
    proj_refine_max_iter: int = 3
    proj_refine_rtol: float = 1e-10
    proj_refine_atol: float = 1e-14

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
        d, multipliers, converged, proj_residual, n_refinements = _solve_kkt_minres_qlp(
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
            proj_refine_max_iter=self.proj_refine_max_iter,
            proj_refine_rtol=self.proj_refine_rtol,
            proj_refine_atol=self.proj_refine_atol,
        )
        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            proj_residual=proj_residual,
            n_proj_refinements=n_refinements,
            projected_grad_norm=jnp.asarray(jnp.inf, dtype=d.dtype),
        )


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
    proj_refine_max_iter: int = 3,
    proj_refine_rtol: float = 1e-10,
    proj_refine_atol: float = 1e-14,
) -> tuple[
    Vector,
    Float[Array, " m"],
    Bool[Array, ""],
    Scalar,
    Int[Array, ""],
]:
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

    # Inactive constraint rows are zeroed in A_work / b_work; the
    # range-space and Schur factorisations need a "1" on those diagonal
    # positions to stay invertible without coupling into the active
    # block.  Hoisted out of the preconditioner branch so the no-precond
    # path can also reuse it for the posterior projection.
    reg_diag = jnp.where(active_mask, 0.0, 1.0)

    # Block-diagonal SPD preconditioner (Section 3.4 of Choi 2006)
    if precond_fn is not None:
        _raw_precond = precond_fn

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

        # M-metric range-space projection: minimise ||δd||_{M^{-1}} s.t.
        # A_work (d - δd) = b_work.  The optimality conditions give
        #   δd = M A_work^T (A_work M A_work^T)^{-1} (A_work d - b_work),
        # which reuses A_M_AT_chol for free.  See Orban & Arioli, SIMAX
        # 36(3) 2014, Lemma 3.1; Benzi-Golub-Liesen, Acta Numerica 2005,
        # §5 (range-space methods).  Required because PMINRES-QLP only
        # minimises the Euclidean residual of the full KKT system; when
        # the Lanczos recurrence is truncated by max_iter or stalled by
        # the SPD preconditioner's free/fixed coupling, the dual block
        # A_work d - b_work stays non-zero and the SLSQP step leaves
        # the linearised feasible region.
        def _project_step(d_in: Vector) -> tuple[Vector, Scalar]:
            r_dual = jnp.where(active_mask, A_work @ d_in - b_work, 0.0)
            r_norm = jnp.linalg.norm(r_dual)
            delta_lambda = _solve_schur(r_dual)
            delta_d = _primal_precond(A_work.T @ delta_lambda)
            return d_in - delta_d, r_norm
    else:
        solution, converged = pminres_qlp_solve(
            kkt_matvec, kkt_rhs, tol=tol, max_iter=max_iter
        )

        # No SPD preconditioner is available, so build a small dedicated
        # m x m Cholesky of A_work A_work^T just for the posterior
        # 2-norm projection.  This is cheap (m << n) and guarantees the
        # same feasibility-restoration property as the preconditioned
        # branch.  Using ``A_work`` (which already incorporates
        # ``free_mask``) keeps the projection consistent with the
        # bound-fixed subspace.
        A_AT = A_work @ A_work.T + jnp.diag(reg_diag) + 1e-8 * jnp.eye(m)
        A_AT_chol = jnp.linalg.cholesky(A_AT)

        def _project_step(d_in: Vector) -> tuple[Vector, Scalar]:
            r_dual = jnp.where(active_mask, A_work @ d_in - b_work, 0.0)
            r_norm = jnp.linalg.norm(r_dual)
            delta_lambda = jax.scipy.linalg.cho_solve((A_AT_chol, True), r_dual)
            return d_in - A_work.T @ delta_lambda, r_norm

    # Iterative refinement of the projection.  HR (2014, Algorithm 4.18,
    # step 1(a)) tightens the projection accuracy until the feasibility
    # residual ||A d - b|| is small relative to ||b||.  We approximate
    # this by a fixed-point loop on the projection itself: each round
    # applies one more correction ``d <- d - M A^T (A M A^T)^{-1} (A d
    # - b)``, which squares the relative feasibility error per round.
    # The loop is ``proj_refine_max_iter + 1`` projections in total
    # (one mandatory shot + ``proj_refine_max_iter`` refinement rounds).
    # Cost per round: one matvec + one Schur back-solve, negligible
    # next to the Krylov solve we already paid for.
    b_norm_floor = jnp.linalg.norm(b_work) + jnp.asarray(1.0, dtype=b_work.dtype)
    proj_atol = jnp.asarray(proj_refine_atol, dtype=b_work.dtype)
    proj_rtol = jnp.asarray(proj_refine_rtol, dtype=b_work.dtype)
    refine_target = proj_atol + proj_rtol * b_norm_floor

    # Initial mandatory projection (round 0).
    d_proj, residual_pre = _project_step(solution[:n])
    n_refinements = jnp.asarray(0)

    def _refine_body(carry, _):
        d_cur, _r_prev, done_prev, n_done = carry
        d_next, r_next = _project_step(d_cur)
        # ``done_prev`` latches once the residual drops below the target
        # so subsequent rounds become no-ops (numerical stability matches
        # exactly via ``jnp.where``, no NaNs from a degenerate Schur
        # solve once the residual is already at floor).
        d_out = jnp.where(done_prev, d_cur, d_next)
        r_out = jnp.where(done_prev, _r_prev, r_next)
        # Count this round only if it actually ran.
        n_out = jnp.where(done_prev, n_done, n_done + 1)
        done_next = done_prev | (r_out <= refine_target)
        return (d_out, r_out, done_next, n_out), r_out

    if proj_refine_max_iter > 0:
        # Compute initial post-shot residual; if it's already at floor
        # we want to skip refinement entirely.  Recomputing it cheaply:
        residual_init = jnp.linalg.norm(
            jnp.where(active_mask, A_work @ d_proj - b_work, 0.0)
        )
        done_init = residual_init <= refine_target
        (d_proj, residual_post, _done_final, n_refinements), _ = jax.lax.scan(
            _refine_body,
            (d_proj, residual_init, done_init, n_refinements),
            None,
            length=proj_refine_max_iter,
        )
    else:
        residual_post = jnp.linalg.norm(
            jnp.where(active_mask, A_work @ d_proj - b_work, 0.0)
        )

    # Re-use ``residual_pre`` so that the unused-variable check passes
    # while still letting downstream code observe the *post*-refinement
    # residual via the InnerSolveResult.  ``residual_pre`` is the
    # residual feeding into the first projection (i.e. the unprojected
    # MINRES iterate's infeasibility); we do not surface it directly
    # but keep the name for readability.
    del residual_pre

    d = d_proj
    if has_fixed:
        # Force the direction to respect the fixed mask.  The KKT matvec
        # and preconditioner branches already keep fixed positions at
        # zero; this final projection guards against floating-point
        # drift and preconditioner implementations that may leak small
        # cross-coupling through the free/fixed boundary.
        d = _free * d + _dfixed
    # KKT system uses L = ... + λ^T(Ad - b), but the active-set QP
    # solver uses L = ... - μ^T(Ad - b) with μ >= 0, so μ = -λ.
    # Multipliers are left as PMINRES-QLP returned them; the projection
    # nudges d by O(||r_dual||), so the resulting KKT-stationarity
    # inconsistency is O(||B|| * ||δd||) and is absorbed by the outer
    # L1 merit / L-BFGS secant blending.
    multipliers = -solution[n:]
    multipliers = jnp.where(active_mask, multipliers, 0.0)

    finite = jnp.isfinite(d).all() & jnp.isfinite(multipliers).all()
    return d, multipliers, converged & finite, residual_post, n_refinements


# ---------------------------------------------------------------------------
# Strategy 4: HR (2014) Algorithm 4.5 — STCG with inexact projections
# ---------------------------------------------------------------------------


class HRInexactSTCG(AbstractInnerSolver):
    """Heinkenschloss-Ridzal (2014) Algorithm 4.5 — Steihaug-Toint CG with
    inexact null-space projections.

    Composes an existing null-space inner solver
    (``ProjectedCGCholesky`` or ``ProjectedCGCraig``) to obtain its
    projector ``W̃_k``, particular solution ``d_p`` and
    multiplier-recovery closure, then runs a *separate* CG iteration
    on top whose three textbook three-term-recurrence cancellations are
    replaced by full H-conjugacy reorthogonalisation against every
    previous search direction.  The residual update follows the
    "modified" recurrence ``r̃_{i+1} = r̃_i − α_i H p̃_i`` (HR Remark
    4.6.i) — the projector is applied only when computing
    ``z̃_{i+1} = W̃(r̃_{i+1})``.

    Together, these two changes give the iteration a fixed-linear-
    operator interpretation under inexact ``W̃`` (HR Lemma 4.10,
    Theorem 4.11): even when each ``W̃(v)`` carries projection
    noise, the iteration is the unique CG sequence for *some* fixed
    operator ``W̃_k``.  This is the technical move that makes the
    iteration converge cleanly to the noise floor instead of grinding
    against it.

    Recommended pairing.  ``ProjectedCGCraig`` exposes the actual
    tightenable knob (``craig_tol``); composing
    ``HRInexactSTCG(inner=ProjectedCGCraig(...))`` lets the user
    trade projector accuracy against runtime explicitly.  The
    Cholesky variant has a fixed ``1e-8`` regularisation so its
    projection floor is not user-controllable.

    ``MinresQLPSolver`` is *not* a valid composition target:
    full-KKT solvers do not expose a separate projection step.  An
    attempted composition raises ``NotImplementedError`` from the
    inherited ``AbstractInnerSolver.build_projection_context``
    default at the first call to ``solve``.

    Cost.  Per HR Algorithm 4.5 step: one HVP, one projector
    application (provided by ``inner``), and an ``O(imax · n)``
    Gram-Schmidt back-substitution against the stored search-direction
    history.  Total reorth cost across all iterations:
    ``O(imax^2 · n)``.  Memory: ``2 · imax · n`` floats for the
    static-shape ``(P, HP)`` buffers.

    Attributes:
        inner: Composed null-space inner solver supplying the
            projector and multiplier-recovery infrastructure.  Must
            implement ``build_projection_context``; the saddle-point
            ``MinresQLPSolver`` does not and will raise on the first
            ``solve`` call.
        max_cg_iter: Static upper bound on the number of inner CG
            iterations.  Determines the size of the reorth buffers.
        cg_tol: Relative convergence tolerance for the projected
            residual ``‖z̃_i‖ ≤ tol · ‖r̃_0‖``.
        cg_regularization: Curvature-guard threshold ``δ²`` used by
            the SNOPT-style scale-invariant short-circuit
            ``⟨p̃, H p̃⟩ ≤ δ² ‖p̃‖²``.  Defaults to ``1e-6``; set to
            ``0.0`` to disable.
    """

    inner: AbstractInnerSolver
    max_cg_iter: int
    cg_tol: Scalar | float
    cg_regularization: float = 1e-6

    def build_projection_context(
        self,
        hvp_fn: Callable[[Vector], Vector],
        g: Vector,
        A: Float[Array, "m n"],
        b: Float[Array, " m"],
        active_mask: Bool[Array, " m"],
        precond_fn: Callable[[Vector], Vector] | None = None,
        free_mask: Bool[Array, " n"] | None = None,
        d_fixed: Vector | None = None,
    ) -> ProjectionContext:
        # Delegate to the composed projector; HRInexactSTCG itself does
        # not provide an additional projector layer.
        return self.inner.build_projection_context(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )

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
        # ``precond_fn`` is accepted for interface compatibility but
        # silently dropped: HR Algorithm 4.5 uses no inner
        # preconditioner.  The composed projector may still consume it
        # internally (e.g. for its constraint-preconditioner path) when
        # ``build_projection_context`` chooses to use it.
        ctx = self.inner.build_projection_context(
            hvp_fn=hvp_fn,
            g=g,
            A=A,
            b=b,
            active_mask=active_mask,
            precond_fn=precond_fn,
            free_mask=free_mask,
            d_fixed=d_fixed,
        )
        effective_tol = adaptive_tol if adaptive_tol is not None else self.cg_tol

        # HR-STCG iterates a null-space step ``t̃`` starting from
        # ``d_p`` (the particular solution that satisfies
        # ``A_work d_p = b_work``).  The "effective gradient" handed
        # to the CG iteration is therefore the gradient of the
        # quadratic at ``d_p``: ``g_k = g_eff + B d_p``.  Without
        # the ``B d_p`` shift the iteration would compute the wrong
        # search direction (the residual would not be projected onto
        # the right vector), and the final ``d = d_p + t̃`` would
        # silently diverge from the true QP optimum.  This mirrors
        # the ``r0 = project(-(g_eff + B d_p))`` initialisation in
        # ``_solve_projected_cg_cholesky``.
        Bd_p = ctx.hvp_work(ctx.d_p)
        g_at_dp = ctx.g_eff + Bd_p

        t_tilde, proj_grad_norm, cg_converged = _hr_stcg(
            hvp_work=ctx.hvp_work,
            g_eff=g_at_dp,
            project=ctx.project,
            cg_tol=effective_tol,
            cg_regularization=self.cg_regularization,
            max_cg_iter=self.max_cg_iter,
        )

        # The HR step lives in the (free) ambient space.  Combine it
        # with the particular solution (which already incorporates
        # ``d_fixed``).
        d = ctx.d_p + t_tilde

        # Multiplier recovery uses the full (unmasked) HVP — same
        # convention as the underlying solver.
        Bd = hvp_fn(d)
        multipliers = ctx.recover_multipliers(Bd + g)

        finite_d = jnp.isfinite(d).all()
        finite_mult = jnp.isfinite(multipliers).all()
        converged = cg_converged & ctx.converged & finite_d & finite_mult

        return InnerSolveResult(
            d=d,
            multipliers=multipliers,
            converged=converged,
            # Null-space projector enforces ``A d = b`` structurally.
            proj_residual=jnp.asarray(0.0, dtype=d.dtype),
            n_proj_refinements=jnp.asarray(0),
            projected_grad_norm=proj_grad_norm.astype(d.dtype),
        )
