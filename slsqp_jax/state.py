"""State / result dataclasses for the SLSQP solver and its sub-components.

This module consolidates every JAX-pytree dataclass and NamedTuple that
flows between the SLSQP outer solver, the QP layer, and the inner
projector / Krylov solvers.  Splitting these out of the algorithmic
modules makes the data flow legible at a glance and removes an
otherwise pervasive set of cross-module imports.

The dataclasses are split into three thematic groups:

* **Outer-loop state** (:class:`SLSQPState`, :class:`SLSQPDiagnostics`,
  :func:`get_diagnostics`, :func:`_init_diagnostics`) — produced and
  consumed by :class:`slsqp_jax.SLSQP` across iterations.
* **QP-layer state** (:class:`QPResult`, :class:`QPSolverResult`,
  :class:`QPState`) — produced and consumed by the QP active-set loop
  and the bound-fixing pass.
* **Inner-solver state** (:class:`InnerSolveResult`,
  :class:`ProjectionContext`) — produced by the equality-constrained
  inner solvers and consumed by the QP active-set loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from slsqp_jax.hessian import LBFGSHistory
from slsqp_jax.types import Scalar, Vector

# ---------------------------------------------------------------------------
# Inner-solver state
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
        projected_grad_norm: Norm of the *projected* initial gradient
            ``W̃_k g`` that the inner solver actually iterated against
            (HR 2014, Theorem 3.5).  This is the noise-aware
            stationarity proxy: when the outer SQP enables
            ``use_inexact_stationarity``, the run is allowed to converge
            once this value drops below ``rtol * max(|L|, 1)``.  Defaults
            to ``inf`` so that solvers which do not produce this
            quantity (i.e. anything other than ``HRInexactSTCG``) cannot
            accidentally satisfy a ``< rtol`` test even if the user
            toggles the flag — the inexact path silently degrades to
            "never converges this way".
    """

    d: Vector
    multipliers: Float[Array, " m"]
    converged: Bool[Array, ""]
    proj_residual: Scalar = jnp.asarray(0.0)
    n_proj_refinements: Int[Array, ""] = jnp.asarray(0)
    projected_grad_norm: Scalar = jnp.asarray(jnp.inf)


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
# QP-layer state
# ---------------------------------------------------------------------------


class QPState(eqx.Module):
    """State for the Active Set QP solver.

    ``any_inner_failure`` accumulates whether any inner-solver call
    during the active-set loop reported non-convergence or produced a
    non-finite direction.  The final ``QPSolverResult.converged`` combines
    this with the active-set completion check.

    ``last_add_idx`` / ``last_drop_idx`` / ``ping_pong_count`` /
    ``ping_ponged`` implement an explicit anti-cycling guard on top of
    EXPAND: when the active-set loop alternately adds and drops the
    *same* constraint several times in a row (typical signature of
    multiplier-recovery noise on a degenerate vertex) the loop is
    short-circuited as ``converged=True`` with the sticky
    ``ping_ponged`` flag set so the outer solver can surface it
    through the diagnostic counters.  Indices are initialised to
    ``-1`` to mean "no add/drop yet".
    """

    d: Vector
    active_set: Bool[Array, " m_ineq"]
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    iteration: Int[Array, ""]
    converged: Bool[Array, ""]
    any_inner_failure: Bool[Array, ""]
    last_add_idx: Int[Array, ""]
    last_drop_idx: Int[Array, ""]
    ping_pong_count: Int[Array, ""]
    ping_ponged: Bool[Array, ""]
    # Latest M-metric projection feasibility residual from the inner
    # solver (relevant for ``MinresQLPSolver``; always 0 for null-space
    # CG / CRAIG solvers where feasibility is enforced structurally).
    proj_residual: Scalar
    # Cumulative number of M-metric projection refinement rounds across
    # all inner solves performed inside this active-set loop.
    n_proj_refinements: Int[Array, ""]
    # Latest projected-gradient norm ``||W̃_k g_k||`` from the inner
    # solver (``HRInexactSTCG`` only; defaults to ``inf`` for every
    # other inner solver).  Surfaced through ``QPSolverResult`` for the
    # opt-in ``use_inexact_stationarity`` outer-loop convergence test.
    # Stores the *latest* inner-solver value rather than accumulating;
    # the active-set loop's final inner solve produces the value that
    # matters.
    projected_grad_norm: Scalar


class QPSolverResult(NamedTuple):
    """Result returned by :func:`slsqp_jax.qp.solve_qp`.

    The Bundle 1 diagnostic fields ``ping_ponged`` / ``reached_max_iter``
    / ``final_working_tol`` are surfaced from the active-set loop so
    the outer solver can track *why* the QP stopped (clean convergence
    versus cycling versus iteration-budget exhaustion).  They default
    to ``False`` / ``False`` / ``0.0`` for the trivial QP paths that
    do not run an active-set loop.

    This is the *internal* QP-pipeline result.  The richer outer-facing
    :class:`QPResult` (returned by :meth:`SLSQP._solve_qp_subproblem`,
    after bound-fixing) wraps this and adds bound-handling diagnostics.
    """

    d: Vector
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    active_set: Bool[Array, " m_ineq"]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]
    ping_ponged: Bool[Array, ""]
    reached_max_iter: Bool[Array, ""]
    final_working_tol: Scalar
    # M-metric projection feasibility residual from the inner solver
    # producing the *final* QP direction.  Always 0 for null-space
    # solvers; surfaces the post-refinement residual of
    # ``MinresQLPSolver`` so the outer SQP can flag QP solves whose
    # feasibility floor is dangerously high.
    proj_residual: Scalar
    # Cumulative number of M-metric projection refinement rounds taken
    # across all inner solves performed inside the active-set loop.
    n_proj_refinements: Int[Array, ""]
    # Latest projected-gradient norm from the final inner solve (HR
    # Inexact STCG only; ``inf`` for every other inner solver).
    # Surfaces the noise-aware stationarity proxy used by the opt-in
    # ``use_inexact_stationarity`` outer-loop convergence test.
    projected_grad_norm: Scalar


class QPResult(eqx.Module):
    """Outer-facing result of solving the SLSQP QP subproblem.

    Returned by :meth:`SLSQP._solve_qp_subproblem` after the bound-fixing
    loop has post-processed the direction returned by :func:`solve_qp`.

    Attributes:
        direction: The search direction d from the QP solution.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        active_set: Boolean mask of active inequality constraints at the solution.
        converged: Whether the QP solver converged successfully.
        iterations: Number of active-set iterations taken.
        bound_fix_solves: Number of non-trivial bound-fixing passes that
            actually ran the reduced-space inner solve (passes where the
            free mask did not change are short-circuited).  Useful for
            debugging bound-handling overhead; 0 when the problem has no
            bound constraints.
        n_bound_fixed: Number of variables pinned to a box bound in the
            final QP direction (counts both lower- and upper-bound
            activations).
        ping_ponged: True when the QP active-set loop short-circuited
            on a detected add/drop ping-pong cycle.
        reached_max_iter: True when the QP active-set loop exhausted
            its iteration budget (``qp_max_iter``).
        lpeca_bypassed: True when the LPEC-A prediction was skipped
            for this step (warm-up window or trust gate fired).
        lpeca_capped: True when the LPEC-A prediction was truncated
            by the rank-aware size cap.
        n_lpeca_bounds_prefixed: Number of box-bound variables
            pre-fixed by LPEC-A before entering the bound-fixing
            loop.  Always 0 when ``active_set_method == "expand"``,
            when ``lpeca_predict_bounds=False``, or when LPEC-A was
            bypassed by the warm-up / trust gates.
        proj_residual: Post-refinement constraint residual ``||A d -
            b||`` from the inner solver producing the final QP
            direction (after bound-fixing).  Always 0 for null-space
            CG / CRAIG; relevant for ``MinresQLPSolver`` where it
            reflects the M-metric range-space projection floor and
            feeds the outer divergence detector via
            ``SLSQPDiagnostics.max_proj_residual``.
        n_proj_refinements: Cumulative number of M-metric projection
            refinement rounds across all inner solves performed for
            this QP step (active-set loop + bound-fixing loop).  Always
            0 for null-space solvers.
        projected_grad_norm: Latest inner-solver projected-gradient
            norm (``HRInexactSTCG`` only; ``inf`` otherwise).
    """

    direction: Vector
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]
    active_set: Bool[Array, " m_ineq"]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]
    bound_fix_solves: Int[Array, ""]
    n_bound_fixed: Int[Array, ""]
    ping_ponged: Bool[Array, ""]
    reached_max_iter: Bool[Array, ""]
    lpeca_bypassed: Bool[Array, ""]
    lpeca_capped: Bool[Array, ""]
    n_lpeca_bounds_prefixed: Int[Array, ""]
    proj_residual: Scalar
    n_proj_refinements: Int[Array, ""]
    projected_grad_norm: Scalar


# ---------------------------------------------------------------------------
# Outer-loop diagnostics and state
# ---------------------------------------------------------------------------


class SLSQPDiagnostics(eqx.Module):
    """Diagnostic counters and statistics accumulated during an SLSQP run.

    These fields are populated inside ``step()`` without Python-side
    branching, and can be inspected by the user after a solve to decide
    whether the ``optimistix`` result code is meaningful (e.g. to detect
    a ``RESULTS.successful`` that actually came from chronic line-search
    failure rather than real convergence).

    Attributes:
        n_qp_inner_failures: Number of iterations where the QP solver
            reported ``converged=False``.
        n_ls_failures: Number of iterations where the line search
            reported ``success=False``.
        n_lbfgs_skips: Number of iterations where the L-BFGS append
            skipped the new curvature pair.
        n_nan_directions: Number of iterations where the QP returned
            a non-finite search direction.
        max_gamma: Maximum L-BFGS ``gamma`` observed across iterations.
        min_diag: Minimum L-BFGS per-variable diagonal entry observed.
        max_diag: Maximum L-BFGS per-variable diagonal entry observed.
        eq_jac_min_sv_est: A lower bound on the smallest singular value
            of ``J_eq`` estimated from the Cholesky factor of
            ``J_eq J_eq^T``.  Small values indicate near rank-deficiency.
        ls_alpha_min: Smallest line-search step accepted across the run.
        tail_ls_failures: Consecutive line-search failure count at
            termination.  Non-zero values suggest stagnation.
        n_bound_fix_solves: Total number of non-trivial bound-fixing inner
            solves that actually ran (no-op passes are skipped via
            ``jax.lax.cond`` and do not count).  Growing unboundedly is a
            sign that the bound active set never stabilises.
        max_bound_fixed: Largest number of variables pinned to their box
            bounds across all iterations (from the final per-iteration
            ``free_mask``).
        max_active_ineq: Largest number of active general inequalities
            observed across iterations.
        n_merit_regressions: Number of iterations where the L1 merit
            function value *increased* despite the line search reporting
            success.  A non-zero count points to merit-function
            mis-calibration (too-small ``rho``) or line-search slippage.
        n_qp_budget_exhausted: Number of SQP steps where the QP
            active-set loop hit ``qp_max_iter`` (i.e. exited because
            the budget ran out, not because of clean convergence or a
            ping-pong short-circuit).  A non-zero count usually
            indicates degeneracy, multiplier-noise cycling, or an
            LPEC-A over-prediction that was not caught by the trust
            gate; consider raising ``mult_drop_floor`` or tightening
            ``lpeca_trust_threshold``.
        n_qp_ping_pong: Number of SQP steps where the QP loop's
            anti-cycling ping-pong detector fired.  These are *good*
            short-circuits (the loop avoided wasting its full budget)
            but a chronic non-zero value still points to a degenerate
            constraint pair worth investigating.
        max_qp_iterations: Peak active-set iteration count observed
            across all SQP steps.  Useful for sizing
            ``qp_max_iter``: if this is consistently equal to
            ``qp_max_iter`` the budget is too tight.
        max_qp_active_size: Peak ``|active_set|`` (general
            inequalities only) observed across all SQP steps.
        n_lpeca_bypassed: Number of SQP steps where the LPEC-A
            prediction was skipped, either because of the warm-up
            window (``step_count < lpeca_warmup_steps``) or because
            the trust gate vetoed it (``rho_bar > lpeca_trust_threshold``).
            Always 0 when ``active_set_method == "expand"``.
        n_lpeca_capped: Number of SQP steps where the LPEC-A
            rank-aware size cap truncated the prediction.
        n_lpeca_bounds_prefixed: Cumulative count of box-bound
            pre-fixes contributed by LPEC-A across all SQP steps
            (summed over the bound predictions that survived the
            trust / warm-up gates and were installed as the initial
            ``free_mask`` for the bound-fixing loop).  Always 0 when
            ``active_set_method == "expand"`` or when
            ``lpeca_predict_bounds=False``.
        n_proj_refinements: Cumulative number of M-metric projection
            refinement rounds taken across all inner solves performed
            by ``MinresQLPSolver`` over the run.  Always 0 for
            null-space CG / CRAIG.
        max_proj_residual: High-water mark of the post-refinement
            constraint residual ``||A d - b||`` reported by
            ``MinresQLPSolver`` on the *accepted* QP direction across
            all SQP steps.
        n_divergence_blowups: Total number of merit blowup events
            observed across the run, whether or not the patience
            threshold was eventually reached.
        divergence_triggered: True when the best-iterate divergence
            rollback fired at least once during the run.
        min_projected_grad_norm: Low-water mark of the inner solver's
            projected-gradient norm ``||W̃_k g_k||`` across the run.
            Always ``inf`` for inner solvers other than
            ``HRInexactSTCG``.  Surfaced for post-hoc inspection of
            how close the run actually got to KKT in the
            inexact-projection sense, regardless of whether
            ``use_inexact_stationarity`` was on.
        n_steps_inexact_below_classical: Number of iterations where the
            inner solver's projected-gradient norm was strictly smaller
            than the classical Lagrangian gradient norm.  A high count
            indicates that multiplier-recovery noise is the limiting
            factor and the user might benefit from
            ``use_inexact_stationarity=True`` paired with
            ``HRInexactSTCG``.
    """

    n_qp_inner_failures: Int[Array, ""]
    n_ls_failures: Int[Array, ""]
    n_lbfgs_skips: Int[Array, ""]
    n_nan_directions: Int[Array, ""]
    max_gamma: Scalar
    min_diag: Scalar
    max_diag: Scalar
    eq_jac_min_sv_est: Scalar
    ls_alpha_min: Scalar
    tail_ls_failures: Int[Array, ""]
    n_bound_fix_solves: Int[Array, ""]
    max_bound_fixed: Int[Array, ""]
    max_active_ineq: Int[Array, ""]
    n_merit_regressions: Int[Array, ""]
    n_qp_budget_exhausted: Int[Array, ""]
    n_qp_ping_pong: Int[Array, ""]
    max_qp_iterations: Int[Array, ""]
    max_qp_active_size: Int[Array, ""]
    n_lpeca_bypassed: Int[Array, ""]
    n_lpeca_capped: Int[Array, ""]
    n_lpeca_bounds_prefixed: Int[Array, ""]
    n_proj_refinements: Int[Array, ""]
    max_proj_residual: Scalar
    n_divergence_blowups: Int[Array, ""]
    divergence_triggered: Bool[Array, ""]
    min_projected_grad_norm: Scalar
    n_steps_inexact_below_classical: Int[Array, ""]


def _init_diagnostics() -> SLSQPDiagnostics:
    """Construct a zero-initialized ``SLSQPDiagnostics`` container."""
    return SLSQPDiagnostics(  # ty: ignore[invalid-return-type]
        n_qp_inner_failures=jnp.array(0),
        n_ls_failures=jnp.array(0),
        n_lbfgs_skips=jnp.array(0),
        n_nan_directions=jnp.array(0),
        max_gamma=jnp.array(0.0),
        min_diag=jnp.array(jnp.inf),
        max_diag=jnp.array(0.0),
        eq_jac_min_sv_est=jnp.array(jnp.inf),
        ls_alpha_min=jnp.array(1.0),
        tail_ls_failures=jnp.array(0),
        n_bound_fix_solves=jnp.array(0),
        max_bound_fixed=jnp.array(0),
        max_active_ineq=jnp.array(0),
        n_merit_regressions=jnp.array(0),
        n_qp_budget_exhausted=jnp.array(0),
        n_qp_ping_pong=jnp.array(0),
        max_qp_iterations=jnp.array(0),
        max_qp_active_size=jnp.array(0),
        n_lpeca_bypassed=jnp.array(0),
        n_lpeca_capped=jnp.array(0),
        n_lpeca_bounds_prefixed=jnp.array(0),
        n_proj_refinements=jnp.array(0),
        max_proj_residual=jnp.array(0.0),
        n_divergence_blowups=jnp.array(0),
        divergence_triggered=jnp.array(False),
        min_projected_grad_norm=jnp.asarray(jnp.inf),
        n_steps_inexact_below_classical=jnp.array(0),
    )


def get_diagnostics(state: "SLSQPState") -> SLSQPDiagnostics:
    """Return the ``SLSQPDiagnostics`` accumulator from a final state.

    Use this after ``optimistix.minimise`` (or a manual ``solve / step``
    loop) to inspect solver health indicators that the Optimistix result
    code alone does not expose.  See :class:`SLSQPDiagnostics` for field
    meanings.
    """
    return state.diagnostics


class SLSQPState(eqx.Module):
    """State for the SLSQP solver.

    This is a JAX PyTree (via eqx.Module) that holds all mutable state
    needed across SLSQP iterations.

    Attributes:
        step_count: Current iteration number.
        f_val: Current objective function value f(x_k).
        grad: Gradient of objective at current point.
        eq_val: Equality constraint values c_eq(x_k).
        ineq_val: Inequality constraint values c_ineq(x_k).
        eq_jac: Jacobian of equality constraints at x_k.
        ineq_jac: Jacobian of inequality constraints at x_k.
        lbfgs_history: L-BFGS history for matrix-free Hessian approximation.
        multipliers_eq: Lagrange multipliers for equality constraints.
        multipliers_ineq: Lagrange multipliers for inequality constraints.
        prev_grad_lagrangian: Previous Lagrangian gradient (for L-BFGS update).
        grad_lagrangian: Current gradient of the Lagrangian evaluated at
            the accepted iterate using the *blended* multipliers
            consistent with the line-search step size.  Reused by
            ``terminate`` so the stationarity check does not fall out
            of sync with the L-BFGS secant pair.
        merit_penalty: Current penalty parameter for L1 merit function.
        bound_jac: Constant Jacobian for bound constraints (computed once in init).
        qp_iterations: Total accumulated QP active-set iterations across all steps.
        qp_converged: Whether the most recent QP solve converged.
        prev_active_set: Active inequality constraint set from the previous QP solve,
            used for warm-starting the next QP subproblem.
        termination_code: Granular ``slsqp_jax.RESULTS`` classification
            (``successful`` / ``merit_stagnation`` / ``line_search_failure``
            / ``iterate_blowup`` / ``infeasible`` / ``qp_subproblem_failure``
            / ``nonlinear_max_steps_reached`` / ``nonfinite``). Surfaced
            via ``Solution.stats["slsqp_result"]``; ``Solution.result``
            itself remains the coarse ``optx.RESULTS`` code accepted by
            optimistix's driver.
    """

    # Iteration tracking
    step_count: Int[Array, ""]

    # Current function values and gradients
    f_val: Scalar
    grad: Vector

    # Constraint information
    eq_val: Float[Array, " m_eq"]
    ineq_val: Float[Array, " m_ineq"]
    eq_jac: Float[Array, "m_eq n"]
    ineq_jac: Float[Array, "m_ineq n"]

    # L-BFGS history for matrix-free Hessian approximation (O(kn) storage)
    lbfgs_history: LBFGSHistory

    # Lagrange multipliers from QP solution
    multipliers_eq: Float[Array, " m_eq"]
    multipliers_ineq: Float[Array, " m_ineq"]

    # Previous Lagrangian gradient for L-BFGS y = grad_L_new - grad_L_old
    prev_grad_lagrangian: Vector

    # Current Lagrangian gradient at the accepted iterate, computed with
    # the *blended* multipliers (matching the L-BFGS secant pair).  Reused
    # by ``terminate`` for stationarity so the check is consistent with
    # what ``step`` saw.
    grad_lagrangian: Vector

    # Merit function penalty parameter
    merit_penalty: Scalar

    # Bound constraint Jacobian (constant, computed once in init)
    bound_jac: Float[Array, "m_bounds n"]

    # QP solver statistics
    qp_iterations: Int[Array, ""]
    qp_converged: Bool[Array, ""]

    # Active-set warm-starting: carry the QP active set across iterations
    # to promote multiplier stability (Wright, SIAM J. Optim., 2002, Sec. 8)
    prev_active_set: Bool[Array, " m_ineq"]

    # Consecutive QP failure tracking for escalating L-BFGS recovery
    consecutive_qp_failures: Int[Array, ""]

    # Consecutive line search failure tracking for escalating L-BFGS recovery
    consecutive_ls_failures: Int[Array, ""]

    # Zero-step convergence detection: consecutive iterations where the QP
    # converged but returned ||d|| < atol, indicating the current point
    # satisfies the QP's KKT conditions.
    consecutive_zero_steps: Int[Array, ""]
    qp_optimal: Bool[Array, ""]

    # Merit-based stagnation detection
    best_merit: Scalar
    steps_without_improvement: Int[Array, ""]
    stagnation: Bool[Array, ""]
    last_alpha: Scalar

    # Norm of the projected gradient ``||W̃_k g_k||`` from the most
    # recent inner solve, surfaced through ``QPResult.projected_grad_norm``.
    # Always ``inf`` for inner solvers that do not produce this value
    # (everything other than ``HRInexactSTCG``).  Used by
    # ``terminate()`` when ``use_inexact_stationarity=True`` so that
    # the run can declare convergence at the inner solver's noise
    # floor instead of waiting for the exact Lagrangian gradient.
    last_projected_grad_norm: Scalar

    # Whether the most recent line search reported success (Armijo satisfied
    # or at least a strictly decreasing merit).  Used by ``terminate`` to
    # distinguish genuine convergence from tiny-alpha stagnation.
    ls_success: Bool[Array, ""]

    # Whether the most recent line search failed AND the escalated
    # identity reset also failed (``consecutive_ls_failures`` exceeded
    # ``2 * ls_failure_patience``).  Triggers early termination with
    # ``RESULTS.line_search_failure`` to surface chronic LS failure.
    ls_fatal: Bool[Array, ""]

    # Whether the QP subproblem has failed for ``2 * qp_failure_patience``
    # consecutive iterations.  Mirrors ``ls_fatal``: the L-BFGS soft- /
    # identity-reset escalation could not produce a usable QP direction,
    # and we should terminate with ``RESULTS.qp_subproblem_failure``.
    qp_fatal: Bool[Array, ""]

    # Best-iterate divergence rollback (see :attr:`SLSQPConfig` /
    # :attr:`ToleranceConfig`).  ``best_x`` is the iterate that achieved
    # ``best_merit``.  When the merit blows up by ``divergence_factor``
    # or returns NaN/Inf for ``divergence_patience`` consecutive steps,
    # ``step()`` overwrites the returned iterate with ``best_x`` and
    # sets ``diverging=True``, which routes ``terminate()`` to
    # ``RESULTS.iterate_blowup``.
    best_x: Vector
    blowup_count: Int[Array, ""]
    diverging: Bool[Array, ""]

    # Granular termination classification using ``slsqp_jax.RESULTS``
    # (see :mod:`slsqp_jax.results`).  Optimistix's ``iterative_solve``
    # driver only accepts members of ``optx.RESULTS`` from
    # ``terminate()``, so the public ``Solution.result`` is the coarse
    # base-class code.  This field carries the finer
    # ``slsqp_jax.RESULTS`` classification (e.g. ``merit_stagnation``,
    # ``line_search_failure``, ``iterate_blowup``,
    # ``qp_subproblem_failure``, ``infeasible``) and is surfaced via
    # ``Solution.stats["slsqp_result"]`` in :meth:`SLSQP.postprocess`.
    # Pre-termination its value is ``RESULTS.successful`` (i.e. the
    # loop is still running, no failure latched).
    termination_code: Any

    # Diagnostic accumulators (see :class:`SLSQPDiagnostics`).
    diagnostics: SLSQPDiagnostics


__all__ = [
    "InnerSolveResult",
    "ProjectionContext",
    "QPState",
    "QPSolverResult",
    "QPResult",
    "SLSQPDiagnostics",
    "SLSQPState",
    "_init_diagnostics",
    "get_diagnostics",
]
