"""Signal evaluators and the registry that wires them in.

A signal is the diagnostics layer's atomic unit of evidence: an
evaluator function decides at fire time whether a known suspicious
pattern is present, and (when it fires) immediately builds the
linalg artifacts it needs from the live state so the report renderer
can surface them without re-running the solver.

The plan distinguishes two flavours of evaluator:

* **Per-step evaluators** receive ``(state, summary, summaries)`` and
  are called inside the runner's iteration loop while the live
  ``SLSQPState`` is in scope.  Each evaluator splits into a cheap
  predicate (``cheap_predicate(summary) -> bool``) that runs every
  step from host scalars only, and an expensive
  ``build_artifacts(state) -> dict`` that pulls device arrays only
  when the predicate fires.
* **End-of-run evaluators** receive
  ``(final_state, summaries, coarse_result)`` and run once after the
  loop exits.  They never see a non-final ``SLSQPState``; trajectory
  inspection happens against the ``StepSummary`` list instead.

Both flavours are registered into :data:`PER_STEP_EVALUATORS` /
:data:`END_OF_RUN_EVALUATORS` (Phase 2 fills them in; Phase 1 ships
empty so the runner has the same call surface from day one).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import jax.numpy as jnp
import numpy as np

from slsqp_jax.diagnostics.playbook import confidence_for, magnitude_for

if TYPE_CHECKING:
    from slsqp_jax.diagnostics.records import StepSummary
    from slsqp_jax.slsqp import SLSQP
    from slsqp_jax.state import SLSQPState


Specificity = Literal["specific", "ambiguous", "generic"]
Magnitude = Literal["marginal", "moderate", "extreme"]
Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class EvalContext:
    """Bundle of solver-derived constants that evaluators need.

    Threaded through the runner so per-step + end-of-run evaluators
    can read the user's :attr:`rtol` / :attr:`atol` (e.g. for the
    ``multiplier_recovery_noise`` test which compares against
    ``rtol * max(|L|, 1)``) without taking a stateful dependency on
    the solver instance.

    Attributes:
        solver: The :class:`slsqp_jax.SLSQP` instance the run used.
            Carried because some artifacts (e.g. the initial L-BFGS
            memory size) live on the solver, not the state.
        rtol: Relative-stationarity tolerance used by ``terminate()``.
        atol: Primal-feasibility tolerance used by ``terminate()``.
        max_steps: Maximum number of SQP iterations the manual loop
            was allowed to run.
    """

    solver: "SLSQP"
    rtol: float
    atol: float
    max_steps: int


@dataclass(frozen=True)
class Signal:
    """Single evidence-backed hypothesis emitted by an evaluator.

    The wording contract is *hypothesis with evidence*, not verdict.
    ``summary`` reads "X looks suspicious because Y"; ``detail``
    expands on the mechanism; ``suggestions`` are starting points for
    the user to investigate, not prescriptions to apply blindly.

    Attributes:
        name: Stable machine-readable identifier (used for the
            playbook scope filter and the cap-policy enforcement
            test).
        specificity: How uniquely this pattern points at one cause.
            ``specific`` = essentially diagnostic for one cause;
            ``ambiguous`` = consistent with two or three named causes;
            ``generic`` = consistent with many causes.  Static
            editorial tag set at registration.
        magnitude: How far past threshold the evidence landed.
            ``marginal`` = within 10x of threshold;
            ``moderate`` = 10x-100x past threshold;
            ``extreme`` = >100x past threshold.  Computed from
            ``evidence`` at fire time.
        confidence: Derived from ``(specificity, magnitude)`` via the
            lookup table in :func:`slsqp_jax.diagnostics.playbook.confidence_for`.
            The single ranking axis the report sorts by.
        summary: One-line "X looks suspicious because Y" framing.
        detail: Multi-paragraph mechanism explanation.
        evidence: Numeric evidence keyed by short name.  Each value
            is the actual measurement; the threshold is documented in
            the evaluator's docstring.
        suggestions: Concrete starting-point fixes (config knobs,
            alternative inner solvers, etc).
        artifacts: Heavy linalg objects the evaluator computed
            immediately from the live state when it fired.  May be
            empty.
        offending_step: 1-indexed step the signal fired on, or ``None``
            for end-of-run evaluators that scope over the whole run.
    """

    name: str
    specificity: Specificity
    magnitude: Magnitude
    confidence: Confidence
    summary: str
    detail: str
    evidence: dict[str, float | int] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    artifacts: dict[str, np.ndarray] = field(default_factory=dict)
    offending_step: Optional[int] = None


# The two flavours of evaluator callable signature.  Kept as bare
# Callable aliases (rather than typing.Protocol) so subclassing /
# duck-typing in tests is trivial.  Both flavours take the
# :class:`EvalContext` as the first positional arg so they can read
# tolerances / solver config without a separate import.
PerStepEvaluator = Callable[
    [EvalContext, "SLSQPState", "StepSummary", list["StepSummary"]],
    Optional[Signal],
]
EndOfRunEvaluator = Callable[
    [EvalContext, "SLSQPState", list["StepSummary"], Any],
    Optional[Signal],
]


@dataclass(frozen=True)
class SignalRegistration:
    """Static registration record for a signal evaluator.

    The ``cap-policy`` enforcement test
    (``tests/diagnostics/test_registry.py``) iterates the registries
    of :data:`PER_STEP_EVALUATORS` / :data:`END_OF_RUN_EVALUATORS`
    and asserts that for every ``name`` here there exists a
    ``test_signal_<name>_synthetic`` and ``test_signal_<name>_integration``
    in the diagnostics test module.  ``specificity`` is captured here
    so it stays adjacent to the evaluator and the docstring (single
    source of truth).
    """

    name: str
    specificity: Specificity
    evaluator: Callable
    flavour: Literal["per_step", "end_of_run"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Phase 1 ships empty registries; Phase 2 populates them via
# :func:`register_evaluator`.  These are the canonical lists the
# runner walks each iteration / at end-of-run.

PER_STEP_EVALUATORS: tuple[PerStepEvaluator, ...] = ()
END_OF_RUN_EVALUATORS: tuple[EndOfRunEvaluator, ...] = ()

# Parallel registry of the static metadata for each evaluator.  Kept
# as a list (not a tuple) so :func:`register_evaluator` can append in
# place without reconstructing the public per-step / end-of-run
# tuples on every call.
SIGNAL_REGISTRY: list[SignalRegistration] = []


def register_evaluator(
    name: str,
    *,
    specificity: Specificity,
    flavour: Literal["per_step", "end_of_run"],
    evaluator: Callable,
) -> None:
    """Append an evaluator to the appropriate registry.

    Phase 2 calls this at module import time for each starter signal.
    The function is idempotent on ``name`` (re-registering the same
    name overwrites the previous record), which simplifies tests
    that exercise alternative thresholds.
    """
    global PER_STEP_EVALUATORS, END_OF_RUN_EVALUATORS

    # Replace any prior registration with the same name.
    for i, reg in enumerate(SIGNAL_REGISTRY):
        if reg.name == name:  # pragma: no cover
            SIGNAL_REGISTRY.pop(i)
            break
    SIGNAL_REGISTRY.append(
        SignalRegistration(
            name=name,
            specificity=specificity,
            evaluator=evaluator,
            flavour=flavour,
        )
    )

    PER_STEP_EVALUATORS = tuple(
        reg.evaluator for reg in SIGNAL_REGISTRY if reg.flavour == "per_step"
    )
    END_OF_RUN_EVALUATORS = tuple(
        reg.evaluator for reg in SIGNAL_REGISTRY if reg.flavour == "end_of_run"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(arr: Any) -> np.ndarray:
    """Materialise a JAX device array to a NumPy host array.

    Used by :func:`build_artifacts` paths to pull the (small) linalg
    objects a fired signal wants the user to be able to dig into
    further.  Idempotent for inputs that are already NumPy arrays.
    """
    return np.asarray(arr)


def _build_signal(
    *,
    name: str,
    specificity: Specificity,
    summary: str,
    detail: str,
    evidence: dict[str, float | int],
    threshold_ratio: float,
    suggestions: list[str],
    artifacts: Optional[dict[str, np.ndarray]] = None,
    offending_step: Optional[int] = None,
) -> Signal:
    """Construct a :class:`Signal` with derived ``magnitude`` /
    ``confidence`` from a ratio-to-threshold input.

    ``threshold_ratio`` is the largest ratio of any single evidence
    value to its documented threshold (always ``>= 1`` whenever the
    signal fires).  Centralising the magnitude / confidence
    derivation here means each evaluator just computes the worst
    ratio and hands it off, instead of duplicating the lookup table.
    """
    magnitude = magnitude_for(threshold_ratio)
    confidence = confidence_for(specificity, magnitude)
    return Signal(
        name=name,
        specificity=specificity,
        magnitude=magnitude,
        confidence=confidence,
        summary=summary,
        detail=detail,
        evidence=dict(evidence),
        suggestions=list(suggestions),
        artifacts=dict(artifacts or {}),
        offending_step=offending_step,
    )


def _just_crossed_below(
    summaries: list["StepSummary"], attr: str, threshold: float
) -> bool:
    """``True`` iff the most-recent summary has ``attr < threshold``
    AND the previous summary did not (or there is no previous summary).

    Used to fire a signal exactly once on the *first* crossing so the
    report does not list the same signal at every step it remains
    tripped.
    """
    if not summaries:
        return False  # pragma: no cover
    cur = float(getattr(summaries[-1], attr))
    if not (cur < threshold):
        return False
    if len(summaries) == 1:
        return True  # pragma: no cover
    prev = float(getattr(summaries[-2], attr))
    return not (prev < threshold)


def _streak_just_reached(
    summaries: list["StepSummary"],
    predicate: Callable[["StepSummary"], bool],
    n: int,
) -> bool:
    """``True`` iff the last ``n`` summaries all satisfy ``predicate``
    AND the immediately-prior summary did not (or did not exist).

    Centralises the consecutive-N pattern used by
    ``lbfgs_conditioning_extreme`` and any future streak signal so a
    signal fires once at the moment the streak first becomes long
    enough, not at every subsequent step.
    """
    if len(summaries) < n:
        return False
    tail = summaries[-n:]
    if not all(predicate(s) for s in tail):
        return False
    if len(summaries) == n:
        return True  # pragma: no cover
    return not predicate(summaries[-n - 1])


# ---------------------------------------------------------------------------
# Per-step evaluators
# ---------------------------------------------------------------------------


# Threshold below which the cumulative low-water singular-value
# estimate of ``J_eq`` is treated as a LICQ violation.  Sourced from
# the AGENTS.md "QP failure diagnostics" + ``min_projected_grad_norm``
# discussions: the projected-CG / CRAIG inner solvers start to suffer
# from regularisation bias once ``sigma_min(J_eq) < 1e-8``.
_RANK_DEF_THRESHOLD = 1e-8


def _eval_eq_jacobian_rank_deficient(
    ctx: EvalContext,
    state: "SLSQPState",
    summary: "StepSummary",
    summaries: list["StepSummary"],
) -> Optional[Signal]:
    """Per-step: equality Jacobian appears rank-deficient.

    Cheap predicate (scalar): the cumulative low-water mark
    ``eq_jac_min_sv_est`` carried on
    :attr:`SLSQPDiagnostics.eq_jac_min_sv_est` just dropped below
    :data:`_RANK_DEF_THRESHOLD` (= ``1e-8``) at this step.  Because
    the diagnostics counter is a low-water mark rather than a
    per-step value, the "first crossing" predicate is the right way
    to fire only once per run.

    Build_artifacts (only invoked if the predicate fires): pulls
    ``state.eq_jac`` (``m_eq * n * 8`` bytes) to host, computes
    ``J J^T`` (``m_eq^2 * 8`` bytes — tiny), and the singular values
    of ``J_eq`` for the user to inspect.

    Specificity: ``specific``.  The (sigma_min < 1e-8) signature
    essentially uniquely points at LICQ violation / projector
    instability.

    Hypothesis: the equality Jacobian is (near-)rank-deficient at the
    current iterate, which makes the null-space projector
    ``v - A^T (A A^T)^{-1} A v`` numerically unstable.  See
    ``AGENTS.md`` "Cholesky-based projection" and the
    ``CraigInnerSolver`` notes for context.
    """
    # Nothing to fire when there are no equality constraints.
    if int(state.eq_val.shape[0]) == 0:
        return None
    if not _just_crossed_below(summaries, "eq_jac_min_sv_est", _RANK_DEF_THRESHOLD):
        return None

    # ---- build_artifacts (live state in scope) ----
    J_eq = _to_numpy(state.eq_jac)
    JJT = _to_numpy(state.eq_jac @ state.eq_jac.T)
    try:
        sv = _to_numpy(jnp.linalg.svd(state.eq_jac, compute_uv=False))
    except Exception:  # pragma: no cover  -- defensive
        sv = np.linalg.svd(J_eq, compute_uv=False)
    sigma_min = float(np.min(sv))
    sigma_max = float(np.max(sv))

    threshold_ratio = _RANK_DEF_THRESHOLD / max(sigma_min, 1e-300)

    detail = (
        "The equality Jacobian J_eq has min singular value "
        f"{sigma_min:.3e} (cumulative low-water "
        f"{summary.eq_jac_min_sv_est:.3e}).  Below ~1e-8 the "
        "null-space projector v - J_eq^T (J_eq J_eq^T)^{-1} J_eq v "
        "is numerically unstable: the Cholesky regularisation "
        "epsilon * I starts to bias the projection, and CRAIG / "
        "MINRES-QLP can stall.  This pattern is essentially "
        "diagnostic for a LICQ violation, but it can also surface "
        "when the equality constraints are well-posed analytically "
        "yet badly scaled."
    )
    return _build_signal(
        name="eq_jacobian_rank_deficient",
        specificity="specific",
        summary=(
            f"J_eq looks suspicious because its smallest singular "
            f"value ({sigma_min:.3e}) is below 1e-8 — the null-space "
            "projector is numerically unstable here."
        ),
        detail=detail,
        evidence={
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "threshold": _RANK_DEF_THRESHOLD,
            "step": summary.step_count,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Check that the equality constraints are linearly "
            "independent at the iterate (LICQ).",
            "If the constraints are well-posed but badly scaled, "
            "rescale them so each row of J_eq has comparable norm.",
            "Try MinresQLPSolver (handles indefinite/singular saddle-"
            "point systems natively) or CraigInnerSolver (no "
            "epsilon * I regularisation in the projector).",
        ],
        artifacts={
            "J_eq": J_eq,
            "JJT": JJT,
            "singular_values": sv,
        },
        offending_step=int(summary.step_count),
    )


_LBFGS_KAPPA_THRESHOLD = 1e6
_LBFGS_KAPPA_STREAK = 3
# Single-step "burst" trigger: catches catastrophic one-shot blow-ups
# that the streak gate misses (the L-BFGS reset chain typically clamps
# back to identity within one or two steps after the spike, so the
# streak never reaches its 3-step minimum).  Two orders of magnitude
# above ``_LBFGS_KAPPA_THRESHOLD`` so a routine "above-but-recovering"
# kappa does not also fire the burst clause.
_LBFGS_KAPPA_BURST = 1e8


def _eval_lbfgs_conditioning_extreme(
    ctx: EvalContext,
    state: "SLSQPState",
    summary: "StepSummary",
    summaries: list["StepSummary"],
) -> Optional[Signal]:
    """Per-step: L-BFGS B0 diagonal condition number is extreme.

    Cheap predicate (scalar): either of two clauses fires.

    1. **Streak**: the L-BFGS diagonal condition-number proxy
       ``max_diag / min_diag`` (carried on
       :attr:`StepSummary.diag_kappa`) has been above
       :data:`_LBFGS_KAPPA_THRESHOLD` (= ``1e6``) for the last
       :data:`_LBFGS_KAPPA_STREAK` (= ``3``) consecutive steps.
       Three is the ``soft reset`` patience the L-BFGS reset chain
       uses; once the streak reaches that length the chain triggers a
       soft reset, so it is also the right point to flag the user.
    2. **Burst**: the *current* step's ``diag_kappa`` is above
       :data:`_LBFGS_KAPPA_BURST` (= ``1e8``), which catches
       catastrophic one-shot blow-ups (e.g. ``kappa = 1e+09`` for a
       single step before the reset chain clamps back to identity).
       The streak gate cannot catch these because the post-spike
       identity reset breaks the streak after one step.

    Build_artifacts (only invoked if the predicate fires): pulls the
    full L-BFGS diagonal (``n * 8`` bytes — fits in L1 even for very
    large ``n``) plus the scalar gamma / history count.

    Specificity: ``ambiguous``.  Extreme L-BFGS conditioning is
    consistent with several causes (stale curvature pairs, post-
    identity-reset skip lock, bad initial scaling), so the report
    cannot pin it on one without more evidence.
    """
    burst_fired = float(summary.diag_kappa) > _LBFGS_KAPPA_BURST
    streak_fired = _streak_just_reached(
        summaries,
        lambda s: s.diag_kappa > _LBFGS_KAPPA_THRESHOLD,
        _LBFGS_KAPPA_STREAK,
    )
    if not (streak_fired or burst_fired):
        return None
    mode = "burst" if burst_fired else "streak"

    diagonal = _to_numpy(state.lbfgs_history.diagonal)
    gamma = float(state.lbfgs_history.gamma)
    history_count = int(state.lbfgs_history.count)

    if burst_fired:
        threshold_ratio = float(summary.diag_kappa) / _LBFGS_KAPPA_BURST
        clause_summary = (
            f"a single-step burst above {_LBFGS_KAPPA_BURST:.0e} "
            "(catastrophic one-shot blow-up)"
        )
    else:
        threshold_ratio = float(summary.diag_kappa) / _LBFGS_KAPPA_THRESHOLD
        clause_summary = (
            f"a streak of {_LBFGS_KAPPA_STREAK} consecutive iterations "
            f"above {_LBFGS_KAPPA_THRESHOLD:.0e}"
        )

    detail = (
        f"L-BFGS B0 diagonal condition number kappa(B0) = "
        f"{summary.diag_kappa:.3e} (min={summary.min_diag:.3e}, "
        f"max={summary.max_diag:.3e}) tripped the "
        f"{mode!r} clause: {clause_summary}.  "
        "Extreme conditioning is consistent with several causes: "
        "(i) stale curvature pairs whose s/y vectors are no longer "
        "informative at the current iterate, (ii) the post-identity-"
        "reset skip lock the AGENTS.md describes (lowering "
        "``lbfgs_skip_floor`` from 1e-8 to 1e-12 was the fix), "
        "(iii) badly-scaled variables, and (iv) a fallback projected-"
        "gradient direction (after a QP-budget exhaustion) being "
        "appended to the L-BFGS history as if it were a Newton "
        "secant.  Each cause has a different fix; check "
        "``state.lbfgs_history.count`` and the ``n_lbfgs_skips`` "
        "counter to disambiguate."
    )
    return _build_signal(
        name="lbfgs_conditioning_extreme",
        specificity="ambiguous",
        summary=(
            f"The L-BFGS B0 diagonal looks suspicious because its "
            f"condition number is {summary.diag_kappa:.3e} ({mode} "
            f"clause: {clause_summary})."
        ),
        detail=detail,
        evidence={
            "diag_kappa": summary.diag_kappa,
            "min_diag": summary.min_diag,
            "max_diag": summary.max_diag,
            "threshold": _LBFGS_KAPPA_THRESHOLD,
            "burst_threshold": _LBFGS_KAPPA_BURST,
            "streak": _LBFGS_KAPPA_STREAK,
            # ``burst_clause = 1`` ⇔ the single-step burst fired,
            # ``burst_clause = 0`` ⇔ the streak fired.  Encoded as int
            # because :attr:`Signal.evidence` is typed numeric-only.
            "burst_clause": 1 if burst_fired else 0,
            "step": summary.step_count,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Try ``use_exact_hvp_in_qp=True`` so the QP no longer "
            "depends on the L-BFGS approximation.",
            "If many curvature pairs are being skipped "
            "(``n_lbfgs_skips`` is large), inspect the secant pairs "
            "for noise.",
            "Consider rescaling the variables so coordinate-wise "
            "curvatures are within a few orders of magnitude.",
        ],
        artifacts={
            "diagonal": diagonal,
            "gamma": np.asarray(gamma),
            "history_count": np.asarray(history_count),
        },
        offending_step=int(summary.step_count),
    )


# ---------------------------------------------------------------------------
# End-of-run evaluators
# ---------------------------------------------------------------------------


def _eval_multiplier_recovery_noise(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: noise-floor stationarity stall pattern.

    The textbook signature documented in ``AGENTS.md``:

    * ``min_projected_grad_norm / |L| < rtol`` (the inner solver's
      noise-aware stationarity proxy already passed)
    * ``|grad_L| / |L| > rtol`` (the classical stationarity proxy
      did not)
    * ``n_steps_inexact_below_classical / step_count > 0.5`` (more
      than half the iterations had a strictly-cleaner projected
      gradient than classical, i.e. multiplier-recovery noise was
      the limiting factor across the bulk of the run)

    All three together are essentially diagnostic for the noise-floor
    stationarity stall mode that ``HRInexactSTCG`` +
    ``use_inexact_stationarity=True`` is designed to rescue.

    Specificity: ``specific``.
    """
    if not summaries:
        return None  # pragma: no cover
    diag = final_state.diagnostics
    last = summaries[-1]
    L_abs = max(abs(last.lagrangian_value), 1.0)
    rtol = ctx.rtol

    classical_ratio = last.grad_lagrangian_norm / L_abs
    proj_ratio = float(diag.min_projected_grad_norm) / L_abs
    n_inexact_below = int(diag.n_steps_inexact_below_classical)
    fraction_below = n_inexact_below / max(last.step_count, 1)

    if not (proj_ratio < rtol):
        return None
    if not (classical_ratio > rtol):
        return None  # pragma: no cover
    if not (fraction_below > 0.5):
        return None  # pragma: no cover

    # Magnitude reflects how clean the projected ratio is relative to
    # rtol versus how stuck the classical ratio is.  We use the
    # ratio of classical-to-projected; the further apart they are,
    # the more obvious the noise-floor diagnosis.
    threshold_ratio = classical_ratio / max(proj_ratio, 1e-300)

    detail = (
        "The inner solver's projected-gradient norm low-water mark "
        f"satisfies ||W g|| / |L| = {proj_ratio:.3e} < rtol "
        f"({rtol:.1e}), while the classical Lagrangian gradient "
        f"ratio ||grad_L|| / |L| = {classical_ratio:.3e} is still "
        "above rtol.  This is the textbook signature of multiplier-"
        "recovery noise contaminating the classical stationarity "
        'test (see AGENTS.md "Inexact stationarity disjunct").  '
        f"More than half the iterations ({n_inexact_below} / "
        f"{last.step_count}) had ||W g|| < ||grad_L||, confirming "
        "the pattern is persistent rather than a single-step "
        "artefact.  HRInexactSTCG paired with "
        "``use_inexact_stationarity=True`` directly rescues this "
        "mode."
    )
    return _build_signal(
        name="multiplier_recovery_noise",
        specificity="specific",
        summary=(
            f"The classical stationarity test ({classical_ratio:.3e}) "
            f"looks suspicious because the inner-solver projected-"
            f"gradient ratio ({proj_ratio:.3e}) already passed rtol "
            f"({rtol:.1e}) but the recovered Lagrangian gradient is "
            "noise-dominated."
        ),
        detail=detail,
        evidence={
            "classical_ratio": classical_ratio,
            "projected_ratio": proj_ratio,
            "rtol": rtol,
            "n_steps_inexact_below_classical": n_inexact_below,
            "fraction_below": fraction_below,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Switch the inner solver to HRInexactSTCG (e.g. "
            "``inner_solver=HRInexactSTCG(inner=ProjectedCGCholesky())``).",
            "Set ``use_inexact_stationarity=True`` so the convergence "
            "test admits the projected-gradient disjunct.",
        ],
        artifacts={},
        offending_step=None,
    )


# Hard floor used as a fallback when the solver's LS budget cannot be
# read off ``EvalContext`` (e.g. synthetic tests that use a stand-in
# solver).  ``2 ** -20 ~= 9.5e-7`` is the SciPy/SLSQP default; multiply
# by 10 so a single backtracking step above the floor still triggers.
_LS_COLLAPSE_FALLBACK_FLOOR = 10.0 * (2.0**-20)


def _ls_floor_for(ctx: EvalContext) -> float:
    """Return ``10 * 2**-line_search.max_steps`` for the active solver.

    The backtracking line search halves ``alpha`` at most
    ``line_search.max_steps`` times before giving up, so the smallest
    accepted ``alpha`` on success is ``2**-max_steps``.  We multiply
    by 10 so an LS that bottomed out within one backtracking step of
    the floor still trips the predicate (matches the ``magnitude_for``
    bucketing of "marginal" = within 10x).

    Falls back to :data:`_LS_COLLAPSE_FALLBACK_FLOOR` (the SciPy default
    ``max_steps == 20``) when the solver instance does not expose the
    ``line_search_max_steps`` accessor — this lets the synthetic test
    suite use a minimal stand-in solver without pulling in the full
    :class:`SLSQP` class.
    """
    solver = ctx.solver
    max_steps = getattr(solver, "line_search_max_steps", None)
    if not isinstance(max_steps, int) or max_steps <= 0:
        return _LS_COLLAPSE_FALLBACK_FLOOR
    return 10.0 * (2.0**-max_steps)


def _eval_line_search_collapse(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: line search collapsed to a tiny step at termination.

    Cheap predicate: the cumulative ``tail_ls_failures`` counter is
    > 0 (the line search failed at the last iteration) and the
    minimum line-search alpha across the run was at or below the
    LS floor ``10 * 2**-line_search.max_steps`` (i.e. within one
    backtracking step of the smallest ``alpha`` the line search will
    ever accept).  See :func:`_ls_floor_for` for the derivation —
    the previous hard-coded ``1e-10`` literal could never fire on
    the default ``LineSearchConfig.max_steps == 20`` solver because
    the LS floor itself is ``2**-20 ~= 9.5e-7``.

    Specificity: ``ambiguous`` — a stuck LS is consistent with a bad
    L-BFGS direction, a too-small merit penalty, or a non-descent
    QP step.

    Artifacts: the ``alpha`` trajectory near the offending step plus
    the final L-BFGS diagonal.  The exact L-BFGS diagonal at the
    offending step is not preserved (that would require checkpointing
    every step); use :func:`capture_state_at_step` if you need it.
    """
    if not summaries:
        return None  # pragma: no cover
    diag = final_state.diagnostics
    tail_failures = int(diag.tail_ls_failures)
    alpha_min = float(diag.ls_alpha_min)
    ls_floor = _ls_floor_for(ctx)
    if not (tail_failures > 0 and alpha_min <= ls_floor):
        return None

    # Locate the offending step (first time alpha drops to the LS
    # floor while the LS was reporting failure).
    offending_step = None
    for s in summaries:
        if (not s.ls_success) and s.last_alpha <= ls_floor:
            offending_step = int(s.step_count)
            break
    if offending_step is None:
        offending_step = int(summaries[-1].step_count)  # pragma: no cover

    # Trajectory window around the offending step (5 before + 5 after,
    # clipped) for the artifact.
    idx = max(0, offending_step - 1)
    window_lo = max(0, idx - 5)
    window_hi = min(len(summaries), idx + 6)
    alpha_window = np.asarray(
        [s.last_alpha for s in summaries[window_lo:window_hi]],
        dtype=float,
    )
    final_diagonal = _to_numpy(final_state.lbfgs_history.diagonal)

    threshold_ratio = ls_floor / max(alpha_min, 1e-300)

    detail = (
        f"The line search reached its smallest accepted alpha "
        f"({alpha_min:.3e}) at or below the LS floor "
        f"({ls_floor:.3e} = 10 * 2**-max_ls_steps) and exited with "
        f"{tail_failures} tail line-search failures.  This is "
        "consistent with (i) a non-descent direction the QP returned "
        "(L-BFGS conditioning, stale curvature), (ii) a too-small "
        "merit penalty rho, or (iii) a step that violates the "
        "Armijo condition by overshooting active constraints.  The "
        "alpha-trajectory window around the offending step is "
        "attached as an artifact.  The L-BFGS diagonal at the "
        "*offending* step is not preserved by the diagnostics layer; "
        "call ``capture_state_at_step(..., step="
        f"{offending_step})`` to recover it exactly."
    )
    return _build_signal(
        name="line_search_collapse",
        specificity="ambiguous",
        summary=(
            f"The line search looks suspicious because alpha "
            f"collapsed to {alpha_min:.3e} (<= LS floor "
            f"{ls_floor:.3e}) and the run exited with "
            f"{tail_failures} tail line-search failures."
        ),
        detail=detail,
        evidence={
            "ls_alpha_min": alpha_min,
            "tail_ls_failures": tail_failures,
            "ls_floor": ls_floor,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Inspect the L-BFGS diagonal / gamma at the offending "
            "step (use ``capture_state_at_step``).  If kappa(B0) is "
            "extreme, try ``use_exact_hvp_in_qp=True``.",
            "Try a larger initial merit penalty (raise the QP "
            "multipliers' bound or set a larger ``proximal_mu_max``).",
            "Try MinresQLPSolver: it solves the saddle-point KKT "
            "directly and is less sensitive to L-BFGS corruption.",
        ],
        artifacts={
            "alpha_window": alpha_window,
            "lbfgs_diagonal_final": final_diagonal,
        },
        offending_step=offending_step,
    )


def _eval_qp_budget_or_pingpong(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: QP active-set loop ran out of budget or ping-ponged.

    Specificity: ``generic``.  Both events have many possible causes
    (LPEC-A over-prediction, degenerate vertex, coupled multipliers
    from the inner solver, etc).
    """
    if not summaries:
        return None  # pragma: no cover
    diag = final_state.diagnostics
    n_budget = int(diag.n_qp_budget_exhausted)
    n_pingpong = int(diag.n_qp_ping_pong)
    if not (n_budget > 0 or n_pingpong > 0):
        return None

    n_steps = max(len(summaries), 1)
    fraction_budget = n_budget / n_steps
    fraction_pingpong = n_pingpong / n_steps
    # The "threshold" for either is "1 occurrence per run" ie 1/n_steps;
    # ratio is just the count itself relative to a threshold of 1.
    threshold_ratio = max(n_budget, n_pingpong) / 1.0

    detail = (
        f"The QP active-set loop hit its iteration budget on "
        f"{n_budget} steps and short-circuited on a ping-pong on "
        f"{n_pingpong} steps (out of {n_steps} total iterations).  "
        "Both events are usually downstream of one of: an LPEC-A "
        "over-prediction the trust gate failed to catch (raise "
        "``lpeca_trust_threshold`` or use "
        "``active_set_method='expand'``), a degenerate vertex with "
        "coupled constraints (raise ``mult_drop_floor``), or "
        "noise-contaminated multipliers from the inner solver "
        "(switch to MinresQLPSolver or HRInexactSTCG)."
    )
    return _build_signal(
        name="qp_budget_or_pingpong",
        specificity="generic",
        summary=(
            f"The QP layer looks suspicious because it exhausted its "
            f"budget on {n_budget} steps and ping-ponged on "
            f"{n_pingpong} steps."
        ),
        detail=detail,
        evidence={
            "n_qp_budget_exhausted": n_budget,
            "n_qp_ping_pong": n_pingpong,
            "fraction_budget": fraction_budget,
            "fraction_pingpong": fraction_pingpong,
            "n_steps": n_steps,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Raise ``qp_max_iter`` to give the active-set loop more "
            "headroom (current default may be too tight for "
            "ill-conditioned problems).",
            "If LPEC-A is enabled, raise ``lpeca_trust_threshold`` "
            "or fall back to ``active_set_method='expand'``.",
            "Raise ``mult_drop_floor`` so noise-flipped multipliers "
            "do not spuriously drop active constraints.",
        ],
        artifacts={},
        offending_step=None,
    )


def _eval_merit_oscillation(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: merit function regressed too often.

    Cheap predicate: ``n_merit_regressions > step_count / 10`` (more
    than 10 % of steps saw merit increase despite line-search
    success).  Specificity: ``generic`` — merit oscillation can be a
    too-small ``rho``, a stale Hessian approximation, or both.
    """
    if not summaries:
        return None  # pragma: no cover
    diag = final_state.diagnostics
    n_regressions = int(diag.n_merit_regressions)
    n_steps = max(len(summaries), 1)
    threshold = max(n_steps // 10, 1)
    if not (n_regressions > threshold):
        return None

    threshold_ratio = n_regressions / threshold

    detail = (
        f"The L1 merit function increased on {n_regressions} of "
        f"{n_steps} iterations despite the line search reporting "
        "success.  The Han-Powell merit function is monotone-"
        "decreasing under a correctly-sized ``rho``; persistent "
        "regressions point at ``rho`` being too small (the penalty "
        "does not dominate the constraint violation) or at a stale "
        "L-BFGS approximation that produces non-descent QP steps."
    )
    return _build_signal(
        name="merit_oscillation",
        specificity="generic",
        summary=(
            f"The merit function looks suspicious because it "
            f"regressed on {n_regressions} of {n_steps} iterations "
            f"(threshold = step_count / 10 = {threshold})."
        ),
        detail=detail,
        evidence={
            "n_merit_regressions": n_regressions,
            "n_steps": n_steps,
            "threshold": threshold,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Inspect the merit penalty ``rho`` trajectory; if it "
            "stays at the initial value, the penalty update is "
            "starved of evidence.",
            "Try ``use_exact_hvp_in_qp=True`` so the QP step "
            "reflects the actual curvature of the Lagrangian.",
        ],
        artifacts={},
        offending_step=None,
    )


def _eval_lpeca_overpredicting(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: LPEC-A over-predicted on > 50 % of steps OR was
    bypassed every step.

    Specificity: ``ambiguous`` — over-prediction can mean LPEC-A's
    rho_bar is too pessimistic (rare) or that the iterate is far
    from KKT (the warm-up gate should already cover this), so the
    signal is informative but not pinpoint.
    """
    if not summaries:
        return None  # pragma: no cover
    diag = final_state.diagnostics
    n_capped = int(diag.n_lpeca_capped)
    n_bypassed = int(diag.n_lpeca_bypassed)
    n_steps = max(len(summaries), 1)
    fraction_capped = n_capped / n_steps
    bypassed_all = n_bypassed == n_steps and n_steps > 0

    if not (fraction_capped > 0.5 or bypassed_all):
        return None

    if bypassed_all:
        threshold_ratio = float(n_bypassed) / 1.0  # pragma: no cover
    else:
        threshold_ratio = fraction_capped / 0.5

    detail = (
        f"LPEC-A's rank-aware size cap truncated the prediction on "
        f"{n_capped} of {n_steps} iterations and the prediction was "
        f"bypassed on {n_bypassed} iterations.  Persistent capping "
        "means rho_bar (the LPEC-A proximity measure) is letting "
        "too many constraints into the predicted active set, which "
        "in turn forces the cap to discard them.  Persistent "
        "bypassing means the trust gate vetoed the prediction every "
        "step, in which case LPEC-A is contributing nothing.  Both "
        "imply the EXPAND fallback is doing all the work."
    )
    return _build_signal(
        name="lpeca_overpredicting",
        specificity="ambiguous",
        summary=(
            f"LPEC-A looks suspicious because it was capped on "
            f"{n_capped} steps and bypassed on {n_bypassed} steps "
            f"(out of {n_steps})."
        ),
        detail=detail,
        evidence={
            "n_lpeca_capped": n_capped,
            "n_lpeca_bypassed": n_bypassed,
            "n_steps": n_steps,
            "fraction_capped": fraction_capped,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Disable LPEC-A by setting "
            "``active_set_method='expand'`` and re-run; if "
            "convergence improves, LPEC-A was hurting more than "
            "helping on this problem.",
            "Raise ``lpeca_trust_threshold`` so the predictor is "
            "trusted more often (only safe when rho_bar is a "
            "reliable proxy near the optimum).",
            "Raise ``lpeca_warmup_steps`` so LPEC-A waits longer before contributing.",
        ],
        artifacts={},
        offending_step=None,
    )


def _eval_infeasible_termination(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: the solver terminated at an infeasible iterate.

    Fires when either (a) the granular ``state.termination_code`` is
    one of {``infeasible``, ``nonfinite``}, or (b) primal feasibility
    was never satisfied across the run (max constraint violation
    above ``atol`` on every recorded summary).  Specificity:
    ``specific``.

    Artifacts: indices and signed values of the worst-violated
    equality + inequality constraints at the final iterate.
    """
    if not summaries:
        return None  # pragma: no cover
    code = final_state.termination_code
    name = _result_name_safe(code)
    code_match = name in {"infeasible", "nonfinite"}

    last = summaries[-1]
    atol = ctx.atol
    feasibility_ever_satisfied = any(
        s.max_eq_violation <= atol and s.max_ineq_violation <= atol for s in summaries
    )
    persistent_infeasibility = not feasibility_ever_satisfied and (
        last.max_eq_violation > atol or last.max_ineq_violation > atol
    )

    if not (code_match or persistent_infeasibility):
        return None

    eq_val = _to_numpy(final_state.eq_val)
    ineq_val = _to_numpy(final_state.ineq_val)

    # Worst-violated indices (top-5).
    eq_worst_idx = (
        np.argsort(np.abs(eq_val))[::-1][:5]
        if eq_val.size > 0
        else np.array([], dtype=int)
    )
    ineq_violations = np.maximum(0.0, -ineq_val) if ineq_val.size > 0 else np.array([])
    ineq_worst_idx = (
        np.argsort(ineq_violations)[::-1][:5]
        if ineq_violations.size > 0
        else np.array([], dtype=int)
    )

    worst_violation = max(last.max_eq_violation, last.max_ineq_violation)
    threshold_ratio = worst_violation / max(atol, 1e-300)

    detail = (
        f"The run terminated at a primally infeasible iterate "
        f"(termination code = {name!r}, max|c_eq| = "
        f"{last.max_eq_violation:.3e}, max(0, -c_ineq) = "
        f"{last.max_ineq_violation:.3e}, atol = {atol:.1e}).  "
        "Either the constraints are mutually infeasible (no point "
        "satisfies all of them simultaneously) or the active-set "
        "machinery could not satisfy them within ``atol`` from the "
        "supplied initial point.  The worst-violated constraint "
        "indices and signed values are attached as artifacts."
    )
    return _build_signal(
        name="infeasible_termination",
        specificity="specific",
        summary=(
            f"Primal feasibility looks suspicious because the run "
            f"ended with max|c_eq|={last.max_eq_violation:.3e} and "
            f"max(0,-c_ineq)={last.max_ineq_violation:.3e} "
            f"(atol={atol:.1e})."
        ),
        detail=detail,
        evidence={
            "max_eq_violation": last.max_eq_violation,
            "max_ineq_violation": last.max_ineq_violation,
            "atol": atol,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Print the worst-violated constraints (see artifacts) "
            "and verify they admit a common feasible point.",
            "Provide a feasible initial iterate, or relax the "
            "constraints incrementally to find a feasibility "
            "boundary.",
            "If ``c_eq`` involves a divide-by-near-zero or other "
            "non-Lipschitz expression, consider rewriting the "
            "constraint to avoid the singularity.",
        ],
        artifacts={
            "eq_values": eq_val,
            "ineq_values": ineq_val,
            "eq_worst_indices": np.asarray(eq_worst_idx, dtype=int),
            "ineq_worst_indices": np.asarray(ineq_worst_idx, dtype=int),
        },
        offending_step=None,
    )


# Default ``divergence_patience`` from :class:`SLSQPConfig` (3).  The
# magnitude of :func:`_eval_divergence_rollback_triggered` is
# normalised by this value so a run that triggered exactly one
# rollback fires at "marginal" magnitude and a run that triggered the
# rollback ten times fires at "moderate" / "extreme".
_DIVERGENCE_PATIENCE_DEFAULT = 3


def _eval_divergence_rollback_triggered(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: best-iterate divergence rollback fired during the run.

    Cheap predicate: ``final_state.diagnostics.divergence_triggered``
    is ``True`` (the rollback latched at least once) OR
    ``n_divergence_blowups > 0`` (a blow-up was observed even if
    patience was not yet reached).  The rollback is the
    "best-iterate-rescue" guardrail described in ``AGENTS.md``
    "Best-iterate divergence rollback"; if it fires, the run hit a
    catastrophic merit blow-up and the iterate the driver returned is
    ``state.best_x``, not the attempted final iterate.

    Specificity: ``specific``.  This datum is essentially diagnostic
    for "the run blew up and was rescued by rollback" — no other
    failure mode flips ``divergence_triggered``.
    """
    if not summaries:
        return None  # pragma: no cover
    diag = final_state.diagnostics
    triggered = bool(diag.divergence_triggered)
    n_blowups = int(diag.n_divergence_blowups)
    if not (triggered or n_blowups > 0):
        return None

    last = summaries[-1]
    # Locate the offending step: first ``StepSummary`` recording a
    # blow-up event (``blowup_count`` > 0 immediately after a step
    # whose ``blowup_count`` was 0, or simply the first step with
    # ``diverging`` set).  Falls back to the final summary when no
    # such step is found.
    offending_step: Optional[int] = None
    prev_blowup = 0
    for s in summaries:
        if s.diverging:
            offending_step = int(s.step_count)
            break
        if s.blowup_count > prev_blowup:
            offending_step = int(s.step_count)
            break
        prev_blowup = int(s.blowup_count)
    if offending_step is None:
        offending_step = int(last.step_count)  # pragma: no cover

    threshold_ratio = max(float(n_blowups), 1.0) / float(_DIVERGENCE_PATIENCE_DEFAULT)

    detail = (
        f"The best-iterate divergence rollback latched (triggered = "
        f"{triggered}, n_divergence_blowups = {n_blowups}).  This "
        "means the merit function suffered a catastrophic blow-up "
        "(``merit_new - best_merit > divergence_factor * "
        "max(|best_merit|, 1)`` for ``divergence_patience`` "
        "consecutive steps; defaults are ``10.0`` and ``3``) and the "
        "iterate the driver returned was overwritten with "
        "``state.best_x``.  The L-BFGS history, multipliers and "
        "gradients still reflect the *attempted* step, so the "
        "report's 'Final iterate metrics' section may show post-"
        "blow-up values that do not correspond to the actually-"
        "returned iterate.  See ``AGENTS.md`` 'Best-iterate "
        "divergence rollback'.  The first blow-up was at step "
        f"{offending_step}; inspect the trajectory around there for "
        "the failure mechanism (typically L-BFGS poisoning + "
        "merit-penalty over-correction)."
    )
    return _build_signal(
        name="divergence_rollback_triggered",
        specificity="specific",
        summary=(
            "The best-iterate divergence rollback looks suspicious "
            f"because it latched (triggered={triggered}) after "
            f"{n_blowups} blow-up event(s); the returned iterate is "
            "``best_x``, not the attempted final iterate."
        ),
        detail=detail,
        evidence={
            "divergence_triggered": 1 if triggered else 0,
            "n_divergence_blowups": n_blowups,
            "patience_default": _DIVERGENCE_PATIENCE_DEFAULT,
            "offending_step": offending_step,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Inspect the trajectory around the offending step "
            "(use ``capture_state_at_step``) for an L-BFGS diagonal "
            "blow-up paired with a merit-penalty jump — the usual "
            "cascade is QP-budget exhaustion → noisy curvature pair "
            "appended → kappa(B0) explosion → unrecoverable QP "
            "direction → merit blow-up → rollback.",
            "Try ``use_exact_hvp_in_qp=True`` so the QP no longer "
            "depends on the L-BFGS approximation (the most common "
            "way to break the cascade).",
            "Lower ``divergence_patience`` so the rollback fires "
            "earlier, before the post-blow-up state corrupts other "
            "quantities the report surfaces.",
        ],
        artifacts={},
        offending_step=offending_step,
    )


# Total-trajectory ratio threshold: ``rho`` grew by 6+ orders of
# magnitude across the run.  The Han-Powell penalty update is meant
# to grow ``rho`` monotonically by small factors per step; a 1e6
# total span is a strong indicator of an over-correction cycle.
_RHO_EXPLOSION_TOTAL_RATIO = 1e6
# Per-step ratio threshold: ``rho`` jumped by 1000x or more in a
# single step.  Either threshold tripping is enough.
_RHO_EXPLOSION_STEP_RATIO = 1e3


def _eval_merit_penalty_explosion(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: merit penalty ``rho`` exploded across the trajectory.

    Cheap predicate: either
    ``max(rho) / max(min(rho), 1) > _RHO_EXPLOSION_TOTAL_RATIO``
    (the trajectory-wide span exceeds 1e6) OR
    ``max(rho[i] / max(rho[i-1], 1)) > _RHO_EXPLOSION_STEP_RATIO``
    (a single-step jump exceeds 1e3).  Either pattern signals that
    the Han-Powell penalty update over-corrected, usually because
    the QP direction was poisoned by a degenerate inner solve and
    the merit function's predicted reduction stopped agreeing with
    the realised one.

    Specificity: ``specific``.  ``rho`` is monotone-non-decreasing
    by construction, and the update cadence is bounded by problem
    geometry; a 6-orders-of-magnitude span across the run or a
    1000x jump in one step has a narrow set of causes (L-BFGS
    poisoning + merit catch-up is the dominant one).
    """
    if not summaries:
        return None  # pragma: no cover
    rhos = [max(float(s.merit_penalty), 1.0) for s in summaries]
    min_rho = min(rhos)
    max_rho = max(rhos)
    ratio_total = max_rho / min_rho
    if len(rhos) > 1:
        per_step_ratios = [rhos[i] / rhos[i - 1] for i in range(1, len(rhos))]
        ratio_step = max(per_step_ratios)
        # Step where the largest single-step jump happened (1-indexed
        # to match ``StepSummary.step_count`` semantics).
        jump_idx_zero = max(range(1, len(rhos)), key=lambda i: rhos[i] / rhos[i - 1])
        jump_step = int(summaries[jump_idx_zero].step_count)
    else:
        ratio_step = 1.0
        jump_step = int(summaries[-1].step_count)

    total_fired = ratio_total > _RHO_EXPLOSION_TOTAL_RATIO
    step_fired = ratio_step > _RHO_EXPLOSION_STEP_RATIO
    if not (total_fired or step_fired):
        return None

    threshold_ratio = max(
        ratio_total / _RHO_EXPLOSION_TOTAL_RATIO,
        ratio_step / _RHO_EXPLOSION_STEP_RATIO,
    )

    detail = (
        "The merit penalty ``rho`` trajectory exploded across the "
        f"run: span ratio ``max(rho) / max(min(rho), 1) = "
        f"{ratio_total:.3e}`` (threshold "
        f"{_RHO_EXPLOSION_TOTAL_RATIO:.0e}), largest single-step "
        f"jump ``{ratio_step:.3e}`` (threshold "
        f"{_RHO_EXPLOSION_STEP_RATIO:.0e}, at step {jump_step}).  "
        "The Han-Powell update grows ``rho`` monotonically by "
        "small factors per step; a span this large is the "
        "merit function trying to catch up after a QP step poisoned "
        "the predicted-vs-realised merit reduction (usually because "
        "an L-BFGS pair from a fallback projected-gradient direction "
        "was appended to the curvature buffer and corrupted the "
        "subsequent Newton direction).  Once ``rho`` is over-sized, "
        "every subsequent line search has to absorb a much larger "
        "merit derivative and typically collapses to the LS floor."
    )
    return _build_signal(
        name="merit_penalty_explosion",
        specificity="specific",
        summary=(
            "The merit penalty ``rho`` looks suspicious because it "
            f"spanned {ratio_total:.3e}x across the run "
            f"(min={min_rho:.3e}, max={max_rho:.3e}; largest "
            f"single-step jump {ratio_step:.3e}x at step {jump_step})."
        ),
        detail=detail,
        evidence={
            "rho_min": min_rho,
            "rho_max": max_rho,
            "rho_ratio_total": ratio_total,
            "rho_ratio_step": ratio_step,
            "rho_threshold_total": _RHO_EXPLOSION_TOTAL_RATIO,
            "rho_threshold_step": _RHO_EXPLOSION_STEP_RATIO,
            "jump_step": jump_step,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "Try ``use_exact_hvp_in_qp=True`` so the QP no longer "
            "depends on the L-BFGS approximation; the merit-penalty "
            "over-correction is almost always downstream of an "
            "L-BFGS poisoning event.",
            "Raise ``qp_max_iter`` so QP-budget exhaustion (which "
            "forces the projected-gradient fallback that poisons the "
            "L-BFGS history) is less likely.",
            "Cap the merit penalty growth by passing a smaller "
            "``proximal_mu_max`` if the equality constraints are "
            "well-scaled; the ceiling prevents runaway ``rho``.",
            "If the run was launched with ``auto_scale=False`` (or via "
            "a path that bypasses :func:`slsqp_jax.minimize_like_scipy`), "
            "enable auto-scaling -- a runaway ``rho`` is almost always "
            "the merit function trying to reconcile mismatched objective "
            "and constraint magnitudes.  The Auto-scaling section of "
            "the report (when present) lists the factors that were "
            "applied.",
            "The new ``auto_scale=True`` (uniform mode, default) uses a "
            "single shared scalar across all constraint rows -- it "
            "prevents the cascade without flattening the relative row "
            "magnitudes.  If your constraints have intentionally "
            "heterogeneous magnitudes (e.g. a top-level budget plus "
            "smaller sub-budgets) this is the right mode.  If a single "
            "constraint row has *vastly* different gradient magnitude "
            "from the rest, ``auto_scale='balanced'`` (per-row, the "
            "old default) flattens those rows individually and may "
            "rescue the run.  ``auto_scale='aggressive'`` raises the "
            "amplification ceiling for very small-gradient problems.",
        ],
        artifacts={
            "rho_trajectory": np.asarray(rhos, dtype=float),
        },
        offending_step=jump_step if step_fired else None,
    )


# ``rho`` is treated as "frozen" for the prefix when its span is
# below this multiplicative tolerance.  ``rho`` is mutated by an
# additive-then-clip update in the solver, so a true "did not move"
# signal needs a tiny epsilon rather than equality testing.
_RHO_FROZEN_TOL = 1.0 + 1e-9
# Minimum violation growth that constitutes "drift": the worst
# infeasibility at the end of the prefix must be at least this
# multiple of the worst infeasibility at the start.
_STARVATION_VIOLATION_GROWTH = 5.0
# The frozen-prefix must span at least this many steps (and at
# least 5 absolute) to fire.  Short prefixes are noise.
_STARVATION_PREFIX_FRACTION = 1.0 / 3.0


def _eval_penalty_starvation(
    ctx: EvalContext,
    final_state: "SLSQPState",
    summaries: list["StepSummary"],
    coarse_result: Any,
) -> Optional[Signal]:
    """End-of-run: ``rho`` stayed at its initial value while feasibility drifted.

    Cheap predicate: the longest prefix of summaries during which
    ``rho`` is constant (within :data:`_RHO_FROZEN_TOL`) spans at
    least ``max(5, n_steps // 3)`` steps, ``max_eq_violation`` is
    monotone non-decreasing across that prefix, and grew by at
    least :data:`_STARVATION_VIOLATION_GROWTH` (= 5x) from the
    first to the last step of the prefix.

    Catches the "started feasible, drifted off, ``rho`` never grew"
    pathology described in the diagnostic notes for the feasible-
    start divergence run: the Han-Powell directional-derivative test
    cannot trigger a ``rho`` increase when the ``f`` reduction alone
    satisfies it, so feasibility can decay silently for many steps.

    Specificity: ``specific``.
    """
    if not summaries:
        return None  # pragma: no cover
    n_steps = len(summaries)
    min_prefix = max(5, n_steps // 3)
    if n_steps < min_prefix:
        return None
    # Identify the longest leading prefix where ``rho`` is frozen.
    rho0 = max(float(summaries[0].merit_penalty), 1.0)
    prefix_end = 0
    for i, s in enumerate(summaries):
        rho_i = max(float(s.merit_penalty), 1.0)
        ratio = max(rho_i / rho0, rho0 / rho_i)
        if ratio > _RHO_FROZEN_TOL:
            break
        prefix_end = i
    prefix_len = prefix_end + 1
    if prefix_len < min_prefix:
        return None

    prefix = summaries[: prefix_end + 1]
    eq_violations = [float(s.max_eq_violation) for s in prefix]
    ineq_violations = [float(s.max_ineq_violation) for s in prefix]
    # Treat eq + ineq as a unified "infeasibility" proxy so the test
    # works for both equality- and inequality-only problems.
    violations = [max(e, i) for e, i in zip(eq_violations, ineq_violations)]
    # Monotone non-decreasing test (with a small tolerance to
    # absorb numerical jitter at the floor).
    monotone = all(
        violations[i] >= violations[i - 1] - 1e-15 for i in range(1, len(violations))
    )
    if not monotone:
        return None
    start_violation = violations[0]
    end_violation = violations[-1]
    # Need a non-zero starting violation to avoid 0/0; if it is at
    # the floor, require an absolute growth above ``atol`` instead.
    atol = ctx.atol
    if start_violation > 0 and start_violation > atol:
        growth_ratio = end_violation / start_violation
    else:
        # Compare against ``atol`` so a near-zero start that grew to
        # ``5 * atol`` qualifies (the canonical feasible-start case).
        growth_ratio = end_violation / max(atol, 1e-300)
    if growth_ratio < _STARVATION_VIOLATION_GROWTH:
        return None

    threshold_ratio = growth_ratio / _STARVATION_VIOLATION_GROWTH

    detail = (
        f"The merit penalty ``rho`` stayed constant at "
        f"{rho0:.3e} for the first {prefix_len} of {n_steps} "
        "iterations while the worst infeasibility grew "
        f"monotonically from {start_violation:.3e} to "
        f"{end_violation:.3e}.  This is the textbook "
        "'penalty-starved feasibility drift' pattern: the "
        "Han-Powell directional-derivative test cannot trigger a "
        "``rho`` increase when the ``f`` reduction alone satisfies "
        "the predicted-vs-realised inequality, so the merit "
        "function happily accepts steps that improve ``f`` at the "
        "cost of feasibility.  When feasibility eventually decays "
        "enough to require a real ``rho`` correction, the catch-up "
        "is usually huge and downstream poisons the L-BFGS history "
        "(see ``merit_penalty_explosion`` and "
        "``divergence_rollback_triggered`` for the typical "
        "downstream signals).  See ``AGENTS.md`` 'L1 merit "
        "directional derivative' for the update rule."
    )
    return _build_signal(
        name="penalty_starvation",
        specificity="specific",
        summary=(
            "The merit penalty ``rho`` looks suspicious because it "
            f"stayed constant at {rho0:.3e} for {prefix_len} of "
            f"{n_steps} steps while max-infeasibility grew "
            f"{growth_ratio:.1f}x (from {start_violation:.3e} to "
            f"{end_violation:.3e})."
        ),
        detail=detail,
        evidence={
            "rho_initial": rho0,
            "frozen_prefix_steps": prefix_len,
            "n_steps": n_steps,
            "violation_start": start_violation,
            "violation_end": end_violation,
            "growth_ratio": growth_ratio,
            "growth_threshold": _STARVATION_VIOLATION_GROWTH,
        },
        threshold_ratio=threshold_ratio,
        suggestions=[
            "If the supplied initial iterate is exactly feasible, "
            "perturb it slightly (e.g. ``x0 + atol * sign_vector``) "
            "so the merit-penalty update mechanism warms ``rho`` up "
            "before feasibility drifts.",
            "Bump the initial ``merit_penalty`` so the feasibility "
            "term has weight from step 1.",
            "Check the magnitude balance between the constraint "
            "Jacobian and the objective gradient: if "
            "``eq_jac_min_sv_est`` (in the diagnostics block) is "
            "much larger than ``||grad_f||``, the constraint is "
            "much steeper than the objective and the merit "
            "function cannot reconcile their natural scales without "
            "a huge ``rho``.  Rescaling the constraint (e.g. drop "
            "the ``1/target`` division if the constraint is "
            "``(h(x) - target) / target``) so ``||J_eq||`` is "
            "comparable to ``||grad_f||`` is the durable fix.",
            "If LPEC-A is enabled, raise ``lpeca_warmup_steps`` so "
            "the predictor does not contaminate the early-iteration "
            "active set on a near-feasible iterate.",
            "If the run was launched with ``auto_scale=False`` (or via "
            "a path that bypasses :func:`slsqp_jax.minimize_like_scipy`), "
            "enable auto-scaling -- the ``||J_eq|| >> ||grad_f||`` "
            "magnitude mismatch that triggers penalty starvation is "
            "exactly what gradient-based scaling is designed to fix.  "
            "The Auto-scaling section of the report (when present) "
            "lists the factors that were applied.",
            "The new ``auto_scale=True`` (uniform mode, default) "
            "preserves the relative row magnitudes between constraints "
            "while bringing the constraint Jacobian into the same "
            "range as the objective gradient -- the right choice when "
            "your constraints encode meaningful inter-row structure "
            "(e.g. a top-level budget plus smaller sub-budgets).  If "
            "*one* constraint row has vastly different gradient "
            "magnitude from the rest and that's not a meaningful "
            "spread, ``auto_scale='balanced'`` (per-row, the old "
            "default) flattens those rows individually and may rescue "
            "the run.  ``auto_scale='aggressive'`` raises the "
            "amplification ceiling.",
        ],
        artifacts={
            "rho_prefix": np.asarray(
                [float(s.merit_penalty) for s in prefix], dtype=float
            ),
            "violations_prefix": np.asarray(violations, dtype=float),
        },
        offending_step=int(prefix[-1].step_count),
    )


def _result_name_safe(result: Any) -> str:
    """Wrap :func:`slsqp_jax.diagnostics.report._result_name` for use here.

    Imported lazily to avoid a circular import: ``signals`` is
    imported by ``__init__`` before ``report``.
    """
    from slsqp_jax.diagnostics.report import _result_name

    return _result_name(result)


# ---------------------------------------------------------------------------
# Registry installation
# ---------------------------------------------------------------------------

# Register all 8 starter signals at import time so the runner picks
# them up by default.  Test code can re-register an alternative
# evaluator under the same ``name`` to shift thresholds without
# editing this module.

register_evaluator(
    "eq_jacobian_rank_deficient",
    specificity="specific",
    flavour="per_step",
    evaluator=_eval_eq_jacobian_rank_deficient,
)
register_evaluator(
    "lbfgs_conditioning_extreme",
    specificity="ambiguous",
    flavour="per_step",
    evaluator=_eval_lbfgs_conditioning_extreme,
)
register_evaluator(
    "multiplier_recovery_noise",
    specificity="specific",
    flavour="end_of_run",
    evaluator=_eval_multiplier_recovery_noise,
)
register_evaluator(
    "line_search_collapse",
    specificity="ambiguous",
    flavour="end_of_run",
    evaluator=_eval_line_search_collapse,
)
register_evaluator(
    "qp_budget_or_pingpong",
    specificity="generic",
    flavour="end_of_run",
    evaluator=_eval_qp_budget_or_pingpong,
)
register_evaluator(
    "merit_oscillation",
    specificity="generic",
    flavour="end_of_run",
    evaluator=_eval_merit_oscillation,
)
register_evaluator(
    "lpeca_overpredicting",
    specificity="ambiguous",
    flavour="end_of_run",
    evaluator=_eval_lpeca_overpredicting,
)
register_evaluator(
    "infeasible_termination",
    specificity="specific",
    flavour="end_of_run",
    evaluator=_eval_infeasible_termination,
)
register_evaluator(
    "divergence_rollback_triggered",
    specificity="specific",
    flavour="end_of_run",
    evaluator=_eval_divergence_rollback_triggered,
)
register_evaluator(
    "merit_penalty_explosion",
    specificity="specific",
    flavour="end_of_run",
    evaluator=_eval_merit_penalty_explosion,
)
register_evaluator(
    "penalty_starvation",
    specificity="specific",
    flavour="end_of_run",
    evaluator=_eval_penalty_starvation,
)


__all__ = [
    "Confidence",
    "EndOfRunEvaluator",
    "END_OF_RUN_EVALUATORS",
    "EvalContext",
    "Magnitude",
    "PER_STEP_EVALUATORS",
    "PerStepEvaluator",
    "SIGNAL_REGISTRY",
    "Signal",
    "SignalRegistration",
    "Specificity",
    "register_evaluator",
]
