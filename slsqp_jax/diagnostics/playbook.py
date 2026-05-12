"""Playbook: termination-code scoping + multi-signal diagnoses.

The plan calls out two distinct pieces of post-fire interpretation:

1. **Termination-code scoping** (:data:`SCOPE_BY_TERMINATION`) — uses
   the granular ``slsqp_jax.RESULTS`` code from the failed run as
   *context* for the rest of the report.  Signals outside the scoped
   set for the run's termination code are still listed (never hidden)
   but visually de-prioritised under "less likely given the
   termination mode".  This is how the tool exploits the failure
   mode the user already arrived with.
2. **Multi-signal diagnoses** (:data:`RULES`) — when several signals
   fire together they sometimes form a textbook pattern that earns a
   named diagnosis.  Single-signal cases are *not* given a rule
   here; they are surfaced directly by the report renderer.

Confidence ranking lives here too: :func:`magnitude_for` derives the
dynamic ``magnitude`` axis from the signal's evidence dict (ratio to
threshold), and :func:`confidence_for` collapses
``(specificity, magnitude)`` to a single ``low`` / ``medium`` / ``high``
tag via the documented lookup table.

Phase 3 populates :data:`RULES` with three starter multi-signal
patterns; Phase 1 / 2 ship empty so the runner / report renderer
have the same call surface from day one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from slsqp_jax.results import RESULTS

if TYPE_CHECKING:
    from slsqp_jax.diagnostics.signals import (
        Confidence,
        Magnitude,
        Signal,
        Specificity,
    )


# ---------------------------------------------------------------------------
# Scoping by termination code
# ---------------------------------------------------------------------------


def _enum_value(item: Any) -> Any:
    """Return the integer ``_value`` of an ``equinox.Enumeration`` item.

    Centralised here so the playbook can compare codes by value
    rather than by ``is``-identity (two items with the same value are
    equal but not ``is``-identical, depending on how / when they were
    constructed).  ``equinox.EnumerationItem`` is not hashable, so we
    must key the scope mapping by the integer value rather than the
    member directly.
    """
    raw = getattr(item, "_value", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# Signals considered "in scope" given each granular ``slsqp_jax.RESULTS``
# termination code.  Keyed by the integer ``_value`` of the
# corresponding ``RESULTS`` member because ``equinox.EnumerationItem``
# is not hashable.  A signal *outside* the scoped set is still
# listed in the report, but under a "less likely given the
# termination mode" sub-section — never hidden, just de-prioritised.
# An empty set (``RESULTS.nonlinear_max_steps_reached``,
# ``RESULTS.successful``) means "everything in scope" — no
# de-prioritisation applies.

SCOPE_BY_TERMINATION: dict[int, set[str]] = {
    _enum_value(RESULTS.merit_stagnation): {
        "lbfgs_conditioning_extreme",
        "multiplier_recovery_noise",
        "line_search_collapse",
        "merit_oscillation",
    },
    _enum_value(RESULTS.line_search_failure): {
        "lbfgs_conditioning_extreme",
        "line_search_collapse",
        "eq_jacobian_rank_deficient",
    },
    _enum_value(RESULTS.qp_subproblem_failure): {
        "qp_budget_or_pingpong",
        "eq_jacobian_rank_deficient",
        "lbfgs_conditioning_extreme",
        "lpeca_overpredicting",
    },
    _enum_value(RESULTS.iterate_blowup): {
        "lbfgs_conditioning_extreme",
        "merit_oscillation",
        "line_search_collapse",
    },
    _enum_value(RESULTS.infeasible): {
        "infeasible_termination",
        "eq_jacobian_rank_deficient",
    },
    _enum_value(RESULTS.nonlinear_max_steps_reached): set(),
    _enum_value(RESULTS.successful): set(),
}


def _scope_for(termination_code: Any) -> Optional[set[str]]:
    """Look up :data:`SCOPE_BY_TERMINATION` by integer value."""
    target = _enum_value(termination_code)
    if target is None:
        return None
    return SCOPE_BY_TERMINATION.get(target)


def signals_in_scope(termination_code: Any, fired_names: set[str]) -> set[str]:
    """Partition ``fired_names`` by whether they lie in the scope for
    ``termination_code``.

    Returns the *in-scope* subset of ``fired_names``.  When
    ``termination_code`` has no entry in :data:`SCOPE_BY_TERMINATION`,
    or its entry is the empty set (the "everything in scope"
    sentinel used for ``successful`` and
    ``nonlinear_max_steps_reached``), every fired signal is treated
    as in-scope.
    """
    scope = _scope_for(termination_code)
    if scope is None or not scope:
        return set(fired_names)
    return {name for name in fired_names if name in scope}


def signals_out_of_scope(termination_code: Any, fired_names: set[str]) -> set[str]:
    """Complement of :func:`signals_in_scope` against ``fired_names``."""
    return set(fired_names) - signals_in_scope(termination_code, fired_names)


# ---------------------------------------------------------------------------
# Confidence: specificity x magnitude lookup
# ---------------------------------------------------------------------------

# Documented lookup table (see plan / README).  Indexed by
# ``(specificity, magnitude)`` strings.  Centralising it here means
# the report renderer / rule engine never has to re-derive it.
_CONFIDENCE_TABLE: dict[tuple[str, str], "Confidence"] = {
    ("specific", "marginal"): "medium",
    ("specific", "moderate"): "high",
    ("specific", "extreme"): "high",
    ("ambiguous", "marginal"): "low",
    ("ambiguous", "moderate"): "medium",
    ("ambiguous", "extreme"): "medium",
    ("generic", "marginal"): "low",
    ("generic", "moderate"): "low",
    ("generic", "extreme"): "low",
}


def confidence_for(specificity: "Specificity", magnitude: "Magnitude") -> "Confidence":
    """Collapse ``(specificity, magnitude)`` to a confidence tag.

    See the README / plan for the table.  Anything not in the table
    falls through to ``"low"`` rather than raising, so a future
    extension to specificity / magnitude vocabularies degrades
    gracefully.
    """
    return _CONFIDENCE_TABLE.get((specificity, magnitude), "low")


def magnitude_for(ratio: float) -> "Magnitude":
    """Bucket a ratio-to-threshold into the documented magnitude classes.

    ``ratio`` is the largest ratio of any single evidence value to its
    documented threshold.  Conventions:

    * ``< 10``    → ``"marginal"``
    * ``< 100``   → ``"moderate"``
    * otherwise   → ``"extreme"``

    Defaults to ``"marginal"`` when the ratio is non-finite (NaN, inf)
    or negative; the latter would indicate a programming error in the
    evaluator (the ratio should always be ``>= 1`` whenever the
    signal fires) but we degrade gracefully rather than raise.
    """
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return "marginal"
    if not (r == r) or r <= 0:  # NaN or non-positive
        return "marginal"
    if r < 10.0:
        return "marginal"
    if r < 100.0:
        return "moderate"
    return "extreme"


# ---------------------------------------------------------------------------
# Multi-signal diagnoses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnosis:
    """Named multi-signal diagnosis surfaced in the report.

    Attributes:
        name: Stable machine-readable identifier.
        cause: One-paragraph mechanism explanation.
        suggestions: Concrete starting-point fixes.
        related_signals: Names of the fired signals the rule combined.
    """

    name: str
    cause: str
    suggestions: list[str] = field(default_factory=list)
    related_signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Rule:
    """An explicit if-then mapping fired-signal-set → :class:`Diagnosis`.

    Attributes:
        name: Stable machine-readable identifier (mirrors the
            diagnosis ``name`` it produces).
        predicate: Callable ``set[str] -> bool`` that returns ``True``
            iff the rule's signals are all present.
        build: Callable ``list[Signal] -> Diagnosis`` that constructs
            the diagnosis from the (filtered) fired signals.
    """

    name: str
    predicate: Callable[[set[str]], bool]
    build: Callable[[list["Signal"]], Diagnosis]


def _signals_by_name(signals: list["Signal"]) -> dict[str, "Signal"]:
    """Return ``{signal.name: signal}`` for the first occurrence of each name."""
    out: dict[str, "Signal"] = {}
    for s in signals:
        if s.name not in out:
            out[s.name] = s
    return out


def _build_stale_lbfgs_curvature(signals: list["Signal"]) -> Diagnosis:
    """Build the "stale L-BFGS curvature poisoning the QP" diagnosis."""
    by_name = _signals_by_name(signals)
    related = sorted(
        n
        for n in (
            "lbfgs_conditioning_extreme",
            "line_search_collapse",
            "merit_oscillation",
        )
        if n in by_name
    )
    return Diagnosis(
        name="stale_lbfgs_curvature",
        cause=(
            "The L-BFGS B0 diagonal has gone extremely ill-conditioned "
            "AND the resulting QP step is failing the line search or "
            "regressing the merit function.  These two patterns "
            "together typically mean the curvature pairs in the L-BFGS "
            "history are stale: the gradient differences ``y_k`` no "
            "longer reflect the local Hessian at the current iterate.  "
            "Soft-then-identity resets in the recovery chain may be "
            "discarding the only useful pair on each cycle "
            "(see ``AGENTS.md`` 'L-BFGS reset strategies')."
        ),
        suggestions=[
            "Try ``use_exact_hvp_in_qp=True`` so the QP no longer "
            "depends on the L-BFGS approximation.  The Newton-CG "
            "mode pays one HVP per CG iteration in exchange for "
            "complete decoupling from L-BFGS quality.",
            "If the problem has equality constraints, try "
            "``inner_solver=MinresQLPSolver()`` which solves the "
            "saddle-point KKT system directly and is less sensitive "
            "to L-BFGS corruption.",
            "Inspect the per-step ``lbfgs_skipped`` flags in the "
            "trajectory: a long unbroken streak of ``True`` is the "
            "signature of the post-identity-reset skip lock.",
        ],
        related_signals=related,
    )


def _build_active_set_churn(signals: list["Signal"]) -> Diagnosis:
    """Build the "active-set churn from rank-deficient working Jacobian" diagnosis."""
    by_name = _signals_by_name(signals)
    related = sorted(
        n
        for n in (
            "eq_jacobian_rank_deficient",
            "qp_budget_or_pingpong",
        )
        if n in by_name
    )
    return Diagnosis(
        name="active_set_churn",
        cause=(
            "The equality Jacobian is near rank-deficient AND the QP "
            "active-set loop is exhausting its budget or ping-ponging "
            "on the same constraint pair.  When ``J_eq`` is rank-"
            "deficient at the current iterate, the working-set "
            "Jacobian inherits the deficiency and the active-set "
            "logic cycles trying to add and drop the same constraints.  "
            "The ping-pong detector + ``mult_drop_floor`` are designed "
            "to short-circuit this, but a chronic rank deficiency "
            "needs a structural fix."
        ),
        suggestions=[
            "Verify the equality constraints are linearly independent "
            "at the iterate (LICQ).  If the constraints are "
            "syntactically distinct but algebraically degenerate, "
            "drop the redundant rows.",
            "Switch to ``inner_solver=MinresQLPSolver()`` which "
            "handles indefinite/singular saddle-point systems "
            "natively (no need for null-space projection).",
            "Raise ``mult_drop_floor`` so noise-flipped multipliers "
            "do not spuriously drop active constraints.",
        ],
        related_signals=related,
    )


def _build_noise_floor_stall(signals: list["Signal"]) -> Diagnosis:
    """Build the "noise-floor stationarity stall" diagnosis."""
    return Diagnosis(
        name="noise_floor_stationarity_stall",
        cause=(
            "The classical Lagrangian-gradient stationarity test "
            "stalled above ``rtol`` while the inner solver's "
            "projected-gradient norm already passed it.  This is "
            "the textbook signature of multiplier-recovery noise "
            "contaminating ``∇L = ∇f − A^T λ`` (see ``AGENTS.md`` "
            "'Inexact stationarity disjunct'): the recovered ``λ`` "
            "carries ``O(eps · cond(A A^T))`` error and that error "
            "swamps the true stationarity residual at high "
            "precision."
        ),
        suggestions=[
            "Set ``inner_solver=HRInexactSTCG(inner=ProjectedCGCholesky())`` "
            "so the noise-aware projected gradient is computed "
            "natively.",
            "Set ``use_inexact_stationarity=True`` so the "
            "convergence test admits the projected-gradient "
            "disjunct.  The two together rescue this exact pattern.",
            "If you cannot change the inner solver, loosening "
            "``rtol`` so it sits above the multiplier-recovery noise "
            "floor is a valid (if less satisfying) workaround.",
        ],
        related_signals=["multiplier_recovery_noise"],
    )


# Single-signal cases do *not* earn a rule — the report renderer
# surfaces them directly from the lone fired signal.  The exception
# is "noise-floor stationarity stall" which warrants its own
# diagnosis because the recommended fix is concrete and non-obvious
# from the signal name alone.

RULES: list[Rule] = [
    Rule(
        name="stale_lbfgs_curvature",
        predicate=lambda fired: (
            "lbfgs_conditioning_extreme" in fired
            and ("line_search_collapse" in fired or "merit_oscillation" in fired)
        ),
        build=_build_stale_lbfgs_curvature,
    ),
    Rule(
        name="active_set_churn",
        predicate=lambda fired: (
            "eq_jacobian_rank_deficient" in fired and "qp_budget_or_pingpong" in fired
        ),
        build=_build_active_set_churn,
    ),
    Rule(
        name="noise_floor_stationarity_stall",
        predicate=lambda fired: "multiplier_recovery_noise" in fired,
        build=_build_noise_floor_stall,
    ),
]


def evaluate_diagnoses(signals: list["Signal"]) -> list[Diagnosis]:
    """Run every registered :class:`Rule` against ``signals``.

    Returns the diagnoses produced by the rules whose predicate
    fired, in registration order.  No deduplication or ranking
    happens here; the report renderer is responsible for sorting by
    confidence and applying the termination-code scope filter.
    """
    fired = {s.name for s in signals}
    return [rule.build(signals) for rule in RULES if rule.predicate(fired)]


# Re-export the granular result class for convenience: callers writing
# their own scope filters / rules can compare against
# ``RESULTS.merit_stagnation`` etc. without a second import.
__all__ = [
    "Diagnosis",
    "RESULTS",
    "RULES",
    "Rule",
    "SCOPE_BY_TERMINATION",
    "confidence_for",
    "evaluate_diagnoses",
    "magnitude_for",
    "signals_in_scope",
    "signals_out_of_scope",
]
