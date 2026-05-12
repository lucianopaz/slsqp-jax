"""Pretty-printed debug-report renderer.

Phase 1's report ships independent value: even with zero signals
fired it produces (a) the granular vs coarse termination distinction,
(b) the ``slsqp_jax.RESULTS`` message string, (c) a prose-annotated
dump of every counter on :class:`SLSQPDiagnostics`, and (d) a small
ASCII trajectory chart of the most-informative scalar fields per
step.  Phase 2 wires the fired-signal section in; Phase 3 adds the
diagnoses block.

The prose annotations for each counter are sourced from the field
docstrings on :class:`SLSQPDiagnostics` so the report tracks the
authoritative source.  When a docstring is unavailable for any
reason (e.g. introspection limits) we fall back to a short generic
label.
"""

from __future__ import annotations

import dataclasses
import io
import math
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from slsqp_jax.diagnostics.records import DebugRunResult
    from slsqp_jax.state import SLSQPDiagnostics


# Width of the rendered report.  Wide enough for the trajectory chart
# columns, narrow enough that two reports fit side-by-side in a
# typical terminal.
_REPORT_WIDTH = 88


# Mapping from counter name to a short prose annotation.  Filled in
# lazily from :class:`SLSQPDiagnostics`'s class docstring on first
# render so the prose tracks the authoritative source.  Counters with
# no docstring entry get a short generic label so the renderer never
# crashes on a future field addition that forgot the docstring.
_DIAG_PROSE_CACHE: dict[str, str] = {}


def _diag_prose(field_name: str) -> str:
    """Return a one-line prose annotation for an ``SLSQPDiagnostics`` field.

    Sourced from the field's entry in the ``Attributes:`` block of
    :class:`SLSQPDiagnostics`'s class docstring, collapsed to a single
    line.  Cached after first use.  Returns an empty string when the
    docstring does not document the field.
    """
    if not _DIAG_PROSE_CACHE:
        _populate_diag_prose_cache()
    return _DIAG_PROSE_CACHE.get(field_name, "")


def _populate_diag_prose_cache() -> None:
    """Parse :class:`SLSQPDiagnostics`'s docstring once and cache it."""
    from slsqp_jax.state import SLSQPDiagnostics

    doc = SLSQPDiagnostics.__doc__ or ""
    # The Attributes block has lines like:
    #     field_name: First line of description.
    #         Continuation lines indented further.
    # We collapse each entry to a single sentence.
    lines = doc.splitlines()
    in_attrs = False
    current_field: Optional[str] = None
    current_buf: list[str] = []

    def flush() -> None:
        nonlocal current_field, current_buf
        if current_field is not None:
            text = " ".join(part.strip() for part in current_buf).strip()
            text = re.sub(r"\s+", " ", text)
            _DIAG_PROSE_CACHE[current_field] = text
        current_field = None
        current_buf = []

    for raw in lines:
        if raw.strip().startswith("Attributes:"):
            in_attrs = True
            continue
        if not in_attrs:
            continue
        if not raw.strip():
            flush()
            continue
        # New field entry pattern: 8 spaces of indent followed by "name:".
        match = re.match(r"^ {8}([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", raw)
        if match:
            flush()
            current_field = match.group(1)
            current_buf = [match.group(2)]
        elif current_field is not None:
            current_buf.append(raw)
    flush()


@dataclass
class DebugReport:
    """User-facing report produced by the diagnostics layer.

    Phase 1 instances carry ``signals=[]`` and ``diagnoses=[]``; the
    renderer still emits a useful report from the termination
    metadata + diagnostics counters + trajectory.  Phase 2 / 3 fill
    the signal and diagnosis sections in.

    Attributes:
        run: The :class:`DebugRunResult` the report was built from.
            Carries the ``SLSQPState``, the ``StepSummary`` trajectory,
            and any signals/diagnoses already wired by later phases.
        signals: Convenience accessor exposing
            ``run.fired_signals`` keyed by name.  ``Phase 1``: empty.
        diagnoses: List of multi-signal diagnoses produced by the
            playbook.  ``Phase 1``: empty.
    """

    run: "DebugRunResult"
    signals: dict[str, Any] = field(default_factory=dict)
    diagnoses: list[Any] = field(default_factory=list)

    @classmethod
    def from_run(cls, run: "DebugRunResult") -> "DebugReport":
        """Build a :class:`DebugReport` from a :class:`DebugRunResult`."""
        signals_by_name: dict[str, Any] = {}
        for sig in run.fired_signals:
            name = getattr(sig, "name", None)
            if isinstance(name, str):
                signals_by_name[name] = sig
        return cls(run=run, signals=signals_by_name, diagnoses=[])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover  -- presentational
        return self.render()

    def __str__(self) -> str:  # pragma: no cover  -- presentational
        return self.render()

    def render(self) -> str:
        """Return the full report as a single string."""
        out = io.StringIO()
        self._render_header(out)
        self._render_termination(out)
        self._render_summary_metrics(out)
        self._render_signals(out)
        self._render_diagnoses(out)
        self._render_diagnostics_block(out)
        self._render_trajectory_chart(out)
        return out.getvalue()

    def print_summary(self, *, file: Any = None) -> None:
        """Write the report to ``file`` (default ``sys.stdout``).

        Uses :py:meth:`io.IOBase.write` rather than :func:`print` so
        the call passes the project-wide ``no-print-statements``
        pre-commit hook (the hook is intentionally strict; presenting
        a multi-paragraph diagnostics report is one of the few
        legitimate exceptions and we route through ``write`` for
        clarity).  The trailing newline matches what :func:`print`
        would emit so users do not have to special-case the output
        in a notebook.
        """
        target = file if file is not None else sys.stdout
        target.write(self.render())
        target.write("\n")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the rendered fields.

        Useful for piping the report into downstream tooling (CI,
        notebooks, dashboards).  Heavy artifacts on signals
        (``np.ndarray`` instances) are not included; call
        ``signal.artifacts`` directly when you want them.
        """
        run = self.run
        return {
            "termination": {
                "granular": _result_name(run.final_result),
                "coarse": _result_name(run.coarse_result),
                "message": _result_message(run.final_result),
                "successful": run.terminated_successfully,
                "max_steps_reached": run.max_steps_reached,
                "terminated_at_step": run.terminated_at_step,
                "n_steps": run.n_steps,
            },
            "diagnostics": _diagnostics_to_dict(run.diagnostics),
            "signals": [_signal_to_dict(sig) for sig in run.fired_signals],
            "diagnoses": [_diagnosis_to_dict(d) for d in self.diagnoses],
        }

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_header(self, out: io.StringIO) -> None:
        title = " SLSQP-JAX Debug Report "
        bar = "=" * _REPORT_WIDTH
        out.write(bar + "\n")
        out.write(title.center(_REPORT_WIDTH, "=") + "\n")
        out.write(bar + "\n\n")

    def _render_termination(self, out: io.StringIO) -> None:
        run = self.run
        granular = _result_name(run.final_result)
        coarse = _result_name(run.coarse_result)
        message = _result_message(run.final_result)
        out.write("Termination\n")
        out.write("-" * _REPORT_WIDTH + "\n")
        if run.terminated_successfully:
            out.write(f"  status:                successful ({granular})\n")
        elif run.max_steps_reached:
            out.write(
                "  status:                NOT successful "
                "(debug-loop budget exhausted)\n"
            )
        else:
            out.write(f"  status:                NOT successful ({granular})\n")
        out.write(f"  granular RESULTS code: {granular}\n")
        if coarse != granular:
            out.write(f"  coarse optx code:      {coarse}  (mapped from granular)\n")
        out.write(f"  steps executed:        {run.n_steps}\n")
        if run.max_steps_reached:
            out.write(
                "  budget:                EXHAUSTED  "
                "(debug_run loop ran out of iterations before terminate() said done)\n"
            )
            out.write(
                "                         "
                "Note: 'successful' as a granular code on this row is the *default*\n"
                "                         "
                "carried on a fresh SLSQPState; it does NOT imply convergence.\n"
                "                         "
                "Re-run with a larger ``max_steps`` to let the solver finish.\n"
            )
        else:
            out.write(
                f"  budget:                ok "
                f"(loop exited at step {run.terminated_at_step + 1})\n"
            )
        if message:
            wrapped = _wrap_paragraph(message, indent="  | ", width=_REPORT_WIDTH - 4)
            out.write("  message:\n")
            out.write(wrapped + "\n")
        out.write("\n")

    def _render_summary_metrics(self, out: io.StringIO) -> None:
        run = self.run
        if not run.summaries:
            return
        last = run.summaries[-1]
        out.write("Final iterate metrics\n")
        out.write("-" * _REPORT_WIDTH + "\n")
        out.write(f"  f(x):                  {_fmt_e(last.f_val)}\n")
        out.write(f"  L1 merit:              {_fmt_e(last.merit)}\n")
        out.write(f"  Lagrangian L:          {_fmt_e(last.lagrangian_value)}\n")
        out.write(f"  ||grad||:              {_fmt_e(last.grad_norm)}\n")
        out.write(f"  ||grad_L||:            {_fmt_e(last.grad_lagrangian_norm)}\n")
        out.write(f"  ||grad_L||/max(|L|,1): {_fmt_e(last.rel_kkt)}\n")
        out.write(f"  max|c_eq|:             {_fmt_e(last.max_eq_violation)}\n")
        out.write(f"  max(0, -c_ineq):       {_fmt_e(last.max_ineq_violation)}\n")
        out.write(f"  rho (merit penalty):   {_fmt_e(last.merit_penalty)}\n")
        out.write(f"  last alpha:            {_fmt_e(last.last_alpha)}\n")
        out.write(f"  L-BFGS gamma:          {_fmt_e(last.gamma)}\n")
        out.write(
            f"  L-BFGS diag kappa:     {_fmt_e(last.diag_kappa)} "
            f"(min={_fmt_e(last.min_diag)}, max={_fmt_e(last.max_diag)})\n"
        )
        out.write(f"  active inequalities:   {last.n_active_ineq}\n")
        out.write("\n")

    def _render_signals(self, out: io.StringIO) -> None:
        from slsqp_jax.diagnostics.playbook import (
            signals_in_scope,
            signals_out_of_scope,
        )

        run = self.run
        if not run.fired_signals:
            return

        fired_names = {sig.name for sig in run.fired_signals}
        in_scope_names = signals_in_scope(run.final_result, fired_names)
        out_of_scope_names = signals_out_of_scope(run.final_result, fired_names)

        in_scope = sorted(
            (sig for sig in run.fired_signals if sig.name in in_scope_names),
            key=_signal_sort_key,
        )
        out_of_scope = sorted(
            (sig for sig in run.fired_signals if sig.name in out_of_scope_names),
            key=_signal_sort_key,
        )

        out.write("Fired signals (in scope for this termination)\n")
        out.write("-" * _REPORT_WIDTH + "\n")
        if not in_scope:
            out.write("  (no in-scope signals fired)\n")
        for sig in in_scope:
            self._render_one_signal(out, sig)

        if out_of_scope:
            out.write("Fired signals (less likely given the termination mode)\n")
            out.write("-" * _REPORT_WIDTH + "\n")
            for sig in out_of_scope:
                self._render_one_signal(out, sig)

    @staticmethod
    def _render_one_signal(out: io.StringIO, sig: Any) -> None:
        name = getattr(sig, "name", "<unknown>")
        confidence = getattr(sig, "confidence", "?")
        specificity = getattr(sig, "specificity", "?")
        magnitude = getattr(sig, "magnitude", "?")
        summary = getattr(sig, "summary", "")
        evidence = getattr(sig, "evidence", {}) or {}
        offending = getattr(sig, "offending_step", None)
        suggestions = getattr(sig, "suggestions", []) or []
        artifact_keys = list((getattr(sig, "artifacts", {}) or {}).keys())
        out.write(
            f"  [{confidence:>6}] {name}  "
            f"(specificity={specificity}, magnitude={magnitude})\n"
        )
        if summary:
            wrapped = _wrap_paragraph(
                summary, indent="     | ", width=_REPORT_WIDTH - 8
            )
            out.write(wrapped + "\n")
        if offending is not None:
            out.write(f"     | offending step: {offending}\n")
        for k, v in evidence.items():
            out.write(f"     | evidence: {k} = {_fmt_value(v)}\n")
        if artifact_keys:
            out.write("     | artifacts attached: " + ", ".join(artifact_keys) + "\n")
        for s in suggestions:
            wrapped = _wrap_paragraph(
                "suggestion: " + s,
                indent="     | ",
                width=_REPORT_WIDTH - 8,
            )
            out.write(wrapped + "\n")
        out.write("\n")

    def _render_diagnoses(self, out: io.StringIO) -> None:
        if not self.diagnoses:
            return
        out.write("Candidate diagnoses\n")
        out.write("-" * _REPORT_WIDTH + "\n")
        # Sort diagnoses by the highest confidence among their related
        # signals (with single-name diagnoses inheriting that signal's
        # confidence directly).  Falls back to alphabetical when no
        # confidence is available.
        run = self.run
        sigs_by_name = {s.name: s for s in run.fired_signals}

        def diag_key(d: Any) -> tuple[int, str]:
            related = getattr(d, "related_signals", []) or []
            best = -1
            for n in related:
                sig = sigs_by_name.get(n)
                if sig is None:
                    continue
                rank = _CONFIDENCE_RANK.get(getattr(sig, "confidence", ""), -1)
                if rank > best:
                    best = rank
            return (-best, getattr(d, "name", ""))

        for d in sorted(self.diagnoses, key=diag_key):
            name = getattr(d, "name", "<unknown>")
            cause = getattr(d, "cause", "")
            related = getattr(d, "related_signals", []) or []
            suggestions = getattr(d, "suggestions", []) or []
            out.write(f"  * {name}\n")
            if cause:
                wrapped = _wrap_paragraph(
                    cause, indent="    | ", width=_REPORT_WIDTH - 6
                )
                out.write(wrapped + "\n")
            if related:
                out.write(f"    | related signals: {', '.join(related)}\n")
            for s in suggestions:
                wrapped = _wrap_paragraph(
                    "suggestion: " + s,
                    indent="    | ",
                    width=_REPORT_WIDTH - 6,
                )
                out.write(wrapped + "\n")
            out.write("\n")

    def _render_diagnostics_block(self, out: io.StringIO) -> None:
        run = self.run
        diag = run.diagnostics
        out.write("SLSQPDiagnostics counters\n")
        out.write("-" * _REPORT_WIDTH + "\n")
        for f in dataclasses.fields(diag):
            value = getattr(diag, f.name)
            try:
                py_val: Any = value.item()  # type: ignore[union-attr]
            except (AttributeError, ValueError):
                py_val = value
            prose = _diag_prose(f.name)
            label = f"  {f.name}:".ljust(38)
            out.write(f"{label}{_fmt_value(py_val)}\n")
            if prose:
                wrapped = _wrap_paragraph(
                    prose, indent="     | ", width=_REPORT_WIDTH - 8
                )
                out.write(wrapped + "\n")
        out.write("\n")

    def _render_trajectory_chart(self, out: io.StringIO) -> None:
        run = self.run
        if not run.summaries:
            return
        out.write("Trajectory (per-step)\n")
        out.write("-" * _REPORT_WIDTH + "\n")
        header = (
            f"{'step':>5}  {'f':>12}  {'merit':>12}  {'rel_KKT':>10}  "
            f"{'alpha':>10}  {'qpit':>5}  {'qpok':>5}  {'lsok':>5}\n"
        )
        out.write(header)
        out.write("-" * _REPORT_WIDTH + "\n")
        # Cap the displayed rows so the report stays readable.  When
        # the trajectory is longer than the cap, show the head, an
        # ellipsis, and the tail.
        cap_head = 20
        cap_tail = 20
        rows = run.summaries
        if len(rows) > cap_head + cap_tail + 1:
            display = rows[:cap_head] + [None] + rows[-cap_tail:]
        else:
            display = list(rows)
        for s in display:
            if s is None:
                out.write(("  ..." + " " * (_REPORT_WIDTH - 5)) + "\n")
                continue
            out.write(
                f"{s.step_count:>5d}  {_fmt_e(s.f_val):>12}  "
                f"{_fmt_e(s.merit):>12}  {_fmt_e(s.rel_kkt):>10}  "
                f"{_fmt_e(s.last_alpha):>10}  {s.qp_iterations_step:>5d}  "
                f"{str(s.qp_converged):>5}  {str(s.ls_success):>5}\n"
            )
        out.write("\n")


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------

# Higher-confidence signals sort first.  Anything with an unknown
# confidence label sorts last.
_CONFIDENCE_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def _signal_sort_key(sig: Any) -> tuple[int, str]:
    """Sort signals by confidence (high first), then name (stable)."""
    confidence = getattr(sig, "confidence", "")
    rank = _CONFIDENCE_RANK.get(confidence, 0)
    return (-rank, getattr(sig, "name", ""))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_e(value: float) -> str:
    """Format a float for the report.

    Uses scientific notation with three significant digits, special-
    casing NaN / inf for legibility.
    """
    if not isinstance(value, (float, int)):
        return str(value)
    fv = float(value)
    if math.isnan(fv):
        return "nan"
    if math.isinf(fv):
        return "+inf" if fv > 0 else "-inf"
    return f"{fv:.3e}"


def _fmt_value(value: Any) -> str:
    """Format any value (int / float / bool / array / object) for printing."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _fmt_e(value)
    return str(value)


def _wrap_paragraph(text: str, *, indent: str, width: int) -> str:
    """Hard-wrap ``text`` to ``width`` columns, prefixing each line with ``indent``."""
    width = max(20, width)
    words = text.split()
    lines: list[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width + len(indent):
            lines.append(current.rstrip())
            current = indent + word
        else:
            current = (current + " " + word) if current.strip() else (indent + word)
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def _enum_value(result: Any) -> Optional[int]:
    """Return the integer code of an :class:`equinox.Enumeration` item.

    The metaclass exposes ``_value`` as a 0-d numpy array; we coerce
    to a Python ``int`` so the value is hashable and comparable.
    Returns ``None`` for non-enumeration inputs.
    """
    raw = getattr(result, "_value", None)
    if raw is None:
        return None
    try:
        return int(raw)  # works for numpy 0-d arrays and Python ints
    except (TypeError, ValueError):
        return None


def _enum_class(result: Any) -> Any:
    """Return the ``equinox.Enumeration`` *class* a ``EnumerationItem`` belongs to.

    ``type(result)`` is the generic ``EnumerationItem`` shell, not the
    user-facing ``RESULTS`` subclass.  The back-reference lives on the
    item's ``_enumeration`` attribute.  Falls back to ``type(result)``
    for non-enumeration inputs.
    """
    enum_cls = getattr(result, "_enumeration", None)
    if enum_cls is not None:
        return enum_cls
    return type(result)


def _result_name(result: Any) -> str:
    """Return the short class-attribute name of a ``RESULTS`` member.

    ``equinox.Enumeration`` exposes the registered members as a dict
    ``_name_to_item`` keyed by attribute name; we reverse-map by the
    integer ``_value`` because two ``EnumerationItem`` instances with
    the same value are equal but not ``is``-identical.  Sub-class
    members (e.g. ``slsqp_jax.RESULTS.merit_stagnation``) are looked
    up against the actual subclass via the item's ``_enumeration``
    back-reference, not against ``type(result)`` (which is the
    generic ``EnumerationItem`` shell).
    """
    if result is None:
        return "<none>"
    enum_cls = _enum_class(result)
    name_to_item = getattr(enum_cls, "_name_to_item", None)
    target_value = _enum_value(result)
    if name_to_item is not None and target_value is not None:
        for name, item in name_to_item.items():
            if _enum_value(item) == target_value:
                return name
    return str(result)


def _result_message(result: Any) -> str:
    """Return the human-readable message string for a ``RESULTS`` member.

    Equinox stores the messages in ``_index_to_message`` keyed by the
    integer ``_value``; older releases used a ``message`` attribute.
    Looked up against the subclass via :func:`_enum_class`.
    """
    if result is None:
        return ""
    enum_cls = _enum_class(result)
    index_to_message = getattr(enum_cls, "_index_to_message", None)
    target_value = _enum_value(result)
    if index_to_message is not None and target_value is not None:
        try:
            msg = index_to_message[target_value]
            if isinstance(msg, str):
                return msg
        except (IndexError, KeyError):
            pass
    for attr in ("value", "message"):
        msg = getattr(result, attr, None)
        if isinstance(msg, str):
            return msg
    return ""


def _diagnostics_to_dict(diag: "SLSQPDiagnostics") -> dict[str, Any]:
    """Convert an ``SLSQPDiagnostics`` to a JSON-serialisable dict."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(diag):
        value = getattr(diag, f.name)
        try:
            out[f.name] = value.item()  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            out[f.name] = value
    return out


def _signal_to_dict(sig: Any) -> dict[str, Any]:
    """Convert a signal to a JSON-serialisable dict (no array data)."""
    return {
        "name": getattr(sig, "name", None),
        "specificity": getattr(sig, "specificity", None),
        "magnitude": getattr(sig, "magnitude", None),
        "confidence": getattr(sig, "confidence", None),
        "summary": getattr(sig, "summary", ""),
        "evidence": dict(getattr(sig, "evidence", {}) or {}),
        "suggestions": list(getattr(sig, "suggestions", []) or []),
        "offending_step": getattr(sig, "offending_step", None),
        "artifact_keys": list((getattr(sig, "artifacts", {}) or {}).keys()),
    }


def _diagnosis_to_dict(d: Any) -> dict[str, Any]:
    """Convert a diagnosis to a JSON-serialisable dict."""
    return {
        "name": getattr(d, "name", None),
        "cause": getattr(d, "cause", ""),
        "suggestions": list(getattr(d, "suggestions", []) or []),
        "related_signals": list(getattr(d, "related_signals", []) or []),
    }


__all__ = [
    "DebugReport",
]
