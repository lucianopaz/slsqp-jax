"""Validate the catalogue against the scipy solvers (``python -m benchmarks validate``).

Two benchmark artefacts dominate the "unsolved by everybody" bucket of the
diagnostics report and both are properties of the *instance*, not of the
solver:

1. **Inequality sign.** sif2jax documents ``constraint(y) -> (eq, ineq)``
   with ``ineq >= 0``, but a number of problems return the raw pycutest
   value ``c(x) - b`` of an L-type group (``c(x) <= b``), i.e. a ``<= 0``
   quantity, and some mix both conventions row by row. Converted with the
   documented sign the feasible side is wrong and the problem is typically
   unbounded (the minimax family: ``u <= f_i(x)`` instead of
   ``u >= f_i(x)``).
2. **Reference objective.** ``expected_objective_value`` may belong to a
   different local minimum than the one every local method reaches from the
   provided start, or be plainly inconsistent with the problem.

For every instance :func:`validate_instance` tries the candidate sign
conventions (documented, rows that the reference minimiser violates, all
rows flipped) with scipy SLSQP, keeps the first one under which the
instance is solved consistently with its reference, then runs
``trust-constr`` on the chosen conversion to obtain a second, independent
objective value. The outcome is written back into ``benchmarks/catalog.csv``
(see :data:`benchmarks.catalog.VALIDATION_COLUMNS`) and read by
:func:`benchmarks.problems.to_sqpdax` at run time:

``ineq_flip``
    rows converted as ``h = c`` instead of ``h = -c``;
``fstar`` / ``fstar_source``
    the reference used for scoring - sif2jax's value when a scipy solver
    reaches it, the scipy consensus when both solvers agree on a feasible
    value that differs from it (or when sif2jax has none);
``valid`` / ``validation``
    ``False`` with ``unbounded`` / ``infeasible`` / ``evaluation_error``
    when no conversion yields a bounded feasible problem; such instances
    are excluded by :func:`benchmarks.catalog.select` unless asked for.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "ValidateSpec",
    "apply_validation",
    "candidate_flips",
    "decide",
    "validate_instance",
]

FEAS_TOL = 1e-6
"""Feasibility tolerance of a validating solve (benchmark default)."""

F_TOL = 1e-4
"""Relative objective tolerance for agreeing with a reference value."""

DIVERGED = -1e10
"""Objective below which a feasible solve counts as unbounded."""


def _rel_gap(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def candidate_flips(
    mineq: int, c_star: Any | None, *, tol: float = 1e-6
) -> list[tuple[str, str]]:
    """Sign conventions to try, most plausible first.

    Parameters
    ----------
    mineq
        Number of inequality rows.
    c_star
        sif2jax's raw inequality values at the reference minimiser, or
        ``None`` when no (finite) reference minimiser exists.
    tol
        Rows with ``c_i(x*) < -tol * max(1, |c_i|)`` violate the documented
        ``c >= 0`` convention and are candidates for flipping.

    Returns
    -------
    list[tuple[str, str]]
        ``(label, ineq_flip)`` pairs: ``("documented", "")``, then
        ``("xstar_rows", "<indices>")`` when the reference minimiser points
        at specific rows, then ``("all", "all")``.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.validate import candidate_flips
    >>> candidate_flips(3, np.array([0.5, -2.0, 0.0]))
    [('documented', ''), ('xstar_rows', '1'), ('all', 'all')]
    >>> candidate_flips(0, None)
    [('documented', '')]
    """
    import numpy as np

    from .problems import format_flip

    out: list[tuple[str, str]] = [("documented", "")]
    if mineq == 0:
        return out
    if c_star is not None:
        c = np.ravel(np.asarray(c_star, dtype=float))
        if c.shape == (mineq,) and np.all(np.isfinite(c)):
            bad = c < -tol * np.maximum(1.0, np.abs(c))
            spec = format_flip(bad, mineq)
            if spec and spec != "all":
                out.append(("xstar_rows", spec))
    out.append(("all", "all"))
    return out


def _summarise(
    r: dict[str, Any], ref: float | None, feas_tol: float, f_tol: float
) -> dict[str, Any]:
    """Derived facts about one candidate conversion."""

    def feasible(key: str) -> bool:
        v = r.get(f"{key}_feas", float("nan"))
        f = r.get(f"{key}_f", float("nan"))
        return bool(v == v and v <= feas_tol and f == f and math.isfinite(f))

    def diverged(key: str) -> bool:
        f = r.get(f"{key}_f", float("nan"))
        v = r.get(f"{key}_feas", float("nan"))
        return bool(f == f and f < DIVERGED and (v != v or v <= 1e-3))

    feasible_f = [float(r[f"{k}_f"]) for k in ("slsqp", "tc") if feasible(k)]
    best = min(feasible_f) if feasible_f else float("nan")
    xf = r.get("xstar_feas", float("nan"))
    xfv = r.get("xstar_f", float("nan"))
    # Published minimisers are rounded to a few digits: allow a looser
    # violation than for a computed point.
    xstar_ok = bool(
        xf == xf
        and xf <= 10.0 * feas_tol
        and xfv == xfv
        and (ref is None or _rel_gap(float(xfv), ref) <= f_tol)
    )
    target = ref if ref is not None else (float(xfv) if xstar_ok else None)
    contradicted = bool(
        xstar_ok
        and feasible_f
        and target is not None
        and best < target - f_tol * max(1.0, abs(target))
    )
    consensus = None
    if feasible("slsqp") and feasible("tc"):
        a, b = float(r["slsqp_f"]), float(r["tc_f"])
        if _rel_gap(a, b) <= f_tol:
            consensus = min(a, b)
    feas_values = [
        float(r[f"{k}_feas"])
        for k in ("slsqp", "tc")
        if r.get(f"{k}_feas", float("nan")) == r.get(f"{k}_feas", float("nan"))
    ]
    return {
        "xstar_ok": xstar_ok,
        "contradicted": contradicted,
        "reaches_ref": bool(
            ref is not None and any(_rel_gap(f, ref) <= f_tol for f in feasible_f)
        ),
        "consensus": consensus,
        "best": best,
        "bounded_feasible": bool(feasible_f) and best > DIVERGED,
        "slsqp_success": bool(r.get("slsqp_success")),
        "diverged": diverged("slsqp") or diverged("tc"),
        "min_feas": min(feas_values) if feas_values else float("nan"),
    }


def decide(
    runs: dict[str, dict[str, Any]],
    *,
    fstar_sif: float | None,
    feas_tol: float = FEAS_TOL,
    f_tol: float = F_TOL,
) -> dict[str, Any]:
    """Pick the conversion and the scoring reference from the validating solves.

    Parameters
    ----------
    runs
        ``{ineq_flip: {"label", "slsqp_f", "slsqp_feas", "slsqp_success",
        "tc_f", "tc_feas", "tc_success", "xstar_feas", "xstar_f"}}`` in
        trial order (documented conversion first); every entry but
        ``label`` is optional.
    fstar_sif
        sif2jax's reference objective, ``None`` when unavailable.
    feas_tol, f_tol
        Tolerances defining a consistent solve.

    Returns
    -------
    dict[str, Any]
        ``ineq_flip``, ``fstar``, ``fstar_source``, ``valid``,
        ``validation`` (``ok``, ``sign_flipped``, ``fstar_mismatch``,
        ``fstar_from_consensus``, ``unverified``, ``unbounded``,
        ``infeasible``), ``validation_note``.

    Notes
    -----
    Evidence is ranked as follows; the first conversion (in trial order)
    satisfying a rank wins.

    1. The reference minimiser ``x*`` is feasible, attains ``f*`` and no
       scipy solver finds a *lower* feasible value (a lower value means the
       feasible side is wrong or ``f*`` is not a minimum).
    2. A scipy solver reaches ``f*`` from ``x0``.
    3. SLSQP and trust-constr agree on a feasible value: that value becomes
       the reference (``scipy-consensus``).
    4. SLSQP alone converges to a bounded feasible point: ``f*`` is kept
       but marked ``unverified`` (or adopted when sif2jax has none).
    5. Some run ends at a bounded feasible point without converging; the
       documented conversion is kept unless it diverges.
    6. Divergence under every conversion -> ``unbounded`` (invalid); no
       feasible point under any -> ``infeasible`` (invalid).

    Examples
    --------
    >>> from benchmarks.validate import decide
    >>> runs = {"": dict(label="documented", slsqp_f=-7e30, slsqp_feas=0.0,
    ...                  slsqp_success=False),
    ...         "all": dict(label="all", slsqp_f=0.0, slsqp_feas=1e-12,
    ...                     slsqp_success=True, tc_f=1e-9, tc_feas=0.0, tc_success=True)}
    >>> d = decide(runs, fstar_sif=0.0)
    >>> d["ineq_flip"], d["validation"], d["fstar_source"]
    ('all', 'sign_flipped', 'sif2jax')
    """
    ref: float | None = (
        float(fstar_sif) if fstar_sif is not None and math.isfinite(fstar_sif) else None
    )
    facts = {flip: _summarise(r, ref, feas_tol, f_tol) for flip, r in runs.items()}
    base = dict(valid=True, validation_note="")

    def label(flip: str) -> str:
        return str(runs[flip].get("label", flip or "documented"))

    def kind(flip: str) -> str:
        return "ok" if flip == "" else "sign_flipped"

    # 1. x* feasible, attains f*, not undercut by scipy.
    for flip, s in facts.items():
        if s["xstar_ok"] and not s["contradicted"]:
            how = (
                "scipy reaches f*"
                if s["reaches_ref"]
                else "scipy does not reach f* from x0"
            )
            fstar = ref if ref is not None else float(runs[flip]["xstar_f"])
            return {
                **base,
                "ineq_flip": flip,
                "fstar": fstar,
                "fstar_source": "sif2jax",
                "validation": kind(flip),
                "validation_note": f"{label(flip)}: x* feasible, {how}",
            }
    # 2. scipy reaches f*.
    for flip, s in facts.items():
        if s["reaches_ref"]:
            return {
                **base,
                "ineq_flip": flip,
                "fstar": ref,
                "fstar_source": "sif2jax",
                "validation": kind(flip),
                "validation_note": f"{label(flip)}: scipy reaches f*",
            }
    # 3. SLSQP / trust-constr consensus.
    for flip, s in facts.items():
        if s["consensus"] is not None and s["consensus"] > DIVERGED:
            r = runs[flip]
            if ref is not None:
                note = (
                    f"{label(flip)}: SLSQP {r['slsqp_f']:.8g} and trust-constr "
                    f"{r['tc_f']:.8g} agree; sif2jax f*={ref:.8g} not reached"
                )
                if s["contradicted"]:
                    note += " and x* is not a minimiser under this conversion"
                validation = "fstar_mismatch"
            else:
                note = f"{label(flip)}: SLSQP and trust-constr agree, no sif2jax f*"
                validation = "fstar_from_consensus"
            if flip:
                note += " (sign flipped)"
            return {
                **base,
                "ineq_flip": flip,
                "fstar": float(s["consensus"]),
                "fstar_source": "scipy-consensus",
                "validation": validation,
                "validation_note": note,
            }
    # 4. SLSQP alone converged to a bounded feasible point.
    for flip, s in facts.items():
        if s["bounded_feasible"] and s["slsqp_success"]:
            r = runs[flip]
            if ref is not None:
                return {
                    **base,
                    "ineq_flip": flip,
                    "fstar": ref,
                    "fstar_source": "sif2jax",
                    "validation": "unverified",
                    "validation_note": (
                        f"{label(flip)}: SLSQP feasible at {r['slsqp_f']:.8g}, "
                        f"f*={ref:.8g} not reached, no trust-constr consensus"
                    ),
                }
            return {
                **base,
                "ineq_flip": flip,
                "fstar": float(r["slsqp_f"]),
                "fstar_source": "scipy-consensus",
                "validation": "fstar_from_consensus",
                "validation_note": f"{label(flip)}: SLSQP alone, no sif2jax f*",
            }
    # 5. bounded feasible end point without convergence.
    documented_diverges = bool(facts.get("", {}).get("diverged"))
    for flip, s in facts.items():
        if s["bounded_feasible"]:
            keep_flip = flip == "" or documented_diverges
            return {
                **base,
                "ineq_flip": flip if keep_flip else "",
                "fstar": ref if ref is not None else float("nan"),
                "fstar_source": "sif2jax",
                "validation": "unverified",
                "validation_note": (
                    f"{label(flip)}: bounded feasible end point at {s['best']:.8g} "
                    "without convergence"
                    + (
                        "; documented conversion diverges"
                        if documented_diverges
                        else ""
                    )
                ),
            }
    # 6. nothing usable.
    fstar = ref if ref is not None else float("nan")
    if any(s["diverged"] for s in facts.values()):
        return {
            **base,
            "ineq_flip": "",
            "fstar": fstar,
            "fstar_source": "sif2jax",
            "valid": False,
            "validation": "unbounded",
            "validation_note": "objective diverges under every sign convention",
        }
    feas_values = [
        s["min_feas"] for s in facts.values() if s["min_feas"] == s["min_feas"]
    ]
    if feas_values and min(feas_values) > 1e-4:
        return {
            **base,
            "ineq_flip": "",
            "fstar": fstar,
            "fstar_source": "sif2jax",
            "valid": False,
            "validation": "infeasible",
            "validation_note": (
                f"best violation {min(feas_values):.3g} under every sign convention"
            ),
        }
    first = next(iter(runs.values()), {})
    return {
        **base,
        "ineq_flip": "",
        "fstar": fstar,
        "fstar_source": "sif2jax",
        "validation": "unverified",
        "validation_note": first.get("error", "") or "no scipy solver converged",
    }


@dataclass(frozen=True)
class ValidateSpec:
    """One instance to validate; runs inside the benchmark worker pool.

    Attributes
    ----------
    problem, y0_iD
        Instance.
    max_steps
        ``maxiter`` of the scipy solves.
    timeout_s
        Wall-clock budget enforced by the pool.
    tc_max_n
        Largest ``n`` for which ``trust-constr`` is run (it is much slower
        than SLSQP on dense problems).
    """

    problem: str
    y0_iD: int = 0
    max_steps: int = 500
    timeout_s: float = 600.0
    tc_max_n: int = 300
    config: str = "validate"

    def run(self, report=None) -> dict[str, Any]:
        return validate_instance(
            self.problem,
            self.y0_iD,
            max_steps=self.max_steps,
            tc_max_n=self.tc_max_n,
            report=report,
        )


def _solve(problem, x0, method: str, max_steps: int) -> dict[str, Any]:
    import jax.numpy as jnp
    import numpy as np

    from .baselines import ScipyBaseline, ScipyRunner
    from .metrics import quality_metrics

    key = "slsqp" if method == "SLSQP" else "tc"
    options = {"ftol": 1e-10} if method == "SLSQP" else {"gtol": 1e-8, "xtol": 1e-10}
    try:
        runner = ScipyRunner(problem, x0, ScipyBaseline(method), options)
        runner.compile()
        out = runner.solve(max_steps)
        q = quality_metrics(problem, jnp.asarray(out.x), None)
        f = q["objective"]
        return {
            f"{key}_f": f if np.isfinite(f) else float("nan"),
            f"{key}_feas": q["feas"],
            f"{key}_success": bool(out.successful),
            f"{key}_status": out.status,
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return {
            f"{key}_f": float("nan"),
            f"{key}_feas": float("nan"),
            f"{key}_success": False,
            f"{key}_status": "error",
            "error": repr(exc)[:300],
        }


def validate_instance(
    name: str,
    y0_iD: int = 0,
    *,
    max_steps: int = 500,
    tc_max_n: int = 300,
    report=None,
) -> dict[str, Any]:
    """Validate one instance (see the module docstring).

    Parameters
    ----------
    name, y0_iD
        Instance.
    max_steps
        ``maxiter`` of the scipy solves.
    tc_max_n
        Largest ``n`` for which ``trust-constr`` is run.
    report
        Optional ``report(kind, payload)`` progress callback.

    Returns
    -------
    dict[str, Any]
        ``name``, ``y0_iD``, the :func:`decide` outcome, ``fstar_sif``, the
        scipy objective / violation under the chosen conversion
        (``ref_slsqp_f``, ``ref_slsqp_feas``, ``ref_tc_f``, ``ref_tc_feas``)
        and ``runs`` (every trial).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import sif2jax

    from .metrics import quality_metrics
    from .problems import to_sqpdax

    jax.config.update("jax_enable_x64", True)
    t0 = time.perf_counter()

    def phase(p: str) -> None:
        if report is not None:
            report("phase", p)

    phase("build")
    raw = sif2jax.cutest.get_problem(name)
    if y0_iD != raw.y0_iD:
        raw = type(raw)(y0_iD=y0_iD)
    problem, x0, meta = to_sqpdax(raw, use_catalog=False)
    fstar_sif = meta.fstar if meta.has_fstar else None
    row: dict[str, Any] = {"name": name, "y0_iD": int(y0_iD), "fstar_sif": meta.fstar}

    f0 = float(problem.fn(x0)[0])
    if not np.isfinite(f0) or not bool(jnp.all(jnp.isfinite(problem.grad(x0)))):
        row.update(
            ineq_flip="",
            fstar=meta.fstar,
            fstar_source="sif2jax",
            valid=False,
            validation="evaluation_error",
            validation_note="objective or gradient not finite at x0",
            runs={},
            wall_s=time.perf_counter() - t0,
        )
        return row

    c_star = None
    xs = None
    try:
        xstar = raw.expected_result
    except NotImplementedError:
        xstar = None
    if xstar is not None:
        cand = jnp.ravel(jnp.asarray(xstar, dtype=jnp.float64))
        if cand.shape == x0.shape and bool(jnp.all(jnp.isfinite(cand))):
            xs = cand
            if meta.mineq:
                c_star = np.asarray(raw.constraint(xs)[1], dtype=float)

    problems: dict[str, Any] = {"": problem}

    def converted(flip: str):
        if flip not in problems:
            problems[flip] = to_sqpdax(raw, ineq_flip=flip, use_catalog=False)[0]
        return problems[flip]

    candidates = candidate_flips(meta.mineq, c_star)
    runs: dict[str, dict[str, Any]] = {
        flip: {"label": label} for label, flip in candidates
    }
    if xs is not None:
        # Feasibility of the reference minimiser under each conversion; when
        # it is feasible under some of them only those are worth solving.
        for _label, flip in candidates:
            q = quality_metrics(converted(flip), xs, None)
            runs[flip]["xstar_feas"] = q["feas"]
            runs[flip]["xstar_f"] = q["objective"]
        consistent = [
            flip
            for _label, flip in candidates
            if runs[flip]["xstar_feas"] <= 10.0 * FEAS_TOL
            and (
                fstar_sif is None or _rel_gap(runs[flip]["xstar_f"], fstar_sif) <= F_TOL
            )
        ]
        if consistent:
            candidates = [
                (label, flip) for label, flip in candidates if flip in consistent
            ]

    for label, flip in candidates:
        phase(f"slsqp:{label}")
        runs[flip].update(_solve(converted(flip), x0, "SLSQP", max_steps))

    ordered = {flip: runs[flip] for _label, flip in candidates}
    # trust-constr gives the second opinion on the conversion SLSQP points
    # at (the preliminary decision); it also certifies boundedness /
    # feasibility when SLSQP converged nowhere.
    chosen = decide(ordered, fstar_sif=fstar_sif)["ineq_flip"]
    if chosen not in ordered:
        chosen = candidates[0][1]
    if meta.n <= tc_max_n:
        phase("trust-constr")
        runs[chosen].update(_solve(converted(chosen), x0, "trust-constr", max_steps))
    decision = decide(ordered, fstar_sif=fstar_sif)
    picked = runs.get(decision["ineq_flip"], runs.get(chosen, {}))
    row.update(decision)
    row.update(
        ref_slsqp_f=picked.get("slsqp_f", float("nan")),
        ref_slsqp_feas=picked.get("slsqp_feas", float("nan")),
        ref_tc_f=picked.get("tc_f", float("nan")),
        ref_tc_feas=picked.get("tc_feas", float("nan")),
        runs=runs,
        wall_s=time.perf_counter() - t0,
    )
    return row


def apply_validation(
    catalog: pd.DataFrame, results: list[dict[str, Any]]
) -> pd.DataFrame:
    """Write validation outcomes into the catalogue.

    Parameters
    ----------
    catalog
        Catalogue frame (validation columns are added when missing).
    results
        Rows from :func:`validate_instance`, or harness rows for timed-out /
        crashed instances (``status`` set, no ``validation`` key): those are
        recorded as ``validation = "timeout"`` / ``"error"`` and keep the
        sif2jax reference.

    Returns
    -------
    pandas.DataFrame
        Updated catalogue; ``fstar`` and ``has_fstar`` reflect the validated
        reference.

    Examples
    --------
    >>> import pandas as pd
    >>> from benchmarks.validate import apply_validation
    >>> cat = pd.DataFrame({"name": ["P"], "y0_iD": [0], "fstar": [1.0],
    ...                     "has_fstar": [True]})
    >>> out = apply_validation(cat, [{"name": "P", "y0_iD": 0, "ineq_flip": "all",
    ...     "fstar": 2.0, "fstar_source": "scipy-consensus", "valid": True,
    ...     "validation": "fstar_mismatch", "validation_note": "", "fstar_sif": 1.0}])
    >>> row = out.loc[0]
    >>> (row["ineq_flip"], float(row["fstar"]), float(row["fstar_sif"]),
    ...  row["validation"])
    ('all', 2.0, 1.0, 'fstar_mismatch')
    """
    from .catalog import VALIDATION_COLUMNS, with_validation_columns

    out = with_validation_columns(catalog)
    index = {
        (str(n), int(y)): i
        for i, (n, y) in enumerate(zip(out["name"], out["y0_iD"], strict=True))
    }
    for res in results:
        key = (str(res.get("name", res.get("problem"))), int(res["y0_iD"]))
        if key not in index:
            continue
        i = out.index[index[key]]
        if "validation" not in res:
            status = str(res.get("status", "error"))
            out.loc[i, "validation"] = "timeout" if status == "timeout" else "error"
            out.loc[i, "validation_note"] = str(res.get("error", ""))[:300]
            continue
        for column in VALIDATION_COLUMNS:
            if column in res:
                out.loc[i, column] = res[column]
        if "fstar_sif" not in res:
            out.loc[i, "fstar_sif"] = out.loc[i, "fstar"]
        fstar = res.get("fstar", float("nan"))
        out.loc[i, "fstar"] = fstar
        out.loc[i, "has_fstar"] = bool(fstar == fstar and math.isfinite(float(fstar)))
    out["ineq_flip"] = out["ineq_flip"].fillna("").astype(str)
    out["valid"] = out["valid"].fillna(True).astype(bool)
    return out


def run_validation(
    catalog: pd.DataFrame,
    *,
    jobs: int = 1,
    max_steps: int = 500,
    timeout_s: float = 600.0,
    tc_max_n: int = 300,
    max_n: int | None = None,
    names: list[str] | None = None,
    log: bool = True,
) -> list[dict[str, Any]]:
    """Validate the catalogue instances on the benchmark worker pool.

    Parameters
    ----------
    catalog
        Catalogue to validate.
    jobs
        Concurrent worker processes.
    max_steps, timeout_s, tc_max_n
        Per-instance settings (see :class:`ValidateSpec`).
    max_n
        Skip instances with more variables than this.
    names
        Restrict to these problem names.
    log
        Print one progress line per instance to stderr.

    Returns
    -------
    list[dict[str, Any]]
        One result per validated instance (``runs`` stripped).
    """
    from .worker import run_tasks

    frame = catalog
    if max_n is not None:
        frame = frame[frame["n"] <= max_n]
    if names:
        wanted = {n.upper() for n in names}
        frame = frame[frame["name"].str.upper().isin(wanted)]
    specs = [
        ValidateSpec(
            problem=str(rec["name"]),
            y0_iD=int(rec["y0_iD"]),
            max_steps=max_steps,
            timeout_s=timeout_s,
            tc_max_n=tc_max_n,
        )
        for rec in frame.sort_values(["n", "name", "y0_iD"]).to_dict("records")
    ]
    results: list[dict[str, Any]] = []
    done = 0

    def on_result(row: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        row = dict(row)
        row.pop("runs", None)
        results.append(row)
        if log:
            name = row.get("name", row.get("problem"))
            print(
                f"[validate {done:4d}/{len(specs)}] {name:12s} "
                f"{row.get('validation', row.get('status')):>20s} "
                f"flip={row.get('ineq_flip', '')!r:8s} f*={row.get('fstar', float('nan')):.6g} "
                f"{row.get('validation_note', row.get('error', ''))[:80]}",
                file=sys.stderr,
                flush=True,
            )

    run_tasks(specs, jobs=jobs, on_result=on_result)
    return results


def validate_catalog_file(
    path: Path,
    *,
    out: Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Validate the instances of a catalogue CSV in place.

    Parameters
    ----------
    path
        Catalogue CSV to read.
    out
        Destination CSV (defaults to ``path``).
    **kwargs
        Forwarded to :func:`run_validation`.

    Returns
    -------
    pandas.DataFrame
        The updated catalogue.
    """
    catalog = pd.read_csv(path)
    results = run_validation(catalog, **kwargs)
    updated = apply_validation(catalog, results)
    updated.to_csv(out or path, index=False)
    return updated
