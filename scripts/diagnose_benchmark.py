"""Diagnose the unsolved tasks of a benchmark run.

Two stages:

```bash
# 1. Re-run every unsolved (problem, config) pair with INFO logging on,
#    collect the per-step trajectory, independent KKT checks and a scipy
#    SLSQP reference solve. Rows go to <out>/diag.jsonl, trajectories to
#    <out>/traj/<problem>__<config>.json. Resumable.
uv run python scripts/diagnose_benchmark.py run results.jsonl --out diagnostics --jobs 6

# 2. Classify every failure by its most likely cause and render the report.
#    ``--previous`` adds a fixed/regressed section against an earlier
#    classified.jsonl.
uv run python scripts/diagnose_benchmark.py report diagnostics --results results.jsonl \\
    --previous diagnostics_old/classified.jsonl
```

A failure is any result row whose ``solved`` flag is false: the minimiser
stopped unsuccessfully, or claimed success at a point that is infeasible or
whose objective disagrees with the CUTEst reference. Only sqpdax rows are
diagnosed; the scipy rows of the same run (``scipy-slsqp``,
``scipy-trust-constr``) are the external baseline: they feed the classifier
(``ref_*`` / ``tc_*`` fields, replacing the former in-run SLSQP reference
solve) and the head-to-head sections of the report. The classification (see
:data:`CAUSES`) distinguishes algorithmic shortcomings, bugs, poor default
parameters and benchmark artefacts (bad reference values).
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIAG_FILE = "diag.jsonl"
TRAJ_DIR = "traj"
REPORT_FILE = "report.html"

# ----------------------------------------------------------------------------
# stage 1: diagnostic re-runs
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagSpec:
    """One diagnostic re-run; executed by :class:`benchmarks.worker.WorkerPool`.

    Attributes
    ----------
    problem, y0_iD, config, max_steps
        Identify the benchmark task to reproduce.
    timeout_s
        Wall-clock budget enforced by the pool.
    ref_max_n
        Largest ``n`` for which the scipy SLSQP reference solve is run.
    ref_time_s
        Wall-clock cap of the reference solve.
    out_dir
        Directory receiving the trajectory file.
    """

    problem: str
    y0_iD: int
    config: str
    max_steps: int
    timeout_s: float = 600.0
    ref_max_n: int = 300
    ref_time_s: float = 60.0
    out_dir: str = "diagnostics"

    @property
    def key(self) -> str:
        return f"{self.problem}__{self.y0_iD}__{self.config}"

    def run(self, report=None) -> dict[str, Any]:  # noqa: C901 - one linear pipeline
        import jax
        import jax.numpy as jnp
        import numpy as np
        import sif2jax

        from benchmarks.configs import CONFIGS
        from benchmarks.metrics import quality_metrics, result_name
        from benchmarks.problems import to_sqpdax
        from slsqp_jax.sqpdax.logging import MemoryHandler
        from slsqp_jax.sqpdax.minimiser import minimise
        from slsqp_jax.sqpdax.results import is_successful

        jax.config.update("jax_enable_x64", True)
        start = time.perf_counter()

        def phase(name: str) -> None:
            if report is not None:
                report("phase", name)

        row: dict[str, Any] = {**asdict(self), "phase": "build", "rerun": True}
        phase("build")
        cfg = CONFIGS[self.config]
        row.update(cfg.tags())
        raw = sif2jax.cutest.get_problem(self.problem)
        problem, x0, meta = to_sqpdax(raw, y0_iD=self.y0_iD)
        if self.y0_iD != raw.y0_iD:
            raw = type(raw)(y0_iD=self.y0_iD)
        row.update({k: v for k, v in meta.as_dict().items() if k != "name"})
        fstar = meta.fstar if meta.has_fstar else None
        xstar = None
        try:
            xstar = raw.expected_result
        except NotImplementedError:
            pass

        handler = MemoryHandler()
        options = dict(cfg.make_options())
        options["logging"] = {"level": "INFO", "handler": handler}
        solver = cfg.make_minimiser()

        def solve(x, max_steps):
            return minimise(
                problem, solver, x, max_steps=max_steps, throw=False, options=options
            )

        phase("compile")
        row["phase"] = "compile"
        t0 = time.perf_counter()
        compiled = jax.jit(solve).lower(x0, jnp.asarray(1, jnp.int32)).compile()
        row["compile_s"] = time.perf_counter() - t0

        phase("pilot")
        row["phase"] = "pilot"
        t0 = time.perf_counter()
        sol = jax.block_until_ready(
            compiled(x0, jnp.asarray(self.max_steps, jnp.int32))
        )
        jax.effects_barrier()
        row["solve_s"] = time.perf_counter() - t0
        row["status"] = result_name(sol.result)
        row["successful"] = bool(is_successful(sol.result))
        row["steps"] = int(sol.stats["num_steps"])
        row.update(_scalar_stats(sol.stats))
        dual = getattr(sol.state, "dual", None)
        row.update(quality_metrics(problem, sol.value, dual, fstar=fstar, xstar=xstar))
        x_final = np.asarray(sol.value, dtype=float)
        row.update(_slack_residuals(problem, sol))

        # --- trajectory from the INFO lines --------------------------------
        phase("analyse")
        row["phase"] = "analyse"
        steps = [r.message for r in handler.records if r.message.startswith("step=")]
        traj = _parse_steps(steps)
        row.update(_trajectory_summary(traj, self.config))
        warnings_ = [r.message for r in handler.records if r.levelno >= 30]
        row["warnings"] = dict(collections.Counter(_warning_key(m) for m in warnings_))
        seen: dict[str, str] = {}
        for m in warnings_:
            seen.setdefault(_warning_key(m), m[:300])
        row["warning_examples"] = seen

        # --- independent checks at the returned point ----------------------
        row.update(_ls_stationarity(problem, x_final))
        if xstar is not None:
            xs = np.ravel(np.asarray(xstar, dtype=float))
            if xs.shape == x_final.shape:
                ref = quality_metrics(problem, jnp.asarray(xs), None, fstar=fstar)
                row["xstar_f"] = ref["objective"]
                row["xstar_feas"] = ref["feas"]
                row["xstar_f_gap"] = ref["f_gap"]

        # --- scipy SLSQP reference from the same starting point ------------
        if meta.n <= self.ref_max_n:
            phase("reference")
            row["phase"] = "reference"
            row.update(_scipy_reference(problem, x0, fstar, self.ref_time_s))

        traj_dir = Path(self.out_dir) / TRAJ_DIR
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"{self.key}.json"
        traj_path.write_text(json.dumps(traj))
        row["traj_path"] = str(traj_path)
        row["error"] = None
        row["phase"] = "done"
        row["task_wall_s"] = time.perf_counter() - start
        return _jsonable(row)


def _slack_residuals(problem: Any, sol: Any) -> dict[str, float]:
    """Residuals of the slack rows of an interior-point iterate.

    Returns ``slack_resid`` (``max`` over ``|h(x) + s|``, ``|x - lb - s_lb|``
    and ``|ub - x - s_ub|`` on the live rows) and ``slack_min``; empty for
    minimisers without slacks.
    """
    import numpy as np

    iterate = getattr(sol.state, "iterate", None)
    slack = getattr(iterate, "slack", None)
    if slack is None:
        return {}
    x = np.asarray(sol.value, dtype=float)
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    s = np.asarray(slack.s, dtype=float)
    s_lb = np.asarray(slack.s_lb, dtype=float)
    s_ub = np.asarray(slack.s_ub, dtype=float)
    live_lb = ~np.asarray(problem.null_lb, dtype=bool)
    live_ub = ~np.asarray(problem.null_ub, dtype=bool)
    parts = []
    mins = []
    if s.size:
        h = np.asarray(problem.ineq_fn(sol.value), dtype=float)
        parts.append(np.abs(h + s))
        mins.append(s)
    if live_lb.any():
        parts.append(np.abs((x - lb - s_lb)[live_lb]))
        mins.append(s_lb[live_lb])
    if live_ub.any():
        parts.append(np.abs((ub - x - s_ub)[live_ub]))
        mins.append(s_ub[live_ub])
    if not parts:
        return {"slack_resid": 0.0, "slack_min": math.inf}
    cat = np.concatenate(parts)
    resid = float(np.max(cat)) if np.isfinite(cat).all() else math.nan
    return {"slack_resid": resid, "slack_min": float(np.min(np.concatenate(mins)))}


_NUM = re.compile(r"^[-+]?(\d+\.?\d*([eE][-+]?\d+)?|inf|nan)$")


def _parse_steps(lines: list[str]) -> dict[str, list[Any]]:
    """Columns of the ``step=...`` INFO lines (numbers parsed, flags as bools)."""
    cols: dict[str, list[Any]] = collections.defaultdict(list)
    for i, line in enumerate(lines):
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        for key, value in fields.items():
            if value in ("True", "False"):
                parsed: Any = value == "True"
            elif _NUM.match(value):
                parsed = float(value)
            else:
                parsed = value
            col = cols[key]
            while len(col) < i:
                col.append(None)
            col.append(parsed)
    n = len(lines)
    for col in cols.values():
        while len(col) < n:
            col.append(None)
    return dict(cols)


def _floats(col: list[Any] | None) -> "Any":
    import numpy as np

    if not col:
        return np.zeros(0)
    return np.asarray(
        [
            v if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan
            for v in col
        ],
        dtype=float,
    )


def _trajectory_summary(traj: dict[str, list[Any]], config: str) -> dict[str, Any]:
    """Scalar features of the trajectory used by the classifier."""
    import numpy as np

    out: dict[str, Any] = {}
    n = len(traj.get("step", []))
    out["traj_steps"] = n
    if n == 0:
        return out
    accepted = traj.get("accepted")
    if accepted:
        acc = np.asarray([bool(a) for a in accepted])
        out["accept_rate"] = float(acc.mean())
        out["rejected_tail"] = int(_tail_run(~acc))
    alpha = _floats(traj.get("alpha"))
    dnorm = _floats(traj.get("dnorm"))
    kkt = _floats(traj.get("kkt"))
    f = _floats(traj.get("f"))
    merit = _floats(traj.get("merit"))
    feas = _floats(traj.get("feas") if "feas" in traj else traj.get("v"))
    if alpha.size:
        out["alpha_last"] = _last(alpha)
        out["alpha_small_frac"] = float(np.nanmean(alpha < 1e-3))
    if dnorm.size:
        out["dnorm_last"] = _last(dnorm)
        out["dnorm_min"] = float(np.nanmin(dnorm)) if np.isfinite(dnorm).any() else None
    if kkt.size:
        out["kkt_first"] = _first(kkt)
        out["kkt_last"] = _last(kkt)
        finite = kkt[np.isfinite(kkt)]
        if finite.size:
            out["kkt_min"] = float(finite.min())
            out["kkt_min_step"] = int(
                np.nanargmin(np.where(np.isfinite(kkt), kkt, np.inf))
            )
        window = min(50, n // 2) if n >= 4 else 0
        if window:
            a, b = kkt[-window - 1], kkt[-1]
            out["kkt_tail_ratio"] = float(b / a) if np.isfinite(a) and a > 0 else None
            tail = kkt[-window:]
            tail = tail[np.isfinite(tail)]
            if tail.size > 2:
                # log-oscillation: std of log10 within the tail
                out["kkt_tail_log_std"] = float(
                    np.std(np.log10(np.maximum(tail, 1e-300)))
                )
    if feas.size:
        out["feas_last"] = _last(feas)
        out["feas_min"] = float(np.nanmin(feas)) if np.isfinite(feas).any() else None
    if merit.size and merit.size > 1:
        d = np.diff(merit)
        d = d[np.isfinite(d)]
        out["merit_increase_frac"] = float(np.mean(d > 0)) if d.size else None
    for key in ("f", "merit", "alpha", "dnorm", "kkt"):
        arr = _floats(traj.get(key))
        if arr.size and (~np.isfinite(arr)).any():
            out["first_nonfinite_step"] = min(
                out.get("first_nonfinite_step", n), int(np.argmax(~np.isfinite(arr)))
            )
            out.setdefault("first_nonfinite_field", key)
    if f.size:
        out["f_first"] = _first(f)
        out["f_last"] = _last(f)
    # family specific tails
    for key in (
        "penalty",
        "stall",
        "qp_fail",
        "ls_fail",
        "qp_iters",
        "cg_iters",
        "mu",
        "radius",
        "radius_v",
        "radius_f",
        "rho",
        "model_fail",
        "nu",
        "pi_f",
        "v_max",
        "y_streak",
        "pairs",
    ):
        if key in traj:
            arr = _floats(traj[key])
            if arr.size:
                out[f"{key}_last"] = _last(arr)
    for key in ("type", "qp_result", "sub_status"):
        if key in traj:
            out[f"{key}_counts"] = dict(collections.Counter(str(v) for v in traj[key]))
    if "ls_fallback" in traj:
        out["ls_fallback_frac"] = float(np.mean([bool(v) for v in traj["ls_fallback"]]))
    if "sub_ok" in traj:
        out["sub_ok_frac"] = float(np.mean([bool(v) for v in traj["sub_ok"]]))
    if "mu_updated" in traj:
        out["mu_updates"] = int(sum(bool(v) for v in traj["mu_updated"]))
    return out


def _tail_run(mask) -> int:
    n = 0
    for v in mask[::-1]:
        if not v:
            break
        n += 1
    return n


def _first(arr) -> float | None:
    import numpy as np

    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else None


def _last(arr) -> float | None:
    v = float(arr[-1])
    return v if math.isfinite(v) else None


def _warning_key(message: str) -> str:
    key = re.split(r"[:(=]", message, maxsplit=1)[0].strip()
    key = re.sub(r"\d+(\.\d+)?([eE][-+]?\d+)?", "#", key)
    return key[:60]


def _scalar_stats(stats: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in stats.items():
        if key == "num_steps":
            continue
        try:
            arr = np.asarray(value)
        except Exception:  # noqa: BLE001
            continue
        if arr.ndim == 0 and arr.dtype.kind in "biuf":
            out[f"stats_{key}"] = arr.item()
    return out


def _ls_stationarity(problem, x, tol: float = 1e-6) -> dict[str, Any]:
    """Stationarity of ``x`` w.r.t. the active constraints with least-squares
    multipliers (independent of the multipliers the solver returned)."""
    import jax.numpy as jnp
    import numpy as np

    xj = jnp.asarray(x)
    g = np.asarray(problem.grad(xj), dtype=float)
    rows = []
    if problem.meq:
        rows.append(np.asarray(problem.eq_fn_jac(xj), dtype=float))
    if problem.mineq:
        h = np.asarray(problem.ineq_fn(xj), dtype=float)
        jh = np.asarray(problem.ineq_fn_jac(xj), dtype=float)
        active = h > -tol
        if active.any():
            rows.append(jh[active])
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    null_lb = np.asarray(problem.null_lb, dtype=bool)
    null_ub = np.asarray(problem.null_ub, dtype=bool)
    at_lb = (~null_lb) & (x - lb < tol)
    at_ub = (~null_ub) & (ub - x < tol)
    bound_idx = np.flatnonzero(at_lb | at_ub)
    if bound_idx.size:
        e = np.zeros((bound_idx.size, x.size))
        e[np.arange(bound_idx.size), bound_idx] = 1.0
        rows.append(e)
    scale = max(1.0, float(np.max(np.abs(g))) if g.size else 1.0)
    n_active = 0
    if rows:
        jac = np.vstack(rows)
        n_active = int(jac.shape[0])
    if not np.all(np.isfinite(g)) or (rows and not np.all(np.isfinite(jac))):
        # Non-finite derivatives at the returned point (e.g. S365 evaluates
        # sqrt of a negative number): stationarity is undefined.
        return {
            "stat_ls": float("nan"),
            "n_active": n_active,
            "n_active_bounds": int(bound_idx.size),
        }
    if rows:
        y, *_ = np.linalg.lstsq(jac.T, -g, rcond=None)
        resid = g + jac.T @ y
    else:
        resid = g
    return {
        "stat_ls": float(np.max(np.abs(resid))) / scale if resid.size else 0.0,
        "n_active": n_active,
        "n_active_bounds": int(bound_idx.size),
    }


class _RefTimeout(Exception):
    pass


def _scipy_reference(problem, x0, fstar, time_s: float) -> dict[str, Any]:
    """scipy SLSQP solve of the same NLP from ``x0`` as an external reference."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize

    from benchmarks.metrics import quality_metrics

    fn = jax.jit(lambda x: problem.fn(x)[0])
    grad = jax.jit(problem.grad)
    eq = jax.jit(problem.eq_fn)
    eq_jac = jax.jit(problem.eq_fn_jac)
    ineq = jax.jit(problem.ineq_fn)
    ineq_jac = jax.jit(problem.ineq_fn_jac)
    f64 = lambda a: np.asarray(a, dtype=float)  # noqa: E731
    cons = []
    if problem.meq:
        cons.append(
            {
                "type": "eq",
                "fun": lambda x: f64(eq(jnp.asarray(x))),
                "jac": lambda x: f64(eq_jac(jnp.asarray(x))),
            }
        )
    if problem.mineq:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x: -f64(ineq(jnp.asarray(x))),
                "jac": lambda x: -f64(ineq_jac(jnp.asarray(x))),
            }
        )
    lb = f64(problem.lb)
    ub = f64(problem.ub)
    bounds = [
        (None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi)
        for lo, hi in zip(lb, ub, strict=True)
    ]
    deadline = time.perf_counter() + time_s

    def callback(_x):
        if time.perf_counter() > deadline:
            raise _RefTimeout

    out: dict[str, Any] = {"ref_solver": "scipy-slsqp"}
    t0 = time.perf_counter()
    try:
        res = minimize(
            lambda x: float(fn(jnp.asarray(x))),
            f64(x0),
            jac=lambda x: f64(grad(jnp.asarray(x))),
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-10},
            callback=callback,
        )
    except _RefTimeout:
        out.update({"ref_status": "timeout", "ref_success": False})
        out["ref_time_s"] = time.perf_counter() - t0
        return out
    except Exception as exc:  # noqa: BLE001 - reference failures are data
        out.update({"ref_status": f"error: {exc}"[:200], "ref_success": False})
        out["ref_time_s"] = time.perf_counter() - t0
        return out
    out["ref_time_s"] = time.perf_counter() - t0
    out["ref_status"] = str(res.message)[:120]
    out["ref_success"] = bool(res.success)
    out["ref_nit"] = int(res.nit)
    q = quality_metrics(problem, jnp.asarray(res.x), None, fstar=fstar)
    out["ref_f"] = q["objective"]
    out["ref_feas"] = q["feas"]
    out["ref_f_gap"] = q["f_gap"]
    out["ref_finite"] = q["finite"]
    return out


def _jsonable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


# ----------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("solved")]


def task_key(row: dict[str, Any]) -> str:
    return f"{row['problem']}__{row.get('y0_iD', 0)}__{row['config']}"


def instance_key(row: dict[str, Any]) -> tuple[str, int]:
    return (row["problem"], int(row.get("y0_iD", 0) or 0))


def is_baseline(row: dict[str, Any]) -> bool:
    return row.get("backend", "sqpdax") != "sqpdax"


# scipy configurations of the benchmark used as external references.
BASELINE_CONFIGS = ("scipy-slsqp", "scipy-trust-constr")
BASELINE_SHORT = {"scipy-slsqp": "slsqp", "scipy-trust-constr": "trust-constr"}


def _baseline_fields(base_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reference fields derived from the scipy rows of the same instance.

    ``ref_*`` mirrors what the in-run scipy SLSQP reference used to provide
    (now taken from the benchmark's own ``scipy-slsqp`` row, which shares
    ``x0``, tolerances and step budget with the sqpdax rows); ``tc_*`` is the
    ``scipy-trust-constr`` row; ``baseline_solved`` / ``baseline_feasible``
    list the baselines that solved / reached a feasible point.
    """
    out: dict[str, Any] = {}
    s = base_rows.get("scipy-slsqp")
    if s is not None:
        out.update(
            ref_solver="scipy-slsqp",
            ref_status=s.get("status"),
            ref_success=s.get("status") == "successful",
            ref_solved=bool(s.get("solved")),
            ref_f=s.get("objective"),
            ref_feas=s.get("feas"),
            ref_f_gap=s.get("f_gap"),
            ref_finite=s.get("finite"),
            ref_steps=s.get("steps"),
            ref_time_s=s.get("time_median_s"),
        )
    t = base_rows.get("scipy-trust-constr")
    if t is not None:
        out.update(
            tc_status=t.get("status"),
            tc_success=t.get("status") == "successful",
            tc_solved=bool(t.get("solved")),
            tc_f=t.get("objective"),
            tc_feas=t.get("feas"),
            tc_f_gap=t.get("f_gap"),
            tc_steps=t.get("steps"),
            tc_time_s=t.get("time_median_s"),
        )
    out["baseline_solved"] = [
        k for k in BASELINE_CONFIGS if (base_rows.get(k) or {}).get("solved")
    ]
    out["baseline_feasible"] = [
        k
        for k in BASELINE_CONFIGS
        if k in base_rows
        and base_rows[k].get("finite", True)
        and _g(base_rows[k], "feas") <= 1e-5
    ]
    return out


def _baseline_cell(row: dict[str, Any], prefix: str) -> str:
    """Compact ``status / gap`` summary of one baseline for the task tables."""
    status = row.get(f"{prefix}_status")
    if status is None:
        return ""
    if row.get(f"{prefix}_solved"):
        return "solved"
    gap = _g(row, f"{prefix}_f_gap")
    feas = _g(row, f"{prefix}_feas")
    bits = [str(status)]
    if math.isfinite(gap):
        bits.append(f"gap {gap:.1e}")
    if math.isfinite(feas) and feas > 1e-6:
        bits.append(f"feas {feas:.1e}")
    return " ".join(bits)


# ----------------------------------------------------------------------------
# stage 2: classification
# ----------------------------------------------------------------------------

CATEGORY_LABELS = {
    "bug": "Bug",
    "algorithmic": "Algorithmic shortcoming",
    "parameter": "Default parameter / tolerance",
    "benchmark": "Benchmark artefact (reference data)",
    "performance": "Performance",
}

# cause id -> (category, title, description)
CAUSES: dict[str, tuple[str, str, str]] = {
    "false_convergence": (
        "bug",
        "Success declared at a non-stationary point",
        "The minimiser returned `successful`, the point is feasible, but it is "
        "not a KKT point: the least-squares stationarity residual over the "
        "active constraints (`stat_ls`, independent of the returned "
        "multipliers) is far above the tolerance. In the previous run this "
        "was the largest group (103 tasks) and was traced to the projected CG "
        "freezing on a non-monotone residual and returning `d = 0` as a "
        "converged QP; that rule was replaced (see the fix attempts) and the "
        "group is essentially gone. What remains is a single exact-Hessian "
        "run (BT9, `||grad L|| = 2e-2` at `f_gap = 1e-4`) where the zero-step "
        "patience of the `qp_kkt` guard still accepts a non-converged QP, "
        "which is the remaining recommendation below.",
    ),
    "stationarity_scale_f": (
        "parameter",
        "Relative stationarity test scaled by `|f|`",
        "`successful` at a non-stationary point because the active-set test "
        "is `||grad_x L||_inf <= rtol * max(|f|, 1)` (`use_mu_max=False`): "
        "with `|f| = 1e3 ... 1e17` a Lagrangian gradient of `1e-3 ... 1e9` "
        "passes. After the CG fix nearly all of these (50 of 51) are tasks no "
        "scipy baseline solves either: the unbounded (sign-flipped) minimax "
        "instances on which `asls-pcg-exact` declares success with `f = -1e6` "
        "instead of running to `-inf` like the other configurations, plus the "
        "DIAG* quadratics. The scaling still deserves a cap (`max(|f|,1)` -> "
        "`min(max(|f|,1), 1e3)` or the SNOPT-style `||grad f||` scaling), but "
        "it is not where solve rate is lost against scipy.",
    ),
    "qp_inactive_multipliers": (
        "bug",
        "Success with multipliers on inactive constraints",
        "`successful` with `||grad_x L(x, lambda)||` tiny for the QP "
        "multipliers but a large complementarity residual `max |lambda_i "
        "h_i(x)|`: the working set still contains rows that are not active "
        "at `x`, the QP reports `working_set_converged`, and the termination "
        "test never checks complementarity (SIPOW1/2, HS20, HS33).",
    ),
    "false_infeasible_certificate": (
        "bug",
        "`infeasible_stationary` certified by a zero QP step",
        "The active-set method stopped with `infeasible_stationary` after one "
        "or two steps on a problem that is feasible (another configuration or "
        "a scipy baseline finds a feasible point from the same start; scipy "
        "SLSQP *solves* 61 of these 115 tasks). Together with "
        "`kkt_solver_failure` and `restoration_failure` this is the signature "
        "of one situation - an infeasible start whose violated constraints "
        "outnumber the variables (DUALC1/2/5/8, HS96/97/98, HS19, HS106, "
        "HS116, OET2-7, TENBARS3, OPTCNTRL): the projected-CG configurations "
        "end here, Craig/MINRES-QLP in `kkt_solver_failure`, pasls in "
        "`restoration_failure`. Those instances are exactly the ones solved "
        "by scipy and by no sqpdax configuration (appendix 3a). "
        "`_infeasible_stationary` tests `stationary & ~feasible` where "
        "`stationary` is `||grad_x L(x, lambda_QP)|| <= rtol*max(|f|,1)`; the "
        "QP multipliers satisfy `H d + grad f + A^T lambda = 0`, so whenever "
        "the QP returns `d ~ 0` (degenerate / over-determined working set, "
        "e.g. HS71: four active bounds plus one equality in n = 4; SIMPLLPB: "
        "three violated inequalities in n = 2) the test is trivially true and "
        "the run is aborted as infeasible without any evidence that the "
        "infeasibility measure is stationary.",
    ),
    "kkt_solver_failure": (
        "bug",
        "KKT solver fails on a rank-deficient / over-determined working set",
        "The QP sub-solver (Craig projector or MINRES-QLP) reports "
        "`kkt_solver_failure` (`singular`, `projector_failure`, `nonfinite`) "
        "on the very first iterations. The initial working set contains all "
        "violated constraints; when there are more of them than variables "
        "(HS23: four violated inequalities in n = 2) or they are linearly "
        "dependent, `A A^T` is singular and the Craig / MINRES-QLP systems "
        "produce NaN or stall at `max_steps`. The NaN direction is counted as "
        "a QP failure, the secant is reset three times and the run ends "
        "`secant_recovery_failure`; the zero direction leads to a false "
        "`infeasible_stationary` certificate. The projected-CG configurations "
        "survive the same working sets because their least-squares projector "
        "tolerates rank deficiency - and then fail one step later with the "
        "false certificate above. Half of these tasks (62/123) are solved by "
        "scipy SLSQP, whose LSQ subproblem relaxes inconsistent linearised "
        "constraints with an elastic variable instead of failing.",
    ),
    "local_minimum": (
        "algorithmic",
        "Converged to a different KKT point than the reference",
        "The returned point is feasible and stationary (small `stat_ls`), but "
        "its objective differs from the CUTEst reference while at least one "
        "other configuration, or scipy SLSQP from the same start, reaches "
        "the reference value. The globalisation steered the run into another "
        "local minimum (or a saddle).",
    ),
    "local_minimum_consensus": (
        "benchmark",
        "Local minimum shared with the scipy baselines",
        "The returned point is feasible and stationary, no sqpdax "
        "configuration reaches the reference objective from this start, and "
        "neither scipy SLSQP nor trust-constr does either (typically all "
        "three agree on the same value: HS5 `-1.913` vs `f* = -5.03`, HS20 "
        "`40.199` vs `38.199`, HS21MOD `-95.96` vs `-99.96`, HS35MOD `0.25` "
        "vs `0.111`, HS64 `5800` vs `6299.8`, BIGGSC4/HATFLDH `-50` vs "
        "`-24.5`, ZECEVIC2 `-21.125` vs `-4.125`, the PALMER*/LEVYMONT* "
        "fits ...). Either the instance has a local minimum on the path from "
        "`x0` or the reference value is wrong for the problem as converted; "
        "in both cases this is not an sqpdax deficiency and the tasks should "
        "be scored against the scipy consensus or excluded.",
    ),
    "reference_mismatch": (
        "benchmark",
        "Reference value inconsistent with the problem",
        "The returned feasible point has a *lower* objective than the "
        "reference, or the reference minimiser `x*` is infeasible / has "
        "`f(x*) != f*` for the problem as converted. The benchmark reference "
        "(or the sif2jax conversion), not the solver, is wrong for these "
        "instances.",
    ),
    "loose_tolerance": (
        "parameter",
        "Converged near the reference but outside `f_tol`",
        "Feasible, stationary and within `1e-2` relative objective gap of the "
        "reference: the run stopped slightly early. A tighter / better scaled "
        "termination test would close the gap.",
    ),
    "secant_recovery_premature": (
        "parameter",
        "Secant recovery declared fatal while still converging",
        "The run ended with `secant_recovery_failure` while the optimality "
        "residual was still decreasing (or already small). The reset policy "
        "escalated to the fatal stage on a streak of rejected steps that a "
        "trust-region / line-search method would normally ride out.",
    ),
    "secant_recovery_stuck": (
        "algorithmic",
        "Secant recovery exhausted far from optimality",
        "`secant_recovery_failure` with a large optimality residual: after the "
        "identity reset the model still produced rejected steps, i.e. the "
        "failure was not caused by a bad curvature estimate but by the step "
        "computation itself (radius collapse, infeasible subproblem, poor "
        "merit penalty).",
    ),
    "slow_progress": (
        "algorithmic",
        "Step budget exhausted while still making progress",
        "`max_steps_reached` with the optimality residual still decreasing "
        "over the last 50 steps. The iteration is sound but too slow: short "
        "steps, many rejected trust-region steps or a slow barrier schedule.",
    ),
    "step_collapse": (
        "algorithmic",
        "Step budget exhausted with collapsed steps",
        "`max_steps_reached` and the accepted steps had become negligible "
        "(tiny `alpha` or `dnorm`) without the optimality residual going "
        "down: the line search / trust region is rejecting the direction or "
        "the direction itself is useless (inconsistent QP, wrong active set).",
    ),
    "cycling": (
        "algorithmic",
        "Step budget exhausted without progress (cycling)",
        "`max_steps_reached`, steps are still being accepted but the "
        "optimality residual oscillates without trend: the active-set / "
        "penalty / barrier logic is cycling.",
    ),
    "restoration_failure": (
        "algorithmic",
        "Line search fails repeatedly at an infeasible point",
        "The active-set method ended `infeasible` (fatal line-search / merit "
        "stagnation streak while primally infeasible) although the problem "
        "is feasible (another configuration or a scipy baseline reached a "
        "feasible point from the same start; scipy SLSQP solves 37 of the 65). "
        "Two traced patterns: (i) the QP step "
        "does not reduce the linearised violation because the working set is "
        "degenerate (HS71: bounds fix every variable, the equality residual "
        "4.96 is never touched, the merit slope is positive and the line "
        "search rejects every step while the penalty stays at 11.9); (ii) "
        "the unsafeguarded normal step is huge on a nearly singular Jacobian "
        "(BYRDSPHR: `||d|| = 1e5`, the penalty is pushed to 2e9, ten line "
        "searches fail). There is no restoration phase, no relaxation of "
        "inconsistent linearisations and no step-norm safeguard.",
    ),
    "ip_bound_slack_residual": (
        "algorithmic",
        "Trust-region IP: infeasible end point with inconsistent slack rows",
        "A `TrustRegionInteriorPointMinimiser` run that ends with the slack "
        "rows `h(x) + s = 0`, `x - lb - s_lb = 0`, `ub - x - s_ub = 0` "
        "violated. In the previous run (173 tasks) this was a bug: the slacks "
        "were seeded as `max(., 1)` and the dogleg normal step never moved "
        "them, so the rows stayed inconsistent even on trivially feasible "
        "problems. Both parts are fixed (IPOPT-style seeding, scaled normal "
        "step over all rows; see the fix attempts) and the group shrank to 36 "
        "tasks; 34 of them end *primally infeasible* with "
        "`feas ~ slack_resid` (TENBARS1-4, TRO*, DUALC*, DEGENLP*, CSFI*, "
        "HS97/98/106/114/116, BATCH, NASH, CLEUVEN7, TABLE7 ...) and the "
        "other two (DECONVC, HS96) are feasible with a round-off-level "
        "residual of 1e-5: the slack residual is the infeasibility itself, "
        "not a bookkeeping error. "
        "These are inequality-heavy problems started far outside the feasible "
        "region, where the Byrd-Omojokun normal step (bounded by `0.8 Delta` "
        "and by the fraction-to-the-boundary rule on the slacks) makes too "
        "little feasibility progress per iteration, the merit penalty is "
        "raised, the ratio test rejects the composite step and the radius "
        "collapses into `secant_recovery_failure`. scipy SLSQP solves 11 of "
        "them, trust-constr 9; the residue is the same infeasible-start "
        "cluster that defeats the active-set configurations.",
    ),
    "ip_bound_violation": (
        "algorithmic",
        "Funnel IP returned a point outside the bounds",
        "The trust-funnel run ended at a point violating `[lb, ub]`. Bounds "
        "are relaxed through slacks (`x - lb - s_lb = 0` is part of the "
        "violation `v`), so a funnel iteration may trade bound violation for "
        "objective decrease; when the normal step cannot close the gap the "
        "run ends `max_steps_reached` / `secant_recovery_failure` outside the "
        "box. Interior-point methods normally keep bounds on x exactly via "
        "the fraction-to-the-boundary rule and only slack general "
        "inequalities.",
    ),
    "ip_barrier_stall": (
        "algorithmic",
        "Barrier parameter never reduced",
        "An interior-point run that ends with `mu` still at (or near) its "
        "initial value: the barrier-subproblem termination test is never met, "
        "so the outer loop never shrinks `mu`; the iterates converge to the "
        "minimiser of the current barrier problem and the trust-region / "
        "funnel machinery then fails on rounding-level model reductions "
        "(69 of 88 end `secant_recovery_failure`, 19 `max_steps_reached`). "
        "Now the largest interior-point group (tfip 52, trip 36) and, unlike "
        "the infeasible residue above, a solver deficit: 38 tasks are solved "
        "by scipy SLSQP and 46 by trust-constr; the end points are feasible "
        "and `kkt_last` is dominated by `compl ~ mu` (CANTILVR: `kkt = 1.20`, "
        "`compl = 1.198`, `mu = 1`, 500 steps; HS100: `kkt = compl = 0.040 = "
        "mu`). 45 tasks never reduce `mu` once (`mu_updates = 0`), i.e. the "
        "first barrier problem is never declared converged although its "
        "iterates have stopped moving. scipy's trust-constr decides the `mu` "
        "update on the *barrier* KKT error with tolerance "
        "`BARRIER_DECAY_RATIO * mu` and additionally forces it whenever the "
        "trust radius falls below `xtol`; sqpdax has no radius-collapse exit "
        "from the inner loop, so a stalled barrier problem is recorded as a "
        "stall of the secant model instead.",
    ),
    "termination_missed": (
        "parameter",
        "Converged but the termination test never fired",
        "The final iterate is feasible, (nearly) optimal (`f_gap` small, "
        "`kkt_last` tiny) but the run ended `secant_recovery_failure` or "
        "`max_steps_reached`: steps at the solution have predicted "
        "reductions at rounding level, the ratio / funnel tests reject them, "
        "rejections are counted as secant failures and the recovery is "
        "exhausted before the KKT test (with its complementarity / mu "
        "requirement) is satisfied.",
    ),
    "near_feasible": (
        "parameter",
        "Stopped just outside the feasibility tolerance",
        "Terminated as infeasible with a violation below `1e-4`: the "
        "feasibility test or its scaling is too strict for the attainable "
        "accuracy.",
    ),
    "problem_infeasible": (
        "benchmark",
        "No solver finds a feasible point",
        "Neither any sqpdax configuration nor scipy SLSQP / trust-constr "
        "reached a feasible point from the given start; the instance is "
        "(locally) infeasible or mis-converted.",
    ),
    "merit_stagnation": (
        "algorithmic",
        "Merit function stagnated",
        "`merit_stagnation`: the penalty merit stopped improving over the "
        "patience window while optimality was not reached. Usually a penalty "
        "parameter that grew too large (the merit is dominated by the "
        "constraint term) or a line search accepting only tiny steps.",
    ),
    "merit_stagnation_near_opt": (
        "parameter",
        "Merit stagnation at an almost-optimal point",
        "`merit_stagnation` with a small optimality residual: the patience "
        "window fires before the termination test gets its chance; a looser "
        "stagnation rule or a final KKT check would convert these to "
        "successes.",
    ),
    "multiplier_failure": (
        "algorithmic",
        "Funnel y-iteration deadlock (normal gate vs. tangential gate)",
        "`stationarity_stall` / `multiplier_solve_failure` after a streak of "
        "y-iterations with `v` far from zero (BT10: v = 0.39 after two "
        "v-iterations, then five y-iterations with `dnorm = 0`). The normal "
        "gate (3.2) `pi_v > omega_n(pi_f_prev) or v >= kappa_vv * v_max` is "
        "false (v_max is still 250 from the seed 1e3), so no normal step is "
        "computed; the tangential gate then finds that infeasibility "
        "dominates and skips the tangential step too. Nothing changes and "
        "the `stall_steps = 5` fallback aborts. Setting `always_normal=True` "
        "fixes HS22, GIGOMEZ3 and HS32 but not the equality-only BT9/BT10/"
        "HS14, where the normal solver itself returns a zero step.",
    ),
    "ip_infeasible_stationary": (
        "algorithmic",
        "Interior point stopped at an infeasible stationary point",
        "The funnel method's Step 8 fired on a problem other methods solve: "
        "the normal step got trapped in a stationary point of the violation.",
    ),
    "unbounded": (
        "benchmark",
        "Problem unbounded as converted (constraint sign)",
        "Every sqpdax configuration, scipy SLSQP (`f = -7e30`) and "
        "trust-constr (`f = -1e50`) drive the objective to -inf while "
        "reporting zero violation. These are the "
        "minimax-type SIF problems (POLAK4, CHACONN1, GOFFIN, MINMAXBD, ...): "
        "sif2jax returns the raw pycutest value `c(x) - b` for L-type groups "
        "(`c(x) <= b`, see the comment in `sif2jax.cutest.POLAK4.constraint`), "
        "i.e. a `<= 0` quantity, whereas its documented convention (and the "
        "adapter `benchmarks.problems.to_sqpdax`, `h = -c <= 0`) assumes "
        "`c(x) >= 0`. The converted constraint `u >= f_i(x)` becomes "
        "`u <= f_i(x)` and `u` is free to go to -inf. Not a solver defect.",
    ),
    "nonfinite": (
        "bug",
        "Non-finite values during the iteration",
        "The run produced NaN / inf (objective, merit or step) outside the "
        "KKT-solver pattern above. Missing safeguards against overflow in the "
        "model, penalty or barrier.",
    ),
    "evaluation_error": (
        "benchmark",
        "Problem functions are not finite at the start / returned point",
        "The objective or its derivatives evaluate to NaN (S365 / S365MOD "
        "take square roots of negative arguments at `x0`). No solver can "
        "progress from here; the instances should be excluded or started "
        "elsewhere.",
    ),
    "compile_timeout": (
        "performance",
        "Compilation exceeded the task budget",
        "The task was killed while XLA was still compiling; the lowered "
        "program is too large for this problem/configuration.",
    ),
    "run_timeout": (
        "performance",
        "Solve exceeded the task budget",
        "Compiled but did not finish within the wall-clock budget.",
    ),
    "unclassified": (
        "algorithmic",
        "Unclassified",
        "None of the rules above matched; see the evidence columns.",
    ),
}


def _g(d: dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = d.get(key)
    if v is None:
        return default
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf}.get(v, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify(
    row: dict[str, Any],
    diag: dict[str, Any] | None,
    peers: list[dict[str, Any]],
    feas_tol: float,
    f_tol: float,
) -> tuple[str, str]:
    """Return ``(cause_id, evidence)`` for one unsolved task."""
    d = diag or {}
    status = d.get("status") or row["status"]
    peers_solved = [p for p in peers if p.get("solved")]
    peers_feasible = [
        p for p in peers if p.get("finite", True) and _g(p, "feas") <= feas_tol
    ]
    ref_ok = bool(d.get("ref_success")) and _g(d, "ref_feas") <= 1e-5
    tc_ok = bool(d.get("tc_success")) and _g(d, "tc_feas") <= 1e-5
    baseline_solved = list(d.get("baseline_solved") or [])
    ref_hits = bool(baseline_solved) or (
        ref_ok and (not row.get("has_fstar") or _g(d, "ref_f_gap") <= f_tol)
    )
    ref_feasible = bool(d.get("baseline_feasible")) or _g(d, "ref_feas") <= 1e-5
    baseline_known = any(k in d for k in ("ref_status", "tc_status"))
    has_fstar = bool(row.get("has_fstar"))
    f_gap = _g(d, "f_gap", _g(row, "f_gap"))
    feas = _g(d, "feas", _g(row, "feas"))
    stat_ls = _g(d, "stat_ls")
    kkt_last = _g(d, "kkt_last")
    kkt_ratio = _g(d, "kkt_tail_ratio")
    ev = []

    family = row.get("family")
    asls_family = family in ("asls", "pasls")
    qp_counts = d.get("qp_result_counts") or {}
    steps = int(_g(d, "steps", _g(row, "steps", 0)) or 0)
    qp_fail_frac = qp_counts.get("kkt_solver_failure", 0) / max(1, steps)
    mu_last = _g(d, "mu_last")

    if status == "timeout":
        phase = row.get("phase")
        ev.append(f"phase={phase}")
        return (
            "compile_timeout" if phase in ("build", "compile") else "run_timeout"
        ), "; ".join(ev)
    if status == "runtime_error" or (status == "nonfinite" and steps == 0):
        ev.append("non-finite objective/derivatives at the evaluated point")
        return "evaluation_error", "; ".join(ev)
    # Problem unbounded as converted: scipy SLSQP diverges to -1e30 as well.
    f_first, f_last = _g(d, "f_first"), _g(d, "f_last")
    ref_f = _g(d, "ref_f")
    tc_f = _g(d, "tc_f")
    diverging = (
        math.isfinite(f_first)
        and math.isfinite(f_last)
        and f_last < f_first - 1e2 * max(1.0, abs(f_first))
    )
    baseline_diverges = (math.isfinite(ref_f) and ref_f < -1e10) or (
        math.isfinite(tc_f) and tc_f < -1e10
    )
    # A run that ends at or above a known ``f*`` is not evidence of an
    # unbounded conversion, whatever the (possibly stale, pre-sign-fix)
    # scipy baseline did.
    fstar = _g(d, "fstar", _g(row, "fstar"))
    above_fstar = (
        has_fstar
        and math.isfinite(fstar)
        and math.isfinite(f_last)
        and f_last >= fstar - 1e-2 * max(1.0, abs(fstar))
    )
    if (
        status != "successful"
        and not above_fstar
        and (baseline_diverges or (diverging and not has_fstar and not baseline_solved))
    ):
        ev.append(
            f"f {f_first:.3g} -> {f_last:.3g}; scipy slsqp f={ref_f:.3g}, trust-constr f={tc_f:.3g}"
        )
        return "unbounded", "; ".join(ev)
    if status == "nonfinite" or d.get("first_nonfinite_step") is not None:
        ev.append(
            f"first non-finite at step {d.get('first_nonfinite_step')} in {d.get('first_nonfinite_field')}"
        )
        if asls_family and qp_counts.get("kkt_solver_failure", 0) > 0:
            ev.append(f"qp results {qp_counts}")
            return "kkt_solver_failure", "; ".join(ev)
        return "nonfinite", "; ".join(ev)
    if asls_family and qp_fail_frac >= 0.5 and status != "successful":
        ev.append(
            f"qp kkt_solver_failure in {qp_counts.get('kkt_solver_failure')}/{steps} steps (status {status})"
        )
        return "kkt_solver_failure", "; ".join(ev)

    bound_viol = max(_g(d, "feas_lb", 0.0), _g(d, "feas_ub", 0.0))
    has_bounds = int(_g(row, "nbounds", 0) or 0) > 0
    barrier_stuck = math.isfinite(mu_last) and mu_last > 1e-3
    # Rows produced after the slack-residual instrumentation carry the measured
    # ``slack_resid``; older rows only allow the indirect ``mu`` inference.
    slack_resid = _g(d, "slack_resid")
    if math.isfinite(slack_resid):
        slack_inconsistent = slack_resid > 1e-6 or bound_viol > 1e-6
    else:
        slack_inconsistent = bound_viol > 1e-6 or barrier_stuck
    if (
        family == "trip"
        and status != "successful"
        and has_bounds
        and slack_inconsistent
    ):
        ev.append(
            f"slack-row residual {slack_resid:.2e}; bound violation {bound_viol:.2e}; "
            f"mu_last={mu_last:.2g} compl={_g(d, 'compl'):.2g} "
            f"kkt_last={kkt_last:.2g} kkt_min={_g(d, 'kkt_min'):.2g} (status {status})"
        )
        return "ip_bound_slack_residual", "; ".join(ev)
    if family == "tfip" and bound_viol > 1e-6:
        ev.append(f"bound violation {bound_viol:.2e} (status {status})")
        return "ip_bound_violation", "; ".join(ev)

    converged_anyway = (
        status in ("secant_recovery_failure", "max_steps_reached")
        and math.isfinite(kkt_last)
        and kkt_last < 1e-3
        and feas <= 1e-4
        and (not has_fstar or f_gap <= 1e-2)
    )
    if converged_anyway:
        ev.append(
            f"kkt_last={kkt_last:.1e} feas={feas:.1e} f_gap={f_gap:.1e} (status {status})"
        )
        return "termination_missed", "; ".join(ev)

    if status == "successful":
        obj = _g(d, "objective", _g(row, "objective"))
        fstar = _g(row, "fstar")
        ev.append(
            f"f={obj:.6g} f*={fstar:.6g} gap={f_gap:.2e} feas={feas:.1e} stat_ls={stat_ls:.1e}"
        )
        stationary = math.isfinite(stat_ls) and stat_ls <= 1e-4
        if family in ("trip", "tfip") and math.isfinite(kkt_last) and kkt_last <= 1e-5:
            # Interior-point iterates satisfy the constraints only to O(mu);
            # the least-squares test with a 1e-6 activity tolerance can miss
            # the active rows.  Trust the solver's own KKT error here.
            stationary = True
        compl = _g(d, "compl")
        xstar_bad = ("xstar_feas" in d) and (
            _g(d, "xstar_feas") > 1e-5 or _g(d, "xstar_f_gap") > f_tol
        )
        below = (
            has_fstar
            and obj < fstar - f_tol * max(1.0, abs(fstar))
            and feas <= feas_tol
        )
        if xstar_bad:
            ev.append(
                f"x* check: feas(x*)={_g(d, 'xstar_feas'):.1e} gap(f(x*),f*)={_g(d, 'xstar_f_gap'):.1e}"
            )
        if below:
            ev.append("objective below reference at a feasible point")
        if not stationary:
            ev.append("not stationary (least-squares multipliers)")
            if math.isfinite(compl) and compl > 1e-4:
                ev.append(f"complementarity {compl:.2e}: multipliers on inactive rows")
                return "qp_inactive_multipliers", "; ".join(ev)
            if abs(obj) > 10.0 and math.isfinite(kkt_last) and kkt_last > 1e-6:
                ev.append(
                    f"|f|={abs(obj):.3g} scales the relative test; ||grad L||={kkt_last:.2e}"
                )
                return "stationarity_scale_f", "; ".join(ev)
            return "false_convergence", "; ".join(ev)
        if below or xstar_bad:
            return "reference_mismatch", "; ".join(ev)
        if f_gap <= 1e-2:
            return "loose_tolerance", "; ".join(ev)
        if peers_solved or ref_hits:
            who = [p["config"] for p in peers_solved] + baseline_solved
            ev.append(f"reference reached by {who or ['scipy-slsqp']}")
            return "local_minimum", "; ".join(ev)
        if baseline_known:
            ev.append(
                f"scipy slsqp f={ref_f:.6g} gap={_g(d, 'ref_f_gap'):.2e}; "
                f"trust-constr f={tc_f:.6g} gap={_g(d, 'tc_f_gap'):.2e}"
            )
        scale = max(1.0, abs(obj))
        if (ref_ok and ref_f < obj - 1e-2 * scale) or (
            tc_ok and tc_f < obj - 1e-2 * scale
        ):
            ev.append("a scipy baseline reaches a lower objective")
            return "local_minimum", "; ".join(ev)
        return "local_minimum_consensus", "; ".join(ev)

    if status in ("multiplier_solve_failure", "stationarity_stall"):
        ev.append(
            f"kkt_last={kkt_last:.1e} feas={feas:.1e} y_streak={d.get('y_streak_last')} "
            f"types={d.get('type_counts')}"
        )
        return "multiplier_failure", "; ".join(ev)

    if (
        family in ("trip", "tfip")
        and barrier_stuck
        and status
        in (
            "secant_recovery_failure",
            "max_steps_reached",
        )
        # An infeasible end point is a feasibility failure first; the stuck
        # barrier is then a consequence, not the cause.
        and _g(d, "feas", 0.0) <= 1e-4
    ):
        ev.append(
            f"mu_last={mu_last:.2g} mu_updates={d.get('mu_updates')} compl={_g(d, 'compl'):.2g} "
            f"kkt_last={kkt_last:.2g} kkt_min={_g(d, 'kkt_min'):.2g} (status {status})"
        )
        return "ip_barrier_stall", "; ".join(ev)

    if status == "secant_recovery_failure":
        ev.append(
            f"kkt_last={kkt_last:.1e} tail_ratio={kkt_ratio:.2f} stage={d.get('stats_secant_recovery_stage')} "
            f"accept_rate={_g(d, 'accept_rate'):.2f} rejected_tail={d.get('rejected_tail')}"
        )
        converging = (math.isfinite(kkt_ratio) and kkt_ratio < 0.5) or (
            math.isfinite(kkt_last) and kkt_last < 1e-3
        )
        return (
            "secant_recovery_premature" if converging else "secant_recovery_stuck"
        ), "; ".join(ev)

    if status == "max_steps_reached":
        ev.append(
            f"kkt_last={kkt_last:.1e} kkt_min={_g(d, 'kkt_min'):.1e} tail_ratio={kkt_ratio:.2f} "
            f"alpha_last={_g(d, 'alpha_last'):.1e} dnorm_last={_g(d, 'dnorm_last'):.1e} "
            f"accept_rate={_g(d, 'accept_rate'):.2f} feas_last={_g(d, 'feas_last'):.1e}"
        )
        if math.isfinite(kkt_ratio) and kkt_ratio < 0.5:
            return "slow_progress", "; ".join(ev)
        collapsed = (
            (_g(d, "alpha_last", 1.0) < 1e-3)
            or (_g(d, "dnorm_last", 1.0) < 1e-8)
            or (_g(d, "accept_rate", 1.0) < 0.2)
        )
        if collapsed:
            return "step_collapse", "; ".join(ev)
        return "cycling", "; ".join(ev)

    if status in ("infeasible", "infeasible_stationary", "infeasible_stationary_point"):
        ev.append(
            f"feas={feas:.1e} steps={steps} feasible peers={[p['config'] for p in peers_feasible]} "
            f"feasible baselines={d.get('baseline_feasible') if baseline_known else 'n/a'}"
        )
        if feas <= 1e-4:
            return "near_feasible", "; ".join(ev)
        if peers_feasible or ref_feasible:
            if status == "infeasible_stationary_point":
                return "ip_infeasible_stationary", "; ".join(ev)
            if status == "infeasible_stationary":
                ev.append(
                    f"stat_ls={stat_ls:.1e} n_active={d.get('n_active')} n={row.get('n')}"
                )
                return "false_infeasible_certificate", "; ".join(ev)
            return "restoration_failure", "; ".join(ev)
        if status == "infeasible_stationary" and steps <= 3:
            # Nobody finds a feasible point, but a 1-3 step abort is not
            # evidence of infeasibility either (see the certificate bug).
            ev.append("aborted within 3 steps; infeasibility unproven")
            return "false_infeasible_certificate", "; ".join(ev)
        return "problem_infeasible", "; ".join(ev)

    if status == "merit_stagnation":
        ev.append(
            f"kkt_last={kkt_last:.1e} penalty={_g(d, 'penalty_last'):.1e} alpha_last={_g(d, 'alpha_last'):.1e} "
            f"stall={d.get('stall_last')} feas={feas:.1e}"
        )
        if math.isfinite(kkt_last) and kkt_last < 1e-3 and feas <= 1e-4:
            return "merit_stagnation_near_opt", "; ".join(ev)
        return "merit_stagnation", "; ".join(ev)

    ev.append(f"status={status}")
    return "unclassified", "; ".join(ev)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

RECOMMENDATIONS: dict[str, str] = {
    "false_convergence": (
        "<ol>"
        "<li><b>Fix the projected-CG freeze rule</b> (<code>subproblem/solver/projected_cg.py</code>): "
        "the preconditioned residual norm <code>r^T z</code> is not monotone in CG, so "
        "<code>freeze = bad | (rz_new >= rz)</code> stops the solve after one or two iterations on "
        "perfectly well-posed systems. Freeze only on a genuine breakdown (non-finite values, "
        "<code>p^T H p <= 0</code> with the negative-curvature exit, or no model decrease over "
        "several consecutive iterations), not on a single non-decreasing residual.</li>"
        "<li><b>Do not label a frozen solve as a success.</b> <code>residual_floor</code> must map to "
        "<code>success=False</code> (or carry an explicit <code>converged=False</code> flag) so the "
        "active-set loop does not report <code>working_set_converged</code> for a zero step.</li>"
        "<li><b>Make the <code>qp_kkt</code> guard check optimality.</b> In "
        "<code>ActiveSetLineSearchMinimiser.termination_flags</code> require "
        "<code>kkt_reason == converged</code> for the last QP and "
        "<code>||grad_x L||_inf <= rtol*scale</code> with the QP multipliers before "
        "<code>zero_step_patience</code> consecutive zero steps can declare success; a zero step from a "
        "non-converged QP should instead trigger a steepest-descent/exact-solve fallback.</li>"
        "<li>The <code>asls-pcg-exact</code> configuration (exact Hessian) is the worst affected (100 "
        "tasks) because indefinite exact Hessians make <code>rz</code> non-monotone immediately.</li>"
        "</ol>"
    ),
    "false_infeasible_certificate": (
        "<ol>"
        "<li><b>Certify infeasible stationarity with the right quantity.</b> Replace "
        "<code>stationary & ~feasible</code> in <code>_infeasible_stationary</code> by a test on the "
        "infeasibility measure: <code>||J_viol^T c_viol|| <= tol * ||c_viol||</code> (gradient of "
        "<code>0.5*||c^+||^2</code>), exactly as the funnel method does with <code>chi_v</code>. A "
        "zero QP step with <code>grad_x L(x, lambda_QP) = 0</code> says nothing about feasibility.</li>"
        "<li><b>Never certify on the first iterations.</b> Require the restoration/feasibility progress "
        "to have stalled over several steps (<code>stall</code> patience) and the QP to have "
        "<code>kkt_reason == converged</code>.</li>"
        "<li><b>Handle degenerate working sets.</b> When the working set has <code>&gt;= n</code> rows "
        "(HS71: 4 active bounds + 1 equality) or inconsistent rows, solve the relaxed QP "
        "(SLSQP's <code>xi</code> elastic variable, or a least-squares normal step "
        "<code>min ||A d + c||</code> subject to the bounds) and drop bounds with wrong-signed "
        "multipliers before giving up.</li>"
        "</ol>"
    ),
    "kkt_solver_failure": (
        "<ol>"
        "<li><b>Regularise the Craig projector.</b> Solve <code>(A A^T + delta I) y = r</code> with a "
        "small <code>delta</code> (or a rank-revealing fallback) so that over-determined / dependent "
        "working sets give a least-squares projection instead of <code>singular</code>/NaN. The "
        "projected-CG solver already behaves this way, which is why <code>asls-pcg</code> survives the "
        "same working sets.</li>"
        "<li><b>Bound the initial working set.</b> Do not activate more violated constraints than "
        "<code>n</code>; pick the most violated ones (or those with independent rows) and let the "
        "active-set iterations add the rest.</li>"
        "<li><b>Never return a non-finite direction.</b> On <code>singular</code>/<code>nonfinite</code> "
        "fall back to the projected steepest-descent direction of the merit instead of a NaN step; "
        "counting the NaN step as a QP failure wastes the three secant resets and ends the run.</li>"
        "<li>MINRES-QLP shows the same behaviour (<code>max_steps_reached</code> after 2 iterations on "
        "rank-deficient saddle systems): raise its iteration budget relative to <code>n + m</code> "
        "and treat the minimum-length solution it returns as usable.</li>"
        "</ol>"
    ),
    "restoration_failure": (
        "<ol>"
        "<li><b>Add a feasibility restoration / relaxed QP.</b> When the line search fails and "
        "<code>feas &gt; atol</code>, minimise <code>||c^+||</code> (bound-constrained Gauss-Newton or "
        "SLSQP's elastic <code>xi</code>) instead of retrying the same QP direction.</li>"
        "<li><b>Safeguard the normal step.</b> Cap the feasibility component at a multiple of the "
        "current violation (Byrd-Omojokun <code>||n|| <= 0.8 Delta</code>, or "
        "<code>||n|| <= kappa * ||c||</code>) so a nearly singular Jacobian (BYRDSPHR) cannot produce "
        "<code>||d|| = 1e5</code>.</li>"
        "<li><b>Do not raise the penalty on a failed line search.</b> Increase <code>penalty</code> only "
        "when the directional derivative of the merit is not sufficiently negative; the "
        "<code>2e9</code> penalties seen here make every later step unacceptable.</li>"
        "<li><b>Release bounds with wrong-signed multipliers</b> before the step is handed to the "
        "line search; a QP step with a positive merit slope (HS71, slope <code>+7.8e-2</code>) must "
        "never reach the line search.</li>"
        "</ol>"
    ),
    "ip_bound_slack_residual": (
        "<p><em>The slack seeding and the scaled normal step are implemented (see the fix "
        "attempts). What is left is a feasibility-restoration problem, the same one the "
        "active-set configurations have on these instances.</em></p>"
        "<ol>"
        "<li><b>Give feasibility priority when the normal step is radius-limited.</b> When "
        "<code>||n|| = 0.8 Delta</code> and the predicted violation reduction is a small fraction "
        "of <code>||c||</code>, enlarge the radius for the normal step (or take a pure normal "
        "step, <code>t = 0</code>) instead of composing a tangential step whose model decrease "
        "is then rejected; scipy's trust-constr does this through the <code>theta</code> "
        "scaling of the normal step and by never counting a rejected step against anything but "
        "the radius.</li>"
        "<li><b>Add a restoration phase.</b> After <code>k</code> consecutive rejected steps "
        "with <code>feas &gt; atol</code> minimise <code>||c(x,s)||</code> (Gauss-Newton on "
        "the scaled rows, bounds kept by fraction-to-the-boundary) until the violation has "
        "dropped by a fixed factor, then resume the barrier iteration from there. Without it "
        "the radius collapse feeds the secant reset ladder and the run ends as "
        "<code>secant_recovery_failure</code> with a misleading diagnosis.</li>"
        "<li><b>Report the right status.</b> A run that stops infeasible should end with an "
        "<code>infeasible</code> result (the analogue of trust-constr status 4, "
        "<code>constraint_violation_exceeds_gtol</code>), not <code>secant_recovery_failure</code> "
        "/ <code>max_steps_reached</code>.</li>"
        "</ol>"
    ),
    "ip_barrier_stall": (
        "<ol>"
        "<li><b>Decide the <code>mu</code> update on the barrier KKT error with a "
        "<code>mu</code>-proportional tolerance.</b> Use the scaled error of the barrier "
        "problem (stationarity, <code>||c(x,s)||</code>, <code>||S z - mu e||</code>) against "
        "<code>kappa_eps * mu</code> (Fiacco-McCormick; trust-constr uses "
        "<code>BARRIER_DECAY_RATIO * mu</code>) instead of the final <code>atol</code>. On "
        "CANTILVR / HS100 / HS106 the iterates satisfy everything except "
        "<code>compl = mu</code>, which by construction can never fall below <code>mu</code>.</li>"
        "<li><b>Add the <code>xtol</code>-style exit to the inner loop.</b> When the trust "
        "radius collapses below <code>radius_floor * max(1, ||x||)</code> inside a barrier "
        "problem, treat that barrier problem as finished: reduce <code>mu</code>, reset the "
        "radius and continue, exactly as trust-constr does (<code>tr_radius &lt; xtol</code>). "
        "Today the collapse is reported to the secant reset policy as a model stall, which "
        "is why 69 of these runs end <code>secant_recovery_failure</code>.</li>"
        "<li>Treat a predicted reduction below <code>eps * |merit|</code> as a converged inner "
        "problem, not as a rejected step (it must not count towards the secant reset streak).</li>"
        "<li>Only once <code>mu</code> is below the barrier tolerance and the radius is still "
        "collapsed should the run terminate - with a dedicated result code.</li>"
        "</ol>"
    ),
    "ip_bound_violation": (
        "<ol>"
        "<li>Keep <code>x</code> inside <code>[lb, ub]</code> exactly: apply the fraction-to-the-boundary "
        "rule to <code>x</code> (not only to the slacks) and remove the bound slacks from the funnel "
        "violation <code>v</code>; with slacked bounds a v-iteration is allowed to trade bound "
        "violation against the funnel budget.</li>"
        "<li>Until then, clip accepted steps to the box and reset the bound slacks after each "
        "accepted iteration (see the trust-region recommendation).</li>"
        "</ol>"
    ),
    "termination_missed": (
        "<ol>"
        "<li>Evaluate the KKT termination test (stationarity, feasibility, complementarity with the "
        "current <code>mu</code>) <em>before</em> the step acceptance logic, and stop when it holds "
        "even if the last step was rejected.</li>"
        "<li>Do not count rejections with <code>|predicted| <= eps * max(1, |merit|)</code> towards the "
        "secant-reset / funnel-failure streak; at that point the model cannot be improved by a "
        "reset.</li>"
        "<li>Scale the stationarity test by <code>max(1, ||grad f||, ||y||)</code> rather than by "
        "<code>max(|f|, 1)</code> so badly scaled least-squares problems (PALMER*, PFIT*) reach the "
        "test at the same relative accuracy.</li>"
        "</ol>"
    ),
    "multiplier_failure": (
        "<ol>"
        "<li><b>Break the y-iteration deadlock.</b> When the tangential gate reports "
        "<code>infeasibility_dominates</code> (or <code>t = 0</code>) while <code>v &gt; eps_v</code>, "
        "force the normal step on the next iteration (the optional Step 10 extension "
        "<code>always_normal=True</code> already fixes HS22, GIGOMEZ3, HS32) and shrink "
        "<code>v_max</code> towards <code>max(kappa * v, eps_v)</code> on every y-iteration so the "
        "gate <code>v &gt;= kappa_vv * v_max</code> eventually opens. Make this the default.</li>"
        "<li><b>Investigate the zero normal step</b> on the equality-only instances (BT9, BT10, HS14): "
        "<code>pi_v &gt; 0</code> but the scaled normal solver returns <code>w_n = 0</code> "
        "(radius <code>radius_v</code>, Cauchy decrease, or the <code>kappa_B</code> multiplier gate "
        "are the suspects).</li>"
        "<li>Only report <code>stationarity_stall</code> when <code>pi_v ~ 0</code> (a genuine "
        "infeasible stationary point); otherwise keep iterating with the normal step forced.</li>"
        "</ol>"
    ),
    "secant_recovery_premature": (
        "<ol>"
        "<li>Count only <em>accepted-model</em> failures towards the recovery streak: rejected steps "
        "whose predicted reduction is below rounding, or that occur right after a <code>mu</code> "
        "update, should not escalate the recovery stage.</li>"
        "<li>After the identity reset, allow at least <code>n</code> iterations before the next "
        "stage, and shrink the trust-region radius with the reset instead of keeping the radius that "
        "produced the rejections.</li>"
        "<li>When <code>kkt_last</code> is already within a factor 10-100 of the tolerance, return "
        "the best iterate with a non-fatal status rather than <code>secant_recovery_failure</code>.</li>"
        "</ol>"
    ),
    "secant_recovery_stuck": (
        "<ol>"
        "<li>These runs are dominated by the trust-region / funnel step computation, not by the "
        "curvature: look at <code>rho</code> and <code>radius_last</code> (radius collapsed to "
        "1e-3 or below with <code>rho &lt;= 0</code>). The model and the merit disagree at tiny radii, "
        "which is only possible if the model omits a term that the merit contains (penalty update "
        "between model and evaluation, barrier term of slacks that were clipped by the "
        "fraction-to-the-boundary rule, or non-finite barrier at <code>s &lt;= 0</code>).</li>"
        "<li>Compare <code>predicted</code> with a finite-difference of the merit along the step in "
        "the diagnostics channel and assert agreement to first order; fix the inconsistency.</li>"
        "<li>Use the exact-Hessian or damped-BFGS curvature on the <code>(x, s)</code> system rather "
        "than resetting to the identity, which wipes out the barrier curvature and makes the next "
        "steps worse.</li>"
        "</ol>"
    ),
    "slow_progress": (
        "Raise the step budget for these (badly scaled PALMER/PFIT/HS268 least-squares) problems or, "
        "better, use the exact Hessian / Gauss-Newton curvature where available; a limited-memory "
        "secant with unit initial scaling needs hundreds of iterations on Hessians with condition "
        "numbers above 1e6. Scaling the initial secant diagonal by <code>s^T y / y^T y</code> "
        "(Shanno-Phua) and tightening the line search (Wolfe instead of Armijo) will reduce the count."
    ),
    "step_collapse": (
        "The line search / trust region keeps producing unusable directions while the optimality "
        "residual does not move. Add the direction checks from the restoration recommendations "
        "(slope must be negative, step norm bounded), fall back to a scaled steepest-descent step on "
        "the merit after two consecutive line-search failures, and reset the secant on a stall of the "
        "merit rather than only on QP failures."
    ),
    "cycling": (
        "Oscillating active sets / penalties. (i) Make the penalty update monotone with hysteresis "
        "(only increase when the directional derivative test fails by a margin, never decrease "
        "within a run); (ii) warm-start the QP working set from the previous iteration; (iii) for "
        "the interior-point methods, prevent <code>mu</code> from oscillating by using a monotone "
        "Fiacco-McCormick schedule until the KKT error of the barrier problem is small. Problems "
        "in this group that also appear under <em>unbounded</em> for other configurations are "
        "benchmark artefacts, not cycling."
    ),
    "merit_stagnation": (
        "Mostly huge penalties (<code>1e3 - 1e8</code>) making the merit insensitive to objective "
        "progress, plus runs stopped by the patience window while the step size was still shrinking. "
        "Bound the penalty growth (increase by at most a factor 10 per iteration and only when "
        "required by the descent test), reset it when the iterate becomes feasible, and extend the "
        "stagnation patience when <code>alpha</code> is still decreasing."
    ),
    "merit_stagnation_near_opt": (
        "Run the KKT test on the current iterate before declaring stagnation; these are solved "
        "instances that simply never reached the test."
    ),
    "loose_tolerance": (
        "The relative stationarity test <code>||grad L|| <= rtol * max(|f|, 1)</code> is too loose "
        "when <code>|f|</code> is large; scale by <code>max(1, ||grad f||)</code> or use an "
        "absolute tolerance on the scaled KKT error. Also report the final objective gap in the "
        "benchmark at a slightly looser <code>f_tol</code> (1e-3) to separate these from true "
        "failures."
    ),
    "local_minimum": (
        "Genuine alternative KKT points reached by one configuration while another reaches the "
        "reference. Not a correctness issue; differences come from the globalisation (merit vs. "
        "funnel, trust region vs. line search). A second-order correction step and the exact Hessian "
        "tend to follow the reference path more often. Score these against the best local value or "
        "mark them as 'alternative optimum' in the report."
    ),
    "local_minimum_consensus": (
        "Benchmark artefact: scipy SLSQP started from the same point converges to the same value "
        "(HS5, HS16, ZECEVIC*, JUDGEB, CB2/CB3, ...). Exclude them from the 'wrong objective' count "
        "or compare against the local value."
    ),
    "reference_mismatch": (
        "Fix the reference: the sif2jax <code>expected_objective_value</code>/<code>expected_result</code> "
        "is inconsistent with the problem as converted (HS34: <code>x*</code> infeasible by 2.2e4; "
        "EGGCRATEB: <code>f* = 0</code> while scipy finds 9.49; several problems return a feasible "
        "point strictly below <code>f*</code>). Validate every catalog entry by checking "
        "<code>c(x*)</code> and <code>f(x*)</code> at build time and drop <code>f*</code> when the "
        "check fails."
    ),
    "unbounded": (
        "Fix the conversion, not the solver. Detect L-type constraints in sif2jax (the raw "
        "<code>c(x) - b &lt;= 0</code> rows) and negate them in <code>benchmarks.problems.to_sqpdax</code>, "
        "or report the issue upstream to sif2jax; verify each problem by checking that "
        "<code>expected_result</code> (when available) satisfies <code>c(x*) &gt;= 0</code> under the "
        "assumed convention. Until then exclude POLAK*, CHACONN*, GOFFIN, MINMAXBD, MADSEN and the "
        "other minimax instances from the scoring. A cheap solver-side safeguard is still worth "
        "adding: stop with an <code>unbounded</code> status when <code>f</code> decreases by more "
        "than <code>1e3 * max(1, |f0|)</code> while feasible."
    ),
    "near_feasible": (
        "The violation is between <code>1e-6</code> and <code>1e-4</code>: let the method keep "
        "iterating (the <code>infeasible</code> fatal flag fires too early) and scale the feasibility "
        "test by <code>max(1, ||c(x0)||)</code>."
    ),
    "problem_infeasible": (
        "No method finds a feasible point. Verify the conversion of these instances (several are "
        "L-type sign issues as well: LOOTSMA, HIMMELBD, DUALC*, NASH, HS96/HS97). Keep them in a "
        "separate 'unverified' bucket until the sign convention is validated."
    ),
    "nonfinite": (
        "Guard the model against overflow: cap the penalty and the merit at <code>1e30</code>, skip "
        "secant pairs with non-finite <code>s^T y</code>, and reject non-finite trial points inside "
        "the line search without counting them as failures."
    ),
    "evaluation_error": (
        "Exclude S365/S365MOD (NaN objective at <code>x0</code>) or start them from a point where "
        "the square roots are defined."
    ),
    "ip_infeasible_stationary": (
        "Single instance; the funnel's Step 8 fired after the normal step stalled. Re-run from a "
        "perturbed point and, if reproducible, apply the normal-step forcing from the "
        "multiplier-failure recommendation."
    ),
    "compile_timeout": (
        "HS105 exceeds the compile budget in every configuration (large <code>mineq</code> with a "
        "dense Jacobian unrolled by XLA). Use <code>jax.lax.map</code>/<code>vmap</code> in the "
        "constraint evaluation, cache compiled executables across configurations, or raise the "
        "compile budget for this tier."
    ),
    "run_timeout": (
        "Large instances that did not finish 500 steps within the wall-clock budget; they fall under "
        "the same slow-progress recommendations once the step count is reduced."
    ),
    "unclassified": "Inspect manually.",
}
"""Per-cause recommendations rendered in the report."""


PRIORITIES: list[tuple[str, list[str], str]] = [
    (
        "Handle infeasible starts with over-determined violated working sets: relaxed / "
        "elastic QP (SLSQP's `xi`), restoration phase with a bounded normal step, "
        "regularised Craig / MINRES-QLP solves, and certify `infeasible_stationary` with the "
        "gradient of the violation measure. This single situation (DUALC*, HS96-98, HS19, "
        "HS106, HS116, OET*, TENBARS3, OPTCNTRL ...) accounts for most of the instances scipy "
        "solves and no sqpdax configuration does",
        [
            "kkt_solver_failure",
            "false_infeasible_certificate",
            "restoration_failure",
            "ip_bound_slack_residual",
            "near_feasible",
            "nonfinite",
        ],
        "`subproblem/active_set.py`, `subproblem/solver/{craig,minres_qlp}.py`, "
        "`minimiser/active_set_linesearch.py::_infeasible_stationary`, "
        "`subproblem/solver/trust_region.py` (normal step / restoration)",
    ),
    (
        "Interior point: decide the `mu` update on the barrier KKT error with a "
        "`mu`-proportional tolerance and add the `xtol`-style radius-collapse exit to the inner "
        "loop (trust-constr's `tr_radius < xtol` rule) instead of feeding the collapse to the "
        "secant reset ladder",
        ["ip_barrier_stall", "ip_bound_violation", "ip_infeasible_stationary"],
        "`minimiser/trust_region_interior_point.py`, `minimiser/trust_funnel_interior_point.py`, "
        "`barrier/update.py`",
    ),
    (
        "Run the KKT test before step acceptance and do not count rounding-level rejections as "
        "secant failures (cheap wins: 43 of the 57 `termination_missed` tasks are solved by "
        "scipy SLSQP, 22 of the 24 `merit_stagnation_near_opt`)",
        [
            "termination_missed",
            "secant_recovery_premature",
            "merit_stagnation_near_opt",
            "loose_tolerance",
        ],
        "`minimiser/base.py` secant recovery, termination tests",
    ),
    (
        "Fix the sif2jax L-type constraint sign in the benchmark adapter and validate `f*`/`x*` "
        "against the scipy consensus at catalog build time (benchmark artefacts, not solver "
        "defects: none of these tasks is solved by a scipy baseline either)",
        [
            "unbounded",
            "reference_mismatch",
            "local_minimum_consensus",
            "problem_infeasible",
            "evaluation_error",
            "stationarity_scale_f",
        ],
        "`benchmarks/problems.py`, `benchmarks/catalog.py`",
    ),
    (
        "Break the funnel y-iteration deadlock (`always_normal` + shrink `v_max` on y-iterations)",
        ["multiplier_failure"],
        "`subproblem/solver/trust_funnel.py` Step 10",
    ),
    (
        "Require a converged QP and check complementarity before the `qp_kkt` guard declares "
        "success",
        ["false_convergence", "qp_inactive_multipliers"],
        "`minimiser/active_set_linesearch.py::termination_flags`",
    ),
    (
        "Globalisation / scaling improvements (penalty hysteresis, Shanno-Phua secant scaling, "
        "direction checks)",
        [
            "cycling",
            "slow_progress",
            "step_collapse",
            "secant_recovery_stuck",
            "merit_stagnation",
            "local_minimum",
        ],
        "merit / secant modules",
    ),
    (
        "Performance",
        ["compile_timeout", "run_timeout"],
        "problem evaluation, compile budget",
    ),
]


def _md_code(text: str) -> str:
    """HTML-escape ``text`` and render `code` spans."""
    out = html.escape(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)


_CATALOG_ATTEMPT: dict[str, str] = {
    "title": "Validate the sign convention and the reference optimum at catalogue build time",
    "compare_dir": "v3/fix_a_catalog",
    "change": _md_code(
        "`benchmarks/validate.py` (new, `python -m benchmarks validate`): every "
        "instance with `n <= 1000` is solved with scipy SLSQP under the candidate "
        "inequality sign conventions - the documented `ineq >= 0`, the rows that "
        "the sif2jax reference point `x*` violates, and all rows flipped - and "
        "the first convention under which the solve is feasible and consistent "
        "with the reference (`f(x*)`, then `expected_objective_value`) is kept; "
        "trust-constr is then run on that conversion as a second, independent "
        "value. The decision ranks `x*`-evidence first, the scipy consensus "
        "second, a single feasible SLSQP point third (`unverified`, reference "
        "kept), and marks the instance `valid = False` when no convention gives "
        "a bounded feasible problem (`unbounded`, `infeasible`, "
        "`evaluation_error`). The outcome is written into "
        "`benchmarks/catalog.csv` (`ineq_flip`, `fstar`, `fstar_sif`, "
        "`fstar_source`, `valid`, `validation`, `validation_note`), "
        "`python -m benchmarks catalog` carries the columns over, "
        "`benchmarks.problems.to_sqpdax` applies the recorded row flips "
        "(`h = c` instead of `h = -c` on L-type rows) and the validated `f*` "
        "at run time, and `catalog.select` drops invalid instances unless "
        "`include_invalid=True`. 28 unit tests cover the decision table, the "
        "flip round trip, the catalogue columns and the adapter."
    ),
    "outcome": _md_code(
        "Validation of the 691 catalogue instances (511 with `n <= 1000`): 209 "
        "`ok`, 23 `sign_flipped` (33 instances with every row flipped, 4 with a "
        "subset such as `2,4,7`), 57 `fstar_from_consensus` (no sif2jax value, "
        "scipy agrees on one), 29 `fstar_mismatch` (both scipy solvers agree on "
        "a feasible value different from sif2jax's; the consensus is used), 37 "
        "`unverified`, and 119 invalid - 110 `infeasible` under every "
        "convention (mostly the `*NE` nonlinear-equation sets, whose equality "
        "rows have no exact solution), 9 `unbounded`; 35 `error` are scipy "
        "SLSQP worker crashes (SIGTRAP/SIGSEGV in the Fortran code) and 2 "
        "timeouts, both kept valid with the sif2jax reference. Mapped onto the "
        "report's diagnoses: of the 27 `unbounded` instances 16 are "
        "`sign_flipped` and 3 get a consensus `f*` (8 stay unbounded and are now "
        "excluded); of the 25 `reference_mismatch` instances 8 are "
        "`fstar_mismatch`, 4 `sign_flipped`, 8 `unverified`; of the 20 "
        "`local_minimum_consensus` instances 11 are `fstar_mismatch`; of the 9 "
        "`problem_infeasible` instances 5 are infeasible under every "
        "convention and excluded. The measured line is the re-run of all 333 "
        "affected tasks on the validated catalogue (together with the solver "
        "changes of the other attempts in this report; the now-invalid "
        "instances were re-run too so the counts are over the original "
        "groups): `unbounded` 80/133 - the 26 rows that keep the label are "
        "20 on instances now excluded as invalid (BURKEHAN, ROSEPETAL, "
        "DIAGIQ*/DIAGNQ* on trip/tfip) and HS108 / POLAK2 on trip/tfip, which "
        "still diverge under the corrected convention although scipy solves "
        "them (a solver defect, now visible); the other 27 unsolved rows "
        "moved to ordinary solver diagnoses (POLAK2 and MAKELA3 converging "
        "slowly at `kkt = 3e-4 .. 1e-3`, ELATTAR and HAIFAM scattered). "
        "`reference_mismatch` 49/85 - the 34 rows that keep the label are 33 "
        "`unverified` instances (DEVGLA2B, FLETCHER, HATFLDH, HS104, HS112, "
        "HS34, HS54, PALMER2E) plus HS13, whose sif2jax value is kept; "
        "`local_minimum_consensus` 58/90 (21 keep it: 15 `unverified`, "
        "EGGCRATEB / HS20 validated `ok`, i.e. genuine local minima); "
        "`problem_infeasible` 4/25 (8 of the 11 remaining are on BATCH, "
        "DEMBO7, HIMMELBD, infeasible under every convention and excluded; "
        "CLEUVEN7 is `unverified`). The `unbounded` classifier rule was also "
        "tightened: a run ending at or above a known `f*` is no longer "
        "labelled unbounded on the strength of a (pre-fix) diverging scipy "
        "baseline alone. What this does not fix are the `unverified` "
        "instances, where scipy SLSQP reaches a feasible point above "
        "sif2jax's `f*` and trust-constr does not converge, so no consensus "
        "exists and the sif2jax value is kept."
    ),
}


_WORKING_SET_CAP_ATTEMPT: dict[str, str] = {
    "title": "Bound the initial working set to at most `n - meq` violated rows",
    "compare_dir": "v3/fix_b1_cap",
    "change": _md_code(
        "`working_set_policy.py::WorkingSetPolicy.initial_set` (new, called by "
        "the active-set loop on the cold / warm-started set): the inequality "
        "and bound rows of the initial working set are capped at the degrees "
        "of freedom left by the equalities, `n - meq`; the most violated rows "
        "fill the slots (`cap_rows`). The same cap is applied by the policies "
        "at every working-set iteration (rows staying in the set first, then "
        "the most violated of the new rows), so the KKT system handed to "
        "Craig / MINRES-QLP is never over-determined and the projected-CG "
        "solver never has to answer with a least-squares particular "
        "solution. Constraints left out are handled by the merit function of "
        "the outer loop and re-examined at the next solve. No regularisation "
        "was added to the Craig projector, as instructed. "
        "`cap_working_set=False` restores the old behaviour. (First version; "
        "refined after the regression check, see the last attempt of this "
        "group.)"
    ),
    "outcome": _md_code(
        "Necessary but not sufficient: 25/122 affected tasks solved with the "
        "cap alone (asls-craig 15/65, asls-minresqlp 10/50). 63 kept the "
        "diagnosis - on DUALC*, HS96-98, OET*, TENBARS3 and the discretised "
        "semi-infinite problems the `n - meq` most violated rows are "
        "near-parallel (duplicate discretisation rows, or bounds that fix the "
        "variables already fixed by an equality), so a full set of `n - meq` "
        "rows is still rank-deficient and the KKT solve fails as before. 13 "
        "moved on to `restoration_failure` and 5 to "
        "`false_infeasible_certificate`: with a solvable QP the loop now runs "
        "but has no mechanism to leave the infeasible region (attempts 2 and "
        "3 below)."
    ),
}

_INDEPENDENCE_SCREEN_ATTEMPT: dict[str, str] = {
    "title": "Screen the capped set for linear independence (cumulative with attempts 1-3 of the infeasible-start group)",
    "compare_dir": "v3/fix_b_restoration",
    "change": _md_code(
        "`working_set_policy.py::independent_rows`: when the capped set would "
        "be full (at least `n - meq` rows requested) the candidate rows are "
        "additionally accepted in priority order through a Gram--Schmidt "
        "screen against the equality Jacobian and the rows already accepted; "
        "a row whose residual is below `independence_tol = 1e-4` times its "
        "norm is skipped and the next most violated row takes the slot. "
        "This keeps near-duplicate violated rows from producing a singular "
        "KKT system (`independence_tol = 0` disables the screen). Measured "
        "together with the violation-measure certificate and the restoration "
        "phase (attempts of `false_infeasible_certificate` / "
        "`restoration_failure`), which are what the capped runs fall through "
        "to."
    ),
    "outcome": _md_code(
        "57/123 affected tasks solved in the combined run (asls-craig 35/66, "
        "asls-minresqlp 22/50; the 7 projected-CG / pasls instances were never "
        "a KKT-solver problem and stay unsolved; DUALC1 solves on all four "
        "KKT configurations). The 24 that keep the diagnosis fall in three "
        "groups: rank-deficient *equality* Jacobians (MSS1/MSS2, HADAMARD, "
        "CLEUVEN7, TABLE7, FLT - the cap and the screen act on the "
        "inequality and bound rows only, and the Craig / MINRES-QLP solves "
        "fail on `A_eq` itself, ending at `kkt ~ 1e-14` on a point with "
        "`||c|| ~ 1`), the DUAL1-3 quadratics on asls-craig / asls-minresqlp "
        "that reach `f*` to `1e-11` but run out of steps at `kkt = 2e-6 .. "
        "4e-6` (slow convergence, the termination group), and the `n = 1000` "
        "DIAGPQB/DIAGPQT bound-constrained quadratics (500 steps at "
        "`kkt = 0.1 .. 0.9`, also on asls-pcg, so not the KKT solver). Of the "
        "rest, 9 end in `secant_recovery_premature` (TENBARS3/4, HS98, OET5, "
        "HS114: the restoration gets the iterate feasible but the secant "
        "recovery is exhausted by the line-search failures on the way) and 7 "
        "in `restoration_failure`. Kept: the cap removes a whole class of "
        "hard failures and the per-iteration cost of the screen is one "
        "Gram--Schmidt pass over at most `n - meq` rows."
    ),
}

_CAP_REFINEMENT_ATTEMPT: dict[str, str] = {
    "title": "Refine the cap after the tiny-tier regression check (final rule; cumulative with the whole round)",
    "compare_dir": "v3/fix_b_final",
    "change": _md_code(
        "The regression run of the tiny tier (273 instances x 7 "
        "configurations, previously solved tasks must stay solved) attributed "
        "eight lost tasks to the first version of the cap: HS13 on all five "
        "active-set configurations and BIGGS3 (the independence screen "
        "dropped one of two nearly parallel constraints that are both active "
        "at a degenerate solution), HS119 and LSNNODOC (the cap truncated a "
        "consistent set of active bounds at a degenerate vertex with more "
        "active bounds than free variables, pinning the QP to a wrong "
        "face). The other 24 flips were platform noise (the baseline was run "
        "on x86-64, the check on darwin; HS38 pasls, DEVGLA1B, HS68, HS1 "
        "trip, HS36 tfip fail at the base commit here too) or catalogue "
        "effects of the artefact fix (BT7, DGOSPEC, HS90, MINMAXBD, HS268). "
        "Two refinements were tried. (i) *Keep every row that is active at "
        "the reference point, cap and screen only the violated rows*: fixed "
        "HS13 / BIGGS3 / LSNNODOC / HS119-pcg-exact but lost DUALC1, HS96 "
        "and HS97, whose start sits on all of its bounds with a violated "
        "equality - keeping the bounds pins the QP to `d = 0` and the "
        "equality can never be met. (ii) *Final rule*: the cold-start set is "
        "kept whole only when the reference point is feasible (no "
        "inequality, bound or equality violated by more than `tol`: every "
        "row is then consistent with `d = 0`); from an infeasible reference "
        "point the whole set is capped by decreasing violation, and rows "
        "staying in the set during the refresh are never dropped or "
        "screened. A third variant that also switched the per-iteration cap "
        "off at feasible iterates was discarded: it re-broke DUALC1 and "
        "GOULDQP1 (the first QP of an infeasible start is where the cap is "
        "needed, and the refresh of that QP leaves the infeasible region)."
    ),
    "outcome": _md_code(
        "Kept. On the 285 tasks shared with the previous measurement the "
        "final round solves 122 (previously 119): DUAL1-3 on asls-craig / "
        "asls-minresqlp / asls-pcg (6 tasks), HS114 and OPTCNTRL "
        "asls-pcg-exact newly solve (the termination work, these had been "
        "stopping at `kkt = 2e-6 .. 4e-6`), while OET3 / OET7 asls-craig and "
        "OPTCNTRL on asls-craig / asls-minresqlp / asls-pcg are lost "
        "(OPTCNTRL ends at `f* + 2e-8`, `||c|| = 3e-8`, flagged by the "
        "secant recovery after four rejected QP steps from the near-KKT "
        "point; OET3 / OET7 fail with the cap on or off). Against the "
        "re-classified baseline the groups read 56/111 "
        "(`kkt_solver_failure`), 52/95 (`false_infeasible_certificate`), "
        "21/54 (`restoration_failure`) and 13/122 (`ip_bound_slack_residual`, "
        "a group that grew from 36 to 122 trip tasks when the `unbounded` "
        "label was retired: the 13 are tasks the catalogue fix solved, trip "
        "itself has no restoration phase). Of the probe set, HS13, BIGGS3, "
        "LSNNODOC asls-craig, HS97, OET2, HS96, HS116, HS19, DUALC1 and "
        "DUALC5 solve; HS119 keeps a near-optimal `merit_stagnation` on "
        "asls-pcg-exact (`f* + 2e-9`, `kkt = 5e-3` after a `7.6e-6` step) "
        "and pasls, GOULDQP1 asls-pcg-exact stops at a stationary point of "
        "the non-convex QP with `f_gap = 1`."
    ),
}

_VIOLATION_CERTIFICATE_ATTEMPT: dict[str, str] = {
    "title": "Certify `infeasible_stationary` with the gradient of the violation measure",
    "compare_dir": "v3/fix_b_restoration",
    "change": _md_code(
        "`active_set_linesearch.py`: `infeasible_stationary` used to fire on "
        "Lagrangian stationarity at an infeasible iterate, which with an "
        "over-determined working set the QP multipliers can produce at *any* "
        "point. The metrics now carry `violation_stationarity = ||P (J+^T c+ / "
        "||c+||_2)||_inf / max_i ||J+_i||_inf` - the bound-projected gradient "
        "of `||c+||_2` (violated inequality rows and the equality residual, "
        "`P` zeroes the components pointing out of the box), relative to the "
        "largest row of the violated Jacobian - and the certificate requires "
        "`violation_stationarity <= sqrt(rtol)` in addition to the iterate "
        "being stuck (stationary Lagrangian or a rejected QP step *and* "
        "restoration fallback). The proximal variant inherits the test. "
        "Cumulative with the working-set cap / screen and the restoration "
        "phase."
    ),
    "outcome": _md_code(
        "67/115 affected tasks solved (asls-pcg-exact 43/65, asls-pcg 15/30, "
        "pasls 7/14): on these the old certificate was simply wrong and the "
        "iteration, once allowed to continue, reaches the optimum - with the "
        "restoration step doing the work on the infeasible starts "
        "(HS96, HS116, HS19, DUALC1/5, OET2/3/6/7, GOULDQP1, DUAL1-4 on "
        "asls-pcg-exact, ...). 14 keep the diagnosis: HS88/HS90 on four "
        "configurations each (scipy fails too), MSS1/MSS2 (rank-deficient "
        "equalities, `||c|| ~ 1` at a point where the violation gradient "
        "vanishes - the certificate is now right to report them), FLETCHER "
        "and HS106 on one configuration. 10 moved to `merit_stagnation` "
        "(DUALC2/8, HS97/98, OPTCNTRL, TENBARS1/2/4 on asls-pcg-exact / "
        "pasls: feasible, a decade above the tolerance), 3 to "
        "`merit_stagnation_near_opt` (DUAL2/3 on asls-pcg at `kkt = 1.4e-6 "
        ".. 1.7e-6`, the termination group), 8 to `restoration_failure` "
        "(HS83 on three configurations, DEGENLPB, DUALC1/pasls, HS87, OET1, "
        "TENBARS3) and 4 to `near_feasible` (DEGENLPA/B, FLETCHER at "
        "`||c+|| ~ 1e-5`). Kept."
    ),
}

_RESTORATION_ATTEMPT: dict[str, str] = {
    "title": "Restoration phase with a bounded Gauss--Newton normal step",
    "compare_dir": "v3/fix_b_restoration",
    "change": _md_code(
        "`active_set_linesearch.py::_restoration_step`: when the QP direction "
        "is rejected by the line search at an infeasible iterate (or the QP "
        "itself fails), the loop falls back to a normal step `d` minimising "
        "`||c_eq + J_eq d||^2 + ||(c+ + J+ d)||^2` over the violated rows "
        "subject to `lb <= x + d <= ub` and `||d||_inf <= restoration_radius * "
        "max(1, ||x||_inf)` (`restoration_radius = 10`). The bound-constrained "
        "least-squares problem is solved by projected CGLS rounds "
        "(matrix-free, `J` / `J^T` products only): variables that hit a bound "
        "are fixed and removed from the next round, and inequality rows that "
        "the current step would newly violate (`c + J d > atol`) are added as "
        "targets; four rounds. The step is accepted by an Armijo test on the "
        "L2 penalty `f + rho_R ||c+||_2` with `rho_R = max(rho, 2 max(grad f . "
        "d, 0) / predicted violation decrease)`, where the predicted decrease "
        "includes the bound violations. Two earlier variants are recorded "
        "under outcome. Implemented for the active-set line-search family "
        "(asls / pasls); the trust-region interior-point loop (`trip`, the "
        "whole `ip_bound_slack_residual` group) has no restoration phase."
    ),
    "outcome": _md_code(
        "27/65 affected `restoration_failure` tasks solved (asls-pcg 10/16, "
        "asls-pcg-exact 7/10, asls-craig 2/3, asls-minresqlp 3/4; pasls 5/32) "
        "and most of the infeasible-start instances listed in the request now "
        "solve on at least one asls configuration: HS96 (all five), HS97 "
        "(`f = 3.136`, three), HS116 (four), HS19 (five), DUALC1/2/5/8, "
        "OPTCNTRL and OET2/6/7 (three to five each), GOULDQP1; still failing "
        "everywhere are HS98 (secant recovery exhausted on the way, "
        "`secant_recovery_premature`), HS106 (`step_collapse` / certificate), "
        "TENBARS3/4 (feasible, then `secant_recovery_premature` or "
        "`merit_stagnation`) and OET5. What did not work on the way: (i) clipping the "
        "unconstrained Gauss--Newton step into the box destroyed the "
        "least-squares property (HS97 got a *negative* predicted decrease) - "
        "replaced by the projected CGLS rounds; (ii) keeping every active row "
        "as a `J d = 0` target over-constrained the normal step to `d ~ 0` on "
        "DUALC1 (215 active rows) - only violated rows are targets now and "
        "rows the step would violate are added in later rounds (DUALC1 then "
        "solves on all four KKT configurations). 19 moved to "
        "`merit_stagnation` (17 of them pasls: the restoration gets the "
        "iterate feasible but the proximal SQP steps that follow stall a "
        "decade above the tolerance) and 8 keep the diagnosis, all pasls "
        "(TRO*, DEGENLPB, DUALC8, HS81, HS107: the proximal QP does not "
        "enforce `A d = -c`, so the fallback is reached with a working set "
        "that bears no relation to the violated rows). "
        "`ip_bound_slack_residual`: 0/36, as expected - all trip, no "
        "restoration phase there; the fix for that loop would be a funnel- or "
        "filter-style restoration of the composite step, which is a separate "
        "piece of work. Kept."
    ),
}

_KKT_FIRST_ATTEMPT: dict[str, str] = {
    "title": "Run the KKT test before step acceptance, ahead of the fatal flags",
    "compare_dir": "v3/fix_c_termination",
    "change": _md_code(
        "`active_set_linesearch.py`, `trust_region_interior_point.py`, "
        "`trust_funnel_interior_point.py` (`termination_flags`): convergence "
        "is now decided from the KKT metrics of the current iterate alone and "
        "outranks the fatal flags accumulated by the last step. Before, "
        "`converged` was gated by `~recovery_fatal` (all three loops), by "
        "`barrier_updated` and `~subproblem_fatal` (trip) and by `~fatal` "
        "(tfip), so an iterate that passed the test after a rejected step or "
        "an exhausted secant recovery was reported as "
        "`secant_recovery_failure` / `max_steps`. `classify_termination` "
        "already ranks `converged` (20) above `fatal` (10); the gates were "
        "redundant with that ranking and wrong whenever a rejected step "
        "landed on a solution. Two guards came out of the measurement: the "
        "asls QP-optimality shortcut (QP reports optimal after a full "
        "accepted step) keeps the recovery gate, because it trusts the model "
        "rather than the measured residual; and the asls stationarity scale "
        "falls back to `max(|f|, 1)` when the last QP solve did not converge, "
        "because the multipliers of an unconverged working set can be "
        "arbitrarily large and `max(|L|, 1)` then passes any gradient "
        "(MATRIX2 on asls-minresqlp was reported `successful` at `kkt = 2e10` "
        "with `kkt / scale = 5e-11` before the guard)."
    ),
    "outcome": _md_code(
        "Small, honest gain: 4/57 `termination_missed` tasks solved (HS111 "
        "and TAME on trip, DUAL1 / DUAL3 on pasls - the cases whose last "
        "iterate really was below `atol` behind a fatal flag), 6/24 "
        "`merit_stagnation_near_opt` together with attempt 2, 0/34 "
        "`secant_recovery_premature`. The diagnosis was too generous: "
        "`termination_missed` fires at `kkt_last < 1e-3` and 52 of the 57 "
        "tasks end with `kkt = 1e-6 .. 5e-4`, three decades above the "
        "`1e-6` test. They are slow convergers (BIGGS5, HS26, HS47, HS57, "
        "DECONVC run 500 steps on every asls configuration), or trip / tfip "
        "runs whose secant recovery is exhausted at `kkt = 1e-5` after a "
        "string of genuine rejections (HS100 ends at `kkt = 1.4e-6`, HS46 "
        "`1.1e-6`, HS95 `1.2e-6` - just above the tolerance). Making the "
        "test outrank the flags cannot help those; the lever there is the "
        "secant reset schedule and the step quality near the solution, not "
        "the termination logic. No false successes among the 115 re-run "
        "tasks after the two guards (MATRIX2 is back to an honest "
        "`secant_recovery_failure`); the tiny-tier regression run is "
        "reported in the summary of this section. Kept, with the guards."
    ),
}

_ROUNDING_LEVEL_ATTEMPT: dict[str, str] = {
    "title": "Do not count rounding-level rejections as secant failures",
    "compare_dir": "v3/fix_c_termination",
    "change": _md_code(
        "`step_controller/base.py`: `StepResult.rounding_level` (new) is set "
        "by every controller when the predicted decrease it compared against "
        "is at the rounding level of the quantity it compared, "
        "`|pred| <= 100 eps max(1, |merit_0|)` (`at_rounding_level`): the line "
        "search uses the directional derivative of the merit, the "
        "trust-region manager its model reduction, the funnel the reduction "
        "governing the iteration type (`f`, `v`, or none for a y-iteration). "
        "Such a rejection compares two numbers that differ by round-off and "
        "says nothing about the model: asls no longer counts it as a "
        "line-search failure (it clears the streak - the reset policy fires on "
        "any positive streak, so holding the count would still escalate the "
        "recovery), trip / tfip no longer count it as a model stall. The "
        "asls stagnation test measures improvements relative to "
        "`max(|phi_best|, atol)` instead of `max(|phi|, 1)`, so a merit of "
        "`1e-14` improving by `1e-16` per step (PRICE4B) is progress, not "
        "stagnation."
    ),
    "outcome": _md_code(
        "6/24 `merit_stagnation_near_opt` solved (PRICE4B on all four "
        "line-search configurations, BQPGASIM on asls-craig / asls-pcg); "
        "11 others now run to the step budget instead of stopping on "
        "stagnation (`termination_missed`: AIRCRFTB, BQPGABIM, PFIT1LS, "
        "DECONVC at `kkt = 2e-6 .. 1e-4`), so the stagnation test no longer "
        "cuts them off but the iteration is simply slow; 6 keep the label "
        "(DIAGPQE `n = 1000` on four configurations at `kkt = 7e-6 .. 3e-5`, "
        "BQPGABIM / DUAL2). 0/34 `secant_recovery_premature`: 28 of these are "
        "trip / tfip and their rejections are not at the rounding level - "
        "`rho < -1` with a resolvable predicted decrease (LEVYMONT*, PALMER*, "
        "HS15/27/87/89, BT13), i.e. the model is genuinely wrong near the "
        "solution and the recovery is right to fire; the premature part is "
        "the schedule exhausting at `kkt = 1e-5 .. 1e-3`, which this change "
        "does not touch. Kept: the flag is cheap, the asls / trip / tfip "
        "streak semantics are now consistent (a rounding-level rejection "
        "clears the streak in all three), and the stagnation scale fixes a "
        "real misclassification."
    ),
}

FIX_ATTEMPTS: dict[str, list[dict[str, str]]] = {
    "unbounded": [_CATALOG_ATTEMPT],
    "reference_mismatch": [_CATALOG_ATTEMPT],
    "local_minimum_consensus": [_CATALOG_ATTEMPT],
    "problem_infeasible": [_CATALOG_ATTEMPT],
    "kkt_solver_failure": [
        _WORKING_SET_CAP_ATTEMPT,
        _INDEPENDENCE_SCREEN_ATTEMPT,
        _CAP_REFINEMENT_ATTEMPT,
    ],
    "false_infeasible_certificate": [
        _VIOLATION_CERTIFICATE_ATTEMPT,
        _CAP_REFINEMENT_ATTEMPT,
    ],
    "restoration_failure": [_RESTORATION_ATTEMPT, _CAP_REFINEMENT_ATTEMPT],
    "termination_missed": [_KKT_FIRST_ATTEMPT, _ROUNDING_LEVEL_ATTEMPT],
    "merit_stagnation_near_opt": [_ROUNDING_LEVEL_ATTEMPT, _KKT_FIRST_ATTEMPT],
    "secant_recovery_premature": [_ROUNDING_LEVEL_ATTEMPT, _KKT_FIRST_ATTEMPT],
    "false_convergence": [
        {
            "title": "Fix the projected-CG freeze rule",
            "compare_dir": "fix1_pcg_freeze",
            "change": _md_code(
                "`projected_cg.py`: the step is no longer frozen when the "
                "projected residual norm `r_z` fails to decrease (a non-monotone "
                "quantity in CG, so the freeze fired on the first iteration of "
                "many healthy QPs and returned `d = 0`). It now freezes on an "
                "increase of the QP model `q(d) = g.d + 0.5 d.B.d` (which CG "
                "decreases monotonically) or on non-finite values; on negative "
                "curvature at the first iteration the step is taken with the "
                "curvature sign flipped (previously the step was `0`). Two "
                "safeguards had to be added because the model test alone "
                "regressed the preconditioned unit tests: once CG has solved "
                "the reduced system to working precision the recomputed "
                "residual is round-off that is *not* in `null(A)`, and a step "
                "along it (`alpha` of two noise quantities, up to `1e14`) drags "
                "the iterate out of the constraint subspace while the model "
                "keeps decreasing — the old `r_z` freeze was catching exactly "
                "that by accident. CG therefore also stops on a relative "
                "tolerance `||r||_M <= rtol * ||r_0||_M` (`rtol = 1e-8`) and on a "
                "round-off-floor detector that compares `r_z = neg_grad . r` "
                "with `r^T M r` (equal in exact arithmetic because the projector "
                "is `M`-self-adjoint; they disagree by orders of magnitude at "
                "the floor)."
            ),
            "outcome": _md_code(
                "Partial but real. Across the three diagnoses the original "
                "308 `false_convergence` tasks were split into by the refined "
                "classifier, 98/300 now solve: 58/103 `false_convergence` "
                "(asls-craig 19/30, asls-pcg 21/31, asls-pcg-exact 18/39), "
                "40/191 `stationarity_scale_f`, 0/6 `qp_inactive_multipliers`. "
                "One task keeps the `false_convergence` label. The rest was "
                "never a CG-freeze problem: 50 are still accepted by the "
                "`rtol*max(|f|,1)` scaling of the stationarity test (the "
                "sign-flipped unbounded minimax problems and DIAG* quadratics), "
                "5 stop with multipliers on inactive working-set rows, and the "
                "others now run on to other pre-existing failure modes "
                "(reference mismatch 19, cycling 20, consensus local minimum "
                "19, local minimum 11, slow progress 10, merit stagnation 10, "
                "step collapse 7, ...). pasls (0/14) and minresqlp (0/8) do "
                "not use the projected CG and were unaffected."
            ),
        },
        {
            "title": "Correct the round-off floor detector for masked metrics",
            "change": _md_code(
                "`projector.py` / `projected_cg.py`: the floor detector of "
                "attempt 1 compared `r_z` with `r^T M r` using the *full* "
                "preconditioner `M`, but the projector works on the free "
                "subspace with `M~^-1 = F M^-1 F`, and the inverse of a "
                "principal block of `M^-1` is not the principal block of `M`. "
                "With a non-diagonal secant preconditioner and at least one "
                "fixed variable (any bound-constrained PASLS iterate) the "
                "identity failed by two to three orders of magnitude and the "
                "detector stopped CG on the first iteration, returning the "
                "particular solution with no null-space step. Found through "
                "the HS71 PASLS regression tests. The projector now exposes "
                "`project_pair(v) = (P v, Q v)` with `P = M~^-1 Q`; "
                "`(Q v)^T (P v) = v^T P v` holds for any masked metric and "
                "costs nothing extra."
            ),
            "outcome": _md_code(
                "Included in the benchmark run this report is based on "
                "(`4a31657`). Relative to the previous diagnosis 189 tasks are "
                "fixed and 41 regressed (section 1c); the regressions worth "
                "following up are OET2/4/5/6/7 on `asls-pcg` / `asls-pcg-exact` "
                "(semi-infinite problems with 1002 inequalities in `n = 3..7`, "
                "previously solved, now `infeasible` / "
                "`false_infeasible_certificate`) and BQPGABIM/BQPGASIM on all "
                "projected-CG configurations (`merit_stagnation` at "
                "`kkt = 1e-5`, one decade above the tolerance), which suggests "
                "the relative tolerance `rtol = 1e-8` or the floor detector "
                "still ends some QP solves one iteration early on heavily "
                "over-determined working sets."
            ),
        },
    ],
    "stationarity_scale_f": [
        {
            "title": "Projected-CG freeze fix (measured on this group too)",
            "compare_dir": "fix1_pcg_freeze",
            "change": _md_code(
                "Same change as attempt 1 of `false_convergence`; this group "
                "was part of the original `false_convergence` set and was "
                "re-run with it."
            ),
            "outcome": _md_code(
                "40/191 solved: these were tasks where the frozen zero step "
                "happened to coincide with a large `|f|`. The remaining 39 "
                "keep the label and are the `max(|f|,1)` scaling itself "
                "(deliberately not changed here: fixing it only turns the "
                "label from success into a failure)."
            ),
        }
    ],
    "qp_inactive_multipliers": [
        {
            "title": "Projected-CG freeze fix (measured on this group too)",
            "compare_dir": "fix1_pcg_freeze",
            "change": _md_code(
                "Same change as attempt 1 of `false_convergence`; re-run with it."
            ),
            "outcome": _md_code(
                "0/6 solved, as expected: the working set, not the CG, is "
                "the problem (the termination test never checks "
                "complementarity)."
            ),
        }
    ],
    "ip_bound_slack_residual": [
        {
            "title": "Seed the slacks consistently (IPOPT-style interior push)",
            "compare_dir": "fix2a_slack_seed",
            "change": _md_code(
                "`interior_point.py::_init_primal`: `initial_slack = 1` is "
                "replaced by `slack_push = 1e-2`. `x0` is clipped into the "
                "interior of `[lb, ub]` by `min(kappa1*max(1,|bound|), "
                "kappa2*(ub-lb))` (IPOPT `bound_push` / `bound_frac`) and the "
                "bound slacks are set exactly, `s_lb = x0 - lb`, `s_ub = ub - "
                "x0`, so the linear rows hold at the start. Inequality slacks "
                "are `max(-h(x0), kappa1*max(1,|h(x0)|))`: an infeasible start "
                "keeps a positive slack and a residual `h + s` that the normal "
                "step has to remove, so infeasible `x0` keep working. Fixed "
                "variables (`lb == ub`) get `s = kappa1` on both rows to keep "
                "the barrier finite (they cannot be consistent in a slack "
                "barrier; BOX2, BIGGS3/5, BQPGABIM, HS35MOD, ... went "
                "non-finite with exact zero slacks in the first run)."
            ),
            "outcome": _md_code(
                "Insufficient on its own: 17/173 affected tasks solved "
                "(HS35, BQP1VAR, CAMEL6, HS2, ... now converge with mu driven to "
                "2e-8). 127 kept the diagnosis: as soon as the dogleg normal "
                "step moves `x` (any problem whose general constraints are "
                "violated at the start or by the nonlinear iterates) the bound "
                "rows `x - lb - s_lb = 0` become inconsistent again, because the "
                "normal step is lifted with zero slack components and the "
                "tangential step preserves `A w = A w_n`. The seeding removes "
                "the *initial* inconsistency but not the mechanism that "
                "recreates it; attempt 2 addresses the mechanism."
            ),
        },
        {
            "title": "Let the normal step move the slacks (scaled normal step, N&W 19.34)",
            "compare_dir": "fix2_slack_normal",
            "change": _md_code(
                "`trust_region.py`: the default `normal_solver` of "
                "`TrustRegionInteriorPointSolver` is now the trust-funnel "
                "`ScaledNormalStepSolver` (Steihaug-Toint CGLS on `min ||c_hat + "
                "A_hat w||` in the scaled `(x, S^-1 s)` space over *all* rows "
                "`[eq | ineq | lb | ub]`, with the fraction-to-the-boundary "
                "backtrack and the Cauchy-decrease safeguard), instead of the "
                "x-only `DogLegSolver` lifted with zero slacks. `jac_mvp` / "
                "`jac_t_mvp` / `hess_mvp` were hoisted from "
                "`FunnelBarrierSubProblem` to `ScaledBarrierSubProblem` so the "
                "solver works on both; a `DogLegSolver` can still be plugged in "
                "and is lifted as before. The slack components of the step are "
                "therefore the linearised-row values `p_s = -(h + s) - J p_x`, "
                "`p_s_lb = (x - lb - s_lb) + p_x`, `p_s_ub = (ub - x - s_ub) - "
                "p_x` to the accuracy of the CGLS solve — the quantities N&W "
                "eq. 19.16 eliminates — but kept as explicit variables so the "
                "trust region and FTB rule still act on them. Cumulative with "
                "attempt 1 (the seeding is kept) and with the projected-CG fix "
                "(irrelevant for trip/tfip)."
            ),
            "outcome": _md_code(
                "The bug itself is fixed; the group is only partly solved. "
                "52/173 affected trip tasks now solve (17 with the seeding "
                "alone). On every re-run task the measured slack-row residual "
                "`max(|h+s|, |x-lb-s_lb|, |ub-x-s_ub|)` is now at round-off "
                "whenever the iterate is feasible (HS35, HS18, ALSOTAME, BOX2, "
                "HS1, PALMER*, ...), the bound rows no longer drift, and `mu` is "
                "driven to `1e-8` on the solved tasks. Of the 121 still failing: "
                "33 have consistent slacks but a stuck barrier "
                "(`ip_barrier_stall`: bound-constrained least squares PALMER*/"
                "PFIT*/DIAG*/HS1/HS38 ending `secant_recovery_failure` with "
                "`mu = 0.04 .. 1`), 15 are solved points that never fired the "
                "termination test, 10 cycle, 7+2 are local minima, 7+4 are "
                "secant-recovery issues, 6 reference mismatches; 35 keep a "
                "non-zero slack residual but *all 35 are infeasible end points* "
                "(TENBARS*, DUALC*, TRO*, DEGENLP*, CSFI*, HS97/98/106/114, "
                "BATCH, NASH, CLEUVEN7, ...): the composite step no longer "
                "breaks the slack rows, but it also fails to reduce the overall "
                "infeasibility (`nu` grows to 1e3, the radius collapses), and "
                "because the bounds are slacked rows `x` is left outside the "
                "box by the same amount (`feas_lb` equals `slack_resid`). That "
                "residue is the formulation issue of the third idea (bounds "
                "kept on `x` directly, not implemented as instructed) plus the "
                "weak feasibility restoration of the trust-region loop, not the "
                "slack-consistency bug. Side effects on the neighbouring "
                "groups: `ip_barrier_stall` 4/87 (trip 3/20, tfip 1/67), "
                "`ip_bound_violation` 1/28 — the funnel loop was essentially "
                "unaffected by either change, as expected (it already moved the "
                "slacks in its normal step). What did not work: exact zero "
                "slacks for fixed variables (non-finite barrier, fixed by the "
                "`kappa1` floor) and seeding alone (attempt 1). In the full "
                "benchmark re-run this report is based on, 51 of the previous "
                "173 tasks are solved and 28 moved to `ip_barrier_stall`; the "
                "36 still carrying this diagnosis are all infeasible end "
                "points, so the cause is re-labelled from *bug* to "
                "*algorithmic* (restoration), see the description."
            ),
        },
        _RESTORATION_ATTEMPT,
    ],
}
"""Per-cause log of fix attempts: ``{cause: [{"title", "change", "outcome",
"compare_dir"}]}``.  ``compare_dir`` names a sub-directory of the diagnostics
directory holding the ``compare.json`` written by the ``compare`` command; the
report pulls the before/after numbers from it."""


WORK_LOG: list[str] = [
    "<strong>Benchmark artefacts</strong> "
    + _md_code(
        "(`unbounded`, `reference_mismatch`, "
        "`local_minimum_consensus`, `problem_infeasible`): sif2jax L-type sign "
        "convention validated per instance against scipy and the reference "
        "optimum re-derived at catalogue build time (`python -m benchmarks "
        "validate`). 191/333 affected tasks solve on the corrected catalogue; "
        "the remainder are instances now excluded as invalid, `unverified` "
        "references, or genuine solver failures that the artefact was hiding. "
        "Kept."
    ),
    "<strong>Infeasible starts with over-determined working sets</strong> "
    + _md_code(
        "(`kkt_solver_failure`, `false_infeasible_certificate`, "
        "`restoration_failure`, `ip_bound_slack_residual`): working set of an "
        "infeasible reference point capped at `n - meq` independent rows (no "
        "Craig regularisation; a feasible degenerate vertex keeps its whole "
        "active set), `infeasible_stationary` certified with the gradient of "
        "the violation measure, restoration phase with a bounded "
        "projected-CGLS normal step in the asls family. 151/339 affected "
        "tasks solved at the first measurement (57/123, 67/115, 27/65, 0/36 "
        "- the last group is trip, which has no restoration phase); the "
        "final rule after the regression check solves 142/382 of the "
        "re-classified groups (56/111, 52/95, 21/54, 13/122). Two "
        "intermediate restoration variants and two cap variants were "
        "discarded on the way (box-clipped Gauss-Newton step; all active "
        "rows kept as targets; all active rows always kept in the working "
        "set; per-iteration cap switched off at feasible iterates). Kept."
    ),
    "<strong>Termination</strong> "
    + _md_code(
        "(`termination_missed`, `merit_stagnation_near_opt`, "
        "`secant_recovery_premature`): KKT test outranks the fatal flags, "
        "`StepResult.rounding_level` keeps rounding-level rejections out of "
        "the secant failure streaks, asls stagnation measured relative to "
        "`max(|phi_best|, atol)`. Only 10/115 affected tasks solve: the "
        "`termination_missed` label (`kkt_last < 1e-3`) mostly covers slow "
        "convergence, not a missed test. Two guards were needed (QP-optimality "
        "shortcut keeps its recovery gate; stationarity scale ignores the "
        "multipliers of an unconverged QP) after MATRIX2 turned into a false "
        "success. Kept, with the guards."
    ),
    "<strong>Regression check</strong> "
    + _md_code(
        "(tiny tier, 273 instances x 7 configurations, `python -m benchmarks "
        "run --tier tiny`, joined with the baseline on instance and "
        "configuration): 370 tasks newly solved, 23 lost. Solved per "
        "configuration before -> after: asls-craig 137 -> 199, asls-minresqlp "
        "147 -> 198, asls-pcg 147 -> 203, asls-pcg-exact 127 -> 211, pasls "
        "144 -> 185, tfip 103 -> 130, trip 123 -> 149. The first pass of the "
        "check (371 / 32) exposed eight regressions of the working-set cap "
        "(HS13 x5, BIGGS3, HS119, LSNNODOC), which led to the final cap rule "
        "(+14 / -6 against that pass: HS13 x5, LSNNODOC x4, BIGGS3, HS114, "
        "HS116, OET4, OET5 recovered; DEGENQP pasls, HS114 / HS116 "
        "asls-pcg-exact, HS116 pasls, OET3 / OET7 asls-craig lost). Of the 23 "
        "remaining losses, 11 are catalogue effects of the artefact fix (BT7 "
        "x7 and DGOSPEC, HS90, MINMAXBD now converge to a different local "
        "optimum than the re-derived `f*`; HS268 stops at `f* + 4e-12`), 9 "
        "are platform noise (the baseline ran on x86-64, the check on "
        "darwin: BIGGS3 tfip, DEVGLA1B x2, HS1 trip, HS36 tfip, HS38 pasls, "
        "HS68, HS113 tfip, PALMER6E asls-pcg-exact fail or stall at the base "
        "commit here as well) and 3 are attributable to the round (HS119 "
        "asls-pcg-exact / pasls, DEGENQP pasls)."
    ),
]
"""Summary bullets of the last fix round, rendered before section 2."""


def _render_attempts(cause: str, out_dir: Path) -> str:
    attempts = FIX_ATTEMPTS.get(cause)
    if not attempts:
        return ""
    parts = ["<div class='attempt'><strong>Fix attempts.</strong>"]
    for i, a in enumerate(attempts, 1):
        parts.append(f"<h4>Attempt {i}: {_md_code(a['title'])}</h4>")
        parts.append(f"<p><em>Change.</em> {a['change']}</p>")
        cmp_path = out_dir / a.get("compare_dir", "") / "compare.json"
        if a.get("compare_dir") and not cmp_path.exists():
            cmp_path = out_dir.parent / a["compare_dir"] / "compare.json"
        if a.get("compare_dir") and cmp_path.exists():
            s = json.loads(cmp_path.read_text()).get("per_cause", {}).get(cause)
            if s:
                parts.append(
                    f"<p><em>Measured.</em> {s['solved']}/{s['tasks']} affected tasks now solved "
                    f"(per config: {html.escape(str(s['per_config']))}); "
                    f"{s['same_cause']} still carry this diagnosis; new diagnoses of the rest: "
                    f"{html.escape(str({k: v for k, v in s['after'].items() if k not in ('solved', cause)}))}.</p>"
                )
        parts.append(f"<p><em>Outcome.</em> {a['outcome']}</p>")
    parts.append("</div>")
    return "".join(parts)


def _fmt(v: Any, spec: str = ".3g") -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return format(v, spec)
    return html.escape(str(v))


def _table(
    headers: list[str], rows: list[list[Any]], specs: dict[str, str] | None = None
) -> str:
    specs = specs or {}
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{_fmt(v, specs.get(h, '.3g'))}</td>"
            for h, v in zip(headers, row, strict=True)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _bar_chart_png(matrix: dict[str, dict[str, int]], title: str) -> str:
    """Stacked horizontal bar chart (configs x causes) as a base64 PNG."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    configs = list(matrix)
    causes = sorted(
        {c for d in matrix.values() for c in d},
        key=lambda c: -sum(matrix[k].get(c, 0) for k in configs),
    )
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(configs) + 2))
    left = np.zeros(len(configs))
    cmap = plt.get_cmap("tab20")
    for i, cause in enumerate(causes):
        vals = np.array([matrix[k].get(cause, 0) for k in configs])
        ax.barh(
            configs,
            vals,
            left=left,
            label=CAUSES.get(cause, ("", cause, ""))[1][:38],
            color=cmap(i % 20),
        )
        left += vals
    ax.set_xlabel("unsolved tasks")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _baseline_section(
    sqpdax_results: list[dict[str, Any]],
    baselines: dict[tuple[str, int], dict[str, dict[str, Any]]],
    classified: list[dict[str, Any]],
    configs: list[str],
) -> str:
    """Head-to-head comparison of every sqpdax configuration with the scipy rows."""
    base_configs = [
        k for k in BASELINE_CONFIGS if any(k in b for b in baselines.values())
    ]
    by_instance: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for r in sqpdax_results:
        by_instance[instance_key(r)][r["config"]] = r
    instances = sorted(by_instance)

    def solved(inst, cfg) -> bool:
        if cfg in BASELINE_CONFIGS:
            return bool((baselines.get(inst, {}).get(cfg) or {}).get("solved"))
        return bool((by_instance[inst].get(cfg) or {}).get("solved"))

    def any_sqpdax(inst) -> bool:
        return any(solved(inst, k) for k in configs)

    parts = ["<h2>1b. Comparison with the scipy baselines</h2>"]
    parts.append(
        "<p>The benchmark ran scipy's <code>SLSQP</code> and <code>trust-constr</code> on the "
        "same instances, from the same starting points, with the same success criterion. "
        "They are used below as the external reference: a task that a baseline solves but a "
        "sqpdax configuration does not is a genuine deficit; a task no solver solves is more "
        "likely a benchmark artefact (wrong reference value, mis-converted constraint, "
        "infeasible or unbounded instance) or a hard instance.</p>"
    )

    # (a) head-to-head per configuration
    headers = ["config", "solved"]
    for b in base_configs:
        s = BASELINE_SHORT[b]
        headers += [f"both ({s})", f"only sqpdax ({s})", f"only {s}", f"neither ({s})"]
    rows = []
    for k in configs + ["any sqpdax config"]:
        if k == "any sqpdax config":
            mine = {i for i in instances if any_sqpdax(i)}
        else:
            mine = {i for i in instances if solved(i, k)}
        row: list[Any] = [k, len(mine)]
        for b in base_configs:
            theirs = {i for i in instances if solved(i, b)}
            row += [
                len(mine & theirs),
                len(mine - theirs),
                len(theirs - mine),
                len(set(instances) - mine - theirs),
            ]
        rows.append(row)
    for b in base_configs:
        rows.append(
            [b, sum(solved(i, b) for i in instances)] + [""] * 4 * len(base_configs)
        )
    parts.append("<h3>Head-to-head</h3>")
    parts.append(_table(headers, rows))

    # (b) causes split by baseline outcome
    parts.append("<h3>Diagnosed causes by baseline outcome</h3>")
    parts.append(
        "<p>For each diagnosed cause: how many of its tasks are solved by each baseline. "
        "Causes whose tasks are mostly solved by scipy are solver deficits; causes whose "
        "tasks nobody solves point at the instance or its reference data.</p>"
    )
    cause_counts = collections.Counter(c["cause"] for c in classified)
    rows = []
    for cause, n in cause_counts.most_common():
        items = [c for c in classified if c["cause"] == cause]
        cells: list[Any] = [
            f"<a href='#{cause}'>{html.escape(CAUSES[cause][1])}</a>",
            CATEGORY_LABELS[CAUSES[cause][0]],
            n,
        ]
        for b in base_configs:
            cells.append(sum(1 for c in items if b in (c.get("baseline_solved") or [])))
        either = sum(1 for c in items if c.get("baseline_solved"))
        cells += [either, n - either, f"{either / n:.0%}"]
        rows.append(cells)
    parts.append(
        _table_raw(
            ["cause", "category", "tasks"]
            + [f"{BASELINE_SHORT[b]} solves" for b in base_configs]
            + ["either solves", "neither solves", "share solved by scipy"],
            rows,
        )
    )

    # (c) instances no sqpdax configuration solves
    gap = [
        i
        for i in instances
        if not any_sqpdax(i) and any(solved(i, b) for b in base_configs)
    ]
    nobody = [
        i
        for i in instances
        if not any_sqpdax(i) and not any(solved(i, b) for b in base_configs)
    ]
    wins = {
        b: [i for i in instances if any_sqpdax(i) and not solved(i, b)]
        for b in base_configs
    }
    both_fail_sqpdax_ok = [
        i
        for i in instances
        if any_sqpdax(i) and not any(solved(i, b) for b in base_configs)
    ]
    parts.append("<h3>Instance-level view</h3>")
    parts.append(
        f"<ul><li><b>{len(gap)}</b> instances are solved by a scipy baseline but by no sqpdax "
        f"configuration (<a href='#gap'>appendix 3a</a>).</li>"
        f"<li><b>{len(nobody)}</b> instances are solved by nobody (<a href='#nobody'>appendix 3b</a>).</li>"
        + "".join(
            f"<li><b>{len(wins[b])}</b> instances are solved by at least one sqpdax configuration but not by "
            f"<code>{b}</code>.</li>"
            for b in base_configs
        )
        + f"<li><b>{len(both_fail_sqpdax_ok)}</b> instances are solved by sqpdax and by neither baseline: "
        + html.escape(", ".join(sorted({p for p, _ in both_fail_sqpdax_ok})))
        + ".</li></ul>"
    )
    return "".join(parts)


def _delta_section(
    classified: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    sqpdax_results: list[dict[str, Any]],
) -> str:
    """Tasks fixed / regressed relative to an earlier ``classified.jsonl``."""
    prev = {task_key(c): c for c in previous}
    now = {task_key(c): c for c in classified}
    present = {task_key(r) for r in sqpdax_results}
    prev_keys = set(prev) & present  # ignore tasks that left the benchmark
    fixed = sorted(prev_keys - set(now))
    regressed = sorted(set(now) - set(prev))
    parts = ["<h2>1c. Changes since the previous diagnosis</h2>"]
    parts.append(
        f"<p>Compared with the previous classification (run "
        f"<code>{html.escape(str((previous or [{}])[0].get('run_id')))}</code>, "
        f"{len(prev_keys)} unsolved tasks): <b>{len(fixed)}</b> tasks are now solved and "
        f"<b>{len(regressed)}</b> tasks that were solved are now unsolved. Note that the two "
        "runs may come from different machines; a handful of flips near a tolerance are "
        "expected from floating-point differences alone.</p>"
    )
    rows = []
    fixed_by = collections.Counter(prev[k]["cause"] for k in fixed)
    for cause, n in fixed_by.most_common():
        per_cfg = collections.Counter(
            prev[k]["config"] for k in fixed if prev[k]["cause"] == cause
        )
        rows.append(
            [
                html.escape(CAUSES.get(cause, ("", cause, ""))[1]),
                n,
                ", ".join(f"{k}:{v}" for k, v in sorted(per_cfg.items())),
            ]
        )
    parts.append("<h3>Fixed, by previous diagnosis</h3>")
    parts.append(_table_raw(["previous cause", "tasks now solved", "per config"], rows))
    parts.append("<h3>Regressed</h3>")
    headers = [
        "problem",
        "config",
        "n",
        "status now",
        "diagnosis now",
        "f gap",
        "feas",
        "kkt_last",
        "scipy-slsqp",
        "scipy-trust-constr",
    ]
    rows = []
    for k in regressed:
        c = now[k]
        rows.append(
            [
                c["problem"],
                c["config"],
                c.get("n"),
                c.get("status"),
                CAUSES[c["cause"]][1],
                c.get("f_gap"),
                c.get("feas"),
                c.get("kkt_last"),
                _baseline_cell(c, "ref"),
                _baseline_cell(c, "tc"),
            ]
        )
    parts.append(_table(headers, rows))
    return "".join(parts)


def build_report(
    results: list[dict[str, Any]],
    diags: list[dict[str, Any]],
    out_dir: Path,
    previous: list[dict[str, Any]] | None = None,
) -> Path:
    feas_tol = float(results[0].get("feas_tol", 1e-6))
    f_tol = float(results[0].get("f_tol", 1e-4))
    by_key = {task_key(d): d for d in diags}
    sqpdax_results = [r for r in results if not is_baseline(r)]
    baseline_results = [r for r in results if is_baseline(r)]
    by_problem: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for r in sqpdax_results:
        by_problem[instance_key(r)].append(r)
    baselines: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for r in baseline_results:
        baselines[instance_key(r)][r["config"]] = r

    classified: list[dict[str, Any]] = []
    for r in failures(sqpdax_results):
        d = by_key.get(task_key(r))
        base_fields = _baseline_fields(baselines.get(instance_key(r), {}))
        peers = [p for p in by_problem[instance_key(r)] if p["config"] != r["config"]]
        cause, evidence = classify(
            r, {**(d or {}), **base_fields}, peers, feas_tol, f_tol
        )
        merged = {**r, **(d or {}), **base_fields}
        merged["cause"] = cause
        merged["category"] = CAUSES[cause][0]
        merged["evidence"] = evidence
        merged["rerun_status"] = (d or {}).get("status")
        merged["reproduced"] = (d is not None) and (d.get("status") == r["status"])
        classified.append(merged)

    configs = sorted({r["config"] for r in sqpdax_results})
    base_configs = [
        k for k in BASELINE_CONFIGS if any(r["config"] == k for r in baseline_results)
    ]
    n_instances = len(by_problem)
    n_total = len(sqpdax_results)
    n_fail = len(classified)
    n_rerun = sum(1 for c in classified if c.get("rerun"))
    n_repro = sum(1 for c in classified if c.get("reproduced"))

    # --- summary tables ------------------------------------------------------
    cause_counts = collections.Counter(c["cause"] for c in classified)
    cat_counts = collections.Counter(c["category"] for c in classified)
    matrix: dict[str, dict[str, int]] = {k: collections.Counter() for k in configs}
    for c in classified:
        matrix[c["config"]][c["cause"]] += 1
    status_matrix: dict[str, dict[str, int]] = {
        k: collections.Counter() for k in configs + base_configs
    }
    for c in classified:
        status_matrix[c["config"]][
            c["status"] if c["status"] != "successful" else "wrong objective"
        ] += 1
    for r in baseline_results:
        if not r.get("solved"):
            status_matrix[r["config"]][
                r["status"] if r["status"] != "successful" else "wrong objective"
            ] += 1

    parts: list[str] = []
    parts.append(
        f"<h1>sqpdax benchmark failure diagnosis</h1>"
        f"<p class='meta'>run <code>{html.escape(str(results[0].get('run_id')))}</code> · "
        f"{n_instances} instances × {len(configs)} sqpdax configurations = {n_total} tasks · "
        f"{n_fail} unsolved ({n_fail / n_total:.0%}) · "
        f"{n_rerun} re-run with diagnostics, {n_repro} reproduced the recorded status · "
        f"baselines: {', '.join(base_configs) or 'none'} · "
        f"tolerances feas ≤ {feas_tol:g}, |f−f*|/max(1,|f*|) ≤ {f_tol:g} · "
        f"generated {time.strftime('%Y-%m-%d %H:%M')}</p>"
    )

    parts.append("<h2>1. Overview</h2>")
    solved_by_cfg = collections.Counter(r["config"] for r in results if r.get("solved"))

    def _status_cells(k: str) -> list[Any]:
        known = sum(status_matrix[k].get(s, 0) for s in STATUS_COLUMNS)
        other = sum(status_matrix[k].values()) - known
        return [status_matrix[k].get(s, 0) for s in STATUS_COLUMNS] + [other]

    rows = [
        [
            k,
            solved_by_cfg[k],
            n_instances - solved_by_cfg[k],
            f"{solved_by_cfg[k] / n_instances:.0%}",
        ]
        + _status_cells(k)
        for k in configs + base_configs
    ]
    parts.append(
        _table(
            ["config", "solved", "unsolved", "solve rate", *STATUS_COLUMNS, "other"],
            rows,
        )
    )
    parts.append(
        "<p class='meta'>Statuses in <em>other</em> are the scipy-specific ones "
        "(<code>positive_directional_derivative_for_linesearch</code>, "
        "<code>singular_matrix_*</code>, <code>inequality_constraints_incompatible</code>, "
        "<code>constraint_violation_exceeds_gtol</code>, <code>runtime_error</code>).</p>"
    )

    # --- comparison with the scipy baselines ----------------------------------
    if base_configs:
        parts.append(_baseline_section(sqpdax_results, baselines, classified, configs))
    if previous:
        parts.append(_delta_section(classified, previous, sqpdax_results))
    parts.append(
        f"<figure><img src='data:image/png;base64,{_bar_chart_png(matrix, 'Unsolved tasks per configuration, by diagnosed cause')}'/>"
        "<figcaption>Each bar is one configuration; segments are the diagnosed causes of its unsolved tasks.</figcaption></figure>"
    )
    rows = []
    for cat, label in CATEGORY_LABELS.items():
        if cat_counts[cat]:
            rows.append([label, cat_counts[cat], f"{cat_counts[cat] / n_fail:.0%}"])
    parts.append("<h3>Unsolved tasks by category</h3>")
    parts.append(_table(["category", "tasks", "share"], rows))
    rows = []
    for cause, n in cause_counts.most_common():
        cat, title, _ = CAUSES[cause]
        cfgs = ", ".join(f"{k}:{matrix[k][cause]}" for k in configs if matrix[k][cause])
        rows.append(
            [
                f"<a href='#{cause}'>{html.escape(title)}</a>",
                CATEGORY_LABELS[cat],
                n,
                cfgs,
            ]
        )
    parts.append("<h3>Unsolved tasks by cause</h3>")
    parts.append(_table_raw(["cause", "category", "tasks", "per config"], rows))

    # --- prioritised action list ---------------------------------------------
    parts.append("<h2>Priorities</h2>")
    parts.append(
        "<p>Actions ordered by the number of unsolved tasks they address (a task is counted once, "
        "under the first action that covers its cause). Details and evidence per cause follow in "
        "section 2.</p>"
    )
    seen: set[str] = set()
    rows = []
    for title, causes_, where in PRIORITIES:
        fresh = [c for c in causes_ if c not in seen]
        seen.update(fresh)
        n = sum(cause_counts[c] for c in fresh)
        if n == 0:
            continue
        links = ", ".join(
            f"<a href='#{c}'>{html.escape(CAUSES[c][1])}</a> ({cause_counts[c]})"
            for c in fresh
            if cause_counts[c]
        )
        rows.append([_md_code(title), n, f"{n / n_fail:.0%}", _md_code(where), links])
    parts.append(
        _table_raw(
            ["action", "tasks", "share of unsolved", "where", "causes addressed"], rows
        )
    )

    # --- work log of the fix round -------------------------------------------
    if WORK_LOG:
        parts.append("<h2>Work log of the last fix round</h2>")
        parts.append(
            "<p>The three actions of the previous report were worked through cause by "
            "cause; each per-cause section below carries the detailed attempt entries "
            "(change, measured effect on the affected tasks, outcome, keep / revert). "
            "This is the summary and the regression check.</p>"
        )
        parts.append(
            "<ul>" + "".join(f"<li>{item}</li>" for item in WORK_LOG) + "</ul>"
        )

    # --- per cause sections --------------------------------------------------
    parts.append("<h2>2. Causes, evidence and recommendations</h2>")
    for cause, n in cause_counts.most_common():
        cat, title, desc = CAUSES[cause]
        items = [c for c in classified if c["cause"] == cause]
        parts.append(
            f"<h3 id='{cause}'>{html.escape(title)} <span class='badge {cat}'>{CATEGORY_LABELS[cat]}</span> <span class='count'>{n} tasks</span></h3>"
        )
        parts.append(f"<p>{_md_code(desc)}</p>")
        per_cfg = collections.Counter(c["config"] for c in items)
        per_problem = collections.Counter(c["problem"] for c in items)
        parts.append(
            "<p class='meta'>per config: "
            + ", ".join(f"{k} {per_cfg[k]}" for k in configs if per_cfg[k])
            + f" · {len(per_problem)} distinct problems"
            + (f" · tiers: {dict(collections.Counter(c['tier'] for c in items))}")
            + "</p>"
        )
        warn = collections.Counter()
        for c in items:
            for k, v in (c.get("warnings") or {}).items():
                warn[k] += 1
        if warn:
            parts.append(
                "<p class='meta'>most common warnings (tasks emitting them): "
                + ", ".join(
                    f"<code>{html.escape(k)}</code> {v}" for k, v in warn.most_common(6)
                )
                + "</p>"
            )
        if cause in RECOMMENDATIONS:
            parts.append(
                f"<div class='reco'><strong>Recommendation.</strong> {RECOMMENDATIONS[cause]}</div>"
            )
        parts.append(_render_attempts(cause, out_dir))
        headers, getter = CAUSE_COLUMNS.get(cause, CAUSE_COLUMNS["default"])
        rows = [
            getter(c) for c in sorted(items, key=lambda c: (c["config"], c["problem"]))
        ]
        parts.append(
            "<details><summary>Affected tasks</summary>"
            + _table(headers, rows)
            + "</details>"
        )

    # --- per problem appendix --------------------------------------------------
    parts.append(
        "<h2>3. Appendix: instances unsolved by every sqpdax configuration</h2>"
    )
    gap_rows, nobody_rows = [], []
    for inst, group in sorted(by_problem.items()):
        if any(p.get("solved") for p in group):
            continue
        prob, y0 = inst
        cs = [c for c in classified if instance_key(c) == inst]
        causes = collections.Counter(c["cause"] for c in cs)
        base_fields = _baseline_fields(baselines.get(inst, {}))
        row = [
            prob,
            group[0]["n"],
            group[0]["meq"],
            group[0]["mineq"],
            group[0].get("fstar"),
            ", ".join(f"{CAUSES[k][1][:30]} ({v})" for k, v in causes.most_common()),
            _baseline_cell(base_fields, "ref"),
            _baseline_cell(base_fields, "tc"),
            (
                f"{_g(base_fields, 'ref_f'):.6g}"
                if math.isfinite(_g(base_fields, "ref_f"))
                else ""
            ),
            (
                f"{_g(base_fields, 'tc_f'):.6g}"
                if math.isfinite(_g(base_fields, "tc_f"))
                else ""
            ),
        ]
        (gap_rows if base_fields.get("baseline_solved") else nobody_rows).append(row)
    appendix_headers = [
        "problem",
        "n",
        "meq",
        "mineq",
        "f*",
        "sqpdax causes",
        "scipy-slsqp",
        "scipy-trust-constr",
        "f (slsqp)",
        "f (trust-constr)",
    ]
    parts.append(
        f"<h3 id='gap'>3a. Solved by a scipy baseline ({len(gap_rows)})</h3>"
        "<p>The hard core of the deficit: every sqpdax configuration fails where scipy does "
        "not. These should be the first instances to study.</p>"
    )
    parts.append(_table(appendix_headers, gap_rows, {"f*": ".6g"}))
    parts.append(
        f"<h3 id='nobody'>3b. Solved by nobody ({len(nobody_rows)})</h3>"
        "<p>Instances where neither scipy baseline meets the success criterion either. When the "
        "scipy objectives agree with each other but not with <code>f*</code> the reference "
        "value is suspect; when they diverge to <code>-1e30</code> the instance is unbounded as "
        "converted.</p>"
    )
    parts.append(_table(appendix_headers, nobody_rows, {"f*": ".6g"}))

    body = "\n".join(parts)
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>sqpdax benchmark failure diagnosis</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.45}}
table{{border-collapse:collapse;font-size:12px;margin:.5rem 0 1rem}} th,td{{border:1px solid #ddd;padding:3px 6px;text-align:left;vertical-align:top}}
th{{background:#f3f3f3;position:sticky;top:0}} tr:nth-child(even){{background:#fafafa}}
details{{margin:.5rem 0 1.5rem}} summary{{cursor:pointer;color:#06c}}
.badge{{font-size:11px;padding:2px 6px;border-radius:4px;color:#fff;vertical-align:middle}}
.bug{{background:#c0392b}} .algorithmic{{background:#2980b9}} .parameter{{background:#f39c12}} .benchmark{{background:#7f8c8d}} .performance{{background:#8e44ad}}
.count{{font-size:12px;color:#666}} .meta{{color:#555;font-size:13px}}
.reco{{background:#eef7ee;border-left:4px solid #2e7d32;padding:.6rem .9rem;margin:.6rem 0}}
.attempt{{background:#eef3fb;border-left:4px solid #1f5fa8;padding:.6rem .9rem;margin:.6rem 0}} .attempt h4{{margin:.6rem 0 .2rem}}
figure{{margin:1rem 0}} figcaption{{font-size:12px;color:#555}} code{{background:#f4f4f4;padding:1px 3px}}
</style></head><body>{body}</body></html>"""
    out = out_dir / REPORT_FILE
    out.write_text(doc)
    (out_dir / "classified.jsonl").write_text(
        "\n".join(
            json.dumps(
                _jsonable(
                    {k: v for k, v in c.items() if k not in ("warning_examples",)}
                )
            )
            for c in classified
        )
        + "\n"
    )
    return out


STATUS_COLUMNS = [
    "wrong objective",
    "max_steps_reached",
    "secant_recovery_failure",
    "infeasible",
    "infeasible_stationary",
    "merit_stagnation",
    "stationarity_stall",
    "multiplier_solve_failure",
    "infeasible_stationary_point",
    "nonfinite",
    "timeout",
]


def _table_raw(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{v if isinstance(v, str) else _fmt(v)}</td>" for v in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _base(c: dict[str, Any]) -> list[Any]:
    return [
        c["problem"],
        c["config"],
        c["tier"],
        c["n"],
        c["meq"],
        c["mineq"],
        c.get("steps"),
        _baseline_cell(c, "ref"),
        _baseline_cell(c, "tc"),
    ]


BASE_HEADERS = [
    "problem",
    "config",
    "tier",
    "n",
    "meq",
    "mineq",
    "steps",
    "scipy-slsqp",
    "scipy-trust-constr",
]

CAUSE_COLUMNS: dict[str, tuple[list[str], Any]] = {
    "default": (
        [
            *BASE_HEADERS,
            "status",
            "f",
            "f*",
            "f gap",
            "feas",
            "kkt_last",
            "stat_ls",
            "scipy f",
            "evidence",
        ],
        lambda c: [
            *_base(c),
            c["status"],
            c.get("objective"),
            c.get("fstar"),
            c.get("f_gap"),
            c.get("feas"),
            c.get("kkt_last"),
            c.get("stat_ls"),
            c.get("ref_f"),
            c.get("evidence"),
        ],
    ),
    "false_convergence": (
        [
            *BASE_HEADERS,
            "f",
            "f*",
            "f gap",
            "feas",
            "stat_ls",
            "solver kkt",
            "kkt_ratio",
            "alpha_last",
            "dnorm_last",
            "scipy f",
            "warnings",
        ],
        lambda c: [
            *_base(c),
            c.get("objective"),
            c.get("fstar"),
            c.get("f_gap"),
            c.get("feas"),
            c.get("stat_ls"),
            c.get("stats_final_lagrangian_grad_norm"),
            c.get("stats_kkt_ratio"),
            c.get("alpha_last"),
            c.get("dnorm_last"),
            c.get("ref_f"),
            ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
        ],
    ),
    "local_minimum": (
        [
            *BASE_HEADERS,
            "f",
            "f*",
            "f gap",
            "stat_ls",
            "configs at f*",
            "scipy f",
            "scipy gap",
        ],
        lambda c: [
            *_base(c),
            c.get("objective"),
            c.get("fstar"),
            c.get("f_gap"),
            c.get("stat_ls"),
            c.get("evidence", "").split("reference reached by ")[-1],
            c.get("ref_f"),
            c.get("ref_f_gap"),
        ],
    ),
    "reference_mismatch": (
        [
            *BASE_HEADERS,
            "f",
            "f*",
            "f gap",
            "feas",
            "stat_ls",
            "f(x*)",
            "feas(x*)",
            "scipy f",
            "scipy feas",
        ],
        lambda c: [
            *_base(c),
            c.get("objective"),
            c.get("fstar"),
            c.get("f_gap"),
            c.get("feas"),
            c.get("stat_ls"),
            c.get("xstar_f"),
            c.get("xstar_feas"),
            c.get("ref_f"),
            c.get("ref_feas"),
        ],
    ),
    "secant_recovery_premature": (
        [
            *BASE_HEADERS,
            "kkt_first",
            "kkt_min",
            "kkt_last",
            "tail ratio",
            "accept rate",
            "rejected tail",
            "stage",
            "pairs",
            "f gap",
            "warnings",
        ],
        lambda c: [
            *_base(c),
            c.get("kkt_first"),
            c.get("kkt_min"),
            c.get("kkt_last"),
            c.get("kkt_tail_ratio"),
            c.get("accept_rate"),
            c.get("rejected_tail"),
            c.get("stats_secant_recovery_stage"),
            c.get("pairs_last"),
            c.get("f_gap"),
            ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
        ],
    ),
    "max_steps": (
        [
            *BASE_HEADERS,
            "kkt_first",
            "kkt_min",
            "kkt_last",
            "tail ratio",
            "tail log-std",
            "accept rate",
            "alpha_last",
            "dnorm_last",
            "feas_last",
            "f gap",
            "warnings",
        ],
        lambda c: [
            *_base(c),
            c.get("kkt_first"),
            c.get("kkt_min"),
            c.get("kkt_last"),
            c.get("kkt_tail_ratio"),
            c.get("kkt_tail_log_std"),
            c.get("accept_rate"),
            c.get("alpha_last"),
            c.get("dnorm_last"),
            c.get("feas_last"),
            c.get("f_gap"),
            ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
        ],
    ),
    "infeasible": (
        [
            *BASE_HEADERS,
            "status",
            "feas",
            "feas_min",
            "kkt_last",
            "penalty",
            "feasible peers / scipy",
            "scipy feas",
            "warnings",
        ],
        lambda c: [
            *_base(c),
            c["status"],
            c.get("feas"),
            c.get("feas_min"),
            c.get("kkt_last"),
            c.get("penalty_last"),
            c.get("evidence"),
            c.get("ref_feas"),
            ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
        ],
    ),
}
CAUSE_COLUMNS["stationarity_scale_f"] = CAUSE_COLUMNS["false_convergence"]
CAUSE_COLUMNS["qp_inactive_multipliers"] = CAUSE_COLUMNS["false_convergence"]
CAUSE_COLUMNS["secant_recovery_stuck"] = CAUSE_COLUMNS["secant_recovery_premature"]
CAUSE_COLUMNS["local_minimum_consensus"] = CAUSE_COLUMNS["reference_mismatch"]
for _k in ("slow_progress", "step_collapse", "cycling"):
    CAUSE_COLUMNS[_k] = CAUSE_COLUMNS["max_steps"]
for _k in (
    "restoration_failure",
    "near_feasible",
    "problem_infeasible",
    "ip_infeasible_stationary",
    "false_infeasible_certificate",
):
    CAUSE_COLUMNS[_k] = CAUSE_COLUMNS["infeasible"]
CAUSE_COLUMNS["kkt_solver_failure"] = (
    [
        *BASE_HEADERS,
        "status",
        "feas",
        "kkt_last",
        "qp results",
        "sub status",
        "first non-finite",
        "warnings",
    ],
    lambda c: [
        *_base(c),
        c["status"],
        c.get("feas"),
        c.get("kkt_last"),
        ", ".join(f"{k}×{v}" for k, v in (c.get("qp_result_counts") or {}).items()),
        ", ".join(f"{k}×{v}" for k, v in (c.get("sub_status_counts") or {}).items()),
        c.get("first_nonfinite_step"),
        ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
    ],
)
CAUSE_COLUMNS["ip_bound_slack_residual"] = (
    [
        *BASE_HEADERS,
        "nbounds",
        "status",
        "f gap",
        "feas_lb",
        "feas_ub",
        "slack_resid",
        "mu_last",
        "mu updates",
        "compl",
        "kkt_min",
        "kkt_last",
        "radius_last",
        "warnings",
    ],
    lambda c: [
        *_base(c),
        c.get("nbounds"),
        c["status"],
        c.get("f_gap"),
        c.get("feas_lb"),
        c.get("feas_ub"),
        c.get("slack_resid"),
        c.get("mu_last"),
        c.get("mu_updates"),
        c.get("compl"),
        c.get("kkt_min"),
        c.get("kkt_last"),
        c.get("radius_last"),
        ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
    ],
)
CAUSE_COLUMNS["ip_barrier_stall"] = CAUSE_COLUMNS["ip_bound_slack_residual"]
CAUSE_COLUMNS["ip_bound_violation"] = CAUSE_COLUMNS["ip_bound_slack_residual"]
CAUSE_COLUMNS["termination_missed"] = CAUSE_COLUMNS["secant_recovery_premature"]
CAUSE_COLUMNS["multiplier_failure"] = (
    [
        *BASE_HEADERS,
        "status",
        "feas",
        "f gap",
        "kkt_last",
        "pi_f",
        "v_max",
        "y streak",
        "iteration types",
        "warnings",
    ],
    lambda c: [
        *_base(c),
        c["status"],
        c.get("feas"),
        c.get("f_gap"),
        c.get("kkt_last"),
        c.get("pi_f_last"),
        c.get("v_max_last"),
        c.get("y_streak_last"),
        ", ".join(f"{k}×{v}" for k, v in (c.get("type_counts") or {}).items()),
        ", ".join(f"{k}×{v}" for k, v in (c.get("warnings") or {}).items()),
    ],
)
CAUSE_COLUMNS["unbounded"] = (
    [
        *BASE_HEADERS,
        "status",
        "f first",
        "f last",
        "f*",
        "feas_last",
        "scipy f",
        "scipy status",
    ],
    lambda c: [
        *_base(c),
        c["status"],
        c.get("f_first"),
        c.get("f_last"),
        c.get("fstar"),
        c.get("feas_last"),
        c.get("ref_f"),
        c.get("ref_status"),
    ],
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from benchmarks.worker import run_tasks

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = read_jsonl(Path(args.results))
    # Only sqpdax runs can be reproduced with trajectory logging; the scipy
    # baselines enter the report through their benchmark rows.
    todo = [r for r in failures(results) if r.get("backend", "sqpdax") == "sqpdax"]
    if args.configs:
        todo = [r for r in todo if r["config"] in args.configs]
    if args.problems:
        todo = [r for r in todo if r["problem"] in args.problems]
    if args.statuses:
        todo = [r for r in todo if r["status"] in args.statuses]
    if args.causes:
        classified = read_jsonl(Path(args.classified))
        keys = {task_key(c) for c in classified if c.get("cause") in args.causes}
        todo = [r for r in todo if task_key(r) in keys]
    if args.max_n is not None:
        todo = [r for r in todo if r["n"] <= args.max_n]
    todo = [r for r in todo if r["status"] != "timeout"]  # nothing to re-run
    diag_path = out_dir / DIAG_FILE
    done: set[str] = set()
    if diag_path.exists() and not args.fresh:
        done = {task_key(d) for d in read_jsonl(diag_path)}
    todo = [r for r in todo if task_key(r) not in done]
    # small problems first so partial results are useful early
    todo.sort(key=lambda r: (r["n"], r["problem"], r["config"]))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} tasks to diagnose ({len(done)} already done) -> {diag_path}")
    if args.dry_run:
        for r in todo:
            print(f"  {r['problem']:<12} {r['config']:<16} n={r['n']:<5} {r['status']}")
        return 0
    specs = [
        DiagSpec(
            problem=r["problem"],
            y0_iD=int(r.get("y0_iD", 0)),
            config=r["config"],
            max_steps=int(r.get("max_steps", 500)),
            timeout_s=args.timeout,
            ref_max_n=args.ref_max_n,
            ref_time_s=args.ref_time,
            out_dir=str(out_dir),
        )
        for r in todo
    ]
    t0 = time.perf_counter()
    count = 0
    with open(diag_path, "a") as fh:

        def on_result(row: dict[str, Any]) -> None:
            nonlocal count
            count += 1
            fh.write(json.dumps(_jsonable(row)) + "\n")
            fh.flush()
            print(
                f"[{count}/{len(specs)} {time.perf_counter() - t0:6.0f}s] {row['problem']:<12} "
                f"{row['config']:<16} {row.get('status', '?'):<26} "
                f"steps={row.get('steps', '')!s:<4} kkt_last={_fmt(row.get('kkt_last'), '.1e'):<8} "
                f"stat_ls={_fmt(row.get('stat_ls'), '.1e'):<8} ref_gap={_fmt(row.get('ref_f_gap'), '.1e')}",
                flush=True,
            )

        run_tasks(specs, jobs=args.jobs, on_result=on_result)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    results = read_jsonl(Path(args.results))
    diag_path = out_dir / DIAG_FILE
    diags = read_jsonl(diag_path) if diag_path.exists() else []
    if args.recommendations:
        RECOMMENDATIONS.update(json.loads(Path(args.recommendations).read_text()))
    previous = read_jsonl(Path(args.previous)) if args.previous else None
    path = build_report(results, diags, out_dir, previous=previous)
    print(f"report written to {path}")
    return 0


def _solved(row: dict[str, Any], feas_tol: float, f_tol: float) -> bool:
    """Benchmark success criterion applied to a diagnostic row."""
    if row.get("status") != "successful":
        return False
    if not (_g(row, "feas", math.inf) <= feas_tol):
        return False
    if row.get("has_fstar"):
        return _g(row, "f_gap", math.inf) <= f_tol
    return True


def compare_runs(
    results: list[dict[str, Any]],
    base_classified: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    causes: list[str] | None,
) -> dict[str, Any]:
    """Before/after comparison of the tasks re-run after a fix attempt.

    Returns a summary with, per original cause, the number of tasks that are
    now solved, still failing with the same diagnosis, or failing with a new
    diagnosis, plus the per-task rows.
    """
    feas_tol = float(results[0].get("feas_tol", 1e-6))
    f_tol = float(results[0].get("f_tol", 1e-4))
    base = {task_key(c): c for c in base_classified}
    by_problem: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    baselines: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for r in results:
        if is_baseline(r):
            baselines[instance_key(r)][r["config"]] = r
        else:
            by_problem[instance_key(r)].append(r)
    rows = []
    for d in fixed:
        key = task_key(d)
        b = base.get(key)
        if b is None or (causes and b.get("cause") not in causes):
            continue
        peers = [p for p in by_problem[instance_key(d)] if p["config"] != d["config"]]
        solved = _solved(d, feas_tol, f_tol)
        d_full = {**d, **_baseline_fields(baselines.get(instance_key(d), {}))}
        new_cause = (
            "solved"
            if solved
            else classify({**b, **d_full}, d_full, peers, feas_tol, f_tol)[0]
        )
        rows.append(
            {
                "problem": d["problem"],
                "config": d["config"],
                "n": d.get("n"),
                "cause_before": b.get("cause"),
                "status_before": b.get("status"),
                "f_gap_before": b.get("f_gap"),
                "status_after": d.get("status"),
                "cause_after": new_cause,
                "solved": solved,
                "steps_after": d.get("steps"),
                "f_gap_after": d.get("f_gap"),
                "feas_after": d.get("feas"),
                "stat_ls_after": d.get("stat_ls"),
                "kkt_last_after": d.get("kkt_last"),
            }
        )
    summary: dict[str, Any] = {"n": len(rows), "per_cause": {}, "rows": rows}
    for cause in sorted({r["cause_before"] for r in rows}):
        sel = [r for r in rows if r["cause_before"] == cause]
        after = collections.Counter(r["cause_after"] for r in sel)
        per_cfg = collections.Counter(r["config"] for r in sel)
        solved_cfg = collections.Counter(r["config"] for r in sel if r["solved"])
        summary["per_cause"][cause] = {
            "tasks": len(sel),
            "solved": sum(r["solved"] for r in sel),
            "same_cause": after.get(cause, 0),
            "after": dict(after.most_common()),
            "per_config": {k: f"{solved_cfg[k]}/{per_cfg[k]}" for k in sorted(per_cfg)},
        }
    return summary


def cmd_compare(args: argparse.Namespace) -> int:
    results = read_jsonl(Path(args.results))
    base = read_jsonl(Path(args.base) / "classified.jsonl")
    fixed_path = Path(args.fixed) / DIAG_FILE
    fixed = read_jsonl(fixed_path) if fixed_path.exists() else []
    summary = compare_runs(results, base, fixed, args.causes)
    out = Path(args.fixed) / "compare.json"
    out.write_text(json.dumps(_jsonable(summary), indent=1))
    for cause, s in summary["per_cause"].items():
        print(
            f"{cause}: {s['solved']}/{s['tasks']} solved; same cause {s['same_cause']}; "
            f"after={s['after']}; per config {s['per_config']}"
        )
    print(f"written {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="re-run unsolved tasks with diagnostics")
    run.add_argument("results", help="results.jsonl of the benchmark run")
    run.add_argument("--out", default="diagnostics")
    run.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--configs", nargs="*")
    run.add_argument("--problems", nargs="*")
    run.add_argument("--statuses", nargs="*")
    run.add_argument("--max-n", type=int, default=None)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--ref-max-n", type=int, default=300)
    run.add_argument("--ref-time", type=float, default=60.0)
    run.add_argument("--fresh", action="store_true", help="ignore existing diag.jsonl")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--causes",
        nargs="*",
        help="only tasks diagnosed with these cause ids in --classified",
    )
    run.add_argument(
        "--classified",
        default="diagnostics/classified.jsonl",
        help="classified.jsonl used to resolve --causes",
    )
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser(
        "compare", help="before/after comparison of a fix-attempt re-run"
    )
    cmp_.add_argument(
        "base", help="diagnostics directory of the baseline (classified.jsonl)"
    )
    cmp_.add_argument("fixed", help="diagnostics directory of the re-run after the fix")
    cmp_.add_argument("--results", required=True)
    cmp_.add_argument("--causes", nargs="*")
    cmp_.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report", help="classify failures and render report.html")
    rep.add_argument("out", help="diagnostics directory produced by `run`")
    rep.add_argument("--results", required=True)
    rep.add_argument(
        "--recommendations", help="JSON file {cause_id: html} merged into the report"
    )
    rep.add_argument(
        "--previous",
        help="classified.jsonl of an earlier diagnosis; adds the fixed/regressed section",
    )
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
