"""Command-line interface: ``python -m benchmarks <command>``.

Commands
--------
catalog
    Regenerate ``benchmarks/catalog.csv`` from the installed sif2jax.
run
    Benchmark a selection of catalogue instances against solver configs,
    appending rows to ``<out>/results.jsonl`` as they complete.
merge
    Concatenate shard run directories into one run directory.
report
    Export the static per-run report (``marimo export html``).
publish
    Copy a run directory into a results tree, update ``index.json`` and
    ``runs.html``, and optionally rebuild the wasm dashboard.
render
    Re-export every indexed run's report plus the dashboard (used when only
    the notebooks changed).
dashboard
    Export the interactive dashboard (``marimo export html-wasm``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPORT_NOTEBOOK = HERE / "report.py"
DASHBOARD_NOTEBOOK = HERE / "dashboard.py"


# --------------------------------------------------------------------------- helpers
def _parse_chunk(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    index, count = text.split("/")
    return int(index), int(count)


def _csv_list(text: str | None) -> list[str] | None:
    if text is None or text == "":
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _configure_environment(device: str | None, jobs: int) -> None:
    """Set JAX/XLA environment variables before any worker is spawned."""
    if device:
        os.environ["JAX_PLATFORMS"] = device
    if jobs > 1:
        # Concurrent workers must not oversubscribe the CPU.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_cpu_multi_thread_eigen" not in flags:
            os.environ["XLA_FLAGS"] = (
                flags + " --xla_cpu_multi_thread_eigen=false"
            ).strip()
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "   -   "
    if seconds < 1e-3:
        return f"{seconds * 1e6:6.1f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:6.1f}ms"
    return f"{seconds:6.2f}s "


# -------------------------------------------------------------------------- commands
def cmd_catalog(args: argparse.Namespace) -> int:
    from .catalog import CATALOG_PATH, COLLECTIONS, build_catalog

    collections = _csv_list(args.collections) or list(COLLECTIONS)
    frame = build_catalog(collections)
    out = Path(args.out) if args.out else CATALOG_PATH
    frame.to_csv(out, index=False)
    print(f"wrote {len(frame)} instances to {out}")
    print(frame.groupby(["collection", "tier"]).size().to_string())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .catalog import DEFAULT_COLLECTIONS, load_catalog, select
    from .configs import get_configs
    from .results import (
        RESULTS_FILE,
        append_row,
        new_run_id,
        run_metadata,
        write_run_metadata,
    )
    from .worker import TaskSpec, run_tasks

    _configure_environment(args.device, args.jobs)

    catalog = load_catalog(Path(args.catalog)) if args.catalog else load_catalog()
    names = _csv_list(args.problems)
    if args.all_collections:
        collections = None
    else:
        collections = _csv_list(args.collections) or DEFAULT_COLLECTIONS
    selection = select(
        catalog,
        # Explicit problem names bypass the tier / collection filters.
        tier=None if names else args.tier,
        collections=None if names else collections,
        names=names,
        max_n=args.max_n,
        chunk=_parse_chunk(args.chunk),
    )
    if args.limit:
        selection = selection.head(args.limit)
    configs = get_configs(_csv_list(args.configs))

    specs: list[TaskSpec] = []
    base_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for rec in selection.to_dict("records"):
        for cfg in configs:
            spec = TaskSpec(
                problem=str(rec["name"]),
                y0_iD=int(rec["y0_iD"]),
                config=cfg.name,
                max_steps=args.max_steps,
                repeats=args.repeats,
                repeat_budget_s=args.repeat_budget,
                max_repeats=args.max_repeats,
                slow_floor=args.slow_floor,
                timeout_s=args.timeout,
            )
            specs.append(spec)
            base = {k: v for k, v in rec.items() if k != "name"}
            base_rows[(spec.problem, spec.y0_iD, spec.config)] = base

    run_id = args.run_id or new_run_id()
    out_dir = Path(args.out) / run_id if not args.flat else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / RESULTS_FILE

    meta = run_metadata(
        run_id=run_id,
        tier=args.tier,
        collections=sorted(selection["collection"].unique().tolist()),
        configs=[c.name for c in configs],
        n_instances=int(len(selection)),
        n_tasks=len(specs),
        max_steps=args.max_steps,
        timeout_s=args.timeout,
        repeats=args.repeats,
        jobs=args.jobs,
        chunk=args.chunk,
        device_requested=args.device,
        status="running",
    )
    write_run_metadata(out_dir, meta)

    print(
        f"run {run_id}: {len(selection)} instances x {len(configs)} configs = {len(specs)} tasks -> {results_path}"
    )
    if args.dry_run:
        for spec in specs:
            print(f"  {spec.problem:12s} y0={spec.y0_iD} {spec.config}")
        return 0

    start = time.perf_counter()
    done = 0
    counts: dict[str, int] = {}

    def on_result(row: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        base = base_rows.get((row["problem"], row["y0_iD"], row["config"]), {})
        full = {**base, **row, "run_id": run_id}
        append_row(results_path, full)
        status = str(full.get("status"))
        counts[status] = counts.get(status, 0) + 1
        steps = full.get("steps")
        print(
            f"[{done:4d}/{len(specs)}] {full['problem']:12s} {full['config']:15s} "
            f"{status:24s} steps={steps if steps is not None else '-':>4} "
            f"t={_fmt_time(full.get('time_median_s'))} "
            f"compile={_fmt_time(full.get('compile_s'))} "
            f"feas={full.get('feas', float('nan')):.1e} stat={full.get('stat_inf', float('nan')):.1e}",
            flush=True,
        )

    run_tasks(specs, jobs=args.jobs, on_result=on_result)

    meta.update(
        status="finished", wall_s=time.perf_counter() - start, status_counts=counts
    )
    write_run_metadata(out_dir, meta)
    print(f"finished {done} tasks in {time.perf_counter() - start:.0f}s: {counts}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from .results import RESULTS_FILE, RUN_FILE, merge_jsonl

    shards = [Path(p) for p in args.shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = merge_jsonl([s / RESULTS_FILE for s in shards], out / RESULTS_FILE)
    metas = [
        json.loads((s / RUN_FILE).read_text())
        for s in shards
        if (s / RUN_FILE).exists()
    ]
    meta = dict(metas[0]) if metas else {}
    meta.update(
        {
            "run_id": args.run_id or meta.get("run_id") or out.name,
            "shards": [s.name for s in shards],
            "n_tasks": sum(int(m.get("n_tasks", 0)) for m in metas),
            "n_instances": sum(int(m.get("n_instances", 0)) for m in metas),
            "configs": sorted({c for m in metas for c in m.get("configs", [])}),
            "status": "merged",
        }
    )
    counts: dict[str, int] = {}
    for m in metas:
        for k, v in (m.get("status_counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    meta["status_counts"] = counts
    (out / RUN_FILE).write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"merged {count} rows from {len(shards)} shards into {out}")
    return 0


def _marimo(*argv: str) -> int:
    cmd = [sys.executable, "-m", "marimo", *argv]
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out) if args.out else run_dir / "report.html"
    return _marimo(
        "export",
        "html",
        str(REPORT_NOTEBOOK),
        "-o",
        str(out),
        "--no-include-code",
        "-f",
        "--",
        "--run-dir",
        str(run_dir),
    )


def cmd_dashboard(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    code = _marimo(
        "export",
        "html-wasm",
        str(DASHBOARD_NOTEBOOK),
        "-o",
        str(out),
        "--mode",
        "run",
        "--no-show-code",
        "-f",
    )
    if code == 0:
        # The wasm notebook cannot import the package; it fetches this helper
        # module from the site instead (see dashboard.py).
        public = out / "public"
        public.mkdir(exist_ok=True)
        shutil.copy2(HERE / "analysis.py", public / "analysis.py")
        (out / "CLAUDE.md").unlink(
            missing_ok=True
        )  # marimo boilerplate, not part of the site
        if results_dir.parent != out:
            print(
                f"warning: dashboard expects results at {out / 'results'}, got {results_dir}"
            )
    return code


def _runs_html(index: list[dict[str, Any]]) -> str:
    rows = []
    for entry in sorted(index, key=lambda e: e["run_id"], reverse=True):
        counts = entry.get("status_counts") or {}
        ok = counts.get("successful", 0)
        total = sum(counts.values()) or entry.get("n_tasks", 0)
        rows.append(
            "<tr>"
            f'<td><a href="results/{entry["run_id"]}/report.html">{entry["run_id"]}</a></td>'
            f"<td>{entry.get('timestamp', '')}</td>"
            f"<td><code>{str(entry.get('sha', ''))[:7]}</code></td>"
            f"<td>{entry.get('tag') or ''}</td>"
            f"<td>{entry.get('trigger', '')}</td>"
            f"<td>{entry.get('device', '')}</td>"
            f"<td>{entry.get('tier', '')}</td>"
            f"<td>{ok}/{total}</td>"
            f'<td><a href="results/{entry["run_id"]}/results.jsonl">jsonl</a></td>'
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>sqpdax benchmark runs</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}table{border-collapse:collapse}"
        "td,th{padding:.3rem .8rem;border-bottom:1px solid #ddd;text-align:left}</style></head><body>"
        "<h1>sqpdax benchmark runs</h1>"
        "<p><a href='index.html'>Interactive dashboard</a></p>"
        "<table><thead><tr><th>run</th><th>timestamp (UTC)</th><th>sha</th><th>tag</th>"
        "<th>trigger</th><th>device</th><th>tier</th><th>successful</th><th>raw</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>\n"
    )


def cmd_publish(args: argparse.Namespace) -> int:
    from .analysis import add_solved, summary_table
    from .results import INDEX_FILE, RUN_FILE, read_jsonl

    run_dir = Path(args.run_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / run_dir.name
    if target != run_dir:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run_dir, target)

    meta_path = target / RUN_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    frame = read_jsonl(target / "results.jsonl")
    if not frame.empty and "status" in frame:
        meta["status_counts"] = {
            str(k): int(v) for k, v in frame["status"].value_counts().items()
        }
        if "device" in frame:
            devices = frame["device"].dropna().unique().tolist()
            meta["device"] = (
                devices[0] if len(devices) == 1 else ",".join(map(str, devices))
            )
        solved = add_solved(frame)
        summary = {
            "by_config": json.loads(summary_table(solved).to_json(orient="records")),
            "by_collection_config": json.loads(
                summary_table(solved, by=("collection", "config")).to_json(
                    orient="records"
                )
            ),
        }
        (target / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        meta["n_solved"] = int(solved["solved"].sum())
    entry = {
        "run_id": target.name,
        **{
            k: meta.get(k)
            for k in (
                "sha",
                "timestamp",
                "tag",
                "trigger",
                "tier",
                "device",
                "configs",
                "n_tasks",
                "n_solved",
                "status_counts",
            )
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")

    index_path = results_dir / INDEX_FILE
    index: list[dict[str, Any]] = (
        json.loads(index_path.read_text()) if index_path.exists() else []
    )
    index = [e for e in index if e.get("run_id") != entry["run_id"]] + [entry]
    index.sort(key=lambda e: e["run_id"])
    index_path.write_text(json.dumps(index, indent=2, default=str) + "\n")

    site_root = results_dir.parent
    (site_root / "runs.html").write_text(_runs_html(index))
    print(f"published {target.name}: {len(index)} runs indexed in {index_path}")

    if not (target / "report.html").exists() or args.render:
        code = cmd_report(argparse.Namespace(run_dir=str(target), out=None))
        if code:
            return code
    if args.dashboard:
        return cmd_dashboard(
            argparse.Namespace(results_dir=str(results_dir), out=str(site_root))
        )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Re-publish every indexed run (reports, summaries, runs.html, dashboard)."""
    results_dir = Path(args.results_dir).resolve()
    index_path = results_dir / "index.json"
    runs = (
        [e["run_id"] for e in json.loads(index_path.read_text())]
        if index_path.exists()
        else []
    )
    for run_id in runs:
        code = cmd_publish(
            argparse.Namespace(
                run_dir=str(results_dir / run_id),
                results_dir=str(results_dir),
                render=True,
                dashboard=False,
            )
        )
        if code:
            return code
    return cmd_dashboard(
        argparse.Namespace(results_dir=str(results_dir), out=str(results_dir.parent))
    )


# ------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalog", help="regenerate benchmarks/catalog.csv")
    p.add_argument("--collections", help="comma-separated subset of collections")
    p.add_argument("--out", help="output CSV path (default: benchmarks/catalog.csv)")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("run", help="run benchmark tasks")
    p.add_argument(
        "--tier",
        default="tiny",
        choices=["tiny", "small", "medium", "large"],
        help="largest size tier to include (default: tiny)",
    )
    p.add_argument(
        "--collections",
        help="comma-separated collections (default: constrained,cqp,bounded,bqp)",
    )
    p.add_argument(
        "--all-collections",
        action="store_true",
        help="include every collection (adds nle)",
    )
    p.add_argument(
        "--problems",
        help="comma-separated problem names to run (bypasses tier/collection filters)",
    )
    p.add_argument("--max-n", type=int, help="additional upper bound on n")
    p.add_argument(
        "--configs",
        "--solvers",
        dest="configs",
        help="comma-separated config names (default: all)",
    )
    p.add_argument("--chunk", help="i/N: run only the i-th of N interleaved slices")
    p.add_argument(
        "--limit", type=int, help="run only the first K instances of the selection"
    )
    p.add_argument("--jobs", type=int, default=1, help="concurrent worker processes")
    p.add_argument("--max-steps", type=int, default=500, help="outer iteration budget")
    p.add_argument(
        "--repeats", type=int, help="exact number of timed runs (default: adaptive)"
    )
    p.add_argument(
        "--repeat-budget",
        type=float,
        default=2.0,
        help="adaptive rule: target total seconds",
    )
    p.add_argument("--max-repeats", type=int, default=1000, help="adaptive rule: cap")
    p.add_argument(
        "--slow-floor",
        type=int,
        default=3,
        help="adaptive rule: minimum repeats below 60 s",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-task wall-clock timeout in seconds",
    )
    p.add_argument("--device", help="JAX platform (cpu, gpu, ...); sets JAX_PLATFORMS")
    p.add_argument("--catalog", help="alternative catalogue CSV")
    p.add_argument(
        "--out",
        default="results",
        help="results root (a run_id subdirectory is created)",
    )
    p.add_argument("--run-id", help="explicit run id (default: <timestamp>_<sha>)")
    p.add_argument(
        "--flat",
        action="store_true",
        help="write directly into --out without a run_id subdirectory",
    )
    p.add_argument("--dry-run", action="store_true", help="list the tasks and exit")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("merge", help="merge shard run directories")
    p.add_argument("shards", nargs="+", help="shard run directories")
    p.add_argument("--out", required=True, help="merged run directory")
    p.add_argument("--run-id", help="run id to record in run.json")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("report", help="export the static per-run report")
    p.add_argument("run_dir")
    p.add_argument("--out", help="output HTML (default: <run_dir>/report.html)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="export the wasm dashboard")
    p.add_argument("--results-dir", default="results")
    p.add_argument(
        "--out", default=".", help="site root receiving index.html and assets"
    )
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("publish", help="publish a run into a results tree")
    p.add_argument("run_dir")
    p.add_argument(
        "--results-dir",
        default="results",
        help="results tree (site root is its parent)",
    )
    p.add_argument(
        "--render", action="store_true", help="re-export report.html even if present"
    )
    p.add_argument(
        "--dashboard", action="store_true", help="also rebuild the wasm dashboard"
    )
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser(
        "render", help="re-export every indexed run's report and the dashboard"
    )
    p.add_argument("--results-dir", default="results")
    p.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
