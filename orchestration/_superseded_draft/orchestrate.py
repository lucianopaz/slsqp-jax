#!/usr/bin/env python3
"""benchmark -> triage (Sonnet) -> diagnose (Fable, 1 turn, no tools) -> implement (Sonnet) -> re-benchmark.

Usage: python orchestrate.py --rounds 3 --hyps-per-round 2 --budget-usd 15
Run from the repo root on a clean git branch. Verify flags with `claude --help` for your version.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(".").resolve()
RUNS = REPO / "runs"
# --- EDIT THESE -------------------------------------------------------------------------------
BENCH_CMD = [
    "python",
    "benchmarks/run_benchmark.py",
]  # must accept: --out <results.json>
CORE_FILES = [
    "src/sqpdax/step.py",
    "src/sqpdax/qp.py",
    "src/sqpdax/hessian.py",
    "src/sqpdax/linesearch.py",
    "src/sqpdax/convergence.py",
]
MODELS = dict(triage="sonnet", diagnose="claude-fable-5-1", implement="sonnet")
# ----------------------------------------------------------------------------------------------
spent = 0.0


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def claude(prompt, model, max_turns, tools=None, deny=None, stdin=None):
    """Headless Claude Code call. Returns the final text; tracks reported cost."""
    global spent
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
    ]
    if tools:
        cmd += ["--allowedTools", *tools]
    if deny:
        cmd += ["--disallowedTools", *deny]
    out = json.loads(sh(cmd, input=stdin).stdout)
    cost = out.get("total_cost_usd") or 0.0
    spent += cost
    print(f"  [{model}] cost={cost:.3f} total={spent:.3f}", flush=True)
    (RUNS / "usage.jsonl").open("a").write(
        json.dumps(dict(t=time.time(), model=model, cost=cost, usage=out.get("usage")))
        + "\n"
    )
    return out.get("result", "")


def benchmark(rdir):
    res = rdir / "results.json"
    sh(BENCH_CMD + ["--out", str(res)])
    recs = json.loads(res.read_text())
    solved = {
        r["problem"]
        for r in recs
        if r["solver"] == "sqpdax" and r["converged_independent"]
    }
    return res, solved


def triage(rdir, res):
    ft = rdir / "failure_table.json"
    claude(
        f"Use the triage subagent. results: {res}  output: {ft}",
        MODELS["triage"],
        20,
        tools=["Read", "Write", "Grep", "Glob", "Bash(python:*)"],
        deny=["Edit"],
    )
    return ft


def diagnose(rdir, ft):
    bundle = [f"# FAILURE TABLE\n{ft.read_text()}"]
    for f in CORE_FILES:  # stable code first would help caching on the API
        bundle.append(f"\n# FILE: {f}\n{(REPO / f).read_text()}")
    prompt = (
        "You are diagnosing why an SQP optimiser (sqpdax, JAX) converges on fewer CUTEst problems than "
        "scipy SLSQP/trust-constr. Convergence is judged by an independent KKT checker, so failures are "
        "real. Using the failure table and source, return ONLY a JSON list (max 6) of hypotheses, ranked "
        'by expected benefit: [{"id": "h1", "category": "bug|hyperparameter|convergence_test", '
        '"claim": ..., "evidence": [problem names / patterns / file:line], "experiment": ..., '
        '"suggested_change": ..., "affected_patterns": [...]}]. Do not write code.'
    )
    text = claude(
        prompt,
        MODELS["diagnose"],
        1,
        deny=["Bash", "Edit", "Write"],
        stdin="\n".join(bundle),
    )
    (rdir / "diagnosis_raw.txt").write_text(text)
    m = re.search(r"\[.*\]", text, re.S)
    hyps = json.loads(m.group(0)) if m else []
    (rdir / "hypotheses.json").write_text(json.dumps(hyps, indent=2))
    return hyps


def implement_and_test(rdir, hyp, baseline):
    claude(
        "Use the implementer subagent on this hypothesis:\n"
        + json.dumps(hyp, indent=2),
        MODELS["implement"],
        40,
        tools=["Read", "Edit", "Write", "Grep", "Glob", "Bash"],
    )
    hdir = rdir / hyp["id"]
    hdir.mkdir(exist_ok=True)
    _, solved = benchmark(hdir)
    gained, lost = solved - baseline, baseline - solved
    print(f"  {hyp['id']}: +{len(gained)} -{len(lost)}")
    if gained and not lost:
        sh(["git", "commit", "-am", f"auto: {hyp['id']} {hyp['claim'][:60]}"])
        return solved
    sh(["git", "reset", "--hard", "HEAD"])
    sh(["git", "clean", "-fd", "src"])
    return baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--hyps-per-round", type=int, default=2)
    ap.add_argument("--budget-usd", type=float, default=10.0)
    a = ap.parse_args()
    if sh(["git", "status", "--porcelain"]).stdout.strip():
        sys.exit("Working tree not clean; commit or stash first.")
    RUNS.mkdir(exist_ok=True)
    rdir = RUNS / "round_0"
    rdir.mkdir(exist_ok=True)
    res, baseline = benchmark(rdir)
    print(f"baseline solved: {len(baseline)}")
    for r in range(1, a.rounds + 1):
        if spent >= a.budget_usd:
            print("budget reached")
            break
        rdir = RUNS / f"round_{r}"
        rdir.mkdir(exist_ok=True)
        ft = triage(rdir, res)
        hyps = diagnose(rdir, ft)
        if not hyps:
            print("no hypotheses parsed; see diagnosis_raw.txt")
            break
        for h in hyps[: a.hyps_per_round]:
            if spent >= a.budget_usd:
                break
            baseline = implement_and_test(rdir, h, baseline)
        res, baseline = benchmark(rdir)  # fresh baseline for next round
        print(f"round {r}: solved {len(baseline)}  spent {spent:.2f}")


if __name__ == "__main__":
    main()
