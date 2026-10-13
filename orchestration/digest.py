#!/usr/bin/env python3
"""Compact evidence digest of one cause group from a ``classified.jsonl``.

    digest.py --classified diagnostics/v2/classified.jsonl --cause ip_barrier_stall [--examples 12]

Prints markdown (about 3k tokens): counts by config and status, share solved by the scipy
baselines, quantiles of the main diagnostics, example rows with their evidence string, the
report's current description/recommendation for the cause, and the ledger history.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import re
import runpy
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def q(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))]


def stats_line(name: str, vals: list[float]) -> str:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return f"- {name}: n/a"
    return f"- {name}: min {min(vals):.2g}, median {statistics.median(vals):.2g}, p90 {q(vals, 0.9):.2g}, max {max(vals):.2g} (n={len(vals)})"


def strip(s: str, limit: int) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", required=True)
    ap.add_argument("--cause", required=True)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.classified)]
    g = [r for r in rows if r["cause"] == a.cause]
    if not g:
        sys.exit(
            f"no rows with cause {a.cause!r}; causes: {collections.Counter(r['cause'] for r in rows).most_common(40)}"
        )
    n = len(g)
    ref = sum(bool(r.get("ref_solved")) for r in g)
    tc = sum(bool(r.get("tc_solved")) for r in g)
    either = sum(bool(r.get("ref_solved") or r.get("tc_solved")) for r in g)

    print(f"# Digest: {a.cause} ({a.classified})\n")
    print(
        f"tasks: {n} | solved by scipy-slsqp {ref}, trust-constr {tc}, either {either}"
    )
    print(
        "by config: "
        + ", ".join(
            f"{k} {v}"
            for k, v in collections.Counter(r["config"] for r in g).most_common()
        )
    )
    print(
        "by status: "
        + ", ".join(
            f"{k} {v}"
            for k, v in collections.Counter(r["status"] for r in g).most_common()
        )
    )
    probs = collections.Counter(r["problem"] for r in g)
    print(
        f"problems ({len(probs)}): "
        + ", ".join(f"{p}x{c}" if c > 1 else p for p, c in probs.most_common(40))
    )
    print("\n## Quantiles")
    for k in ("steps", "kkt_last", "feas", "mu_last", "f_gap", "n"):
        print(
            stats_line(k, [r.get(k) for r in g if isinstance(r.get(k), (int, float))])
        )

    print("\n## Examples (scipy-solved first, then by kkt_last)")
    ex = sorted(
        g,
        key=lambda r: (
            not (r.get("ref_solved") or r.get("tc_solved")),
            r.get("kkt_last") or 0,
        ),
    )
    seen: set[str] = set()
    shown = 0
    for r in ex:
        if r["problem"] in seen:
            continue
        seen.add(r["problem"])
        shown += 1
        print(
            f"- {r['problem']} [{r['config']}] n={r.get('n')} meq={r.get('meq')} mineq={r.get('mineq')} "
            f"status={r['status']} steps={r.get('steps')} kkt={r.get('kkt_last'):.2g} feas={r.get('feas'):.2g} "
            f"mu={r.get('mu_last')} f_gap={r.get('f_gap')} scipy={bool(r.get('ref_solved'))}/{bool(r.get('tc_solved'))}"
            if all(isinstance(r.get(k), (int, float)) for k in ("kkt_last", "feas"))
            else f"- {r['problem']} [{r['config']}] status={r['status']} steps={r.get('steps')}"
        )
        if r.get("evidence"):
            print(f"    evidence: {strip(str(r['evidence']), 260)}")
        if shown >= a.examples:
            break

    try:
        m = runpy.run_path(
            str(ROOT / "scripts" / "diagnose_benchmark.py"), run_name="digest_import"
        )
        cat, title, desc = m["CAUSES"][a.cause]
        print(
            f"\n## Report text\n\ncategory: {cat}; title: {title}\n\n{strip(desc, 900)}"
        )
        if a.cause in m["RECOMMENDATIONS"]:
            print(f"\nRecommendations: {strip(m['RECOMMENDATIONS'][a.cause], 1800)}")
        for at in m["FIX_ATTEMPTS"].get(a.cause, []):
            print(
                f"\nPrevious attempt: {strip(at['title'], 160)} -> {strip(at['outcome'], 400)}"
            )
    except Exception as e:  # report constants are optional context
        print(f"\n(report text unavailable: {e!r})")

    print("\n## Ledger\n")
    out = subprocess.run(
        [sys.executable, str(HERE / "ledger.py"), "list", "--cause", a.cause, "-v"],
        capture_output=True,
        text=True,
    )
    print(out.stdout.strip() or "(no entries)")


if __name__ == "__main__":
    main()
