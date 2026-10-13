#!/usr/bin/env python3
"""Head-to-head of sqpdax against the (frozen) scipy baselines, optionally with measured fixes overlaid.

    gap.py RESULTS [--overlay compare.json ...] [--list 40]

RESULTS is the frozen benchmark ``results.jsonl`` (scipy rows are never re-run). Each ``--overlay``
is a ``compare.json`` from ``diagnose_benchmark.py compare``; a task unsolved in RESULTS counts as
solved when the overlay marks (problem, config) solved (all duplicates of that key must be solved).
Goal: every sqpdax config and their union solve more instances than scipy-slsqp / trust-constr / both.
"""

from __future__ import annotations

import argparse
import collections
import json

SQ = [
    "asls-craig",
    "asls-pcg",
    "asls-pcg-exact",
    "asls-minresqlp",
    "pasls",
    "trip",
    "tfip",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--list", type=int, default=30)
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.results)]
    ov: dict[tuple, list[bool]] = collections.defaultdict(list)
    for p in a.overlay:
        for r in json.load(open(p))["rows"]:
            ov[(r["problem"], r["config"])].append(bool(r["solved"]))
    solved: dict[tuple, bool] = {}
    for r in rows:
        k = (r["problem"], r.get("y0_iD"), r["config"])
        s = bool(r.get("solved"))
        if (
            not s
            and r["config"] in SQ
            and (r["problem"], r["config"]) in ov
            and all(ov[(r["problem"], r["config"])])
        ):
            s = True
        solved[k] = s
    inst = sorted({(p, y) for p, y, _ in solved}, key=lambda t: (t[0], str(t[1])))

    def has(i, cfgs):
        return any(solved.get((i[0], i[1], c), False) for c in cfgs)

    n = len(inst)
    sl = sum(has(i, ["scipy-slsqp"]) for i in inst)
    tc = sum(has(i, ["scipy-trust-constr"]) for i in inst)
    both = sum(has(i, ["scipy-slsqp", "scipy-trust-constr"]) for i in inst)
    sq = sum(has(i, SQ) for i in inst)
    print(
        f"instances {n} | scipy-slsqp {sl}, trust-constr {tc}, scipy union {both} | sqpdax union {sq} ({sq - both:+d} vs scipy union)"
    )
    for c in SQ:
        k = sum(has(i, [c]) for i in inst)
        print(
            f"  {c:16s} {k:4d}  ({k - sl:+d} vs scipy-slsqp, {k - tc:+d} vs trust-constr)"
        )
    deficit = [
        i
        for i in inst
        if has(i, ["scipy-slsqp", "scipy-trust-constr"]) and not has(i, SQ)
    ]
    ahead = [
        i
        for i in inst
        if has(i, SQ) and not has(i, ["scipy-slsqp", "scipy-trust-constr"])
    ]
    print(
        f"\ndeficit (scipy solves, no sqpdax config does): {len(deficit)}; sqpdax-only wins: {len(ahead)}"
    )
    print(
        "  "
        + ", ".join(f"{p}" for p, _ in deficit[: a.list])
        + (" ..." if len(deficit) > a.list else "")
    )


if __name__ == "__main__":
    main()
