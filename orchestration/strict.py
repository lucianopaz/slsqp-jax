#!/usr/bin/env python3
"""Strict-convergence accounting for a benchmark ``results.jsonl``.

The benchmark flag ``solved`` means: the solver reported success, the point is
feasible and ``f_gap <= f_tol``.  It never looks at stationarity, so a solver
that stops on a small change of the objective (SciPy SLSQP, ``ftol``) can be
counted as solved at a non-stationary point of a flat valley.  This script
re-scores every row with the KKT metrics the benchmark already stores
(``stat_scaled``, ``compl``, ``mult_min``) and compares the two views.

    strict.py RESULTS.jsonl [--tau 1e-4] [--f-tol 1e-4] [--feas-tol 1e-6]

Definitions (all rows, scipy included):

- ``solved``: the benchmark flag.
- ``strict``: ``solved`` and a KKT point: ``stat_scaled <= tau``,
  ``compl <= tau`` and ``mult_min >= -tau``.
- ``stat``: ``solved`` and ``stat_scaled <= tau`` only (no complementarity, so
  barrier methods whose final complementarity is ~mu are not penalised).
- ``kkt``: ``feas <= feas_tol``, ``f_gap <= f_tol`` and the same KKT test,
  whatever the solver's exit status (a true solution that was labelled as a
  failure counts).

Rows without multipliers (``NaN`` metrics) are reported separately and never
counted as strict.
"""

from __future__ import annotations

import argparse
import collections
import json
import math


def _ok(v: float | None) -> bool:
    return v is not None and isinstance(v, (int, float)) and not math.isnan(v)


def kkt_point(r: dict, tau: float) -> bool | None:
    """True/False for the KKT test, ``None`` when the metrics are missing."""
    stat, compl, mult = r.get("stat_scaled"), r.get("compl"), r.get("mult_min")
    if not (_ok(stat) and _ok(compl) and _ok(mult)):
        return None
    return stat <= tau and compl <= tau and mult >= -tau


def score(r: dict, tau: float, f_tol: float, feas_tol: float) -> dict:
    k = kkt_point(r, tau)
    solved = bool(r.get("solved"))
    f_ok = _ok(r.get("f_gap")) and r["f_gap"] <= f_tol
    feas_ok = _ok(r.get("feas")) and r["feas"] <= feas_tol
    stat = r.get("stat_scaled")
    return {
        "solved": solved,
        "stat": solved and _ok(stat) and stat <= tau,
        "strict": solved and k is True,
        "kkt": f_ok and feas_ok and k is True,
        "unknown": k is None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--f-tol", type=float, default=1e-4)
    ap.add_argument("--feas-tol", type=float, default=1e-6)
    ap.add_argument("--taus", default="1e-6,1e-5,1e-4,1e-3")
    a = ap.parse_args()
    rows = [r for r in map(json.loads, open(a.results)) if r.get("problem")]
    cfgs = sorted({r["config"] for r in rows}, key=lambda c: (c.startswith("scipy"), c))
    inst = sorted({(r["problem"], r.get("y0_iD", 0)) for r in rows})

    def table(tau: float) -> dict:
        out = {}
        for c in cfgs:
            sc = [score(r, tau, a.f_tol, a.feas_tol) for r in rows if r["config"] == c]
            out[c] = {k: sum(s[k] for s in sc) for k in sc[0]}
        return out

    print(f"{len(inst)} instances; tau = {a.tau:g}, f_tol = {a.f_tol:g}\n")
    t = table(a.tau)
    print(
        f"{'config':20s} {'solved':>6s} {'stat':>5s} {'strict':>6s} {'kkt':>5s} {'solved-not-strict':>18s} {'unknown':>8s}"
    )
    for c in cfgs:
        d = t[c]
        print(
            f"{c:20s} {d['solved']:6d} {d['stat']:5d} {d['strict']:6d} {d['kkt']:5d} {d['solved'] - d['strict']:18d} {d['unknown']:8d}"
        )
    print("\nsensitivity of 'stat' (solved and stat_scaled <= tau) to tau:")
    taus = [float(x) for x in a.taus.split(",")]
    print(f"{'config':20s} " + " ".join(f"{x:>8g}" for x in taus))
    tabs = {x: table(x) for x in taus}
    for c in cfgs:
        print(f"{c:20s} " + " ".join(f"{tabs[x][c]['stat']:8d}" for x in taus))

    # instance-level union view: an instance is solved by a group if any member solves it
    def inst_set(group, key):
        s = set()
        for r in rows:
            if r["config"] in group and score(r, a.tau, a.f_tol, a.feas_tol)[key]:
                s.add((r["problem"], r.get("y0_iD", 0)))
        return s

    sq = [c for c in cfgs if not c.startswith("scipy")]
    sc = [c for c in cfgs if c.startswith("scipy")]
    sc1 = ["scipy-slsqp"]
    print("\ninstance-level union (any config solves it):")
    for key in ("solved", "stat", "strict", "kkt"):
        S, C = inst_set(sq, key), inst_set(sc, key)
        print(
            f"  {key:7s} sqpdax {len(S):3d}  scipy {len(C):3d}  scipy-slsqp {len(inst_set(sc1, key)):3d}  deficit (scipy yes, sqpdax no) {len(C - S):2d}  sqpdax-only {len(S - C):2d}"
        )
        print("          deficit:", sorted(p for p, _ in (C - S)))
    print(
        "\nscipy 'solved' rows that fail the strict test (instance: stat_scaled, compl, mult_min, f_gap):"
    )
    bad = collections.defaultdict(list)
    for r in rows:
        if (
            r["config"].startswith("scipy")
            and r.get("solved")
            and kkt_point(r, a.tau) is False
        ):
            bad[r["config"]].append(r)
    for c, rs in bad.items():
        print(f"  {c}: {len(rs)}")
        for r in sorted(rs, key=lambda r: -(r.get("stat_scaled") or 0))[:12]:
            print(
                f"    {r['problem']:12s} stat={r.get('stat_scaled'):.1e} compl={r.get('compl'):.1e} mult_min={r.get('mult_min'):+.1e} f_gap={r.get('f_gap'):.1e}"
            )


if __name__ == "__main__":
    main()
