#!/usr/bin/env python3
"""Join two benchmark ``results.jsonl`` files on (problem, y0_iD, config).

    regress_join.py BASELINE.jsonl NEW.jsonl [--configs a,b] [--out loss.json]

Prints solved per config before/after, the newly solved and the lost tasks, and the command
lines to re-run the lost tasks at the base commit (attribution step I7).
"""

from __future__ import annotations

import argparse
import collections
import json


def load(path: str) -> dict:
    return {
        (r["problem"], r.get("y0_iD"), r["config"]): r
        for r in map(json.loads, open(path))
        if r.get("problem")
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--configs")
    ap.add_argument("--out")
    a = ap.parse_args()
    base, new = load(a.baseline), load(a.new)
    cfgs = set(a.configs.split(",")) if a.configs else None
    keys = [k for k in new if k in base and (cfgs is None or k[2] in cfgs)]
    gained = sorted(
        k for k in keys if new[k].get("solved") and not base[k].get("solved")
    )
    lost = sorted(k for k in keys if base[k].get("solved") and not new[k].get("solved"))
    per = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys:
        per[k[2]][0] += bool(base[k].get("solved"))
        per[k[2]][1] += bool(new[k].get("solved"))
        per[k[2]][2] += 1
    print(
        f"shared tasks: {len(keys)} (baseline-only {len(set(base) - set(new))}, new-only {len(set(new) - set(base))})"
    )
    for c, (b, n, t) in sorted(per.items()):
        print(f"  {c:16s} {b:4d} -> {n:4d} / {t}")
    print(f"gained {len(gained)}, lost {len(lost)}")
    for k in lost:
        print(
            f"  LOST {k[0]} y0={k[1]} {k[2]}: {base[k].get('status')} -> {new[k].get('status')} (kkt/feas {new[k].get('feas')})"
        )
    if lost:
        by_cfg = collections.defaultdict(set)
        for p, _, c in lost:
            by_cfg[c].add(p)
        print("\nattribution (run at the base worktree, one line per config):")
        for c, ps in sorted(by_cfg.items()):
            print(
                f"  python -m benchmarks run --tier tiny --problems {','.join(sorted(ps))} --configs {c} --repeats 1 --jobs 2 --out <OUT>/attr_{c} --flat"
            )
    if a.out:
        json.dump(
            {
                "gained": gained,
                "lost": lost,
                "per_config": {c: v for c, v in per.items()},
            },
            open(a.out, "w"),
        )


if __name__ == "__main__":
    main()
