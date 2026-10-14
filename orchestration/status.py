#!/usr/bin/env python3
"""Progress of both streams: ledger tail, hypothesis branches per stream, running jobs, latest logs.

python orchestration/status.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
B = os.path.join(os.path.dirname(MAIN), "slsqp-jax-streamB")


def sh(*cmd: str, cwd: str = MAIN) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()


print("== ledger (last 12) ==")
for line in open(os.path.join(HERE, "ledger.jsonl")).read().splitlines()[-12:]:
    e = json.loads(line)
    print(
        f"{e.get('date')} {e.get('id')}: {str(e.get('decision')).upper()} "
        f"{e.get('solved')}/{e.get('tasks')} lost={e.get('regress_lost')}"
    )
for name, d, base in (("A", MAIN, "local_bench_cleanup"), ("B", B, "stream/b")):
    print(f"\n== stream {name}: {base} ==")
    print(sh("git", "log", "--oneline", "-3", base))
    print("open worktrees:")
    for wt in sh("git", "worktree", "list", cwd=d).splitlines():
        if f"{d}/.worktrees" in wt:
            print("  ", wt)
print("\n== running jobs ==")
print(
    sh(
        "bash",
        "-c",
        "ps -eo etime,command | grep -E 'diagnose_benchmark|benchmarks run|pytest' "
        "| grep -v grep | cut -c1-170",
    )
)
print("\n== latest diagnostics logs ==")
logs = sorted(
    glob.glob(os.path.join(MAIN, "diagnostics/v4/*/*/*.log")), key=os.path.getmtime
)[-6:]
for p in logs:
    age = int(time.time() - os.path.getmtime(p))
    tail = sh("tail", "-c", "140", p).replace("\n", " ")
    print(f"{os.path.relpath(p, MAIN)} ({age}s ago): ...{tail}")
