#!/usr/bin/env python3
"""Append-only ledger of hypotheses (``ledger.jsonl``) and a session lock.

    ledger.py add --id ID --cause C --title T --decision accepted|rejected|inconclusive [...]
    ledger.py list [--cause C] [--decision D]
    ledger.py tail [N]
    ledger.py lock CAUSE | unlock CAUSE | status

Fields (all optional except id/cause/title/decision): date, branch, commit, baseline_dir,
compare_dir, tasks, solved, regress_lost (``{"regression": k, "noise": k, "catalogue": k}``),
notes. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
LOCK = HERE / ".lock"


def read() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [
        json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()
    ]


def fmt(e: dict) -> str:
    solved = ""
    if e.get("tasks") is not None and e.get("solved") is not None:
        solved = f" solved {e['solved']}/{e['tasks']}"
    lost = f" lost={e['regress_lost']}" if e.get("regress_lost") else ""
    return f"[{e.get('date', '?')}] {e['id']} ({e['cause']}) {e['decision'].upper()}{solved}{lost}: {e['title']}"


def cmd_add(a: argparse.Namespace) -> None:
    if any(e["id"] == a.id for e in read()):
        sys.exit(f"ledger: id {a.id!r} already exists")
    entry = {
        "id": a.id,
        "date": a.date or dt.date.today().isoformat(),
        "cause": a.cause,
        "title": a.title,
        "decision": a.decision,
    }
    for k in ("branch", "commit", "baseline_dir", "compare_dir", "notes"):
        if getattr(a, k):
            entry[k] = getattr(a, k)
    if a.tasks is not None:
        entry["tasks"] = a.tasks
    if a.solved is not None:
        entry["solved"] = a.solved
    if a.regress_lost:
        entry["regress_lost"] = json.loads(a.regress_lost)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(fmt(entry))


def cmd_list(a: argparse.Namespace) -> None:
    for e in read():
        if a.cause and a.cause not in e["cause"].split(","):
            continue
        if a.decision and e["decision"] != a.decision:
            continue
        print(fmt(e))
        if a.verbose and e.get("notes"):
            print("    " + e["notes"])


def cmd_tail(a: argparse.Namespace) -> None:
    for e in read()[-a.n :]:
        print(fmt(e))


def held() -> dict | None:
    """Current lock holder, or None (a released lock file counts as no lock)."""
    if not LOCK.exists() or not LOCK.read_text().strip():
        return None
    data = json.loads(LOCK.read_text())
    return None if data.get("released") else data


def cmd_lock(a: argparse.Namespace) -> None:
    if held():
        sys.exit(f"ledger: locked: {LOCK.read_text().strip()}")
    LOCK.write_text(
        json.dumps(
            {"cause": a.cause, "since": dt.datetime.now().isoformat(timespec="seconds")}
        )
    )
    print(f"locked {a.cause}")


def cmd_unlock(a: argparse.Namespace) -> None:
    cur = held()
    if cur is None:
        print("not locked")
        return
    if cur["cause"] != a.cause and not a.force:
        sys.exit(f"ledger: lock is held for {cur['cause']!r}; use --force to override")
    try:
        LOCK.unlink()
    except PermissionError:  # deletion disabled: mark as released instead
        LOCK.write_text(
            json.dumps({"released": dt.datetime.now().isoformat(timespec="seconds")})
        )
    print("unlocked")


def cmd_status(_: argparse.Namespace) -> None:
    print(LOCK.read_text() if held() else "not locked")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add")
    s.add_argument("--id", required=True)
    s.add_argument(
        "--cause", required=True, help="cause id; several separated by commas"
    )
    s.add_argument("--title", required=True)
    s.add_argument(
        "--decision", required=True, choices=["accepted", "rejected", "inconclusive"]
    )
    s.add_argument("--date")
    s.add_argument("--branch")
    s.add_argument("--commit")
    s.add_argument("--baseline-dir", dest="baseline_dir")
    s.add_argument("--compare-dir", dest="compare_dir")
    s.add_argument("--tasks", type=int)
    s.add_argument("--solved", type=int)
    s.add_argument(
        "--regress-lost",
        dest="regress_lost",
        help='JSON, e.g. \'{"regression":0,"noise":2}\'',
    )
    s.add_argument("--notes")
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("list")
    s.add_argument("--cause")
    s.add_argument("--decision")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("tail")
    s.add_argument("n", nargs="?", type=int, default=5)
    s.set_defaults(fn=cmd_tail)
    s = sub.add_parser("lock")
    s.add_argument("cause")
    s.set_defaults(fn=cmd_lock)
    s = sub.add_parser("unlock")
    s.add_argument("cause")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_unlock)
    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
