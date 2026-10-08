"""Result storage: append-only JSONL rows plus per-run metadata.

This module is pure ``pandas`` / standard library so the marimo notebooks
can import it under Pyodide.

Layout of one run directory ``results/<YYYYMMDDTHHMMSSZ>_<shortsha>/``:

- ``results.jsonl`` - one JSON object per ``(problem, y0_iD, config)`` task,
  appended as soon as the task completes.
- ``run.json`` - run-level metadata (see :func:`run_metadata`).
- ``report.html`` - static per-run report exported by marimo.

``results/index.json`` at the parent level lists every published run.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "HARNESS_STATUSES",
    "RESULTS_FILE",
    "RUN_FILE",
    "INDEX_FILE",
    "append_row",
    "git_sha",
    "load_index",
    "load_run",
    "merge_jsonl",
    "new_run_id",
    "read_jsonl",
    "run_metadata",
    "write_run_metadata",
]

RESULTS_FILE = "results.jsonl"
RUN_FILE = "run.json"
INDEX_FILE = "index.json"

HARNESS_STATUSES: tuple[str, ...] = ("timeout", "compile_error", "runtime_error")
"""Statuses produced by the harness rather than by a minimiser."""


def git_sha(short: bool = False) -> str:
    """Current git commit, or ``"unknown"`` outside a repository.

    Parameters
    ----------
    short
        Return the 7-character abbreviation.
    """
    # BENCH_SHA lets CI record the benchmarked commit when the workflow
    # itself runs from another ref (workflow_run); GITHUB_SHA is the default.
    sha = os.environ.get("BENCH_SHA") or os.environ.get("GITHUB_SHA")
    if sha is None:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    return sha[:7] if short else sha


def new_run_id(now: datetime | None = None, sha: str | None = None) -> str:
    """Build a run id ``<YYYYMMDDTHHMMSSZ>_<shortsha>`` that sorts by time.

    Parameters
    ----------
    now
        Timestamp (UTC assumed when naive); defaults to the current time.
    sha
        Commit to embed; defaults to :func:`git_sha`.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> from benchmarks.results import new_run_id
    >>> new_run_id(datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc), "0123456789abcdef")
    '20261008T093000Z_0123456'
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    sha = (sha or git_sha())[:7]
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{sha}"


def run_metadata(**extra: Any) -> dict[str, Any]:
    """Collect run-level metadata.

    Parameters
    ----------
    **extra
        Additional fields (tier, configs, trigger, ...) merged into the
        result.

    Returns
    -------
    dict[str, Any]
        ``run_id``-free mapping with ``sha``, ``timestamp``, ``hostname``,
        ``platform``, ``python`` and any ``extra`` entries.
    """
    meta: dict[str, Any] = {
        "sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "trigger": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "tag": os.environ.get("BENCH_TAG"),
    }
    meta.update(extra)
    return meta


def write_run_metadata(run_dir: Path, meta: Mapping[str, Any]) -> Path:
    """Write ``run.json`` into ``run_dir`` (created if needed)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_FILE
    path.write_text(
        json.dumps(dict(meta), indent=2, sort_keys=True, default=str) + "\n"
    )
    return path


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line to ``path`` (created if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(dict(row), default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def read_jsonl(path: Path | str) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame (empty frame if missing/empty)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def merge_jsonl(sources: Iterable[Path | str], target: Path) -> int:
    """Concatenate several JSONL files into ``target``.

    Parameters
    ----------
    sources
        Input files (missing ones are skipped).
    target
        Output file, overwritten.

    Returns
    -------
    int
        Number of rows written.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w") as out:
        for source in sources:
            source = Path(source)
            if not source.exists():
                continue
            with source.open() as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    return count


def load_run(run_dir: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load ``results.jsonl`` and ``run.json`` from a run directory.

    Parameters
    ----------
    run_dir
        The run directory.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Rows (with a ``run_id`` column added) and the metadata mapping
        (empty when ``run.json`` is missing).
    """
    run_dir = Path(run_dir)
    frame = read_jsonl(run_dir / RESULTS_FILE)
    meta_path = run_dir / RUN_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not frame.empty:
        frame["run_id"] = run_dir.name
    return frame, meta


def load_index(results_dir: Path | str) -> pd.DataFrame:
    """Load ``results/index.json`` as a DataFrame (empty if missing).

    Parameters
    ----------
    results_dir
        Directory holding the run directories and ``index.json``.
    """
    path = Path(results_dir) / INDEX_FILE
    if not path.exists():
        return pd.DataFrame()
    entries = json.loads(path.read_text())
    return pd.DataFrame(entries)
