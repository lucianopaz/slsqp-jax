"""Catalogue of sif2jax benchmark instances with size tiers.

The catalogue is a CSV with one row per ``(problem, y0_iD)``; computing it
takes about a minute (every problem's constraint function is evaluated once
to size it), so the generated ``benchmarks/catalog.csv`` is committed and
only regenerated with ``python -m benchmarks catalog``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "CATALOG_PATH",
    "COLLECTIONS",
    "TIERS",
    "build_catalog",
    "load_catalog",
    "select",
    "tier_of",
]

CATALOG_PATH = Path(__file__).with_name("catalog.csv")

COLLECTIONS: dict[str, str] = {
    "constrained": "constrained_minimisation_problems",
    "cqp": "constrained_quadratic_problems",
    "bounded": "bounded_minimisation_problems",
    "bqp": "bounded_quadratic_problems",
    "nle": "nonlinear_equations_problems",
}
"""Catalogue collection name -> ``sif2jax`` attribute holding the problems."""

TIERS: dict[str, int] = {"tiny": 20, "small": 100, "medium": 1000}
"""Upper bound on ``n`` for each tier; anything larger is ``large``."""

DEFAULT_COLLECTIONS: tuple[str, ...] = ("constrained", "cqp", "bounded", "bqp")

COLUMNS: tuple[str, ...] = (
    "name",
    "collection",
    "n",
    "meq",
    "mineq",
    "nbounds",
    "y0_iD",
    "n_y0s",
    "has_fstar",
    "fstar",
    "has_xstar",
    "tier",
)


def tier_of(n: int) -> str:
    """Map a problem dimension to its tier name.

    Parameters
    ----------
    n
        Number of decision variables.

    Returns
    -------
    str
        ``tiny`` (n <= 20), ``small`` (n <= 100), ``medium`` (n <= 1000) or
        ``large``.

    Examples
    --------
    >>> from benchmarks.catalog import tier_of
    >>> [tier_of(n) for n in (2, 20, 21, 1000, 1001)]
    ['tiny', 'tiny', 'small', 'medium', 'large']
    """
    for name, bound in TIERS.items():
        if n <= bound:
            return name
    return "large"


def _iter_instances(collections: Iterable[str]) -> Iterable[tuple[str, Any, int]]:
    import sif2jax

    for collection in collections:
        for problem in getattr(sif2jax, COLLECTIONS[collection]):
            for y0_iD in sorted(problem.provided_y0s):
                yield collection, problem, int(y0_iD)


def build_catalog(
    collections: Sequence[str] = tuple(COLLECTIONS),
    *,
    log: bool = True,
) -> pd.DataFrame:
    """Compute the catalogue by instantiating every sif2jax problem.

    Parameters
    ----------
    collections
        Catalogue collections to include (keys of :data:`COLLECTIONS`).
    log
        Print a progress line per collection to stderr.

    Returns
    -------
    pandas.DataFrame
        One row per ``(name, y0_iD)`` with the columns in :data:`COLUMNS`.
    """
    import jax

    from .problems import to_sqpdax

    jax.config.update("jax_enable_x64", True)
    rows: list[dict[str, Any]] = []
    for collection, problem, y0_iD in _iter_instances(collections):
        try:
            _, _, meta = to_sqpdax(problem, y0_iD=y0_iD)
        except Exception as exc:  # noqa: BLE001 - catalogue must survive bad problems
            if log:
                print(
                    f"[catalog] skipping {problem.name} ({collection}): {exc!r}",
                    file=sys.stderr,
                )
            continue
        # ``meta.collection`` is class-based, so a quadratic problem listed in
        # both ``constrained_*`` and ``constrained_quadratic_*`` gets the same
        # label from either list and collapses in ``drop_duplicates`` below.
        row = meta.as_dict()
        row["tier"] = tier_of(meta.n)
        rows.append(row)
        if log and len(rows) % 100 == 0:
            print(f"[catalog] {len(rows)} instances", file=sys.stderr)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame = frame.drop_duplicates(subset=["name", "y0_iD"], keep="first")
    return frame.sort_values(["collection", "name", "y0_iD"]).reset_index(drop=True)


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Read the committed catalogue CSV.

    Parameters
    ----------
    path
        Location of the CSV; defaults to :data:`CATALOG_PATH`.

    Returns
    -------
    pandas.DataFrame
        The catalogue with the columns in :data:`COLUMNS`.
    """
    return pd.read_csv(path)


def select(
    catalog: pd.DataFrame,
    *,
    tier: str | None = None,
    collections: Sequence[str] | None = DEFAULT_COLLECTIONS,
    names: Sequence[str] | None = None,
    max_n: int | None = None,
    chunk: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Filter the catalogue down to the instances to benchmark.

    Parameters
    ----------
    catalog
        Full catalogue from :func:`load_catalog`.
    tier
        Keep instances in this tier and every smaller one (``small`` keeps
        ``tiny`` too). ``None`` keeps all tiers.
    collections
        Collections to keep; ``None`` keeps all. By default the nonlinear
        equations are excluded.
    names
        Explicit problem names to keep (applied after the other filters;
        case-insensitive).
    max_n
        Additional upper bound on ``n``.
    chunk
        ``(index, count)`` to keep only the ``index``-th of ``count`` equal
        slices of the sorted selection (for CI sharding).

    Returns
    -------
    pandas.DataFrame
        The selected rows, sorted by ``(n, name, y0_iD)`` so chunks are
        balanced in problem size.
    """
    frame = catalog
    if tier is not None:
        order = [*TIERS, "large"]
        if tier not in order:
            raise ValueError(f"unknown tier {tier!r}; expected one of {order}")
        keep = order[: order.index(tier) + 1]
        frame = frame[frame["tier"].isin(keep)]
    if collections is not None:
        frame = frame[frame["collection"].isin(list(collections))]
    if names:
        wanted = {name.upper() for name in names}
        frame = frame[frame["name"].str.upper().isin(wanted)]
    if max_n is not None:
        frame = frame[frame["n"] <= max_n]
    frame = frame.sort_values(["n", "name", "y0_iD"]).reset_index(drop=True)
    if chunk is not None:
        index, count = chunk
        if not 0 <= index < count:
            raise ValueError(f"chunk index {index} out of range for {count} chunks")
        frame = frame.iloc[index::count].reset_index(drop=True)
    return frame
