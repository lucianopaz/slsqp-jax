"""Adapt sif2jax problems to :class:`slsqp_jax.sqpdax.problem.Problem`.

sif2jax expresses constraints as ``constraint(y) -> (eq, ineq)`` with
``eq == 0`` and ``ineq >= 0``; sqpdax expects ``g(x) = 0`` and ``h(x) <= 0``.
The adapter negates the inequality block and builds the problem through
:func:`~slsqp_jax.sqpdax.problem.builder.build_problem` with
``autodiff_mode="jax"`` and ``force_hvp_in_jax_mode=True`` so gradients,
Jacobians and Hessian-vector products are all available (exact-curvature
configurations rely on the latter).

Not every sif2jax problem honours the ``ineq >= 0`` convention: some return
the raw pycutest value ``c(x) - b`` of an L-type group (``c(x) <= b``), which
is a ``<= 0`` quantity (POLAK4, CHACONN1, GOFFIN, ... the minimax family), and
a few mix both conventions row by row (HS108). The benchmark catalogue
(``benchmarks/catalog.csv``, see :mod:`benchmarks.validate`) records, per
instance, which inequality rows have to be flipped and which reference
objective to score against; :func:`to_sqpdax` applies those overrides by
default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from slsqp_jax.sqpdax.problem import Problem, build_problem

__all__ = [
    "ProblemMeta",
    "catalog_overrides",
    "collection_of",
    "constraint_sizes",
    "flip_signs",
    "format_flip",
    "to_sqpdax",
]


@dataclass(frozen=True)
class ProblemMeta:
    """Static description of one ``(problem, y0_iD)`` benchmark instance.

    Attributes
    ----------
    name
        CUTEst problem name (the sif2jax class name).
    collection
        One of ``constrained``, ``cqp``, ``bounded``, ``bqp``, ``nle``.
    n, meq, mineq
        Problem sizes as seen by sqpdax.
    nbounds
        Number of finite bound entries (lower and upper counted separately).
    y0_iD
        Which of the SIF-provided starting points is used.
    n_y0s
        Number of starting points the problem provides.
    has_fstar, fstar
        Whether a reference objective value is available for scoring, and
        its value. This is the *validated* reference: sif2jax's
        ``expected_objective_value`` unless the catalogue replaced it by the
        scipy consensus (see ``fstar_source``).
    has_xstar
        Whether a reference minimiser is known.
    ineq_flip
        Inequality rows whose sign convention the catalogue corrected:
        ``""`` (none), ``"all"`` or comma-separated zero-based row indices
        (see :func:`flip_signs`).
    fstar_sif
        sif2jax's own ``expected_objective_value`` (NaN when unavailable).
    fstar_source
        ``"sif2jax"`` or ``"scipy-consensus"``.
    valid
        ``False`` when the catalogue validation found the instance
        unusable as converted (unbounded, infeasible, NaN at the start).
    """

    name: str
    collection: str
    n: int
    meq: int
    mineq: int
    nbounds: int
    y0_iD: int
    n_y0s: int
    has_fstar: bool
    fstar: float
    has_xstar: bool
    ineq_flip: str = ""
    fstar_sif: float = float("nan")
    fstar_source: str = "sif2jax"
    valid: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary."""
        return asdict(self)


def flip_signs(spec: str | None, mineq: int) -> np.ndarray:
    """Per-row sign multipliers that turn sif2jax inequalities into ``h <= 0``.

    Parameters
    ----------
    spec
        ``""``/``None`` (no row flipped), ``"all"`` or comma-separated
        zero-based indices of the rows that sif2jax already returns as
        ``<= 0`` quantities.
    mineq
        Number of inequality rows.

    Returns
    -------
    numpy.ndarray
        Vector of ``-1.0`` (documented ``c >= 0`` rows, ``h = -c``) and
        ``+1.0`` (flipped rows, ``h = c``) of length ``mineq``.

    Examples
    --------
    >>> from benchmarks.problems import flip_signs
    >>> flip_signs("", 3).tolist()
    [-1.0, -1.0, -1.0]
    >>> flip_signs("all", 2).tolist()
    [1.0, 1.0]
    >>> flip_signs("0,2", 3).tolist()
    [1.0, -1.0, 1.0]
    """
    signs = -np.ones(mineq)
    if _missing(spec):
        return signs
    spec = str(spec).strip()
    if not spec:
        return signs
    if spec == "all":
        return -signs
    idx = np.asarray([int(tok) for tok in spec.split(",") if tok.strip()], dtype=int)
    if idx.size and (idx.min() < 0 or idx.max() >= mineq):
        raise ValueError(f"ineq_flip {spec!r} out of range for mineq={mineq}")
    signs[idx] = 1.0
    return signs


def format_flip(flipped: np.ndarray, mineq: int) -> str:
    """Inverse of :func:`flip_signs`: encode the flipped rows as a spec string.

    Parameters
    ----------
    flipped
        Boolean mask (length ``mineq``) or integer indices of flipped rows.
    mineq
        Number of inequality rows.

    Returns
    -------
    str
        ``""``, ``"all"`` or the comma-separated indices.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems import format_flip
    >>> format_flip(np.array([True, False, True]), 3)
    '0,2'
    >>> format_flip(np.ones(2, dtype=bool), 2)
    'all'
    """
    arr = np.asarray(flipped)
    idx = np.flatnonzero(arr) if arr.dtype == bool else np.unique(arr.astype(int))
    if idx.size == 0:
        return ""
    if idx.size == mineq:
        return "all"
    return ",".join(str(int(i)) for i in idx)


OVERRIDE_COLUMNS: tuple[str, ...] = ("ineq_flip", "fstar", "fstar_source", "valid")
"""Catalogue columns :func:`to_sqpdax` reads back as instance overrides."""


@lru_cache(maxsize=4)
def _catalog_overrides(path: str) -> dict[tuple[str, int], dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path)
    present = [c for c in OVERRIDE_COLUMNS if c in frame.columns]
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not present or "fstar_source" not in frame.columns:
        # Catalogue not validated yet: nothing to override.
        return out
    for rec in frame[["name", "y0_iD", *present]].to_dict("records"):
        out[(str(rec["name"]), int(rec["y0_iD"]))] = {k: rec[k] for k in present}
    return out


def catalog_overrides(
    name: str, y0_iD: int, path: Path | None = None
) -> dict[str, Any]:
    """Validated overrides for one instance from the benchmark catalogue.

    Parameters
    ----------
    name
        CUTEst problem name.
    y0_iD
        Starting-point selector.
    path
        Catalogue CSV; defaults to :data:`benchmarks.catalog.CATALOG_PATH`.

    Returns
    -------
    dict[str, Any]
        Subset of :data:`OVERRIDE_COLUMNS` present in the catalogue for this
        instance; empty when the instance is unknown or the catalogue has
        not been validated (see :mod:`benchmarks.validate`).
    """
    if path is None:
        from .catalog import CATALOG_PATH

        path = CATALOG_PATH
    path = Path(path)
    if not path.exists():
        return {}
    return dict(_catalog_overrides(str(path)).get((name, int(y0_iD)), {}))


def collection_of(problem: Any) -> str:
    """Classify a sif2jax problem by its abstract base class.

    Parameters
    ----------
    problem
        A sif2jax problem instance.

    Returns
    -------
    str
        ``cqp``, ``bqp``, ``nle``, ``constrained``, ``bounded`` or
        ``unconstrained``. Quadratic sub-types are checked first because
        they subclass the general constrained / bounded bases.
    """
    import sif2jax

    if isinstance(problem, sif2jax.AbstractConstrainedQuadraticProblem):
        return "cqp"
    if isinstance(problem, sif2jax.AbstractBoundedQuadraticProblem):
        return "bqp"
    if isinstance(problem, sif2jax.AbstractNonlinearEquations):
        return "nle"
    if isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
        return "constrained"
    if isinstance(problem, sif2jax.AbstractBoundedMinimisation):
        return "bounded"
    return "unconstrained"


def constraint_sizes(problem: Any, y0: Array) -> tuple[int, int]:
    """Number of equality and inequality constraints of a sif2jax problem.

    Parameters
    ----------
    problem
        A sif2jax problem instance.
    y0
        A point at which ``constraint`` can be evaluated.

    Returns
    -------
    tuple[int, int]
        ``(meq, mineq)``; both zero when the problem has no ``constraint``.
    """
    if not hasattr(problem, "constraint"):
        return 0, 0
    eq, ineq = problem.constraint(y0)
    meq = 0 if eq is None else int(jnp.ravel(jnp.asarray(eq)).size)
    mineq = 0 if ineq is None else int(jnp.ravel(jnp.asarray(ineq)).size)
    return meq, mineq


def _missing(value: Any) -> bool:
    """``True`` for ``None`` and float NaN (an empty CSV cell)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _optional(getter):
    """Evaluate ``getter()`` and swallow ``NotImplementedError``."""
    try:
        return getter()
    except NotImplementedError:
        return None


def _bounds(problem: Any, n: int) -> tuple[Array | None, Array | None]:
    bounds = getattr(problem, "bounds", None)
    if bounds is None:
        return None, None
    lb, ub = bounds
    lb = jnp.ravel(jnp.asarray(lb, dtype=jnp.float64))
    ub = jnp.ravel(jnp.asarray(ub, dtype=jnp.float64))
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError(
            f"{problem.name}: bounds of shape {lb.shape}/{ub.shape} do not match n={n}"
        )
    return lb, ub


def to_sqpdax(
    problem: Any,
    y0_iD: int | None = None,
    *,
    ineq_flip: str | None = None,
    fstar: float | None = None,
    use_catalog: bool = True,
) -> tuple[Problem, Array, ProblemMeta]:
    """Convert a sif2jax problem into an sqpdax :class:`Problem`.

    Parameters
    ----------
    problem
        A sif2jax problem instance. If ``y0_iD`` is given and differs from
        ``problem.y0_iD`` the problem is re-instantiated with that id.
    y0_iD
        Optional starting-point selector among ``problem.provided_y0s``.
    ineq_flip
        Inequality rows to treat as ``<= 0`` quantities (see
        :func:`flip_signs`). ``None`` takes the catalogue value when
        ``use_catalog`` is set, otherwise no row is flipped.
    fstar
        Reference objective for scoring. ``None`` takes the catalogue value
        when ``use_catalog`` is set, otherwise sif2jax's
        ``expected_objective_value``.
    use_catalog
        Read the validated overrides from ``benchmarks/catalog.csv``
        (:func:`catalog_overrides`). The catalogue builder passes ``False``.

    Returns
    -------
    tuple[Problem, jax.Array, ProblemMeta]
        The sqpdax problem, its starting point (float64 vector) and the
        instance metadata.

    Examples
    --------
    >>> import sif2jax
    >>> from benchmarks.problems import to_sqpdax
    >>> prob, x0, meta = to_sqpdax(sif2jax.cutest.get_problem("HS71"))
    >>> (meta.n, meta.meq, meta.mineq, meta.nbounds)
    (4, 1, 1, 8)
    >>> bool(prob.ineq_fn(x0)[0] <= 0.0)  # h(x0) = 25 - prod(x0) <= 0
    True
    """
    if y0_iD is not None and y0_iD != problem.y0_iD:
        problem = type(problem)(y0_iD=y0_iD)

    x0 = jnp.ravel(jnp.asarray(problem.y0, dtype=jnp.float64))
    n = int(x0.size)
    meq, mineq = constraint_sizes(problem, x0)
    lb, ub = _bounds(problem, n)
    args = problem.args

    overrides: dict[str, Any] = {}
    if use_catalog and (ineq_flip is None or fstar is None):
        overrides = catalog_overrides(problem.name, int(problem.y0_iD))
    if ineq_flip is None:
        ineq_flip = overrides.get("ineq_flip", "")
    if not isinstance(ineq_flip, str):  # NaN from an empty CSV cell
        ineq_flip = ""
    signs = jnp.asarray(flip_signs(ineq_flip, mineq), dtype=jnp.float64)

    def fn(x: Array) -> Array:
        return jnp.asarray(problem.objective(x, args), dtype=x.dtype)

    eq_fn = ineq_fn = None
    if hasattr(problem, "constraint"):
        if meq:

            def eq_fn(x: Array) -> Array:
                return jnp.ravel(jnp.asarray(problem.constraint(x)[0], dtype=x.dtype))

        if mineq:

            def ineq_fn(x: Array) -> Array:
                # sif2jax: c(x) >= 0  ->  sqpdax: h(x) = -c(x) <= 0; rows the
                # catalogue flagged as raw L-type (already <= 0) keep their sign.
                c = jnp.ravel(jnp.asarray(problem.constraint(x)[1], dtype=x.dtype))
                return signs.astype(x.dtype) * c

    sq_problem = build_problem(
        fn,
        n=n,
        meq=meq,
        mineq=mineq,
        eq_fn=eq_fn,
        ineq_fn=ineq_fn,
        lb=lb,
        ub=ub,
        autodiff_mode="jax",
        force_hvp_in_jax_mode=True,
    )

    nbounds = 0
    if lb is not None and ub is not None:
        nbounds = int(jnp.sum(jnp.isfinite(lb)) + jnp.sum(jnp.isfinite(ub)))

    fstar_sif = _optional(lambda: problem.expected_objective_value)
    xstar = _optional(lambda: problem.expected_result)
    fstar_sif_value = (
        float("nan") if fstar_sif is None else float(np.asarray(fstar_sif))
    )

    fstar_source = "sif2jax"
    if fstar is None:
        fstar = fstar_sif_value
        cat_fstar = overrides.get("fstar")
        if not _missing(cat_fstar) and np.isfinite(float(cat_fstar)):
            fstar = float(cat_fstar)
            fstar_source = str(overrides.get("fstar_source", "sif2jax"))
    fstar_value = float(fstar)
    valid_raw = overrides.get("valid", True)
    valid = True if _missing(valid_raw) else bool(valid_raw)

    meta = ProblemMeta(
        name=problem.name,
        collection=collection_of(problem),
        n=n,
        meq=meq,
        mineq=mineq,
        nbounds=nbounds,
        y0_iD=int(problem.y0_iD),
        n_y0s=len(problem.provided_y0s),
        has_fstar=bool(np.isfinite(fstar_value)),
        fstar=fstar_value,
        has_xstar=xstar is not None,
        ineq_flip=ineq_flip,
        fstar_sif=fstar_sif_value,
        fstar_source=fstar_source,
        valid=valid,
    )
    return sq_problem, x0, meta
