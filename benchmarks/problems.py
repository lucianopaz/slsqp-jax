"""Adapt sif2jax problems to :class:`slsqp_jax.sqpdax.problem.Problem`.

sif2jax expresses constraints as ``constraint(y) -> (eq, ineq)`` with
``eq == 0`` and ``ineq >= 0``; sqpdax expects ``g(x) = 0`` and ``h(x) <= 0``.
The adapter negates the inequality block and builds the problem through
:func:`~slsqp_jax.sqpdax.problem.builder.build_problem` with
``autodiff_mode="jax"`` and ``force_hvp_in_jax_mode=True`` so gradients,
Jacobians and Hessian-vector products are all available (exact-curvature
configurations rely on the latter).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from slsqp_jax.sqpdax.problem import Problem, build_problem

__all__ = [
    "ProblemMeta",
    "collection_of",
    "constraint_sizes",
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
        Whether the reference objective value is known, and its value.
    has_xstar
        Whether a reference minimiser is known.
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

    def as_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary."""
        return asdict(self)


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
    problem: Any, y0_iD: int | None = None
) -> tuple[Problem, Array, ProblemMeta]:
    """Convert a sif2jax problem into an sqpdax :class:`Problem`.

    Parameters
    ----------
    problem
        A sif2jax problem instance. If ``y0_iD`` is given and differs from
        ``problem.y0_iD`` the problem is re-instantiated with that id.
    y0_iD
        Optional starting-point selector among ``problem.provided_y0s``.

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

    def fn(x: Array) -> Array:
        return jnp.asarray(problem.objective(x, args), dtype=x.dtype)

    eq_fn = ineq_fn = None
    if hasattr(problem, "constraint"):
        if meq:

            def eq_fn(x: Array) -> Array:
                return jnp.ravel(jnp.asarray(problem.constraint(x)[0], dtype=x.dtype))

        if mineq:

            def ineq_fn(x: Array) -> Array:
                # sif2jax: c(x) >= 0  ->  sqpdax: h(x) = -c(x) <= 0
                return -jnp.ravel(jnp.asarray(problem.constraint(x)[1], dtype=x.dtype))

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

    fstar = _optional(lambda: problem.expected_objective_value)
    xstar = _optional(lambda: problem.expected_result)
    fstar_value = float("nan") if fstar is None else float(np.asarray(fstar))

    meta = ProblemMeta(
        name=problem.name,
        collection=collection_of(problem),
        n=n,
        meq=meq,
        mineq=mineq,
        nbounds=nbounds,
        y0_iD=int(problem.y0_iD),
        n_y0s=len(problem.provided_y0s),
        has_fstar=fstar is not None and np.isfinite(fstar_value),
        fstar=fstar_value,
        has_xstar=xstar is not None,
    )
    return sq_problem, x0, meta
