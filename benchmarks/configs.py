"""Registry of solver configurations benchmarked by default.

Each :class:`BenchConfig` pairs a minimiser factory with the ``options``
bag that :func:`slsqp_jax.sqpdax.minimiser.minimise` forwards to
``solver.init``; subproblem-solver variants (CRAIG projector, MINRES-QLP)
are selected there rather than through the minimiser constructor.

Two SciPy baselines (``scipy-slsqp``, ``scipy-trust-constr``) share the
registry. Their ``backend`` is ``"scipy"``: ``make_minimiser`` returns a
:class:`benchmarks.baselines.ScipyBaseline` and ``make_options`` the
``options`` mapping handed to :func:`scipy.optimize.minimize`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BenchConfig",
    "CONFIGS",
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    "get_configs",
]

DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-6


@dataclass(frozen=True)
class BenchConfig:
    """One benchmarked solver configuration.

    Attributes
    ----------
    name
        Short identifier used on the command line and in result rows.
    family
        Minimiser family (``asls``, ``pasls``, ``trip``, ``tfip``, or
        ``scipy`` for the baselines).
    subproblem
        Human-readable tag for the subproblem solver variant.
    curvature
        ``secant`` or ``exact``.
    make_minimiser
        Zero-argument factory returning a configured, un-initialised
        minimiser instance (a :class:`~benchmarks.baselines.ScipyBaseline`
        for the ``scipy`` backend).
    make_options
        Zero-argument factory returning the ``options`` mapping passed to
        :func:`~slsqp_jax.sqpdax.minimiser.minimise` (or to
        :func:`scipy.optimize.minimize`). A factory (rather than a stored
        mapping) keeps the module importable without JAX and avoids
        sharing Equinox modules across processes.
    backend
        ``"sqpdax"`` (default) or ``"scipy"``; selects the runner used by
        :mod:`benchmarks.worker`.
    """

    name: str
    family: str
    subproblem: str
    curvature: str
    make_minimiser: Callable[[], Any]
    make_options: Callable[[], Mapping[str, Any]] = field(default=dict)
    backend: str = "sqpdax"

    def tags(self) -> dict[str, str]:
        """Return the descriptive fields written into every result row."""
        return {
            "config": self.name,
            "family": self.family,
            "subproblem": self.subproblem,
            "curvature": self.curvature,
            "backend": self.backend,
        }


def _asls(curvature: str = "secant"):
    from slsqp_jax.sqpdax.minimiser import ActiveSetLineSearchMinimiser

    return ActiveSetLineSearchMinimiser(
        curvature=curvature, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


def _pasls():
    from slsqp_jax.sqpdax.minimiser import ProximalActiveSetLineSearchMinimiser

    return ProximalActiveSetLineSearchMinimiser(
        curvature="secant", rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


def _trip():
    # Interior-point minimisers only expose an absolute tolerance.
    from slsqp_jax.sqpdax.minimiser import TrustRegionInteriorPointMinimiser

    return TrustRegionInteriorPointMinimiser(curvature="secant", atol=DEFAULT_ATOL)


def _tfip():
    from slsqp_jax.sqpdax.minimiser import TrustFunnelInteriorPointMinimiser

    return TrustFunnelInteriorPointMinimiser(curvature="secant", atol=DEFAULT_ATOL)


def _craig_options() -> dict[str, Any]:
    from slsqp_jax.sqpdax.subproblem.solver import CraigProjector

    return {"subproblem": {"subproblem_solver": {"projector": CraigProjector()}}}


def _minres_qlp_options() -> dict[str, Any]:
    from slsqp_jax.sqpdax.subproblem.solver import MinresQLPSubProblemSolver

    return {"subproblem": {"subproblem_solver": MinresQLPSubProblemSolver()}}


def _scipy_slsqp():
    from .baselines import ScipyBaseline

    return ScipyBaseline("SLSQP")


def _scipy_trust_constr():
    from .baselines import ScipyBaseline

    return ScipyBaseline("trust-constr")


def _slsqp_options() -> dict[str, Any]:
    # SLSQP stops on the change of the objective; ``ftol`` is its only
    # tolerance, so it takes the same absolute value as the sqpdax configs.
    return {"ftol": DEFAULT_ATOL}


def _trust_constr_options() -> dict[str, Any]:
    # Stationarity tolerance aligned with the sqpdax configs; ``xtol`` and
    # ``barrier_tol`` keep SciPy's defaults (1e-8). Note that trust-constr
    # declares success as soon as the Lagrangian gradient is below ``gtol``,
    # whatever the current barrier parameter, so complementarity at its
    # returned point is O(mu) rather than O(gtol); the harness's ``feas`` /
    # ``f_gap`` checks decide whether such a point counts as solved.
    return {"gtol": DEFAULT_ATOL}


CONFIGS: dict[str, BenchConfig] = {
    cfg.name: cfg
    for cfg in (
        BenchConfig("asls-pcg", "asls", "projected-cg", "secant", _asls),
        BenchConfig(
            "asls-craig", "asls", "projected-cg+craig", "secant", _asls, _craig_options
        ),
        BenchConfig(
            "asls-minresqlp", "asls", "minres-qlp", "secant", _asls, _minres_qlp_options
        ),
        BenchConfig(
            "asls-pcg-exact",
            "asls",
            "projected-cg",
            "exact",
            lambda: _asls("exact"),
        ),
        BenchConfig("pasls", "pasls", "proximal-active-set", "secant", _pasls),
        BenchConfig("trip", "trip", "trust-region-ip", "secant", _trip),
        BenchConfig("tfip", "tfip", "trust-funnel-ip", "secant", _tfip),
        BenchConfig(
            "scipy-slsqp",
            "scipy",
            "slsqp",
            "secant",
            _scipy_slsqp,
            _slsqp_options,
            backend="scipy",
        ),
        BenchConfig(
            "scipy-trust-constr",
            "scipy",
            "trust-constr",
            "exact",
            _scipy_trust_constr,
            _trust_constr_options,
            backend="scipy",
        ),
    )
}
"""Default benchmark configurations keyed by name (sqpdax configs first, then the SciPy baselines)."""


def get_configs(names: list[str] | None = None) -> list[BenchConfig]:
    """Resolve configuration names to :class:`BenchConfig` instances.

    Parameters
    ----------
    names
        Configuration names; ``None`` or empty selects every default
        configuration.

    Returns
    -------
    list[BenchConfig]
        Configurations in the requested order.

    Raises
    ------
    KeyError
        If a name is not in :data:`CONFIGS`.

    Examples
    --------
    >>> from benchmarks.configs import get_configs
    >>> [c.name for c in get_configs()]  # doctest: +NORMALIZE_WHITESPACE
    ['asls-pcg', 'asls-craig', 'asls-minresqlp', 'asls-pcg-exact', 'pasls', 'trip', 'tfip',
     'scipy-slsqp', 'scipy-trust-constr']
    >>> get_configs(["tfip"])[0].family
    'tfip'
    >>> get_configs(["scipy-slsqp"])[0].backend
    'scipy'
    """
    if not names:
        return list(CONFIGS.values())
    missing = [n for n in names if n not in CONFIGS]
    if missing:
        raise KeyError(f"unknown config(s) {missing}; known: {list(CONFIGS)}")
    return [CONFIGS[n] for n in names]
