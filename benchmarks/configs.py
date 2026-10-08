"""Registry of solver configurations benchmarked by default.

Each :class:`BenchConfig` pairs a minimiser factory with the ``options``
bag that :func:`slsqp_jax.sqpdax.minimiser.minimise` forwards to
``solver.init``; subproblem-solver variants (CRAIG projector, MINRES-QLP)
are selected there rather than through the minimiser constructor.
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
        Minimiser family (``asls``, ``pasls``, ``trip``, ``tfip``).
    subproblem
        Human-readable tag for the subproblem solver variant.
    curvature
        ``secant`` or ``exact``.
    make_minimiser
        Zero-argument factory returning a configured, un-initialised
        minimiser instance.
    make_options
        Zero-argument factory returning the ``options`` mapping passed to
        :func:`~slsqp_jax.sqpdax.minimiser.minimise`. A factory (rather
        than a stored mapping) keeps the module importable without JAX
        and avoids sharing Equinox modules across processes.
    """

    name: str
    family: str
    subproblem: str
    curvature: str
    make_minimiser: Callable[[], Any]
    make_options: Callable[[], Mapping[str, Any]] = field(default=dict)

    def tags(self) -> dict[str, str]:
        """Return the descriptive fields written into every result row."""
        return {
            "config": self.name,
            "family": self.family,
            "subproblem": self.subproblem,
            "curvature": self.curvature,
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
    )
}
"""Default benchmark configurations keyed by name."""


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
    >>> [c.name for c in get_configs()]
    ['asls-pcg', 'asls-craig', 'asls-minresqlp', 'asls-pcg-exact', 'pasls', 'trip', 'tfip']
    >>> get_configs(["tfip"])[0].family
    'tfip'
    """
    if not names:
        return list(CONFIGS.values())
    missing = [n for n in names if n not in CONFIGS]
    if missing:
        raise KeyError(f"unknown config(s) {missing}; known: {list(CONFIGS)}")
    return [CONFIGS[n] for n in names]
