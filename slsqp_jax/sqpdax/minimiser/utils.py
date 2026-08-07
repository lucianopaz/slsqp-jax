"""Option-key helpers for constrained-minimiser configuration bags."""

from dataclasses import fields
from typing import Any, cast

from ..registry import kind_family_field_names, static_field_names

__all__ = [
    "minimiser_option_keys",
    "solver_option_keys",
]


def minimiser_option_keys(cls: type) -> set[str]:
    """Recognised ``options['minimiser']`` keys for ``cls``.

    Includes static tunable Equinox fields plus any ``kind``-family fields
    (``secant`` / ``barrier_update``). ``options`` itself is a static field
    but is the bag being validated, so it is never returned as a key.

    Parameters
    ----------
    cls
        Concrete :class:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser`
        (or subclass) type.

    Returns
    -------
    set of str
        Allowed keys under ``options['minimiser']``.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.minimiser.base import CommonMinimiser
    >>> from slsqp_jax.sqpdax.minimiser.utils import minimiser_option_keys
    >>> sorted(minimiser_option_keys(CommonMinimiser))
    ['atol', 'min_steps', 'rtol', 'secant', 'secant_memory']
    """
    return (static_field_names(cls) | kind_family_field_names(cls)) - {"options"}


def solver_option_keys(cls: type) -> set[str]:
    """Recognised ``options['subproblem']`` keys for a subproblem-solver type.

    All dataclass fields of ``cls`` except ``lagrangian`` (injected per
    step, not user-set). Nested ``SubProblemSolver`` fields are validated
    recursively by
    :meth:`~slsqp_jax.sqpdax.minimiser.base.CommonMinimiser._validate_solver_options`.

    Parameters
    ----------
    cls
        A :class:`~slsqp_jax.sqpdax.subproblem.solver.base.SubProblemSolver`
        subclass.

    Returns
    -------
    set of str
        Allowed keys under ``options['subproblem']`` (and nested sections).

    Examples
    --------
    >>> from slsqp_jax.sqpdax.minimiser.utils import solver_option_keys
    >>> from slsqp_jax.sqpdax.subproblem.solver import ProjectedCGSubProblemSolver
    >>> "tol" in solver_option_keys(ProjectedCGSubProblemSolver)
    True
    >>> "lagrangian" in solver_option_keys(ProjectedCGSubProblemSolver)
    False
    """
    return {f.name for f in fields(cast(Any, cls))} - {"lagrangian"}
