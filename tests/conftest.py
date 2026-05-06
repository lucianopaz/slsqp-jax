"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from typing import Any

import pytest

from slsqp_jax import (
    SLSQP,
    AdaptiveCGConfig,
    LBFGSConfig,
    LineSearchConfig,
    LPECAConfig,
    PreconditionerConfig,
    ProximalConfig,
    QPConfig,
    SLSQPConfig,
    ToleranceConfig,
)

# Mapping from the legacy flat ``SLSQP`` kwargs to ``(sub_config_name,
# nested_field_name)``.  Only fields that moved into a sub-config are
# listed; fields that stay on ``SLSQP`` proper (constraint functions,
# ``inner_solver``, ``verbose``, derivative overrides, etc.) flow
# through unchanged.
_FLAT_TO_NESTED: dict[str, tuple[str, str]] = {
    # Tolerance
    "rtol": ("tolerance", "rtol"),
    "atol": ("tolerance", "atol"),
    "max_steps": ("tolerance", "max_steps"),
    "min_steps": ("tolerance", "min_steps"),
    "stagnation_tol": ("tolerance", "stagnation_tol"),
    "divergence_factor": ("tolerance", "divergence_factor"),
    "divergence_patience": ("tolerance", "divergence_patience"),
    # L-BFGS
    "lbfgs_memory": ("lbfgs", "memory"),
    "damping_threshold": ("lbfgs", "damping_threshold"),
    "lbfgs_diag_floor": ("lbfgs", "diag_floor"),
    "lbfgs_diag_ceil": ("lbfgs", "diag_ceil"),
    # Line search
    "line_search_max_steps": ("line_search", "max_steps"),
    "armijo_c1": ("line_search", "armijo_c1"),
    "ls_failure_patience": ("line_search", "failure_patience"),
    # QP
    "qp_max_iter": ("qp", "max_iter"),
    "qp_max_cg_iter": ("qp", "max_cg_iter"),
    "qp_failure_patience": ("qp", "failure_patience"),
    "zero_step_patience": ("qp", "zero_step_patience"),
    "qp_ping_pong_threshold": ("qp", "ping_pong_threshold"),
    "mult_drop_floor": ("qp", "mult_drop_floor"),
    "cg_regularization": ("qp", "cg_regularization"),
    "use_exact_hvp_in_qp": ("qp", "use_exact_hvp"),
    # Proximal
    "proximal_tau": ("proximal", "tau"),
    "proximal_mu_min": ("proximal", "mu_min"),
    "proximal_mu_max": ("proximal", "mu_max"),
    # Preconditioner
    "use_preconditioner": ("preconditioner", "enabled"),
    "preconditioner_type": ("preconditioner", "type"),
    "diagonal_n_probes": ("preconditioner", "diagonal_n_probes"),
    # LPEC-A
    "active_set_method": ("lpeca", "method"),
    "lpeca_sigma": ("lpeca", "sigma"),
    "lpeca_beta": ("lpeca", "beta"),
    "lpeca_use_lp": ("lpeca", "use_lp"),
    "lpeca_trust_threshold": ("lpeca", "trust_threshold"),
    "lpeca_warmup_steps": ("lpeca", "warmup_steps"),
    "lpeca_predict_bounds": ("lpeca", "predict_bounds"),
    # Adaptive CG
    "adaptive_cg_tol": ("adaptive_cg", "enabled"),
    "use_inexact_stationarity": ("adaptive_cg", "use_inexact_stationarity"),
}

_SUBCONFIG_FACTORIES = {
    "tolerance": ToleranceConfig,
    "lbfgs": LBFGSConfig,
    "line_search": LineSearchConfig,
    "qp": QPConfig,
    "proximal": ProximalConfig,
    "preconditioner": PreconditionerConfig,
    "lpeca": LPECAConfig,
    "adaptive_cg": AdaptiveCGConfig,
}


def _make_slsqp(**kwargs: Any) -> SLSQP:
    """Build an :class:`SLSQP` instance from legacy flat kwargs.

    Translates the legacy 40+ flat keyword arguments to the new
    nested :class:`SLSQPConfig` and forwards everything else
    (constraint functions, ``inner_solver``, ``verbose``, derivative
    overrides) directly to :class:`SLSQP`.

    Test files use this in place of the historical ``SLSQP(...)``
    constructor so the test bodies stay readable while exercising the
    canonical ``config=`` API.
    """
    if "config" in kwargs:
        return SLSQP(**kwargs)

    nested: dict[str, dict[str, Any]] = {}
    passthrough: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in _FLAT_TO_NESTED:
            section, field = _FLAT_TO_NESTED[key]
            nested.setdefault(section, {})[field] = value
        else:
            passthrough[key] = value

    sub_configs = {
        section: _SUBCONFIG_FACTORIES[section](**fields)
        for section, fields in nested.items()
    }
    config = SLSQPConfig(**sub_configs)
    return SLSQP(config=config, **passthrough)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Make ``very_slow`` imply ``slow`` so that ``-m "not slow"`` deselects both."""
    slow_marker = pytest.mark.slow
    for item in items:
        if item.get_closest_marker("very_slow") is not None:
            item.add_marker(slow_marker)
