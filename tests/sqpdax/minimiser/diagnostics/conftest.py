"""Fixtures for :mod:`slsqp_jax.sqpdax.minimiser.diagnostics` tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.active_set_prediction import LPECAPrediction
from slsqp_jax.sqpdax.minimiser.diagnostics import ActiveSetLineSearchDiagnostics


def make_prediction(
    *, valid: bool = True, capped: bool = False, n_bounds_prefixed: int = 0
) -> LPECAPrediction:
    """Minimal :class:`LPECAPrediction` carrying only the counted outcome fields."""
    return LPECAPrediction(
        active_set=None,  # type: ignore[arg-type]
        valid=jnp.asarray(valid),
        capped=jnp.asarray(capped),
        rho_bar=jnp.asarray(0.0),
        n_bounds_prefixed=jnp.asarray(n_bounds_prefixed, jnp.int32),
    )


@pytest.fixture
def zero_als_diagnostics() -> ActiveSetLineSearchDiagnostics:
    """Fresh all-zero active-set line-search carry."""
    return ActiveSetLineSearchDiagnostics.zero()
