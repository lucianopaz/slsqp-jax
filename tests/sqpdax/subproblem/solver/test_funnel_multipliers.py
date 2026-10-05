"""Unit tests for :mod:`slsqp_jax.sqpdax.subproblem.solver.funnel_multipliers`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.subproblem.solver import (
    LinearForcing,
    MultiplierCase,
    classify_multiplier_case,
    default_omega_n,
    default_omega_t,
    satisfies_forcing_condition,
)

EPS_PI, EPS_V, KAPPA_CHI = 1e-6, 1e-6, 0.1
OMEGA_T = LinearForcing(0.5)

# (pi_f, chi_f, pi_v, v) -> expected branch / predicates.  ``chi_f`` is chosen
# so that (3.15c) is decided independently of the first two tests.
CASES = {
    "terminate": dict(
        inputs=(1e-8, 1e-9, 0.0, 1e-8),
        case=MultiplierCase.terminate,
        kkt=True,
        dominates=False,
        cauchy=True,
    ),
    "terminate-beats-skip": dict(
        # πᶠ ≤ ε_π ∧ v ≤ ε_v *and* πᶠ ≤ ω_t(πᵛ): Algorithm 2 terminates first.
        inputs=(1e-8, 0.0, 1e-3, 1e-8),
        case=MultiplierCase.terminate,
        kkt=True,
        dominates=True,
        cauchy=False,
    ),
    "skip-tangential": dict(
        # Stationary enough relative to infeasibility but v is too large to stop.
        inputs=(0.2, 0.0, 1.0, 0.5),
        case=MultiplierCase.skip_tangential,
        kkt=False,
        dominates=True,
        cauchy=False,
    ),
    "tangential-with-cauchy": dict(
        inputs=(1.0, 0.5, 0.1, 0.5),
        case=MultiplierCase.tangential,
        kkt=False,
        dominates=False,
        cauchy=True,
    ),
    "tangential-without-cauchy": dict(
        # Neither gate holds and χᶠ < κ_χ πᶠ: the multiplier solve was too loose.
        inputs=(1.0, 0.01, 0.1, 0.5),
        case=MultiplierCase.tangential,
        kkt=False,
        dominates=False,
        cauchy=False,
    ),
    "small-pi-f-large-v": dict(
        # πᶠ ≤ ε_π but v > ε_v and πᵛ = 0 with πᶠ > 0: (3.15a)/(3.15b) fail.
        inputs=(1e-8, 1e-8, 0.0, 0.5),
        case=MultiplierCase.tangential,
        kkt=False,
        dominates=False,
        cauchy=True,
    ),
}


def _classify(inputs, jit: bool):
    pi_f, chi_f, pi_v, v = (jnp.asarray(x, jnp.float32) for x in inputs)

    def run(pi_f, chi_f, pi_v, v):
        return classify_multiplier_case(
            pi_f,
            chi_f,
            pi_v,
            v,
            eps_pi=EPS_PI,
            eps_v=EPS_V,
            kappa_chi=KAPPA_CHI,
            omega_t=OMEGA_T,
        )

    return (jax.jit(run) if jit else run)(pi_f, chi_f, pi_v, v)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("spec", CASES.values(), ids=CASES.keys())
def test_classification_follows_algorithm_2_precedence(spec, jit):
    """Branch = first of (3.15a), (3.15b) that holds; predicates reported raw."""
    out = _classify(spec["inputs"], jit)
    assert out.case == spec["case"]
    assert bool(out.kkt_satisfied) is spec["kkt"]
    assert bool(out.infeasibility_dominates) is spec["dominates"]
    assert bool(out.cauchy_satisfied) is spec["cauchy"]
    assert bool(out.acceptable) is (spec["kkt"] or spec["dominates"] or spec["cauchy"])
    # Branch is consistent with the predicates it is built from.
    if spec["kkt"]:
        assert out.case == MultiplierCase.terminate
    elif spec["dominates"]:
        assert out.case == MultiplierCase.skip_tangential
    else:
        assert out.case == MultiplierCase.tangential


def test_classification_rejects_bad_kappa_chi():
    with pytest.raises(ValueError, match="kappa_chi"):
        _ = classify_multiplier_case(
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            eps_pi=EPS_PI,
            eps_v=EPS_V,
            kappa_chi=1.0,
            omega_t=OMEGA_T,
        )


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0])
def test_linear_forcing_is_a_forcing_function(scale):
    """Continuous, strictly increasing, ``ω(0) = 0``."""
    omega = LinearForcing(scale)
    taus = jnp.linspace(0.0, 3.0, 7)
    values = jax.vmap(omega)(taus)
    assert values[0] == 0.0
    assert jnp.all(jnp.diff(values) > 0)
    assert jnp.allclose(values, scale * taus)


def test_linear_forcing_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale"):
        LinearForcing(0.0)


@pytest.mark.parametrize(
    "omega_n, omega_t, kappa_omega, expected",
    [
        (default_omega_n(), default_omega_t(), 0.25, True),  # c_n c_t = κ_ω exactly
        (default_omega_n(), default_omega_t(), 0.9, True),
        (default_omega_n(), default_omega_t(), 0.2, False),  # κ_ω below c_n c_t
        (LinearForcing(2.0), LinearForcing(0.6), 0.99, False),  # c_n c_t > 1
    ],
    ids=["defaults-tight", "defaults-loose", "defaults-fail", "product-above-one"],
)
def test_forcing_condition_3_16(omega_n, omega_t, kappa_omega, expected):
    """(3.16) ``ω_t(ω_n(τ)) ≤ κ_ω τ`` checked on a positive grid."""
    taus = jnp.array([1e-6, 1e-3, 1.0, 10.0])
    assert bool(satisfies_forcing_condition(omega_n, omega_t, kappa_omega, taus)) is (
        expected
    )


def test_forcing_condition_rejects_bad_kappa_omega():
    with pytest.raises(ValueError, match="kappa_omega"):
        satisfies_forcing_condition(
            default_omega_n(), default_omega_t(), 1.0, jnp.ones(1)
        )
