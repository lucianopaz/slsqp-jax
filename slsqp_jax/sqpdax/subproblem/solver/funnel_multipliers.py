"""Multiplier acceptance tests (3.15) and forcing functions of the trust-funnel method.

Curtis, Gould, Robinson & Toint (2017) accept an inexact multiplier estimate
``y_k`` — through the f-criticality measures ``πᶠ`` and ``χᶠ`` it induces —
as soon as one of three conditions holds:

```
(3.15a)  πᶠ ≤ ε_π  and  v ≤ ε_v        → approximate KKT point: terminate
(3.15b)  πᶠ ≤ ω_t(πᵛ)                 → infeasibility dominates: t_k ← 0
(3.15c)  χᶠ ≥ κ_χ πᶠ                  → a Cauchy decrease exists: tangential step
```

Algorithm 2 checks them in this order, so the *case* reported here is the
first that holds; (3.15c) is reported separately because the orchestrator
needs it to decide whether the multiplier solve was accurate enough.

The forcing functions ``ω_n`` (normal-step gate (3.2)) and ``ω_t`` (3.15b)
must satisfy the compatibility condition (3.16),
``ω_t(ω_n(τ)) ≤ κ_ω τ`` for all ``τ ≥ 0`` with ``κ_ω ∈ (0, 1)``; the linear
defaults ``ω(τ) = c τ`` satisfy it exactly whenever ``c_n c_t ≤ κ_ω``.
"""

from abc import abstractmethod
from typing import cast

from equinox import Enumeration, Module, field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Scalar

__all__ = [
    "MultiplierCase",
    "MultiplierClassification",
    "classify_multiplier_case",
    "ForcingFunction",
    "LinearForcing",
    "default_omega_n",
    "default_omega_t",
    "satisfies_forcing_condition",
]


class MultiplierCase(Enumeration):
    """Which acceptance branch of (3.15) the multiplier estimate falls into."""

    terminate = "(3.15a): πᶠ ≤ ε_π and v ≤ ε_v; approximate KKT point of the barrier subproblem."
    skip_tangential = "(3.15b): πᶠ ≤ ω_t(πᵛ); the tangential step is set to zero."
    tangential = "Neither (3.15a) nor (3.15b): a tangential step is computed."


class MultiplierClassification(Module):
    """Outcome of :func:`classify_multiplier_case`.

    Attributes
    ----------
    case
        First branch of (3.15) that holds, in Algorithm 2 order.
    kkt_satisfied
        (3.15a) ``πᶠ ≤ ε_π and v ≤ ε_v``.
    infeasibility_dominates
        (3.15b) ``πᶠ ≤ ω_t(πᵛ)``.
    cauchy_satisfied
        (3.15c) ``χᶠ ≥ κ_χ πᶠ``.
    acceptable
        Any of the three holds (Lemma 3.8: an accurate enough multiplier
        solve always achieves this). When ``False`` the orchestrator should
        improve the multiplier estimate before proceeding.
    """

    case: MultiplierCase
    kkt_satisfied: Bool[Array, ""]
    infeasibility_dominates: Bool[Array, ""]
    cauchy_satisfied: Bool[Array, ""]
    acceptable: Bool[Array, ""]


class ForcingFunction(Module):
    """Continuous, strictly increasing ``ω : [0, ∞) → [0, ∞)`` with ``ω(0) = 0``."""

    @abstractmethod
    def __call__(self, tau: Scalar) -> Scalar:
        """Evaluate ``ω(τ)`` for ``τ ≥ 0``.

        Parameters
        ----------
        tau
            Nonnegative argument (a criticality measure).

        Returns
        -------
        Scalar
            ``ω(τ)``.
        """


class LinearForcing(ForcingFunction):
    """``ω(τ) = scale · τ`` with ``scale > 0``.

    Attributes
    ----------
    scale
        Positive slope.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.subproblem.solver import LinearForcing
    >>> float(LinearForcing(0.5)(jnp.asarray(2.0)))
    1.0
    """

    scale: float = field(static=True, default=0.5)

    def __check_init__(self):
        if not self.scale > 0.0:
            raise ValueError(f"LinearForcing.scale must be positive; got {self.scale}")

    def __call__(self, tau: Scalar) -> Scalar:
        return self.scale * tau


def default_omega_n() -> LinearForcing:
    """Default normal-step gate ``ω_n(τ) = τ / 2`` used in (3.2)."""
    return cast(LinearForcing, LinearForcing(0.5))


def default_omega_t() -> LinearForcing:
    """Default tangential gate ``ω_t(τ) = τ / 2`` used in (3.15b).

    Together with :func:`default_omega_n` this gives
    ``ω_t(ω_n(τ)) = τ / 4``, so (3.16) holds for any ``κ_ω ≥ 1/4``.
    """
    return cast(LinearForcing, LinearForcing(0.5))


def satisfies_forcing_condition(
    omega_n: ForcingFunction,
    omega_t: ForcingFunction,
    kappa_omega: float,
    taus: Float[Array, " k"],
) -> Bool[Array, ""]:
    """Check (3.16) ``ω_t(ω_n(τ)) ≤ κ_ω τ`` on a grid of arguments.

    Parameters
    ----------
    omega_n, omega_t
        Forcing functions of the normal gate (3.2) and the multiplier test
        (3.15b).
    kappa_omega
        Constant ``κ_ω ∈ (0, 1)`` of (3.16).
    taus
        Nonnegative sample points. For linear forcing functions a single
        positive point is exact; general functions are only checked on the
        samples.

    Returns
    -------
    Bool[Array, ""]
        ``True`` when the inequality holds at every sample.

    Raises
    ------
    ValueError
        If ``kappa_omega`` is outside ``(0, 1)``.
    """
    if not 0.0 < kappa_omega < 1.0:
        raise ValueError(f"kappa_omega must lie in (0, 1); got {kappa_omega}")
    taus = jnp.asarray(taus)
    composed = jnp.vectorize(lambda t: omega_t(omega_n(t)))(taus)
    return jnp.all(composed <= kappa_omega * taus)


def classify_multiplier_case(
    pi_f: Scalar,
    chi_f: Scalar,
    pi_v: Scalar,
    v: Scalar,
    *,
    eps_pi: Scalar | float,
    eps_v: Scalar | float,
    kappa_chi: float,
    omega_t: ForcingFunction,
) -> MultiplierClassification:
    """Evaluate the three acceptance tests (3.15) and pick the Algorithm 2 branch.

    Parameters
    ----------
    pi_f
        f-criticality ``πᶠ_k(y_k)`` (3.14a).
    chi_f
        Cauchy-angle measure ``χᶠ_k(y_k)`` (3.14b).
    pi_v
        v-criticality ``πᵛ_k`` (3.1a).
    v
        Constraint violation ``v_k``.
    eps_pi
        Stationarity tolerance ``ε_π`` of the current barrier subproblem.
    eps_v
        Feasibility tolerance ``ε_v`` of the current barrier subproblem.
    kappa_chi
        Constant ``κ_χ ∈ (0, 1)`` of (3.15c).
    omega_t
        Forcing function of (3.15b).

    Returns
    -------
    MultiplierClassification
        Branch and the three individual predicates.

    Raises
    ------
    ValueError
        If ``kappa_chi`` is outside ``(0, 1)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.subproblem.solver import (
    ...     LinearForcing, MultiplierCase, classify_multiplier_case,
    ... )
    >>> out = classify_multiplier_case(
    ...     pi_f=jnp.asarray(1e-9), chi_f=jnp.asarray(1e-9),
    ...     pi_v=jnp.asarray(0.0), v=jnp.asarray(0.0),
    ...     eps_pi=1e-6, eps_v=1e-6, kappa_chi=0.1, omega_t=LinearForcing(0.5),
    ... )
    >>> bool(out.case == MultiplierCase.terminate), bool(out.acceptable)
    (True, True)
    """
    if not 0.0 < kappa_chi < 1.0:
        raise ValueError(f"kappa_chi must lie in (0, 1); got {kappa_chi}")
    kkt = (pi_f <= eps_pi) & (v <= eps_v)
    dominates = pi_f <= omega_t(pi_v)
    cauchy = chi_f >= kappa_chi * pi_f
    case = MultiplierCase.where(
        kkt,
        MultiplierCase.terminate,
        MultiplierCase.where(
            dominates, MultiplierCase.skip_tangential, MultiplierCase.tangential
        ),
    )
    return cast(
        MultiplierClassification,
        MultiplierClassification(
            case=case,
            kkt_satisfied=kkt,
            infeasibility_dominates=dominates,
            cauchy_satisfied=cauchy,
            acceptable=kkt | dominates | cauchy,
        ),
    )
