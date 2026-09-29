"""Failure- and conditioning-driven escalation of secant resets."""

from typing import Self

import jax
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Int

from ..types import InitializableModule
from .base import Secant

__all__ = [
    "SecantResetSignals",
    "SecantResetPolicy",
]


class SecantResetSignals(Module):
    """Failure streaks a minimiser reports to :class:`SecantResetPolicy`.

    The policy is algorithm-agnostic: each minimiser maps its own failure
    counters onto the three channels below (e.g. the active-set line search
    reports QP failures as ``subproblem`` and line-search failures as
    ``step``; the trust-region interior-point loop reports model-quality
    stalls as ``model``).

    Attributes
    ----------
    subproblem_streak
        Consecutive subproblem (QP / KKT) failures, including this step.
    step_streak
        Consecutive globalisation (line-search / step-acceptance) failures.
    model_streak
        Consecutive model-quality stalls reported by trust-region
        minimisers: a collapsed radius while the KKT error is still above
        tolerance, or a rejected step whose actual / predicted reduction
        ratio is far below zero (the model predicted no decrease).
    subproblem_patience, step_patience, model_patience
        Streak length at which the patience-level reset fires (``≥ 1``).
    """

    subproblem_streak: Int[Array, ""]
    step_streak: Int[Array, ""]
    model_streak: Int[Array, ""] = field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    subproblem_patience: int = field(static=True, default=1)
    step_patience: int = field(static=True, default=1)
    model_patience: int = field(static=True, default=3)

    def __check_init__(self) -> None:
        if (
            self.subproblem_patience < 1
            or self.step_patience < 1
            or self.model_patience < 1
        ):
            raise ValueError(
                "reset patience must be at least 1; got "
                f"subproblem_patience={self.subproblem_patience}, "
                f"step_patience={self.step_patience}, "
                f"model_patience={self.model_patience}"
            )

    @classmethod
    def none(cls) -> Self:
        """Signals reporting no failures (conditioning-only resets).

        Returns
        -------
        Self
            Zero streaks.
        """
        zero = jnp.asarray(0, jnp.int32)
        return cls(subproblem_streak=zero, step_streak=zero, model_streak=zero)


class SecantResetPolicy(InitializableModule):
    """Choose when and how hard to reset a secant approximation.

    Mirrors the recovery cascade of the legacy solver:

    * the approximation is ill-conditioned — at least
      :attr:`condition_min_pairs` stored pairs and
      ``upper / lower > condition_threshold`` from
      :meth:`~slsqp_jax.sqpdax.secant.base.Secant.curvature_bounds` — fires
      :attr:`condition_severity`;
    * the first failure of a streak (``streak == 1``) fires
      :attr:`first_failure_severity`;
    * a streak reaching its patience fires :attr:`patience_severity`.

    The streak triggers apply identically to each of the three
    :class:`SecantResetSignals` channels (``subproblem``, ``step``,
    ``model``). When several triggers fire the strongest severity wins.
    Severities follow
    :meth:`~slsqp_jax.sqpdax.secant.base.Secant.reset` (for
    :class:`~slsqp_jax.sqpdax.secant.lbfgs.LBFGS`: ``0`` soft, ``1`` diagonal,
    ``2`` identity).

    Attributes
    ----------
    enabled
        Master switch; ``False`` never resets.
    condition_threshold
        Condition-estimate threshold for the ill-conditioning trigger.
    condition_min_pairs
        Minimum stored pairs before the conditioning trigger may fire.
    condition_severity, first_failure_severity, patience_severity
        Severity applied by each trigger.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.secant import LBFGS, SecantResetPolicy, SecantResetSignals
    >>> policy = SecantResetPolicy()
    >>> signals = SecantResetSignals(
    ...     subproblem_streak=jnp.asarray(3), step_streak=jnp.asarray(0),
    ...     subproblem_patience=3, step_patience=3,
    ... )
    >>> int(policy.severity(LBFGS(n=2, memory=2), signals))
    2
    >>> stalled = SecantResetSignals(
    ...     subproblem_streak=jnp.asarray(0), step_streak=jnp.asarray(0),
    ...     model_streak=jnp.asarray(1), model_patience=3,
    ... )
    >>> int(policy.severity(LBFGS(n=2, memory=2), stalled))
    0
    """

    enabled: bool = field(static=True, default=True)
    condition_threshold: float = field(static=True, default=1e6)
    condition_min_pairs: int = field(static=True, default=2)
    condition_severity: int = field(static=True, default=0)
    first_failure_severity: int = field(static=True, default=0)
    patience_severity: int = field(static=True, default=2)

    def severity(self, secant: Secant, signals: SecantResetSignals) -> Int[Array, ""]:
        """Strongest triggered severity, or ``-1`` when nothing fires.

        Parameters
        ----------
        secant
            Current approximation.
        signals
            Failure streaks reported by the minimiser.

        Returns
        -------
        Int[Array, ""]
            Reset severity to apply (negative for no reset).
        """
        none = jnp.asarray(-1, jnp.int32)
        if not self.enabled:
            return none
        lower, upper = secant.curvature_bounds()
        condition = upper / jnp.maximum(lower, 1e-30)
        ill_conditioned = (secant.num_pairs >= self.condition_min_pairs) & (
            condition > self.condition_threshold
        )
        triggers = [(ill_conditioned, self.condition_severity)]
        for streak, patience in (
            (signals.subproblem_streak, signals.subproblem_patience),
            (signals.step_streak, signals.step_patience),
            (signals.model_streak, signals.model_patience),
        ):
            triggers.append((streak == 1, self.first_failure_severity))
            triggers.append((streak >= patience, self.patience_severity))
        severity = none
        for fired, level in triggers:
            severity = jnp.where(fired, jnp.maximum(severity, level), severity)
        return severity

    def apply(
        self, secant: Secant, signals: SecantResetSignals
    ) -> tuple[Secant, Int[Array, ""]]:
        """Reset ``secant`` at the selected severity, if any.

        Parameters
        ----------
        secant
            Current approximation.
        signals
            Failure streaks reported by the minimiser.

        Returns
        -------
        secant
            Reset approximation, or ``secant`` unchanged.
        severity
            Applied severity (negative when no reset fired).
        """
        severity = self.severity(secant, signals)
        if not self.enabled:
            return secant, severity
        new = jax.lax.cond(
            severity >= 0,
            lambda sev: secant.reset(sev),
            lambda _: secant,
            severity,
        )
        return new, severity
