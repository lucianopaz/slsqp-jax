"""Failure- and conditioning-driven escalation of secant resets."""

from typing import Self, cast

import jax
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from ..types import InitializableModule
from .base import Secant

__all__ = [
    "FailureRecoverySchedule",
    "SecantRecoveryState",
    "SecantResetSignals",
    "SecantResetPolicy",
]


class FailureRecoverySchedule(InitializableModule):
    """Ordered global failure thresholds for staged recovery.

    Attributes
    ----------
    soft, diagonal, identity
        Exact streak values at which the corresponding secant reset fires.
    fatal
        Global failure count at which recovery terminates, after an identity
        reset has already been attempted.
    """

    soft: int = field(static=True, default=1)
    diagonal: int = field(static=True, default=2)
    identity: int = field(static=True, default=3)
    fatal: int = field(static=True, default=4)

    def __check_init__(self) -> None:
        if not (0 < self.soft < self.diagonal < self.identity < self.fatal):
            raise ValueError(
                "failure recovery thresholds must satisfy "
                "0 < soft < diagonal < identity < fatal; got "
                f"({self.soft}, {self.diagonal}, {self.identity}, {self.fatal})"
            )


class SecantRecoveryState(Module):
    """Dynamic state of one monotone secant-recovery episode.

    Attributes
    ----------
    failure_streak
        Number of consecutive outer iterations on which at least one recovery
        channel fired. Multiple channels on one iteration count once.
    stage
        Strongest reset already applied in the current episode (``-1`` for
        none, ``0`` soft, ``1`` diagonal, ``2`` identity).
    fatal
        Whether the configured failure threshold was reached after identity
        recovery.
    """

    failure_streak: Int[Array, ""] = field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
    )
    stage: Int[Array, ""] = field(default_factory=lambda: jnp.asarray(-1, jnp.int32))
    fatal: Bool[Array, ""] = field(default_factory=lambda: jnp.asarray(False))


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
    The individual streaks identify the failure source for diagnostics only.
    :class:`SecantResetPolicy` collapses all nonzero channels into one global
    recovery event per outer iteration.
    """

    subproblem_streak: Int[Array, ""]
    step_streak: Int[Array, ""]
    model_streak: Int[Array, ""] = field(
        default_factory=lambda: jnp.asarray(0, jnp.int32)
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
    """Advance one global, monotone secant-recovery lifecycle.

    A recovery event occurs when any failure channel is nonzero or the
    approximation is ill-conditioned. Events from several channels on the
    same outer iteration increment the global streak only once. The default
    schedule applies soft, diagonal, and identity resets on failures 1, 2,
    and 3, then marks failure 4 fatal. A healthy iteration clears the episode.

    A stage is applied only if it is stronger than the stage already attempted
    in the current episode, so a failure in another channel cannot downgrade
    or repeat recovery.

    Attributes
    ----------
    enabled
        Master switch; ``False`` never resets.
    condition_threshold
        Condition-estimate threshold for the ill-conditioning trigger.
    condition_min_pairs
        Minimum stored pairs before the conditioning trigger may fire.
    recovery_schedule
        Global soft / diagonal / identity / fatal thresholds.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.secant import LBFGS, SecantResetPolicy, SecantResetSignals
    >>> policy = SecantResetPolicy()
    >>> signals = SecantResetSignals(
    ...     subproblem_streak=jnp.asarray(1), step_streak=jnp.asarray(0),
    ... )
    >>> state = SecantRecoveryState()
    >>> _, state, severity = policy.apply(LBFGS(n=2, memory=2), signals, state)
    >>> int(severity), int(state.failure_streak)
    (0, 1)
    >>> mixed = SecantResetSignals(
    ...     subproblem_streak=jnp.asarray(2), step_streak=jnp.asarray(1),
    ... )
    >>> _, state, severity = policy.apply(LBFGS(n=2, memory=2), mixed, state)
    >>> int(severity), int(state.failure_streak)
    (1, 2)
    >>> _, state, severity = policy.apply(
    ...     LBFGS(n=2, memory=2), SecantResetSignals.none(), state
    ... )
    >>> int(severity), int(state.failure_streak)
    (-1, 0)
    """

    enabled: bool = field(static=True, default=True)
    condition_threshold: float = field(static=True, default=1e6)
    condition_min_pairs: int = field(static=True, default=2)
    recovery_schedule: FailureRecoverySchedule = field(
        static=True, default_factory=FailureRecoverySchedule
    )

    def recovery_event(
        self, secant: Secant, signals: SecantResetSignals
    ) -> Bool[Array, ""]:
        """Whether this outer iteration advances global recovery.

        Parameters
        ----------
        secant
            Current approximation.
        signals
            Raw failure channels from the owning minimiser.

        Returns
        -------
        Bool[Array, ""]
            ``True`` for one global recovery event, regardless of how many
            channels fired.
        """
        if not self.enabled:
            return jnp.asarray(False)
        lower, upper = secant.curvature_bounds()
        condition = upper / jnp.maximum(lower, 1e-30)
        ill_conditioned = (secant.num_pairs >= self.condition_min_pairs) & (
            condition > self.condition_threshold
        )
        channel_failure = (
            (signals.subproblem_streak > 0)
            | (signals.step_streak > 0)
            | (signals.model_streak > 0)
        )
        return ill_conditioned | channel_failure

    def severity(
        self,
        secant: Secant,
        signals: SecantResetSignals,
        state: SecantRecoveryState,
    ) -> Int[Array, ""]:
        """Next stronger reset severity, or ``-1`` when none fires.

        Parameters
        ----------
        secant
            Current approximation.
        signals
            Failure streaks reported by the minimiser.
        state
            Current global recovery state.

        Returns
        -------
        Int[Array, ""]
            Reset severity to apply (negative for no reset).
        """
        none = jnp.asarray(-1, jnp.int32)
        if not self.enabled:
            return none
        event = self.recovery_event(secant, signals)
        streak = state.failure_streak + 1
        schedule = self.recovery_schedule
        requested = none
        requested = jnp.where(streak == schedule.soft, 0, requested)
        requested = jnp.where(streak == schedule.diagonal, 1, requested)
        requested = jnp.where(streak == schedule.identity, 2, requested)
        return jnp.where(event & (requested > state.stage), requested, none)

    def apply(
        self,
        secant: Secant,
        signals: SecantResetSignals,
        state: SecantRecoveryState,
    ) -> tuple[Secant, SecantRecoveryState, Int[Array, ""]]:
        """Advance recovery and reset ``secant`` at a new stage, if any.

        Parameters
        ----------
        secant
            Current approximation.
        signals
            Failure streaks reported by the minimiser.
        state
            Current global recovery state.

        Returns
        -------
        secant
            Reset approximation, or ``secant`` unchanged.
        state
            Advanced global recovery state.
        severity
            Applied severity (negative when no reset fired).
        """
        severity = self.severity(secant, signals, state)
        if not self.enabled:
            return secant, cast(SecantRecoveryState, SecantRecoveryState()), severity
        event = self.recovery_event(secant, signals)
        failure_streak = jnp.where(event, state.failure_streak + 1, 0).astype(jnp.int32)
        stage = jnp.where(event, jnp.maximum(state.stage, severity), -1).astype(
            jnp.int32
        )
        fatal = (
            event
            & (state.stage >= 2)
            & (failure_streak >= self.recovery_schedule.fatal)
        )
        next_state = cast(
            SecantRecoveryState,
            SecantRecoveryState(
                failure_streak=failure_streak,
                stage=stage,
                fatal=fatal,
            ),
        )
        new = jax.lax.cond(
            severity >= 0,
            lambda sev: secant.reset(sev),
            lambda _: secant,
            severity,
        )
        return new, next_state, severity
