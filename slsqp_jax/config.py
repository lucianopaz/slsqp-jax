"""Grouped configuration dataclasses for the :class:`SLSQP` solver.

The :class:`SLSQP` outer-loop solver previously exposed ~40 keyword
arguments in a single flat namespace.  This module groups them into
small, semantically related ``eqx.Module`` dataclasses so the user-facing
surface of :class:`SLSQP` collapses to a single ``config: SLSQPConfig``
field plus the constraint structure (functions, counts, bounds), the
optional derivative overrides, the optional pluggable inner solver, and
the verbose printer.

Example::

    from slsqp_jax import SLSQP, SLSQPConfig, ToleranceConfig, LBFGSConfig

    solver = SLSQP(
        eq_constraint_fn=eq_fn,
        n_eq_constraints=1,
        config=SLSQPConfig(
            tolerance=ToleranceConfig(rtol=1e-8, atol=1e-8, max_steps=200),
            lbfgs=LBFGSConfig(memory=20),
        ),
    )

The static / non-static distinction on each field mirrors the legacy
field-by-field annotations so JAX retracing behaviour is unchanged.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx


class ToleranceConfig(eqx.Module):
    """Outer-loop tolerances and iteration / divergence budgets.

    Attributes:
        rtol: Relative tolerance for the stationarity convergence check.
            ``||grad_L|| <= rtol * max(|L|, 1)``.  Default ``1e-6``.
        atol: Absolute tolerance for primal feasibility and a number of
            internal heuristic floors (steepest-descent fallback,
            adaptive CG tolerance floor, default proximal mu floor).
            Default ``1e-6``.
        max_steps: Maximum number of outer SQP iterations.  Default 100.
        min_steps: Minimum iterations before convergence is allowed.
            Prevents premature termination at trivial starting points.
            Default 1.
        stagnation_tol: Relative-improvement threshold for the
            merit-based stagnation counter.  Default ``1e-12``.
        divergence_factor: Best-iterate divergence rollback fires when
            the merit grows by more than
            ``divergence_factor * max(|best_merit|, 1)`` for
            ``divergence_patience`` consecutive steps.  Default ``10.0``.
        divergence_patience: Number of consecutive blow-up steps required
            before the divergence rollback latches.  Default 3.
    """

    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 100
    min_steps: int = 1
    stagnation_tol: float = 1e-12
    divergence_factor: float = 10.0
    divergence_patience: int = 3


class LBFGSConfig(eqx.Module):
    """L-BFGS Hessian-approximation parameters.

    Attributes:
        memory: Number of curvature pairs ``(s, y)`` stored in the
            ring buffer.  Default 10.
        damping_threshold: VARCHEN damping threshold applied to each
            curvature pair before storage; set to ``0.0`` to disable.
            Default ``0.2``.
        diag_floor: Lower clip for the per-variable secant diagonal
            ``B_0 = diag(d)``.  Default ``1e-4``.
        diag_ceil: Upper clip for the per-variable secant diagonal.
            Default ``1e6``.
    """

    memory: int = eqx.field(static=True, default=10)
    damping_threshold: float = 0.2
    diag_floor: float = 1e-4
    diag_ceil: float = 1e6


class LineSearchConfig(eqx.Module):
    """Backtracking L1-merit line-search parameters.

    Attributes:
        max_steps: Maximum number of backtracking iterations.  Default 20.
        armijo_c1: Armijo condition coefficient ``c_1``.  Default ``1e-4``.
        failure_patience: After this many consecutive line-search
            failures, the L-BFGS history is hard-reset to identity.
            Default 3.
    """

    max_steps: int = eqx.field(static=True, default=20)
    armijo_c1: float = 1e-4
    failure_patience: int = 3


class QPConfig(eqx.Module):
    """QP-subproblem parameters.

    Attributes:
        max_iter: Maximum number of active-set iterations per QP solve.
            Default 100.
        max_cg_iter: Maximum number of CG iterations per inner solve.
            Default 50.
        failure_patience: After this many consecutive QP failures, the
            L-BFGS history is hard-reset to identity.  Default 3.
        zero_step_patience: After this many consecutive iterations where
            the QP returns ``||d|| < atol`` and primal feasibility holds,
            convergence is declared via the guarded ``qp_kkt_success``
            disjunct.  Default 3.
        ping_pong_threshold: Threshold for the QP add/drop ping-pong
            short-circuit.  Default ``2**31 - 1`` (effectively disabled);
            opt in by setting to ``3``-``8`` on degenerate problems.
        mult_drop_floor: Floor on the negative-multiplier drop test
            inside the QP active-set loop.  Default ``1e-6``.
        cg_regularization: Minimum eigenvalue threshold ``delta**2`` for
            the CG curvature guard.  Default ``1e-6``.
        use_exact_hvp: When True, the QP inner CG uses the exact
            Lagrangian HVP (via AD) instead of the L-BFGS approximation.
            Default False.
    """

    max_iter: int = eqx.field(static=True, default=100)
    max_cg_iter: int = eqx.field(static=True, default=50)
    failure_patience: int = 3
    zero_step_patience: int = 3
    ping_pong_threshold: int = eqx.field(static=True, default=2**31 - 1)
    mult_drop_floor: float = eqx.field(static=True, default=1e-6)
    cg_regularization: float = 1e-6
    use_exact_hvp: bool = eqx.field(static=True, default=False)


class ProximalConfig(eqx.Module):
    """Adaptive proximal multiplier stabilization (sSQP, Wright 2002).

    Attributes:
        tau: Exponent in ``mu = clip(kkt_residual^tau, mu_min, mu_max)``.
            Must lie in the half-open interval ``[0, 1)``.  Set to ``0.0``
            to disable sSQP entirely (equality constraints are then
            enforced via direct null-space projection).  Default ``0.5``.
        mu_min: Floor on the adaptive proximal mu.  ``None`` resolves to
            ``ToleranceConfig.atol`` at runtime.  Default ``None``.
        mu_max: Ceiling on the adaptive proximal mu.  Default ``0.1``.
    """

    tau: float = 0.5
    mu_min: Optional[float] = None
    mu_max: float = eqx.field(static=True, default=0.1)


class PreconditionerConfig(eqx.Module):
    """QP-inner-solver preconditioner configuration.

    Attributes:
        enabled: Whether to use a preconditioner at all.  Default True.
        type: Either ``"lbfgs"`` (default) or ``"diagonal"``.  The
            diagonal estimator requires an exact HVP (set
            :attr:`QPConfig.use_exact_hvp` or provide ``obj_hvp_fn``).
        diagonal_n_probes: Number of Rademacher probes for the
            stochastic diagonal estimator.  Default 20.
    """

    enabled: bool = eqx.field(static=True, default=True)
    type: str = eqx.field(static=True, default="lbfgs")
    diagonal_n_probes: int = eqx.field(static=True, default=20)


class LPECAConfig(eqx.Module):
    """LPEC-A active-set identification (Oberlin & Wright, 2005).

    Attributes:
        method: One of ``"expand"`` (default), ``"lpeca_init"`` or
            ``"lpeca"``.  ``"expand"`` disables LPEC-A entirely.
        sigma: Threshold exponent (``sigma_bar`` in the paper).  Must
            lie in the open interval ``(0, 1)``.  Default ``0.9``.
        beta: Threshold scaling factor.  ``None`` resolves to
            ``1 / (m_ineq + n + m_eq)`` at runtime.  Default ``None``.
        use_lp: When True, solve the LPEC-A LP (via ``mpax.r2HPDHG``)
            for tighter multiplier estimates.  Requires ``mpax``.
            Default False.
        trust_threshold: Trust gate on ``rho_bar``.  When ``rho_bar``
            exceeds this value the prediction is replaced with an
            empty set.  Default ``1.0``.
        warmup_steps: The first ``warmup_steps`` outer SQP iterations
            bypass LPEC-A.  Default 3.
        predict_bounds: When True (default), extend the LPEC-A prediction
            to box constraints (warm-start the bound-fixing loop).
    """

    method: str = eqx.field(static=True, default="expand")
    sigma: float = eqx.field(static=True, default=0.9)
    beta: Optional[float] = eqx.field(static=True, default=None)
    use_lp: bool = eqx.field(static=True, default=False)
    trust_threshold: float = eqx.field(static=True, default=1.0)
    warmup_steps: int = eqx.field(static=True, default=3)
    predict_bounds: bool = eqx.field(static=True, default=True)


class AdaptiveCGConfig(eqx.Module):
    """Adaptive CG / inexact stationarity configuration.

    Attributes:
        enabled: When True, the CG convergence tolerance is adapted from
            the outer KKT residual (Eisenstat-Walker style).  Default
            False to preserve baseline behaviour.
        use_inexact_stationarity: When True, the projected-gradient norm
            from a noise-aware inner solver (e.g. :class:`HRInexactSTCG`)
            is added as a logical-OR disjunct to the classical
            stationarity test.  Default False.
    """

    enabled: bool = eqx.field(static=True, default=False)
    use_inexact_stationarity: bool = eqx.field(static=True, default=False)


class SLSQPConfig(eqx.Module):
    """Aggregate configuration for :class:`SLSQP`.

    Replaces the legacy 40+ flat keyword arguments with a small set of
    grouped sub-configs.  Pass directly to :class:`SLSQP` via the
    ``config=`` keyword.  All sub-configs default to their dataclass
    defaults so ``SLSQPConfig()`` reproduces the legacy default settings.
    """

    tolerance: ToleranceConfig = eqx.field(default_factory=ToleranceConfig)
    lbfgs: LBFGSConfig = eqx.field(default_factory=LBFGSConfig)
    line_search: LineSearchConfig = eqx.field(default_factory=LineSearchConfig)
    qp: QPConfig = eqx.field(default_factory=QPConfig)
    proximal: ProximalConfig = eqx.field(default_factory=ProximalConfig)
    preconditioner: PreconditionerConfig = eqx.field(
        default_factory=PreconditionerConfig
    )
    lpeca: LPECAConfig = eqx.field(default_factory=LPECAConfig)
    adaptive_cg: AdaptiveCGConfig = eqx.field(default_factory=AdaptiveCGConfig)


__all__ = [
    "ToleranceConfig",
    "LBFGSConfig",
    "LineSearchConfig",
    "QPConfig",
    "ProximalConfig",
    "PreconditionerConfig",
    "LPECAConfig",
    "AdaptiveCGConfig",
    "SLSQPConfig",
]
