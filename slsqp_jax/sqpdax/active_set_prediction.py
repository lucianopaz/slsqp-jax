"""LPEC-A active-set prediction for the active-set line-search minimisers.

Implements the LPEC-A identification test of Oberlin & Wright (2005,
§3.3): a proximity measure ``ρ̄`` is built from the constraint values, the
multiplier estimates and the Lagrangian stationarity residual, and every
inequality (general or bound) within ``(β ρ̄)^σ`` of its boundary is
predicted active. Under MFCQ and second-order sufficiency the prediction
is asymptotically exact, so seeding the QP working set with it removes the
combinatorial add / drop iterations near the solution.

Far from the solution the raw threshold is meaningless, so the predictor
layers three guards on top of the test:

* a **trust gate** (``rho_bar > trust_threshold`` empties the prediction);
* a **warm-up** (the first ``warmup_steps`` outer iterations are skipped);
* a **rank-aware cap** keeping at most ``n - meq - 1`` predicted rows so the
  working-set Jacobian ``[A_eq; A_active]`` retains a LICQ-like margin.

Optionally the multipliers are refined by the LPEC-A linear programme
(Eq. 42) solved with ``mpax``.

Sign convention: ``sqpdax`` treats ``h(x) <= 0`` as feasible (the paper's
convention), with bounds written as ``lb - x <= 0`` and ``x - ub <= 0``.

References
----------
Oberlin, C. & Wright, S. J. (2006). Active set identification in nonlinear
programming. *SIAM Journal on Optimization*, 17(2), 577-605.
"""

from __future__ import annotations

from typing import Literal, cast

import equinox as eqx
import jax
from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

from .active_set import ActiveSet
from .dual import Dual
from .lagrangian import EvaluatedLagrangian
from .primal import Primal
from .types import InitializableModule, Scalar

__all__ = [
    "LPECAPrediction",
    "LPECAPredictor",
    "solve_lpeca_lp",
]

ActiveSetMethod = Literal["expand", "lpeca_init", "lpeca"]


class LPECAPrediction(Module):
    """Outcome of :meth:`LPECAPredictor.predict`.

    Attributes
    ----------
    active_set
        Predicted working set. All-inactive when ``valid`` is ``False``.
    valid
        ``True`` when the prediction is used: the trust gate passed
        (``rho_bar <= trust_threshold``) *and* the warm-up is over.
    capped
        ``True`` when more rows passed the threshold test than the
        rank-aware cap allows, so only the most violated ones were kept.
    rho_bar
        Proximity measure ``ρ̄`` (Eq. 36) at the current point.
    n_bounds_prefixed
        Number of bounds (lower plus upper) in ``active_set``.
    """

    active_set: ActiveSet
    valid: Bool[Array, ""]
    capped: Bool[Array, ""]
    rho_bar: Scalar
    n_bounds_prefixed: Int[Array, ""]


def _stack_inequalities(
    lag: EvaluatedLagrangian[Primal],
) -> tuple[Array, Bool[Array, " mineq+2n"]]:
    """Stack ``h(x)``, ``lb - x`` and ``x - ub`` with a presence mask."""
    x = lag.x_ref
    values = jnp.concatenate(
        [
            lag.ineq_fn_val,
            jnp.where(lag.null_lb, 0.0, lag.lb - x),
            jnp.where(lag.null_ub, 0.0, x - lag.ub),
        ]
    )
    present = jnp.concatenate([jnp.ones(lag.mineq, bool), ~lag.null_lb, ~lag.null_ub])
    return values, present


def _stack_multipliers(dual: Dual) -> Array:
    """Stack the inequality and bound multipliers in the same order."""
    return jnp.concatenate(
        [dual.ineq_multipliers, dual.lb_multipliers, dual.ub_multipliers]
    )


def compute_rho_bar(lag: EvaluatedLagrangian[Primal]) -> Scalar:
    """LPEC-A proximity measure ``ρ̄`` (Oberlin & Wright, Eq. 36).

    ```
    ρ̄ = Σ_i φ_i + ‖c_eq‖₁ + ‖∇_x L‖₁,
    φ_i = sqrt(max(-h_i λ_i, 0))  if h_i < 0 (feasible),  φ_i = h_i  otherwise
    ```

    where ``i`` runs over the general inequalities and the finite bounds,
    and ``λ`` are the multipliers carried by ``lag.dual``.

    Parameters
    ----------
    lag
        Lagrangian evaluated at the current iterate **and** current
        multiplier estimates (``lag.dual``).

    Returns
    -------
    Scalar
        ``ρ̄ >= 0``; zero exactly at a KKT point.
    """
    values, present = _stack_inequalities(lag)
    multipliers = _stack_multipliers(lag.dual)
    feasible = values < 0.0
    complementarity = jnp.where(
        feasible, jnp.sqrt(jnp.maximum(-values * multipliers, 0.0)), values
    )
    ineq_term = jnp.sum(jnp.where(present, complementarity, 0.0))
    eq_term = jnp.sum(jnp.abs(lag.eq_fn_val))
    stationarity_term = jnp.sum(jnp.abs(lag.x_grad))
    return ineq_term + eq_term + stationarity_term


def solve_lpeca_lp(
    lag: EvaluatedLagrangian[Primal],
    *,
    lambda_bound: float = 1e6,
    eps: float = 1e-6,
    max_iter: int = 1000,
) -> Dual:
    """Refine the multiplier estimates with the LPEC-A LP (Eq. 42).

    Solves

    ```
    min_{λ, μ, u, v}  Σ_{feasible i} (-h_i) λ_i + eᵀu + eᵀv
    s.t.              ∇f + J_eqᵀ μ + Jᵀ λ = u - v,   0 <= λ <= K,   u, v >= 0
    ```

    where ``J`` stacks the general-inequality Jacobian with the ``∓I`` bound
    rows, using ``mpax``'s reflected restarted Halpern PDHG (``r2HPDHG``).

    Parameters
    ----------
    lag
        Lagrangian evaluated at the current iterate (its ``dual`` is ignored).
    lambda_bound
        Upper bound ``K`` on the inequality / bound multipliers.
    eps
        Absolute and relative LP tolerance.
    max_iter
        LP iteration limit.

    Returns
    -------
    Dual
        LP-optimal multipliers (zero on absent bounds).

    Raises
    ------
    ImportError
        If ``mpax`` is not installed (``uv sync --group extras``).
    """
    try:
        from mpax import create_lp, r2HPDHG
    except ImportError:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "LPECAPredictor(use_lp=True) requires the optional 'mpax' package; "
            "install it with `pip install slsqp-jax[extras]` or "
            "`uv sync --group extras`."
        ) from None

    n, meq, mineq = lag.n, lag.meq, lag.mineq
    dtype = lag.x_ref.dtype
    values, present = _stack_inequalities(lag)
    m_all = values.shape[0]

    # Costs: slack magnitude on feasible rows, nothing on violated ones.
    cost_lambda = jnp.where(present & (values < 0.0), -values, 0.0)
    cost = jnp.concatenate(
        [cost_lambda, jnp.zeros(meq, dtype), jnp.ones(n, dtype), jnp.ones(n, dtype)]
    )
    # Stationarity: J_ineqᵀ λ + J_eqᵀ μ - λ_lb + λ_ub - u + v = -∇f.
    eye = jnp.eye(n, dtype=dtype)
    a_eq = jnp.concatenate(
        [lag.ineq_fn_jac_val.T, -eye, eye, lag.eq_fn_jac_val.T, -eye, eye], axis=1
    )
    b_eq = -lag.grad_val
    lower = jnp.concatenate(
        [
            jnp.zeros(m_all, dtype),
            jnp.full(meq, -jnp.inf, dtype),
            jnp.zeros(2 * n, dtype),
        ]
    )
    upper = jnp.concatenate(
        [
            jnp.where(present, lambda_bound, 0.0).astype(dtype),
            jnp.full(meq, jnp.inf, dtype),
            jnp.full(2 * n, jnp.inf, dtype),
        ]
    )
    n_vars = cost.shape[0]
    lp = create_lp(
        c=cost,
        A=a_eq,
        b=b_eq,
        G=jnp.zeros((0, n_vars), dtype),
        h=jnp.zeros(0, dtype),
        l=lower,
        u=upper,
    )
    solver = r2HPDHG(
        eps_abs=eps, eps_rel=eps, iteration_limit=max_iter, jit=True, verbose=False
    )
    z = solver.optimize(lp).primal_solution
    lam = jnp.where(present, z[:m_all], 0.0)
    return cast(
        Dual,
        Dual(
            eq_multipliers=z[m_all : m_all + meq],
            ineq_multipliers=lam[:mineq],
            lb_multipliers=lam[mineq : mineq + n],
            ub_multipliers=lam[mineq + n :],
        ),
    )


class LPECAPredictor(InitializableModule):
    """LPEC-A working-set predictor with trust gate, warm-up and rank cap.

    All fields are static so the predictor can live on the minimiser (a
    ``while_loop`` carry) and be configured through
    ``options['minimiser']['active_set_predictor']``.

    Attributes
    ----------
    method
        ``"expand"`` (default) disables prediction; ``"lpeca_init"`` seeds
        the QP working set and keeps the policy's EXPAND ramp;
        ``"lpeca"`` seeds the QP and forces ``expand_factor = 0`` on a
        :class:`~slsqp_jax.sqpdax.subproblem.solver.working_set_policy.ThresholdWorkingSetPolicy`.
    sigma
        Threshold exponent ``σ ∈ (0, 1)``.
    beta
        Threshold scale; ``None`` resolves to ``1 / (mineq + n + meq)``.
    trust_threshold
        Largest ``ρ̄`` for which the prediction is trusted.
    warmup_steps
        Outer iterations skipped before predicting.
    predict_bounds
        Include finite bounds in the predicted set.
    use_lp
        Refine the multipliers with :func:`solve_lpeca_lp` (needs ``mpax``).
    lp_lambda_bound, lp_eps, lp_max_iter
        Parameters forwarded to :func:`solve_lpeca_lp`.

    Examples
    --------
    >>> from slsqp_jax.sqpdax.active_set_prediction import LPECAPredictor
    >>> predictor = LPECAPredictor().init(method="lpeca", warmup_steps=0)
    >>> predictor.enabled, predictor.disables_expand
    (True, True)
    """

    method: ActiveSetMethod = eqx.field(static=True, default="expand")
    sigma: float = eqx.field(static=True, default=0.9)
    beta: float | None = eqx.field(static=True, default=None)
    trust_threshold: float = eqx.field(static=True, default=1.0)
    warmup_steps: int = eqx.field(static=True, default=3)
    predict_bounds: bool = eqx.field(static=True, default=True)
    use_lp: bool = eqx.field(static=True, default=False)
    lp_lambda_bound: float = eqx.field(static=True, default=1e6)
    lp_eps: float = eqx.field(static=True, default=1e-6)
    lp_max_iter: int = eqx.field(static=True, default=1000)

    def __check_init__(self) -> None:
        """Validate the method name and the threshold parameters."""
        if self.method not in ("expand", "lpeca_init", "lpeca"):
            raise ValueError(
                "method must be one of 'expand', 'lpeca_init', 'lpeca'; "
                f"got {self.method!r}"
            )
        if not 0.0 < self.sigma < 1.0:
            raise ValueError(f"sigma must lie in (0, 1); got {self.sigma}")
        if self.beta is not None and self.beta <= 0.0:
            raise ValueError(f"beta must be positive or None; got {self.beta}")
        if self.trust_threshold < 0.0:
            raise ValueError(
                f"trust_threshold must be non-negative; got {self.trust_threshold}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative; got {self.warmup_steps}"
            )

    @property
    def enabled(self) -> bool:
        """Whether :meth:`predict` should seed the QP (``method != "expand"``)."""
        return self.method != "expand"

    @property
    def disables_expand(self) -> bool:
        """Whether the EXPAND ramp must be switched off (``method == "lpeca"``)."""
        return self.method == "lpeca"

    def predict(
        self, lag: EvaluatedLagrangian[Primal], step_count: Int[Array, ""] | int
    ) -> LPECAPrediction:
        """Predict the active set at ``lag``'s reference point.

        Parameters
        ----------
        lag
            Lagrangian evaluated at the current iterate with the current
            multiplier estimates in ``lag.dual`` (replaced by the LP solution
            when ``use_lp`` is set).
        step_count
            Outer iteration counter, compared against ``warmup_steps``.

        Returns
        -------
        LPECAPrediction
            Predicted working set plus diagnostics.
        """
        if self.use_lp:
            lag = eqx.tree_at(
                lambda lagrangian: lagrangian.dual,
                lag,
                solve_lpeca_lp(
                    lag,
                    lambda_bound=self.lp_lambda_bound,
                    eps=self.lp_eps,
                    max_iter=self.lp_max_iter,
                ),
            )

        n, meq, mineq = lag.n, lag.meq, lag.mineq
        values, present = _stack_inequalities(lag)
        if not self.predict_bounds:
            present = present & jnp.concatenate(
                [jnp.ones(mineq, bool), jnp.zeros(2 * n, bool)]
            )
        rho_bar = compute_rho_bar(lag)
        beta = 1.0 / max(mineq + n + meq, 1) if self.beta is None else self.beta

        in_warmup = jnp.asarray(step_count) < self.warmup_steps
        valid = (rho_bar <= self.trust_threshold) & ~in_warmup
        threshold = jnp.where(valid, (beta * rho_bar) ** self.sigma, 0.0)
        # Eq. 43 in the ``h <= 0`` convention: within ``threshold`` of the
        # boundary, on either side.
        raw = present & valid & (values >= -threshold)

        # Rank-aware cap: keep the ``n_dof`` most violated rows so that
        # ``[A_eq; A_active]`` keeps a LICQ-like rank margin. Static
        # ``top_k`` avoids materialising a rank vector.
        m_all = values.shape[0]
        n_dof = min(max(n - meq - 1, 1), m_all)
        capped = jnp.asarray(False)
        predicted = raw
        if m_all > 0:
            scores = jnp.where(raw, values, -jnp.inf)
            _, top = jax.lax.top_k(scores, n_dof)
            selected = jnp.zeros_like(raw).at[top.astype(jnp.int32)].set(True)
            predicted = raw & selected
            capped = jnp.sum(raw.astype(jnp.int32)) > n_dof

        active_ineq = predicted[:mineq]
        active_lb = predicted[mineq : mineq + n]
        active_ub = predicted[mineq + n :] & ~active_lb
        n_bounds = jnp.sum(active_lb.astype(jnp.int32)) + jnp.sum(
            active_ub.astype(jnp.int32)
        )
        return cast(
            LPECAPrediction,
            LPECAPrediction(
                active_set=cast(
                    ActiveSet,
                    ActiveSet(
                        meq=meq,
                        active_inequalities=active_ineq,
                        active_lb=active_lb,
                        active_ub=active_ub,
                    ),
                ),
                valid=valid,
                capped=capped & valid,
                rho_bar=rho_bar,
                n_bounds_prefixed=n_bounds.astype(jnp.int32),
            ),
        )
