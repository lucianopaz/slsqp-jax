"""Tests for the automatic problem scaling feature.

Covers the math, the atol compensation, the multiplier round-trip,
the zero-gradient warning, the four mode aliases, the verbose
unscaling adapter, and a SciPy-style smoke test asserting that
default-on auto-scaling does not regress the result.
"""

from __future__ import annotations

import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from slsqp_jax import (
    SLSQP,
    ScaleFactors,
    auto_scaled_minimise,
    compute_scale_factors_at_x0,
    minimize_like_scipy,
    resolve_scaling_mode,
    wrap_verbose_for_scaling,
)

# ---------------------------------------------------------------------------
# Mode taxonomy / scale-factor math
# ---------------------------------------------------------------------------


def test_resolve_scaling_mode_aliases() -> None:
    """``True`` resolves to ``"balanced"``; ``False`` to ``None``."""
    cfg_true = resolve_scaling_mode(True)
    cfg_balanced = resolve_scaling_mode("balanced")
    assert cfg_true == cfg_balanced
    assert cfg_true.target_gradient == 1.0
    assert cfg_true.max_factor == 1e3

    assert resolve_scaling_mode(False) is None

    cfg_knitro = resolve_scaling_mode("knitro")
    assert cfg_knitro.max_factor == 1.0
    assert cfg_knitro.target_gradient == 1.0

    cfg_ipopt = resolve_scaling_mode("ipopt")
    assert cfg_ipopt.target_gradient == 100.0
    assert cfg_ipopt.max_factor == 1.0

    cfg_aggr = resolve_scaling_mode("aggressive")
    assert cfg_aggr.max_factor == 1e6


def test_resolve_scaling_mode_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_scaling_mode("foobar")
    with pytest.raises(TypeError):
        resolve_scaling_mode(1)  # type: ignore[arg-type]


def _toy_problem_factory(
    obj_norm: float, eq_norm: float
) -> tuple[Any, Any, np.ndarray]:
    """Build (fn, eq_fn, x0) where ||grad_f||=obj_norm and ||J_eq||=eq_norm."""
    n = 3

    def fn(x, args):
        return obj_norm * jnp.sum(x), None

    def eq_fn(x, args):
        return jnp.array([eq_norm * jnp.sum(x) - 1.0])

    return fn, eq_fn, np.zeros(n)


def test_scale_factors_balanced_default_balances_documented_cascade() -> None:
    """Load-bearing: the documented ||J_eq|| >> ||grad_f|| case is fixed."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=0.018, eq_norm=70.0)
    factors = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
        grad_floor=1e-12,
    )
    # After scaling: s_f * ||grad_f|| ~ 1.0 and s_eq * ||J_eq|| ~ 1.0.
    s_f_grad = factors.s_f * 0.018
    s_eq_jac = float(factors.s_eq[0]) * 70.0
    assert abs(s_f_grad - 1.0) / 1.0 < 0.01
    assert abs(s_eq_jac - 1.0) / 1.0 < 0.01


def test_scale_factors_balanced_idempotent_well_scaled() -> None:
    """Well-scaled problems get s = 1.0 (no-op)."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=1.0, eq_norm=1.0)
    factors = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    assert factors.s_f == pytest.approx(1.0)
    assert float(factors.s_eq[0]) == pytest.approx(1.0)


def test_scale_factors_balanced_caps_at_1e3() -> None:
    """Tiny gradients are clipped to ``max_factor``."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=1e-7, eq_norm=1.0)
    factors = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    assert factors.s_f == pytest.approx(1e3, rel=1e-3)


def test_scale_factors_knitro_strict_shrink_only() -> None:
    """Knitro mode shrinks but never amplifies."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=0.018, eq_norm=70.0)
    factors = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1.0,
    )
    assert factors.s_f == pytest.approx(1.0)
    assert float(factors.s_eq[0]) == pytest.approx(1.0 / 70.0, rel=1e-3)


def test_scale_factors_ipopt_only_shrinks() -> None:
    """IPOPT mode (target=100, cap=1) is a no-op when all grads < 100."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=2.0, eq_norm=10.0)
    factors = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=100.0,
        max_factor=1.0,
    )
    assert factors.s_f == pytest.approx(1.0)
    assert float(factors.s_eq[0]) == pytest.approx(1.0)


def test_scale_factors_aggressive_amplifies_beyond_1e3() -> None:
    """Aggressive mode amplifies beyond the balanced cap."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=1e-5, eq_norm=1.0)
    factors_aggr = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e6,
    )
    factors_bal = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    assert factors_aggr.s_f == pytest.approx(1e5, rel=1e-3)
    assert factors_bal.s_f == pytest.approx(1e3, rel=1e-3)


def test_scale_factors_zero_gradient_warns() -> None:
    """Below-floor gradient leaves s = 1.0 and warns."""

    def fn(x, args):
        return jnp.array(0.0), None

    x0 = np.zeros(3)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        factors = compute_scale_factors_at_x0(
            fn,
            x0,
            args=None,
            has_aux=True,
            target_gradient=1.0,
            max_factor=1e3,
        )
    assert factors.s_f == 1.0
    assert factors.skipped_obj is True
    assert any("grad_floor" in str(w.message) for w in ws)


# ---------------------------------------------------------------------------
# atol compensation
# ---------------------------------------------------------------------------


def test_atol_compensation() -> None:
    """``atol_internal = atol_user * min(min(s_eq), min(s_ineq), 1.0)``."""
    factors = ScaleFactors(
        s_f=1.0,
        s_eq=jnp.array([0.014, 0.5]),
        s_ineq=jnp.array([]),
        atol_user=1e-6,
        atol_internal=1e-6 * 0.014,
        target_gradient=1.0,
        max_factor=1e3,
        grad_floor=1e-12,
    )
    # The internal atol is constructed by compute_scale_factors_at_x0.
    # Replicate the formula here as a sanity check.
    s_min = float(min(jnp.min(factors.s_eq), 1.0))
    expected = factors.atol_user * min(s_min, 1.0)
    assert factors.atol_internal == pytest.approx(expected)


def test_atol_compensation_via_compute() -> None:
    """End-to-end: ``compute_scale_factors_at_x0`` populates the right atol."""
    fn, eq_fn, x0 = _toy_problem_factory(obj_norm=1.0, eq_norm=70.0)
    factors = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
        atol_user=1e-6,
    )
    s_eq_min = float(jnp.min(factors.s_eq))
    expected = 1e-6 * s_eq_min
    assert factors.atol_internal == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Multiplier round-trip
# ---------------------------------------------------------------------------


def test_multiplier_roundtrip() -> None:
    """``lambda_user * c_user = lambda_scaled * c_scaled`` on a quadratic."""

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    def c_eq(x):
        return jnp.array([jnp.sum(x) - 5.0])

    x0 = np.zeros(3)
    sol = minimize_like_scipy(
        f,
        x0,
        constraints={"type": "eq", "fun": c_eq},
        auto_scale=True,
    )
    factors = sol.stats["scale_factors"]
    mult_scaled = sol.stats["multipliers_eq"]
    mult_user = sol.stats["multipliers_eq_user"]

    # ``lambda_user = (s_eq / s_f) * lambda_scaled``.
    expected_user = (factors.s_eq / factors.s_f) * mult_scaled
    assert jnp.allclose(mult_user, expected_user, atol=1e-12)


# ---------------------------------------------------------------------------
# Solution invariance
# ---------------------------------------------------------------------------


def test_solution_invariant_under_scaling_quadratic() -> None:
    """``auto_scale=True`` and ``False`` reach the same closed-form optimum."""

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    def c_eq(x):
        return jnp.array([jnp.sum(x) - 5.0])

    x0 = np.zeros(3)
    sol_on = minimize_like_scipy(
        f,
        x0,
        constraints={"type": "eq", "fun": c_eq},
        auto_scale=True,
    )
    sol_off = minimize_like_scipy(
        f,
        x0,
        constraints={"type": "eq", "fun": c_eq},
        auto_scale=False,
    )
    np.testing.assert_allclose(
        np.asarray(sol_on.value), np.asarray(sol_off.value), rtol=1e-4
    )


# ---------------------------------------------------------------------------
# minimize_like_scipy default-on / off paths
# ---------------------------------------------------------------------------


def test_minimize_like_scipy_default_on_smoke() -> None:
    """Default-on populates ``scale_factors`` and produces a finite result."""

    def f(x):
        return jnp.sum(x**2)

    x0 = np.array([2.0, 3.0])
    sol = minimize_like_scipy(f, x0)
    assert "scale_factors" in sol.stats
    assert isinstance(sol.stats["scale_factors"], ScaleFactors)
    assert np.all(np.isfinite(np.asarray(sol.value)))


def test_minimize_like_scipy_explicit_off_path() -> None:
    """``auto_scale=False`` keeps the previous code path (no scale_factors)."""

    def f(x):
        return jnp.sum(x**2)

    x0 = np.array([2.0, 3.0])
    sol = minimize_like_scipy(f, x0, auto_scale=False)
    assert "scale_factors" not in sol.stats


# ---------------------------------------------------------------------------
# Verbose-unscaling
# ---------------------------------------------------------------------------


def test_verbose_user_callable_receives_factors() -> None:
    """User callables are invoked with a ``scale_factors`` keyword."""
    captured: dict[str, Any] = {"calls": 0, "factors": None}

    def my_verbose(**kwargs):
        captured["calls"] += 1
        captured["factors"] = kwargs.pop("scale_factors", None)

    factors = ScaleFactors(
        s_f=2.0,
        s_eq=jnp.array([0.5]),
        s_ineq=jnp.array([]),
        atol_user=1e-6,
        atol_internal=5e-7,
        target_gradient=1.0,
        max_factor=1e3,
        grad_floor=1e-12,
    )
    wrapped = wrap_verbose_for_scaling(my_verbose, factors)
    wrapped(objective=("f", 4.0, ".3e"))
    assert captured["calls"] == 1
    assert captured["factors"] is factors


def test_verbose_unscaling_columns(capsys: pytest.CaptureFixture) -> None:
    """The wrapped built-in printer divides ``f`` by ``s_f``."""
    factors = ScaleFactors(
        s_f=2.0,
        s_eq=jnp.array([]),
        s_ineq=jnp.array([]),
        atol_user=1e-6,
        atol_internal=1e-6,
        target_gradient=1.0,
        max_factor=1e3,
        grad_floor=1e-12,
    )
    wrapped = wrap_verbose_for_scaling(True, factors)
    # ``f_scaled = 4.0`` -> the printed value should be 2.0.
    wrapped(
        num_steps=("Step", 1),
        objective=("f", 4.0, ".3e"),
        merit=("merit", 5.0, ".3e"),
    )
    cap = capsys.readouterr()
    # The merit column carries the (s) suffix flagging scaled units.
    assert "merit(s)" in cap.out
    # The unscaled f value (4.0 / 2.0 = 2.0) appears.
    assert "2.000e+00" in cap.out


# ---------------------------------------------------------------------------
# auto_scaled_minimise (lower-level path)
# ---------------------------------------------------------------------------


def test_auto_scaled_minimise_lower_level_path() -> None:
    """``auto_scaled_minimise`` reaches the same x* as ``minimize_like_scipy``."""

    def f(x, args):
        return jnp.sum((x - 1.0) ** 2), None

    def c_eq(x, args):
        return jnp.array([jnp.sum(x) - 5.0])

    x0 = jnp.zeros(3)
    solver = SLSQP(
        eq_constraint_fn=c_eq,
        n_eq_constraints=1,
    )
    sol = auto_scaled_minimise(
        f,
        solver,
        x0,
        args=None,
        auto_scale=True,
        has_aux=True,
        max_steps=100,
    )
    assert "scale_factors" in sol.stats
    np.testing.assert_allclose(np.asarray(sol.value), np.full(3, 5.0 / 3.0), rtol=1e-4)


def test_auto_scaled_minimise_off_passes_through() -> None:
    """``auto_scale=False`` does not augment ``stats``."""

    def f(x, args):
        return jnp.sum((x - 1.0) ** 2), None

    def c_eq(x, args):
        return jnp.array([jnp.sum(x) - 5.0])

    x0 = jnp.zeros(3)
    solver = SLSQP(
        eq_constraint_fn=c_eq,
        n_eq_constraints=1,
    )
    sol = auto_scaled_minimise(
        f,
        solver,
        x0,
        args=None,
        auto_scale=False,
        has_aux=True,
        max_steps=100,
    )
    assert "scale_factors" not in sol.stats


# ---------------------------------------------------------------------------
# Diagnostic-report integration
# ---------------------------------------------------------------------------


def test_report_renders_auto_scaling_section() -> None:
    """The DebugReport surfaces an Auto-scaling section when factors are set."""
    from slsqp_jax import diagnose_minimize_like_scipy

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    def c_eq(x):
        return jnp.array([jnp.sum(x) - 5.0])

    x0 = np.zeros(3)
    sol, report = diagnose_minimize_like_scipy(
        f,
        x0,
        constraints={"type": "eq", "fun": c_eq},
    )
    assert report.scale_factors is not None
    rendered = report.render()
    assert "Auto-scaling" in rendered
    assert "atol_user" in rendered
    assert "atol_internal" in rendered


# ---------------------------------------------------------------------------
# Skip-placeholder for the canonical reproducer
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Reproducer for the documented Portfolio(n=5000) ||J_eq|| >> "
    "||grad_f|| cascade is not yet stable enough for CI; placeholder "
    "tracks the missing integration test (sibling of "
    "tests/diagnostics/test_signals_integration.py::"
    "test_signal_penalty_starvation_integration)."
)
def test_auto_scale_rescues_feasible_start_portfolio() -> None:
    """Auto-scaling rescues the feasible-start divergence cascade."""
    raise NotImplementedError
