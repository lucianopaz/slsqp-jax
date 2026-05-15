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
    compute_uniform_scale_factors_at_x0,
    minimize_like_scipy,
    resolve_scaling_mode,
    wrap_verbose_for_scaling,
)

# ---------------------------------------------------------------------------
# Mode taxonomy / scale-factor math
# ---------------------------------------------------------------------------


def test_resolve_scaling_mode_aliases() -> None:
    """``True`` resolves to ``"uniform"``; ``False`` to ``None``;
    per-row modes remain available via their string aliases."""
    cfg_true = resolve_scaling_mode(True)
    cfg_uniform = resolve_scaling_mode("uniform")
    assert cfg_true == cfg_uniform
    assert cfg_true.target_gradient == 1.0
    assert cfg_true.max_factor == 1e3
    assert cfg_true.uniform is True

    cfg_balanced = resolve_scaling_mode("balanced")
    assert cfg_balanced.target_gradient == 1.0
    assert cfg_balanced.max_factor == 1e3
    assert cfg_balanced.uniform is False
    # Sanity: balanced and uniform differ only in the ``uniform`` flag
    # at default settings (same target/max_factor/grad_floor).
    assert cfg_true != cfg_balanced

    assert resolve_scaling_mode(False) is None

    cfg_knitro = resolve_scaling_mode("knitro")
    assert cfg_knitro.max_factor == 1.0
    assert cfg_knitro.target_gradient == 1.0
    assert cfg_knitro.uniform is False

    cfg_ipopt = resolve_scaling_mode("ipopt")
    assert cfg_ipopt.target_gradient == 100.0
    assert cfg_ipopt.max_factor == 1.0
    assert cfg_ipopt.uniform is False

    cfg_aggr = resolve_scaling_mode("aggressive")
    assert cfg_aggr.max_factor == 1e6
    assert cfg_aggr.uniform is False


def test_resolve_scaling_mode_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_scaling_mode("foobar")
    with pytest.raises(TypeError):
        resolve_scaling_mode(1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Uniform-mode unit tests (compute_uniform_scale_factors_at_x0)
# ---------------------------------------------------------------------------


def _uniform_problem(
    eq_rows: list[float] | None = None,
    ineq_rows: list[float] | None = None,
    *,
    obj_grad_norm: float = 1.0,
    n: int = 3,
):
    """Build a toy uniform-test problem with one row per requested norm.

    Each constraint row ``r`` evaluates to ``r * x[0] - 1.0`` (linear in
    ``x[0]``), so its Jacobian row is ``[r, 0, ..., 0]`` and the
    inf-norm of that row is exactly ``|r|``.  This gives us tight
    control over the per-row gradient inf-norm sequence the
    ``compute_uniform_scale_factors_at_x0`` helper sees.
    """
    obj_grad_norm = float(obj_grad_norm)

    def fn(x, args):
        return obj_grad_norm * jnp.sum(x), None

    eq_fn = None
    if eq_rows:
        eq_rows_arr = jnp.asarray(eq_rows, dtype=float)

        def eq_fn(x, args):
            return eq_rows_arr * x[0] - 1.0

    ineq_fn = None
    if ineq_rows:
        ineq_rows_arr = jnp.asarray(ineq_rows, dtype=float)

        def ineq_fn(x, args):
            return ineq_rows_arr * x[0] - 1.0

    x0 = np.zeros(n)
    return fn, eq_fn, ineq_fn, x0


def test_uniform_mode_single_scalar_across_constraints() -> None:
    """One row eq + one row ineq with disparate norms get the same ``s_c``."""
    fn, eq_fn, ineq_fn, x0 = _uniform_problem(eq_rows=[1e2], ineq_rows=[1e6])
    factors = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        ineq_constraint_fn=ineq_fn,
        target_gradient=1.0,
        max_factor=1e3,
        atol_user=1e-6,
    )
    assert factors.uniform is True
    assert factors.s_eq.shape == (1,)
    assert factors.s_ineq.shape == (1,)
    s_c_eq = float(factors.s_eq[0])
    s_c_ineq = float(factors.s_ineq[0])
    assert s_c_eq == s_c_ineq
    # max row norm = 1e6, target = 1, raw = 1e-6, clipped at 1/max_factor = 1e-3
    np.testing.assert_allclose(s_c_eq, 1e-3)


def test_uniform_mode_cross_group_max_driven_by_larger_group() -> None:
    """``s_c`` is driven by the larger of eq/ineq row norms (symmetric)."""
    # Case 1: ineq is larger.
    fn, eq_fn, ineq_fn, x0 = _uniform_problem(eq_rows=[1e2], ineq_rows=[1e6])
    fac_a = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        ineq_constraint_fn=ineq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    # Case 2: eq is larger (symmetric mirror).
    fn, eq_fn, ineq_fn, x0 = _uniform_problem(eq_rows=[1e6], ineq_rows=[1e2])
    fac_b = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        ineq_constraint_fn=ineq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    # Both should produce the same s_c (max(eq, ineq) is symmetric).
    np.testing.assert_allclose(float(fac_a.s_eq[0]), float(fac_b.s_eq[0]))


def test_uniform_mode_preserves_row_ratios() -> None:
    """Every eq/ineq row shares the same ``s_c``; ratios survive scaling."""
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e2, 1e4, 1e0])
    factors = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e6,  # generous cap so no clipping fires
    )
    s_eq = np.asarray(factors.s_eq)
    assert s_eq.shape == (3,)
    np.testing.assert_allclose(s_eq, s_eq[0])
    # Post-scaling Jacobian row ratios equal pre-scaling ratios.
    pre = np.array([1e2, 1e4, 1e0])
    post = pre * float(s_eq[0])
    np.testing.assert_allclose(post / post[0], pre / pre[0])


def test_uniform_mode_atol_can_exceed_user_atol() -> None:
    """When ``s_c > 1`` (tiny constraint gradient), ``atol_internal > atol_user``."""
    # ||grad c|| = 1e-4 -> raw s_c = 1e4 -> clipped at max_factor = 1e3.
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e-4])
    factors = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
        atol_user=1e-6,
    )
    s_c = float(factors.s_eq[0])
    assert s_c > 1.0
    assert factors.atol_internal > factors.atol_user
    np.testing.assert_allclose(factors.atol_internal, s_c * factors.atol_user)


def test_uniform_mode_symmetric_max_factor_clipping() -> None:
    """Both upper and lower clipping bounds fire under uniform mode."""
    # Lower clip (huge constraint gradient): raw s_c < 1/max_factor.
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e9])
    f_lo = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    np.testing.assert_allclose(float(f_lo.s_eq[0]), 1e-3)

    # Upper clip (tiny constraint gradient): raw s_c > max_factor.
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e-9])
    f_hi = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    np.testing.assert_allclose(float(f_hi.s_eq[0]), 1e3)


def test_uniform_mode_max_factor_one_disables_scaling_and_warns() -> None:
    """``max_factor=1.0`` collapses the symmetric interval and emits a warning."""
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e6])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        factors = compute_uniform_scale_factors_at_x0(
            fn,
            x0,
            args=None,
            has_aux=True,
            eq_constraint_fn=eq_fn,
            target_gradient=1.0,
            max_factor=1.0,
            atol_user=1e-6,
        )
        msgs = [str(wi.message) for wi in w]
    # s_f = 1, s_c = 1, atol_internal = atol_user
    np.testing.assert_allclose(factors.s_f, 1.0)
    np.testing.assert_allclose(float(factors.s_eq[0]), 1.0)
    np.testing.assert_allclose(factors.atol_internal, factors.atol_user)
    # Single UserWarning naming the alternative.
    assert any("max_factor=1.0 disables scaling" in m and "knitro" in m for m in msgs)


def test_uniform_mode_max_factor_below_one_raises() -> None:
    """``max_factor < 1.0`` raises ValueError naming the alternative."""
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e6])
    with pytest.raises(ValueError, match="knitro"):
        compute_uniform_scale_factors_at_x0(
            fn,
            x0,
            args=None,
            has_aux=True,
            eq_constraint_fn=eq_fn,
            max_factor=0.5,
        )


def test_uniform_mode_constraint_jacobian_below_floor_warns_once() -> None:
    """All-degenerate Jacobian: one warning, ``s_c = 1.0``."""
    # Three eq rows all at gradient inf-norm 1e-15 (below 1e-12 default floor).
    fn, eq_fn, _, x0 = _uniform_problem(eq_rows=[1e-15, 5e-16, 1e-14])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        factors = compute_uniform_scale_factors_at_x0(
            fn,
            x0,
            args=None,
            has_aux=True,
            eq_constraint_fn=eq_fn,
            target_gradient=1.0,
            max_factor=1e3,
            grad_floor=1e-12,
        )
        msgs = [str(wi.message) for wi in w]
    # All rows share s_c = 1 (no per-row skipping under uniform).
    np.testing.assert_allclose(np.asarray(factors.s_eq), np.ones(3))
    # Single grad_floor warning, not three.
    floor_msgs = [m for m in msgs if "all constraint Jacobian row inf-norms" in m]
    assert len(floor_msgs) == 1


def test_uniform_mode_no_constraints_is_no_op() -> None:
    """No eq/ineq inputs: ``s_c = 1``, ``atol_internal = atol_user``."""
    fn, _, _, x0 = _uniform_problem()  # eq_rows=ineq_rows=None
    factors = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        target_gradient=1.0,
        max_factor=1e3,
        atol_user=1e-6,
    )
    assert factors.uniform is True
    assert factors.s_eq.size == 0
    assert factors.s_ineq.size == 0
    np.testing.assert_allclose(factors.atol_internal, factors.atol_user)


def test_balanced_mode_still_per_row_after_default_change() -> None:
    """Explicit ``auto_scale='balanced'`` reproduces the old per-row behavior.

    Load-bearing: after the default-mode flip, users who explicitly opt
    into ``"balanced"`` must still get the documented cascade-fixing
    behavior (each row's gradient inf-norm gets driven to
    ``target_gradient``).
    """
    fn, eq_fn, ineq_fn, x0 = _uniform_problem(eq_rows=[70.0], ineq_rows=[0.018])
    cfg = resolve_scaling_mode("balanced")
    assert cfg.uniform is False
    factors_b = compute_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        ineq_constraint_fn=ineq_fn,
        target_gradient=cfg.target_gradient,
        max_factor=cfg.max_factor,
        grad_floor=cfg.grad_floor,
    )
    # Per-row balancing: each row hit ~ target/norm.
    np.testing.assert_allclose(float(factors_b.s_eq[0]), 1.0 / 70.0, rtol=1e-2)
    np.testing.assert_allclose(float(factors_b.s_ineq[0]), 1.0 / 0.018, rtol=1e-2)

    # Same problem under uniform: both rows share the *same* s_c
    # (driven by max(70, 0.018) = 70), so the ratio 70/0.018 ≈ 3889 is
    # preserved post-scaling.
    factors_u = compute_uniform_scale_factors_at_x0(
        fn,
        x0,
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        ineq_constraint_fn=ineq_fn,
        target_gradient=1.0,
        max_factor=1e3,
    )
    s_c = float(factors_u.s_eq[0])
    assert float(factors_u.s_ineq[0]) == s_c  # cross-group uniform
    post_ratio = (70.0 * s_c) / (0.018 * s_c)
    pre_ratio = 70.0 / 0.018
    np.testing.assert_allclose(post_ratio, pre_ratio)


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
    """Default-on populates ``scale_factors`` and produces a finite result.

    Under the new default the populated ``ScaleFactors`` is in uniform
    mode (``uniform=True``).  We additionally pin a constrained problem
    so ``s_eq`` is non-empty and confirm that it is constant-valued
    (the load-bearing uniform-mode invariant).
    """

    def f(x):
        return jnp.sum(x**2)

    x0 = np.array([2.0, 3.0])
    sol = minimize_like_scipy(f, x0)
    assert "scale_factors" in sol.stats
    factors = sol.stats["scale_factors"]
    assert isinstance(factors, ScaleFactors)
    assert factors.uniform is True
    assert np.all(np.isfinite(np.asarray(sol.value)))

    # Constrained variant: confirm s_eq / s_ineq are constant-valued.
    def c_eq(x):
        return jnp.array([jnp.sum(x) - 5.0])

    def c_ineq(x):
        return jnp.array([x[0] - 0.1, x[1] - 0.2])

    sol2 = minimize_like_scipy(
        f,
        x0,
        constraints=(
            {"type": "eq", "fun": c_eq},
            {"type": "ineq", "fun": c_ineq},
        ),
        options={"max_steps": 100},
    )
    factors2 = sol2.stats["scale_factors"]
    assert factors2.uniform is True
    s_eq = np.asarray(factors2.s_eq)
    s_ineq = np.asarray(factors2.s_ineq)
    # All eq rows share the same s_c.
    if s_eq.size > 0:
        np.testing.assert_allclose(s_eq, s_eq[0])
    # All ineq rows share the same s_c.
    if s_ineq.size > 0:
        np.testing.assert_allclose(s_ineq, s_ineq[0])
    # Eq and ineq share the *same* s_c (cross-group uniform).
    if s_eq.size > 0 and s_ineq.size > 0:
        assert float(s_eq[0]) == float(s_ineq[0])


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


def test_verbose_adapter_under_jit_with_sf_equal_one() -> None:
    """Regression: the verbose adapter must not concretise ``s_eq`` / ``s_ineq``.

    The verbose callback runs inside the ``jax.jit``-compiled
    ``SLSQP.step`` (Optimistix's outer driver and ``debug_run``'s
    inner ``jit_step`` both jit it).  Earlier the ``(s)`` suffix
    decision in ``_adapt_entry`` did ``float(jnp.min(factors.s_eq))``
    and ``float(jnp.min(factors.s_ineq))`` lazily, guarded by
    ``factors.s_f != 1.0`` short-circuit.  When ``s_f == 1.0``
    (``auto_scale="knitro"``, ``"ipopt"``, or any
    ``auto_scale_max_factor=1.0`` configuration on a problem with
    ``||grad_f||_inf < target_gradient``), the short-circuit failed
    open and the concrete reads triggered ``ConcretizationTypeError``.

    The fix precomputes the bool eagerly in
    ``wrap_verbose_for_scaling``; this test pins that contract by
    invoking the wrapper on a tracer-typed value with ``s_f == 1.0``
    and non-trivial ``s_eq`` / ``s_ineq``.
    """
    import jax

    factors = ScaleFactors(
        s_f=1.0,
        s_eq=jnp.array([0.5]),
        s_ineq=jnp.array([0.25, 2.0]),
        atol_user=1e-6,
        atol_internal=2.5e-7,
        target_gradient=1.0,
        max_factor=1.0,
        grad_floor=1e-12,
    )
    captured: dict[str, Any] = {}

    def collector(**kwargs: tuple) -> None:
        captured.update(kwargs)

    wrapped = wrap_verbose_for_scaling(collector, factors)

    @jax.jit
    def call_under_jit(merit_value: jax.Array) -> jax.Array:
        wrapped(merit=("merit", merit_value, ".3e"))
        return merit_value + 1.0

    out = call_under_jit(jnp.asarray(3.5))
    assert float(out) == pytest.approx(4.5)
    assert captured["scale_factors"] is factors
    assert captured["merit"][0] == "merit"


def test_minimize_like_scipy_max_factor_one_runs_under_jit() -> None:
    """End-to-end smoke test for ``auto_scale_max_factor=1.0`` under the
    legacy per-row ``"balanced"`` mode (``s_f=1``).

    Reproducer for the ConcretizationTypeError observed when the
    verbose adapter was called from inside Optimistix's jitted
    iteration loop with ``s_f`` clipped exactly to ``1.0``.

    Note: under the new default ``auto_scale=True`` (uniform mode),
    ``max_factor=1.0`` collapses the symmetric clipping interval to
    ``[1, 1]`` and disables scaling entirely (with a UserWarning).
    This test pins ``auto_scale="balanced"`` so the original
    "shrink-only" intent of ``max_factor=1.0`` is preserved.
    """

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    def c_eq(x):
        return jnp.array([jnp.sum(x) - 5.0])

    def c_ineq(x):
        return jnp.array([x[0] - 0.1])

    x0 = jnp.zeros(3)
    sol = minimize_like_scipy(
        fun=f,
        x0=x0,
        constraints=(
            {"type": "eq", "fun": c_eq},
            {"type": "ineq", "fun": c_ineq},
        ),
        auto_scale="balanced",
        auto_scale_max_factor=1.0,
        options={"max_steps": 100},
    )
    np.testing.assert_allclose(np.asarray(sol.value), np.full(3, 5.0 / 3.0), rtol=1e-4)


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


# ---------------------------------------------------------------------------
# User-unit scaling stats (final_objective_user, final_lagrangian_grad_norm_user,
# multipliers_*_qp_user)
# ---------------------------------------------------------------------------


def test_scaled_stats_user_unit_objective() -> None:
    """``final_objective_user = final_objective / s_f`` under auto-scaling."""

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
    s_f = float(factors.s_f)

    assert "final_objective_user" in sol.stats
    np.testing.assert_allclose(
        float(sol.stats["final_objective_user"]),
        float(sol.stats["final_objective"]) / s_f,
        rtol=1e-12,
    )


def test_scaled_stats_user_unit_lagrangian_grad_norm() -> None:
    """``final_lagrangian_grad_norm_user = ||grad_L||_scaled / s_f``."""

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
    s_f = float(factors.s_f)

    assert "final_lagrangian_grad_norm" in sol.stats
    assert "final_lagrangian_grad_norm_user" in sol.stats
    np.testing.assert_allclose(
        float(sol.stats["final_lagrangian_grad_norm_user"]),
        float(sol.stats["final_lagrangian_grad_norm"]) / s_f,
        rtol=1e-12,
    )


def test_scaled_stats_user_unit_qp_multipliers() -> None:
    """``multipliers_eq_qp_user`` and ``multipliers_ineq_qp_user`` exist
    and follow the same scale recipe as the LS variants."""

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

    assert "multipliers_eq_qp_user" in sol.stats
    expected_eq_user = (factors.s_eq / factors.s_f) * sol.stats["multipliers_eq_qp"]
    np.testing.assert_allclose(
        np.asarray(sol.stats["multipliers_eq_qp_user"]),
        np.asarray(expected_eq_user),
        atol=1e-12,
    )


def test_unscaled_stats_no_user_unit_keys() -> None:
    """Without auto-scaling the user-unit keys are absent.

    They are added by ``unscale_solution``; running ``optx.minimise``
    directly (no scaling wrapper) keeps the bare scaled/unscaled-units
    contract unchanged.
    """

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    x0 = np.zeros(3)
    sol = minimize_like_scipy(f, x0, auto_scale=False)
    assert "final_objective_user" not in sol.stats
    assert "final_lagrangian_grad_norm_user" not in sol.stats
    assert "multipliers_eq_qp_user" not in sol.stats


# ---------------------------------------------------------------------------
# Proximal mu_min floor under auto-scaling
# ---------------------------------------------------------------------------


def test_scaled_proximal_mu_min_preserves_user_atol_compat_path(
    monkeypatch,
) -> None:
    """``minimize_like_scipy`` pins ``_proximal_mu_min`` to user-atol.

    Without the fix in ``compat.py``, the freshly-constructed
    ``SLSQP`` resolved its ``_proximal_mu_min`` against
    ``factors.atol_internal``, which under auto-scaling can be orders
    of magnitude smaller than the user's pre-scaling ``atol``.  This
    test monkey-patches ``optx.minimise`` to capture the constructed
    solver and asserts the resolved floor.
    """
    from slsqp_jax import compat as _compat

    captured: dict[str, object] = {}

    real_minimise = _compat.optx.minimise

    def capturing_minimise(fn, solver, x0, *args, **kwargs):
        captured["solver"] = solver
        return real_minimise(fn, solver, x0, *args, **kwargs)

    monkeypatch.setattr(_compat.optx, "minimise", capturing_minimise)

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    # ||J_eq|| = 70 forces s_eq < 1 and atol_internal < user atol.
    def c_eq(x):
        return jnp.array([70.0 * jnp.sum(x) - 5.0])

    x0 = np.zeros(3)
    user_atol = 1e-6
    minimize_like_scipy(
        f,
        x0,
        constraints={"type": "eq", "fun": c_eq},
        auto_scale=True,
        options={"atol": user_atol},
    )

    captured_solver = captured.get("solver")
    assert captured_solver is not None, "optx.minimise was never called"
    # Because ``proximal_mu_min`` is left unset by ``minimize_like_scipy``,
    # the compat fix should pin the static field to the user's atol.
    assert float(captured_solver._proximal_mu_min) == pytest.approx(user_atol)


def test_scaled_proximal_mu_min_preserves_user_atol_replace_callables_path() -> None:
    """``auto_scaled_minimise`` already preserves user-atol resolution.

    ``scaling._replace_solver_callables`` shallow-copies an
    already-constructed ``SLSQP`` whose static ``_proximal_mu_min`` was
    resolved against the user's pre-scaling ``atol``; this test pins
    that contract.
    """

    def f(x, args):
        return jnp.sum((x - 1.0) ** 2), None

    def c_eq(x, args):
        return jnp.array([70.0 * jnp.sum(x) - 5.0])

    from slsqp_jax import SLSQPConfig, ToleranceConfig

    user_atol = 1e-6
    config = SLSQPConfig(tolerance=ToleranceConfig(atol=user_atol))
    solver = SLSQP(
        config=config,
        eq_constraint_fn=c_eq,
        n_eq_constraints=1,
    )
    # Pre-scaling resolution is the only path here — the field is set
    # by ``__init__`` against the user's tolerance.
    assert float(solver._proximal_mu_min) == pytest.approx(user_atol)

    x0 = jnp.zeros(3)
    auto_scaled_minimise(
        f,
        solver,
        x0,
        args=None,
        auto_scale=True,
        has_aux=True,
        max_steps=20,
    )
    # ``auto_scaled_minimise`` does not reconstruct the solver; the
    # static field is unchanged after the run.
    assert float(solver._proximal_mu_min) == pytest.approx(user_atol)


def test_scaled_proximal_mu_min_honours_explicit_user_value(
    monkeypatch,
) -> None:
    """An explicit ``proximal_mu_min`` is honoured even under scaling."""
    from slsqp_jax import compat as _compat

    captured: dict[str, object] = {}

    real_minimise = _compat.optx.minimise

    def capturing_minimise(fn, solver, x0, *args, **kwargs):
        captured["solver"] = solver
        return real_minimise(fn, solver, x0, *args, **kwargs)

    monkeypatch.setattr(_compat.optx, "minimise", capturing_minimise)

    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    def c_eq(x):
        return jnp.array([70.0 * jnp.sum(x) - 5.0])

    x0 = np.zeros(3)
    explicit_mu_min = 1e-4
    minimize_like_scipy(
        f,
        x0,
        constraints={"type": "eq", "fun": c_eq},
        auto_scale=True,
        options={"proximal_mu_min": explicit_mu_min},
    )

    captured_solver = captured.get("solver")
    assert captured_solver is not None
    assert float(captured_solver._proximal_mu_min) == pytest.approx(explicit_mu_min)


# ---------------------------------------------------------------------------
# Helper coverage: empty-array branches + low-level wrappers
# ---------------------------------------------------------------------------


def test_grad_inf_norm_empty_array_returns_zero() -> None:
    """``_grad_inf_norm`` short-circuits for size-zero gradients."""
    from slsqp_jax.scaling import _grad_inf_norm

    assert _grad_inf_norm(jnp.zeros((0,))) == 0.0


def test_row_inf_norms_empty_jacobian_returns_zero_array() -> None:
    """``_row_inf_norms`` returns an empty ``np.ndarray`` for size-zero input."""
    from slsqp_jax.scaling import _row_inf_norms

    out = _row_inf_norms(jnp.zeros((0, 3)))
    assert out.shape == (0,)


def test_compute_scale_factors_with_no_aux_objective() -> None:
    """``has_aux=False`` exercises the alternate ``_scalar_fn`` branch."""

    def fn(x, args):
        return jnp.sum(x**2)

    factors = compute_scale_factors_at_x0(
        fn,
        np.array([2.0, 3.0]),
        args=None,
        has_aux=False,
        target_gradient=1.0,
        max_factor=1e3,
    )
    assert factors.s_f > 0


def test_compute_scale_factors_with_user_eq_jac_returning_1d() -> None:
    """A user-supplied 1-D ``eq_jac_fn`` lifts to a (1, n) matrix."""

    def fn(x, args):
        return jnp.sum(x**2), None

    def eq_fn(x, args):
        return jnp.array([jnp.sum(x) - 1.0])

    def eq_jac(x, args):
        # 1-D row that the wrapper must reshape to (1, n).
        return jnp.array([1.0, 1.0, 1.0])

    factors = compute_scale_factors_at_x0(
        fn,
        np.array([1.0, 2.0, 3.0]),
        args=None,
        has_aux=True,
        eq_constraint_fn=eq_fn,
        eq_jac_fn=eq_jac,
        target_gradient=1.0,
        max_factor=1e3,
    )
    assert factors.s_eq.shape == (1,)


def test_compute_scale_factors_with_user_ineq_jac_returning_1d() -> None:
    """A user-supplied 1-D ``ineq_jac_fn`` lifts to a (1, n) matrix."""

    def fn(x, args):
        return jnp.sum(x**2), None

    def ineq_fn(x, args):
        return jnp.array([jnp.sum(x) - 1.0])

    def ineq_jac(x, args):
        return jnp.array([1.0, 1.0, 1.0])

    factors = compute_scale_factors_at_x0(
        fn,
        np.array([1.0, 2.0, 3.0]),
        args=None,
        has_aux=True,
        ineq_constraint_fn=ineq_fn,
        ineq_jac_fn=ineq_jac,
        target_gradient=1.0,
        max_factor=1e3,
    )
    assert factors.s_ineq.shape == (1,)


def test_compute_scale_factors_eq_row_below_floor_warns() -> None:
    """A near-zero equality row triggers the per-row skip warning."""

    def fn(x, args):
        return jnp.sum(x**2), None

    def eq_fn(x, args):
        return jnp.array([1e-20 * jnp.sum(x)])  # effectively zero gradient

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        factors = compute_scale_factors_at_x0(
            fn,
            np.array([1.0, 2.0]),
            args=None,
            has_aux=True,
            eq_constraint_fn=eq_fn,
            target_gradient=1.0,
            max_factor=1e3,
            grad_floor=1e-12,
        )
    assert factors.n_skipped_eq == 1
    assert any("eq[0]" in str(w.message) for w in ws)
    assert float(factors.s_eq[0]) == 1.0


def test_compute_scale_factors_ineq_row_below_floor_warns() -> None:
    """A near-zero inequality row triggers the per-row skip warning."""

    def fn(x, args):
        return jnp.sum(x**2), None

    def ineq_fn(x, args):
        return jnp.array([1e-20 * jnp.sum(x)])

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        factors = compute_scale_factors_at_x0(
            fn,
            np.array([1.0, 2.0]),
            args=None,
            has_aux=True,
            ineq_constraint_fn=ineq_fn,
            target_gradient=1.0,
            max_factor=1e3,
            grad_floor=1e-12,
        )
    assert factors.n_skipped_ineq == 1
    assert any("ineq[0]" in str(w.message) for w in ws)
    assert float(factors.s_ineq[0]) == 1.0


def test_wrap_objective_no_aux_returns_aux_none() -> None:
    """``_wrap_objective(has_aux=False)`` returns ``(s_f * value, None)``."""
    from slsqp_jax.scaling import _wrap_objective

    def fn(x, args):
        return jnp.sum(x**2)

    wrapped = _wrap_objective(fn, s_f=2.0, has_aux=False)
    value, aux = wrapped(jnp.array([1.0, 2.0]), None)
    assert float(value) == pytest.approx(2.0 * 5.0)
    assert aux is None


def test_wrap_constraint_hvp_scales_per_row() -> None:
    """``_wrap_constraint_hvp`` multiplies row ``i`` of the (m, n) HVP stack
    by ``s_row[i]``."""
    from slsqp_jax.scaling import _wrap_constraint_hvp

    def hvp(x, v, args):
        # Two constraint rows; per-component HVP convention.
        return jnp.array([[1.0, 1.0], [2.0, 2.0]])

    s_row = jnp.array([3.0, 5.0])
    wrapped = _wrap_constraint_hvp(hvp, s_row)
    out = wrapped(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), None)
    np.testing.assert_allclose(np.asarray(out), np.array([[3.0, 3.0], [10.0, 10.0]]))


def test_make_user_unit_value_short_circuits_when_sf_is_one() -> None:
    """``_make_user_unit_value`` is a no-op when ``s_f == 1.0``."""
    from slsqp_jax.scaling import _make_user_unit_value

    factors = ScaleFactors(
        s_f=1.0,
        s_eq=jnp.array([]),
        s_ineq=jnp.array([]),
        atol_user=1e-6,
        atol_internal=1e-6,
        target_gradient=1.0,
        max_factor=1.0,
        grad_floor=1e-12,
    )
    assert _make_user_unit_value("objective", 7.0, factors) == 7.0


def test_make_user_unit_value_returns_value_for_unknown_key() -> None:
    """Keys not in :data:`_UNSCALABLE_KEYS_OBJ_DIVIDE` are passed through."""
    from slsqp_jax.scaling import _make_user_unit_value

    factors = ScaleFactors(
        s_f=2.0,
        s_eq=jnp.array([]),
        s_ineq=jnp.array([]),
        atol_user=1e-6,
        atol_internal=1e-6,
        target_gradient=1.0,
        max_factor=1.0,
        grad_floor=1e-12,
    )
    # ``"merit"`` is in the scaled-label set, not the unscalable-divide
    # set; ``_make_user_unit_value`` returns it unchanged.
    assert _make_user_unit_value("merit", 9.0, factors) == 9.0


def test_scaling_is_active_via_ineq_branch() -> None:
    """``_scaling_is_active`` returns ``True`` when only ``s_ineq`` is non-trivial."""
    from slsqp_jax.scaling import _scaling_is_active

    factors = ScaleFactors(
        s_f=1.0,
        s_eq=jnp.array([1.0]),
        s_ineq=jnp.array([0.5]),
        atol_user=1e-6,
        atol_internal=5e-7,
        target_gradient=1.0,
        max_factor=1e3,
        grad_floor=1e-12,
    )
    assert _scaling_is_active(factors) is True


def test_scaling_is_active_returns_false_when_all_trivial() -> None:
    """All trivial factors -> ``False``."""
    from slsqp_jax.scaling import _scaling_is_active

    factors = ScaleFactors(
        s_f=1.0,
        s_eq=jnp.array([1.0]),
        s_ineq=jnp.array([1.0]),
        atol_user=1e-6,
        atol_internal=1e-6,
        target_gradient=1.0,
        max_factor=1.0,
        grad_floor=1e-12,
    )
    assert _scaling_is_active(factors) is False


# ---------------------------------------------------------------------------
# auto_scaled_minimise: adjoint forwarding + non-SLSQP guard
# ---------------------------------------------------------------------------


def test_auto_scaled_minimise_passthrough_forwards_adjoint() -> None:
    """``adjoint`` is forwarded into ``optx.minimise`` on the
    ``auto_scale=False`` passthrough path."""
    import optimistix as optx

    def f(x, args):
        return jnp.sum((x - 1.0) ** 2), None

    x0 = jnp.array([0.0, 0.0])
    solver = SLSQP()
    sol = auto_scaled_minimise(
        f,
        solver,
        x0,
        args=None,
        auto_scale=False,
        has_aux=True,
        max_steps=50,
        adjoint=optx.RecursiveCheckpointAdjoint(),
    )
    np.testing.assert_allclose(np.asarray(sol.value), np.array([1.0, 1.0]), rtol=1e-5)


def test_auto_scaled_minimise_scaled_path_forwards_adjoint() -> None:
    """``adjoint`` is forwarded into ``optx.minimise`` on the scaled path."""
    import optimistix as optx

    def f(x, args):
        return jnp.sum((x - 1.0) ** 2), None

    def c_eq(x, args):
        return jnp.array([jnp.sum(x) - 5.0])

    x0 = jnp.zeros(3)
    solver = SLSQP(eq_constraint_fn=c_eq, n_eq_constraints=1)
    sol = auto_scaled_minimise(
        f,
        solver,
        x0,
        args=None,
        auto_scale=True,
        has_aux=True,
        max_steps=50,
        adjoint=optx.RecursiveCheckpointAdjoint(),
    )
    np.testing.assert_allclose(np.asarray(sol.value), np.full(3, 5.0 / 3.0), rtol=1e-4)


def test_auto_scaled_minimise_rejects_non_slsqp_solver() -> None:
    """Auto-scaling requires an SLSQP solver to forward callables to."""
    import optimistix as optx

    class _NotSLSQP(optx.AbstractMinimiser):
        rtol = 1e-6
        atol = 1e-6
        norm = optx.max_norm

        def init(self, *_args, **_kwargs):  # pragma: no cover -- never called
            ...

        def step(self, *_args, **_kwargs):  # pragma: no cover -- never called
            ...

        def terminate(self, *_args, **_kwargs):  # pragma: no cover -- never called
            ...

        def postprocess(self, *_args, **_kwargs):  # pragma: no cover -- never called
            ...

    def f(x, args):
        return jnp.sum(x**2), None

    with pytest.raises(TypeError, match="auto_scaled_minimise: solver"):
        auto_scaled_minimise(
            f,
            _NotSLSQP(),
            jnp.array([1.0, 2.0]),
            args=None,
            auto_scale=True,
            has_aux=True,
        )
