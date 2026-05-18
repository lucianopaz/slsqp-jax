"""Tests for the filterSQP-style multiplier-scale-aware termination.

Covers (in this order):

1. Unit-level checks of :func:`slsqp_jax.slsqp.termination.compute_mu_max`
   — manual computations on small fixtures and a JIT smoke test.
2. Behavioural regressions on end-to-end runs that pin the *intended
   difference* between the legacy ``max(|L|, 1)`` denominator and the
   filterSQP ``max(mu_max, 1)`` denominator (eq. 5 of
   *User manual for filterSQP*, Fletcher & Leyffer):
   * the large-``|f|`` case no longer loosens the threshold,
   * the unconstrained / empty-constraint case still terminates,
   * the near-rank-deficient equality case where multipliers grow
     still converges within budget.
3. Complementarity (eqs. 3 and 4) checks at the final iterate of
   representative problems, codifying the "guaranteed constructively
   by the QP / LS active-set logic" claim that lets us avoid a
   runtime complementarity check.
4. ``sol.stats["kkt_scale"]`` exposure (scaled and user-unit
   variants under auto-scaling).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from slsqp_jax import is_successful, minimize_like_scipy
from slsqp_jax.slsqp.termination import compute_mu_max
from tests.conftest import _make_slsqp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. Unit tests for ``compute_mu_max``
# ---------------------------------------------------------------------------


class TestComputeMuMax:
    """Manual computations on hand-crafted inputs."""

    def test_empty_constraints_reduce_to_grad_norm(self):
        # No equalities, no general inequalities, no bounds.
        # μ_max should reduce to ||grad_f||_2.
        g = jnp.array([3.0, 4.0])  # ||g|| = 5
        mu = compute_mu_max(
            grad_f=g,
            eq_jac=jnp.zeros((0, 2)),
            ineq_jac_general=jnp.zeros((0, 2)),
            mult_eq=jnp.zeros((0,)),
            mult_ineq_general=jnp.zeros((0,)),
            mult_bound=jnp.zeros((0,)),
        )
        assert float(mu) == pytest.approx(5.0)

    def test_equality_only_picks_max_row_contribution(self):
        # Two equality rows:
        #   row 0 = [1, 1, 0] → ||a_0|| = sqrt(2), λ_0 = 3 → term ≈ 4.243
        #   row 1 = [0, 2, 0] → ||a_1|| = 2,        λ_1 = -4 → term = 8
        # ||g|| = sqrt(14) ≈ 3.742.  Max = 8.
        g = jnp.array([1.0, 2.0, 3.0])
        eq_jac = jnp.array([[1.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        mult_eq = jnp.array([3.0, -4.0])
        mu = compute_mu_max(
            grad_f=g,
            eq_jac=eq_jac,
            ineq_jac_general=jnp.zeros((0, 3)),
            mult_eq=mult_eq,
            mult_ineq_general=jnp.zeros((0,)),
            mult_bound=jnp.zeros((0,)),
        )
        assert float(mu) == pytest.approx(8.0)

    def test_bound_only_collapses_to_max_abs_multiplier(self):
        # Bound Jacobian rows have ||a_i|| = 1 by construction, so the
        # filterSQP eq. (5) term for each bound reduces to |ν_i|.
        g = jnp.array([0.1, 0.2])
        mult_bound = jnp.array([5.0, -7.0, 2.0])
        mu = compute_mu_max(
            grad_f=g,
            eq_jac=jnp.zeros((0, 2)),
            ineq_jac_general=jnp.zeros((0, 2)),
            mult_eq=jnp.zeros((0,)),
            mult_ineq_general=jnp.zeros((0,)),
            mult_bound=mult_bound,
        )
        assert float(mu) == pytest.approx(7.0)

    def test_mixed_three_vars_two_eq_one_ineq_one_bound(self):
        """Hand-computed mixed example matching the plan.

        ``n=3``, ``m_eq=2``, ``m_ineq_general=1``, ``m_bound=1``.
        The four kinds of contributors must all be considered.
        """
        g = jnp.array([1.0, 2.0, 2.0])  # ||g|| = 3
        # Equality rows: row 0 = [1, 1, 0] (norm sqrt(2)), row 1 = [0, 0, 1]
        # (norm 1).  Multipliers [0.5, 10] → contributions sqrt(2)/2≈0.707
        # and 10.
        eq_jac = jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mult_eq = jnp.array([0.5, 10.0])
        # General inequality row: [1, 2, 2] (norm 3).  Multiplier 4 →
        # contribution 12.
        ineq_jac_general = jnp.array([[1.0, 2.0, 2.0]])
        mult_ineq_general = jnp.array([4.0])
        # Bound multiplier 6 → contribution 6.
        mult_bound = jnp.array([6.0])

        # Expected: max(3, 0.707, 10, 12, 6) = 12.
        mu = compute_mu_max(
            grad_f=g,
            eq_jac=eq_jac,
            ineq_jac_general=ineq_jac_general,
            mult_eq=mult_eq,
            mult_ineq_general=mult_ineq_general,
            mult_bound=mult_bound,
        )
        assert float(mu) == pytest.approx(12.0)

    def test_negative_multipliers_take_absolute_value(self):
        # Signs in eq. (5) are inside ``|·|``; a large-magnitude negative
        # multiplier must still drive μ_max up.
        g = jnp.array([0.1])
        eq_jac = jnp.array([[2.0]])
        mu = compute_mu_max(
            grad_f=g,
            eq_jac=eq_jac,
            ineq_jac_general=jnp.zeros((0, 1)),
            mult_eq=jnp.array([-50.0]),  # contribution: 2 * 50 = 100
            mult_ineq_general=jnp.zeros((0,)),
            mult_bound=jnp.zeros((0,)),
        )
        assert float(mu) == pytest.approx(100.0)

    def test_jit_compatibility(self):
        """``compute_mu_max`` traces cleanly under ``jax.jit``."""

        @jax.jit
        def f(g, J_eq, lam):
            return compute_mu_max(
                grad_f=g,
                eq_jac=J_eq,
                ineq_jac_general=jnp.zeros((0, g.shape[0])),
                mult_eq=lam,
                mult_ineq_general=jnp.zeros((0,)),
                mult_bound=jnp.zeros((0,)),
            )

        out = f(
            jnp.array([1.0, 0.0]),
            jnp.array([[1.0, 1.0]]),
            jnp.array([2.0]),
        )
        # max(||g||=1, sqrt(2)*2) = 2*sqrt(2).
        assert float(out) == pytest.approx(2.0 * np.sqrt(2))


# ---------------------------------------------------------------------------
# 2. Behavioural regressions
# ---------------------------------------------------------------------------


class TestStationarityThresholdBehavior:
    """End-to-end checks that pin the *intended difference* between the
    legacy ``max(|L|, 1)`` denominator and the new ``max(μ_max, 1)``
    denominator.
    """

    def test_unconstrained_quadratic_still_terminates(self):
        """Empty-constraint case: μ_max collapses to ``||∇f||``.

        Pin the unconstrained convergence path so the empty-constraint
        reduction inside :func:`compute_mu_max` does not regress.
        """

        def objective(x, args):
            return jnp.sum((x - jnp.array([1.0, 2.0, 3.0])) ** 2), None

        x0 = jnp.array([0.0, 0.0, 0.0])
        solver = _make_slsqp(rtol=1e-8, atol=1e-8, max_steps=50)
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        assert is_successful(sol.stats["slsqp_result"])
        np.testing.assert_allclose(sol.value, jnp.array([1.0, 2.0, 3.0]), atol=1e-6)
        # At the optimum ||∇f|| ≈ 0 so μ_max ≈ 0 and the test reduces
        # to ||∇L|| ≤ rtol * 1 = rtol — an *absolute* requirement.
        assert float(sol.stats["final_lagrangian_grad_norm"]) <= 1e-6

    def test_large_f_does_not_loosen_threshold(self):
        """Adding a large constant to ``f`` no longer loosens ``rtol``.

        Under the legacy ``rtol * max(|L|, 1)`` denominator, adding
        ``+1e6`` to the objective inflated the stationarity threshold
        by ``|L| ≈ 1e6``, allowing termination at
        ``||∇L|| ~ 1e6 * rtol``.  Under the filterSQP denominator
        μ_max only sees ``||∇f||``, ``|ν_i|`` and ``||a_i||·|λ_i|`` —
        all of which collapse to ``0`` near the optimum — so the
        threshold reduces to ``rtol * 1 = rtol`` and the iterate is
        required to be much more accurate.
        """

        def objective(x, args):
            return 1e6 + jnp.sum((x - jnp.array([1.0, 2.0])) ** 2), None

        x0 = jnp.array([0.0, 0.0])
        rtol = 1e-6
        solver = _make_slsqp(rtol=rtol, atol=1e-8, max_steps=50)
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        assert is_successful(sol.stats["slsqp_result"])
        grad_L = float(sol.stats["final_lagrangian_grad_norm"])
        kkt_scale = float(sol.stats["kkt_scale"])
        # The convergence contract: ||∇L|| ≤ rtol * max(μ_max, 1).
        # We allow a small numerical slack on top of the contract.
        assert grad_L <= rtol * max(kkt_scale, 1.0) + 1e-12, (
            f"||∇L|| = {grad_L:.3e} exceeds rtol·max(μ_max, 1) = "
            f"{rtol * max(kkt_scale, 1.0):.3e}; new denominator broken"
        )
        # Tighter: explicitly verify we did NOT terminate at the
        # legacy ``|L|``-loose tolerance (~1e0 here).  The achieved
        # ||∇L|| should be vastly tighter than ``rtol * |L| = 1``.
        assert grad_L <= 1e-2, (
            f"||∇L|| = {grad_L:.3e} is looser than 1e-2; the new "
            f"denominator must be tighter than the legacy "
            f"`rtol·|L|` ≈ 1 slack."
        )

    def test_kkt_scale_exposed_in_stats(self):
        """``sol.stats`` exposes ``μ_max`` and its exact residual ratio.

        Problem: ``min x²+y² s.t. x+y=2`` with analytic optimum
        ``x*=(1,1)`` and ``λ*=2``.  ``μ_max`` should equal the larger
        of ``||∇f||_2 = 2√2`` and ``||a_eq||_2·|λ| = √2·2 = 2√2``,
        i.e. ``2√2``.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def eq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = _make_slsqp(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=1,
        )
        x0 = jnp.array([0.0, 0.0])
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        assert "kkt_scale" in sol.stats
        assert "kkt_ratio" in sol.stats
        assert float(sol.stats["kkt_scale"]) == pytest.approx(
            2.0 * np.sqrt(2), rel=1e-3
        )
        assert float(sol.stats["kkt_ratio"]) == pytest.approx(
            float(sol.stats["final_lagrangian_grad_norm"])
            / max(float(sol.stats["kkt_scale"]), 1.0)
        )

    def test_auto_scaling_unscales_kkt_scale(self):
        """``minimize_like_scipy`` auto-scales the problem; ``kkt_scale``
        must come back in user units on the public ``kkt_scale`` key.

        We pick ``x0 = (0.5, 0.5)`` (so ``||grad_f(x0)|| > 0`` and
        auto-scaling actually fires).  The analytic solution to
        ``min x²+y² s.t. x+y=2`` is ``x*=y*=1`` with ``λ*=2``, so
        ``μ_max = 2√2`` in user units regardless of the internal
        scaling.
        """

        def fun(x):
            return jnp.sum(x**2)

        def eq_constraint_fn(x):
            return x[0] + x[1] - 2.0

        x0 = np.array([0.5, 0.5])
        sol = minimize_like_scipy(
            fun,
            x0,
            constraints={"type": "eq", "fun": eq_constraint_fn},
            options={"rtol": 1e-8, "atol": 1e-8, "max_steps": 50},
            throw=False,
        )
        assert "kkt_scale" in sol.stats
        assert "kkt_ratio" in sol.stats
        # The user-unit value matches the analytic μ_max even when
        # the internal scaling stretched the problem.
        assert float(sol.stats["kkt_scale"]) == pytest.approx(
            2.0 * np.sqrt(2), rel=5e-3
        )
        # ``kkt_ratio`` is the exact dimensionless quantity tested by
        # the scaled internal solver, avoiding ambiguity from the hard
        # ``max(mu_max, 1)`` floor after public stats are unscaled.
        assert float(sol.stats["kkt_ratio"]) <= 1e-8 + 1e-12


class TestDegenerateMultipliers:
    """Near rank-deficient equality Jacobian — multipliers can grow, but
    the filterSQP-normalised test stays well-defined.
    """

    def test_near_parallel_equality_constraints_converges(self):
        """Two nearly-parallel equality rows so ``cond(J Jᵀ) ~ 1/ε²``.

        The QP-recovered multipliers are large because the active
        Jacobian is poorly conditioned, but the filterSQP-normalised
        test stays satisfiable: ``||∇L||`` and ``μ_max`` both inflate
        with the multipliers, so the ratio is bounded.
        """
        eps = 1e-3

        def objective(x, args):
            return jnp.sum((x - jnp.array([1.0, 2.0])) ** 2), None

        def eq_constraint(x, args):
            # Two nearly-parallel constraints; both satisfied at the
            # origin (where the unconstrained optimum is *not*).
            return jnp.array([x[0] + x[1], x[0] + (1.0 + eps) * x[1]])

        solver = _make_slsqp(
            rtol=1e-6,
            atol=1e-6,
            max_steps=80,
            eq_constraint_fn=eq_constraint,
            n_eq_constraints=2,
        )
        x0 = jnp.array([0.1, 0.1])
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=80
        )
        assert is_successful(sol.stats["slsqp_result"]), (
            f"Run did not converge under filterSQP normalisation; "
            f"slsqp_result = {sol.stats['slsqp_result']}"
        )
        np.testing.assert_allclose(sol.value, jnp.array([0.0, 0.0]), atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Complementarity (filterSQP eqs. 3 and 4)
# ---------------------------------------------------------------------------


class TestComplementarity:
    """Pin the "guaranteed constructively by QP / LS" claim into a regression.

    The filterSQP manual lists three KKT conditions besides
    stationarity (eq. 1) and primal feasibility (eq. 2): the sign +
    complementary-slackness conditions on bound multipliers ``ν``
    (eq. 3) and on general-constraint multipliers ``λ`` (eq. 4).  Our
    QP + LS active-set logic *enforces* these by construction, so we
    do not check them at runtime — but we lock the contract in here.
    """

    def test_inactive_lower_bound_multiplier_is_zero(self):
        """Strictly-interior bound ⇒ multiplier = 0 (eq. 3, strict case).

        ``min (x-1)² s.t. x ≥ -10`` — bound is far inactive at
        ``x* = 1``, so the bound multiplier ``ν`` must be zero.
        """

        def objective(x, args):
            return (x[0] - 1.0) ** 2, None

        bounds = jnp.array([[-10.0, jnp.inf]])
        solver = _make_slsqp(rtol=1e-8, atol=1e-8, max_steps=50, bounds=bounds)
        x0 = jnp.array([5.0])
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        np.testing.assert_allclose(sol.value, jnp.array([1.0]), atol=1e-6)
        # Layout: [general; lower; upper] with general=0, upper=0.
        mults = np.asarray(sol.stats["multipliers_ineq"])
        assert mults.shape == (1,)
        assert abs(float(mults[0])) < 1e-6, (
            f"Strictly-interior bound has multiplier "
            f"{float(mults[0]):.3e}; eq. (3) demands ≈ 0."
        )

    def test_active_lower_bound_multiplier_sign_and_value(self):
        """Active lower bound ⇒ multiplier ≥ 0 (eq. 3, lower-active case).

        ``min (x-1)² s.t. x ≥ 2`` — the active lower bound forces
        ``x* = 2``; the KKT system gives ``ν = 2(x* - 1) = 2``.
        """

        def objective(x, args):
            return (x[0] - 1.0) ** 2, None

        bounds = jnp.array([[2.0, jnp.inf]])
        solver = _make_slsqp(rtol=1e-8, atol=1e-8, max_steps=50, bounds=bounds)
        x0 = jnp.array([3.0])
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        np.testing.assert_allclose(sol.value, jnp.array([2.0]), atol=1e-6)
        mults = np.asarray(sol.stats["multipliers_ineq"])
        assert mults.shape == (1,)
        nu_lower = float(mults[0])
        assert nu_lower >= -1e-10, (
            f"Active lower-bound multiplier {nu_lower:.3e} violates "
            f"eq. (3) sign condition (must be ≥ 0)."
        )
        assert nu_lower == pytest.approx(2.0, rel=1e-3)

    def test_general_inequality_complementarity(self):
        """``λ_i · c_i ≈ 0`` (eq. 4) on a general inequality problem.

        ``min x²+y² s.t. x+y ≥ 2`` — the analytical optimum is
        ``(1, 1)``, where the inequality is *active* (equality
        ``x+y = 2`` holds) and ``λ ≥ 0``.  The complementary-slackness
        product ``λ · c`` must be ≈ 0 in both senses: ``c ≈ 0`` because
        the constraint is active, and ``λ`` is finite.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 2.0])

        solver = _make_slsqp(
            rtol=1e-8,
            atol=1e-8,
            max_steps=80,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([0.0, 0.0])
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=80
        )
        np.testing.assert_allclose(sol.value, jnp.array([1.0, 1.0]), atol=1e-5)

        # ``multipliers_ineq`` is [general; lower; upper]; here only
        # the single general inequality exists.
        mults = np.asarray(sol.stats["multipliers_ineq"])
        assert mults.shape == (1,)
        c_val = float(ineq_constraint(sol.value, None)[0])
        lam = float(mults[0])
        # Complementary slackness (eq. 4): λ · c ≈ 0.
        assert abs(lam * c_val) < 1e-6, (
            f"|λ · c| = {abs(lam * c_val):.3e} fails eq. (4); "
            f"λ = {lam:.3e}, c = {c_val:.3e}"
        )
        # Sign on active row: λ ≥ 0.
        assert lam >= -1e-10, (
            f"Active general-inequality multiplier {lam:.3e} violates "
            f"eq. (4) sign condition."
        )

    def test_strictly_inactive_general_inequality_multiplier_is_zero(self):
        """Strictly inactive general inequality ⇒ multiplier = 0 (eq. 4).

        ``min x²+y² s.t. x+y ≥ -10`` — constraint is strictly inactive
        at the optimum ``(0, 0)`` (LHS = 0, RHS = -10).  The LS
        multiplier recovery clamps inactive-row multipliers to zero by
        construction (the active-set rule is value-based at the final
        iterate); this test pins that contract.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - (-10.0)])

        solver = _make_slsqp(
            rtol=1e-8,
            atol=1e-8,
            max_steps=50,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([1.0, 1.0])
        sol = optx.minimise(
            objective, solver, x0, has_aux=True, throw=False, max_steps=50
        )
        np.testing.assert_allclose(sol.value, jnp.array([0.0, 0.0]), atol=1e-5)
        mults = np.asarray(sol.stats["multipliers_ineq"])
        assert mults.shape == (1,)
        assert abs(float(mults[0])) < 1e-6, (
            f"Strictly-interior general inequality has multiplier "
            f"{float(mults[0]):.3e}; eq. (4) demands ≈ 0."
        )
