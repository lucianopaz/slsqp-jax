"""Tests for the SciPy compatibility layer.

Covers constraint parsing (dict, LinearConstraint, NonlinearConstraint),
bounds conversion, caching, verbose output, and end-to-end
``minimize_like_scipy`` comparisons against ``scipy.optimize.minimize``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
)
from scipy.optimize import (
    minimize as scipy_minimize,
)

from slsqp_jax.compat import (
    _CachedEvaluator,
    _convert_bounds,
    minimize_like_scipy,
    parse_constraints,
)

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Tests for _CachedEvaluator
# ===================================================================


class TestCachedEvaluator:
    def test_cache_hit_same_object(self):
        """Second call with the same x should not re-evaluate."""
        call_count = 0

        def fn(x, args):
            nonlocal call_count
            call_count += 1
            return x * 2

        cached = _CachedEvaluator(fn)
        x = jnp.array([1.0, 2.0])
        r1 = cached(x, None)
        r2 = cached(x, None)
        assert call_count == 1
        np.testing.assert_array_equal(r1, r2)

    def test_cache_miss_different_object(self):
        """Different x object should re-evaluate."""
        call_count = 0

        def fn(x, args):
            nonlocal call_count
            call_count += 1
            return x * 2

        cached = _CachedEvaluator(fn)
        # Hold references so GC doesn't recycle the id
        x1 = jnp.array([1.0])
        x2 = jnp.array([2.0])
        cached(x1, None)
        cached(x2, None)
        assert call_count == 2

    def test_shared_cache_eq_ineq(self):
        """eq_fn and ineq_fn sharing a cache should evaluate once."""
        call_count = 0

        def raw(x, args):
            nonlocal call_count
            call_count += 1
            return jnp.array([x[0] + x[1], x[0] - x[1]])

        cached = _CachedEvaluator(raw)

        def eq_fn(x, args):
            return cached(x, args)[0:1]

        def ineq_fn(x, args):
            return cached(x, args)[1:2]

        x = jnp.array([3.0, 1.0])
        eq_val = eq_fn(x, None)
        ineq_val = ineq_fn(x, None)
        assert call_count == 1
        np.testing.assert_allclose(eq_val, [4.0])
        np.testing.assert_allclose(ineq_val, [2.0])


# ===================================================================
# Tests for _convert_bounds
# ===================================================================


class TestConvertBounds:
    def test_none_bounds(self):
        assert _convert_bounds(None, 3) is None

    def test_tuple_bounds(self):
        b = _convert_bounds([(0, 10), (None, 5), (-1, None)], 3)
        expected = np.array([[0, 10], [-np.inf, 5], [-1, np.inf]])
        np.testing.assert_array_equal(b, expected)

    def test_scipy_bounds(self):
        b = _convert_bounds(Bounds(lb=[0, -np.inf], ub=[np.inf, 5]), 2)
        expected = np.array([[0, np.inf], [-np.inf, 5]])
        np.testing.assert_array_equal(b, expected)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="bounds has 2 entries"):
            _convert_bounds([(0, 1), (0, 1)], 3)


# ===================================================================
# Tests for parse_constraints – dict constraints
# ===================================================================


class TestParseDictConstraints:
    def test_equality_dict(self):
        con = {"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0}
        x0 = jnp.array([0.5, 0.5])
        pc = parse_constraints(con, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 0
        assert pc.eq_constraint_fn is not None
        np.testing.assert_allclose(pc.eq_constraint_fn(x0, None), [0.0], atol=1e-12)

    def test_inequality_dict(self):
        con = {"type": "ineq", "fun": lambda x: x[0] - 1.0}
        x0 = jnp.array([2.0])
        pc = parse_constraints(con, x0)
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 1
        assert pc.ineq_constraint_fn is not None
        np.testing.assert_allclose(pc.ineq_constraint_fn(x0, None), [1.0], atol=1e-12)

    def test_dict_with_jac(self):
        con = {
            "type": "eq",
            "fun": lambda x: x[0] + x[1] - 1.0,
            "jac": lambda x: np.array([1.0, 1.0]),
        }
        x0 = jnp.array([0.5, 0.5])
        pc = parse_constraints(con, x0)
        assert pc.eq_jac_fn is not None
        jac = pc.eq_jac_fn(x0, None)
        np.testing.assert_allclose(jac, [[1.0, 1.0]])

    def test_dict_with_args(self):
        con = {
            "type": "eq",
            "fun": lambda x, c: x[0] + x[1] - c,
            "args": (1.0,),
        }
        x0 = jnp.array([0.3, 0.7])
        pc = parse_constraints(con, x0)
        assert pc.eq_constraint_fn is not None
        np.testing.assert_allclose(pc.eq_constraint_fn(x0, None), [0.0], atol=1e-12)

    def test_dict_no_hvp(self):
        con = {"type": "eq", "fun": lambda x: x[0]}
        x0 = jnp.array([1.0])
        pc = parse_constraints(con, x0)
        assert pc.eq_hvp_fn is None

    def test_multiple_dict_constraints(self):
        cons = [
            {"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0},
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: x[1]},
        ]
        x0 = jnp.array([0.5, 0.5])
        pc = parse_constraints(cons, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 2

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown constraint type"):
            parse_constraints({"type": "bad", "fun": lambda x: x[0]}, jnp.array([1.0]))


# ===================================================================
# Tests for parse_constraints – LinearConstraint
# ===================================================================


class TestParseLinearConstraint:
    def test_equality(self):
        """A @ x == b  <=>  lb == ub."""
        A = np.array([[1.0, 1.0]])
        lc = LinearConstraint(A, 1.0, 1.0)
        x0 = jnp.array([0.5, 0.5])
        pc = parse_constraints(lc, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 0
        assert pc.eq_constraint_fn is not None
        np.testing.assert_allclose(pc.eq_constraint_fn(x0, None), [0.0], atol=1e-12)

    def test_equality_jac(self):
        A = np.array([[1.0, 2.0]])
        lc = LinearConstraint(A, 3.0, 3.0)
        x0 = jnp.array([1.0, 1.0])
        pc = parse_constraints(lc, x0)
        assert pc.eq_jac_fn is not None
        np.testing.assert_allclose(pc.eq_jac_fn(x0, None), [[1.0, 2.0]])

    def test_equality_hvp_is_zero(self):
        A = np.array([[1.0, 1.0]])
        lc = LinearConstraint(A, 1.0, 1.0)
        x0 = jnp.array([0.5, 0.5])
        pc = parse_constraints(lc, x0)
        assert pc.eq_hvp_fn is not None
        hvp = pc.eq_hvp_fn(x0, jnp.array([1.0, 0.0]), None)
        np.testing.assert_allclose(hvp, [[0.0, 0.0]])

    def test_one_sided_lower(self):
        """A @ x >= lb  (ub = inf)."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        lc = LinearConstraint(A, [0.0, 0.0], [np.inf, np.inf])
        x0 = jnp.array([1.0, 2.0])
        pc = parse_constraints(lc, x0)
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 2
        assert pc.ineq_constraint_fn is not None
        np.testing.assert_allclose(pc.ineq_constraint_fn(x0, None), [1.0, 2.0])

    def test_one_sided_upper(self):
        """A @ x <= ub  (lb = -inf)."""
        A = np.array([[1.0, 1.0]])
        lc = LinearConstraint(A, -np.inf, 5.0)
        x0 = jnp.array([1.0, 2.0])
        pc = parse_constraints(lc, x0)
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 1
        # ub - A@x = 5 - 3 = 2
        assert pc.ineq_constraint_fn is not None
        np.testing.assert_allclose(pc.ineq_constraint_fn(x0, None), [2.0])

    def test_two_sided(self):
        """lb <= A @ x <= ub with lb != ub, both finite."""
        A = np.array([[1.0, 0.0]])
        lc = LinearConstraint(A, 0.0, 5.0)
        x0 = jnp.array([3.0, 0.0])
        pc = parse_constraints(lc, x0)
        assert pc.n_ineq_constraints == 2
        assert pc.ineq_constraint_fn is not None
        vals = pc.ineq_constraint_fn(x0, None)
        # [A@x - lb, ub - A@x] = [3, 2]
        np.testing.assert_allclose(vals, [3.0, 2.0])

    def test_ineq_jac(self):
        A = np.array([[1.0, 0.0]])
        lc = LinearConstraint(A, 0.0, 5.0)
        x0 = jnp.array([3.0, 0.0])
        pc = parse_constraints(lc, x0)
        assert pc.ineq_jac_fn is not None
        jac = pc.ineq_jac_fn(x0, None)
        # lower row is A, upper row is -A
        np.testing.assert_allclose(jac, [[1.0, 0.0], [-1.0, 0.0]])

    def test_ineq_hvp_is_zero(self):
        """Inequality HVP for linear constraints is always zero."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        lc = LinearConstraint(A, [0.0, 0.0], [np.inf, np.inf])
        x0 = jnp.array([1.0, 2.0])
        pc = parse_constraints(lc, x0)
        assert pc.ineq_hvp_fn is not None
        hvp = pc.ineq_hvp_fn(x0, jnp.array([1.0, 0.0]), None)
        np.testing.assert_allclose(hvp, np.zeros((2, 2)))


# ===================================================================
# Tests for parse_constraints – NonlinearConstraint
# ===================================================================


class TestParseNonlinearConstraint:
    def test_equality(self):
        nlc = NonlinearConstraint(lambda x: x[0] ** 2 + x[1] ** 2, 1.0, 1.0)
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 0
        assert pc.eq_constraint_fn is not None
        np.testing.assert_allclose(pc.eq_constraint_fn(x0, None), [0.0], atol=1e-12)

    def test_inequality_lower(self):
        nlc = NonlinearConstraint(lambda x: x[0] ** 2, 1.0, np.inf)
        x0 = jnp.array([2.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_ineq_constraints == 1
        assert pc.ineq_constraint_fn is not None
        # x^2 - 1 = 3
        np.testing.assert_allclose(pc.ineq_constraint_fn(x0, None), [3.0])

    def test_inequality_upper(self):
        nlc = NonlinearConstraint(lambda x: x[0], -np.inf, 10.0)
        x0 = jnp.array([3.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_ineq_constraints == 1
        assert pc.ineq_constraint_fn is not None
        # ub - fun(x) = 10 - 3 = 7
        np.testing.assert_allclose(pc.ineq_constraint_fn(x0, None), [7.0])

    def test_with_callable_jac(self):
        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        )
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_jac_fn is not None
        np.testing.assert_allclose(pc.eq_jac_fn(x0, None), [[2.0, 0.0]])

    def test_non_callable_jac_ignored(self):
        nlc = NonlinearConstraint(
            lambda x: x[0],
            0.0,
            np.inf,
            jac="2-point",
        )
        x0 = jnp.array([1.0])
        pc = parse_constraints(nlc, x0)
        assert pc.ineq_jac_fn is None

    def test_with_callable_hess(self):
        # hess(x, v) returns sum_i v_i * H_i as (n, n) matrix.
        # For c(x) = x0^2 + x1^2, the Hessian is diag(2, 2).
        def hess_fn(x, v):
            return v[0] * np.diag([2.0, 2.0])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
            hess=hess_fn,
        )
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None
        v = jnp.array([1.0, 0.0])
        hvp = pc.eq_hvp_fn(x0, v, None)
        # hess(x, e_0) = diag(2,2); diag(2,2) @ [1,0] = [2, 0]
        np.testing.assert_allclose(hvp, [[2.0, 0.0]])

    def test_non_callable_hess_ignored(self):
        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0]]]),
            hess="2-point",
        )
        x0 = jnp.array([1.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is None

    def test_mixed_bounds_caching(self):
        """NonlinearConstraint with eq + ineq components shares evaluation."""
        call_count = 0

        def my_fun(x):
            nonlocal call_count
            call_count += 1
            return np.array([x[0] + x[1], x[0] - x[1]])

        # First component: equality (lb==ub==1)
        # Second component: inequality (lb=0, ub=inf)
        nlc = NonlinearConstraint(my_fun, [1.0, 0.0], [1.0, np.inf])
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 1

        # Evaluate both with the same x object
        call_count = 0
        x = jnp.array([0.6, 0.4])
        assert pc.eq_constraint_fn is not None
        assert pc.ineq_constraint_fn is not None
        eq_val = pc.eq_constraint_fn(x, None)
        ineq_val = pc.ineq_constraint_fn(x, None)

        assert call_count == 1  # cached
        np.testing.assert_allclose(eq_val, [0.0], atol=1e-12)  # 0.6+0.4-1
        np.testing.assert_allclose(ineq_val, [0.2], atol=1e-12)  # 0.6-0.4-0

    def test_mixed_bounds_hvp_caching(self):
        """HVP from NonlinearConstraint with mixed eq+ineq should share computation."""
        hess_call_count = 0

        def my_fun(x):
            return np.array([x[0] ** 2 + x[1] ** 2, x[0] - x[1]])

        def my_hess(x, v):
            nonlocal hess_call_count
            hess_call_count += 1
            return v[0] * np.diag([2.0, 2.0]) + v[1] * np.zeros((2, 2))

        nlc = NonlinearConstraint(
            my_fun,
            [1.0, 0.0],
            [1.0, np.inf],
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]], [1.0, -1.0]]),
            hess=my_hess,
        )
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None
        assert pc.ineq_hvp_fn is not None

        hess_call_count = 0
        x = jnp.array([0.6, 0.4])
        v = jnp.array([1.0, 0.0])
        eq_hvp = pc.eq_hvp_fn(x, v, None)
        ineq_hvp = pc.ineq_hvp_fn(x, v, None)

        # Both should share the cached evaluation: 2 calls to raw_hess
        # (one per component) but only on the first accessor.
        # On the second accessor the cache is hit, so no additional calls.
        assert hess_call_count == 2  # one per component, computed once total
        np.testing.assert_allclose(eq_hvp, [[2.0, 0.0]])
        np.testing.assert_allclose(ineq_hvp, [[0.0, 0.0]])

    def test_ineq_only_hvp(self):
        """NonlinearConstraint with only inequality and callable hess."""

        def hess_fn(x, v):
            return v[0] * np.diag([2.0, 2.0])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            np.inf,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
            hess=hess_fn,
        )
        x0 = jnp.array([2.0, 1.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 1
        assert pc.ineq_hvp_fn is not None
        assert pc.eq_hvp_fn is None
        v = jnp.array([1.0, 0.0])
        hvp = pc.ineq_hvp_fn(x0, v, None)
        np.testing.assert_allclose(hvp, [[2.0, 0.0]])

    def test_ineq_only_upper_hvp(self):
        """NonlinearConstraint with upper-bounded ineq-only and callable hess."""

        def hess_fn(x, v):
            return v[0] * np.diag([2.0, 2.0])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            -np.inf,
            10.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
            hess=hess_fn,
        )
        x0 = jnp.array([1.0, 1.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 1
        assert pc.ineq_hvp_fn is not None
        v = jnp.array([1.0, 0.0])
        hvp = pc.ineq_hvp_fn(x0, v, None)
        # upper ineq: ub - f(x) >= 0, so HVP is negated
        np.testing.assert_allclose(hvp, [[-2.0, 0.0]])

    def test_mixed_bounds_with_jac_caching(self):
        """NonlinearConstraint with mixed eq+ineq and callable jac uses cache."""
        jac_call_count = 0

        def my_fun(x):
            return np.array([x[0] + x[1], x[0] - x[1]])

        def my_jac(x):
            nonlocal jac_call_count
            jac_call_count += 1
            return np.array([[1.0, 1.0], [1.0, -1.0]])

        nlc = NonlinearConstraint(my_fun, [1.0, 0.0], [1.0, np.inf], jac=my_jac)
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 1
        assert pc.eq_jac_fn is not None
        assert pc.ineq_jac_fn is not None

        jac_call_count = 0
        x = jnp.array([0.6, 0.4])
        eq_jac = pc.eq_jac_fn(x, None)
        ineq_jac = pc.ineq_jac_fn(x, None)

        assert jac_call_count == 1  # cached
        np.testing.assert_allclose(eq_jac, [[1.0, 1.0]])
        np.testing.assert_allclose(ineq_jac, [[1.0, -1.0]])


# ===================================================================
# Tests for non-standard NonlinearConstraint.hessp extension
# ===================================================================


class TestParseNonlinearConstraintHessp:
    """Covers the non-standard ``NonlinearConstraint.hessp`` attribute.

    SciPy does not ship ``hessp`` on ``NonlinearConstraint``.  The compat
    layer treats a user-attached ``hessp`` attribute, if present and
    callable, as the per-component constraint HVP with precedence over
    ``hess``.  See the module docstring of ``slsqp_jax.compat`` for the
    full contract.
    """

    def test_hessp_takes_precedence_over_hess(self):
        """Both attributes set: ``hess`` must never be called."""
        hess_call_count = 0
        hessp_call_count = 0

        def my_hess(x, v):
            nonlocal hess_call_count
            hess_call_count += 1
            return v[0] * np.diag([2.0, 2.0])

        def my_hessp(x, p):
            nonlocal hessp_call_count
            hessp_call_count += 1
            # c(x) = x0^2 + x1^2, Hessian = diag(2, 2), so HVP is 2 * p.
            return np.array([[2.0 * p[0], 2.0 * p[1]]])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
            hess=my_hess,
        )
        nlc.hessp = my_hessp  # non-standard

        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None

        p = jnp.array([3.0, 4.0])
        hvp = pc.eq_hvp_fn(x0, p, None)
        np.testing.assert_allclose(hvp, [[6.0, 8.0]])
        assert hessp_call_count == 1
        assert hess_call_count == 0

    def test_hessp_only_no_hess(self):
        """hessp alone is enough to produce eq_hvp_fn."""

        def my_hessp(x, p):
            return np.array([[2.0 * p[0], 2.0 * p[1]]])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        )
        nlc.hessp = my_hessp

        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None
        p = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(pc.eq_hvp_fn(x0, p, None), [[2.0, 0.0]])

    def test_hessp_mixed_eq_ineq_shapes_and_caching(self):
        """Mixed eq+ineq: hessp is called once, rows are split and sign-flipped."""
        hessp_call_count = 0

        def my_fun(x):
            return np.array([x[0] ** 2 + x[1] ** 2, x[0] - x[1]])

        # Row 0: (d^2 c_0 / dx^2) @ p = diag(2,2) @ p = 2*p
        # Row 1: (d^2 c_1 / dx^2) @ p = 0
        def my_hessp(x, p):
            nonlocal hessp_call_count
            hessp_call_count += 1
            return np.array(
                [[2.0 * p[0], 2.0 * p[1]], [0.0, 0.0]],
            )

        nlc = NonlinearConstraint(
            my_fun,
            [1.0, 0.0],
            [1.0, np.inf],
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]], [1.0, -1.0]]),
        )
        nlc.hessp = my_hessp

        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 1
        assert pc.eq_hvp_fn is not None
        assert pc.ineq_hvp_fn is not None

        hessp_call_count = 0
        x = jnp.array([0.6, 0.4])
        p = jnp.array([1.0, 0.0])
        eq_hvp = pc.eq_hvp_fn(x, p, None)
        ineq_hvp = pc.ineq_hvp_fn(x, p, None)

        # hessp called once; second accessor hits the cache.
        assert hessp_call_count == 1
        np.testing.assert_allclose(eq_hvp, [[2.0, 0.0]])
        np.testing.assert_allclose(ineq_hvp, [[0.0, 0.0]])

    def test_hessp_upper_bound_sign_flip(self):
        """Upper-bounded ineq row is negated (ub - f(x) >= 0 convention)."""

        def my_hessp(x, p):
            return np.array([[2.0 * p[0], 2.0 * p[1]]])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            -np.inf,
            10.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        )
        nlc.hessp = my_hessp

        x0 = jnp.array([1.0, 1.0])
        pc = parse_constraints(nlc, x0)
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 1
        assert pc.ineq_hvp_fn is not None
        p = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(pc.ineq_hvp_fn(x0, p, None), [[-2.0, 0.0]])

    def test_hessp_non_callable_falls_back_to_hess(self):
        """A non-callable hessp (e.g. a sentinel string) is ignored."""
        hess_call_count = 0

        def my_hess(x, v):
            nonlocal hess_call_count
            hess_call_count += 1
            return v[0] * np.diag([2.0, 2.0])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
            hess=my_hess,
        )
        nlc.hessp = "2-point"  # non-callable sentinel

        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None
        p = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(pc.eq_hvp_fn(x0, p, None), [[2.0, 0.0]])
        assert hess_call_count >= 1

    def test_hessp_wrong_arity_raises(self):
        """Arity != 2 (and no *args) must raise at parse time."""
        base_kwargs = dict(
            fun=lambda x: x[0] ** 2 + x[1] ** 2,
            lb=1.0,
            ub=1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        )
        x0 = jnp.array([1.0, 0.0])

        nlc1 = NonlinearConstraint(**base_kwargs)
        nlc1.hessp = lambda x: x  # arity 1
        with pytest.raises(TypeError, match="hessp must accept exactly two"):
            parse_constraints(nlc1, x0)

        nlc3 = NonlinearConstraint(**base_kwargs)
        nlc3.hessp = lambda x, p, extra: x  # arity 3
        with pytest.raises(TypeError, match="hessp must accept exactly two"):
            parse_constraints(nlc3, x0)

    def test_hessp_accepts_varargs(self):
        """A *args callable is accepted without arity enforcement."""

        def my_hessp(*args, **kwargs):
            p = args[1]
            return np.array([[2.0 * p[0], 2.0 * p[1]]])

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        )
        nlc.hessp = my_hessp
        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None
        p = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(pc.eq_hvp_fn(x0, p, None), [[2.0, 0.0]])

    def test_hessp_non_introspectable_callable_accepted(self):
        """Callables whose signature cannot be inspected bypass arity check.

        Mirrors the C-level / exotic-wrapper case noted in the docstring:
        ``inspect.signature`` raises ``ValueError`` and the validator must
        silently accept the callable so parsing does not abort.  We force
        the branch with a synthetic callable whose ``__signature__``
        descriptor itself raises ``ValueError``.
        """

        class _NoSignatureHessp:
            def __call__(self, x, p):
                return np.array([[2.0 * p[0], 2.0 * p[1]]])

            @property
            def __signature__(self):
                raise ValueError("synthetic: no signature available")

        nlc = NonlinearConstraint(
            lambda x: x[0] ** 2 + x[1] ** 2,
            1.0,
            1.0,
            jac=lambda x: np.array([[2 * x[0], 2 * x[1]]]),
        )
        nlc.hessp = _NoSignatureHessp()

        x0 = jnp.array([1.0, 0.0])
        pc = parse_constraints(nlc, x0)
        assert pc.eq_hvp_fn is not None
        p = jnp.array([1.0, 0.0])
        np.testing.assert_allclose(pc.eq_hvp_fn(x0, p, None), [[2.0, 0.0]])


# ===================================================================
# Tests for mixed constraint lists
# ===================================================================


class TestMixedConstraints:
    def test_dict_and_linear(self):
        cons = [
            {"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0},
            LinearConstraint(np.array([[0.0, 1.0]]), 0.0, np.inf),
        ]
        x0 = jnp.array([0.5, 0.5])
        pc = parse_constraints(cons, x0)
        assert pc.n_eq_constraints == 1
        assert pc.n_ineq_constraints == 1

    def test_all_types(self):
        cons = [
            {"type": "eq", "fun": lambda x: x[0] - 1.0},
            {"type": "ineq", "fun": lambda x: x[1]},
            LinearConstraint(np.eye(2), [0, 0], [10, 10]),
            NonlinearConstraint(lambda x: x[0] ** 2 + x[1] ** 2, -np.inf, 25.0),
        ]
        x0 = jnp.array([1.0, 2.0])
        pc = parse_constraints(cons, x0)
        assert pc.n_eq_constraints == 1  # dict eq
        assert pc.n_ineq_constraints > 0

    def test_jac_none_if_any_missing(self):
        """If one source lacks jac, the combined group is None."""
        cons = [
            {
                "type": "ineq",
                "fun": lambda x: x[0],
                "jac": lambda x: np.array([[1.0, 0.0]]),
            },
            {"type": "ineq", "fun": lambda x: x[1]},  # no jac
        ]
        x0 = jnp.array([1.0, 1.0])
        pc = parse_constraints(cons, x0)
        assert pc.ineq_jac_fn is None

    def test_empty_constraints(self):
        pc = parse_constraints((), jnp.array([1.0]))
        assert pc.n_eq_constraints == 0
        assert pc.n_ineq_constraints == 0
        assert pc.eq_constraint_fn is None

    def test_combined_jac_and_hvp_from_two_sources(self):
        """Two LinearConstraints providing eq parts combine jac and hvp."""
        lc1 = LinearConstraint(np.array([[1.0, 0.0]]), 1.0, 1.0)
        lc2 = LinearConstraint(np.array([[0.0, 1.0]]), 2.0, 2.0)
        x0 = jnp.array([1.0, 2.0])
        pc = parse_constraints([lc1, lc2], x0)
        assert pc.n_eq_constraints == 2
        assert pc.eq_jac_fn is not None
        assert pc.eq_hvp_fn is not None

        jac = pc.eq_jac_fn(x0, None)
        np.testing.assert_allclose(jac, [[1.0, 0.0], [0.0, 1.0]])

        v = jnp.array([1.0, 0.0])
        hvp = pc.eq_hvp_fn(x0, v, None)
        np.testing.assert_allclose(hvp, np.zeros((2, 2)))

    def test_combined_ineq_hvp_from_two_sources(self):
        """Two NonlinearConstraints providing ineq HVPs combine correctly."""

        def hess1(x, v):
            return v[0] * np.diag([2.0, 0.0])

        def hess2(x, v):
            return v[0] * np.diag([0.0, 6.0])

        nlc1 = NonlinearConstraint(
            lambda x: x[0] ** 2,
            1.0,
            np.inf,
            jac=lambda x: np.array([[2 * x[0], 0.0]]),
            hess=hess1,
        )
        nlc2 = NonlinearConstraint(
            lambda x: 3 * x[1] ** 2,
            0.0,
            np.inf,
            jac=lambda x: np.array([[0.0, 6 * x[1]]]),
            hess=hess2,
        )
        x0 = jnp.array([1.0, 1.0])
        pc = parse_constraints([nlc1, nlc2], x0)
        assert pc.n_ineq_constraints == 2
        assert pc.ineq_jac_fn is not None
        assert pc.ineq_hvp_fn is not None

        jac = pc.ineq_jac_fn(x0, None)
        np.testing.assert_allclose(jac, [[2.0, 0.0], [0.0, 6.0]])

        v = jnp.array([1.0, 0.0])
        hvp = pc.ineq_hvp_fn(x0, v, None)
        np.testing.assert_allclose(hvp, [[2.0, 0.0], [0.0, 0.0]])


# ===================================================================
# Tests for minimize_like_scipy  (end-to-end)
# ===================================================================


class TestMinimizeLikeScipy:
    def test_unconstrained(self):
        """Minimise x^2 + 2y^2."""

        def fun(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        x0 = np.array([3.0, -2.0])
        ref = scipy_minimize(fun, x0, method="SLSQP")
        sol = minimize_like_scipy(fun, x0, options={"max_steps": 50})
        np.testing.assert_allclose(sol.value, ref.x, atol=1e-4)

    def test_equality_constraint_dict(self):
        """Nocedal & Wright Example 16.4 (equality only)."""

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

        cons = {"type": "eq", "fun": lambda x: x[0] + x[1] - 4.0}
        x0 = np.array([2.0, 0.0])
        ref = scipy_minimize(fun, x0, method="SLSQP", constraints=cons)
        sol = minimize_like_scipy(fun, x0, constraints=cons, options={"max_steps": 100})
        np.testing.assert_allclose(sol.value, ref.x, atol=2e-3)

    def test_inequality_constraints_dict(self):
        """Minimise (x-1)^2 + (y-2.5)^2  s.t. x - 2y + 2 >= 0 etc."""

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

        cons = [
            {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
            {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
            {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
        ]
        bnds = [(0, None), (0, None)]
        x0 = np.array([2.0, 0.0])
        ref = scipy_minimize(fun, x0, method="SLSQP", bounds=bnds, constraints=cons)
        sol = minimize_like_scipy(
            fun, x0, bounds=bnds, constraints=cons, options={"max_steps": 100}
        )
        np.testing.assert_allclose(sol.value, ref.x, atol=1e-2)

    def test_linear_constraint(self):
        """Minimise sum(x^2) s.t. sum(x) == 3."""

        def fun(x):
            return jnp.sum(jnp.asarray(x) ** 2)

        A = np.ones((1, 3))
        lc = LinearConstraint(A, 3.0, 3.0)
        x0 = np.array([1.0, 1.0, 1.0])
        ref = scipy_minimize(
            lambda x: float(fun(x)), x0, method="SLSQP", constraints=lc
        )
        sol = minimize_like_scipy(fun, x0, constraints=lc, options={"max_steps": 100})
        np.testing.assert_allclose(sol.value, ref.x, atol=1e-3)

    def test_nonlinear_constraint(self):
        """Minimise sum(x^2) s.t. x0^2 + x1^2 >= 1."""

        def fun(x):
            return jnp.sum(jnp.asarray(x) ** 2)

        nlc = NonlinearConstraint(lambda x: x[0] ** 2 + x[1] ** 2, 1.0, np.inf)
        x0 = np.array([2.0, 2.0])
        ref = scipy_minimize(
            lambda x: float(fun(x)), x0, method="SLSQP", constraints=nlc
        )
        sol = minimize_like_scipy(fun, x0, constraints=nlc, options={"max_steps": 100})
        np.testing.assert_allclose(sol.value, ref.x, atol=1e-2)

    def test_bounds_only(self):
        """Minimise (x-5)^2 s.t. 0 <= x <= 3."""

        def fun(x):
            return (x[0] - 5.0) ** 2

        x0 = np.array([1.0])
        ref = scipy_minimize(fun, x0, method="SLSQP", bounds=[(0, 3)])
        sol = minimize_like_scipy(fun, x0, bounds=[(0, 3)], options={"max_steps": 50})
        np.testing.assert_allclose(sol.value, ref.x, atol=1e-3)

    def test_with_jac_callable(self):
        """Supply an explicit gradient function."""

        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        def grad(x):
            return jnp.array([2 * x[0], 2 * x[1]])

        x0 = np.array([3.0, -2.0])
        sol = minimize_like_scipy(fun, x0, jac=grad, options={"max_steps": 50})
        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-4)

    def test_with_jac_true(self):
        """fun returns (f, g) when jac=True."""

        def fun(x):
            f = x[0] ** 2 + x[1] ** 2
            g = jnp.array([2 * x[0], 2 * x[1]])
            return f, g

        x0 = np.array([3.0, -2.0])
        sol = minimize_like_scipy(fun, x0, jac=True, options={"max_steps": 50})
        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-4)

    def test_has_aux(self):
        """fun returns (f, aux) when has_aux=True."""

        def fun(x):
            return x[0] ** 2 + x[1] ** 2, {"info": 42}

        x0 = np.array([3.0, -2.0])
        sol = minimize_like_scipy(fun, x0, has_aux=True, options={"max_steps": 50})
        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-4)

    def test_with_hessp(self):
        """Supply a Hessian-vector product function."""

        def fun(x):
            return x[0] ** 2 + 2 * x[1] ** 2

        def hessp(x, p):
            return jnp.array([2 * p[0], 4 * p[1]])

        x0 = np.array([3.0, -2.0])
        sol = minimize_like_scipy(fun, x0, hessp=hessp, options={"max_steps": 50})
        np.testing.assert_allclose(sol.value, [0.0, 0.0], atol=1e-4)

    def test_verbose_runs(self, capsys):
        """verbose=True should produce output without errors."""

        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([3.0, -2.0])
        sol = minimize_like_scipy(fun, x0, verbose=True, options={"max_steps": 10})
        # Just check it ran without error and produced some output
        assert sol.value is not None

    def test_throw_false(self):
        """throw=False should return a solution even if not converged."""

        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([100.0, 100.0])
        sol = minimize_like_scipy(fun, x0, throw=False, options={"max_steps": 1})
        assert sol.value is not None

    def test_options_routed_into_subconfigs(self):
        """Non-``max_steps`` options reach their nested ``*Config`` slot.

        Exercises the per-section translation loop in
        ``_pop_section`` (the branch that actually pops a key from the
        options dict) by passing one option from every sub-config.
        """

        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([3.0, -2.0])
        sol = minimize_like_scipy(
            fun,
            x0,
            options={
                "max_steps": 20,
                "min_steps": 1,
                "lbfgs_memory": 5,
                "line_search_max_steps": 8,
                "qp_max_iter": 30,
                "proximal_tau": 0.0,
                "preconditioner_type": "lbfgs",
                "active_set_method": "expand",
                "adaptive_cg_tol": False,
            },
        )
        assert sol.value is not None

    def test_unrecognized_option_raises(self):
        """Unknown keys in the SciPy-style ``options`` dict should error.

        Covers the trailing ``if opts: raise TypeError(...)`` guard in
        ``minimize_like_scipy`` after every section has been popped.
        """

        def fun(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([3.0, -2.0])
        with pytest.raises(TypeError, match="unrecognized option"):
            minimize_like_scipy(
                fun, x0, options={"max_steps": 5, "definitely_not_a_real_option": 42}
            )
