"""Regression tests for SLSQP -> solve_qp tolerance routing.

The outer SLSQP solver passes ``tol=self.atol`` into ``solve_qp`` so the
QP active-set loop's working tolerance, the EXPAND ramp, and the inner
CG fallback (when ``cg_tol is None``) all live in the same unit system
as the outer NLP feasibility test.  These tests pin that contract so a
future refactor cannot silently revert it back to the hard-coded
``1e-8``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from slsqp_jax.qp import solve_qp
from tests.conftest import _make_slsqp

jax.config.update("jax_enable_x64", True)


class TestQPToleranceRouting:
    """Verify ``stats["qp_tolerance"]`` reflects the resolved ``self.atol``."""

    def test_qp_tolerance_follows_self_atol(self):
        """A non-default ``atol`` flows through to ``stats["qp_tolerance"]``.

        Previously the value was hard-coded to ``1e-8`` in
        ``SLSQP.postprocess``, so this assertion would have failed
        regardless of the user's ``atol``.
        """

        def objective(x, args):
            return jnp.sum(x**2), None

        def ineq_constraint(x, args):
            return jnp.array([x[0] + x[1] - 1.0])

        atol = 1e-4
        solver = _make_slsqp(
            atol=atol,
            ineq_constraint_fn=ineq_constraint,
            n_ineq_constraints=1,
        )
        x0 = jnp.array([0.6, 0.6])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=20)

        assert sol.stats["qp_tolerance"] == atol

    def test_qp_tolerance_default_matches_default_atol(self):
        """The default ``atol = 1e-6`` shows up as ``stats["qp_tolerance"]``."""

        def objective(x, args):
            return jnp.sum(x**2), None

        solver = _make_slsqp()
        x0 = jnp.array([3.0, -2.0])
        sol = optx.minimise(objective, solver, x0, has_aux=True, max_steps=20)

        assert sol.stats["qp_tolerance"] == solver.atol

    def test_solve_qp_final_working_tol_respects_user_tol(self):
        """``solve_qp(..., tol=...)`` plumbs the value into the active-set loop.

        EXPAND ramps the working tolerance from ``tol`` to
        ``tol + tol * expand_factor`` over ``max_iter`` steps.  At any
        point during the loop ``final_working_tol >= tol`` must hold,
        which we assert post-hoc on a tiny inequality QP.
        """
        H = jnp.eye(3)
        g = jnp.array([1.0, 0.0, -2.0])
        A_ineq = jnp.array([[1.0, 1.0, 1.0]])
        b_ineq = jnp.array([0.0])

        tol = 1e-4
        result = solve_qp(
            lambda v: H @ v,
            g,
            jnp.zeros((0, 3)),
            jnp.zeros(0),
            A_ineq,
            b_ineq,
            tol=tol,
        )
        np.testing.assert_array_less(tol - 1e-15, float(result.final_working_tol))
