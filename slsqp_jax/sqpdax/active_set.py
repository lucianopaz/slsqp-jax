"""Working-set masks for active inequalities and variable bounds."""

import equinox as eqx
from equinox import Module, tree_at
from jax import numpy as jnp
from jaxtyping import Array, Bool

from .dual import Dual
from .lagrangian import EvaluatedLagrangian
from .primal import PrimalType
from .problem import EvaluatedProblem

__all__ = [
    "ActiveSet",
]


class ActiveSet(Module):
    """Boolean working set for inequalities and bounds in an SQP subproblem.

    Attributes
    ----------
    meq
        Number of equality constraints (always active).
    active_inequalities
        Mask of length ``mineq``; ``True`` where an inequality is in the
        working set.
    active_lb
        Mask of length ``n``; ``True`` where a lower bound is active.
    active_ub
        Mask of length ``n``; ``True`` where an upper bound is active.

    Notes
    -----
    Equalities are always treated as active. Inactive inequality / bound rows
    are zeroed by :meth:`mask_dual`, :meth:`mask_problem`, and
    :meth:`mask_lagrangian` so the reduced KKT system only sees the working
    set. Inactive finite bounds are also marked null so bound Jacobian rows
    drop out of the KKT MVP.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.active_set import ActiveSet
    >>> from slsqp_jax.sqpdax.dual import Dual
    >>> active = ActiveSet(
    ...     meq=1,
    ...     active_inequalities=jnp.array([True, False]),
    ...     active_lb=jnp.array([True, False]),
    ...     active_ub=jnp.array([False, True]),
    ... )
    >>> active.active_gen.tolist()
    [True, True, False]
    >>> dual = Dual(
    ...     eq_multipliers=jnp.array([1.0]),
    ...     ineq_multipliers=jnp.array([2.0, 3.0]),
    ...     lb_multipliers=jnp.array([4.0, 5.0]),
    ...     ub_multipliers=jnp.array([6.0, 7.0]),
    ... )
    >>> masked = active.mask_dual(dual)
    >>> masked.ineq_multipliers.tolist()
    [2.0, 0.0]
    >>> masked.lb_multipliers.tolist()
    [4.0, 0.0]
    >>> masked.ub_multipliers.tolist()
    [0.0, 7.0]
    """

    meq: int = eqx.field(static=True)
    active_inequalities: Bool[Array, " mineq"]
    active_lb: Bool[Array, " n"]
    active_ub: Bool[Array, " n"]

    def mask_dual(self, dual: Dual) -> Dual:
        """Zero multipliers for inactive inequalities and bounds.

        Equality multipliers are left unchanged. Inactive inequality,
        lower-bound, and upper-bound multipliers are replaced with zeros.

        Parameters
        ----------
        dual
            Multipliers to restrict to the working set.

        Returns
        -------
        Dual
            Copy of ``dual`` with inactive rows zeroed.
        """
        dual = tree_at(
            lambda d: d.ineq_multipliers,
            dual,
            jnp.where(
                self.active_inequalities,
                dual.ineq_multipliers,
                jnp.zeros_like(dual.ineq_multipliers),
            ),
        )
        dual = tree_at(
            lambda d: d.lb_multipliers,
            dual,
            jnp.where(
                self.active_lb,
                dual.lb_multipliers,
                jnp.zeros_like(dual.lb_multipliers),
            ),
        )
        return tree_at(
            lambda d: d.ub_multipliers,
            dual,
            jnp.where(
                self.active_ub,
                dual.ub_multipliers,
                jnp.zeros_like(dual.ub_multipliers),
            ),
        )

    def mask_problem(
        self, problem: EvaluatedProblem[PrimalType]
    ) -> EvaluatedProblem[PrimalType]:
        """Zero inactive inequality / bound rows of an evaluated problem.

        Parameters
        ----------
        problem
            Pointwise NLP evaluation to restrict to the working set.

        Returns
        -------
        EvaluatedProblem
            Copy of ``problem`` with inactive inequality and bound rows
            removed from the reduced KKT view.

        Notes
        -----
        Inactive inequality values and Jacobian rows are set to zero.
        Inactive bound values are zeroed, and inactive finite bounds are
        marked null (``null_lb`` / ``null_ub``): the KKT MVP off-diagonal
        blocks read the ``±1`` bound Jacobian from those masks (``null`` →
        no row), so an inactive-but-finite bound is only removed by marking
        it null. Equality residuals and Jacobians are unchanged.

        ``ineq_fn_qvp`` is left as-is: it is a static field (``tree_at``
        cannot replace it) and is only ever contracted with inequality
        multipliers, which :meth:`mask_dual` already zeroes on inactive rows.
        """
        masked_ineq_fn_val = jnp.where(
            self.active_inequalities,
            problem.ineq_fn_val,
            jnp.zeros_like(problem.ineq_fn_val),
        )
        masked_ineq_fn_jac_val = jnp.where(
            self.active_inequalities[:, None],
            problem.ineq_fn_jac_val,
            jnp.zeros_like(problem.ineq_fn_jac_val),
        )
        masked_lb = jnp.where(
            self.active_lb,
            problem.lb,
            jnp.zeros_like(problem.lb),
        )
        masked_ub = jnp.where(
            self.active_ub,
            problem.ub,
            jnp.zeros_like(problem.ub),
        )
        masked_null_lb = problem.null_lb | ~self.active_lb
        masked_null_ub = problem.null_ub | ~self.active_ub

        return tree_at(
            lambda p: (
                p.ineq_fn_val,
                p.ineq_fn_jac_val,
                p.lb,
                p.ub,
                p.null_lb,
                p.null_ub,
            ),
            problem,
            (
                masked_ineq_fn_val,
                masked_ineq_fn_jac_val,
                masked_lb,
                masked_ub,
                masked_null_lb,
                masked_null_ub,
            ),
        )

    def mask_lagrangian(
        self, lagrangian: EvaluatedLagrangian[PrimalType]
    ) -> EvaluatedLagrangian[PrimalType]:
        """Apply :meth:`mask_problem` and :meth:`mask_dual` to a Lagrangian.

        Parameters
        ----------
        lagrangian
            Cached Lagrangian at a reference point.

        Returns
        -------
        EvaluatedLagrangian
            Copy whose evaluated problem and dual are restricted to the
            working set. The secant (if any) is left unchanged.
        """
        return tree_at(
            lambda lag: (lag.evaluated, lag.dual),
            lagrangian,
            (self.mask_problem(lagrangian.evaluated), self.mask_dual(lagrangian.dual)),
        )

    @property
    def active_gen(self) -> Bool[Array, " meq+mineq"]:
        """Concatenated active mask for equalities then inequalities.

        Equalities are always ``True``; inequality entries copy
        :attr:`active_inequalities`.

        Returns
        -------
        jax.Array
            Boolean vector of length ``meq + mineq``.
        """
        return jnp.concatenate(
            [jnp.ones((self.meq,), dtype=bool), self.active_inequalities]
        )
