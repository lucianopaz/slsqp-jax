"""Merit functions for globalization of SQP / interior-point steps.

A merit combines the objective with a measure of constraint violation
(and optionally an interior-point barrier). Step controllers use it to
accept or reject candidate steps without requiring the full KKT residual
to decrease.
"""

from abc import abstractmethod
from typing import cast

from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array

from .autodiff_utils import ad_proxy_from_constants
from .barrier import Barrier
from .primal import InteriorPointPrimal, Primal
from .problem.basic import ProblemProtocol
from .types import Scalar

__all__ = [
    "safe_norm",
    "Merit",
    "NormMerit",
]


def safe_norm(v: Array, ord: int = 2) -> Scalar:
    """Norm of ``v`` with a zero subgradient at the origin.

    :func:`jax.numpy.linalg.norm` is not differentiable at ``v == 0``
    (the gradient of ‖·‖ is undefined / NaN there). This helper evaluates
    the norm on a filled-in surrogate when ``v`` is identically zero and
    replaces both the value and the gradient contribution with zero.

    Parameters
    ----------
    v
        Vector whose norm is requested.
    ord
        Norm order forwarded to :func:`jax.numpy.linalg.norm`
        (e.g. ``1`` or ``2``).

    Returns
    -------
    Scalar
        ``‖v‖_ord``, or ``0`` when every entry of ``v`` is zero.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from slsqp_jax.sqpdax.merit import safe_norm
    >>> float(safe_norm(jnp.zeros(3), ord=2))
    0.0
    >>> jax.grad(lambda v: safe_norm(v, ord=2))(jnp.zeros(3)).tolist()
    [0.0, 0.0, 0.0]
    >>> float(safe_norm(jnp.array([3.0, 4.0]), ord=2))
    5.0
    """
    is_zero = jnp.all(v == 0)
    v_safe = jnp.where(is_zero, jnp.ones_like(v), v)
    n = jnp.linalg.norm(v_safe, ord=ord)
    return jnp.where(is_zero, jnp.zeros_like(n), n)


class Merit(Module):
    """Abstract merit that scores a primal iterate for globalization.

    Concrete subclasses implement :meth:`__call__`. When ``barrier`` is
    set the merit expects an
    :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal` and includes the
    barrier contribution; otherwise it uses an exterior penalty on
    inequality / bound violations.

    Attributes
    ----------
    problem
        NLP whose objective and constraints enter the merit.
    barrier
        Optional interior-point barrier on slacks. ``None`` selects the
        exterior (penalty) formulation.
    problem_weight
        Weight on the objective term.
    barrier_weight
        Weight on the barrier term when ``barrier`` is set.
    feasibility_weight
        Weight on the constraint-violation norms.
    """

    problem: ProblemProtocol
    barrier: Barrier | None = None
    problem_weight: Scalar = field(default=1.0)
    barrier_weight: Scalar = field(default=1.0)
    feasibility_weight: Scalar = field(default=1.0)

    @property
    def has_interior_point(self) -> bool:
        """Whether this merit uses the interior-point (barrier) form."""
        return self.barrier is not None

    @abstractmethod
    def __call__(self, x: Primal, *args, **kwargs) -> Scalar:
        """Evaluate the merit at primal ``x``.

        Parameters
        ----------
        x
            Primal iterate (or :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal`
            when :attr:`has_interior_point` is true).
        *args, **kwargs
            Extra arguments forwarded to ``problem(x, ...)``.

        Returns
        -------
        Scalar
            Merit value at ``x``.
        """
        ...


class NormMerit(Merit):
    """Norm-penalized merit with optional interior-point barrier.

    Uses :func:`~slsqp_jax.sqpdax.autodiff_utils.ad_proxy_from_constants` so
    reverse-mode AD of the merit recovers the problem Jacobians stored in
    the evaluated NLP, without differentiating the residual callables
    themselves.

    Exterior form (``barrier is None``)::

        φ(x) = ω_f f(x)
             + ω_c ( ‖g(x)‖ + ‖ [h(x)]₊ ⊕ [ℓ-x]₊ ⊕ [x-u]₊ ‖ )

    where ``⊕`` denotes concatenation, ``[·]₊ = max(0, ·)``, and inactive
    (null) bounds contribute zeros.

    Interior-point form (``barrier`` set, ``x`` an
    :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal`)::

        φ(x,s) = ω_f f(x) + ω_b B(s)
               + ω_c ( ‖g(x)‖ + ‖ (h(x)+s) ⊕ (ℓ-x+s_ℓ) ⊕ (x-u+s_u) ‖ )

    with null-bound slack entries zeroed. ``ω_f``, ``ω_b``, and ``ω_c`` are
    :attr:`problem_weight`, :attr:`barrier_weight`, and
    :attr:`feasibility_weight`.

    Attributes
    ----------
    norm
        Order of the feasibility norms (``1`` or ``2`` typical).
    """

    norm: int = field(default=1)

    def __call__(self, x: Primal, *args, **kwargs) -> Scalar:
        """Evaluate the norm merit at ``x``.

        Parameters
        ----------
        x
            Primal iterate. Must be an
            :class:`~slsqp_jax.sqpdax.primal.InteriorPointPrimal` when
            :attr:`has_interior_point` is true.
        *args, **kwargs
            Extra arguments forwarded to ``problem(x, ...)``.

        Returns
        -------
        Scalar
            Merit value ``φ(x)`` (see class docstring for the formula).
        """
        evaled_problem = self.problem(x, *args, **kwargs)

        # Custom first-order proxies so AD of the merit uses the stored
        # Jacobians rather than differentiating the residual callables.
        fn = ad_proxy_from_constants(evaled_problem.fn_val, evaled_problem.grad_val)
        eq_fn = ad_proxy_from_constants(
            evaled_problem.eq_fn_val, evaled_problem.eq_fn_jac_val
        )
        ineq_fn = ad_proxy_from_constants(
            evaled_problem.ineq_fn_val, evaled_problem.ineq_fn_jac_val
        )

        output = self.problem_weight * fn(x)
        eq_fn_val = eq_fn(x)
        ineq_fn_val = ineq_fn(x)
        lb_fn_val = jnp.where(
            evaled_problem.null_lb,
            0.0,
            evaled_problem.lb - x.x,
        )
        ub_fn_val = jnp.where(
            evaled_problem.null_ub,
            0.0,
            x.x - evaled_problem.ub,
        )

        if self.has_interior_point:
            slack = cast(InteriorPointPrimal, x).slack
            evaled_barrier = self.barrier_weight * cast(Barrier, self.barrier).fn(slack)
            ineq_fn_val = ineq_fn_val + slack.s
            lb_fn_val = lb_fn_val + jnp.where(
                evaled_problem.null_lb,
                0.0,
                slack.s_lb,
            )
            ub_fn_val = ub_fn_val + jnp.where(
                evaled_problem.null_ub,
                0.0,
                slack.s_ub,
            )
            ineq = jnp.concatenate([ineq_fn_val, lb_fn_val, ub_fn_val])
        else:
            evaled_barrier = 0.0
            ineq = jnp.concatenate(
                [
                    jnp.maximum(0.0, ineq_fn_val),
                    jnp.maximum(0.0, lb_fn_val),
                    jnp.maximum(0.0, ub_fn_val),
                ]
            )

        output += evaled_barrier + self.feasibility_weight * (
            safe_norm(eq_fn_val, ord=self.norm) + safe_norm(ineq, ord=self.norm)
        )
        return output
