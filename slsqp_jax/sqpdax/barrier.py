"""Interior-point barrier terms on nonnegative slack variables."""

from typing import Callable
from abc import abstractmethod
import jax
from jax import numpy as jnp
from equinox import Module
import equinox as eqx
from jaxtyping import Array, Bool
from .primal import Slack
from .types import Scalar, InitializableModule


__all__ = ["EvaluatedBarrier", "Barrier", "LogBarrier"]


class EvaluatedBarrier(Module):
    """Barrier value, gradient, and HVP cached at a slack reference point.

    Produced by :meth:`Barrier.__call__`. The ``hvp`` field is a closure
    bound to ``slack_ref`` that maps a slack tangent to the barrier
    Hessian-vector product at that point.

    Attributes
    ----------
    slack_ref
        Slack vector at which the barrier was evaluated.
    weight
        Barrier weight (e.g. the interior-point parameter ``μ``) copied from
        the parent :class:`Barrier`.
    null_lb
        Mask of inactive lower bounds (``True`` entries contribute nothing
        to the barrier).
    null_ub
        Mask of inactive upper bounds.
    fn_val
        Scalar barrier value ``B(s)``.
    grad_val
        Barrier gradient w.r.t. the slack block, as a :class:`Slack`.
    hvp
        Map ``tangent ↦ ∇²B(slack_ref)[tangent]``.
    original
        The :class:`Barrier` instance that produced this evaluation (used
        for runtime type checks such as ``isinstance(..., LogBarrier)``).
    """

    slack_ref: Slack
    weight: Scalar
    null_lb: Bool[Array, " n"]
    null_ub: Bool[Array, " n"]
    fn_val: Scalar
    grad_val: Slack
    hvp: Callable[[Slack], Slack] = eqx.field(static=True)
    # Dynamic, not static: ``original`` is the source ``Barrier`` whose ``weight``
    # (mu), ``null_lb`` / ``null_ub`` are JAX arrays.  Storing it static froze
    # those arrays into the treedef ("A JAX array is being set as static!").  It is
    # only read for the ``isinstance(..., LogBarrier)`` type check, which works on
    # the dynamic pytree just the same.
    original: "Barrier"


class Barrier(InitializableModule):
    """Unevaluated barrier on inequality / bound slacks.

    Subclasses implement :meth:`fn`. Calling an instance evaluates the
    barrier at a :class:`~slsqp_jax.sqpdax.primal.Slack` and returns an
    :class:`EvaluatedBarrier` whose gradient and HVP are obtained via JAX
    autodiff of :meth:`fn`.

    Attributes
    ----------
    weight
        Nonnegative barrier weight (typically the IP parameter ``μ``).
    null_lb
        Mask of inactive lower-bound slacks.
    null_ub
        Mask of inactive upper-bound slacks.
    """

    weight: Scalar
    null_lb: Bool[Array, " n"]
    null_ub: Bool[Array, " n"]

    def __call__(self, slack: Slack, *args, **kwargs) -> EvaluatedBarrier:
        """Evaluate the barrier and its first- and second-order maps at ``slack``.

        Parameters
        ----------
        slack
            Slack variables at which to evaluate.
        *args, **kwargs
            Extra arguments forwarded to :meth:`fn`.

        Returns
        -------
        EvaluatedBarrier
            Cached value, gradient, and HVP closure at ``slack``.
        """
        fn_val = self.fn(slack, *args, **kwargs)

        def wrapped_grad(slack: Slack) -> Slack:
            return jax.grad(self.fn)(slack, *args, **kwargs)

        grad_val = wrapped_grad(slack)

        def hvp(tangent: Slack) -> Slack:
            return jax.jvp(wrapped_grad, (slack,), (tangent,))[1]

        return EvaluatedBarrier(
            slack_ref=slack,
            weight=self.weight,
            fn_val=fn_val,
            grad_val=grad_val,
            hvp=hvp,
            null_lb=self.null_lb,
            null_ub=self.null_ub,
            original=self,
        )

    @abstractmethod
    def fn(self, slack: Slack, *args, **kwargs) -> Scalar:
        """Scalar barrier value ``B(slack)``.

        Parameters
        ----------
        slack
            Slack variables.
        *args, **kwargs
            Extra arguments (ignored by :class:`LogBarrier`).

        Returns
        -------
        Scalar
            Barrier value.
        """
        ...

    def grad(self, slack: Slack) -> Slack:
        """Barrier gradient at ``slack``.

        Parameters
        ----------
        slack
            Slack variables.

        Returns
        -------
        Slack
            ``∇B(slack)`` with the same block structure as ``slack``.
        """
        return self(slack).grad_val

    def hvp(self, slack: Slack, tangent: Slack) -> Slack:
        """Barrier Hessian-vector product at ``slack``.

        Parameters
        ----------
        slack
            Slack variables (evaluation point).
        tangent
            Slack-shaped tangent direction.

        Returns
        -------
        Slack
            ``∇²B(slack)[tangent]``.
        """
        return self(slack).hvp(tangent)


class LogBarrier(Barrier):
    """Logarithmic barrier ``-μ (∑ log s + ∑ log s_lb + ∑ log s_ub)``.

    Inactive bound slacks (entries where ``null_lb`` / ``null_ub`` are
    ``True``) are replaced by ``1`` inside the logarithm so they contribute
    nothing. When ``weight == 0`` the barrier evaluates to zero (with zero
    derivatives under JAX autodiff of the guarded expression).
    """

    def fn(self, slack: Slack, *args, **kwargs) -> Scalar:
        """Evaluate the log barrier at ``slack``.

        Parameters
        ----------
        slack
            Positive slack variables.
        *args, **kwargs
            Ignored; accepted for :class:`Barrier` compatibility.

        Returns
        -------
        Scalar
            ``-weight * (∑ log s + ∑_{active} log s_lb + ∑_{active} log s_ub)``,
            or ``0`` when ``weight == 0``.
        """
        return -jnp.where(
            self.weight == 0,
            0.0,
            self.weight
            * (
                jnp.sum(jnp.log(slack.s))
                + jnp.sum(
                    jnp.log(
                        jnp.where(
                            self.null_lb,
                            1.0,
                            slack.s_lb,
                        ),
                    ),
                )
                + jnp.sum(
                    jnp.log(
                        jnp.where(
                            self.null_ub,
                            1.0,
                            slack.s_ub,
                        ),
                    )
                )
            ),
        )
