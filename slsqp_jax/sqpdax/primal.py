"""Primal decision variables and interior-point slack variables."""

from typing import Self

from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Float

from .types import Vector_mineq, Vector_n

__all__ = [
    "Primal",
    "Slack",
    "InteriorPointPrimal",
]


class Primal(Module):
    """Decision variables for an equality/inequality constrained NLP.

    Attributes
    ----------
    x
        Primal vector of length ``n``.
    """

    x: Vector_n

    @property
    def n(self) -> int:
        """Number of decision variables."""
        return self.x.shape[-1]

    def flatten(self) -> Vector_n:
        """Return the contiguous 1-D layout of this primal.

        Returns
        -------
        Vector_n
            The decision vector ``x``.
        """
        return self.x

    @classmethod
    def from_flat(cls, arr: Vector_n, *sizes: int) -> Self:
        """Build a :class:`Primal` by slicing a flat array.

        Parameters
        ----------
        arr
            Flat array whose leading ``n`` entries become ``x``.
        *sizes
            A single integer ``n``, the number of decision variables.

        Returns
        -------
        Primal
            Module with ``x = arr[:n]``.

        Raises
        ------
        ValueError
            If ``sizes`` does not contain exactly one integer.
        """
        try:
            [n] = sizes
        except ValueError:
            raise ValueError("Primal.from_flat expects 1 size argument: n") from None
        return cls(arr[:n])

    @property
    def sizes(self) -> tuple[int]:
        """Size tuple ``(n,)`` accepted by :meth:`from_flat`."""
        return (self.n,)


class Slack(Module):
    """Nonnegative slack variables for inequalities and bound constraints.

    Used by interior-point formulations to convert inequalities into
    equalities. Bound slacks ``s_lb`` / ``s_ub`` have length ``n``; the
    inequality slack ``s`` has length ``mineq``.

    Attributes
    ----------
    s
        Slacks for general inequality constraints, length ``mineq``.
    s_lb
        Slacks for lower bounds on ``x``, length ``n``.
    s_ub
        Slacks for upper bounds on ``x``, length ``n``.
    """

    s: Vector_mineq
    s_lb: Vector_n
    s_ub: Vector_n

    @property
    def n(self) -> int:
        """Number of decision variables (length of bound slacks)."""
        return self.s_lb.shape[-1]

    @property
    def mineq(self) -> int:
        """Number of general inequality constraints."""
        return self.s.shape[-1]

    def flatten(self) -> Float[Array, " 2*n+mineq"]:
        """Concatenate slacks as ``[s, s_lb, s_ub]``.

        Returns
        -------
        jax.Array
            Flat vector of length ``mineq + 2 * n``.
        """
        return jnp.concatenate([self.s, self.s_lb, self.s_ub], axis=0)

    @classmethod
    def from_flat(cls, arr: Float[Array, " 2*n+mineq"], *sizes: int) -> Self:
        """Build a :class:`Slack` by slicing a flat array.

        The flat layout matches :meth:`flatten`: ``[s, s_lb, s_ub]``.

        Parameters
        ----------
        arr
            Flat array of length at least ``mineq + 2 * n``.
        *sizes
            Integers ``(n, mineq)`` in that order.

        Returns
        -------
        Slack
            Module reconstructed from the leading slices of ``arr``.

        Raises
        ------
        ValueError
            If ``sizes`` does not contain exactly two integers.
        """
        try:
            n, mineq = sizes
        except ValueError:
            raise ValueError(
                "Slack.from_flat expects 2 sizes arguments: n, mineq"
            ) from None
        s = arr[:mineq]
        s_lb = arr[mineq : mineq + n]
        s_ub = arr[mineq + n : mineq + n * 2]
        return cls(s, s_lb, s_ub)

    @property
    def sizes(self) -> tuple[int, int]:
        """Size tuple ``(n, mineq)`` accepted by :meth:`from_flat`."""
        return (self.n, self.mineq)


class InteriorPointPrimal(Primal):
    """Primal variables augmented with interior-point slacks.

    Extends :class:`Primal` with a :class:`Slack` block so the combined
    flat layout is ``[x, s, s_lb, s_ub]``.

    Attributes
    ----------
    x
        Decision vector of length ``n`` (inherited from :class:`Primal`).
    slack
        Associated inequality and bound slacks.
    """

    slack: Slack

    @property
    def mineq(self) -> int:
        """Number of general inequality constraints."""
        return self.slack.mineq

    def flatten(self) -> Float[Array, " 3*n+mineq"]:
        """Concatenate the decision vector and slack block.

        Returns
        -------
        jax.Array
            Flat vector ``[x, slack.flatten()]`` of length
            ``n + mineq + 2 * n``.
        """
        return jnp.concatenate([self.x, self.slack.flatten()], axis=0)

    @classmethod
    def from_flat(cls, arr: Float[Array, " 3*n+mineq"], *sizes: int) -> Self:
        """Build an :class:`InteriorPointPrimal` by slicing a flat array.

        The flat layout matches :meth:`flatten`:
        ``[x, s, s_lb, s_ub]``.

        Parameters
        ----------
        arr
            Flat array of length at least ``3 * n + mineq``.
        *sizes
            Integers ``(n, mineq)`` in that order.

        Returns
        -------
        InteriorPointPrimal
            Module reconstructed from the leading slices of ``arr``.

        Raises
        ------
        ValueError
            If ``sizes`` does not contain exactly two integers.
        """
        try:
            n, mineq = sizes
        except ValueError:
            raise ValueError(
                "InteriorPointPrimal.from_flat expects 2 sizes arguments: n, mineq"
            ) from None
        x = arr[:n]
        slack = Slack.from_flat(arr[n:], n, mineq)
        return cls(x, slack)

    @property
    def sizes(self) -> tuple[int, int]:
        """Size tuple ``(n, mineq)`` accepted by :meth:`from_flat`."""
        return (self.n, self.mineq)
