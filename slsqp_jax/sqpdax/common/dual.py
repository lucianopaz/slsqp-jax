"""Dual multipliers for equality, inequality, and bound constraints."""

from typing import Self

from equinox import Module
from jax import numpy as jnp
from jaxtyping import Array, Float

from .types import Vector_meq, Vector_mineq, Vector_n

__all__ = [
    "Dual",
]


class Dual(Module):
    """Lagrange multipliers for equalities, inequalities, and bounds.

    The flat layout used by :meth:`flatten` / :meth:`from_flat` is
    ``[eq_multipliers, ineq_multipliers, lb_multipliers, ub_multipliers]``.

    Attributes
    ----------
    eq_multipliers
        Multipliers for equality constraints, length ``meq``.
    ineq_multipliers
        Multipliers for inequality constraints, length ``mineq``.
    lb_multipliers
        Multipliers for lower bounds on the primal, length ``n``.
    ub_multipliers
        Multipliers for upper bounds on the primal, length ``n``.
    """

    eq_multipliers: Vector_meq
    ineq_multipliers: Vector_mineq
    lb_multipliers: Vector_n
    ub_multipliers: Vector_n

    @property
    def n(self) -> int:
        """Number of decision variables (length of bound multipliers)."""
        return self.lb_multipliers.shape[-1]

    @property
    def meq(self) -> int:
        """Number of equality constraints."""
        return self.eq_multipliers.shape[-1]

    @property
    def mineq(self) -> int:
        """Number of inequality constraints."""
        return self.ineq_multipliers.shape[-1]

    def flatten(self) -> Float[Array, " meq+mineq+2*n"]:
        """Concatenate multipliers as ``[eq, ineq, lb, ub]``.

        Returns
        -------
        jax.Array
            Flat vector of length ``meq + mineq + 2 * n``.
        """
        return jnp.concatenate(
            [
                self.eq_multipliers,
                self.ineq_multipliers,
                self.lb_multipliers,
                self.ub_multipliers,
            ],
            axis=0,
        )

    @classmethod
    def from_flat(cls, arr: Float[Array, " meq+mineq+2*n"], *sizes: int) -> Self:
        """Build a :class:`Dual` by slicing a flat array.

        The flat layout matches :meth:`flatten`:
        ``[eq_multipliers, ineq_multipliers, lb_multipliers, ub_multipliers]``.

        Parameters
        ----------
        arr
            Flat array of length at least ``meq + mineq + 2 * n``.
        *sizes
            Integers ``(n, mineq, meq)`` in that order.

        Returns
        -------
        Dual
            Module reconstructed from the leading slices of ``arr``.

        Raises
        ------
        ValueError
            If ``sizes`` does not contain exactly three integers.
        """
        try:
            n, mineq, meq = sizes
        except ValueError:
            raise ValueError(
                "Dual.from_flat expects 3 sizes arguments: n, mineq, meq"
            ) from None
        eq_multipliers = arr[:meq]
        ineq_multipliers = arr[meq : meq + mineq]
        lb_multipliers = arr[meq + mineq : meq + mineq + n]
        ub_multipliers = arr[meq + mineq + n : meq + mineq + n * 2]
        return cls(eq_multipliers, ineq_multipliers, lb_multipliers, ub_multipliers)

    @property
    def sizes(self) -> tuple[int, int, int]:
        """Size tuple ``(n, mineq, meq)`` accepted by :meth:`from_flat`."""
        return (self.n, self.mineq, self.meq)
