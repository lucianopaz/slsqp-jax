"""Native, fine-grained result types for sqpdax minimisers.

The owned sqpdax driver reports these project-native enumerations. Conversion
to :class:`optimistix.RESULTS` is deliberately isolated in
:class:`ResultAdapter`, which is used by the Optimistix compatibility layer.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import optimistix as optx
from equinox import Enumeration, Module
from equinox._enum import EnumerationItem
from jax import numpy as jnp
from jaxtyping import Array, Bool

__all__ = [
    "MINIMISER_RESULTS",
    "ResultType",
    "ResultAdapter",
    "is_successful",
]


class MINIMISER_RESULTS(Enumeration):
    """Outcomes shared by every native sqpdax minimiser."""

    running = "The minimiser has not terminated."
    successful = "The minimiser converged successfully."
    nonfinite = "A non-finite iterate or optimality quantity was encountered."
    max_steps_reached = "The maximum number of outer iterations was reached."


ResultType = TypeVar("ResultType", bound=MINIMISER_RESULTS)


class ResultAdapter(Module, Generic[ResultType]):
    """Construct universal native outcomes and map them to Optimistix."""

    @property
    @abstractmethod
    def result_type(self) -> type[ResultType]:
        """Concrete native enumeration class."""
        ...

    def promote(self, result: MINIMISER_RESULTS) -> ResultType:
        """Promote a universal result into the concrete algorithm enumeration."""
        return self.result_type.promote(result)

    @property
    def running(self) -> ResultType:
        """Native running sentinel."""
        return self.promote(MINIMISER_RESULTS.running)

    @property
    def successful(self) -> ResultType:
        """Native successful result."""
        return self.promote(MINIMISER_RESULTS.successful)

    @property
    def nonfinite(self) -> ResultType:
        """Native non-finite result."""
        return self.promote(MINIMISER_RESULTS.nonfinite)

    @property
    def max_steps_reached(self) -> ResultType:
        """Native outer-iteration budget result."""
        return self.promote(MINIMISER_RESULTS.max_steps_reached)

    def is_successful(self, result: ResultType) -> Bool[Array, ""]:
        """Whether ``result`` denotes successful convergence."""
        return result == self.successful

    @abstractmethod
    def to_optimistix(self, result: ResultType) -> optx.RESULTS:
        """Collapse a native result to an Optimistix-compatible result."""
        ...


def is_successful(result: Any) -> Bool[Array, ""]:
    """Safely test native sqpdax or Optimistix results."""
    if isinstance(result, EnumerationItem):
        enum = result._enumeration
        if issubclass(enum, MINIMISER_RESULTS):
            base = MINIMISER_RESULTS
            successful = MINIMISER_RESULTS.successful
        elif issubclass(enum, optx.RESULTS):
            base = optx.RESULTS
            successful = optx.RESULTS.successful
        return (
            result == successful if enum is base else result == enum.promote(successful)
        )
    return jnp.asarray(False)  # pragma: no cover
