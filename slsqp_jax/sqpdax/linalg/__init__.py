"""Shared linear-algebra routines used across ``sqpdax`` solvers.

Pure, matrix-free building blocks with no dependency on the subproblem or
solver classes, so that every solver reuses one implementation:

* :mod:`~slsqp_jax.sqpdax.linalg.steihaug` — trust-region boundary rules of
  the Steihaug–Toint truncated conjugate-gradient family
  (:func:`~slsqp_jax.sqpdax.linalg.steihaug.boundary_step_length`,
  :func:`~slsqp_jax.sqpdax.linalg.steihaug.steihaug_step`).
"""

from . import steihaug
from .steihaug import boundary_step_length, steihaug_step

__all__ = [
    "steihaug",
    "boundary_step_length",
    "steihaug_step",
]
