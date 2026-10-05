"""Shared linear-algebra routines used across ``sqpdax`` solvers.

Pure, matrix-free building blocks with no dependency on the subproblem or
solver classes, so that every solver reuses one implementation:

* :mod:`~slsqp_jax.sqpdax.linalg.steihaug` — the Steihaug–Toint truncated
  conjugate-gradient kernel (:func:`~slsqp_jax.sqpdax.linalg.steihaug.steihaug_cg`,
  optionally projected onto a subspace, which also covers CGLS) and its
  boundary rules (:func:`~slsqp_jax.sqpdax.linalg.steihaug.boundary_step_length`,
  :func:`~slsqp_jax.sqpdax.linalg.steihaug.steihaug_step`).
* :mod:`~slsqp_jax.sqpdax.linalg.box` — ray / box intersections behind the
  fraction-to-boundary backtracks
  (:func:`~slsqp_jax.sqpdax.linalg.box.box_ray_length`,
  :func:`~slsqp_jax.sqpdax.linalg.box.box_fraction`).
* :mod:`~slsqp_jax.sqpdax.linalg.projection` — matrix-free orthogonal
  projection onto ``null(A)`` via CG on the normal equations
  (:func:`~slsqp_jax.sqpdax.linalg.projection.null_space_projector`,
  :func:`~slsqp_jax.sqpdax.linalg.projection.cg_normal_equations`).
"""

from . import box, projection, steihaug
from .box import box_fraction, box_ray_length
from .projection import cg_normal_equations, null_space_projector
from .steihaug import (
    SteihaugCGResult,
    boundary_step_length,
    steihaug_cg,
    steihaug_step,
)

__all__ = [
    "box",
    "projection",
    "steihaug",
    "SteihaugCGResult",
    "boundary_step_length",
    "steihaug_cg",
    "steihaug_step",
    "box_fraction",
    "box_ray_length",
    "cg_normal_equations",
    "null_space_projector",
]
