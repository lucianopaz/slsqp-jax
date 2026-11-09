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
  :func:`~slsqp_jax.sqpdax.linalg.projection.cg_normal_equations`,
  :func:`~slsqp_jax.sqpdax.linalg.projection.pcg`).
* :mod:`~slsqp_jax.sqpdax.linalg.scaled_normal_equations` — bound/slack-row
  elimination for the interior-point normal equations ``Â Âᵀ``: the
  explicit pseudo-inverse Schur complement
  (:class:`~slsqp_jax.sqpdax.linalg.scaled_normal_equations.SchurNormalEquations`)
  and the matrix-free nested PCG
  (:func:`~slsqp_jax.sqpdax.linalg.scaled_normal_equations.slack_eliminated_normal_solver`).
* :mod:`~slsqp_jax.sqpdax.linalg.operator_norm` — power-iteration
  estimates of spectral norms used by the diagnostics
  (:func:`~slsqp_jax.sqpdax.linalg.operator_norm.power_iteration_norm`,
  :func:`~slsqp_jax.sqpdax.linalg.operator_norm.spectral_norm_estimate`).
* :mod:`~slsqp_jax.sqpdax.linalg.reduced_hessian` — dense kernels of the
  limited-memory reduced-Hessian secant: QR with a non-negative diagonal
  (:func:`~slsqp_jax.sqpdax.linalg.reduced_hessian.positive_qr`) and
  twice-iterated Gram–Schmidt
  (:func:`~slsqp_jax.sqpdax.linalg.reduced_hessian.orthogonalize_against`),
  the BFGS update of the reduced-Hessian factor
  (:func:`~slsqp_jax.sqpdax.linalg.reduced_hessian.bfgs_factor_update`) and
  the removal of the oldest basis vector
  (:func:`~slsqp_jax.sqpdax.linalg.reduced_hessian.drop_oldest_basis_vector`).
"""

from . import (
    box,
    operator_norm,
    projection,
    reduced_hessian,
    scaled_normal_equations,
    steihaug,
)
from .box import box_fraction, box_ray_length
from .operator_norm import power_iteration_norm, spectral_norm_estimate
from .projection import cg_normal_equations, null_space_projector, pcg
from .reduced_hessian import (
    bfgs_factor_update,
    drop_oldest_basis_vector,
    orthogonalize_against,
    positive_qr,
)
from .scaled_normal_equations import (
    NormalEquationsStrategy,
    ResolvedNormalEquationsStrategy,
    SchurNormalEquations,
    resolve_normal_equations_strategy,
    slack_eliminated_normal_solver,
    strategy_code,
)
from .steihaug import (
    SteihaugCGResult,
    boundary_step_length,
    steihaug_cg,
    steihaug_step,
)

__all__ = [
    "box",
    "operator_norm",
    "projection",
    "reduced_hessian",
    "scaled_normal_equations",
    "steihaug",
    "power_iteration_norm",
    "spectral_norm_estimate",
    "bfgs_factor_update",
    "drop_oldest_basis_vector",
    "orthogonalize_against",
    "positive_qr",
    "NormalEquationsStrategy",
    "ResolvedNormalEquationsStrategy",
    "SchurNormalEquations",
    "resolve_normal_equations_strategy",
    "slack_eliminated_normal_solver",
    "strategy_code",
    "pcg",
    "SteihaugCGResult",
    "boundary_step_length",
    "steihaug_cg",
    "steihaug_step",
    "box_fraction",
    "box_ray_length",
    "cg_normal_equations",
    "null_space_projector",
]
