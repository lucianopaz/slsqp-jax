"""Inner equality-constrained QP solvers used by the SLSQP QP layer.

This subpackage replaces the legacy monolithic ``slsqp_jax/inner_solver.py``
with one module per concern: the abstract base, the shared masking /
bound-fixing helper, the Krylov primitives, the shared projected-PCG
driver, and one module per concrete strategy.
"""

from slsqp_jax.inner.base import AbstractInnerSolver
from slsqp_jax.inner.cholesky import ProjectedCGCholesky
from slsqp_jax.inner.craig import ProjectedCGCraig
from slsqp_jax.inner.hr_stcg import HRInexactSTCG
from slsqp_jax.inner.krylov import (
    build_cg_step,
    craig_solve,
    pminres_qlp_solve,
    solve_unconstrained_cg,
)
from slsqp_jax.inner.minres_qlp import MinresQLPSolver
from slsqp_jax.state import InnerSolveResult, ProjectionContext

__all__ = [
    "AbstractInnerSolver",
    "HRInexactSTCG",
    "InnerSolveResult",
    "MinresQLPSolver",
    "ProjectedCGCholesky",
    "ProjectedCGCraig",
    "ProjectionContext",
    "build_cg_step",
    "craig_solve",
    "pminres_qlp_solve",
    "solve_unconstrained_cg",
]
