"""Unit tests for :mod:`slsqp_jax.sqpdax.preconditioner.strategy`."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import Array

from slsqp_jax.sqpdax.dual import Dual
from slsqp_jax.sqpdax.lagrangian import Lagrangian
from slsqp_jax.sqpdax.preconditioner import (
    DiagonalPreconditioner,
    NoPreconditioner,
    PreconditionerContext,
    PreconditionerStrategy,
    SecantPreconditioner,
    StochasticDiagonalPreconditioner,
)
from slsqp_jax.sqpdax.primal import Primal
from slsqp_jax.sqpdax.problem.basic import Problem
from slsqp_jax.sqpdax.secant import LBFGS
from slsqp_jax.sqpdax.types import Aux
from tests.sqpdax.lagrangian.conftest import (
    empty_fn,
    empty_hvp,
    empty_jac,
)

from .conftest import make_spd_matrix


def _problem_with_hessian(matrix: Array) -> Problem:
    """Unconstrained ``½ xᵀ H x`` with exact HVP ``H v``."""
    n = matrix.shape[0]

    def fn(x: Array) -> tuple[Array, Aux]:
        return (0.5 * x @ (matrix @ x), None)

    lb = jnp.full(n, -jnp.inf)
    ub = jnp.full(n, jnp.inf)
    return Problem(
        fn=fn,
        grad=lambda x: matrix @ x,
        hvp=lambda x, v: matrix @ v,
        eq_fn=empty_fn,
        ineq_fn=empty_fn,
        eq_fn_jac=empty_jac,
        ineq_fn_jac=empty_jac,
        eq_fn_hvp=empty_hvp,
        ineq_fn_hvp=empty_hvp,
        lb=lb,
        ub=ub,
        null_lb=jnp.ones(n, dtype=bool),
        null_ub=jnp.ones(n, dtype=bool),
        n=n,
        meq=0,
        mineq=0,
    )


def _lbfgs(n: int) -> LBFGS:
    secant = LBFGS(n=n, memory=4)
    for k in range(3):
        s = jnp.zeros(n).at[k % n].set(1.0) + 0.1
        secant = secant.append(s, (k + 2.0) * s)
    return secant


def _context(
    matrix: Array | None = None,
    *,
    secant: LBFGS | None = None,
    step: int = 0,
) -> PreconditionerContext:
    """Context at ``x = 1`` with an exact Lagrangian iff ``matrix`` is given."""
    n = 3 if matrix is None else matrix.shape[0]
    x = jnp.ones(n)
    lagrangian = None
    if matrix is not None:
        dual = Dual(
            eq_multipliers=jnp.zeros(0),
            ineq_multipliers=jnp.zeros(0),
            lb_multipliers=jnp.zeros(n),
            ub_multipliers=jnp.zeros(n),
        )
        lagrangian = Lagrangian(_problem_with_hessian(matrix))(Primal(x=x), dual)
    return PreconditionerContext(
        x_ref=x, secant=secant, lagrangian=lagrangian, step_count=jnp.asarray(step)
    )


@pytest.mark.parametrize(
    ("spec", "cls", "requires_secant", "requires_exact_hvp", "is_active"),
    [
        ({"kind": "none"}, NoPreconditioner, False, False, False),
        ({"kind": "lbfgs"}, SecantPreconditioner, True, False, True),
        (
            {"kind": "lbfgs", "require_secant": False},
            SecantPreconditioner,
            False,
            False,
            True,
        ),
        (
            {"kind": "stochastic_diagonal", "n_probes": 5},
            StochasticDiagonalPreconditioner,
            False,
            True,
            True,
        ),
    ],
    ids=["none", "lbfgs", "lbfgs-optional", "stochastic-diagonal"],
)
def test_strategy_registry_and_requirements(
    spec, cls, requires_secant, requires_exact_hvp, is_active
):
    """``from_spec`` resolves each kind and exposes its curvature requirements."""
    strategy = PreconditionerStrategy.from_spec(spec)
    assert type(strategy) is cls
    assert strategy.requires_secant is requires_secant
    assert strategy.requires_exact_hvp is requires_exact_hvp
    assert strategy.is_active is is_active
    params = {k: v for k, v in spec.items() if k != "kind"}
    for name, value in params.items():
        assert getattr(strategy, name) == value


@pytest.mark.parametrize(
    ("strategy", "with_secant"),
    [
        (NoPreconditioner(), True),
        (SecantPreconditioner(), False),
    ],
    ids=["none", "lbfgs-without-secant"],
)
def test_build_returns_none_without_curvature(strategy, with_secant):
    """Strategies without usable curvature defer to the solver's own default."""
    ctx = _context(secant=_lbfgs(3) if with_secant else None)
    assert strategy.build(ctx) is None


def test_secant_strategy_inverse_matches_secant():
    """The L-BFGS strategy applies ``H ≈ B⁻¹`` as the preconditioner inverse."""
    secant = _lbfgs(3)
    prec = SecantPreconditioner().build(_context(secant=secant))
    assert prec is not None
    v = jnp.array([0.3, -1.0, 2.0])
    assert jnp.allclose(prec.invert(v), secant.inverse_hvp(v), atol=1e-5)
    assert jnp.allclose(prec.pushforward(v), secant.hvp(v), atol=1e-5)


@pytest.mark.parametrize(
    ("hessian_diag", "expected"),
    [
        ((1.0, 4.0, 9.0), (1.0, 4.0, 9.0)),
        # Median |d| is 2: the floor is 2e-6, so |d| stays positive and SPD.
        ((-3.0, 0.0, 2.0), (3.0, 2e-6, 2.0)),
    ],
    ids=["spd", "indefinite-floored"],
)
def test_stochastic_diagonal_strategy_is_spd(hessian_diag, expected):
    """The exact-HVP diagonal is made SPD by ``|d|`` and a relative floor."""
    strategy = StochasticDiagonalPreconditioner(n_probes=3)
    prec = strategy.build(_context(jnp.diag(jnp.asarray(hessian_diag))))
    assert isinstance(prec, DiagonalPreconditioner)
    assert jnp.allclose(prec.diagonal, jnp.asarray(expected), rtol=1e-5)
    assert bool(jnp.all(prec.diagonal > 0))


def test_stochastic_diagonal_strategy_is_seeded_by_step():
    """The same step reproduces the estimate; a different step draws new probes."""
    strategy = StochasticDiagonalPreconditioner(n_probes=2)
    matrix = make_spd_matrix(3)
    first = strategy.build(_context(matrix, step=0))
    again = strategy.build(_context(matrix, step=0))
    later = strategy.build(_context(matrix, step=1))
    assert isinstance(first, DiagonalPreconditioner)
    assert isinstance(again, DiagonalPreconditioner)
    assert isinstance(later, DiagonalPreconditioner)
    assert jnp.array_equal(first.diagonal, again.diagonal)
    assert not jnp.array_equal(first.diagonal, later.diagonal)


def test_stochastic_diagonal_strategy_is_jittable():
    """``build`` traces with the step count as a dynamic value."""
    strategy = StochasticDiagonalPreconditioner(n_probes=4)
    diag = jnp.array([1.0, 2.0, 3.0])

    @eqx.filter_jit
    def run(ctx):
        return strategy.build(ctx).diagonal

    assert jnp.allclose(run(_context(jnp.diag(diag), step=3)), diag)


def test_stochastic_diagonal_strategy_requires_exact_lagrangian():
    """Without an exact Lagrangian view the strategy cannot probe curvature."""
    with pytest.raises(ValueError, match="exact"):
        StochasticDiagonalPreconditioner().build(_context(secant=_lbfgs(3)))


def test_stochastic_diagonal_strategy_validates_probe_count():
    """At least one probe is required."""
    with pytest.raises(Exception, match="n_probes"):
        StochasticDiagonalPreconditioner(n_probes=0)
