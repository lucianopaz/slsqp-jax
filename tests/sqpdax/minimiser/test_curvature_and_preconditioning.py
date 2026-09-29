"""Curvature mode, preconditioner strategies and secant resets on the minimisers."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from slsqp_jax.sqpdax.minimiser import (
    ActiveSetLineSearchMinimiser,
    ProximalActiveSetLineSearchMinimiser,
    TrustRegionInteriorPointMinimiser,
    minimise,
)
from slsqp_jax.sqpdax.preconditioner import (
    DiagonalPreconditioner,
    IdentityPreconditioner,
    NoPreconditioner,
    SecantPreconditioner,
    StochasticDiagonalPreconditioner,
)
from slsqp_jax.sqpdax.secant import SecantResetPolicy
from slsqp_jax.sqpdax.subproblem.solver import (
    MinresQLPSubProblemSolver,
    ProjectedCGSubProblemSolver,
)
from tests.sqpdax.lagrangian.conftest import make_problem

from .conftest import make_scaled_quartic

MINIMISERS = {
    "active-set": ActiveSetLineSearchMinimiser,
    "proximal": ProximalActiveSetLineSearchMinimiser,
}

X_STAR = (0.8684468143545903, 0.11959380513219049, 0.01195938051321905)

SECANT_STATS_KEYS = (
    "secant_n_appends",
    "secant_n_skips",
    "secant_n_damped",
    "secant_n_resets",
    "secant_min_damping_theta",
    "secant_last_condition",
    "secant_max_condition",
)


def _x0():
    # Built lazily so the dtype follows the x64 flag at run time.
    return jnp.array([0.5, 0.3, 0.2])


@pytest.mark.parametrize("minimiser_cls", MINIMISERS.values(), ids=MINIMISERS.keys())
@pytest.mark.parametrize(
    ("curvature", "preconditioner", "with_hvp", "maintains", "model_uses"),
    [
        ("auto", NoPreconditioner(), True, False, False),
        ("auto", NoPreconditioner(), False, True, True),
        ("exact", NoPreconditioner(), True, False, False),
        ("secant", NoPreconditioner(), True, True, True),
        ("exact", SecantPreconditioner(), True, True, False),
        ("auto", SecantPreconditioner(), False, True, True),
        ("secant", StochasticDiagonalPreconditioner(), True, True, True),
    ],
    ids=[
        "auto-exact",
        "auto-secant",
        "exact",
        "secant-with-hvp",
        "exact-lbfgs-precond",
        "secant-lbfgs-precond",
        "secant-diag-precond",
    ],
)
def test_curvature_mode_resolves_secant_roles(
    minimiser_cls, curvature, preconditioner, with_hvp, maintains, model_uses
):
    """The secant is kept for the model or preconditioner, and only modelled on demand."""
    problem = make_scaled_quartic(with_curvature=with_hvp)
    solver = minimiser_cls(curvature=curvature, preconditioner=preconditioner).init(
        problem, _x0()
    )
    assert (solver.secant is not None) is maintains
    assert (solver.secant_stats is not None) is maintains
    assert (solver._model_secant(problem) is not None) is model_uses
    ctx = solver._init_subproblem(problem)
    assert (ctx.lagrangian.secant is not None) is model_uses


@pytest.mark.parametrize("minimiser_cls", MINIMISERS.values(), ids=MINIMISERS.keys())
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"curvature": "exact"}, "curvature='exact'"),
        (
            {"preconditioner": StochasticDiagonalPreconditioner()},
            "stochastic_diagonal",
        ),
    ],
    ids=["exact", "stochastic-diagonal"],
)
def test_exact_curvature_requests_need_hvps(minimiser_cls, kwargs, match):
    """Exact-HVP configurations fail loudly on problems without HVPs."""
    problem = make_scaled_quartic(with_curvature=False)
    with pytest.raises(ValueError, match=match):
        minimiser_cls(**kwargs).init(problem, _x0())


def test_unknown_curvature_mode_is_rejected():
    """Only ``auto`` / ``exact`` / ``secant`` are valid modes."""
    with pytest.raises(ValueError, match="curvature"):
        ActiveSetLineSearchMinimiser(curvature="bfgs")  # type: ignore[arg-type]


@pytest.mark.parametrize("minimiser_cls", MINIMISERS.values(), ids=MINIMISERS.keys())
@pytest.mark.parametrize(
    ("minimiser_opts", "with_hvp"),
    [
        ({"curvature": "exact", "preconditioner": {"kind": "lbfgs"}}, True),
        ({"curvature": "secant", "preconditioner": {"kind": "lbfgs"}}, True),
        (
            {
                "curvature": "exact",
                "preconditioner": {"kind": "stochastic_diagonal", "n_probes": 4},
            },
            True,
        ),
        ({"preconditioner": {"kind": "lbfgs"}}, False),
    ],
    ids=["exact-lbfgs", "secant-lbfgs", "exact-diag", "auto-secant-lbfgs"],
)
def test_preconditioned_configurations_converge(
    minimiser_cls, minimiser_opts, with_hvp
):
    """Every curvature / preconditioner pairing reaches the analytic minimiser."""
    options = {"minimiser": minimiser_opts}
    # In float32 the proximal secant runs stall on merit stagnation within
    # ~1e-6 of the minimiser, so the convergence flag needs x64.
    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("error")
        problem = make_scaled_quartic(with_curvature=with_hvp)
        sol = minimise(
            problem,
            minimiser_cls(rtol=1e-6, atol=1e-6, min_steps=1),
            _x0(),
            max_steps=100,
            throw=False,
            options=options,
        )
        assert bool(sol.state.result_adapter.is_successful(sol.result))
        assert jnp.allclose(sol.value, jnp.asarray(X_STAR), atol=1e-4)
    maintains = minimiser_opts.get("curvature") != "exact" or (
        minimiser_opts["preconditioner"]["kind"] == "lbfgs"
    )
    assert all((key in sol.stats) is maintains for key in SECANT_STATS_KEYS)


def test_secant_stats_track_appends_in_solution():
    """The recorded counters account for every accepted secant update."""
    problem = make_scaled_quartic(with_curvature=False)
    sol = minimise(
        problem,
        ActiveSetLineSearchMinimiser(rtol=1e-6, atol=1e-6, min_steps=1),
        _x0(),
        max_steps=100,
        throw=False,
    )
    stats = sol.stats
    n_updates = int(stats["secant_n_appends"]) + int(stats["secant_n_skips"])
    assert 0 < int(stats["secant_n_appends"]) <= n_updates <= int(stats["num_steps"])
    assert int(stats["secant_n_damped"]) <= int(stats["secant_n_appends"])
    assert stats["secant_n_resets"].shape == (3,)
    assert 0.0 < float(stats["secant_min_damping_theta"]) <= 1.0
    assert float(stats["secant_max_condition"]) >= float(stats["secant_last_condition"])


@pytest.mark.parametrize(
    ("make_inner", "expected_inverse"),
    [
        (ProjectedCGSubProblemSolver, lambda m, v: m.secant.inverse_hvp(v)),
        (MinresQLPSubProblemSolver, lambda m, v: m.secant.inverse_hvp(v)),
        (
            lambda: ProjectedCGSubProblemSolver(
                preconditioner=DiagonalPreconditioner(jnp.full(3, 2.0))
            ),
            lambda m, v: v / 2.0,
        ),
    ],
    ids=["pcg", "user-minres", "user-preconditioner-kept"],
)
def test_proximal_default_preconditioner_reaches_inner_solver(
    make_inner, expected_inverse
):
    """The proximal L-BFGS default reaches user inner solvers but never overrides."""
    problem = make_scaled_quartic(with_curvature=False)
    inner = make_inner()
    options = {"subproblem": {"subproblem_solver": inner}}
    solver = ProximalActiveSetLineSearchMinimiser().init(
        problem, _x0(), options=options
    )
    solver = _with_pairs(solver)
    leaf = solver._init_subproblem(problem).solver.subproblem_solver
    assert type(leaf) is type(inner)
    v = jnp.array([1.0, -2.0, 0.5])
    assert jnp.allclose(
        leaf.preconditioner.invert(v), expected_inverse(solver, v), atol=1e-5
    )


def test_proximal_default_preconditioner_is_off_with_exact_hvp():
    """With exact HVPs no secant is kept, so the optional default builds nothing."""
    problem = make_scaled_quartic(with_curvature=True)
    solver = ProximalActiveSetLineSearchMinimiser().init(problem, _x0())
    assert solver.secant is None
    leaf = solver._init_subproblem(problem).solver.subproblem_solver
    assert leaf.preconditioner is None


def _with_pairs(solver):
    secant = solver.secant
    for k in range(3):
        s = jnp.zeros(3).at[k].set(1.0) + 0.1
        secant = secant.append(s, (k + 2.0) * s)
    return eqx.tree_at(lambda m: m.secant, solver, secant)


@pytest.mark.parametrize(
    ("qp_streak", "ls_streak", "severity"),
    [
        (0, 0, -1),
        (1, 0, 0),
        (0, 1, 0),
        (2, 0, -1),
        (3, 0, 2),
        (0, 3, 2),
        (3, 1, 2),
    ],
    ids=["none", "qp-first", "ls-first", "qp-mid", "qp-patience", "ls-patience", "mix"],
)
def test_reset_secant_follows_failure_streaks(qp_streak, ls_streak, severity):
    """QP / line-search streaks escalate the reset; counts land in the stats."""
    problem = make_scaled_quartic(with_curvature=False)
    solver = ActiveSetLineSearchMinimiser(
        qp_failure_patience=3, ls_failure_patience=3
    ).init(problem, _x0())
    solver = _with_pairs(solver)
    solver = eqx.tree_at(
        lambda m: (m.consecutive_qp_failures, m.consecutive_ls_failures),
        solver,
        (jnp.asarray(qp_streak), jnp.asarray(ls_streak)),
    )
    before = solver.secant
    after = solver._reset_secant()

    expected_secant = before if severity < 0 else before.reset(jnp.asarray(severity))
    assert eqx.tree_equal(after.secant, expected_secant)
    expected_resets = jnp.zeros(3, jnp.int32)
    if severity >= 0:
        expected_resets = expected_resets.at[severity].set(1)
    assert jnp.array_equal(after.secant_stats.n_resets, expected_resets)


def test_reset_secant_respects_disabled_policy():
    """A disabled policy leaves the secant untouched even at full patience."""
    problem = make_scaled_quartic(with_curvature=False)
    solver = ActiveSetLineSearchMinimiser(
        qp_failure_patience=1, secant_reset=SecantResetPolicy(enabled=False)
    ).init(problem, _x0())
    solver = eqx.tree_at(
        lambda m: m.consecutive_qp_failures, _with_pairs(solver), jnp.asarray(5)
    )
    after = solver._reset_secant()
    assert eqx.tree_equal(after.secant, solver.secant)
    assert int(jnp.sum(after.secant_stats.n_resets)) == 0


def test_trust_region_warns_when_preconditioner_is_ignored():
    """Trust-region subproblem solvers take no preconditioner, so it is reported."""
    problem = make_problem()
    solver = TrustRegionInteriorPointMinimiser(
        preconditioner=SecantPreconditioner()
    ).init(problem, jnp.array([0.5, 0.5]))
    with pytest.warns(UserWarning, match="ignored"):
        solver._init_subproblem(problem)


@pytest.mark.parametrize(
    ("minimiser_cls", "spec", "expected"),
    [
        (
            ActiveSetLineSearchMinimiser,
            {"kind": "stochastic_diagonal", "n_probes": 3},
            StochasticDiagonalPreconditioner(n_probes=3),
        ),
        (
            ProximalActiveSetLineSearchMinimiser,
            {"require_secant": True},
            SecantPreconditioner(require_secant=True),
        ),
        (
            ActiveSetLineSearchMinimiser,
            SecantPreconditioner(),
            SecantPreconditioner(),
        ),
    ],
    ids=["kind-spec", "field-update", "instance"],
)
def test_preconditioner_option_parsing(minimiser_cls, spec, expected):
    """Kind specs, field updates and instances are all accepted as options."""
    options = {
        "minimiser": {
            "preconditioner": spec,
            "curvature": "exact",
            "secant_reset": {"condition_threshold": 1e3},
        }
    }
    parsed = minimiser_cls()._parse_options(options)
    assert parsed.preconditioner == expected
    assert parsed.curvature == "exact"
    assert parsed.secant_reset == SecantResetPolicy(condition_threshold=1e3)


def test_preconditioner_option_rejects_bare_kind_string():
    """A kind must be given as ``{"kind": ...}``, not as a bare string."""
    with pytest.raises(TypeError, match="preconditioner"):
        ActiveSetLineSearchMinimiser()._parse_options(
            {"minimiser": {"preconditioner": "lbfgs"}}
        )


def test_identity_preconditioner_instance_is_not_a_strategy():
    """Raw preconditioners are solver-level options, not minimiser strategies."""
    with pytest.raises(TypeError, match="preconditioner"):
        ActiveSetLineSearchMinimiser()._parse_options(
            {"minimiser": {"preconditioner": IdentityPreconditioner(jnp.zeros(3))}}
        )
