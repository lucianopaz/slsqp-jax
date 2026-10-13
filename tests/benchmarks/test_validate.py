"""Tests for the catalogue validation logic (``benchmarks.validate``) and the
sign-flip helpers of ``benchmarks.problems`` - pure functions, no sif2jax."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.catalog import (
    COLUMNS,
    VALIDATION_COLUMNS,
    select,
    with_validation_columns,
)
from benchmarks.problems import flip_signs, format_flip
from benchmarks.validate import apply_validation, candidate_flips, decide

F_HUGE = -7e30


def _run(label, **kw):
    return {"label": label, **kw}


@pytest.mark.parametrize(
    ("spec", "mineq", "expected"),
    [
        ("", 3, [-1.0, -1.0, -1.0]),
        (None, 2, [-1.0, -1.0]),
        (float("nan"), 2, [-1.0, -1.0]),
        ("all", 2, [1.0, 1.0]),
        ("0,2", 3, [1.0, -1.0, 1.0]),
        ("1", 2, [-1.0, 1.0]),
    ],
)
def test_flip_signs(spec, mineq, expected):
    assert flip_signs(spec, mineq).tolist() == expected


@pytest.mark.parametrize("mask", [[False] * 4, [True] * 4, [True, False, False, True]])
def test_format_flip_roundtrips(mask):
    arr = np.asarray(mask)
    spec = format_flip(arr, arr.size)
    assert (flip_signs(spec, arr.size) > 0).tolist() == mask


def test_flip_signs_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        flip_signs("3", 3)


@pytest.mark.parametrize(
    ("c_star", "expected"),
    [
        (None, [("documented", ""), ("all", "all")]),
        (np.array([0.5, 0.1, 2.0]), [("documented", ""), ("all", "all")]),
        (
            np.array([0.5, -2.0, 0.0]),
            [("documented", ""), ("xstar_rows", "1"), ("all", "all")],
        ),
        # every row violated -> the row candidate coincides with "all"
        (np.array([-1.0, -1.0, -1.0]), [("documented", ""), ("all", "all")]),
        (np.array([np.nan, 1.0, 1.0]), [("documented", ""), ("all", "all")]),
    ],
)
def test_candidate_flips(c_star, expected):
    assert candidate_flips(3, c_star) == expected
    assert candidate_flips(0, c_star) == [("documented", "")]


@pytest.mark.parametrize(
    ("runs", "fstar_sif", "expected"),
    [
        # documented sign, scipy reaches f*
        (
            {"": _run("documented", slsqp_f=1.0, slsqp_feas=0.0, slsqp_success=True)},
            1.0,
            dict(
                ineq_flip="",
                validation="ok",
                fstar=1.0,
                fstar_source="sif2jax",
                valid=True,
            ),
        ),
        # documented diverges, all-flipped reaches f* -> sign flipped
        (
            {
                "": _run(
                    "documented", slsqp_f=F_HUGE, slsqp_feas=0.0, slsqp_success=False
                ),
                "all": _run("all", slsqp_f=2.0, slsqp_feas=1e-9, slsqp_success=True),
            },
            2.0,
            dict(
                ineq_flip="all",
                validation="sign_flipped",
                fstar=2.0,
                fstar_source="sif2jax",
            ),
        ),
        # x* feasible only under the row flip, scipy fails everywhere (HS108)
        (
            {
                "": _run("documented", xstar_feas=1.0, xstar_f=-0.866),
                "2,4,7": _run(
                    "xstar_rows",
                    xstar_feas=0.0,
                    xstar_f=-0.866,
                    slsqp_f=-4e15,
                    slsqp_feas=3e11,
                    slsqp_success=False,
                ),
                "all": _run("all", xstar_feas=0.866, xstar_f=-0.866),
            },
            -0.866,
            dict(
                ineq_flip="2,4,7", validation="sign_flipped", fstar=-0.866, valid=True
            ),
        ),
        # x* feasible under both, but scipy undercuts f* under the documented
        # sign -> the flipped conversion where f* is a minimum wins (HS64)
        (
            {
                "": _run(
                    "documented",
                    xstar_feas=0.0,
                    xstar_f=6299.84,
                    slsqp_f=5800.0,
                    slsqp_feas=0.0,
                    slsqp_success=True,
                ),
                "all": _run(
                    "all",
                    xstar_feas=0.0,
                    xstar_f=6299.84,
                    slsqp_f=6299.84,
                    slsqp_feas=0.0,
                    slsqp_success=True,
                ),
            },
            6299.84,
            dict(ineq_flip="all", validation="sign_flipped", fstar=6299.84),
        ),
        # both scipy solvers agree on a value that differs from f* (HS5)
        (
            {
                "": _run(
                    "documented",
                    xstar_feas=0.0,
                    xstar_f=-1.57,
                    slsqp_f=-1.9132,
                    slsqp_feas=0.0,
                    slsqp_success=True,
                    tc_f=-1.9132,
                    tc_feas=0.0,
                    tc_success=True,
                )
            },
            -5.03,
            dict(
                ineq_flip="",
                validation="fstar_mismatch",
                fstar=-1.9132,
                fstar_source="scipy-consensus",
            ),
        ),
        # no sif2jax f*, consensus adopted (sign flipped) (HAIFAM)
        (
            {
                "": _run(
                    "documented", slsqp_f=F_HUGE, slsqp_feas=0.0, slsqp_success=False
                ),
                "all": _run(
                    "all",
                    slsqp_f=-45.0003,
                    slsqp_feas=0.0,
                    slsqp_success=False,
                    tc_f=-45.0003,
                    tc_feas=0.0,
                    tc_success=False,
                ),
            },
            None,
            dict(
                ineq_flip="all",
                validation="fstar_from_consensus",
                fstar=-45.0003,
                fstar_source="scipy-consensus",
            ),
        ),
        # SLSQP alone, f* known but not reached -> unverified, f* kept
        (
            {"": _run("documented", slsqp_f=3.0, slsqp_feas=0.0, slsqp_success=True)},
            1.0,
            dict(
                ineq_flip="", validation="unverified", fstar=1.0, fstar_source="sif2jax"
            ),
        ),
        # bounded feasible end point without convergence, documented diverges
        (
            {
                "": _run(
                    "documented", slsqp_f=F_HUGE, slsqp_feas=0.0, slsqp_success=False
                ),
                "all": _run("all", slsqp_f=-45.0, slsqp_feas=0.0, slsqp_success=False),
            },
            None,
            dict(ineq_flip="all", validation="unverified", valid=True),
        ),
        # diverges under every conversion (DIAGIQB)
        (
            {
                "": _run(
                    "documented", slsqp_f=-1e13, slsqp_feas=0.0, slsqp_success=False
                )
            },
            None,
            dict(ineq_flip="", validation="unbounded", valid=False),
        ),
        # nothing feasible under any conversion
        (
            {
                "": _run(
                    "documented", slsqp_f=1.0, slsqp_feas=0.5, slsqp_success=False
                ),
                "all": _run("all", slsqp_f=1.0, slsqp_feas=0.2, slsqp_success=False),
            },
            1.0,
            dict(ineq_flip="", validation="infeasible", valid=False, fstar=1.0),
        ),
        # solver errors only
        (
            {"": _run("documented", error="boom")},
            1.0,
            dict(
                ineq_flip="",
                validation="unverified",
                valid=True,
                validation_note="boom",
            ),
        ),
    ],
)
def test_decide(runs, fstar_sif, expected):
    out = decide(runs, fstar_sif=fstar_sif)
    for key, value in expected.items():
        if isinstance(value, float):
            assert out[key] == pytest.approx(value), key
        else:
            assert out[key] == value, key
    assert set(out) >= {
        "ineq_flip",
        "fstar",
        "fstar_source",
        "valid",
        "validation",
        "validation_note",
    }


@pytest.fixture
def base_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "collection": ["constrained"] * 3,
            "n": [2, 3, 4],
            "meq": [0, 0, 0],
            "mineq": [1, 2, 0],
            "nbounds": [0, 0, 0],
            "y0_iD": [0, 0, 0],
            "n_y0s": [1, 1, 1],
            "has_fstar": [True, False, True],
            "fstar": [1.0, float("nan"), 3.0],
            "has_xstar": [False, False, False],
            "tier": ["tiny"] * 3,
        }
    )


def test_with_validation_columns_defaults_and_carry_over(base_catalog):
    out = with_validation_columns(base_catalog)
    assert list(out.columns) == [*COLUMNS, *VALIDATION_COLUMNS]
    assert (out["validation"] == "unvalidated").all()
    assert out["valid"].all()
    assert out["fstar_sif"].equals(out["fstar"])

    previous = out.copy()
    previous.loc[0, ["ineq_flip", "fstar", "fstar_source", "validation"]] = [
        "all",
        1.5,
        "scipy-consensus",
        "fstar_mismatch",
    ]
    previous.loc[2, ["valid", "validation"]] = [False, "unbounded"]
    rebuilt = with_validation_columns(base_catalog, previous)
    assert rebuilt.loc[0, "ineq_flip"] == "all"
    assert rebuilt.loc[0, "fstar"] == 1.5
    assert rebuilt.loc[0, "validation"] == "fstar_mismatch"
    assert rebuilt.loc[1, "validation"] == "unvalidated"
    assert not rebuilt.loc[2, "valid"]
    # an unvalidated previous catalogue carries nothing over
    assert with_validation_columns(base_catalog, base_catalog).equals(out)


def test_apply_validation_and_select(base_catalog):
    results = [
        {
            "name": "A",
            "y0_iD": 0,
            "ineq_flip": "0",
            "fstar": 1.0,
            "fstar_sif": 1.0,
            "fstar_source": "sif2jax",
            "valid": True,
            "validation": "sign_flipped",
            "validation_note": "x",
            "ref_slsqp_f": 1.0,
            "ref_slsqp_feas": 0.0,
        },
        {
            "name": "B",
            "y0_iD": 0,
            "ineq_flip": "",
            "fstar": -2.0,
            "fstar_sif": float("nan"),
            "fstar_source": "scipy-consensus",
            "valid": True,
            "validation": "fstar_from_consensus",
            "validation_note": "",
        },
        # harness row of a timed-out instance
        {"problem": "C", "y0_iD": 0, "status": "timeout", "error": "killed"},
        # unknown instance is ignored
        {"name": "Z", "y0_iD": 0, "validation": "ok", "fstar": 0.0},
    ]
    out = apply_validation(base_catalog, results)
    a, b, c = (out[out["name"] == k].iloc[0] for k in "ABC")
    assert a["ineq_flip"] == "0" and a["validation"] == "sign_flipped"
    assert (
        b["fstar"] == -2.0
        and bool(b["has_fstar"])
        and b["fstar_source"] == "scipy-consensus"
    )
    assert c["validation"] == "timeout" and c["fstar"] == 3.0 and bool(c["valid"])
    assert len(out) == 3

    out.loc[out["name"] == "C", "valid"] = False
    assert set(select(out, tier=None)["name"]) == {"A", "B"}
    assert set(select(out, tier=None, include_invalid=True)["name"]) == {"A", "B", "C"}
    # explicit names bypass the validity filter
    assert set(select(out, tier=None, names=["C"])["name"]) == {"C"}
