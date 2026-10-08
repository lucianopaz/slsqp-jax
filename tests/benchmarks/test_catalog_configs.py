"""Tests for ``benchmarks.catalog``, ``benchmarks.configs`` and the repeat rule."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarks import __main__ as cli
from benchmarks.catalog import COLUMNS, TIERS, load_catalog, select, tier_of
from benchmarks.configs import CONFIGS, get_configs
from benchmarks.worker import TaskSpec, adaptive_repeats


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    return load_catalog()


def test_catalog_schema(catalog):
    assert list(catalog.columns) == list(COLUMNS)
    assert not catalog.duplicated(["name", "y0_iD"]).any()
    assert (catalog["n"] > 0).all()
    assert catalog["tier"].map(lambda t: t in {*TIERS, "large"}).all()
    assert (catalog["tier"] == catalog["n"].map(tier_of)).all()


@pytest.mark.parametrize(
    ("n", "tier"),
    [
        (1, "tiny"),
        (20, "tiny"),
        (21, "small"),
        (100, "small"),
        (1000, "medium"),
        (1001, "large"),
    ],
)
def test_tier_of(n, tier):
    assert tier_of(n) == tier


@pytest.mark.parametrize("tier", ["tiny", "small", "medium", "large"])
def test_select_tier_is_cumulative(catalog, tier):
    sel = select(catalog, tier=tier)
    order = [*TIERS, "large"]
    allowed = set(order[: order.index(tier) + 1])
    assert set(sel["tier"]) <= allowed
    assert set(sel["collection"]) <= {"constrained", "cqp", "bounded", "bqp"}
    assert sel["n"].is_monotonic_increasing


def test_select_filters(catalog):
    by_name = select(catalog, tier=None, collections=None, names=["hs71", "HS35"])
    assert set(by_name["name"]) == {"HS71", "HS35"}
    assert (
        select(catalog, tier=None, collections=["nle"])["collection"] == "nle"
    ).all()
    assert (select(catalog, tier=None, max_n=5)["n"] <= 5).all()
    with pytest.raises(ValueError, match="unknown tier"):
        select(catalog, tier="huge")


def test_select_chunks_partition(catalog):
    full = select(catalog, tier="small")
    chunks = [select(catalog, tier="small", chunk=(i, 3)) for i in range(3)]
    assert sum(len(c) for c in chunks) == len(full)
    merged = (
        pd.concat(chunks).sort_values(["n", "name", "y0_iD"]).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(merged, full.reset_index(drop=True))
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1


def test_configs_registry():
    assert len(CONFIGS) == 7
    assert list(CONFIGS) == [
        "asls-pcg",
        "asls-craig",
        "asls-minresqlp",
        "asls-pcg-exact",
        "pasls",
        "trip",
        "tfip",
    ]
    assert [c.name for c in CONFIGS.values() if c.curvature == "exact"] == [
        "asls-pcg-exact"
    ]
    assert get_configs(None) == list(CONFIGS.values())
    assert [c.name for c in get_configs(["tfip", "pasls"])] == ["tfip", "pasls"]
    with pytest.raises(KeyError):
        get_configs(["nope"])


@pytest.mark.parametrize("config", list(CONFIGS.values()), ids=list(CONFIGS))
def test_config_factories_build(config):
    minimiser = config.make_minimiser()
    options = config.make_options()
    assert minimiser is not None
    assert isinstance(options, dict)
    tags = config.tags()
    assert tags["config"] == config.name and tags["curvature"] == config.curvature


@pytest.mark.parametrize(
    ("pilot_s", "expected"),
    [
        (1e-7, 1000),
        (1e-3, 1000),
        (0.01, 200),
        (0.1, 20),
        (1.0, 3),
        (30.0, 3),
        (90.0, 1),
    ],
)
def test_adaptive_repeats(pilot_s, expected):
    assert (
        adaptive_repeats(
            pilot_s, budget_s=2.0, max_repeats=1000, slow_floor=3, slow_threshold_s=60.0
        )
        == expected
    )


def test_adaptive_repeats_respects_caps():
    assert (
        adaptive_repeats(
            1e-9, budget_s=2.0, max_repeats=50, slow_floor=3, slow_threshold_s=60.0
        )
        == 50
    )
    assert (
        adaptive_repeats(
            10.0, budget_s=2.0, max_repeats=50, slow_floor=5, slow_threshold_s=60.0
        )
        == 5
    )


def test_cli_parser_defaults():
    args = cli.build_parser().parse_args(
        ["run", "--problems", "HS71", "--chunk", "1/3"]
    )
    assert args.tier == "tiny" and args.timeout == 300.0 and args.repeats is None
    assert cli._parse_chunk(args.chunk) == (1, 3)
    assert cli._csv_list(" a, b ,") == ["a", "b"]
    assert cli._csv_list(None) is None
    spec = TaskSpec(problem="HS71", config="pasls", timeout_s=args.timeout)
    assert spec.max_steps == 500
