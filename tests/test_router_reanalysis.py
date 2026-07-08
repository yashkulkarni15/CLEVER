"""Tests for the --reanalyze-existing mode of scripts/21_judge_router_threshold.py.

The reanalysis filters an existing judged-verdicts JSONL (phase 7 output) to
hit distances at or below the router-selected threshold and recomputes the
judged-equivalence rate, so the router operating point can be audited without
re-running the eviction pipeline.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "21_judge_router_threshold.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("judge_router_threshold", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rec(ds, orig, new, d, v, policy="lfu"):
    return {
        "dataset": ds,
        "policy": policy,
        "orig_query": orig,
        "new_query": new,
        "distance_l2sq": d,
        "verdict": v,
    }


def test_conditional_split_and_rates(mod):
    records = [
        rec("lmsys", "a", "b", 0.5, True),
        rec("lmsys", "c", "d", 0.6, False),
        rec("lmsys", "e", "f", 0.8, False),
    ]
    out = mod.conditional_yes(records, 0.772)
    stats = out["lmsys"]
    assert stats["at_or_below"]["n"] == 2
    assert stats["at_or_below"]["yes"] == 1
    assert stats["at_or_below"]["mean_yes"] == pytest.approx(0.5)
    assert stats["above"]["n"] == 1
    assert stats["above"]["yes"] == 0
    assert stats["all"]["n"] == 3


def test_pairs_deduplicated_across_policies(mod):
    records = [
        rec("lmsys", "a", "b", 0.5, True, policy="lfu"),
        rec("lmsys", "a", "b", 0.5, True, policy="arc"),
    ]
    out = mod.conditional_yes(records, 0.772)
    assert out["lmsys"]["all"]["n"] == 1


def test_none_verdicts_skipped(mod):
    records = [
        rec("lmsys", "a", "b", 0.5, None),
        rec("lmsys", "c", "d", 0.5, True),
    ]
    out = mod.conditional_yes(records, 0.772)
    assert out["lmsys"]["all"]["n"] == 1
    assert out["lmsys"]["all"]["yes"] == 1


def test_datasets_kept_separate(mod):
    records = [
        rec("lmsys", "a", "b", 0.5, True),
        rec("qqp", "a", "b", 0.5, False),
    ]
    out = mod.conditional_yes(records, 0.772)
    assert out["lmsys"]["all"]["yes"] == 1
    assert out["qqp"]["all"]["yes"] == 0


def test_boundary_inclusive(mod):
    records = [rec("lmsys", "a", "b", 0.772, True)]
    out = mod.conditional_yes(records, 0.772)
    assert out["lmsys"]["at_or_below"]["n"] == 1
    assert out["lmsys"]["above"]["n"] == 0


def test_wilson_ci_matches_phase7_numbers(mod):
    lo, hi = mod.wilson_ci(174, 3912)
    assert 0.038 < lo < 0.039
    assert 0.051 < hi < 0.052
