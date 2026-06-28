import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "sample_hits",
    Path(__file__).resolve().parent.parent / "scripts" / "16_sample_hits_for_judge.py",
)
sample_hits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample_hits)


def _recs(n):
    return [{"dataset": "qqp", "policy": "lfu", "distance_l2sq": i / n,
             "orig_query": f"o{i}", "new_query": f"n{i}"} for i in range(n)]


def test_sample_is_capped_and_deterministic():
    recs = _recs(500)
    a = sample_hits.stratified_sample(recs, n=100, n_bins=5, seed=7)
    b = sample_hits.stratified_sample(recs, n=100, n_bins=5, seed=7)
    assert len(a) == 100
    assert [r["new_query"] for r in a] == [r["new_query"] for r in b]  # deterministic


def test_sample_returns_all_when_fewer_than_n():
    recs = _recs(20)
    out = sample_hits.stratified_sample(recs, n=100, n_bins=5, seed=7)
    assert len(out) == 20


def test_sample_spreads_across_distance_bins():
    recs = _recs(500)
    out = sample_hits.stratified_sample(recs, n=50, n_bins=5, seed=7)
    lo = sum(1 for r in out if r["distance_l2sq"] < 0.2)
    hi = sum(1 for r in out if r["distance_l2sq"] >= 0.8)
    assert lo > 0 and hi > 0  # both tails represented
