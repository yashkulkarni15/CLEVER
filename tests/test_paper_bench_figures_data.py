"""Unit tests for the data-shaping logic in scripts/19_paper_index_routing_figures.py.

Covers:
  - benchmark runs -> per-index-type pareto points (workload filter)
  - pareto frontier extraction (monotone recall/latency staircase)
  - multi-seed threshold sweep -> mean curves

Plot rendering is intentionally untested; it is verified by running the
script and checking that nonzero PNG/PDF files are produced.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "19_paper_index_routing_figures.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "paper_index_routing_figures", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(index_type="hnsw", recall=0.9, p50=1.0, workload="uniform", params=None):
    return {
        "index_type": index_type,
        "params": params or {},
        "workload": workload,
        "recall_at_1": recall,
        "search_latency_ms": {"p50": p50},
    }


def test_pareto_points_filters_workload_and_groups_by_type(mod):
    runs = [
        make_run("hnsw", 0.95, 0.5),
        make_run("flat", 1.0, 17.0),
        make_run("hnsw", 0.99, 9.9, workload="bursty"),
    ]
    pts = mod.pareto_points(runs, workload="uniform")
    assert set(pts.keys()) == {"hnsw", "flat"}
    assert pts["hnsw"] == [(0.5, 0.95)]
    assert pts["flat"] == [(17.0, 1.0)]


def test_pareto_frontier_is_a_staircase(mod):
    points = [(0.1, 0.90), (0.5, 0.99), (0.3, 0.80), (17.0, 1.0), (2.0, 0.95)]
    frontier = mod.pareto_frontier(points)
    # Dominated points (0.3, 0.80) and (2.0, 0.95) are excluded.
    assert frontier == [(0.1, 0.90), (0.5, 0.99), (17.0, 1.0)]
    lats = [p[0] for p in frontier]
    recs = [p[1] for p in frontier]
    assert lats == sorted(lats)
    assert recs == sorted(recs)


def test_mean_sweep_averages_across_seeds(mod):
    per_seed = {
        "1": {"random": {"threshold_sweep": [
            {"threshold": 0.5, "hit_rate": 0.4, "retrieval_similarity": 0.9,
             "latency_saving_pct": 40.0},
            {"threshold": 1.0, "hit_rate": 0.8, "retrieval_similarity": 0.7,
             "latency_saving_pct": 80.0},
        ]}},
        "2": {"random": {"threshold_sweep": [
            {"threshold": 0.5, "hit_rate": 0.6, "retrieval_similarity": 0.8,
             "latency_saving_pct": 60.0},
            {"threshold": 1.0, "hit_rate": 1.0, "retrieval_similarity": 0.6,
             "latency_saving_pct": 100.0},
        ]}},
    }
    curve = mod.mean_sweep(per_seed, fill="random")
    assert curve["threshold"] == [0.5, 1.0]
    assert curve["hit_rate"] == pytest.approx([0.5, 0.9])
    assert curve["quality"] == pytest.approx([0.85, 0.65])
    assert curve["latency_saving_pct"] == pytest.approx([50.0, 90.0])


def test_mean_sweep_rejects_mismatched_thresholds(mod):
    per_seed = {
        "1": {"random": {"threshold_sweep": [
            {"threshold": 0.5, "hit_rate": 0.4, "retrieval_similarity": 0.9,
             "latency_saving_pct": 40.0}]}},
        "2": {"random": {"threshold_sweep": [
            {"threshold": 0.6, "hit_rate": 0.6, "retrieval_similarity": 0.8,
             "latency_saving_pct": 60.0}]}},
    }
    with pytest.raises(ValueError):
        mod.mean_sweep(per_seed, fill="random")
