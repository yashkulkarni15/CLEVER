"""Unit tests for the pure data-shaping logic in scripts/14_visualize_phase_figures.py.

Covers:
  - adaptive summary CSV -> tidy plotting dataframe (Figure A)
  - density input discovery / missing-input detection (Figure B)
  - density gap CSV -> per-dataset characterization frame (Figure B)

Plot rendering is intentionally untested here; it is verified by running the
script and checking that nonzero PNG/PDF files are produced.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "14_visualize_phase_figures.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "visualize_phase_figures", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────
# Figure A: adaptive summary -> tidy frame
# ─────────────────────────────────────────────────────────────────

def _summary_frame(dataset: str, rows: dict) -> pd.DataFrame:
    """Build a frame matching the real adaptive_subset_summary.csv schema."""
    records = []
    for policy, (hit_rate, time_ms) in rows.items():
        records.append({
            "dataset": dataset,
            "embedding_model": "all-MiniLM-L6-v2",
            "size": "100k",
            "policy": policy,
            "cache_size_pct": 0.1,
            "max_cache_size": 3000,
            "workload": "temporal",
            "seed": 42,
            "n_queries": 30000,
            "n_stream": 21000,
            "final_hit_rate": hit_rate,
            "semantic_coverage_avg_dist": 0.7,
            "avg_query_time_ms": time_ms,
            "stream_time_s": 12.0,
            "n_evictions": 12000,
            "policy_stats_json": "{}",
        })
    return pd.DataFrame(records)


FULL_POLICY_ROWS = {
    "lru": (0.417, 0.60),
    "lfu": (0.436, 0.72),
    "semantic": (0.422, 2.53),
    "adaptive_hard": (0.434, 2.95),
    "adaptive_blend": (0.422, 3.82),
}


def test_tidy_adaptive_summary_orders_datasets_and_policies(mod):
    # Deliberately concatenate out of canonical order.
    raw = pd.concat([
        _summary_frame("moss", FULL_POLICY_ROWS),
        _summary_frame("lmsys", FULL_POLICY_ROWS),
        _summary_frame("qqp", FULL_POLICY_ROWS),
    ], ignore_index=True)

    tidy = mod.tidy_adaptive_summary(raw)

    assert list(tidy.columns) == [
        "dataset", "policy", "final_hit_rate", "avg_query_time_ms",
    ]
    # Dataset-major canonical order, policies in fixed order within each.
    assert tidy["dataset"].tolist() == (
        ["lmsys"] * 5 + ["qqp"] * 5 + ["moss"] * 5
    )
    assert tidy["policy"].tolist()[:5] == [
        "lru", "lfu", "semantic", "adaptive_hard", "adaptive_blend",
    ]


def test_tidy_adaptive_summary_preserves_values(mod):
    raw = _summary_frame("qqp", FULL_POLICY_ROWS)
    tidy = mod.tidy_adaptive_summary(raw)

    lfu = tidy[(tidy["dataset"] == "qqp") & (tidy["policy"] == "lfu")]
    assert len(lfu) == 1
    assert lfu["final_hit_rate"].iloc[0] == pytest.approx(0.436)
    assert lfu["avg_query_time_ms"].iloc[0] == pytest.approx(0.72)


def test_tidy_adaptive_summary_rejects_missing_policy(mod):
    rows = {k: v for k, v in FULL_POLICY_ROWS.items() if k != "adaptive_blend"}
    raw = _summary_frame("lmsys", rows)

    with pytest.raises(ValueError, match="adaptive_blend"):
        mod.tidy_adaptive_summary(raw)


def test_tidy_adaptive_summary_rejects_missing_column(mod):
    raw = _summary_frame("lmsys", FULL_POLICY_ROWS).drop(
        columns=["final_hit_rate"]
    )
    with pytest.raises(ValueError, match="final_hit_rate"):
        mod.tidy_adaptive_summary(raw)


def test_find_adaptive_summary_files(mod, tmp_path):
    for ds in ("lmsys", "qqp"):
        d = tmp_path / f"phase3_subset_{ds}_minilm_30000"
        d.mkdir()
        (d / "adaptive_subset_summary.csv").write_text("dataset\n")
    # A run directory without a summary must be ignored.
    (tmp_path / "phase3_subset_moss_minilm_30000").mkdir()

    found = mod.find_adaptive_summary_files(tmp_path)

    assert len(found) == 2
    assert all(p.name == "adaptive_subset_summary.csv" for p in found)


# ─────────────────────────────────────────────────────────────────
# Figure B: density input discovery (missing-input detection)
# ─────────────────────────────────────────────────────────────────

def test_find_density_inputs_all_missing_when_dir_absent(mod, tmp_path):
    found, missing = mod.find_density_inputs(tmp_path / "results" / "density")

    assert found == []
    # 3 datasets x 2 thetas x 2 files = 12 expected inputs.
    assert len(missing) == 12
    names = {p.name for p in missing}
    assert names == {"density_log.csv", "density_gap.csv"}
    dirs = {p.parent.name for p in missing}
    assert "phase2_lmsys_minilm_100k_theta_0p30" in dirs
    assert "phase2_moss_minilm_100k_theta_0p90" in dirs


def test_find_density_inputs_detects_present_files(mod, tmp_path):
    density_dir = tmp_path / "density"
    for ds in ("lmsys", "qqp", "moss"):
        for tag in ("0p30", "0p90"):
            d = density_dir / f"phase2_{ds}_minilm_100k_theta_{tag}"
            d.mkdir(parents=True)
            (d / "density_log.csv").write_text("policy\n")
            (d / "density_gap.csv").write_text("dataset\n")

    found, missing = mod.find_density_inputs(density_dir)

    assert missing == []
    assert len(found) == 12


def test_find_density_inputs_partial(mod, tmp_path):
    density_dir = tmp_path / "density"
    d = density_dir / "phase2_qqp_minilm_100k_theta_0p30"
    d.mkdir(parents=True)
    (d / "density_log.csv").write_text("policy\n")
    (d / "density_gap.csv").write_text("dataset\n")

    found, missing = mod.find_density_inputs(density_dir)

    assert len(found) == 2
    assert len(missing) == 10


# ─────────────────────────────────────────────────────────────────
# Figure B: density gap CSV -> characterization frame
# ─────────────────────────────────────────────────────────────────

def _gap_frame(dataset: str, theta: float, checkpoints: list) -> pd.DataFrame:
    """Build a frame matching the real density_gap.csv schema
    (see build_gap_rows in scripts/12_run_density_profile.py)."""
    records = []
    for query_idx, density, hr_lru, hr_lfu in checkpoints:
        records.append({
            "dataset": dataset,
            "embedding_model": "all-MiniLM-L6-v2",
            "size": "100k",
            "workload": "temporal",
            "seed": 42,
            "cache_size_pct": 0.1,
            "max_cache_size": 3000,
            "query_idx": query_idx,
            "timestamp": query_idx,
            "density_theta": theta,
            "mean_density_reference": density,
            "mean_density_lru": density,
            "mean_density_lfu": density * 1.1,
            "hit_rate_lru": hr_lru,
            "hit_rate_lfu": hr_lfu,
            "hit_rate_gap_lfu_minus_lru": hr_lfu - hr_lru,
        })
    return pd.DataFrame(records)


def test_tidy_density_gap_takes_final_checkpoint(mod):
    raw = pd.concat([
        _gap_frame("qqp", 0.30, [
            (1000, 0.002, 0.30, 0.31),
            (21000, 0.004, 0.398, 0.423),   # final: gap = +2.49pp
        ]),
        _gap_frame("qqp", 0.90, [
            (1000, 0.10, 0.30, 0.31),
            (21000, 0.20, 0.398, 0.423),
        ]),
        _gap_frame("lmsys", 0.30, [
            (21000, 0.003, 0.420, 0.4347),  # gap = +1.47pp
        ]),
    ], ignore_index=True)

    tidy = mod.tidy_density_gap(raw)

    assert list(tidy.columns) == [
        "dataset", "density_theta", "mean_density", "gap_pp",
    ]
    qqp_30 = tidy[
        (tidy["dataset"] == "qqp") & (tidy["density_theta"] == 0.30)
    ]
    assert len(qqp_30) == 1
    assert qqp_30["mean_density"].iloc[0] == pytest.approx(0.004)
    assert qqp_30["gap_pp"].iloc[0] == pytest.approx(2.49, abs=0.01)

    lmsys_30 = tidy[
        (tidy["dataset"] == "lmsys") & (tidy["density_theta"] == 0.30)
    ]
    assert lmsys_30["gap_pp"].iloc[0] == pytest.approx(1.47, abs=0.01)


def test_tidy_density_gap_rejects_missing_column(mod):
    raw = _gap_frame("qqp", 0.30, [(21000, 0.004, 0.398, 0.423)]).drop(
        columns=["hit_rate_gap_lfu_minus_lru"]
    )
    with pytest.raises(ValueError, match="hit_rate_gap_lfu_minus_lru"):
        mod.tidy_density_gap(raw)
