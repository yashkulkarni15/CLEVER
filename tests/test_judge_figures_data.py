"""Unit tests for the pure data-shaping logic in scripts/18_visualize_judge.py.

Covers:
  - Wilson 95% CI for binomial YES rates
  - judge_scores.json loading / completeness validation
  - scores -> tidy plotting frame (quality-adjusted + CI propagation)
  - verdict records -> per-dataset YES-rate-vs-distance curve (pair dedupe,
    quantile bins, tied-distance robustness, None-verdict exclusion)
  - LaTeX table emission

Plot rendering is intentionally untested here; it is verified by running the
script and checking that nonzero PNG/PDF files are produced.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "18_visualize_judge.py"
)

DATASETS = ["lmsys", "qqp", "moss"]
POLICIES = ["lru", "lfu", "semantic", "arc", "gdsf", "siso"]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("visualize_judge", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_scores(overrides: dict | None = None) -> dict:
    """Full 18-cell scores dict matching the judge_scores.json schema."""
    scores = {}
    for ds in DATASETS:
        for policy in POLICIES:
            scores[f"{ds}/{policy}"] = {
                "n_records": 1000,
                "n_judged": 1000,
                "n_unparsed": 0,
                "n_failed": 0,
                "n_pending": 0,
                "mean_yes": 0.039,
                "raw_hit_rate": 0.570114,
                "quality_adjusted_hit_rate": 0.022234,
            }
    if overrides:
        for key, cell in overrides.items():
            scores[key] = {**scores[key], **cell}
    return scores


# ─────────────────────────────────────────────────────────────────
# Wilson CI
# ─────────────────────────────────────────────────────────────────

def test_wilson_ci_interior_brackets_point_estimate(mod):
    lo, hi = mod.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert lo == pytest.approx(0.4038, abs=0.005)
    assert hi == pytest.approx(0.5962, abs=0.005)


def test_wilson_ci_zero_successes(mod):
    lo, hi = mod.wilson_ci(0, 100)
    assert lo == 0.0
    assert hi == pytest.approx(0.0370, abs=0.002)


def test_wilson_ci_all_successes(mod):
    lo, hi = mod.wilson_ci(1000, 1000)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert 0.99 < lo < 1.0


def test_wilson_ci_zero_n(mod):
    lo, hi = mod.wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# load_scores
# ─────────────────────────────────────────────────────────────────

def test_load_scores_reads_complete_file(mod, tmp_path):
    path = tmp_path / "judge_scores.json"
    path.write_text(json.dumps(make_scores()))
    scores = mod.load_scores(path)
    assert len(scores) == 18
    assert scores["lmsys/lru"]["raw_hit_rate"] == pytest.approx(0.570114)


def test_load_scores_rejects_missing_cell(mod, tmp_path):
    scores = make_scores()
    del scores["qqp/siso"]
    path = tmp_path / "judge_scores.json"
    path.write_text(json.dumps(scores))
    with pytest.raises(ValueError, match="qqp/siso"):
        mod.load_scores(path)


def test_load_scores_rejects_zero_judged_cell(mod, tmp_path):
    path = tmp_path / "judge_scores.json"
    path.write_text(json.dumps(make_scores({"moss/arc": {"n_judged": 0}})))
    with pytest.raises(ValueError, match="moss/arc"):
        mod.load_scores(path)


# ─────────────────────────────────────────────────────────────────
# scores_frame
# ─────────────────────────────────────────────────────────────────

def test_scores_frame_has_all_cells_in_canonical_order(mod):
    frame = mod.scores_frame(make_scores())
    assert len(frame) == 18
    assert (frame[0]["dataset"], frame[0]["policy"]) == ("lmsys", "lru")
    assert (frame[-1]["dataset"], frame[-1]["policy"]) == ("moss", "siso")


def test_scores_frame_quality_adjustment_and_ci_propagation(mod):
    scores = make_scores({
        "lmsys/lru": {"mean_yes": 0.05, "raw_hit_rate": 0.6, "n_judged": 1000},
    })
    row = mod.scores_frame(scores)[0]
    assert row["raw"] == pytest.approx(0.6)
    assert row["mean_yes"] == pytest.approx(0.05)
    assert row["qadj"] == pytest.approx(0.03)
    # CI on mean_yes comes from Wilson at k=50, n=1000, scaled by raw.
    yes_lo, yes_hi = mod.wilson_ci(50, 1000)
    assert row["yes_lo"] == pytest.approx(yes_lo)
    assert row["yes_hi"] == pytest.approx(yes_hi)
    assert row["qadj_lo"] == pytest.approx(0.6 * yes_lo)
    assert row["qadj_hi"] == pytest.approx(0.6 * yes_hi)


# ─────────────────────────────────────────────────────────────────
# distance_yes_curve
# ─────────────────────────────────────────────────────────────────

def make_record(ds="lmsys", dist=0.5, verdict=False, orig="q1", new="q2"):
    return {
        "dataset": ds,
        "distance_l2sq": dist,
        "verdict": verdict,
        "orig_query": orig,
        "new_query": new,
    }


def test_distance_yes_curve_bins_by_quantile(mod):
    records = [
        make_record(dist=i / 100, verdict=(i < 20), orig=f"o{i}", new=f"n{i}")
        for i in range(100)
    ]
    curve = mod.distance_yes_curve(records, n_bins=5)
    bins = curve["lmsys"]
    assert len(bins) == 5
    assert sum(b["n"] for b in bins) == 100
    assert bins[0]["yes_rate"] == pytest.approx(1.0)
    for b in bins[1:]:
        assert b["yes_rate"] == pytest.approx(0.0)
    assert bins[0]["dist_median"] < bins[-1]["dist_median"]


def test_distance_yes_curve_dedupes_pairs(mod):
    records = [
        make_record(dist=i / 100, verdict=(i < 20), orig=f"o{i}", new=f"n{i}")
        for i in range(100)
    ]
    # Same pair judged under several policies: must count once.
    records.append(make_record(dist=0.0, verdict=True, orig="o0", new="n0"))
    curve = mod.distance_yes_curve(records, n_bins=5)
    assert sum(b["n"] for b in curve["lmsys"]) == 100


def test_distance_yes_curve_skips_none_verdicts(mod):
    records = [
        make_record(dist=i / 10, verdict=True, orig=f"o{i}", new=f"n{i}")
        for i in range(10)
    ]
    records.append(make_record(dist=0.5, verdict=None, orig="ox", new="nx"))
    curve = mod.distance_yes_curve(records, n_bins=2)
    assert sum(b["n"] for b in curve["lmsys"]) == 10


def test_distance_yes_curve_survives_tied_distances(mod):
    records = [
        make_record(dist=0.25, verdict=(i % 2 == 0), orig=f"o{i}", new=f"n{i}")
        for i in range(40)
    ]
    curve = mod.distance_yes_curve(records, n_bins=5)
    bins = curve["lmsys"]
    assert sum(b["n"] for b in bins) == 40
    total_yes = sum(b["yes_rate"] * b["n"] for b in bins)
    assert total_yes == pytest.approx(20.0)


def test_distance_yes_curve_groups_by_dataset(mod):
    records = [make_record(ds="lmsys", orig="a", new="b", verdict=True),
               make_record(ds="qqp", orig="c", new="d", verdict=False)]
    curve = mod.distance_yes_curve(records, n_bins=1)
    assert curve["lmsys"][0]["yes_rate"] == pytest.approx(1.0)
    assert curve["qqp"][0]["yes_rate"] == pytest.approx(0.0)


def test_distance_yes_curve_bin_ci_brackets_rate(mod):
    records = [
        make_record(dist=i / 100, verdict=(i % 2 == 0), orig=f"o{i}", new=f"n{i}")
        for i in range(100)
    ]
    curve = mod.distance_yes_curve(records, n_bins=2)
    for b in curve["lmsys"]:
        assert b["lo"] <= b["yes_rate"] <= b["hi"]
        assert b["hi"] - b["lo"] < 1.0


# ─────────────────────────────────────────────────────────────────
# latex_table
# ─────────────────────────────────────────────────────────────────

def test_latex_table_structure_and_values(mod):
    scores = make_scores({
        "lmsys/lru": {
            "mean_yes": 0.036, "raw_hit_rate": 0.555357,
            "quality_adjusted_hit_rate": 0.019993, "n_judged": 1000,
        },
    })
    table = mod.latex_table(scores)
    assert "\\begin{tabular}" in table and "\\end{tabular}" in table
    for label in ["LRU", "LFU", "Semantic", "ARC", "GDSF", "SISO"]:
        assert label in table
    # lmsys/lru row: raw 55.5%, yes 3.6%, qadj 2.0%.
    assert "55.5" in table
    assert "3.6" in table
    assert "2.0" in table


def test_latex_table_has_one_row_per_policy(mod):
    table = mod.latex_table(make_scores())
    assert table.count("\\\\") >= len(POLICIES)
