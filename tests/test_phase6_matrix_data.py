"""Unit tests for the data-loading logic in scripts/15_visualize_phase6_matrix.py.

Covers:
  - primary Phase 6 cell loading (phase6_matrix / phase6_recal)
  - overlay loading of policies that live in their own run dirs
    (phase10_fifo_*), merged into the same matrix cells
  - missing-input accounting: required cells are reported, optional
    overlay cells are reported separately and never block the figures
  - policy presentation metadata staying in sync with POLICY_ORDER

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
    / "15_visualize_phase6_matrix.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("phase6_matrix", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cell(eviction_dir: Path, prefix: str, ds: str, tag: str,
                cache: int, policies: dict[str, float]):
    """Write a minimal multi-seed results JSON for one matrix cell."""
    cell = eviction_dir / f"{prefix}_{ds}_{tag}_100k_c0p{cache}"
    cell.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregated": {
            policy: {
                f"{cache/100:.2f}_temporal": {
                    "hit_rate_mean": rate,
                    "hit_rate_std": 0.0001,
                }
            }
            for policy, rate in policies.items()
        }
    }
    (cell / "eviction_results_multi_seed.json").write_text(json.dumps(payload))


def _write_full_primary(mod, eviction_dir: Path, policies: dict[str, float]):
    """Populate every required (encoder, dataset, cache) cell."""
    for _enc_key, prefix, tag, _label in mod.ENCODERS:
        for ds in mod.DATASET_ORDER:
            for cache in mod.CACHE_SIZES:
                _write_cell(eviction_dir, prefix, ds, tag, cache, policies)


# ═════════════════════════════════════════════════════════════════
# Primary cell loading
# ═════════════════════════════════════════════════════════════════

class TestPrimaryCells:

    def test_reads_hit_rate_from_primary_cell(self, mod, tmp_path):
        _write_full_primary(mod, tmp_path, {"lru": 0.51, "lfu": 0.57})

        loaded = mod.load_matrix(tmp_path)

        assert loaded["matrix"]["minilm"]["lmsys"]["lfu"][10] == (0.57, 0.0001)

    def test_reports_missing_required_cell(self, mod, tmp_path):
        _write_full_primary(mod, tmp_path, {"lfu": 0.57})
        target = (tmp_path / "phase6_matrix_lmsys_minilm_100k_c0p10"
                  / "eviction_results_multi_seed.json")
        target.unlink()

        loaded = mod.load_matrix(tmp_path)

        assert target in loaded["missing"]


# ═════════════════════════════════════════════════════════════════
# Overlay cells (policies run in their own job, e.g. FIFO)
# ═════════════════════════════════════════════════════════════════

class TestOverlayCells:

    def test_overlay_policy_merges_into_matrix_cell(self, mod, tmp_path):
        """FIFO lives in phase10_fifo_* but belongs in the same cell."""
        _write_full_primary(mod, tmp_path, {"lru": 0.51, "lfu": 0.57})
        _write_cell(tmp_path, "phase10_fifo", "lmsys", "minilm", 10,
                    {"fifo": 0.42})

        loaded = mod.load_matrix(tmp_path)

        assert loaded["matrix"]["minilm"]["lmsys"]["fifo"][10] == (0.42, 0.0001)
        # Primary policies in the same cell are untouched.
        assert loaded["matrix"]["minilm"]["lmsys"]["lfu"][10] == (0.57, 0.0001)

    def test_overlay_does_not_override_primary_policy(self, mod, tmp_path):
        """A primary cell already carrying the policy wins."""
        _write_full_primary(mod, tmp_path, {"lfu": 0.57, "fifo": 0.40})
        _write_cell(tmp_path, "phase10_fifo", "lmsys", "minilm", 10,
                    {"fifo": 0.99})

        loaded = mod.load_matrix(tmp_path)

        assert loaded["matrix"]["minilm"]["lmsys"]["fifo"][10] == (0.40, 0.0001)

    def test_absent_overlay_is_not_a_required_missing_cell(self, mod, tmp_path):
        """Figures must still render before the FIFO job lands."""
        _write_full_primary(mod, tmp_path, {"lru": 0.51, "lfu": 0.57})

        loaded = mod.load_matrix(tmp_path)

        assert loaded["missing"] == []

    def test_absent_overlay_is_reported_separately(self, mod, tmp_path):
        """Silent omission would read as 'FIFO was covered' when it wasn't."""
        _write_full_primary(mod, tmp_path, {"lru": 0.51, "lfu": 0.57})

        loaded = mod.load_matrix(tmp_path)

        expected = (tmp_path / "phase10_fifo_lmsys_minilm_100k_c0p10"
                    / "eviction_results_multi_seed.json")
        assert expected in loaded["overlay_missing"]

    def test_present_overlay_is_not_reported_missing(self, mod, tmp_path):
        _write_full_primary(mod, tmp_path, {"lfu": 0.57})
        for _enc_key, _prefix, tag, _label in mod.ENCODERS:
            for ds in mod.DATASET_ORDER:
                for cache in mod.CACHE_SIZES:
                    _write_cell(tmp_path, "phase10_fifo", ds, tag, cache,
                                {"fifo": 0.42})

        loaded = mod.load_matrix(tmp_path)

        assert loaded["overlay_missing"] == []


# ═════════════════════════════════════════════════════════════════
# Presentation metadata
# ═════════════════════════════════════════════════════════════════

class TestPresentPolicies:
    """Policies with no data anywhere must not render as blank rows."""

    def test_excludes_policy_absent_from_every_cell(self, mod, tmp_path):
        _write_full_primary(mod, tmp_path, {"lru": 0.51, "lfu": 0.57})

        present = mod.present_policies(mod.load_matrix(tmp_path)["matrix"])

        assert "fifo" not in present

    def test_includes_partially_covered_policy(self, mod, tmp_path):
        """One cell of real data is enough to earn a row."""
        _write_full_primary(mod, tmp_path, {"lru": 0.51, "lfu": 0.57})
        _write_cell(tmp_path, "phase10_fifo", "lmsys", "minilm", 10,
                    {"fifo": 0.42})

        present = mod.present_policies(mod.load_matrix(tmp_path)["matrix"])

        assert "fifo" in present

    def test_preserves_policy_order(self, mod, tmp_path):
        _write_full_primary(mod, tmp_path, {"lfu": 0.57, "lru": 0.51})

        present = mod.present_policies(mod.load_matrix(tmp_path)["matrix"])

        assert present == [p for p in mod.POLICY_ORDER if p in present]


class TestPolicyMetadata:

    def test_fifo_is_in_policy_order(self, mod):
        assert "fifo" in mod.POLICY_ORDER

    def test_every_policy_has_label_color_and_marker(self, mod):
        for policy in mod.POLICY_ORDER:
            assert policy in mod.POLICY_LABELS
            assert policy in mod.POLICY_COLORS
            assert policy in mod.POLICY_MARKERS

    def test_policy_colors_are_distinct(self, mod):
        colors = [mod.POLICY_COLORS[p] for p in mod.POLICY_ORDER]
        assert len(set(colors)) == len(colors)
