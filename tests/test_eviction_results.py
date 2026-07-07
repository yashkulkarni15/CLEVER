# tests/test_eviction_results.py
import json

import pytest

from src.utils.eviction_results import load_aggregated_hit_rates


def _write_result(tmp_path, dirname, policy_rates, cache_key="0.10",
                  command_line=None, dataset="qqp"):
    """Fixture mirroring eviction_results_multi_seed.json: manifest.command_line
    carries the dataset; aggregated → policy → cache key → hit_rate_mean."""
    d = tmp_path / dirname
    d.mkdir(parents=True)
    if command_line is None:
        command_line = (
            f"scripts/08_run_eviction.py --dataset {dataset} "
            f"--embedding-model all-MiniLM-L6-v2 --size 100k "
            f"--cache-sizes {cache_key} --workloads temporal --multi-seed"
        )
    data = {
        "manifest": {"timestamp_utc": "2026-06-12T22:03:57Z",
                     "command_line": command_line},
        "per_seed": {},
        "aggregated": {
            pol: {cache_key: {"hit_rate_mean": hr, "hit_rate_std": 0.0,
                              "n_seeds": 3, "seeds": [42, 123, 456],
                              "workloads": ["temporal"]}}
            for pol, hr in policy_rates.items()
        },
    }
    path = d / "eviction_results_multi_seed.json"
    path.write_text(json.dumps(data))
    return path


def test_dataset_read_from_manifest_not_dirname(tmp_path):
    # Directory names deliberately contradict the manifest: the loader must
    # trust JSON content only.
    _write_result(tmp_path, "phase6_matrix_lmsys_minilm_100k_c0p10",
                  {"lru": 0.575043, "lfu": 0.5543}, dataset="qqp")
    _write_result(tmp_path, "phase6_matrix_qqp_minilm_100k_c0p10",
                  {"lru": 0.31}, dataset="lmsys")
    out = load_aggregated_hit_rates(
        str(tmp_path / "*" / "eviction_results_multi_seed.json"), "0.10")
    assert out[("qqp", "lru")] == 0.575043
    assert out[("qqp", "lfu")] == 0.5543
    assert out[("lmsys", "lru")] == 0.31
    assert ("lmsys", "lfu") not in out


def test_missing_dataset_flag_raises_naming_file(tmp_path):
    path = _write_result(
        tmp_path, "run_a", {"lru": 0.5},
        command_line="scripts/08_run_eviction.py --size 100k --multi-seed")
    with pytest.raises(SystemExit) as ei:
        load_aggregated_hit_rates(
            str(tmp_path / "*" / "eviction_results_multi_seed.json"), "0.10")
    assert str(path) in str(ei.value)
    assert "--dataset" in str(ei.value)


def test_strict_cache_key_no_fallback(tmp_path):
    path = _write_result(tmp_path, "run_a", {"lru": 0.5}, cache_key="0.10")
    with pytest.raises(SystemExit) as ei:
        load_aggregated_hit_rates(
            str(tmp_path / "*" / "eviction_results_multi_seed.json"), "0.20")
    msg = str(ei.value)
    assert str(path) in msg
    assert "0.10" in msg  # available keys listed


def test_duplicate_dataset_policy_raises_naming_both_files(tmp_path):
    # Two files claim (qqp, lru) at the same cache key with different rates:
    # a silent last-wins join would pick one arbitrarily.
    path_a = _write_result(tmp_path, "phase6_matrix_qqp_minilm_100k_c0p10",
                           {"lru": 0.575043}, dataset="qqp")
    path_b = _write_result(tmp_path, "phase6_recal_qqp_minilm_100k_c0p10",
                           {"lru": 0.31}, dataset="qqp")
    with pytest.raises(SystemExit) as ei:
        load_aggregated_hit_rates(
            str(tmp_path / "*" / "eviction_results_multi_seed.json"), "0.10")
    msg = str(ei.value)
    assert str(path_a) in msg
    assert str(path_b) in msg
    assert "qqp" in msg
    assert "lru" in msg


def test_zero_files_matched_raises_naming_pattern(tmp_path):
    pattern = str(tmp_path / "nothing_here" / "*.json")
    with pytest.raises(SystemExit) as ei:
        load_aggregated_hit_rates(pattern, "0.10")
    assert pattern in str(ei.value)
