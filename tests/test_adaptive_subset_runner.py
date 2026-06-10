"""CLI smoke tests for the Phase 3A adaptive subset runner."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def test_adaptive_subset_runner_writes_result_and_summary_csvs(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.99, 0.01, 0.0],
            [0.0, 0.98, -0.01, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.99, 0.02, 0.0],
            [0.0, 0.98, -0.02, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01, 0.0],
            [0.0, 0.98, -0.01, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.99, 0.02, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    emb_path = tmp_path / "embeddings.npy"
    np.save(emb_path, embeddings)

    queries_path = tmp_path / "queries.parquet"
    pd.DataFrame(
        {
            "query_id": list(range(len(embeddings))),
            "query_text": [f"query-{i}" for i in range(len(embeddings))],
        }
    ).to_parquet(queries_path, index=False)

    config_path = tmp_path / "eviction.yaml"
    config = {
        "cache": {
            "index_type": "flat",
            "index_params": {},
            "cache_sizes_pct": [0.25],
        },
        "eviction": {
            "policies": ["lru", "lfu", "semantic"],
            "semantic": {
                "similarity_threshold": 0.05,
                "recompute_interval": 1,
                "dynamic_impute": True,
            },
            "adaptive": {
                "similarity_threshold": 0.05,
                "density_floor": 0.01,
                "density_threshold_scale": 0.0,
                "frequency_skew_threshold": 1.5,
                "min_observations": 0,
                "recompute_interval": 1,
            },
        },
        "evaluation": {
            "warmup_pct": 0.25,
            "hit_threshold": 0.05,
            "rolling_window": 4,
            "log_interval": 500,
            "workloads": ["temporal"],
        },
        "seed": 42,
        "seeds": [42],
    }
    config_path.write_text(yaml.safe_dump(config))

    output_dir = tmp_path / "adaptive"
    script = Path(__file__).resolve().parents[1] / "scripts" / "13_run_adaptive_subset.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--embeddings",
            str(emb_path),
            "--queries",
            str(queries_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--policies",
            "lru",
            "lfu",
            "semantic",
            "adaptive_hard",
            "adaptive_blend",
            "--cache-size",
            "0.25",
            "--max-queries",
            "12",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr

    result_csv = output_dir / "adaptive_subset_results.csv"
    summary_csv = output_dir / "adaptive_subset_summary.csv"
    assert result_csv.exists()
    assert summary_csv.exists()

    results = pd.read_csv(result_csv)
    summary = pd.read_csv(summary_csv)
    expected = {"lru", "lfu", "semantic", "adaptive_hard", "adaptive_blend"}
    assert set(results["policy"]) == expected
    assert set(summary["policy"]) == expected
    assert {
        "policy",
        "final_hit_rate",
        "semantic_coverage_avg_dist",
        "avg_query_time_ms",
        "n_evictions",
        "policy_stats_json",
    }.issubset(results.columns)
    assert summary["final_hit_rate"].between(0.0, 1.0).all()
