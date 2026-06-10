"""CLI smoke tests for the Phase 2 density profile runner."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def test_density_profile_runner_writes_density_and_gap_csvs(tmp_path):
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [11.0, 10.0],
        ],
        dtype=np.float32,
    )
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
        "eviction": {"policies": ["lru", "lfu"]},
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

    output_dir = tmp_path / "density"
    script = Path(__file__).resolve().parents[1] / "scripts" / "12_run_density_profile.py"

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
            "--cache-size",
            "0.25",
            "--density-theta",
            "0.05",
            "--checkpoints",
            "2",
            "--exact-density",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr

    density_csv = output_dir / "density_log.csv"
    gap_csv = output_dir / "density_gap.csv"
    assert density_csv.exists()
    assert gap_csv.exists()

    density = pd.read_csv(density_csv)
    gap = pd.read_csv(gap_csv)

    assert set(density["policy"]) == {"lru", "lfu"}
    assert len(density) == 4
    assert len(gap) == 2
    assert {
        "query_idx",
        "policy",
        "mean_density",
        "cache_snapshot_size",
        "cumulative_hit_rate",
        "n_evictions",
    }.issubset(density.columns)
    assert {
        "query_idx",
        "mean_density_reference",
        "mean_density_lru",
        "mean_density_lfu",
        "hit_rate_lru",
        "hit_rate_lfu",
        "hit_rate_gap_lfu_minus_lru",
    }.issubset(gap.columns)
    assert gap["mean_density_reference"].between(0.0, 1.0).all()
