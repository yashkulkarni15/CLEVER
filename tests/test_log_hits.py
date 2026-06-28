# tests/test_log_hits.py
# Run with FAISS thread pins (see Global Constraints).
import importlib.util
import json
import numpy as np
from pathlib import Path

# 08_run_eviction.py inserts repo-root on sys.path at import time, so `src.*`
# imports resolve once exec_module runs.
SPEC = importlib.util.spec_from_file_location(
    "run_eviction",
    Path(__file__).resolve().parent.parent / "scripts" / "08_run_eviction.py",
)
run_eviction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_eviction)


def _config():
    return {
        "cache": {"index_type": "hnsw", "index_params": {}},
        "evaluation": {"warmup_pct": 0.3, "hit_threshold": 0.90,
                       "rolling_window": 100, "log_interval": 50},
        "eviction": {},
    }


def test_log_hits_writes_pairs(tmp_path):
    # 40 unit-norm embeddings with deliberate duplicates so hits fire.
    rng = np.random.RandomState(0)
    base = rng.randn(8, 16).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    embs = np.repeat(base, 5, axis=0)          # 40 rows, each repeated 5x
    texts = [f"q{i % 8}" for i in range(40)]

    res = run_eviction.evaluate_policy(
        "lru", embs, texts, _config(), cache_size_pct=0.25, seed=42,
        workload_type="temporal", log_hits=True, hits_dir=tmp_path,
        dataset="lmsys",
    )
    out = tmp_path / "hits_lmsys_lru_c0p25_seed42_temporal.jsonl"
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == res["n_hits"] >= 1
    row = lines[0]
    assert set(row) >= {"dataset", "policy", "stream_idx", "distance_l2sq",
                        "matched_cache_id", "orig_query", "new_query"}
    assert row["dataset"] == "lmsys" and row["policy"] == "lru"
