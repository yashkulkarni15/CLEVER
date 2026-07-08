# tests/test_log_hits.py
# Run with FAISS thread pins (see Global Constraints).
import importlib.util
import json
import numpy as np
import pytest
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


def _hitty_data():
    # 40 unit-norm embeddings with deliberate duplicates so hits fire.
    rng = np.random.RandomState(0)
    base = rng.randn(8, 16).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    embs = np.repeat(base, 5, axis=0)          # 40 rows, each repeated 5x
    texts = [f"q{i % 8}" for i in range(40)]
    return embs, texts


def test_hits_path_uses_round_not_truncate(tmp_path):
    p = run_eviction._hits_path(tmp_path, "lmsys", "lru", 0.29, 42, "temporal")
    assert p.name == "hits_lmsys_lru_c0p29_seed42_temporal.jsonl"
    p = run_eviction._hits_path(tmp_path, "lmsys", "lru", 0.10, 42, "temporal")
    assert p.name == "hits_lmsys_lru_c0p10_seed42_temporal.jsonl"


def test_log_hits_writes_pairs(tmp_path):
    embs, texts = _hitty_data()

    res = run_eviction.evaluate_policy(
        "lru", embs, texts, _config(), cache_size_pct=0.25, seed=42,
        workload_type="temporal", log_hits=True, hits_dir=tmp_path,
        dataset="lmsys",
    )
    out = tmp_path / "hits_lmsys_lru_c0p25_seed42_temporal.jsonl"
    assert out.exists()
    assert not list(tmp_path.glob("*.tmp")), "temp file left after success"
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == res["n_hits"] >= 1
    row = lines[0]
    assert set(row) >= {"dataset", "policy", "stream_idx", "distance_l2sq",
                        "matched_cache_id", "orig_query", "new_query"}
    assert row["dataset"] == "lmsys" and row["policy"] == "lru"


def test_hit_records_carry_embedding_model(tmp_path):
    embs, texts = _hitty_data()

    tagged_dir = tmp_path / "tagged"
    run_eviction.evaluate_policy(
        "lru", embs, texts, _config(), cache_size_pct=0.25, seed=42,
        workload_type="temporal", log_hits=True, hits_dir=tagged_dir,
        dataset="lmsys", embedding_model="all-MiniLM-L6-v2",
    )
    out = tagged_dir / "hits_lmsys_lru_c0p25_seed42_temporal.jsonl"
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines and all(
        r["embedding_model"] == "all-MiniLM-L6-v2" for r in lines
    )

    default_dir = tmp_path / "default"
    run_eviction.evaluate_policy(
        "lru", embs, texts, _config(), cache_size_pct=0.25, seed=42,
        workload_type="temporal", log_hits=True, hits_dir=default_dir,
        dataset="lmsys",
    )
    out = default_dir / "hits_lmsys_lru_c0p25_seed42_temporal.jsonl"
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines and all(r["embedding_model"] == "unknown" for r in lines)


def test_checkpoint_resume_regenerates_missing_hits(tmp_path):
    embs, texts = _hitty_data()
    cfg = _config()
    cfg["eviction"]["policies"] = ["lru"]
    cfg["cache"]["cache_sizes_pct"] = [0.25]
    cfg["evaluation"]["workloads"] = ["temporal"]
    ckpt = tmp_path / "checkpoint.json"
    hits_dir = tmp_path / "hits"

    # First pass: checkpoint written, no hit logging requested.
    run_eviction.run_full_experiment(
        embs, texts, cfg, seeds=[42], checkpoint_path=ckpt,
    )
    assert ckpt.exists()
    assert not hits_dir.exists()

    # Second pass with log_hits: the checkpointed run must be re-executed
    # (its hits file is missing) and must not duplicate the per-seed entry.
    results = run_eviction.run_full_experiment(
        embs, texts, cfg, seeds=[42], checkpoint_path=ckpt,
        log_hits=True, hits_dir=hits_dir, dataset="lmsys",
    )
    out = hits_dir / "hits_lmsys_lru_c0p25_seed42_temporal.jsonl"
    assert out.exists(), "hits file not regenerated for checkpointed run"
    assert len(results["per_seed"]["lru"]["0.25"]) == 1

    # Third pass: hits file now present → run is skipped, file untouched.
    mtime = out.stat().st_mtime_ns
    run_eviction.run_full_experiment(
        embs, texts, cfg, seeds=[42], checkpoint_path=ckpt,
        log_hits=True, hits_dir=hits_dir, dataset="lmsys",
    )
    assert out.stat().st_mtime_ns == mtime


def test_interrupted_run_leaves_no_partial_jsonl(tmp_path, monkeypatch):
    embs, texts = _hitty_data()

    calls = {"n": 0}
    orig_lookup = run_eviction.SemanticCache.lookup

    def dying_lookup(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 8:
            raise RuntimeError("simulated mid-stream kill")
        return orig_lookup(self, *args, **kwargs)

    monkeypatch.setattr(run_eviction.SemanticCache, "lookup", dying_lookup)

    with pytest.raises(RuntimeError, match="simulated mid-stream kill"):
        run_eviction.evaluate_policy(
            "lru", embs, texts, _config(), cache_size_pct=0.25, seed=42,
            workload_type="temporal", log_hits=True, hits_dir=tmp_path,
            dataset="lmsys",
        )
    assert not list(tmp_path.glob("*.jsonl")), "partial hits .jsonl left behind"
