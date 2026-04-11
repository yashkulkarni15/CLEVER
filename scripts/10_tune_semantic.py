#!/usr/bin/env python3
"""
Quick local parameter sweep for semantic eviction policy.

Tests different configurations (cache sizes, recompute intervals,
hit thresholds) on the dev subset to find where semantic shines
before committing to a full HPC run.

Usage:
    python scripts/10_tune_semantic.py
"""

import sys
import time
import logging
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache.semantic_cache import SemanticCache
from src.cache.eviction.lru import LRUPolicy
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.semantic import SemanticPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_single(
    embeddings: np.ndarray,
    policy_name: str,
    cache_pct: float,
    hit_threshold: float,
    semantic_cfg: dict | None = None,
    seed: int = 42,
) -> dict:
    """Run one eviction experiment and return results."""
    n = len(embeddings)
    rng = np.random.RandomState(seed)
    dim = embeddings.shape[1]

    max_cache = int(n * cache_pct)
    warmup_pct = 0.30
    n_warmup = max_cache  # warmup fills the cache
    n_stream = n - n_warmup

    warmup_embs = embeddings[:n_warmup]
    stream_embs = embeddings[n_warmup:]

    # Dummy texts (just indices)
    warmup_texts = [f"q{i}" for i in range(n_warmup)]
    stream_texts = [f"q{n_warmup + i}" for i in range(n_stream)]

    # Create policy
    if policy_name == "lru":
        policy = LRUPolicy()
    elif policy_name == "lfu":
        policy = LFUPolicy()
    elif policy_name == "semantic":
        cfg = semantic_cfg or {}
        policy = SemanticPolicy(
            similarity_threshold=cfg.get("similarity_threshold", 0.30),
            alpha=cfg.get("alpha", 1.0),
            beta=cfg.get("beta", 1.0),
            recompute_interval=cfg.get("recompute_interval", 500),
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Build cache
    cache = SemanticCache(
        dim=dim,
        index_type="hnsw",
        index_params={"M": 32, "efConstruction": 128, "efSearch": 128},
        max_size=max_cache,
        eviction_policy=policy,
    )
    cache.build(warmup_embs, warmup_texts)

    # Stream
    n_hits = 0
    t0 = time.perf_counter()
    for i in range(n_stream):
        result = cache.lookup(stream_embs[i], k=1, threshold=hit_threshold)
        if result.hit:
            n_hits += 1
        else:
            cache.insert(stream_embs[i], stream_texts[i])
    elapsed = time.perf_counter() - t0

    hit_rate = n_hits / n_stream if n_stream > 0 else 0
    avg_ms = elapsed / n_stream * 1000 if n_stream > 0 else 0

    return {
        "policy": policy_name,
        "cache_pct": cache_pct,
        "hit_threshold": hit_threshold,
        "hit_rate": round(hit_rate, 6),
        "n_hits": n_hits,
        "n_misses": n_stream - n_hits,
        "avg_query_ms": round(avg_ms, 4),
        "elapsed_s": round(elapsed, 2),
        "semantic_cfg": semantic_cfg if policy_name == "semantic" else None,
    }


def main():
    # Load dev embeddings
    emb_path = "results/embeddings/full_embeddings.npy"
    embeddings = np.load(emb_path)
    logger.info(f"Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    # ── Experiment grid ──────────────────────────────────────────
    cache_sizes = [0.05, 0.10, 0.15, 0.20]
    hit_thresholds = [0.90, 0.70, 0.50]
    semantic_configs = [
        {"name": "sem_default", "similarity_threshold": 0.30, "alpha": 1.0, "beta": 1.0, "recompute_interval": 5000},
        {"name": "sem_freq500", "similarity_threshold": 0.30, "alpha": 1.0, "beta": 1.0, "recompute_interval": 500},
        {"name": "sem_freq100", "similarity_threshold": 0.30, "alpha": 1.0, "beta": 1.0, "recompute_interval": 100},
        {"name": "sem_recency", "similarity_threshold": 0.30, "alpha": 2.0, "beta": 0.5, "recompute_interval": 500},
        {"name": "sem_frequency", "similarity_threshold": 0.30, "alpha": 0.5, "beta": 2.0, "recompute_interval": 500},
    ]

    results = []
    total = len(cache_sizes) * len(hit_thresholds) * (2 + len(semantic_configs))
    run_num = 0

    for cache_pct in cache_sizes:
        for ht in hit_thresholds:
            # Baselines
            for baseline in ["lru", "lfu"]:
                run_num += 1
                logger.info(f"[{run_num}/{total}] {baseline} cache={cache_pct:.0%} ht={ht}")
                r = run_single(embeddings, baseline, cache_pct, ht)
                results.append(r)

            # Semantic variants
            for scfg in semantic_configs:
                run_num += 1
                name = scfg["name"]
                cfg = {k: v for k, v in scfg.items() if k != "name"}
                logger.info(f"[{run_num}/{total}] {name} cache={cache_pct:.0%} ht={ht}")
                r = run_single(embeddings, "semantic", cache_pct, ht, semantic_cfg=cfg)
                r["policy"] = name  # override for readability
                results.append(r)

    # ── Print results table ──────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Policy':<16s} {'Cache%':>6s} {'HitThr':>6s} {'HitRate':>8s} {'Hits':>6s} {'Miss':>6s} {'ms/q':>7s} {'Time':>6s}")
    print("-" * 95)

    for r in results:
        print(
            f"{r['policy']:<16s} {r['cache_pct']:>5.0%} {r['hit_threshold']:>6.2f} "
            f"{r['hit_rate']:>7.4f} {r['n_hits']:>6d} {r['n_misses']:>6d} "
            f"{r['avg_query_ms']:>7.3f} {r['elapsed_s']:>5.1f}s"
        )

    # ── Summary: best semantic vs best baseline per config ───────
    print("\n" + "=" * 95)
    print("SUMMARY: Semantic vs Best Baseline")
    print("-" * 95)

    for cache_pct in cache_sizes:
        for ht in hit_thresholds:
            subset = [r for r in results if r["cache_pct"] == cache_pct and r["hit_threshold"] == ht]
            baselines = [r for r in subset if r["policy"] in ("lru", "lfu")]
            semantics = [r for r in subset if r["policy"] not in ("lru", "lfu")]

            best_base = max(baselines, key=lambda r: r["hit_rate"])
            best_sem = max(semantics, key=lambda r: r["hit_rate"])

            delta = best_sem["hit_rate"] - best_base["hit_rate"]
            marker = "✅" if delta > 0.005 else ("⚠️ " if delta > 0 else "❌")

            print(
                f"  cache={cache_pct:.0%} ht={ht:.2f}: "
                f"best_baseline={best_base['policy']}({best_base['hit_rate']:.4f}) "
                f"best_semantic={best_sem['policy']}({best_sem['hit_rate']:.4f}) "
                f"Δ={delta:+.4f} {marker}"
            )


if __name__ == "__main__":
    main()
