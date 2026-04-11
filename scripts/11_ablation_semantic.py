#!/usr/bin/env python3
"""
Formal Ablation suite for Semantic Cache Eviction.

Evaluates the theoretical invariants introduced:
1. Eventual Evictability (via mu parameter smoothing)
2. Drift Symmetry (via dynamic imputation to counter 0.0 defaults)
3. Under-the-hood Vectorization (speed tested inherently on all sem configs vs LRU)

Usage:
    ./venv/bin/python scripts/11_ablation_semantic.py
"""

import sys
import time
import logging
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

def run_ablation(
    embeddings: np.ndarray,
    texts: list[str],
    policy_name: str,
    cache_pct: float,
    hit_threshold: float,
    semantic_cfg: dict | None = None,
) -> dict:
    """Run one cache simulation stream."""
    n = len(embeddings)
    dim = embeddings.shape[1]

    max_cache = int(n * cache_pct)
    warmup_pct = 0.30
    n_warmup = max_cache  # Fill cache
    n_stream = n - n_warmup

    warmup_embs = embeddings[:n_warmup]
    stream_embs = embeddings[n_warmup:]
    warmup_texts = texts[:n_warmup]
    stream_texts = texts[n_warmup:]

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
            recompute_interval=cfg.get("recompute_interval", 100),
            mu=cfg.get("mu", 0.1),
            dynamic_impute=cfg.get("dynamic_impute", True)
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    cache = SemanticCache(
        dim=dim,
        index_type="hnsw",
        index_params={"M": 32, "efConstruction": 128, "efSearch": 128},
        max_size=max_cache,
        eviction_policy=policy,
    )
    
    t_start_build = time.perf_counter()
    cache.build(warmup_embs, warmup_texts)
    build_time = time.perf_counter() - t_start_build

    n_hits = 0
    t_start_stream = time.perf_counter()
    for i in range(n_stream):
        result = cache.lookup(stream_embs[i], k=1, threshold=hit_threshold)
        if result.hit:
            n_hits += 1
        else:
            cache.insert(stream_embs[i], stream_texts[i])
    stream_time = time.perf_counter() - t_start_stream

    hit_rate = n_hits / n_stream if n_stream > 0 else 0
    avg_query_ms = stream_time / n_stream * 1000 if n_stream > 0 else 0

    return {
        "policy": policy_name,
        "cache_pct": cache_pct,
        "hit_rate": hit_rate,
        "avg_query_ms": avg_query_ms,
        "stream_time": stream_time
    }

def main():
    emb_path = "results/embeddings/full_embeddings.npy"
    if not Path(emb_path).exists():
        logger.error(f"Embeddings not found at {emb_path}")
        return

    embeddings = np.load(emb_path)
    
    # Use 15,000 requests to make ablations fast locally
    subset_size = min(15000, len(embeddings))
    embeddings = embeddings[:subset_size]
    texts = [f"doc_{i}" for i in range(subset_size)]
    
    logger.info(f"Loaded structured stream of {subset_size} queries.")

    cache_sizes = [0.10, 0.20]
    hit_threshold = 0.50 # Use rigorous matching to separate dense clusters
    
    ablation_cfgs = [
        {"name": "Sem(μ=0,Dyn=F)", "mu": 0.0, "dynamic_impute": False, "recompute_interval": 200},
        {"name": "Sem(μ=0.1,Dyn=F)", "mu": 0.1, "dynamic_impute": False, "recompute_interval": 200},
        {"name": "Sem(μ=0.1,Dyn=T)", "mu": 0.1, "dynamic_impute": True, "recompute_interval": 200},
    ]

    results = []
    
    run_num = 0
    total = len(cache_sizes) * (2 + len(ablation_cfgs))

    for cache_pct in cache_sizes:
        for baseline in ["lru", "lfu"]:
            run_num += 1
            logger.info(f"[{run_num}/{total}] Running {baseline} @ {cache_pct*100:.0f}%")
            r = run_ablation(embeddings, texts, baseline, cache_pct, hit_threshold)
            results.append(r)

        for cfg in ablation_cfgs:
            run_num += 1
            logger.info(f"[{run_num}/{total}] Running {cfg['name']} @ {cache_pct*100:.0f}%")
            scfg = {"similarity_threshold": 0.30, "alpha": 1.0, "beta": 1.0, **cfg}
            r = run_ablation(embeddings, texts, "semantic", cache_pct, hit_threshold, semantic_cfg=scfg)
            r["policy"] = cfg["name"]
            results.append(r)

    print("\n" + "=" * 80)
    print(f"{'Policy Variants':<20s} {'Cache%':>8s} {'HitRate':>10s} {'ms/query':>10s} {'Total Time':>12s}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['policy']:<20s} {r['cache_pct']:>7.0%} {r['hit_rate']:>10.4f} {r['avg_query_ms']:>10.4f} {r['stream_time']:>11.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
