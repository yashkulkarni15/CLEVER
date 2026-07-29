#!/usr/bin/env python3
"""
Phase 4 — Eviction policy evaluation.

Simulates a realistic cache lifecycle:
1. **Warmup**: Build a bounded-size cache from the first portion of queries.
2. **Stream**: Replay the remaining queries as a live query stream.
   - On **hit**: record success, update eviction bookkeeping.
   - On **miss**: insert the new query into the cache (may trigger eviction).
3. **Metrics**: Track cumulative hit rate, rolling hit rate, semantic
   coverage, and eviction overhead per policy.

Supports workload diversity (temporal, uniform, clustered, bursty) and
multi-seed evaluation for statistical rigor.

Usage
-----
Local smoke test (8.5K dev subset)::

    python scripts/08_run_eviction.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/eviction.yaml \\
        --output results/eviction/

Great Lakes (full dataset)::

    python scripts/08_run_eviction.py \\
        --embeddings results/embeddings/full_embeddings.npy \\
        --queries data/full_queries.parquet \\
        --config configs/eviction.yaml \\
        --output results/eviction/ \\
        --multi-seed
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.paths import embeddings_file, queries_file
from src.cache.semantic_cache import SemanticCache
from src.cache.eviction.fifo import FIFOPolicy
from src.cache.eviction.lru import LRUPolicy
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.semantic import SemanticPolicy
from src.cache.eviction.adaptive import AdaptiveHardSwitchPolicy, AdaptiveBlendPolicy
from src.cache.eviction.arc import ARCPolicy
from src.cache.eviction.gdsf import GDSFPolicy
from src.cache.eviction.siso import SISOPolicy
from src.cache.eviction.oracle import OraclePolicy
from src.benchmark.workload import generate_workload
from src.utils.env_check import require_supported_runtime, pin_numpy_threads
from src.utils.manifest import generate_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(
    embeddings_path: str, queries_path: str
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and query texts."""
    embeddings = np.load(embeddings_path).astype(np.float32)
    df = pd.read_parquet(queries_path)
    texts = df["query_text"].tolist()

    # Truncate to match (in case lengths differ)
    n = min(len(embeddings), len(texts))
    embeddings = embeddings[:n]
    texts = texts[:n]

    logger.info(f"Loaded {n} queries, embedding dim={embeddings.shape[1]}")
    return embeddings, texts


def create_policy(
    policy_name: str,
    config: dict,
    cache_embs: np.ndarray,
    cache_ids: list[int],
    stream_embs: np.ndarray,
    seed: int = 0,
    capacity: int | None = None,
) -> "EvictionPolicy":
    """Create an eviction policy instance from config.

    Args:
        policy_name: One of 'fifo', 'lru', 'lfu', 'semantic',
            'adaptive_hard', 'adaptive_blend', 'arc', 'gdsf', 'siso',
            'oracle'.
        config: Full experiment config dict.
        cache_embs: Initial cache embeddings (for oracle pre-computation).
        cache_ids: Initial cache IDs.
        stream_embs: Future query stream embeddings (for oracle).
        seed: Base seed forwarded to semantic policy's imputation RNG.
        capacity: Max cache entries (bounds ARC's ghost lists / target).
    """
    eviction_cfg = config.get("eviction", {})

    if policy_name == "fifo":
        return FIFOPolicy()
    elif policy_name == "lru":
        return LRUPolicy()
    elif policy_name == "lfu":
        return LFUPolicy()
    elif policy_name == "semantic":
        sem_cfg = eviction_cfg.get("semantic", {})
        return SemanticPolicy(
            similarity_threshold=sem_cfg.get("similarity_threshold", 0.30),
            alpha=sem_cfg.get("alpha", 1.0),
            beta=sem_cfg.get("beta", 1.0),
            recompute_interval=sem_cfg.get("recompute_interval", 50),
            mu=sem_cfg.get("mu", 0.1),
            dynamic_impute=sem_cfg.get("dynamic_impute", True),
            seed=seed,
        )
    elif policy_name == "adaptive_hard":
        adaptive_cfg = {
            **eviction_cfg.get("adaptive", {}),
            **eviction_cfg.get("adaptive_hard", {}),
        }
        return AdaptiveHardSwitchPolicy(
            similarity_threshold=adaptive_cfg.get(
                "similarity_threshold",
                eviction_cfg.get("semantic", {}).get("similarity_threshold", 0.90),
            ),
            recompute_interval=adaptive_cfg.get("recompute_interval", 2000),
            max_redundancy_samples=adaptive_cfg.get("max_redundancy_samples", 1024),
            density_history_size=adaptive_cfg.get("density_history_size", 64),
            density_threshold_scale=adaptive_cfg.get("density_threshold_scale", 1.0),
            density_floor=adaptive_cfg.get("density_floor", 1e-4),
            frequency_skew_threshold=adaptive_cfg.get("frequency_skew_threshold", 1.5),
            min_observations=adaptive_cfg.get("min_observations", 50),
            dynamic_impute=adaptive_cfg.get("dynamic_impute", True),
            seed=seed,
        )
    elif policy_name == "adaptive_blend":
        adaptive_cfg = {
            **eviction_cfg.get("adaptive", {}),
            **eviction_cfg.get("adaptive_blend", {}),
        }
        return AdaptiveBlendPolicy(
            similarity_threshold=adaptive_cfg.get(
                "similarity_threshold",
                eviction_cfg.get("semantic", {}).get("similarity_threshold", 0.90),
            ),
            recompute_interval=adaptive_cfg.get("recompute_interval", 2000),
            max_redundancy_samples=adaptive_cfg.get("max_redundancy_samples", 1024),
            density_history_size=adaptive_cfg.get("density_history_size", 64),
            density_threshold_scale=adaptive_cfg.get("density_threshold_scale", 1.0),
            density_floor=adaptive_cfg.get("density_floor", 1e-4),
            frequency_skew_threshold=adaptive_cfg.get("frequency_skew_threshold", 1.5),
            min_observations=adaptive_cfg.get("min_observations", 50),
            dynamic_impute=adaptive_cfg.get("dynamic_impute", True),
            seed=seed,
            base_recency_weight=adaptive_cfg.get("base_recency_weight", 1.0),
            max_frequency_weight=adaptive_cfg.get("max_frequency_weight", 1.0),
            max_semantic_weight=adaptive_cfg.get("max_semantic_weight", 2.0),
        )
    elif policy_name == "arc":
        arc_cfg = eviction_cfg.get("arc", {})
        return ARCPolicy(
            capacity=capacity,
            ghost_similarity_threshold=arc_cfg.get(
                "ghost_similarity_threshold", 0.20,
            ),
            seed=seed,
        )
    elif policy_name == "gdsf":
        return GDSFPolicy(seed=seed)
    elif policy_name == "siso":
        siso_cfg = eviction_cfg.get("siso", {})
        return SISOPolicy(
            theta_c=siso_cfg.get("theta_c", 0.86),
            theta_r=siso_cfg.get("theta_r", 0.86),
            theta_r_min=siso_cfg.get("theta_r_min", 0.60),
            theta_r_max=siso_cfg.get("theta_r_max", 0.98),
            theta_r_step=siso_cfg.get("theta_r_step", 0.02),
            decay_factor=siso_cfg.get("decay_factor", 1.1),
            maintenance_interval=siso_cfg.get("maintenance_interval", 1000),
            adjust_interval=siso_cfg.get("adjust_interval", 100),
            llm_latency=siso_cfg.get("llm_latency", 1.0),
            arrival_rate=siso_cfg.get("arrival_rate", 0.5),
            slo_latency=siso_cfg.get("slo_latency", 1.3),
            max_merge_samples=siso_cfg.get("max_merge_samples", 1024),
            seed=seed,
        )
    elif policy_name == "oracle":
        oracle_cfg = eviction_cfg.get("oracle", {})
        return OraclePolicy(
            future_stream_embeddings=stream_embs,
            cache_embeddings=cache_embs,
            cache_ids=cache_ids,
            similarity_threshold=config["evaluation"].get("hit_threshold", 0.90),
            refresh_interval=oracle_cfg.get("refresh_interval", 100),
            horizon=oracle_cfg.get("horizon", 0),
            use_gpu=oracle_cfg.get("use_gpu", False),
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def reorder_stream(
    stream_embs: np.ndarray,
    stream_texts: list[str],
    warmup_embs: np.ndarray,
    workload_type: str,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Reorder the query stream according to a workload pattern.

    For 'temporal', returns the stream as-is (chronological order).
    For 'uniform', 'clustered', 'bursty', uses generate_workload() to
    produce a realistic access pattern by sampling with replacement.

    Args:
        stream_embs: Stream embeddings, shape (S, D).
        stream_texts: Stream query texts.
        warmup_embs: Warmup embeddings (clustering context for workloads).
        workload_type: One of 'temporal', 'uniform', 'clustered', 'bursty'.
        seed: Random seed.

    Returns:
        Reordered (stream_embs, stream_texts).
    """
    if workload_type == "temporal":
        return stream_embs, stream_texts

    n_stream = len(stream_embs)
    indices = generate_workload(
        query_vectors=stream_embs,
        db_vectors=warmup_embs,
        workload_type=workload_type,
        n_queries=n_stream,
        seed=seed,
    )
    reordered_embs = stream_embs[indices]
    reordered_texts = [stream_texts[i] for i in indices]
    return reordered_embs, reordered_texts


# ═════════════════════════════════════════════════════════════════
# Checkpointing
# ═════════════════════════════════════════════════════════════════

def _ckpt_key(policy: str, cache_pct: float, workload: str, seed: int) -> str:
    """Unique identifier for a single evaluation run."""
    return f"{policy}|{cache_pct:.2f}|{workload}|{seed}"


def _save_checkpoint(all_results: dict, path: Path) -> None:
    """Atomically save partial results to disk."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    tmp.rename(path)
    logger.info(f"  ✓ Checkpoint saved → {path}")


def _load_checkpoint(path: Path) -> tuple[dict, set[str]]:
    """Load existing checkpoint and return (results_dict, completed_keys)."""
    if not path.exists():
        return {}, set()
    with open(path) as f:
        data = json.load(f)
    completed: set[str] = set()
    for policy_name, pct_dict in data.items():
        for pct_key, runs in pct_dict.items():
            for run in runs:
                key = _ckpt_key(
                    policy_name, float(pct_key),
                    run.get("workload", "temporal"),
                    run["seed"],
                )
                completed.add(key)
    logger.info(f"Checkpoint loaded: {len(completed)} completed runs from {path}")
    return data, completed


# ═════════════════════════════════════════════════════════════════
# Core evaluation
# ═════════════════════════════════════════════════════════════════

def _hits_path(
    hits_dir: Path,
    dataset: "str | None",
    policy_name: str,
    cache_size_pct: float,
    seed: int,
    workload_type: str,
) -> Path:
    """Canonical hits-log path for one run (shared by writer and skip check)."""
    ds_tag = dataset or "unknown"
    return Path(hits_dir) / (
        f"hits_{ds_tag}_{policy_name}_c0p{round(cache_size_pct * 100):02d}"
        f"_seed{seed}_{workload_type}.jsonl"
    )


def evaluate_policy(
    policy_name: str,
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    cache_size_pct: float,
    seed: int,
    workload_type: str = "temporal",
    log_hits: bool = False,
    hits_dir: "Path | None" = None,
    dataset: "str | None" = None,
    embedding_model: "str | None" = None,
) -> dict:
    """Run a single eviction evaluation for one policy + cache size.

    Returns:
        Dict with cumulative_hit_rate, rolling_hit_rates, semantic
        coverage, eviction stats, and per-phase timing.
    """
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    dim = embeddings.shape[1]

    # ── Data split ───────────────────────────────────────────────
    warmup_pct = config["evaluation"].get("warmup_pct", 0.30)
    n_warmup = int(n * warmup_pct)
    max_cache_size = int(n * cache_size_pct)

    # (P0.6) Enforce chronological/temporal splitting for realistic
    # evaluation of concept drift and caching.
    warmup_embs = embeddings[:n_warmup]
    warmup_texts = texts[:n_warmup]
    stream_embs = embeddings[n_warmup:]
    stream_texts = texts[n_warmup:]

    # If warmup is larger than max_cache_size, truncate warmup to fill cache
    if n_warmup > max_cache_size:
        warmup_embs = warmup_embs[:max_cache_size]
        warmup_texts = warmup_texts[:max_cache_size]
        n_warmup = max_cache_size

    # ── Apply workload reordering to stream ──────────────────────
    stream_embs, stream_texts = reorder_stream(
        stream_embs, stream_texts, warmup_embs, workload_type, seed,
    )
    n_stream = len(stream_embs)

    logger.info(
        f"  [{policy_name}] cache_size={max_cache_size} "
        f"({cache_size_pct:.0%}), warmup={n_warmup}, stream={n_stream}, "
        f"workload={workload_type}, seed={seed}"
    )

    # ── Create policy ────────────────────────────────────────────
    cache_ids = list(range(n_warmup))
    policy = create_policy(
        policy_name, config, warmup_embs, cache_ids, stream_embs, seed=seed,
        capacity=max_cache_size,
    )

    # ── Build cache ──────────────────────────────────────────────
    cache_cfg = config.get("cache", {})
    t_build_start = time.perf_counter()
    cache = SemanticCache(
        dim=dim,
        index_type=cache_cfg.get("index_type", "hnsw"),
        index_params=cache_cfg.get("index_params", {}),
        max_size=max_cache_size,
        eviction_policy=policy,
    )
    cache.build(warmup_embs, warmup_texts)
    build_time = time.perf_counter() - t_build_start

    # ── Replay query stream ──────────────────────────────────────
    eval_cfg = config.get("evaluation", {})
    hit_threshold = eval_cfg.get("hit_threshold", 0.90)
    rolling_window = eval_cfg.get("rolling_window", 1000)
    log_interval = eval_cfg.get("log_interval", 500)

    n_hits = 0
    rolling_hits = deque(maxlen=rolling_window)
    cumulative_hit_rates = []

    hit_records = None
    hits_path = None
    if log_hits and hits_dir is not None:
        hits_dir = Path(hits_dir)
        hits_dir.mkdir(parents=True, exist_ok=True)
        hits_path = _hits_path(
            hits_dir, dataset, policy_name, cache_size_pct, seed, workload_type,
        )
        hit_records = []

    t_stream_start = time.perf_counter()

    for i in range(n_stream):
        query_emb = stream_embs[i]
        query_text = stream_texts[i]

        # Advance oracle stream position
        if policy_name == "oracle":
            policy.advance_stream(i)

        # Lookup
        result = cache.lookup(query_emb, k=1, threshold=hit_threshold)

        if result.hit:
            n_hits += 1
            rolling_hits.append(1)
            if hit_records is not None:
                matched = result.cache_entry
                hit_records.append({
                    "dataset": dataset or "unknown",
                    "policy": policy_name,
                    "cache_size_pct": cache_size_pct,
                    "seed": seed,
                    "workload": workload_type,
                    "stream_idx": i,
                    "distance_l2sq": round(float(result.distance), 6),
                    "matched_cache_id": int(matched.cache_id) if matched else -1,
                    "orig_query": matched.query_text if matched else "",
                    "new_query": query_text,
                    "embedding_model": embedding_model or "unknown",
                })
        else:
            rolling_hits.append(0)
            # Insert on miss → may trigger eviction
            cache.insert(query_emb, query_text)

        # Record metrics periodically
        if (i + 1) % log_interval == 0 or i == n_stream - 1:
            cum_rate = n_hits / (i + 1)
            roll_rate = sum(rolling_hits) / len(rolling_hits) if rolling_hits else 0
            cumulative_hit_rates.append({
                "query_idx": i + 1,
                "cumulative_hit_rate": round(cum_rate, 6),
                "rolling_hit_rate": round(roll_rate, 6),
                "n_evictions": cache._n_evictions,
                "cache_size": cache.size,
            })

            if (i + 1) % log_interval == 0:
                logger.info(
                    f"    [{policy_name}] {i+1}/{n_stream} queries — "
                    f"cum_hit={cum_rate:.4f} roll_hit={roll_rate:.4f} "
                    f"evictions={cache._n_evictions} size={cache.size}"
                )

    stream_time = time.perf_counter() - t_stream_start

    # Serialize hits outside the timed loop; atomic rename so a killed run
    # never leaves a partial .jsonl for the sampler to pick up.
    if hit_records is not None:
        tmp_path = hits_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w") as f:
            for rec in hit_records:
                f.write(json.dumps(rec) + "\n")
            f.flush()
        os.replace(tmp_path, hits_path)

    # ── Compute semantic coverage ────────────────────────────────
    t_coverage_start = time.perf_counter()
    sample_size = min(2000, n_stream)
    sample_idx = rng.choice(n_stream, sample_size, replace=False)
    sample_embs = stream_embs[sample_idx]
    dists, _ = cache.batch_lookup(sample_embs, k=1)
    semantic_coverage = float(np.mean(dists[:, 0]))
    coverage_time = time.perf_counter() - t_coverage_start

    # ── Collect results ──────────────────────────────────────────
    final_hit_rate = n_hits / n_stream if n_stream > 0 else 0
    cache_stats = cache.stats

    result = {
        "policy": policy_name,
        "cache_size_pct": cache_size_pct,
        "max_cache_size": max_cache_size,
        "workload": workload_type,
        "seed": seed,
        "n_queries": n,
        "n_warmup": n_warmup,
        "n_stream": n_stream,
        "final_hit_rate": round(final_hit_rate, 6),
        "n_hits": n_hits,
        "n_misses": n_stream - n_hits,
        "semantic_coverage_avg_dist": round(semantic_coverage, 6),
        "cumulative_hit_rates": cumulative_hit_rates,
        "timing": {
            "build_time_s": round(build_time, 3),
            "stream_time_s": round(stream_time, 3),
            "coverage_time_s": round(coverage_time, 3),
            "avg_query_time_ms": round(
                stream_time / n_stream * 1000, 4
            ) if n_stream > 0 else 0,
        },
        "cache_stats": cache_stats,
    }

    logger.info(
        f"  [{policy_name}] DONE — hit_rate={final_hit_rate:.4f} "
        f"evictions={cache_stats['n_evictions']} "
        f"coverage={semantic_coverage:.4f} "
        f"time={stream_time:.1f}s"
    )

    return result


# ═════════════════════════════════════════════════════════════════
# Multi-seed runner
# ═════════════════════════════════════════════════════════════════

def run_full_experiment(
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    seeds: list[int],
    max_workers: int = 1,
    checkpoint_path: Path | None = None,
    log_hits: bool = False,
    hits_dir: Path | None = None,
    dataset: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Run all policy x cache_size x workload x seed combinations.

    All policies run sequentially to avoid pickling issues with
    ProcessPoolExecutor (nested functions / FAISS objects are not
    safely serializable across process boundaries).

    Supports checkpoint/resume: after each run completes, results are
    saved to ``checkpoint_path``.  On restart, completed runs are
    skipped automatically.

    Returns:
        Dict with nested results and aggregated statistics.
    """
    if max_workers > 1:
        logger.warning(
            f"--workers {max_workers} requested but parallel execution is "
            f"disabled (closure pickling incompatible with ProcessPoolExecutor). "
            f"All policies will run sequentially."
        )

    policies = config["eviction"]["policies"]
    cache_sizes = config["cache"]["cache_sizes_pct"]
    workloads = config["evaluation"].get("workloads", ["temporal"])

    # ── Load checkpoint (if available) ───────────────────────────
    if checkpoint_path is not None:
        all_results, completed_keys = _load_checkpoint(checkpoint_path)
    else:
        all_results = {}
        completed_keys = set()

    aggregated = {}
    total_runs = len(policies) * len(cache_sizes) * len(workloads) * len(seeds)
    run_num = 0
    n_skipped = 0
    run_times: list[float] = []

    # Prepare data structure (preserve any checkpoint data)
    for policy_name in policies:
        if policy_name not in all_results:
            all_results[policy_name] = {}
        aggregated[policy_name] = {}
        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"
            if pct_key not in all_results[policy_name]:
                all_results[policy_name][pct_key] = []

    # ── Run all policies sequentially ────────────────────────────
    logger.info(f"Running {total_runs} tasks sequentially...")
    if completed_keys:
        logger.info(f"  ({len(completed_keys)} already completed — will skip)")

    for policy_name in policies:
        for cache_pct in cache_sizes:
            for workload_type in workloads:
                for seed in seeds:
                    run_num += 1
                    key = _ckpt_key(policy_name, cache_pct, workload_type, seed)

                    if key in completed_keys:
                        hits_missing = False
                        if log_hits and hits_dir is not None:
                            expected_hits = _hits_path(
                                hits_dir, dataset, policy_name,
                                cache_pct, seed, workload_type,
                            )
                            hits_missing = not expected_hits.exists()
                        if hits_missing:
                            logger.info(
                                f"RERUN {run_num}/{total_runs} "
                                f"(checkpointed but hits log missing): "
                                f"{expected_hits.name}"
                            )
                            # Drop the stale checkpoint entry so the re-run
                            # replaces it instead of duplicating the seed.
                            runs = all_results[policy_name][f"{cache_pct:.2f}"]
                            runs[:] = [
                                r for r in runs
                                if not (
                                    r.get("workload", "temporal") == workload_type
                                    and r["seed"] == seed
                                )
                            ]
                        else:
                            n_skipped += 1
                            logger.info(
                                f"SKIP {run_num}/{total_runs} "
                                f"(checkpoint): {key}"
                            )
                            continue

                    logger.info(
                        f"RUN {run_num}/{total_runs}: "
                        f"policy={policy_name}, cache={cache_pct:.0%}, "
                        f"workload={workload_type}, seed={seed}"
                    )
                    t_run = time.perf_counter()
                    res = evaluate_policy(
                        policy_name, embeddings, texts, config,
                        cache_pct, seed, workload_type,
                        log_hits=log_hits, hits_dir=hits_dir, dataset=dataset,
                        embedding_model=embedding_model,
                    )
                    run_time = time.perf_counter() - t_run
                    run_times.append(run_time)

                    all_results[policy_name][f"{cache_pct:.2f}"].append(res)

                    # Progress estimate
                    remaining = total_runs - run_num
                    avg_t = sum(run_times) / len(run_times)
                    eta_h = remaining * avg_t / 3600
                    logger.info(
                        f"  ✓ Completed in {run_time/60:.1f} min "
                        f"| {remaining} runs left "
                        f"| ETA ~{eta_h:.1f}h"
                    )

                    # Save checkpoint after each run
                    if checkpoint_path is not None:
                        _save_checkpoint(all_results, checkpoint_path)

    # ── Aggregation ──────────────────────────────────────────────
    for policy_name in policies:
        for cache_pct in cache_sizes:
            pct_key = f"{cache_pct:.2f}"
            seed_results = all_results[policy_name][pct_key]

            # Sort by (workload, seed) for deterministic ordering
            seed_results.sort(key=lambda x: (x.get("workload", ""), x["seed"]))

            if not seed_results:
                continue

            hit_rates = [r["final_hit_rate"] for r in seed_results]
            coverages = [r["semantic_coverage_avg_dist"] for r in seed_results]
            stream_times = [r["timing"]["stream_time_s"] for r in seed_results]
            eviction_counts = [r["cache_stats"]["n_evictions"] for r in seed_results]

            aggregated[policy_name][pct_key] = {
                "hit_rate_mean": round(float(np.mean(hit_rates)), 6),
                "hit_rate_std": round(float(np.std(hit_rates)), 6),
                "coverage_mean": round(float(np.mean(coverages)), 6),
                "coverage_std": round(float(np.std(coverages)), 6),
                "stream_time_mean_s": round(float(np.mean(stream_times)), 3),
                "evictions_mean": round(float(np.mean(eviction_counts)), 1),
                "n_seeds": len(seeds),
                "seeds": seeds,
                "workloads": workloads,
            }

    # ── Oracle dominance check ───────────────────────────────────
    _check_oracle_dominance(aggregated, policies)

    return {
        "manifest": generate_manifest(config),
        "per_seed": all_results,
        "aggregated": aggregated,
        "config": {
            "policies": policies,
            "cache_sizes_pct": cache_sizes,
            "workloads": workloads,
            "seeds": seeds,
            "n_queries": len(embeddings),
        },
    }


def _check_oracle_dominance(aggregated: dict, policies: list[str]) -> None:
    """Warn if oracle hit rate is below any other policy (should never happen)."""
    if "oracle" not in policies:
        return

    oracle_agg = aggregated.get("oracle", {})
    for pct_key, oracle_stats in oracle_agg.items():
        oracle_hr = oracle_stats["hit_rate_mean"]
        for other in policies:
            if other == "oracle":
                continue
            other_hr = aggregated.get(other, {}).get(pct_key, {}).get("hit_rate_mean", 0)
            if oracle_hr < other_hr - 0.001:
                logger.warning(
                    f"ORACLE VIOLATION: oracle ({oracle_hr:.4f}) < "
                    f"{other} ({other_hr:.4f}) at cache={pct_key}. "
                    f"This may indicate a correctness issue."
                )


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Eviction policy evaluation"
    )
    parser.add_argument(
        "--embeddings", default=None,
        help="Path to embeddings .npy file. If omitted, resolved from "
             "--dataset/--embedding-model/--size (Option A convention).",
    )
    parser.add_argument(
        "--queries", default=None,
        help="Path to queries .parquet file. If omitted, resolved from "
             "--dataset/--size.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name (lmsys/moss/qqp) for path resolution when "
             "--embeddings/--queries are not given explicitly.",
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Embedding model tag for resolving the embeddings path.",
    )
    parser.add_argument(
        "--size", default="full",
        help="Scale subset to run (e.g. 10k, 100k, full). Default: full.",
    )
    parser.add_argument(
        "--config", default="configs/eviction.yaml",
        help="Path to eviction config YAML",
    )
    parser.add_argument(
        "--output", default="results/eviction/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Run with multiple seeds from config",
    )
    parser.add_argument(
        "--policies", nargs="+", default=None,
        help="Override policies to run (e.g. --policies lru semantic)",
    )
    parser.add_argument(
        "--cache-sizes", nargs="+", type=float, default=None,
        help="Override cache sizes (e.g. --cache-sizes 0.10 0.20)",
    )
    parser.add_argument(
        "--workloads", nargs="+", default=None,
        help="Override workloads (e.g. --workloads temporal clustered)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for multi-processing. Defaults to 1 (sequential).",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU-accelerated FAISS for oracle batch search (requires faiss-gpu).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoint and start from scratch.",
    )
    parser.add_argument(
        "--log-hits", action="store_true",
        help="Dump per-hit (orig_query, new_query, distance) JSONL for the LLM judge.",
    )
    parser.add_argument(
        "--hits-dir", default=None,
        help="Directory for --log-hits JSONL output (default: <output>/hits).",
    )
    return parser.parse_args()


def main():
    require_supported_runtime()
    pin_numpy_threads()

    args = parse_args()

    # Resolve data paths (Option A): explicit --embeddings/--queries win;
    # otherwise resolve from --dataset/--embedding-model/--size convention.
    emb_path, qry_path = args.embeddings, args.queries
    if emb_path is None or qry_path is None:
        if args.dataset is None:
            parser_error = (
                "Provide either --embeddings AND --queries, or --dataset "
                "(+ optional --embedding-model/--size) to resolve them."
            )
            raise SystemExit(parser_error)
        if emb_path is None:
            emb_path = str(embeddings_file(
                "results/embeddings", args.dataset, args.embedding_model, args.size,
            ))
        if qry_path is None:
            qry_path = str(queries_file(args.dataset, args.size))
        logger.info(
            f"Resolved paths for dataset={args.dataset} "
            f"model={args.embedding_model} size={args.size}:"
        )
        logger.info(f"  embeddings → {emb_path}")
        logger.info(f"  queries    → {qry_path}")

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.policies:
        config["eviction"]["policies"] = args.policies
    if args.cache_sizes:
        config["cache"]["cache_sizes_pct"] = args.cache_sizes
    if args.workloads:
        config["evaluation"]["workloads"] = args.workloads
    if args.gpu:
        config.setdefault("eviction", {}).setdefault("oracle", {})["use_gpu"] = True

    # Load data
    embeddings, texts = load_data(emb_path, qry_path)

    # Determine seeds
    if args.multi_seed:
        seeds = config.get("seeds", [42, 123, 456])
    else:
        seeds = [config.get("seed", 42)]

    # Checkpoint setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    if args.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Fresh start: removed existing checkpoint")

    hits_dir = Path(args.hits_dir) if args.hits_dir else (output_dir / "hits")
    if args.fresh and args.log_hits and hits_dir.exists():
        stale = sorted(hits_dir.glob("hits_*.jsonl")) + sorted(
            hits_dir.glob("*.jsonl.tmp")
        )
        for p in stale:
            p.unlink()
        logger.info(f"Fresh start: removed {len(stale)} stale hits file(s)")

    # Run experiment
    t_start = time.perf_counter()
    results = run_full_experiment(
        embeddings, texts, config, seeds,
        max_workers=args.workers,
        checkpoint_path=checkpoint_path,
        log_hits=args.log_hits,
        hits_dir=hits_dir if args.log_hits else None,
        dataset=args.dataset,
        embedding_model=args.embedding_model,
    )
    total_time = time.perf_counter() - t_start

    results["total_time_s"] = round(total_time, 1)

    # Save results
    suffix = "_multi_seed" if args.multi_seed else ""
    output_file = output_dir / f"eviction_results{suffix}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (experiment complete)")

    logger.info(f"\n{'═' * 60}")
    logger.info(f"RESULTS SAVED → {output_file}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"{'═' * 60}")

    # Print summary table
    print("\n\n" + "=" * 80)
    print("EVICTION POLICY COMPARISON")
    print("=" * 80)
    agg = results["aggregated"]
    print(f"{'Policy':<12} {'Cache %':<10} {'Hit Rate':<18} {'Coverage':<18} {'Evictions':<12}")
    print("-" * 80)
    for policy in config["eviction"]["policies"]:
        for pct in config["cache"]["cache_sizes_pct"]:
            pct_key = f"{pct:.2f}"
            if policy in agg and pct_key in agg[policy]:
                a = agg[policy][pct_key]
                hr = f"{a['hit_rate_mean']:.4f} ± {a['hit_rate_std']:.4f}"
                cov = f"{a['coverage_mean']:.4f} ± {a['coverage_std']:.4f}"
                ev = f"{a['evictions_mean']:.0f}"
                print(f"{policy:<12} {pct:<10.0%} {hr:<18} {cov:<18} {ev:<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
