#!/usr/bin/env python3
"""
Phase 2 — Workload density profiling.

Runs a narrow cache replay and logs density checkpoints without modifying the
validated eviction experiment harness. The output is designed for the Phase 2
sanity check and Phase 3 kill-gate preparation.
"""

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.workload import generate_workload
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.lru import LRUPolicy
from src.cache.semantic_cache import SemanticCache
from src.data.paths import embeddings_file, queries_file
from src.profiler.density import active_embedding_matrix, compute_density_snapshot
from src.utils.env_check import pin_numpy_threads, require_supported_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(embeddings_path: str, queries_path: str) -> tuple[np.ndarray, list[str]]:
    embeddings = np.load(embeddings_path).astype(np.float32)
    queries = pd.read_parquet(queries_path)["query_text"].tolist()
    n = min(len(embeddings), len(queries))
    return embeddings[:n], queries[:n]


def create_policy(policy_name: str):
    if policy_name == "lru":
        return LRUPolicy()
    if policy_name == "lfu":
        return LFUPolicy()
    raise ValueError("Density profile currently supports policies: lru, lfu")


def checkpoint_positions(n_stream: int, n_checkpoints: int) -> set[int]:
    if n_stream <= 0 or n_checkpoints <= 0:
        return set()
    if n_checkpoints >= n_stream:
        return set(range(1, n_stream + 1))

    raw = np.linspace(n_stream / n_checkpoints, n_stream, n_checkpoints)
    return {max(1, min(n_stream, int(round(x)))) for x in raw}


def reorder_stream(
    stream_embs: np.ndarray,
    stream_texts: list[str],
    warmup_embs: np.ndarray,
    workload_type: str,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    if workload_type == "temporal":
        return stream_embs, stream_texts

    indices = generate_workload(
        query_vectors=stream_embs,
        db_vectors=warmup_embs,
        workload_type=workload_type,
        n_queries=len(stream_embs),
        seed=seed,
    )
    return stream_embs[indices], [stream_texts[i] for i in indices]


def run_policy_density_profile(
    *,
    policy_name: str,
    embeddings: np.ndarray,
    texts: list[str],
    config: dict,
    cache_size_pct: float,
    workload_type: str,
    seed: int,
    density_theta: float,
    n_checkpoints: int,
    probe_size: int | None,
    anchor_size: int | None,
    exact_density: bool,
    metadata: dict,
) -> list[dict]:
    n = len(embeddings)
    dim = embeddings.shape[1]
    warmup_pct = config["evaluation"].get("warmup_pct", 0.30)
    n_warmup = int(n * warmup_pct)
    max_cache_size = int(n * cache_size_pct)

    warmup_embs = embeddings[:n_warmup]
    warmup_texts = texts[:n_warmup]
    stream_embs = embeddings[n_warmup:]
    stream_texts = texts[n_warmup:]

    if n_warmup > max_cache_size:
        warmup_embs = warmup_embs[:max_cache_size]
        warmup_texts = warmup_texts[:max_cache_size]
        n_warmup = max_cache_size

    stream_embs, stream_texts = reorder_stream(
        stream_embs, stream_texts, warmup_embs, workload_type, seed,
    )
    n_stream = len(stream_embs)
    checkpoints = checkpoint_positions(n_stream, n_checkpoints)

    cache_cfg = config.get("cache", {})
    cache = SemanticCache(
        dim=dim,
        index_type=cache_cfg.get("index_type", "hnsw"),
        index_params=cache_cfg.get("index_params", {}),
        max_size=max_cache_size,
        eviction_policy=create_policy(policy_name),
    )
    cache.build(warmup_embs, warmup_texts)

    eval_cfg = config.get("evaluation", {})
    hit_threshold = eval_cfg.get("hit_threshold", 0.90)
    rolling_window = eval_cfg.get("rolling_window", 1000)
    rolling_hits = deque(maxlen=rolling_window)
    n_hits = 0
    rows: list[dict] = []
    seed_offset = sum(ord(ch) for ch in policy_name) * 1_000_003

    for i, query_emb in enumerate(stream_embs):
        query_idx = i + 1
        result = cache.lookup(query_emb, k=1, threshold=hit_threshold)
        if result.hit:
            n_hits += 1
            rolling_hits.append(1)
        else:
            rolling_hits.append(0)
            cache.insert(query_emb, stream_texts[i])

        if query_idx not in checkpoints:
            continue

        active_ids, active_embs = active_embedding_matrix(cache)
        snapshot = compute_density_snapshot(
            active_embs,
            theta_l2sq=density_theta,
            probe_size=None if exact_density else probe_size,
            anchor_size=None if exact_density else anchor_size,
            seed=int(seed + seed_offset + query_idx),
        )
        cumulative_hit_rate = n_hits / query_idx
        rolling_hit_rate = sum(rolling_hits) / len(rolling_hits)

        rows.append({
            **metadata,
            "policy": policy_name,
            "cache_size_pct": cache_size_pct,
            "max_cache_size": max_cache_size,
            "workload": workload_type,
            "seed": seed,
            "query_idx": query_idx,
            "timestamp": query_idx,
            "hit_threshold": hit_threshold,
            "density_theta": density_theta,
            "mean_density": snapshot.mean_density,
            "cache_snapshot_size": snapshot.cache_snapshot_size,
            "density_probe_size": snapshot.density_probe_size,
            "density_anchor_size": snapshot.density_anchor_size,
            "density_sampled": snapshot.sampled,
            "n_hits": n_hits,
            "n_misses": query_idx - n_hits,
            "cumulative_hit_rate": cumulative_hit_rate,
            "rolling_hit_rate": rolling_hit_rate,
            "n_evictions": cache._n_evictions,
            "active_id_count": len(active_ids),
        })

    return rows


def build_gap_rows(density_rows: list[dict]) -> list[dict]:
    if not density_rows:
        return []

    df = pd.DataFrame(density_rows)
    if not {"lru", "lfu"}.issubset(set(df["policy"])):
        return []

    key_cols = [
        "dataset",
        "embedding_model",
        "size",
        "workload",
        "seed",
        "cache_size_pct",
        "max_cache_size",
        "density_theta",
        "query_idx",
    ]
    lru = df[df["policy"] == "lru"].set_index(key_cols)
    lfu = df[df["policy"] == "lfu"].set_index(key_cols)
    joined = lru.join(lfu, lsuffix="_lru", rsuffix="_lfu", how="inner").reset_index()

    rows = []
    for _, row in joined.iterrows():
        rows.append({
            "dataset": row["dataset"],
            "embedding_model": row["embedding_model"],
            "size": row["size"],
            "workload": row["workload"],
            "seed": int(row["seed"]),
            "cache_size_pct": float(row["cache_size_pct"]),
            "max_cache_size": int(row["max_cache_size"]),
            "query_idx": int(row["query_idx"]),
            "timestamp": int(row["query_idx"]),
            "density_theta": float(row["density_theta"]),
            "mean_density_reference": float(row["mean_density_lru"]),
            "mean_density_lru": float(row["mean_density_lru"]),
            "mean_density_lfu": float(row["mean_density_lfu"]),
            "hit_rate_lru": float(row["cumulative_hit_rate_lru"]),
            "hit_rate_lfu": float(row["cumulative_hit_rate_lfu"]),
            "hit_rate_gap_lfu_minus_lru": (
                float(row["cumulative_hit_rate_lfu"])
                - float(row["cumulative_hit_rate_lru"])
            ),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 density profile runner")
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--size", default="full")
    parser.add_argument("--config", default="configs/eviction.yaml")
    parser.add_argument("--output", default="results/density")
    parser.add_argument("--policies", nargs="+", default=["lru", "lfu"])
    parser.add_argument("--cache-size", type=float, default=0.10)
    parser.add_argument("--workload", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--density-theta", type=float, required=True)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--probe-size", type=int, default=512)
    parser.add_argument("--anchor-size", type=int, default=1024)
    parser.add_argument("--exact-density", action="store_true")
    return parser.parse_args()


def main():
    require_supported_runtime()
    pin_numpy_threads()
    args = parse_args()

    emb_path = args.embeddings
    qry_path = args.queries
    if emb_path is None or qry_path is None:
        if args.dataset is None:
            raise SystemExit(
                "Provide either --embeddings AND --queries, or --dataset "
                "(+ optional --embedding-model/--size)."
            )
        if emb_path is None:
            emb_path = str(embeddings_file(
                "results/embeddings", args.dataset, args.embedding_model, args.size,
            ))
        if qry_path is None:
            qry_path = str(queries_file(args.dataset, args.size))

    config = load_config(args.config)
    embeddings, texts = load_data(emb_path, qry_path)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    metadata = {
        "dataset": args.dataset or "custom",
        "embedding_model": args.embedding_model,
        "size": args.size,
        "embeddings_path": emb_path,
        "queries_path": qry_path,
    }

    logger.info(
        "Running density profile: dataset=%s model=%s size=%s theta=%.4f",
        metadata["dataset"],
        args.embedding_model,
        args.size,
        args.density_theta,
    )

    density_rows: list[dict] = []
    for policy_name in args.policies:
        density_rows.extend(run_policy_density_profile(
            policy_name=policy_name,
            embeddings=embeddings,
            texts=texts,
            config=config,
            cache_size_pct=args.cache_size,
            workload_type=args.workload,
            seed=seed,
            density_theta=args.density_theta,
            n_checkpoints=args.checkpoints,
            probe_size=args.probe_size,
            anchor_size=args.anchor_size,
            exact_density=args.exact_density,
            metadata=metadata,
        ))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    density_path = output_dir / "density_log.csv"
    gap_path = output_dir / "density_gap.csv"
    pd.DataFrame(density_rows).to_csv(density_path, index=False)
    pd.DataFrame(build_gap_rows(density_rows)).to_csv(gap_path, index=False)

    logger.info("Density log saved -> %s", density_path)
    logger.info("Density gap saved -> %s", gap_path)


if __name__ == "__main__":
    main()
