#!/usr/bin/env python3
"""Phase 3A adaptive eviction subset comparison.

Runs a small, configurable comparison before scaling adaptive eviction:

- lru
- lfu
- semantic
- adaptive_hard
- adaptive_blend

The script intentionally reuses ``scripts/08_run_eviction.py`` replay logic so
Phase 3A differs only in policy set and subset size.
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.paths import embeddings_file, queries_file
from src.utils.env_check import pin_numpy_threads, require_supported_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_POLICIES = ["lru", "lfu", "semantic", "adaptive_hard", "adaptive_blend"]


def load_eviction_runner():
    script_path = Path(__file__).resolve().parent / "08_run_eviction.py"
    spec = importlib.util.spec_from_file_location("clever_eviction_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import eviction runner from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3A adaptive subset comparison")
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--size", default="100k")
    parser.add_argument("--config", default="configs/eviction.yaml")
    parser.add_argument("--output", default="results/adaptive/phase3_subset")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--cache-size", type=float, default=0.10)
    parser.add_argument("--workload", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=20_000,
        help="Cap rows after loading data. Use 0 or negative to disable.",
    )
    return parser.parse_args()


def resolve_paths(args) -> tuple[str, str]:
    emb_path = args.embeddings
    qry_path = args.queries
    if emb_path is not None and qry_path is not None:
        return emb_path, qry_path

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
    return emb_path, qry_path


def flatten_result(result: dict, metadata: dict) -> dict:
    timing = result.get("timing", {})
    cache_stats = result.get("cache_stats", {})
    return {
        **metadata,
        "policy": result["policy"],
        "cache_size_pct": result["cache_size_pct"],
        "max_cache_size": result["max_cache_size"],
        "workload": result["workload"],
        "seed": result["seed"],
        "n_queries": result["n_queries"],
        "n_warmup": result["n_warmup"],
        "n_stream": result["n_stream"],
        "final_hit_rate": result["final_hit_rate"],
        "n_hits": result["n_hits"],
        "n_misses": result["n_misses"],
        "semantic_coverage_avg_dist": result["semantic_coverage_avg_dist"],
        "build_time_s": timing.get("build_time_s", 0.0),
        "stream_time_s": timing.get("stream_time_s", 0.0),
        "coverage_time_s": timing.get("coverage_time_s", 0.0),
        "avg_query_time_ms": timing.get("avg_query_time_ms", 0.0),
        "n_evictions": cache_stats.get("n_evictions", 0),
        "n_rebuilds": cache_stats.get("n_rebuilds", 0),
        "policy_stats_json": json.dumps(
            cache_stats.get("policy_stats", {}),
            sort_keys=True,
            default=str,
        ),
    }


def main():
    require_supported_runtime()
    pin_numpy_threads()
    args = parse_args()
    runner = load_eviction_runner()

    emb_path, qry_path = resolve_paths(args)
    config = runner.load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)

    embeddings, texts = runner.load_data(emb_path, qry_path)
    if args.max_queries > 0:
        n = min(args.max_queries, len(embeddings), len(texts))
        embeddings = embeddings[:n]
        texts = texts[:n]
        logger.info("Subset cap applied: %s queries", n)

    config.setdefault("cache", {})["cache_sizes_pct"] = [args.cache_size]
    config.setdefault("evaluation", {})["workloads"] = [args.workload]
    config.setdefault("eviction", {})["policies"] = args.policies

    metadata = {
        "dataset": args.dataset or "custom",
        "embedding_model": args.embedding_model,
        "size": args.size,
        "embeddings_path": emb_path,
        "queries_path": qry_path,
        "max_queries": args.max_queries,
    }

    rows = []
    for policy in args.policies:
        logger.info("Running Phase 3A subset policy=%s", policy)
        result = runner.evaluate_policy(
            policy,
            embeddings,
            texts,
            config,
            args.cache_size,
            seed,
            args.workload,
        )
        rows.append(flatten_result(result, metadata))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "adaptive_subset_results.csv"
    summary_path = output_dir / "adaptive_subset_summary.csv"

    results = pd.DataFrame(rows)
    summary_cols = [
        "dataset",
        "embedding_model",
        "size",
        "policy",
        "cache_size_pct",
        "max_cache_size",
        "workload",
        "seed",
        "n_queries",
        "n_stream",
        "final_hit_rate",
        "semantic_coverage_avg_dist",
        "avg_query_time_ms",
        "stream_time_s",
        "n_evictions",
        "policy_stats_json",
    ]

    results.to_csv(results_path, index=False)
    results[summary_cols].to_csv(summary_path, index=False)

    logger.info("Results saved -> %s", results_path)
    logger.info("Summary saved -> %s", summary_path)

    print("\nPHASE 3A ADAPTIVE SUBSET SUMMARY")
    print(results[[
        "policy",
        "final_hit_rate",
        "semantic_coverage_avg_dist",
        "avg_query_time_ms",
        "n_evictions",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
