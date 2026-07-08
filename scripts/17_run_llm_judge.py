#!/usr/bin/env python3
"""Phase 7 — run the independent query-equivalence LLM judge over a hit sample.

Provider-agnostic (OpenAI-compatible). Reads credentials from .env via
JudgeConfig.from_env(). Caches verdicts by (orig,new) pair — each verdict is
appended to disk the moment it arrives — so interrupted or quota-limited runs
resume without re-spending API calls. Computes quality-adjusted hit rate =
raw_hit_rate * mean(verdict == YES) per (dataset, policy).
"""
import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge.config import JudgeConfig
from src.judge.client import JudgeClient, DailyLimitError
from src.judge.rubric import build_messages, parse_verdict
from src.utils.eviction_results import load_aggregated_hit_rates

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Per-record run-config fields every scoring group must agree on.
REQUIRED_CONFIG_FIELDS = ("embedding_model", "cache_size_pct", "seed", "workload")


def pair_key(orig: str, new: str) -> str:
    h = hashlib.sha1()
    h.update(orig.encode("utf-8"))
    h.update(b"\x00")
    h.update(new.encode("utf-8"))
    return h.hexdigest()


def record_keys(records) -> list[str]:
    return [pair_key(r["orig_query"], r["new_query"]) for r in records]


def judge_samples(records, client, cache, max_workers: int, cache_path=None,
                  max_calls: int = 0, keys=None, stats=None) -> dict:
    """Return {pair_key: Optional[bool]} for every record, judging each unique
    (orig,new) pair at most once. `cache` is mutated in place; when `cache_path`
    is given every new verdict is also appended there the moment it arrives, so
    an interrupted run loses nothing. `max_calls` > 0 caps how many uncached
    pairs are submitted this run (the rest are left pending). `stats`, when
    provided, receives per-key statuses ("judged"/"unparsed"/"failed"/
    "pending"), aggregate counts, and a `daily_limit` flag set when the
    provider's daily quota ran out and the remaining pairs were left pending."""
    if keys is None:
        keys = record_keys(records)
    unique = {}
    for k, r in zip(keys, records):
        if k not in cache:
            unique[k] = (r["orig_query"], r["new_query"])

    status = {k: "judged" for k in keys if k in cache}
    submitted = list(unique.items())
    if max_calls > 0 and len(submitted) > max_calls:
        for k, _ in submitted[max_calls:]:
            status[k] = "pending"
        submitted = submitted[:max_calls]
    daily_limit = False

    def _judge_one(item):
        k, (orig, new) = item
        return parse_verdict(client.complete(build_messages(orig, new)))

    cache_fh = None
    if cache_path is not None and submitted:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_fh = open(cache_path, "a")
    try:
        if submitted:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_judge_one, item): item[0] for item in submitted}
                for fut in as_completed(futures):
                    k = futures[fut]
                    try:
                        verdict = fut.result()
                    except DailyLimitError as exc:
                        daily_limit = True
                        logger.warning(f"daily quota exhausted, leaving remaining "
                                       f"pairs for the next run: {exc}")
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as exc:  # noqa: BLE001 — one bad pair must not sink the run
                        status[k] = "failed"
                        logger.warning(f"pair {k} failed after retries: {exc}")
                        continue
                    if verdict is None:
                        status[k] = "unparsed"
                        continue
                    cache[k] = verdict
                    status[k] = "judged"
                    if cache_fh is not None:
                        cache_fh.write(json.dumps({"key": k, "verdict": verdict}) + "\n")
                        cache_fh.flush()
                for k in futures.values():
                    status.setdefault(k, "pending")
    finally:
        if cache_fh is not None:
            cache_fh.close()

    if stats is not None:
        stats["status"] = status
        stats["daily_limit"] = daily_limit
        for name in ("judged", "unparsed", "failed", "pending"):
            stats[f"n_{name}"] = sum(1 for s in status.values() if s == name)
    return {k: cache.get(k) for k in keys}


def quality_adjusted(records, verdicts, raw_hit_rates, keys=None, statuses=None) -> dict:
    """Per (dataset, policy): mean YES over parsed verdicts, quality-adjusted
    hit rate, and bookkeeping counts. Unresolved records (unparsed / failed /
    pending) never contribute to mean_yes; a group with nothing judged reports
    mean_yes = None rather than a fabricated 0."""
    if keys is None:
        keys = record_keys(records)
    statuses = statuses or {}
    groups = defaultdict(lambda: {"ys": [], "n_records": 0, "n_unparsed": 0,
                                  "n_failed": 0, "n_pending": 0})
    for k, r in zip(keys, records):
        g = groups[(r["dataset"], r["policy"])]
        g["n_records"] += 1
        v = verdicts.get(k)
        if v is not None:
            g["ys"].append(1 if v else 0)
        else:
            g[f"n_{statuses.get(k, 'pending')}"] += 1
    out = {}
    for gk, g in groups.items():
        ys = g["ys"]
        mean_yes = round(sum(ys) / len(ys), 6) if ys else None
        raw = raw_hit_rates.get(gk)
        out[gk] = {
            "n_records": g["n_records"],
            "n_judged": len(ys),
            "n_unparsed": g["n_unparsed"],
            "n_failed": g["n_failed"],
            "n_pending": g["n_pending"],
            "mean_yes": mean_yes,
            "raw_hit_rate": raw,
            "quality_adjusted_hit_rate": (
                round(raw * mean_yes, 6)
                if raw is not None and mean_yes is not None else None),
        }
    return out


def load_cache(path: Path) -> dict:
    """Last-wins over the append-only JSONL. Rows with a null verdict (from
    older runs) are skipped so those pairs get re-judged."""
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["verdict"] is None:
                continue
            cache[row["key"]] = row["verdict"]
    return cache


def validate_homogeneity(records, raw_cache_key: str) -> None:
    """Every (dataset, policy) scoring group must come from one run config
    (embedding model, cache size, seed, workload), and that cache size must
    equal the raw-results cache key — otherwise the quality adjustment would
    mix incomparable runs."""
    configs = defaultdict(set)
    for r in records:
        missing = [f for f in REQUIRED_CONFIG_FIELDS if f not in r]
        if missing:
            raise SystemExit(
                f"sample record for {r.get('dataset')}/{r.get('policy')} is "
                f"missing {missing}; regenerate the hit logs with the current "
                "harness (scripts/08_run_eviction.py --log-hits) and re-sample"
            )
        configs[(r["dataset"], r["policy"])].add(
            (r["embedding_model"], f"{float(r['cache_size_pct']):.2f}",
             str(r["seed"]), r["workload"]))
    expected_pct = f"{float(raw_cache_key):.2f}"
    for (ds, pol), cfgs in sorted(configs.items()):
        if len(cfgs) > 1:
            raise SystemExit(
                f"group {ds}/{pol} mixes run configs (embedding_model, "
                f"cache_size_pct, seed, workload): {sorted(cfgs)}")
        ((_, pct, _, _),) = cfgs
        if pct != expected_pct:
            raise SystemExit(
                f"group {ds}/{pol} was sampled at cache_size_pct {pct} but "
                f"--raw-cache-key is {raw_cache_key}; raw and judged hit "
                "rates must come from the same cache size")


def main():
    ap = argparse.ArgumentParser(description="Run the query-equivalence LLM judge")
    ap.add_argument("--sample", default="results/judge/sample.jsonl")
    ap.add_argument("--out-dir", default="results/judge")
    ap.add_argument("--raw-results-glob",
                    default="results/eviction/phase6_matrix_*_minilm_100k_c0p10/eviction_results_multi_seed.json")
    ap.add_argument("--raw-cache-key", default="0.10")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="Cap on uncached judge API calls this run (0 = unlimited).")
    args = ap.parse_args()

    load_dotenv()  # read .env
    cfg = JudgeConfig.from_env()
    client = JudgeClient(cfg)
    logger.info(f"Judge model={cfg.model} base_url={cfg.base_url} concurrency={cfg.max_concurrency}")

    records = [json.loads(l) for l in Path(args.sample).read_text().splitlines()]
    validate_homogeneity(records, args.raw_cache_key)
    keys = record_keys(records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "verdict_cache.jsonl"
    cache = load_cache(cache_path)
    logger.info(f"Loaded {len(cache)} cached verdicts; {len(records)} sample rows")

    stats = {}
    try:
        verdicts = judge_samples(records, client, cache, cfg.max_concurrency,
                                 cache_path=cache_path, max_calls=args.max_calls,
                                 keys=keys, stats=stats)
    except DailyLimitError as exc:
        # judge_samples handles the in-flight case itself; this guards a
        # DailyLimitError surfacing outside the worker pool.
        stats.setdefault("status", {})
        stats["daily_limit"] = True
        logger.warning(f"daily quota exhausted: {exc}")
        verdicts = {k: cache.get(k) for k in keys}

    with open(out_dir / "verdicts.jsonl", "w") as fh:
        for k, r in zip(keys, records):
            fh.write(json.dumps({**r, "verdict": verdicts.get(k)}) + "\n")

    raw = load_aggregated_hit_rates(args.raw_results_glob, args.raw_cache_key)
    scores = quality_adjusted(records, verdicts, raw, keys=keys,
                              statuses=stats.get("status"))
    scores_out = {f"{ds}/{pol}": v for (ds, pol), v in sorted(scores.items())}
    (out_dir / "judge_scores.json").write_text(json.dumps(scores_out, indent=2))
    logger.info(
        f"Wrote judge_scores.json with {len(scores_out)} (dataset,policy) cells "
        f"(judged={stats.get('n_judged', 0)} unparsed={stats.get('n_unparsed', 0)} "
        f"failed={stats.get('n_failed', 0)} pending={stats.get('n_pending', 0)})")
    if stats.get("daily_limit"):
        logger.warning(
            f"stopped early: provider daily quota exhausted. Verdicts so far "
            f"are saved in {cache_path}; re-run this script after the quota "
            "resets to resume from the cache.")


if __name__ == "__main__":
    main()
