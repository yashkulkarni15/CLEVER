#!/usr/bin/env python3
"""Phase 7 — run the independent query-equivalence LLM judge over a hit sample.

Provider-agnostic (OpenAI-compatible). Reads credentials from .env via
JudgeConfig.from_env(). Caches verdicts by (orig,new) pair so converged policies
and repeated queries do not re-spend API calls. Computes quality-adjusted hit
rate = raw_hit_rate * mean(verdict == YES) per (dataset, policy).
"""
import argparse
import hashlib
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.judge.config import JudgeConfig
from src.judge.client import JudgeClient
from src.judge.rubric import build_messages, parse_verdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def pair_key(orig: str, new: str) -> str:
    h = hashlib.sha1()
    h.update(orig.encode("utf-8"))
    h.update(b"\x00")
    h.update(new.encode("utf-8"))
    return h.hexdigest()


def judge_samples(records, client, cache, max_workers: int) -> dict:
    """Return {pair_key: Optional[bool]} for every record, judging each unique
    (orig,new) pair at most once. `cache` is mutated in place (resumable)."""
    unique = {}
    for r in records:
        k = pair_key(r["orig_query"], r["new_query"])
        if k not in cache and k not in unique:
            unique[k] = (r["orig_query"], r["new_query"])

    def _run(item):
        k, (orig, new) = item
        text = client.complete(build_messages(orig, new))
        return k, parse_verdict(text)

    if unique:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for k, verdict in ex.map(_run, list(unique.items())):
                cache[k] = verdict

    return {pair_key(r["orig_query"], r["new_query"]): cache[
        pair_key(r["orig_query"], r["new_query"])] for r in records}


def quality_adjusted(records, verdicts, raw_hit_rates) -> dict:
    by_group = defaultdict(list)
    for r in records:
        k = pair_key(r["orig_query"], r["new_query"])
        v = verdicts.get(k)
        if v is not None:
            by_group[(r["dataset"], r["policy"])].append(1 if v else 0)
    out = {}
    for g, ys in by_group.items():
        mean_yes = sum(ys) / len(ys) if ys else 0.0
        raw = raw_hit_rates.get(g)
        out[g] = {
            "n": len(ys),
            "mean_yes": round(mean_yes, 6),
            "raw_hit_rate": raw,
            "quality_adjusted_hit_rate": round(raw * mean_yes, 6) if raw is not None else None,
        }
    return out


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["verdict"]
    return cache


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for k, v in cache.items():
            fh.write(json.dumps({"key": k, "verdict": v}) + "\n")


def load_raw_hit_rates(glob_pat: str, cache_key: str) -> dict:
    """Map (dataset, policy) → aggregated hit_rate_mean from eviction result JSONs.
    Expects dir names like phase6_matrix_<ds>_minilm_100k_c0p10."""
    import glob as _glob
    out = {}
    for f in _glob.glob(glob_pat):
        ds = Path(f).parent.name.split("_")[2]  # phase6_matrix_<ds>_...
        data = json.loads(Path(f).read_text())
        for pol, block in data.get("aggregated", {}).items():
            key = cache_key if cache_key in block else (next(iter(block)) if block else None)
            if key:
                out[(ds, pol)] = block[key]["hit_rate_mean"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the query-equivalence LLM judge")
    ap.add_argument("--sample", default="results/judge/sample.jsonl")
    ap.add_argument("--out-dir", default="results/judge")
    ap.add_argument("--raw-results-glob",
                    default="results/eviction/phase6_matrix_*_minilm_100k_c0p10/eviction_results_multi_seed.json")
    ap.add_argument("--raw-cache-key", default="0.10")
    args = ap.parse_args()

    load_dotenv()  # read .env
    cfg = JudgeConfig.from_env()
    client = JudgeClient(cfg)
    logger.info(f"Judge model={cfg.model} base_url={cfg.base_url} concurrency={cfg.max_concurrency}")

    records = [json.loads(l) for l in Path(args.sample).read_text().splitlines()]
    out_dir = Path(args.out_dir)
    cache_path = out_dir / "verdict_cache.jsonl"
    cache = load_cache(cache_path)
    logger.info(f"Loaded {len(cache)} cached verdicts; {len(records)} sample rows")

    verdicts = judge_samples(records, client, cache, cfg.max_concurrency)
    save_cache(cache_path, cache)

    with open(out_dir / "verdicts.jsonl", "w") as fh:
        for r in records:
            k = pair_key(r["orig_query"], r["new_query"])
            fh.write(json.dumps({**r, "verdict": cache.get(k)}) + "\n")

    raw = load_raw_hit_rates(args.raw_results_glob, args.raw_cache_key)
    scores = quality_adjusted(records, verdicts, raw)
    scores_out = {f"{ds}/{pol}": v for (ds, pol), v in sorted(scores.items())}
    (out_dir / "judge_scores.json").write_text(json.dumps(scores_out, indent=2))
    logger.info(f"Wrote judge_scores.json with {len(scores_out)} (dataset,policy) cells")


if __name__ == "__main__":
    main()
