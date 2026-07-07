#!/usr/bin/env python3
"""Phase 7 — stratified sampler over --log-hits JSONL for the LLM judge.

Reads hits_*.jsonl, groups by the full run config (dataset, policy,
embedding_model, cache_size_pct, seed, workload), and draws up to N records per
group spread across quantile bins of L2² distance so borderline hits (near the
hit threshold) are represented, not just easy near-duplicates.
"""
import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def stratified_sample(records: list[dict], n: int, n_bins: int, seed: int) -> list[dict]:
    if len(records) <= n:
        return list(records)
    rng = np.random.RandomState(seed)
    dists = np.array([r["distance_l2sq"] for r in records], dtype=float)
    # Quantile edges → assign each record to a bin. Heavy ties produce duplicate
    # quantile edges; dedupe so tied mass forms one stratum instead of emptying
    # the others and degrading the stratification into uniform top-up.
    edges = np.quantile(dists, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    edges = np.unique(edges)
    bin_ids = np.clip(np.searchsorted(edges, dists, side="right") - 1, 0, len(edges) - 2)
    non_empty = np.unique(bin_ids)
    if n < len(non_empty):
        # Fewer draws than populated strata: the rng picks which strata
        # contribute one record each, instead of drawing from all of them
        # and truncating (which would bias toward low-index records).
        non_empty = rng.choice(non_empty, size=n, replace=False)
    per_bin = max(1, n // len(non_empty))
    chosen: list[int] = []
    for b in non_empty:
        pool = np.nonzero(bin_ids == b)[0]
        take = min(per_bin, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    # Top up to exactly n (or as close as the pool allows) from the remainder.
    if len(chosen) < n:
        remaining = sorted(set(range(len(records))) - set(chosen))
        if remaining:
            extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)),
                               replace=False).tolist()
            chosen.extend(extra)
    return [records[i] for i in sorted(chosen)]


def main():
    ap = argparse.ArgumentParser(description="Stratified hit sampler for the LLM judge")
    ap.add_argument("--hits-glob", default="results/eviction/**/hits/hits_*.jsonl",
                    help="Glob for hits JSONL (recursive).")
    ap.add_argument("--out", default="results/judge/sample.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = glob.glob(args.hits_glob, recursive=True)
    if not files:
        raise SystemExit(f"no hits files matched: {args.hits_glob}")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    total_skipped = 0
    n_duplicates = 0
    for f in files:
        skipped = 0
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            key = (r["dataset"], r["policy"], r.get("embedding_model", "unknown"),
                   f"{float(r['cache_size_pct']):.2f}", str(r["seed"]), r["workload"])
            # stream_idx pins record identity within a run config: a query slot
            # can hit at most once per run, so a repeat means the same log was
            # matched twice (re-run output dir, copied results tree) and would
            # double-weight the pair in mean_yes.
            identity = key + (r["stream_idx"],)
            if identity in seen:
                n_duplicates += 1
                continue
            seen.add(identity)
            groups[key].append(r)
        if skipped:
            logger.warning(f"{f}: skipped {skipped} malformed line(s)")
        total_skipped += skipped
    if total_skipped:
        logger.warning(f"skipped {total_skipped} malformed line(s) across {len(files)} file(s)")
    if n_duplicates:
        logger.warning(f"dropped {n_duplicates} duplicate record(s) across {len(files)} file(s)")
    if not groups:
        raise SystemExit("no valid hit records found in matched files")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w") as fh:
        for key, recs in sorted(groups.items()):
            sample = stratified_sample(recs, args.n, args.n_bins, args.seed)
            for r in sample:
                fh.write(json.dumps(r) + "\n")
            total += len(sample)
            logger.info(f"{'/'.join(key)}: {len(sample)} sampled from {len(recs)} hits")
    logger.info(f"Wrote {total} sampled pairs → {out_path}")


if __name__ == "__main__":
    main()
