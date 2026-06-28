#!/usr/bin/env python3
"""Phase 7 — stratified sampler over --log-hits JSONL for the LLM judge.

Reads hits_*.jsonl, groups by (dataset, policy), and draws up to N records per
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
    # Quantile edges → assign each record to a bin.
    edges = np.quantile(dists, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    bins: dict[int, list[int]] = defaultdict(list)
    for idx, d in enumerate(dists):
        b = int(np.searchsorted(edges, d, side="right") - 1)
        bins[max(0, min(b, n_bins - 1))].append(idx)
    per_bin = max(1, n // n_bins)
    chosen: list[int] = []
    for b in range(n_bins):
        pool = bins.get(b, [])
        if not pool:
            continue
        take = min(per_bin, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    # Top up to exactly n (or as close as the pool allows) from the remainder.
    if len(chosen) < n:
        remaining = sorted(set(range(len(records))) - set(chosen))
        if remaining:
            extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)),
                               replace=False).tolist()
            chosen.extend(extra)
    chosen = sorted(chosen)[:n]
    return [records[i] for i in chosen]


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
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            groups[(r["dataset"], r["policy"])].append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w") as fh:
        for (ds, pol), recs in sorted(groups.items()):
            sample = stratified_sample(recs, args.n, args.n_bins, args.seed)
            for r in sample:
                fh.write(json.dumps(r) + "\n")
            total += len(sample)
            logger.info(f"{ds}/{pol}: {len(sample)} sampled from {len(recs)} hits")
    logger.info(f"Wrote {total} sampled pairs → {out_path}")


if __name__ == "__main__":
    main()
