#!/usr/bin/env python3
"""Phase 9 (scaffold) — judged hit quality at the ROUTER-selected threshold.

Review-driven experiment (reviewer Major Concern #2 / fix-first #2): the paper
reports 60.33% latency savings at the adaptive router's operating point
(theta = 0.772 L2^2), but the LLM-judge audit was run at the *eviction* hit
threshold (0.90 L2^2). Because cosine similarity is shown to be an unreliable
quality proxy, a reviewer will ask whether the router's savings are
quality-preserving. This script audits judged hit quality at the router
threshold so the two can be compared on equal footing.

Mechanism (reuses the existing judge pipeline end to end):
  1. copy the base eviction config, set hit_threshold (and the semantic
     redundancy threshold) to --router-theta, write a temp config;
  2. run 08_run_eviction.py with --log-hits to emit hits_*.jsonl at that
     threshold (policy is immaterial to lookup quality; default LFU);
  3. run 16_sample_hits_for_judge.py to stratify-sample the logged hits;
  4. run 17_run_llm_judge.py to compute the quality-adjusted hit rate;
  5. print the router-threshold quality-adjusted rate next to the paper's
     0.90-threshold number so the delta is explicit.

STATUS: scaffold for the full pipeline (steps 1-4 wired to the real CLIs with
observed flags; judge credentials are read from .env by 17). Confirm the two
NOTE markers against your config schema before a live run.

A cheaper, already-runnable variant is --reanalyze-existing: because the
router threshold (0.772 L2^2) is TIGHTER than the audited eviction threshold
(0.90 L2^2), every hit the router would accept is already contained in the
phase 7 judged sample. Filtering those verdicts to distance <= theta gives the
judged-equivalence rate at the router's operating point with zero new judge
calls. Caveat: this conditions on the 0.90-run hit distribution instead of
re-simulating cache dynamics at 0.772 (a tighter threshold turns some hits
into misses, which are then inserted); the full pipeline above removes that
approximation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
S = REPO / "scripts"


def load_yaml(p: Path) -> dict:
    with open(p) as fh:
        return yaml.safe_load(fh)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval; same convention as scripts/18_visualize_judge.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def conditional_yes(records, theta: float) -> dict:
    """Judged-equivalence rate conditioned on hit distance, per dataset.

    Pairs are deduplicated across policies on (dataset, orig_query, new_query)
    and records with a null verdict are skipped, matching scripts/18. Buckets:
    'all' (every judged pair, i.e. the 0.90-threshold sample), 'at_or_below'
    (distance_l2sq <= theta, the hits the router operating point would keep),
    and 'above' (the rest).
    """
    pairs = {}
    for r in records:
        v = r.get("verdict")
        if v is None:
            continue
        key = (r["dataset"], r["orig_query"], r["new_query"])
        pairs.setdefault(key, (float(r["distance_l2sq"]), bool(v)))

    def stats(sel):
        n = len(sel)
        yes = sum(1 for _, v in sel if v)
        lo, hi = wilson_ci(yes, n)
        return {"n": n, "yes": yes,
                "mean_yes": (yes / n) if n else float("nan"),
                "ci_lo": lo, "ci_hi": hi}

    out = {}
    for ds in sorted({k[0] for k in pairs}):
        sub = [dv for k, dv in pairs.items() if k[0] == ds]
        out[ds] = {
            "all": stats(sub),
            "at_or_below": stats([x for x in sub if x[0] <= theta]),
            "above": stats([x for x in sub if x[0] > theta]),
        }
    return out


def reanalyze(verdicts_path: Path, theta: float) -> dict:
    records = []
    with open(verdicts_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    results = conditional_yes(records, theta)
    print(f"[21] distance-conditioned reanalysis of {verdicts_path} at theta={theta}")
    for ds, buckets in results.items():
        for name, s in buckets.items():
            print(f"    {ds:8s} {name:12s} n={s['n']:6d} yes={s['yes']:5d} "
                  f"mean_yes={s['mean_yes']:.4f} "
                  f"wilson95=({s['ci_lo']:.4f},{s['ci_hi']:.4f})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Judged hit quality at the router threshold (scaffold)")
    ap.add_argument("--router-theta", type=float, default=0.772,
                    help="router-selected L2^2 threshold to audit (paper: 0.772)")
    ap.add_argument("--base-config", default="configs/eviction.yaml")
    ap.add_argument("--dataset", default="lmsys")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--size", default="100k")
    ap.add_argument("--policies", default="lfu")
    ap.add_argument("--n", type=int, default=1000, help="hits to sample for judging")
    ap.add_argument("--max-calls", type=int, default=0, help="cap judge API calls (0 = no cap)")
    ap.add_argument("--output", default="results/judge_router")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reanalyze-existing", metavar="VERDICTS_JSONL",
                    help="skip the pipeline and recompute judged quality at "
                         "--router-theta by filtering an existing judged-verdicts "
                         "file (e.g. results/judge/phase7_minilm_c0p10_local/"
                         "verdicts.jsonl); conditions on the source run's hit "
                         "distribution instead of re-simulating cache dynamics")
    args = ap.parse_args()

    if args.reanalyze_existing:
        reanalyze(Path(args.reanalyze_existing), args.router_theta)
        return

    out = REPO / args.output
    (out / "hits").mkdir(parents=True, exist_ok=True)

    # --- Step 1: config variant at the router threshold ---------------------
    cfg = copy.deepcopy(load_yaml(REPO / args.base_config))
    cfg["hit_threshold"] = args.router_theta                      # NOTE 1: lookup threshold
    cfg.setdefault("policies", ["lfu"])
    cfg["policies"] = args.policies.split(",")
    # NOTE 2: keep the semantic redundancy threshold <= hit_threshold invariant.
    if "semantic" in cfg and cfg["semantic"].get("similarity_threshold", 0) > args.router_theta:
        cfg["semantic"]["similarity_threshold"] = args.router_theta
    cfg_path = out / "config_router_theta.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    steps = [
        # Step 2: run eviction with hit logging at the router threshold
        [sys.executable, str(S / "08_run_eviction.py"),
         "--config", str(cfg_path), "--dataset", args.dataset,
         "--embedding-model", args.embedding_model, "--size", args.size,
         "--output", str(out), "--log-hits", "--hits-dir", str(out / "hits"),
         "--multi-seed"],
        # Step 3: stratified sample of the logged hits
        [sys.executable, str(S / "16_sample_hits_for_judge.py"),
         "--hits-glob", str(out / "hits" / "hits_*.jsonl"),
         "--out", str(out / "sample.jsonl"), "--n", str(args.n)],
        # Step 4: run the query-equivalence judge -> quality-adjusted hit rate
        [sys.executable, str(S / "17_run_llm_judge.py"),
         "--sample", str(out / "sample.jsonl"), "--out-dir", str(out),
         "--raw-results-glob", str(out / "*.json"),
         "--raw-cache-key", "0.10", "--max-calls", str(args.max_calls)],
    ]

    print(f"[21] auditing judged hit quality at router theta = {args.router_theta}")
    for cmd in steps:
        print("   $", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if args.dry_run:
        return

    # --- Step 5: report ------------------------------------------------------
    verdict_files = sorted(out.glob("*judge*.json")) + sorted(out.glob("*quality*.json"))
    print("\n[21] Router-threshold judge outputs:", [f.name for f in verdict_files])
    print("[21] Compare the reported quality-adjusted hit rate here (theta=%.3f) against"
          % args.router_theta)
    print("     the paper's 0.90-threshold audit (LMSYS/QQP ~1-2%%, MOSS ~25%%).")


if __name__ == "__main__":
    main()
