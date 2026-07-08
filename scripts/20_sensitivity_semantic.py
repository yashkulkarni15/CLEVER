#!/usr/bin/env python3
"""Phase 9 (scaffold) — sensitivity sweep for the semantic eviction policy.

Review-driven experiment (reviewer fix-first #4): quantify how the semantic
policy's hit rate responds to its hyperparameters — the smoothing constant
``mu``, the semantic neighbour / redundancy threshold ``theta_e``
(``semantic.similarity_threshold``), and the lookup ``hit_threshold`` — rather
than arguing degradation-to-LFU only analytically (paper Sec. "Necessity of
mu-Smoothing").

Mechanism: this is a thin, config-driven wrapper over the *existing* pipeline.
For each grid point it (1) copies the base eviction config, (2) overrides the
swept key, (3) writes a temp config, (4) runs ``08_run_eviction.py`` on the
chosen cell, and (5) reads the resulting ``eviction_results_multi_seed.json``
to record the semantic-vs-LFU hit-rate gap. It adds no new cache logic, so its
numbers are directly comparable to the paper's matrix.

Grids live in configs/sensitivity_semantic.yaml.

STATUS: scaffold. The subprocess plumbing and result parsing are complete and
match the observed CLI/artifact schema; run with --dry-run first to inspect the
planned invocations, then drop --dry-run to execute on HPC. Verify the two
NOTE markers below against your local config schema before a full run.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EVICTION = REPO / "scripts" / "08_run_eviction.py"


def load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def set_nested(cfg: dict, dotted_key: str, value) -> None:
    """Set cfg['a']['b'] = value for dotted_key 'a.b', creating dicts as needed."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


# NOTE 1: maps a swept grid name -> the dotted config path 08 reads it from.
# Confirm these paths against configs/eviction.yaml (they match it as of writing).
KEY_PATHS = {
    "mu": "semantic.mu",
    "similarity_threshold": "semantic.similarity_threshold",
    "hit_threshold": "hit_threshold",
}


def grid_points(spec: dict) -> list[dict]:
    grids = spec["grids"]
    base_vals = {k: v[len(v) // 2] for k, v in grids.items()}  # mid value = base
    if spec.get("mode") == "full_grid":
        keys = list(grids)
        return [dict(zip(keys, combo)) for combo in itertools.product(*grids.values())]
    # coordinate sweep: vary one key at a time, others at base
    points = []
    for k, vals in grids.items():
        for v in vals:
            pt = dict(base_vals)
            pt[k] = v
            points.append(pt)
    # de-dup (the all-base point repeats once per key)
    seen, uniq = set(), []
    for pt in points:
        key = tuple(sorted(pt.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(pt)
    return uniq


def run_cell(point: dict, spec: dict, base_cfg: dict, out_root: Path, dry: bool) -> dict:
    sweep = spec["sweep"]
    # enforce the config invariant theta_e <= hit_threshold
    if point["similarity_threshold"] > point["hit_threshold"]:
        return {"point": point, "skipped": "similarity_threshold > hit_threshold"}

    cfg = copy.deepcopy(base_cfg)
    for gk, val in point.items():
        set_nested(cfg, KEY_PATHS[gk], val)
    # NOTE 2: restrict policies/capacity/seeds to the sweep cell to save compute.
    set_nested(cfg, "policies", sweep["policies"])
    set_nested(cfg, "cache_sizes_pct", sweep["cache_sizes_pct"])
    set_nested(cfg, "seeds", sweep["seeds"])

    tag = "_".join(f"{k}{point[k]}" for k in sorted(point))
    cell_dir = out_root / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cell_dir / "config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    cmd = [
        sys.executable, str(EVICTION),
        "--config", str(cfg_path),
        "--dataset", sweep["dataset"],
        "--embedding-model", sweep["embedding_model"],
        "--size", sweep["size"],
        "--output", str(cell_dir),
        "--multi-seed",
    ]
    if dry:
        return {"point": point, "cmd": " ".join(cmd), "dry_run": True}

    subprocess.run(cmd, check=True)
    res = json.loads((cell_dir / "eviction_results_multi_seed.json").read_text())
    agg = res["aggregated"]
    frac = str(sweep["cache_sizes_pct"][0])
    frac = frac if frac in agg["lfu"] else list(agg["lfu"])[0]
    lfu = agg["lfu"][frac]["hit_rate_mean"]
    sem = agg["semantic"][frac]["hit_rate_mean"]
    return {
        "point": point,
        "lfu_hit_rate": lfu,
        "semantic_hit_rate": sem,
        "semantic_minus_lfu_pp": round((sem - lfu) * 100, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic-policy sensitivity sweep (scaffold)")
    ap.add_argument("--sensitivity-config", default="configs/sensitivity_semantic.yaml")
    ap.add_argument("--output", default="results/sensitivity")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned 08_run_eviction.py invocations without running them")
    args = ap.parse_args()

    spec = load_yaml(REPO / args.sensitivity_config)
    base_cfg = load_yaml(REPO / spec["base_config"])
    out_root = REPO / args.output
    out_root.mkdir(parents=True, exist_ok=True)

    points = grid_points(spec)
    print(f"[20] {len(points)} grid points ({spec.get('mode','coordinate')} sweep) "
          f"on {spec['sweep']['dataset']}/{spec['sweep']['embedding_model']}")
    results = [run_cell(pt, spec, base_cfg, out_root, args.dry_run) for pt in points]

    summary = out_root / "semantic_sensitivity.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"[20] wrote {summary}")
    for r in results:
        if "semantic_minus_lfu_pp" in r:
            print(f"    {r['point']} -> semantic-LFU = {r['semantic_minus_lfu_pp']:+.4f} pp")


if __name__ == "__main__":
    main()
