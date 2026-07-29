#!/usr/bin/env python3
"""
Phase 6 — Full-matrix paper figures (cross-encoder eviction comparison).

Consumes the completed Phase 6 matrix:

  MiniLM  : results/eviction/phase6_matrix_{ds}_minilm_100k_c0p{10,20,30}/
            eviction_results_multi_seed.json
  gte-base: results/eviction/phase6_recal_{ds}_gte_100k_c0p{10,20,30}/
            eviction_results_multi_seed.json   (recalibrated; the original
            phase6_matrix_*_gte_* cells are degenerate and are NOT read here)

Generates two figures:

  Figure C — Cache-size ablation grid (the negative-result money figure)
      2 rows (encoder: MiniLM, gte-base) x 3 cols (dataset: LMSYS, QQP, MOSS).
      Each panel: hit rate vs cache size {10,20,30}% with one line per policy.
      Story: LFU/ARC lead at tight budgets and the lead shrinks as cache grows;
      SISO is worst on the sparse datasets; MOSS is a saturated control. The
      same ordering holds under both encoders. Per-panel y-scales because
      absolute hit rates do NOT transfer across encoders (per-encoder
      hit-threshold calibration) — only the relative ordering does.
      Output: results/figures/phase6_cache_size_ablation.{png,pdf}

  Figure D — Final policy heatmap
      Rows = policies, columns = dataset x cache size, one heatmap per encoder
      (stacked). Cell = mean hit rate (%). Annotated. Sequential colormap is
      normalised per dataset-column-block so within-dataset policy differences
      stay visible despite MOSS saturating near 99%.
      Output: results/figures/phase6_policy_heatmap.{png,pdf}

Usage::

    python scripts/15_visualize_phase6_matrix.py \\
        --eviction-dir results/eviction \\
        --output-dir results/figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Canonical orderings ──────────────────────────────────────────
DATASET_ORDER = ["lmsys", "qqp", "moss"]
DATASET_LABELS = {"lmsys": "LMSYS", "qqp": "QQP", "moss": "MOSS"}

CACHE_SIZES = [10, 20, 30]  # percent

# (encoder key, dir prefix, embedding tag, display label)
ENCODERS = [
    ("minilm", "phase6_matrix", "minilm", "MiniLM (384d)"),
    ("gte", "phase6_recal", "gte", "gte-base (768d)"),
]

# Policies whose runs live in their own result dirs rather than in the
# Phase 6 cells, keyed by the dir prefix that carries them. FIFO was added
# after the matrix had already been run, so re-running the other six purely
# to co-locate it would waste allocation. These are merged into the matching
# matrix cell at load time and are optional: the figures still render before
# the overlay job lands.
OVERLAY_PREFIXES = ["phase10_fifo"]

POLICY_ORDER = ["fifo", "lru", "lfu", "semantic", "arc", "gdsf", "siso"]
POLICY_LABELS = {
    "fifo": "FIFO", "lru": "LRU", "lfu": "LFU", "semantic": "Semantic",
    "arc": "ARC", "gdsf": "GDSF", "siso": "SISO",
}
# Okabe-Ito colorblind-safe palette (7 distinct). FIFO takes black as the
# weakest-baseline control; Okabe-Ito's remaining hue (#F0E442 yellow) is
# too light to read on white.
POLICY_COLORS = {
    "fifo": "#000000",      # black
    "lru": "#0072B2",       # blue
    "lfu": "#E69F00",       # orange
    "semantic": "#009E73",  # green
    "arc": "#CC79A7",       # reddish purple
    "gdsf": "#56B4E9",      # sky blue
    "siso": "#D55E00",      # vermillion
}
POLICY_MARKERS = {
    "fifo": "P", "lru": "o", "lfu": "s", "semantic": "^",
    "arc": "D", "gdsf": "v", "siso": "X",
}


def setup_style():
    """Configure matplotlib for paper-quality figures (8-10pt fonts)."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_fig(fig, output_dir: Path, name: str):
    out_path = output_dir / f"{name}.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path} (+ .pdf)")


# ═════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════

def cell_path(eviction_dir: Path, prefix: str, ds: str, tag: str, cache: int) -> Path:
    return (eviction_dir / f"{prefix}_{ds}_{tag}_100k_c0p{cache}"
            / "eviction_results_multi_seed.json")


def _merge_cell(cell: dict, path: Path, cache: int):
    """Merge one results file's policies into a cell, without overwriting.

    A policy already present from an earlier (higher-priority) source wins,
    so an overlay can only add policies the primary run did not carry.
    """
    data = json.loads(path.read_text())
    agg = data["aggregated"]
    for policy in POLICY_ORDER:
        if cache in cell[policy]:
            continue
        pol_block = agg.get(policy, {})
        if not pol_block:
            continue
        # Robust: take the single cache key actually present per policy.
        only_key = next(iter(pol_block))
        v = pol_block[only_key]
        cell[policy][cache] = (v["hit_rate_mean"], v["hit_rate_std"])


def load_matrix(eviction_dir: Path) -> dict:
    """Load the full Phase 6 matrix into nested dicts.

    Returns matrix[encoder][dataset][policy][cache_pct] = (mean, std), a
    `missing` list of required cell paths that were absent, and an
    `overlay_missing` list of optional overlay paths (see OVERLAY_PREFIXES)
    that were absent. Overlay absence is reported rather than swallowed so a
    partially covered matrix is never mistaken for a complete one.
    """
    matrix: dict = {}
    missing: list[Path] = []
    overlay_missing: list[Path] = []
    for enc_key, prefix, tag, _label in ENCODERS:
        matrix[enc_key] = {}
        for ds in DATASET_ORDER:
            cell = {p: {} for p in POLICY_ORDER}
            matrix[enc_key][ds] = cell
            for cache in CACHE_SIZES:
                path = cell_path(eviction_dir, prefix, ds, tag, cache)
                if path.exists():
                    _merge_cell(cell, path, cache)
                else:
                    missing.append(path)
                for overlay in OVERLAY_PREFIXES:
                    opath = cell_path(eviction_dir, overlay, ds, tag, cache)
                    if opath.exists():
                        _merge_cell(cell, opath, cache)
                    else:
                        overlay_missing.append(opath)
    return {
        "matrix": matrix,
        "missing": missing,
        "overlay_missing": overlay_missing,
    }


def present_policies(matrix: dict) -> list[str]:
    """POLICY_ORDER restricted to policies with data in at least one cell.

    Keeps a policy whose overlay job has not landed yet out of the figures
    entirely, rather than drawing it as an empty row.
    """
    return [
        policy for policy in POLICY_ORDER
        if any(
            matrix[enc_key][ds][policy]
            for enc_key, _prefix, _tag, _label in ENCODERS
            for ds in DATASET_ORDER
        )
    ]


# ═════════════════════════════════════════════════════════════════
# Figure C — cache-size ablation grid
# ═════════════════════════════════════════════════════════════════

def plot_cache_size_ablation(matrix: dict, output_dir: Path):
    n_rows, n_cols = len(ENCODERS), len(DATASET_ORDER)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(9.6, 5.4), squeeze=False,
    )

    for r, (enc_key, _prefix, _tag, enc_label) in enumerate(ENCODERS):
        for c, ds in enumerate(DATASET_ORDER):
            ax = axes[r][c]
            for policy in POLICY_ORDER:
                pts = matrix[enc_key][ds][policy]
                xs = [cs for cs in CACHE_SIZES if cs in pts]
                if not xs:
                    continue
                ys = [pts[cs][0] * 100 for cs in xs]
                es = [pts[cs][1] * 100 for cs in xs]
                ax.errorbar(
                    xs, ys, yerr=es,
                    marker=POLICY_MARKERS[policy], markersize=5,
                    color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
                    linewidth=1.3, capsize=2, elinewidth=0.8,
                )
            ax.set_xticks(CACHE_SIZES)
            ax.set_xticklabels([f"{cs}%" for cs in CACHE_SIZES])
            if r == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Cache size (% of stream)")
            if c == 0:
                ax.set_ylabel(f"{enc_label}\nHit rate", fontsize=9)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels) or 6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02), columnspacing=1.4, handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_fig(fig, output_dir, "phase6_cache_size_ablation")


# ═════════════════════════════════════════════════════════════════
# Figure D — final policy heatmap (per encoder)
# ═════════════════════════════════════════════════════════════════

def plot_policy_heatmap(matrix: dict, output_dir: Path):
    policies = present_policies(matrix)
    col_labels = [
        f"{DATASET_LABELS[ds]}\n{cs}%"
        for ds in DATASET_ORDER for cs in CACHE_SIZES
    ]
    col_keys = [(ds, cs) for ds in DATASET_ORDER for cs in CACHE_SIZES]

    fig, axes = plt.subplots(
        len(ENCODERS), 1, figsize=(8.2, 6.2), squeeze=False,
    )

    for r, (enc_key, _prefix, _tag, enc_label) in enumerate(ENCODERS):
        ax = axes[r][0]
        grid = np.full((len(policies), len(col_keys)), np.nan)
        for i, policy in enumerate(policies):
            for j, (ds, cs) in enumerate(col_keys):
                pt = matrix[enc_key][ds][policy].get(cs)
                if pt is not None:
                    grid[i, j] = pt[0] * 100

        # Normalise colour PER dataset-block (3 cols each) so within-dataset
        # policy differences stay visible despite MOSS saturating near 99%.
        norm_grid = np.full_like(grid, np.nan)
        for b in range(len(DATASET_ORDER)):
            block = grid[:, b * 3:(b + 1) * 3]
            lo, hi = np.nanmin(block), np.nanmax(block)
            rng = hi - lo if hi > lo else 1.0
            norm_grid[:, b * 3:(b + 1) * 3] = (block - lo) / rng

        ax.imshow(norm_grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_keys)))
        ax.set_xticklabels(col_labels, fontsize=7)
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels([POLICY_LABELS[p] for p in policies])
        ax.set_title(f"{enc_label}", loc="left", fontsize=9, fontweight="bold")
        ax.grid(False)

        # Annotate each cell with the absolute hit-rate %.
        for i in range(len(policies)):
            for j in range(len(col_keys)):
                if np.isnan(grid[i, j]):
                    continue
                txt_color = "white" if norm_grid[i, j] > 0.55 else "black"
                ax.text(
                    j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color,
                )
        # Block separators between datasets.
        for b in range(1, len(DATASET_ORDER)):
            ax.axvline(b * 3 - 0.5, color="white", linewidth=2)

    fig.tight_layout()
    _save_fig(fig, output_dir, "phase6_policy_heatmap")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Phase 6 full-matrix paper figures "
                    "(cross-encoder cache-size ablation + policy heatmap)"
    )
    parser.add_argument(
        "--eviction-dir", default="results/eviction",
        help="Directory containing phase6_matrix_* / phase6_recal_* run dirs",
    )
    parser.add_argument(
        "--output-dir", default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figures", choices=["c", "d", "all"], default="all",
        help="Which figure(s) to generate (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_matrix(Path(args.eviction_dir))
    if loaded["missing"]:
        logger.error("missing input: the following Phase 6 cells were not found:")
        for p in loaded["missing"]:
            logger.error(f"  missing input: {p}")
        sys.exit(1)

    # Overlay cells are optional, but their absence must be stated: a figure
    # silently drawn without FIFO reads as "FIFO was covered" when it wasn't.
    if loaded["overlay_missing"]:
        omitted = [p for p in POLICY_ORDER
                   if p not in present_policies(loaded["matrix"])]
        logger.warning(
            f"{len(loaded['overlay_missing'])} overlay cell(s) absent; "
            f"policies omitted from figures: {omitted or 'none'}"
        )
        for p in loaded["overlay_missing"]:
            logger.warning(f"  absent overlay cell: {p}")
    matrix = loaded["matrix"]

    if args.figures in ("c", "all"):
        logger.info("Figure C: cache-size ablation grid")
        plot_cache_size_ablation(matrix, output_dir)

    if args.figures in ("d", "all"):
        logger.info("Figure D: final policy heatmap")
        plot_policy_heatmap(matrix, output_dir)


if __name__ == "__main__":
    main()
