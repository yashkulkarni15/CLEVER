#!/usr/bin/env python3
"""
Paper figures for index benchmarking and adaptive routing.

Replaces the hand-assembled dashboard images with paper-styled charts
generated directly from the result artifacts:

  Figure A — hnsw_pareto_tradeoff.{png,pdf}
      Recall@1 vs P50 search latency (log scale) for every index
      configuration in the 499K-vector uniform-workload benchmark,
      with the Pareto frontier overlaid.
      Source: results/benchmarks/index_benchmark_500k.json

  Figure B — router_threshold_sweep.{png,pdf}
      Mean five-seed threshold sweep under random cache fill: hit rate
      and latency savings (top panel) and accepted-hit cosine quality
      (bottom panel) vs the routing threshold, with the adaptive
      operating point marked.
      Source: results/routing/routing_eval_multi_seed.json

Usage::

    python scripts/19_paper_index_routing_figures.py \\
        --benchmark results/benchmarks/index_benchmark_500k.json \\
        --routing results/routing/routing_eval_multi_seed.json \\
        --output-dir results/figures
"""

import argparse
import json
import logging
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

INDEX_ORDER = ["flat", "hnsw", "ivf", "lsh"]
INDEX_LABELS = {"flat": "Flat (exact)", "hnsw": "HNSW", "ivf": "IVF", "lsh": "LSH"}
# Okabe-Ito colorblind-safe hues (same instances as the other paper figures).
INDEX_COLORS = {
    "flat": "#555555",
    "hnsw": "#0072B2",
    "ivf": "#E69F00",
    "lsh": "#D55E00",
}
INDEX_MARKERS = {"flat": "*", "hnsw": "o", "ivf": "s", "lsh": "^"}


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
# Data shaping (unit-tested in tests/test_paper_bench_figures_data.py)
# ═════════════════════════════════════════════════════════════════

def pareto_points(runs, workload: str = "uniform") -> dict:
    """Group benchmark runs of one workload into {index_type: [(p50, recall)]}."""
    pts: dict[str, list] = {}
    for r in runs:
        if r.get("workload") != workload:
            continue
        pts.setdefault(r["index_type"], []).append(
            (r["search_latency_ms"]["p50"], r["recall_at_1"])
        )
    return pts


def pareto_frontier(points) -> list:
    """Non-dominated (latency, recall) staircase, sorted by latency."""
    frontier = []
    best_recall = -1.0
    for lat, rec in sorted(points):
        if rec > best_recall:
            frontier.append((lat, rec))
            best_recall = rec
    return frontier


def mean_sweep(per_seed: dict, fill: str = "random") -> dict:
    """Average the threshold sweep across seeds.

    All seeds must share the same threshold grid; a mismatch means the
    sweeps are not comparable and is an error.
    """
    sweeps = [per_seed[s][fill]["threshold_sweep"] for s in sorted(per_seed)]
    grids = [[p["threshold"] for p in sw] for sw in sweeps]
    if any(g != grids[0] for g in grids[1:]):
        raise ValueError(f"threshold grids differ across seeds: {grids}")
    curve = {"threshold": grids[0], "hit_rate": [], "quality": [],
             "latency_saving_pct": []}
    for i in range(len(grids[0])):
        curve["hit_rate"].append(
            float(np.mean([sw[i]["hit_rate"] for sw in sweeps])))
        curve["quality"].append(
            float(np.mean([sw[i]["retrieval_similarity"] for sw in sweeps])))
        curve["latency_saving_pct"].append(
            float(np.mean([sw[i]["latency_saving_pct"] for sw in sweeps])))
    return curve


# ═════════════════════════════════════════════════════════════════
# Figure A — index pareto tradeoff
# ═════════════════════════════════════════════════════════════════

def plot_index_pareto(pts: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    all_points = [p for v in pts.values() for p in v]
    frontier = pareto_frontier(all_points)
    ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
            linestyle="--", color="#999999", linewidth=1.2, zorder=1,
            label="Pareto frontier")

    for itype in INDEX_ORDER:
        if itype not in pts:
            continue
        lats = [p[0] for p in pts[itype]]
        recs = [p[1] for p in pts[itype]]
        ax.scatter(lats, recs, s=48 if itype == "flat" else 34,
                   color=INDEX_COLORS[itype], marker=INDEX_MARKERS[itype],
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label=INDEX_LABELS[itype])

    # Anchor annotations: the exact baseline and the chosen HNSW config.
    flat_lat, flat_rec = max(pts["flat"])
    ax.annotate("Flat: 17.4 ms", (flat_lat, flat_rec),
                textcoords="offset points", xytext=(-8, -14),
                ha="right", fontsize=8, color="#555555")
    hnsw_best = max(pts["hnsw"], key=lambda p: (p[1], -p[0]))
    ax.annotate("HNSW: 0.989 at 0.52 ms", hnsw_best,
                xytext=(1.6, 0.92), textcoords="data",
                ha="left", fontsize=8, color=INDEX_COLORS["hnsw"],
                arrowprops={"arrowstyle": "-", "color": INDEX_COLORS["hnsw"],
                            "linewidth": 0.7, "shrinkB": 4})

    ax.set_xscale("log")
    ax.set_xlabel("P50 search latency (ms, log scale)")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0.55, 1.02)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, output_dir, "hnsw_pareto_tradeoff")


# ═════════════════════════════════════════════════════════════════
# Figure B — routing threshold sweep
# ═════════════════════════════════════════════════════════════════

def plot_threshold_sweep(curve: dict, adaptive_mean: float,
                         adaptive_std: float, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.4, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    th = curve["threshold"]
    ax1.plot(th, [100 * h for h in curve["hit_rate"]], marker="o",
             markersize=4, color="#0072B2", linewidth=1.5, label="Hit rate")
    ax1.plot(th, curve["latency_saving_pct"], marker="s", markersize=4,
             color="#E69F00", linewidth=1.5, linestyle="--",
             label="Latency savings")
    ax1.set_ylabel("Percent")
    ax1.set_ylim(0, 104)
    ax1.legend(frameon=False, loc="lower right")

    ax2.plot(th, curve["quality"], marker="^", markersize=4,
             color="#009E73", linewidth=1.5, label="Accepted-hit quality")
    ax2.set_ylabel("Mean cosine\nof accepted hits")
    ax2.set_xlabel(r"Routing threshold $\theta$ ($L_2^2$ distance)")
    ax2.set_ylim(0.6, 1.0)

    for ax in (ax1, ax2):
        ax.axvspan(adaptive_mean - adaptive_std, adaptive_mean + adaptive_std,
                   color="#0072B2", alpha=0.08, linewidth=0)
        ax.axvline(adaptive_mean, color="#555555", linewidth=1,
                   linestyle=":")
    ax1.text(adaptive_mean + 0.015, 8,
             rf"adaptive $\theta = {adaptive_mean:.3f}$",
             fontsize=8, color="#555555")

    fig.tight_layout()
    _save_fig(fig, output_dir, "router_threshold_sweep")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-styled index-benchmark and "
                    "routing-sweep figures"
    )
    parser.add_argument(
        "--benchmark", default="results/benchmarks/index_benchmark_500k.json",
        help="499K-vector index benchmark JSON",
    )
    parser.add_argument(
        "--routing", default="results/routing/routing_eval_multi_seed.json",
        help="Multi-seed routing evaluation JSON",
    )
    parser.add_argument(
        "--workload", default="uniform",
        help="Benchmark workload to plot (default: uniform)",
    )
    parser.add_argument(
        "--output-dir", default="results/figures",
        help="Output directory for figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads(Path(args.benchmark).read_text())
    pts = pareto_points(bench["runs"], workload=args.workload)
    flat_p50 = pts["flat"][0][0]
    best = max(pts["hnsw"], key=lambda p: (p[1], -p[0]))
    logger.info(f"flat p50 {flat_p50:.2f} ms; best HNSW recall {best[1]:.3f} "
                f"at {best[0]:.2f} ms ({flat_p50 / best[0]:.0f}x)")
    plot_index_pareto(pts, output_dir)

    routing = json.loads(Path(args.routing).read_text())
    curve = mean_sweep(routing["per_seed"], fill="random")
    agg = routing["aggregated"]["random"]["best_threshold"]
    plot_threshold_sweep(curve, agg["mean"], agg["std"], output_dir)


if __name__ == "__main__":
    main()
