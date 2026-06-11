#!/usr/bin/env python3
"""
Phase 2/3 — Paper figure generation.

Generates the two figures justified by current results:

  Figure A — Adaptive dry-run comparison (Phase 3A subset runs)
      Two panels: final hit rate and average per-query time per policy,
      grouped by dataset (LMSYS, QQP, MOSS).
      Inputs : results/adaptive/phase3_subset_*/adaptive_subset_summary.csv
      Output : results/figures/adaptive_dry_run_comparison.{png,pdf}

  Figure B — Density characterization (Phase 2 profiling)
      Dataset-level cache density at L2^2 theta 0.30 / 0.90 paired with
      the LFU-LRU hit-rate gap.
      Inputs : results/density/phase2_{ds}_minilm_100k_theta_{0p30,0p90}/
               {density_log.csv,density_gap.csv}
      Output : results/figures/density_characterization.{png,pdf}
      If the inputs are missing locally (they may exist only on GLC), the
      script reports exactly which files are needed and exits nonzero.

Usage::

    python scripts/14_visualize_phase_figures.py \\
        --adaptive-dir results/adaptive \\
        --density-dir results/density \\
        --output-dir results/figures
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Canonical orderings ──────────────────────────────────────────
DATASET_ORDER = ["lmsys", "qqp", "moss"]
DATASET_LABELS = {"lmsys": "LMSYS", "qqp": "QQP", "moss": "MOSS"}

POLICY_ORDER = ["lru", "lfu", "semantic", "adaptive_hard", "adaptive_blend"]
POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "semantic": "Semantic",
    "adaptive_hard": "Adaptive (hard)",
    "adaptive_blend": "Adaptive (blend)",
}

# Okabe-Ito colorblind-safe palette.
POLICY_COLORS = {
    "lru": "#0072B2",
    "lfu": "#E69F00",
    "semantic": "#009E73",
    "adaptive_hard": "#CC79A7",
    "adaptive_blend": "#D55E00",
}

DENSITY_THETAS = (0.30, 0.90)
THETA_COLORS = {0.30: "#56B4E9", 0.90: "#0072B2"}


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
    """Save figure as PNG and PDF."""
    out_path = output_dir / f"{name}.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved: {out_path} (+ .pdf)")


# ═════════════════════════════════════════════════════════════════
# Figure A — data shaping
# ═════════════════════════════════════════════════════════════════

ADAPTIVE_REQUIRED_COLUMNS = [
    "dataset", "policy", "final_hit_rate", "avg_query_time_ms",
]


def find_adaptive_summary_files(adaptive_dir: Path) -> list[Path]:
    """Locate adaptive_subset_summary.csv files under phase3 run directories."""
    adaptive_dir = Path(adaptive_dir)
    return sorted(adaptive_dir.glob("phase3_subset_*/adaptive_subset_summary.csv"))


def tidy_adaptive_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce concatenated adaptive_subset_summary rows to a tidy plotting frame.

    Returns one row per (dataset, policy) with columns
    [dataset, policy, final_hit_rate, avg_query_time_ms], ordered
    dataset-major (DATASET_ORDER) and policy-minor (POLICY_ORDER).

    Raises:
        ValueError: if required columns are absent, or any present dataset
            is missing one of the expected policies.
    """
    missing_cols = [c for c in ADAPTIVE_REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(
            f"Adaptive summary is missing required columns: {missing_cols}"
        )

    df = raw[ADAPTIVE_REQUIRED_COLUMNS].copy()
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]

    # Completeness check: every dataset must have every expected policy.
    for ds in datasets:
        have = set(df.loc[df["dataset"] == ds, "policy"])
        absent = [p for p in POLICY_ORDER if p not in have]
        if absent:
            raise ValueError(
                f"Dataset '{ds}' is missing policies: {absent}"
            )

    df = df[df["policy"].isin(POLICY_ORDER)]
    df["dataset"] = pd.Categorical(df["dataset"], categories=datasets, ordered=True)
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER, ordered=True)
    df = df.sort_values(["dataset", "policy"]).reset_index(drop=True)
    df["dataset"] = df["dataset"].astype(str)
    df["policy"] = df["policy"].astype(str)
    return df


def load_adaptive_data(adaptive_dir: Path) -> pd.DataFrame:
    """Load and tidy all adaptive subset summaries; exits if none found."""
    files = find_adaptive_summary_files(adaptive_dir)
    if not files:
        logger.error(
            "missing input: no adaptive_subset_summary.csv found under "
            f"{adaptive_dir}/phase3_subset_*/"
        )
        sys.exit(1)
    for f in files:
        logger.info(f"Loading: {f}")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return tidy_adaptive_summary(raw)


# ═════════════════════════════════════════════════════════════════
# Figure A — plotting
# ═════════════════════════════════════════════════════════════════

def _grouped_bars(ax, tidy: pd.DataFrame, value_col: str, label_fmt: str):
    """Draw policy-grouped bars per dataset; returns dataset tick positions."""
    datasets = [d for d in DATASET_ORDER if d in set(tidy["dataset"])]
    n_policies = len(POLICY_ORDER)
    bar_width = 0.8 / n_policies
    x = np.arange(len(datasets))

    for i, policy in enumerate(POLICY_ORDER):
        vals = [
            tidy[(tidy["dataset"] == ds) & (tidy["policy"] == policy)][value_col].iloc[0]
            for ds in datasets
        ]
        bars = ax.bar(
            x + (i - (n_policies - 1) / 2) * bar_width,
            vals, bar_width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
            edgecolor="white", linewidth=0.4,
        )
        ax.bar_label(bars, fmt=label_fmt, fontsize=6, padding=1, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    return x, datasets


def plot_adaptive_dry_run(tidy: pd.DataFrame, output_dir: Path):
    """Figure A: final hit rate (left) and avg per-query time (right, log)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # Panel 1: final hit rate.
    x, datasets = _grouped_bars(ax1, tidy, "final_hit_rate", "%.3f")
    ax1.set_ylabel("Final hit rate")
    ax1.set_ylim(0, 1.12)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_title("(a) Hit rate", loc="left", fontsize=9)

    # Mark MOSS saturation: all five policies tie.
    if "moss" in datasets:
        moss_x = x[datasets.index("moss")]
        moss_max = tidy[tidy["dataset"] == "moss"]["final_hit_rate"].max()
        # Anchor the leader at the left edge of the MOSS group so it does
        # not cross the bars.
        ax1.annotate(
            "saturated\n(all tie)",
            xy=(moss_x - 0.40, moss_max), xytext=(moss_x - 0.72, moss_max - 0.24),
            fontsize=7, style="italic", color="#444444", ha="center",
            arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.7),
        )

    # Panel 2: avg per-query time. Spread is >10x, so log scale.
    _grouped_bars(ax2, tidy, "avg_query_time_ms", "%.2f")
    ax2.set_yscale("log")
    ax2.set_ylabel("Avg. query time (ms, log)")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 2.2)
    ax2.yaxis.set_major_locator(ticker.FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.set_title("(b) Overhead", loc="left", fontsize=9)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.04), columnspacing=1.2, handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_fig(fig, output_dir, "adaptive_dry_run_comparison")


# ═════════════════════════════════════════════════════════════════
# Figure B — data shaping
# ═════════════════════════════════════════════════════════════════

DENSITY_GAP_REQUIRED_COLUMNS = [
    "dataset", "density_theta", "query_idx",
    "mean_density_reference", "hit_rate_gap_lfu_minus_lru",
]


def expected_density_paths(
    density_dir: Path,
    datasets: tuple = tuple(DATASET_ORDER),
    thetas: tuple = DENSITY_THETAS,
) -> list[Path]:
    """Expected Phase 2 density profile outputs (see slurm/phase2_density.sbatch)."""
    density_dir = Path(density_dir)
    paths = []
    for ds in datasets:
        for theta in thetas:
            tag = f"{theta:.2f}".replace(".", "p")
            run_dir = density_dir / f"phase2_{ds}_minilm_100k_theta_{tag}"
            paths.append(run_dir / "density_log.csv")
            paths.append(run_dir / "density_gap.csv")
    return paths


def find_density_inputs(density_dir: Path) -> tuple[list[Path], list[Path]]:
    """Split expected density inputs into (found, missing)."""
    found, missing = [], []
    for path in expected_density_paths(density_dir):
        (found if path.exists() else missing).append(path)
    return found, missing


def tidy_density_gap(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce density_gap rows to one final-checkpoint row per (dataset, theta).

    Returns columns [dataset, density_theta, mean_density, gap_pp] where
    mean_density is the reference (LRU) cache density at the last checkpoint
    and gap_pp is the LFU-LRU hit-rate gap in percentage points.

    Raises:
        ValueError: if required columns are absent.
    """
    missing_cols = [
        c for c in DENSITY_GAP_REQUIRED_COLUMNS if c not in raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"density_gap data is missing required columns: {missing_cols}"
        )

    rows = []
    for (ds, theta), group in raw.groupby(["dataset", "density_theta"]):
        final = group.loc[group["query_idx"].idxmax()]
        rows.append({
            "dataset": ds,
            "density_theta": float(theta),
            "mean_density": float(final["mean_density_reference"]),
            "gap_pp": float(final["hit_rate_gap_lfu_minus_lru"]) * 100.0,
        })

    tidy = pd.DataFrame(
        rows, columns=["dataset", "density_theta", "mean_density", "gap_pp"]
    )
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    tidy["_ord"] = tidy["dataset"].map(lambda d: order.get(d, len(order)))
    tidy = (
        tidy.sort_values(["_ord", "density_theta"])
        .drop(columns="_ord")
        .reset_index(drop=True)
    )
    return tidy


def load_density_data(density_dir: Path) -> pd.DataFrame | None:
    """Load density gap CSVs; returns None (after logging) if inputs missing."""
    found, missing = find_density_inputs(density_dir)
    if missing:
        logger.error("missing input: the following Phase 2 density files were "
                     "not found locally (fetch from GLC):")
        for p in missing:
            logger.error(f"  missing input: {p}")
        return None

    gap_files = [p for p in found if p.name == "density_gap.csv"]
    raw = pd.concat([pd.read_csv(f) for f in gap_files], ignore_index=True)
    return tidy_density_gap(raw)


# ═════════════════════════════════════════════════════════════════
# Figure B — plotting
# ═════════════════════════════════════════════════════════════════

def plot_density_characterization(tidy: pd.DataFrame, output_dir: Path):
    """Figure B: per-dataset density at theta 0.30/0.90 vs LFU-LRU gap."""
    datasets = [d for d in DATASET_ORDER if d in set(tidy["dataset"])]
    x = np.arange(len(datasets))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(4.2, 2.9))

    for i, theta in enumerate(DENSITY_THETAS):
        vals = [
            tidy[(tidy["dataset"] == ds) & (tidy["density_theta"] == theta)][
                "mean_density"
            ].iloc[0]
            for ds in datasets
        ]
        ax.bar(
            x + (i - 0.5) * bar_width, vals, bar_width,
            color=THETA_COLORS[theta],
            label=rf"density ($\theta$={theta:.2f})",
            edgecolor="white", linewidth=0.4,
        )

    ax.set_yscale("log")
    ax.set_ylabel(r"Mean cache density (L2$^2$, log)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])

    # Overlay LFU-LRU gap on a twin axis (theta-independent; take theta=0.30).
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines.top.set_visible(False)
    gaps = [
        tidy[(tidy["dataset"] == ds) & (tidy["density_theta"] == DENSITY_THETAS[0])][
            "gap_pp"
        ].iloc[0]
        for ds in datasets
    ]
    ax2.plot(
        x, gaps, "D--", color="#D55E00", markersize=6, linewidth=1.2,
        label="LFU−LRU gap",
    )
    for xi, gap, ds in zip(x, gaps, datasets):
        note = f"{gap:+.2f}pp"
        if ds == "moss" and abs(gap) < 0.1:
            note += " (saturated)"
        ax2.annotate(
            note, (xi, gap), textcoords="offset points", xytext=(0, 7),
            fontsize=7, ha="center", color="#D55E00",
        )
    ax2.set_ylabel("LFU−LRU hit-rate gap (pp)")
    lo, hi = min(gaps + [0.0]), max(gaps + [0.0])
    pad = 0.25 * max(hi - lo, 1.0)
    ax2.set_ylim(lo - pad, hi + pad)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=7)

    fig.tight_layout()
    _save_fig(fig, output_dir, "density_characterization")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Phase 2/3 paper figures "
                    "(adaptive dry-run comparison, density characterization)"
    )
    parser.add_argument(
        "--adaptive-dir", default="results/adaptive",
        help="Directory containing phase3_subset_* run directories",
    )
    parser.add_argument(
        "--density-dir", default="results/density",
        help="Directory containing phase2_* density profile directories",
    )
    parser.add_argument(
        "--output-dir", default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figures", choices=["a", "b", "all"], default="all",
        help="Which figure(s) to generate (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0

    if args.figures in ("a", "all"):
        logger.info("Figure A: adaptive dry-run comparison")
        tidy_a = load_adaptive_data(Path(args.adaptive_dir))
        plot_adaptive_dry_run(tidy_a, output_dir)

    if args.figures in ("b", "all"):
        logger.info("Figure B: density characterization")
        tidy_b = load_density_data(Path(args.density_dir))
        if tidy_b is None:
            exit_code = 1
        else:
            plot_density_characterization(tidy_b, output_dir)

    if exit_code:
        logger.error(
            "Figure B was NOT generated: inputs missing locally "
            "(see 'missing input:' lines above)."
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
