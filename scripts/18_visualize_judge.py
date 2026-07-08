#!/usr/bin/env python3
"""
Phase 8 — LLM-judge paper figures and table (quality-adjusted hit rate).

Consumes the completed Phase 7 judge run:

  results/judge/phase7_minilm_c0p10_local/judge_scores.json
      18 (dataset,policy) cells: raw hit rate, judged-YES rate,
      quality-adjusted hit rate (10% cache, MiniLM, seed 42, n=1000/cell).
  results/judge/phase7_minilm_c0p10_local/verdicts.jsonl
      per-pair records (dataset, policy, distance_l2sq, verdict, texts).

Generates:

  Figure E — Raw vs quality-adjusted hit rate
      3 panels (dataset), paired bars per policy: raw hit rate (gray,
      context) next to quality-adjusted hit rate (blue, the point), shared
      0-100% y-scale. Wilson 95% CIs on the judged-YES rate propagate to the
      quality-adjusted bars. Story: the measured hit rate is mostly illusory
      (LMSYS/QQP collapse ~57-60% -> ~2%; MOSS 97.6% -> ~25%) and no policy
      escapes the collapse.
      Output: results/figures/judge_quality_adjusted.{png,pdf}

  Figure F — Judged-YES rate vs hit distance
      Per-dataset quantile-binned P(judge says equivalent) against the hit's
      embedding distance (L2^2 on unit-norm), pairs deduped across policies.
      Mechanism figure: equivalence decays with distance, and even the
      nearest LMSYS hits are mostly non-equivalent (templated prompts).
      Output: results/figures/judge_yes_vs_distance.{png,pdf}

  Table — LaTeX tabular (policies x datasets: raw, YES [95% CI], QA)
      Output: results/judge/phase8_judge_table.tex

Usage::

    python scripts/18_visualize_judge.py \\
        --judge-dir results/judge/phase7_minilm_c0p10_local \\
        --output-dir results/figures
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Canonical orderings (match scripts/15_visualize_phase6_matrix.py) ──
DATASET_ORDER = ["lmsys", "qqp", "moss"]
DATASET_LABELS = {"lmsys": "LMSYS", "qqp": "QQP", "moss": "MOSS"}

POLICY_ORDER = ["lru", "lfu", "semantic", "arc", "gdsf", "siso"]
POLICY_LABELS = {
    "lru": "LRU", "lfu": "LFU", "semantic": "Semantic",
    "arc": "ARC", "gdsf": "GDSF", "siso": "SISO",
}

# Okabe-Ito colorblind-safe hues (same instances as the Phase 6 figures).
RAW_COLOR = "#ADADAD"       # de-emphasis gray: raw hit rate is context
QADJ_COLOR = "#0072B2"      # accent blue: quality-adjusted is the point
DATASET_COLORS = {
    "lmsys": "#0072B2",     # blue
    "qqp": "#E69F00",       # orange
    "moss": "#009E73",      # green
}
DATASET_MARKERS = {"lmsys": "o", "qqp": "s", "moss": "^"}


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
# Data shaping (unit-tested in tests/test_judge_figures_data.py)
# ═════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_scores(path: Path) -> dict:
    """Load judge_scores.json and require all 18 judged cells."""
    scores = json.loads(Path(path).read_text())
    problems = []
    for ds in DATASET_ORDER:
        for policy in POLICY_ORDER:
            key = f"{ds}/{policy}"
            cell = scores.get(key)
            if cell is None:
                problems.append(f"{key}: missing")
            elif cell.get("n_judged", 0) <= 0:
                problems.append(f"{key}: n_judged=0")
    if problems:
        raise ValueError(f"incomplete judge scores: {'; '.join(problems)}")
    return scores


def scores_frame(scores: dict) -> list[dict]:
    """Tidy per-cell rows in canonical order, with Wilson CIs propagated.

    The raw hit rate's seed variance is negligible (sigma ~ 1e-4), so the
    quality-adjusted CI is raw * [CI of the judged-YES rate].
    """
    frame = []
    for ds in DATASET_ORDER:
        for policy in POLICY_ORDER:
            cell = scores[f"{ds}/{policy}"]
            n = cell["n_judged"]
            mean_yes = cell["mean_yes"]
            k = int(round(mean_yes * n))
            yes_lo, yes_hi = wilson_ci(k, n)
            raw = cell["raw_hit_rate"]
            frame.append({
                "dataset": ds,
                "policy": policy,
                "raw": raw,
                "mean_yes": mean_yes,
                "yes_lo": yes_lo,
                "yes_hi": yes_hi,
                "qadj": raw * mean_yes,
                "qadj_lo": raw * yes_lo,
                "qadj_hi": raw * yes_hi,
                "n_judged": n,
            })
    return frame


def distance_yes_curve(records, n_bins: int = 10) -> dict:
    """Per-dataset quantile-binned YES rate vs hit distance.

    Pairs are deduped (a pair judged under several policies counts once);
    None verdicts (unparsed) are excluded. Tied distances collapse duplicate
    quantile edges, so heavily tied data degrades to fewer bins rather than
    crashing or producing empty bins.
    """
    per_ds: dict[str, list[tuple[float, bool]]] = {}
    seen = set()
    for r in records:
        verdict = r.get("verdict")
        if verdict is None:
            continue
        key = (r["dataset"], r["orig_query"], r["new_query"])
        if key in seen:
            continue
        seen.add(key)
        per_ds.setdefault(r["dataset"], []).append(
            (float(r["distance_l2sq"]), bool(verdict))
        )

    curves: dict[str, list[dict]] = {}
    for ds, pts in per_ds.items():
        dists = np.array([p[0] for p in pts])
        yes = np.array([p[1] for p in pts], dtype=float)
        edges = np.unique(np.quantile(dists, np.linspace(0, 1, n_bins + 1)))
        if len(edges) == 1:
            bin_idx = np.zeros(len(dists), dtype=int)
            n_eff = 1
        else:
            bin_idx = np.clip(
                np.searchsorted(edges, dists, side="right") - 1,
                0, len(edges) - 2,
            )
            n_eff = len(edges) - 1
        bins = []
        for b in range(n_eff):
            mask = bin_idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            k = int(yes[mask].sum())
            lo, hi = wilson_ci(k, n)
            bins.append({
                "dist_median": float(np.median(dists[mask])),
                "yes_rate": k / n,
                "n": n,
                "lo": lo,
                "hi": hi,
            })
        curves[ds] = bins
    return curves


def latex_table(scores: dict) -> str:
    """Policies x datasets tabular: raw %, judged-YES % [95% CI], QA %."""
    frame = scores_frame(scores)
    by_cell = {(r["dataset"], r["policy"]): r for r in frame}

    lines = [
        "\\begin{tabular}{l" + " rrr" * len(DATASET_ORDER) + "}",
        "\\toprule",
        " & " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{DATASET_LABELS[ds]}}}"
            for ds in DATASET_ORDER
        ) + " \\\\",
        "Policy & " + " & ".join(
            ["Raw & YES [95\\% CI] & QA"] * len(DATASET_ORDER)
        ) + " \\\\",
        "\\midrule",
    ]
    for policy in POLICY_ORDER:
        cells = []
        for ds in DATASET_ORDER:
            r = by_cell[(ds, policy)]
            cells.append(
                f"{100 * r['raw']:.1f} & "
                f"{100 * r['mean_yes']:.1f} "
                f"[{100 * r['yes_lo']:.1f}, {100 * r['yes_hi']:.1f}] & "
                f"{100 * r['qadj']:.1f}"
            )
        lines.append(f"{POLICY_LABELS[policy]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def load_verdict_records(path: Path):
    """Stream verdicts.jsonl, skipping malformed lines with a warning."""
    n_bad = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        logger.warning(f"skipped {n_bad} malformed lines in {path}")


# ═════════════════════════════════════════════════════════════════
# Figure E — raw vs quality-adjusted hit rate
# ═════════════════════════════════════════════════════════════════

def plot_quality_adjusted(frame: list[dict], output_dir: Path):
    by_cell = {(r["dataset"], r["policy"]): r for r in frame}
    fig, axes = plt.subplots(
        1, len(DATASET_ORDER), figsize=(9.6, 3.4),
        sharey=True, squeeze=False,
    )

    x = np.arange(len(POLICY_ORDER))
    width = 0.38
    for c, ds in enumerate(DATASET_ORDER):
        ax = axes[0][c]
        rows = [by_cell[(ds, p)] for p in POLICY_ORDER]
        raw = [100 * r["raw"] for r in rows]
        qadj = [100 * r["qadj"] for r in rows]
        err_lo = [100 * (r["qadj"] - r["qadj_lo"]) for r in rows]
        err_hi = [100 * (r["qadj_hi"] - r["qadj"]) for r in rows]

        ax.bar(x - width / 2, raw, width, color=RAW_COLOR,
               label="Raw hit rate")
        ax.bar(x + width / 2, qadj, width, color=QADJ_COLOR,
               yerr=[err_lo, err_hi], capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
               label="Quality-adjusted")
        # QA bars are unreadable on a 0-100% axis: direct-label them.
        for xi, r in zip(x, rows):
            ax.text(xi + width / 2, 100 * r["qadj_hi"] + 1.5,
                    f"{100 * r['qadj']:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#1A1A1A")

        ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 104)
        if c == 0:
            ax.set_ylabel("Hit rate (%)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Quality-adjusted hit rate (raw × judged-equivalent fraction) — "
        "10% cache, MiniLM, n=1000 judged hits/cell",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save_fig(fig, output_dir, "judge_quality_adjusted")


# ═════════════════════════════════════════════════════════════════
# Figure F — judged-YES rate vs hit distance
# ═════════════════════════════════════════════════════════════════

def plot_yes_vs_distance(curves: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    for ds in DATASET_ORDER:
        bins = curves.get(ds, [])
        if not bins:
            continue
        xs = [b["dist_median"] for b in bins]
        ys = [100 * b["yes_rate"] for b in bins]
        los = [100 * b["lo"] for b in bins]
        his = [100 * b["hi"] for b in bins]
        color = DATASET_COLORS[ds]
        ax.plot(xs, ys, marker=DATASET_MARKERS[ds], markersize=5,
                markeredgecolor="white", markeredgewidth=0.7,
                color=color, linewidth=1.5, label=DATASET_LABELS[ds])
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
        # Direct label at the curve's left end (nearest hits).
        ax.annotate(
            DATASET_LABELS[ds], (xs[0], ys[0]),
            textcoords="offset points", xytext=(-4, 6),
            ha="right", fontsize=8, color=color,
        )

    ax.axvline(0.90, color="#888888", linewidth=1, linestyle="--")
    ax.text(0.888, 0.45, "hit threshold (0.90)", rotation=90,
            transform=ax.get_xaxis_transform(),
            ha="right", va="center", fontsize=7, color="#666666")
    ax.set_xlabel("Hit distance (squared L2 on unit-norm embeddings)")
    ax.set_ylabel("Judged equivalent (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        "Semantic equivalence decays with embedding distance\n"
        "(deciles per dataset, unique pairs, Wilson 95% bands)",
        fontsize=9,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir, "judge_yes_vs_distance")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Phase 8 LLM-judge figures and LaTeX table "
                    "(quality-adjusted hit rate)"
    )
    parser.add_argument(
        "--judge-dir", default="results/judge/phase7_minilm_c0p10_local",
        help="Judge run directory containing judge_scores.json and "
             "verdicts.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--table-out", default="results/judge/phase8_judge_table.tex",
        help="Output path for the LaTeX table",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Quantile bins per dataset for the distance curve",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()

    judge_dir = Path(args.judge_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        scores = load_scores(judge_dir / "judge_scores.json")
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"cannot load judge scores: {exc}")
        sys.exit(1)

    frame = scores_frame(scores)
    logger.info("Figure E: raw vs quality-adjusted hit rate")
    plot_quality_adjusted(frame, output_dir)

    verdicts_path = judge_dir / "verdicts.jsonl"
    if verdicts_path.exists():
        logger.info("Figure F: judged-YES rate vs hit distance")
        curves = distance_yes_curve(
            load_verdict_records(verdicts_path), n_bins=args.n_bins,
        )
        plot_yes_vs_distance(curves, output_dir)
    else:
        logger.warning(f"{verdicts_path} not found — skipping Figure F")

    table_out = Path(args.table_out)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table_out.write_text(latex_table(scores) + "\n")
    logger.info(f"Saved: {table_out}")


if __name__ == "__main__":
    main()
