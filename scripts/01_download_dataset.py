#!/usr/bin/env python3
"""
01_download_dataset.py — Build a query dataset (lmsys / moss / qqp).

Extracts first-user queries into the shared schema, preprocesses (filter +
dedup), and writes scale subsets. All three datasets feed the same downstream
pipeline (scripts/02 embeddings → scripts/08 eviction).

Usage:
    # LMSYS (downloads from HuggingFace)
    python scripts/01_download_dataset.py --dataset lmsys --output data/

    # MOSS (reads local English JSON files; single-turn English by default)
    python scripts/01_download_dataset.py --dataset moss --raw-path data/raw/moss/

    # QQP (reads local Quora TSV; flattens to unique questions)
    python scripts/01_download_dataset.py --dataset qqp --raw-path data/raw/qqp/

    # dev/smoke: cap rows
    python scripts/01_download_dataset.py --dataset qqp --max-rows 5000 --skip-subsets
"""

import csv
import json
import logging
import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    extract_first_user_queries,
    extract_moss_queries,
    extract_qqp_queries,
    load_lmsys_dataset,
    save_raw_queries,
)
from src.data.preprocessor import preprocess_queries, save_processed_queries
from src.data.sampler import create_subsets


# ── Per-dataset raw-query extraction (Option A dispatch) ─────────────
# All branches return a DataFrame with the shared schema so the
# downstream preprocess → subset → embed → eviction pipeline is identical.

def _build_lmsys_raw(raw_path, cache_dir, max_rows):
    """LMSYS: download from HuggingFace, extract first user turn."""
    dataset = load_lmsys_dataset(cache_dir=cache_dir)
    return extract_first_user_queries(dataset, max_rows=max_rows)


def _build_moss_raw(raw_path, max_rows, single_turn_only):
    """MOSS: read local en_*.json file(s), extract first human turn.

    ``raw_path`` may be a single ``en_*.json`` file or a directory containing
    them (e.g. ``data/raw/moss/``). Chinese (``zh_*``) files are ignored.
    """
    raw_path = Path(raw_path)
    if raw_path.is_dir():
        files = sorted(raw_path.glob("en_*.json"))
    else:
        files = [raw_path]
    if not files:
        raise click.ClickException(f"No MOSS en_*.json files found under {raw_path}")

    records = []
    for fp in files:
        logger.info(f"  loading MOSS file {fp.name}")
        with open(fp, encoding="utf-8") as f:
            records.extend(json.load(f))
    logger.info(f"  total MOSS raw records: {len(records)}")
    return extract_moss_queries(
        records, max_rows=max_rows, single_turn_only=single_turn_only,
    )


def _build_qqp_raw(raw_path, max_rows):
    """QQP: read the local Quora TSV, flatten to unique questions."""
    raw_path = Path(raw_path)
    if raw_path.is_dir():
        cands = sorted(raw_path.glob("*.tsv"))
        if not cands:
            raise click.ClickException(f"No .tsv found under {raw_path}")
        raw_path = cands[0]
    with open(raw_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"  total QQP raw pairs: {len(rows)}")
    return extract_qqp_queries(rows, max_rows=max_rows)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["lmsys", "moss", "qqp"]),
    default="lmsys",
    help="Which dataset to build (Option A multi-dataset dispatch).",
)
@click.option(
    "--raw-path",
    type=click.Path(),
    default=None,
    help="For moss/qqp: path to the local raw file or directory "
         "(e.g. data/raw/moss/ or data/raw/qqp/quora_duplicate_questions.tsv).",
)
@click.option(
    "--single-turn-only/--all-turns",
    default=True,
    help="MOSS only: keep only num_turns==1 conversations (default) "
         "or take the first human turn from every conversation.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output directory for processed data. "
         "Default: data/ for lmsys, data/<dataset>/ for moss/qqp.",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=None,
    help="HuggingFace cache directory for raw dataset download (lmsys).",
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Max rows to process (for development/debugging). None = all.",
)
@click.option(
    "--min-tokens",
    type=int,
    default=3,
    help="Minimum word count for query filtering.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum word count for query filtering.",
)
@click.option(
    "--skip-subsets",
    is_flag=True,
    default=False,
    help="Skip creating scale subsets (just process queries).",
)
def main(dataset, raw_path, single_turn_only, output, cache_dir, max_rows,
         min_tokens, max_tokens, skip_subsets):
    """Build a query dataset (lmsys/moss/qqp): extract, preprocess, subset."""
    # Default output: data/ for lmsys (back-compat), data/<dataset>/ otherwise.
    if output is None:
        output = "data/" if dataset == "lmsys" else f"data/{dataset}/"
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1+2: Load + extract raw queries (per-dataset dispatch)
    logger.info("=" * 60)
    logger.info(f"Step 1-2: Building raw queries for dataset='{dataset}'")
    logger.info("=" * 60)
    if dataset == "lmsys":
        raw_df = _build_lmsys_raw(raw_path, cache_dir, max_rows)
    elif dataset == "moss":
        if raw_path is None:
            raw_path = "data/raw/moss/"
        raw_df = _build_moss_raw(raw_path, max_rows, single_turn_only)
    elif dataset == "qqp":
        if raw_path is None:
            raw_path = "data/raw/qqp/"
        raw_df = _build_qqp_raw(raw_path, max_rows)
    else:  # pragma: no cover — click.Choice guards this
        raise click.ClickException(f"Unknown dataset {dataset}")

    raw_out = output_dir / "raw_queries.parquet"
    save_raw_queries(raw_df, raw_out)
    logger.info(f"Raw queries saved: {len(raw_df)} rows → {raw_out}")

    # Step 3: Preprocess (filter + deduplicate)
    logger.info("=" * 60)
    logger.info("Step 3: Preprocessing queries")
    logger.info("=" * 60)
    processed_df = preprocess_queries(
        raw_df, min_tokens=min_tokens, max_tokens=max_tokens
    )
    processed_path = output_dir / "processed_queries.parquet"
    save_processed_queries(processed_df, processed_path)

    # Step 4: Create scale subsets
    if not skip_subsets:
        logger.info("=" * 60)
        logger.info("Step 4: Creating scale subsets")
        logger.info("=" * 60)
        create_subsets(processed_df, output_dir=output_dir, seed=42)

    # Summary
    logger.info("=" * 60)
    logger.info("DONE — Data Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"  Raw queries:       {len(raw_df)}")
    logger.info(f"  After processing:  {len(processed_df)}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info(f"  Token count range: [{min_tokens}, {max_tokens}]")

    if "frequency" in processed_df.columns:
        dups = processed_df["frequency"].sum() - len(processed_df)
        logger.info(f"  Duplicates removed: {int(dups)}")

    logger.info("")
    logger.info("Next step: python scripts/02_generate_embeddings.py")


if __name__ == "__main__":
    main()
