#!/usr/bin/env python3
"""
02_generate_embeddings.py — Generate embeddings from processed queries.

Usage (local, small subsets):
    python scripts/02_generate_embeddings.py \
        --input data/processed_queries.parquet \
        --output results/embeddings/ \
        --device cpu --batch-size 64 \
        --sizes 10k,50k

Usage (Great Lakes, full dataset):
    python scripts/02_generate_embeddings.py \
        --input data/processed_queries.parquet \
        --output results/embeddings/ \
        --device cuda --batch-size 512 \
        --sizes 10k,50k,100k,500k,full
"""

import json
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_processed_queries
from src.data.paths import embeddings_dir
from src.data.sampler import DEFAULT_SUBSET_SIZES, create_embedding_subsets
from src.embeddings.encoder import QueryEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def requested_embedding_output_names(sizes: str, n_queries: int | None = None) -> list[str]:
    """Return the embedding output names this invocation is expected to write."""
    names = []
    for raw_name in sizes.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name == "full":
            if name not in names:
                names.append(name)
            continue
        if name in DEFAULT_SUBSET_SIZES:
            if n_queries is not None and DEFAULT_SUBSET_SIZES[name] > n_queries:
                continue
            if name not in names:
                names.append(name)

    if "full" not in names:
        names.append("full")

    return names


def _metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.meta.json")


def _valid_embedding_file(
    path: Path,
    *,
    model_name: str | None = None,
    require_model_metadata: bool = False,
) -> bool:
    """Check that a saved embedding file has a readable float32 2D npy header."""
    if not path.exists():
        return False

    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError, EOFError):
        return False

    valid_array = (
        arr.dtype == np.float32
        and arr.ndim == 2
        and arr.shape[0] > 0
        and arr.shape[1] > 0
    )
    if not valid_array:
        return False

    meta_path = _metadata_path(path)
    if not meta_path.exists():
        return not require_model_metadata

    try:
        metadata = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    if model_name is not None and metadata.get("model_name") != model_name:
        return False
    if metadata.get("n_rows") != int(arr.shape[0]):
        return False
    if metadata.get("embedding_dim") != int(arr.shape[1]):
        return False

    return True


def embedding_outputs_complete(
    output_dir: str | Path,
    sizes: str,
    *,
    n_queries: int | None = None,
    model_name: str | None = None,
    require_model_metadata: bool = False,
) -> bool:
    """Return True when all requested embedding output files are already usable."""
    output_dir = Path(output_dir)
    return all(
        _valid_embedding_file(
            output_dir / f"{name}_embeddings.npy",
            model_name=model_name,
            require_model_metadata=require_model_metadata,
        )
        for name in requested_embedding_output_names(sizes, n_queries=n_queries)
    )


def write_embedding_metadata(saved: dict[str, Path], *, model_name: str) -> None:
    """Write small sidecar metadata so flat output resume is model-aware."""
    for path in saved.values():
        arr = np.load(path, mmap_mode="r")
        metadata = {
            "model_name": model_name,
            "n_rows": int(arr.shape[0]),
            "embedding_dim": int(arr.shape[1]),
        }
        _metadata_path(path).write_text(json.dumps(metadata, sort_keys=True, indent=2))


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to processed_queries.parquet.",
)
@click.option(
    "--dataset",
    default=None,
    help="If set (e.g. moss/qqp/lmsys), nest output as "
         "<output>/<dataset>/<model_tag>/ so datasets+models don't overwrite.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="results/embeddings/",
    help="Base output directory for embedding .npy files.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cpu",
    help="Device for encoding. Use 'cuda' on Great Lakes GPU nodes.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Encoding batch size. Use 512 for GPU, 64 for CPU.",
)
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    help="Sentence-transformers model name.",
)
@click.option(
    "--sizes",
    default="10k,50k",
    help="Comma-separated subset sizes to generate (e.g., '10k,50k,100k,500k,full').",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip generation when all requested embedding .npy outputs already exist.",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Max queries to encode (for dev/debugging). None = all.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for subset creation.",
)
def main(input_path, dataset, output, device, batch_size, model, sizes,
         skip_existing, max_queries, seed):
    """Generate embeddings from processed queries and create scale subsets."""
    # Option A: when --dataset is given, nest output by <dataset>/<model_tag>/
    # so the 3×2 (dataset × embedding) matrix never overwrites itself.
    if dataset:
        output_dir = embeddings_dir(output, dataset, model)
        logger.info(f"Nesting embeddings under {output_dir} (dataset={dataset}, model={model})")
    else:
        output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load processed queries
    logger.info("=" * 60)
    logger.info("Step 1: Loading processed queries")
    logger.info("=" * 60)
    df = load_processed_queries(input_path)

    if max_queries and max_queries < len(df):
        logger.info(f"Limiting to {max_queries} queries (dev mode)")
        df = df.head(max_queries).reset_index(drop=True)

    queries = df["query_text"].tolist()
    logger.info(f"Loaded {len(queries)} queries")

    if skip_existing and embedding_outputs_complete(
        output_dir,
        sizes,
        n_queries=len(df),
        model_name=model,
        require_model_metadata=dataset is None,
    ):
        logger.info(
            f"Skipping embedding generation; requested outputs already exist under {output_dir}"
        )
        return

    # Step 2: Generate embeddings
    logger.info("=" * 60)
    logger.info(f"Step 2: Generating embeddings (device={device}, batch_size={batch_size})")
    logger.info("=" * 60)

    encoder = QueryEncoder(model_name=model, device=device)

    start_time = time.time()
    embeddings = encoder.encode(queries, batch_size=batch_size, normalize=True)
    elapsed = time.time() - start_time

    logger.info(f"Encoding complete in {elapsed:.1f}s ({len(queries)/elapsed:.0f} queries/sec)")

    # Validate embeddings
    logger.info("Validating embeddings...")
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
    assert embeddings.shape == (len(queries), encoder.embedding_dim)
    norms = np.linalg.norm(embeddings, axis=1)
    # float32-appropriate tolerance: norm error grows with embedding_dim, so 1e-5
    # is too strict for higher-dim models (e.g. gte-base, 768-dim).
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"Embeddings not properly L2-normalized "
        f"(max deviation {np.abs(norms - 1.0).max():.2e})"
    )
    nan_count = np.isnan(embeddings).any(axis=1).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} embeddings with NaN — removing them")
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        embeddings = embeddings[valid_mask]
        df = df[valid_mask].reset_index(drop=True)
    logger.info("✓ Embeddings validated (normalized, no NaN)")

    # Step 3: Create subsets
    logger.info("=" * 60)
    logger.info("Step 3: Creating embedding subsets")
    logger.info("=" * 60)

    # Parse requested sizes
    requested = [s.strip() for s in sizes.split(",")]
    subset_sizes = {}
    for name in requested:
        if name == "full":
            continue  # Full is always saved
        if name in DEFAULT_SUBSET_SIZES:
            subset_sizes[name] = DEFAULT_SUBSET_SIZES[name]
        else:
            logger.warning(f"Unknown subset size '{name}', skipping")

    saved = create_embedding_subsets(
        embeddings=embeddings,
        df=df,
        output_dir=output_dir,
        sizes=subset_sizes,
        seed=seed,
    )
    write_embedding_metadata(saved, model_name=model)

    # Summary
    logger.info("=" * 60)
    logger.info("DONE — Embedding Generation Summary")
    logger.info("=" * 60)
    logger.info(f"  Total queries encoded: {len(queries)}")
    logger.info(f"  Embedding dim:         {encoder.embedding_dim}")
    logger.info(f"  Encoding time:         {elapsed:.1f}s")
    logger.info(f"  Throughput:            {len(queries)/elapsed:.0f} queries/sec")
    logger.info(f"  Device:                {device}")
    logger.info(f"  Output directory:      {output_dir}")
    logger.info(f"  Subsets created:       {list(saved.keys())}")
    for name, path in saved.items():
        emb = np.load(path)
        size_mb = emb.nbytes / (1024 * 1024)
        logger.info(f"    {name}: {emb.shape} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
