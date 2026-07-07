"""
Option-A multi-dataset path resolver.

Single source of truth for where per-(dataset, embedding-model) artifacts live,
so that the generation step (scripts/02) and the consumption step (scripts/08)
never disagree and never overwrite each other when the experiment matrix grows
to ``3 datasets × 2 embeddings``.

Conventions
-----------
- Queries:    ``<data_root>/<dataset>/<size>_queries.parquet``
- Embeddings: ``<emb_root>/<dataset>/<model_tag>/<size>_embeddings.npy``

``model_tag`` strips any HuggingFace org prefix (e.g. ``thenlper/gte-base`` →
``gte-base``) so model names with slashes don't create nested directories.

Note: the original LMSYS/all-MiniLM-L6-v2 artifacts predate this scheme and
live at the flat legacy paths (``results/embeddings/<size>_embeddings.npy``,
``data/<size>_queries.parquet``). When the nested path is missing but the
legacy flat file exists, the resolver falls back to it, so callers never need
to special-case LMSYS. A nested file always wins over the legacy one.
"""

from pathlib import Path

DEFAULT_EMB_ROOT = "results/embeddings"
DEFAULT_DATA_ROOT = "data"

# The pre-Option-A flat layout only ever held LMSYS embeddings from this model.
LEGACY_DATASET = "lmsys"
LEGACY_MODEL_TAG = "all-MiniLM-L6-v2"


def model_tag(model_name: str) -> str:
    """Last path component of a model name (HF org prefix stripped)."""
    return model_name.rstrip("/").split("/")[-1]


def embeddings_dir(emb_root: str, dataset: str, model_name: str) -> Path:
    """Directory holding a given (dataset, model)'s embedding subsets."""
    return Path(emb_root) / dataset / model_tag(model_name)


def embeddings_file(
    emb_root: str, dataset: str, model_name: str, size: str = "full",
) -> Path:
    """Path to a specific ``<size>_embeddings.npy`` for (dataset, model).

    Falls back to the legacy flat ``<emb_root>/<size>_embeddings.npy`` when
    the nested file is missing but the legacy one exists (LMSYS/MiniLM only).
    """
    nested = embeddings_dir(emb_root, dataset, model_name) / f"{size}_embeddings.npy"
    if not nested.exists() and (
        dataset == LEGACY_DATASET and model_tag(model_name) == LEGACY_MODEL_TAG
    ):
        legacy = Path(emb_root) / f"{size}_embeddings.npy"
        if legacy.exists():
            return legacy
    return nested


def queries_file(
    dataset: str, size: str = "full", data_root: str = DEFAULT_DATA_ROOT,
) -> Path:
    """Path to a specific ``<size>_queries.parquet`` for a dataset.

    Falls back to the legacy flat ``<data_root>/<size>_queries.parquet`` when
    the nested file is missing but the legacy one exists (LMSYS only; the
    queries file does not depend on the embedding model).
    """
    nested = Path(data_root) / dataset / f"{size}_queries.parquet"
    if not nested.exists() and dataset == LEGACY_DATASET:
        legacy = Path(data_root) / f"{size}_queries.parquet"
        if legacy.exists():
            return legacy
    return nested
