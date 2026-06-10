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

Note: the original LMSYS artifacts predate this scheme and live at the flat
legacy paths (``results/embeddings/full_embeddings.npy``,
``data/full_queries.parquet``). LMSYS runs therefore pass explicit
``--embeddings``/``--queries`` overrides (as the existing slurm jobs do); this
resolver is the default for the new ``moss``/``qqp`` datasets.
"""

from pathlib import Path

DEFAULT_EMB_ROOT = "results/embeddings"
DEFAULT_DATA_ROOT = "data"


def model_tag(model_name: str) -> str:
    """Last path component of a model name (HF org prefix stripped)."""
    return model_name.rstrip("/").split("/")[-1]


def embeddings_dir(emb_root: str, dataset: str, model_name: str) -> Path:
    """Directory holding a given (dataset, model)'s embedding subsets."""
    return Path(emb_root) / dataset / model_tag(model_name)


def embeddings_file(
    emb_root: str, dataset: str, model_name: str, size: str = "full",
) -> Path:
    """Path to a specific ``<size>_embeddings.npy`` for (dataset, model)."""
    return embeddings_dir(emb_root, dataset, model_name) / f"{size}_embeddings.npy"


def queries_file(
    dataset: str, size: str = "full", data_root: str = DEFAULT_DATA_ROOT,
) -> Path:
    """Path to a specific ``<size>_queries.parquet`` for a dataset."""
    return Path(data_root) / dataset / f"{size}_queries.parquet"
