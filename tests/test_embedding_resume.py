"""Tests for resumable embedding generation output checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_embedding_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "02_generate_embeddings.py"
    spec = importlib.util.spec_from_file_location("generate_embeddings", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_embedding_outputs_complete_requires_each_requested_file(tmp_path):
    module = _load_embedding_script()
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir()

    for name in ("10k", "50k", "full"):
        np.save(output_dir / f"{name}_embeddings.npy", np.zeros((2, 3), dtype=np.float32))

    assert module.embedding_outputs_complete(output_dir, "10k,50k,full")

    (output_dir / "50k_embeddings.npy").unlink()

    assert not module.embedding_outputs_complete(output_dir, "10k,50k,full")


def test_embedding_outputs_complete_rejects_corrupt_npy(tmp_path):
    module = _load_embedding_script()
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir()

    np.save(output_dir / "10k_embeddings.npy", np.zeros((2, 3), dtype=np.float32))
    (output_dir / "full_embeddings.npy").write_bytes(b"incomplete")

    assert not module.embedding_outputs_complete(output_dir, "10k")


def test_embedding_outputs_complete_ignores_impossible_subset_sizes(tmp_path):
    module = _load_embedding_script()
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir()

    np.save(output_dir / "full_embeddings.npy", np.zeros((5_000, 3), dtype=np.float32))

    assert module.embedding_outputs_complete(
        output_dir,
        "10k,50k",
        n_queries=5_000,
    )


def test_embedding_outputs_complete_rejects_stale_flat_model_metadata(tmp_path):
    module = _load_embedding_script()
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir()

    np.save(output_dir / "full_embeddings.npy", np.zeros((5, 3), dtype=np.float32))
    (output_dir / "full_embeddings.meta.json").write_text(json.dumps({
        "model_name": "old-model",
        "n_rows": 5,
        "embedding_dim": 3,
    }))

    assert not module.embedding_outputs_complete(
        output_dir,
        "full",
        n_queries=5,
        model_name="new-model",
        require_model_metadata=True,
    )

    (output_dir / "full_embeddings.meta.json").write_text(json.dumps({
        "model_name": "new-model",
        "n_rows": 5,
        "embedding_dim": 3,
    }))

    assert module.embedding_outputs_complete(
        output_dir,
        "full",
        n_queries=5,
        model_name="new-model",
        require_model_metadata=True,
    )
