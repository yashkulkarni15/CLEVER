"""
Tests for the Option-A multi-dataset path resolver (`src/data/paths.py`).

These pure functions are the single source of truth for where per-(dataset,
embedding-model) artifacts live, so scripts/02 (generate) and scripts/08
(consume) never disagree and never overwrite each other.
"""

from pathlib import Path

from src.data.paths import (
    embeddings_dir,
    embeddings_file,
    model_tag,
    queries_file,
)


class TestModelTag:
    def test_plain_model_name_unchanged(self):
        assert model_tag("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"

    def test_hf_org_prefix_stripped(self):
        assert model_tag("thenlper/gte-base") == "gte-base"

    def test_trailing_slash_ignored(self):
        assert model_tag("thenlper/gte-base/") == "gte-base"


class TestEmbeddingsDir:
    def test_nests_dataset_and_model(self):
        p = embeddings_dir("results/embeddings", "moss", "thenlper/gte-base")
        assert p == Path("results/embeddings/moss/gte-base")

    def test_minilm_under_dataset(self):
        p = embeddings_dir("results/embeddings", "qqp", "all-MiniLM-L6-v2")
        assert p == Path("results/embeddings/qqp/all-MiniLM-L6-v2")


class TestEmbeddingsFile:
    def test_full_size_filename(self):
        p = embeddings_file("results/embeddings", "moss", "all-MiniLM-L6-v2", "full")
        assert p == Path("results/embeddings/moss/all-MiniLM-L6-v2/full_embeddings.npy")

    def test_subset_size_filename(self):
        p = embeddings_file("results/embeddings", "qqp", "thenlper/gte-base", "10k")
        assert p == Path("results/embeddings/qqp/gte-base/10k_embeddings.npy")


class TestQueriesFile:
    def test_nests_under_dataset(self):
        p = queries_file("moss", "full", data_root="data")
        assert p == Path("data/moss/full_queries.parquet")

    def test_subset_size(self):
        p = queries_file("qqp", "10k", data_root="data")
        assert p == Path("data/qqp/10k_queries.parquet")

    def test_default_data_root(self):
        assert queries_file("moss", "full") == Path("data/moss/full_queries.parquet")
