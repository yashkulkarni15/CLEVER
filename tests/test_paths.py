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


class TestEmbeddingsFileLegacyFallback:
    """LMSYS/MiniLM artifacts predate the nested layout: when the nested file
    is missing but the legacy flat file exists, the resolver returns the
    legacy path. Everything else keeps the nested path."""

    def test_lmsys_minilm_falls_back_to_legacy_flat(self, tmp_path):
        legacy = tmp_path / "100k_embeddings.npy"
        legacy.touch()
        p = embeddings_file(str(tmp_path), "lmsys", "all-MiniLM-L6-v2", "100k")
        assert p == legacy

    def test_org_prefixed_minilm_falls_back(self, tmp_path):
        legacy = tmp_path / "full_embeddings.npy"
        legacy.touch()
        p = embeddings_file(
            str(tmp_path), "lmsys", "sentence-transformers/all-MiniLM-L6-v2", "full"
        )
        assert p == legacy

    def test_nested_wins_when_both_exist(self, tmp_path):
        nested = tmp_path / "lmsys" / "all-MiniLM-L6-v2" / "100k_embeddings.npy"
        nested.parent.mkdir(parents=True)
        nested.touch()
        (tmp_path / "100k_embeddings.npy").touch()
        p = embeddings_file(str(tmp_path), "lmsys", "all-MiniLM-L6-v2", "100k")
        assert p == nested

    def test_non_lmsys_never_falls_back(self, tmp_path):
        (tmp_path / "100k_embeddings.npy").touch()
        p = embeddings_file(str(tmp_path), "moss", "all-MiniLM-L6-v2", "100k")
        assert p == tmp_path / "moss" / "all-MiniLM-L6-v2" / "100k_embeddings.npy"

    def test_other_model_never_falls_back(self, tmp_path):
        (tmp_path / "100k_embeddings.npy").touch()
        p = embeddings_file(str(tmp_path), "lmsys", "thenlper/gte-base", "100k")
        assert p == tmp_path / "lmsys" / "gte-base" / "100k_embeddings.npy"

    def test_neither_exists_returns_nested(self, tmp_path):
        p = embeddings_file(str(tmp_path), "lmsys", "all-MiniLM-L6-v2", "100k")
        assert p == tmp_path / "lmsys" / "all-MiniLM-L6-v2" / "100k_embeddings.npy"


class TestQueriesFileLegacyFallback:
    def test_lmsys_falls_back_to_legacy_flat(self, tmp_path):
        legacy = tmp_path / "100k_queries.parquet"
        legacy.touch()
        p = queries_file("lmsys", "100k", data_root=str(tmp_path))
        assert p == legacy

    def test_nested_wins_when_both_exist(self, tmp_path):
        nested = tmp_path / "lmsys" / "100k_queries.parquet"
        nested.parent.mkdir()
        nested.touch()
        (tmp_path / "100k_queries.parquet").touch()
        p = queries_file("lmsys", "100k", data_root=str(tmp_path))
        assert p == nested

    def test_non_lmsys_never_falls_back(self, tmp_path):
        (tmp_path / "100k_queries.parquet").touch()
        p = queries_file("moss", "100k", data_root=str(tmp_path))
        assert p == tmp_path / "moss" / "100k_queries.parquet"

    def test_neither_exists_returns_nested(self, tmp_path):
        p = queries_file("lmsys", "100k", data_root=str(tmp_path))
        assert p == tmp_path / "lmsys" / "100k_queries.parquet"
