"""Tests for workload density profiling."""

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache
from src.profiler.density import (
    active_embedding_matrix,
    compute_density_snapshot,
)


def test_empty_and_singleton_density_zero():
    empty = np.empty((0, 2), dtype=np.float32)
    singleton = np.array([[0.0, 0.0]], dtype=np.float32)

    assert compute_density_snapshot(empty, theta_l2sq=0.3).mean_density == 0.0
    assert compute_density_snapshot(singleton, theta_l2sq=0.3).mean_density == 0.0


def test_density_excludes_self_matches_for_duplicate_pair():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    snapshot = compute_density_snapshot(embeddings, theta_l2sq=0.3)

    assert snapshot.cache_snapshot_size == 3
    assert snapshot.mean_density == np.float32(1.0 / 3.0)
    assert snapshot.per_entry_density.tolist() == [0.5, 0.5, 0.0]


def test_fully_connected_triangle_density_one():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
        ],
        dtype=np.float32,
    )

    snapshot = compute_density_snapshot(embeddings, theta_l2sq=0.3)

    assert snapshot.mean_density == 1.0
    assert snapshot.per_entry_density.tolist() == [1.0, 1.0, 1.0]


def test_triangle_plus_isolated_density_half():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    snapshot = compute_density_snapshot(embeddings, theta_l2sq=0.3)

    assert snapshot.mean_density == 0.5
    assert snapshot.per_entry_density.tolist() == pytest.approx(
        [2 / 3, 2 / 3, 2 / 3, 0.0]
    )


def test_threshold_boundary_is_inclusive():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    snapshot = compute_density_snapshot(embeddings, theta_l2sq=1.0)

    assert snapshot.mean_density == 1.0


def test_density_uses_active_ids_not_total_rows():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )

    snapshot = compute_density_snapshot(
        embeddings,
        theta_l2sq=0.3,
        active_ids=[0, 1, 3],
    )

    assert snapshot.cache_snapshot_size == 3
    assert snapshot.mean_density == 1.0


def test_active_embedding_matrix_ignores_dead_cache_entries():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    texts = ["a", "b", "c"]
    cache = SemanticCache(
        dim=2,
        index_type="flat",
        max_size=2,
        eviction_policy="lru",
        rebuild_threshold=1.0,
    )
    cache.build(embeddings[:2], texts[:2])
    cache.insert(embeddings[2], texts[2])

    active_ids, active_embeddings = active_embedding_matrix(cache)
    snapshot = compute_density_snapshot(active_embeddings, theta_l2sq=0.3)

    assert active_ids == [1, 2]
    assert snapshot.cache_snapshot_size == 2
    assert snapshot.mean_density == 0.0


def test_density_is_permutation_invariant():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    a = compute_density_snapshot(embeddings, theta_l2sq=0.3)
    b = compute_density_snapshot(embeddings[[3, 2, 1, 0]], theta_l2sq=0.3)

    assert a.mean_density == b.mean_density


def test_float32_and_float64_are_consistent():
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
        ]
    )

    a = compute_density_snapshot(embeddings.astype(np.float32), theta_l2sq=0.3)
    b = compute_density_snapshot(embeddings.astype(np.float64), theta_l2sq=0.3)

    assert a.mean_density == b.mean_density


def test_sampled_density_seed_reproducible_and_bounded():
    rng = np.random.RandomState(7)
    embeddings = rng.normal(size=(50, 8)).astype(np.float32)

    a = compute_density_snapshot(
        embeddings,
        theta_l2sq=2.0,
        probe_size=10,
        anchor_size=20,
        seed=123,
    )
    b = compute_density_snapshot(
        embeddings,
        theta_l2sq=2.0,
        probe_size=10,
        anchor_size=20,
        seed=123,
    )

    assert a.mean_density == b.mean_density
    assert 0.0 <= a.mean_density <= 1.0
    assert a.density_probe_size == 10
    assert a.density_anchor_size == 20
