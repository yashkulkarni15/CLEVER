"""Tests for adaptive eviction policies.

These tests define Phase 3A behavior before implementation:
- hard-switch adaptive eviction chooses LRU, LFU, or Semantic based on
  online workload/cache signals;
- blended adaptive eviction combines recency, frequency, and redundancy.
"""

import numpy as np

from src.cache.eviction.adaptive import (
    AdaptiveBlendPolicy,
    AdaptiveHardSwitchPolicy,
)


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _cluster_embeddings() -> dict[int, np.ndarray]:
    return {
        0: _unit([1.0, 0.0, 0.0, 0.0]),
        1: _unit([0.0, 1.0, 0.0, 0.0]),
        2: _unit([0.0, 0.995, 0.01, 0.0]),
        3: _unit([0.0, 0.99, -0.01, 0.0]),
    }


def _sparse_embeddings() -> dict[int, np.ndarray]:
    return {
        0: _unit([1.0, 0.0, 0.0, 0.0]),
        1: _unit([0.0, 1.0, 0.0, 0.0]),
        2: _unit([0.0, 0.0, 1.0, 0.0]),
        3: _unit([0.0, 0.0, 0.0, 1.0]),
    }


def _insert_all(policy, embeddings: dict[int, np.ndarray]) -> None:
    for cid, emb in embeddings.items():
        policy.on_insert(cid, emb)


class TestAdaptiveHardSwitchPolicy:
    def test_dense_cache_switches_to_semantic_arm(self):
        policy = AdaptiveHardSwitchPolicy(
            similarity_threshold=0.30,
            density_floor=0.05,
            density_threshold_scale=0.0,
            frequency_skew_threshold=100.0,
            min_observations=0,
        )
        embeddings = _cluster_embeddings()
        _insert_all(policy, embeddings)
        policy.recompute_density({0, 1, 2, 3})

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim in {1, 2, 3}
        assert policy.stats["last_selected_arm"] == "semantic"
        assert policy.stats["arm_counts"]["semantic"] == 1

    def test_density_warmup_does_not_force_semantic_arm(self):
        policy = AdaptiveHardSwitchPolicy(
            similarity_threshold=0.30,
            density_floor=0.0001,
            min_observations=50,
        )
        embeddings = _cluster_embeddings()
        _insert_all(policy, embeddings)

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim == 0
        assert policy.stats["last_selected_arm"] == "lru"

    def test_frequency_skew_switches_to_lfu_arm_when_density_is_low(self):
        policy = AdaptiveHardSwitchPolicy(
            similarity_threshold=0.01,
            density_floor=1.0,
            frequency_skew_threshold=1.5,
            min_observations=0,
        )
        _insert_all(policy, _sparse_embeddings())
        for _ in range(6):
            policy.on_access(0)

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim == 1
        assert policy.stats["last_selected_arm"] == "lfu"
        assert policy.stats["arm_counts"]["lfu"] == 1

    def test_low_density_low_skew_falls_back_to_lru_arm(self):
        policy = AdaptiveHardSwitchPolicy(
            similarity_threshold=0.01,
            density_floor=1.0,
            frequency_skew_threshold=100.0,
            min_observations=0,
        )
        _insert_all(policy, _sparse_embeddings())
        policy.on_access(0)

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim == 1
        assert policy.stats["last_selected_arm"] == "lru"
        assert policy.stats["arm_counts"]["lru"] == 1

    def test_rebuild_remaps_adaptive_policy_state(self):
        policy = AdaptiveHardSwitchPolicy(
            similarity_threshold=0.30,
            density_floor=0.05,
            min_observations=0,
        )
        _insert_all(policy, _cluster_embeddings())
        policy.on_access(2)

        policy.on_rebuild({0: 2, 1: 0, 2: 1, 3: 3})

        victim = policy.select_victim({0, 1, 2, 3})
        assert victim in {0, 1, 2, 3}
        assert set(policy._embeddings) == {0, 1, 2, 3}


class TestAdaptiveBlendPolicy:
    def test_dense_cache_assigns_nonzero_semantic_weight(self):
        policy = AdaptiveBlendPolicy(
            similarity_threshold=0.30,
            density_floor=0.05,
            density_threshold_scale=0.0,
            max_semantic_weight=3.0,
            min_observations=0,
        )
        _insert_all(policy, _cluster_embeddings())
        policy.recompute_density({0, 1, 2, 3})

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim in {1, 2, 3}
        assert policy.stats["last_weights"]["semantic"] > 0

    def test_density_warmup_keeps_semantic_weight_zero(self):
        policy = AdaptiveBlendPolicy(
            similarity_threshold=0.30,
            density_floor=0.0001,
            min_observations=50,
        )
        _insert_all(policy, _cluster_embeddings())

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim == 0
        assert policy.stats["last_weights"]["semantic"] == 0.0

    def test_sparse_cache_blend_behaves_like_lru_when_other_weights_are_zero(self):
        policy = AdaptiveBlendPolicy(
            similarity_threshold=0.01,
            density_floor=1.0,
            max_frequency_weight=0.0,
            max_semantic_weight=0.0,
            min_observations=0,
        )
        _insert_all(policy, _sparse_embeddings())
        policy.on_access(0)

        victim = policy.select_victim({0, 1, 2, 3})

        assert victim == 1
        assert policy.stats["last_weights"]["semantic"] == 0.0
        assert policy.stats["last_weights"]["frequency"] == 0.0

    def test_frequency_weight_rises_with_frequency_skew(self):
        policy = AdaptiveBlendPolicy(
            similarity_threshold=0.01,
            density_floor=1.0,
            frequency_skew_threshold=1.5,
            max_frequency_weight=2.0,
            min_observations=0,
        )
        _insert_all(policy, _sparse_embeddings())
        for _ in range(6):
            policy.on_access(0)

        policy.select_victim({0, 1, 2, 3})

        assert policy.stats["last_weights"]["frequency"] > 0

    def test_rebuild_remaps_blended_policy_state(self):
        policy = AdaptiveBlendPolicy(
            similarity_threshold=0.30,
            density_floor=0.05,
            min_observations=0,
        )
        _insert_all(policy, _cluster_embeddings())
        policy.on_access(2)

        policy.on_rebuild({0: 2, 1: 0, 2: 1, 3: 3})

        victim = policy.select_victim({0, 1, 2, 3})
        assert victim in {0, 1, 2, 3}
        assert set(policy._embeddings) == {0, 1, 2, 3}
