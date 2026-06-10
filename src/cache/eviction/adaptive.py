"""Adaptive eviction policies for Phase 3A.

The policies in this module use online workload/cache signals instead of
dataset labels:

- frequency skew among active cache entries,
- active-cache semantic density from a sampled neighbour graph.

``AdaptiveHardSwitchPolicy`` chooses exactly one arm per eviction. 
``AdaptiveBlendPolicy`` combines the same ingredients into one victim score.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class _AdaptiveSignalState:
    """Shared online state for adaptive eviction policies."""

    def _init_adaptive_state(
        self,
        *,
        similarity_threshold: float = 0.90,
        recompute_interval: int = 2000,
        max_redundancy_samples: int = 1024,
        density_history_size: int = 64,
        density_threshold_scale: float = 1.0,
        density_floor: float = 1e-4,
        frequency_skew_threshold: float = 1.5,
        min_observations: int = 50,
        dynamic_impute: bool = True,
        seed: int = 0,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.recompute_interval = int(recompute_interval)
        self.max_redundancy_samples = int(max_redundancy_samples)
        self.density_threshold_scale = float(density_threshold_scale)
        self.density_floor = float(density_floor)
        self.frequency_skew_threshold = float(frequency_skew_threshold)
        self.min_observations = int(min_observations)
        self.dynamic_impute = bool(dynamic_impute)
        self.seed = int(seed)

        self._access_order: OrderedDict[int, None] = OrderedDict()
        self._access_counts: OrderedDict[int, int] = OrderedDict()
        self._embeddings: dict[int, np.ndarray] = {}
        self._neighbors: dict[int, set[int]] = {}
        self._density_history: deque[float] = deque(maxlen=density_history_size)
        self._evictions_since_recompute = 0
        self._n_recomputes = 0

        self._last_density = 0.0
        self._last_density_threshold = self.density_floor
        self._last_frequency_skew = 1.0

    # ── Lifecycle helpers ───────────────────────────────────────

    def _record_access(self, cache_id: int) -> None:
        if cache_id in self._access_order:
            self._access_order.move_to_end(cache_id)
        if cache_id in self._access_counts:
            self._access_counts[cache_id] += 1

    def _record_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        self._access_order[cache_id] = None
        self._access_counts[cache_id] = 0
        self._embeddings[cache_id] = embedding.astype(np.float32).copy()
        self._neighbors[cache_id] = set()

        if self.dynamic_impute:
            self._impute_new_neighbors(cache_id)

    def _record_evict(self, cache_id: int) -> None:
        nbrs = self._neighbors.pop(cache_id, None)
        if nbrs:
            for nid in nbrs:
                other = self._neighbors.get(nid)
                if other is not None:
                    other.discard(cache_id)

        self._access_order.pop(cache_id, None)
        self._access_counts.pop(cache_id, None)
        self._embeddings.pop(cache_id, None)
        self._evictions_since_recompute += 1

    def _record_rebuild(self, id_remap: dict[int, int]) -> None:
        new_order: OrderedDict[int, None] = OrderedDict()
        for old_id in self._access_order:
            if old_id in id_remap:
                new_order[id_remap[old_id]] = None
        self._access_order = new_order

        new_counts: OrderedDict[int, int] = OrderedDict()
        for old_id, count in self._access_counts.items():
            if old_id in id_remap:
                new_counts[id_remap[old_id]] = count
        self._access_counts = new_counts

        self._embeddings = {
            id_remap[old_id]: emb
            for old_id, emb in self._embeddings.items()
            if old_id in id_remap
        }

        new_neighbors: dict[int, set[int]] = {}
        for old_id, nbrs in self._neighbors.items():
            if old_id not in id_remap:
                continue
            new_id = id_remap[old_id]
            new_neighbors[new_id] = {
                id_remap[nid] for nid in nbrs if nid in id_remap
            }
        self._neighbors = new_neighbors
        self._evictions_since_recompute = self.recompute_interval

    # ── Density and workload signals ────────────────────────────

    def recompute_density(self, active_ids: set[int]) -> float:
        """Force a neighbour-graph refresh and return current mean density."""
        self._recompute_neighbors(active_ids)
        density = self._mean_density(active_ids)
        self._last_density = density
        return density

    def _refresh_signals(self, active_ids: set[int]) -> dict[str, float]:
        if self._evictions_since_recompute >= self.recompute_interval:
            self._recompute_neighbors(active_ids)
            self._evictions_since_recompute = 0

        density = self._mean_density(active_ids)
        threshold = self._density_threshold()
        frequency_skew = self._frequency_skew(active_ids)
        density_ready = self._density_ready()

        self._last_density = density
        self._last_density_threshold = threshold
        self._last_frequency_skew = frequency_skew
        self._density_history.append(density)

        return {
            "density": density,
            "density_threshold": threshold,
            "density_ready": density_ready,
            "frequency_skew": frequency_skew,
        }

    def _density_ready(self) -> bool:
        return len(self._density_history) >= self.min_observations

    def _density_threshold(self) -> float:
        if not self._density_history:
            return self.density_floor
        if not self._density_ready():
            return self.density_floor

        vals = np.array(self._density_history, dtype=np.float32)
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        return max(self.density_floor, median + self.density_threshold_scale * mad)

    def _mean_density(self, active_ids: set[int]) -> float:
        active = [cid for cid in active_ids if cid in self._embeddings]
        n = len(active)
        if n <= 1:
            return 0.0

        denom = float(n - 1)
        return float(np.mean([
            len(self._neighbors.get(cid, set()) & active_ids) / denom
            for cid in active
        ]))

    def _frequency_skew(self, active_ids: set[int]) -> float:
        counts = [
            self._access_counts.get(cid, 0) + 1
            for cid in active_ids
            if cid in self._access_counts
        ]
        if not counts:
            return 1.0
        arr = np.array(counts, dtype=np.float32)
        mean = max(float(np.mean(arr)), 1e-9)
        return float(np.max(arr) / mean)

    # ── Victim-selection primitives ─────────────────────────────

    def _active_in_recency_order(self, active_ids: set[int]) -> list[int]:
        return [cid for cid in self._access_order if cid in active_ids]

    def _select_lru(self, active_ids: set[int]) -> Optional[int]:
        for cid in self._access_order:
            if cid in active_ids:
                return cid
        return None

    def _select_lfu(self, active_ids: set[int]) -> Optional[int]:
        victim: Optional[int] = None
        min_count = float("inf")
        for cid, count in self._access_counts.items():
            if cid not in active_ids:
                continue
            if count < min_count:
                min_count = count
                victim = cid
        return victim

    def _select_semantic(self, active_ids: set[int]) -> Optional[int]:
        active_cids = self._active_in_recency_order(active_ids)
        n = len(active_cids)
        if n == 0:
            return None

        recency, frequency, redundancy = self._score_components(active_cids)
        utility = recency + frequency + 1e-9
        score = (redundancy + 0.1) / utility
        return int(active_cids[int(np.argmax(score))])

    def _score_components(
        self, active_cids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(active_cids)
        if n == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty

        ranks = np.arange(n, dtype=np.float32)
        recency = (ranks + 1.0) / float(n)

        counts = np.array(
            [self._access_counts.get(cid, 0) for cid in active_cids],
            dtype=np.float32,
        )
        max_count = max(float(np.max(counts)), 1.0)
        frequency = counts / max_count

        denom = max(n - 1, 1)
        active_set = set(active_cids)
        redundancy = np.array([
            len(self._neighbors.get(cid, set()) & active_set)
            for cid in active_cids
        ], dtype=np.float32) / float(denom)

        return recency, frequency, redundancy

    # ── Neighbour graph ─────────────────────────────────────────

    def _impute_new_neighbors(self, cache_id: int) -> None:
        other_ids = [cid for cid in self._embeddings if cid != cache_id]
        if not other_ids:
            return

        sample_ids = self._sample_ids(other_ids, salt=cache_id)
        sample_embs = np.array([self._embeddings[cid] for cid in sample_ids], dtype=np.float32)
        emb = self._embeddings[cache_id].reshape(1, -1)
        dists = self._l2sq(emb, sample_embs)[0]

        hits = np.flatnonzero(dists <= self.similarity_threshold)
        for idx in hits:
            nid = sample_ids[int(idx)]
            self._neighbors[cache_id].add(nid)
            self._neighbors.setdefault(nid, set()).add(cache_id)

    def _recompute_neighbors(self, active_ids: set[int]) -> None:
        ids = sorted(cid for cid in active_ids if cid in self._embeddings)
        n = len(ids)
        self._neighbors = {cid: set() for cid in ids}
        if n <= 1:
            self._n_recomputes += 1
            return

        anchor_ids = self._sample_ids(ids, salt=self._n_recomputes + 10_000)
        embs = np.array([self._embeddings[cid] for cid in ids], dtype=np.float32)
        anchors = np.array([self._embeddings[cid] for cid in anchor_ids], dtype=np.float32)

        batch_size = 2048
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            dists = self._l2sq(embs[start:end], anchors)
            is_neighbor = dists <= self.similarity_threshold

            for local_row, cols in enumerate(is_neighbor):
                row_id = ids[start + local_row]
                for col_idx in np.flatnonzero(cols):
                    col_id = anchor_ids[int(col_idx)]
                    if row_id == col_id:
                        continue
                    self._neighbors[row_id].add(col_id)
                    self._neighbors[col_id].add(row_id)

        self._n_recomputes += 1

    def _sample_ids(self, ids: list[int], *, salt: int) -> list[int]:
        sample_size = min(len(ids), self.max_redundancy_samples)
        if len(ids) <= sample_size:
            return list(ids)
        rng_seed = (self.seed * 1_000_003 + int(salt) * 97 + 17) & 0x7FFFFFFF
        rng = np.random.RandomState(rng_seed or 1)
        idx = rng.choice(len(ids), sample_size, replace=False)
        idx.sort()
        return [ids[int(i)] for i in idx]

    @staticmethod
    def _l2sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        return (
            np.sum(a * a, axis=1, keepdims=True)
            + np.sum(b * b, axis=1)[None, :]
            - 2.0 * (a @ b.T)
        )

    @property
    def _common_stats(self) -> dict:
        return {
            "similarity_threshold": self.similarity_threshold,
            "density_floor": self.density_floor,
            "density_threshold_scale": self.density_threshold_scale,
            "frequency_skew_threshold": self.frequency_skew_threshold,
            "min_observations": self.min_observations,
            "last_density": round(self._last_density, 8),
            "last_density_threshold": round(self._last_density_threshold, 8),
            "last_frequency_skew": round(self._last_frequency_skew, 6),
            "n_recomputes": self._n_recomputes,
        }


class AdaptiveHardSwitchPolicy(_AdaptiveSignalState, EvictionPolicy):
    """Choose LRU, LFU, or Semantic at each eviction from online signals."""

    def __init__(self, **kwargs) -> None:
        self._init_adaptive_state(**kwargs)
        self._last_selected_arm = "lru"
        self._arm_counts = {"lru": 0, "lfu": 0, "semantic": 0}

    def on_access(self, cache_id: int) -> None:
        self._record_access(cache_id)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        self._record_insert(cache_id, embedding)

    def on_evict(self, cache_id: int) -> None:
        self._record_evict(cache_id)

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        self._record_rebuild(id_remap)

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        if not active_ids:
            return None

        signals = self._refresh_signals(active_ids)
        if signals["density_ready"] and signals["density"] > signals["density_threshold"]:
            arm = "semantic"
            victim = self._select_semantic(active_ids)
        elif signals["frequency_skew"] >= self.frequency_skew_threshold:
            arm = "lfu"
            victim = self._select_lfu(active_ids)
        else:
            arm = "lru"
            victim = self._select_lru(active_ids)

        if victim is None:
            arm = "lru"
            victim = self._select_lru(active_ids)

        self._last_selected_arm = arm
        self._arm_counts[arm] += 1
        return victim

    @property
    def name(self) -> str:
        return "adaptive_hard"

    @property
    def stats(self) -> dict:
        return {
            **self._common_stats,
            "last_selected_arm": self._last_selected_arm,
            "arm_counts": dict(self._arm_counts),
        }


class AdaptiveBlendPolicy(_AdaptiveSignalState, EvictionPolicy):
    """Evict by a dynamic blended recency/frequency/redundancy score."""

    def __init__(
        self,
        *,
        base_recency_weight: float = 1.0,
        max_frequency_weight: float = 1.0,
        max_semantic_weight: float = 2.0,
        **kwargs,
    ) -> None:
        self._init_adaptive_state(**kwargs)
        self.base_recency_weight = float(base_recency_weight)
        self.max_frequency_weight = float(max_frequency_weight)
        self.max_semantic_weight = float(max_semantic_weight)
        self._last_weights = {
            "recency": self.base_recency_weight,
            "frequency": 0.0,
            "semantic": 0.0,
        }
        self._last_victim: Optional[int] = None

    def on_access(self, cache_id: int) -> None:
        self._record_access(cache_id)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        self._record_insert(cache_id, embedding)

    def on_evict(self, cache_id: int) -> None:
        self._record_evict(cache_id)

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        self._record_rebuild(id_remap)

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        active_cids = self._active_in_recency_order(active_ids)
        n = len(active_cids)
        if n == 0:
            return None

        signals = self._refresh_signals(active_ids)
        recency, frequency, redundancy = self._score_components(active_cids)
        inverse_recency = 1.0 - ((recency * n - 1.0) / max(n - 1, 1))
        inverse_frequency = 1.0 - frequency

        weights = self._compute_weights(signals)
        score = (
            weights["recency"] * inverse_recency
            + weights["frequency"] * inverse_frequency
            + weights["semantic"] * redundancy
        )

        victim = int(active_cids[int(np.argmax(score))])
        self._last_victim = victim
        self._last_weights = weights
        return victim

    def _compute_weights(self, signals: dict[str, float]) -> dict[str, float]:
        if signals["density_ready"]:
            threshold = max(signals["density_threshold"], 1e-12)
            density_ratio = signals["density"] / threshold
            semantic_scale = min(max(density_ratio - 1.0, 0.0), 1.0)
        else:
            semantic_scale = 0.0

        skew_denominator = max(self.frequency_skew_threshold - 1.0, 1e-9)
        frequency_scale = min(
            max((signals["frequency_skew"] - 1.0) / skew_denominator, 0.0),
            1.0,
        )

        return {
            "recency": self.base_recency_weight,
            "frequency": self.max_frequency_weight * frequency_scale,
            "semantic": self.max_semantic_weight * semantic_scale,
        }

    @property
    def name(self) -> str:
        return "adaptive_blend"

    @property
    def stats(self) -> dict:
        return {
            **self._common_stats,
            "base_recency_weight": self.base_recency_weight,
            "max_frequency_weight": self.max_frequency_weight,
            "max_semantic_weight": self.max_semantic_weight,
            "last_weights": dict(self._last_weights),
            "last_victim": self._last_victim,
        }
