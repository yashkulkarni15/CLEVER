"""
Semantic-aware eviction policy — the paper's novel contribution.

Scores each cached entry by a ratio of its *redundancy* (how many
similar entries are nearby) to its *utility* (recency + frequency).
Entries that are highly redundant and rarely used are evicted first;
entries that are isolated (no similar neighbours) or heavily used are
protected.

Eviction score
--------------
    score(e) = redundancy(e) / (α·recency(e) + β·frequency(e) + ε)

- ``redundancy(e)``: fraction of active cache entries whose L2²
  distance to *e* is ≤ ``similarity_threshold``.  High redundancy
  means evicting *e* won't reduce the cache's topical coverage.

- ``recency(e)``: normalised recency in [0, 1] where 1 = most
  recently accessed.  Recency is measured by rank in access order.

- ``frequency(e)``: normalised access count in [0, 1] where 1 = the
  most frequently accessed entry.

- ``α``, ``β``: tunable weights (default 1.0 each).

- ``ε``: small constant (1e-9) to avoid division by zero.

Batch optimisation
------------------
Computing neighbor counts for every entry on every eviction would be
O(N²) per eviction.  Instead we **batch-recompute** redundancy scores
every ``recompute_interval`` evictions using the FAISS index, and
cache the scores in between.  Between recomputations we update only
recency and frequency (which are cheap O(1) updates).
"""

import logging
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy

logger = logging.getLogger(__name__)


class SemanticPolicy(EvictionPolicy):
    """Semantic-aware eviction policy.

    Args:
        similarity_threshold: L2² distance threshold for counting an
            entry as a "neighbour".  For normalised embeddings,
            ``cosine_sim ≈ 1 - L2²/2``, so ``L2² ≤ 0.30`` corresponds
            to ``cosine_sim ≥ 0.85``.
        alpha: Weight for the recency component.
        beta: Weight for the frequency component.
        recompute_interval: Number of evictions between full
            redundancy re-computations.
    """

    # Maximum number of anchor entries for redundancy estimation.
    # Instead of O(N²) all-pairs, we use O(N×S) sampled anchors.
    MAX_REDUNDANCY_SAMPLES = 1024

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        alpha: float = 1.0,
        beta: float = 1.0,
        recompute_interval: int = 50,
        mu: float = 0.1,
        dynamic_impute: bool = True,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.beta = beta
        self.recompute_interval = recompute_interval
        self.mu = mu
        self.dynamic_impute = dynamic_impute
        self._epsilon = 1e-9

        # ── Internal state ───────────────────────────────────────
        # Access order: front = oldest access, back = most recent.
        self._access_order: OrderedDict[int, None] = OrderedDict()
        # Access counts per entry.
        self._access_counts: dict[int, int] = {}
        # Cached redundancy scores (fraction of neighbours).
        self._redundancy: dict[int, float] = {}
        # Embeddings stored per cache_id (needed for neighbour counting).
        self._embeddings: dict[int, np.ndarray] = {}
        # Counter of evictions since last redundancy recomputation.
        self._evictions_since_recompute: int = 0

        # Timing
        self._total_eviction_time_s: float = 0.0
        self._n_evictions: int = 0
        self._n_recomputes: int = 0

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Move to back of access order and increment count."""
        if cache_id in self._access_order:
            self._access_order.move_to_end(cache_id)
        if cache_id in self._access_counts:
            self._access_counts[cache_id] += 1

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Register a newly inserted entry."""
        self._access_order[cache_id] = None
        self._access_counts[cache_id] = 0
        self._embeddings[cache_id] = embedding.copy()
        
        # Dynamic drift-resilient imputation
        if self.dynamic_impute and self._redundancy:
            # Randomly sample active entries to quickly approximate the new entry's redundancy
            active_ids = list(self._embeddings.keys())
            S = min(len(active_ids), self.MAX_REDUNDANCY_SAMPLES)
            
            if len(active_ids) > S:
                import random
                sample_ids = random.sample(active_ids, S)
            else:
                sample_ids = active_ids
                
            sample_embs = np.array([self._embeddings[cid] for cid in sample_ids], dtype=np.float32)
            
            emb = embedding.reshape(1, -1).astype(np.float32)
            norm_q = np.sum(emb ** 2)
            norms_S = np.sum(sample_embs ** 2, axis=1)
            dot = emb @ sample_embs.T
            dist_sq = norm_q + norms_S - 2 * dot
            
            is_neighbor = dist_sq[0] <= self.similarity_threshold
            r = is_neighbor.sum() / max(S, 1)
            self._redundancy[cache_id] = float(r)
        else:
            self._redundancy[cache_id] = getattr(self, '_mean_redundancy', 0.5)

    def on_evict(self, cache_id: int) -> None:
        """Clean up all bookkeeping for the evicted entry."""
        self._access_order.pop(cache_id, None)
        self._access_counts.pop(cache_id, None)
        self._redundancy.pop(cache_id, None)
        self._embeddings.pop(cache_id, None)
        self._evictions_since_recompute += 1
        self._n_evictions += 1

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Select the entry with the highest eviction score."""
        if not active_ids:
            return None

        t_start = time.perf_counter()

        # Recompute redundancy scores periodically
        if self._evictions_since_recompute >= self.recompute_interval:
            self._recompute_redundancy(active_ids)
            self._evictions_since_recompute = 0

        # ── Vectorized Score Computation ─────────────────────────
        # Safely extract strictly ordered ids filtered to the active set
        active_cids = [cid for cid in self._access_order if cid in active_ids]
        n_active = len(active_cids)
        
        if n_active == 0:
            return None
            
        cids = np.array(active_cids, dtype=np.int32)
        
        # 1. Vectorized utility: Recency
        ranks = np.arange(n_active, dtype=np.float32)
        recency = ranks / max(n_active - 1, 1)

        # 2. Vectorized utility: Frequency 
        # Safely extracted in the exact order of active_cids
        counts = np.array([self._access_counts.get(cid, 0) for cid in active_cids], dtype=np.float32)
        max_count = max(float(np.max(counts)), 1.0)
        freq = counts / max_count

        # 3. Vectorized Redundancy
        mean_r = getattr(self, '_mean_redundancy', 0.5)
        r = np.array([self._redundancy.get(cid, mean_r) for cid in active_cids], dtype=np.float32)

        # 4. Evaluation 
        utility = self.alpha * recency + self.beta * freq + self._epsilon
        # Eventual Evictability invariant enforced by mu offset
        score = (r + self.mu) / utility

        best_idx = int(np.argmax(score))
        victim = int(cids[best_idx])

        self._total_eviction_time_s += time.perf_counter() - t_start
        return victim

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap all internal state after cache compaction."""
        # Access order
        new_order: OrderedDict[int, None] = OrderedDict()
        for old_id in self._access_order:
            if old_id in id_remap:
                new_order[id_remap[old_id]] = None
        self._access_order = new_order

        # Access counts
        new_counts: dict[int, int] = {}
        for old_id, cnt in self._access_counts.items():
            if old_id in id_remap:
                new_counts[id_remap[old_id]] = cnt
        self._access_counts = new_counts

        # Redundancy scores
        new_red: dict[int, float] = {}
        for old_id, score in self._redundancy.items():
            if old_id in id_remap:
                new_red[id_remap[old_id]] = score
        self._redundancy = new_red

        # Embeddings
        new_embs: dict[int, np.ndarray] = {}
        for old_id, emb in self._embeddings.items():
            if old_id in id_remap:
                new_embs[id_remap[old_id]] = emb
        self._embeddings = new_embs

        # Force a recomputation on the next eviction
        self._evictions_since_recompute = self.recompute_interval

    # ── Redundancy computation ───────────────────────────────────

    def _recompute_redundancy(self, active_ids: set[int]) -> None:
        """Batch-recompute redundancy scores using sampled anchors.

        Instead of O(N²) all-pairs distances, samples up to
        MAX_REDUNDANCY_SAMPLES anchor entries and estimates each entry's
        redundancy as the fraction of *anchors* within the similarity
        threshold.  This gives O(N × S) complexity where S ≪ N.

        Uses GPU acceleration via PyTorch if available, falling back
        to NumPy.
        """
        if not active_ids:
            return

        ids = sorted(active_ids)
        n = len(ids)

        embs = np.array(
            [self._embeddings[cid] for cid in ids],
            dtype=np.float32,
        )

        # ── Sample anchors if N is large ─────────────────────────
        S = min(n, self.MAX_REDUNDANCY_SAMPLES)
        if S < n:
            rng = np.random.RandomState(self._n_recomputes)
            anchor_idx = rng.choice(n, S, replace=False)
            anchor_idx.sort()
            anchor_embs = embs[anchor_idx]
        else:
            anchor_idx = np.arange(n)
            anchor_embs = embs

        try:
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            import torch
            if torch.cuda.is_available():
                self._recompute_redundancy_torch(
                    ids, embs, n, anchor_embs, anchor_idx, S
                )
                return
        except ImportError:
            pass

        self._recompute_redundancy_numpy(
            ids, embs, n, anchor_embs, anchor_idx, S
        )

    def _recompute_redundancy_torch(
        self,
        ids: list[int],
        embs: np.ndarray,
        n: int,
        anchor_embs: np.ndarray,
        anchor_idx: np.ndarray,
        S: int,
    ) -> None:
        """GPU-accelerated redundancy computation using PyTorch."""
        import torch

        device = torch.device("cuda")
        embs_t = torch.from_numpy(embs).to(device)
        anch_t = torch.from_numpy(anchor_embs).to(device)

        norms_sq_all = torch.sum(embs_t ** 2, dim=1)     # (N,)
        norms_sq_anc = torch.sum(anch_t ** 2, dim=1)      # (S,)

        neighbour_counts = torch.zeros(n, dtype=torch.int32, device=device)

        # Process all entries against anchors in batches
        batch_size = 8192
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch = embs_t[i:end]                          # (B, D)

            dot = torch.matmul(batch, anch_t.T)            # (B, S)
            dist_sq = (
                norms_sq_all[i:end].unsqueeze(1)
                + norms_sq_anc.unsqueeze(0)
                - 2 * dot
            )                                               # (B, S)

            is_neighbour = dist_sq <= self.similarity_threshold  # (B, S)

            # Exclude self: if row i+r corresponds to anchor_idx[j],
            # zero out that cell.  Build a mask vectorised.
            if S == n:
                # Full mode — anchor_idx is identity range
                row_ids = torch.arange(i, end, device=device)  # (B,)
                # Self-match: row_ids == col index (anchor_idx is 0..N-1)
                self_mask = row_ids.unsqueeze(1) == torch.arange(
                    S, device=device
                ).unsqueeze(0)
                is_neighbour = is_neighbour & ~self_mask
            else:
                # Sampled mode — check if global index appears in
                # anchor_idx.  Much cheaper than looping.
                anchor_set_t = torch.from_numpy(anchor_idx).to(device)
                row_globals = torch.arange(i, end, device=device)
                self_mask = row_globals.unsqueeze(1) == anchor_set_t.unsqueeze(0)
                is_neighbour = is_neighbour & ~self_mask

            neighbour_counts[i:end] = is_neighbour.sum(dim=1).to(torch.int32)

        counts_cpu = neighbour_counts.cpu().numpy()
        denom = max(S - 1, 1) if S == n else max(S, 1)
        for i, cid in enumerate(ids):
            self._redundancy[cid] = float(counts_cpu[i]) / denom

        red_vals = list(self._redundancy.values())
        if red_vals:
            self._mean_redundancy = float(np.mean(red_vals))

        self._n_recomputes += 1

    def _recompute_redundancy_numpy(
        self,
        ids: list[int],
        embs: np.ndarray,
        n: int,
        anchor_embs: np.ndarray,
        anchor_idx: np.ndarray,
        S: int,
    ) -> None:
        """CPU fallback redundancy computation (sampled anchors)."""
        norms_sq_all = np.sum(embs ** 2, axis=1)          # (N,)
        norms_sq_anc = np.sum(anchor_embs ** 2, axis=1)    # (S,)

        neighbour_counts = np.zeros(n, dtype=np.int32)

        # Batch rows against anchor columns
        batch_size = 2048
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch = embs[i:end]                             # (B, D)

            dot = batch @ anchor_embs.T                     # (B, S)
            dist_sq = (
                norms_sq_all[i:end, None]
                + norms_sq_anc[None, :]
                - 2 * dot
            )                                                # (B, S)

            is_neighbour = dist_sq <= self.similarity_threshold

            # Vectorised self-exclusion
            if S == n:
                row_ids = np.arange(i, end)[:, None]        # (B, 1)
                col_ids = np.arange(S)[None, :]             # (1, S)
                is_neighbour &= row_ids != col_ids
            else:
                row_ids = np.arange(i, end)[:, None]
                anch_ids = anchor_idx[None, :]
                is_neighbour &= row_ids != anch_ids

            neighbour_counts[i:end] = is_neighbour.sum(axis=1)

        denom = max(S - 1, 1) if S == n else max(S, 1)
        for i, cid in enumerate(ids):
            self._redundancy[cid] = neighbour_counts[i] / denom

        red_vals = list(self._redundancy.values())
        if red_vals:
            self._mean_redundancy = float(np.mean(red_vals))

        self._n_recomputes += 1

    # ── Stats ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return timing and scoring statistics."""
        avg_time = (
            self._total_eviction_time_s / self._n_evictions
            if self._n_evictions > 0 else 0.0
        )
        return {
            "n_evictions": self._n_evictions,
            "n_recomputes": self._n_recomputes,
            "avg_eviction_time_ms": round(avg_time * 1000, 4),
            "total_eviction_time_s": round(self._total_eviction_time_s, 3),
            "similarity_threshold": self.similarity_threshold,
            "recompute_interval": self.recompute_interval,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "dynamic_impute": self.dynamic_impute,
        }

    @property
    def name(self) -> str:
        return "semantic"

    def __repr__(self) -> str:
        return (
            f"SemanticPolicy(threshold={self.similarity_threshold}, "
            f"α={self.alpha}, β={self.beta}, μ={self.mu}, "
            f"recompute_every={self.recompute_interval}, "
            f"dyn_impute={self.dynamic_impute})"
        )
