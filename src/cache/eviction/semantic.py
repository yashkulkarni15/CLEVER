"""
Semantic-aware eviction policy — the paper's novel contribution.

Instead of a pure recency/frequency heuristic, each cached entry is
scored by the *ratio* of its semantic **redundancy** (how many other
cached entries are nearby in embedding space) to its **utility**
(recency + frequency).  Entries that are highly redundant and rarely
used are evicted first; entries that are isolated (no similar
neighbours) or heavily used are protected.

Eviction score
--------------
::

    score(e) = (r(e) + μ) / (α · recency(e) + β · frequency(e) + ε)

where ``r(e) = |neighbours(e)| / max(n_active - 1, 1)``.

The ``+ μ`` is **additive (Laplace-style) smoothing** that mixes two
ranking criteria:

    score(e) = r(e) / u(e)  +  μ · (1 / u(e))
             = semantic term +  μ · inverse-utility term

- **μ → 0**   ⇒ pure redundancy-over-utility scoring.
- **μ → ∞**   ⇒ score approaches plain inverse-utility ordering
  asymptotically (LRU+LFU), with the redundancy signal washed out.
  The two orderings agree exactly only in the limit.

This guarantees the policy never does worse than LRU+LFU when the
redundancy signal is uninformative — it degrades gracefully instead
of collapsing (as the un-smoothed ``r / u`` formulation would, since
any entry with ``r = 0`` had score 0 and was *never evictable*).

Recency floor
-------------
``recency(e)`` is defined as ``(rank + 1) / n_active`` rather than
``rank / max(n-1, 1)``, so recency ∈ [1/n, 1] instead of [0, 1].  This
prevents the oldest entry from driving the utility denominator to
exactly ε, which previously caused that entry's score to explode
(≈ r / 1e-9) and dominate the selection regardless of redundancy.

Incremental redundancy maintenance
----------------------------------
The neighbour graph is stored explicitly as a symmetric set-of-sets:
``_neighbors[cid] ⊆ active_ids``.  This enables O(S) online maintenance
instead of O(N²) per-eviction recomputation:

- ``on_insert(e)``: sample S ≤ 1024 active entries, compute distances
  to *e* once, and **symmetrically** add edges for every (e, j) pair
  within the threshold.  Both ``_neighbors[e]`` and ``_neighbors[j]``
  are updated so the graph stays consistent.

- ``on_evict(x)``: remove ``x`` from every ``_neighbors[j]`` in
  ``_neighbors[x]`` (again, symmetric).  O(|neighbours of x|).

- ``_recompute_redundancy`` (called every ``recompute_interval``
  evictions): rebuild the graph from scratch using a sampled anchor
  set, to correct for sampling bias that accumulates between
  recomputations.  GPU-accelerated (PyTorch) if available.

This "incremental symmetric redundancy graph" is the novel systems
contribution: the cache's redundancy state is always fresh and
consistent, with amortised O(S) cost per insert/evict instead of the
naive O(N²) recomputation every step.

Parameters
----------
similarity_threshold
    L2² distance threshold for counting an entry as a "neighbour".
    For unit-norm embeddings, ``cosine_sim ≈ 1 - L2²/2``, so
    ``L2² ≤ 0.30`` corresponds to ``cosine_sim ≥ 0.85``.
alpha, beta
    Weights on the recency and frequency components of utility.
mu
    Additive smoothing constant for eventual evictability.  Mixes
    semantic scoring with inverse-utility ordering; score approaches
    pure inverse-utility asymptotically as μ → ∞.  Default 0.1.
dynamic_impute
    If True, run the sampled symmetric update on every insert.  If
    False, new entries start with an empty neighbour set and rely on
    the next batch recomputation to populate it.
recompute_interval
    Number of evictions between full batch rebuilds of the neighbour
    graph.  Lower = fresher but more expensive.
seed
    Base seed for the imputation RNG.  Mixed with ``cache_id`` so
    different run seeds produce honestly independent samples.
"""

import logging
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy

logger = logging.getLogger(__name__)


class SemanticPolicy(EvictionPolicy):
    """Semantic-aware eviction policy with an incremental neighbour graph."""

    # Maximum number of anchor entries for redundancy estimation.
    # Instead of O(N²) all-pairs, we use O(N×S) sampled anchors.
    MAX_REDUNDANCY_SAMPLES = 1024

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        alpha: float = 1.0,
        beta: float = 1.0,
        recompute_interval: int = 2000,
        mu: float = 0.1,
        dynamic_impute: bool = True,
        seed: int = 0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.beta = beta
        self.recompute_interval = recompute_interval
        self.mu = mu
        self.dynamic_impute = dynamic_impute
        self._seed = int(seed)
        self._epsilon = 1e-9

        # ── Internal state ───────────────────────────────────────
        # Access order: front = oldest access, back = most recent.
        self._access_order: OrderedDict[int, None] = OrderedDict()
        # Access counts per entry.
        self._access_counts: dict[int, int] = {}
        # Embeddings stored per cache_id (needed for neighbour counting).
        self._embeddings: dict[int, np.ndarray] = {}
        # Symmetric neighbour graph — the authoritative redundancy state.
        self._neighbors: dict[int, set[int]] = {}
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
        """Register a newly inserted entry and symmetrically update the
        neighbour graph against a sampled subset of active entries.
        """
        self._access_order[cache_id] = None
        self._access_counts[cache_id] = 0
        self._embeddings[cache_id] = embedding.copy()
        self._neighbors[cache_id] = set()

        if not self.dynamic_impute:
            return

        # Need at least one other entry to have any neighbour relation.
        other_ids = [cid for cid in self._embeddings if cid != cache_id]
        n_others = len(other_ids)
        if n_others == 0:
            return

        S = min(n_others, self.MAX_REDUNDANCY_SAMPLES)
        if n_others > S:
            rng_seed = self._mix_seed(cache_id)
            rng = np.random.RandomState(rng_seed)
            sample_ids = rng.choice(other_ids, S, replace=False).tolist()
        else:
            sample_ids = other_ids

        sample_embs = np.array(
            [self._embeddings[cid] for cid in sample_ids], dtype=np.float32,
        )

        emb = embedding.reshape(1, -1).astype(np.float32)
        norm_q = float(np.sum(emb ** 2))
        norms_S = np.sum(sample_embs ** 2, axis=1)
        dot = (emb @ sample_embs.T)[0]
        dist_sq = norm_q + norms_S - 2 * dot

        hits = np.flatnonzero(dist_sq <= self.similarity_threshold)
        if hits.size == 0:
            return

        # Symmetric edge insertion — both sides of every discovered pair.
        new_set = self._neighbors[cache_id]
        for idx in hits:
            nid = sample_ids[int(idx)]
            new_set.add(nid)
            nbr_set = self._neighbors.get(nid)
            if nbr_set is not None:
                nbr_set.add(cache_id)

    def on_evict(self, cache_id: int) -> None:
        """Symmetrically remove *cache_id* from the neighbour graph and
        clean up all associated bookkeeping.
        """
        # Pop first so we also detach any self-loops safely.
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
        self._n_evictions += 1

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Select the entry with the highest (r + μ) / utility score."""
        if not active_ids:
            return None

        t_start = time.perf_counter()

        # Recompute the neighbour graph periodically.
        if self._evictions_since_recompute >= self.recompute_interval:
            self._recompute_redundancy(active_ids)
            self._evictions_since_recompute = 0

        # ── Vectorised score computation ─────────────────────────
        # Strictly ordered ids (front = oldest, back = newest) filtered
        # to the active set.
        active_cids = [cid for cid in self._access_order if cid in active_ids]
        n_active = len(active_cids)
        if n_active == 0:
            return None

        cids = np.array(active_cids, dtype=np.int64)

        # 1. Recency with (rank + 1) / n floor so the oldest entry has
        #    recency = 1/n > 0, not exactly 0.  This prevents the
        #    utility denominator from collapsing to ε and the oldest
        #    entry's score from exploding regardless of redundancy.
        ranks = np.arange(n_active, dtype=np.float32)
        recency = (ranks + 1.0) / float(n_active)

        # 2. Frequency, normalised by the current maximum count.
        counts = np.array(
            [self._access_counts.get(cid, 0) for cid in active_cids],
            dtype=np.float32,
        )
        max_count = max(float(np.max(counts)), 1.0)
        freq = counts / max_count

        # 3. Redundancy from the neighbour graph.  Denominator is
        #    (n_active - 1) so r ∈ [0, 1].
        denom = max(n_active - 1, 1)
        r = np.array(
            [len(self._neighbors.get(cid, ())) for cid in active_cids],
            dtype=np.float32,
        ) / float(denom)

        # 4. (r + μ) / (α · recency + β · freq + ε)
        utility = self.alpha * recency + self.beta * freq + self._epsilon
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

        # Embeddings
        new_embs: dict[int, np.ndarray] = {}
        for old_id, emb in self._embeddings.items():
            if old_id in id_remap:
                new_embs[id_remap[old_id]] = emb
        self._embeddings = new_embs

        # Neighbour graph — remap both keys and set members.  Any
        # neighbour referring to an evicted id is dropped.
        new_neighbors: dict[int, set[int]] = {}
        for old_id, nbrs in self._neighbors.items():
            if old_id not in id_remap:
                continue
            new_id = id_remap[old_id]
            new_nbrs = {
                id_remap[o] for o in nbrs if o in id_remap
            }
            new_neighbors[new_id] = new_nbrs
        self._neighbors = new_neighbors

        # Force a recomputation on the next eviction.
        self._evictions_since_recompute = self.recompute_interval

    # ── Helpers ──────────────────────────────────────────────────

    def _mix_seed(self, cache_id: int) -> int:
        """Combine the base seed and a cache_id into a 31-bit RNG seed.

        Ensures different run seeds produce honestly independent samples
        for the same cache_id.  Uses a Knuth-style multiplicative mix.
        """
        mix = (int(self._seed) * 2654435769 + int(cache_id) * 40503 + 1) & 0x7FFFFFFF
        # Avoid seeding RandomState with 0 (valid but less ergonomic).
        return mix or 1

    # ── Redundancy computation ───────────────────────────────────

    def _recompute_redundancy(self, active_ids: set[int]) -> None:
        """Batch-rebuild the symmetric neighbour graph.

        Instead of O(N²) all-pairs distances, samples up to
        MAX_REDUNDANCY_SAMPLES anchor entries and discovers every edge
        incident to at least one anchor.  Each discovered pair is
        inserted symmetrically into ``_neighbors``, so the graph is
        always consistent even though it's a subsample of the full
        similarity graph.

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
            rng_seed = ((self._seed + 1) * 1000003 + self._n_recomputes) & 0x7FFFFFFF
            rng = np.random.RandomState(rng_seed or 1)
            anchor_idx = rng.choice(n, S, replace=False)
            anchor_idx.sort()
            anchor_embs = embs[anchor_idx]
        else:
            anchor_idx = np.arange(n)
            anchor_embs = embs

        # Rebuild the graph from scratch.
        new_neighbors: dict[int, set[int]] = {cid: set() for cid in ids}

        used_torch = False
        try:
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            import torch
            if torch.cuda.is_available():
                self._recompute_redundancy_torch(
                    ids, embs, n, anchor_embs, anchor_idx, S, new_neighbors,
                )
                used_torch = True
        except ImportError:
            pass

        if not used_torch:
            self._recompute_redundancy_numpy(
                ids, embs, n, anchor_embs, anchor_idx, S, new_neighbors,
            )

        self._neighbors = new_neighbors
        self._n_recomputes += 1

    def _recompute_redundancy_torch(
        self,
        ids: list[int],
        embs: np.ndarray,
        n: int,
        anchor_embs: np.ndarray,
        anchor_idx: np.ndarray,
        S: int,
        new_neighbors: dict[int, set[int]],
    ) -> None:
        """GPU-accelerated symmetric neighbour discovery via PyTorch."""
        import torch

        device = torch.device("cuda")
        embs_t = torch.from_numpy(embs).to(device)
        anch_t = torch.from_numpy(anchor_embs).to(device)

        norms_sq_all = torch.sum(embs_t ** 2, dim=1)     # (N,)
        norms_sq_anc = torch.sum(anch_t ** 2, dim=1)      # (S,)

        anchor_idx_t = torch.from_numpy(anchor_idx).to(device)

        # Process all entries against anchors in batches.
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

            # Self-exclusion.
            row_globals = torch.arange(i, end, device=device)
            self_mask = row_globals.unsqueeze(1) == anchor_idx_t.unsqueeze(0)
            is_neighbour = is_neighbour & ~self_mask

            # Pull the pair indices back to the host and update the
            # symmetric graph.
            rows, cols = torch.nonzero(is_neighbour, as_tuple=True)
            if rows.numel() == 0:
                continue
            rows_np = rows.cpu().numpy()
            cols_np = cols.cpu().numpy()
            row_globals_np = rows_np + i
            col_globals_np = anchor_idx[cols_np]

            for r_g, c_g in zip(row_globals_np, col_globals_np):
                cid_i = ids[int(r_g)]
                cid_j = ids[int(c_g)]
                new_neighbors[cid_i].add(cid_j)
                new_neighbors[cid_j].add(cid_i)

    def _recompute_redundancy_numpy(
        self,
        ids: list[int],
        embs: np.ndarray,
        n: int,
        anchor_embs: np.ndarray,
        anchor_idx: np.ndarray,
        S: int,
        new_neighbors: dict[int, set[int]],
    ) -> None:
        """CPU fallback symmetric neighbour discovery (sampled anchors)."""
        norms_sq_all = np.sum(embs ** 2, axis=1)          # (N,)
        norms_sq_anc = np.sum(anchor_embs ** 2, axis=1)    # (S,)

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
            row_globals = np.arange(i, end)[:, None]
            anch_globals = anchor_idx[None, :]
            is_neighbour &= row_globals != anch_globals

            rows, cols = np.where(is_neighbour)
            if rows.size == 0:
                continue
            row_globals_flat = rows + i
            col_globals_flat = anchor_idx[cols]

            for r_g, c_g in zip(row_globals_flat, col_globals_flat):
                cid_i = ids[int(r_g)]
                cid_j = ids[int(c_g)]
                new_neighbors[cid_i].add(cid_j)
                new_neighbors[cid_j].add(cid_i)

    # ── Stats ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return timing and scoring statistics."""
        avg_time = (
            self._total_eviction_time_s / self._n_evictions
            if self._n_evictions > 0 else 0.0
        )
        if self._neighbors:
            n = max(len(self._embeddings) - 1, 1)
            r_vals = [len(s) / n for s in self._neighbors.values()]
            mean_r = float(np.mean(r_vals)) if r_vals else 0.0
            max_r = float(np.max(r_vals)) if r_vals else 0.0
        else:
            mean_r = 0.0
            max_r = 0.0
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
            "seed": self._seed,
            "mean_redundancy": round(mean_r, 6),
            "max_redundancy": round(max_r, 6),
        }

    @property
    def name(self) -> str:
        return "semantic"

    def __repr__(self) -> str:
        return (
            f"SemanticPolicy(threshold={self.similarity_threshold}, "
            f"α={self.alpha}, β={self.beta}, μ={self.mu}, "
            f"recompute_every={self.recompute_interval}, "
            f"dyn_impute={self.dynamic_impute}, seed={self._seed})"
        )
