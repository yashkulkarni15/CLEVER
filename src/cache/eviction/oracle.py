"""
Oracle (Belady's optimal) eviction policy.

Evicts the entry whose next access is furthest in the future.  This
requires knowledge of the entire future query stream and therefore
cannot be used online -- it serves purely as a theoretical upper bound.

Implementation
--------------
The oracle maintains a mapping from each active cache entry to a queue
of future stream positions where that entry would be the nearest
neighbor (within the similarity threshold).

Because the cache *mutates* during the stream (entries are inserted on
miss, evicted to make room), nearest-neighbor relationships change over
time.  A one-shot pre-computation at construction becomes stale.

To stay correct, the oracle performs a **periodic full refresh**: every
``refresh_interval`` evictions it builds a fresh FAISS flat index of
all *current* active entries, batch-searches the remaining stream
against it, and rebuilds every entry's future-use queue from scratch.

Complexity
----------
- Full refresh: O(S * C) via FAISS flat search  (S = remaining stream, C = active cache)
- ``select_victim``: O(C)  (linear scan of active entries)
- Amortized refresh cost per eviction: O(S * C / refresh_interval)
"""

import collections
import logging
from typing import Optional

import faiss
import numpy as np

from src.cache.eviction.base import EvictionPolicy

logger = logging.getLogger(__name__)

_INF = float("inf")


class OraclePolicy(EvictionPolicy):
    """Belady's optimal eviction -- evict the entry used furthest away.

    Args:
        future_stream_embeddings: Shape (S, D) -- the full future query
            stream in chronological order.
        cache_embeddings: Shape (C, D) -- the initial cache entries.
        cache_ids: Corresponding cache IDs (typically ``list(range(C))``).
        similarity_threshold: L2-squared distance threshold -- a future
            query "accesses" a cache entry only if the NN distance is
            below this value.
        refresh_interval: Number of evictions between full future-use
            rebuilds.  Lower = more accurate but slower.
        use_gpu: If True, use GPU-accelerated FAISS for batch search
            in ``_full_refresh()``.  Requires ``faiss-gpu``.
    """

    def __init__(
        self,
        future_stream_embeddings: np.ndarray,
        cache_embeddings: np.ndarray,
        cache_ids: list[int],
        similarity_threshold: float = 0.90,
        refresh_interval: int = 100,
        horizon: int = 0,
        use_gpu: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.refresh_interval = refresh_interval
        self.horizon = horizon  # 0 = unlimited lookahead

        # GPU acceleration for batch search in _full_refresh().
        self._use_gpu = use_gpu
        self._gpu_res = None
        if use_gpu:
            try:
                self._gpu_res = faiss.StandardGpuResources()
                logger.info("Oracle: GPU-accelerated FAISS enabled")
            except (AttributeError, RuntimeError):
                logger.warning("Oracle: faiss-gpu not available, falling back to CPU")
                self._use_gpu = False

        # Full stream (read-only reference for refreshes).
        self._stream_embs = future_stream_embeddings
        self._stream_pos: int = 0

        # Active cache state.
        self._active_ids: set[int] = set(cache_ids)
        self._cache_embs: dict[int, np.ndarray] = {}
        for cid, emb in zip(cache_ids, cache_embeddings):
            self._cache_embs[cid] = emb

        # Future-use queues: cache_id -> deque of future stream indices.
        self._future_uses: dict[int, collections.deque] = {}
        # Scalar next-use for fast victim selection.
        self._next_use: dict[int, float] = {}

        # Refresh bookkeeping.
        self._evictions_since_refresh: int = 0
        self._n_refreshes: int = 0

        # Initial full refresh to populate future-use queues.
        self._full_refresh()

    # ── Core: periodic full refresh ───────────────────────────────

    def _full_refresh(self) -> None:
        """Rebuild all future-use queues from the current cache state.

        Builds a FAISS flat index of active entries, searches the
        remaining stream against it, and records which cache entry each
        future query would hit.
        """
        active_list = sorted(self._active_ids)

        # Reset queues for all active entries.
        for cid in active_list:
            self._future_uses[cid] = collections.deque()

        if not active_list or self._stream_pos >= len(self._stream_embs):
            for cid in active_list:
                self._next_use[cid] = _INF
            return

        # Build flat index of current active entries.
        active_embs = np.array(
            [self._cache_embs[cid] for cid in active_list],
            dtype=np.float32,
        )
        dim = active_embs.shape[1]
        cpu_index = faiss.IndexFlatL2(dim)
        if self._use_gpu and self._gpu_res is not None:
            index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
        else:
            index = cpu_index
        index.add(active_embs)

        # Batch-search remaining stream against active cache.
        remaining = self._stream_embs[self._stream_pos:]
        if self.horizon > 0:
            remaining = remaining[:self.horizon]
        n_remaining = len(remaining)
        batch_size = 10_000

        logger.info(
            f"Oracle refresh #{self._n_refreshes + 1}: "
            f"{n_remaining} stream queries × {len(active_list)} cache entries"
            f"{f' (horizon={self.horizon})' if self.horizon > 0 else ''}"
        )

        for batch_start in range(0, n_remaining, batch_size):
            batch_end = min(batch_start + batch_size, n_remaining)
            batch = remaining[batch_start:batch_end].astype(np.float32)
            dists, idxs = index.search(batch, k=1)

            # Vectorized filtering: only iterate over valid matches.
            nn_idxs = idxs[:, 0]
            nn_dists = dists[:, 0]
            valid_mask = (nn_idxs >= 0) & (nn_dists <= self.similarity_threshold)
            valid_positions = np.where(valid_mask)[0]

            for i in valid_positions:
                cid = active_list[nn_idxs[i]]
                stream_idx = self._stream_pos + batch_start + int(i)
                self._future_uses[cid].append(stream_idx)

        # Sync scalar next_use from queues.
        for cid in active_list:
            self._sync_next_use(cid)

        self._n_refreshes += 1
        self._evictions_since_refresh = 0

    def _sync_next_use(self, cid: int) -> None:
        """Set ``_next_use[cid]`` from the front of its queue."""
        q = self._future_uses.get(cid)
        if not q:
            self._next_use[cid] = _INF
            return

        # Drain any entries that are strictly in the past.
        while q and q[0] < self._stream_pos:
            q.popleft()

        self._next_use[cid] = float(q[0]) if q else _INF

    # ── Lifecycle hooks ───────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Pop the current use and advance to next future use."""
        q = self._future_uses.get(cache_id)
        if q:
            # Pop the current access (at stream_pos) and any earlier ones.
            while q and q[0] <= self._stream_pos:
                q.popleft()
        self._sync_next_use(cache_id)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Register a newly inserted entry with a protective next_use.

        The oracle evicts entries with the HIGHEST ``_next_use`` (furthest
        in the future).  We must NOT assign ``_INF`` here because INF means
        "never used again" — the worst possible victim — which would cause
        every newly inserted entry to be immediately evicted on the next miss,
        creating a thrashing loop.

        Instead we assign ``stream_pos`` (the current query index) as a
        temporary sentinel.  Since stream_pos < any future index, this entry
        effectively appears as "next used right now" → lowest eviction priority.

        The sentinel is replaced with the real next_use at the next periodic
        full refresh (governed by ``refresh_interval``).
        """
        self._active_ids.add(cache_id)
        self._cache_embs[cache_id] = embedding.copy()
        self._future_uses[cache_id] = collections.deque()

        # Protective sentinel: appears "just accessed", won't be chosen
        # as victim (oracle picks MAX next_use, not MIN).
        self._next_use[cache_id] = float(self._stream_pos)

    def on_evict(self, cache_id: int) -> None:
        """Clean up state and trigger refresh when interval is reached."""
        self._active_ids.discard(cache_id)
        self._next_use.pop(cache_id, None)
        self._cache_embs.pop(cache_id, None)
        self._future_uses.pop(cache_id, None)

        self._evictions_since_refresh += 1
        if self._evictions_since_refresh >= self.refresh_interval:
            self._full_refresh()

    def advance_stream(self, new_pos: int) -> None:
        """Update the current stream position (called by the runner)."""
        self._stream_pos = new_pos

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Evict the entry whose next access is furthest in the future."""
        if not active_ids:
            return None

        victim: Optional[int] = None
        max_next_use = -1.0

        for cid in active_ids:
            nu = self._next_use.get(cid, _INF)
            if nu > max_next_use:
                max_next_use = nu
                victim = cid

        return victim

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap all internal state after cache compaction."""
        # Active IDs.
        self._active_ids = {
            id_remap[old] for old in self._active_ids if old in id_remap
        }

        # Cache embeddings.
        new_embs: dict[int, np.ndarray] = {}
        for old_id, emb in self._cache_embs.items():
            if old_id in id_remap:
                new_embs[id_remap[old_id]] = emb
        self._cache_embs = new_embs

        # Future-use queues.
        new_futures: dict[int, collections.deque] = {}
        for old_id, q in self._future_uses.items():
            if old_id in id_remap:
                new_futures[id_remap[old_id]] = q
        self._future_uses = new_futures

        # Next-use scalars.
        new_next: dict[int, float] = {}
        for old_id, nu in self._next_use.items():
            if old_id in id_remap:
                new_next[id_remap[old_id]] = nu
        self._next_use = new_next

        # Force a full refresh to re-establish correctness.
        self._full_refresh()

    # ── Representation ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "oracle"

    @property
    def stats(self) -> dict:
        """Return oracle-specific statistics."""
        return {
            "n_refreshes": self._n_refreshes,
            "refresh_interval": self.refresh_interval,
            "horizon": self.horizon,
            "stream_len": len(self._stream_embs),
            "active_entries": len(self._active_ids),
        }

    def __repr__(self) -> str:
        return (
            f"OraclePolicy(threshold={self.similarity_threshold}, "
            f"refresh_interval={self.refresh_interval}, "
            f"horizon={self.horizon}, "
            f"stream_len={len(self._stream_embs)}, "
            f"active={len(self._active_ids)}, "
            f"refreshes={self._n_refreshes})"
        )
