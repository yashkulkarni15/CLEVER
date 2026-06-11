"""
ARC (Adaptive Replacement Cache) eviction policy — Megiddo & Modha 2003.

Classic ARC partitions a cache of capacity ``c`` into a recency list T1
(resident, seen once) and a frequency list T2 (resident, seen 2+), plus
two ghost lists B1/B2 holding metadata of entries recently evicted from
T1/T2.  An adaptation parameter ``p`` (the target size of T1) shifts the
T1/T2 split: a hit in B1 ("we evicted from the recency side too soon")
grows ``p``; a hit in B2 shrinks it.

Mapping onto CLEVER's eviction-only interface
---------------------------------------------
The policy does not control admission — the cache decides when to insert
and evict, and cache ids are unique per slot (an evicted id is never
re-inserted).  Classic ARC's "request for x currently in B1/B2" can
therefore never be observed by id equality.  This implementation adapts:

- **Ghost hits are detected semantically.**  Each ghost remembers the
  evicted entry's embedding; ``on_insert`` checks whether the incoming
  embedding lies within ``ghost_similarity_threshold`` (squared L2;
  the default 0.20 ≈ cosine 0.90 for unit-norm vectors) of a ghost.
  A match means the workload re-requested something recently evicted —
  the semantic-cache analogue of a classic ghost hit.  On a match, ``p``
  adapts with the classic deltas (+max(|B2|/|B1|, 1) for B1, clamped to
  c; −max(|B1|/|B2|, 1) for B2, clamped to 0), the matched ghost is
  retired, and the new entry goes straight to T2 (classic ARC moves a
  ghost-hit item to the MRU of T2).
- ``on_access`` of a T1 resident promotes it to T2 (classic T1 hit);
  a T2 resident moves to T2's MRU position.
- ``on_evict`` moves the victim's id+embedding to B1 (from T1) or B2
  (from T2); each ghost list is bounded by ``c`` (LRU-trimmed).
- ``select_victim`` implements classic REPLACE: evict T1's LRU when
  |T1| > p (or T2 is empty), otherwise T2's LRU.

Deviations from classic ARC
---------------------------
- No admission control, so the classic tweak "x ∈ B2 and |T1| == p →
  still evict from T1" is unavailable (``select_victim`` does not see
  the incoming request); the strict |T1| > p test is used instead.
- Ghost identity is approximate (embedding proximity), not exact key
  equality.
- ``capacity`` may be inferred as the max observed ``|active_ids|``
  (updated on each ``select_victim``, floored by the resident count)
  when not given explicitly.
- ``on_rebuild`` remaps every id present in the remap and drops the
  rest.  The harness only remaps live ids, so ghost lists are cleared
  on rebuild — required for safety, since compaction reuses ids.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class ARCPolicy(EvictionPolicy):
    """ARC eviction — adaptive balance of recency (T1) and frequency (T2)."""

    def __init__(
        self,
        capacity: Optional[int] = None,
        seed: Optional[int] = None,
        ghost_similarity_threshold: float = 0.20,
    ) -> None:
        """
        Args:
            capacity: Cache capacity ``c`` (ghost-list bound and ``p``
                clamp).  When ``None``, inferred online as the max
                observed ``|active_ids|``.
            seed: Unused; accepted for constructor-signature uniformity
                with the other policies.
            ghost_similarity_threshold: Max squared-L2 distance for an
                incoming embedding to count as a ghost hit.
        """
        self.capacity = capacity
        self.seed = seed
        self.ghost_similarity_threshold = float(ghost_similarity_threshold)

        # Resident lists (front = LRU, back = MRU).
        self._t1: OrderedDict[int, None] = OrderedDict()
        self._t2: OrderedDict[int, None] = OrderedDict()
        # Ghost lists: id → embedding of the evicted entry.
        self._b1: OrderedDict[int, np.ndarray] = OrderedDict()
        self._b2: OrderedDict[int, np.ndarray] = OrderedDict()
        # Embeddings of resident entries (needed to populate ghosts).
        self._embeddings: dict[int, np.ndarray] = {}
        # Adaptation parameter: target size of T1, 0 ≤ p ≤ c.
        self._p: float = 0.0
        self._observed_capacity = 0

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """T1 hit → promote to T2 (seen 2+); T2 hit → move to MRU."""
        if cache_id in self._t1:
            del self._t1[cache_id]
            self._t2[cache_id] = None
        elif cache_id in self._t2:
            self._t2.move_to_end(cache_id)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """New entry → MRU of T1; ghost hit → adapt p, MRU of T2."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
        self._forget(cache_id)  # defensive: id reuse must not alias state
        self._embeddings[cache_id] = emb

        match = self._match_ghost(emb)
        if match is None:
            self._t1[cache_id] = None
            return

        list_name, ghost_id = match
        self._adapt_p(list_name)
        ghosts = self._b1 if list_name == "b1" else self._b2
        ghosts.pop(ghost_id, None)
        self._t2[cache_id] = None

    def on_evict(self, cache_id: int) -> None:
        """Move the evicted resident into the matching ghost list."""
        emb = self._embeddings.pop(cache_id, None)
        if cache_id in self._t1:
            del self._t1[cache_id]
            if emb is not None:
                self._b1[cache_id] = emb
        elif cache_id in self._t2:
            del self._t2[cache_id]
            if emb is not None:
                self._b2[cache_id] = emb
        self._trim_ghosts()

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Classic REPLACE: T1's LRU while |T1| > p, else T2's LRU.

        Defensive: ids never seen by the policy fall back to the
        smallest active id (deterministic, always valid).
        """
        if not active_ids:
            return None
        self._observed_capacity = max(self._observed_capacity, len(active_ids))
        t1_active = [cid for cid in self._t1 if cid in active_ids]
        t2_active = [cid for cid in self._t2 if cid in active_ids]

        if t1_active and (len(t1_active) > self._p or not t2_active):
            return t1_active[0]
        if t2_active:
            return t2_active[0]
        return min(active_ids)

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap T1/T2/B1/B2 and embeddings; drop ids not in the map.

        Ghost ids are normally absent from ``id_remap`` (only live ids
        are remapped), so ghost lists are effectively cleared — stale
        ghost ids must not survive, since compaction reuses ids.
        ``p`` is preserved across rebuilds.
        """
        self._t1 = self._remap_ordered(self._t1, id_remap)
        self._t2 = self._remap_ordered(self._t2, id_remap)
        self._b1 = self._remap_ordered(self._b1, id_remap)
        self._b2 = self._remap_ordered(self._b2, id_remap)
        self._embeddings = {
            id_remap[old_id]: emb
            for old_id, emb in self._embeddings.items()
            if old_id in id_remap
        }

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _remap_ordered(
        items: OrderedDict, id_remap: dict[int, int],
    ) -> OrderedDict:
        """Remap keys preserving order; drop keys absent from the map."""
        remapped: OrderedDict[int, object] = OrderedDict()
        for old_id, value in items.items():
            if old_id in id_remap:
                remapped[id_remap[old_id]] = value
        return remapped

    def _forget(self, cache_id: int) -> None:
        """Purge any stale state for *cache_id* (defensive vs id reuse)."""
        self._t1.pop(cache_id, None)
        self._t2.pop(cache_id, None)
        self._b1.pop(cache_id, None)
        self._b2.pop(cache_id, None)
        self._embeddings.pop(cache_id, None)

    def _match_ghost(self, emb: np.ndarray) -> Optional[tuple[str, int]]:
        """Find the closest ghost within the similarity threshold.

        Returns ``("b1"|"b2", ghost_id)`` for the nearest matching ghost
        (squared L2 ≤ ``ghost_similarity_threshold``), or ``None``.
        Ties prefer B1, matching the order of evaluation.
        """
        candidates: list[tuple[float, str, int]] = []
        for list_name, ghosts in (("b1", self._b1), ("b2", self._b2)):
            if not ghosts:
                continue
            ids = list(ghosts)
            mat = np.stack([ghosts[gid] for gid in ids])
            diff = mat - emb[None, :]
            dists = np.einsum("ij,ij->i", diff, diff)
            best = int(np.argmin(dists))
            if float(dists[best]) <= self.ghost_similarity_threshold:
                candidates.append((float(dists[best]), list_name, ids[best]))

        if not candidates:
            return None
        dist, list_name, ghost_id = min(candidates, key=lambda t: t[0])
        return list_name, ghost_id

    def _adapt_p(self, hit_list: str) -> None:
        """Classic ARC deltas: B1 hit grows p, B2 hit shrinks it.

        Ghost-list sizes are measured with the matched ghost still
        resident, mirroring classic ARC (x ∈ B1/B2 during adaptation).
        """
        c = float(self._effective_capacity())
        n_b1, n_b2 = len(self._b1), len(self._b2)
        if hit_list == "b1":
            delta = max(n_b2 / n_b1, 1.0) if n_b1 else 1.0
            self._p = min(c, self._p + delta)
        else:
            delta = max(n_b1 / n_b2, 1.0) if n_b2 else 1.0
            self._p = max(0.0, self._p - delta)

    def _effective_capacity(self) -> int:
        """Explicit capacity, or the inferred one (≥ resident count, ≥ 1)."""
        if self.capacity is not None:
            return max(int(self.capacity), 1)
        resident = len(self._t1) + len(self._t2)
        return max(self._observed_capacity, resident, 1)

    def _trim_ghosts(self) -> None:
        """Bound each ghost list at capacity, dropping LRU ghosts."""
        c = self._effective_capacity()
        while len(self._b1) > c:
            self._b1.popitem(last=False)
        while len(self._b2) > c:
            self._b2.popitem(last=False)

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "arc"

    def __repr__(self) -> str:
        return (
            f"ARCPolicy(t1={len(self._t1)}, t2={len(self._t2)}, "
            f"b1={len(self._b1)}, b2={len(self._b2)}, p={self._p:.2f})"
        )
