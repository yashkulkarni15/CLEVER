"""
Greedy-Dual-Size-Frequency (GDSF) eviction policy (Cherkasova, 1998).

Each entry gets a priority::

    priority(e) = L + frequency(e) * cost(e) / size(e)

where ``L`` is the global inflation clock ("aging"): whenever a victim
is evicted, ``L`` is raised to the victim's priority, so entries
inserted/accessed later start from a higher baseline and old, cold
entries gradually lose out — this is what distinguishes GDSF from pure
LFU.

Project-specific cost/size (per HANDOFF.md):
- ``cost(e) = 1 / ||embedding||``
- ``size(e) = embedding dimension``

Note for the paper: CLEVER's embeddings are unit-norm 384-d vectors, so
``cost(e) ≈ 1`` and ``size(e)`` is constant across entries.  On this
workload the cost/size term is therefore uniform and GDSF reduces to
**LFU-with-aging**.  The full general formula is implemented anyway
(cost from the actual embedding norm, size from the actual dimension)
so the policy remains correct for non-uniform embeddings.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class GDSFPolicy(EvictionPolicy):
    """GDSF eviction — evict the entry with the lowest inflated priority.

    Tie-breaking: among entries with the same priority, evict the one
    that was inserted *earliest* (preserved via insertion order in
    ``_freq``), matching the LFU policy's convention.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        # ``seed`` is accepted for constructor uniformity across
        # policies; GDSF is deterministic and does not use it.
        self._seed = seed
        # Global inflation clock: raised to the victim's priority on
        # each eviction, so later entries start from a higher baseline.
        self._L: float = 0.0
        # OrderedDict preserves insertion order for tie-breaking.
        # Per entry: [frequency, cost/size ratio, priority].
        self._entries: OrderedDict[int, list[float]] = OrderedDict()

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Increment frequency and recompute priority at the current L."""
        entry = self._entries.get(cache_id)
        if entry is not None:
            entry[0] += 1
            entry[2] = self._L + entry[0] * entry[1]

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """New entry starts with frequency = 1 at the current L.

        cost(e) = 1/||e|| and size(e) = dim(e) are computed here, once,
        from the actual embedding.  Degenerate inputs (zero norm or
        zero dimension) fall back to cost = 1 / size = 1 so the
        priority stays finite.
        """
        norm = float(np.linalg.norm(embedding))
        cost = 1.0 / norm if norm > 0.0 else 1.0
        size = embedding.size if embedding.size > 0 else 1
        ratio = cost / size
        self._entries[cache_id] = [1, ratio, self._L + ratio]

    def on_evict(self, cache_id: int) -> None:
        """Drop the evicted entry and inflate L to its priority."""
        entry = self._entries.pop(cache_id, None)
        if entry is not None:
            # max() guards against the cache evicting a non-minimum
            # entry; L must never decrease.
            self._L = max(self._L, entry[2])

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Evict the entry with the lowest priority.

        Ties at the minimum are broken by insertion order (the first
        entry encountered in ``_entries`` is the oldest-inserted).
        Only ids in ``active_ids`` are considered; if none of them is
        tracked, returns ``None``.
        """
        victim_id: Optional[int] = None
        min_priority = float("inf")

        for cid, (_, _, priority) in self._entries.items():
            if cid not in active_ids:
                continue
            if priority < min_priority:
                min_priority = priority
                victim_id = cid

        return victim_id

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap keys, preserving insertion order, priorities, and L.

        IDs absent from ``id_remap`` are stale (dead after compaction)
        and dropped.  The inflation clock ``L`` is untouched: aging is
        a property of the eviction history, not of the id numbering.
        """
        new_entries: OrderedDict[int, list[float]] = OrderedDict()
        for old_id, entry in self._entries.items():
            if old_id in id_remap:
                new_entries[id_remap[old_id]] = entry
        self._entries = new_entries

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "gdsf"

    def __repr__(self) -> str:
        return f"GDSFPolicy(tracked={len(self._entries)}, L={self._L:.4f})"
