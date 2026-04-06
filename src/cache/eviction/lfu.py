"""
Least-Frequently-Used (LFU) eviction policy.

Evicts the entry with the fewest total accesses.  Ties are broken by
insertion order (oldest inserted first) — this prevents pathological
cases where a newly inserted entry with count 0 is immediately evicted
before it ever receives a hit.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class LFUPolicy(EvictionPolicy):
    """LFU eviction — evict the least frequently accessed entry.

    Tie-breaking: among entries with the same access count, evict the
    one that was inserted *earliest* (preserved via insertion order in
    ``_counts``).
    """

    def __init__(self) -> None:
        # OrderedDict preserves insertion order for tie-breaking.
        # Values are access counts.
        self._counts: OrderedDict[int, int] = OrderedDict()

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Increment access count."""
        if cache_id in self._counts:
            self._counts[cache_id] += 1

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """New entry starts with count = 0."""
        self._counts[cache_id] = 0

    def on_evict(self, cache_id: int) -> None:
        """Remove evicted entry from counts."""
        self._counts.pop(cache_id, None)

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Evict the entry with the lowest access count.

        Among entries tied at the minimum count, evict the first one
        in insertion order (front of the OrderedDict subset).

        Single-pass: ``_counts`` is an ``OrderedDict`` that preserves
        insertion order, so the first entry encountered at the global
        minimum count is guaranteed to be the oldest-inserted entry
        with that count.  We therefore do two logical steps in one
        loop: (1) find the global minimum, and (2) record the first
        entry that has it.
        """
        victim_id: Optional[int] = None
        min_count = float("inf")

        for cid in self._counts:
            if cid not in active_ids:
                continue
            cnt = self._counts[cid]
            if cnt < min_count:
                min_count = cnt
                victim_id = cid

        return victim_id

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap keys, preserving insertion order and counts."""
        new_counts: OrderedDict[int, int] = OrderedDict()
        for old_id, count in self._counts.items():
            if old_id in id_remap:
                new_counts[id_remap[old_id]] = count
        self._counts = new_counts

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "lfu"

    def __repr__(self) -> str:
        return f"LFUPolicy(tracked={len(self._counts)})"
