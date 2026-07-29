"""
First-In-First-Out (FIFO) eviction policy.

Evicts the entry that was inserted longest ago, regardless of how often
or how recently it has been read.  Accesses are ignored entirely, which
is the only thing separating FIFO from LRU: under LRU a hit moves an
entry to the back of the queue, under FIFO it does not move at all.

FIFO is the weakest reasonable baseline.  It is included because
recency and frequency signals are worth nothing on a workload with no
temporal locality, and a policy that exploits neither is the control
that shows this.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class FIFOPolicy(EvictionPolicy):
    """FIFO eviction — evict the oldest insertion."""

    def __init__(self) -> None:
        # Front = first inserted, back = most recently inserted.
        self._queue: OrderedDict[int, None] = OrderedDict()

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """No-op.  FIFO does not react to reads."""

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Append the new entry at the back of the queue."""
        self._queue[cache_id] = None

    def on_evict(self, cache_id: int) -> None:
        """Remove the evicted entry from the queue."""
        self._queue.pop(cache_id, None)

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Return the oldest insertion that is still active.

        Iterates from the front and skips ids the cache has already
        dropped without telling us, mirroring ``LRUPolicy``.
        """
        for cid in self._queue:
            if cid in active_ids:
                return cid
        return None

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap queue keys, preserving insertion order."""
        new_queue: OrderedDict[int, None] = OrderedDict()
        for old_id in self._queue:
            if old_id in id_remap:
                new_queue[id_remap[old_id]] = None
        self._queue = new_queue

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "fifo"

    def __repr__(self) -> str:
        return f"FIFOPolicy(tracked={len(self._queue)})"
