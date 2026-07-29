"""
Unit tests for the FIFO (First-In-First-Out) eviction policy.

FIFO evicts in strict insertion order and ignores accesses entirely.
This is what separates it from LRU: an entry that is hit repeatedly
still leaves the cache at its original position in the queue.

Tests cover:
- Victim selection = oldest insertion, not oldest access
- on_access being a no-op (the discriminating property vs LRU)
- Re-insertion after eviction returning to the back of the queue
- Defensive select_victim behaviour (empty set, inactive ids)
- on_rebuild ID remapping preserving insertion order
- Registration in POLICY_REGISTRY and the harness factory

Run with: pytest tests/test_fifo.py -v
"""

import numpy as np

from src.cache.eviction.fifo import FIFOPolicy


# ── Helpers ──────────────────────────────────────────────────────────

def _emb(dim: int = 16) -> np.ndarray:
    """Zero embedding — FIFO ignores embeddings entirely."""
    return np.zeros(dim, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
# Insertion-order eviction
# ═════════════════════════════════════════════════════════════════════

class TestFIFOInsertionOrder:
    """Victims come out in the order they went in."""

    def test_evicts_first_inserted(self):
        """The oldest insertion is the victim."""
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        assert policy.select_victim({0, 1, 2}) == 0

    def test_evicts_in_strict_insertion_sequence(self):
        """Repeated evictions drain the queue front to back."""
        policy = FIFOPolicy()
        for i in range(4):
            policy.on_insert(i, _emb())

        active = {0, 1, 2, 3}
        order = []
        while active:
            victim = policy.select_victim(active)
            order.append(victim)
            policy.on_evict(victim)
            active.discard(victim)

        assert order == [0, 1, 2, 3]

    def test_evict_then_insert_advances_queue(self):
        """A new insert goes to the back, not the front."""
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        victim = policy.select_victim({0, 1, 2})
        policy.on_evict(victim)
        policy.on_insert(3, _emb())

        assert policy.select_victim({1, 2, 3}) == 1

    def test_reinsert_after_evict_goes_to_back(self):
        """A re-admitted id loses its original queue position."""
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        policy.on_evict(0)
        policy.on_insert(0, _emb())

        assert policy.select_victim({0, 1, 2}) == 1


# ═════════════════════════════════════════════════════════════════════
# Access independence (the property that distinguishes FIFO from LRU)
# ═════════════════════════════════════════════════════════════════════

class TestFIFOIgnoresAccess:
    """on_access must not reorder the queue."""

    def test_access_does_not_protect_entry(self):
        """Hitting the oldest entry does not save it (LRU would)."""
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        policy.on_access(0)

        assert policy.select_victim({0, 1, 2}) == 0

    def test_repeated_access_does_not_protect_entry(self):
        """Frequency is irrelevant too (LFU would protect this entry)."""
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        for _ in range(10):
            policy.on_access(0)

        assert policy.select_victim({0, 1, 2}) == 0

    def test_access_on_unknown_id_is_safe(self):
        """Accessing an id the policy never saw must not raise."""
        policy = FIFOPolicy()
        policy.on_insert(0, _emb())

        policy.on_access(99)

        assert policy.select_victim({0}) == 0


# ═════════════════════════════════════════════════════════════════════
# Defensive behaviour
# ═════════════════════════════════════════════════════════════════════

class TestFIFODefensive:
    """Edge cases the cache may hand the policy."""

    def test_single_entry(self):
        policy = FIFOPolicy()
        policy.on_insert(0, _emb())
        assert policy.select_victim({0}) == 0

    def test_empty_active_set_returns_none(self):
        policy = FIFOPolicy()
        policy.on_insert(0, _emb())
        assert policy.select_victim(set()) is None

    def test_skips_inactive_ids(self):
        """Stale queue entries are passed over, not returned."""
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        # 0 and 1 are no longer active but were never on_evict'ed.
        assert policy.select_victim({2}) == 2

    def test_evict_unknown_id_is_safe(self):
        policy = FIFOPolicy()
        policy.on_insert(0, _emb())

        policy.on_evict(99)

        assert policy.select_victim({0}) == 0

    def test_victim_is_always_in_active_set(self):
        policy = FIFOPolicy()
        active = {10, 20, 30}
        for cid in active:
            policy.on_insert(cid, _emb())

        assert policy.select_victim(active) in active


# ═════════════════════════════════════════════════════════════════════
# Rebuild remapping
# ═════════════════════════════════════════════════════════════════════

class TestFIFORebuild:
    """Cache compaction renumbers ids; insertion order must survive."""

    def test_on_rebuild_preserves_insertion_order(self):
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        # old 0 → new 2, old 1 → new 0, old 2 → new 1
        policy.on_rebuild({0: 2, 1: 0, 2: 1})

        # Oldest insertion was old-0, now numbered 2.
        assert policy.select_victim({0, 1, 2}) == 2

    def test_on_rebuild_drops_stale_ids(self):
        policy = FIFOPolicy()
        for i in range(3):
            policy.on_insert(i, _emb())

        # old 0 is gone; only 1 and 2 survive compaction.
        policy.on_rebuild({1: 0, 2: 1})

        assert policy.select_victim({0, 1}) == 0


# ═════════════════════════════════════════════════════════════════════
# Naming and registration
# ═════════════════════════════════════════════════════════════════════

class TestFIFORegistration:
    """The harness resolves policies by name in two places."""

    def test_name_is_fifo(self):
        assert FIFOPolicy().name == "fifo"

    def test_registered_in_policy_registry(self):
        from src.cache.eviction import POLICY_REGISTRY

        assert POLICY_REGISTRY["fifo"] is FIFOPolicy

    def test_harness_factory_builds_fifo(self):
        """08_run_eviction.py hard-codes an if/else, bypassing the registry."""
        import importlib.util
        from pathlib import Path

        script = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "08_run_eviction.py"
        )
        spec = importlib.util.spec_from_file_location("run_eviction", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        policy = module.create_policy(
            "fifo",
            config={},
            cache_embs=np.zeros((2, 16), dtype=np.float32),
            cache_ids=[0, 1],
            stream_embs=np.zeros((2, 16), dtype=np.float32),
        )

        assert isinstance(policy, FIFOPolicy)
