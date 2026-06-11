"""
Unit tests for the GDSF (Greedy-Dual-Size-Frequency) eviction policy.

Tests cover:
- Victim selection = lowest priority (frequency dominates when
  cost/size is uniform)
- on_access frequency increments raising priority
- The global inflation clock L (aging) distinguishing GDSF from LFU
- Cost/size computed from the actual embedding (norm and dimension)
- Defensive select_victim behaviour (empty set, unseen ids)
- on_rebuild ID remapping (stale ids dropped, L preserved)

Run with: pytest tests/test_gdsf.py -v
"""

import numpy as np

from src.cache.eviction.gdsf import GDSFPolicy


# ── Helpers ──────────────────────────────────────────────────────────

def _unit_emb() -> np.ndarray:
    """1-d unit-norm embedding: cost = 1, size = 1 → cost/size = 1."""
    return np.array([1.0], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
# Priority ordering (frequency dominates with uniform cost/size)
# ═════════════════════════════════════════════════════════════════════

class TestGDSFPriorityOrdering:
    """With uniform cost/size, the lowest-frequency entry loses."""

    def test_evicts_lowest_priority_entry(self):
        """Entry with the fewest accesses has the lowest priority."""
        policy = GDSFPolicy(seed=42)
        for i in range(3):
            policy.on_insert(i, _unit_emb())

        # freq: 0 → 3, 1 → 2, 2 → 1
        policy.on_access(0)
        policy.on_access(0)
        policy.on_access(1)

        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, "Entry 2 (lowest frequency) should be evicted"

    def test_access_raises_priority_above_unaccessed_peer(self):
        """A single on_access must protect an entry over an untouched one."""
        policy = GDSFPolicy()
        policy.on_insert(0, _unit_emb())
        policy.on_insert(1, _unit_emb())

        # Without the access, the tie-break would evict 0 (older).
        policy.on_access(0)

        victim = policy.select_victim({0, 1})
        assert victim == 1, "Accessed entry 0 should outrank untouched entry 1"

    def test_tie_breaks_by_insertion_order(self):
        """Equal priorities → evict the oldest-inserted entry."""
        policy = GDSFPolicy()
        for i in range(3):
            policy.on_insert(i, _unit_emb())

        victim = policy.select_victim({0, 1, 2})
        assert victim == 0, "Tie-break should favor oldest insertion"

    def test_name(self):
        assert GDSFPolicy().name == "gdsf"


# ═════════════════════════════════════════════════════════════════════
# Aging via the global inflation clock L
# ═════════════════════════════════════════════════════════════════════

class TestGDSFAging:
    """The L clock must inflate after evictions — GDSF is NOT pure LFU."""

    def test_inflation_clock_ages_out_old_frequent_entry(self):
        """A new freq-1 entry inserted after several evictions must
        outrank an old freq-3 entry: priority(old) = 0 + 3 = 3 was
        fixed when L = 0, while priority(new) = L + 1 = 4 after three
        evictions pushed L to 3.  Pure LFU would evict the new entry
        (freq 1 < 3); GDSF must evict the old one."""
        policy = GDSFPolicy()

        # Old entry A: freq 3 → priority = 0 + 3*1/1 = 3.
        policy.on_insert(0, _unit_emb())
        policy.on_access(0)
        policy.on_access(0)

        # Three fillers, each inserted at the current L and immediately
        # evicted: L walks 0 → 1 → 2 → 3.  (The policy is advisory, so
        # the test plays the cache's role and evicts each filler;
        # select_victim is sanity-checked once where the comparison is
        # strict — the last filler would tie A at priority 3.)
        policy.on_insert(10, _unit_emb())
        assert policy.select_victim({0, 10}) == 10, \
            "Filler (priority 1 < 3) should lose first"
        policy.on_evict(10)
        for fid in (11, 12):
            policy.on_insert(fid, _unit_emb())
            policy.on_evict(fid)

        # New entry B: freq 1 → priority = 3 + 1*1/1 = 4 > 3.
        policy.on_insert(1, _unit_emb())

        victim = policy.select_victim({0, 1})
        assert victim == 0, (
            "Aging must evict the old freq-3 entry over the new freq-1 "
            "entry — pure LFU would pick the new one"
        )


# ═════════════════════════════════════════════════════════════════════
# Cost/size computed from the actual embedding
# ═════════════════════════════════════════════════════════════════════

class TestGDSFCostAndSize:
    """cost(e) = 1/||e|| and size(e) = dim(e) must come from the
    embedding, not be assumed constant."""

    def test_cost_from_embedding_norm(self):
        """Smaller norm → higher cost → higher priority → survives.

        Entry 0: ||e|| = 0.5 → cost 2, size 1 → priority 2.
        Entry 1: ||e|| = 1.0 → cost 1, size 1 → priority 1.
        Both freq 1; entry 1 has the lower priority and must lose
        (insertion-order tie-break would wrongly pick 0 if cost were
        assumed uniform)."""
        policy = GDSFPolicy()
        policy.on_insert(0, np.array([0.5], dtype=np.float32))
        policy.on_insert(1, np.array([1.0], dtype=np.float32))

        victim = policy.select_victim({0, 1})
        assert victim == 1, "Low-cost (large-norm) entry should be evicted"

    def test_size_from_embedding_dim(self):
        """Larger dimension → larger size → lower priority → evicted.

        Entry 0: 1-d unit norm → cost/size = 1/1 = 1.
        Entry 1: 4-d unit norm → cost/size = 1/4 = 0.25.
        Both freq 1; entry 1 has the lower priority and must lose."""
        policy = GDSFPolicy()
        policy.on_insert(0, np.array([1.0], dtype=np.float32))
        policy.on_insert(1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        victim = policy.select_victim({0, 1})
        assert victim == 1, "Larger-size entry should be evicted"


# ═════════════════════════════════════════════════════════════════════
# Defensive behaviour
# ═════════════════════════════════════════════════════════════════════

class TestGDSFDefensive:
    """Edge cases: empty/foreign active sets, unseen ids."""

    def test_empty_active_set_returns_none(self):
        policy = GDSFPolicy()
        policy.on_insert(0, _unit_emb())
        assert policy.select_victim(set()) is None

    def test_victim_always_in_active_ids(self):
        """Tracked entries outside active_ids must be skipped, even if
        they have the lowest priority."""
        policy = GDSFPolicy()
        for i in range(3):
            policy.on_insert(i, _unit_emb())
        policy.on_access(2)  # entry 2 has the HIGHEST priority

        victim = policy.select_victim({2})
        assert victim == 2, "Only active ids may be returned"

    def test_unseen_active_ids_return_none(self):
        """An active set of ids the policy never saw yields None."""
        policy = GDSFPolicy()
        policy.on_insert(0, _unit_emb())
        assert policy.select_victim({99}) is None

    def test_on_access_unseen_id_is_ignored(self):
        """on_access for an untracked id must not raise."""
        policy = GDSFPolicy()
        policy.on_insert(0, _unit_emb())
        policy.on_access(99)  # must be a no-op
        assert policy.select_victim({0}) == 0


# ═════════════════════════════════════════════════════════════════════
# Rebuild / ID compaction
# ═════════════════════════════════════════════════════════════════════

class TestGDSFRebuild:
    """on_rebuild must remap ids, drop stale ids, and preserve L."""

    def test_on_rebuild_remaps_drops_stale_and_preserves_L(self):
        policy = GDSFPolicy()

        # Entry 0: freq 2 → priority 2.  Entry 1: freq 1 → priority 1.
        policy.on_insert(0, _unit_emb())
        policy.on_insert(1, _unit_emb())
        policy.on_access(0)

        # One eviction to push L above zero: filler priority = 1 → L = 1.
        policy.on_insert(5, _unit_emb())
        policy.on_evict(5)
        assert policy._L == 1.0

        # Entry 7 dies during compaction (absent from the remap).
        policy.on_insert(7, _unit_emb())

        # Compaction: old 0 → new 1, old 1 → new 0; 7 is dropped.
        policy.on_rebuild({0: 1, 1: 0})

        assert set(policy._entries) == {0, 1}, \
            "Stale ids must be dropped, survivors renumbered"
        assert policy._L == 1.0, "The inflation clock must survive rebuilds"

        # Priorities follow the remap: new 0 (old 1, priority 1) loses
        # to new 1 (old 0, priority 2).
        victim = policy.select_victim({0, 1})
        assert victim == 0, "After remap, new 0 (old 1) has lowest priority"

    def test_on_rebuild_preserves_insertion_order_for_ties(self):
        """Tie-breaking must follow the *original* insertion order even
        after ids are renumbered."""
        policy = GDSFPolicy()
        for i in range(3):
            policy.on_insert(i, _unit_emb())  # all priority 1 (tied)

        # Renumber: old 0 → new 2, old 1 → new 0, old 2 → new 1.
        policy.on_rebuild({0: 2, 1: 0, 2: 1})

        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, \
            "New id 2 (old 0, oldest inserted) should win the tie-break"
