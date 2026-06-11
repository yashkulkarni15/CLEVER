"""
Unit tests for the ARC (Adaptive Replacement Cache) eviction policy.

Tests cover:
- Fresh inserts land in T1; T1 is preferred for eviction when p is small
- on_access promotes T1 entries to T2; promoted entries survive longer
- Evictions populate the correct ghost list (B1/B2); ghosts stay bounded
- p adapts on (semantic) ghost hits, in the classic ARC directions
- Defensive select_victim behaviour (empty set, stale/unseen ids)
- on_rebuild id remapping for T1/T2/B1/B2

Run with: pytest tests/test_arc.py -v
"""

import numpy as np

from src.cache.eviction.arc import ARCPolicy


# ── Helpers ──────────────────────────────────────────────────────────

def _random_embedding(dim: int = 16, rng=None) -> np.ndarray:
    """Generate a random unit-norm embedding."""
    if rng is None:
        rng = np.random.RandomState(42)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _fill_t1(policy: ARCPolicy, ids, rng) -> None:
    """Insert each id with a distinct random embedding (lands in T1)."""
    for cid in ids:
        policy.on_insert(cid, _random_embedding(rng=rng))


# ═════════════════════════════════════════════════════════════════════
# Behavior 1: fresh inserts land in T1; T1-LRU evicted when p is small
# ═════════════════════════════════════════════════════════════════════

class TestFreshInsertsAndT1Preference:
    """New entries are 'seen once' → T1; small p prefers T1 victims."""

    def test_fresh_inserts_land_in_t1(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)

        assert list(policy._t1) == [0, 1, 2], "Fresh inserts must go to T1"
        assert len(policy._t2) == 0, "Nothing was accessed twice yet"

    def test_select_victim_prefers_t1_lru_when_p_small(self):
        """p starts at 0 → |T1| > p → classic REPLACE evicts T1's LRU."""
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)

        victim = policy.select_victim({0, 1, 2})
        assert victim == 0, "T1 LRU (first insert) should be the victim"
        assert policy.name == "arc"


# ═════════════════════════════════════════════════════════════════════
# Behavior 2: on_access promotes T1 → T2; promoted entries survive longer
# ═════════════════════════════════════════════════════════════════════

class TestPromotionToT2:
    """A second touch makes an entry 'frequent' (T2)."""

    def test_access_promotes_t1_entry_to_t2(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)

        policy.on_access(1)

        assert 1 not in policy._t1, "Accessed entry must leave T1"
        assert 1 in policy._t2, "Accessed entry must be promoted to T2"

    def test_promoted_entry_survives_one_time_entries(self):
        """With p = 0, every one-time (T1) entry is evicted before the
        promoted (T2) entry — the key recency/frequency separation."""
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)
        policy.on_access(1)  # 1 → T2

        victim = policy.select_victim({0, 1, 2})
        assert victim == 0, "One-time entry 0 should go before promoted 1"
        policy.on_evict(0)

        victim = policy.select_victim({1, 2})
        assert victim == 2, "One-time entry 2 should go before promoted 1"
        policy.on_evict(2)

        victim = policy.select_victim({1})
        assert victim == 1, "Only the promoted entry remains"

    def test_access_of_t2_entry_moves_it_to_mru(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)
        policy.on_access(0)  # T2 = [0]
        policy.on_access(1)  # T2 = [0, 1]
        policy.on_access(0)  # T2 = [1, 0]

        assert list(policy._t2) == [1, 0], "Re-accessed T2 entry moves to MRU"


# ═════════════════════════════════════════════════════════════════════
# Behavior 3: evictions populate the right ghost list; ghosts bounded
# ═════════════════════════════════════════════════════════════════════

class TestGhostLists:
    """Evicted ids become ghosts in B1 (from T1) or B2 (from T2)."""

    def test_t1_eviction_goes_to_b1(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1], rng)

        policy.on_evict(0)

        assert 0 not in policy._t1, "Evicted id must leave T1"
        assert 0 in policy._b1, "T1 eviction must land in B1"
        assert 0 not in policy._b2
        assert 0 not in policy._embeddings, "Resident embedding released"

    def test_t2_eviction_goes_to_b2(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1], rng)
        policy.on_access(0)  # 0 → T2

        policy.on_evict(0)

        assert 0 not in policy._t2, "Evicted id must leave T2"
        assert 0 in policy._b2, "T2 eviction must land in B2"
        assert 0 not in policy._b1

    def test_ghost_lists_bounded_by_explicit_capacity(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=2, seed=0)
        _fill_t1(policy, range(5), rng)

        for cid in range(5):
            policy.on_evict(cid)

        assert len(policy._b1) <= 2, "B1 must stay bounded by capacity"
        assert list(policy._b1) == [3, 4], "Oldest ghosts trimmed first"

    def test_ghost_lists_bounded_by_inferred_capacity(self):
        """capacity=None → c inferred as max observed |active_ids|."""
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=None, seed=0)
        _fill_t1(policy, range(5), rng)

        policy.select_victim({0, 1, 2, 3})  # observe |active| = 4
        assert policy._observed_capacity == 4

        for cid in range(5):
            policy.on_evict(cid)

        assert len(policy._b1) == 4, "B1 bounded by inferred capacity"
        assert list(policy._b1) == [1, 2, 3, 4]


# ═════════════════════════════════════════════════════════════════════
# Behavior 4: p adapts on ghost hits (semantic re-request detection)
# ═════════════════════════════════════════════════════════════════════

class TestPAdaptation:
    """Inserting an embedding near a ghost is a ghost hit: B1 hits grow
    p (favor recency side), B2 hits shrink it, and the re-requested
    entry lands directly in T2 (classic ARC ghost-hit placement)."""

    def test_b1_ghost_hit_increases_p_and_lands_in_t2(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        e0 = _random_embedding(rng=rng)

        policy.on_insert(0, e0)
        policy.on_evict(0)  # 0 → B1, ghost remembers e0
        assert policy._p == 0.0

        # Re-request of (essentially) the same item → B1 ghost hit.
        policy.on_insert(10, e0.copy())

        assert policy._p > 0.0, "B1 ghost hit must increase p"
        assert 10 in policy._t2, "Ghost-hit insert goes to MRU of T2"
        assert 10 not in policy._t1
        assert 0 not in policy._b1, "Matched ghost must be retired"

    def test_b2_ghost_hit_decreases_p(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        e0 = _random_embedding(rng=rng)

        policy.on_insert(0, e0)
        policy.on_access(0)  # 0 → T2
        policy.on_evict(0)   # 0 → B2
        policy._p = 3.0

        policy.on_insert(10, e0.copy())

        assert policy._p < 3.0, "B2 ghost hit must decrease p"
        assert 10 in policy._t2, "Ghost-hit insert goes to MRU of T2"
        assert 0 not in policy._b2, "Matched ghost must be retired"

    def test_dissimilar_insert_is_not_a_ghost_hit(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)

        policy.on_insert(0, _random_embedding(rng=rng))
        policy.on_evict(0)
        policy.on_insert(10, _random_embedding(rng=rng))  # unrelated

        assert policy._p == 0.0, "No ghost hit → p unchanged"
        assert 10 in policy._t1, "Non-ghost insert is a fresh T1 entry"
        assert 0 in policy._b1, "Unmatched ghost stays"

    def test_p_is_clamped_to_capacity(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=2, seed=0)
        e0 = _random_embedding(rng=rng)
        policy._p = 1.9

        policy.on_insert(0, e0)
        policy.on_evict(0)
        policy.on_insert(10, e0.copy())

        assert policy._p <= 2.0, "p must never exceed capacity"

    def test_large_p_shifts_victim_to_t2(self):
        """When |T1| ≤ p, classic REPLACE evicts from T2 instead."""
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2, 3], rng)
        policy.on_access(2)  # T2 = [2]
        policy.on_access(3)  # T2 = [2, 3]
        policy._p = 4.0      # T1 target ≥ |T1| → protect T1

        victim = policy.select_victim({0, 1, 2, 3})
        assert victim == 2, "With large p, T2's LRU must be the victim"


# ═════════════════════════════════════════════════════════════════════
# Behavior 5: defensive select_victim
# ═════════════════════════════════════════════════════════════════════

class TestDefensiveSelection:
    """select_victim must always return None or a member of active_ids."""

    def test_empty_active_set_returns_none(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1], rng)

        assert policy.select_victim(set()) is None

    def test_victim_always_within_active_ids(self):
        """Stale bookkeeping (tracked ids no longer active) is skipped."""
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)
        policy.on_access(1)  # 1 → T2

        victim = policy.select_victim({1, 2})  # 0 is stale
        assert victim in {1, 2}

    def test_unseen_active_ids_still_yield_valid_victim(self):
        """Ids the policy never observed must still produce a victim."""
        policy = ARCPolicy(capacity=4, seed=0)

        victim = policy.select_victim({100, 200, 300})
        assert victim in {100, 200, 300}, \
            "Defensive fallback must return a member of active_ids"


# ═════════════════════════════════════════════════════════════════════
# Behavior 6: on_rebuild remaps all internal ids; stale ids dropped
# ═════════════════════════════════════════════════════════════════════

class TestRebuild:
    """Compaction renumbers ids; all four lists must follow."""

    def test_rebuild_remaps_all_lists_preserving_order(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2, 3], rng)
        policy.on_access(1)  # 1 → T2
        policy.on_access(3)  # 3 → T2
        policy.on_evict(0)   # 0 → B1
        policy.on_evict(1)   # 1 → B2
        # State: T1 = [2], T2 = [3], B1 = [0], B2 = [1]

        # Generic remap covering ghosts too (harness normally remaps
        # only live ids, but the implementation must be id-agnostic).
        policy.on_rebuild({2: 10, 3: 11, 0: 12, 1: 13})

        assert list(policy._t1) == [10]
        assert list(policy._t2) == [11]
        assert list(policy._b1) == [12]
        assert list(policy._b2) == [13]
        assert set(policy._embeddings) == {10, 11}, \
            "Resident embeddings must be remapped too"

    def test_rebuild_drops_stale_ids(self):
        """Ids absent from the remap (dead/evicted) must vanish —
        compaction reuses ids, so stale ghosts would alias new slots."""
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)
        policy.on_access(1)  # 1 → T2
        policy.on_evict(0)   # 0 → B1

        policy.on_rebuild({1: 0, 2: 1})  # live ids only, ghost dropped

        assert list(policy._t1) == [1]
        assert list(policy._t2) == [0]
        assert len(policy._b1) == 0, "Ghost not in remap must be dropped"
        assert len(policy._b2) == 0
        assert set(policy._embeddings) == {0, 1}

    def test_rebuild_preserves_p_and_order_semantics(self):
        rng = np.random.RandomState(42)
        policy = ARCPolicy(capacity=4, seed=0)
        _fill_t1(policy, [0, 1, 2], rng)
        policy._p = 1.5

        policy.on_rebuild({0: 2, 1: 0, 2: 1})

        assert policy._p == 1.5, "p survives a rebuild"
        # Insertion order 0,1,2 → remapped order 2,0,1: new 2 is LRU.
        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, "Remapped T1 must preserve LRU order"
