"""
Comprehensive tests for eviction policies and SemanticCache integration.

Tests cover:
- Each policy's core eviction logic
- Edge cases (empty cache, single entry, ties)
- on_rebuild ID remapping
- SemanticCache integration (insert → evict → lookup correctness)
- Cache size invariant (never exceeds max_size)
- Policy interchangeability
- Semantic policy redundancy scoring

Run with: pytest tests/test_eviction.py -v
"""

import numpy as np
import pytest

from src.cache.eviction.lru import LRUPolicy
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.semantic import SemanticPolicy
from src.cache.eviction.oracle import OraclePolicy
from src.cache.semantic_cache import SemanticCache


# ── Helpers ──────────────────────────────────────────────────────────

def _random_embedding(dim: int = 16, rng=None) -> np.ndarray:
    """Generate a random unit-norm embedding."""
    if rng is None:
        rng = np.random.RandomState(42)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _cluster_embedding(center: np.ndarray, noise: float = 0.01,
                        rng=None) -> np.ndarray:
    """Generate an embedding near a cluster center."""
    if rng is None:
        rng = np.random.RandomState(42)
    v = center + rng.randn(*center.shape).astype(np.float32) * noise
    v /= np.linalg.norm(v)
    return v


# ═════════════════════════════════════════════════════════════════════
# LRU Policy Tests
# ═════════════════════════════════════════════════════════════════════

class TestLRUPolicy:
    """Test LRU eviction logic."""

    def test_evicts_oldest_untouched(self):
        """The first inserted entry (never accessed) should be evicted."""
        policy = LRUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        # Insert entries 0, 1, 2
        for i in range(3):
            policy.on_insert(i, emb)

        active = {0, 1, 2}
        victim = policy.select_victim(active)
        assert victim == 0, "LRU should evict entry 0 (oldest)"

    def test_access_protects_entry(self):
        """Accessing an entry moves it to the back; oldest untouched is evicted."""
        policy = LRUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        for i in range(3):
            policy.on_insert(i, emb)

        # Access entry 0 — it moves to the back
        policy.on_access(0)

        active = {0, 1, 2}
        victim = policy.select_victim(active)
        assert victim == 1, "After accessing 0, entry 1 should be oldest"

    def test_evict_then_insert(self):
        """After evicting, the next victim should update correctly."""
        policy = LRUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        for i in range(3):
            policy.on_insert(i, emb)

        # Evict 0
        victim = policy.select_victim({0, 1, 2})
        policy.on_evict(victim)

        # Insert new entry 3
        policy.on_insert(3, emb)

        # Next victim should be 1 (oldest remaining)
        victim = policy.select_victim({1, 2, 3})
        assert victim == 1

    def test_single_entry(self):
        """With one entry, it should be selected for eviction."""
        policy = LRUPolicy()
        policy.on_insert(0, np.zeros(16, dtype=np.float32))
        victim = policy.select_victim({0})
        assert victim == 0

    def test_empty_active_set(self):
        """Empty active set should return None."""
        policy = LRUPolicy()
        policy.on_insert(0, np.zeros(16, dtype=np.float32))
        victim = policy.select_victim(set())
        assert victim is None

    def test_on_rebuild_remaps_ids(self):
        """After rebuild, internal order should use new IDs."""
        policy = LRUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        for i in range(3):
            policy.on_insert(i, emb)

        # Simulate rebuild: old {0, 1, 2} → new {0, 1, 2}
        # but with different mapping (e.g. old 2 → new 0)
        id_remap = {0: 2, 1: 0, 2: 1}
        policy.on_rebuild(id_remap)

        # Old insertion order was 0, 1, 2
        # After remap: new order is 2 (was 0), 0 (was 1), 1 (was 2)
        # So new 2 should be evicted first
        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, "After remap, new ID 2 (was old 0) should be oldest"


# ═════════════════════════════════════════════════════════════════════
# LFU Policy Tests
# ═════════════════════════════════════════════════════════════════════

class TestLFUPolicy:
    """Test LFU eviction logic."""

    def test_evicts_least_accessed(self):
        """Entry with fewest accesses should be evicted."""
        policy = LFUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        for i in range(3):
            policy.on_insert(i, emb)

        # Access entry 0 twice, entry 1 once, entry 2 zero times
        policy.on_access(0)
        policy.on_access(0)
        policy.on_access(1)

        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, "Entry 2 (0 accesses) should be evicted"

    def test_tie_breaks_by_insertion_order(self):
        """Among entries with same count, oldest inserted is evicted."""
        policy = LFUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        for i in range(3):
            policy.on_insert(i, emb)

        # All have 0 accesses — entry 0 was inserted first
        victim = policy.select_victim({0, 1, 2})
        assert victim == 0, "Tie-break should favor oldest insertion"

    def test_newly_inserted_not_immediately_evicted_if_others_exist(self):
        """A new entry (count 0) should be evicted only if all others
        also have count 0, in which case the oldest is evicted."""
        policy = LFUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        for i in range(3):
            policy.on_insert(i, emb)
            policy.on_access(i)  # each has count 1

        # Insert new entry 3 with count 0
        policy.on_insert(3, emb)

        victim = policy.select_victim({0, 1, 2, 3})
        assert victim == 3, "New entry with count 0 should be evicted"

    def test_on_rebuild_preserves_counts(self):
        """Rebuild should remap IDs but preserve access counts."""
        policy = LFUPolicy()
        emb = np.zeros(16, dtype=np.float32)

        policy.on_insert(0, emb)
        policy.on_insert(1, emb)
        policy.on_access(0)  # count 0 → 1
        policy.on_access(0)  # count 0 → 2
        policy.on_access(1)  # count 1 → 1

        # Remap: old 0 → new 1, old 1 → new 0
        policy.on_rebuild({0: 1, 1: 0})

        # New ID 0 (was old 1, count 1) should be evicted over
        # new ID 1 (was old 0, count 2)
        victim = policy.select_victim({0, 1})
        assert victim == 0, "After remap, new 0 (old 1, count=1) should be evicted"


# ═════════════════════════════════════════════════════════════════════
# Semantic Policy Tests
# ═════════════════════════════════════════════════════════════════════

class TestSemanticPolicy:
    """Test semantic-aware eviction logic."""

    def test_evicts_redundant_over_isolated(self):
        """Entries in a dense cluster should be evicted before
        isolated entries (which are irreplaceable)."""
        rng = np.random.RandomState(42)
        dim = 16

        # Create a cluster center
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,  # L2² threshold
            recompute_interval=1,       # recompute every eviction
        )

        # Entry 0: isolated (unique direction)
        isolated_emb = _random_embedding(dim, rng)
        policy.on_insert(0, isolated_emb)

        # Entries 1, 2, 3: clustered (very similar to each other)
        for i in range(1, 4):
            emb = _cluster_embedding(center, noise=0.01, rng=rng)
            policy.on_insert(i, emb)

        # Force redundancy recomputation
        policy._recompute_redundancy({0, 1, 2, 3})

        # The clustered entries should have higher redundancy
        assert len(policy._neighbors[0]) < len(policy._neighbors[1]), \
            "Isolated entry should have fewer neighbours than clustered"

        # Victim should be one of the clustered entries (1, 2, or 3)
        victim = policy.select_victim({0, 1, 2, 3})
        assert victim in {1, 2, 3}, \
            f"Victim should be clustered, got {victim}"

    def test_access_protects_redundant_entry(self):
        """High access count / recency should protect even redundant entries."""
        rng = np.random.RandomState(42)
        dim = 16
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,
            recompute_interval=1,
        )

        # All entries are clustered (all redundant)
        for i in range(4):
            emb = _cluster_embedding(center, noise=0.01, rng=rng)
            policy.on_insert(i, emb)

        # Heavily access entry 0
        for _ in range(100):
            policy.on_access(0)

        policy._recompute_redundancy({0, 1, 2, 3})

        # Entry 0 should NOT be evicted despite redundancy
        # because its high frequency protects it
        victim = policy.select_victim({0, 1, 2, 3})
        assert victim != 0, "Heavily accessed entry should be protected"

    def test_on_rebuild_remaps_all_state(self):
        """Rebuild should remap embeddings, scores, and order."""
        policy = SemanticPolicy(recompute_interval=1)
        dim = 16
        rng = np.random.RandomState(42)

        for i in range(3):
            policy.on_insert(i, _random_embedding(dim, rng))

        policy._recompute_redundancy({0, 1, 2})

        # Remap
        remap = {0: 2, 1: 0, 2: 1}
        policy.on_rebuild(remap)

        # Check all state is remapped
        assert 2 in policy._access_order
        assert 0 in policy._access_order
        assert 1 in policy._access_order
        assert 2 in policy._embeddings
        assert 0 in policy._embeddings
        assert 1 in policy._embeddings

    def test_new_entry_not_immediately_evicted(self):
        """New entries start with redundancy 0 (isolated), so they
        should NOT be the first evicted."""
        rng = np.random.RandomState(42)
        dim = 16
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,
            recompute_interval=100,  # don't recompute yet
        )

        # Entries 0-2: clustered, with redundancy pre-set
        for i in range(3):
            emb = _cluster_embedding(center, noise=0.01, rng=rng)
            policy.on_insert(i, emb)

        # Force a recompute so 0-2 have redundancy scores
        policy._recompute_redundancy({0, 1, 2})

        # Insert new entry 3 (gets redundancy=0 by default)
        policy.on_insert(3, _random_embedding(dim, rng))

        # Entry 3 should not be evicted (redundancy=0 → low score)
        victim = policy.select_victim({0, 1, 2, 3})
        assert victim != 3, "Newly inserted entry should not be immediately evicted"

    def test_dynamic_impute_populates_neighbors_for_close_entries(self):
        """on_insert with dynamic_impute=True must discover and store the
        new entry's neighbours in the symmetric graph immediately, instead
        of relying on the next batch recompute."""
        rng = np.random.RandomState(42)
        dim = 16
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,
            dynamic_impute=True,
            recompute_interval=100_000,  # never fires in this test
        )

        # Seed the cache with a dense cluster.
        for i in range(5):
            policy.on_insert(i, _cluster_embedding(center, noise=0.01, rng=rng))

        # Insert a new clustered entry — it should see the existing 5.
        policy.on_insert(99, _cluster_embedding(center, noise=0.01, rng=rng))

        assert len(policy._neighbors[99]) >= 1, \
            "Dynamic imputation should discover at least one neighbour"

    def test_on_insert_is_symmetric(self):
        """When on_insert discovers that e_new is a neighbour of some
        existing entry j, both _neighbors[e_new] AND _neighbors[j] must
        be updated — the graph is always symmetric."""
        rng = np.random.RandomState(42)
        dim = 16
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,
            dynamic_impute=True,
            recompute_interval=100_000,
        )

        # Insert a cluster seed first.
        policy.on_insert(0, _cluster_embedding(center, noise=0.01, rng=rng))
        assert policy._neighbors[0] == set()

        # Insert a second clustered entry.
        policy.on_insert(1, _cluster_embedding(center, noise=0.01, rng=rng))

        # Both sides of the edge must be present.
        assert 1 in policy._neighbors[0], \
            "Existing entry's neighbour set must include the newly inserted id"
        assert 0 in policy._neighbors[1], \
            "New entry's neighbour set must include the existing id"

    def test_on_evict_removes_from_neighbors_symmetrically(self):
        """Evicting an entry must purge it from every other entry's
        neighbour set, not just its own."""
        rng = np.random.RandomState(42)
        dim = 16
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,
            dynamic_impute=True,
            recompute_interval=100_000,
        )

        for i in range(3):
            policy.on_insert(i, _cluster_embedding(center, noise=0.01, rng=rng))

        # Precondition: all three are mutually connected.
        assert 1 in policy._neighbors[0]
        assert 2 in policy._neighbors[0]

        policy.on_evict(1)

        assert 1 not in policy._neighbors[0], \
            "Evicted id must be removed from surviving entries' neighbour sets"
        assert 1 not in policy._neighbors[2]
        assert 1 not in policy._neighbors, \
            "Evicted id must have no entry of its own in _neighbors"

    def test_recency_floor_prevents_ancient_entry_explosion(self):
        """Even when the oldest entry has never been accessed, its
        utility must not collapse to ε (which would make its score
        explode regardless of redundancy).  The recency floor (rank+1)/n
        keeps utility ≥ 1/n."""
        policy = SemanticPolicy(
            similarity_threshold=0.30,
            alpha=1.0,
            beta=1.0,
            mu=0.1,
            dynamic_impute=False,  # isolate the recency-floor effect
            recompute_interval=100_000,
        )

        dim = 16
        rng = np.random.RandomState(42)

        # Insert 10 isolated entries (no neighbours → r = 0 for all).
        for i in range(10):
            policy.on_insert(i, _random_embedding(dim, rng))

        # With r = 0 for everyone and freq = 0 for everyone, the score
        # reduces to mu/recency.  The oldest entry (rank 0) has recency
        # = 1/10 (NOT 0), so its score is mu * 10, not mu / ε.
        # Victim should still be the oldest entry (rank 0 = id 0).
        victim = policy.select_victim({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        assert victim == 0, \
            "Oldest untouched entry should still win, but via a finite denominator"

        # And the result must be an int, not NaN/Inf-driven.
        assert isinstance(victim, int)

    def test_large_mu_approaches_inverse_utility(self):
        """With mu → ∞, the semantic policy's score is dominated by
        mu/utility, so the ordering asymptotically approaches
        `argmax(1/utility)` = `argmin(utility)` = plain LRU+LFU
        behaviour.  This is the graceful-degradation guarantee."""
        rng = np.random.RandomState(42)
        dim = 16
        center = _random_embedding(dim, rng)

        policy = SemanticPolicy(
            similarity_threshold=0.30,
            alpha=1.0,
            beta=1.0,
            mu=1e6,  # dominates the numerator
            dynamic_impute=True,
            recompute_interval=100_000,
        )

        # Three clustered entries — all mutually redundant.
        for i in range(3):
            policy.on_insert(i, _cluster_embedding(center, noise=0.01, rng=rng))

        # Heavily access entry 0 (makes it the most frequent AND the
        # most recently accessed).
        for _ in range(50):
            policy.on_access(0)

        # Under LRU+LFU: entry 0 has highest utility → lowest 1/utility
        # → lowest score → NOT the victim.  Entry 1 was inserted before
        # entry 2 (oldest untouched remaining).
        victim = policy.select_victim({0, 1, 2})
        assert victim != 0, \
            "With mu → ∞, the heavily-accessed entry must be protected"
        assert victim == 1, \
            "Under LRU+LFU degeneration, the oldest untouched entry wins"


# ═════════════════════════════════════════════════════════════════════
# Oracle Policy Tests
# ═════════════════════════════════════════════════════════════════════

class TestOraclePolicy:
    """Test oracle (Bélády's optimal) eviction."""

    def test_evicts_entry_used_furthest_away(self):
        """Oracle should evict the entry whose next use is latest."""
        rng = np.random.RandomState(42)
        dim = 16

        # Create 3 cache entries
        cache_embs = np.array([
            _random_embedding(dim, rng) for _ in range(3)
        ], dtype=np.float32)
        cache_ids = [0, 1, 2]

        # Future stream: query similar to entry 0, then entry 1, then entry 2
        # Entry 2 is used last → should be evicted first
        stream = np.array([
            cache_embs[0],  # step 0: accesses entry 0
            cache_embs[1],  # step 1: accesses entry 1
            cache_embs[2],  # step 2: accesses entry 2
        ], dtype=np.float32)

        policy = OraclePolicy(
            future_stream_embeddings=stream,
            cache_embeddings=cache_embs,
            cache_ids=cache_ids,
            similarity_threshold=0.90,  # exact match
        )

        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, \
            f"Oracle should evict entry 2 (used furthest away), got {victim}"

    def test_evicts_never_used_entry(self):
        """Entry never accessed in the future → next_use = ∞ → evicted first."""
        rng = np.random.RandomState(42)
        dim = 16

        cache_embs = np.array([
            _random_embedding(dim, rng) for _ in range(3)
        ], dtype=np.float32)
        cache_ids = [0, 1, 2]

        # Stream only accesses entries 0 and 1
        stream = np.array([
            cache_embs[0],
            cache_embs[1],
        ], dtype=np.float32)

        policy = OraclePolicy(
            future_stream_embeddings=stream,
            cache_embeddings=cache_embs,
            cache_ids=cache_ids,
            similarity_threshold=0.90,
        )

        victim = policy.select_victim({0, 1, 2})
        assert victim == 2, "Entry 2 (never used) should be evicted"

    def test_on_rebuild_remaps_next_use(self):
        """Rebuild should remap next_use keys."""
        rng = np.random.RandomState(42)
        dim = 16
        e0 = _random_embedding(dim, rng)
        e1 = _random_embedding(dim, rng)
        cache_embs = np.array([e0, e1], dtype=np.float32)

        stream = np.array([e1, e0], dtype=np.float32)

        policy = OraclePolicy(
            future_stream_embeddings=stream,
            cache_embeddings=cache_embs,
            cache_ids=[0, 1],
            similarity_threshold=0.90,
        )

        # Before remap: entry 0 next_use=1, entry 1 next_use=0
        # Remap: 0→1, 1→0
        policy.on_rebuild({0: 1, 1: 0})

        # After remap: new 1 (was 0) next_use=1, new 0 (was 1) next_use=0
        # So new 1 has the furthest next_use → should be evicted
        victim = policy.select_victim({0, 1})
        assert victim == 1


# ═════════════════════════════════════════════════════════════════════
# SemanticCache Integration Tests
# ═════════════════════════════════════════════════════════════════════

class TestCacheEvictionIntegration:
    """Test eviction policies integrated into SemanticCache."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample embeddings and texts."""
        rng = np.random.RandomState(42)
        n = 20
        dim = 16
        embeddings = rng.randn(n, dim).astype(np.float32)
        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        texts = [f"query_{i}" for i in range(n)]
        return embeddings, texts, dim

    def test_cache_size_never_exceeds_max(self, sample_data):
        """For each policy, cache size must never exceed max_size."""
        embeddings, texts, dim = sample_data
        max_size = 10

        for policy_name in ["lru", "lfu", "semantic"]:
            cache = SemanticCache(
                dim=dim,
                index_type="flat",
                max_size=max_size,
                eviction_policy=policy_name,
            )
            # Build with first 10 entries
            cache.build(embeddings[:10], texts[:10])
            assert cache.size == 10

            # Insert 10 more — each should trigger eviction
            for i in range(10, 20):
                cache.insert(embeddings[i], texts[i])
                assert cache.size <= max_size, \
                    f"[{policy_name}] size={cache.size} > max={max_size}"

    def test_lru_eviction_order(self, sample_data):
        """LRU cache should evict the oldest untouched entry first."""
        embeddings, texts, dim = sample_data

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=5, eviction_policy="lru",
        )
        cache.build(embeddings[:5], texts[:5])

        # Access entries 0, 1, 2, 3 (not 4)
        for i in range(4):
            cache.lookup(embeddings[i])

        # Insert new entry → should evict entry 4 (least recently used)
        cache.insert(embeddings[5], texts[5])
        assert 4 not in cache._active, "Entry 4 should have been evicted"
        assert cache.size == 5

    def test_policy_string_backward_compat(self, sample_data):
        """'none' policy should mean no eviction (unbounded)."""
        embeddings, texts, dim = sample_data

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=0,  # unbounded
            eviction_policy="none",
        )
        cache.build(embeddings, texts)
        assert cache.size == 20
        assert cache._policy is None

    def test_policy_instance_accepted(self, sample_data):
        """Cache should accept a pre-built EvictionPolicy instance."""
        embeddings, texts, dim = sample_data

        policy = LRUPolicy()
        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=10, eviction_policy=policy,
        )
        cache.build(embeddings[:10], texts[:10])
        assert cache._policy is policy
        assert cache.eviction_policy == "lru"

    def test_unknown_policy_raises(self, sample_data):
        """Unknown policy string should raise ValueError."""
        _, _, dim = sample_data
        with pytest.raises(ValueError, match="Unknown eviction policy"):
            SemanticCache(
                dim=dim, index_type="flat",
                eviction_policy="nonexistent",
            )

    def test_oracle_via_instance(self, sample_data):
        """Oracle policy must be passed as an instance."""
        embeddings, texts, dim = sample_data

        # Oracle needs future stream
        oracle = OraclePolicy(
            future_stream_embeddings=embeddings[10:],
            cache_embeddings=embeddings[:10],
            cache_ids=list(range(10)),
            similarity_threshold=0.90,
        )
        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=10, eviction_policy=oracle,
        )
        cache.build(embeddings[:10], texts[:10])
        assert cache.eviction_policy == "oracle"

    def test_rebuild_triggers_policy_remap(self, sample_data):
        """When cache rebuilds, policy internal state must be remapped."""
        embeddings, texts, dim = sample_data

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=5, eviction_policy="lru",
            rebuild_threshold=0.20,  # rebuild at 20% dead
        )
        cache.build(embeddings[:5], texts[:5])

        # Insert many entries to trigger evictions and eventually rebuild
        for i in range(5, 15):
            cache.insert(embeddings[i], texts[i])

        assert cache.size <= 5
        # After rebuild, cache should still function correctly
        result = cache.lookup(embeddings[0])
        assert result is not None

    def test_stats_include_eviction_count(self, sample_data):
        """Cache stats should track eviction count."""
        embeddings, texts, dim = sample_data

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=5, eviction_policy="lru",
        )
        cache.build(embeddings[:5], texts[:5])

        for i in range(5, 10):
            cache.insert(embeddings[i], texts[i])

        stats = cache.stats
        assert stats["n_evictions"] == 5
        assert stats["eviction_policy"] == "lru"

    def test_semantic_policy_stats(self, sample_data):
        """Semantic policy should expose timing stats."""
        embeddings, texts, dim = sample_data

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=5, eviction_policy="semantic",
            policy_params={"recompute_interval": 2},
        )
        cache.build(embeddings[:5], texts[:5])

        for i in range(5, 10):
            cache.insert(embeddings[i], texts[i])

        stats = cache.stats
        assert "policy_stats" in stats
        assert "n_evictions" in stats["policy_stats"]


# ═════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_max_size_one(self):
        """Cache with max_size=1 should always have exactly 1 entry."""
        rng = np.random.RandomState(42)
        dim = 16
        embs = rng.randn(5, dim).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=1, eviction_policy="lru",
        )
        cache.build(embs[:1], ["q0"])
        assert cache.size == 1

        for i in range(1, 5):
            cache.insert(embs[i], f"q{i}")
            assert cache.size == 1, f"Size should be 1, got {cache.size}"

    def test_insert_without_build_raises(self):
        """Inserting before build should raise ValueError."""
        cache = SemanticCache(dim=16, index_type="flat", eviction_policy="lru")
        with pytest.raises(ValueError, match="not built"):
            cache.insert(np.zeros(16, dtype=np.float32), "test")

    def test_lookup_after_all_evicted_and_refilled(self):
        """After evicting all initial entries and inserting new ones,
        lookups should still work correctly."""
        rng = np.random.RandomState(42)
        dim = 16
        n = 10
        embs = rng.randn(n, dim).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        cache = SemanticCache(
            dim=dim, index_type="flat",
            max_size=3, eviction_policy="lru",
        )
        cache.build(embs[:3], [f"q{i}" for i in range(3)])

        # Insert enough to evict all original entries
        for i in range(3, 10):
            cache.insert(embs[i], f"q{i}")

        assert cache.size <= 3
        # Lookup should find one of the newer entries
        result = cache.lookup(embs[9])
        assert result.hit is True
        assert result.cache_entry is not None

    def test_all_policies_produce_valid_victim(self):
        """Every policy should return a valid cache_id from active_ids."""
        emb = np.zeros(16, dtype=np.float32)
        active = {10, 20, 30}

        for PolicyClass in [LRUPolicy, LFUPolicy, SemanticPolicy]:
            policy = PolicyClass()
            for cid in active:
                policy.on_insert(cid, emb)

            victim = policy.select_victim(active)
            assert victim in active, \
                f"{PolicyClass.__name__} returned {victim}, not in {active}"


# ═════════════════════════════════════════════════════════════════════
# Oracle Dominance Tests
# ═════════════════════════════════════════════════════════════════════

class TestOracleDominance:
    """Verify that oracle hit rate >= all other policies.

    This is the critical correctness property: Belady's algorithm
    is the theoretical upper bound for any online eviction policy.
    """

    @staticmethod
    def _run_eviction_simulation(
        policy, stream_embs, cache, n_stream, hit_threshold, policy_name,
    ) -> float:
        """Replay a stream and return the final hit rate."""
        n_hits = 0
        for i in range(n_stream):
            if policy_name == "oracle":
                policy.advance_stream(i)

            result = cache.lookup(stream_embs[i], k=1, threshold=hit_threshold)
            if result.hit:
                n_hits += 1
            else:
                cache.insert(stream_embs[i], f"stream_{i}")

        return n_hits / n_stream if n_stream > 0 else 0.0

    def test_oracle_dominates_all_policies(self):
        """Run all 4 policies on synthetic data.  Oracle must win."""
        rng = np.random.RandomState(42)
        dim = 32
        n_cache = 50
        n_stream = 200
        max_cache_size = n_cache
        hit_threshold = 0.90

        # Create cache entries: random unit-norm embeddings.
        cache_embs = rng.randn(n_cache, dim).astype(np.float32)
        cache_embs /= np.linalg.norm(cache_embs, axis=1, keepdims=True)
        cache_texts = [f"cache_{i}" for i in range(n_cache)]

        # Create stream: mix of near-duplicates (should be hits)
        # and novel queries (should be misses).
        stream_embs = np.empty((n_stream, dim), dtype=np.float32)
        for i in range(n_stream):
            if rng.rand() < 0.4:
                # Near-duplicate of a random cache entry
                base = cache_embs[rng.randint(n_cache)]
                noise = rng.randn(dim).astype(np.float32) * 0.05
                v = base + noise
            else:
                # Novel query
                v = rng.randn(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            stream_embs[i] = v

        hit_rates = {}
        for policy_name in ["lru", "lfu", "semantic", "oracle"]:
            cache_ids = list(range(n_cache))

            if policy_name == "lru":
                policy = LRUPolicy()
            elif policy_name == "lfu":
                policy = LFUPolicy()
            elif policy_name == "semantic":
                policy = SemanticPolicy(
                    similarity_threshold=0.30,
                    recompute_interval=50,
                )
            else:
                policy = OraclePolicy(
                    future_stream_embeddings=stream_embs,
                    cache_embeddings=cache_embs,
                    cache_ids=cache_ids,
                    similarity_threshold=hit_threshold,
                    refresh_interval=20,
                )

            cache = SemanticCache(
                dim=dim,
                index_type="flat",
                max_size=max_cache_size,
                eviction_policy=policy,
            )
            cache.build(cache_embs.copy(), cache_texts[:])

            hr = self._run_eviction_simulation(
                policy, stream_embs, cache, n_stream,
                hit_threshold, policy_name,
            )
            hit_rates[policy_name] = hr

        oracle_hr = hit_rates["oracle"]
        for other, hr in hit_rates.items():
            if other == "oracle":
                continue
            assert oracle_hr >= hr - 0.01, (
                f"Oracle ({oracle_hr:.4f}) must dominate {other} ({hr:.4f}). "
                f"All rates: {hit_rates}"
            )


# ═════════════════════════════════════════════════════════════════════
# Oracle Refresh Correctness Tests
# ═════════════════════════════════════════════════════════════════════

class TestOracleRefresh:
    """Verify the periodic full-refresh mechanism."""

    def test_refresh_updates_next_use(self):
        """After a refresh, next_use should reflect current cache state."""
        rng = np.random.RandomState(42)
        dim = 16

        e0 = rng.randn(dim).astype(np.float32)
        e0 /= np.linalg.norm(e0)
        e1 = rng.randn(dim).astype(np.float32)
        e1 /= np.linalg.norm(e1)

        cache_embs = np.array([e0, e1], dtype=np.float32)

        # Stream: e0 at step 0, e1 at step 1
        stream = np.array([e0, e1], dtype=np.float32)

        policy = OraclePolicy(
            future_stream_embeddings=stream,
            cache_embeddings=cache_embs,
            cache_ids=[0, 1],
            similarity_threshold=0.90,
            refresh_interval=5,
        )

        # Entry 0 should be used at step 0, entry 1 at step 1
        assert policy._next_use[0] <= policy._next_use[1], \
            "Entry 0 should have earlier next_use than entry 1"

        # Advance past step 0 and manually refresh
        policy._stream_pos = 1
        policy._full_refresh()

        # Now entry 0 should have no future use (step 0 is past)
        # Entry 0's step-0 access is now in the past → should be INF
        # Entry 1's step-1 access is at current pos → should be ≤ 1
        assert policy._next_use[0] == float("inf"), \
            f"Entry 0 should have no future use after advancing past step 0, got {policy._next_use[0]}"
        assert policy._next_use[1] <= 1.0, \
            f"Entry 1 should have next_use at step 1, got {policy._next_use[1]}"

    def test_refresh_after_eviction(self):
        """After evicting an entry, refresh should redistribute its queries."""
        rng = np.random.RandomState(42)
        dim = 16

        # Create 3 entries and a stream that uses all 3
        embs = rng.randn(3, dim).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        stream = np.array([embs[0], embs[1], embs[2]], dtype=np.float32)

        policy = OraclePolicy(
            future_stream_embeddings=stream,
            cache_embeddings=embs,
            cache_ids=[0, 1, 2],
            similarity_threshold=0.90,
            refresh_interval=1,  # refresh after every eviction
        )

        # Evict entry 2 — should trigger refresh
        policy.on_evict(2)

        # After eviction, entry 2 should be gone
        assert 2 not in policy._active_ids
        assert 2 not in policy._next_use

    def test_on_insert_uses_protective_sentinel(self):
        """on_insert must assign _next_use = stream_pos (not INF) so the
        new entry is NOT immediately chosen as an eviction victim.

        Oracle evicts entries with the HIGHEST next_use.  Assigning INF
        would make new entries the first victim.  stream_pos is a low
        value (below all future indices) so the entry is protected until
        the next full refresh corrects it.
        """
        rng = np.random.RandomState(42)
        dim = 16

        embs = rng.randn(2, dim).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        stream = np.array([embs[0]], dtype=np.float32)

        policy = OraclePolicy(
            future_stream_embeddings=stream,
            cache_embeddings=embs,
            cache_ids=[0, 1],
            similarity_threshold=0.90,
            refresh_interval=100,
        )

        # Advance stream to a known position
        policy.advance_stream(42)

        # Insert a new entry
        new_emb = rng.randn(dim).astype(np.float32)
        new_emb /= np.linalg.norm(new_emb)
        policy.on_insert(2, new_emb)

        # _next_use should be stream_pos (42), NOT INF
        assert policy._next_use[2] == 42.0, (
            f"Expected _next_use=42 (stream_pos), got {policy._next_use[2]}. "
            "New entries must NOT get INF or they become immediate eviction victims."
        )

        # Refresh counter should NOT be bumped — the protective sentinel
        # is sufficient; periodic refreshes handle the rest.
        assert policy._evictions_since_refresh == 0, (
            f"on_insert should not schedule early refreshes, "
            f"got counter={policy._evictions_since_refresh}"
        )


# ═════════════════════════════════════════════════════════════════════
# Workload Reordering Tests
# ═════════════════════════════════════════════════════════════════════

class TestWorkloadReordering:
    """Test that workload reordering produces valid indices."""

    @pytest.mark.slow
    def test_uniform_workload_valid(self):
        """Uniform workload should produce valid index array."""
        from src.benchmark.workload import generate_workload

        rng = np.random.RandomState(42)
        n = 100
        dim = 16
        embs = rng.randn(n, dim).astype(np.float32)
        db_embs = rng.randn(50, dim).astype(np.float32)

        indices = generate_workload(embs, db_embs, "uniform", n, seed=42)
        assert len(indices) == n, f"Expected {n} indices, got {len(indices)}"
        assert indices.max() < n, "Index out of bounds"
        assert indices.min() >= 0, "Negative index"

    @pytest.mark.slow
    def test_clustered_workload_valid(self):
        """Clustered workload should produce valid index array."""
        from src.benchmark.workload import generate_workload

        rng = np.random.RandomState(42)
        n = 100
        dim = 16
        query_embs = rng.randn(n, dim).astype(np.float32)
        db_embs = rng.randn(50, dim).astype(np.float32)

        indices = generate_workload(query_embs, db_embs, "clustered", n, seed=42)
        assert len(indices) == n
        assert indices.max() < n
        assert indices.min() >= 0

    @pytest.mark.slow
    def test_bursty_workload_valid(self):
        """Bursty workload should produce valid index array."""
        from src.benchmark.workload import generate_workload

        rng = np.random.RandomState(42)
        n = 100
        dim = 16
        query_embs = rng.randn(n, dim).astype(np.float32)
        db_embs = rng.randn(50, dim).astype(np.float32)

        indices = generate_workload(query_embs, db_embs, "bursty", n, seed=42)
        assert len(indices) == n
        assert indices.max() < n
        assert indices.min() >= 0
