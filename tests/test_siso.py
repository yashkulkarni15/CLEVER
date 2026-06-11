"""
Unit tests for the SISO eviction policy baseline.

SISO: "Rethinking Caching for LLM Serving Systems: Beyond Traditional
Heuristics" — Kim et al., arXiv:2508.18736 (Aug 2025).

Behaviors under test (paper Algorithm 1 + §4.3):
- select_victim returns None on an empty active set.
- Cold start: all-equal (cluster_size, access_count) ties break by
  insertion order (FIFO), deterministically.
- Victims never come from outside ``active_ids``.
- Merge step: an insert within theta_c of an existing entry increments
  the *closest* entry's cluster_size (Alg. 1 lines 9-10); the merged
  insert itself gets access_count = 0, not infinity.
- New-region inserts (no neighbour within theta_c) are protected with
  access_count = infinity (Alg. 1 lines 12-13).
- Maintenance round: cluster_size /= decay_factor and access_count
  reset to 0 (Alg. 1 lines 19-21).
- Eviction rule: ascending (cluster_size, access_count) (lines 17-18).
- Dynamic threshold adjustment moves theta_r down under overload and
  up under light load, clamped to [theta_r_min, theta_r_max] (§4.3).
- on_rebuild remaps all internal state.

Run with:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE ./venv/bin/python -m pytest tests/test_siso.py -q
"""

import numpy as np
import pytest

from src.cache.eviction.siso import SISOPolicy


# ── Helpers ──────────────────────────────────────────────────────────

DIM = 16


def _axis(i: int, dim: int = DIM) -> np.ndarray:
    """Unit-norm basis vector e_i — mutually orthogonal (cosine = 0)."""
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _near(base: np.ndarray, other_axis: int, eps: float = 0.1) -> np.ndarray:
    """Unit-norm vector close to *base* (cosine = 1/sqrt(1+eps^2) ≈ 0.995
    for eps = 0.1, comfortably above theta_c = 0.86)."""
    v = base.astype(np.float32).copy()
    v[other_axis] += eps
    v /= np.linalg.norm(v)
    return v


# ═════════════════════════════════════════════════════════════════════
# Victim selection basics
# ═════════════════════════════════════════════════════════════════════

class TestVictimSelectionBasics:

    def test_empty_active_set_returns_none(self):
        """select_victim on an empty active set must return None."""
        policy = SISOPolicy()
        policy.on_insert(0, _axis(0))
        assert policy.select_victim(set()) is None

    def test_cold_start_tie_breaks_by_insertion_order(self):
        """Cold start (paper-specific edge case): all entries tie on
        (cluster_size, access_count) — three mutually orthogonal inserts
        never merge, so all carry identical metadata.  Alg. 1 line 17
        sorts without specifying further tie-breaks; we break by
        insertion order (FIFO), so the first insert is the victim."""
        policy = SISOPolicy()
        for i in range(3):
            policy.on_insert(i, _axis(i))

        victim = policy.select_victim({0, 1, 2})
        assert victim == 0, f"Cold-start FIFO tie-break expected 0, got {victim}"

    def test_victim_always_within_active_ids(self):
        """select_victim must never return an ID outside active_ids,
        even when the globally-minimal entry is not active."""
        policy = SISOPolicy()
        for i in range(4):
            policy.on_insert(i, _axis(i))

        # Entry 0 is the global FIFO minimum but is NOT active.
        victim = policy.select_victim({1, 3})
        assert victim in {1, 3}, f"Victim {victim} outside active set"
        assert victim == 1, "FIFO tie-break among active entries only"


# ═════════════════════════════════════════════════════════════════════
# Merge step (Alg. 1 lines 7-13)
# ═════════════════════════════════════════════════════════════════════

class TestMergeStep:

    def test_merge_increments_closest_cluster_size(self):
        """An insert with cosine > theta_c to a cached entry folds one
        unit of cluster_size into the *closest* entry (lines 9-10).

        Hand computation (theta_c = 0.86):
          e0 = axis(0), e1 = axis(1):  cos(e0, e1) = 0          → no merge
          d  = near(e0): cos(e0, d) = 1/sqrt(1.01) ≈ 0.995 > 0.86 → merge
                          cos(e1, d) ≈ 0.0995 < 0.86

        Expected: cluster_size = {e0: 2, e1: 1, d: 1};
        the merged near-duplicate d gets access_count = 0 (it would not
        have been admitted as a new centroid by the paper, so it does
        NOT receive the new-centroid ∞ protection).
        Eviction consequence: between {e0, e1}, e1 has the smaller
        cluster_size and is the victim.
        """
        policy = SISOPolicy()
        policy.on_insert(0, _axis(0))
        policy.on_insert(1, _axis(1))
        policy.on_insert(2, _near(_axis(0), other_axis=5))

        assert policy._cluster_size[0] == 2.0, \
            f"Closest entry must absorb the merge, got {policy._cluster_size[0]}"
        assert policy._cluster_size[1] == 1.0, \
            "Non-closest entry must be untouched"
        assert policy._cluster_size[2] == 1.0, \
            "Merged insert itself starts at cluster_size 1"
        assert policy._access_count[2] == 0.0, \
            "Merged (near-duplicate) insert must NOT get ∞ protection"

        victim = policy.select_victim({0, 1})
        assert victim == 1, \
            "Entry with the larger cluster_size (semantic locality) is protected"


# ═════════════════════════════════════════════════════════════════════
# Access counts (Alg. 1 lines 17-18 tie-break)
# ═════════════════════════════════════════════════════════════════════

class TestAccessCount:

    def test_access_count_breaks_cluster_size_ties(self):
        """Among entries with equal cluster_size, the one with the
        smallest access_count is removed (Alg. 1 lines 17-18).

        d1 and d2 are both near-duplicates of e0 (closest to e0, so
        both merge into e0 and start with access_count = 0 and
        cluster_size = 1).  After one hit on d1, d2 has the smaller
        access_count and must be the victim — even though d1 was
        inserted first (FIFO alone would pick d1).
        """
        policy = SISOPolicy()
        e0 = _axis(0)
        policy.on_insert(0, e0)
        policy.on_insert(1, _near(e0, other_axis=5))   # merged, ac = 0
        policy.on_insert(2, _near(e0, other_axis=6))   # merged, ac = 0

        policy.on_access(1)

        assert policy._access_count[1] == 1.0
        assert policy._access_count[2] == 0.0

        victim = policy.select_victim({1, 2})
        assert victim == 2, \
            f"Smallest access_count must lose the tie, got {victim}"

    def test_new_region_insert_protected_with_infinite_access_count(self):
        """Alg. 1 lines 12-13: a centroid added as NEW (no cached entry
        within theta_c) gets access_count = ∞, prioritising it over old
        centroids until the next maintenance round resets counts.

        d1 (near-duplicate of e0) merges → access_count 0, then gets one
        hit (access_count 1).  e2 is orthogonal to everything → new
        region → access_count ∞.  At equal cluster_size, d1 (finite
        count) must be the victim, NOT the brand-new e2.
        """
        policy = SISOPolicy()
        e0 = _axis(0)
        policy.on_insert(0, e0)
        policy.on_insert(1, _near(e0, other_axis=5))   # merged, ac = 0
        policy.on_access(1)                             # ac = 1

        policy.on_insert(2, _axis(2))                   # new region

        assert policy._access_count[2] == float("inf"), \
            "New-region insert must receive ∞ access_count"

        victim = policy.select_victim({1, 2})
        assert victim == 1, \
            f"Brand-new centroid must be protected by ∞, got victim {victim}"


# ═════════════════════════════════════════════════════════════════════
# Maintenance round (Alg. 1 lines 19-21)
# ═════════════════════════════════════════════════════════════════════

class TestMaintenanceRound:

    def test_maintenance_decays_cluster_size_and_resets_access_count(self):
        """After each filtering round the paper divides every
        cluster_size by 1.1 (line 20) and zeroes every access_count
        (line 21) — clearing the ∞ protection of new centroids.

        With maintenance_interval = 3, the third request (insert)
        triggers the round: all three orthogonal entries end with
        cluster_size = 1/1.1 and access_count = 0.
        """
        policy = SISOPolicy(maintenance_interval=3)
        for i in range(3):
            policy.on_insert(i, _axis(i))

        for i in range(3):
            assert policy._cluster_size[i] == pytest.approx(1.0 / 1.1), \
                f"cluster_size[{i}] not decayed: {policy._cluster_size[i]}"
            assert policy._access_count[i] == 0.0, \
                f"access_count[{i}] not reset: {policy._access_count[i]}"

    def test_accesses_count_toward_maintenance_interval(self):
        """The paper's re-clustering trigger counts *accumulated
        queries* — hits and misses alike.  Two inserts + one access
        must fire the round at interval 3."""
        policy = SISOPolicy(maintenance_interval=3)
        policy.on_insert(0, _axis(0))
        policy.on_insert(1, _axis(1))
        policy.on_access(0)  # third request → maintenance fires

        assert policy._cluster_size[0] == pytest.approx(1.0 / 1.1)
        assert policy._access_count[0] == 0.0, \
            "Reset must also clear the count incremented by this access"
        assert policy._access_count[1] == 0.0, "∞ must be cleared"


# ═════════════════════════════════════════════════════════════════════
# Rebuild remapping
# ═════════════════════════════════════════════════════════════════════


class TestRebuild:

    def test_rebuild_remaps_metadata_and_drops_stale_ids(self):
        """on_rebuild must move cluster_size/access_count/embeddings to
        the new ids, preserve insertion order, and drop ids absent from
        the remap (the compaction convention shared with LRU/LFU)."""
        policy = SISOPolicy()
        policy.on_insert(0, _axis(0))            # new region → inf
        policy.on_insert(1, _near(_axis(0), 3))  # merges into 0
        policy.on_insert(2, _axis(5))            # new region → inf

        policy.on_rebuild({0: 10, 2: 11})        # id 1 compacted away

        assert set(policy._cluster_size) == {10, 11}
        assert policy._cluster_size[10] == pytest.approx(2.0)
        assert policy._cluster_size[11] == pytest.approx(1.0)
        assert policy._access_count[10] == float("inf")
        assert set(policy._embeddings) == {10, 11}
        assert np.allclose(policy._embeddings[10], _axis(0))
        assert list(policy._order) == [10, 11]

        # Victim selection works on the new ids: smaller cluster wins.
        assert policy.select_victim({10, 11}) == 11


# ═════════════════════════════════════════════════════════════════════
# Dynamic retrieval-threshold adjustment (§4.3)
# ═════════════════════════════════════════════════════════════════════


class TestDynamicThreshold:
    """θ_R adapts every ``adjust_interval`` requests from the observed
    window hit ratio via the paper's M/D/1 model:
    E = L(1−h), W = E + λE²/(2(1−λE)). W > SLO → lower θ_R (favor
    hits); otherwise raise it (favor quality). Clamped to
    [theta_r_min, theta_r_max]. θ_R never affects victim selection."""

    def test_overload_lowers_theta_r(self):
        # All misses: h=0, E=1, W = 1 + 0.5/(2·0.5) = 1.5 > SLO 1.3.
        policy = SISOPolicy(adjust_interval=4, theta_r=0.86,
                            theta_r_step=0.02)
        for i in range(4):
            policy.on_insert(i, _axis(i))
        assert policy.current_threshold == pytest.approx(0.84)

    def test_light_load_raises_theta_r(self):
        # 1 miss + 3 hits: h=0.75, E=0.25, W≈0.268 < SLO 1.3.
        policy = SISOPolicy(adjust_interval=4, theta_r=0.86,
                            theta_r_step=0.02)
        policy.on_insert(0, _axis(0))
        for _ in range(3):
            policy.on_access(0)
        assert policy.current_threshold == pytest.approx(0.88)

    def test_theta_r_clamped_to_bounds(self):
        policy = SISOPolicy(adjust_interval=2, theta_r=0.97,
                            theta_r_step=0.02, theta_r_max=0.98)
        policy.on_insert(0, _axis(0))
        policy.on_access(0)        # h=0.5 → W<SLO → raise, clamped
        assert policy.current_threshold == pytest.approx(0.98)
        policy.on_access(0)
        policy.on_access(0)        # second round: stays at the cap
        assert policy.current_threshold == pytest.approx(0.98)

    def test_l2sq_view_of_threshold(self):
        policy = SISOPolicy(theta_r=0.86)
        assert policy.current_threshold_l2sq == pytest.approx(
            2.0 * (1.0 - 0.86))
