"""
SISO eviction policy — faithful baseline reimplementation.

Paper
-----
"Rethinking Caching for LLM Serving Systems: Beyond Traditional
Heuristics" — Jungwoo Kim, Minsang Kim, Jaeheon Lee, Chanwoo Moon,
Heejin Kim, Taeho Hwang, Woosuk Chung, Yeseong Kim, Sungjin Lee.
arXiv:2508.18736 (submitted 26 Aug 2025).

SISO is a semantic caching system built on three ideas:

1. **Centroid-based caching** — each cached entry represents a semantic
   cluster.  Two metadata fields drive replacement:
   ``cluster_size`` (number of queries the centroid represents — the
   paper's measure of *semantic locality*, long-term value) and
   ``access_count`` (hits while cached — short-term popularity).

2. **Semantic-locality-aware replacement** (paper Algorithm 1,
   SISO-CacheManager):

   - *Merge* (lines 7-13): for each incoming centroid, find the closest
     cached centroid by cosine similarity.  If similarity > θ_C
     (clustering threshold, paper default **0.86**), fold its
     cluster_size into the closest centroid.  Otherwise it is a new
     centroid and its access_count is initialised to **∞** so it is
     prioritised over old ones until the next filtering round.
   - *Filtering* (lines 16-21): while over capacity, sort by
     ``(cluster_size, access_count)`` ascending and remove the first
     element.  Afterwards every survivor's cluster_size is divided by
     **1.1** (10 % decay) and every access_count is reset to **0**.

3. **Dynamic threshold adjustment** (§4.3, SISO-Server): the retrieval
   threshold θ_R is tuned within **[0.60, 0.98]** (the paper's T2H sweep
   range; the paper fixes θ_R = 0.86 for non-adaptive baselines).  The
   LLM system is modelled as an M/D/1 queue (paper Eq. 1-2):

       E = L · (1 − h(θ_R)),   W = E + λE² / (2(1 − λE))

   where ``L`` is the average LLM serving time, ``λ`` the arrival rate
   and ``h`` the cache hit ratio.  SISO picks the *highest* θ_R whose
   expected waiting time ``W`` stays below the SLO latency ``S``:
   lower θ_R under overload (more hits, less LLM traffic), raise it
   when there is slack (better answer quality).

Adaptations / deviations from the paper
---------------------------------------
The paper's cache manager runs *offline* over clustered query logs;
this codebase exposes an *online per-entry* hook interface
(``EvictionPolicy``).  Every deviation needed to bridge the two:

1. **Online merging.**  The paper's merge step *refuses* to add a
   centroid that lies within θ_C of a cached one (it only folds its
   cluster_size in).  Here the cache layer has already decided to
   insert every miss, so ``on_insert`` instead increments the closest
   cached entry's cluster_size by 1 when cosine > θ_C.  The
   near-duplicate itself is still cached, with cluster_size = 1 and
   access_count = 0 (no ∞ protection — mirroring that the paper would
   not have admitted it at all, it is the first filtering victim).
2. **∞ protection** (Alg. 1 lines 12-13) is applied to inserts that
   open a *new* semantic region (no cached entry within θ_C).
3. **Filtering-round bookkeeping** (lines 19-21: cluster_size /= 1.1,
   access_count = 0) cannot run at offline re-clustering time, so it
   runs every ``maintenance_interval`` requests (hits + misses).  The
   paper triggers re-clustering when newly accumulated queries reach
   ~10 % of the initial query set; the request-count interval is the
   online analogue and is a constructor parameter.
4. **One victim at a time.**  ``select_victim`` applies Alg. 1
   lines 17-18 (ascending ``(cluster_size, access_count)``) once per
   call instead of a batch while-loop.  Ties beyond the two keys are
   unspecified in the paper; broken here by insertion order (FIFO),
   matching the codebase's LFU convention and keeping runs
   deterministic.
5. **Dynamic θ_R without a T2H table.**  The paper builds a
   threshold-to-hit-ratio table by replaying sampled query logs against
   the cache — impossible through this hook interface (hits arrive as
   bare ``on_access(cache_id)`` without similarity scores).  Instead,
   the observed hit ratio over a sliding window of
   ``adjust_interval`` requests feeds the paper's M/D/1 formula
   directly, and θ_R moves by ``theta_r_step`` in the indicated
   direction (the paper itself applies such corrective stepping when
   the estimated W deviates >10 % from the actual).  ``arrival_rate``
   (λ), ``llm_latency`` (L) and ``slo_latency`` (S) are normalised
   constructor parameters because offline replay has no wall clock;
   the paper re-measures λ every 10 s.  As in the paper, θ_R does NOT
   influence victim selection — it is a serving-layer knob, exposed
   via ``current_threshold`` (cosine) / ``current_threshold_l2sq`` for
   the harness to consume when wired centrally.
6. **Bounded merge search.**  The paper finds the closest centroid via
   (an HNSW-accelerated) full search; to keep per-insert cost bounded
   like ``semantic.py``, at most ``max_merge_samples`` cached entries
   are scanned (seeded random sample when exceeded).
7. **Distance convention.**  Embeddings are unit-norm in this codebase
   and distances are squared L2 with ``cosine = 1 − L2²/2``.  The
   paper's cosine thresholds convert as ``L2² < 2·(1 − θ)``;
   e.g. θ_C = 0.86 ⇔ L2² < 0.28.

Hyperparameter defaults (paper values where published)
-------------------------------------------------------
- ``theta_c`` = 0.86            (§3.1/§4.1, clustering threshold θ_C)
- ``theta_r`` = 0.86            (§5.1, fixed θ_R of non-adaptive baselines)
- ``theta_r_min/max`` = 0.60 / 0.98   (§4.3, T2H sweep range)
- ``decay_factor`` = 1.1        (Alg. 1 line 20)
- ``slo_latency`` = 1.3, ``llm_latency`` = 1.0   (§5.1, SLO = 1.3 × E2E
  latency, normalised so L = 1)
- ``theta_r_step`` = 0.02, ``adjust_interval`` = 100,
  ``maintenance_interval`` = 1000, ``arrival_rate`` = 0.5,
  ``max_merge_samples`` = 1024   (adaptation parameters, see above)
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.cache.eviction.base import EvictionPolicy


class SISOPolicy(EvictionPolicy):
    """SISO semantic-locality eviction (Kim et al., arXiv:2508.18736)."""

    def __init__(
        self,
        theta_c: float = 0.86,
        theta_r: float = 0.86,
        theta_r_min: float = 0.60,
        theta_r_max: float = 0.98,
        theta_r_step: float = 0.02,
        decay_factor: float = 1.1,
        maintenance_interval: int = 1000,
        adjust_interval: int = 100,
        llm_latency: float = 1.0,
        arrival_rate: float = 0.5,
        slo_latency: float = 1.3,
        max_merge_samples: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        self.theta_c = float(theta_c)
        self.theta_r_min = float(theta_r_min)
        self.theta_r_max = float(theta_r_max)
        self.theta_r_step = float(theta_r_step)
        self.decay_factor = float(decay_factor)
        self.maintenance_interval = int(maintenance_interval)
        self.adjust_interval = int(adjust_interval)
        self.llm_latency = float(llm_latency)
        self.arrival_rate = float(arrival_rate)
        self.slo_latency = float(slo_latency)
        self.max_merge_samples = int(max_merge_samples)
        self._seed = 0 if seed is None else int(seed)

        # Cosine merge threshold → squared-L2 on unit-norm embeddings.
        self._theta_c_l2sq = 2.0 * (1.0 - self.theta_c)

        # ── Dynamic retrieval threshold θ_R (§4.3) ───────────────
        self._theta_r = float(theta_r)

        # ── Centroid metadata (Alg. 1) ───────────────────────────
        # Insertion order doubles as the deterministic final tie-break
        # (front = oldest insert).  Values are unused (set semantics).
        self._order: OrderedDict[int, None] = OrderedDict()
        # cluster_size — semantic locality (float: decays by /1.1).
        self._cluster_size: dict[int, float] = {}
        # access_count — short-term popularity (float: ∞ for new regions).
        self._access_count: dict[int, float] = {}
        # Embeddings, needed for the merge-step closest-centroid search.
        self._embeddings: dict[int, np.ndarray] = {}

        # Requests (hits + misses) since the last maintenance round —
        # the online analogue of the paper's accumulated-query trigger
        # (deviation #3).
        self._request_count = 0

        # Sliding window feeding the θ_R adjustment (deviation #5).
        self._window_requests = 0
        self._window_hits = 0

    # ── Lifecycle hooks ──────────────────────────────────────────

    def on_access(self, cache_id: int) -> None:
        """Cache hit — increment the entry's access_count (short-term
        popularity, §4.2)."""
        if cache_id in self._access_count:
            self._access_count[cache_id] += 1.0
        self._tick(hit=True)

    def on_insert(self, cache_id: int, embedding: np.ndarray) -> None:
        """Register a new entry and run the merge step (Alg. 1 lines 7-13).

        If the closest cached entry lies within θ_C (cosine), one unit
        of cluster_size is folded into it — the online analogue of the
        paper merging an incoming centroid of size 1.
        """
        emb = embedding.astype(np.float32).copy()
        self._order[cache_id] = None
        self._cluster_size[cache_id] = 1.0
        self._embeddings[cache_id] = emb

        closest_id = self._find_closest(cache_id, emb)
        if closest_id is not None:
            # Lines 9-10: fold the incoming unit mass into the closest
            # centroid's cluster_size.  The near-duplicate itself gets
            # NO ∞ protection (the paper would not have admitted it).
            self._cluster_size[closest_id] += 1.0
            self._access_count[cache_id] = 0.0
        else:
            # Lines 12-13: a genuinely new centroid is prioritised over
            # old ones until the next maintenance round resets counts.
            self._access_count[cache_id] = float("inf")
        self._tick(hit=False)

    def on_evict(self, cache_id: int) -> None:
        """Drop all metadata for the evicted entry."""
        self._order.pop(cache_id, None)
        self._cluster_size.pop(cache_id, None)
        self._access_count.pop(cache_id, None)
        self._embeddings.pop(cache_id, None)

    def select_victim(self, active_ids: set[int]) -> Optional[int]:
        """Alg. 1 lines 17-18: remove the entry with the smallest
        ``(cluster_size, access_count)``, ascending.

        Single pass over insertion order: strict ``<`` comparison means
        the first entry seen at the global minimum key is the
        oldest-inserted one — the FIFO tie-break for free.  Only IDs in
        *active_ids* are considered.
        """
        victim: Optional[int] = None
        best_key: Optional[tuple[float, float]] = None

        for cid in self._order:
            if cid not in active_ids:
                continue
            key = (self._cluster_size[cid], self._access_count[cid])
            if best_key is None or key < best_key:
                best_key = key
                victim = cid

        return victim

    def on_rebuild(self, id_remap: dict[int, int]) -> None:
        """Remap all per-entry state to the new ids, preserving
        insertion order; ids absent from the remap are dropped."""
        new_order: OrderedDict[int, None] = OrderedDict()
        new_cluster: dict[int, float] = {}
        new_access: dict[int, float] = {}
        new_embeddings: dict[int, np.ndarray] = {}
        for old_id in self._order:
            if old_id not in id_remap:
                continue
            new_id = id_remap[old_id]
            new_order[new_id] = None
            new_cluster[new_id] = self._cluster_size[old_id]
            new_access[new_id] = self._access_count[old_id]
            new_embeddings[new_id] = self._embeddings[old_id]
        self._order = new_order
        self._cluster_size = new_cluster
        self._access_count = new_access
        self._embeddings = new_embeddings

    # ── Helpers ──────────────────────────────────────────────────

    def _tick(self, hit: bool) -> None:
        """Count one request (hit or miss) toward the maintenance round
        and the θ_R adjustment window."""
        self._request_count += 1
        if self._request_count >= self.maintenance_interval:
            self._maintenance_round()
            self._request_count = 0

        self._window_requests += 1
        if hit:
            self._window_hits += 1
        if self._window_requests >= self.adjust_interval:
            self._adjust_theta_r()
            self._window_requests = 0
            self._window_hits = 0

    def _adjust_theta_r(self) -> None:
        """§4.3: step θ_R against the M/D/1 waiting-time estimate.

        E = L·(1−h);  W = E + λE² / (2(1−λE)).  W above the SLO means
        the system is overloaded → lower θ_R (more cache hits, less LLM
        traffic); otherwise raise it (better answer quality).  An
        unstable queue (λE ≥ 1) counts as overload.
        """
        h = self._window_hits / self._window_requests
        expected_service = self.llm_latency * (1.0 - h)
        rho = self.arrival_rate * expected_service
        if rho >= 1.0:
            overloaded = True
        else:
            waiting = expected_service + (
                self.arrival_rate * expected_service ** 2
            ) / (2.0 * (1.0 - rho))
            overloaded = waiting > self.slo_latency
        step = -self.theta_r_step if overloaded else self.theta_r_step
        self._theta_r = float(
            np.clip(self._theta_r + step, self.theta_r_min, self.theta_r_max)
        )

    def _maintenance_round(self) -> None:
        """Alg. 1 lines 19-21, run every ``maintenance_interval``
        requests: divide every cluster_size by ``decay_factor`` and
        zero every access_count (clearing ∞ protection of new
        centroids)."""
        for cid in self._cluster_size:
            self._cluster_size[cid] /= self.decay_factor
        for cid in self._access_count:
            self._access_count[cid] = 0.0

    def _find_closest(
        self, cache_id: int, emb: np.ndarray
    ) -> Optional[int]:
        """Closest cached entry within θ_C of *emb*, or None.

        FindClosestCentroid (Alg. 1 line 8) over at most
        ``max_merge_samples`` cached entries (seeded sample when the
        cache is larger — deviation #6, bounds per-insert cost like
        ``semantic.py``).  Distances are squared L2 on unit-norm
        embeddings; cosine > θ_C ⇔ L2² < 2(1 − θ_C) (deviation #7).
        """
        other_ids = [cid for cid in self._embeddings if cid != cache_id]
        n_others = len(other_ids)
        if n_others == 0:
            return None

        if n_others > self.max_merge_samples:
            rng = np.random.RandomState(self._mix_seed(cache_id))
            other_ids = rng.choice(
                other_ids, self.max_merge_samples, replace=False,
            ).tolist()

        others = np.array(
            [self._embeddings[cid] for cid in other_ids], dtype=np.float32,
        )
        q = emb.reshape(1, -1)
        dist_sq = (
            float(np.sum(q ** 2))
            + np.sum(others ** 2, axis=1)
            - 2.0 * (q @ others.T)[0]
        )

        best = int(np.argmin(dist_sq))
        # Strict inequality, matching the paper's "CosineSimilarity > θ_C".
        if dist_sq[best] < self._theta_c_l2sq:
            return other_ids[best]
        return None

    def _mix_seed(self, cache_id: int) -> int:
        """Mix the base seed with a cache_id into a 31-bit RNG seed
        (same Knuth-style mix as ``semantic.py``)."""
        mix = (
            int(self._seed) * 2654435769 + int(cache_id) * 40503 + 1
        ) & 0x7FFFFFFF
        return mix or 1

    # ── Serving-layer views (θ_R is not used in victim selection) ──

    @property
    def current_threshold(self) -> float:
        """Current retrieval threshold θ_R as cosine similarity."""
        return self._theta_r

    @property
    def current_threshold_l2sq(self) -> float:
        """θ_R as squared-L2 distance on unit-norm embeddings."""
        return 2.0 * (1.0 - self._theta_r)

    # ── Representation ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "siso"

    def __repr__(self) -> str:
        return (
            f"SISOPolicy(theta_c={self.theta_c}, theta_r={self._theta_r:.3f}, "
            f"decay={self.decay_factor}, "
            f"maint_every={self.maintenance_interval}, "
            f"adjust_every={self.adjust_interval}, seed={self._seed})"
        )
