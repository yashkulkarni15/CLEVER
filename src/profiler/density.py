"""
Workload density metric for semantic-cache experiments.

The metric is defined over a cache snapshot C:

    r(e) = fraction of other cached entries within L2-squared distance theta
    mean_density = average r(e) over cached entries

Self-neighbors are excluded. Empty and singleton snapshots have density 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DensitySnapshot:
    """Density statistics for one cache snapshot."""

    mean_density: float
    cache_snapshot_size: int
    theta_l2sq: float
    distance_metric: str
    include_self: bool
    density_probe_size: int
    density_anchor_size: int
    sampled: bool
    seed: int
    per_entry_density: np.ndarray


def active_embedding_matrix(cache) -> tuple[list[int], np.ndarray]:
    """
    Return active cache IDs and embeddings from a SemanticCache-like object.

    This is intentionally read-only and avoids depending on FAISS search output,
    which can include logically deleted entries in this project.
    """
    active_ids = sorted(cache._active)
    embeddings = np.array(
        [cache._entries[cache_id].embedding for cache_id in active_ids],
        dtype=np.float32,
    )
    if embeddings.size == 0:
        embeddings = np.empty((0, getattr(cache, "dim", 0)), dtype=np.float32)
    return active_ids, embeddings


def compute_density_snapshot(
    embeddings: np.ndarray,
    *,
    theta_l2sq: float,
    active_ids: Sequence[int] | None = None,
    probe_size: int | None = None,
    anchor_size: int | None = None,
    seed: int = 0,
) -> DensitySnapshot:
    """
    Compute mean cached-entry density for an embedding snapshot.

    Args:
        embeddings: Embedding matrix. If ``active_ids`` is provided, rows are
            selected from this matrix before density is computed.
        theta_l2sq: Inclusive L2-squared neighbor threshold.
        active_ids: Optional row indices representing active cache entries.
        probe_size: Optional number of entries to estimate density for.
        anchor_size: Optional number of comparison entries for sampled mode.
        seed: Deterministic sampling seed.

    Returns:
        DensitySnapshot with exact or sampled density values.
    """
    if theta_l2sq < 0:
        raise ValueError("theta_l2sq must be non-negative")

    matrix = _as_embedding_matrix(embeddings, active_ids)
    n = matrix.shape[0]
    if n <= 1:
        return DensitySnapshot(
            mean_density=0.0,
            cache_snapshot_size=n,
            theta_l2sq=float(theta_l2sq),
            distance_metric="l2_squared",
            include_self=False,
            density_probe_size=n,
            density_anchor_size=n,
            sampled=False,
            seed=int(seed),
            per_entry_density=np.zeros(n, dtype=np.float32),
        )

    sampled = _uses_sampling(n, probe_size, anchor_size)
    probe_idx, anchor_idx = _choose_probe_and_anchor_indices(
        n, probe_size=probe_size, anchor_size=anchor_size, seed=seed,
    )
    density = _density_for_probe_anchor(matrix, probe_idx, anchor_idx, theta_l2sq)

    return DensitySnapshot(
        mean_density=float(np.mean(density)) if density.size else 0.0,
        cache_snapshot_size=n,
        theta_l2sq=float(theta_l2sq),
        distance_metric="l2_squared",
        include_self=False,
        density_probe_size=int(len(probe_idx)),
        density_anchor_size=int(len(anchor_idx)),
        sampled=sampled,
        seed=int(seed),
        per_entry_density=density.astype(np.float32, copy=False),
    )


def _as_embedding_matrix(
    embeddings: np.ndarray,
    active_ids: Sequence[int] | None,
) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {matrix.shape}")

    if active_ids is not None:
        active = np.asarray(list(active_ids), dtype=np.int64)
        matrix = matrix[active]

    return np.ascontiguousarray(matrix, dtype=np.float32)


def _uses_sampling(
    n: int,
    probe_size: int | None,
    anchor_size: int | None,
) -> bool:
    return (
        probe_size is not None
        and probe_size < n
    ) or (
        anchor_size is not None
        and anchor_size < n
    )


def _choose_probe_and_anchor_indices(
    n: int,
    *,
    probe_size: int | None,
    anchor_size: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    probe_n = _bounded_sample_size(probe_size, n)
    anchor_n = _bounded_sample_size(anchor_size, n)

    if probe_n == n and anchor_n == n:
        full = np.arange(n, dtype=np.int64)
        return full, full

    rng = np.random.RandomState(seed)
    probe_idx = rng.choice(n, size=probe_n, replace=False)
    anchor_idx = rng.choice(n, size=anchor_n, replace=False)
    probe_idx.sort()
    anchor_idx.sort()
    return probe_idx.astype(np.int64), anchor_idx.astype(np.int64)


def _bounded_sample_size(size: int | None, n: int) -> int:
    if size is None or size <= 0 or size >= n:
        return n
    return int(size)


def _density_for_probe_anchor(
    matrix: np.ndarray,
    probe_idx: np.ndarray,
    anchor_idx: np.ndarray,
    theta_l2sq: float,
) -> np.ndarray:
    probes = matrix[probe_idx]
    anchors = matrix[anchor_idx]

    probe_norms = np.sum(probes * probes, axis=1, keepdims=True)
    anchor_norms = np.sum(anchors * anchors, axis=1, keepdims=True).T
    distances = probe_norms + anchor_norms - 2.0 * (probes @ anchors.T)
    distances = np.maximum(distances, 0.0)

    neighbors = distances <= float(theta_l2sq)
    same_entry = probe_idx[:, None] == anchor_idx[None, :]
    neighbors[same_entry] = False

    denominators = len(anchor_idx) - same_entry.sum(axis=1)
    counts = neighbors.sum(axis=1)

    density = np.divide(
        counts,
        denominators,
        out=np.zeros(len(probe_idx), dtype=np.float32),
        where=denominators > 0,
    )
    return density.astype(np.float32, copy=False)
