"""
Eviction policy implementations for the semantic cache.

Available policies:

- ``LRUPolicy``      — Least Recently Used (baseline)
- ``LFUPolicy``      — Least Frequently Used (baseline)
- ``SemanticPolicy``  — Semantic-aware (novel contribution)
- ``AdaptiveHardSwitchPolicy`` — Online LRU/LFU/Semantic hard switch
- ``AdaptiveBlendPolicy`` — Online blended recency/frequency/redundancy score
- ``ARCPolicy``       — Adaptive Replacement Cache (Megiddo & Modha 2003)
- ``GDSFPolicy``      — Greedy-Dual-Size-Frequency (Cherkasova 1998)
- ``SISOPolicy``      — Semantic-locality eviction (Kim et al. 2025)
- ``OraclePolicy``    — Bélády's optimal (upper bound, offline only)
"""

from src.cache.eviction.base import EvictionPolicy
from src.cache.eviction.lru import LRUPolicy
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.semantic import SemanticPolicy
from src.cache.eviction.adaptive import AdaptiveHardSwitchPolicy, AdaptiveBlendPolicy
from src.cache.eviction.arc import ARCPolicy
from src.cache.eviction.gdsf import GDSFPolicy
from src.cache.eviction.siso import SISOPolicy
from src.cache.eviction.oracle import OraclePolicy

POLICY_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "semantic": SemanticPolicy,
    "adaptive_hard": AdaptiveHardSwitchPolicy,
    "adaptive_blend": AdaptiveBlendPolicy,
    "arc": ARCPolicy,
    "gdsf": GDSFPolicy,
    "siso": SISOPolicy,
    "oracle": OraclePolicy,
}

__all__ = [
    "EvictionPolicy",
    "LRUPolicy",
    "LFUPolicy",
    "SemanticPolicy",
    "AdaptiveHardSwitchPolicy",
    "AdaptiveBlendPolicy",
    "ARCPolicy",
    "GDSFPolicy",
    "SISOPolicy",
    "OraclePolicy",
    "POLICY_REGISTRY",
]
