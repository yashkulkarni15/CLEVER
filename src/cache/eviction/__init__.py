"""
Eviction policy implementations for the semantic cache.

Available policies:

- ``LRUPolicy``      — Least Recently Used (baseline)
- ``LFUPolicy``      — Least Frequently Used (baseline)
- ``SemanticPolicy``  — Semantic-aware (novel contribution)
- ``AdaptiveHardSwitchPolicy`` — Online LRU/LFU/Semantic hard switch
- ``AdaptiveBlendPolicy`` — Online blended recency/frequency/redundancy score
- ``OraclePolicy``    — Bélády's optimal (upper bound, offline only)
"""

from src.cache.eviction.base import EvictionPolicy
from src.cache.eviction.lru import LRUPolicy
from src.cache.eviction.lfu import LFUPolicy
from src.cache.eviction.semantic import SemanticPolicy
from src.cache.eviction.adaptive import AdaptiveHardSwitchPolicy, AdaptiveBlendPolicy
from src.cache.eviction.oracle import OraclePolicy

POLICY_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "semantic": SemanticPolicy,
    "adaptive_hard": AdaptiveHardSwitchPolicy,
    "adaptive_blend": AdaptiveBlendPolicy,
    "oracle": OraclePolicy,
}

__all__ = [
    "EvictionPolicy",
    "LRUPolicy",
    "LFUPolicy",
    "SemanticPolicy",
    "AdaptiveHardSwitchPolicy",
    "AdaptiveBlendPolicy",
    "OraclePolicy",
    "POLICY_REGISTRY",
]
