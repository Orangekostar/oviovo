"""Core types for the simplified V2 semantic mapping pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np


class V2ObjectState(str, Enum):
    """Minimal lifecycle states for the simplified V2 semantic map."""

    PROVISIONAL = "provisional"
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class ObjectPool:
    """Fine object-centric point pool with frozen semantic label support."""

    object_id: int
    canonical_label: str = ""
    label_confidence: float = 0.0
    label_frozen: bool = False
    label_candidate_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    point_observation_count: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    point_last_seen_frame: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    point_confidence: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    spatial_index_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    last_seen_frame: int = 0
    observation_count: int = 0
    state: V2ObjectState = V2ObjectState.ACTIVE
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return int(len(self.points))


@dataclass
class ProvisionalSpatialBucket:
    """Spatial bucket for unresolved object candidates before promotion."""

    bucket_id: int
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    point_observation_count: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    point_last_seen_frame: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    point_confidence: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    spatial_index_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    candidate_label_votes: Dict[str, float] = field(default_factory=dict)
    label_candidate_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hit_count: int = 0
    last_seen_frame: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoarseVoxelOwner:
    """Compact semantic ownership record for one voxel."""

    voxel_key: Tuple[int, int, int]
    owner_object_id: int = -1
    owner_confidence: float = 0.0
    canonical_label: str = ""
    last_seen_frame: int = 0


@dataclass
class SemanticMapV2State:
    """State container for the simplified V2 semantic mapping pipeline."""

    object_pools: Dict[int, ObjectPool] = field(default_factory=dict)
    provisional_buckets: Dict[int, ProvisionalSpatialBucket] = field(default_factory=dict)
    coarse_owners: Dict[Tuple[int, int, int], CoarseVoxelOwner] = field(default_factory=dict)
    coarse_support: Dict[Tuple[int, int, int], Dict[int, float]] = field(default_factory=dict)
    geometry_accum: Dict[Tuple[int, int, int], list[np.ndarray | int]] = field(default_factory=dict)
    next_object_id: int = 0
    next_bucket_id: int = 0
    frame_count: int = 0
    last_frame_debug: Dict[str, Any] = field(default_factory=dict)
