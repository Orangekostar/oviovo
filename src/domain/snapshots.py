from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.domain.entities import PersistentEntity
from src.domain.mapping import OwnershipLayer


class SnapshotScope(str, Enum):
    CURRENT = "current"
    HISTORY = "history"


@dataclass(frozen=True)
class MapSnapshot:
    scene_id: str
    timestamp: float
    revision: int
    scope: SnapshotScope
    entities: tuple[PersistentEntity, ...]
    geometry_revision: int
    voxel_evidence_revision: int
    ownership: OwnershipLayer
    lifecycle_revision: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("snapshot timestamp must be finite")
        entities = tuple(self.entities)
        entity_ids = [entity.entity_id for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique within a snapshot")
        revisions = (
            self.revision,
            self.geometry_revision,
            self.voxel_evidence_revision,
            self.lifecycle_revision,
        )
        if any(int(value) < 0 for value in revisions):
            raise ValueError("snapshot revisions must be non-negative")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "entities", entities)
