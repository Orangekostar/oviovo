from __future__ import annotations

from typing import Protocol

from src.core.data_structures import Frame
from src.domain.association import AssociationDecisionBatch, EntityResolutionBatch
from src.domain.entities import LifecycleDelta
from src.domain.mapping import FusionDelta, OwnershipDelta
from src.domain.observations import ObservationBatch
from src.domain.snapshots import MapSnapshot
from src.domain.tracking import LocalTrackBatch
from src.domain.visibility import VisibilityEvidenceBatch


class ObservationFrontend(Protocol):
    def observe(self, frame: Frame) -> ObservationBatch: ...


class LocalTemporalModule(Protocol):
    def update(self, observations: ObservationBatch) -> LocalTrackBatch: ...


class VisibilityModule(Protocol):
    def evaluate(
        self,
        frame: Frame,
        tracks: LocalTrackBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> VisibilityEvidenceBatch: ...


class EntityAssociation(Protocol):
    def associate(
        self,
        tracks: LocalTrackBatch,
        visibility: VisibilityEvidenceBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> AssociationDecisionBatch: ...


class EntityRegistry(Protocol):
    def resolve(
        self,
        tracks: LocalTrackBatch,
        decisions: AssociationDecisionBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> EntityResolutionBatch: ...


class VoxelFusion(Protocol):
    def fuse(
        self,
        frame: Frame,
        observations: ObservationBatch,
        resolutions: EntityResolutionBatch,
        previous_snapshot: MapSnapshot | None,
    ) -> FusionDelta: ...


class OwnershipManager(Protocol):
    def update(
        self,
        fusion: FusionDelta,
        previous_snapshot: MapSnapshot | None,
    ) -> OwnershipDelta: ...


class LifecycleManager(Protocol):
    def update(
        self,
        visibility: VisibilityEvidenceBatch,
        decisions: AssociationDecisionBatch,
        ownership: OwnershipDelta,
        previous_snapshot: MapSnapshot | None,
    ) -> LifecycleDelta: ...


class MapCommitter(Protocol):
    def commit(
        self,
        frame: Frame,
        resolutions: EntityResolutionBatch,
        fusion: FusionDelta,
        ownership: OwnershipDelta,
        lifecycle: LifecycleDelta,
        previous_snapshot: MapSnapshot | None,
    ) -> MapSnapshot: ...


class MapQuery(Protocol):
    def query(self, snapshot: MapSnapshot, text: str, top_k: int = 5) -> tuple[str, ...]: ...
