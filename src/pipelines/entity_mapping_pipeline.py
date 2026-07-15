from __future__ import annotations

from dataclasses import dataclass

from src.core.data_structures import Frame
from src.domain.snapshots import MapSnapshot
from src.pipelines.entity_mapping_interfaces import (
    EntityAssociation,
    EntityRegistry,
    LifecycleManager,
    LocalTemporalModule,
    MapCommitter,
    ObservationFrontend,
    OwnershipManager,
    VisibilityModule,
    VoxelFusion,
)


@dataclass(frozen=True)
class EntityMappingModules:
    frontend: ObservationFrontend
    temporal: LocalTemporalModule
    visibility: VisibilityModule
    association: EntityAssociation
    registry: EntityRegistry
    fusion: VoxelFusion
    ownership: OwnershipManager
    lifecycle: LifecycleManager
    committer: MapCommitter


class EntityMappingPipeline:
    def __init__(self, modules: EntityMappingModules) -> None:
        self.modules = modules
        self.current_snapshot: MapSnapshot | None = None

    def process_frame(self, frame: Frame) -> MapSnapshot:
        previous = self.current_snapshot
        observations = self.modules.frontend.observe(frame)
        tracks = self.modules.temporal.update(observations)
        visibility = self.modules.visibility.evaluate(frame, tracks, previous)
        decisions = self.modules.association.associate(tracks, visibility, previous)
        resolutions = self.modules.registry.resolve(tracks, decisions, previous)
        fusion = self.modules.fusion.fuse(frame, observations, resolutions, previous)
        ownership = self.modules.ownership.update(fusion, previous)
        lifecycle = self.modules.lifecycle.update(visibility, decisions, ownership, previous)
        snapshot = self.modules.committer.commit(
            frame,
            resolutions,
            fusion,
            ownership,
            lifecycle,
            previous,
        )
        if previous is not None and snapshot.revision <= previous.revision:
            raise ValueError("map snapshot revision must increase monotonically")
        self.current_snapshot = snapshot
        return snapshot
