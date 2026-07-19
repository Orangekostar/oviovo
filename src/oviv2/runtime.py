from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.addressing import VoxelKey
from src.oviv2.entities import EntityRegistry, EntityRegistryConfig
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig


@dataclass(frozen=True)
class Oviv2RuntimeConfig:
    tsdf: TsdfConfig = TsdfConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    tracker: LocalTrackerConfig = LocalTrackerConfig()
    registry: EntityRegistryConfig = EntityRegistryConfig()
    semantic_support_scale: float = 1.0
    entity_support_scale: float = 1.0
    ownership_min_net_support: float = 1e-6

    def __post_init__(self) -> None:
        if self.evidence.block_resolution != self.tsdf.block_resolution:
            raise ValueError("evidence and TSDF block resolutions must match")
        for name in ("semantic_support_scale", "entity_support_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.ownership_min_net_support)
            or self.ownership_min_net_support < 0.0
        ):
            raise ValueError("ownership_min_net_support must be finite and non-negative")


@dataclass(frozen=True)
class RuntimeFrameResult:
    frame_id: int
    revision: int
    geometry_blocks_touched: int
    observation_count: int
    updated_track_count: int
    accepted_entity_ids: tuple[int, ...]


class Oviv2Runtime:
    """Voxel-first OVIV2 runtime with no dependency on legacy mutable map state."""

    def __init__(
        self,
        scene_id: str,
        config: Oviv2RuntimeConfig = Oviv2RuntimeConfig(),
    ) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not isinstance(config, Oviv2RuntimeConfig):
            raise TypeError("config must be Oviv2RuntimeConfig")
        self.scene_id = scene_id.strip()
        self.config = config
        self.geometry = SparseTsdfVolume(config.tsdf)
        self.evidence = SparseEvidenceStore(config.evidence)
        self.ownership = ReversibleOwnershipStore(config.tsdf.block_resolution)
        self.tracker = LocalTracker(config.tracker)
        self.registry = EntityRegistry(config.registry)
        self.revision = 0
        self.last_frame_id = -1
        self.last_timestamp = 0.0

    def process_frame(
        self,
        frame: Frame,
        observations: tuple[FrameObservation, ...],
    ) -> RuntimeFrameResult:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if frame.frame_id <= self.last_frame_id:
            raise ValueError("frame IDs must increase monotonically")
        if any(item.frame_id != frame.frame_id for item in observations):
            raise ValueError("all observations must belong to the current frame")

        next_revision = self.revision + 1
        blocks_touched = self.geometry.integrate(
            frame.depth,
            frame.rgb,
            frame.intrinsics.to_matrix(),
            frame.pose,
        )

        for item in observations:
            if item.kind is ObservationKind.STRUCTURE and item.semantic_id > 0:
                support = max(item.confidence * self.config.semantic_support_scale, 1e-9)
                for voxel_key in item.voxel_keys:
                    self.evidence.update_semantic(
                        voxel_key,
                        item.semantic_id,
                        support,
                        next_revision,
                    )

        track_batch = self.tracker.update(observations, frame.frame_id)
        accepted_entity_ids: list[int] = []
        ownership_keys: set[VoxelKey] = set()
        for track in track_batch.accepted:
            entity = self.registry.resolve(track, next_revision)
            accepted_entity_ids.append(entity.entity_id)
            current_observation = track.observations[-1]
            semantic_support = max(
                current_observation.confidence * self.config.semantic_support_scale,
                1e-9,
            )
            entity_support = max(
                current_observation.confidence * self.config.entity_support_scale,
                1e-9,
            )
            for voxel_key in current_observation.voxel_keys:
                if entity.semantic_id > 0:
                    self.evidence.update_semantic(
                        voxel_key,
                        entity.semantic_id,
                        semantic_support,
                        next_revision,
                    )
                self.evidence.update_entity(
                    voxel_key,
                    entity.entity_id,
                    positive_delta=entity_support,
                    negative_delta=0.0,
                    timestamp=frame.timestamp,
                    revision=next_revision,
                )
                ownership_keys.add(voxel_key)
        self.recompute_ownership(tuple(sorted(ownership_keys)), revision=next_revision)

        self.revision = next_revision
        self.last_frame_id = int(frame.frame_id)
        self.last_timestamp = float(frame.timestamp)
        return RuntimeFrameResult(
            frame_id=self.last_frame_id,
            revision=self.revision,
            geometry_blocks_touched=int(blocks_touched),
            observation_count=len(observations),
            updated_track_count=len(track_batch.updated),
            accepted_entity_ids=tuple(accepted_entity_ids),
        )

    def recompute_ownership(
        self,
        voxel_keys: tuple[VoxelKey, ...],
        *,
        revision: int,
    ) -> None:
        for voxel_key in voxel_keys:
            candidates = self.evidence.entity_candidates(voxel_key)
            ranked = sorted(
                (
                    (item.positive_support - item.negative_support, item.entity_id)
                    for item in candidates
                ),
                key=lambda item: (-item[0], item[1]),
            )
            positive = [item for item in ranked if item[0] > self.config.ownership_min_net_support]
            if positive:
                best_support, best_entity = positive[0]
                total_support = sum(item[0] for item in positive)
                self.ownership.assign(
                    voxel_key,
                    best_entity,
                    confidence=float(best_support / total_support),
                    evidence_revision=revision,
                )
                continue
            current = self.ownership.owner_of(voxel_key)
            if current is not None:
                self.ownership.release(voxel_key, current.entity_id, revision)

    def commit(self, target_dir: str | Path) -> VoxelMapSnapshot:
        return VoxelMapSnapshot.commit(
            target_dir,
            VoxelSnapshotMetadata(
                scene_id=self.scene_id,
                frame_id=max(self.last_frame_id, 0),
                timestamp=self.last_timestamp,
                revision=self.revision,
                voxel_size_m=self.config.tsdf.voxel_size_m,
                block_resolution=self.config.tsdf.block_resolution,
            ),
            self.geometry,
            self.evidence,
            self.ownership,
        )
