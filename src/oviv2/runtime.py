from __future__ import annotations

import copy
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
from src.oviv2.visibility import VisibilityConfig, VisibilityStatus, VoxelVisibilityProjector


@dataclass(frozen=True)
class Oviv2RuntimeConfig:
    tsdf: TsdfConfig = TsdfConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    tracker: LocalTrackerConfig = LocalTrackerConfig()
    registry: EntityRegistryConfig = EntityRegistryConfig()
    visibility_depth_tolerance_m: float = 0.1
    absence_negative_support: float = 1.0
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
        for name in ("visibility_depth_tolerance_m", "absence_negative_support"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RuntimeFrameResult:
    frame_id: int
    revision: int
    geometry_blocks_touched: int
    observation_count: int
    updated_track_count: int
    accepted_entity_ids: tuple[int, ...]
    matched_entity_count: int = 0
    new_entity_count: int = 0
    association_conflict_count: int = 0
    revoked_edge_count: int = 0


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
        self.visibility = VoxelVisibilityProjector(
            VisibilityConfig(
                voxel_size_m=config.tsdf.voxel_size_m,
                depth_tolerance_m=config.visibility_depth_tolerance_m,
            )
        )
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
        trial_tracker = copy.deepcopy(self.tracker)
        track_batch = trial_tracker.update(observations, frame.frame_id)
        accepted_tracks = tuple(
            sorted(track_batch.accepted, key=lambda track: track.track_id)
        )
        trial_registry = copy.deepcopy(self.registry)
        entity_ids_before = set(trial_registry.entities)
        resolved_entities = trial_registry.resolve_batch(accepted_tracks, next_revision)
        accepted_entity_ids = tuple(entity.entity_id for entity in resolved_entities)
        matched_entity_count = sum(
            entity.entity_id in entity_ids_before for entity in resolved_entities
        )
        new_entity_count = len(resolved_entities) - matched_entity_count

        blocks_touched = self.geometry.integrate(
            frame.depth,
            frame.rgb,
            frame.intrinsics.to_matrix(),
            frame.pose,
        )

        self.apply_visibility(
            frame,
            revision=next_revision,
            entity_voxels={
                entity_id: entity.voxel_keys
                for entity_id, entity in self.registry.entities.items()
            },
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

        ownership_keys: set[VoxelKey] = set()
        for track, entity in zip(accepted_tracks, resolved_entities):
            current_observation = track.observations[-1]
            entity_support = max(
                current_observation.confidence * self.config.entity_support_scale,
                1e-9,
            )
            for voxel_key in current_observation.voxel_keys:
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

        self.tracker = trial_tracker
        self.registry = trial_registry
        self.revision = next_revision
        self.last_frame_id = int(frame.frame_id)
        self.last_timestamp = float(frame.timestamp)
        return RuntimeFrameResult(
            frame_id=self.last_frame_id,
            revision=self.revision,
            geometry_blocks_touched=int(blocks_touched),
            observation_count=len(observations),
            updated_track_count=len(track_batch.updated),
            accepted_entity_ids=accepted_entity_ids,
            matched_entity_count=matched_entity_count,
            new_entity_count=new_entity_count,
            association_conflict_count=track_batch.conflict_count,
            revoked_edge_count=track_batch.revoked_edge_count,
        )

    def apply_visibility(
        self,
        frame: Frame,
        *,
        revision: int,
        entity_voxels: dict[int, frozenset[VoxelKey]],
    ) -> None:
        changed: set[VoxelKey] = set()
        for entity_id in sorted(entity_voxels):
            grouped = self.visibility.classify_many(
                tuple(entity_voxels[entity_id]),
                frame,
            )
            for voxel_key in grouped[VisibilityStatus.ABSENT]:
                candidates = {
                    item.entity_id for item in self.evidence.entity_candidates(voxel_key)
                }
                if entity_id not in candidates:
                    continue
                self.evidence.update_entity(
                    voxel_key,
                    entity_id,
                    positive_delta=0.0,
                    negative_delta=self.config.absence_negative_support,
                    timestamp=frame.timestamp,
                    revision=revision,
                )
                changed.add(voxel_key)
        self.recompute_ownership(tuple(sorted(changed)), revision=revision)

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
                schema_version=2,
            ),
            self.geometry,
            self.evidence,
            self.ownership,
            registry=self.registry,
        )
