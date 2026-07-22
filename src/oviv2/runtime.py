from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d

from src.core.data_structures import Frame
from src.oviv2.addressing import VoxelKey
from src.oviv2.dense_projection import (
    DenseProjectionResult,
    DenseSemanticConfig,
    DenseSemanticIntegrator,
)
from src.oviv2.dense_semantics import DenseSemanticFrame, DenseSemanticProvenance
from src.oviv2.entities import EntityRegistry, EntityRegistryConfig
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata
from src.oviv2.compact_checkpoint import (
    CompactOwnershipCommitReceipt,
    CompactOwnershipCheckpoint,
    CompactOwnershipMetadata,
)
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
    missing_observation_policy: Literal[
        "signed_depth", "missing_as_absence"
    ] = "signed_depth"
    dense_semantics: DenseSemanticConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.missing_observation_policy, str) or (
            self.missing_observation_policy
            not in {"signed_depth", "missing_as_absence"}
        ):
            raise ValueError(
                "missing_observation_policy must be 'signed_depth' or "
                "'missing_as_absence'"
            )
        if self.dense_semantics is not None and not isinstance(
            self.dense_semantics,
            DenseSemanticConfig,
        ):
            raise TypeError("dense_semantics must be DenseSemanticConfig or None")
        if self.dense_semantics is not None and not np.isclose(
            self.dense_semantics.voxel_size_m,
            self.tsdf.voxel_size_m,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("dense voxel_size_m must match TSDF voxel_size_m")
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
    dense_sampled_pixel_count: int = 0
    dense_valid_pixel_count: int = 0
    dense_updated_voxel_count: int = 0

    def __post_init__(self) -> None:
        try:
            dense_result = DenseProjectionResult(
                self.dense_sampled_pixel_count,
                self.dense_valid_pixel_count,
                self.dense_updated_voxel_count,
            )
        except ValueError as exc:
            message = str(exc)
            for field_name in (
                "sampled_pixel_count",
                "valid_pixel_count",
                "updated_voxel_count",
            ):
                message = message.replace(field_name, f"dense_{field_name}")
            raise ValueError(message) from exc
        object.__setattr__(
            self,
            "dense_sampled_pixel_count",
            dense_result.sampled_pixel_count,
        )
        object.__setattr__(
            self,
            "dense_valid_pixel_count",
            dense_result.valid_pixel_count,
        )
        object.__setattr__(
            self,
            "dense_updated_voxel_count",
            dense_result.updated_voxel_count,
        )


def _clone_geometry_for_frame(
    geometry: SparseTsdfVolume,
    frame: Frame,
) -> SparseTsdfVolume:
    config = geometry.config
    depth = np.asarray(frame.depth)
    intrinsic = np.asarray(frame.intrinsics.to_matrix(), dtype=np.float64)
    pose = np.asarray(frame.pose, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite 4x4 matrix")
    try:
        world_to_camera = np.linalg.inv(pose)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera_to_world must be invertible") from exc

    clean_depth = np.array(depth, dtype=np.float32, copy=True, order="C")
    valid_depth = (
        np.isfinite(clean_depth)
        & (clean_depth > 0.0)
        & (clean_depth <= config.depth_max_m)
    )
    clean_depth[~valid_depth] = 0.0
    block_coords = geometry._grid.compute_unique_block_coordinates(
        o3d.t.geometry.Image(o3d.core.Tensor(clean_depth)),
        o3d.core.Tensor(intrinsic, dtype=o3d.core.float64),
        o3d.core.Tensor(world_to_camera, dtype=o3d.core.float64),
        depth_scale=1.0,
        depth_max=config.depth_max_m,
        trunc_voxel_multiplier=config.trunc_voxel_multiplier,
    )
    source_indices = geometry._grid.hashmap().active_buf_indices()
    source_count = int(source_indices.shape[0])
    required_capacity = min(
        config.block_count,
        max(1, source_count + int(block_coords.shape[0])),
    )
    trial = SparseTsdfVolume(replace(config, block_count=required_capacity))
    if source_count:
        source_keys = geometry._grid.hashmap().key_tensor()[source_indices]
        destination_indices, activated = trial._grid.hashmap().activate(source_keys)
        if not np.all(activated.numpy()):
            raise RuntimeError("failed to clone active TSDF blocks")
        for name in geometry._ATTRIBUTE_NAMES:
            trial._grid.attribute(name)[destination_indices] = geometry._grid.attribute(
                name
            )[source_indices]
    trial._config = config
    return trial


class Oviv2Runtime:
    """Voxel-first OVIV2 runtime with no dependency on legacy mutable map state."""

    def __init__(
        self,
        scene_id: str,
        config: Oviv2RuntimeConfig = Oviv2RuntimeConfig(),
        *,
        dense_semantic_provenance: DenseSemanticProvenance | None = None,
    ) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not isinstance(config, Oviv2RuntimeConfig):
            raise TypeError("config must be Oviv2RuntimeConfig")
        if dense_semantic_provenance is not None and not isinstance(
            dense_semantic_provenance,
            DenseSemanticProvenance,
        ):
            raise TypeError(
                "dense_semantic_provenance must be DenseSemanticProvenance or None"
            )
        dense_enabled = config.dense_semantics is not None
        provenance_present = dense_semantic_provenance is not None
        if dense_enabled != provenance_present:
            raise ValueError(
                "dense semantics config and dense semantic provenance must be paired"
            )
        self.scene_id = scene_id.strip()
        self.config = config
        self.dense_semantic_provenance = dense_semantic_provenance
        self.dense_semantic_integrator = (
            DenseSemanticIntegrator(config.dense_semantics)
            if config.dense_semantics is not None
            else None
        )
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
        dense_semantics: DenseSemanticFrame | None = None,
    ) -> RuntimeFrameResult:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if (
            not isinstance(frame.frame_id, (int, np.integer))
            or isinstance(frame.frame_id, (bool, np.bool_))
            or int(frame.frame_id) < 0
        ):
            raise ValueError("frame_id must be a non-negative integer")
        current_frame_id = int(frame.frame_id)
        try:
            current_timestamp = float(frame.timestamp)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("frame timestamp must be finite") from exc
        if not np.isfinite(current_timestamp):
            raise ValueError("frame timestamp must be finite")
        if current_frame_id <= self.last_frame_id:
            raise ValueError("frame IDs must increase monotonically")
        if any(item.frame_id != current_frame_id for item in observations):
            raise ValueError("all observations must belong to the current frame")
        if self.dense_semantic_integrator is None:
            if dense_semantics is not None:
                raise ValueError("dense semantics were supplied while dense mode is disabled")
        elif dense_semantics is None:
            raise ValueError("dense semantics are required while dense mode is enabled")
        elif not isinstance(dense_semantics, DenseSemanticFrame):
            raise TypeError("dense_semantics must be a DenseSemanticFrame")

        next_revision = self.revision + 1
        trial_tracker = copy.deepcopy(self.tracker)
        track_batch = trial_tracker.update(observations, current_frame_id)
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

        trial_geometry = _clone_geometry_for_frame(self.geometry, frame)
        trial_evidence = copy.deepcopy(self.evidence)
        trial_ownership = copy.deepcopy(self.ownership)
        blocks_touched = trial_geometry.integrate(
            frame.depth,
            frame.rgb,
            frame.intrinsics.to_matrix(),
            frame.pose,
        )

        dense_result = DenseProjectionResult(0, 0, 0)
        if self.dense_semantic_integrator is not None:
            assert dense_semantics is not None
            dense_result = self.dense_semantic_integrator.integrate(
                frame,
                dense_semantics,
                trial_evidence,
                next_revision,
            )

        self._apply_visibility_to(
            frame,
            revision=next_revision,
            entity_voxels={
                entity_id: entity.voxel_keys
                for entity_id, entity in self.registry.entities.items()
            },
            evidence=trial_evidence,
            ownership=trial_ownership,
            matched_entity_ids=frozenset(accepted_entity_ids),
        )

        for item in observations:
            if item.kind is ObservationKind.STRUCTURE and item.semantic_id > 0:
                support = max(item.confidence * self.config.semantic_support_scale, 1e-9)
                for voxel_key in item.voxel_keys:
                    trial_evidence.update_semantic(
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
                trial_evidence.update_entity(
                    voxel_key,
                    entity.entity_id,
                    positive_delta=entity_support,
                    negative_delta=0.0,
                    timestamp=frame.timestamp,
                    revision=next_revision,
                )
                ownership_keys.add(voxel_key)
        self._recompute_ownership_in(
            tuple(sorted(ownership_keys)),
            revision=next_revision,
            evidence=trial_evidence,
            ownership=trial_ownership,
        )

        self.geometry = trial_geometry
        self.evidence = trial_evidence
        self.ownership = trial_ownership
        self.tracker = trial_tracker
        self.registry = trial_registry
        self.revision = next_revision
        self.last_frame_id = current_frame_id
        self.last_timestamp = current_timestamp
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
            dense_sampled_pixel_count=dense_result.sampled_pixel_count,
            dense_valid_pixel_count=dense_result.valid_pixel_count,
            dense_updated_voxel_count=dense_result.updated_voxel_count,
        )

    def apply_visibility(
        self,
        frame: Frame,
        *,
        revision: int,
        entity_voxels: dict[int, frozenset[VoxelKey]],
        matched_entity_ids: frozenset[int] = frozenset(),
    ) -> None:
        self._apply_visibility_to(
            frame,
            revision=revision,
            entity_voxels=entity_voxels,
            evidence=self.evidence,
            ownership=self.ownership,
            matched_entity_ids=matched_entity_ids,
        )

    def _apply_visibility_to(
        self,
        frame: Frame,
        *,
        revision: int,
        entity_voxels: dict[int, frozenset[VoxelKey]],
        evidence: SparseEvidenceStore,
        ownership: ReversibleOwnershipStore,
        matched_entity_ids: frozenset[int],
    ) -> None:
        changed: set[VoxelKey] = set()
        for entity_id in sorted(entity_voxels):
            grouped = self.visibility.classify_many(
                tuple(entity_voxels[entity_id]),
                frame,
            )
            penalized = list(grouped[VisibilityStatus.ABSENT])
            if (
                self.config.missing_observation_policy == "missing_as_absence"
                and entity_id not in matched_entity_ids
            ):
                penalized.extend(grouped[VisibilityStatus.PRESENT])
                penalized.extend(grouped[VisibilityStatus.OCCLUDED])
            for voxel_key in penalized:
                candidates = {
                    item.entity_id for item in evidence.entity_candidates(voxel_key)
                }
                if entity_id not in candidates:
                    continue
                evidence.update_entity(
                    voxel_key,
                    entity_id,
                    positive_delta=0.0,
                    negative_delta=self.config.absence_negative_support,
                    timestamp=frame.timestamp,
                    revision=revision,
                )
                changed.add(voxel_key)
        self._recompute_ownership_in(
            tuple(sorted(changed)),
            revision=revision,
            evidence=evidence,
            ownership=ownership,
        )

    def recompute_ownership(
        self,
        voxel_keys: tuple[VoxelKey, ...],
        *,
        revision: int,
    ) -> None:
        self._recompute_ownership_in(
            voxel_keys,
            revision=revision,
            evidence=self.evidence,
            ownership=self.ownership,
        )

    def _recompute_ownership_in(
        self,
        voxel_keys: tuple[VoxelKey, ...],
        *,
        revision: int,
        evidence: SparseEvidenceStore,
        ownership: ReversibleOwnershipStore,
    ) -> None:
        for voxel_key in voxel_keys:
            candidates = evidence.entity_candidates(voxel_key)
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
                ownership.assign(
                    voxel_key,
                    best_entity,
                    confidence=float(best_support / total_support),
                    evidence_revision=revision,
                )
                continue
            current = ownership.owner_of(voxel_key)
            if current is not None:
                ownership.release(voxel_key, current.entity_id, revision)

    def _snapshot_metadata(self) -> VoxelSnapshotMetadata:
        return VoxelSnapshotMetadata(
            scene_id=self.scene_id,
            frame_id=max(self.last_frame_id, 0),
            timestamp=self.last_timestamp,
            revision=self.revision,
            voxel_size_m=self.config.tsdf.voxel_size_m,
            block_resolution=self.config.tsdf.block_resolution,
            schema_version=(
                3 if self.dense_semantic_provenance is not None else 2
            ),
            dense_semantic_provenance=self.dense_semantic_provenance,
        )

    def commit(self, target_dir: str | Path) -> VoxelMapSnapshot:
        return VoxelMapSnapshot.commit(
            target_dir,
            self._snapshot_metadata(),
            self.geometry,
            self.evidence,
            self.ownership,
            registry=self.registry,
        )

    def commit_new(self, target_dir: str | Path) -> VoxelMapSnapshot:
        return VoxelMapSnapshot.commit_new(
            target_dir,
            self._snapshot_metadata(),
            self.geometry,
            self.evidence,
            self.ownership,
            registry=self.registry,
        )

    def commit_compact_ownership_new(
        self,
        target_dir: str | Path,
    ) -> CompactOwnershipCommitReceipt:
        metadata = self._snapshot_metadata()
        if metadata.dense_semantic_provenance is None:
            raise ValueError(
                "compact ownership checkpoints require dense semantic provenance"
            )
        return CompactOwnershipCheckpoint.commit_receipt_new(
            target_dir,
            CompactOwnershipMetadata(
                scene_id=metadata.scene_id,
                frame_id=metadata.frame_id,
                timestamp=metadata.timestamp,
                revision=metadata.revision,
                voxel_size_m=metadata.voxel_size_m,
                block_resolution=metadata.block_resolution,
                dense_semantic_provenance=metadata.dense_semantic_provenance,
            ),
            self.ownership,
        )
