from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from numbers import Integral, Real
import re

import numpy as np
from scipy.spatial import cKDTree

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.temporal_association import (
    TemporalAssociationTarget,
    associate_temporal_observations,
)
from src.oviv2.temporal_background_ledger import (
    BackgroundContribution,
    BackgroundLedgerEvidence,
    LedgerDecision,
    ReversibleBackgroundLedger,
)
import src.oviv2.temporal_background_ledger as _ledger_module
from src.oviv2.temporal_background import (
    TemporalBackgroundVolume,
    build_background_depth,
)
import src.oviv2.temporal_background as _background_module
from src.oviv2.temporal_config import (
    ExecutionProfile,
    TemporalBackgroundLedgerConfig,
    TemporalGeometryConfig,
    TemporalReadoutConfig,
    temporal_config_from_json,
    temporal_config_to_json,
)
from src.oviv2.temporal_geometry import (
    MotionDecision,
    ObjectSubmap,
    backproject_observation,
    estimate_object_motion,
    estimate_object_translation,
    integrate_object_submap,
)
from src.oviv2.temporal_epoch import GeometryEpoch, start_new_epoch
from src.oviv2.temporal_export import (
    DynamicEvidenceState,
    TemporalExportBatch,
    TemporalExportSample,
    TemporalLifecycleEvent,
    advance_dynamic_state,
    timestamp_seconds_to_ns,
)
from src.oviv2.temporal_identity import IdentityMemoryBank, IdentityMemoryRecord
from src.oviv2.temporal_proposals import (
    ProjectedIdentitySearchRegion,
    ProposalRecoveryInput,
    recover_temporal_proposals,
)
from src.oviv2.temporal_lifecycle import (
    TemporalEvidence,
    TemporalEvidenceKind,
    TemporalLifecycle,
    TemporalLifecycleState,
    advance_lifecycle,
)
from src.oviv2.temporal_observation_merge import regularize_temporal_object_extents
from src.oviv2.temporal_state import (
    TEMPORAL_MECHANISM_RECORD_KEYS,
    TemporalDiagnostics,
    TemporalEntityState,
    TemporalExportTracker,
    TemporalExportTrackerEntry,
    TemporalFrameDiagnostics,
    TemporalGeometryState,
    TemporalRuntimeState,
)
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig


_NEAREST_QUERY_CHUNK_SIZE = 1024


def _ledger_group_records(
    ledger: ReversibleBackgroundLedger | None,
    *,
    committed: bool,
) -> set[str]:
    if ledger is None:
        return set()
    records = ledger._committed if committed else ledger._provisional
    first_frames: dict[tuple[int, int, tuple[int, int, int]], int] = {}
    for entity_id, epoch_id, frame_id, block in records:
        key = (entity_id, epoch_id, block)
        first_frames[key] = min(frame_id, first_frames.get(key, frame_id))
    return {
        f"ledger:{entity_id}:{epoch_id}:{block[0]},{block[1]},{block[2]}:{first_frame}"
        for (entity_id, epoch_id, block), first_frame in first_frames.items()
    }


_SPARSE_PIXEL_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class TemporalFrameResult:
    frame_id: int
    revision: int
    active_entity_ids: tuple[int, ...]
    dormant_entity_ids: tuple[int, ...]
    new_entity_ids: tuple[int, ...]
    reactivated_entity_ids: tuple[int, ...]
    background_blocks_touched: int
    export: TemporalExportBatch | None = None
    proposal_opportunity_count: int = 0
    proposal_trigger_count: int = 0
    reid_opportunity_count: int = 0
    reid_trigger_count: int = 0
    expired_identity_ids: tuple[int, ...] = ()
    diagnostics: TemporalFrameDiagnostics | None = None

    def __post_init__(self) -> None:
        for name in ("frame_id", "revision", "background_blocks_touched"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        for name in (
            "active_entity_ids",
            "dormant_entity_ids",
            "new_entity_ids",
            "reactivated_entity_ids",
            "expired_identity_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or int(value) < 0
                for value in values
            ):
                raise ValueError(f"{name} must contain nonnegative integer IDs")
            normalized = tuple(int(value) for value in values)
            if normalized != tuple(sorted(set(normalized))):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, normalized)
        if set(self.active_entity_ids) & set(self.dormant_entity_ids):
            raise ValueError("active and dormant entity IDs must be disjoint")
        if not set(self.new_entity_ids).issubset(self.active_entity_ids) or not set(
            self.reactivated_entity_ids
        ).issubset(self.active_entity_ids):
            raise ValueError("new and reactivated entity IDs must be active subsets")
        if set(self.new_entity_ids) & set(self.reactivated_entity_ids):
            raise ValueError("new and reactivated entity IDs must be disjoint")
        if self.export is None:
            object.__setattr__(self, "export", TemporalExportBatch(self.frame_id, 0, (), ()))
        elif type(self.export) is not TemporalExportBatch:
            raise TypeError("export must be a TemporalExportBatch")
        assert self.export is not None
        if self.export.frame_index != self.frame_id:
            raise RuntimeError("dual runtime frame mismatch between result and export")
        for name in (
            "proposal_opportunity_count", "proposal_trigger_count",
            "reid_opportunity_count", "reid_trigger_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.proposal_trigger_count > self.proposal_opportunity_count:
            raise ValueError("proposal triggers cannot exceed opportunities")
        if self.reid_trigger_count > self.reid_opportunity_count:
            raise ValueError("re-ID triggers cannot exceed opportunities")
        if self.diagnostics is not None:
            if type(self.diagnostics) is not TemporalFrameDiagnostics:
                raise TypeError("diagnostics must be TemporalFrameDiagnostics or None")
            if self.diagnostics.frame_id != self.frame_id:
                raise ValueError("diagnostics frame must match result")


def _finite_float64(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    wide = np.longdouble(value)
    if not np.isfinite(wide):
        raise ValueError(f"{name} must be finite")
    limit = np.longdouble(np.finfo(np.float64).max)
    if wide < -limit or wide > limit:
        raise ValueError(f"{name} must lie within the float64 range")
    result = float(wide)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_float64_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric")
    wide = np.asarray(raw, dtype=np.longdouble)
    limit = np.longdouble(np.finfo(np.float64).max)
    if wide.shape != shape or not np.isfinite(wide).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    if np.any(wide < -limit) or np.any(wide > limit):
        raise ValueError(f"{name} must lie within the float64 range")
    result = np.asarray(wide, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float64 values")
    return result


def _validate_frame(frame: object) -> Frame:
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if isinstance(frame.frame_id, (bool, np.bool_)) or not isinstance(frame.frame_id, Integral):
        raise TypeError("frame.frame_id must be an integer")
    if int(frame.frame_id) < 0:
        raise ValueError("frame.frame_id must be nonnegative")
    if _finite_float64(frame.timestamp, "frame.timestamp") < 0.0:
        raise ValueError("frame.timestamp must be nonnegative for native export")
    depth = np.asarray(frame.depth)
    rgb = np.asarray(frame.rgb)
    pose = np.asarray(frame.pose)
    if depth.dtype.kind != "f" or depth.ndim != 2:
        raise TypeError("frame.depth must be a two-dimensional floating array")
    if rgb.shape != (*depth.shape, 3):
        raise ValueError("frame.rgb shape must match depth")
    if rgb.dtype != np.dtype(np.uint8):
        if rgb.dtype.kind != "f":
            raise TypeError("frame.rgb must be uint8 or floating point")
        if not np.isfinite(rgb).all() or (rgb.size and (float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0)):
            raise ValueError("floating frame.rgb must be finite and lie in [0, 1]")
    pose64 = _finite_float64_array(pose, (4, 4), "frame.pose")
    if not np.allclose(pose64[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError("frame.pose must be homogeneous")
    rotation = pose64[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("frame.pose must be rigid")
    if not isinstance(frame.intrinsics, CameraIntrinsics):
        raise TypeError("frame.intrinsics must be CameraIntrinsics")
    intrinsics = frame.intrinsics
    for name in ("width", "height"):
        value = getattr(intrinsics, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"frame.intrinsics.{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"frame.intrinsics.{name} must be positive")
    if (intrinsics.height, intrinsics.width) != depth.shape:
        raise ValueError("frame intrinsics dimensions must match depth")
    values = tuple(
        _finite_float64(value, f"frame.intrinsics.{name}")
        for name, value in zip(("fx", "fy", "cx", "cy"), (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy))
    )
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("frame intrinsics must be finite with positive focal lengths")
    return frame


def _validate_inputs(
    frame: object,
    observations: object,
    dense_semantics: object,
    state: TemporalRuntimeState,
) -> tuple[Frame, tuple[FrameObservation, ...], DenseSemanticFrame | None]:
    frame = _validate_frame(frame)
    if frame.frame_id <= state.last_frame_id:
        raise ValueError("frame.frame_id must increase strictly")
    if state.last_frame_id != -1 and float(frame.timestamp) <= state.last_timestamp:
        raise ValueError("frame.timestamp must increase strictly")
    if frame.source_frame_id is not None:
        if isinstance(frame.source_frame_id, (bool, np.bool_)) or not isinstance(frame.source_frame_id, Integral):
            raise TypeError("frame.source_frame_id must be an integer or None")
        if frame.source_frame_id < 0 or frame.source_frame_id > np.iinfo(np.int64).max:
            raise ValueError("frame.source_frame_id must be a nonnegative int64")
    if type(observations) is not tuple:
        raise TypeError("observations must be an exact tuple")
    if any(not isinstance(item, FrameObservation) for item in observations):
        raise TypeError("observations must contain FrameObservation values")
    ids: list[int] = []
    for observation in observations:
        if isinstance(observation.observation_id, (bool, np.bool_)) or not isinstance(
            observation.observation_id, Integral
        ):
            raise TypeError("observation_id must be an integer")
        if observation.observation_id < 0:
            raise ValueError("observation_id must be nonnegative")
        if isinstance(observation.frame_id, (bool, np.bool_)) or not isinstance(
            observation.frame_id, Integral
        ):
            raise TypeError("observation frame_id must be an integer")
        observation_timestamp = _finite_float64(
            observation.timestamp, "observation timestamp"
        )
        ids.append(observation.observation_id)
        if observation.frame_id != frame.frame_id:
            raise ValueError("observation frame_id must match frame")
        if observation_timestamp != _finite_float64(frame.timestamp, "frame.timestamp"):
            raise ValueError("observation timestamp must match frame")
        if observation.mask.shape != frame.depth.shape:
            raise ValueError("observation mask shape must match frame depth")
        geometry = _finite_float64_array(
            (
                observation.centroid_xyz,
                observation.bounds_min_xyz,
                observation.bounds_max_xyz,
            ),
            (3, 3),
            "observation geometry",
        )
        if observation.kind is ObservationKind.OBJECT and np.any(
            geometry[2] - geometry[1] <= 0.0
        ):
            raise ValueError("object observation extent must be positive")
    if len(ids) != len(set(ids)):
        raise ValueError("observation IDs must be unique")
    if dense_semantics is not None:
        if not isinstance(dense_semantics, DenseSemanticFrame):
            raise TypeError("dense_semantics must be DenseSemanticFrame or None")
        expected_source = frame.frame_id if frame.source_frame_id is None else frame.source_frame_id
        if dense_semantics.cache_frame_id != frame.frame_id:
            raise ValueError("dense_semantics cache_frame_id must match frame.frame_id")
        if dense_semantics.source_frame_id != expected_source:
            raise ValueError("dense_semantics source_frame_id must match frame")
        if dense_semantics.image_shape != frame.depth.shape:
            raise ValueError("dense_semantics image_shape must match frame")
    return frame, observations, dense_semantics


def _dense_semantic_support(dense: DenseSemanticFrame) -> np.ndarray:
    sampled = (dense.class_ids[..., 0] > 0) & (dense.probabilities[..., 0] > 0.0)
    expanded = np.repeat(
        np.repeat(sampled, dense.sample_stride, axis=0),
        dense.sample_stride,
        axis=1,
    )
    return np.asarray(expanded[: dense.image_shape[0], : dense.image_shape[1]], dtype=bool)


def _proposal_semantic_provenance_hash(
    frame: Frame, dense: DenseSemanticFrame
) -> str:
    digest = hashlib.sha256()
    metadata = (
        frame.frame_id,
        frame.source_frame_id,
        float(frame.timestamp),
        frame.intrinsics.fx,
        frame.intrinsics.fy,
        frame.intrinsics.cx,
        frame.intrinsics.cy,
        frame.intrinsics.width,
        frame.intrinsics.height,
        dense.cache_frame_id,
        dense.source_frame_id,
        dense.image_shape,
        dense.sample_stride,
        dense.class_count,
    )
    digest.update(repr(metadata).encode("ascii"))
    for value in (
        frame.rgb,
        frame.depth,
        frame.pose,
        dense.class_ids,
        dense.probabilities,
        dense.entropy,
        dense.margin,
    ):
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _current_world_xyz(frame: Frame, depth_m: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(depth_m.shape, dtype=np.float64)
    camera = np.stack(
        (
            (columns - frame.intrinsics.cx) * depth_m / frame.intrinsics.fx,
            (rows - frame.intrinsics.cy) * depth_m / frame.intrinsics.fy,
            depth_m,
        ),
        axis=-1,
    )
    return camera @ np.asarray(frame.pose[:3, :3], dtype=np.float64).T + np.asarray(
        frame.pose[:3, 3], dtype=np.float64
    )


def _proposal_appearance_provenance_hash(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    model_id: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        repr((frame.frame_id, frame.source_frame_id, float(frame.timestamp), model_id)).encode(
            "ascii"
        )
    )
    for observation in observations:
        if (
            observation.kind not in (ObservationKind.OBJECT, ObservationKind.UNKNOWN)
            or observation.image_feature is None
            or observation.feature_model_id != model_id
        ):
            continue
        digest.update(repr(observation.observation_id).encode("ascii"))
        for value in (observation.mask, observation.image_feature):
            array = np.asarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class _UnprojectableIdentitySearchRegion(ValueError):
    pass


def _projected_identity_search_region(
    frame: Frame,
    state: TemporalRuntimeState,
    identity_id: int,
    config: TemporalReadoutConfig,
    *,
    record: IdentityMemoryRecord | None = None,
) -> ProjectedIdentitySearchRegion:
    entity = next(
        (
            item for item in state.entities
            if item.lifecycle.entity_id == identity_id
        ),
        None,
    )
    if record is None:
        record = state.identities.get(identity_id)
    elif record.identity_id != identity_id:
        raise ValueError("search region identity record does not match identity_id")
    if entity is None or record is None:
        raise ValueError("search region identity lacks retained authoritative geometry")
    local_points = entity.submap.local_points_xyz[
        : config.geometry.maximum_visibility_points_per_entity
    ]
    points = (
        local_points @ entity.object_to_world[:3, :3].T
        + entity.object_to_world[:3, 3]
    )
    mask = np.zeros(frame.depth.shape, dtype=bool)
    expected_depth = np.zeros(frame.depth.shape, dtype=np.float32)
    if points.shape[0]:
        world_to_camera = np.linalg.inv(np.asarray(frame.pose, dtype=np.float64))
        camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > 0.0)
        camera = camera[valid]
        z = z[valid]
        if z.size:
            columns = np.rint(
                frame.intrinsics.fx * camera[:, 0] / z + frame.intrinsics.cx
            ).astype(np.int64)
            rows = np.rint(
                frame.intrinsics.fy * camera[:, 1] / z + frame.intrinsics.cy
            ).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < frame.depth.shape[0])
                & (columns >= 0)
                & (columns < frame.depth.shape[1])
            )
            for row, column, depth in zip(rows[inside], columns[inside], z[inside]):
                previous = expected_depth[row, column]
                if previous == 0.0 or depth < previous:
                    expected_depth[row, column] = depth
                    mask[row, column] = True
    if not mask.any():
        raise _UnprojectableIdentitySearchRegion(
            "search region identity has no projectable retained geometry"
        )
    epoch = state.geometry.current(identity_id)
    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                identity_id,
                record.last_frame_id,
                frame.frame_id,
                frame.source_frame_id,
                epoch.epoch_id,
            )
        ).encode("ascii")
    )
    for value in (frame.pose, mask, expected_depth):
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return ProjectedIdentitySearchRegion(
        identity_id,
        record.last_frame_id,
        mask,
        expected_depth,
        digest.hexdigest(),
    )


def _canonical_projected_identity_search_regions(
    frame: Frame,
    state: TemporalRuntimeState,
    config: TemporalReadoutConfig,
) -> tuple[ProjectedIdentitySearchRegion, ...]:
    records_by_id = {
        record.identity_id: record for record in state.identities.records
    }
    candidates: list[IdentityMemoryRecord] = []
    for entity in state.entities:
        identity_id = entity.lifecycle.entity_id
        record = records_by_id.get(identity_id)
        if record is None:
            raise ValueError("retained entity lacks identity provenance")
        state.geometry.current(identity_id)
        candidates.append(record)
    candidates.sort(key=lambda record: (-record.last_frame_id, record.identity_id))

    assert config.proposal is not None
    capacity = 4 * config.proposal.maximum_recovered_proposals
    search_regions: list[ProjectedIdentitySearchRegion] = []
    for record in candidates:
        try:
            region = _projected_identity_search_region(
                frame,
                state,
                record.identity_id,
                config,
                record=record,
            )
        except _UnprojectableIdentitySearchRegion:
            continue
        search_regions.append(region)
        if len(search_regions) == capacity:
            break
    return tuple(search_regions)


def build_proposal_recovery_evidence(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame,
    config: TemporalReadoutConfig,
    state: TemporalRuntimeState,
) -> ProposalRecoveryInput:
    if not isinstance(config, TemporalReadoutConfig):
        raise TypeError("config must be a TemporalReadoutConfig")
    frame, observations, validated_dense = _validate_inputs(
        frame, observations, dense_semantics, state
    )
    if validated_dense is None:
        raise TypeError("dense_semantics must be a DenseSemanticFrame")
    dense_semantics = validated_dense

    depth_m = np.where(
        np.isfinite(frame.depth)
        & (frame.depth > 0.0)
        & (frame.depth <= config.geometry.depth_max_m),
        frame.depth,
        0.0,
    )
    segmentation_occupied = np.zeros(frame.depth.shape, dtype=bool)
    for observation in observations:
        if observation.kind in (ObservationKind.OBJECT, ObservationKind.UNKNOWN):
            segmentation_occupied |= observation.mask

    return ProposalRecoveryInput(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        depth_m=depth_m,
        current_xyz=_current_world_xyz(frame, depth_m),
        segmentation_occupied=segmentation_occupied,
        semantic_support=_dense_semantic_support(dense_semantics),
        semantic_source_frame_id=frame.frame_id,
        semantic_provenance_hash=_proposal_semantic_provenance_hash(
            frame, dense_semantics
        ),
        appearance_support=None,
        appearance_source_frame_id=None,
        appearance_model_id=None,
        appearance_provenance_hash=None,
        search_regions=_canonical_projected_identity_search_regions(
            frame, state, config
        ),
    )


def _validate_proposal_evidence(
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense: DenseSemanticFrame | None,
    evidence: ProposalRecoveryInput | None,
    config: TemporalReadoutConfig,
    state: TemporalRuntimeState,
) -> None:
    if evidence is None:
        return
    if not isinstance(evidence, ProposalRecoveryInput):
        raise TypeError("proposal_evidence must be ProposalRecoveryInput or None")
    if dense is None:
        raise ValueError("proposal evidence requires validated current dense semantics")
    if evidence.frame_id != frame.frame_id or evidence.timestamp != frame.timestamp:
        raise ValueError("proposal evidence must match the current frame")
    expected_depth = np.where(
        np.isfinite(frame.depth)
        & (frame.depth > 0.0)
        & (frame.depth <= config.geometry.depth_max_m),
        frame.depth,
        0.0,
    )
    if evidence.depth_m.dtype != expected_depth.dtype:
        raise ValueError("proposal depth evidence dtype is not canonical")
    if not np.array_equal(evidence.depth_m, expected_depth):
        raise ValueError("proposal depth evidence is not bound to the current frame")
    expected_xyz = _current_world_xyz(frame, expected_depth)
    if evidence.current_xyz.dtype != expected_xyz.dtype:
        raise ValueError("proposal XYZ evidence dtype is not canonical")
    if not np.array_equal(evidence.current_xyz, expected_xyz):
        raise ValueError("proposal XYZ evidence is not bound to current geometry")
    if not np.array_equal(evidence.semantic_support, _dense_semantic_support(dense)):
        raise ValueError("proposal semantic support is not bound to dense semantics")
    if evidence.semantic_source_frame_id != frame.frame_id:
        raise ValueError("proposal semantic source must be the current frame")
    if evidence.semantic_provenance_hash != _proposal_semantic_provenance_hash(frame, dense):
        raise ValueError("proposal semantic provenance does not bind current inputs")
    occupied = np.zeros(frame.depth.shape, dtype=bool)
    for observation in observations:
        if observation.kind in (ObservationKind.OBJECT, ObservationKind.UNKNOWN):
            occupied |= observation.mask
    if np.any(occupied & ~evidence.segmentation_occupied):
        raise ValueError("proposal segmentation does not cover current observations")
    if evidence.appearance_support is not None:
        assert evidence.appearance_model_id is not None
        expected_appearance = np.zeros(frame.depth.shape, dtype=bool)
        for observation in observations:
            if (
                observation.kind in (ObservationKind.OBJECT, ObservationKind.UNKNOWN)
                and observation.image_feature is not None
                and observation.feature_model_id == evidence.appearance_model_id
            ):
                expected_appearance |= observation.mask
        if not np.array_equal(evidence.appearance_support, expected_appearance):
            raise ValueError("proposal appearance support is not bound to current features")
        expected_hash = _proposal_appearance_provenance_hash(
            frame, observations, evidence.appearance_model_id
        )
        if evidence.appearance_provenance_hash != expected_hash:
            raise ValueError("proposal appearance provenance is not bound to current features")
    assert config.proposal is not None
    capacity = 4 * config.proposal.maximum_recovered_proposals
    if len(evidence.search_regions) > capacity:
        raise ValueError("proposal search region count exceeds configured capacity")
    expected_regions = _canonical_projected_identity_search_regions(
        frame, state, config
    )
    if len(evidence.search_regions) != len(expected_regions):
        raise ValueError("proposal search regions are not canonical")
    for region, expected_region in zip(
        evidence.search_regions, expected_regions, strict=True
    ):
        observed_key = (
            region.identity_id,
            region.source_frame_id,
            region.projection_provenance_hash,
        )
        expected_key = (
            expected_region.identity_id,
            expected_region.source_frame_id,
            expected_region.projection_provenance_hash,
        )
        if observed_key != expected_key:
            raise ValueError("proposal search region order is not canonical")
        if region.expected_depth_m.dtype != expected_region.expected_depth_m.dtype:
            raise ValueError("proposal search region depth dtype is not canonical")
        if region != expected_region:
            raise ValueError("proposal search region is not bound to retained identity geometry")


def _centroid(entity: TemporalEntityState) -> tuple[float, float, float]:
    transform = entity.object_to_world
    local_points = entity.submap.local_points_xyz
    if local_points.shape[0]:
        world_points = (
            transform[:3, :3] @ local_points.T
        ).T + transform[:3, 3]
        world_points = np.ascontiguousarray(world_points, dtype=np.float64)
        if not np.all(np.isfinite(world_points)):
            raise ValueError("world point transformation produced non-finite values")
        mean = world_points.mean(axis=0, dtype=np.float64)
        return tuple(float(value) for value in mean)
    return tuple(float(value) for value in transform[:3, 3])


def _association_target(entity: TemporalEntityState) -> TemporalAssociationTarget:
    centroid = _centroid(entity)
    return TemporalAssociationTarget(
        entity_id=entity.lifecycle.entity_id,
        lifecycle=entity.lifecycle.lifecycle,
        centroid_xyz=centroid,
        extent_xyz=entity.extent_xyz,
        image_prototype=entity.image_prototype,
        semantic_probabilities=entity.semantic_probabilities,
        predicted_centroid_xyz=centroid,
        feature_model_id=entity.feature_model_id,
    )


def _identity_association_target(
    record: IdentityMemoryRecord,
    lifecycle: TemporalLifecycleState,
    timestamp: float,
) -> TemporalAssociationTarget:
    dt = max(0.0, float(timestamp) - record.last_timestamp)
    predicted = tuple(
        position + velocity * dt
        for position, velocity in zip(
            record.last_centroid_xyz, record.motion_velocity_xyz
        )
    )
    return TemporalAssociationTarget(
        entity_id=record.identity_id,
        lifecycle=lifecycle.lifecycle,
        centroid_xyz=record.last_centroid_xyz,
        extent_xyz=record.extent_xyz,
        image_prototype=record.appearance_prototype,
        semantic_probabilities=record.semantic_probabilities,
        predicted_centroid_xyz=predicted,
        feature_model_id=record.feature_model_id,
    )


def _semantic_update(
    old: tuple[tuple[int, float], ...], observation: FrameObservation
) -> tuple[tuple[int, float], ...]:
    values = {class_id: probability for class_id, probability in old}
    if observation.semantic_id > 0 and observation.confidence > 0.0:
        retention = 1.0 - float(observation.confidence)
        values = {class_id: probability * retention for class_id, probability in values.items()}
        values[observation.semantic_id] = values.get(observation.semantic_id, 0.0) + float(
            observation.confidence
        )
    values = {class_id: value for class_id, value in values.items() if value > 0.0}
    total = math.fsum(values.values())
    if total == 0.0:
        return ()
    normalized = [(class_id, value / total) for class_id, value in sorted(values.items())]
    correction = 1.0 - math.fsum(value for _, value in normalized)
    class_id, value = normalized[-1]
    normalized[-1] = (class_id, value + correction)
    return tuple(normalized)


def _prototype_update(
    old: np.ndarray | None, old_model: str | None, observation: FrameObservation
) -> tuple[np.ndarray | None, str | None]:
    if observation.image_feature is None:
        return old, old_model
    incoming = np.asarray(observation.image_feature, dtype=np.float64)
    if old is None or old_model != observation.feature_model_id or old.shape != incoming.shape:
        value = incoming
    else:
        confidence = float(observation.confidence)
        value = (1.0 - confidence) * old + confidence * incoming
        if float(np.linalg.norm(value)) == 0.0:
            value = incoming
    return value / np.linalg.norm(value), observation.feature_model_id


def _initial_lifecycle(
    entity_id: int, frame: Frame, confidence: float, config: TemporalReadoutConfig
) -> TemporalLifecycleState:
    lifecycle = config.lifecycle
    delta = lifecycle.present_log_likelihood * confidence
    if not math.isfinite(delta):
        raise ValueError("initial PRESENT lifecycle delta must be finite")
    if lifecycle.initial_log_odds >= lifecycle.log_odds_limit - delta:
        log_odds = lifecycle.log_odds_limit
    else:
        log_odds = lifecycle.initial_log_odds + delta
    if not math.isfinite(log_odds):
        raise ValueError("initial PRESENT lifecycle log odds must be finite")
    if log_odds >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-log_odds))
    else:
        exp_value = math.exp(log_odds)
        probability = exp_value / (1.0 + exp_value)
    return TemporalLifecycleState(
        entity_id=entity_id,
        lifecycle=TemporalLifecycle.ACTIVE if probability >= lifecycle.active_on_probability else TemporalLifecycle.UNCERTAIN,
        existence_log_odds=log_odds,
        last_frame_id=frame.frame_id,
        last_timestamp=float(frame.timestamp),
        absent_streak=0,
        absence_view_bins=(),
    )


def _observed_extent(observation: FrameObservation) -> tuple[float, float, float]:
    value = np.asarray(observation.bounds_max_xyz) - np.asarray(observation.bounds_min_xyz)
    if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError("observation extent must be finite and positive")
    return tuple(float(item) for item in value)


def _view_bin(
    frame: Frame, centroid: tuple[float, float, float], config: TemporalReadoutConfig
) -> int:
    direction = np.asarray(centroid, dtype=np.float64) - np.asarray(frame.pose[:3, 3], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        azimuth = elevation = 0.0
    else:
        direction /= norm
        azimuth = math.atan2(float(direction[1]), float(direction[0])) % (2.0 * math.pi)
        elevation = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
    lifecycle = config.lifecycle
    azimuth_index = min(
        lifecycle.view_bin_azimuth_count - 1,
        max(0, math.floor(azimuth / (2.0 * math.pi) * lifecycle.view_bin_azimuth_count)),
    )
    elevation_index = min(
        lifecycle.view_bin_elevation_count - 1,
        max(
            0,
            math.floor(
                (elevation + math.pi / 2.0)
                / math.pi
                * lifecycle.view_bin_elevation_count
            ),
        ),
    )
    return elevation_index * lifecycle.view_bin_azimuth_count + azimuth_index


def _absence_evidence(
    entity: TemporalEntityState, frame: Frame, config: TemporalReadoutConfig
) -> TemporalEvidence:
    evidence, _ = _absence_evidence_with_release(entity, frame, config)
    return evidence


def _absence_evidence_with_release(
    entity: TemporalEntityState, frame: Frame, config: TemporalReadoutConfig
) -> tuple[TemporalEvidence, np.ndarray]:
    maximum = config.geometry.maximum_visibility_points_per_entity
    points = entity.submap.world_points(entity.object_to_world)[:maximum]
    kind = TemporalEvidenceKind.OUT_OF_VIEW
    strength = 0.0
    released = np.zeros(frame.depth.shape, dtype=bool)
    if points.shape[0]:
        world_to_camera = np.linalg.inv(np.asarray(frame.pose, dtype=np.float64))
        camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = camera[:, 2]
        front = np.isfinite(camera).all(axis=1) & (z > 0.0)
        if front.any():
            camera = camera[front]
            z = camera[:, 2]
            intrinsics = frame.intrinsics
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                u = intrinsics.fx * camera[:, 0] / z + intrinsics.cx
                v = intrinsics.fy * camera[:, 1] / z + intrinsics.cy
            height, width = frame.depth.shape
            inside = (
                np.isfinite(u)
                & np.isfinite(v)
                & (u >= -0.5)
                & (u < width - 0.5)
                & (v >= -0.5)
                & (v < height - 0.5)
            )
            if inside.any():
                x = np.floor(u[inside] + 0.5).astype(np.int64)
                y = np.floor(v[inside] + 0.5).astype(np.int64)
                predicted = z[inside]
                nearest: dict[tuple[int, int], float] = {}
                for row, column, depth in zip(y, x, predicted):
                    pixel = (int(row), int(column))
                    nearest[pixel] = min(nearest.get(pixel, math.inf), float(depth))
                valid_count = absent_count = occluded_count = unknown_count = 0
                tolerance = config.lifecycle.visibility_depth_tolerance_m
                for (row, column), predicted_depth in sorted(nearest.items()):
                    observed_depth = float(frame.depth[row, column])
                    if not math.isfinite(observed_depth) or observed_depth <= 0.0 or observed_depth > config.geometry.depth_max_m:
                        unknown_count += 1
                        continue
                    valid_count += 1
                    difference = observed_depth - predicted_depth
                    if difference > tolerance:
                        absent_count += 1
                        released[row, column] = True
                    elif difference < -tolerance:
                        occluded_count += 1
                fraction = absent_count / valid_count if valid_count else 0.0
                if (
                    absent_count >= config.lifecycle.minimum_visible_pixel_count
                    and fraction >= config.lifecycle.minimum_visible_fraction
                ):
                    kind = TemporalEvidenceKind.VISIBLE_ABSENT
                    strength = fraction
                elif occluded_count:
                    kind = TemporalEvidenceKind.OCCLUDED
                elif unknown_count:
                    kind = TemporalEvidenceKind.DEPTH_UNKNOWN
                else:
                    kind = TemporalEvidenceKind.OCCLUDED
    if kind is not TemporalEvidenceKind.VISIBLE_ABSENT:
        released.fill(False)
    evidence = TemporalEvidence(
        kind=kind,
        strength=strength,
        frame_id=frame.frame_id,
        timestamp=float(frame.timestamp),
        view_bin=_view_bin(frame, _centroid(entity), config)
        if kind is TemporalEvidenceKind.VISIBLE_ABSENT
        else None,
    )
    return evidence, released


def _sparse_background_block_keys(
    frame: Frame, depth_m: np.ndarray, config: TemporalGeometryConfig
) -> tuple[tuple[int, int, int], ...]:
    flat_depth = np.asarray(depth_m).reshape(-1)
    width = int(depth_m.shape[1])
    rotation = np.asarray(frame.pose[:3, :3], dtype=np.float64)
    origin = np.asarray(frame.pose[:3, 3], dtype=np.float64)
    truncation = 4.0 * config.voxel_size_m
    offsets = np.linspace(-truncation, truncation, 9, dtype=np.float64)
    block_size = 8.0 * config.voxel_size_m
    touched: set[tuple[int, int, int]] = set()
    for start in range(0, flat_depth.size, _SPARSE_PIXEL_CHUNK_SIZE):
        chunk = flat_depth[start : start + _SPARSE_PIXEL_CHUNK_SIZE]
        indices = np.flatnonzero(chunk > 0.0) + start
        if indices.size == 0:
            continue
        rows, columns = np.divmod(indices, width)
        depth = flat_depth[indices].astype(np.float64)
        camera = np.column_stack(
            (
                (columns - frame.intrinsics.cx) * depth / frame.intrinsics.fx,
                (rows - frame.intrinsics.cy) * depth / frame.intrinsics.fy,
                depth,
            )
        )
        world = camera @ rotation.T + origin
        rays = world - origin
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        samples = world[:, None, :] + rays[:, None, :] * offsets[None, :, None]
        keys = np.floor(samples.reshape(-1, 3) / block_size).astype(np.int64)
        touched.update(tuple(int(value) for value in row) for row in keys)
        if len(touched) > config.background_block_count:
            raise ValueError(
                "sparse background keys exceed background_block_count capacity"
            )
    return tuple(sorted(touched))


def _is_no_block_candidate_error(error: RuntimeError) -> bool:
    message = str(error)
    if message == "No block is touched in TSDF volume":
        return True
    reason = (
        "No block is touched in TSDF volume, abort integration. Please check "
        "specified parameters, especially depth_scale and voxel_size"
    )
    functions = (
        "(void open3d::t::geometry::kernel::voxel_grid::DepthTouchCPU())",
        (
            "(void open3d::t::geometry::kernel::voxel_grid::DepthTouchCPU("
            "std::shared_ptr<open3d::core::HashMap>&, const open3d::core::Tensor&, "
            "const open3d::core::Tensor&, const open3d::core::Tensor&, "
            "open3d::core::Tensor&, open3d::t::geometry::kernel::voxel_grid::"
            "index_t, float, float, float, float, open3d::t::geometry::kernel::"
            "voxel_grid::index_t))"
        ),
        (
            "(void open3d::t::geometry::kernel::voxel_grid::DepthTouchCPU("
            "std::shared_ptr<open3d::core::HashMap>&, const open3d::core::Tensor&, "
            "const open3d::core::Tensor&, const open3d::core::Tensor&, "
            "open3d::core::Tensor&, index_t, float, float, float, float, index_t))"
        ),
    )
    for function in functions:
        body = (
            r"\[Open3D Error\] "
            + re.escape(function)
            + r" (?:[^\r\n]*[\\/])?VoxelBlockGridCPU\.cpp:[1-9][0-9]*: "
            + re.escape(reason)
        )
        if re.fullmatch(body, message) is not None or re.fullmatch(
            r"\x1b\[1;31m" + body + r"(?:\r?\n)?\x1b\[0;m", message
        ) is not None:
            return True
    return False


class _SparseBackgroundVolume(TemporalBackgroundVolume):
    @classmethod
    def preallocated(
        cls, config: TemporalGeometryConfig, capacity: int
    ) -> _SparseBackgroundVolume:
        volume = cls.__new__(cls)
        volume._config = _background_module._validate_config(config)
        volume._volume = _background_module._new_sparse_volume(
            volume._config, max(1, capacity)
        )
        volume._last_blocks_touched = 0
        volume._ledger_volume_digest = None
        volume._ledger_active_block_keys = None
        return volume

    def _integrate_owned(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intrinsic: np.ndarray,
        pose: np.ndarray,
    ) -> None:
        self._ledger_volume_digest = None
        self._ledger_active_block_keys = None
        super()._integrate_owned(depth, rgb, intrinsic, pose)

    def _integrate_owned_blocks(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intrinsic: np.ndarray,
        pose: np.ndarray,
        block_keys: tuple[tuple[int, int, int], ...],
    ) -> None:
        self._ledger_volume_digest = None
        self._ledger_active_block_keys = None
        super()._integrate_owned_blocks(
            depth, rgb, intrinsic, pose, block_keys
        )

    def active_block_key_set(self) -> frozenset[tuple[int, int, int]]:
        cached = getattr(self, "_ledger_active_block_keys", None)
        if cached is None:
            cached = frozenset(
                tuple(int(value) for value in row)
                for row in _background_module._active_block_keys(self._volume)
            )
            self._ledger_active_block_keys = cached
        return cached

    def candidate_block_keys(
        self, frame: Frame, masked_depth: np.ndarray
    ) -> tuple[tuple[int, int, int], ...]:
        try:
            return super().candidate_block_keys(frame, masked_depth)
        except RuntimeError as error:
            if not _is_no_block_candidate_error(error):
                raise
            return _sparse_background_block_keys(
                frame, masked_depth, self.config
            )

    def integrate_blocks_owned(
        self,
        frame: Frame,
        masked_depth: np.ndarray,
        block_keys: tuple[tuple[int, int, int], ...],
    ) -> None:
        self._integrate_blocks_owned(
            frame, masked_depth, block_keys, validate_touched=True
        )

    def _integrate_prevalidated_blocks_owned(
        self,
        frame: Frame,
        masked_depth: np.ndarray,
        block_keys: tuple[tuple[int, int, int], ...],
    ) -> None:
        self._integrate_blocks_owned(
            frame, masked_depth, block_keys, validate_touched=False
        )

    def _integrate_blocks_owned(
        self,
        frame: Frame,
        masked_depth: np.ndarray,
        block_keys: tuple[tuple[int, int, int], ...],
        *,
        validate_touched: bool,
    ) -> None:
        _, frame_depth, rgb, pose, intrinsic = _background_module._validate_frame(
            frame
        )
        depth = _background_module._validate_masked_depth(
            masked_depth, frame_depth, self.config
        )
        if not isinstance(block_keys, tuple) or not block_keys:
            raise TypeError("block_keys must be a non-empty tuple")
        if any(
            not isinstance(key, tuple)
            or len(key) != 3
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                for value in key
            )
            for key in block_keys
        ):
            raise TypeError("block_keys must contain canonical three-integer tuples")
        normalized = tuple(tuple(int(value) for value in key) for key in block_keys)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("block_keys must be sorted and unique")
        if validate_touched:
            candidate = self.candidate_block_keys(frame, depth)
            if not set(normalized).issubset(candidate):
                raise ValueError("block_keys must be touched by the observation")
        existing = self.active_block_key_set()
        if len(existing | set(normalized)) > self.config.background_block_count:
            raise ValueError(
                "TSDF block capacity would exceed background_block_count"
            )
        self._integrate_owned_blocks(depth, rgb, intrinsic, pose, normalized)


def _candidate_keys_with_sparse_fallback(
    volume: TemporalBackgroundVolume,
    frame: Frame,
    depth_m: np.ndarray,
    config: TemporalGeometryConfig,
) -> tuple[tuple[int, int, int], ...]:
    try:
        return volume.candidate_block_keys(frame, depth_m)
    except RuntimeError as error:
        if not _is_no_block_candidate_error(error):
            raise
        return _sparse_background_block_keys(frame, depth_m, config)


def _stage_ledger_evidence(
    ledger: ReversibleBackgroundLedger,
    evidence: BackgroundLedgerEvidence,
    block_keys: tuple[tuple[int, int, int], ...] | None = None,
) -> LedgerDecision:
    del block_keys
    return ledger.stage(evidence)


@dataclass(frozen=True)
class _SparseRebuildObservation:
    key: object
    frame: Frame
    depth_m: np.ndarray
    block_keys: tuple[tuple[int, int, int], ...]


def _rebuild_sparse_background_blocks(
    config: TemporalGeometryConfig,
    observations: tuple[
        _SparseRebuildObservation
        | tuple[object, tuple[int, int, int], Frame, np.ndarray],
        ...,
    ],
) -> TemporalBackgroundVolume:
    if not isinstance(observations, tuple):
        raise TypeError("observations must be a tuple")
    if all(type(item) is _SparseRebuildObservation for item in observations):
        grouped_observations = observations
    else:
        if any(type(item) is _SparseRebuildObservation for item in observations):
            raise TypeError("sparse rebuild observation formats cannot be mixed")
        grouped: dict[
            object, tuple[object, Frame, np.ndarray, set[tuple[int, int, int]]]
        ] = {}
        for observation_key, block_key, frame, depth_m in sorted(
            observations, key=lambda item: (repr(item[0]), item[1])
        ):
            _, frame_depth, _, _, _ = _background_module._validate_frame(frame)
            depth = _background_module._validate_masked_depth(
                depth_m, frame_depth, config
            )
            native_key = _ledger_module._native_identity(frame)
            existing = grouped.get(native_key)
            if existing is None:
                grouped[native_key] = (
                    observation_key,
                    frame,
                    np.array(depth, copy=True),
                    {block_key},
                )
                continue
            _, first_frame, merged, block_keys = existing
            if (
                _ledger_module._native_frame_payload(first_frame)
                != _ledger_module._native_frame_payload(frame)
            ):
                raise ValueError("native frame content conflicts during sparse rebuild")
            positive = depth > 0.0
            overlap = positive & (merged > 0.0)
            if np.any(overlap & (merged != depth)):
                raise ValueError("masked depth conflict during sparse rebuild")
            merged[positive] = depth[positive]
            block_keys.add(block_key)
        grouped_observations = tuple(
            _SparseRebuildObservation(key, frame, depth, tuple(sorted(keys)))
            for key, frame, depth, keys in grouped.values()
        )
    all_keys = {
        key for item in grouped_observations for key in item.block_keys
    }
    if len(all_keys) > config.background_block_count:
        raise ValueError("TSDF block capacity would exceed background_block_count")
    rebuilt = _SparseBackgroundVolume.preallocated(config, len(all_keys))
    for item in sorted(
        grouped_observations, key=lambda observation: repr(observation.key)
    ):
        rebuilt.integrate_blocks_owned(
            item.frame, item.depth_m, item.block_keys
        )
    return rebuilt


class _SparseBackgroundLedger(ReversibleBackgroundLedger):
    def __init__(
        self,
        geometry_config: TemporalGeometryConfig,
        config: TemporalBackgroundLedgerConfig,
    ) -> None:
        super().__init__(geometry_config, config)
        release_volume = _SparseBackgroundVolume.preallocated(
            geometry_config, 1
        )
        base_volume = _SparseBackgroundVolume.preallocated(
            geometry_config, 1
        )
        self._state = replace(
            self._state,
            volume=release_volume,
            base_volume=base_volume,
            combined_volume=base_volume,
            derivation_digest=self._derivation_digest(
                base_volume, {}, base_volume
            ),
        )

    def clone(self) -> _SparseBackgroundLedger:
        clone = object.__new__(type(self))
        clone._config = self._config
        clone._maximum_ownership_records_per_observation = (
            self._maximum_ownership_records_per_observation
        )
        clone._state = replace(
            self._state,
            provisional=dict(self._provisional),
            committed=dict(self._committed),
            event_digests=dict(self._event_digests),
            native_frames=dict(self._native_frames),
            native_view_bins=dict(self._native_view_bins),
            processed_frames=dict(self._processed_frames),
        )
        return clone

    @staticmethod
    def _volume_digest(volume: TemporalBackgroundVolume) -> str:
        cached = getattr(volume, "_ledger_volume_digest", None)
        if isinstance(cached, str):
            return cached
        payload = tuple(
            (dtype, shape, hashlib.sha256(data).hexdigest())
            for dtype, shape, data in volume.canonical_block_state()
        )
        digest = _ledger_module._digest(payload)
        if isinstance(volume, _SparseBackgroundVolume):
            volume._ledger_volume_digest = digest
        return digest

    @classmethod
    def _derivation_digest(
        cls,
        base_volume: TemporalBackgroundVolume,
        committed: dict[object, object],
        combined_volume: TemporalBackgroundVolume,
    ) -> str:
        return _ledger_module._digest(
            (
                cls._volume_digest(base_volume),
                tuple(
                    _ledger_module._record_payload(committed[key])
                    for key in sorted(committed)
                ),
                cls._volume_digest(combined_volume),
            )
        )

    @property
    def _base_volume(self) -> TemporalBackgroundVolume:
        value = self._state.base_volume
        if not isinstance(value, _SparseBackgroundVolume):
            raise RuntimeError("sparse ledger base volume is unavailable")
        return value

    def integrate_base(self, frame: Frame, masked_depth: np.ndarray) -> int:
        current_base = self._base_volume
        block_keys = current_base.candidate_block_keys(frame, masked_depth)
        native_identity = _ledger_module._native_identity(frame)
        watermark = (
            int(frame.frame_id),
            native_identity,
            _ledger_module._native_frame_index(frame),
            _ledger_module._observation_payload(frame, masked_depth),
        )
        previous = self._state.base_observation_watermark
        if previous is not None:
            if frame.frame_id == previous[0] or native_identity == previous[1]:
                if watermark == previous:
                    return 0
                raise ValueError("conflicting duplicate base observation")
            if frame.frame_id <= previous[0] or watermark[2] <= previous[2]:
                raise ValueError("base observation cannot move backwards")
        journal_blocks = {
            record.contribution.block_key
            for record in (*self._provisional.values(), *self._committed.values())
        }
        active_base = current_base.active_block_key_set()
        if (
            len(active_base | set(block_keys) | journal_blocks)
            > current_base.config.background_block_count
        ):
            raise ValueError(
                "combined TSDF block capacity would exceed background_block_count"
            )
        base_volume = current_base.clone()
        if not isinstance(base_volume, _SparseBackgroundVolume):
            raise TypeError("sparse ledger base volume has an invalid type")
        if block_keys:
            base_volume.integrate_blocks_owned(frame, masked_depth, block_keys)
        else:
            base_volume._last_blocks_touched = 0
        next_state = self._finalize_state(
            replace(
                self._state,
                base_volume=base_volume,
                base_observation_watermark=watermark,
            )
        )
        self._before_publish(next_state)
        self._state = next_state
        return base_volume.last_blocks_touched

    @staticmethod
    def _record_indexes(
        provisional: dict[object, object],
        committed: dict[object, object],
        current_frame_id: int,
    ) -> tuple[
        dict[object, tuple[object, ...]],
        dict[object, int],
        dict[int, tuple[object, ...]],
    ]:
        native_frames: dict[object, tuple[object, ...]] = {}
        processed_frames: dict[int, tuple[object, ...]] = {}
        event_view_bins: dict[tuple[int, int, int], int] = {}
        for record in (*provisional.values(), *committed.values()):
            native = _ledger_module._native_identity(record.frame)
            native_payload = _ledger_module._native_frame_payload(record.frame)
            previous_native = native_frames.setdefault(native, native_payload)
            if previous_native != native_payload:
                raise ValueError(
                    "native frame content conflicts while indexing sparse records"
                )
            event_key = (
                record.entity_id,
                record.geometry_epoch,
                record.frame_id,
            )
            previous_view_bin = event_view_bins.setdefault(
                event_key, record.view_bin
            )
            if previous_view_bin != record.view_bin:
                raise ValueError(
                    "entity event view_bin conflicts while indexing sparse records"
                )
            if record.frame_id == current_frame_id:
                processed_payload = (record.frame_id, native_payload)
                previous_processed = processed_frames.setdefault(
                    record.frame_id, processed_payload
                )
                if previous_processed != processed_payload:
                    raise ValueError(
                        "processed frame content conflicts while indexing sparse records"
                    )
        # The shared index schema is native-frame scoped; entity-scoped bins are
        # validated above and consumed directly from committed records.
        return native_frames, {}, processed_frames

    def _aggregate_observations(
        self, committed: dict[object, object]
    ) -> tuple[_SparseRebuildObservation, ...]:
        grouped: dict[
            object,
            tuple[
                Frame,
                tuple[object, ...],
                np.ndarray,
                set[tuple[int, int, int]],
                dict[tuple[object, ...], np.ndarray],
                set[int],
            ],
        ] = {}
        for record_key in sorted(committed):
            record = committed[record_key]
            if record.observation_canonical is None:
                raise ValueError("sparse committed record lacks observation provenance")
            native_key = _ledger_module._native_identity(record.frame)
            existing = grouped.get(native_key)
            if existing is None:
                native_payload = _ledger_module._native_frame_payload(record.frame)
                merged = np.zeros_like(record.depth_m)
                existing = (
                    record.frame,
                    native_payload,
                    merged,
                    set(),
                    {},
                    {id(record.frame)},
                )
                grouped[native_key] = existing
            frame, native_payload, merged, block_keys, observations, frame_ids = (
                existing
            )
            block_keys.add(record.contribution.block_key)
            if id(record.frame) not in frame_ids:
                if _ledger_module._native_frame_payload(record.frame) != native_payload:
                    raise ValueError(
                        "native frame content conflicts during sparse rebuild"
                    )
                frame_ids.add(id(record.frame))
            previous_depth = observations.get(record.observation_canonical)
            if previous_depth is not None:
                if previous_depth is not record.depth_m and not np.array_equal(
                    previous_depth, record.depth_m
                ):
                    raise ValueError("masked depth conflict during sparse rebuild")
                continue
            positive = record.depth_m > 0.0
            overlap = positive & (merged > 0.0)
            if np.any(overlap & (merged != record.depth_m)):
                raise ValueError("masked depth conflict during sparse rebuild")
            merged[positive] = record.depth_m[positive]
            observations[record.observation_canonical] = record.depth_m
        return tuple(
            _SparseRebuildObservation(
                native_key,
                item[0],
                _ledger_module._readonly(item[2]),
                tuple(sorted(item[3])),
            )
            for native_key, item in sorted(grouped.items(), key=lambda pair: pair[0])
        )

    def _rebuild_committed_volume(
        self, committed: dict[object, object]
    ) -> TemporalBackgroundVolume:
        observations = self._aggregate_observations(committed)
        all_keys = {
            key for observation in observations for key in observation.block_keys
        }
        if len(all_keys) > self._volume.config.background_block_count:
            raise ValueError(
                "TSDF block capacity would exceed background_block_count"
            )
        rebuilt = _SparseBackgroundVolume.preallocated(
            self._volume.config, len(all_keys)
        )
        for observation in sorted(observations, key=lambda item: repr(item.key)):
            rebuilt._integrate_prevalidated_blocks_owned(
                observation.frame,
                observation.depth_m,
                observation.block_keys,
            )
        return rebuilt

    def _compose_combined_volume(
        self,
        base_volume: TemporalBackgroundVolume,
        committed: dict[object, object],
    ) -> TemporalBackgroundVolume:
        if not committed:
            return base_volume
        rebuilt = base_volume.clone()
        if not isinstance(rebuilt, _SparseBackgroundVolume):
            raise TypeError("sparse ledger base volume has an invalid type")
        for observation in self._aggregate_observations(committed):
            rebuilt._integrate_prevalidated_blocks_owned(
                observation.frame,
                observation.depth_m,
                observation.block_keys,
            )
        return rebuilt

    def _finalize_state(self, next_state):
        base_volume = next_state.base_volume
        if not isinstance(base_volume, _SparseBackgroundVolume):
            raise TypeError("sparse ledger state requires a base volume")
        if (
            base_volume is self._state.base_volume
            and next_state.volume is self._state.volume
            and self._state.combined_volume is not None
        ):
            return next_state
        if not next_state.committed:
            combined = base_volume
        elif base_volume.active_block_count == 0:
            combined = next_state.volume
        else:
            combined = self._compose_combined_volume(
                base_volume, next_state.committed
            )
        return replace(
            next_state,
            combined_volume=combined,
            derivation_digest=self._derivation_digest(
                base_volume, next_state.committed, combined
            ),
        )

    def validate_combined_volume(self, *, rebuild: bool = True) -> None:
        combined = self._state.combined_volume
        base = self._state.base_volume
        if not isinstance(combined, _SparseBackgroundVolume) or not isinstance(
            base, _SparseBackgroundVolume
        ):
            raise ValueError("sparse ledger combined volume state is incomplete")
        digest = self._derivation_digest(base, self._committed, combined)
        if digest != self._state.derivation_digest:
            raise ValueError("ledger combined volume derivation is invalid")
        if not rebuild:
            return
        expected = self._compose_combined_volume(base, self._committed)
        if (
            expected.config != combined.config
            or expected.canonical_block_state() != combined.canonical_block_state()
            or expected.last_blocks_touched != combined.last_blocks_touched
        ):
            raise ValueError("ledger combined volume derivation is invalid")

    def _within_capacity(
        self,
        provisional: dict[object, object],
        committed: dict[object, object],
    ) -> bool:
        if not super()._within_capacity(provisional, committed):
            return False
        base_blocks = self._base_volume.active_block_key_set()
        journal_blocks = {
            record.contribution.block_key
            for record in (*provisional.values(), *committed.values())
        }
        return (
            len(base_blocks | journal_blocks)
            <= self._base_volume.config.background_block_count
        )

    def stage(self, evidence: BackgroundLedgerEvidence) -> LedgerDecision:
        if not isinstance(evidence, BackgroundLedgerEvidence):
            return super().stage(evidence)
        if not evidence.contributions:
            return super().stage(evidence)
        event_key = (evidence.entity_id, evidence.geometry_epoch, evidence.frame_id)
        digest = self._evidence_digest(evidence)
        previous_digest = self._event_digests.get(event_key)
        if previous_digest is not None:
            if previous_digest == digest:
                return LedgerDecision.NO_OP
            raise ValueError("conflicting duplicate ledger evidence")
        if evidence.frame_id < self._last_frame_id:
            raise ValueError("evidence.frame_id cannot move backwards")
        if evidence.timestamp < self._last_timestamp:
            raise ValueError("evidence.timestamp cannot move backwards")
        if (
            evidence.frame_id == self._last_frame_id
            and evidence.timestamp != self._last_timestamp
        ):
            raise ValueError("evidence.timestamp must agree within one frame")
        event_count = (
            len(self._event_digests)
            if evidence.frame_id == self._last_frame_id
            else 0
        )
        if event_count >= self._volume.config.maximum_entities:
            return LedgerDecision.REJECTED_CAPACITY
        assert evidence.frame is not None
        assert evidence.depth_m is not None
        try:
            touched = _candidate_keys_with_sparse_fallback(
                self._volume,
                evidence.frame,
                evidence.depth_m,
                self._volume.config,
            )
        except (TypeError, ValueError):
            raise
        except Exception:
            return LedgerDecision.REJECTED_INTEGRATION
        return self._stage_sparse(evidence, touched, digest)

    def _stage_sparse(
        self,
        evidence: BackgroundLedgerEvidence,
        touched: tuple[tuple[int, int, int], ...],
        digest: str,
    ) -> LedgerDecision:
        assert evidence.frame is not None
        assert evidence.depth_m is not None
        assert evidence._native_frame_canonical is not None
        assert evidence._processed_frame_canonical is not None
        native = _ledger_module._native_identity(evidence.frame)
        existing_native = self._native_frames.get(native)
        if (
            existing_native is not None
            and existing_native != evidence._native_frame_canonical
        ):
            raise ValueError("native frame content conflicts for the same identity")
        existing_view_bin = self._native_view_bins.get(native)
        if existing_view_bin is not None and existing_view_bin != evidence.view_bin:
            raise ValueError("native frame identity is already bound to another view_bin")
        existing_processed = self._processed_frames.get(evidence.frame_id)
        if (
            existing_processed is not None
            and existing_processed != evidence._processed_frame_canonical
        ):
            raise ValueError("processed frame_id maps to conflicting native frame content")
        declared = tuple(item.block_key for item in evidence.contributions)
        if declared != touched:
            raise ValueError("contribution block_key values must exactly match touched blocks")

        provisional = dict(self._provisional)
        committed = dict(self._committed)
        assert evidence.view_bin is not None
        assert evidence._observation_canonical is not None
        for contribution in evidence.contributions:
            record = _ledger_module._Record(
                evidence.entity_id,
                evidence.geometry_epoch,
                evidence.frame_id,
                evidence.timestamp,
                evidence.view_bin,
                contribution,
                evidence.frame,
                evidence.depth_m,
                evidence._observation_canonical,
            )
            if record.key in provisional or record.key in committed:
                raise ValueError("duplicate contribution key")
            provisional[record.key] = record
        if not self._within_capacity(provisional, committed):
            return LedgerDecision.REJECTED_CAPACITY

        grouped: dict[tuple[object, ...], list[object]] = {}
        for record in provisional.values():
            grouped.setdefault(_ledger_module._record_group_key(record), []).append(record)
        eligible: set[tuple[object, ...]] = set()
        for group_key, records in grouped.items():
            physical: dict[object, object] = {}
            for record in sorted(records, key=lambda item: item.key):
                physical.setdefault(_ledger_module._native_identity(record.frame), record)
            observations = tuple(physical.values())
            view_bins = {record.view_bin for record in observations}
            indices = {
                _ledger_module._native_frame_index(record.frame)
                for record in observations
            }
            if (
                len(observations) >= self.config.commit_support_frames
                and len(view_bins) >= self.config.commit_distinct_view_bins
                and max(indices) - min(indices) >= self.config.minimum_commit_frame_gap
            ):
                eligible.add(group_key)
        committing = {
            key: record for key, record in provisional.items()
            if _ledger_module._record_group_key(record) in eligible
        }
        if committing:
            committed.update(committing)
            for key in committing:
                del provisional[key]
            try:
                rebuilt = self._rebuild_committed_volume(committed)
            except Exception:
                return LedgerDecision.REJECTED_INTEGRATION
            self._publish_event(
                evidence,
                digest,
                provisional,
                committed,
                volume=rebuilt,
                generation=self._generation + 1,
            )
            return LedgerDecision.COMMITTED
        self._publish_event(evidence, digest, provisional, committed)
        return LedgerDecision.STAGED


def _empty_submap(reference: tuple[float, float, float]) -> ObjectSubmap:
    return ObjectSubmap(
        reference_centroid_xyz=reference,
        local_voxel_keys=(),
        local_points_xyz=np.empty((0, 3), dtype=np.float64),
        weights=np.empty((0,), dtype=np.float64),
        last_seen_frame_ids=np.empty((0,), dtype=np.int64),
    )


def _proposal_observation(
    proposal: object,
    identities: IdentityMemoryBank,
    voxel_size_m: float,
    capacity: int,
) -> FrameObservation:
    from src.oviv2.temporal_proposals import RecoveredTemporalProposal

    if type(proposal) is not RecoveredTemporalProposal:
        raise TypeError("proposal must be a RecoveredTemporalProposal")
    record = identities.get(proposal.identity_hint)
    semantic_id = 0
    confidence = 1.0
    prototype = None
    model_id = None
    if record is not None:
        if record.semantic_probabilities:
            semantic_id, confidence = max(record.semantic_probabilities, key=lambda item: (item[1], -item[0]))
        prototype = record.appearance_prototype if proposal.appearance_available else None
        model_id = record.feature_model_id if prototype is not None else None
    key = tuple(int(math.floor(value / voxel_size_m)) for value in proposal.centroid_xyz)
    observation_id = (1 << 62) + proposal.frame_id * capacity + proposal.proposal_id
    if observation_id >= 3 * (1 << 61):
        raise OverflowError(
            "recovered proposal observation ID exceeds reserved tag10 namespace"
        )
    observation = FrameObservation(
        observation_id=observation_id,
        frame_id=proposal.frame_id,
        timestamp=proposal.timestamp,
        kind=ObservationKind.OBJECT,
        label=(
            f"temporal-recovery:{proposal.identity_hint}:"
            f"{proposal.projection_provenance_hash}:{proposal.semantic_provenance_hash}"
        ),
        semantic_id=semantic_id,
        confidence=float(confidence),
        mask=proposal.mask,
        bbox_xyxy=tuple(float(value) for value in proposal.bbox_xyxy),
        voxel_keys=frozenset({key}),
        centroid_xyz=proposal.centroid_xyz,
        bounds_min_xyz=proposal.bounds_min_xyz,
        bounds_max_xyz=proposal.bounds_max_xyz,
        image_feature=prototype,
        feature_model_id=model_id,
        visible_pixel_count=proposal.area_px,
    )
    regularized = regularize_temporal_object_extents(
        (observation,), voxel_size_m
    )[0]
    if regularized is observation:
        return observation
    lower = np.asarray(observation.bounds_min_xyz, dtype=np.float64)
    upper = np.asarray(observation.bounds_max_xyz, dtype=np.float64)
    centroid = np.asarray(observation.centroid_xyz, dtype=np.float64)
    cell_lower = np.asarray(regularized.bounds_min_xyz, dtype=np.float64)
    cell_upper = np.asarray(regularized.bounds_max_xyz, dtype=np.float64)
    positive = upper > lower
    adjusted_lower = np.minimum(np.where(positive, lower, cell_lower), centroid)
    adjusted_upper = np.maximum(np.where(positive, upper, cell_upper), centroid)
    object.__setattr__(
        regularized,
        "bounds_min_xyz",
        tuple(float(value) for value in adjusted_lower),
    )
    object.__setattr__(
        regularized,
        "bounds_max_xyz",
        tuple(float(value) for value in adjusted_upper),
    )
    return regularized


def _motion_confidence(
    motion: object,
    config: TemporalReadoutConfig,
    previous_submap: ObjectSubmap | None = None,
    points_world: np.ndarray | None = None,
) -> float:
    if motion.decision is MotionDecision.REJECTED:
        return 0.0
    if motion.decision is MotionDecision.ICP_ACCEPTED:
        return float(motion.fitness)
    assert config.motion is not None
    if previous_submap is None or points_world is None:
        return 0.0
    limit = config.geometry.maximum_visibility_points_per_entity
    previous = previous_submap.world_points(motion.object_to_world)[:limit]
    current = np.asarray(points_world, dtype=np.float64)[:limit]
    if (
        previous.ndim != 2
        or current.ndim != 2
        or previous.shape[1:] != (3,)
        or current.shape[1:] != (3,)
        or previous.shape[0] == 0
        or current.shape[0] == 0
        or not np.isfinite(previous).all()
        or not np.isfinite(current).all()
    ):
        return 0.0
    residual = _symmetric_nearest_residual(previous, current)
    residual_limit = float(config.motion.maximum_translation_residual_m)
    confidence = max(0.0, min(1.0, 1.0 - residual / residual_limit))
    return (
        confidence
        if confidence >= config.motion.minimum_translation_confidence else 0.0
    )


def _symmetric_nearest_residual(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1:] != (3,)
        or right.shape[1:] != (3,)
        or left.shape[0] == 0
        or right.shape[0] == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("nearest residual requires finite nonempty (N, 3) arrays")

    def mean_minimum_squared(source: np.ndarray, target: np.ndarray) -> float:
        tree = cKDTree(target)
        total = 0.0
        for start in range(0, source.shape[0], _NEAREST_QUERY_CHUNK_SIZE):
            query = source[start : start + _NEAREST_QUERY_CHUNK_SIZE]
            distances, _ = tree.query(query, k=1, workers=1)
            total += float(np.dot(distances, distances))
        return total / source.shape[0]

    return math.sqrt(
        0.5
        * (
            mean_minimum_squared(left, right)
            + mean_minimum_squared(right, left)
        )
    )


class TemporalCurrentRuntime:
    def __init__(
        self,
        scene_id: str,
        config: TemporalReadoutConfig,
        tracker_config: LocalTrackerConfig,
    ) -> None:
        if not isinstance(config, TemporalReadoutConfig):
            raise TypeError("config must be TemporalReadoutConfig")
        if not isinstance(tracker_config, LocalTrackerConfig):
            raise TypeError("tracker_config must be LocalTrackerConfig")
        if config.execution_profile not in (
            ExecutionProfile.A2,
            ExecutionProfile.A3,
            ExecutionProfile.A4,
        ):
            raise ValueError(
                "TemporalCurrentRuntime requires execution profile A2, A3, or A4"
            )
        self.config = temporal_config_from_json(temporal_config_to_json(config))
        self.tracker_config = tracker_config
        assert self.config.identity is not None
        assert self.config.geometry_epoch is not None
        assert self.config.background_ledger is not None
        identities = IdentityMemoryBank(self.config.identity)
        geometry = TemporalGeometryState(
            (),
            self.config.geometry_epoch.maximum_epochs_per_identity,
            self.config.geometry_epoch.maximum_retained_epochs,
        )
        ledger = (
            None
            if self.config.execution_profile is ExecutionProfile.A2
            or not self.config.background_ledger_enabled
            else _SparseBackgroundLedger(
                self.config.geometry, self.config.background_ledger
            )
        )
        self.state = TemporalRuntimeState(
            scene_id=scene_id,
            revision=0,
            last_frame_id=-1,
            last_timestamp=-float(np.finfo(np.float64).max),
            next_entity_id=1,
            entities=(),
            background=TemporalBackgroundVolume(self.config.geometry),
            tracker=LocalTracker(tracker_config),
            identities=identities,
            geometry=geometry,
            lifecycle_beliefs=(),
            background_ledger=ledger,
            background_mode=(
                "profile_locked"
                if self.config.background_ledger_enabled
                else "masking_only"
            ),
            export_tracker=TemporalExportTracker(),
            diagnostics=TemporalDiagnostics(),
        )

    def _before_publish(self, next_state: TemporalRuntimeState) -> None:
        del next_state

    def advance_frame_without_update(self, frame: Frame) -> TemporalFrameResult:
        current = self.state
        frame = _validate_frame(frame)
        if frame.frame_id != current.last_frame_id + 1:
            raise ValueError("advance frame_id must immediately follow runtime progress")
        if current.last_frame_id != -1 and float(frame.timestamp) <= current.last_timestamp:
            raise ValueError("frame.timestamp must increase strictly")
        timestamp = float(frame.timestamp)
        timestamp_ns = timestamp_seconds_to_ns(timestamp)
        assert current.lifecycle_beliefs is not None
        lifecycle_beliefs = tuple(
            replace(
                belief,
                last_frame_id=frame.frame_id,
                last_timestamp=timestamp,
            )
            for belief in current.lifecycle_beliefs
        )
        lifecycle_by_id = {
            belief.entity_id: belief for belief in lifecycle_beliefs
        }
        entities = tuple(
            replace(
                entity,
                lifecycle=lifecycle_by_id[entity.lifecycle.entity_id],
            )
            for entity in current.entities
        )
        tracker = current._advanced_tracker_snapshot(frame.frame_id)
        export = TemporalExportBatch(frame.frame_id, timestamp_ns, (), ())
        assert current.export_tracker is not None
        export_tracker = TemporalExportTracker(
            current.export_tracker.entries, export
        )
        frame_diagnostics = TemporalFrameDiagnostics(frame_id=frame.frame_id)
        assert current.diagnostics is not None
        diagnostics = replace(
            current.diagnostics,
            processed_frame_count=current.diagnostics.processed_frame_count + 1,
            last_frame=frame_diagnostics,
            assignment_diagnostics=(),
        )
        assert current.geometry is not None
        next_state = TemporalRuntimeState._adopt_owned(
            scene_id=current.scene_id,
            revision=current.revision + 1,
            last_frame_id=frame.frame_id,
            last_timestamp=timestamp,
            next_entity_id=current.next_entity_id,
            entities=entities,
            background=current._owned_background(),
            tracker=tracker,
            identities=object.__getattribute__(current, "_identities_state"),
            geometry=current.geometry,
            lifecycle_beliefs=lifecycle_beliefs,
            background_ledger=object.__getattribute__(current, "_ledger_state"),
            background_mode=current.background_mode,
            export_tracker=export_tracker,
            diagnostics=diagnostics,
        )
        result = TemporalFrameResult(
            frame_id=frame.frame_id,
            revision=next_state.revision,
            active_entity_ids=tuple(
                entity.lifecycle.entity_id
                for entity in entities
                if entity.lifecycle.lifecycle
                in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
            ),
            dormant_entity_ids=tuple(
                entity.lifecycle.entity_id
                for entity in entities
                if entity.lifecycle.lifecycle is TemporalLifecycle.DORMANT
            ),
            new_entity_ids=(),
            reactivated_entity_ids=(),
            background_blocks_touched=0,
            export=export,
            diagnostics=frame_diagnostics,
        )
        self._before_publish(next_state)
        self.state = next_state
        return result

    def process_frame(
        self,
        frame: Frame,
        observations: tuple[FrameObservation, ...],
        dense_semantics: DenseSemanticFrame | None = None,
        *,
        proposal_evidence: ProposalRecoveryInput | None = None,
    ) -> TemporalFrameResult:
        current = self.state
        frame, observations, dense_semantics = _validate_inputs(
            frame, observations, dense_semantics, current
        )
        _validate_proposal_evidence(
            frame, observations, dense_semantics, proposal_evidence,
            self.config, current,
        )
        trial_identities = current._mutable_identities_snapshot()
        geometry_transaction = current.geometry.transaction()
        trial_ledger = (
            None
            if self.config.execution_profile is ExecutionProfile.A2
            or not self.config.background_ledger_enabled
            else current._mutable_ledger_snapshot()
        )
        ledger_before_staged = _ledger_group_records(trial_ledger, committed=False)
        ledger_before_committed = _ledger_group_records(trial_ledger, committed=True)
        lifecycle_by_id = {
            item.entity_id: item for item in current.lifecycle_beliefs
        }
        expiry = trial_identities.expire_dormant(
            current_frame_id=frame.frame_id
        )
        expired_ids = set(expiry.expired_identity_ids)
        for identity_id in expired_ids:
            lifecycle_by_id.pop(identity_id, None)
            geometry_transaction.remove_identity(identity_id, forget=True)
        retained_current_entities = tuple(
            item for item in current.entities
            if item.lifecycle.entity_id not in expired_ids
        )
        proposal_opportunities = proposal_triggers = 0
        proposal_opportunity_records: tuple[str, ...] = ()
        proposal_trigger_records: tuple[str, ...] = ()
        if proposal_evidence is not None and self.config.proposal_recovery_enabled:
            assert self.config.proposal is not None
            recovery = recover_temporal_proposals(proposal_evidence, self.config.proposal)
            proposal_opportunities = recovery.opportunity_count
            proposal_triggers = recovery.trigger_count
            proposal_opportunity_records = recovery.opportunity_records
            proposal_trigger_records = recovery.trigger_records
            recovered = tuple(
                _proposal_observation(
                    item,
                    trial_identities,
                    self.config.geometry.voxel_size_m,
                    self.config.proposal.maximum_recovered_proposals,
                )
                for item in recovery.proposals
            )
            existing_ids = {item.observation_id for item in observations}
            if existing_ids & {item.observation_id for item in recovered}:
                raise ValueError("recovered proposal observation IDs collide")
            observations = observations + recovered
        trial_tracker = current._mutable_tracker_snapshot()
        batch = trial_tracker.update(observations, frame.frame_id)
        confirmed = tuple(
            sorted(
                (track.observations[-1] for track in batch.accepted),
                key=lambda item: item.observation_id,
            )
        )
        allow_dormant_reid = (
            self.config.execution_profile is ExecutionProfile.A4
            and self.config.dormant_reid_enabled
        )
        association_entities = tuple(
            entity
            for entity in retained_current_entities
            if entity.lifecycle.lifecycle is not TemporalLifecycle.DORMANT
        )
        excluded_dormant = tuple(
            entity
            for entity in retained_current_entities
            if entity.lifecycle.lifecycle is TemporalLifecycle.DORMANT
            and not allow_dormant_reid
        )
        targets_by_id = {
            entity.lifecycle.entity_id: _association_target(entity)
            for entity in association_entities
        }
        if allow_dormant_reid:
            for record in trial_identities.records:
                belief = lifecycle_by_id[record.identity_id]
                if belief.lifecycle is TemporalLifecycle.DORMANT:
                    targets_by_id[record.identity_id] = _identity_association_target(
                        record, belief, float(frame.timestamp)
                    )
        targets = tuple(targets_by_id[key] for key in sorted(targets_by_id))
        association = associate_temporal_observations(
            confirmed, targets, self.config.association,
            dormant_reid=(
                self.config.identity
                if self.config.dormant_reid_enabled
                else None
            ),
        )
        entities_by_id = {
            item.lifecycle.entity_id: item for item in retained_current_entities
        }
        bank_only_assignments = tuple(
            pair for pair in association.assignments
            if pair[1] not in entities_by_id
        )
        matched_current_ids = {
            entity_id for _, entity_id in association.assignments
            if entity_id in entities_by_id
        }
        reclaimable_dormant_count = sum(
            entity.lifecycle.lifecycle is TemporalLifecycle.DORMANT
            and entity.lifecycle.entity_id not in matched_current_ids
            for entity in retained_current_entities
        )
        maximum_bank_reactivations = (
            self.config.geometry.maximum_entities
            - len(retained_current_entities)
            + reclaimable_dormant_count
        )
        if len(bank_only_assignments) > maximum_bank_reactivations:
            diagnostic_by_pair = {
                (item.observation_id, item.entity_id): item
                for item in association.assignment_diagnostics
            }
            retained_bank_assignments = set(sorted(
                bank_only_assignments,
                key=lambda pair: (
                    -diagnostic_by_pair[pair].score,
                    pair[0],
                    pair[1],
                ),
            )[:maximum_bank_reactivations])
            dropped_assignments = set(bank_only_assignments) - retained_bank_assignments
            retained_assignments = tuple(
                pair for pair in association.assignments
                if pair not in dropped_assignments
            )
            retained_assignment_set = set(retained_assignments)
            retained_reid_triggers = tuple(
                pair for pair in association.reid_trigger_pairs
                if pair in retained_assignment_set
            )
            association = replace(
                association,
                assignments=retained_assignments,
                unmatched_observation_ids=tuple(sorted({
                    *association.unmatched_observation_ids,
                    *(pair[0] for pair in dropped_assignments),
                })),
                unmatched_entity_ids=tuple(sorted({
                    *association.unmatched_entity_ids,
                    *(pair[1] for pair in dropped_assignments),
                })),
                reid_trigger_count=len(retained_reid_triggers),
                assignment_diagnostics=tuple(
                    item for item in association.assignment_diagnostics
                    if (item.observation_id, item.entity_id)
                    in retained_assignment_set
                ),
                reid_trigger_pairs=retained_reid_triggers,
            )
        observations_by_id = {item.observation_id: item for item in confirmed}
        next_entities: dict[int, TemporalEntityState] = {}
        reactivated: list[int] = []
        forced_new_observation_ids: list[int] = []
        motion_by_entity: dict[int, tuple[float, float]] = {}
        evidence_by_entity: dict[int, TemporalEvidence] = {}
        released_pixels_by_entity: dict[int, np.ndarray] = {}
        motion_events: list[tuple[int, int, MotionDecision]] = []
        epoch_reset_triggers = 0
        epoch_reset_trigger_records: list[str] = []
        old_lifecycle_by_entity = {
            item.entity_id: item
            for item in current.lifecycle_beliefs
            if item.entity_id not in expired_ids
        }
        diagnostic_by_pair = {
            (item.observation_id, item.entity_id): item
            for item in association.assignment_diagnostics
        }
        suppressed_reid_trigger_pairs: set[tuple[int, int]] = set()

        for old in excluded_dormant:
            absence, released = _absence_evidence_with_release(
                old, frame, self.config
            )
            released_pixels_by_entity[old.lifecycle.entity_id] = released
            lifecycle = advance_lifecycle(
                old.lifecycle,
                absence,
                self.config.lifecycle,
            )
            lifecycle_by_id[old.lifecycle.entity_id] = lifecycle
            evidence_by_entity[old.lifecycle.entity_id] = absence
            geometry_transaction.replace_current(
                geometry_transaction.current(old.lifecycle.entity_id).apply_evidence(absence.kind)
            )
            next_entities[old.lifecycle.entity_id] = TemporalEntityState(
                lifecycle=lifecycle,
                semantic_probabilities=old.semantic_probabilities,
                image_prototype=old.image_prototype,
                extent_xyz=old.extent_xyz,
                object_to_world=old.object_to_world,
                submap=old.submap,
                first_seen_frame_id=old.first_seen_frame_id,
                last_seen_frame_id=old.last_seen_frame_id,
                feature_model_id=old.feature_model_id,
            )

        for observation_id, entity_id in association.assignments:
            observation = observations_by_id[observation_id]
            points = backproject_observation(frame, observation, self.config.geometry)
            old = entities_by_id.get(entity_id)
            if points.shape[0] == 0:
                pair = (observation_id, entity_id)
                if pair in association.reid_trigger_pairs:
                    suppressed_reid_trigger_pairs.add(pair)
                evidence_kind = (
                    TemporalEvidenceKind.OUT_OF_VIEW
                    if old is None
                    else TemporalEvidenceKind.OCCLUDED
                )
                evidence = TemporalEvidence(
                    evidence_kind,
                    0.0,
                    frame.frame_id,
                    float(frame.timestamp),
                    None,
                )
                belief = (
                    lifecycle_by_id[entity_id]
                    if old is None
                    else old.lifecycle
                )
                lifecycle = advance_lifecycle(
                    belief, evidence, self.config.lifecycle,
                )
                lifecycle_by_id[entity_id] = lifecycle
                evidence_by_entity[entity_id] = evidence
                if old is not None:
                    next_entities[entity_id] = replace(old, lifecycle=lifecycle)
                continue
            if old is None:
                record = trial_identities.get(entity_id)
                if record is None:
                    raise RuntimeError("association selected an expired identity")
                belief = lifecycle_by_id[entity_id]
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = observation.centroid_xyz
                submap = integrate_object_submap(
                    _empty_submap(observation.centroid_xyz), points,
                    frame.frame_id, self.config.geometry,
                    object_to_world=pose,
                )
                epoch_id = geometry_transaction.next_epoch_id(entity_id)
                epoch = GeometryEpoch(
                    entity_id, epoch_id, pose, submap, True, None,
                    frame.frame_id,
                )
                geometry_transaction.append(epoch)
                present = TemporalEvidence(
                    TemporalEvidenceKind.PRESENT,
                    float(observation.confidence), frame.frame_id,
                    float(frame.timestamp), None,
                )
                lifecycle = advance_lifecycle(
                    belief, present, self.config.lifecycle,
                )
                lifecycle_by_id[entity_id] = lifecycle
                if belief.lifecycle is TemporalLifecycle.DORMANT and lifecycle.lifecycle is TemporalLifecycle.ACTIVE:
                    reactivated.append(entity_id)
                prototype, feature_model_id = _prototype_update(
                    record.appearance_prototype, record.feature_model_id,
                    observation,
                )
                next_entities[entity_id] = TemporalEntityState(
                    lifecycle=lifecycle,
                    semantic_probabilities=_semantic_update(
                        record.semantic_probabilities, observation
                    ),
                    image_prototype=prototype,
                    extent_xyz=tuple(
                        (old_value + new_value) / 2.0
                        for old_value, new_value in zip(
                            record.extent_xyz, _observed_extent(observation)
                        )
                    ),
                    object_to_world=pose,
                    submap=submap,
                    first_seen_frame_id=record.first_frame_id,
                    last_seen_frame_id=frame.frame_id,
                    feature_model_id=feature_model_id,
                )
                evidence_by_entity[entity_id] = present
                motion_by_entity[entity_id] = (0.0, 0.0)
                continue
            motion_estimator = (
                estimate_object_motion
                if self.config.execution_profile is ExecutionProfile.A4
                and self.config.icp_enabled
                else estimate_object_translation
            )
            motion = motion_estimator(
                old.submap,
                points,
                observation.centroid_xyz,
                self.config.geometry,
                previous_object_to_world=old.object_to_world,
            )
            motion_events.append((observation_id, entity_id, motion.decision))
            epoch = geometry_transaction.current(entity_id)
            diagnostic = diagnostic_by_pair[(observation_id, entity_id)]
            if motion.decision is MotionDecision.REJECTED:
                if not diagnostic.high_confidence_identity_match:
                    forced_new_observation_ids.append(observation_id)
                    occluded = TemporalEvidence(
                        TemporalEvidenceKind.OCCLUDED, 0.0, frame.frame_id,
                        float(frame.timestamp), None,
                    )
                    evidence_by_entity[entity_id] = occluded
                    lifecycle = advance_lifecycle(
                        old.lifecycle, occluded, self.config.lifecycle,
                    )
                    lifecycle_by_id[entity_id] = lifecycle
                    next_entities[entity_id] = replace(old, lifecycle=lifecycle)
                    continue
                epoch = start_new_epoch(
                    epoch,
                    motion,
                    points,
                    observation.centroid_xyz,
                    frame.frame_id,
                    self.config.geometry,
                )
                epoch_reset_triggers += 1
                epoch_reset_trigger_records.append(
                    f"motion:{frame.frame_id}:{observation_id}:{entity_id}"
                )
                geometry_transaction.append(epoch)
            else:
                epoch = epoch.integrate(
                    motion, points, frame.frame_id, self.config.geometry
                )
                geometry_transaction.replace_current(epoch)
            submap = epoch.submap
            motion_confidence = _motion_confidence(
                motion, self.config, old.submap, points
            )
            displacement = float(
                np.linalg.norm(
                    np.asarray(epoch.object_to_world[:3, 3], dtype=np.float64)
                    - np.asarray(old.object_to_world[:3, 3], dtype=np.float64)
                )
            )
            motion_by_entity[entity_id] = (displacement, motion_confidence)
            present = TemporalEvidence(
                TemporalEvidenceKind.PRESENT,
                float(observation.confidence),
                frame.frame_id,
                float(frame.timestamp),
                None,
            )
            evidence_by_entity[entity_id] = present
            lifecycle = advance_lifecycle(
                old.lifecycle, present, self.config.lifecycle,
            )
            lifecycle_by_id[entity_id] = lifecycle
            if old.lifecycle.lifecycle is TemporalLifecycle.DORMANT and lifecycle.lifecycle is TemporalLifecycle.ACTIVE:
                reactivated.append(entity_id)
            observed_extent = _observed_extent(observation)
            extent = tuple(
                (old_value + new_value) / 2.0
                for old_value, new_value in zip(old.extent_xyz, observed_extent)
            )
            prototype, feature_model_id = _prototype_update(
                old.image_prototype, old.feature_model_id, observation
            )
            next_entities[entity_id] = TemporalEntityState(
                lifecycle=lifecycle,
                semantic_probabilities=_semantic_update(old.semantic_probabilities, observation),
                image_prototype=prototype,
                extent_xyz=extent,
                object_to_world=epoch.object_to_world,
                submap=submap,
                first_seen_frame_id=old.first_seen_frame_id,
                last_seen_frame_id=frame.frame_id,
                feature_model_id=feature_model_id,
            )

        if suppressed_reid_trigger_pairs:
            retained_reid_triggers = tuple(
                pair for pair in association.reid_trigger_pairs
                if pair not in suppressed_reid_trigger_pairs
            )
            association = replace(
                association,
                reid_trigger_count=len(retained_reid_triggers),
                reid_trigger_pairs=retained_reid_triggers,
            )

        for entity_id in association.unmatched_entity_ids:
            old = entities_by_id.get(entity_id)
            if old is None:
                belief = lifecycle_by_id[entity_id]
                out_of_view = TemporalEvidence(
                    TemporalEvidenceKind.OUT_OF_VIEW, 0.0,
                    frame.frame_id, float(frame.timestamp), None,
                )
                lifecycle_by_id[entity_id] = advance_lifecycle(
                    belief, out_of_view, self.config.lifecycle,
                )
                evidence_by_entity[entity_id] = out_of_view
                continue
            absence, released = _absence_evidence_with_release(
                old, frame, self.config
            )
            released_pixels_by_entity[entity_id] = released
            lifecycle = advance_lifecycle(
                old.lifecycle,
                absence,
                self.config.lifecycle,
            )
            lifecycle_by_id[entity_id] = lifecycle
            evidence_by_entity[entity_id] = absence
            geometry_transaction.replace_current(
                geometry_transaction.current(entity_id).apply_evidence(absence.kind)
            )
            next_entities[entity_id] = TemporalEntityState(
                lifecycle=lifecycle,
                semantic_probabilities=old.semantic_probabilities,
                image_prototype=old.image_prototype,
                extent_xyz=old.extent_xyz,
                object_to_world=old.object_to_world,
                submap=old.submap,
                first_seen_frame_id=old.first_seen_frame_id,
                last_seen_frame_id=old.last_seen_frame_id,
                feature_model_id=old.feature_model_id,
            )

        reclaimed_ids: list[int] = []
        geometry_reclaim_records: list[str] = []

        def reclaim_dormant_wrapper(*, protected_ids: set[int]) -> bool:
            candidates = sorted(
                (
                    item for item in next_entities.values()
                    if item.lifecycle.lifecycle is TemporalLifecycle.DORMANT
                    and item.lifecycle.entity_id not in protected_ids
                ),
                key=lambda item: (
                    item.last_seen_frame_id, item.lifecycle.entity_id
                ),
            )
            if not candidates:
                return False
            victim_id = candidates[0].lifecycle.entity_id
            epoch_id = geometry_transaction.current(victim_id).epoch_id
            del next_entities[victim_id]
            geometry_transaction.remove_identity(victim_id)
            reclaimed_ids.append(victim_id)
            geometry_reclaim_records.append(
                f"geometry:{victim_id}:{epoch_id}:{frame.frame_id}"
            )
            return True

        assigned_entity_ids = {entity_id for _, entity_id in association.assignments}
        while len(next_entities) > self.config.geometry.maximum_entities:
            if not reclaim_dormant_wrapper(protected_ids=assigned_entity_ids):
                raise RuntimeError("geometry capacity cannot retain observed identities")

        next_entity_id = trial_identities._next_identity_id
        new_ids: list[int] = []
        for observation_id in tuple(sorted((*association.unmatched_observation_ids, *forced_new_observation_ids))):
            observation = observations_by_id[observation_id]
            points = backproject_observation(frame, observation, self.config.geometry)
            if points.shape[0] == 0:
                continue
            if (
                len(trial_identities.records)
                >= trial_identities._config.maximum_identities
            ):
                continue
            if len(next_entities) >= self.config.geometry.maximum_entities:
                if not reclaim_dormant_wrapper(protected_ids=assigned_entity_ids):
                    continue
            entity_id = next_entity_id
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = observation.centroid_xyz
            submap = integrate_object_submap(
                _empty_submap(observation.centroid_xyz),
                points,
                frame.frame_id,
                self.config.geometry,
                object_to_world=pose,
            )
            present = TemporalEvidence(
                TemporalEvidenceKind.PRESENT,
                float(observation.confidence),
                frame.frame_id,
                float(frame.timestamp),
                None,
            )
            if current.last_frame_id >= 0:
                seed = TemporalLifecycleState(
                    entity_id=entity_id,
                    lifecycle=TemporalLifecycle.UNCERTAIN,
                    existence_log_odds=self.config.lifecycle.initial_log_odds,
                    last_frame_id=current.last_frame_id,
                    last_timestamp=current.last_timestamp,
                    absent_streak=0,
                    absence_view_bins=(),
                )
                lifecycle = advance_lifecycle(
                    seed, present, self.config.lifecycle,
                )
            else:
                lifecycle = _initial_lifecycle(
                    entity_id, frame, float(observation.confidence), self.config
                )
            lifecycle_by_id[entity_id] = lifecycle
            prototype, feature_model_id = _prototype_update(
                None, None, observation
            )
            next_entities[entity_id] = TemporalEntityState(
                lifecycle=lifecycle,
                semantic_probabilities=_semantic_update((), observation),
                image_prototype=prototype,
                extent_xyz=_observed_extent(observation),
                object_to_world=pose,
                submap=submap,
                first_seen_frame_id=frame.frame_id,
                last_seen_frame_id=frame.frame_id,
                feature_model_id=feature_model_id,
            )
            geometry_transaction.append(
                GeometryEpoch(
                    entity_id, 0, pose, submap, True, None, frame.frame_id
                )
            )
            evidence_by_entity[entity_id] = present
            motion_by_entity[entity_id] = (0.0, 0.0)
            new_ids.append(entity_id)
            next_entity_id += 1

        trial_geometry = geometry_transaction.finalize()
        ordered_entities = tuple(next_entities[key] for key in sorted(next_entities))
        if self.config.execution_profile is ExecutionProfile.A2:
            if trial_ledger is not None:
                raise RuntimeError("A2 must not own a background ledger")
            trial_background = current._mutable_background_snapshot()
            if trial_background.active_block_count:
                raise RuntimeError("A2 temporal background must remain empty")
            trial_background._last_blocks_touched = 0
            background_blocks_touched = 0
        elif not self.config.background_ledger_enabled:
            if trial_ledger is not None:
                raise RuntimeError("masking-only control must not own a background ledger")
            protected = tuple(
                entity.submap.world_points(entity.object_to_world)[
                    : self.config.geometry.maximum_visibility_points_per_entity
                ]
                for entity in ordered_entities
                if entity.lifecycle.lifecycle
                in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
                and trial_geometry.current(entity.lifecycle.entity_id).readout_valid
            )
            background_depth = build_background_depth(
                frame, observations, protected, self.config.geometry
            )
            trial_background = current.background.trial_integrate(
                frame, background_depth.depth_m
            )
            background_blocks_touched = trial_background.last_blocks_touched
            ledger_decisions = []
        else:
            if trial_ledger is None:
                raise RuntimeError("A3/A4 require a background ledger")
            protected = tuple(
                entity.submap.world_points(entity.object_to_world)[
                    : self.config.geometry.maximum_visibility_points_per_entity
                ]
                for entity in ordered_entities
                if entity.lifecycle.lifecycle in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
                and trial_geometry.current(entity.lifecycle.entity_id).readout_valid
            )
            background_depth = build_background_depth(
                frame, observations, protected, self.config.geometry
            )
            released_mask = np.zeros(background_depth.depth_m.shape, dtype=bool)
            for entity_id, temporal_evidence in evidence_by_entity.items():
                if temporal_evidence.kind is TemporalEvidenceKind.VISIBLE_ABSENT:
                    released_mask |= released_pixels_by_entity[entity_id]
            base_depth = np.where(
                released_mask, 0.0, background_depth.depth_m
            ).astype(background_depth.depth_m.dtype, copy=False)
            base_blocks_touched = trial_ledger.integrate_base(frame, base_depth)
            ledger_decisions: list[LedgerDecision] = []
            ledger_frame: Frame | None = None
            for entity in ordered_entities:
                entity_id = entity.lifecycle.entity_id
                temporal_evidence = evidence_by_entity.get(entity_id)
                kind = (
                    None if temporal_evidence is None else temporal_evidence.kind
                )
                if kind not in (
                    TemporalEvidenceKind.PRESENT,
                    TemporalEvidenceKind.OCCLUDED,
                    TemporalEvidenceKind.VISIBLE_ABSENT,
                ):
                    continue
                epoch = trial_geometry.current(entity_id)
                if kind is TemporalEvidenceKind.VISIBLE_ABSENT:
                    assert temporal_evidence is not None
                    if temporal_evidence.view_bin is None:
                        raise RuntimeError(
                            "visible-absent lifecycle evidence lacks view_bin"
                        )
                    released_mask = released_pixels_by_entity[entity_id]
                    released_depth = np.where(
                        released_mask, background_depth.depth_m, 0.0
                    ).astype(background_depth.depth_m.dtype, copy=False)
                    if ledger_frame is None:
                        ledger_frame = _ledger_module._frozen_frame(frame)
                    block_keys = _candidate_keys_with_sparse_fallback(
                        trial_ledger._volume,
                        ledger_frame,
                        released_depth,
                        self.config.geometry,
                    )
                    if not block_keys:
                        continue
                    evidence = BackgroundLedgerEvidence(
                        entity_id=entity_id,
                        geometry_epoch=epoch.epoch_id,
                        frame_id=frame.frame_id,
                        timestamp=float(frame.timestamp),
                        kind=kind,
                        view_bin=temporal_evidence.view_bin,
                        contributions=tuple(BackgroundContribution(key) for key in block_keys),
                        frame=ledger_frame,
                        depth_m=released_depth,
                    )
                else:
                    evidence = BackgroundLedgerEvidence(
                        entity_id=entity_id,
                        geometry_epoch=epoch.epoch_id,
                        frame_id=frame.frame_id,
                        timestamp=float(frame.timestamp),
                        kind=kind,
                        view_bin=None,
                        contributions=(),
                    )
                decision = _stage_ledger_evidence(
                    trial_ledger,
                    evidence,
                    block_keys if kind is TemporalEvidenceKind.VISIBLE_ABSENT else None,
                )
                ledger_decisions.append(decision)
                if decision in (
                    LedgerDecision.REJECTED_CAPACITY,
                    LedgerDecision.REJECTED_INTEGRATION,
                ):
                    raise RuntimeError(f"background ledger rejected frame: {decision.value}")
            # The trial ledger owns this freshly finalized volume. Runtime state
            # adopts it immutably, while public ledger access still returns a clone.
            trial_background = trial_ledger._published_volume
            background_blocks_touched = max(
                base_blocks_touched, trial_background.last_blocks_touched
            )
        if self.config.execution_profile is ExecutionProfile.A2:
            ledger_decisions = []
        ledger_after_staged = _ledger_group_records(trial_ledger, committed=False)
        ledger_after_committed = _ledger_group_records(trial_ledger, committed=True)
        ledger_stage_records = tuple(sorted(
            (ledger_after_staged | ledger_after_committed)
            - (ledger_before_staged | ledger_before_committed)
        ))
        ledger_commit_records = tuple(sorted(
            ledger_after_committed - ledger_before_committed
        ))
        ledger_reclaim_records = tuple(sorted(
            ledger_before_committed - ledger_after_committed
        ))
        for entity in ordered_entities:
            entity_id = entity.lifecycle.entity_id
            centroid = _centroid(entity)
            record = trial_identities.get(entity_id)
            values = dict(
                frame_id=entity.last_seen_frame_id,
                timestamp=entity.lifecycle.last_timestamp if entity.last_seen_frame_id == frame.frame_id else (
                    record.last_timestamp if record is not None else entity.lifecycle.last_timestamp
                ),
                semantic_probabilities=entity.semantic_probabilities,
                appearance_prototype=entity.image_prototype,
                feature_model_id=entity.feature_model_id,
                lifecycle=entity.lifecycle.lifecycle,
                centroid_xyz=centroid,
                extent_xyz=entity.extent_xyz,
                motion_velocity_xyz=(0.0, 0.0, 0.0),
                motion_uncertainty_m=0.0,
            )
            if record is None:
                inserted = trial_identities.insert(**values)
                if inserted.identity_id != entity_id:
                    raise RuntimeError("identity allocation diverged from runtime entity IDs")
            elif entity.last_seen_frame_id > record.last_frame_id:
                dt = float(frame.timestamp) - record.last_timestamp
                velocity = tuple(
                    (new - old) / dt
                    for new, old in zip(centroid, record.last_centroid_xyz)
                )
                values["motion_velocity_xyz"] = velocity
                trial_identities.update(entity_id, **values)
            elif record.lifecycle is not entity.lifecycle.lifecycle:
                trial_identities._records[entity_id] = replace(
                    record, lifecycle=entity.lifecycle.lifecycle
                )

        for record in trial_identities.records:
            belief = lifecycle_by_id[record.identity_id]
            if belief.last_frame_id < frame.frame_id:
                belief = advance_lifecycle(
                    belief,
                    TemporalEvidence(
                        TemporalEvidenceKind.OUT_OF_VIEW, 0.0,
                        frame.frame_id, float(frame.timestamp), None,
                    ),
                    self.config.lifecycle,
                )
                lifecycle_by_id[record.identity_id] = belief
            if record.lifecycle is not belief.lifecycle:
                trial_identities._records[record.identity_id] = replace(
                    record, lifecycle=belief.lifecycle
                )
        if trial_identities._next_identity_id != next_entity_id:
            raise RuntimeError("identity allocator and runtime next ID diverged")

        timestamp_ns = timestamp_seconds_to_ns(frame.timestamp)
        old_export = {item.entity_id: item for item in current.export_tracker.entries}
        export_entries: list[TemporalExportTrackerEntry] = []
        samples: list[TemporalExportSample] = []
        events: list[TemporalLifecycleEvent] = []
        assert self.config.dynamic_state is not None
        for entity in ordered_entities:
            entity_id = entity.lifecycle.entity_id
            epoch = trial_geometry.current(entity_id)
            previous = old_export.get(entity_id)
            observed = entity.last_seen_frame_id == frame.frame_id
            if previous is None:
                count = 1 if observed else 0
                dynamic = DynamicEvidenceState.static()
            else:
                count = previous.observation_count + (1 if observed else 0)
                displacement, confidence = motion_by_entity.get(entity_id, (0.0, 0.0))
                dynamic = (
                    advance_dynamic_state(
                        previous.dynamic_evidence,
                        accepted_motion=observed and confidence > 0.0,
                        displacement_m=displacement,
                        confidence=confidence,
                        config=self.config.dynamic_state,
                    )
                    if observed else previous.dynamic_evidence
                )
            centroid = _centroid(entity)
            export_entries.append(
                TemporalExportTrackerEntry(
                    entity_id, count, dynamic,
                    centroid if observed else (previous.last_centroid_xyz if previous else None),
                    epoch.readout_valid, epoch.epoch_id,
                )
            )
            if observed:
                _, confidence = motion_by_entity.get(entity_id, (0.0, 0.0))
                samples.append(
                    TemporalExportSample(
                        frame.frame_id, timestamp_ns, entity_id, centroid, count,
                        dynamic.dynamic_state, confidence, epoch.epoch_id,
                        epoch.readout_valid,
                    )
                )
            old_lifecycle = old_lifecycle_by_entity.get(entity_id)
            old_epoch = None
            try:
                old_epoch = current.geometry.current(entity_id)
            except KeyError:
                pass
            lifecycle_changed = (
                old_lifecycle is not None
                and old_lifecycle.lifecycle is not entity.lifecycle.lifecycle
            )
            readout_changed = old_epoch is not None and old_epoch.readout_valid is not epoch.readout_valid
            if lifecycle_changed or readout_changed:
                assert old_lifecycle is not None
                events.append(
                    TemporalLifecycleEvent(
                        frame.frame_id, timestamp_ns, entity_id,
                        old_lifecycle.lifecycle, entity.lifecycle.lifecycle,
                        evidence_by_entity[entity_id].kind, epoch.epoch_id,
                        epoch.readout_valid,
                    )
                )
        wrapper_ids = {item.lifecycle.entity_id for item in ordered_entities}
        for record in trial_identities.records:
            entity_id = record.identity_id
            if entity_id in wrapper_ids:
                continue
            previous = old_export.get(entity_id)
            belief = lifecycle_by_id[entity_id]
            epoch_id = max(
                0,
                trial_geometry.next_epoch_id(entity_id) - 1,
                previous.geometry_epoch if previous is not None else 0,
            )
            export_entries.append(
                TemporalExportTrackerEntry(
                    entity_id=entity_id,
                    observation_count=(
                        0 if previous is None else previous.observation_count
                    ),
                    dynamic_evidence=(
                        DynamicEvidenceState.static()
                        if previous is None else previous.dynamic_evidence
                    ),
                    last_centroid_xyz=record.last_centroid_xyz,
                    readout_valid=False,
                    geometry_epoch=epoch_id,
                )
            )
            old_lifecycle = old_lifecycle_by_entity.get(entity_id)
            if (
                entity_id in reclaimed_ids
                or (
                    old_lifecycle is not None
                    and old_lifecycle.lifecycle is not belief.lifecycle
                )
            ):
                before = (
                    belief.lifecycle
                    if old_lifecycle is None else old_lifecycle.lifecycle
                )
                events.append(
                    TemporalLifecycleEvent(
                        frame.frame_id, timestamp_ns, entity_id,
                        before, belief.lifecycle,
                        TemporalEvidenceKind.OUT_OF_VIEW,
                        epoch_id, False,
                    )
                )
        export = TemporalExportBatch(
            frame.frame_id, timestamp_ns, tuple(samples), tuple(events)
        )
        export_tracker = TemporalExportTracker(tuple(export_entries), export)
        motion_records = tuple(
            f"motion:{frame.frame_id}:{observation_id}:{entity_id}"
            for observation_id, entity_id, decision in motion_events
            if decision is MotionDecision.REJECTED
        )
        icp_records = tuple(
            f"motion:{frame.frame_id}:{observation_id}:{entity_id}"
            for observation_id, entity_id, _decision in motion_events
        ) if (
            self.config.execution_profile is ExecutionProfile.A4
            and self.config.icp_enabled
        ) else ()
        icp_accept_records = tuple(
            f"motion:{frame.frame_id}:{observation_id}:{entity_id}"
            for observation_id, entity_id, decision in motion_events
            if decision is MotionDecision.ICP_ACCEPTED
        ) if icp_records else ()
        icp_reject_records = tuple(record for record in icp_records if record not in set(icp_accept_records))
        reid_opportunity_records = tuple(
            f"reid:{frame.frame_id}:{observation_id}:{entity_id}"
            for observation_id, entity_id in association.reid_opportunity_pairs
        )
        reid_trigger_records = tuple(
            f"reid:{frame.frame_id}:{observation_id}:{entity_id}"
            for observation_id, entity_id in association.reid_trigger_pairs
        )
        epoch_reset_records = motion_records
        mechanism_records = {
            "proposal_opportunity_count": proposal_opportunity_records,
            "proposal_trigger_count": proposal_trigger_records,
            "reid_opportunity_count": reid_opportunity_records,
            "reid_trigger_count": reid_trigger_records,
            "identity_expiry_count": tuple(f"identity:{item}" for item in sorted(expired_ids)),
            "geometry_reclaim_count": tuple(sorted(geometry_reclaim_records)),
            "motion_rejection_count": motion_records,
            "ledger_rejection_count": (),
            "epoch_reset_opportunity_count": epoch_reset_records,
            "epoch_reset_trigger_count": tuple(epoch_reset_trigger_records),
            "icp_opportunity_count": icp_records,
            "icp_accept_count": icp_accept_records,
            "icp_reject_count": icp_reject_records,
            "ledger_stage_count": ledger_stage_records,
            "ledger_commit_count": ledger_commit_records,
            "ledger_reclaim_count": ledger_reclaim_records,
        }
        frame_diagnostics = TemporalFrameDiagnostics(
            frame_id=frame.frame_id,
            proposal_opportunity_count=proposal_opportunities,
            proposal_trigger_count=proposal_triggers,
            reid_opportunity_count=association.reid_opportunity_count,
            reid_trigger_count=association.reid_trigger_count,
            epoch_reset_opportunity_count=sum(
                item is MotionDecision.REJECTED for _, _, item in motion_events
            ),
            epoch_reset_trigger_count=epoch_reset_triggers,
            icp_opportunity_count=(
                len(motion_events)
                if self.config.execution_profile is ExecutionProfile.A4
                and self.config.icp_enabled else 0
            ),
            icp_accept_count=(
                sum(item is MotionDecision.ICP_ACCEPTED for _, _, item in motion_events)
                if self.config.execution_profile is ExecutionProfile.A4
                and self.config.icp_enabled else 0
            ),
            icp_reject_count=(
                sum(item is not MotionDecision.ICP_ACCEPTED for _, _, item in motion_events)
                if self.config.execution_profile is ExecutionProfile.A4
                and self.config.icp_enabled else 0
            ),
            motion_rejection_count=sum(
                item is MotionDecision.REJECTED for _, _, item in motion_events
            ),
            ledger_stage_count=len(ledger_stage_records),
            ledger_commit_count=len(ledger_commit_records),
            ledger_reclaim_count=len(ledger_reclaim_records),
            ledger_rejection_count=sum(
                item in (
                    LedgerDecision.REJECTED_CAPACITY,
                    LedgerDecision.REJECTED_INTEGRATION,
                )
                for item in ledger_decisions
            ),
            identity_expiry_count=len(expired_ids),
            geometry_reclaim_count=len(reclaimed_ids),
            mechanism_records=tuple(
                (name, mechanism_records[name])
                for name in TEMPORAL_MECHANISM_RECORD_KEYS
            ),
        )
        diagnostics = replace(
            current.diagnostics,
            processed_frame_count=current.diagnostics.processed_frame_count + 1,
            proposal_opportunity_count=current.diagnostics.proposal_opportunity_count + proposal_opportunities,
            proposal_trigger_count=current.diagnostics.proposal_trigger_count + proposal_triggers,
            reid_opportunity_count=current.diagnostics.reid_opportunity_count + association.reid_opportunity_count,
            reid_trigger_count=current.diagnostics.reid_trigger_count + association.reid_trigger_count,
            identity_expiry_count=(
                current.diagnostics.identity_expiry_count + len(expired_ids)
            ),
            geometry_reclaim_count=(
                current.diagnostics.geometry_reclaim_count + len(reclaimed_ids)
            ),
            motion_rejection_count=(
                current.diagnostics.motion_rejection_count
                + frame_diagnostics.motion_rejection_count
            ),
            ledger_rejection_count=(
                current.diagnostics.ledger_rejection_count
                + frame_diagnostics.ledger_rejection_count
            ),
            epoch_reset_opportunity_count=(
                current.diagnostics.epoch_reset_opportunity_count
                + frame_diagnostics.epoch_reset_opportunity_count
            ),
            epoch_reset_trigger_count=(
                current.diagnostics.epoch_reset_trigger_count
                + frame_diagnostics.epoch_reset_trigger_count
            ),
            icp_opportunity_count=(
                current.diagnostics.icp_opportunity_count
                + frame_diagnostics.icp_opportunity_count
            ),
            icp_accept_count=(
                current.diagnostics.icp_accept_count
                + frame_diagnostics.icp_accept_count
            ),
            icp_reject_count=(
                current.diagnostics.icp_reject_count
                + frame_diagnostics.icp_reject_count
            ),
            ledger_stage_count=(
                current.diagnostics.ledger_stage_count
                + frame_diagnostics.ledger_stage_count
            ),
            ledger_commit_count=(
                current.diagnostics.ledger_commit_count
                + frame_diagnostics.ledger_commit_count
            ),
            ledger_reclaim_count=(
                current.diagnostics.ledger_reclaim_count
                + frame_diagnostics.ledger_reclaim_count
            ),
            last_frame=frame_diagnostics,
            assignment_diagnostics=association.assignment_diagnostics,
        )
        next_state = TemporalRuntimeState._adopt_owned(
            scene_id=current.scene_id,
            revision=current.revision + 1,
            last_frame_id=frame.frame_id,
            last_timestamp=float(frame.timestamp),
            next_entity_id=trial_identities._next_identity_id,
            entities=ordered_entities,
            background=trial_background,
            tracker=trial_tracker,
            identities=trial_identities,
            geometry=trial_geometry,
            lifecycle_beliefs=tuple(
                lifecycle_by_id[key] for key in sorted(lifecycle_by_id)
                if key not in expired_ids
            ),
            background_ledger=trial_ledger,
            background_mode=current.background_mode,
            export_tracker=export_tracker,
            diagnostics=diagnostics,
        )
        result = TemporalFrameResult(
            frame_id=frame.frame_id,
            revision=next_state.revision,
            active_entity_ids=tuple(
                entity.lifecycle.entity_id
                for entity in ordered_entities
                if entity.lifecycle.lifecycle in (TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN)
            ),
            dormant_entity_ids=tuple(
                entity.lifecycle.entity_id
                for entity in ordered_entities
                if entity.lifecycle.lifecycle is TemporalLifecycle.DORMANT
            ),
            new_entity_ids=tuple(sorted(new_ids)),
            reactivated_entity_ids=tuple(sorted(reactivated)),
            background_blocks_touched=background_blocks_touched,
            export=export,
            proposal_opportunity_count=proposal_opportunities,
            proposal_trigger_count=proposal_triggers,
            reid_opportunity_count=association.reid_opportunity_count,
            reid_trigger_count=association.reid_trigger_count,
            expired_identity_ids=tuple(sorted(expired_ids)),
            diagnostics=frame_diagnostics,
        )
        self._before_publish(next_state)
        self.state = next_state
        return result
