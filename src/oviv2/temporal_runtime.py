from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

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
from src.oviv2.temporal_background import (
    TemporalBackgroundVolume,
    build_background_depth,
)
from src.oviv2.temporal_config import (
    ExecutionProfile,
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
from src.oviv2.temporal_identity import IdentityMemoryBank
from src.oviv2.temporal_proposals import (
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
from src.oviv2.temporal_state import (
    TemporalDiagnostics,
    TemporalEntityState,
    TemporalExportTracker,
    TemporalExportTrackerEntry,
    TemporalGeometryState,
    TemporalRuntimeState,
)
from src.oviv2.tracking import LocalTracker, LocalTrackerConfig


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
            raise ValueError("export frame must match result")
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


def _centroid(entity: TemporalEntityState) -> tuple[float, float, float]:
    points = entity.submap.world_points(entity.object_to_world)
    if points.shape[0]:
        mean = points.mean(axis=0, dtype=np.float64)
        return tuple(float(value) for value in mean)
    return tuple(float(value) for value in entity.object_to_world[:3, 3])


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
    maximum = config.geometry.maximum_visibility_points_per_entity
    points = entity.submap.world_points(entity.object_to_world)[:maximum]
    kind = TemporalEvidenceKind.OUT_OF_VIEW
    strength = 0.0
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
    return TemporalEvidence(
        kind=kind,
        strength=strength,
        frame_id=frame.frame_id,
        timestamp=float(frame.timestamp),
        view_bin=_view_bin(frame, _centroid(entity), config)
        if kind is TemporalEvidenceKind.VISIBLE_ABSENT
        else None,
    )


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
    if observation_id > np.iinfo(np.int64).max:
        raise OverflowError("recovered proposal observation ID exceeds int64")
    return FrameObservation(
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


def _motion_confidence(motion: object, config: TemporalReadoutConfig) -> float:
    if motion.decision is MotionDecision.REJECTED:
        return 0.0
    if motion.decision is MotionDecision.ICP_ACCEPTED:
        return float(motion.fitness)
    assert config.motion is not None
    residual_limit = float(config.motion.maximum_translation_residual_m)
    confidence = max(0.0, min(1.0, 1.0 - float(motion.rmse_m) / residual_limit))
    return confidence if confidence >= config.motion.minimum_translation_confidence else 0.0


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
        ledger = ReversibleBackgroundLedger(
            self.config.geometry, self.config.background_ledger
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
            export_tracker=TemporalExportTracker(),
            diagnostics=TemporalDiagnostics(),
        )

    def _before_publish(self, next_state: TemporalRuntimeState) -> None:
        del next_state

    def process_frame(
        self,
        frame: Frame,
        observations: tuple[FrameObservation, ...],
        dense_semantics: DenseSemanticFrame | None = None,
        *,
        proposal_evidence: ProposalRecoveryInput | None = None,
    ) -> TemporalFrameResult:
        current = self.state
        frame, observations, _ = _validate_inputs(
            frame, observations, dense_semantics, current
        )
        trial_identities = current._mutable_identities_snapshot()
        trial_geometry = current.geometry
        trial_ledger = current._mutable_ledger_snapshot()
        proposal_opportunities = proposal_triggers = 0
        if proposal_evidence is not None:
            if not isinstance(proposal_evidence, ProposalRecoveryInput):
                raise TypeError("proposal_evidence must be ProposalRecoveryInput or None")
            if proposal_evidence.frame_id != frame.frame_id or proposal_evidence.timestamp != frame.timestamp:
                raise ValueError("proposal evidence must match the current frame")
            assert self.config.proposal is not None
            recovery = recover_temporal_proposals(proposal_evidence, self.config.proposal)
            proposal_opportunities = recovery.opportunity_count
            proposal_triggers = recovery.trigger_count
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
        allow_dormant_reid = self.config.execution_profile is ExecutionProfile.A4
        association_entities = tuple(
            entity
            for entity in current.entities
            if allow_dormant_reid
            or entity.lifecycle.lifecycle is not TemporalLifecycle.DORMANT
        )
        excluded_dormant = tuple(
            entity
            for entity in current.entities
            if entity.lifecycle.lifecycle is TemporalLifecycle.DORMANT
            and not allow_dormant_reid
        )
        targets = tuple(_association_target(entity) for entity in association_entities)
        if allow_dormant_reid:
            association = associate_temporal_observations(
                confirmed, targets, self.config.association,
                dormant_reid=self.config.identity,
            )
        else:
            association = associate_temporal_observations(
                confirmed, targets, self.config.association
            )
        observations_by_id = {item.observation_id: item for item in confirmed}
        entities_by_id = {item.lifecycle.entity_id: item for item in current.entities}
        next_entities: dict[int, TemporalEntityState] = {}
        reactivated: list[int] = []
        forced_new_observation_ids: list[int] = []
        motion_by_entity: dict[int, tuple[float, float]] = {}
        evidence_by_entity: dict[int, TemporalEvidenceKind] = {}
        old_lifecycle_by_entity = {
            item.lifecycle.entity_id: item.lifecycle for item in current.entities
        }
        diagnostic_by_pair = {
            (item.observation_id, item.entity_id): item
            for item in association.assignment_diagnostics
        }

        for old in excluded_dormant:
            absence = _absence_evidence(old, frame, self.config)
            lifecycle = advance_lifecycle(
                old.lifecycle,
                absence,
                self.config.lifecycle,
            )
            evidence_by_entity[old.lifecycle.entity_id] = absence.kind
            trial_geometry = trial_geometry.replace_current(
                trial_geometry.current(old.lifecycle.entity_id).apply_evidence(absence.kind)
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
            old = entities_by_id[entity_id]
            points = backproject_observation(frame, observation, self.config.geometry)
            motion_estimator = (
                estimate_object_motion
                if self.config.execution_profile is ExecutionProfile.A4
                else estimate_object_translation
            )
            motion = motion_estimator(
                old.submap,
                points,
                observation.centroid_xyz,
                self.config.geometry,
                previous_object_to_world=old.object_to_world,
            )
            epoch = trial_geometry.current(entity_id)
            diagnostic = diagnostic_by_pair[(observation_id, entity_id)]
            if motion.decision is MotionDecision.REJECTED:
                if not diagnostic.high_confidence_identity_match:
                    forced_new_observation_ids.append(observation_id)
                    evidence_by_entity[entity_id] = TemporalEvidenceKind.OCCLUDED
                    lifecycle = advance_lifecycle(
                        old.lifecycle,
                        TemporalEvidence(
                            TemporalEvidenceKind.OCCLUDED, 0.0, frame.frame_id,
                            float(frame.timestamp), None,
                        ),
                        self.config.lifecycle,
                    )
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
                trial_geometry = trial_geometry.append(epoch)
            else:
                epoch = epoch.integrate(
                    motion, points, frame.frame_id, self.config.geometry
                )
                trial_geometry = trial_geometry.replace_current(epoch)
            submap = epoch.submap
            motion_confidence = _motion_confidence(motion, self.config)
            displacement = float(
                np.linalg.norm(
                    np.asarray(epoch.object_to_world[:3, 3], dtype=np.float64)
                    - np.asarray(old.object_to_world[:3, 3], dtype=np.float64)
                )
            )
            motion_by_entity[entity_id] = (displacement, motion_confidence)
            evidence_by_entity[entity_id] = TemporalEvidenceKind.PRESENT
            lifecycle = advance_lifecycle(
                old.lifecycle,
                TemporalEvidence(
                    TemporalEvidenceKind.PRESENT,
                    float(observation.confidence),
                    frame.frame_id,
                    float(frame.timestamp),
                    None,
                ),
                self.config.lifecycle,
            )
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

        for entity_id in association.unmatched_entity_ids:
            old = entities_by_id[entity_id]
            absence = _absence_evidence(old, frame, self.config)
            lifecycle = advance_lifecycle(
                old.lifecycle,
                absence,
                self.config.lifecycle,
            )
            evidence_by_entity[entity_id] = absence.kind
            trial_geometry = trial_geometry.replace_current(
                trial_geometry.current(entity_id).apply_evidence(absence.kind)
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

        next_entity_id = current.next_entity_id
        new_ids: list[int] = []
        for observation_id in tuple(sorted((*association.unmatched_observation_ids, *forced_new_observation_ids))):
            observation = observations_by_id[observation_id]
            points = backproject_observation(frame, observation, self.config.geometry)
            if points.shape[0] == 0:
                continue
            if len(next_entities) >= self.config.geometry.maximum_entities:
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
                    seed,
                    TemporalEvidence(
                        TemporalEvidenceKind.PRESENT,
                        float(observation.confidence),
                        frame.frame_id,
                        float(frame.timestamp),
                        None,
                    ),
                    self.config.lifecycle,
                )
            else:
                lifecycle = _initial_lifecycle(
                    entity_id, frame, float(observation.confidence), self.config
                )
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
            trial_geometry = trial_geometry.append(
                GeometryEpoch(
                    entity_id, 0, pose, submap, True, None, frame.frame_id
                )
            )
            evidence_by_entity[entity_id] = TemporalEvidenceKind.PRESENT
            motion_by_entity[entity_id] = (0.0, 0.0)
            new_ids.append(entity_id)
            next_entity_id += 1

        ordered_entities = tuple(next_entities[key] for key in sorted(next_entities))
        if self.config.execution_profile is ExecutionProfile.A2:
            trial_background = current._mutable_background_snapshot()
            if trial_background.active_block_count:
                raise RuntimeError("A2 temporal background must remain empty")
            trial_background._last_blocks_touched = 0
            background_blocks_touched = 0
        else:
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
            for entity in ordered_entities:
                entity_id = entity.lifecycle.entity_id
                kind = evidence_by_entity.get(entity_id)
                if kind not in (
                    TemporalEvidenceKind.PRESENT,
                    TemporalEvidenceKind.OCCLUDED,
                    TemporalEvidenceKind.VISIBLE_ABSENT,
                ):
                    continue
                epoch = trial_geometry.current(entity_id)
                if kind is TemporalEvidenceKind.VISIBLE_ABSENT:
                    block_keys = trial_ledger.committed_volume.candidate_block_keys(
                        frame, background_depth.depth_m
                    )
                    if not block_keys:
                        continue
                    evidence = BackgroundLedgerEvidence(
                        entity_id=entity_id,
                        geometry_epoch=epoch.epoch_id,
                        frame_id=frame.frame_id,
                        timestamp=float(frame.timestamp),
                        kind=kind,
                        view_bin=_view_bin(frame, _centroid(entity), self.config),
                        contributions=tuple(BackgroundContribution(key) for key in block_keys),
                        frame=frame,
                        depth_m=background_depth.depth_m,
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
                decision = trial_ledger.stage(evidence)
                if decision in (
                    LedgerDecision.REJECTED_CAPACITY,
                    LedgerDecision.REJECTED_INTEGRATION,
                ):
                    raise RuntimeError(f"background ledger rejected frame: {decision.value}")
            trial_background = trial_ledger.committed_volume
            background_blocks_touched = trial_background.last_blocks_touched
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
                        evidence_by_entity[entity_id], epoch.epoch_id,
                        epoch.readout_valid,
                    )
                )
        export = TemporalExportBatch(
            frame.frame_id, timestamp_ns, tuple(samples), tuple(events)
        )
        export_tracker = TemporalExportTracker(tuple(export_entries), export)
        diagnostics = replace(
            current.diagnostics,
            processed_frame_count=current.diagnostics.processed_frame_count + 1,
            proposal_opportunity_count=current.diagnostics.proposal_opportunity_count + proposal_opportunities,
            proposal_trigger_count=current.diagnostics.proposal_trigger_count + proposal_triggers,
            reid_opportunity_count=current.diagnostics.reid_opportunity_count + association.reid_opportunity_count,
            reid_trigger_count=current.diagnostics.reid_trigger_count + association.reid_trigger_count,
            motion_rejection_count=current.diagnostics.motion_rejection_count + sum(
                1 for observation_id, entity_id in association.assignments
                if observation_id in forced_new_observation_ids
                or trial_geometry.current(entity_id).motion_decision is MotionDecision.REJECTED
            ),
            assignment_diagnostics=association.assignment_diagnostics,
        )
        next_state = TemporalRuntimeState._adopt_owned(
            scene_id=current.scene_id,
            revision=current.revision + 1,
            last_frame_id=frame.frame_id,
            last_timestamp=float(frame.timestamp),
            next_entity_id=next_entity_id,
            entities=ordered_entities,
            background=trial_background,
            tracker=trial_tracker,
            identities=trial_identities,
            geometry=trial_geometry,
            lifecycle_beliefs=tuple(entity.lifecycle for entity in ordered_entities),
            background_ledger=trial_ledger,
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
        )
        self._before_publish(next_state)
        self.state = next_state
        return result
