from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.addressing import VoxelKey
from src.oviv2.entities import PersistentEntity
from src.oviv2.runtime import Oviv2Runtime, RuntimeFrameResult
from src.oviv2.temporal_config import ExecutionProfile, TemporalReadoutConfig
from src.oviv2.temporal_export import (
    DynamicEvidenceState,
    TemporalExportBatch,
    TemporalExportSample,
    TemporalLifecycleEvent,
    advance_dynamic_state,
    timestamp_seconds_to_ns,
)
from src.oviv2 import temporal_lifecycle
from src.oviv2.temporal_lifecycle import (
    TemporalEvidence,
    TemporalEvidenceKind,
    TemporalLifecycle,
    TemporalLifecycleState,
)
from src.oviv2.visibility import (
    VisibilityConfig,
    VisibilityStatus,
    VoxelVisibilityProjector,
)


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


@dataclass(frozen=True)
class CumulativeEntityView:
    entity_id: int
    lifecycle_state: str
    voxel_keys: frozenset[VoxelKey]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _integer(self.entity_id, "entity_id", minimum=1))
        if self.lifecycle_state not in {"active", "dormant"}:
            raise ValueError("lifecycle_state must be active or dormant")
        if not isinstance(self.voxel_keys, frozenset):
            raise TypeError("voxel_keys must be a frozenset")
        keys: set[VoxelKey] = set()
        for key in self.voxel_keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise ValueError("voxel_keys must contain integer 3-tuples")
            keys.add(tuple(_integer(item, "voxel key", minimum=-2**63) for item in key))  # type: ignore[arg-type]
        object.__setattr__(self, "voxel_keys", frozenset(keys))
        minimum = _point(self.bounds_min_xyz, "bounds_min_xyz")
        maximum = _point(self.bounds_max_xyz, "bounds_max_xyz")
        if any(lower > upper for lower, upper in zip(minimum, maximum)):
            raise ValueError("bounds_min_xyz must not exceed bounds_max_xyz")
        object.__setattr__(self, "centroid_xyz", _point(self.centroid_xyz, "centroid_xyz"))
        object.__setattr__(self, "bounds_min_xyz", minimum)
        object.__setattr__(self, "bounds_max_xyz", maximum)

    @classmethod
    def from_persistent_entity(cls, entity: PersistentEntity) -> CumulativeEntityView:
        if type(entity) is not PersistentEntity:
            raise TypeError("entity must be an exact PersistentEntity")
        result = object.__new__(cls)
        object.__setattr__(result, "entity_id", entity.entity_id)
        object.__setattr__(result, "lifecycle_state", entity.lifecycle_state)
        object.__setattr__(result, "voxel_keys", entity.voxel_keys)
        object.__setattr__(result, "centroid_xyz", entity.centroid_xyz)
        object.__setattr__(result, "bounds_min_xyz", entity.bounds_min_xyz)
        object.__setattr__(result, "bounds_max_xyz", entity.bounds_max_xyz)
        return result


@dataclass(frozen=True)
class CumulativeReadoutView:
    scene_id: str
    revision: int
    last_frame_id: int
    last_timestamp: float
    voxel_size_m: float
    depth_max_m: float
    visibility_depth_tolerance_m: float
    entities: tuple[CumulativeEntityView, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        object.__setattr__(self, "scene_id", self.scene_id.strip())
        object.__setattr__(self, "revision", _integer(self.revision, "revision", minimum=0))
        object.__setattr__(self, "last_frame_id", _integer(self.last_frame_id, "last_frame_id", minimum=-1))
        object.__setattr__(self, "last_timestamp", _finite(self.last_timestamp, "last_timestamp"))
        for name in ("voxel_size_m", "depth_max_m", "visibility_depth_tolerance_m"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if type(self.entities) is not tuple or any(
            not isinstance(item, CumulativeEntityView) for item in self.entities
        ):
            raise TypeError("entities must be an exact tuple of CumulativeEntityView values")
        ordered = tuple(sorted(self.entities, key=lambda item: item.entity_id))
        if len({item.entity_id for item in ordered}) != len(ordered):
            raise ValueError("entity IDs must be unique")
        object.__setattr__(self, "entities", ordered)

    @classmethod
    def capture(cls, runtime: Oviv2Runtime) -> CumulativeReadoutView:
        if not isinstance(runtime, Oviv2Runtime):
            raise TypeError("runtime must be an Oviv2Runtime")
        return cls(
            scene_id=runtime.scene_id,
            revision=runtime.revision,
            last_frame_id=runtime.last_frame_id,
            last_timestamp=runtime.last_timestamp,
            voxel_size_m=runtime.config.tsdf.voxel_size_m,
            depth_max_m=runtime.config.tsdf.depth_max_m,
            visibility_depth_tolerance_m=runtime.config.visibility_depth_tolerance_m,
            entities=tuple(
                CumulativeEntityView.from_persistent_entity(item)
                for item in sorted(
                    runtime.registry.entities.values(), key=lambda value: value.entity_id
                )
            ),
        )


@dataclass(frozen=True)
class ReferenceReadoutState:
    scene_id: str
    revision: int
    last_frame_id: int
    last_timestamp: float
    entity_lifecycles: tuple[tuple[int, str], ...]
    lifecycle_states: tuple[TemporalLifecycleState, ...]
    cumulative_view: CumulativeReadoutView | None
    export_tracker: tuple[_ReferenceExportTrackerEntry, ...] = ()

    @property
    def latest_cumulative_view(self) -> CumulativeReadoutView | None:
        return self.cumulative_view


@dataclass(frozen=True)
class ReferenceFrameResult:
    scene_id: str
    frame_id: int
    timestamp: float
    revision: int
    active_entity_ids: tuple[int, ...]
    dormant_entity_ids: tuple[int, ...]
    new_entity_ids: tuple[int, ...]
    reactivated_entity_ids: tuple[int, ...]
    background_blocks_touched: int
    entity_lifecycles: tuple[tuple[int, str], ...]
    lifecycle_states: tuple[TemporalLifecycleState, ...]
    cumulative_view: CumulativeReadoutView
    export: TemporalExportBatch | None = None

    @property
    def latest_cumulative_view(self) -> CumulativeReadoutView:
        return self.cumulative_view

    def __post_init__(self) -> None:
        if self.scene_id != self.cumulative_view.scene_id:
            raise ValueError("result scene must match cumulative_view")
        if self.frame_id != self.cumulative_view.last_frame_id:
            raise ValueError("result frame must match cumulative_view")
        if self.revision != self.cumulative_view.revision:
            raise ValueError("result revision must match cumulative_view")
        if self.timestamp != self.cumulative_view.last_timestamp:
            raise ValueError("result timestamp must match cumulative_view")
        if self.export is None:
            object.__setattr__(
                self,
                "export",
                TemporalExportBatch(
                    self.frame_id,
                    timestamp_seconds_to_ns(self.timestamp),
                    (),
                    (),
                ),
            )
        elif type(self.export) is not TemporalExportBatch:
            raise TypeError("export must be a TemporalExportBatch or None")
        assert self.export is not None
        if self.export.frame_index != self.frame_id:
            raise ValueError("export frame must match result")
        if self.export.timestamp_ns != timestamp_seconds_to_ns(self.timestamp):
            raise ValueError("export timestamp must match result")
        for name in (
            "active_entity_ids",
            "dormant_entity_ids",
            "new_entity_ids",
            "reactivated_entity_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be a sorted unique tuple")
        if set(self.active_entity_ids) & set(self.dormant_entity_ids):
            raise ValueError("active and dormant entity IDs must be disjoint")


@dataclass(frozen=True)
class _ReferenceExportTrackerEntry:
    entity_id: int
    observation_count: int
    dynamic_evidence: DynamicEvidenceState
    last_centroid_xyz: tuple[float, float, float] | None
    readout_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _integer(self.entity_id, "entity_id", minimum=1))
        object.__setattr__(
            self,
            "observation_count",
            _integer(self.observation_count, "observation_count", minimum=0),
        )
        if type(self.dynamic_evidence) is not DynamicEvidenceState:
            raise TypeError("dynamic_evidence must be a DynamicEvidenceState")
        if self.last_centroid_xyz is not None:
            object.__setattr__(
                self,
                "last_centroid_xyz",
                _point(self.last_centroid_xyz, "last_centroid_xyz"),
            )
        if (self.observation_count == 0) != (self.last_centroid_xyz is None):
            raise ValueError(
                "last_centroid_xyz is required exactly when observations exist"
            )
        if type(self.readout_valid) is not bool:
            raise TypeError("readout_valid must be an exact bool")


class _BaseReferenceReadout:
    _PROFILE: ExecutionProfile

    def __init__(self, scene_id: str, config: TemporalReadoutConfig) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if not isinstance(config, TemporalReadoutConfig):
            raise TypeError("config must be a TemporalReadoutConfig")
        if config.execution_profile is not self._PROFILE:
            raise ValueError(f"{type(self).__name__} requires {self._PROFILE.name}")
        self.config = config
        self.state = ReferenceReadoutState(
            scene_id=scene_id.strip(),
            revision=0,
            last_frame_id=-1,
            last_timestamp=0.0,
            entity_lifecycles=(),
            lifecycle_states=(),
            export_tracker=(),
            cumulative_view=None,
        )

    def bind_initial_cumulative_view(self, view: CumulativeReadoutView) -> None:
        if not isinstance(view, CumulativeReadoutView):
            raise TypeError("view must be a CumulativeReadoutView")
        if self.state.cumulative_view is not None:
            raise ValueError("initial cumulative view is already bound")
        if (
            view.scene_id != self.state.scene_id
            or view.revision != 0
            or view.last_frame_id != -1
            or self.state.revision != 0
            or self.state.last_frame_id != -1
        ):
            raise ValueError("initial cumulative view progress mismatch")
        self.state = ReferenceReadoutState(
            scene_id=self.state.scene_id,
            revision=0,
            last_frame_id=-1,
            last_timestamp=view.last_timestamp,
            entity_lifecycles=(),
            lifecycle_states=(),
            export_tracker=(),
            cumulative_view=view,
        )

    def transaction_snapshot(self) -> ReferenceReadoutState:
        return self.state

    def restore_transaction(self, snapshot: ReferenceReadoutState) -> None:
        if not isinstance(snapshot, ReferenceReadoutState):
            raise TypeError("snapshot must be a ReferenceReadoutState")
        self.state = snapshot

    def _before_publish(self, next_state: ReferenceReadoutState) -> None:
        del next_state

    def _validate_binding(
        self,
        frame: Frame,
        before: CumulativeReadoutView,
        after: CumulativeReadoutView,
        cumulative_result: RuntimeFrameResult,
    ) -> tuple[int, ...]:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        frame_id = _integer(frame.frame_id, "frame.frame_id", minimum=0)
        timestamp = _finite(frame.timestamp, "frame.timestamp")
        if not isinstance(before, CumulativeReadoutView) or not isinstance(
            after, CumulativeReadoutView
        ):
            raise TypeError("before and after must be CumulativeReadoutView values")
        if not isinstance(cumulative_result, RuntimeFrameResult):
            raise TypeError("cumulative_result must be a RuntimeFrameResult")
        if before.scene_id != self.state.scene_id or after.scene_id != self.state.scene_id:
            raise ValueError("scene binding mismatch")
        if self.state.cumulative_view is None:
            if (before.revision, before.last_frame_id) != (0, -1):
                raise ValueError("initial before progress mismatch")
        elif before != self.state.cumulative_view:
            raise ValueError("before view does not match current readout state")
        if after.revision != before.revision + 1:
            raise ValueError("after revision must equal before revision plus one")
        if before.revision != self.state.revision:
            raise ValueError("before revision does not match readout progress")
        if after.last_frame_id != frame_id or after.last_timestamp != timestamp:
            raise ValueError("after view does not match frame progress")
        if cumulative_result.frame_id != frame_id or cumulative_result.revision != after.revision:
            raise ValueError("cumulative result does not match frame/after progress")
        if frame_id <= before.last_frame_id:
            raise ValueError("frame ID must increase beyond before view")
        if before.last_frame_id >= 0 and timestamp <= before.last_timestamp:
            raise ValueError("timestamp must increase beyond before view")
        accepted = cumulative_result.accepted_entity_ids
        if type(accepted) is not tuple or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or int(value) < 1
            for value in accepted
        ):
            raise ValueError("accepted_entity_ids must contain positive integer IDs")
        normalized = tuple(int(value) for value in accepted)
        if len(set(normalized)) != len(normalized):
            raise ValueError("accepted_entity_ids must be unique")
        after_ids = {item.entity_id for item in after.entities}
        if not set(normalized).issubset(after_ids):
            raise ValueError("accepted_entity_ids must exist in after view")
        return normalized

    @staticmethod
    def _result(
        state: ReferenceReadoutState,
        *,
        new_ids: tuple[int, ...],
        reactivated_ids: tuple[int, ...],
        export: TemporalExportBatch,
    ) -> ReferenceFrameResult:
        assert state.cumulative_view is not None
        active = tuple(
            entity_id
            for entity_id, lifecycle in state.entity_lifecycles
            if lifecycle in {TemporalLifecycle.ACTIVE.value, TemporalLifecycle.UNCERTAIN.value}
        )
        dormant = tuple(
            entity_id
            for entity_id, lifecycle in state.entity_lifecycles
            if lifecycle == TemporalLifecycle.DORMANT.value
        )
        return ReferenceFrameResult(
            scene_id=state.scene_id,
            frame_id=state.last_frame_id,
            timestamp=state.last_timestamp,
            revision=state.revision,
            active_entity_ids=active,
            dormant_entity_ids=dormant,
            new_entity_ids=tuple(sorted(new_ids)),
            reactivated_entity_ids=tuple(sorted(reactivated_ids)),
            background_blocks_touched=0,
            entity_lifecycles=state.entity_lifecycles,
            lifecycle_states=state.lifecycle_states,
            cumulative_view=state.cumulative_view,
            export=export,
        )

    def _advance_export_tracker(
        self,
        *,
        frame: Frame,
        after: CumulativeReadoutView,
        accepted: tuple[int, ...] | frozenset[int],
        events: tuple[TemporalLifecycleEvent, ...] = (),
        readout_valid_by_entity: dict[int, bool] | None = None,
    ) -> tuple[tuple[_ReferenceExportTrackerEntry, ...], TemporalExportBatch]:
        timestamp_ns = timestamp_seconds_to_ns(frame.timestamp)
        accepted_ids = frozenset(accepted)
        old_tracker = {item.entity_id: item for item in self.state.export_tracker}
        after_entities = {item.entity_id: item for item in after.entities}
        tracked_ids = (
            accepted_ids
            if readout_valid_by_entity is None
            else frozenset(readout_valid_by_entity)
        )
        next_tracker = {
            entity_id: item
            for entity_id, item in old_tracker.items()
            if entity_id in after_entities
        }
        samples: list[TemporalExportSample] = []
        dynamic_config = self.config.dynamic_state
        assert dynamic_config is not None

        for entity_id in sorted(tracked_ids):
            centroid = after_entities[entity_id].centroid_xyz
            old = old_tracker.get(entity_id)
            if entity_id in accepted_ids and (
                old is None or old.last_centroid_xyz is None
            ):
                observation_count = 1
                dynamic_evidence = DynamicEvidenceState.static()
                motion_confidence = 0.0
                last_centroid = centroid
            elif entity_id in accepted_ids:
                assert old is not None and old.last_centroid_xyz is not None
                observation_count = old.observation_count + 1
                displacement = float(
                    np.linalg.norm(
                        np.asarray(centroid, dtype=np.float64)
                        - np.asarray(old.last_centroid_xyz, dtype=np.float64)
                    )
                )
                motion_confidence = 1.0
                dynamic_evidence = advance_dynamic_state(
                    old.dynamic_evidence,
                    accepted_motion=True,
                    displacement_m=displacement,
                    confidence=motion_confidence,
                    config=dynamic_config,
                )
                last_centroid = centroid
            elif old is None:
                observation_count = 0
                dynamic_evidence = DynamicEvidenceState.static()
                motion_confidence = 0.0
                last_centroid = None
            else:
                observation_count = old.observation_count
                dynamic_evidence = old.dynamic_evidence
                motion_confidence = 0.0
                last_centroid = old.last_centroid_xyz
            readout_valid = (
                True
                if readout_valid_by_entity is None
                else readout_valid_by_entity[entity_id]
            )
            next_tracker[entity_id] = _ReferenceExportTrackerEntry(
                entity_id=entity_id,
                observation_count=observation_count,
                dynamic_evidence=dynamic_evidence,
                last_centroid_xyz=last_centroid,
                readout_valid=readout_valid,
            )
            if entity_id in accepted_ids:
                samples.append(
                    TemporalExportSample(
                        frame_index=frame.frame_id,
                        timestamp_ns=timestamp_ns,
                        entity_id=entity_id,
                        centroid_xyz=centroid,
                        observation_count=observation_count,
                        dynamic_state=dynamic_evidence.dynamic_state,
                        motion_confidence=motion_confidence,
                        geometry_epoch=0,
                        readout_valid=readout_valid,
                    )
                )

        tracker = tuple(sorted(next_tracker.values(), key=lambda item: item.entity_id))
        export = TemporalExportBatch(
            frame_index=frame.frame_id,
            timestamp_ns=timestamp_ns,
            samples=tuple(samples),
            events=events,
        )
        return tracker, export


class ReferenceCurrentReadout(_BaseReferenceReadout):
    _PROFILE = ExecutionProfile.A0

    def process_cumulative_frame(
        self,
        frame: Frame,
        *,
        before: CumulativeReadoutView,
        after: CumulativeReadoutView,
        cumulative_result: RuntimeFrameResult,
    ) -> ReferenceFrameResult:
        accepted = self._validate_binding(frame, before, after, cumulative_result)
        before_ids = {item.entity_id for item in before.entities}
        lifecycles = tuple(
            (item.entity_id, item.lifecycle_state) for item in after.entities
        )
        export_tracker, export = self._advance_export_tracker(
            frame=frame,
            after=after,
            accepted=accepted,
        )
        next_state = ReferenceReadoutState(
            scene_id=self.state.scene_id,
            revision=after.revision,
            last_frame_id=after.last_frame_id,
            last_timestamp=after.last_timestamp,
            entity_lifecycles=lifecycles,
            lifecycle_states=(),
            export_tracker=export_tracker,
            cumulative_view=after,
        )
        result = self._result(
            next_state,
            new_ids=tuple(item.entity_id for item in after.entities if item.entity_id not in before_ids),
            reactivated_ids=(),
            export=export,
        )
        self._before_publish(next_state)
        self.state = next_state
        return result


class LifecycleOverlayReadout(_BaseReferenceReadout):
    _PROFILE = ExecutionProfile.A1

    @staticmethod
    def _view_bin(frame: Frame, centroid: tuple[float, float, float], config: TemporalReadoutConfig) -> int:
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
            max(0, math.floor((elevation + math.pi / 2.0) / math.pi * lifecycle.view_bin_elevation_count)),
        )
        return elevation_index * lifecycle.view_bin_azimuth_count + azimuth_index

    def _absence_evidence(
        self, entity: CumulativeEntityView, frame: Frame, before: CumulativeReadoutView
    ) -> TemporalEvidence:
        projector = VoxelVisibilityProjector(
            VisibilityConfig(
                voxel_size_m=before.voxel_size_m,
                depth_tolerance_m=before.visibility_depth_tolerance_m,
            )
        )
        depth = np.asarray(frame.depth)
        clean_depth = np.array(depth, dtype=np.float64, copy=True)
        valid = np.isfinite(clean_depth) & (clean_depth > 0.0) & (
            clean_depth <= before.depth_max_m
        )
        clean_depth[~valid] = 0.0
        visibility_frame = Frame(
            frame_id=frame.frame_id,
            rgb=frame.rgb,
            depth=clean_depth,
            pose=frame.pose,
            intrinsics=frame.intrinsics,
            timestamp=frame.timestamp,
            source_frame_id=frame.source_frame_id,
        )
        grouped = projector.classify_many(tuple(entity.voxel_keys), visibility_frame)
        absent_count = len(grouped[VisibilityStatus.ABSENT])
        valid_count = sum(
            len(grouped[status])
            for status in (
                VisibilityStatus.PRESENT,
                VisibilityStatus.ABSENT,
                VisibilityStatus.OCCLUDED,
            )
        )
        fraction = absent_count / valid_count if valid_count else 0.0
        lifecycle = self.config.lifecycle
        if (
            absent_count >= lifecycle.minimum_visible_pixel_count
            and fraction >= lifecycle.minimum_visible_fraction
        ):
            kind = TemporalEvidenceKind.VISIBLE_ABSENT
            strength = fraction
            view_bin = self._view_bin(frame, entity.centroid_xyz, self.config)
        elif grouped[VisibilityStatus.OCCLUDED]:
            kind = TemporalEvidenceKind.OCCLUDED
            strength = 0.0
            view_bin = None
        elif grouped[VisibilityStatus.UNOBSERVED]:
            kind = TemporalEvidenceKind.DEPTH_UNKNOWN
            strength = 0.0
            view_bin = None
        else:
            kind = TemporalEvidenceKind.OCCLUDED
            strength = 0.0
            view_bin = None
        return TemporalEvidence(kind, strength, frame.frame_id, float(frame.timestamp), view_bin)

    def _initial_state(self, entity_id: int, frame: Frame, *, present: bool) -> TemporalLifecycleState:
        lifecycle = self.config.lifecycle
        log_odds = lifecycle.initial_log_odds + (
            lifecycle.present_log_likelihood if present else 0.0
        )
        log_odds = max(-lifecycle.log_odds_limit, min(lifecycle.log_odds_limit, log_odds))
        probability = 1.0 / (1.0 + math.exp(-log_odds))
        state = (
            TemporalLifecycle.ACTIVE
            if present and probability >= lifecycle.active_on_probability
            else TemporalLifecycle.UNCERTAIN
        )
        return TemporalLifecycleState(
            entity_id=entity_id,
            lifecycle=state,
            existence_log_odds=log_odds,
            last_frame_id=frame.frame_id,
            last_timestamp=float(frame.timestamp),
            absent_streak=0,
            absence_view_bins=(),
        )

    def process_cumulative_frame(
        self,
        frame: Frame,
        *,
        before: CumulativeReadoutView,
        after: CumulativeReadoutView,
        cumulative_result: RuntimeFrameResult,
    ) -> ReferenceFrameResult:
        accepted = frozenset(self._validate_binding(frame, before, after, cumulative_result))
        old_states = {item.entity_id: item for item in self.state.lifecycle_states}
        before_entities = {item.entity_id: item for item in before.entities}
        after_ids = {item.entity_id for item in after.entities}
        if set(old_states) != set(before_entities):
            if self.state.cumulative_view is not None:
                raise ValueError("lifecycle IDs do not match before view IDs")

        next_states: list[TemporalLifecycleState] = []
        old_export_tracker = {
            item.entity_id: item for item in self.state.export_tracker
        }
        readout_valid_by_entity: dict[int, bool] = {}
        lifecycle_events: list[
            tuple[
                TemporalLifecycleState,
                TemporalLifecycleState,
                TemporalEvidenceKind,
                bool,
            ]
        ] = []
        new_ids: list[int] = []
        reactivated_ids: list[int] = []
        for entity_id in sorted(after_ids):
            old = old_states.get(entity_id)
            if old is None:
                next_states.append(self._initial_state(entity_id, frame, present=entity_id in accepted))
                readout_valid_by_entity[entity_id] = True
                new_ids.append(entity_id)
                continue
            old_readout_valid = old_export_tracker.get(entity_id)
            previous_readout_valid = (
                True
                if old_readout_valid is None
                else old_readout_valid.readout_valid
            )
            if entity_id in accepted:
                evidence = TemporalEvidence(
                    TemporalEvidenceKind.PRESENT,
                    1.0,
                    frame.frame_id,
                    float(frame.timestamp),
                    None,
                )
            else:
                evidence = self._absence_evidence(before_entities[entity_id], frame, before)
            updated = temporal_lifecycle.advance_lifecycle(old, evidence, self.config.lifecycle)
            if evidence.kind is TemporalEvidenceKind.PRESENT:
                readout_valid = True
            elif evidence.kind is TemporalEvidenceKind.VISIBLE_ABSENT:
                readout_valid = False
            else:
                readout_valid = previous_readout_valid
            readout_valid_by_entity[entity_id] = readout_valid
            if (
                old.lifecycle is not updated.lifecycle
                or previous_readout_valid is not readout_valid
            ):
                lifecycle_events.append(
                    (old, updated, evidence.kind, readout_valid)
                )
            if old.lifecycle is TemporalLifecycle.DORMANT and updated.lifecycle is TemporalLifecycle.ACTIVE:
                reactivated_ids.append(entity_id)
            next_states.append(updated)

        lifecycle_states = tuple(next_states)
        lifecycles = tuple(
            (item.entity_id, item.lifecycle.value) for item in lifecycle_states
        )
        timestamp_ns = timestamp_seconds_to_ns(frame.timestamp)
        events = tuple(
            TemporalLifecycleEvent(
                frame_index=frame.frame_id,
                timestamp_ns=timestamp_ns,
                entity_id=old.entity_id,
                before=old.lifecycle,
                after=updated.lifecycle,
                evidence=evidence_kind,
                geometry_epoch=0,
                readout_valid=readout_valid,
            )
            for old, updated, evidence_kind, readout_valid in lifecycle_events
        )
        export_tracker, export = self._advance_export_tracker(
            frame=frame,
            after=after,
            accepted=accepted,
            events=events,
            readout_valid_by_entity=readout_valid_by_entity,
        )
        next_state = ReferenceReadoutState(
            scene_id=self.state.scene_id,
            revision=after.revision,
            last_frame_id=after.last_frame_id,
            last_timestamp=after.last_timestamp,
            entity_lifecycles=lifecycles,
            lifecycle_states=lifecycle_states,
            export_tracker=export_tracker,
            cumulative_view=after,
        )
        result = self._result(
            next_state,
            new_ids=tuple(new_ids),
            reactivated_ids=tuple(reactivated_ids),
            export=export,
        )
        self._before_publish(next_state)
        self.state = next_state
        return result
