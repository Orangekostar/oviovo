from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_association import TemporalAssignmentDiagnostic
from src.oviv2.temporal_background_ledger import ReversibleBackgroundLedger
from src.oviv2.temporal_config import (
    TemporalBackgroundLedgerConfig,
    TemporalIdentityConfig,
)
from src.oviv2.temporal_epoch import GeometryEpoch
from src.oviv2.temporal_export import DynamicEvidenceState, TemporalExportBatch
from src.oviv2.temporal_geometry import ObjectSubmap
from src.oviv2.temporal_identity import IdentityMemoryBank
from src.oviv2.temporal_lifecycle import TemporalLifecycleState
from src.oviv2.tracking import LocalTracker


def _readonly_float_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise TypeError
        wide = np.asarray(raw, dtype=np.longdouble)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    limit = np.longdouble(np.finfo(np.float64).max)
    if wide.shape != shape or not np.isfinite(wide).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    if np.any(wide < -limit) or np.any(wide > limit):
        raise ValueError(f"{name} must lie within the float64 range")
    array = np.array(wide, dtype=np.float64, copy=True, order="C")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _positive_extent(value: object) -> tuple[float, float, float]:
    try:
        extent = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError("extent_xyz must contain numeric values") from exc
    if len(extent) != 3 or not all(math.isfinite(item) and item > 0.0 for item in extent):
        raise ValueError("extent_xyz must contain three finite positive values")
    return extent


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return ("ndarray", contiguous.dtype.str, contiguous.shape, contiguous.tobytes())
    if isinstance(value, Enum):
        return (type(value).__qualname__, value.value)
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            tuple((field.name, _canonical(getattr(value, field.name))) for field in fields(value)),
        )
    if isinstance(value, dict):
        return tuple(sorted((_canonical(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_canonical(item) for item in value))
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical state value: {type(value).__qualname__}")


def _tracker_dump(tracker: LocalTracker) -> tuple[object, ...]:
    graph = tracker.graph
    return (
        _canonical(tracker.config),
        _canonical(tracker.tracks),
        int(tracker._next_track_id),
        tracker._last_frame_id,
        int(graph._window_size),
        _canonical(graph._nodes),
        tuple(int(item) for item in graph._frame_ids),
        _canonical(graph._edges),
    )


def _background_snapshot(
    background: TemporalBackgroundVolume,
) -> TemporalBackgroundVolume:
    snapshot = background._clone(max(1, background.active_block_count))
    snapshot._last_blocks_touched = background.last_blocks_touched
    return snapshot


@dataclass(frozen=True, eq=False)
class TemporalEntityState:
    lifecycle: TemporalLifecycleState
    semantic_probabilities: tuple[tuple[int, float], ...]
    image_prototype: np.ndarray | None
    extent_xyz: tuple[float, float, float]
    object_to_world: np.ndarray
    submap: ObjectSubmap
    first_seen_frame_id: int
    last_seen_frame_id: int
    feature_model_id: str | None = None

    __hash__ = None

    def __post_init__(self) -> None:
        self._initialize_owned(adopt=False)

    def _initialize_owned(self, *, adopt: bool) -> None:
        if not isinstance(self.lifecycle, TemporalLifecycleState):
            raise TypeError("lifecycle must be a TemporalLifecycleState")
        if type(self.semantic_probabilities) is not tuple:
            raise TypeError("semantic_probabilities must be an exact tuple")
        normalized: list[tuple[int, float]] = []
        previous = 0
        for item in self.semantic_probabilities:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("semantic probabilities must be exact pairs")
            class_id, probability = item
            if isinstance(class_id, (bool, np.bool_)) or not isinstance(class_id, Integral):
                raise TypeError("semantic class IDs must be integers")
            class_id = int(class_id)
            if class_id <= previous:
                raise ValueError("semantic class IDs must be sorted, unique, and positive")
            if isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real):
                raise TypeError("semantic probabilities must be numeric")
            probability = float(probability)
            if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
                raise ValueError("semantic probabilities must be finite and positive")
            normalized.append((class_id, probability))
            previous = class_id
        if normalized and not math.isclose(
            math.fsum(probability for _, probability in normalized),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("semantic probabilities must sum to one")
        if self.image_prototype is not None:
            prototype = np.array(self.image_prototype, dtype=np.float64, copy=True, order="C")
            if prototype.ndim != 1 or prototype.size == 0 or not np.isfinite(prototype).all():
                raise ValueError("image_prototype must be a finite non-empty vector")
            norm = float(np.linalg.norm(prototype))
            if norm == 0.0:
                raise ValueError("image_prototype must be nonzero")
            prototype = np.ascontiguousarray(prototype / norm)
            object.__setattr__(
                self,
                "image_prototype",
                np.frombuffer(prototype.tobytes(), dtype=prototype.dtype),
            )
        if self.feature_model_id is not None:
            if not isinstance(self.feature_model_id, str) or not self.feature_model_id.strip():
                raise ValueError("feature_model_id must be a non-empty string or None")
            object.__setattr__(self, "feature_model_id", self.feature_model_id.strip())
        if self.image_prototype is None and self.feature_model_id is not None:
            raise ValueError("feature_model_id requires image_prototype")
        pose = _readonly_float_array(self.object_to_world, (4, 4), "object_to_world")
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
            raise ValueError("object_to_world must be homogeneous")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("object_to_world must be rigid")
        if not isinstance(self.submap, ObjectSubmap):
            raise TypeError("submap must be an ObjectSubmap")
        for name in ("first_seen_frame_id", "last_seen_frame_id"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.first_seen_frame_id > self.last_seen_frame_id:
            raise ValueError("first_seen_frame_id cannot exceed last_seen_frame_id")
        if self.lifecycle.last_frame_id < self.last_seen_frame_id:
            raise ValueError("lifecycle frame cannot precede last_seen_frame_id")
        object.__setattr__(self, "semantic_probabilities", tuple(normalized))
        object.__setattr__(self, "extent_xyz", _positive_extent(self.extent_xyz))
        object.__setattr__(self, "object_to_world", pose)

    def __eq__(self, other: object) -> bool:
        if type(other) is not TemporalEntityState:
            return False
        assert isinstance(other, TemporalEntityState)
        return bool(
            self.lifecycle == other.lifecycle
            and self.semantic_probabilities == other.semantic_probabilities
            and self.extent_xyz == other.extent_xyz
            and self.submap == other.submap
            and self.first_seen_frame_id == other.first_seen_frame_id
            and self.last_seen_frame_id == other.last_seen_frame_id
            and self.feature_model_id == other.feature_model_id
            and (
                (self.image_prototype is None and other.image_prototype is None)
                or (
                    self.image_prototype is not None
                    and other.image_prototype is not None
                    and np.array_equal(self.image_prototype, other.image_prototype)
                )
            )
            and np.array_equal(self.object_to_world, other.object_to_world)
        )

    def canonical_dump(self) -> tuple[object, ...]:
        return _canonical(self)


@dataclass(frozen=True)
class TemporalGeometryState:
    epochs: tuple[GeometryEpoch, ...]
    maximum_epochs_per_identity: int
    maximum_retained_epochs: int
    next_epoch_ids: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if type(self.epochs) is not tuple or any(type(item) is not GeometryEpoch for item in self.epochs):
            raise TypeError("epochs must be an exact tuple of GeometryEpoch values")
        for name in ("maximum_epochs_per_identity", "maximum_retained_epochs"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        ordered = tuple(sorted((copy.deepcopy(item) for item in self.epochs), key=lambda item: (item.entity_id, item.epoch_id)))
        keys = tuple((item.entity_id, item.epoch_id) for item in ordered)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("geometry epochs must have unique identity/epoch keys")
        for entity_id in sorted({item.entity_id for item in ordered}):
            ids = tuple(item.epoch_id for item in ordered if item.entity_id == entity_id)
            if ids != tuple(sorted(set(ids))) or len(ids) > self.maximum_epochs_per_identity:
                raise ValueError("geometry epoch IDs must be monotonic, unique, and bounded")
        if len(ordered) > self.maximum_retained_epochs:
            raise ValueError("geometry epochs exceed maximum_retained_epochs")
        if type(self.next_epoch_ids) is not tuple:
            raise TypeError("next_epoch_ids must be an exact tuple")
        next_ids: list[tuple[int, int]] = []
        for item in self.next_epoch_ids:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("next_epoch_ids must contain exact pairs")
            identity_id, next_epoch_id = item
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in item
            ):
                raise TypeError("next_epoch_ids must contain integers")
            if int(identity_id) <= 0 or int(next_epoch_id) <= 0:
                raise ValueError("next_epoch_ids must contain positive values")
            next_ids.append((int(identity_id), int(next_epoch_id)))
        next_ids = sorted(next_ids)
        if tuple(item[0] for item in next_ids) != tuple(sorted({item[0] for item in next_ids})):
            raise ValueError("next_epoch_ids identity IDs must be unique")
        for identity_id in sorted({item.entity_id for item in ordered}):
            required = max(item.epoch_id for item in ordered if item.entity_id == identity_id) + 1
            declared = dict(next_ids).get(identity_id)
            if declared is not None and declared < required:
                raise ValueError("next epoch ID cannot precede retained geometry")
            if declared is None:
                next_ids.append((identity_id, required))
        object.__setattr__(self, "epochs", ordered)
        object.__setattr__(self, "next_epoch_ids", tuple(sorted(next_ids)))

    def current(self, identity_id: int) -> GeometryEpoch:
        matches = tuple(item for item in self.epochs if item.entity_id == identity_id)
        if not matches:
            raise KeyError(identity_id)
        return matches[-1]

    def replace_current(self, epoch: GeometryEpoch) -> TemporalGeometryState:
        current = self.current(epoch.entity_id)
        if epoch.epoch_id != current.epoch_id:
            raise ValueError("replacement must preserve current epoch ID")
        return TemporalGeometryState(
            tuple(item for item in self.epochs if (item.entity_id, item.epoch_id) != (epoch.entity_id, epoch.epoch_id)) + (epoch,),
            self.maximum_epochs_per_identity,
            self.maximum_retained_epochs,
            self.next_epoch_ids,
        )

    def append(self, epoch: GeometryEpoch) -> TemporalGeometryState:
        retained = list(self.epochs)
        next_ids = dict(self.next_epoch_ids)
        expected = next_ids.get(epoch.entity_id, 0)
        if epoch.epoch_id != expected:
            raise ValueError("new geometry epoch must use the reserved next epoch ID")
        next_ids[epoch.entity_id] = epoch.epoch_id + 1
        retained.append(epoch)
        while sum(item.entity_id == epoch.entity_id for item in retained) > self.maximum_epochs_per_identity:
            victim = next(item for item in retained if item.entity_id == epoch.entity_id)
            retained.remove(victim)
        while len(retained) > self.maximum_retained_epochs:
            current_keys = {(item.entity_id, max(e.epoch_id for e in retained if e.entity_id == item.entity_id)) for item in retained}
            victim = next((item for item in retained if (item.entity_id, item.epoch_id) not in current_keys), None)
            if victim is None:
                raise OverflowError("current geometry epochs exceed retention capacity")
            retained.remove(victim)
        return TemporalGeometryState(
            tuple(retained),
            self.maximum_epochs_per_identity,
            self.maximum_retained_epochs,
            tuple(sorted(next_ids.items())),
        )

    def remove_identity(self, identity_id: int, *, forget: bool = False) -> TemporalGeometryState:
        next_ids = dict(self.next_epoch_ids)
        if forget:
            next_ids.pop(identity_id, None)
        return TemporalGeometryState(
            tuple(item for item in self.epochs if item.entity_id != identity_id),
            self.maximum_epochs_per_identity,
            self.maximum_retained_epochs,
            tuple(sorted(next_ids.items())),
        )

    def next_epoch_id(self, identity_id: int) -> int:
        return dict(self.next_epoch_ids).get(identity_id, 0)

    def canonical_dump(self) -> tuple[object, ...]:
        return (
            self.maximum_epochs_per_identity,
            self.maximum_retained_epochs,
            self.next_epoch_ids,
            _canonical(self.epochs),
        )


@dataclass(frozen=True)
class TemporalExportTrackerEntry:
    entity_id: int
    observation_count: int
    dynamic_evidence: DynamicEvidenceState
    last_centroid_xyz: tuple[float, float, float] | None
    readout_valid: bool
    geometry_epoch: int

    def __post_init__(self) -> None:
        for name, minimum in (("entity_id", 1), ("observation_count", 0), ("geometry_epoch", 0)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            object.__setattr__(self, name, int(value))
        if type(self.dynamic_evidence) is not DynamicEvidenceState:
            raise TypeError("dynamic_evidence must be DynamicEvidenceState")
        if self.last_centroid_xyz is not None:
            centroid = np.asarray(self.last_centroid_xyz, dtype=np.float64)
            if centroid.shape != (3,) or not np.isfinite(centroid).all():
                raise ValueError("last_centroid_xyz must contain three finite values")
            object.__setattr__(self, "last_centroid_xyz", tuple(float(item) for item in centroid))
        if type(self.readout_valid) is not bool:
            raise TypeError("readout_valid must be an exact bool")


@dataclass(frozen=True)
class TemporalExportTracker:
    entries: tuple[TemporalExportTrackerEntry, ...] = ()
    last_batch: TemporalExportBatch | None = None

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(type(item) is not TemporalExportTrackerEntry for item in self.entries):
            raise TypeError("entries must contain TemporalExportTrackerEntry values")
        ordered = tuple(sorted(self.entries, key=lambda item: item.entity_id))
        if tuple(item.entity_id for item in ordered) != tuple(sorted({item.entity_id for item in ordered})):
            raise ValueError("export tracker entity IDs must be unique")
        if self.last_batch is not None and type(self.last_batch) is not TemporalExportBatch:
            raise TypeError("last_batch must be a TemporalExportBatch or None")
        object.__setattr__(self, "entries", ordered)

    def get(self, entity_id: int) -> TemporalExportTrackerEntry | None:
        return next((item for item in self.entries if item.entity_id == entity_id), None)

    def canonical_dump(self) -> tuple[object, ...]:
        return _canonical(self)


@dataclass(frozen=True)
class TemporalFrameDiagnostics:
    frame_id: int
    proposal_opportunity_count: int = 0
    proposal_trigger_count: int = 0
    reid_opportunity_count: int = 0
    reid_trigger_count: int = 0
    epoch_reset_opportunity_count: int = 0
    epoch_reset_trigger_count: int = 0
    icp_opportunity_count: int = 0
    icp_accept_count: int = 0
    icp_reject_count: int = 0
    motion_rejection_count: int = 0
    ledger_stage_count: int = 0
    ledger_commit_count: int = 0
    ledger_reclaim_count: int = 0
    ledger_rejection_count: int = 0
    identity_expiry_count: int = 0
    geometry_reclaim_count: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
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
        if self.epoch_reset_trigger_count > self.epoch_reset_opportunity_count:
            raise ValueError("epoch reset triggers cannot exceed opportunities")
        if self.icp_accept_count + self.icp_reject_count != self.icp_opportunity_count:
            raise ValueError("ICP accept/reject counts must partition opportunities")


@dataclass(frozen=True)
class TemporalDiagnostics:
    processed_frame_count: int = 0
    proposal_opportunity_count: int = 0
    proposal_trigger_count: int = 0
    reid_opportunity_count: int = 0
    reid_trigger_count: int = 0
    motion_rejection_count: int = 0
    ledger_rejection_count: int = 0
    identity_expiry_count: int = 0
    geometry_reclaim_count: int = 0
    epoch_reset_opportunity_count: int = 0
    epoch_reset_trigger_count: int = 0
    icp_opportunity_count: int = 0
    icp_accept_count: int = 0
    icp_reject_count: int = 0
    ledger_stage_count: int = 0
    ledger_commit_count: int = 0
    ledger_reclaim_count: int = 0
    last_frame: TemporalFrameDiagnostics | None = None
    assignment_diagnostics: tuple[TemporalAssignmentDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "processed_frame_count", "proposal_opportunity_count",
            "proposal_trigger_count", "reid_opportunity_count",
            "reid_trigger_count", "motion_rejection_count",
            "ledger_rejection_count",
            "identity_expiry_count", "geometry_reclaim_count",
            "epoch_reset_opportunity_count", "epoch_reset_trigger_count",
            "icp_opportunity_count", "icp_accept_count", "icp_reject_count",
            "ledger_stage_count", "ledger_commit_count", "ledger_reclaim_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.last_frame is not None and type(self.last_frame) is not TemporalFrameDiagnostics:
            raise TypeError("last_frame must be TemporalFrameDiagnostics or None")
        if self.last_frame is not None and self.processed_frame_count == 0:
            raise ValueError("initial diagnostics cannot have a last frame")
        if type(self.assignment_diagnostics) is not tuple or any(
            type(item) is not TemporalAssignmentDiagnostic
            for item in self.assignment_diagnostics
        ):
            raise TypeError("assignment_diagnostics must contain exact diagnostics")
        pairs = tuple(
            (item.observation_id, item.entity_id)
            for item in self.assignment_diagnostics
        )
        if pairs != tuple(sorted(set(pairs))):
            raise ValueError("assignment diagnostics must be sorted and unique")


@dataclass(frozen=True, eq=False, repr=False, init=False)
class TemporalRuntimeState:
    __slots__ = (
        "scene_id",
        "revision",
        "last_frame_id",
        "last_timestamp",
        "next_entity_id",
        "entities",
        "background",
        "tracker",
        "identities",
        "geometry",
        "lifecycle_beliefs",
        "background_ledger",
        "export_tracker",
        "diagnostics",
        "_background_state",
        "_tracker_state",
        "_identities_state",
        "_ledger_state",
    )

    scene_id: str
    revision: int
    last_frame_id: int
    last_timestamp: float
    next_entity_id: int
    entities: tuple[TemporalEntityState, ...]
    background: TemporalBackgroundVolume
    tracker: LocalTracker
    identities: IdentityMemoryBank | None
    geometry: TemporalGeometryState | None
    lifecycle_beliefs: tuple[TemporalLifecycleState, ...] | None
    background_ledger: ReversibleBackgroundLedger | None
    export_tracker: TemporalExportTracker | None
    diagnostics: TemporalDiagnostics | None

    __hash__ = None

    def __init__(
        self,
        scene_id: str,
        revision: int,
        last_frame_id: int,
        last_timestamp: float,
        next_entity_id: int,
        entities: tuple[TemporalEntityState, ...],
        background: TemporalBackgroundVolume,
        tracker: LocalTracker,
        identities: IdentityMemoryBank | None = None,
        geometry: TemporalGeometryState | None = None,
        lifecycle_beliefs: tuple[TemporalLifecycleState, ...] | None = None,
        background_ledger: ReversibleBackgroundLedger | None = None,
        export_tracker: TemporalExportTracker | None = None,
        diagnostics: TemporalDiagnostics | None = None,
    ) -> None:
        for name, value in locals().copy().items():
            if name != "self":
                object.__setattr__(self, name, value)
        self._initialize_owned(adopt=False)

    def __getattribute__(self, name: str) -> Any:
        if name == "tracker":
            try:
                raw = object.__getattribute__(self, "_tracker_state")
            except AttributeError:
                return object.__getattribute__(self, name)
            return copy.deepcopy(raw)
        if name == "background":
            try:
                raw = object.__getattribute__(self, "_background_state")
            except AttributeError:
                return object.__getattribute__(self, name)
            return _background_snapshot(raw)
        if name == "identities":
            try:
                raw = object.__getattribute__(self, "_identities_state")
            except AttributeError:
                return object.__getattribute__(self, name)
            return raw.clone()
        if name == "background_ledger":
            try:
                raw = object.__getattribute__(self, "_ledger_state")
            except AttributeError:
                return object.__getattribute__(self, name)
            return None if raw is None else raw.clone()
        return object.__getattribute__(self, name)

    def _initialize_owned(self, *, adopt: bool) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        object.__setattr__(self, "scene_id", self.scene_id.strip())
        for name, minimum in (("revision", 0), ("last_frame_id", -1), ("next_entity_id", 0)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            object.__setattr__(self, name, int(value))
        if isinstance(self.last_timestamp, (bool, np.bool_)) or not isinstance(self.last_timestamp, Real):
            raise TypeError("last_timestamp must be numeric")
        wide_timestamp = np.longdouble(self.last_timestamp)
        limit = np.longdouble(np.finfo(np.float64).max)
        if not np.isfinite(wide_timestamp) or not -limit <= wide_timestamp <= limit:
            raise ValueError("last_timestamp must be finite")
        timestamp = float(wide_timestamp)
        object.__setattr__(self, "last_timestamp", timestamp)
        if type(self.entities) is not tuple or any(
            not isinstance(item, TemporalEntityState) for item in self.entities
        ):
            raise TypeError("entities must be an exact tuple of TemporalEntityState")
        ids = tuple(item.lifecycle.entity_id for item in self.entities)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("entities must have sorted unique lifecycle entity IDs")
        if ids and self.next_entity_id <= ids[-1]:
            raise ValueError("next_entity_id must exceed every entity ID")
        if any(entity.lifecycle.last_frame_id > self.last_frame_id for entity in self.entities):
            raise ValueError("entity lifecycle frame cannot exceed runtime last_frame_id")
        background = object.__getattribute__(self, "background")
        tracker = object.__getattribute__(self, "tracker")
        if not isinstance(background, TemporalBackgroundVolume):
            raise TypeError("background must be a TemporalBackgroundVolume")
        if not isinstance(tracker, LocalTracker):
            raise TypeError("tracker must be a LocalTracker")
        background_state = background if adopt else _background_snapshot(background)
        tracker_state = tracker if adopt else copy.deepcopy(tracker)
        identities = object.__getattribute__(self, "identities")
        geometry = object.__getattribute__(self, "geometry")
        lifecycle_beliefs = object.__getattribute__(self, "lifecycle_beliefs")
        ledger = object.__getattribute__(self, "background_ledger")
        export_tracker = object.__getattribute__(self, "export_tracker")
        diagnostics = object.__getattribute__(self, "diagnostics")
        if identities is None:
            identities = IdentityMemoryBank(
                TemporalIdentityConfig(
                    max(self.next_entity_id, background.config.maximum_entities),
                    1, 0.0, 1.0,
                )
            )
            for entity in self.entities:
                record = IdentityMemoryBank._new_record(
                    entity.lifecycle.entity_id,
                    {
                        "frame_id": entity.last_seen_frame_id,
                        "timestamp": entity.lifecycle.last_timestamp,
                        "semantic_probabilities": entity.semantic_probabilities,
                        "appearance_prototype": entity.image_prototype,
                        "feature_model_id": entity.feature_model_id,
                        "lifecycle": entity.lifecycle.lifecycle,
                        "centroid_xyz": tuple(float(item) for item in entity.object_to_world[:3, 3]),
                        "extent_xyz": entity.extent_xyz,
                        "motion_velocity_xyz": (0.0, 0.0, 0.0),
                        "motion_uncertainty_m": 0.0,
                    },
                )
                identities._records[record.identity_id] = record
            identities._next_identity_id = self.next_entity_id
        if geometry is None:
            geometry = TemporalGeometryState(
                tuple(
                    GeometryEpoch(
                        entity.lifecycle.entity_id, 0, entity.object_to_world,
                        entity.submap, True, None, entity.last_seen_frame_id,
                    )
                    for entity in self.entities
                ),
                1,
                max(1, background.config.maximum_entities),
            )
        if lifecycle_beliefs is None:
            lifecycle_beliefs = tuple(entity.lifecycle for entity in self.entities)
        if ledger is None and identities is None:
            ledger = ReversibleBackgroundLedger(
                background.config,
                TemporalBackgroundLedgerConfig(
                    background.config.background_block_count, 2, 2, 1, 8
                ),
            )
        if export_tracker is None:
            export_tracker = TemporalExportTracker()
        if diagnostics is None:
            diagnostics = TemporalDiagnostics(processed_frame_count=self.revision)
        if not isinstance(identities, IdentityMemoryBank):
            raise TypeError("identities must be an IdentityMemoryBank")
        if type(geometry) is not TemporalGeometryState:
            raise TypeError("geometry must be a TemporalGeometryState")
        if type(lifecycle_beliefs) is not tuple or any(type(item) is not TemporalLifecycleState for item in lifecycle_beliefs):
            raise TypeError("lifecycle_beliefs must contain TemporalLifecycleState values")
        if ledger is not None and not isinstance(ledger, ReversibleBackgroundLedger):
            raise TypeError("background_ledger must be a ReversibleBackgroundLedger or None")
        if type(export_tracker) is not TemporalExportTracker:
            raise TypeError("export_tracker must be a TemporalExportTracker")
        if type(diagnostics) is not TemporalDiagnostics:
            raise TypeError("diagnostics must be TemporalDiagnostics")
        if diagnostics.processed_frame_count != self.revision:
            raise ValueError("diagnostics processed frame count must match revision")
        if diagnostics.last_frame is not None and (
            diagnostics.last_frame.frame_id != self.last_frame_id
        ):
            raise ValueError("diagnostics last frame must match runtime progress")
        if (
            diagnostics.proposal_trigger_count > diagnostics.proposal_opportunity_count
            or diagnostics.reid_trigger_count > diagnostics.reid_opportunity_count
            or diagnostics.epoch_reset_trigger_count
            > diagnostics.epoch_reset_opportunity_count
        ):
            raise ValueError("diagnostic triggers cannot exceed opportunities")
        if (
            diagnostics.icp_accept_count + diagnostics.icp_reject_count
            != diagnostics.icp_opportunity_count
        ):
            raise ValueError("diagnostic ICP outcomes must partition opportunities")
        identity_ids = tuple(item.identity_id for item in identities.records)
        lifecycle_ids = tuple(item.entity_id for item in lifecycle_beliefs)
        geometry_ids = tuple(sorted({item.entity_id for item in geometry.epochs}))
        if identities._next_identity_id != self.next_entity_id:
            raise ValueError("bank next identity ID must match runtime next_entity_id")
        if lifecycle_ids != identity_ids:
            raise ValueError("lifecycle beliefs must exactly cover identity memory")
        if not set(ids).issubset(identity_ids) or geometry_ids != ids:
            raise ValueError("geometry and legacy wrappers must be an identity subset")
        if any(
            item.last_frame_id > self.last_frame_id
            or item.last_timestamp > self.last_timestamp
            for item in identities.records
        ):
            raise ValueError("identity observation cannot be in the future of runtime progress")
        if any(identities.get(item.entity_id).lifecycle is not item.lifecycle for item in lifecycle_beliefs):
            raise ValueError("identity and lifecycle states must agree")
        if any(
            item.last_frame_id != self.last_frame_id
            or item.last_timestamp != self.last_timestamp
            for item in lifecycle_beliefs
        ):
            raise ValueError("lifecycle belief progress must equal runtime progress")
        lifecycle_by_id = {item.entity_id: item for item in lifecycle_beliefs}
        export_by_id = {item.entity_id: item for item in export_tracker.entries}
        if not set(export_by_id).issubset(identity_ids):
            raise ValueError("export tracker IDs must belong to identity memory")
        for entity in self.entities:
            entity_id = entity.lifecycle.entity_id
            record = identities.get(entity_id)
            assert record is not None
            belief = lifecycle_by_id[entity_id]
            if entity.lifecycle != belief or record.lifecycle is not belief.lifecycle:
                raise ValueError("wrapper, identity, and lifecycle values must agree")
            if entity.semantic_probabilities != record.semantic_probabilities:
                raise ValueError("wrapper and identity semantic probabilities must agree")
            if (
                entity.extent_xyz != record.extent_xyz
                or entity.first_seen_frame_id != record.first_frame_id
                or entity.last_seen_frame_id != record.last_frame_id
            ):
                raise ValueError("wrapper and identity extent/frame history must agree")
            if entity.feature_model_id != record.feature_model_id or not (
                (entity.image_prototype is None and record.appearance_prototype is None)
                or (
                    entity.image_prototype is not None
                    and record.appearance_prototype is not None
                    and np.array_equal(entity.image_prototype, record.appearance_prototype)
                )
            ):
                raise ValueError("wrapper and identity appearance provenance must agree")
            epoch = geometry.current(entity_id)
            if (
                not np.array_equal(epoch.object_to_world, entity.object_to_world)
                or epoch.submap != entity.submap
            ):
                raise ValueError("current geometry epoch pose/submap must match wrapper")
            points = entity.submap.world_points(entity.object_to_world)
            centroid = (
                tuple(float(value) for value in points.mean(axis=0, dtype=np.float64))
                if points.shape[0]
                else tuple(float(value) for value in entity.object_to_world[:3, 3])
            )
            if centroid != record.last_centroid_xyz:
                raise ValueError("wrapper geometry centroid and identity memory must agree")
            tracked = export_by_id.get(entity_id)
            if tracked is not None and (
                tracked.geometry_epoch != epoch.epoch_id
                or tracked.readout_valid is not epoch.readout_valid
            ):
                raise ValueError("export tracker must match current geometry epoch/readout")
        if export_tracker.last_batch is not None and (
            export_tracker.last_batch.frame_index != self.last_frame_id
            or export_tracker.last_batch.timestamp_ns < 0
        ):
            raise ValueError("export tracker batch must match runtime progress")
        object.__setattr__(self, "background", None)
        object.__setattr__(self, "tracker", None)
        object.__setattr__(self, "identities", None)
        object.__setattr__(self, "background_ledger", None)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "lifecycle_beliefs", tuple(copy.deepcopy(item) for item in lifecycle_beliefs))
        object.__setattr__(self, "export_tracker", copy.deepcopy(export_tracker))
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "_background_state", background_state)
        object.__setattr__(self, "_tracker_state", tracker_state)
        object.__setattr__(self, "_identities_state", identities if adopt else identities.clone())
        object.__setattr__(
            self, "_ledger_state",
            ledger if adopt or ledger is None else ledger.clone(),
        )

        if (self.revision == 0) != (self.last_frame_id == -1):
            raise ValueError("revision zero must identify the initial state")
        tracker_frame = tracker_state._last_frame_id
        if (self.last_frame_id == -1 and tracker_frame is not None) or (
            self.last_frame_id >= 0 and tracker_frame != self.last_frame_id
        ):
            raise ValueError("tracker last frame must match runtime last_frame_id")
        if any(entity.lifecycle.last_frame_id != self.last_frame_id for entity in self.entities):
            raise ValueError("entity lifecycle frame must equal runtime last_frame_id")
        if len(self.entities) > background_state.config.maximum_entities:
            raise ValueError("entities exceed configured maximum_entities")

    def __repr__(self) -> str:
        return (
            f"TemporalRuntimeState(scene_id={self.scene_id!r}, revision={self.revision}, "
            f"last_frame_id={self.last_frame_id}, entities={len(self.entities)})"
        )

    def __copy__(self) -> TemporalRuntimeState:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> TemporalRuntimeState:
        del memo
        return self

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError("TemporalRuntimeState does not support pickle; use canonical_dump")

    def _mutable_background_snapshot(self) -> TemporalBackgroundVolume:
        return _background_snapshot(object.__getattribute__(self, "_background_state"))

    def _integrate_background_owned(
        self, frame: Any, masked_depth: np.ndarray
    ) -> TemporalBackgroundVolume:
        background = object.__getattribute__(self, "_background_state")
        return background._owned_trial_integrate(frame, masked_depth)

    def _mutable_tracker_snapshot(self) -> LocalTracker:
        return copy.deepcopy(object.__getattribute__(self, "_tracker_state"))

    def _mutable_identities_snapshot(self) -> IdentityMemoryBank:
        return object.__getattribute__(self, "_identities_state").clone()

    def _mutable_ledger_snapshot(self) -> ReversibleBackgroundLedger:
        ledger = object.__getattribute__(self, "_ledger_state")
        if ledger is None:
            raise RuntimeError("background ledger is disabled for this profile")
        return ledger.clone()

    def _owned_background(self) -> TemporalBackgroundVolume:
        return object.__getattribute__(self, "_background_state")

    @classmethod
    def _adopt_owned(
        cls,
        *,
        scene_id: str,
        revision: int,
        last_frame_id: int,
        last_timestamp: float,
        next_entity_id: int,
        entities: tuple[TemporalEntityState, ...],
        background: TemporalBackgroundVolume,
        tracker: LocalTracker,
        identities: IdentityMemoryBank,
        geometry: TemporalGeometryState,
        lifecycle_beliefs: tuple[TemporalLifecycleState, ...],
        background_ledger: ReversibleBackgroundLedger | None,
        export_tracker: TemporalExportTracker,
        diagnostics: TemporalDiagnostics,
    ) -> TemporalRuntimeState:
        state = object.__new__(cls)
        for name, value in (
            ("scene_id", scene_id),
            ("revision", revision),
            ("last_frame_id", last_frame_id),
            ("last_timestamp", last_timestamp),
            ("next_entity_id", next_entity_id),
            ("entities", entities),
            ("background", background),
            ("tracker", tracker),
            ("identities", identities),
            ("geometry", geometry),
            ("lifecycle_beliefs", lifecycle_beliefs),
            ("background_ledger", background_ledger),
            ("export_tracker", export_tracker),
            ("diagnostics", diagnostics),
        ):
            object.__setattr__(state, name, value)
        state._initialize_owned(adopt=True)
        return state

    def canonical_dump(self) -> tuple[object, ...]:
        background = object.__getattribute__(self, "_background_state")
        tracker = object.__getattribute__(self, "_tracker_state")
        identities = object.__getattribute__(self, "_identities_state")
        ledger = object.__getattribute__(self, "_ledger_state")
        return (
            self.scene_id,
            self.revision,
            self.last_frame_id,
            self.last_timestamp,
            self.next_entity_id,
            tuple(entity.canonical_dump() for entity in self.entities),
            _canonical(background.config),
            background.canonical_block_state(),
            background.last_blocks_touched,
            _tracker_dump(tracker),
            identities.canonical_dump(),
            self.geometry.canonical_dump(),
            _canonical(self.lifecycle_beliefs),
            None if ledger is None else ledger.journal_digest(),
            self.export_tracker.canonical_dump(),
            _canonical(self.diagnostics),
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is TemporalRuntimeState and self.canonical_dump() == other.canonical_dump()
