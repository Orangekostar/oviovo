from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_geometry import ObjectSubmap
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
        if (self.image_prototype is None) != (self.feature_model_id is None):
            raise ValueError("image_prototype and feature_model_id must be provided together")
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


@dataclass(frozen=True, eq=False, slots=True, repr=False)
class TemporalRuntimeState:
    scene_id: str
    revision: int
    last_frame_id: int
    last_timestamp: float
    next_entity_id: int
    entities: tuple[TemporalEntityState, ...]
    background: TemporalBackgroundVolume
    tracker: LocalTracker
    _background_state: TemporalBackgroundVolume = field(init=False, repr=False)
    _tracker_state: LocalTracker = field(init=False, repr=False)

    __hash__ = None

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
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        self._initialize_owned(adopt=False)

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
        object.__setattr__(self, "background", None)
        object.__setattr__(self, "tracker", None)
        object.__setattr__(self, "_background_state", background_state)
        object.__setattr__(self, "_tracker_state", tracker_state)

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
        ):
            object.__setattr__(state, name, value)
        state._initialize_owned(adopt=True)
        return state

    def canonical_dump(self) -> tuple[object, ...]:
        background = object.__getattribute__(self, "_background_state")
        tracker = object.__getattribute__(self, "_tracker_state")
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
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is TemporalRuntimeState and self.canonical_dump() == other.canonical_dump()
