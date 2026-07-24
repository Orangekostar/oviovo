from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
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
        array = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
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

    __hash__ = None

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, eq=False)
class TemporalRuntimeState:
    scene_id: str
    revision: int
    last_frame_id: int
    last_timestamp: float
    next_entity_id: int
    entities: tuple[TemporalEntityState, ...]
    background: TemporalBackgroundVolume
    tracker: LocalTracker

    __hash__ = None

    def __post_init__(self) -> None:
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
        timestamp = float(self.last_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("last_timestamp must be finite")
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
        if not isinstance(self.background, TemporalBackgroundVolume):
            raise TypeError("background must be a TemporalBackgroundVolume")
        if not isinstance(self.tracker, LocalTracker):
            raise TypeError("tracker must be a LocalTracker")

    def canonical_dump(self) -> tuple[object, ...]:
        return (
            self.scene_id,
            self.revision,
            self.last_frame_id,
            self.last_timestamp,
            self.next_entity_id,
            tuple(entity.canonical_dump() for entity in self.entities),
            _canonical(self.background.config),
            self.background.canonical_block_state(),
            self.background.last_blocks_touched,
            _tracker_dump(self.tracker),
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is TemporalRuntimeState and self.canonical_dump() == other.canonical_dump()
