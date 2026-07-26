from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.oviv2.temporal_config import TemporalIdentityConfig
from src.oviv2.temporal_lifecycle import TemporalLifecycle


_INT64_MAX = 2**63 - 1


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if value > _INT64_MAX:
        raise ValueError(f"{name} must fit signed int64")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _point3(value: object, name: str, *, positive: bool = False) -> tuple[float, float, float]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain three finite values")
    result = tuple(float(item) for item in point)
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} must contain positive values")
    return result  # type: ignore[return-value]


def _probabilities(value: object) -> tuple[tuple[int, float], ...]:
    if type(value) is not tuple:
        raise TypeError("semantic_probabilities must be an exact tuple")
    output: list[tuple[int, float]] = []
    previous = 0
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("semantic probabilities must be exact pairs")
        class_id = _exact_int(item[0], "semantic class_id", minimum=1)
        probability = _finite(item[1], "semantic probability", minimum=0.0)
        if probability > 1.0:
            raise ValueError("semantic probability must lie in [0, 1]")
        if class_id <= previous:
            raise ValueError("semantic probabilities must have sorted unique class IDs")
        output.append((class_id, probability))
        previous = class_id
    if output and not math.isclose(math.fsum(item[1] for item in output), 1.0, abs_tol=1e-9):
        raise ValueError("semantic probabilities must sum to 1")
    return tuple(output)


def _prototype(value: object, model_id: object) -> tuple[np.ndarray | None, str | None]:
    if value is None:
        if model_id is not None:
            raise ValueError("feature_model_id requires appearance_prototype")
        return None, None
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("appearance_prototype requires a non-empty feature_model_id")
    try:
        vector = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError("appearance_prototype must be a finite non-zero vector") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("appearance_prototype must be a finite non-zero vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("appearance_prototype must be a finite non-zero vector")
    normalized = np.ascontiguousarray(vector / norm)
    frozen = np.frombuffer(normalized.tobytes(), dtype=normalized.dtype)
    return frozen, model_id.strip()


@dataclass(frozen=True, eq=False)
class IdentityMemoryRecord:
    identity_id: int
    semantic_probabilities: tuple[tuple[int, float], ...]
    appearance_prototype: np.ndarray | None
    feature_model_id: str | None
    lifecycle: TemporalLifecycle
    first_frame_id: int
    first_timestamp: float
    first_centroid_xyz: tuple[float, float, float]
    last_frame_id: int
    last_timestamp: float
    last_centroid_xyz: tuple[float, float, float]
    extent_xyz: tuple[float, float, float]
    motion_velocity_xyz: tuple[float, float, float]
    motion_uncertainty_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _exact_int(self.identity_id, "identity_id", minimum=1))
        object.__setattr__(self, "semantic_probabilities", _probabilities(self.semantic_probabilities))
        prototype, model_id = _prototype(self.appearance_prototype, self.feature_model_id)
        object.__setattr__(self, "appearance_prototype", prototype)
        object.__setattr__(self, "feature_model_id", model_id)
        if not isinstance(self.lifecycle, TemporalLifecycle):
            raise TypeError("lifecycle must be a TemporalLifecycle")
        first_frame = _exact_int(self.first_frame_id, "first_frame_id")
        last_frame = _exact_int(self.last_frame_id, "last_frame_id")
        first_time = _finite(self.first_timestamp, "first_timestamp")
        last_time = _finite(self.last_timestamp, "last_timestamp")
        if last_frame < first_frame or last_time < first_time:
            raise ValueError("last observation cannot precede first observation")
        if last_frame > first_frame and last_time <= first_time:
            raise ValueError("last_timestamp must increase strictly with last_frame_id")
        if last_frame == first_frame and last_time != first_time:
            raise ValueError("one frame cannot have two observation timestamps")
        object.__setattr__(self, "first_frame_id", first_frame)
        object.__setattr__(self, "last_frame_id", last_frame)
        object.__setattr__(self, "first_timestamp", first_time)
        object.__setattr__(self, "last_timestamp", last_time)
        object.__setattr__(self, "first_centroid_xyz", _point3(self.first_centroid_xyz, "first_centroid_xyz"))
        object.__setattr__(self, "last_centroid_xyz", _point3(self.last_centroid_xyz, "last_centroid_xyz"))
        object.__setattr__(self, "extent_xyz", _point3(self.extent_xyz, "extent_xyz", positive=True))
        object.__setattr__(self, "motion_velocity_xyz", _point3(self.motion_velocity_xyz, "motion_velocity_xyz"))
        object.__setattr__(self, "motion_uncertainty_m", _finite(self.motion_uncertainty_m, "motion_uncertainty_m", minimum=0.0))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdentityMemoryRecord):
            return NotImplemented
        return self._canonical() == other._canonical()

    def __deepcopy__(self, memo: dict[int, object]) -> IdentityMemoryRecord:
        del memo
        return IdentityMemoryRecord(**self._constructor_fields())

    def _constructor_fields(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "semantic_probabilities": self.semantic_probabilities,
            "appearance_prototype": self.appearance_prototype,
            "feature_model_id": self.feature_model_id,
            "lifecycle": self.lifecycle,
            "first_frame_id": self.first_frame_id,
            "first_timestamp": self.first_timestamp,
            "first_centroid_xyz": self.first_centroid_xyz,
            "last_frame_id": self.last_frame_id,
            "last_timestamp": self.last_timestamp,
            "last_centroid_xyz": self.last_centroid_xyz,
            "extent_xyz": self.extent_xyz,
            "motion_velocity_xyz": self.motion_velocity_xyz,
            "motion_uncertainty_m": self.motion_uncertainty_m,
        }

    def _canonical(self) -> tuple[object, ...]:
        prototype = None if self.appearance_prototype is None else self.appearance_prototype.tobytes().hex()
        return (
            self.identity_id, self.semantic_probabilities, prototype, self.feature_model_id,
            self.lifecycle.value, self.first_frame_id, self.first_timestamp,
            self.first_centroid_xyz, self.last_frame_id, self.last_timestamp,
            self.last_centroid_xyz, self.extent_xyz, self.motion_velocity_xyz,
            self.motion_uncertainty_m,
        )


@dataclass(frozen=True)
class IdentityExpiryDiagnostics:
    current_frame_id: int
    expired_identity_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_frame_id",
            _exact_int(self.current_frame_id, "current_frame_id"),
        )
        if type(self.expired_identity_ids) is not tuple:
            raise TypeError("expired_identity_ids must be an exact tuple")
        normalized = tuple(
            _exact_int(value, "expired identity_id", minimum=1)
            for value in self.expired_identity_ids
        )
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("expired_identity_ids must be sorted and unique")
        object.__setattr__(self, "expired_identity_ids", normalized)


class IdentityMemoryBank:
    def __init__(self, config: TemporalIdentityConfig) -> None:
        if not isinstance(config, TemporalIdentityConfig):
            raise TypeError("config must be a TemporalIdentityConfig")
        maximum = _exact_int(config.maximum_identities, "config.maximum_identities", minimum=1)
        dormant = _exact_int(config.maximum_dormant_frames, "config.maximum_dormant_frames", minimum=1)
        similarity = _finite(config.minimum_reid_similarity, "config.minimum_reid_similarity")
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("config.minimum_reid_similarity must lie in [0, 1]")
        distance = _finite(config.maximum_reid_distance_m, "config.maximum_reid_distance_m")
        if distance <= 0.0:
            raise ValueError("config.maximum_reid_distance_m must be positive")
        self._config = config
        self._maximum_identities = maximum
        self._maximum_dormant_frames = dormant
        self._next_identity_id = 1
        self._records: dict[int, IdentityMemoryRecord] = {}

    @property
    def records(self) -> tuple[IdentityMemoryRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def get(self, identity_id: int) -> IdentityMemoryRecord | None:
        identity_id = _exact_int(identity_id, "identity_id", minimum=1)
        return self._records.get(identity_id)

    def insert(self, **values: object) -> IdentityMemoryRecord:
        if len(self._records) >= self._maximum_identities:
            raise OverflowError("maximum_identities capacity has been exhausted")
        identity_id = self._next_identity_id
        record = self._new_record(identity_id, values)
        self._records[identity_id] = record
        self._next_identity_id += 1
        return record

    def update(self, identity_id: int, **values: object) -> IdentityMemoryRecord:
        identity_id = _exact_int(identity_id, "identity_id", minimum=1)
        previous = self._records.get(identity_id)
        if previous is None:
            raise KeyError(identity_id)
        frame_id = _exact_int(values.get("frame_id"), "frame_id")
        timestamp = _finite(values.get("timestamp"), "timestamp")
        if frame_id <= previous.last_frame_id or timestamp <= previous.last_timestamp:
            raise ValueError("frame_id and timestamp must increase strictly")
        record = self._new_record(
            identity_id,
            values,
            first=(previous.first_frame_id, previous.first_timestamp, previous.first_centroid_xyz),
        )
        self._records[identity_id] = record
        return record

    def expire_dormant(self, *, current_frame_id: int) -> IdentityExpiryDiagnostics:
        current = _exact_int(current_frame_id, "current_frame_id")
        if self._records and current < max(item.last_frame_id for item in self._records.values()):
            raise ValueError("current_frame_id cannot precede a retained identity")
        expired = tuple(
            record.identity_id for record in self.records
            if record.lifecycle is TemporalLifecycle.DORMANT
            and current - record.last_frame_id > self._maximum_dormant_frames
        )
        for identity_id in expired:
            del self._records[identity_id]
        return IdentityExpiryDiagnostics(current, expired)

    def clone(self) -> IdentityMemoryBank:
        clone = IdentityMemoryBank(self._config)
        clone._next_identity_id = self._next_identity_id
        clone._records = {item.identity_id: item.__deepcopy__({}) for item in self.records}
        return clone

    def canonical_dump(self) -> tuple[object, ...]:
        return (self._next_identity_id, tuple(item._canonical() for item in self.records))

    @staticmethod
    def _new_record(
        identity_id: int,
        values: dict[str, object],
        *,
        first: tuple[int, float, tuple[float, float, float]] | None = None,
    ) -> IdentityMemoryRecord:
        expected = {
            "frame_id", "timestamp", "semantic_probabilities", "appearance_prototype",
            "feature_model_id", "lifecycle", "centroid_xyz", "extent_xyz",
            "motion_velocity_xyz", "motion_uncertainty_m",
        }
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown or missing:
            raise TypeError(f"identity fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
        frame_id = _exact_int(values["frame_id"], "frame_id")
        timestamp = _finite(values["timestamp"], "timestamp")
        centroid = _point3(values["centroid_xyz"], "centroid_xyz")
        if first is None:
            first = (frame_id, timestamp, centroid)
        return IdentityMemoryRecord(
            identity_id=identity_id,
            semantic_probabilities=values["semantic_probabilities"],  # type: ignore[arg-type]
            appearance_prototype=values["appearance_prototype"],  # type: ignore[arg-type]
            feature_model_id=values["feature_model_id"],  # type: ignore[arg-type]
            lifecycle=values["lifecycle"],  # type: ignore[arg-type]
            first_frame_id=first[0], first_timestamp=first[1], first_centroid_xyz=first[2],
            last_frame_id=frame_id, last_timestamp=timestamp, last_centroid_xyz=centroid,
            extent_xyz=values["extent_xyz"],  # type: ignore[arg-type]
            motion_velocity_xyz=values["motion_velocity_xyz"],  # type: ignore[arg-type]
            motion_uncertainty_m=values["motion_uncertainty_m"],  # type: ignore[arg-type]
        )
