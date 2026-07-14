"""Process-neutral mapping, query, and evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

SnapshotScope = Literal["current", "history"]
ExecutionMode = Literal["online", "offline"]
IntegrationMode = Literal["native", "composed"]


def _readonly_array(
    value: Any,
    *,
    name: str,
    dtype: Any | None = None,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    finite: bool = True,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if finite and array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _points(value: Any, *, name: str) -> np.ndarray:
    array = _readonly_array(value, name=name, dtype=np.float32, ndim=2)
    if array.shape[1:] != (3,):
        raise ValueError(f"{name} shape must be (N, 3), got {array.shape}")
    return array


@dataclass(frozen=True)
class FramePacket:
    """Runtime RGB-D input. Ground-truth fields are intentionally absent."""

    scene_id: str
    frame_id: int
    timestamp: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        rgb = _readonly_array(self.rgb, name="rgb", dtype=np.uint8, ndim=3, finite=False)
        if rgb.shape[2:] != (3,):
            raise ValueError(f"rgb shape must be (H, W, 3), got {rgb.shape}")
        depth = _readonly_array(self.depth_m, name="depth_m", dtype=np.float32, ndim=2, finite=False)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth_m shape must match rgb spatial shape {rgb.shape[:2]}, got {depth.shape}")
        intrinsics = _readonly_array(self.intrinsics, name="intrinsics", dtype=np.float64, shape=(3, 3))
        pose = _readonly_array(
            self.camera_to_world,
            name="camera_to_world",
            dtype=np.float64,
            shape=(4, 4),
        )
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("camera_to_world must be a homogeneous transform")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", pose)


@dataclass
class EntityPrediction:
    entity_id: str
    points_xyz: np.ndarray
    semantic_embedding: np.ndarray | None
    semantic_label: str | None
    semantic_score: float
    lifecycle_state: str
    first_seen: float
    last_seen: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.entity_id):
            raise ValueError("entity_id must be non-empty")
        self.entity_id = str(self.entity_id)
        self.points_xyz = _points(self.points_xyz, name="points_xyz")
        if self.semantic_embedding is not None:
            self.semantic_embedding = _readonly_array(
                self.semantic_embedding,
                name="semantic_embedding",
                dtype=np.float32,
                ndim=1,
            )
        if self.semantic_label is not None:
            self.semantic_label = str(self.semantic_label)
        self.semantic_score = float(self.semantic_score)
        if not np.isfinite(self.semantic_score):
            raise ValueError("semantic_score must be finite")
        self.lifecycle_state = str(self.lifecycle_state)
        if not self.lifecycle_state:
            raise ValueError("lifecycle_state must be non-empty")
        self.first_seen = float(self.first_seen)
        self.last_seen = float(self.last_seen)
        if self.first_seen > self.last_seen:
            raise ValueError("first_seen cannot exceed last_seen")
        self.metadata = dict(self.metadata)


@dataclass
class MapSnapshot:
    method: str
    scene_id: str
    timestamp: float
    entities: list[EntityPrediction]
    background_xyz: np.ndarray | None
    scope: SnapshotScope
    runtime: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method or not self.scene_id:
            raise ValueError("method and scene_id must be non-empty")
        self.timestamp = float(self.timestamp)
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if self.scope not in {"current", "history"}:
            raise ValueError(f"invalid snapshot scope: {self.scope}")
        self.entities = list(self.entities)
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id must be unique within a snapshot")
        if self.background_xyz is not None:
            self.background_xyz = _points(self.background_xyz, name="background_xyz")
        self.runtime = {str(key): float(value) for key, value in self.runtime.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in self.runtime.values()):
            raise ValueError("runtime values must be finite and non-negative")


@dataclass(frozen=True)
class QueryRequest:
    query_id: str
    scene_id: str
    text: str
    timestamp: float
    scope: SnapshotScope = "current"

    def __post_init__(self) -> None:
        if not self.query_id or not self.scene_id or not self.text.strip():
            raise ValueError("query_id, scene_id, and text must be non-empty")
        if self.scope not in {"current", "history"}:
            raise ValueError(f"invalid query scope: {self.scope}")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass
class QueryResult:
    query_id: str
    ranked_entity_ids: list[str]
    raw_scores: list[float]
    returned_entity_id: str | None
    rejected_as_not_found: bool
    latency_ms: float

    def __post_init__(self) -> None:
        self.ranked_entity_ids = [str(value) for value in self.ranked_entity_ids]
        self.raw_scores = [float(value) for value in self.raw_scores]
        if len(self.ranked_entity_ids) != len(self.raw_scores):
            raise ValueError("ranked_entity_ids and raw_scores must have equal length")
        if any(not np.isfinite(value) for value in self.raw_scores):
            raise ValueError("raw_scores must be finite")
        self.returned_entity_id = None if self.returned_entity_id is None else str(self.returned_entity_id)
        self.latency_ms = float(self.latency_ms)
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")
        if self.rejected_as_not_found and self.returned_entity_id is not None:
            raise ValueError("a rejected query cannot return an entity")
        if self.returned_entity_id is not None and self.returned_entity_id not in self.ranked_entity_ids:
            raise ValueError("returned_entity_id must appear in ranked_entity_ids")


@dataclass(frozen=True)
class QueryTarget:
    query_id: str
    valid_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        object.__setattr__(self, "valid_entity_ids", tuple(str(value) for value in self.valid_entity_ids))


@dataclass(frozen=True)
class RunMetadata:
    method: str
    integration: IntegrationMode
    execution_mode: ExecutionMode

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("method must be non-empty")
        if self.integration not in {"native", "composed"}:
            raise ValueError(f"invalid integration mode: {self.integration}")
        if self.execution_mode not in {"online", "offline"}:
            raise ValueError(f"invalid execution mode: {self.execution_mode}")
        if (
            self.integration == "composed"
            and " + " not in self.method
            and "composed" not in self.method.lower()
        ):
            raise ValueError("composed method label must disclose the added component")
        if self.execution_mode == "offline" and "offline" not in self.method.lower():
            raise ValueError("offline method label must disclose offline execution")

    @property
    def eligible_for_online_latency(self) -> bool:
        return self.execution_mode == "online"


@dataclass
class GroundTruthSnapshot:
    scene_id: str
    timestamp: float
    points_xyz: np.ndarray
    semantic_labels: np.ndarray
    instance_ids: np.ndarray

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("ground-truth scene_id must be non-empty")
        self.timestamp = float(self.timestamp)
        if not np.isfinite(self.timestamp):
            raise ValueError("ground-truth timestamp must be finite")
        self.points_xyz = _points(self.points_xyz, name="gt.points_xyz")
        labels = _readonly_array(
            self.semantic_labels,
            name="gt.semantic_labels",
            dtype=object,
            ndim=1,
            finite=False,
        )
        instance_ids = _readonly_array(
            self.instance_ids,
            name="gt.instance_ids",
            dtype=np.int64,
            ndim=1,
        )
        if len(labels) != len(self.points_xyz) or len(instance_ids) != len(self.points_xyz):
            raise ValueError("ground-truth arrays must have equal length")
        self.semantic_labels = labels
        self.instance_ids = instance_ids


@dataclass(frozen=True)
class GroundTruthSequence:
    scene_id: str
    snapshots: tuple[GroundTruthSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("ground-truth sequence scene_id must be non-empty")
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        timestamps = [snapshot.timestamp for snapshot in self.snapshots]
        if timestamps != sorted(timestamps):
            raise ValueError("ground-truth timestamps must be monotonically non-decreasing")
        if any(snapshot.scene_id != self.scene_id for snapshot in self.snapshots):
            raise ValueError("ground-truth snapshot scene mismatch")


@runtime_checkable
class DatasetAdapter(Protocol):
    def scenes(self) -> Sequence[str]: ...

    def frames(self, scene_id: str) -> Iterator[FramePacket]: ...

    def ground_truth(self, scene_id: str) -> GroundTruthSequence: ...


@runtime_checkable
class BaselineRunner(Protocol):
    def reset(self, scene_id: str) -> None: ...

    def update(self, frame: FramePacket) -> None: ...

    def snapshot(self, timestamp: float, scope: SnapshotScope = "current") -> MapSnapshot: ...

    def query(self, request: QueryRequest) -> QueryResult: ...

    def close(self) -> None: ...


@runtime_checkable
class UnifiedEvaluator(Protocol):
    def evaluate_snapshot(self, prediction: MapSnapshot, ground_truth: GroundTruthSnapshot) -> dict[str, Any]: ...

    def evaluate_sequence(
        self,
        predictions: Sequence[MapSnapshot],
        ground_truth: GroundTruthSequence,
    ) -> dict[str, Any]: ...

    def evaluate_queries(
        self,
        results: Sequence[QueryResult],
        ground_truth: Sequence[QueryTarget],
    ) -> dict[str, Any]: ...
