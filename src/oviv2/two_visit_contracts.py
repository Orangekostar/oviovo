"""Immutable contracts for source-bound two-visit mapping."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from src.evaluation.contracts import MapSnapshot


RelationState = Literal[
    "persistent_static",
    "persistent_moved",
    "appeared",
    "removed_candidate",
    "uncertain",
    "split",
    "merge",
]
CompositionAction = Literal[
    "emit_t1",
    "retain_t0_unobserved",
    "retain_t0_occluded",
    "suppress_t0_visible_free",
    "uncertain",
]
VisibilityState = Literal["occupied", "visible_free", "occluded", "unobserved"]
GeometrySource = Literal["ovi_t0", "ovi_t1"]
IdentitySource = Literal["rescene", "geometric_baseline", "unmatched"]
StateSource = Literal["t1_visibility", "pair_reasoner", "fallback"]
SemanticSource = Literal["ovi_t0", "ovi_t1", "fused_ovi"]
EvidenceStatus = Literal[
    "PASS",
    "BLOCKED_EXTERNAL_SOURCE",
    "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
    "PLUMBING_ONLY_RANDOM_INITIALIZATION",
]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_RELATION_STATES = {
    "persistent_static",
    "persistent_moved",
    "appeared",
    "removed_candidate",
    "uncertain",
    "split",
    "merge",
}
_COMPOSITION_ACTIONS = {
    "emit_t1",
    "retain_t0_unobserved",
    "retain_t0_occluded",
    "suppress_t0_visible_free",
    "uncertain",
}
_VISIBILITY_STATES = {"occupied", "visible_free", "occluded", "unobserved"}
_IDENTITY_SOURCES = {"rescene", "geometric_baseline", "unmatched"}
_STATE_SOURCES = {"t1_visibility", "pair_reasoner", "fallback"}
_SEMANTIC_SOURCES = {"ovi_t0", "ovi_t1", "fused_ovi"}
_EVIDENCE_STATUSES = {
    "PASS",
    "BLOCKED_EXTERNAL_SOURCE",
    "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
    "PLUMBING_ONLY_RANDOM_INITIALIZATION",
}


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")
    return text


def _finite_float(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" and at least {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{qualifier}")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _readonly_array(
    value: object,
    name: str,
    *,
    dtype: np.dtype[Any] | type,
    ndim: int,
    finite: bool = True,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if finite and array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _readonly_integer_array(value: object, name: str, *, dtype: np.dtype[Any] | type) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must have 1 dimension")
    if raw.size and (
        not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain integers")
    result = np.array(raw, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _canonical(value: object, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} contains a non-finite number")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, item in value.items():
            key = _nonempty_string(raw_key, f"{name} key")
            normalized[key] = _canonical(item, f"{name}.{key}")
        return dict(sorted(normalized.items()))
    if isinstance(value, (tuple, list)):
        return [_canonical(item, f"{name} item") for item in value]
    if isinstance(value, Path):
        return str(value)
    raise ValueError(f"{name} contains unsupported value type {type(value).__name__}")


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _canonical(payload, "content"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_content_sha256(snapshot: MapSnapshot) -> str:
    """Return a stable digest of every mapping field used by this pipeline."""

    if not isinstance(snapshot, MapSnapshot):
        raise TypeError("snapshot must be a MapSnapshot")
    entities: list[dict[str, object]] = []
    for entity in snapshot.entities:
        entities.append(
            {
                "entity_id": entity.entity_id,
                "points_xyz": entity.points_xyz,
                "semantic_embedding": entity.semantic_embedding,
                "semantic_label": entity.semantic_label,
                "semantic_score": entity.semantic_score,
                "lifecycle_state": entity.lifecycle_state,
                "first_seen": entity.first_seen,
                "last_seen": entity.last_seen,
                "metadata": entity.metadata,
            }
        )
    return _digest(
        {
            "method": snapshot.method,
            "scene_id": snapshot.scene_id,
            "timestamp": snapshot.timestamp,
            "scope": snapshot.scope,
            "entities": entities,
            "background_xyz": snapshot.background_xyz,
            "runtime": snapshot.runtime,
        }
    )


@dataclass(frozen=True, slots=True)
class VisitMap:
    """One independently built OVI map with an immutable source witness."""

    visit_id: int
    snapshot: MapSnapshot
    coordinate_frame_id: str
    source_manifest_sha256: str
    map_voxel_size_m: float
    observed_frame_start: int
    observed_frame_end: int
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        visit_id = _integer(self.visit_id, "visit_id")
        if visit_id not in {0, 1}:
            raise ValueError("visit_id must be exactly 0 or 1")
        if not isinstance(self.snapshot, MapSnapshot):
            raise TypeError("snapshot must be a MapSnapshot")
        if self.snapshot.scope != "current":
            raise ValueError("visit snapshot scope must be current")
        coordinate_frame_id = _nonempty_string(
            self.coordinate_frame_id, "coordinate_frame_id"
        )
        manifest_sha256 = _sha256(
            self.source_manifest_sha256, "source_manifest_sha256"
        )
        voxel_size = _finite_float(
            self.map_voxel_size_m, "map_voxel_size_m", minimum=np.nextafter(0.0, 1.0)
        )
        start = _integer(self.observed_frame_start, "observed_frame_start")
        end = _integer(self.observed_frame_end, "observed_frame_end")
        if end < start:
            raise ValueError("observed_frame_end cannot precede observed_frame_start")
        object.__setattr__(self, "visit_id", visit_id)
        object.__setattr__(self, "coordinate_frame_id", coordinate_frame_id)
        object.__setattr__(self, "source_manifest_sha256", manifest_sha256)
        object.__setattr__(self, "map_voxel_size_m", voxel_size)
        object.__setattr__(self, "observed_frame_start", start)
        object.__setattr__(self, "observed_frame_end", end)
        object.__setattr__(self, "snapshot_sha256", snapshot_content_sha256(self.snapshot))

    def assert_unchanged(self) -> None:
        if snapshot_content_sha256(self.snapshot) != self.snapshot_sha256:
            raise ValueError("visit snapshot changed after contract construction")


@dataclass(frozen=True, slots=True)
class OviEntitySemanticEvidence:
    """OVI-owned semantics kept outside the ReScene neural feature tensor."""

    visit_id: int
    entity_id: str
    semantic_label: str | None
    semantic_score: float
    semantic_embedding: np.ndarray | None

    def __post_init__(self) -> None:
        visit_id = _integer(self.visit_id, "semantic visit_id")
        if visit_id not in {0, 1}:
            raise ValueError("semantic visit_id must be exactly 0 or 1")
        entity_id = _nonempty_string(self.entity_id, "semantic entity_id")
        label = (
            None
            if self.semantic_label is None
            else _nonempty_string(self.semantic_label, "semantic_label")
        )
        score = _finite_float(self.semantic_score, "semantic_score")
        embedding = None
        if self.semantic_embedding is not None:
            embedding = _readonly_array(
                self.semantic_embedding,
                "semantic_embedding",
                dtype=np.float32,
                ndim=1,
            )
            if not len(embedding):
                raise ValueError("semantic_embedding cannot be empty")
        object.__setattr__(self, "visit_id", visit_id)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(self, "semantic_score", score)
        object.__setattr__(self, "semantic_embedding", embedding)


def validate_visit_pair(t0: VisitMap, t1: VisitMap) -> None:
    """Validate the shared frame and independence boundary for two visits."""

    if not isinstance(t0, VisitMap) or not isinstance(t1, VisitMap):
        raise TypeError("t0 and t1 must be VisitMap values")
    if (t0.visit_id, t1.visit_id) != (0, 1):
        raise ValueError("visit pair must be ordered as t0 then t1")
    t0.assert_unchanged()
    t1.assert_unchanged()
    if t0.snapshot is t1.snapshot:
        raise ValueError("t0 and t1 must be independently built snapshots")
    if t0.snapshot.scene_id != t1.snapshot.scene_id:
        raise ValueError("visit pair must describe one scene")
    if t0.coordinate_frame_id != t1.coordinate_frame_id:
        raise ValueError("visit pair coordinate frame must match")
    if t0.source_manifest_sha256 != t1.source_manifest_sha256:
        raise ValueError("visit pair source manifest must match")
    if t0.map_voxel_size_m != t1.map_voxel_size_m:
        raise ValueError("visit pair mapping voxel size must match")
    if t0.observed_frame_end >= t1.observed_frame_start:
        raise ValueError("visit windows must be non-overlapping and ordered")


@dataclass(frozen=True, slots=True)
class NeuralSampleMap:
    """Two-visit neural tokens with lossless CSR provenance to OVI samples."""

    coordinates_xyzt: np.ndarray
    features: np.ndarray
    visit_ids: np.ndarray
    source_visit_ids: np.ndarray
    source_entity_ids: tuple[str, ...]
    source_point_indices: np.ndarray
    source_to_token_offsets: np.ndarray
    neural_voxel_size_m: float
    feature_schema: str
    coordinate_frame_id: str
    source_manifest_sha256: str
    source_visit_map_sha256: tuple[str, str]
    entity_semantics: tuple[OviEntitySemanticEvidence, ...] = ()

    def __post_init__(self) -> None:
        coordinates = _readonly_array(
            self.coordinates_xyzt,
            "coordinates_xyzt",
            dtype=np.float64,
            ndim=2,
        )
        if coordinates.shape[1:] != (4,):
            raise ValueError("coordinates_xyzt shape must be (N, 4)")
        features = _readonly_array(
            self.features, "features", dtype=np.float32, ndim=2
        )
        if features.shape[0] != coordinates.shape[0] or features.shape[1] < 1:
            raise ValueError("features shape must be (N, F) with F >= 1")
        visits = _readonly_integer_array(self.visit_ids, "visit_ids", dtype=np.int8)
        if visits.shape != (coordinates.shape[0],):
            raise ValueError("visit_ids shape must be (N,)")
        if set(visits.tolist()) != {0, 1}:
            raise ValueError("visit_ids must contain both exact visits 0 and 1")
        if not np.array_equal(coordinates[:, 3], visits.astype(np.float64)):
            raise ValueError("coordinates_xyzt time must equal the exact visit ID")

        source_visits = _readonly_integer_array(
            self.source_visit_ids, "source_visit_ids", dtype=np.int8
        )
        point_indices = _readonly_integer_array(
            self.source_point_indices, "source_point_indices", dtype=np.int64
        )
        offsets = _readonly_integer_array(
            self.source_to_token_offsets,
            "source_to_token_offsets",
            dtype=np.int64,
        )
        entity_ids = tuple(
            _nonempty_string(item, "source_entity_ids item")
            for item in self.source_entity_ids
        )
        contributor_count = len(entity_ids)
        if source_visits.shape != (contributor_count,) or point_indices.shape != (
            contributor_count,
        ):
            raise ValueError("source contributor arrays must have equal lengths")
        if offsets.shape != (coordinates.shape[0] + 1,):
            raise ValueError("source_to_token_offsets shape must be (N + 1,)")
        if offsets[0] != 0 or offsets[-1] != contributor_count:
            raise ValueError("CSR offsets must start at zero and end at contributor count")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("every token must have a non-empty contributor span")
        if not np.array_equal(
            np.sort(point_indices), np.arange(contributor_count, dtype=np.int64)
        ):
            raise ValueError("source_point_indices must be a conserving permutation")
        for token_index in range(coordinates.shape[0]):
            start, end = int(offsets[token_index]), int(offsets[token_index + 1])
            if np.any(source_visits[start:end] != visits[token_index]):
                raise ValueError("a token contributor span mixes visits")
            if len(set(entity_ids[start:end])) != 1:
                raise ValueError("a token contributor span mixes OVI entities")

        voxel_size = _finite_float(
            self.neural_voxel_size_m,
            "neural_voxel_size_m",
            minimum=np.nextafter(0.0, 1.0),
        )
        feature_schema = _nonempty_string(self.feature_schema, "feature_schema")
        coordinate_frame_id = _nonempty_string(
            self.coordinate_frame_id, "coordinate_frame_id"
        )
        manifest_sha256 = _sha256(
            self.source_manifest_sha256, "source_manifest_sha256"
        )
        if not isinstance(self.source_visit_map_sha256, tuple) or len(
            self.source_visit_map_sha256
        ) != 2:
            raise ValueError("source_visit_map_sha256 must contain t0 and t1 hashes")
        visit_hashes = (
            _sha256(self.source_visit_map_sha256[0], "t0 map SHA-256"),
            _sha256(self.source_visit_map_sha256[1], "t1 map SHA-256"),
        )
        semantics = tuple(self.entity_semantics)
        if any(not isinstance(item, OviEntitySemanticEvidence) for item in semantics):
            raise ValueError("entity_semantics must contain OviEntitySemanticEvidence")
        semantics = tuple(sorted(semantics, key=lambda item: (item.visit_id, item.entity_id)))
        semantic_keys = tuple((item.visit_id, item.entity_id) for item in semantics)
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError("entity_semantics must be unique by visit and entity")
        token_keys = set(zip(visits.tolist(), self.token_entity_ids, strict=True))
        if semantics and set(semantic_keys) != token_keys:
            raise ValueError("entity_semantics must cover every token entity exactly once")

        object.__setattr__(self, "coordinates_xyzt", coordinates)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "visit_ids", visits)
        object.__setattr__(self, "source_visit_ids", source_visits)
        object.__setattr__(self, "source_entity_ids", entity_ids)
        object.__setattr__(self, "source_point_indices", point_indices)
        object.__setattr__(self, "source_to_token_offsets", offsets)
        object.__setattr__(self, "neural_voxel_size_m", voxel_size)
        object.__setattr__(self, "feature_schema", feature_schema)
        object.__setattr__(self, "coordinate_frame_id", coordinate_frame_id)
        object.__setattr__(self, "source_manifest_sha256", manifest_sha256)
        object.__setattr__(self, "source_visit_map_sha256", visit_hashes)
        object.__setattr__(self, "entity_semantics", semantics)

    @property
    def source_point_count(self) -> int:
        return len(self.source_entity_ids)

    @property
    def token_entity_ids(self) -> tuple[str, ...]:
        return tuple(
            self.source_entity_ids[int(self.source_to_token_offsets[index])]
            for index in range(self.coordinates_xyzt.shape[0])
        )

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.coordinates_xyzt,
            self.features,
            self.visit_ids,
            self.source_visit_ids,
            self.source_point_indices,
            self.source_to_token_offsets,
        )

    def content_sha256(self) -> str:
        semantics = [
            {
                "visit_id": item.visit_id,
                "entity_id": item.entity_id,
                "semantic_label": item.semantic_label,
                "semantic_score": item.semantic_score,
                "semantic_embedding": item.semantic_embedding,
            }
            for item in self.entity_semantics
        ]
        return _digest(
            {
                "coordinates_xyzt": self.coordinates_xyzt,
                "features": self.features,
                "visit_ids": self.visit_ids,
                "source_visit_ids": self.source_visit_ids,
                "source_entity_ids": self.source_entity_ids,
                "source_point_indices": self.source_point_indices,
                "source_to_token_offsets": self.source_to_token_offsets,
                "neural_voxel_size_m": self.neural_voxel_size_m,
                "feature_schema": self.feature_schema,
                "coordinate_frame_id": self.coordinate_frame_id,
                "source_manifest_sha256": self.source_manifest_sha256,
                "source_visit_map_sha256": self.source_visit_map_sha256,
                "entity_semantics": semantics,
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class TemporalQueryEvidence:
    """Backend-neutral temporal query masks and confidence evidence."""

    status: EvidenceStatus
    backend_name: str
    backend_config_sha256: str
    pair_sha256: str
    temporal_query_ids: tuple[str, ...]
    query_masks: np.ndarray | None
    token_scores: np.ndarray | None
    query_scores: np.ndarray | None
    checkpoint_sha256: str | None
    ranking_eligible: bool
    runtime_s: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError(f"unsupported temporal evidence status: {self.status}")
        backend_name = _nonempty_string(self.backend_name, "backend_name")
        backend_hash = _sha256(
            self.backend_config_sha256, "backend_config_sha256"
        )
        pair_hash = _sha256(self.pair_sha256, "pair_sha256")
        query_ids = tuple(
            _nonempty_string(item, "temporal_query_ids item")
            for item in self.temporal_query_ids
        )
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("temporal_query_ids must be unique")
        runtime = _finite_float(self.runtime_s, "runtime_s", minimum=0.0)
        peak_memory = _integer(self.peak_memory_bytes, "peak_memory_bytes")
        if type(self.ranking_eligible) is not bool:
            raise ValueError("ranking_eligible must be boolean")
        checkpoint = (
            None
            if self.checkpoint_sha256 is None
            else _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        )

        arrays = (self.query_masks, self.token_scores, self.query_scores)
        if self.status != "PASS":
            if query_ids or any(value is not None for value in arrays):
                raise ValueError("blocked evidence cannot carry query predictions")
            if self.ranking_eligible:
                raise ValueError("blocked evidence cannot be ranking eligible")
            masks = token_scores = query_scores = None
        else:
            if any(value is None for value in arrays):
                raise ValueError("PASS evidence requires masks and scores")
            if not query_ids:
                raise ValueError("PASS evidence requires at least one temporal query")
            raw_masks = np.asarray(self.query_masks)
            if raw_masks.dtype != np.bool_:
                raise ValueError("query_masks must be boolean")
            masks = _readonly_array(
                self.query_masks, "query_masks", dtype=np.bool_, ndim=2, finite=False
            )
            token_scores = _readonly_array(
                self.token_scores, "token_scores", dtype=np.float32, ndim=2
            )
            query_scores = _readonly_array(
                self.query_scores, "query_scores", dtype=np.float32, ndim=1
            )
            expected_q = len(query_ids)
            if masks.shape[0] != expected_q or token_scores.shape != masks.shape:
                raise ValueError("query mask and token score shapes must be (Q, N)")
            if masks.shape[1] < 1 or query_scores.shape != (expected_q,):
                raise ValueError("query_scores shape must be (Q,) and N must be positive")
            if np.any(token_scores < 0.0) or np.any(token_scores > 1.0):
                raise ValueError("token_scores must be in [0, 1]")
            if np.any(query_scores < 0.0) or np.any(query_scores > 1.0):
                raise ValueError("query_scores must be in [0, 1]")

        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(self, "backend_config_sha256", backend_hash)
        object.__setattr__(self, "pair_sha256", pair_hash)
        object.__setattr__(self, "temporal_query_ids", query_ids)
        object.__setattr__(self, "query_masks", masks)
        object.__setattr__(self, "token_scores", token_scores)
        object.__setattr__(self, "query_scores", query_scores)
        object.__setattr__(self, "checkpoint_sha256", checkpoint)
        object.__setattr__(self, "runtime_s", runtime)
        object.__setattr__(self, "peak_memory_bytes", peak_memory)

    @classmethod
    def blocked(
        cls,
        *,
        status: EvidenceStatus,
        backend_name: str,
        backend_config_sha256: str,
        pair_sha256: str,
    ) -> TemporalQueryEvidence:
        if status == "PASS":
            raise ValueError("blocked evidence status cannot be PASS")
        return cls(
            status=status,
            backend_name=backend_name,
            backend_config_sha256=backend_config_sha256,
            pair_sha256=pair_sha256,
            temporal_query_ids=(),
            query_masks=None,
            token_scores=None,
            query_scores=None,
            checkpoint_sha256=None,
            ranking_eligible=False,
            runtime_s=0.0,
            peak_memory_bytes=0,
        )

    def content_sha256(self) -> str:
        return _digest(
            {
                "status": self.status,
                "backend_name": self.backend_name,
                "backend_config_sha256": self.backend_config_sha256,
                "pair_sha256": self.pair_sha256,
                "temporal_query_ids": self.temporal_query_ids,
                "query_masks": self.query_masks,
                "token_scores": self.token_scores,
                "query_scores": self.query_scores,
                "checkpoint_sha256": self.checkpoint_sha256,
                "ranking_eligible": self.ranking_eligible,
                "runtime_s": self.runtime_s,
                "peak_memory_bytes": self.peak_memory_bytes,
            }
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TemporalQueryEvidence) and (
            self.content_sha256() == other.content_sha256()
        )


@dataclass(frozen=True, slots=True)
class PairRelation:
    """One explicit many-sided relation between OVI entities across visits."""

    temporal_query_id: int | str
    t0_entity_ids: tuple[str, ...]
    t1_entity_ids: tuple[str, ...]
    state: RelationState
    query_confidence: float
    evidence: Mapping[str, float]
    identity_source: IdentitySource

    def __post_init__(self) -> None:
        query_id = _nonempty_string(str(self.temporal_query_id), "temporal_query_id")
        t0_ids = tuple(
            _nonempty_string(item, "t0_entity_ids item") for item in self.t0_entity_ids
        )
        t1_ids = tuple(
            _nonempty_string(item, "t1_entity_ids item") for item in self.t1_entity_ids
        )
        if len(t0_ids) != len(set(t0_ids)) or len(t1_ids) != len(set(t1_ids)):
            raise ValueError("relation entity IDs must be unique within each visit")
        if self.state not in _RELATION_STATES:
            raise ValueError(f"unsupported relation state: {self.state}")
        cardinality = (len(t0_ids), len(t1_ids))
        if self.state in {"persistent_static", "persistent_moved"} and cardinality != (1, 1):
            raise ValueError("persistent relation must have 1-to-1 cardinality")
        if self.state == "appeared" and cardinality != (0, 1):
            raise ValueError("appeared relation must have 0-to-1 cardinality")
        if self.state == "removed_candidate" and cardinality != (1, 0):
            raise ValueError("removed candidate relation must have 1-to-0 cardinality")
        if self.state == "split" and not (cardinality[0] == 1 and cardinality[1] > 1):
            raise ValueError("split relation must have 1-to-N cardinality")
        if self.state == "merge" and not (cardinality[0] > 1 and cardinality[1] == 1):
            raise ValueError("merge relation must have N-to-1 cardinality")
        if self.state == "uncertain" and cardinality == (0, 0):
            raise ValueError("uncertain relation must reference at least one entity")
        confidence = _finite_float(self.query_confidence, "query_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("query_confidence must be in [0, 1]")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")
        evidence = {
            _nonempty_string(key, "evidence key"): _finite_float(
                value, f"evidence.{key}"
            )
            for key, value in self.evidence.items()
        }
        if self.identity_source not in _IDENTITY_SOURCES:
            raise ValueError(f"unsupported identity source: {self.identity_source}")
        object.__setattr__(self, "temporal_query_id", query_id)
        object.__setattr__(self, "t0_entity_ids", t0_ids)
        object.__setattr__(self, "t1_entity_ids", t1_ids)
        object.__setattr__(self, "query_confidence", confidence)
        object.__setattr__(self, "evidence", MappingProxyType(dict(sorted(evidence.items()))))


@dataclass(frozen=True, slots=True)
class CurrentCompositionDecision:
    """One t1-first decision with explicit geometry and semantic authority."""

    source_entity_id: str
    source_visit: int
    decision: CompositionAction
    visibility_status: VisibilityState
    visibility_score: float
    relation_id: str | None
    geometry_source: GeometrySource | None
    identity_source: IdentitySource
    state_source: StateSource
    semantic_source: SemanticSource | None

    def __post_init__(self) -> None:
        entity_id = _nonempty_string(self.source_entity_id, "source_entity_id")
        source_visit = _integer(self.source_visit, "source_visit")
        if source_visit not in {0, 1}:
            raise ValueError("source_visit must be exactly 0 or 1")
        if self.decision not in _COMPOSITION_ACTIONS:
            raise ValueError(f"unsupported composition decision: {self.decision}")
        if self.visibility_status not in _VISIBILITY_STATES:
            raise ValueError(f"unsupported visibility status: {self.visibility_status}")
        score = _finite_float(self.visibility_score, "visibility_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("visibility_score must be in [0, 1]")
        relation_id = (
            None
            if self.relation_id is None
            else _nonempty_string(self.relation_id, "relation_id")
        )
        if self.geometry_source not in {None, "ovi_t0", "ovi_t1"}:
            raise ValueError("geometry_source must be OVI t0, OVI t1, or absent")
        if self.identity_source not in _IDENTITY_SOURCES:
            raise ValueError(f"unsupported identity source: {self.identity_source}")
        if self.state_source not in _STATE_SOURCES:
            raise ValueError(f"unsupported state source: {self.state_source}")
        if self.semantic_source not in {None, *_SEMANTIC_SOURCES}:
            raise ValueError("semantic_source must remain under OVI authority")

        required = {
            "emit_t1": (1, "occupied", "ovi_t1", {"ovi_t1", "fused_ovi"}),
            "retain_t0_unobserved": (
                0,
                "unobserved",
                "ovi_t0",
                {"ovi_t0", "fused_ovi"},
            ),
            "retain_t0_occluded": (
                0,
                "occluded",
                "ovi_t0",
                {"ovi_t0", "fused_ovi"},
            ),
        }
        if self.decision in required:
            expected_visit, expected_visibility, expected_geometry, allowed_semantics = required[
                self.decision
            ]
            if (
                source_visit != expected_visit
                or self.visibility_status != expected_visibility
                or self.geometry_source != expected_geometry
                or self.semantic_source not in allowed_semantics
            ):
                raise ValueError(f"{self.decision} authority contract mismatch")
        elif self.decision == "suppress_t0_visible_free":
            if (
                source_visit != 0
                or self.visibility_status != "visible_free"
                or self.geometry_source is not None
                or self.semantic_source is not None
                or self.state_source != "t1_visibility"
            ):
                raise ValueError("t0 suppression requires t1 visible-free authority")
        elif self.geometry_source is not None:
            expected_geometry = "ovi_t0" if source_visit == 0 else "ovi_t1"
            if self.geometry_source != expected_geometry:
                raise ValueError("uncertain geometry must match its OVI source visit")
        object.__setattr__(self, "source_entity_id", entity_id)
        object.__setattr__(self, "source_visit", source_visit)
        object.__setattr__(self, "visibility_score", score)
        object.__setattr__(self, "relation_id", relation_id)
