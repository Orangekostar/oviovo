"""t1-first composition of dense OVI visit maps with signed visibility."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot
from src.oviv2.two_visit_contracts import (
    CurrentCompositionDecision,
    PairRelation,
    VisitMap,
    snapshot_content_sha256,
    validate_visit_pair,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_VISIBILITY_STATES = ("occupied", "visible_free", "occluded", "unobserved")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")
    return value


def _points(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (N, 3) with finite values")
    return array


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    quantized = np.floor(points / voxel_size_m)
    limit = np.iinfo(np.int64).max
    if not np.all(np.isfinite(quantized)) or np.any(np.abs(quantized) > limit):
        raise ValueError("composition points exceed int64 voxel range")
    return quantized.astype(np.int64)


@dataclass(frozen=True, slots=True)
class SignedVisibilityGrid:
    voxel_size_m: float
    voxel_keys: np.ndarray
    statuses: tuple[str, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.voxel_size_m, bool) or not isinstance(
            self.voxel_size_m, (int, float)
        ):
            raise ValueError("voxel_size_m must be numeric")
        voxel_size = float(self.voxel_size_m)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        raw_keys = np.asarray(self.voxel_keys)
        if raw_keys.ndim != 2 or raw_keys.shape[1:] != (3,):
            raise ValueError("voxel_keys must have shape (N, 3)")
        if raw_keys.size and not np.issubdtype(raw_keys.dtype, np.integer):
            raise ValueError("voxel_keys must contain integers")
        keys = np.array(raw_keys, dtype=np.int64, copy=True)
        statuses = tuple(str(item) for item in self.statuses)
        if len(statuses) != len(keys) or any(
            item not in _VISIBILITY_STATES for item in statuses
        ):
            raise ValueError("statuses must classify every visibility voxel")
        if len(keys):
            order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
            if not np.array_equal(order, np.arange(len(keys))) or np.any(
                np.all(keys[1:] == keys[:-1], axis=1)
            ):
                raise ValueError("voxel_keys must be sorted and unique")
        keys.setflags(write=False)
        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(self, "voxel_keys", keys)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(
            self, "source_sha256", _sha256(self.source_sha256, "source_sha256")
        )

    @classmethod
    def empty(cls, voxel_size_m: float, source_sha256: str) -> SignedVisibilityGrid:
        return cls(
            voxel_size_m=voxel_size_m,
            voxel_keys=np.empty((0, 3), dtype=np.int64),
            statuses=(),
            source_sha256=source_sha256,
        )

    @classmethod
    def from_payload(cls, payload: object, source_sha256: str) -> SignedVisibilityGrid:
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "status",
            "voxel_size_m",
            "cells",
        }:
            raise ValueError("signed visibility payload schema is invalid")
        if payload["schema_version"] != 1 or payload["status"] != "PASS":
            raise ValueError("signed visibility payload identity is invalid")
        cells = payload["cells"]
        if not isinstance(cells, list):
            raise ValueError("signed visibility cells must be a list")
        records: list[tuple[tuple[int, int, int], str]] = []
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != {"voxel_key", "status"}:
                raise ValueError("signed visibility cell is invalid")
            key = cell["voxel_key"]
            if (
                not isinstance(key, list)
                or len(key) != 3
                or any(type(item) is not int for item in key)
            ):
                raise ValueError("signed visibility voxel key is invalid")
            records.append((tuple(key), cell["status"]))
        records.sort(key=lambda item: item[0])
        return cls(
            voxel_size_m=payload["voxel_size_m"],
            voxel_keys=np.asarray(
                [item[0] for item in records], dtype=np.int64
            ).reshape(-1, 3),
            statuses=tuple(item[1] for item in records),
            source_sha256=source_sha256,
        )

    def as_mapping(self) -> dict[tuple[int, int, int], str]:
        return {
            tuple(int(item) for item in key): status
            for key, status in zip(self.voxel_keys, self.statuses, strict=True)
        }


@dataclass(frozen=True, slots=True)
class CompositionConfig:
    composition_voxel_size_m: float = 0.05
    method_name: str = "OVI-MAP + two-visit current composer"
    object_semantic_labels: frozenset[str] = frozenset()
    minimum_entity_visible_free_fraction: float = 0.8
    minimum_entity_visible_free_voxels: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.composition_voxel_size_m, bool) or not isinstance(
            self.composition_voxel_size_m, (int, float)
        ):
            raise ValueError("composition_voxel_size_m must be numeric")
        voxel = float(self.composition_voxel_size_m)
        if not math.isfinite(voxel) or voxel <= 0.0:
            raise ValueError("composition_voxel_size_m must be finite and positive")
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be non-empty")
        if not isinstance(self.object_semantic_labels, frozenset) or any(
            not isinstance(label, str) or not label.strip()
            for label in self.object_semantic_labels
        ):
            raise ValueError("object_semantic_labels must be a frozenset of labels")
        fraction = self.minimum_entity_visible_free_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("minimum_entity_visible_free_fraction must be in [0, 1]")
        minimum_voxels = self.minimum_entity_visible_free_voxels
        if type(minimum_voxels) is not int or minimum_voxels < 1:
            raise ValueError("minimum_entity_visible_free_voxels must be positive")
        object.__setattr__(self, "composition_voxel_size_m", voxel)
        object.__setattr__(self, "method_name", self.method_name.strip())
        object.__setattr__(
            self,
            "object_semantic_labels",
            frozenset(label.strip() for label in self.object_semantic_labels),
        )
        object.__setattr__(
            self, "minimum_entity_visible_free_fraction", float(fraction)
        )


@dataclass(frozen=True, slots=True)
class CompositionPointGroup:
    decision: CurrentCompositionDecision
    source_point_indices: np.ndarray
    source_snapshot_sha256: str
    output_entity_id: str | None
    output_point_start: int | None
    output_point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, CurrentCompositionDecision):
            raise TypeError("decision must be CurrentCompositionDecision")
        raw = np.asarray(self.source_point_indices)
        if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("source_point_indices must be a non-empty integer vector")
        indices = np.array(raw, dtype=np.int64, copy=True)
        if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError(
                "source_point_indices must be sorted, unique, and non-negative"
            )
        indices.setflags(write=False)
        count = int(self.output_point_count)
        emitted = self.decision.geometry_source is not None
        if emitted:
            if (
                not isinstance(self.output_entity_id, str)
                or not self.output_entity_id
                or type(self.output_point_start) is not int
                or self.output_point_start < 0
                or count != len(indices)
            ):
                raise ValueError("emitted point group output span is invalid")
        elif (
            self.output_entity_id is not None
            or self.output_point_start is not None
            or count != 0
        ):
            raise ValueError("suppressed point group cannot have an output span")
        object.__setattr__(self, "source_point_indices", indices)
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _sha256(self.source_snapshot_sha256, "source_snapshot_sha256"),
        )
        object.__setattr__(self, "output_point_count", count)


@dataclass(frozen=True, slots=True)
class TwoVisitCurrentMap:
    snapshot: MapSnapshot
    provenance: tuple[CompositionPointGroup, ...]
    source_visit_map_sha256: tuple[str, str]
    source_manifest_sha256: str
    visibility_source_sha256: str
    relation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.snapshot, MapSnapshot)
            or self.snapshot.scope != "current"
        ):
            raise ValueError("current map snapshot must have current scope")
        if not self.provenance or any(
            not isinstance(item, CompositionPointGroup) for item in self.provenance
        ):
            raise ValueError("current map requires point-group provenance")
        if len(self.source_visit_map_sha256) != 2:
            raise ValueError("source_visit_map_sha256 must bind t0 and t1")
        object.__setattr__(
            self,
            "source_visit_map_sha256",
            tuple(
                _sha256(item, "source visit map SHA-256")
                for item in self.source_visit_map_sha256
            ),
        )
        object.__setattr__(
            self,
            "source_manifest_sha256",
            _sha256(self.source_manifest_sha256, "source_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "visibility_source_sha256",
            _sha256(self.visibility_source_sha256, "visibility_source_sha256"),
        )
        if len(self.relation_ids) != len(set(self.relation_ids)):
            raise ValueError("relation_ids must be unique")

    @property
    def decisions(self) -> tuple[CurrentCompositionDecision, ...]:
        return tuple(item.decision for item in self.provenance)

    def content_sha256(self) -> str:
        records = [
            {
                "decision": asdict(item.decision),
                "source_point_indices": item.source_point_indices.tolist(),
                "source_snapshot_sha256": item.source_snapshot_sha256,
                "output_entity_id": item.output_entity_id,
                "output_point_start": item.output_point_start,
                "output_point_count": item.output_point_count,
            }
            for item in self.provenance
        ]
        payload = {
            "snapshot_sha256": snapshot_content_sha256(self.snapshot),
            "provenance": records,
            "source_visit_map_sha256": self.source_visit_map_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "visibility_source_sha256": self.visibility_source_sha256,
            "relation_ids": self.relation_ids,
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


@dataclass(slots=True)
class _EntityBuilder:
    output_id: str
    semantic_source: EntityPrediction
    chunks: list[np.ndarray]

    @property
    def point_count(self) -> int:
        return sum(len(item) for item in self.chunks)


def _relation_indexes(
    relations: tuple[PairRelation, ...],
    t0_ids: set[str],
    t1_ids: set[str],
) -> tuple[dict[str, PairRelation], dict[str, PairRelation]]:
    by_t0: dict[str, PairRelation] = {}
    by_t1: dict[str, PairRelation] = {}
    query_ids: set[str] = set()
    for relation in relations:
        if not isinstance(relation, PairRelation):
            raise TypeError("relations must contain PairRelation values")
        if relation.temporal_query_id in query_ids:
            raise ValueError("relation query IDs must be unique")
        query_ids.add(str(relation.temporal_query_id))
        if not set(relation.t0_entity_ids).issubset(t0_ids) or not set(
            relation.t1_entity_ids
        ).issubset(t1_ids):
            raise ValueError("relation references an entity outside the visit maps")
        for entity_id in relation.t0_entity_ids:
            if entity_id in by_t0:
                raise ValueError("t0 entity appears in multiple relations")
            by_t0[entity_id] = relation
        for entity_id in relation.t1_entity_ids:
            if entity_id in by_t1:
                raise ValueError("t1 entity appears in multiple relations")
            by_t1[entity_id] = relation
    return by_t0, by_t1


def _decision(
    *,
    source_entity_id: str,
    source_visit: int,
    action: str,
    visibility: str,
    relation: PairRelation | None,
    visibility_score: float | None = None,
) -> CurrentCompositionDecision:
    retained = action in {"emit_t1", "retain_t0_occluded", "retain_t0_unobserved"}
    semantic = ("ovi_t1" if source_visit == 1 else "ovi_t0") if retained else None
    geometry = ("ovi_t1" if source_visit == 1 else "ovi_t0") if retained else None
    return CurrentCompositionDecision(
        source_entity_id=source_entity_id,
        source_visit=source_visit,
        decision=action,
        visibility_status=visibility,
        visibility_score=(0.0 if visibility == "unobserved" else 1.0)
        if visibility_score is None
        else visibility_score,
        relation_id=None if relation is None else str(relation.temporal_query_id),
        geometry_source=geometry,
        identity_source="unmatched" if relation is None else relation.identity_source,
        state_source=(
            "fallback"
            if action in {"retain_t0_occluded", "retain_t0_unobserved"}
            else "t1_visibility"
        ),
        semantic_source=semantic,
    )


def _status_groups(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    occupied_keys: set[tuple[int, int, int]],
    visibility: dict[tuple[int, int, int], str],
) -> tuple[tuple[str, np.ndarray], ...]:
    keys = _voxel_keys(points, voxel_size_m)
    grouped: dict[str, list[int]] = {item: [] for item in _VISIBILITY_STATES}
    for index, key_array in enumerate(keys):
        key = tuple(int(item) for item in key_array)
        status = (
            "occupied" if key in occupied_keys else visibility.get(key, "unobserved")
        )
        grouped[status].append(index)
    return tuple(
        (status, np.asarray(grouped[status], dtype=np.int64))
        for status in _VISIBILITY_STATES
        if grouped[status]
    )


def _action(status: str) -> str:
    return {
        "occupied": "suppress_t0_occupied_by_t1",
        "visible_free": "suppress_t0_visible_free",
        "occluded": "retain_t0_occluded",
        "unobserved": "retain_t0_unobserved",
    }[status]


def _entity_visible_free_evidence(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    occupied_keys: set[tuple[int, int, int]],
    visibility: dict[tuple[int, int, int], str],
) -> tuple[int, float]:
    informative: set[tuple[int, int, int]] = set()
    visible_free: set[tuple[int, int, int]] = set()
    for key_array in _voxel_keys(points, voxel_size_m):
        key = tuple(int(item) for item in key_array)
        status = (
            "occupied" if key in occupied_keys else visibility.get(key, "unobserved")
        )
        if status in {"visible_free", "occupied"}:
            informative.add(key)
        if status == "visible_free":
            visible_free.add(key)
    fraction = len(visible_free) / len(informative) if informative else 0.0
    return len(visible_free), fraction


def _fallback_output_id(entity_id: str, occupied: set[str]) -> str:
    output_id = f"t0-fallback:{entity_id}"
    if output_id in occupied:
        raise ValueError(f"fallback entity ID collides with t1 entity: {output_id}")
    return output_id


def compose_current_map(
    t0: VisitMap,
    t1: VisitMap,
    relations: tuple[PairRelation, ...],
    signed_visibility: SignedVisibilityGrid,
    config: CompositionConfig,
) -> TwoVisitCurrentMap:
    """Compose a dense current map with t1 visibility as deletion authority."""

    if not isinstance(signed_visibility, SignedVisibilityGrid):
        raise TypeError("signed_visibility must be SignedVisibilityGrid")
    if not isinstance(config, CompositionConfig):
        raise TypeError("config must be CompositionConfig")
    if signed_visibility.voxel_size_m != config.composition_voxel_size_m:
        raise ValueError("visibility and composition voxel sizes must match")
    validate_visit_pair(t0, t1)
    if not isinstance(relations, tuple):
        raise TypeError("relations must be a tuple")
    t0_entities = {item.entity_id: item for item in t0.snapshot.entities}
    t1_entities = {item.entity_id: item for item in t1.snapshot.entities}
    if any(
        not len(item.points_xyz)
        for item in (*t0.snapshot.entities, *t1.snapshot.entities)
    ):
        raise ValueError("OVI entities must contain at least one surface point")
    by_t0, by_t1 = _relation_indexes(relations, set(t0_entities), set(t1_entities))
    before = (
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    )

    t1_point_sets = [item.points_xyz for item in t1.snapshot.entities]
    if t1.snapshot.background_xyz is not None:
        t1_point_sets.append(t1.snapshot.background_xyz)
    occupied_keys = {
        tuple(int(item) for item in key)
        for points in t1_point_sets
        for key in _voxel_keys(
            _points(points, "t1 OVI points"), config.composition_voxel_size_m
        )
    }
    visibility = signed_visibility.as_mapping()
    builders: dict[str, _EntityBuilder] = {}
    provenance: list[CompositionPointGroup] = []

    for entity_id in sorted(t1_entities):
        entity = t1_entities[entity_id]
        points = np.asarray(entity.points_xyz, dtype=np.float32)
        builders[entity_id] = _EntityBuilder(entity_id, entity, [points])
        relation = by_t1.get(entity_id)
        provenance.append(
            CompositionPointGroup(
                decision=_decision(
                    source_entity_id=entity_id,
                    source_visit=1,
                    action="emit_t1",
                    visibility="occupied",
                    relation=relation,
                ),
                source_point_indices=np.arange(len(points), dtype=np.int64),
                source_snapshot_sha256=t1.snapshot_sha256,
                output_entity_id=entity_id,
                output_point_start=0,
                output_point_count=len(points),
            )
        )

    background_chunks: list[np.ndarray] = []
    background_count = 0
    if t1.snapshot.background_xyz is not None and len(t1.snapshot.background_xyz):
        points = np.asarray(t1.snapshot.background_xyz, dtype=np.float32)
        background_chunks.append(points)
        background_count = len(points)
        provenance.append(
            CompositionPointGroup(
                decision=_decision(
                    source_entity_id="__background__",
                    source_visit=1,
                    action="emit_t1",
                    visibility="occupied",
                    relation=None,
                ),
                source_point_indices=np.arange(len(points), dtype=np.int64),
                source_snapshot_sha256=t1.snapshot_sha256,
                output_entity_id="__background__",
                output_point_start=0,
                output_point_count=len(points),
            )
        )

    for entity_id in sorted(t0_entities):
        entity = t0_entities[entity_id]
        points = np.asarray(entity.points_xyz, dtype=np.float32)
        relation = by_t0.get(entity_id)
        groups = _status_groups(
            points,
            voxel_size_m=config.composition_voxel_size_m,
            occupied_keys=occupied_keys,
            visibility=visibility,
        )
        visible_free_voxels, visible_free_fraction = _entity_visible_free_evidence(
            points,
            voxel_size_m=config.composition_voxel_size_m,
            occupied_keys=occupied_keys,
            visibility=visibility,
        )
        suppress_residue = (
            entity.semantic_label in config.object_semantic_labels
            and visible_free_voxels >= config.minimum_entity_visible_free_voxels
            and visible_free_fraction >= config.minimum_entity_visible_free_fraction
        )
        for status, indices in groups:
            action = _action(status)
            if suppress_residue and status in {"occluded", "unobserved"}:
                action = "suppress_t0_entity_visible_free"
            output_id: str | None = None
            output_start: int | None = None
            output_count = 0
            if action in {"retain_t0_occluded", "retain_t0_unobserved"}:
                if (
                    relation is not None
                    and relation.state in {"persistent_static", "persistent_moved"}
                    and len(relation.t1_entity_ids) == 1
                ):
                    output_id = relation.t1_entity_ids[0]
                else:
                    output_id = _fallback_output_id(entity_id, set(t1_entities))
                if output_id not in builders:
                    builders[output_id] = _EntityBuilder(output_id, entity, [])
                builder = builders[output_id]
                output_start = builder.point_count
                chunk = points[indices]
                builder.chunks.append(chunk)
                output_count = len(chunk)
            provenance.append(
                CompositionPointGroup(
                    decision=_decision(
                        source_entity_id=entity_id,
                        source_visit=0,
                        action=action,
                        visibility=status,
                        relation=relation,
                        visibility_score=(
                            visible_free_fraction
                            if action == "suppress_t0_entity_visible_free"
                            else None
                        ),
                    ),
                    source_point_indices=indices,
                    source_snapshot_sha256=t0.snapshot_sha256,
                    output_entity_id=output_id,
                    output_point_start=output_start,
                    output_point_count=output_count,
                )
            )

    if t0.snapshot.background_xyz is not None and len(t0.snapshot.background_xyz):
        points = np.asarray(t0.snapshot.background_xyz, dtype=np.float32)
        for status, indices in _status_groups(
            points,
            voxel_size_m=config.composition_voxel_size_m,
            occupied_keys=occupied_keys,
            visibility=visibility,
        ):
            action = _action(status)
            output_start = None
            output_count = 0
            output_id = None
            if action in {"retain_t0_occluded", "retain_t0_unobserved"}:
                output_id = "__background__"
                output_start = background_count
                chunk = points[indices]
                background_chunks.append(chunk)
                background_count += len(chunk)
                output_count = len(chunk)
            provenance.append(
                CompositionPointGroup(
                    decision=_decision(
                        source_entity_id="__background__",
                        source_visit=0,
                        action=action,
                        visibility=status,
                        relation=None,
                    ),
                    source_point_indices=indices,
                    source_snapshot_sha256=t0.snapshot_sha256,
                    output_entity_id=output_id,
                    output_point_start=output_start,
                    output_point_count=output_count,
                )
            )

    output_entities: list[EntityPrediction] = []
    for output_id, builder in builders.items():
        if not builder.chunks:
            continue
        points = np.concatenate(builder.chunks, axis=0)
        source = builder.semantic_source
        metadata = dict(source.metadata)
        metadata.update(
            {
                "geometry_authority": (
                    "ovi_t1_with_ovi_t0_fallback"
                    if output_id in t1_entities and len(builder.chunks) > 1
                    else "ovi_t1"
                    if output_id in t1_entities
                    else "ovi_t0_fallback"
                ),
                "semantic_authority": "ovi_t1"
                if output_id in t1_entities
                else "ovi_t0",
                "two_visit_output_entity_id": output_id,
            }
        )
        output_entities.append(
            EntityPrediction(
                entity_id=output_id,
                points_xyz=points,
                semantic_embedding=source.semantic_embedding,
                semantic_label=source.semantic_label,
                semantic_score=source.semantic_score,
                lifecycle_state="current",
                first_seen=source.first_seen,
                last_seen=source.last_seen,
                metadata=metadata,
            )
        )

    background = (
        np.concatenate(background_chunks, axis=0).astype(np.float32, copy=False)
        if background_chunks
        else None
    )
    snapshot = MapSnapshot(
        method=config.method_name,
        scene_id=t1.snapshot.scene_id,
        timestamp=t1.snapshot.timestamp,
        entities=output_entities,
        background_xyz=background,
        scope="current",
        runtime={
            "composition_source_point_groups": float(len(provenance)),
            "composition_suppressed_point_count": float(
                sum(
                    len(item.source_point_indices)
                    for item in provenance
                    if item.output_point_count == 0
                )
            ),
        },
    )
    after = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    if before != after or before != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise ValueError("OVI source maps changed during current-map composition")
    return TwoVisitCurrentMap(
        snapshot=snapshot,
        provenance=tuple(provenance),
        source_visit_map_sha256=(t0.snapshot_sha256, t1.snapshot_sha256),
        source_manifest_sha256=t0.source_manifest_sha256,
        visibility_source_sha256=signed_visibility.source_sha256,
        relation_ids=tuple(str(item.temporal_query_id) for item in relations),
    )


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _record(path: Path, root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def write_two_visit_current_map(
    current_map: TwoVisitCurrentMap, output_root: str | Path
) -> Path:
    """Atomically publish the neutral map, provenance, and source manifest."""

    if not isinstance(current_map, TwoVisitCurrentMap):
        raise TypeError("current_map must be TwoVisitCurrentMap")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"current map output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot_paths = write_map_snapshot(current_map.snapshot, staging)
        provenance_path = staging / "provenance.jsonl"
        records = []
        for item in current_map.provenance:
            records.append(
                json.dumps(
                    {
                        "decision": asdict(item.decision),
                        "source_point_indices": item.source_point_indices.tolist(),
                        "source_snapshot_sha256": item.source_snapshot_sha256,
                        "output_entity_id": item.output_entity_id,
                        "output_point_start": item.output_point_start,
                        "output_point_count": item.output_point_count,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        _write_bytes(
            provenance_path,
            ("\n".join(records) + "\n").encode("utf-8"),
        )
        artifacts = {
            name: _record(path, staging) for name, path in snapshot_paths.items()
        }
        artifacts["provenance"] = _record(provenance_path, staging)
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_TWO_VISIT_CURRENT_MAP_V1",
            "status": "PASS",
            "method": current_map.snapshot.method,
            "current_map_sha256": current_map.content_sha256(),
            "source_visit_map_sha256": list(current_map.source_visit_map_sha256),
            "source_manifest_sha256": current_map.source_manifest_sha256,
            "visibility_source_sha256": current_map.visibility_source_sha256,
            "relation_ids": list(current_map.relation_ids),
            "geometry_sources": ["ovi_t0", "ovi_t1"],
            "artifacts": artifacts,
        }
        manifest_path = staging / "manifest.json"
        _write_bytes(
            manifest_path,
            (
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8"),
        )
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise ValueError(f"current map output already exists: {output}")
        staging.rename(output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"
