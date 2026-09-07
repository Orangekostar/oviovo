"""Geometry-preserving dense instance readout from raw ReScene queries."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace

import numpy as np

from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView
from src.oviv2.rescene_input_bridge import SurfaceAttributeBundle
from src.oviv2.rescene_supported_view import SupportedInferenceView


OWNER_QUERY = 1
OWNER_OVI_RESIDUAL = 2
OWNER_BACKGROUND = 3
OWNER_UNKNOWN = 4
_OWNER_LABELS = {
    OWNER_QUERY: "query",
    OWNER_OVI_RESIDUAL: "ovi_residual",
    OWNER_BACKGROUND: "background",
    OWNER_UNKNOWN: "unknown",
}


class DenseInstanceReadoutError(ValueError):
    """Raised when dense query projection violates the ownership contract."""


def _readonly(value: object, dtype: np.dtype | type, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise DenseInstanceReadoutError(f"array must have {ndim} dimensions")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DenseQueryProposal:
    visit_id: int
    raw_query_index: int
    query_id: str
    query_score: float
    point_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.visit_id not in (0, 1):
            raise DenseInstanceReadoutError("query proposal visit must be 0 or 1")
        if type(self.raw_query_index) is not int or self.raw_query_index < 0:
            raise DenseInstanceReadoutError("raw query index must be non-negative")
        if not isinstance(self.query_id, str) or not self.query_id:
            raise DenseInstanceReadoutError("query proposal ID must be non-empty")
        if not math.isfinite(self.query_score) or not 0.0 <= self.query_score <= 1.0:
            raise DenseInstanceReadoutError("query proposal score must lie in [0, 1]")
        points = _readonly(self.point_indices, np.int64, 1)
        if not len(points) or np.any(points < 0) or len(np.unique(points)) != len(points):
            raise DenseInstanceReadoutError("query proposal points must be unique")
        object.__setattr__(self, "point_indices", points)


@dataclass(frozen=True, slots=True)
class DenseInstance:
    visit_id: int
    instance_index: int
    instance_id: str
    owner_source: str
    point_indices: np.ndarray
    raw_query_index: int | None
    temporal_identity_id: str | None
    parent_ovi_entity_point_counts: tuple[tuple[str, int], ...]
    semantic_embedding: np.ndarray | None
    semantic_labels: tuple[str, ...]
    semantic_provenance: str
    component_count: int
    multi_object_conflict: bool

    def __post_init__(self) -> None:
        if self.visit_id not in (0, 1) or type(self.instance_index) is not int or self.instance_index < 0:
            raise DenseInstanceReadoutError("dense instance identity is invalid")
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise DenseInstanceReadoutError("dense instance ID must be non-empty")
        if self.owner_source not in {"query", "ovi_residual"}:
            raise DenseInstanceReadoutError("dense instance source is invalid")
        points = _readonly(self.point_indices, np.int64, 1)
        if not len(points) or np.any(points < 0) or len(np.unique(points)) != len(points):
            raise DenseInstanceReadoutError("dense instance points must be unique")
        parents = tuple(self.parent_ovi_entity_point_counts)
        if (
            not parents
            or tuple(name for name, _count in parents) != tuple(sorted(name for name, _count in parents))
            or any(not name or type(count) is not int or count <= 0 for name, count in parents)
            or sum(count for _name, count in parents) != len(points)
        ):
            raise DenseInstanceReadoutError("dense instance parent counts are invalid")
        embedding = None
        if self.semantic_embedding is not None:
            embedding = _readonly(self.semantic_embedding, np.float32, 1)
            if not len(embedding) or not np.all(np.isfinite(embedding)):
                raise DenseInstanceReadoutError(
                    "dense instance semantic embedding is invalid"
                )
        if type(self.component_count) is not int or self.component_count < 1:
            raise DenseInstanceReadoutError("component count must be positive")
        if self.multi_object_conflict != (self.component_count > 1):
            raise DenseInstanceReadoutError("multi-object conflict must match components")
        object.__setattr__(self, "point_indices", points)
        object.__setattr__(self, "parent_ovi_entity_point_counts", parents)
        object.__setattr__(self, "semantic_embedding", embedding)

    @property
    def point_count(self) -> int:
        return len(self.point_indices)


@dataclass(frozen=True, slots=True)
class DenseVisitReadout:
    visit_id: int
    scan_id: str
    owner_instance_indices: np.ndarray
    owner_source_codes: np.ndarray
    neural_valid: np.ndarray
    dense_to_model_indices: np.ndarray
    instances: tuple[DenseInstance, ...]

    def __post_init__(self) -> None:
        owners = _readonly(self.owner_instance_indices, np.int64, 1)
        sources = _readonly(self.owner_source_codes, np.uint8, 1)
        valid = _readonly(self.neural_valid, np.bool_, 1)
        dense_to_model = _readonly(self.dense_to_model_indices, np.int64, 1)
        if not len(owners) or any(len(value) != len(owners) for value in (sources, valid, dense_to_model)):
            raise DenseInstanceReadoutError("dense visit arrays must have equal rows")
        if np.any(dense_to_model < -1) or not np.array_equal(valid, dense_to_model >= 0):
            raise DenseInstanceReadoutError("neural validity must match dense-to-model rows")
        if np.any(~np.isin(sources, tuple(_OWNER_LABELS))):
            raise DenseInstanceReadoutError("owner source code is invalid")
        instances = tuple(self.instances)
        if tuple(value.instance_index for value in instances) != tuple(range(len(instances))):
            raise DenseInstanceReadoutError("dense instances must use contiguous indices")
        expected = np.full(len(owners), -1, dtype=np.int64)
        for instance in instances:
            if instance.visit_id != self.visit_id or np.any(instance.point_indices >= len(owners)):
                raise DenseInstanceReadoutError("dense instance points name the wrong visit")
            if np.any(expected[instance.point_indices] >= 0):
                raise DenseInstanceReadoutError("dense instance points overlap")
            expected[instance.point_indices] = instance.instance_index
            required_source = OWNER_QUERY if instance.owner_source == "query" else OWNER_OVI_RESIDUAL
            if np.any(sources[instance.point_indices] != required_source):
                raise DenseInstanceReadoutError("instance source does not match dense ownership")
        if not np.array_equal(expected, owners):
            raise DenseInstanceReadoutError("instance records do not cover owned dense rows")
        if np.any((owners < 0) != np.isin(sources, (OWNER_BACKGROUND, OWNER_UNKNOWN))):
            raise DenseInstanceReadoutError("unowned rows must be background or unknown")
        object.__setattr__(self, "owner_instance_indices", owners)
        object.__setattr__(self, "owner_source_codes", sources)
        object.__setattr__(self, "neural_valid", valid)
        object.__setattr__(self, "dense_to_model_indices", dense_to_model)
        object.__setattr__(self, "instances", instances)

    @property
    def owner_source_labels(self) -> tuple[str, ...]:
        return tuple(_OWNER_LABELS[int(value)] for value in self.owner_source_codes)


@dataclass(frozen=True, slots=True)
class DensePairReadout:
    pair_content_sha256: str
    minimum_query_score: float
    raw_query_indices: np.ndarray
    query_scores: np.ndarray
    visits: tuple[DenseVisitReadout, DenseVisitReadout]
    raw_proposals: tuple[tuple[DenseQueryProposal, ...], tuple[DenseQueryProposal, ...]]
    geometry_unchanged: bool = True
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_indices = _readonly(self.raw_query_indices, np.int64, 1)
        scores = _readonly(self.query_scores, np.float32, 1)
        if len(raw_indices) != len(scores) or np.any(raw_indices < 0):
            raise DenseInstanceReadoutError("raw query scores are inconsistent")
        if tuple(value.visit_id for value in self.visits) != (0, 1):
            raise DenseInstanceReadoutError("dense pair visits must be ordered")
        if not self.geometry_unchanged:
            raise DenseInstanceReadoutError("dense instance readout cannot change geometry")
        proposals = tuple(tuple(rows) for rows in self.raw_proposals)
        if len(proposals) != 2:
            raise DenseInstanceReadoutError("raw proposals must contain both visits")
        score_by_query = dict(zip(raw_indices.tolist(), scores.tolist(), strict=True))
        for visit, rows in zip(self.visits, proposals, strict=True):
            if any(not isinstance(row, DenseQueryProposal) for row in rows):
                raise DenseInstanceReadoutError("raw proposal type is invalid")
            query_indices = tuple(row.raw_query_index for row in rows)
            if query_indices != tuple(sorted(set(query_indices))):
                raise DenseInstanceReadoutError("raw proposals must be unique and sorted")
            for row in rows:
                if (
                    row.visit_id != visit.visit_id
                    or row.raw_query_index not in score_by_query
                    or row.query_score != score_by_query[row.raw_query_index]
                    or np.any(row.point_indices >= len(visit.owner_instance_indices))
                    or not np.all(visit.neural_valid[row.point_indices])
                ):
                    raise DenseInstanceReadoutError("raw proposal binding is invalid")
        object.__setattr__(self, "raw_query_indices", raw_indices)
        object.__setattr__(self, "query_scores", scores)
        object.__setattr__(self, "raw_proposals", proposals)
        object.__setattr__(
            self,
            "_content_sha256",
            _canonical_sha256(
                {
                    "pair_content_sha256": self.pair_content_sha256,
                    "minimum_query_score": self.minimum_query_score,
                    "raw_query_indices": _array_record(raw_indices),
                    "query_scores": _array_record(scores),
                    "visits": [
                        {
                            "visit_id": visit.visit_id,
                            "scan_id": visit.scan_id,
                            "owners": _array_record(visit.owner_instance_indices),
                            "sources": _array_record(visit.owner_source_codes),
                            "neural_valid": _array_record(visit.neural_valid),
                            "dense_to_model": _array_record(visit.dense_to_model_indices),
                            "instances": [
                                {
                                    "instance_id": row.instance_id,
                                    "points": _array_record(row.point_indices),
                                    "raw_query_index": row.raw_query_index,
                                    "temporal_identity_id": row.temporal_identity_id,
                                    "parents": row.parent_ovi_entity_point_counts,
                                    "embedding": (
                                        None
                                        if row.semantic_embedding is None
                                        else _array_record(row.semantic_embedding)
                                    ),
                                    "labels": row.semantic_labels,
                                    "semantic_provenance": row.semantic_provenance,
                                    "component_count": row.component_count,
                                    "multi_object_conflict": row.multi_object_conflict,
                                }
                                for row in visit.instances
                            ],
                        }
                        for visit in self.visits
                    ],
                    "raw_proposals": [
                        [
                            {
                                "visit_id": row.visit_id,
                                "raw_query_index": row.raw_query_index,
                                "query_id": row.query_id,
                                "query_score": row.query_score,
                                "points": _array_record(row.point_indices),
                            }
                            for row in rows
                        ]
                        for rows in proposals
                    ],
                }
            ),
        )

    def content_sha256(self) -> str:
        return self._content_sha256


def build_dense_to_model_indices(
    pair: OviObjectPairView,
    *,
    surface: SurfaceAttributeBundle,
    supported_view: SupportedInferenceView,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each original dense visit row to compact model M, or -1."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be OviObjectPairView")
    if not isinstance(surface, SurfaceAttributeBundle):
        raise TypeError("surface must be SurfaceAttributeBundle")
    if not isinstance(supported_view, SupportedInferenceView):
        raise TypeError("supported_view must be SupportedInferenceView")
    if supported_view.model_input.surface_attributes_sha256 != surface.content_sha256():
        raise DenseInstanceReadoutError("supported view names a different surface bundle")
    compact_source = supported_view.new_to_old_source_point_indices
    offsets = supported_view.pair.source_to_token_offsets
    counts = np.diff(offsets)
    if len(offsets) != len(supported_view.model_input.adapter_to_model) + 1 or int(offsets[-1]) != len(compact_source):
        raise DenseInstanceReadoutError("supported source offsets are inconsistent")
    source_to_adapter = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    source_to_model = supported_view.model_input.adapter_to_model[source_to_adapter]
    source_visits = surface.source_visit_ids[compact_source]
    original_vertices = surface.original_vertex_indices[compact_source]
    result = [
        np.full(visit.point_count, -1, dtype=np.int64) for visit in pair.visits
    ]
    for visit_id in (0, 1):
        selected = source_visits == visit_id
        dense_rows = original_vertices[selected]
        if np.any(dense_rows >= pair.visits[visit_id].point_count) or len(np.unique(dense_rows)) != len(dense_rows):
            raise DenseInstanceReadoutError("surface rows do not uniquely map to a dense visit")
        result[visit_id][dense_rows] = source_to_model[selected]
        result[visit_id].setflags(write=False)
    return result[0], result[1]


def _query_scores(
    masks_mq: np.ndarray, logits_qc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        masks_mq.ndim != 2
        or logits_qc.ndim != 2
        or masks_mq.shape[1] != logits_qc.shape[0]
        or logits_qc.shape[1] < 2
        or not np.issubdtype(masks_mq.dtype, np.floating)
        or not np.issubdtype(logits_qc.dtype, np.floating)
        or not np.all(np.isfinite(masks_mq))
        or not np.all(np.isfinite(logits_qc))
    ):
        raise DenseInstanceReadoutError("raw ReScene output shape is invalid")
    positive = masks_mq > 0.0
    counts = np.count_nonzero(positive, axis=0)
    retained = np.flatnonzero(counts > 0).astype(np.int64)
    shifted = logits_qc.astype(np.float64) - np.max(logits_qc, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    foreground = np.max(probabilities[:, :-1], axis=1)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(masks_mq.astype(np.float64), -80.0, 80.0)))
    mean_positive = np.zeros(masks_mq.shape[1], dtype=np.float64)
    nonempty = counts > 0
    mean_positive[nonempty] = np.sum(sigmoid * positive, axis=0)[nonempty] / counts[nonempty]
    scores = np.asarray(foreground * mean_positive, dtype=np.float32)
    return retained, scores, np.asarray(sigmoid, dtype=np.float32)


def _component_count(points: np.ndarray, *, voxel_size_m: float = 0.10) -> int:
    voxels = {tuple(int(axis) for axis in row) for row in np.floor(points / voxel_size_m).astype(np.int64)}
    if not voxels:
        return 0
    components = 0
    remaining = set(voxels)
    neighbors = tuple(
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    )
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            linked = {
                (current[0] + dx, current[1] + dy, current[2] + dz)
                for dx, dy, dz in neighbors
            } & remaining
            remaining.difference_update(linked)
            stack.extend(linked)
    return components


def _exclusive_query_winners(
    *,
    dense_to_model: np.ndarray,
    eligible_queries: np.ndarray,
    query_scores: np.ndarray,
    masks_mq: np.ndarray,
    sigmoid_mq: np.ndarray,
    point_chunk_size: int,
) -> np.ndarray:
    winners = np.full(len(dense_to_model), -1, dtype=np.int64)
    valid_dense = np.flatnonzero(dense_to_model >= 0)
    if not len(eligible_queries):
        return winners
    for start in range(0, len(valid_dense), point_chunk_size):
        dense_rows = valid_dense[start : start + point_chunk_size]
        model_rows = dense_to_model[dense_rows]
        positive = masks_mq[np.ix_(model_rows, eligible_queries)] > 0.0
        strength = (
            sigmoid_mq[np.ix_(model_rows, eligible_queries)]
            * query_scores[eligible_queries]
        )
        strength = np.where(positive, strength, -np.inf)
        chosen = np.argmax(strength, axis=1)
        has_owner = np.any(positive, axis=1)
        winners[dense_rows[has_owner]] = eligible_queries[chosen[has_owner]]
    return winners


def _semantic_summary(
    visit: OviObjectVisitView, point_indices: np.ndarray
) -> tuple[
    tuple[tuple[str, int], ...],
    np.ndarray | None,
    tuple[str, ...],
    str,
]:
    owners = visit.entity_owner_indices[point_indices]
    if np.any(owners < 0):
        raise DenseInstanceReadoutError("instance points must have OVI parents")
    counts = tuple(
        (visit.entities[int(index)].entity_id, int(np.count_nonzero(owners == index)))
        for index in sorted(set(int(value) for value in owners))
    )
    count_by_entity = dict(counts)
    valid_entities = tuple(
        visit.entities[index]
        for index in sorted(set(int(value) for value in owners))
        if visit.entities[index].semantic_embedding is not None
    )
    if valid_entities:
        widths = {len(entity.semantic_embedding) for entity in valid_entities}
        if len(widths) != 1:
            raise DenseInstanceReadoutError(
                "parent OVI semantic embedding widths differ"
            )
        embeddings = np.stack(
            [entity.semantic_embedding for entity in valid_entities]
        )
        weights = np.asarray(
            [count_by_entity[entity.entity_id] for entity in valid_entities],
            dtype=np.float64,
        )
        embedding = np.average(embeddings, axis=0, weights=weights).astype(
            np.float32
        )
        provenance = "ovi_point_count_weighted_available_parent_embeddings"
    else:
        embedding = None
        provenance = "ovi_parent_embeddings_unavailable"
    labels = tuple(
        sorted(
            {
                visit.entities[int(index)].semantic_label
                for index in set(owners.tolist())
                if visit.entities[int(index)].semantic_label is not None
            }
        )
    )
    return counts, embedding, labels, provenance


def _visit_readout(
    visit: OviObjectVisitView,
    *,
    dense_to_model: np.ndarray,
    winner_query: np.ndarray,
    retained_queries: np.ndarray,
    all_query_scores: np.ndarray,
    masks_mq: np.ndarray,
    unknown_mask: np.ndarray,
) -> tuple[DenseVisitReadout, tuple[DenseQueryProposal, ...]]:
    sources = np.full(visit.point_count, OWNER_BACKGROUND, dtype=np.uint8)
    sources[unknown_mask] = OWNER_UNKNOWN
    sources[visit.entity_owner_indices >= 0] = OWNER_OVI_RESIDUAL
    sources[winner_query >= 0] = OWNER_QUERY
    owners = np.full(visit.point_count, -1, dtype=np.int64)
    instances: list[DenseInstance] = []
    for raw_query in sorted(set(int(value) for value in winner_query if value >= 0)):
        points = np.flatnonzero(winner_query == raw_query).astype(np.int64)
        parent_counts, embedding, labels, semantic_provenance = _semantic_summary(
            visit, points
        )
        components = _component_count(visit.points_xyz[points])
        index = len(instances)
        owners[points] = index
        instances.append(
            DenseInstance(
                visit_id=visit.visit_id,
                instance_index=index,
                instance_id=f"P2:t{visit.visit_id}:query_{raw_query:04d}",
                owner_source="query",
                point_indices=points,
                raw_query_index=raw_query,
                temporal_identity_id=None,
                parent_ovi_entity_point_counts=parent_counts,
                semantic_embedding=embedding,
                semantic_labels=labels,
                semantic_provenance=semantic_provenance,
                component_count=components,
                multi_object_conflict=components > 1,
            )
        )
    for entity_index, entity in enumerate(visit.entities):
        points = np.flatnonzero(
            (visit.entity_owner_indices == entity_index) & (winner_query < 0)
        ).astype(np.int64)
        if not len(points):
            continue
        index = len(instances)
        owners[points] = index
        instances.append(
            DenseInstance(
                visit_id=visit.visit_id,
                instance_index=index,
                instance_id=f"P2:t{visit.visit_id}:residual:{entity.entity_id}",
                owner_source="ovi_residual",
                point_indices=points,
                raw_query_index=None,
                temporal_identity_id=None,
                parent_ovi_entity_point_counts=((entity.entity_id, len(points)),),
                semantic_embedding=entity.semantic_embedding,
                semantic_labels=(
                    () if entity.semantic_label is None else (entity.semantic_label,)
                ),
                semantic_provenance=(
                    "inherited_single_ovi_parent"
                    if entity.semantic_embedding is not None
                    else "single_ovi_parent_embedding_unavailable"
                ),
                component_count=1,
                multi_object_conflict=False,
            )
        )
    raw_proposals: list[DenseQueryProposal] = []
    valid_rows = dense_to_model >= 0
    for raw_query in retained_queries:
        points = np.flatnonzero(
            valid_rows & (masks_mq[np.maximum(dense_to_model, 0), raw_query] > 0.0)
        ).astype(np.int64)
        if len(points):
            raw_proposals.append(
                DenseQueryProposal(
                    visit_id=visit.visit_id,
                    raw_query_index=int(raw_query),
                    query_id=f"query_{int(raw_query):04d}",
                    query_score=float(all_query_scores[raw_query]),
                    point_indices=points,
                )
            )
    return (
        DenseVisitReadout(
            visit_id=visit.visit_id,
            scan_id=visit.scan_id,
            owner_instance_indices=owners,
            owner_source_codes=sources,
            neural_valid=valid_rows,
            dense_to_model_indices=dense_to_model,
            instances=tuple(instances),
        ),
        tuple(raw_proposals),
    )


def build_dense_instance_readout(
    pair: OviObjectPairView,
    *,
    surface: SurfaceAttributeBundle,
    supported_view: SupportedInferenceView,
    pred_masks_mq: np.ndarray,
    pred_logits_qc: np.ndarray,
    minimum_query_score: float = 0.3,
    point_chunk_size: int = 131072,
    unknown_point_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> DensePairReadout:
    """Project raw model queries to every OVI row with exclusive ownership."""

    if (
        isinstance(minimum_query_score, bool)
        or not isinstance(minimum_query_score, (int, float))
        or not 0.0 <= float(minimum_query_score) <= 1.0
        or type(point_chunk_size) is not int
        or point_chunk_size < 1
    ):
        raise DenseInstanceReadoutError("readout thresholds or chunk size are invalid")
    masks = np.asarray(pred_masks_mq)
    logits = np.asarray(pred_logits_qc)
    if masks.shape[0] != len(supported_view.model_input.model_visit_ids):
        raise DenseInstanceReadoutError("mask M domain differs from supported input")
    before = pair.content_sha256()
    dense_maps = build_dense_to_model_indices(
        pair, surface=surface, supported_view=supported_view
    )
    retained, all_scores, sigmoid = _query_scores(masks, logits)
    eligible = retained[all_scores[retained] >= float(minimum_query_score)]
    unknown = unknown_point_masks or tuple(
        np.zeros(visit.point_count, dtype=np.bool_) for visit in pair.visits
    )
    if len(unknown) != 2:
        raise DenseInstanceReadoutError("unknown masks must contain both visits")
    normalized_unknown: list[np.ndarray] = []
    for visit, value in zip(pair.visits, unknown, strict=True):
        mask = np.asarray(value)
        if mask.shape != (visit.point_count,) or mask.dtype != np.bool_:
            raise DenseInstanceReadoutError("unknown mask must be dense boolean rows")
        if np.any(mask & (visit.entity_owner_indices >= 0)):
            raise DenseInstanceReadoutError("unknown rows cannot hide OVI-owned points")
        normalized_unknown.append(mask)

    winner_queries = [
        _exclusive_query_winners(
            dense_to_model=dense_to_model,
            eligible_queries=eligible,
            query_scores=all_scores,
            masks_mq=masks,
            sigmoid_mq=sigmoid,
            point_chunk_size=point_chunk_size,
        )
        for dense_to_model in dense_maps
    ]

    built = tuple(
        _visit_readout(
            visit,
            dense_to_model=dense_map,
            winner_query=winners,
            retained_queries=retained,
            all_query_scores=all_scores,
            masks_mq=masks,
            unknown_mask=unknown_mask,
        )
        for visit, dense_map, winners, unknown_mask in zip(
            pair.visits, dense_maps, winner_queries, normalized_unknown, strict=True
        )
    )
    visits = [built[0][0], built[1][0]]
    query_by_visit = [
        {
            row.raw_query_index: row
            for row in visit.instances
            if row.raw_query_index is not None
        }
        for visit in visits
    ]
    shared_queries = set(query_by_visit[0]) & set(query_by_visit[1])
    for visit_id in (0, 1):
        replaced = []
        for row in visits[visit_id].instances:
            identity = (
                f"query_{row.raw_query_index:04d}"
                if row.raw_query_index in shared_queries
                and not row.multi_object_conflict
                and not query_by_visit[1 - visit_id][row.raw_query_index].multi_object_conflict
                else None
            )
            replaced.append(replace(row, temporal_identity_id=identity))
        visits[visit_id] = replace(visits[visit_id], instances=tuple(replaced))
    if pair.content_sha256() != before:
        raise DenseInstanceReadoutError("dense readout mutated the OVI pair")
    return DensePairReadout(
        pair_content_sha256=before,
        minimum_query_score=float(minimum_query_score),
        raw_query_indices=retained,
        query_scores=all_scores[retained],
        visits=(visits[0], visits[1]),
        raw_proposals=(built[0][1], built[1][1]),
    )


__all__ = [
    "DenseInstance",
    "DenseInstanceReadoutError",
    "DensePairReadout",
    "DenseQueryProposal",
    "DenseVisitReadout",
    "OWNER_BACKGROUND",
    "OWNER_OVI_RESIDUAL",
    "OWNER_QUERY",
    "OWNER_UNKNOWN",
    "build_dense_instance_readout",
    "build_dense_to_model_indices",
]
