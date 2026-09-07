"""Observation-query adapter for raw queries, dense instances, and pair identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.dense_instance_repair_metrics import (
    DenseMethodView,
    build_p2_method_view,
)
from src.evaluation.ovi_pair_views import OviObjectPairView, OviObjectVisitView
from src.oviv2.rescene_dense_instance_readout import (
    OWNER_UNKNOWN,
    DensePairReadout,
    build_dense_instance_readout,
)
from src.oviv2.rescene_input_bridge import SurfaceAttributeBundle
from src.oviv2.rescene_supported_view import SupportedInferenceView
from src.oviv2.two_visit_contracts import PairRelation, VisitMap, validate_visit_pair


class ObservationDenseAdapterError(ValueError):
    """Raised when an observation result violates dense evaluation contracts."""


def _method_id(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("OBS_") or len(value) <= 4:
        raise ObservationDenseAdapterError(
            "method_id must use a non-empty OBS_* namespace"
        )
    return value


@dataclass(frozen=True, slots=True)
class ObservationDenseLayers:
    method_id: str
    readout: DensePairReadout
    method_view: DenseMethodView
    raw_query_count: int
    retained_query_count: int
    dense_point_count: int
    xyz_changed_count: int

    def content_sha256(self) -> str:
        payload = {
            "method_id": self.method_id,
            "readout_sha256": self.readout.content_sha256(),
            "method_view_sha256": self.method_view.content_sha256(),
            "raw_query_count": self.raw_query_count,
            "retained_query_count": self.retained_query_count,
            "dense_point_count": self.dense_point_count,
            "xyz_changed_count": self.xyz_changed_count,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationComposerInputs:
    """Dense visit maps and candidate-only relations for the existing composer."""

    method_id: str
    visit_maps: tuple[VisitMap, VisitMap]
    relations: tuple[PairRelation, ...]
    source_d_rows: tuple[
        tuple[tuple[str, np.ndarray], ...], tuple[tuple[str, np.ndarray], ...]
    ]
    background_source_d_rows: tuple[np.ndarray, np.ndarray]
    unknown_source_d_rows: tuple[np.ndarray, np.ndarray]
    source_pair_sha256: str

    def content_sha256(self) -> str:
        def rows_payload(
            rows: tuple[tuple[str, np.ndarray], ...],
        ) -> list[dict[str, object]]:
            return [
                {
                    "entity_id": entity_id,
                    "rows": np.asarray(indices, dtype=np.int64).tolist(),
                }
                for entity_id, indices in rows
            ]

        payload = {
            "method_id": self.method_id,
            "visit_map_sha256": [item.snapshot_sha256 for item in self.visit_maps],
            "relations": [
                {
                    "temporal_query_id": item.temporal_query_id,
                    "t0_entity_ids": item.t0_entity_ids,
                    "t1_entity_ids": item.t1_entity_ids,
                    "state": item.state,
                    "query_confidence": item.query_confidence,
                    "evidence": dict(item.evidence),
                    "identity_source": item.identity_source,
                }
                for item in self.relations
            ],
            "source_d_rows": [rows_payload(rows) for rows in self.source_d_rows],
            "background_source_d_rows": [
                value.tolist() for value in self.background_source_d_rows
            ],
            "unknown_source_d_rows": [
                value.tolist() for value in self.unknown_source_d_rows
            ],
            "source_pair_sha256": self.source_pair_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _namespaced_readout(readout: DensePairReadout, method_id: str) -> DensePairReadout:
    visits = []
    for visit in readout.visits:
        instances = tuple(
            replace(
                instance,
                instance_id=f"{method_id}:{instance.instance_id.split(':', 1)[1]}",
                temporal_identity_id=(
                    None
                    if instance.temporal_identity_id is None
                    else f"{method_id}:{instance.temporal_identity_id}"
                ),
            )
            for instance in visit.instances
        )
        visits.append(replace(visit, instances=instances))
    proposals = tuple(
        tuple(
            replace(proposal, query_id=f"{method_id}:{proposal.query_id}")
            for proposal in rows
        )
        for rows in readout.raw_proposals
    )
    return DensePairReadout(
        pair_content_sha256=readout.pair_content_sha256,
        minimum_query_score=readout.minimum_query_score,
        raw_query_indices=readout.raw_query_indices,
        query_scores=readout.query_scores,
        visits=(visits[0], visits[1]),
        raw_proposals=(proposals[0], proposals[1]),
        geometry_unchanged=readout.geometry_unchanged,
    )


def _namespaced_method_view(
    pair: OviObjectPairView, readout: DensePairReadout, method_id: str
) -> DenseMethodView:
    historical = build_p2_method_view(pair, readout)
    candidates = tuple(
        tuple(
            replace(
                candidate,
                raw_query_id=(
                    None
                    if candidate.raw_query_id is None
                    else f"{method_id}:{candidate.raw_query_id}"
                ),
            )
            for candidate in rows
        )
        for rows in historical.candidates
    )
    return replace(
        historical,
        method_id=method_id,
        candidates=(candidates[0], candidates[1]),
    )


def build_observation_dense_layers(
    pair: OviObjectPairView,
    *,
    surface: SurfaceAttributeBundle,
    supported_view: SupportedInferenceView,
    pred_masks_mq: np.ndarray,
    pred_logits_qc: np.ndarray,
    method_id: str,
    minimum_query_score: float = 0.3,
    point_chunk_size: int = 131072,
) -> ObservationDenseLayers:
    """Build all three observable layers while retaining the source D geometry."""

    method = _method_id(method_id)
    masks = np.asarray(pred_masks_mq)
    if masks.ndim != 2:
        raise ObservationDenseAdapterError("pred_masks_mq must have shape M x Q")
    historical = build_dense_instance_readout(
        pair,
        surface=surface,
        supported_view=supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=pred_logits_qc,
        minimum_query_score=minimum_query_score,
        point_chunk_size=point_chunk_size,
    )
    readout = _namespaced_readout(historical, method)
    view = _namespaced_method_view(pair, readout, method)
    dense_count = sum(visit.point_count for visit in pair.visits)
    if any(
        len(readout.visits[index].owner_instance_indices)
        != pair.visits[index].point_count
        for index in (0, 1)
    ):
        raise ObservationDenseAdapterError("dense readout changed the D row domain")
    return ObservationDenseLayers(
        method_id=method,
        readout=readout,
        method_view=view,
        raw_query_count=int(masks.shape[1]),
        retained_query_count=len(readout.raw_query_indices),
        dense_point_count=dense_count,
        xyz_changed_count=0,
    )


def _source_rows(value: np.ndarray) -> np.ndarray:
    rows = np.array(value, dtype=np.int64, copy=True)
    rows.setflags(write=False)
    return rows


def _parent_semantic_score(
    pair_visit: OviObjectVisitView, parent_counts: tuple[tuple[str, int], ...]
) -> float:
    entities = {item.entity_id: item for item in pair_visit.entities}
    total = sum(count for _entity_id, count in parent_counts)
    return sum(
        entities[entity_id].semantic_score * count
        for entity_id, count in parent_counts
    ) / total


def _observation_span(
    pair_visit: OviObjectVisitView,
    parent_counts: tuple[tuple[str, int], ...],
    frame_offset: int,
) -> tuple[float, float]:
    entities = {item.entity_id: item for item in pair_visit.entities}
    frames = tuple(
        frame_id
        for entity_id, _count in parent_counts
        for frame_id in entities[entity_id].observation_frame_ids
    )
    if not frames:
        return float(frame_offset), float(frame_offset + pair_visit.frame_count - 1)
    return float(frame_offset + min(frames)), float(frame_offset + max(frames))


def build_observation_composer_inputs(
    pair: OviObjectPairView,
    layers: ObservationDenseLayers,
    *,
    map_voxel_size_m: float = 0.01,
) -> ObservationComposerInputs:
    """Convert an OBS dense readout into conservative t1-first composer inputs."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(layers, ObservationDenseLayers):
        raise TypeError("layers must be ObservationDenseLayers")
    method = _method_id(layers.method_id)
    pair_sha256 = pair.content_sha256()
    if layers.readout.pair_content_sha256 != pair_sha256:
        raise ObservationDenseAdapterError("dense layers name a different source pair")
    if (
        isinstance(map_voxel_size_m, bool)
        or not isinstance(map_voxel_size_m, (int, float))
        or not np.isfinite(map_voxel_size_m)
        or map_voxel_size_m <= 0.0
    ):
        raise ObservationDenseAdapterError("map_voxel_size_m must be finite and positive")

    visit_maps: list[VisitMap] = []
    source_rows: list[tuple[tuple[str, np.ndarray], ...]] = []
    background_rows: list[np.ndarray] = []
    unknown_rows: list[np.ndarray] = []
    frame_offset = 0
    for pair_visit, readout_visit in zip(
        pair.visits, layers.readout.visits, strict=True
    ):
        entities: list[EntityPrediction] = []
        visit_rows: list[tuple[str, np.ndarray]] = []
        for instance in readout_visit.instances:
            rows = _source_rows(instance.point_indices)
            first_seen, last_seen = _observation_span(
                pair_visit, instance.parent_ovi_entity_point_counts, frame_offset
            )
            label = (
                instance.semantic_labels[0]
                if len(instance.semantic_labels) == 1
                else None
            )
            entities.append(
                EntityPrediction(
                    entity_id=instance.instance_id,
                    points_xyz=pair_visit.points_xyz[rows],
                    semantic_embedding=instance.semantic_embedding,
                    semantic_label=label,
                    semantic_score=_parent_semantic_score(
                        pair_visit, instance.parent_ovi_entity_point_counts
                    ),
                    lifecycle_state="active",
                    first_seen=first_seen,
                    last_seen=last_seen,
                    metadata={
                        "visit_id": pair_visit.visit_id,
                        "owner_source": instance.owner_source,
                        "raw_query_index": instance.raw_query_index,
                        "temporal_identity_candidate": instance.temporal_identity_id,
                        "parent_ovi_entity_point_counts": [
                            [entity_id, count]
                            for entity_id, count in instance.parent_ovi_entity_point_counts
                        ],
                        "semantic_source": "parent_ovi_control",
                        "semantic_provenance": instance.semantic_provenance,
                        "component_count": instance.component_count,
                        "multi_object_conflict": instance.multi_object_conflict,
                        "source_d_row_count": len(rows),
                        "source_d_row_sha256": hashlib.sha256(
                            np.ascontiguousarray(rows).tobytes(order="C")
                        ).hexdigest(),
                        "geometry_authority": f"ovi_t{pair_visit.visit_id}_reassigned",
                    },
                )
            )
            visit_rows.append((instance.instance_id, rows))

        unassigned = _source_rows(
            np.flatnonzero(readout_visit.owner_instance_indices < 0)
        )
        unknown = _source_rows(
            np.flatnonzero(readout_visit.owner_source_codes == OWNER_UNKNOWN)
        )
        claimed = np.concatenate(
            [*(rows for _entity_id, rows in visit_rows), unassigned]
        )
        if not np.array_equal(
            np.sort(claimed), np.arange(pair_visit.point_count, dtype=np.int64)
        ) or len(np.unique(claimed)) != pair_visit.point_count:
            raise ObservationDenseAdapterError(
                "reassigned visit does not conserve every source D row exactly once"
            )
        snapshot = MapSnapshot(
            method=f"{method} dense reassigned t{pair_visit.visit_id}",
            scene_id=pair.pair_id,
            timestamp=float(frame_offset + pair_visit.frame_count - 1),
            entities=entities,
            background_xyz=pair_visit.points_xyz[unassigned],
            scope="current",
        )
        visit_maps.append(
            VisitMap(
                visit_id=pair_visit.visit_id,
                snapshot=snapshot,
                coordinate_frame_id=pair.coordinate_frame_id,
                source_manifest_sha256=pair.source_manifest_sha256,
                map_voxel_size_m=float(map_voxel_size_m),
                observed_frame_start=frame_offset,
                observed_frame_end=frame_offset + pair_visit.frame_count - 1,
            )
        )
        source_rows.append(tuple(visit_rows))
        background_rows.append(unassigned)
        unknown_rows.append(unknown)
        frame_offset += pair_visit.frame_count

    visit_pair = (visit_maps[0], visit_maps[1])
    validate_visit_pair(*visit_pair)
    instance_by_identity = [
        {
            instance.temporal_identity_id: instance
            for instance in visit.instances
            if instance.temporal_identity_id is not None
        }
        for visit in layers.readout.visits
    ]
    query_score = dict(
        zip(
            layers.readout.raw_query_indices.tolist(),
            layers.readout.query_scores.tolist(),
            strict=True,
        )
    )
    relations: list[PairRelation] = []
    for identity in sorted(set(instance_by_identity[0]) & set(instance_by_identity[1])):
        t0 = instance_by_identity[0][identity]
        t1 = instance_by_identity[1][identity]
        if t0.raw_query_index is None or t1.raw_query_index != t0.raw_query_index:
            raise ObservationDenseAdapterError(
                "temporal identity candidate is inconsistent with its raw query"
            )
        centroids = (
            np.mean(pair.visits[0].points_xyz[t0.point_indices], axis=0),
            np.mean(pair.visits[1].points_xyz[t1.point_indices], axis=0),
        )
        score = float(query_score[t0.raw_query_index])
        relations.append(
            PairRelation(
                temporal_query_id=identity,
                t0_entity_ids=(t0.instance_id,),
                t1_entity_ids=(t1.instance_id,),
                state="uncertain",
                query_confidence=score,
                evidence={
                    "candidate_only": 1.0,
                    "query_score": score,
                    "t0_point_support": float(t0.point_count),
                    "t1_point_support": float(t1.point_count),
                    "centroid_distance_m": float(
                        np.linalg.norm(centroids[0] - centroids[1])
                    ),
                },
                identity_source="rescene",
            )
        )
    return ObservationComposerInputs(
        method_id=method,
        visit_maps=visit_pair,
        relations=tuple(relations),
        source_d_rows=(source_rows[0], source_rows[1]),
        background_source_d_rows=(background_rows[0], background_rows[1]),
        unknown_source_d_rows=(unknown_rows[0], unknown_rows[1]),
        source_pair_sha256=pair_sha256,
    )


__all__ = [
    "ObservationComposerInputs",
    "ObservationDenseAdapterError",
    "ObservationDenseLayers",
    "build_observation_composer_inputs",
    "build_observation_dense_layers",
]
