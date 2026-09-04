"""Visibility-gated dense historical completion over a frozen B3 map."""

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

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import read_map_snapshot, write_map_snapshot
from src.oviv2.two_visit_contracts import (
    PairRelation,
    VisitMap,
    snapshot_content_sha256,
    validate_visit_pair,
)
from src.oviv2.two_visit_current_map import (
    SignedVisibilityGrid,
    TwoVisitCurrentMap,
)
from src.oviv2.two_visit_registration import (
    RegistrationEvidence,
    apply_rigid_transform,
    point_cloud_sha256,
)

DENSE_RECOVERY_ARTIFACT_ID = "OVI_TWO_VISIT_B7_DENSE_RECOVERY_V1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VISIBILITY_STATES = ("occupied", "visible_free", "occluded", "unobserved")
_DECISIONS = frozenset(
    {
        "reject_occupied",
        "reject_visible_free",
        "reject_incompatible",
        "recover_occluded",
        "recover_unobserved",
    }
)


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    scaled = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m)
    limit = np.iinfo(np.int64).max
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > limit):
        raise ValueError("dense recovery points exceed int64 voxel range")
    return scaled.astype(np.int64)


def _snapshot_entity(snapshot: MapSnapshot, entity_id: str) -> EntityPrediction:
    matches = [entity for entity in snapshot.entities if entity.entity_id == entity_id]
    if len(matches) != 1:
        raise ValueError(f"snapshot does not contain exactly one entity {entity_id}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class DenseRecoveryConfig:
    composition_voxel_size_m: float = 0.05
    maximum_target_surface_distance_m: float = 0.20
    target_bbox_margin_m: float = 0.10
    method_name: str = "OVI_B7_GEOMETRIC_REGISTRATION_VISIBILITY"

    def __post_init__(self) -> None:
        for name in (
            "composition_voxel_size_m",
            "maximum_target_surface_distance_m",
            "target_bbox_margin_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be non-empty")
        object.__setattr__(self, "method_name", self.method_name.strip())

    def to_json_record(self) -> dict[str, object]:
        return asdict(self)

    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_json_record(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class DenseRecoveryPointGroup:
    relation_id: str
    relation_state: str
    identity_source: str
    t0_entity_id: str
    t1_entity_id: str
    baseline_current_map_sha256: str
    source_snapshot_sha256: str
    source_point_indices: np.ndarray
    registration_sha256: str
    transform_sha256: str
    recovery_config_sha256: str
    visibility_status: str
    compatible: bool
    decision: str
    output_entity_id: str | None
    output_point_start: int | None
    output_point_count: int

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "relation_state",
            "identity_source",
            "t0_entity_id",
            "t1_entity_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value.strip())
        for name in (
            "baseline_current_map_sha256",
            "source_snapshot_sha256",
            "registration_sha256",
            "transform_sha256",
            "recovery_config_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        raw_indices = np.asarray(self.source_point_indices)
        if (
            raw_indices.ndim != 1
            or not len(raw_indices)
            or not np.issubdtype(raw_indices.dtype, np.integer)
        ):
            raise ValueError("source_point_indices must be a non-empty integer vector")
        indices = np.array(raw_indices, dtype=np.int64, copy=True)
        if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError(
                "source_point_indices must be sorted, unique, and non-negative"
            )
        indices.setflags(write=False)
        object.__setattr__(self, "source_point_indices", indices)
        if self.visibility_status not in _VISIBILITY_STATES:
            raise ValueError("visibility_status is invalid")
        if type(self.compatible) is not bool:
            raise TypeError("compatible must be boolean")
        if self.decision not in _DECISIONS:
            raise ValueError("dense recovery decision is invalid")
        expected = {
            "reject_occupied": ("occupied", None),
            "reject_visible_free": ("visible_free", None),
            "reject_incompatible": (
                self.visibility_status
                if self.visibility_status in {"occluded", "unobserved"}
                else "__invalid__",
                False,
            ),
            "recover_occluded": ("occluded", True),
            "recover_unobserved": ("unobserved", True),
        }[self.decision]
        if self.visibility_status != expected[0] or (
            expected[1] is not None and self.compatible is not expected[1]
        ):
            raise ValueError("decision, visibility, and compatibility disagree")
        emitted = self.decision.startswith("recover_")
        count = self.output_point_count
        if type(count) is not int or count < 0:
            raise ValueError("output_point_count must be non-negative")
        if emitted:
            if (
                self.output_entity_id != self.t1_entity_id
                or type(self.output_point_start) is not int
                or self.output_point_start < 0
                or count != len(indices)
            ):
                raise ValueError("recovered point group output span is invalid")
        elif (
            self.output_entity_id is not None
            or self.output_point_start is not None
            or count != 0
        ):
            raise ValueError("rejected point group cannot have an output span")

    def to_json_record(self) -> dict[str, object]:
        return {
            "relation_id": self.relation_id,
            "relation_state": self.relation_state,
            "identity_source": self.identity_source,
            "t0_entity_id": self.t0_entity_id,
            "t1_entity_id": self.t1_entity_id,
            "baseline_current_map_sha256": self.baseline_current_map_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "source_point_indices": self.source_point_indices.tolist(),
            "registration_sha256": self.registration_sha256,
            "transform_sha256": self.transform_sha256,
            "recovery_config_sha256": self.recovery_config_sha256,
            "visibility_status": self.visibility_status,
            "compatible": self.compatible,
            "decision": self.decision,
            "output_entity_id": self.output_entity_id,
            "output_point_start": self.output_point_start,
            "output_point_count": self.output_point_count,
        }


@dataclass(frozen=True, slots=True)
class DenseRecoveryResult:
    snapshot: MapSnapshot
    provenance: tuple[DenseRecoveryPointGroup, ...]
    registrations: tuple[RegistrationEvidence, ...]
    baseline_current_map_sha256: str
    source_visit_map_sha256: tuple[str, str]
    source_manifest_sha256: str
    candidate_visibility_source_sha256: str
    config_sha256: str
    relation_ids: tuple[str, ...]
    recovered_point_count: int
    removed_baseline_point_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.snapshot, MapSnapshot)
            or self.snapshot.scope != "current"
        ):
            raise ValueError("dense recovery requires a current MapSnapshot")
        if not isinstance(self.provenance, tuple) or any(
            not isinstance(group, DenseRecoveryPointGroup) for group in self.provenance
        ):
            raise TypeError("provenance must contain DenseRecoveryPointGroup values")
        if not isinstance(self.registrations, tuple) or any(
            not isinstance(value, RegistrationEvidence) for value in self.registrations
        ):
            raise TypeError("registrations must contain RegistrationEvidence values")
        for name in (
            "baseline_current_map_sha256",
            "source_manifest_sha256",
            "candidate_visibility_source_sha256",
            "config_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if len(self.source_visit_map_sha256) != 2:
            raise ValueError("source_visit_map_sha256 must bind t0 and t1")
        object.__setattr__(
            self,
            "source_visit_map_sha256",
            tuple(
                _sha256(value, name="source visit SHA-256")
                for value in self.source_visit_map_sha256
            ),
        )
        if len(self.relation_ids) != len(set(self.relation_ids)):
            raise ValueError("relation_ids must be unique")
        for name in ("recovered_point_count", "removed_baseline_point_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.recovered_point_count != sum(
            group.output_point_count for group in self.provenance
        ):
            raise ValueError("recovered point count disagrees with provenance")
        output_spans: dict[str, list[tuple[int, int]]] = {}
        entities = {entity.entity_id: entity for entity in self.snapshot.entities}
        registration_sha256 = {value.content_sha256() for value in self.registrations}
        for group in self.provenance:
            if (
                group.baseline_current_map_sha256 != self.baseline_current_map_sha256
                or group.source_snapshot_sha256 != self.source_visit_map_sha256[0]
                or group.recovery_config_sha256 != self.config_sha256
                or group.relation_id not in self.relation_ids
                or group.registration_sha256 not in registration_sha256
            ):
                raise ValueError("provenance result binding mismatch")
            if group.output_entity_id is None:
                continue
            entity = entities.get(group.output_entity_id)
            if entity is None:
                raise ValueError(
                    "recovery provenance references a missing output entity"
                )
            assert group.output_point_start is not None
            stop = group.output_point_start + group.output_point_count
            if stop > len(entity.points_xyz):
                raise ValueError("recovery provenance output span exceeds its entity")
            spans = output_spans.setdefault(group.output_entity_id, [])
            if any(
                group.output_point_start < right and left < stop
                for left, right in spans
            ):
                raise ValueError("recovery provenance output spans overlap")
            spans.append((group.output_point_start, stop))

    def content_sha256(self) -> str:
        payload = {
            "snapshot_sha256": snapshot_content_sha256(self.snapshot),
            "provenance": [group.to_json_record() for group in self.provenance],
            "registration_sha256": [
                value.content_sha256() for value in self.registrations
            ],
            "baseline_current_map_sha256": self.baseline_current_map_sha256,
            "source_visit_map_sha256": list(self.source_visit_map_sha256),
            "source_manifest_sha256": self.source_manifest_sha256,
            "candidate_visibility_source_sha256": self.candidate_visibility_source_sha256,
            "config_sha256": self.config_sha256,
            "relation_ids": list(self.relation_ids),
            "recovered_point_count": self.recovered_point_count,
            "removed_baseline_point_count": self.removed_baseline_point_count,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _GroupDraft:
    evidence: RegistrationEvidence
    source_indices: np.ndarray
    transformed_points: np.ndarray
    visibility_status: str
    compatible: bool
    decision: str


def _relation_index(
    relations: tuple[PairRelation, ...],
    t0_ids: set[str],
    t1_ids: set[str],
) -> dict[str, PairRelation]:
    if not isinstance(relations, tuple):
        raise TypeError("relations must be a tuple")
    result: dict[str, PairRelation] = {}
    used_t0: set[str] = set()
    used_t1: set[str] = set()
    for relation in relations:
        if not isinstance(relation, PairRelation):
            raise TypeError("relations must contain PairRelation values")
        relation_id = str(relation.temporal_query_id)
        if relation_id in result:
            raise ValueError("relation IDs must be unique")
        if not set(relation.t0_entity_ids).issubset(t0_ids) or not set(
            relation.t1_entity_ids
        ).issubset(t1_ids):
            raise ValueError("relation references an entity outside the visit maps")
        if used_t0 & set(relation.t0_entity_ids) or used_t1 & set(
            relation.t1_entity_ids
        ):
            raise ValueError("an entity cannot occur in multiple relations")
        used_t0.update(relation.t0_entity_ids)
        used_t1.update(relation.t1_entity_ids)
        result[relation_id] = relation
    return result


def _transform_sha256(transform: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(transform).tobytes(order="C")
    ).hexdigest()


def _baseline_removal_masks(
    baseline: TwoVisitCurrentMap,
    recovered_source_ids: set[str],
) -> tuple[dict[str, np.ndarray], int]:
    entities = {entity.entity_id: entity for entity in baseline.snapshot.entities}
    masks = {
        entity_id: np.zeros(len(entity.points_xyz), dtype=np.bool_)
        for entity_id, entity in entities.items()
    }
    removed = 0
    for group in baseline.provenance:
        decision = group.decision
        if (
            decision.source_visit != 0
            or decision.source_entity_id not in recovered_source_ids
            or group.output_point_count == 0
        ):
            continue
        output_id = group.output_entity_id
        if output_id is None or output_id == "__background__" or output_id not in masks:
            raise ValueError("B3 source entity output span is invalid")
        assert group.output_point_start is not None
        start = group.output_point_start
        stop = start + group.output_point_count
        mask = masks[output_id]
        if stop > len(mask) or np.any(mask[start:stop]):
            raise ValueError(
                "B3 source entity output spans overlap or exceed the entity"
            )
        mask[start:stop] = True
        removed += group.output_point_count
    return masks, removed


def _validate_registration_binding(
    evidence: RegistrationEvidence,
    relation: PairRelation,
    source: EntityPrediction,
    target: EntityPrediction,
) -> None:
    if (
        evidence.relation_id != str(relation.temporal_query_id)
        or evidence.relation_state != relation.state
        or evidence.identity_source != relation.identity_source
        or evidence.t0_entity_ids != relation.t0_entity_ids
        or evidence.t1_entity_ids != relation.t1_entity_ids
    ):
        raise ValueError("registration relation binding mismatch")
    if (
        evidence.source_points_sha256 != point_cloud_sha256(source.points_xyz)
        or evidence.target_points_sha256 != point_cloud_sha256(target.points_xyz)
        or evidence.source_point_count != len(source.points_xyz)
        or evidence.target_point_count != len(target.points_xyz)
    ):
        raise ValueError("registration point binding mismatch")
    source_label = source.semantic_label
    target_label = target.semantic_label
    if (
        source_label is None
        or target_label is None
        or source_label.casefold() != target_label.casefold()
        or evidence.semantic_label is None
        or evidence.semantic_label.casefold() != target_label.casefold()
    ):
        raise ValueError("registration semantic binding mismatch")


def _drafts_for_registration(
    evidence: RegistrationEvidence,
    relation: PairRelation,
    source: EntityPrediction,
    target: EntityPrediction,
    occupied_keys: set[tuple[int, int, int]],
    visibility: dict[tuple[int, int, int], str],
    config: DenseRecoveryConfig,
) -> tuple[_GroupDraft, ...]:
    if evidence.transform_world_from_t0 is None:
        return ()
    _validate_registration_binding(evidence, relation, source, target)
    transformed = apply_rigid_transform(
        source.points_xyz, evidence.transform_world_from_t0
    )
    keys = _voxel_keys(transformed, config.composition_voxel_size_m)
    missing = {
        tuple(int(value) for value in key)
        for key in keys
        if tuple(int(value) for value in key) not in visibility
    }
    if missing:
        raise ValueError("candidate visibility is incomplete for transformed voxels")
    target_points = np.asarray(target.points_xyz, dtype=np.float64)
    distances, _ = cKDTree(target_points).query(transformed, k=1, workers=1)
    lower = target_points.min(axis=0) - config.target_bbox_margin_m
    upper = target_points.max(axis=0) + config.target_bbox_margin_m
    inside_box = np.all((transformed >= lower) & (transformed <= upper), axis=1)
    compatible = (
        np.asarray(distances) <= config.maximum_target_surface_distance_m
    ) | inside_box
    statuses: list[str] = []
    for key in keys:
        key_tuple = tuple(int(value) for value in key)
        statuses.append(
            "occupied" if key_tuple in occupied_keys else visibility[key_tuple]
        )
    drafts: list[_GroupDraft] = []
    for status in _VISIBILITY_STATES:
        status_mask = np.asarray([value == status for value in statuses])
        partitions = (
            (False, status_mask & ~compatible),
            (True, status_mask & compatible),
        )
        for is_compatible, mask in partitions:
            indices = np.flatnonzero(mask).astype(np.int64)
            if not len(indices):
                continue
            decision = {
                "occupied": "reject_occupied",
                "visible_free": "reject_visible_free",
                "occluded": (
                    "recover_occluded" if is_compatible else "reject_incompatible"
                ),
                "unobserved": (
                    "recover_unobserved" if is_compatible else "reject_incompatible"
                ),
            }[status]
            drafts.append(
                _GroupDraft(
                    evidence=evidence,
                    source_indices=indices,
                    transformed_points=np.asarray(
                        transformed[indices], dtype=np.float32
                    ),
                    visibility_status=status,
                    compatible=is_compatible,
                    decision=decision,
                )
            )
    return tuple(drafts)


def recover_dense_history(
    baseline: TwoVisitCurrentMap,
    t0: VisitMap,
    t1: VisitMap,
    relations: tuple[PairRelation, ...],
    registrations: tuple[RegistrationEvidence, ...],
    candidate_visibility: SignedVisibilityGrid,
    config: DenseRecoveryConfig,
) -> DenseRecoveryResult:
    """Replace eligible unwarped B3 spans with current-compatible rigid history."""

    if not isinstance(baseline, TwoVisitCurrentMap):
        raise TypeError("baseline must be a TwoVisitCurrentMap")
    if not isinstance(candidate_visibility, SignedVisibilityGrid):
        raise TypeError("candidate_visibility must be a SignedVisibilityGrid")
    if not isinstance(config, DenseRecoveryConfig):
        raise TypeError("config must be a DenseRecoveryConfig")
    validate_visit_pair(t0, t1)
    if baseline.source_visit_map_sha256 != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise ValueError("B3 source VisitMap binding mismatch")
    if baseline.source_manifest_sha256 != t0.source_manifest_sha256:
        raise ValueError("B3 source manifest binding mismatch")
    if (
        baseline.snapshot.scene_id != t1.snapshot.scene_id
        or baseline.snapshot.timestamp != t1.snapshot.timestamp
    ):
        raise ValueError("B3 snapshot and t1 identity mismatch")
    if candidate_visibility.voxel_size_m != config.composition_voxel_size_m:
        raise ValueError("candidate visibility and recovery voxel sizes differ")
    t0_entities = {entity.entity_id: entity for entity in t0.snapshot.entities}
    t1_entities = {entity.entity_id: entity for entity in t1.snapshot.entities}
    relation_by_id = _relation_index(relations, set(t0_entities), set(t1_entities))
    if not isinstance(registrations, tuple) or any(
        not isinstance(value, RegistrationEvidence) for value in registrations
    ):
        raise TypeError("registrations must contain RegistrationEvidence values")
    registration_by_id: dict[str, RegistrationEvidence] = {}
    for evidence in registrations:
        if evidence.relation_id in registration_by_id:
            raise ValueError("registration relation IDs must be unique")
        if evidence.relation_id not in relation_by_id:
            raise ValueError("registration references an unknown relation")
        relation = relation_by_id[evidence.relation_id]
        if (
            relation.state not in {"persistent_static", "persistent_moved"}
            or len(relation.t0_entity_ids) != 1
            or len(relation.t1_entity_ids) != 1
        ):
            raise ValueError("registration references an ineligible relation")
        _validate_registration_binding(
            evidence,
            relation,
            t0_entities[relation.t0_entity_ids[0]],
            t1_entities[relation.t1_entity_ids[0]],
        )
        registration_by_id[evidence.relation_id] = evidence
    before = (
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    )
    occupied_chunks = [
        np.asarray(entity.points_xyz, dtype=np.float64)
        for entity in t1.snapshot.entities
    ]
    if t1.snapshot.background_xyz is not None:
        occupied_chunks.append(np.asarray(t1.snapshot.background_xyz, dtype=np.float64))
    occupied_keys = {
        tuple(int(value) for value in key)
        for chunk in occupied_chunks
        for key in _voxel_keys(chunk, config.composition_voxel_size_m)
    }
    visibility = candidate_visibility.as_mapping()
    drafts: list[_GroupDraft] = []
    for relation_id in sorted(registration_by_id):
        evidence = registration_by_id[relation_id]
        if not evidence.accepted:
            continue
        relation = relation_by_id[relation_id]
        if relation.state not in {"persistent_static", "persistent_moved"}:
            raise ValueError("accepted registration has an ineligible relation state")
        source = t0_entities[relation.t0_entity_ids[0]]
        target = t1_entities[relation.t1_entity_ids[0]]
        drafts.extend(
            _drafts_for_registration(
                evidence,
                relation,
                source,
                target,
                occupied_keys,
                visibility,
                config,
            )
        )
    recovered_source_ids = {
        draft.evidence.t0_entity_ids[0]
        for draft in drafts
        if draft.decision.startswith("recover_")
    }
    if not recovered_source_ids:
        groups = tuple(
            DenseRecoveryPointGroup(
                relation_id=draft.evidence.relation_id,
                relation_state=draft.evidence.relation_state,
                identity_source=draft.evidence.identity_source,
                t0_entity_id=draft.evidence.t0_entity_ids[0],
                t1_entity_id=draft.evidence.t1_entity_ids[0],
                baseline_current_map_sha256=baseline.content_sha256(),
                source_snapshot_sha256=t0.snapshot_sha256,
                source_point_indices=draft.source_indices,
                registration_sha256=draft.evidence.content_sha256(),
                transform_sha256=_transform_sha256(
                    draft.evidence.transform_world_from_t0
                ),
                recovery_config_sha256=config.content_sha256(),
                visibility_status=draft.visibility_status,
                compatible=draft.compatible,
                decision=draft.decision,
                output_entity_id=None,
                output_point_start=None,
                output_point_count=0,
            )
            for draft in drafts
        )
        result = DenseRecoveryResult(
            snapshot=baseline.snapshot,
            provenance=groups,
            registrations=registrations,
            baseline_current_map_sha256=baseline.content_sha256(),
            source_visit_map_sha256=baseline.source_visit_map_sha256,
            source_manifest_sha256=baseline.source_manifest_sha256,
            candidate_visibility_source_sha256=candidate_visibility.source_sha256,
            config_sha256=config.content_sha256(),
            relation_ids=tuple(sorted(relation_by_id)),
            recovered_point_count=0,
            removed_baseline_point_count=0,
        )
        after = (
            snapshot_content_sha256(t0.snapshot),
            snapshot_content_sha256(t1.snapshot),
        )
        if before != after or before != (t0.snapshot_sha256, t1.snapshot_sha256):
            raise ValueError("OVI source maps changed during dense recovery")
        return result

    removal_masks, removed_count = _baseline_removal_masks(
        baseline, recovered_source_ids
    )
    recovered_by_target: dict[str, list[int]] = {}
    for draft_index, draft in enumerate(drafts):
        if draft.decision.startswith("recover_"):
            recovered_by_target.setdefault(draft.evidence.t1_entity_ids[0], []).append(
                draft_index
            )
    output_entities: list[EntityPrediction] = []
    output_spans: dict[int, tuple[int, int]] = {}
    for baseline_entity in baseline.snapshot.entities:
        mask = removal_masks[baseline_entity.entity_id]
        chunks = [np.asarray(baseline_entity.points_xyz)[~mask]]
        count = len(chunks[0])
        for draft_index in recovered_by_target.get(baseline_entity.entity_id, []):
            chunk = drafts[draft_index].transformed_points
            output_spans[draft_index] = (count, len(chunk))
            chunks.append(chunk)
            count += len(chunk)
        if count == 0:
            continue
        metadata = dict(baseline_entity.metadata)
        if baseline_entity.entity_id in recovered_by_target:
            metadata["geometry_authority"] = (
                "ovi_t1_with_visibility_gated_registered_ovi_t0"
            )
            metadata["b7_dense_recovery"] = True
        output_entities.append(
            EntityPrediction(
                entity_id=baseline_entity.entity_id,
                points_xyz=np.concatenate(chunks, axis=0),
                semantic_embedding=baseline_entity.semantic_embedding,
                semantic_label=baseline_entity.semantic_label,
                semantic_score=baseline_entity.semantic_score,
                lifecycle_state=baseline_entity.lifecycle_state,
                first_seen=baseline_entity.first_seen,
                last_seen=baseline_entity.last_seen,
                metadata=metadata,
            )
        )
    recovered_count = sum(count for _start, count in output_spans.values())
    groups: list[DenseRecoveryPointGroup] = []
    for draft_index, draft in enumerate(drafts):
        emitted = draft_index in output_spans
        start, count = output_spans.get(draft_index, (None, 0))
        groups.append(
            DenseRecoveryPointGroup(
                relation_id=draft.evidence.relation_id,
                relation_state=draft.evidence.relation_state,
                identity_source=draft.evidence.identity_source,
                t0_entity_id=draft.evidence.t0_entity_ids[0],
                t1_entity_id=draft.evidence.t1_entity_ids[0],
                baseline_current_map_sha256=baseline.content_sha256(),
                source_snapshot_sha256=t0.snapshot_sha256,
                source_point_indices=draft.source_indices,
                registration_sha256=draft.evidence.content_sha256(),
                transform_sha256=_transform_sha256(
                    draft.evidence.transform_world_from_t0
                ),
                recovery_config_sha256=config.content_sha256(),
                visibility_status=draft.visibility_status,
                compatible=draft.compatible,
                decision=draft.decision,
                output_entity_id=(draft.evidence.t1_entity_ids[0] if emitted else None),
                output_point_start=start,
                output_point_count=count,
            )
        )
    runtime = dict(baseline.snapshot.runtime)
    runtime.update(
        {
            "b7_registration_count": float(len(registrations)),
            "b7_recovered_point_count": float(recovered_count),
            "b7_removed_baseline_point_count": float(removed_count),
        }
    )
    snapshot = MapSnapshot(
        method=config.method_name,
        scene_id=baseline.snapshot.scene_id,
        timestamp=baseline.snapshot.timestamp,
        entities=output_entities,
        background_xyz=(
            None
            if baseline.snapshot.background_xyz is None
            else np.asarray(baseline.snapshot.background_xyz, dtype=np.float32)
        ),
        scope="current",
        runtime=runtime,
    )
    after = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    if before != after or before != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise ValueError("OVI source maps changed during dense recovery")
    return DenseRecoveryResult(
        snapshot=snapshot,
        provenance=tuple(groups),
        registrations=registrations,
        baseline_current_map_sha256=baseline.content_sha256(),
        source_visit_map_sha256=baseline.source_visit_map_sha256,
        source_manifest_sha256=baseline.source_manifest_sha256,
        candidate_visibility_source_sha256=candidate_visibility.source_sha256,
        config_sha256=config.content_sha256(),
        relation_ids=tuple(sorted(relation_by_id)),
        recovered_point_count=recovered_count,
        removed_baseline_point_count=removed_count,
    )


def _span_points(
    snapshot: MapSnapshot,
    output_entity_id: str,
    start: int,
    count: int,
) -> np.ndarray:
    if output_entity_id == "__background__":
        if snapshot.background_xyz is None:
            raise ValueError("historical provenance references absent background")
        points = snapshot.background_xyz
    else:
        points = _snapshot_entity(snapshot, output_entity_id).points_xyz
    stop = start + count
    if start < 0 or stop > len(points):
        raise ValueError("historical provenance span exceeds output geometry")
    return np.asarray(points[start:stop], dtype=np.float32)


def historical_output_points(
    result: DenseRecoveryResult,
    baseline: TwoVisitCurrentMap,
) -> np.ndarray:
    """Return final points whose geometry authority includes historical OVI t0."""

    if not isinstance(result, DenseRecoveryResult) or not isinstance(
        baseline, TwoVisitCurrentMap
    ):
        raise TypeError("result and baseline contracts are required")
    if result.baseline_current_map_sha256 != baseline.content_sha256():
        raise ValueError("dense recovery and B3 baseline hashes differ")
    replaced = {
        group.t0_entity_id for group in result.provenance if group.output_point_count
    }
    chunks: list[np.ndarray] = []
    for group in baseline.provenance:
        if (
            group.decision.source_visit != 0
            or group.output_point_count == 0
            or group.decision.source_entity_id in replaced
        ):
            continue
        assert group.output_entity_id is not None
        assert group.output_point_start is not None
        chunks.append(
            _span_points(
                baseline.snapshot,
                group.output_entity_id,
                group.output_point_start,
                group.output_point_count,
            )
        )
    for group in result.provenance:
        if group.output_point_count == 0:
            continue
        assert group.output_entity_id is not None
        assert group.output_point_start is not None
        chunks.append(
            _span_points(
                result.snapshot,
                group.output_entity_id,
                group.output_point_start,
                group.output_point_count,
            )
        )
    output = (
        np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float32)
    )
    output.setflags(write=False)
    return output


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _record(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _read_artifact(record: object, root: Path, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise TypeError(f"{label} path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be artifact-relative")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if _record(path, root) != record:
        raise ValueError(f"{label} binding mismatch")
    return path


def _registration_from_record(record: object) -> RegistrationEvidence:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("registration record is invalid")
    values = dict(record)
    values.pop("schema_version")
    for name in ("t0_entity_ids", "t1_entity_ids", "rejection_reasons"):
        values[name] = tuple(values[name])
    if values["principal_extent_ratios"] is not None:
        values["principal_extent_ratios"] = tuple(values["principal_extent_ratios"])
    if values["transform_world_from_t0"] is not None:
        values["transform_world_from_t0"] = np.asarray(
            values["transform_world_from_t0"], dtype=np.float64
        )
    return RegistrationEvidence(**values)


def read_dense_recovery(manifest_path: str | Path) -> DenseRecoveryResult:
    """Reopen and verify a B7 dense-recovery artifact without pickle."""

    path = Path(manifest_path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("dense recovery manifest must be a regular file")
    try:
        manifest = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dense recovery manifest is invalid JSON") from error
    required = {
        "schema_version",
        "artifact_id",
        "status",
        "method",
        "dense_recovery_sha256",
        "baseline_current_map_sha256",
        "source_visit_map_sha256",
        "source_manifest_sha256",
        "candidate_visibility_source_sha256",
        "config_sha256",
        "relation_ids",
        "recovered_point_count",
        "removed_baseline_point_count",
        "artifacts",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != DENSE_RECOVERY_ARTIFACT_ID
        or manifest.get("status") != "PASS"
    ):
        raise ValueError("dense recovery manifest identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "snapshot",
        "entities",
        "registration",
        "recovery_provenance",
    }:
        raise ValueError("dense recovery artifact inventory is invalid")
    root = path.parent
    snapshot_path = _read_artifact(artifacts["snapshot"], root, label="snapshot")
    entities_path = _read_artifact(artifacts["entities"], root, label="entities")
    registration_path = _read_artifact(
        artifacts["registration"], root, label="registration"
    )
    provenance_path = _read_artifact(
        artifacts["recovery_provenance"], root, label="recovery provenance"
    )
    snapshot = read_map_snapshot(snapshot_path, entities_path)
    registration_payload = json.loads(registration_path.read_bytes())
    if (
        not isinstance(registration_payload, dict)
        or set(registration_payload) != {"schema_version", "registrations"}
        or registration_payload.get("schema_version") != 1
        or not isinstance(registration_payload.get("registrations"), list)
    ):
        raise ValueError("registration artifact is invalid")
    registrations = tuple(
        _registration_from_record(record)
        for record in registration_payload["registrations"]
    )
    groups: list[DenseRecoveryPointGroup] = []
    with provenance_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("recovery provenance record is invalid")
            values = dict(record)
            values["source_point_indices"] = np.asarray(
                values["source_point_indices"], dtype=np.int64
            )
            groups.append(DenseRecoveryPointGroup(**values))
    result = DenseRecoveryResult(
        snapshot=snapshot,
        provenance=tuple(groups),
        registrations=registrations,
        baseline_current_map_sha256=manifest["baseline_current_map_sha256"],
        source_visit_map_sha256=tuple(manifest["source_visit_map_sha256"]),
        source_manifest_sha256=manifest["source_manifest_sha256"],
        candidate_visibility_source_sha256=manifest[
            "candidate_visibility_source_sha256"
        ],
        config_sha256=manifest["config_sha256"],
        relation_ids=tuple(manifest["relation_ids"]),
        recovered_point_count=manifest["recovered_point_count"],
        removed_baseline_point_count=manifest["removed_baseline_point_count"],
    )
    if result.snapshot.method != manifest["method"]:
        raise ValueError("dense recovery method identity mismatch")
    if result.content_sha256() != manifest["dense_recovery_sha256"]:
        raise ValueError("dense recovery content hash mismatch")
    return result


def write_dense_recovery(result: DenseRecoveryResult, output_root: str | Path) -> Path:
    """Atomically publish one B7 dense-recovery artifact."""

    if not isinstance(result, DenseRecoveryResult):
        raise TypeError("result must be a DenseRecoveryResult")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"dense recovery output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot_paths = write_map_snapshot(result.snapshot, staging)
        registration_path = staging / "registration.json"
        _write_bytes(
            registration_path,
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "registrations": [
                            value.to_json_record() for value in result.registrations
                        ],
                    },
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        provenance_path = staging / "recovery-provenance.jsonl"
        records = [
            json.dumps(
                group.to_json_record(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for group in result.provenance
        ]
        _write_bytes(
            provenance_path,
            ("\n".join(records) + ("\n" if records else "")).encode("utf-8"),
        )
        artifacts = {
            name: _record(path, staging) for name, path in snapshot_paths.items()
        }
        artifacts.update(
            {
                "registration": _record(registration_path, staging),
                "recovery_provenance": _record(provenance_path, staging),
            }
        )
        manifest = {
            "schema_version": 1,
            "artifact_id": DENSE_RECOVERY_ARTIFACT_ID,
            "status": "PASS",
            "method": result.snapshot.method,
            "dense_recovery_sha256": result.content_sha256(),
            "baseline_current_map_sha256": result.baseline_current_map_sha256,
            "source_visit_map_sha256": list(result.source_visit_map_sha256),
            "source_manifest_sha256": result.source_manifest_sha256,
            "candidate_visibility_source_sha256": result.candidate_visibility_source_sha256,
            "config_sha256": result.config_sha256,
            "relation_ids": list(result.relation_ids),
            "recovered_point_count": result.recovered_point_count,
            "removed_baseline_point_count": result.removed_baseline_point_count,
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
            raise ValueError(f"dense recovery output already exists: {output}")
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


__all__ = [
    "DENSE_RECOVERY_ARTIFACT_ID",
    "DenseRecoveryConfig",
    "DenseRecoveryPointGroup",
    "DenseRecoveryResult",
    "historical_output_points",
    "read_dense_recovery",
    "recover_dense_history",
    "write_dense_recovery",
]
