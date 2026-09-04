"""Evaluator-only attribution for visibility-gated B7 dense recovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.two_visit_snapshot_metrics import TwoVisitEvaluationContext
from src.oviv2.two_visit_contracts import (
    PairRelation,
    VisitMap,
    snapshot_content_sha256,
)
from src.oviv2.two_visit_current_map import TwoVisitCurrentMap
from src.oviv2.two_visit_dense_recovery import (
    DenseRecoveryResult,
)
from src.oviv2.two_visit_registration import point_cloud_sha256

ATTRIBUTION_ARTIFACT_ID = "OVI_TWO_VISIT_B7_ATTRIBUTION_V1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _voxel_centers(voxels: np.ndarray, voxel_size_m: float) -> np.ndarray:
    return (np.asarray(voxels, dtype=np.float64) + 0.5) * voxel_size_m


def _snapshot_entity_points(snapshot, entity_id: str) -> np.ndarray:
    matches = [
        entity.points_xyz
        for entity in snapshot.entities
        if entity.entity_id == entity_id
    ]
    if len(matches) != 1:
        raise ValueError("attribution output entity binding is invalid")
    return np.asarray(matches[0], dtype=np.float64)


def _all_entity_points(snapshot) -> np.ndarray:
    chunks = [
        np.asarray(entity.points_xyz, dtype=np.float64) for entity in snapshot.entities
    ]
    return (
        np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float64)
    )


def _within(points: np.ndarray, targets: np.ndarray, threshold: float) -> np.ndarray:
    if not len(points) or not len(targets):
        return np.zeros(len(points), dtype=np.bool_)
    distances, _indices = cKDTree(targets).query(points, k=1, workers=1)
    return np.asarray(distances) <= threshold


@dataclass(frozen=True, slots=True)
class B7AttributionRow:
    relation_id: str
    relation_state: str
    identity_source: str
    registration_quality_bin: str
    registration_rejection_reason: str | None
    visibility_state: str
    source_visit: int
    source_entity_id: str
    target_entity_id: str
    semantic_label: str | None
    decision: str
    candidate_point_count: int
    recovered_point_count: int
    newly_covered_gt_surface_count: int
    introduced_confirmed_free_match_count: int
    revealed_background_conflict_count: int

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "relation_state",
            "identity_source",
            "registration_quality_bin",
            "visibility_state",
            "source_entity_id",
            "target_entity_id",
            "decision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if self.source_visit != 0:
            raise ValueError("B7 attribution source visit must be t0")
        for name in (
            "candidate_point_count",
            "recovered_point_count",
            "newly_covered_gt_surface_count",
            "introduced_confirmed_free_match_count",
            "revealed_background_conflict_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.recovered_point_count > self.candidate_point_count:
            raise ValueError("recovered points cannot exceed candidates")

    def to_json_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class B7AttributionResult:
    rows: tuple[B7AttributionRow, ...]
    summary: Mapping[str, int]
    dense_recovery_sha256: str
    baseline_current_map_sha256: str
    source_t0_snapshot_sha256: str
    source_t1_snapshot_sha256: str
    evaluation_source_sha256: str
    distance_threshold_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or any(
            not isinstance(row, B7AttributionRow) for row in self.rows
        ):
            raise TypeError("rows must contain B7AttributionRow values")
        if not isinstance(self.summary, Mapping):
            raise TypeError("summary must be a mapping")
        summary = {str(name): value for name, value in self.summary.items()}
        if any(type(value) is not int or value < 0 for value in summary.values()):
            raise ValueError("attribution summary values must be non-negative integers")
        for name in (
            "dense_recovery_sha256",
            "baseline_current_map_sha256",
            "source_t0_snapshot_sha256",
            "source_t1_snapshot_sha256",
            "evaluation_source_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self, "summary", MappingProxyType(dict(sorted(summary.items())))
        )
        object.__setattr__(
            self,
            "distance_threshold_m",
            _finite(self.distance_threshold_m, name="distance_threshold_m"),
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "rows": [row.to_json_record() for row in self.rows],
                    "summary": dict(self.summary),
                    "dense_recovery_sha256": self.dense_recovery_sha256,
                    "baseline_current_map_sha256": self.baseline_current_map_sha256,
                    "source_t0_snapshot_sha256": self.source_t0_snapshot_sha256,
                    "source_t1_snapshot_sha256": self.source_t1_snapshot_sha256,
                    "evaluation_source_sha256": self.evaluation_source_sha256,
                    "distance_threshold_m": self.distance_threshold_m,
                }
            )
        ).hexdigest()


def _quality_bin(evidence) -> str:
    if not evidence.accepted:
        return "rejected"
    overlap = min(
        float(evidence.source_to_target_overlap),
        float(evidence.target_to_source_overlap),
    )
    p90 = max(
        float(evidence.source_to_target_p90_m),
        float(evidence.target_to_source_p90_m),
    )
    if overlap >= 0.75 and p90 <= 0.05:
        return "high"
    if overlap >= 0.50 and p90 <= 0.10:
        return "medium"
    return "low"


def _recovered_groups(
    result: DenseRecoveryResult,
) -> tuple[list[np.ndarray], list[int]]:
    points: list[np.ndarray] = []
    row_indices: list[int] = []
    for group_index, group in enumerate(result.provenance):
        if not group.output_point_count:
            continue
        assert group.output_entity_id is not None
        assert group.output_point_start is not None
        entity_points = _snapshot_entity_points(result.snapshot, group.output_entity_id)
        start = group.output_point_start
        stop = start + group.output_point_count
        points.append(entity_points[start:stop])
        row_indices.extend([group_index] * group.output_point_count)
    return points, row_indices


def attribute_b7_recovery(
    *,
    result: DenseRecoveryResult,
    baseline: TwoVisitCurrentMap,
    t0: VisitMap,
    t1: VisitMap,
    relations: tuple[PairRelation, ...],
    evaluation: TwoVisitEvaluationContext,
    evaluation_source_sha256: str,
    distance_threshold_m: float = 0.05,
) -> B7AttributionResult:
    """Attribute B7 additions after method output is immutable."""

    if not isinstance(result, DenseRecoveryResult):
        raise TypeError("result must be DenseRecoveryResult")
    if not isinstance(baseline, TwoVisitCurrentMap):
        raise TypeError("baseline must be TwoVisitCurrentMap")
    if (
        not isinstance(t0, VisitMap)
        or t0.visit_id != 0
        or not isinstance(t1, VisitMap)
        or t1.visit_id != 1
    ):
        raise TypeError("t0 and t1 must be ordered VisitMap values")
    if not isinstance(evaluation, TwoVisitEvaluationContext):
        raise TypeError("evaluation must be TwoVisitEvaluationContext")
    threshold = _finite(distance_threshold_m, name="distance_threshold_m")
    evaluation_sha = _sha256(evaluation_source_sha256, name="evaluation_source_sha256")
    if result.baseline_current_map_sha256 != baseline.content_sha256():
        raise ValueError("attribution baseline hash mismatch")
    if result.source_visit_map_sha256 != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise ValueError("attribution visit source hash mismatch")
    relation_by_id = {
        str(relation.temporal_query_id): relation for relation in relations
    }
    if len(relation_by_id) != len(relations) or set(result.relation_ids) != set(
        relation_by_id
    ):
        raise ValueError("attribution relation binding mismatch")
    before = (
        snapshot_content_sha256(result.snapshot),
        snapshot_content_sha256(baseline.snapshot),
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    )
    t0_entities = {entity.entity_id: entity for entity in t0.snapshot.entities}
    t1_entities = {entity.entity_id: entity for entity in t1.snapshot.entities}
    for evidence in result.registrations:
        relation = relation_by_id.get(evidence.relation_id)
        if (
            relation is None
            or evidence.relation_state != relation.state
            or evidence.identity_source != relation.identity_source
            or evidence.t0_entity_ids != relation.t0_entity_ids
            or evidence.t1_entity_ids != relation.t1_entity_ids
            or len(evidence.t0_entity_ids) != 1
            or len(evidence.t1_entity_ids) != 1
        ):
            raise ValueError("attribution registration relation binding mismatch")
        source = t0_entities.get(evidence.t0_entity_ids[0])
        target_entity = t1_entities.get(evidence.t1_entity_ids[0])
        if (
            source is None
            or target_entity is None
            or evidence.source_points_sha256 != point_cloud_sha256(source.points_xyz)
            or evidence.target_points_sha256
            != point_cloud_sha256(target_entity.points_xyz)
            or evidence.source_point_count != len(source.points_xyz)
            or evidence.target_point_count != len(target_entity.points_xyz)
        ):
            raise ValueError("attribution registration point binding mismatch")
    registration_by_hash = {
        evidence.content_sha256(): evidence for evidence in result.registrations
    }
    recovered_chunks, recovered_group_indices = _recovered_groups(result)
    recovered = (
        np.concatenate(recovered_chunks, axis=0)
        if recovered_chunks
        else np.empty((0, 3), dtype=np.float64)
    )
    target = _voxel_centers(evaluation.current_semantic_voxels[:, :3], 0.05)
    baseline_points = _all_entity_points(baseline.snapshot)
    previously_covered = _within(target, baseline_points, threshold)
    newly_covered_group_counts = np.zeros(len(result.provenance), dtype=np.int64)
    if len(recovered) and len(target):
        distances, indices = cKDTree(recovered).query(target, k=1, workers=1)
        newly_covered = (~previously_covered) & (np.asarray(distances) <= threshold)
        for recovered_index in np.asarray(indices)[newly_covered]:
            newly_covered_group_counts[
                recovered_group_indices[int(recovered_index)]
            ] += 1
    confirmed_free = _voxel_centers(evaluation.confirmed_free_voxels, 0.05)
    revealed_background = _voxel_centers(evaluation.revealed_background_voxels, 0.05)
    rows: list[B7AttributionRow] = []
    for group_index, group in enumerate(result.provenance):
        relation = relation_by_id[group.relation_id]
        evidence = registration_by_hash.get(group.registration_sha256)
        if evidence is None:
            raise ValueError("attribution registration hash mismatch")
        source = t0_entities.get(group.t0_entity_id)
        if source is None:
            raise ValueError("attribution source entity is missing")
        emitted = np.empty((0, 3), dtype=np.float64)
        if group.output_point_count:
            assert group.output_entity_id is not None
            assert group.output_point_start is not None
            entity_points = _snapshot_entity_points(
                result.snapshot, group.output_entity_id
            )
            start = group.output_point_start
            emitted = entity_points[start : start + group.output_point_count]
        rows.append(
            B7AttributionRow(
                relation_id=group.relation_id,
                relation_state=relation.state,
                identity_source=relation.identity_source,
                registration_quality_bin=_quality_bin(evidence),
                registration_rejection_reason=None,
                visibility_state=group.visibility_status,
                source_visit=0,
                source_entity_id=group.t0_entity_id,
                target_entity_id=group.t1_entity_id,
                semantic_label=source.semantic_label,
                decision=group.decision,
                candidate_point_count=len(group.source_point_indices),
                recovered_point_count=group.output_point_count,
                newly_covered_gt_surface_count=int(
                    newly_covered_group_counts[group_index]
                ),
                introduced_confirmed_free_match_count=int(
                    np.count_nonzero(_within(emitted, confirmed_free, threshold))
                ),
                revealed_background_conflict_count=int(
                    np.count_nonzero(_within(emitted, revealed_background, threshold))
                ),
            )
        )
    represented_registrations = {
        group.registration_sha256 for group in result.provenance
    }
    for evidence in result.registrations:
        if evidence.accepted or evidence.content_sha256() in represented_registrations:
            continue
        relation = relation_by_id[evidence.relation_id]
        source_id = evidence.t0_entity_ids[0]
        target_id = evidence.t1_entity_ids[0]
        source = t0_entities.get(source_id)
        rows.append(
            B7AttributionRow(
                relation_id=evidence.relation_id,
                relation_state=relation.state,
                identity_source=relation.identity_source,
                registration_quality_bin="rejected",
                registration_rejection_reason=",".join(evidence.rejection_reasons),
                visibility_state="not_classified",
                source_visit=0,
                source_entity_id=source_id,
                target_entity_id=target_id,
                semantic_label=None if source is None else source.semantic_label,
                decision="registration_rejected",
                candidate_point_count=evidence.source_point_count,
                recovered_point_count=0,
                newly_covered_gt_surface_count=0,
                introduced_confirmed_free_match_count=0,
                revealed_background_conflict_count=0,
            )
        )
    rows.sort(
        key=lambda row: (
            row.relation_id,
            row.decision,
            row.visibility_state,
            row.source_entity_id,
        )
    )
    summary = {
        "candidate_point_count": sum(row.candidate_point_count for row in rows),
        "recovered_point_count": sum(row.recovered_point_count for row in rows),
        "newly_covered_gt_surface_count": sum(
            row.newly_covered_gt_surface_count for row in rows
        ),
        "introduced_confirmed_free_match_count": sum(
            row.introduced_confirmed_free_match_count for row in rows
        ),
        "revealed_background_conflict_count": sum(
            row.revealed_background_conflict_count for row in rows
        ),
        "rejected_occupied_point_count": sum(
            row.candidate_point_count
            for row in rows
            if row.decision == "reject_occupied"
        ),
        "rejected_visible_free_point_count": sum(
            row.candidate_point_count
            for row in rows
            if row.decision == "reject_visible_free"
        ),
        "rejected_incompatible_point_count": sum(
            row.candidate_point_count
            for row in rows
            if row.decision == "reject_incompatible"
        ),
        "registration_rejected_count": sum(
            row.decision == "registration_rejected" for row in rows
        ),
    }
    after = (
        snapshot_content_sha256(result.snapshot),
        snapshot_content_sha256(baseline.snapshot),
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    )
    if before != after:
        raise ValueError("method output changed during evaluator-only attribution")
    return B7AttributionResult(
        rows=tuple(rows),
        summary=summary,
        dense_recovery_sha256=result.content_sha256(),
        baseline_current_map_sha256=baseline.content_sha256(),
        source_t0_snapshot_sha256=t0.snapshot_sha256,
        source_t1_snapshot_sha256=t1.snapshot_sha256,
        evaluation_source_sha256=evaluation_sha,
        distance_threshold_m=threshold,
    )


def _write(path: Path, content: bytes) -> None:
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


def write_b7_attribution(
    result: B7AttributionResult,
    output_root: str | Path,
) -> Path:
    """Atomically publish compact B7 attribution JSON and JSONL."""

    if not isinstance(result, B7AttributionResult):
        raise TypeError("result must be B7AttributionResult")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"attribution output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        summary_path = stage / "summary.json"
        rows_path = stage / "rows.jsonl"
        _write(
            summary_path,
            _canonical(
                {
                    "schema_version": 1,
                    "summary": dict(result.summary),
                    "dense_recovery_sha256": result.dense_recovery_sha256,
                    "baseline_current_map_sha256": result.baseline_current_map_sha256,
                    "source_t0_snapshot_sha256": result.source_t0_snapshot_sha256,
                    "source_t1_snapshot_sha256": result.source_t1_snapshot_sha256,
                    "evaluation_source_sha256": result.evaluation_source_sha256,
                    "distance_threshold_m": result.distance_threshold_m,
                }
            )
            + b"\n",
        )
        row_lines = b"\n".join(_canonical(row.to_json_record()) for row in result.rows)
        _write(rows_path, row_lines + (b"\n" if row_lines else b""))
        manifest = {
            "schema_version": 1,
            "artifact_id": ATTRIBUTION_ARTIFACT_ID,
            "status": "PASS",
            "attribution_sha256": result.content_sha256(),
            "artifacts": {
                "summary": _record(summary_path, stage),
                "rows": _record(rows_path, stage),
            },
        }
        manifest_path = stage / "manifest.json"
        _write(
            manifest_path,
            json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        )
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output / "manifest.json"


__all__ = [
    "ATTRIBUTION_ARTIFACT_ID",
    "B7AttributionResult",
    "B7AttributionRow",
    "attribute_b7_recovery",
    "write_b7_attribution",
]
