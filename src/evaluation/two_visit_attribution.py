"""Exact, source-bound failure attribution for two-visit current maps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from src.oviv2.two_visit_contracts import PairRelation
from src.oviv2.two_visit_current_map import CompositionPointGroup, TwoVisitCurrentMap

FailureType = Literal["ghost_fp", "object_fp", "object_fn", "change_error"]
GhostCategory = Literal[
    "T0_RETAINED_DESPITE_VISIBLE_FREE",
    "T0_RETAINED_UNOBSERVED",
    "T0_RETAINED_OCCLUDED",
    "T1_OVI_FALSE_POSITIVE",
    "PAIR_REASONER_WRONG_ID",
    "VISIBILITY_DEPTH_ERROR",
    "BACKGROUND_COMPOSITION_ERROR",
    "UNATTRIBUTED",
]

FAILURE_TYPES = ("ghost_fp", "object_fp", "object_fn", "change_error")
GHOST_CATEGORIES: tuple[GhostCategory, ...] = (
    "T0_RETAINED_DESPITE_VISIBLE_FREE",
    "T0_RETAINED_UNOBSERVED",
    "T0_RETAINED_OCCLUDED",
    "T1_OVI_FALSE_POSITIVE",
    "PAIR_REASONER_WRONG_ID",
    "VISIBILITY_DEPTH_ERROR",
    "BACKGROUND_COMPOSITION_ERROR",
    "UNATTRIBUTED",
)
ATTRIBUTION_PRECEDENCE = (
    "BACKGROUND_COMPOSITION_ERROR",
    "VISIBILITY_DEPTH_ERROR",
    "PAIR_REASONER_WRONG_ID",
    "T1_OVI_FALSE_POSITIVE",
    "T0_RETAINED_DESPITE_VISIBLE_FREE",
    "T0_RETAINED_UNOBSERVED",
    "T0_RETAINED_OCCLUDED",
    "UNATTRIBUTED",
)
_VISIBILITY_STATES = {"occupied", "visible_free", "occluded", "unobserved"}
_SHA256_LENGTH = 64


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, label: str) -> str | None:
    return None if value is None else _nonempty(value, label)


def _sha256(value: object, label: str) -> str:
    text = _nonempty(value, label)
    if len(text) != _SHA256_LENGTH or any(item not in "0123456789abcdef" for item in text):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return text


@dataclass(frozen=True, slots=True)
class EvaluatorFailure:
    """One metric-contributing evaluator error before causal attribution."""

    failure_id: str
    failure_type: FailureType
    predicted_entity_id: str | None = None
    predicted_point_index: int | None = None
    target_entity_id: str | None = None
    evaluator_visibility: str | None = None
    visibility_depth_correct: bool | None = None
    observed_relation_id: str | None = None
    expected_relation_id: str | None = None
    relation_correct: bool | None = None

    def __post_init__(self) -> None:
        failure_id = _nonempty(self.failure_id, "failure_id")
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError("failure_type is invalid")
        predicted = _optional_string(self.predicted_entity_id, "predicted_entity_id")
        target = _optional_string(self.target_entity_id, "target_entity_id")
        observed_relation = _optional_string(
            self.observed_relation_id, "observed_relation_id"
        )
        expected_relation = _optional_string(
            self.expected_relation_id, "expected_relation_id"
        )
        point_index = self.predicted_point_index
        if point_index is not None and (
            type(point_index) is not int or point_index < 0
        ):
            raise ValueError("predicted_point_index must be null or non-negative")
        visibility = self.evaluator_visibility
        if visibility is not None and visibility not in _VISIBILITY_STATES:
            raise ValueError("evaluator_visibility is invalid")
        for value, label in (
            (self.visibility_depth_correct, "visibility_depth_correct"),
            (self.relation_correct, "relation_correct"),
        ):
            if value is not None and type(value) is not bool:
                raise ValueError(f"{label} must be boolean or null")
        if self.failure_type == "ghost_fp" and (
            predicted is None or point_index is None
        ):
            raise ValueError("ghost_fp requires an exact predicted point identity")
        if self.failure_type == "object_fp" and predicted is None:
            raise ValueError("object_fp requires a predicted entity identity")
        if self.failure_type == "object_fn" and target is None:
            raise ValueError("object_fn requires a target entity identity")
        if self.failure_type == "change_error" and not any(
            (predicted, target, observed_relation, expected_relation)
        ):
            raise ValueError("change_error requires an entity or relation identity")
        object.__setattr__(self, "failure_id", failure_id)
        object.__setattr__(self, "predicted_entity_id", predicted)
        object.__setattr__(self, "target_entity_id", target)
        object.__setattr__(self, "observed_relation_id", observed_relation)
        object.__setattr__(self, "expected_relation_id", expected_relation)


@dataclass(frozen=True, slots=True)
class FailureAttributionRow:
    failure_id: str
    failure_type: FailureType
    category: GhostCategory
    attribution_basis: str
    predicted_entity_id: str | None
    predicted_point_index: int | None
    target_entity_id: str | None
    source_visit: int | None
    source_ovi_entity_id: str | None
    t0_ovi_entity_id: str | None
    t1_ovi_entity_id: str | None
    t0_relation_entity_ids: tuple[str, ...]
    t1_relation_entity_ids: tuple[str, ...]
    relation_id: str | None
    expected_relation_id: str | None
    pair_relation_state: str | None
    identity_source: str | None
    visibility_classification: str | None
    evaluator_visibility: str | None
    visibility_depth_correct: bool | None
    composition_decision: str | None
    geometry_source: str | None
    semantic_source: str | None
    source_snapshot_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureAttributionResult:
    current_map_sha256: str
    evaluator_source_sha256: str
    relation_source_sha256: str
    source_manifest_sha256: str
    source_visit_map_sha256: tuple[str, str]
    visibility_source_sha256: str
    relation_ids: tuple[str, ...]
    rows: tuple[FailureAttributionRow, ...]
    category_counts: dict[str, int]
    ghost_counts: dict[str, int]
    total_failure_count: int
    total_ghost_count: int
    unattributed_count: int
    unattributed_share: float

    def __post_init__(self) -> None:
        for name in (
            "current_map_sha256",
            "evaluator_source_sha256",
            "relation_source_sha256",
            "source_manifest_sha256",
            "visibility_source_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        visit_hashes = tuple(
            _sha256(item, "source_visit_map_sha256")
            for item in self.source_visit_map_sha256
        )
        if len(visit_hashes) != 2:
            raise ValueError("source_visit_map_sha256 must contain exactly two hashes")
        if tuple(row.failure_id for row in self.rows) != tuple(
            sorted(row.failure_id for row in self.rows)
        ):
            raise ValueError("attribution rows must be sorted by failure ID")
        if len({row.failure_id for row in self.rows}) != len(self.rows):
            raise ValueError("attribution row failure IDs must be unique")
        expected_categories = set(GHOST_CATEGORIES)
        if set(self.category_counts) != expected_categories or set(
            self.ghost_counts
        ) != expected_categories:
            raise ValueError("attribution count categories are incomplete")
        if any(type(value) is not int or value < 0 for value in self.category_counts.values()):
            raise ValueError("category counts must be non-negative integers")
        if any(type(value) is not int or value < 0 for value in self.ghost_counts.values()):
            raise ValueError("ghost counts must be non-negative integers")
        if sum(self.category_counts.values()) != len(self.rows):
            raise ValueError("category counts do not conserve failure mass")
        observed_ghost_count = sum(
            row.failure_type == "ghost_fp" for row in self.rows
        )
        if sum(self.ghost_counts.values()) != observed_ghost_count:
            raise ValueError("ghost counts do not conserve Ghost mass")
        unattributed = sum(row.category == "UNATTRIBUTED" for row in self.rows)
        expected_share = unattributed / len(self.rows) if self.rows else 0.0
        if (
            self.total_failure_count != len(self.rows)
            or self.total_ghost_count != observed_ghost_count
            or self.unattributed_count != unattributed
            or not math.isclose(self.unattributed_share, expected_share)
        ):
            raise ValueError("attribution totals are inconsistent")
        object.__setattr__(self, "source_visit_map_sha256", visit_hashes)
        object.__setattr__(
            self,
            "category_counts",
            MappingProxyType(dict(self.category_counts)),
        )
        object.__setattr__(
            self,
            "ghost_counts",
            MappingProxyType(dict(self.ghost_counts)),
        )

    def content_sha256(self) -> str:
        payload = {
            "current_map_sha256": self.current_map_sha256,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "relation_source_sha256": self.relation_source_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_visit_map_sha256": self.source_visit_map_sha256,
            "visibility_source_sha256": self.visibility_source_sha256,
            "relation_ids": self.relation_ids,
            "rows": [row.as_dict() for row in self.rows],
            "category_counts": dict(self.category_counts),
            "ghost_counts": dict(self.ghost_counts),
            "total_failure_count": self.total_failure_count,
            "total_ghost_count": self.total_ghost_count,
            "unattributed_count": self.unattributed_count,
            "unattributed_share": self.unattributed_share,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _OutputSpan:
    start: int
    stop: int
    group: CompositionPointGroup


def _output_spans(current_map: TwoVisitCurrentMap) -> dict[str, tuple[_OutputSpan, ...]]:
    counts = {entity.entity_id: len(entity.points_xyz) for entity in current_map.snapshot.entities}
    if len(counts) != len(current_map.snapshot.entities):
        raise ValueError("current-map entity IDs must be unique")
    if "__background__" in counts:
        raise ValueError("__background__ is reserved for background geometry")
    if current_map.snapshot.background_xyz is not None:
        counts["__background__"] = len(current_map.snapshot.background_xyz)
    spans: dict[str, list[_OutputSpan]] = {key: [] for key in counts}
    for group in current_map.provenance:
        decision = group.decision
        expected_snapshot = current_map.source_visit_map_sha256[decision.source_visit]
        if group.source_snapshot_sha256 != expected_snapshot:
            raise ValueError("provenance source snapshot identity mismatch")
        if group.output_point_count == 0:
            continue
        output_id = group.output_entity_id
        if output_id not in counts or group.output_point_start is None:
            raise ValueError("provenance references an unknown output entity")
        start = group.output_point_start
        stop = start + group.output_point_count
        if stop > counts[output_id]:
            raise ValueError("provenance output span exceeds current-map geometry")
        spans[output_id].append(_OutputSpan(start, stop, group))
    result: dict[str, tuple[_OutputSpan, ...]] = {}
    for output_id, values in spans.items():
        ordered = tuple(sorted(values, key=lambda item: (item.start, item.stop)))
        cursor = 0
        for value in ordered:
            if value.start != cursor or value.stop <= value.start:
                raise ValueError("provenance output spans must be contiguous and disjoint")
            cursor = value.stop
        if cursor != counts[output_id]:
            raise ValueError("provenance output spans do not cover current-map geometry")
        result[output_id] = ordered
    return result


def _source_group(
    failure: EvaluatorFailure,
    spans: dict[str, tuple[_OutputSpan, ...]],
) -> CompositionPointGroup | None:
    if failure.predicted_entity_id is None:
        return None
    if failure.predicted_entity_id not in spans:
        raise ValueError("failure references an unknown current-map entity")
    candidates = spans[failure.predicted_entity_id]
    if failure.predicted_point_index is None:
        return candidates[0].group if len(candidates) == 1 else None
    for span in candidates:
        if span.start <= failure.predicted_point_index < span.stop:
            return span.group
    raise ValueError("failure point index is outside current-map geometry")


def _category(
    failure: EvaluatorFailure,
    group: CompositionPointGroup | None,
    relation_id: str | None,
) -> tuple[GhostCategory, str]:
    if group is None:
        return "UNATTRIBUTED", "no_unique_emitted_source_group"
    decision = group.decision
    if (
        failure.predicted_entity_id == "__background__"
        or decision.source_entity_id == "__background__"
    ):
        return "BACKGROUND_COMPOSITION_ERROR", "background_output_failure"
    if failure.visibility_depth_correct is False:
        return "VISIBILITY_DEPTH_ERROR", "evaluator_depth_visibility_disagrees"
    relation_mismatch = failure.relation_correct is False or (
        failure.expected_relation_id is not None
        and relation_id is not None
        and failure.expected_relation_id != relation_id
    )
    if relation_mismatch:
        return "PAIR_REASONER_WRONG_ID", "evaluator_relation_disagrees"
    if failure.failure_type not in {"ghost_fp", "object_fp"}:
        return "UNATTRIBUTED", "no_supported_causal_precedence_for_failure_type"
    if decision.source_visit == 1:
        return "T1_OVI_FALSE_POSITIVE", "failure_originates_in_t1_ovi_geometry"
    if failure.evaluator_visibility == "visible_free":
        return (
            "T0_RETAINED_DESPITE_VISIBLE_FREE",
            "evaluator_marks_retained_t0_location_visible_free",
        )
    if decision.visibility_status == "unobserved":
        return "T0_RETAINED_UNOBSERVED", "t0_fallback_was_unobserved_at_t1"
    if decision.visibility_status == "occluded":
        return "T0_RETAINED_OCCLUDED", "t0_fallback_was_occluded_at_t1"
    return "UNATTRIBUTED", "no_precedence_rule_matched"


def _row(
    failure: EvaluatorFailure,
    group: CompositionPointGroup | None,
    relations: dict[str, PairRelation],
) -> FailureAttributionRow:
    decision = None if group is None else group.decision
    relation_id = (
        failure.observed_relation_id
        if decision is None
        else decision.relation_id
    )
    if (
        decision is not None
        and failure.observed_relation_id is not None
        and failure.observed_relation_id != decision.relation_id
    ):
        raise ValueError("evaluator and composer observed relation identities disagree")
    relation = None if relation_id is None else relations.get(relation_id)
    if relation_id is not None and relation is None:
        raise ValueError("failure references an unknown pair relation")
    if relation is not None and decision is not None:
        relation_entities = (
            relation.t0_entity_ids
            if decision.source_visit == 0
            else relation.t1_entity_ids
        )
        if decision.source_entity_id not in relation_entities:
            raise ValueError("provenance source entity is not a member of its relation")
    if relation is not None:
        t0_ids = relation.t0_entity_ids
        t1_ids = relation.t1_entity_ids
    elif decision is None:
        t0_ids = ()
        t1_ids = ()
    elif decision.source_visit == 0:
        t0_ids = (decision.source_entity_id,)
        t1_ids = ()
    else:
        t0_ids = ()
        t1_ids = (decision.source_entity_id,)
    category, basis = _category(failure, group, relation_id)
    return FailureAttributionRow(
        failure_id=failure.failure_id,
        failure_type=failure.failure_type,
        category=category,
        attribution_basis=basis,
        predicted_entity_id=failure.predicted_entity_id,
        predicted_point_index=failure.predicted_point_index,
        target_entity_id=failure.target_entity_id,
        source_visit=None if decision is None else decision.source_visit,
        source_ovi_entity_id=(
            None if decision is None else decision.source_entity_id
        ),
        t0_ovi_entity_id=t0_ids[0] if len(t0_ids) == 1 else None,
        t1_ovi_entity_id=t1_ids[0] if len(t1_ids) == 1 else None,
        t0_relation_entity_ids=t0_ids,
        t1_relation_entity_ids=t1_ids,
        relation_id=relation_id,
        expected_relation_id=failure.expected_relation_id,
        pair_relation_state=None if relation is None else relation.state,
        identity_source=None if decision is None else decision.identity_source,
        visibility_classification=(
            None if decision is None else decision.visibility_status
        ),
        evaluator_visibility=failure.evaluator_visibility,
        visibility_depth_correct=failure.visibility_depth_correct,
        composition_decision=None if decision is None else decision.decision,
        geometry_source=None if decision is None else decision.geometry_source,
        semantic_source=None if decision is None else decision.semantic_source,
        source_snapshot_sha256=(
            None if group is None else group.source_snapshot_sha256
        ),
    )


def attribute_two_visit_failures(
    current_map: TwoVisitCurrentMap,
    failures: tuple[EvaluatorFailure, ...],
    relations: tuple[PairRelation, ...],
    *,
    evaluator_source_sha256: str,
    relation_source_sha256: str,
) -> FailureAttributionResult:
    """Assign every evaluator failure to exactly one explicit causal category."""

    if not isinstance(current_map, TwoVisitCurrentMap):
        raise TypeError("current_map must be a TwoVisitCurrentMap")
    if not isinstance(failures, tuple) or any(
        not isinstance(item, EvaluatorFailure) for item in failures
    ):
        raise TypeError("failures must be a tuple of EvaluatorFailure values")
    if len({item.failure_id for item in failures}) != len(failures):
        raise ValueError("failure IDs must be unique")
    if not isinstance(relations, tuple) or any(
        not isinstance(item, PairRelation) for item in relations
    ):
        raise TypeError("relations must be a tuple of PairRelation values")
    relation_map = {item.temporal_query_id: item for item in relations}
    if len(relation_map) != len(relations) or set(relation_map) != set(
        current_map.relation_ids
    ):
        raise ValueError("pair relation identities do not match the current map")
    spans = _output_spans(current_map)
    rows = tuple(
        sorted(
            (
                _row(failure, _source_group(failure, spans), relation_map)
                for failure in failures
            ),
            key=lambda item: item.failure_id,
        )
    )
    counts = Counter(row.category for row in rows)
    ghost_counts = Counter(
        row.category for row in rows if row.failure_type == "ghost_fp"
    )
    category_counts = {category: counts[category] for category in GHOST_CATEGORIES}
    complete_ghost_counts = {
        category: ghost_counts[category] for category in GHOST_CATEGORIES
    }
    unattributed = counts["UNATTRIBUTED"]
    return FailureAttributionResult(
        current_map_sha256=current_map.content_sha256(),
        evaluator_source_sha256=evaluator_source_sha256,
        relation_source_sha256=relation_source_sha256,
        source_manifest_sha256=current_map.source_manifest_sha256,
        source_visit_map_sha256=current_map.source_visit_map_sha256,
        visibility_source_sha256=current_map.visibility_source_sha256,
        relation_ids=tuple(sorted(relation_map)),
        rows=rows,
        category_counts=category_counts,
        ghost_counts=complete_ghost_counts,
        total_failure_count=len(rows),
        total_ghost_count=sum(complete_ghost_counts.values()),
        unattributed_count=unattributed,
        unattributed_share=unattributed / len(rows) if rows else 0.0,
    )


def _json_bytes(value: object, *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
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


def write_two_visit_attribution(
    result: FailureAttributionResult, output_root: str | Path
) -> Path:
    """Atomically publish exact rows and a source-bound aggregate manifest."""

    if not isinstance(result, FailureAttributionResult):
        raise TypeError("result must be a FailureAttributionResult")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"attribution output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        rows_path = staging / "failure_rows.jsonl"
        rows_content = b"".join(
            _json_bytes(row.as_dict(), indent=None) for row in result.rows
        )
        _write(rows_path, rows_content)
        summary_path = staging / "summary.json"
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "attribution_precedence": list(ATTRIBUTION_PRECEDENCE),
            "category_counts": dict(result.category_counts),
            "ghost_counts": dict(result.ghost_counts),
            "total_failure_count": result.total_failure_count,
            "total_ghost_count": result.total_ghost_count,
            "unattributed_count": result.unattributed_count,
            "unattributed_share": result.unattributed_share,
        }
        _write(summary_path, _json_bytes(summary))
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_TWO_VISIT_ATTRIBUTION_V1",
            "status": "PASS",
            "attribution_sha256": result.content_sha256(),
            "current_map_sha256": result.current_map_sha256,
            "evaluator_source_sha256": result.evaluator_source_sha256,
            "relation_source_sha256": result.relation_source_sha256,
            "source_manifest_sha256": result.source_manifest_sha256,
            "source_visit_map_sha256": list(result.source_visit_map_sha256),
            "visibility_source_sha256": result.visibility_source_sha256,
            "relation_ids": list(result.relation_ids),
            "artifacts": {
                "rows": _record(rows_path, staging),
                "summary": _record(summary_path, staging),
            },
        }
        manifest_path = staging / "manifest.json"
        _write(manifest_path, _json_bytes(manifest))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise ValueError(f"attribution output already exists: {output}")
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
    "ATTRIBUTION_PRECEDENCE",
    "FAILURE_TYPES",
    "GHOST_CATEGORIES",
    "EvaluatorFailure",
    "FailureAttributionResult",
    "FailureAttributionRow",
    "attribute_two_visit_failures",
    "write_two_visit_attribution",
]
