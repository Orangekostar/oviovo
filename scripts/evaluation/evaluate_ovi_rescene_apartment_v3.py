#!/usr/bin/env python3
"""Build an audited Apartment B4/ReScene comparison from one native forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.compare_ovi_rescene_relations import (
    compare_relation_topologies,
    topology_key,
)
from scripts.evaluation.prepare_ovi_rescene_supported_v3 import audit_supported_input
from scripts.evaluation.run_ovi_rescene_b7 import (
    authorize_b7_scene,
    load_and_validate_b7_config,
    load_frozen_b7_inputs,
)
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.query_entity_resolver import (
    ResolverConfig,
    ResolverDomainCoverage,
    SupportedResolverResult,
    resolve_supported_queries,
)
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
)
from src.oviv2.rescene_input_bridge import load_model_input_artifact
from src.oviv2.rescene_supported_view import EntityCoverage
from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    PairRelation,
    TemporalQueryEvidence,
)

_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "backend_name",
    "pair_sha256",
    "checkpoint_sha256",
    "temporal_query_ids",
    "runtime_s",
    "peak_memory_bytes",
    "output_arrays",
}
_ARRAY_KEYS = {"token_indices", "query_masks", "token_scores", "query_scores"}
_COVERAGE_FIELDS = {
    "visit_id",
    "entity_id",
    "full_adapter_token_count",
    "supported_adapter_token_count",
    "full_source_point_count",
    "supported_source_point_count",
    "full_model_token_count",
    "supported_model_token_count",
}


class ApartmentEvaluationError(ValueError):
    """Raised when an Apartment evidence package is not source-bound."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _regular_path(path: str | Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        record = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ApartmentEvaluationError(f"{label} is unavailable") from error
    if not stat.S_ISREG(record.st_mode):
        raise ApartmentEvaluationError(f"{label} must be a regular non-symlink file")
    return absolute


def _stream_record(
    path: str | Path, *, label: str, recorded_path: str | None = None
) -> dict[str, object]:
    absolute = _regular_path(path, label=label)
    before = absolute.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with absolute.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise ApartmentEvaluationError(f"{label} cannot be read") from error
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or byte_count != before.st_size:
        raise ApartmentEvaluationError(f"{label} changed while being read")
    return {
        "path": recorded_path if recorded_path is not None else str(absolute),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    absolute = _regular_path(path, label=label)
    try:
        value = json.loads(absolute.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApartmentEvaluationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ApartmentEvaluationError(f"{label} must contain a JSON object")
    return value


def _validate_source_record(record: object, *, label: str) -> dict[str, object]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ApartmentEvaluationError(f"{label} record schema is invalid")
    path = record.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ApartmentEvaluationError(f"{label} path must be absolute")
    observed = _stream_record(path, label=label)
    if dict(record) != observed:
        raise ApartmentEvaluationError(f"{label} binding mismatch")
    return observed


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_query_evidence_artifact(
    manifest_path: str | Path, pair: NeuralSampleMap
) -> TemporalQueryEvidence:
    """Load one executor result after validating its manifest and array permutation."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be a NeuralSampleMap")
    path = _regular_path(manifest_path, label="query evidence manifest")
    manifest_record = _stream_record(path, label="query evidence manifest")
    manifest = _json_object(path, label="query evidence manifest")
    if (
        set(manifest) != _MANIFEST_KEYS
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("backend_name") != "concerto"
        or manifest.get("pair_sha256") != pair.content_sha256()
    ):
        raise ApartmentEvaluationError("query evidence identity is invalid")
    query_ids = manifest.get("temporal_query_ids")
    if (
        not isinstance(query_ids, list)
        or not query_ids
        or any(not isinstance(value, str) or not value for value in query_ids)
        or len(query_ids) != len(set(query_ids))
    ):
        raise ApartmentEvaluationError("query evidence IDs are invalid")
    binding = manifest.get("output_arrays")
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ApartmentEvaluationError("query evidence array binding is invalid")
    relative = binding.get("path")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or Path(relative).name != relative
    ):
        raise ApartmentEvaluationError("query evidence array path is invalid")
    arrays_path = path.parent / relative
    observed = _stream_record(
        arrays_path, label="query evidence arrays", recorded_path=relative
    )
    if dict(binding) != observed:
        raise ApartmentEvaluationError("query evidence array binding mismatch")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_KEYS:
                raise ApartmentEvaluationError("query evidence array schema is invalid")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ApartmentEvaluationError):
            raise
        raise ApartmentEvaluationError("query evidence arrays cannot be decoded") from error
    token_count = len(pair.visit_ids)
    query_count = len(query_ids)
    indices = arrays["token_indices"]
    if (
        indices.shape != (token_count,)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.issubdtype(indices.dtype, np.bool_)
        or not np.array_equal(np.sort(indices), np.arange(token_count))
    ):
        raise ApartmentEvaluationError("query token indices are not a permutation")
    raw_masks = arrays["query_masks"]
    raw_scores = arrays["token_scores"]
    if (
        raw_masks.dtype != np.bool_
        or raw_masks.shape != (query_count, token_count)
        or raw_scores.shape != raw_masks.shape
        or not np.issubdtype(raw_scores.dtype, np.floating)
    ):
        raise ApartmentEvaluationError("query token arrays have invalid shapes")
    restored_masks = np.empty_like(raw_masks)
    restored_scores = np.empty_like(raw_scores, dtype=np.float32)
    restored_masks[:, indices] = raw_masks
    restored_scores[:, indices] = raw_scores
    method_identity = {
        "backend_name": manifest["backend_name"],
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "pair_sha256": manifest["pair_sha256"],
        "schema_version": manifest["schema_version"],
    }
    try:
        evidence = TemporalQueryEvidence(
            status="PASS",
            backend_name=f"rescene:{manifest['backend_name']}",
            backend_config_sha256=_sha256_json(method_identity),
            pair_sha256=manifest["pair_sha256"],
            temporal_query_ids=tuple(query_ids),
            query_masks=restored_masks,
            token_scores=restored_scores,
            query_scores=arrays["query_scores"],
            checkpoint_sha256=manifest.get("checkpoint_sha256"),
            ranking_eligible=True,
            runtime_s=manifest.get("runtime_s"),
            peak_memory_bytes=manifest.get("peak_memory_bytes"),
            diagnostics={
                "output_arrays_sha256": str(observed["sha256"]),
                "output_manifest_sha256": str(manifest_record["sha256"]),
                "token_order_restored": "true",
            },
        )
        validate_query_evidence(pair, evidence)
    except (TypeError, ValueError) as error:
        raise ApartmentEvaluationError("query evidence contract is invalid") from error
    return evidence


def _load_entity_coverage(root: Path) -> tuple[EntityCoverage, ...]:
    payload = _json_object(root / "entity_coverage.json", label="entity coverage")
    entities = payload.get("entities")
    if (
        set(payload) != {
            "schema_version",
            "artifact_id",
            "status",
            "unsupported_entity_keys",
            "entities",
        }
        or payload.get("schema_version") != 1
        or payload.get("artifact_id") != "OVI_RESCENE_ENTITY_COVERAGE_V3"
        or payload.get("status") != "PASS"
        or not isinstance(entities, list)
        or not entities
    ):
        raise ApartmentEvaluationError("entity coverage identity is invalid")
    try:
        rows = tuple(
            EntityCoverage(**row)
            for row in entities
            if isinstance(row, dict) and set(row) == _COVERAGE_FIELDS
        )
    except (TypeError, ValueError) as error:
        raise ApartmentEvaluationError("entity coverage record is invalid") from error
    if len(rows) != len(entities):
        raise ApartmentEvaluationError("entity coverage record schema is invalid")
    return rows


def _relation_payload(relation: PairRelation) -> dict[str, object]:
    return {
        "temporal_query_id": str(relation.temporal_query_id),
        "t0_entity_ids": list(relation.t0_entity_ids),
        "t1_entity_ids": list(relation.t1_entity_ids),
        "state": relation.state,
        "query_confidence": relation.query_confidence,
        "evidence": dict(relation.evidence),
        "identity_source": relation.identity_source,
    }


def _validated_relations(
    values: Sequence[PairRelation], *, label: str
) -> tuple[PairRelation, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ApartmentEvaluationError(f"{label} relations must be a sequence")
    relations = tuple(values)
    if any(not isinstance(value, PairRelation) for value in relations):
        raise ApartmentEvaluationError(f"{label} contains an invalid relation")
    identifiers = [str(value.temporal_query_id) for value in relations]
    if len(identifiers) != len(set(identifiers)):
        raise ApartmentEvaluationError(f"{label} relation IDs are duplicated")
    return relations


def _relation_collection(
    *, artifact_id: str, relations: tuple[PairRelation, ...]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "status": "PASS",
        "raw_count": len(relations),
        "unique_count": len({topology_key(value) for value in relations}),
        "state_counts": dict(sorted(Counter(value.state for value in relations).items())),
        "relations": [
            _relation_payload(value)
            for value in sorted(relations, key=lambda item: str(item.temporal_query_id))
        ],
    }


def _delta_payload(delta: object) -> dict[str, object]:
    result = asdict(delta)
    for key in ("intersection", "b4_only", "b5_only"):
        result[key] = [
            [state, list(t0), list(t1)] for state, t0, t1 in result[key]
        ]
    return result


def _visual_cases(
    result: SupportedResolverResult, evidence: TemporalQueryEvidence
) -> tuple[dict[str, object], ...]:
    cases: list[dict[str, object]] = []
    used: set[str] = set()
    if result.relations:
        relation = min(
            result.relations,
            key=lambda value: (-value.query_confidence, str(value.temporal_query_id)),
        )
        query_id = str(relation.temporal_query_id)
        used.add(query_id)
        cases.append(
            {
                "role": "representative",
                "query_id": query_id,
                "selection_rule": "highest_confidence_supported_relation",
                "entity_keys": [
                    [0, value] for value in relation.t0_entity_ids
                ]
                + [[1, value] for value in relation.t1_entity_ids],
            }
        )
    candidates = sorted(
        (
            value
            for value in result.evidence.values()
            if value.token_intersection_count > 0 and value.query_id not in used
        ),
        key=lambda value: (
            value.supported_entity_token_coverage,
            value.full_entity_token_evidence,
            value.query_id,
            value.visit_id,
            value.entity_id,
        ),
    )
    if candidates:
        selected = candidates[0]
        related = sorted(
            (
                value
                for value in result.evidence.values()
                if value.query_id == selected.query_id
                and value.token_intersection_count > 0
            ),
            key=lambda value: (
                value.visit_id,
                -value.dominance_score,
                value.entity_id,
            ),
        )
        best_by_visit: dict[int, object] = {selected.visit_id: selected}
        for value in related:
            best_by_visit.setdefault(value.visit_id, value)
        cases.append(
            {
                "role": "low_support",
                "query_id": selected.query_id,
                "selection_rule": "minimum_nonzero_supported_entity_coverage",
                "selected_supported_entity_token_coverage": (
                    selected.supported_entity_token_coverage
                ),
                "entity_keys": [
                    [visit, best_by_visit[visit].entity_id]
                    for visit in sorted(best_by_visit)
                ],
            }
        )
    return tuple(cases)


def _project_points_to_pixels(
    coordinates: np.ndarray,
    reference: np.ndarray,
    *,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Project points with axes and bounds fixed by the full entity geometry."""

    points = np.asarray(coordinates, dtype=np.float64)
    basis = np.asarray(reference, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ApartmentEvaluationError("visual points must have shape (N, 3)")
    if basis.ndim != 2 or basis.shape[1:] != (3,) or not len(basis):
        raise ApartmentEvaluationError("visual reference must have shape (N, 3)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(basis)):
        raise ApartmentEvaluationError("visual geometry must be finite")
    if not len(coordinates):
        return np.empty((0, 2), dtype=np.int32)
    left, top, right, bottom = bounds
    spans = np.ptp(basis, axis=0)
    axes = tuple(np.argsort(-spans, kind="stable")[:2])
    projected = points[:, axes]
    reference_projected = basis[:, axes]
    low = np.min(reference_projected, axis=0)
    high = np.max(reference_projected, axis=0)
    extent = np.maximum(high - low, 1e-9)
    normalized = (projected - low) / extent
    pixels_x = left + np.rint(normalized[:, 0] * (right - left - 1)).astype(int)
    pixels_y = bottom - np.rint(normalized[:, 1] * (bottom - top - 1)).astype(int)
    return np.column_stack((pixels_x, pixels_y)).astype(np.int32, copy=False)


def _paint_points(
    canvas: np.ndarray,
    coordinates: np.ndarray,
    *,
    reference: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    pixels = _project_points_to_pixels(coordinates, reference, bounds=bounds)
    if not len(pixels):
        return
    left, top, right, bottom = bounds
    pixels_x = pixels[:, 0]
    pixels_y = pixels[:, 1]
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = np.clip(pixels_x + dx, left, right - 1)
        y = np.clip(pixels_y + dy, top, bottom - 1)
        canvas[y, x] = color


def _render_case(
    path: Path,
    *,
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    case: Mapping[str, object],
) -> None:
    query_id = str(case["query_id"])
    query_index = evidence.temporal_query_ids.index(query_id)
    assert evidence.query_masks is not None
    query_mask = evidence.query_masks[query_index]
    keys = tuple((int(value[0]), str(value[1])) for value in case["entity_keys"])
    panel_count = max(1, len(keys))
    panel_width = 420
    height = 340
    canvas = np.full((height, panel_count * panel_width, 3), 248, dtype=np.uint8)
    for panel, (visit_id, entity_id) in enumerate(keys):
        left = panel * panel_width + 24
        right = (panel + 1) * panel_width - 24
        bounds = (left, 52, right, height - 24)
        entity_mask = (pair.visit_ids == visit_id) & np.asarray(
            [value == entity_id for value in pair.token_entity_ids], dtype=np.bool_
        )
        indices = np.flatnonzero(entity_mask)
        selected = np.flatnonzero(entity_mask & query_mask)
        _paint_points(
            canvas,
            pair.coordinates_xyzt[indices, :3],
            reference=pair.coordinates_xyzt[indices, :3],
            bounds=bounds,
            color=(80, 125, 160),
        )
        _paint_points(
            canvas,
            pair.coordinates_xyzt[selected, :3],
            reference=pair.coordinates_xyzt[indices, :3],
            bounds=bounds,
            color=(220, 55, 45),
        )
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), f"{case['role']} | {query_id}", fill=(20, 20, 20))
    for panel, (visit_id, entity_id) in enumerate(keys):
        draw.text(
            (panel * panel_width + 24, 30),
            f"t{visit_id} {entity_id}",
            fill=(20, 20, 20),
        )
    image.save(path, format="PNG", optimize=True)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def evaluate_apartment_artifact(
    *,
    supported_root: str | Path,
    evidence_manifest_path: str | Path,
    b4_relations: Sequence[PairRelation],
    b4_source_record: Mapping[str, object],
    output_root: str | Path,
    projection_config: ProjectionConfig,
    resolver_config: ResolverConfig,
) -> dict[str, Any]:
    """Audit inputs, resolve one forward twice, and atomically publish the report."""

    if not isinstance(projection_config, ProjectionConfig):
        raise TypeError("projection_config must be ProjectionConfig")
    if not isinstance(resolver_config, ResolverConfig):
        raise TypeError("resolver_config must be ResolverConfig")
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise ApartmentEvaluationError(f"report output already exists: {output}")
    supported = Path(os.path.abspath(os.fspath(supported_root)))
    try:
        receipt = audit_supported_input(supported)
        pair = load_neural_sample_artifact(supported / "adapter_pair")
        model_input = load_model_input_artifact(supported / "model_input")
    except (OSError, TypeError, ValueError) as error:
        raise ApartmentEvaluationError("supported V3 input audit failed") from error
    evidence = load_query_evidence_artifact(evidence_manifest_path, pair)
    entity_coverage = _load_entity_coverage(supported)
    b4 = _validated_relations(b4_relations, label="B4")
    b4_source = _validate_source_record(b4_source_record, label="B4 source")
    try:
        legacy = project_queries_to_instances(
            pair, evidence, projection_config
        ).relations
        domains = ResolverDomainCoverage(
            full_source_count=receipt["full_domains"]["source"],
            supported_source_count=receipt["supported_domains"]["source"],
            full_adapter_count=receipt["full_domains"]["adapter"],
            supported_adapter_count=receipt["supported_domains"]["adapter"],
            full_model_count=receipt["full_domains"]["model"],
            supported_model_count=receipt["supported_domains"]["model"],
        )
        resolved = resolve_supported_queries(
            pair=pair,
            evidence=evidence,
            entity_coverage=entity_coverage,
            adapter_to_model=model_input.adapter_to_model,
            domain_coverage=domains,
            config=resolver_config,
        )
    except (TypeError, ValueError) as error:
        raise ApartmentEvaluationError("Apartment relation resolution failed") from error
    supported_relations = resolved.relations
    b4_delta_legacy = compare_relation_topologies(b4, legacy)
    b4_delta_supported = compare_relation_topologies(b4, supported_relations)
    cases = _visual_cases(resolved, evidence)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        payloads = {
            "b4_relations": (
                "b4_relations.json",
                _relation_collection(
                    artifact_id="OVI_RESCENE_APARTMENT_B4_RELATIONS_V3",
                    relations=b4,
                ),
            ),
            "legacy_relations": (
                "legacy_relations.json",
                _relation_collection(
                    artifact_id="OVI_RESCENE_APARTMENT_LEGACY_RELATIONS_V3",
                    relations=legacy,
                ),
            ),
            "supported_relations": (
                "supported_relations.json",
                _relation_collection(
                    artifact_id="OVI_RESCENE_APARTMENT_SUPPORTED_RELATIONS_V3",
                    relations=supported_relations,
                ),
            ),
            "resolver_diagnostics": (
                "resolver_diagnostics.json",
                {
                    "schema_version": 1,
                    "artifact_id": "OVI_RESCENE_APARTMENT_RESOLVER_DIAGNOSTICS_V3",
                    "status": "PASS",
                    "fallback_query_ids": list(resolved.fallback_query_ids),
                    "conflict_query_ids": list(resolved.conflict_query_ids),
                    "one_sided_query_ids": list(resolved.one_sided_query_ids),
                    "collision_model_count": resolved.collision_model_count,
                    "duplicate_topology_count": resolved.duplicate_topology_count,
                    "evidence": [
                        asdict(value)
                        for _, value in sorted(resolved.evidence.items())
                    ],
                },
            ),
        }
        artifacts: dict[str, dict[str, object]] = {}
        for role, (relative, payload) in payloads.items():
            path = staging / relative
            _write(path, _json_bytes(payload))
            artifacts[role] = _stream_record(
                path, label=role, recorded_path=relative
            )
        visual_root = staging / "visuals"
        visual_root.mkdir()
        for case in cases:
            relative = f"visuals/{case['role']}.png"
            path = staging / relative
            _render_case(path, pair=pair, evidence=evidence, case=case)
            role = f"{case['role']}_visual"
            artifacts[role] = _stream_record(
                path, label=role, recorded_path=relative
            )
        legacy_unique = len({topology_key(value) for value in legacy})
        supported_unique = len({topology_key(value) for value in supported_relations})
        b4_unique = len({topology_key(value) for value in b4})
        coverage = dict(resolved.input_coverage)
        coverage["conditional_query_coverage"] = resolved.conditional_query_coverage
        evidence_manifest = _regular_path(
            evidence_manifest_path, label="query evidence manifest"
        )
        evidence_payload = _json_object(
            evidence_manifest, label="query evidence manifest"
        )
        evidence_arrays = evidence_manifest.parent / evidence_payload["output_arrays"]["path"]
        summary: dict[str, Any] = {
            "schema_version": 3,
            "artifact_id": "OVI_RESCENE_APARTMENT_EVIDENCE_V3",
            "status": "PASS",
            "scene": "apartment",
            "pair_sha256": pair.content_sha256(),
            "prediction_sha256": evidence.prediction_sha256(),
            "source_bindings": {
                "supported_input_contract": _stream_record(
                    supported / "input_contract_v3.json",
                    label="supported input contract",
                ),
                "query_evidence_manifest": _stream_record(
                    evidence_manifest, label="query evidence manifest"
                ),
                "query_evidence_arrays": _stream_record(
                    evidence_arrays, label="query evidence arrays"
                ),
                "b4_source": b4_source,
            },
            "thresholds": {
                "legacy_projection": asdict(projection_config),
                "supported_resolver": asdict(resolver_config),
            },
            "runtime": {
                "native_forward_s": evidence.runtime_s,
                "peak_gpu_memory_bytes": evidence.peak_memory_bytes,
            },
            "counts": {
                "b4_raw": len(b4),
                "b4_unique": b4_unique,
                "legacy_raw": len(legacy),
                "legacy_unique": legacy_unique,
                "supported_raw": len(supported_relations),
                "supported_unique": supported_unique,
                "supported_fallback": len(resolved.fallback_query_ids),
                "supported_conflict": len(resolved.conflict_query_ids),
                "supported_one_sided": len(resolved.one_sided_query_ids),
                "model_collision": resolved.collision_model_count,
                "duplicate_topology": resolved.duplicate_topology_count,
            },
            "coverage": coverage,
            "b4_vs_legacy": _delta_payload(b4_delta_legacy),
            "b4_vs_supported": _delta_payload(b4_delta_supported),
            "visual_selection": list(cases),
            "artifacts": artifacts,
        }
        _write(staging / "summary.json", _json_bytes(summary))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def evaluate_from_frozen_b7(
    *,
    supported_root: str | Path,
    evidence_manifest_path: str | Path,
    b7_config_path: str | Path,
    output_root: str | Path,
    projection_config: ProjectionConfig,
    resolver_config: ResolverConfig,
) -> dict[str, Any]:
    """Rebuild frozen B4 once, then publish the Apartment report."""

    config_path = _regular_path(b7_config_path, label="frozen B7 config")
    try:
        config = load_and_validate_b7_config(
            config_path,
            verify_source_bindings=True,
            verify_frozen_inputs=False,
        )
        authorize_b7_scene(config, "apartment")
        frozen = load_frozen_b7_inputs(config)
    except (OSError, TypeError, ValueError) as error:
        raise ApartmentEvaluationError("frozen B4 rebuild failed") from error
    return evaluate_apartment_artifact(
        supported_root=supported_root,
        evidence_manifest_path=evidence_manifest_path,
        b4_relations=frozen.relations,
        b4_source_record=_stream_record(config_path, label="frozen B7 config"),
        output_root=output_root,
        projection_config=projection_config,
        resolver_config=resolver_config,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supported-root", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--b7-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        summary = evaluate_from_frozen_b7(
            supported_root=args.supported_root,
            evidence_manifest_path=args.evidence_manifest,
            b7_config_path=args.b7_config,
            output_root=args.output,
            projection_config=ProjectionConfig(),
            resolver_config=ResolverConfig(),
        )
    except (ApartmentEvaluationError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(Path(args.output).absolute() / "summary.json"),
                "status": summary["status"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ApartmentEvaluationError",
    "evaluate_apartment_artifact",
    "evaluate_from_frozen_b7",
    "load_query_evidence_artifact",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
