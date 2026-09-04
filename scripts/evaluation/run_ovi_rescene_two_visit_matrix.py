#!/usr/bin/env python3
"""Validate and execute the frozen OVI-ReScene B0-B6 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.freeze_tesse_two_visit_protocol import (
    OFFICE_HELD_OUT_STATUS,
    PROTOCOL_ID,
    protocol_content_sha256,
)

MATRIX_ID = "OVI_RESCENE_TWO_VISIT_B0_B6_V1"
MATRIX_STATUS = "FROZEN_BEFORE_EXECUTION"
BLOCKED_STATUS = "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
_REQUIRED_STATUS = "REQUIRED"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_RECORD_KEYS = {"path", "sha256", "byte_count"}
_METRIC_GROUPS = (
    "current_state",
    "geometry",
    "identity",
    "semantics",
    "systems",
    "artifacts",
)
_SOURCE_ROLES = {
    "external_sources",
    "adapter",
    "deterministic_executor",
    "geometric_reasoner",
    "composer",
    "composition_contracts",
    "native_ovi_runner",
    "metric_implementation",
    "snapshot_metrics",
    "common_v2_evaluator",
    "matrix_runner",
    "semantic_crosswalk",
    "semantic_vocabulary_apartment",
    "semantic_vocabulary_office",
    "visibility_execution",
    "visit_loader",
}
_ROW_ARTIFACT_KEYS = {
    "command",
    "stdout_log",
    "stderr_log",
    "output_artifact",
    "metric_receipt",
    "status_receipt",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "matrix_id",
    "status",
    "dataset",
    "diagnostic_amendment",
    "composition_amendment",
    "protocol",
    "freeze_parent_commit",
    "source_bindings",
    "method_config",
    "metric_contract",
    "success_gates",
    "office_policy",
    "preregistration_amendment",
    "execution_contract",
    "rows",
}
_ROW_KEYS = {
    "id",
    "name",
    "uses_ovi_t0",
    "uses_ovi_t1",
    "pair_reasoner",
    "signed_visibility",
    "final_geometry_sources",
    "purpose",
    "availability",
    "blocked_reason",
    "artifacts",
}
_ROW_CONTRACTS = (
    (
        "B0",
        "OVI_T0_ONLY",
        True,
        False,
        "none",
        False,
        ("ovi_t0",),
        "old_map_ghost",
    ),
    (
        "B1",
        "OVI_UNION",
        True,
        True,
        "none",
        False,
        ("ovi_t0", "ovi_t1"),
        "completeness_and_ghost_upper_case",
    ),
    (
        "B2",
        "OVI_T1_ONLY",
        False,
        True,
        "none",
        False,
        ("ovi_t1",),
        "low_ghost_floor_and_completeness_lower_bound",
    ),
    (
        "B3",
        "OVI_VISIBILITY_COMPOSE",
        True,
        True,
        "none",
        True,
        ("ovi_t0", "ovi_t1"),
        "signed_visibility_contribution",
    ),
    (
        "B4",
        "OVI_GEOMETRIC_PAIRING_VISIBILITY",
        True,
        True,
        "geometric_semantic",
        True,
        ("ovi_t0", "ovi_t1"),
        "nonlearned_identity_baseline",
    ),
    (
        "B5",
        "OVI_RESCENE_VISIBILITY",
        True,
        True,
        "rescene_concerto",
        True,
        ("ovi_t0", "ovi_t1"),
        "proposed_full_system",
    ),
    (
        "B6",
        "OVI_RESCENE_NO_VISIBILITY",
        True,
        True,
        "rescene_concerto",
        False,
        ("ovi_t0", "ovi_t1"),
        "signed_visibility_necessity",
    ),
)


class MatrixError(ValueError):
    """Raised when a frozen matrix or execution artifact is invalid."""


@dataclass(frozen=True, slots=True)
class VariantExecutionArtifacts:
    command_line: tuple[str, ...]
    stdout_log: Path
    stderr_log: Path
    output_artifact: Path
    metric_receipt: Path

    def __post_init__(self) -> None:
        if (
            not isinstance(self.command_line, tuple)
            or not self.command_line
            or any(not isinstance(item, str) or not item for item in self.command_line)
        ):
            raise MatrixError("variant command line must be a non-empty string tuple")
        for name in (
            "stdout_log",
            "stderr_log",
            "output_artifact",
            "metric_receipt",
        ):
            path = getattr(self, name)
            if not isinstance(path, Path):
                raise MatrixError(f"{name} must be a Path")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise MatrixError(f"symlink path component is not allowed: {current}")


def _regular_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(absolute)
    before = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise MatrixError(f"{label} must be a regular file")
    data = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(data) != after.st_size:
        raise MatrixError(f"{label} changed while reading")
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatrixError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise MatrixError(f"{label} must contain a JSON object")
    return value


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    data = _regular_bytes(absolute, label="artifact")
    record_path = str(absolute)
    if root is not None:
        try:
            record_path = absolute.relative_to(root).as_posix()
        except ValueError as error:
            raise MatrixError("artifact escapes matrix run root") from error
    return {
        "path": record_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _bound_record(
    record: object,
    *,
    label: str,
    verify: bool,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise MatrixError(f"{label} record schema is invalid")
    raw_path = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise MatrixError(f"{label} record values are invalid")
    declared = Path(raw_path)
    if declared.is_absolute() or ".." in declared.parts:
        raise MatrixError(f"{label} path must be repository-relative")
    path = REPO_ROOT / declared
    normalized = dict(record)
    if verify and _record(path) != {**normalized, "path": str(path.absolute())}:
        raise MatrixError(f"{label} source binding mismatch")
    return path, normalized


def _validate_metric_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "distance_threshold_m",
        "groups",
        "na_policy",
        "aggregation",
    }:
        raise MatrixError("metric contract schema is invalid")
    if value.get("distance_threshold_m") != 0.05:
        raise MatrixError("metric distance threshold must remain 0.05 m")
    if value.get("na_policy") != "null_with_reason_never_zero_fill":
        raise MatrixError("metric N/A policy is invalid")
    if value.get("aggregation") != "pareto_vector_no_hidden_scalar":
        raise MatrixError("metric aggregation must remain Pareto-only")
    groups = value.get("groups")
    if not isinstance(groups, Mapping) or tuple(groups) != _METRIC_GROUPS:
        raise MatrixError("metric groups are invalid")
    for name, fields in groups.items():
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(item, str) or not item for item in fields)
            or len(fields) != len(set(fields))
        ):
            raise MatrixError(f"metric group {name} is invalid")
    return dict(value)


def _validate_method_config(value: object) -> None:
    expected = {
        "ovi_mapping_voxel_size_m": 0.01,
        "ovi_semantics": {
            "classifier": "siglip_l_16_384_canonical_relative",
            "vocabulary_paths": {
                "apartment": (
                    "configs/evaluation/vocabularies/tesse_cd_apartment_ovi_full.json"
                ),
                "office": (
                    "configs/evaluation/vocabularies/tesse_cd_office_ovi_full.json"
                ),
            },
            "canonical_prompts": ["object", "things", "stuff", "texture"],
            "maximum_text_length": 64,
            "minimum_observation_count": 2,
        },
        "signed_visibility": {
            "voxel_size_m": 0.05,
            "depth_tolerance_m": 0.1,
            "depth_max_m": 10.0,
            "minimum_absent_fraction": 0.8,
            "minimum_absent_observations": 6,
            "minimum_distinct_viewpoints": 3,
            "minimum_viewpoint_baseline_m": 0.25,
        },
        "adapter": {"neural_voxel_size_m": 0.02, "feature_schema": "rgb"},
        "geometric_reasoner": {
            "comparison_voxel_size_m": 0.05,
            "maximum_centroid_distance_m": 2.0,
            "minimum_pair_score": 0.5,
            "require_known_label_compatibility": True,
            "weights": {
                "voxel_iou": 0.2,
                "symmetric_coverage": 0.1,
                "centroid": 0.25,
                "size": 0.15,
                "extent": 0.1,
                "semantic": 0.2,
            },
        },
        "rescene": {
            "backend": "concerto",
            "neural_voxel_size_m": 0.02,
            "checkpoint_status": BLOCKED_STATUS,
            "random_initialization_ranking_eligible": False,
        },
        "composer": {
            "composition_voxel_size_m": 0.05,
            "authority_rule": (
                "t1_occupied_then_visible_free_then_entity_visible_free_"
                "then_t0_occluded_or_unobserved"
            ),
            "removal_authority": "t1_signed_visible_free_only",
            "semantic_authority": "ovi",
            "entity_scope": "complete_scene_object_vocabulary_only",
            "minimum_entity_visible_free_fraction": 0.8,
            "minimum_entity_visible_free_voxels": 3,
        },
        "evaluator_voxel_size_m": 0.05,
    }
    if value != expected:
        raise MatrixError("method configuration differs from the frozen contract")


def _validate_success_gates(value: object) -> None:
    expected = {
        "ghost_delta_over_b2_max": 0.02,
        "practical_ghost_max": 0.2,
        "strong_ghost_max": 0.1,
        "minimum_object_f1": 0.367952,
        "preferred_object_f1": 0.372762,
        "change_f1_reference": 0.088458,
        "maximum_change_f1_drop": 0.005,
        "minimum_unobserved_recall_gain_over_b2": 0.05,
        "minimum_rescene_gain_over_b4": 0.02,
        "selection": "all_applicable_gates_no_weighted_average",
    }
    if value != expected:
        raise MatrixError("success gates differ from the preregistration")


def _validate_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(_ROW_CONTRACTS):
        raise MatrixError("matrix must contain exactly B0-B6")
    normalized: list[dict[str, Any]] = []
    for row, expected in zip(value, _ROW_CONTRACTS, strict=True):
        variant_id = expected[0]
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            raise MatrixError(f"{variant_id} contract fields are invalid")
        observed = (
            row.get("id"),
            row.get("name"),
            row.get("uses_ovi_t0"),
            row.get("uses_ovi_t1"),
            row.get("pair_reasoner"),
            row.get("signed_visibility"),
            tuple(row.get("final_geometry_sources", ())),
            row.get("purpose"),
        )
        if observed != expected:
            raise MatrixError(f"{variant_id} contract differs from the frozen row")
        availability = row.get("availability")
        blocked_reason = row.get("blocked_reason")
        if variant_id in {"B5", "B6"}:
            if availability != BLOCKED_STATUS or blocked_reason != BLOCKED_STATUS:
                raise MatrixError(f"{variant_id} blocked status is invalid")
        elif availability != _REQUIRED_STATUS or blocked_reason is not None:
            raise MatrixError(f"{variant_id} required status is invalid")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != _ROW_ARTIFACT_KEYS:
            raise MatrixError(f"{variant_id} artifact contract is invalid")
        prefix = f"{variant_id}/"
        if any(
            not isinstance(path, str)
            or not path.startswith(prefix)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in artifacts.values()
        ):
            raise MatrixError(f"{variant_id} artifact paths are invalid")
        normalized.append(dict(row))
    return normalized


def load_and_validate_matrix(
    path: Path,
    *,
    verify_source_bindings: bool = True,
) -> dict[str, Any]:
    """Load the frozen matrix and verify every static source binding."""

    payload = _json_object(_regular_bytes(path, label="matrix"), label="matrix")
    if (
        set(payload) != _TOP_LEVEL_KEYS
        or payload.get("schema_version") != 1
        or payload.get("matrix_id") != MATRIX_ID
        or payload.get("status") != MATRIX_STATUS
        or payload.get("dataset") != "TESSE-CD"
    ):
        raise MatrixError("matrix identity or schema is invalid")
    parent = payload.get("freeze_parent_commit")
    if not isinstance(parent, str) or _GIT_SHA.fullmatch(parent) is None:
        raise MatrixError("freeze parent commit is invalid")
    if payload.get("preregistration_amendment") != {
        "status": "FROZEN_BEFORE_FIRST_METHOD_SCORE",
        "parent_commit": "1c94cee0ce16deb79c184d19de6f30936bbe9de6",
        "reason": "complete_exact_execution_and_method_bindings",
        "method_results_inspected": False,
    }:
        raise MatrixError("preregistration amendment is invalid")
    if payload.get("diagnostic_amendment") != {
        "status": "FROZEN_AFTER_BASELINE_DIAGNOSIS_BEFORE_TEMPORAL_METHOD_SCORE",
        "parent_commit": "3fb383a5f4e64b2465aa8459d9da57d643ba7eaf",
        "reason": (
            "complete_ovi_native_label_space_and_partition_known_nonobjects_"
            "from_object_metrics"
        ),
        "baseline_results_inspected": ["B0", "B1", "B2"],
        "method_results_inspected": False,
    }:
        raise MatrixError("diagnostic amendment is invalid")
    if payload.get("composition_amendment") != {
        "status": "FROZEN_AFTER_B3_B4_FAILURE_DIAGNOSIS_BEFORE_CORRECTED_RERUN",
        "parent_commit": "185a26ad75b1d0bce71cf983ebb0310842d59c58",
        "reason": "lift_strong_voxel_visible_free_evidence_to_ovi_object_entity",
        "method_results_inspected": ["B3", "B4"],
        "observed_ghost": {
            "B2": 0.0,
            "B3": 0.9961844725,
            "B4": 0.9961844725,
        },
        "threshold_source": "reuse_frozen_signed_visibility_thresholds",
    }:
        raise MatrixError("composition amendment is invalid")

    protocol_path, _protocol_record = _bound_record(
        payload.get("protocol"),
        label="protocol",
        verify=verify_source_bindings,
    )
    protocol_payload = _json_object(
        _regular_bytes(protocol_path, label="protocol"), label="protocol"
    )
    if protocol_payload.get("protocol_id") != PROTOCOL_ID or protocol_content_sha256(
        protocol_payload
    ) != payload.get("execution_contract", {}).get("protocol_content_sha256"):
        raise MatrixError("matrix protocol content binding is invalid")

    sources = payload.get("source_bindings")
    if not isinstance(sources, Mapping) or set(sources) != _SOURCE_ROLES:
        raise MatrixError("matrix source binding roles are invalid")
    for role in sorted(sources):
        _bound_record(
            sources[role], label=f"source {role}", verify=verify_source_bindings
        )
    _validate_method_config(payload.get("method_config"))
    _validate_metric_contract(payload.get("metric_contract"))
    _validate_success_gates(payload.get("success_gates"))
    office = payload.get("office_policy")
    if office != {
        "status": OFFICE_HELD_OUT_STATUS,
        "attempt_count": 0,
        "release_required_bindings": [
            "adapter_config",
            "temporal_reasoner",
            "composer_policy",
            "baseline_matrix",
            "metric_contract",
            "success_gates",
        ],
    }:
        raise MatrixError("Office held-out policy is invalid")
    execution = payload.get("execution_contract")
    if not isinstance(execution, Mapping) or set(execution) != {
        "protocol_content_sha256",
        "source_commit_policy",
        "required_variants",
        "blocked_variants",
        "publish_policy",
    }:
        raise MatrixError("matrix execution contract is invalid")
    if (
        execution.get("source_commit_policy") != "record_exact_runtime_head"
        or execution.get("required_variants") != ["B0", "B1", "B2", "B3", "B4"]
        or execution.get("blocked_variants") != ["B5", "B6"]
        or execution.get("publish_policy") != "atomic_only_after_all_required_pass"
        or not isinstance(execution.get("protocol_content_sha256"), str)
        or _SHA256.fullmatch(execution["protocol_content_sha256"]) is None
    ):
        raise MatrixError("matrix execution settings are invalid")
    payload["rows"] = _validate_rows(payload.get("rows"))
    return payload


def _validate_metric_receipt(
    path: Path,
    *,
    variant_id: str,
    run_id: str,
    metric_contract: Mapping[str, Any],
) -> None:
    payload = _json_object(
        _regular_bytes(path, label=f"{variant_id} metric receipt"),
        label=f"{variant_id} metric receipt",
    )
    if set(payload) != {
        "schema_version",
        "status",
        "variant_id",
        "run_id",
        "metric_groups",
        "unavailable",
        "method_input_bindings",
        "evaluation_only_bindings",
    } or (
        payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or payload.get("variant_id") != variant_id
        or payload.get("run_id") != run_id
    ):
        raise MatrixError(f"{variant_id} metric receipt identity is invalid")
    groups = payload.get("metric_groups")
    unavailable = payload.get("unavailable")
    method_bindings = payload.get("method_input_bindings")
    evaluation_bindings = payload.get("evaluation_only_bindings")
    expected_groups = metric_contract["groups"]
    if not isinstance(groups, Mapping) or set(groups) != set(_METRIC_GROUPS):
        raise MatrixError(f"{variant_id} metric groups are invalid")
    if not isinstance(unavailable, Mapping):
        raise MatrixError(f"{variant_id} unavailable reasons are invalid")
    if (
        not isinstance(method_bindings, Mapping)
        or not method_bindings
        or not isinstance(evaluation_bindings, Mapping)
        or not evaluation_bindings
        or set(method_bindings) & set(evaluation_bindings)
    ):
        raise MatrixError(f"{variant_id} method/evaluation bindings are not isolated")
    null_names: set[str] = set()
    for group_name, metric_names in expected_groups.items():
        values = groups.get(group_name)
        if not isinstance(values, Mapping) or set(values) != set(metric_names):
            raise MatrixError(f"{variant_id} {group_name} metrics are incomplete")
        for metric_name, metric_value in values.items():
            if metric_value is None:
                null_names.add(metric_name)
            elif (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(float(metric_value))
                or float(metric_value) < 0.0
            ):
                raise MatrixError(f"{variant_id} metric {metric_name} is invalid")
    if set(unavailable) != null_names or any(
        not isinstance(reason, str) or not reason for reason in unavailable.values()
    ):
        raise MatrixError(f"{variant_id} N/A metrics lack exact reasons")


def _write_atomic(path: Path, payload: object) -> None:
    content = _canonical_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _GIT_SHA.fullmatch(result) is None:
        raise MatrixError("runtime Git HEAD is invalid")
    return result


def execute_required_matrix(
    *,
    matrix_path: Path,
    scene: str,
    output_root: Path,
    source_commit: str,
    executor: Callable[[Mapping[str, Any], str, Path], VariantExecutionArtifacts],
) -> Path:
    """Run all required baselines atomically and emit learned-backend blockers."""

    matrix = load_and_validate_matrix(matrix_path)
    if scene != "apartment":
        raise MatrixError(OFFICE_HELD_OUT_STATUS)
    if not isinstance(source_commit, str) or _GIT_SHA.fullmatch(source_commit) is None:
        raise MatrixError("source commit is invalid")
    if source_commit != _runtime_head():
        raise MatrixError("source commit does not match runtime Git HEAD")
    if not callable(executor):
        raise TypeError("executor must be callable")
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise MatrixError(f"matrix output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    matrix_bytes = _regular_bytes(matrix_path, label="matrix")
    matrix_digest = hashlib.sha256(matrix_bytes).hexdigest()
    variants: list[dict[str, object]] = []
    artifact_witnesses: list[tuple[Path, dict[str, object]]] = []
    try:
        for row in matrix["rows"]:
            variant_id = row["id"]
            run_id = f"{matrix_digest[:12]}-{scene}-{variant_id}"
            variant_root = staging / variant_id
            variant_root.mkdir()
            if row["availability"] == BLOCKED_STATUS:
                blocker = {
                    "schema_version": 1,
                    "status": BLOCKED_STATUS,
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "ranking_eligible": False,
                    "reason": BLOCKED_STATUS,
                    "metrics": None,
                    "source_commit": source_commit,
                    "matrix_sha256": matrix_digest,
                }
                blocker_path = variant_root / "blocked-receipt.json"
                _write_atomic(blocker_path, blocker)
                blocker_record = _record(blocker_path, root=staging)
                artifact_witnesses.append((blocker_path, blocker_record))
                variants.append(
                    {
                        "variant_id": variant_id,
                        "run_id": run_id,
                        "status": BLOCKED_STATUS,
                        "receipt": blocker_record,
                    }
                )
                continue

            artifacts = executor(row, run_id, variant_root)
            if not isinstance(artifacts, VariantExecutionArtifacts):
                raise MatrixError(f"{variant_id} executor returned an invalid result")
            declared_artifacts = row["artifacts"]
            paths = {
                "stdout_log": artifacts.stdout_log,
                "stderr_log": artifacts.stderr_log,
                "output_artifact": artifacts.output_artifact,
                "metric_receipt": artifacts.metric_receipt,
            }
            for name, path in paths.items():
                observed_path = Path(os.path.abspath(os.fspath(path)))
                expected_path = staging / declared_artifacts[name]
                if observed_path != expected_path:
                    raise MatrixError(
                        f"{variant_id} {name} differs from the frozen artifact path"
                    )
            command_path = staging / declared_artifacts["command"]
            _write_atomic(
                command_path,
                {
                    "schema_version": 1,
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "argv": list(artifacts.command_line),
                },
            )
            records = {
                name: _record(path, root=staging) for name, path in paths.items()
            }
            records["command"] = _record(command_path, root=staging)
            artifact_witnesses.extend(
                (paths[name], record)
                for name, record in records.items()
                if name in paths
            )
            artifact_witnesses.append((command_path, records["command"]))
            _validate_metric_receipt(
                artifacts.metric_receipt,
                variant_id=variant_id,
                run_id=run_id,
                metric_contract=matrix["metric_contract"],
            )
            receipt = {
                "schema_version": 1,
                "status": "PASS",
                "variant_id": variant_id,
                "run_id": run_id,
                "source_commit": source_commit,
                "matrix_sha256": matrix_digest,
                "command_line": list(artifacts.command_line),
                "artifacts": records,
            }
            receipt_path = variant_root / "run-receipt.json"
            if receipt_path != staging / declared_artifacts["status_receipt"]:
                raise MatrixError(f"{variant_id} status receipt path is invalid")
            _write_atomic(receipt_path, receipt)
            receipt_record = _record(receipt_path, root=staging)
            artifact_witnesses.append((receipt_path, receipt_record))
            variants.append(
                {
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "status": "PASS",
                    "receipt": receipt_record,
                }
            )
        if [item["status"] for item in variants[:5]] != ["PASS"] * 5:
            raise MatrixError("not all required B0-B4 variants passed")
        if _regular_bytes(matrix_path, label="matrix") != matrix_bytes:
            raise MatrixError("matrix changed before publish")
        if source_commit != _runtime_head():
            raise MatrixError("source commit changed before publish")
        load_and_validate_matrix(matrix_path)
        for path, expected in artifact_witnesses:
            if _record(path, root=staging) != expected:
                raise MatrixError(f"artifact changed before matrix publish: {path}")
        summary = {
            "schema_version": 1,
            "matrix_id": MATRIX_ID,
            "status": "MATRIX_EXECUTION_COMPLETE_WITH_BLOCKED_VARIANTS",
            "scene": scene,
            "source_commit": source_commit,
            "matrix": {
                "path": str(Path(matrix_path).absolute()),
                "sha256": matrix_digest,
                "byte_count": len(matrix_bytes),
            },
            "variants": variants,
        }
        summary_path = staging / "matrix-summary.json"
        _write_atomic(summary_path, summary)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "matrix-summary.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/ovi_rescene_two_visit_matrix.json",
    )
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    matrix = load_and_validate_matrix(args.matrix)
    print(
        json.dumps(
            {
                "status": "MATRIX_CONTRACT_PASS",
                "matrix_id": matrix["matrix_id"],
                "rows": [row["id"] for row in matrix["rows"]],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BLOCKED_STATUS",
    "MATRIX_ID",
    "MatrixError",
    "VariantExecutionArtifacts",
    "execute_required_matrix",
    "load_and_validate_matrix",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
