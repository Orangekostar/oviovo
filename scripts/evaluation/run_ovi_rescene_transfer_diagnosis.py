#!/usr/bin/env python3
"""Build the source-bound D0/D1/D2 transfer diagnosis and training gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.decide_ovi_rescene_adaptation import (
    build_transfer_evidence_from_rows,
    decide_transfer_adaptation,
)
from scripts.evaluation.rescene_pair_executor import NativeForwardResult
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    AssociationForwardOutputs,
    build_d1_sensor_support_view,
    build_d2_inference_bundle,
    build_temporal_query_evidence,
    load_object_transfer_config,
    measure_native_candidate_representation,
    project_cached_relations_to_sensor_support,
)
from src.evaluation.ovi_pair_views import build_ovi_object_pair_view
from src.evaluation.rscan_association_metrics import (
    evaluate_pair_relations,
)
from src.evaluation.rscan_gt_instances import load_pair_ground_truth
from src.evaluation.rscan_method_views import (
    build_method_pair_view_from_manifest,
)
from src.oviv2.two_visit_contracts import PairRelation

_DOMAINS = (
    "D0_NATIVE_PROCESSED",
    "D1_NATIVE_SENSOR_SUPPORT",
    "D2_OVI_RECONSTRUCTION",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_CONFIG_KEYS = {
    "schema_version",
    "config_id",
    "status",
    "pair_id",
    "pair_role",
    "held_out_pair_ids",
    "metric_iou_threshold",
    "confidence_threshold",
    "d1_support",
    "decision_thresholds",
    "source_bindings",
    "outputs",
}
_BINDING_KEYS = {
    "object_transfer_config",
    "selection_manifest",
    "d2_mapping_receipt",
    "d2_pair_receipt",
    "d0_pair_result",
    "d0_native_manifest",
    "d2_forward_metadata",
    "d2_forward_arrays",
    "d2_association_metrics",
    "d2_instance_metrics",
    "ground_truth_manifest",
}


class TransferDiagnosisError(ValueError):
    """Raised when transfer diagnostic evidence is malformed or unbound."""


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransferDiagnosisError(f"{label} must be a non-empty string")
    return value.strip()


def _digest(value: object, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TransferDiagnosisError(f"{label} is invalid")
    return value


def _count(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TransferDiagnosisError(f"{label} must be a non-negative integer")
    return value


def _optional_metric(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TransferDiagnosisError(f"{label} must be in [0, 1] or null")
    return float(value)


def _optional_nonnegative(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise TransferDiagnosisError(f"{label} must be non-negative or null")
    return float(value)


def build_per_pair_rows(
    *,
    pair_id: str,
    evaluated_commit: str,
    config_sha256: str,
    checkpoint_sha256: str,
    iou_threshold: float,
    domains: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Flatten three domain measurements without replacing nulls by zero."""

    pair = _nonempty(pair_id, label="pair_id")
    commit = _digest(evaluated_commit, label="evaluated commit", pattern=_GIT_SHA)
    config_hash = _digest(config_sha256, label="config SHA-256", pattern=_SHA256)
    checkpoint = _digest(
        checkpoint_sha256, label="checkpoint SHA-256", pattern=_SHA256
    )
    if iou_threshold not in {0.5, 0.25}:
        raise TransferDiagnosisError("diagnostic IoU threshold is unsupported")
    if not isinstance(domains, tuple) or tuple(
        domain.get("domain_id") if isinstance(domain, Mapping) else None
        for domain in domains
    ) != _DOMAINS:
        raise TransferDiagnosisError("diagnosis must contain ordered D0/D1/D2 domains")
    required = {
        "domain_id",
        "execution_mode",
        "input_point_count",
        "sensor_supported_point_count",
        "raw_query_count",
        "raw_nonempty_query_count",
        "confident_query_count",
        "t0_candidate_count",
        "t1_candidate_count",
        "persistent_gt_count",
        "represented_persistent_gt_count",
        "forward_runtime_s",
        "peak_memory_bytes",
        "input_sha256",
        "methods",
    }
    rows: list[dict[str, object]] = []
    for domain in domains:
        if set(domain) != required:
            raise TransferDiagnosisError("domain measurement schema is invalid")
        domain_id = str(domain["domain_id"])
        execution_mode = _nonempty(
            domain["execution_mode"], label=f"{domain_id} execution mode"
        )
        input_count = _count(
            domain["input_point_count"], label=f"{domain_id} input point count"
        )
        support_count = _count(
            domain["sensor_supported_point_count"],
            label=f"{domain_id} supported point count",
        )
        raw_count = _count(
            domain["raw_query_count"], label=f"{domain_id} raw query count"
        )
        nonempty_count = _count(
            domain["raw_nonempty_query_count"],
            label=f"{domain_id} nonempty query count",
        )
        confident_count = _count(
            domain["confident_query_count"],
            label=f"{domain_id} confident query count",
        )
        persistent_count = _count(
            domain["persistent_gt_count"], label=f"{domain_id} persistent GT count"
        )
        represented_count = _count(
            domain["represented_persistent_gt_count"],
            label=f"{domain_id} represented GT count",
        )
        if (
            input_count == 0
            or raw_count == 0
            or persistent_count == 0
            or support_count > input_count
            or nonempty_count > raw_count
            or confident_count > nonempty_count
            or represented_count > persistent_count
        ):
            raise TransferDiagnosisError("domain measurement counts are inconsistent")
        methods = domain["methods"]
        if not isinstance(methods, Mapping) or not methods:
            raise TransferDiagnosisError("domain method measurements are empty")
        for method_id, raw_method in methods.items():
            method = _nonempty(method_id, label="method_id")
            if not isinstance(raw_method, Mapping) or set(raw_method) != {
                "prediction_count",
                "true_positive_count",
                "false_positive_count",
                "precision",
                "recall",
                "postprocess_runtime_s",
            }:
                raise TransferDiagnosisError("method measurement schema is invalid")
            predictions = _count(
                raw_method["prediction_count"], label=f"{method} prediction count"
            )
            true_positives = _count(
                raw_method["true_positive_count"], label=f"{method} TP count"
            )
            false_positives = _count(
                raw_method["false_positive_count"], label=f"{method} FP count"
            )
            if true_positives + false_positives != predictions:
                raise TransferDiagnosisError("method prediction counts are inconsistent")
            precision = _optional_metric(
                raw_method["precision"], label=f"{method} precision"
            )
            recall = _optional_metric(raw_method["recall"], label=f"{method} recall")
            association_f1 = (
                None
                if precision is None or recall is None
                else 0.0
                if precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            rows.append(
                {
                    "pair_id": pair,
                    "domain_id": domain_id,
                    "status": "MEASURED",
                    "execution_mode": execution_mode,
                    "method_id": method,
                    "iou_threshold": float(iou_threshold),
                    "evaluated_commit": commit,
                    "input_point_count": input_count,
                    "sensor_supported_point_count": support_count,
                    "sensor_support_fraction": support_count / input_count,
                    "raw_query_count": raw_count,
                    "raw_nonempty_query_count": nonempty_count,
                    "raw_nonempty_fraction": nonempty_count / raw_count,
                    "confident_query_count": confident_count,
                    "confident_query_fraction": confident_count / raw_count,
                    "t0_candidate_count": _count(
                        domain["t0_candidate_count"],
                        label=f"{domain_id} t0 candidate count",
                    ),
                    "t1_candidate_count": _count(
                        domain["t1_candidate_count"],
                        label=f"{domain_id} t1 candidate count",
                    ),
                    "persistent_gt_count": persistent_count,
                    "represented_persistent_gt_count": represented_count,
                    "proposal_representation_coverage": represented_count
                    / persistent_count,
                    "prediction_count": predictions,
                    "true_positive_count": true_positives,
                    "false_positive_count": false_positives,
                    "association_precision": precision,
                    "association_recall": recall,
                    "association_f1": association_f1,
                    "forward_runtime_s": _optional_nonnegative(
                        domain["forward_runtime_s"],
                        label=f"{domain_id} forward runtime",
                    ),
                    "postprocess_runtime_s": _optional_nonnegative(
                        raw_method["postprocess_runtime_s"],
                        label=f"{method} postprocess runtime",
                    ),
                    "peak_memory_bytes": (
                        None
                        if domain["peak_memory_bytes"] is None
                        else _count(
                            domain["peak_memory_bytes"],
                            label=f"{domain_id} peak memory",
                        )
                    ),
                    "input_sha256": _digest(
                        domain["input_sha256"],
                        label=f"{domain_id} input SHA-256",
                        pattern=_SHA256,
                    ),
                    "config_sha256": config_hash,
                    "checkpoint_sha256": checkpoint,
                }
            )
    return tuple(rows)


def build_experiment_rows(
    *,
    pair_id: str,
    evaluated_commit: str,
    config_sha256: str,
    decision: Mapping[str, object],
    domain_rows: tuple[Mapping[str, object], ...],
    output_root: str,
) -> tuple[dict[str, object], ...]:
    """Record cache reuse and the gate-controlled adaptation outcome."""

    pair = _nonempty(pair_id, label="pair_id")
    commit = _digest(evaluated_commit, label="evaluated commit", pattern=_GIT_SHA)
    config_hash = _digest(config_sha256, label="config SHA-256", pattern=_SHA256)
    artifact_root = _nonempty(output_root, label="output root")
    if not isinstance(decision, Mapping):
        raise TransferDiagnosisError("decision must be a mapping")
    action = decision.get("action")
    selected_rule = _nonempty(decision.get("selected_rule"), label="selected rule")
    if action not in {
        "NO_ADAPTATION",
        "ADAPT_DECODER_MASK_HEAD",
        "UPSTREAM_PROPOSAL_GROUPING_REPAIR",
    }:
        raise TransferDiagnosisError("decision action is invalid")
    experiments: list[dict[str, object]] = []
    for domain_id in _DOMAINS:
        selected = [row for row in domain_rows if row.get("domain_id") == domain_id]
        if not selected:
            raise TransferDiagnosisError("experiment ledger lacks a domain row")
        reference = selected[0]
        experiments.append(
            {
                "run_id": f"transfer_diagnosis_{domain_id.lower()}",
                "command": reference.get("execution_mode"),
                "commit": commit,
                "source_hashes": f"input:{reference.get('input_sha256')}",
                "config_hash": config_hash,
                "dataset": "3RScan",
                "pair": pair,
                "domain": domain_id,
                "seed": "not_applicable",
                "device": "cpu" if domain_id.startswith("D1_") else "cuda:0",
                "runtime_s": reference.get("forward_runtime_s"),
                "peak_memory_bytes": reference.get("peak_memory_bytes"),
                "status": "PASS_REUSED" if domain_id != _DOMAINS[1] else "PASS_CACHED_PROJECTION",
                "artifact_root": artifact_root,
            }
        )
    status = {
        "NO_ADAPTATION": "NOT_RUN_GATE_SELECTED_NO_ADAPTATION",
        "ADAPT_DECODER_MASK_HEAD": "PENDING_GATE_REQUIRED_ADAPTATION",
        "UPSTREAM_PROPOSAL_GROUPING_REPAIR": "NOT_RUN_GATE_SELECTED_UPSTREAM_REPAIR",
    }[action]
    experiments.append(
        {
            "run_id": "decoder_mask_head_adaptation",
            "command": f"GATED_BY:{selected_rule}",
            "commit": commit,
            "source_hashes": "decision:adaptation_decision_v2.json",
            "config_hash": config_hash,
            "dataset": "3RScan",
            "pair": pair,
            "domain": "TRAIN_OVI_ADAPT",
            "seed": "not_applicable",
            "device": "not_run",
            "runtime_s": None,
            "peak_memory_bytes": None,
            "status": status,
            "artifact_root": artifact_root,
        }
    )
    return tuple(experiments)


def _regular_bytes(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise TransferDiagnosisError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise TransferDiagnosisError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise TransferDiagnosisError(f"{label} changed while being read")
    return absolute, content


def _json_object(content: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransferDiagnosisError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise TransferDiagnosisError(f"{label} must contain a JSON object")
    return value


def _bound_content(record: object, *, label: str) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise TransferDiagnosisError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TransferDiagnosisError(f"{label} binding path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute():
        path = relative
    else:
        if ".." in relative.parts:
            raise TransferDiagnosisError(f"{label} binding path escapes the repository")
        path = REPO_ROOT / relative
    absolute, content = _regular_bytes(path, label=label)
    if (
        record.get("sha256") != hashlib.sha256(content).hexdigest()
        or record.get("byte_count") != len(content)
    ):
        raise TransferDiagnosisError(f"{label} binding mismatch")
    return absolute, content


def _load_config(path: str | Path) -> tuple[Path, bytes, dict[str, object]]:
    source, content = _regular_bytes(path, label="transfer diagnosis config")
    config = _json_object(content, label="transfer diagnosis config")
    if (
        set(config) != _CONFIG_KEYS
        or config.get("schema_version") != 1
        or config.get("config_id") != "OVI_RESCENE_TRANSFER_DIAGNOSIS_V1"
        or config.get("status") != "FROZEN_BEFORE_D2_EVAL"
        or config.get("pair_role") != "D2_DEV"
    ):
        raise TransferDiagnosisError("transfer diagnosis config identity is invalid")
    pair_id = _nonempty(config.get("pair_id"), label="configured pair ID")
    held_out = config.get("held_out_pair_ids")
    if (
        not isinstance(held_out, list)
        or len(held_out) != 2
        or any(not isinstance(value, str) or not value for value in held_out)
        or pair_id in held_out
    ):
        raise TransferDiagnosisError("held-out pair declaration is invalid")
    if config.get("metric_iou_threshold") not in {0.5, 0.25}:
        raise TransferDiagnosisError("configured diagnostic IoU is invalid")
    confidence = _optional_metric(
        config.get("confidence_threshold"), label="confidence threshold"
    )
    if confidence is None:
        raise TransferDiagnosisError("confidence threshold cannot be null")
    support = config.get("d1_support")
    if (
        not isinstance(support, Mapping)
        or set(support)
        != {"definition", "maximum_distance_m", "ground_truth_used"}
        or support.get("definition")
        != "nearest_d2_dense_surface_within_distance"
        or support.get("ground_truth_used") is not False
        or _optional_nonnegative(
            support.get("maximum_distance_m"), label="D1 support distance"
        )
        in {None, 0.0}
    ):
        raise TransferDiagnosisError("D1 support config is invalid")
    bindings = config.get("source_bindings")
    outputs = config.get("outputs")
    if not isinstance(bindings, Mapping) or set(bindings) != _BINDING_KEYS:
        raise TransferDiagnosisError("diagnosis source binding schema is invalid")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "per_pair_metrics",
        "experiments",
        "decision",
    }:
        raise TransferDiagnosisError("diagnosis output schema is invalid")
    return source, content, config


def _pair_record(selection: Mapping[str, object], pair_id: str) -> Mapping[str, object]:
    pairs = selection.get("pairs")
    if not isinstance(pairs, list):
        raise TransferDiagnosisError("selection manifest pair list is invalid")
    selected = tuple(
        value
        for value in pairs
        if isinstance(value, Mapping) and value.get("pair_id") == pair_id
    )
    if len(selected) != 1:
        raise TransferDiagnosisError("configured pair is missing or duplicated")
    return selected[0]


def _relation(record: object) -> PairRelation:
    required = {
        "temporal_query_id",
        "t0_entity_ids",
        "t1_entity_ids",
        "state",
        "query_confidence",
        "identity_source",
        "evidence",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise TransferDiagnosisError("cached D0 relation schema is invalid")
    try:
        return PairRelation(
            temporal_query_id=record["temporal_query_id"],
            t0_entity_ids=tuple(record["t0_entity_ids"]),
            t1_entity_ids=tuple(record["t1_entity_ids"]),
            state=record["state"],
            query_confidence=record["query_confidence"],
            identity_source=record["identity_source"],
            evidence=record["evidence"],
        )
    except (TypeError, ValueError) as error:
        raise TransferDiagnosisError("cached D0 relation is invalid") from error


def _method_records(pair_result: Mapping[str, object]) -> dict[str, tuple[Mapping[str, object], ...]]:
    methods = pair_result.get("methods")
    method_ids = ("G_full", "F", "R_legacy", "R_supported")
    if not isinstance(methods, Mapping):
        raise TransferDiagnosisError("D0 method result schema is invalid")
    output: dict[str, tuple[Mapping[str, object], ...]] = {}
    for method_id in method_ids:
        method = methods.get(method_id)
        relations = method.get("relations") if isinstance(method, Mapping) else None
        if not isinstance(relations, list):
            raise TransferDiagnosisError(f"D0 {method_id} relations are invalid")
        output[method_id] = tuple(relations)
    return output


def _method_measurement(
    result: Mapping[str, object], *, threshold_key: str, runtime_s: float | None
) -> dict[str, object]:
    thresholds = result.get("thresholds")
    row = thresholds.get(threshold_key) if isinstance(thresholds, Mapping) else None
    if not isinstance(row, Mapping):
        raise TransferDiagnosisError("association threshold result is invalid")
    return {
        "prediction_count": row.get("paired_prediction_edges"),
        "true_positive_count": row.get("true_positive_edges"),
        "false_positive_count": row.get("false_positive_edges"),
        "precision": row.get("paired_precision"),
        "recall": row.get("paired_recall"),
        "postprocess_runtime_s": runtime_s,
    }


def _d2_method_measurement(row: Mapping[str, str]) -> dict[str, object]:
    def integer(key: str) -> int:
        try:
            return int(row[key])
        except (KeyError, ValueError) as error:
            raise TransferDiagnosisError(f"D2 metric {key} is invalid") from error

    def metric(key: str) -> float | None:
        raw = row.get(key)
        if raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError) as error:
            raise TransferDiagnosisError(f"D2 metric {key} is invalid") from error

    return {
        "prediction_count": integer("paired_prediction_count"),
        "true_positive_count": integer("true_positive_count"),
        "false_positive_count": integer("false_positive_count"),
        "precision": metric("precision"),
        "recall": metric("end_to_end_recall"),
        "postprocess_runtime_s": None,
    }


def _csv_dicts(content: bytes, *, label: str) -> tuple[dict[str, str], ...]:
    try:
        decoded = content.decode("utf-8")
        rows = tuple(csv.DictReader(io.StringIO(decoded, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise TransferDiagnosisError(f"{label} is not valid CSV") from error
    if not rows:
        raise TransferDiagnosisError(f"{label} has no rows")
    return rows


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _csv_bytes(rows: tuple[Mapping[str, object], ...]) -> bytes:
    if not rows or any(tuple(row) != tuple(rows[0]) for row in rows):
        raise TransferDiagnosisError("CSV rows are empty or inconsistent")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=tuple(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


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


def _output_path(value: object, *, label: str) -> Path:
    raw = _nonempty(value, label=label)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise TransferDiagnosisError(f"{label} must be repository-relative")
    return REPO_ROOT / relative


def run_transfer_diagnosis(
    config_path: str | Path, *, evaluated_commit: str
) -> dict[str, object]:
    """Revalidate cached artifacts and publish one deterministic diagnosis."""

    commit = _digest(evaluated_commit, label="evaluated commit", pattern=_GIT_SHA)
    config_source, config_content, config = _load_config(config_path)
    config_sha256 = hashlib.sha256(config_content).hexdigest()
    bindings = config["source_bindings"]
    outputs_config = config["outputs"]
    assert isinstance(bindings, Mapping)
    assert isinstance(outputs_config, Mapping)
    loaded = {
        name: _bound_content(bindings[name], label=name)
        for name in sorted(_BINDING_KEYS)
    }
    object_config = load_object_transfer_config(loaded["object_transfer_config"][0])
    selection = _json_object(loaded["selection_manifest"][1], label="selection manifest")
    pair_id = str(config["pair_id"])
    record = _pair_record(selection, pair_id)
    selection_sha256 = hashlib.sha256(loaded["selection_manifest"][1]).hexdigest()
    d0_pair = build_method_pair_view_from_manifest(
        pair_record=record,
        source_manifest_sha256=selection_sha256,
        domain_id="D0_NATIVE_PROCESSED",
    )
    mapping = _json_object(
        loaded["d2_mapping_receipt"][1], label="D2 mapping receipt"
    )
    visits = mapping.get("visits")
    if not isinstance(visits, list) or len(visits) != 2:
        raise TransferDiagnosisError("D2 mapping receipt visit schema is invalid")
    try:
        native_manifests = tuple(
            Path(visit["native_mapping"]["manifest"]["path"]) for visit in visits
        )
        materialized_manifests = tuple(
            Path(visit["materialized"]["manifest"]["path"]) for visit in visits
        )
    except (KeyError, TypeError) as error:
        raise TransferDiagnosisError("D2 mapping receipt paths are invalid") from error
    d2_pair = build_ovi_object_pair_view(
        pair_record=record,
        source_manifest_sha256=selection_sha256,
        native_manifests=native_manifests,
        materialized_manifests=materialized_manifests,
    )
    pair_receipt = _json_object(
        loaded["d2_pair_receipt"][1], label="D2 pair receipt"
    )
    if (
        pair_receipt.get("pair_id") != pair_id
        or pair_receipt.get("pair_content_sha256") != d2_pair.content_sha256()
    ):
        raise TransferDiagnosisError("D2 pair receipt identity mismatch")
    d0_result = _json_object(loaded["d0_pair_result"][1], label="D0 pair result")
    d0_manifest = _json_object(
        loaded["d0_native_manifest"][1], label="D0 native manifest"
    )
    if d0_result.get("pair_id") != pair_id or d0_manifest.get("pair_id") != pair_id:
        raise TransferDiagnosisError("D0 evidence names a different pair")
    ground_truth = load_pair_ground_truth(loaded["ground_truth_manifest"][0])
    support_config = config["d1_support"]
    assert isinstance(support_config, Mapping)
    d1_started = time.perf_counter()
    d1_pair, d1_audit = build_d1_sensor_support_view(
        d0_pair,
        d2_pair,
        maximum_distance_m=float(support_config["maximum_distance_m"]),
    )
    method_records = _method_records(d0_result)
    threshold = float(config["metric_iou_threshold"])
    threshold_key = "iou_0_50" if threshold == 0.5 else "iou_0_25"
    native_methods: dict[str, dict[str, object]] = {}
    supported_methods: dict[str, dict[str, object]] = {}
    d1_relations: dict[str, tuple[PairRelation, ...]] = {}
    source_methods = d0_result.get("methods")
    assert isinstance(source_methods, Mapping)
    for method_id, records_for_method in method_records.items():
        relations = tuple(_relation(value) for value in records_for_method)
        native_result = evaluate_pair_relations(
            d0_pair,
            relations,
            ground_truth,
            matching_policy="max_valid_count_then_iou",
        )
        source_method = source_methods.get(method_id)
        source_runtime = (
            source_method.get("cpu_postprocess_s")
            if isinstance(source_method, Mapping)
            else None
        )
        native_methods[method_id] = _method_measurement(
            native_result,
            threshold_key=threshold_key,
            runtime_s=float(source_runtime) if source_runtime is not None else None,
        )
        projected = project_cached_relations_to_sensor_support(
            d1_pair,
            records_for_method,
            static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        )
        d1_relations[method_id] = projected
        evaluation_started = time.perf_counter()
        supported_result = evaluate_pair_relations(
            d1_pair,
            projected,
            ground_truth,
            matching_policy="max_valid_count_then_iou",
        )
        supported_methods[method_id] = _method_measurement(
            supported_result,
            threshold_key=threshold_key,
            runtime_s=time.perf_counter() - evaluation_started,
        )
    d1_runtime_s = time.perf_counter() - d1_started
    d0_raw = d0_result.get("native_raw_queries")
    if not isinstance(d0_raw, Mapping):
        raise TransferDiagnosisError("D0 raw-query counts are invalid")
    raw_total = _count(d0_raw.get("total_count"), label="D0 raw query count")
    raw_nonempty = _count(
        d0_raw.get("nonempty_count"), label="D0 nonempty query count"
    )
    confidence_threshold = float(config["confidence_threshold"])
    d0_raw_relations = tuple(_relation(value) for value in method_records["R_legacy"])
    d0_confident = sum(
        relation.temporal_query_id.startswith("query_")
        and relation.query_confidence >= confidence_threshold
        for relation in d0_raw_relations
    )
    d1_raw_relations = tuple(
        relation
        for relation in d1_relations["R_legacy"]
        if relation.temporal_query_id.startswith("query_")
    )
    d1_nonempty = len(d1_raw_relations)
    d1_confident = sum(
        relation.query_confidence >= confidence_threshold
        for relation in d1_raw_relations
    )
    d0_representation = measure_native_candidate_representation(
        d0_pair, ground_truth, iou_threshold=threshold
    )
    d1_representation = measure_native_candidate_representation(
        d1_pair, ground_truth, iou_threshold=threshold
    )

    forward_metadata = _json_object(
        loaded["d2_forward_metadata"][1], label="D2 forward metadata"
    )
    try:
        with np.load(io.BytesIO(loaded["d2_forward_arrays"][1]), allow_pickle=False) as archive:
            if set(archive.files) != {
                "independent_t0",
                "independent_t1",
                "pred_masks_mq",
                "pred_logits_qc",
            }:
                raise TransferDiagnosisError("D2 forward array schema is invalid")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, TransferDiagnosisError):
            raise
        raise TransferDiagnosisError("D2 forward arrays are invalid") from error
    bundle = build_d2_inference_bundle(
        d2_pair,
        sampler_source_path=object_config.native_sampler_source.path,
        sampler_source_sha256=object_config.native_sampler_source.sha256,
    )
    forward = forward_metadata.get("forward")
    if not isinstance(forward, Mapping):
        raise TransferDiagnosisError("D2 forward metadata is invalid")
    try:
        d2_outputs = AssociationForwardOutputs(
            independent_model_features=(
                arrays["independent_t0"],
                arrays["independent_t1"],
            ),
            visit_forward_sha256=tuple(forward["visit_forward_sha256"]),
            independent_runtime_s=tuple(forward["independent_runtime_s"]),
            joint_forward=NativeForwardResult(
                pred_masks_mq=arrays["pred_masks_mq"],
                pred_logits_qc=arrays["pred_logits_qc"],
                runtime_s=float(forward["joint_runtime_s"]),
                peak_memory_bytes=int(forward["peak_memory_bytes"]),
                peak_reserved_memory_bytes=int(forward["peak_reserved_memory_bytes"]),
                rss_peak_bytes=int(forward["rss_peak_bytes"]),
                device_name=str(forward["device_name"]),
                model_tensor_count=int(forward["model_tensor_count"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TransferDiagnosisError("D2 cached forward metadata is malformed") from error
    d2_query_evidence = build_temporal_query_evidence(
        bundle,
        d2_outputs,
        backend_config_sha256=object_config.source_sha256,
        checkpoint_sha256=object_config.rescene_checkpoint.sha256,
    )
    d2_total = arrays["pred_logits_qc"].shape[0]
    d2_nonempty = len(d2_query_evidence.temporal_query_ids)
    d2_confident = int(
        np.count_nonzero(d2_query_evidence.query_scores >= confidence_threshold)
    )
    association_rows = tuple(
        row
        for row in _csv_dicts(
            loaded["d2_association_metrics"][1], label="D2 association metrics"
        )
        if row.get("pair_id") == pair_id
        and float(row.get("endpoint_iou_threshold", "nan")) == threshold
    )
    d2_method_ids = ("G_full", "G_supported", "F_obj", "R_obj")
    if {row.get("method_id") for row in association_rows} != set(d2_method_ids):
        raise TransferDiagnosisError("D2 association method rows are incomplete")
    association_by_method = {str(row["method_id"]): row for row in association_rows}
    d2_methods = {
        method_id: _d2_method_measurement(association_by_method[method_id])
        for method_id in d2_method_ids
    }
    proposal_row = association_by_method["G_full"]
    d2_persistent = int(proposal_row["persistent_gt_count"])
    d2_represented = int(proposal_row["representation_conditional_gt_count"])
    d0_runtime = d0_manifest.get("runtime")
    if not isinstance(d0_runtime, Mapping):
        raise TransferDiagnosisError("D0 runtime metadata is invalid")
    dense_count = sum(visit.point_count for visit in d2_pair.visits)
    d2_support_count = sum(
        int(np.count_nonzero(visit.appearance_valid)) for visit in d2_pair.visits
    )
    d0_point_count = sum(visit.point_count for visit in d0_pair.visits)
    domains = (
        {
            "domain_id": "D0_NATIVE_PROCESSED",
            "execution_mode": "REUSED_NATIVE_FORWARD_METRIC_V2_REEVALUATION",
            "input_point_count": d0_point_count,
            "sensor_supported_point_count": d0_point_count,
            "raw_query_count": raw_total,
            "raw_nonempty_query_count": raw_nonempty,
            "confident_query_count": d0_confident,
            "t0_candidate_count": len(d0_pair.visits[0].candidate_ids),
            "t1_candidate_count": len(d0_pair.visits[1].candidate_ids),
            "persistent_gt_count": d0_representation["persistent_gt_count"],
            "represented_persistent_gt_count": d0_representation[
                "represented_persistent_gt_count"
            ],
            "forward_runtime_s": d0_runtime.get("forward_s"),
            "peak_memory_bytes": d0_runtime.get("peak_gpu_bytes"),
            "input_sha256": d0_pair.method_tensor_sha256(),
            "methods": native_methods,
        },
        {
            "domain_id": "D1_NATIVE_SENSOR_SUPPORT",
            "execution_mode": "CACHED_D0_SENSOR_SUPPORT_PROJECTION_METRIC_V2",
            "input_point_count": d0_point_count,
            "sensor_supported_point_count": sum(
                int(value) for value in d1_audit["visit_support_counts"]
            ),
            "raw_query_count": raw_total,
            "raw_nonempty_query_count": d1_nonempty,
            "confident_query_count": d1_confident,
            "t0_candidate_count": len(d1_pair.visits[0].candidate_ids),
            "t1_candidate_count": len(d1_pair.visits[1].candidate_ids),
            "persistent_gt_count": d1_representation["persistent_gt_count"],
            "represented_persistent_gt_count": d1_representation[
                "represented_persistent_gt_count"
            ],
            "forward_runtime_s": None,
            "peak_memory_bytes": None,
            "input_sha256": d1_pair.method_tensor_sha256(),
            "methods": supported_methods,
        },
        {
            "domain_id": "D2_OVI_RECONSTRUCTION",
            "execution_mode": "REUSED_REAL_OVI_FORWARD_FIXED_CANDIDATES",
            "input_point_count": dense_count,
            "sensor_supported_point_count": d2_support_count,
            "raw_query_count": d2_total,
            "raw_nonempty_query_count": d2_nonempty,
            "confident_query_count": d2_confident,
            "t0_candidate_count": len(d2_pair.candidate_ids[0]),
            "t1_candidate_count": len(d2_pair.candidate_ids[1]),
            "persistent_gt_count": d2_persistent,
            "represented_persistent_gt_count": d2_represented,
            "forward_runtime_s": float(forward["joint_runtime_s"]),
            "peak_memory_bytes": int(forward["peak_memory_bytes"]),
            "input_sha256": d2_pair.content_sha256(),
            "methods": d2_methods,
        },
    )
    per_pair_rows = build_per_pair_rows(
        pair_id=pair_id,
        evaluated_commit=commit,
        config_sha256=config_sha256,
        checkpoint_sha256=object_config.rescene_checkpoint.sha256,
        iou_threshold=threshold,
        domains=domains,
    )
    evidence = build_transfer_evidence_from_rows(per_pair_rows)
    thresholds = config.get("decision_thresholds")
    if not isinstance(thresholds, Mapping):
        raise TransferDiagnosisError("decision thresholds are invalid")
    decision = decide_transfer_adaptation(evidence, thresholds)
    decision["pair_id"] = pair_id
    decision["evaluated_commit"] = commit
    decision["config_sha256"] = config_sha256
    decision["d1_cpu_runtime_s"] = d1_runtime_s
    per_pair_path = _output_path(
        outputs_config["per_pair_metrics"], label="per-pair metrics output"
    )
    experiment_path = _output_path(
        outputs_config["experiments"], label="experiment output"
    )
    decision_path = _output_path(outputs_config["decision"], label="decision output")
    for path in (per_pair_path, experiment_path, decision_path):
        if path.exists() or path.is_symlink():
            raise TransferDiagnosisError(f"diagnosis output already exists: {path}")
    per_pair_content = _csv_bytes(per_pair_rows)
    decision["source_bindings"] = {
        "diagnosis_config": {
            "path": str(config_source),
            "sha256": config_sha256,
            "byte_count": len(config_content),
        },
        "per_pair_metrics": {
            "path": str(per_pair_path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(per_pair_content).hexdigest(),
            "byte_count": len(per_pair_content),
        },
        **{name: dict(bindings[name]) for name in sorted(bindings)},
    }
    experiments = build_experiment_rows(
        pair_id=pair_id,
        evaluated_commit=commit,
        config_sha256=config_sha256,
        decision=decision,
        domain_rows=per_pair_rows,
        output_root=str(per_pair_path.parent),
    )
    _write_exclusive(per_pair_path, per_pair_content)
    _write_exclusive(decision_path, _json_bytes(decision))
    _write_exclusive(experiment_path, _csv_bytes(experiments))
    return {
        "pair_id": pair_id,
        "action": decision["action"],
        "selected_rule": decision["selected_rule"],
        "per_pair_metrics": str(per_pair_path),
        "experiments": str(experiment_path),
        "decision": str(decision_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluated-commit", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_transfer_diagnosis(
            arguments.config,
            evaluated_commit=arguments.evaluated_commit,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "TransferDiagnosisError",
    "build_experiment_rows",
    "build_per_pair_rows",
    "main",
    "run_transfer_diagnosis",
]


if __name__ == "__main__":
    raise SystemExit(main())
