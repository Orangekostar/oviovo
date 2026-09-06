#!/usr/bin/env python3
"""Build a provenance-bound Apartment ownership and completion report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.prepare_ovi_rescene_supported_v3 import audit_supported_input
from src.evaluation.ovi_ownership_completion import build_ownership_readout
from src.evaluation.ovi_pair_views import OviObjectPairView
from src.evaluation.rscan_gt_instances import GroundTruthPair
from src.evaluation.temporal_object_groups import (
    TemporalObjectGrouping,
    evaluate_current_instance_grouping,
)
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.two_visit_contracts import NeuralSampleMap, PairRelation

_RECORD_KEYS = {"path", "sha256", "byte_count"}
_CONFIG_KEYS = {
    "schema_version",
    "config_id",
    "scene",
    "supported_root",
    "apartment_report",
    "b3_metrics",
    "b7_metrics",
    "rscan_matrix",
    "output",
}
_RELATION_KEYS = {
    "temporal_query_id",
    "t0_entity_ids",
    "t1_entity_ids",
    "state",
    "query_confidence",
    "evidence",
    "identity_source",
}
_DENSE_VARIANTS = ("A_ID", "U0", "U1", "U2", "U3")
_INSTANCE_METRIC_KEYS = (
    "status",
    "predicted_instance_count",
    "ground_truth_instance_count",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "precision",
    "recall",
    "f_score",
    "fragment_gt_count",
    "merge_prediction_count",
    "merge_prediction_rate",
    "duplicate_prediction_count",
    "mean_matched_iou",
)


class OwnershipReportError(ValueError):
    """Raised when an ownership/completion input is invalid or unbound."""


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


def _regular_bytes(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise OwnershipReportError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise OwnershipReportError(f"{label} must be a regular non-symlink file")
    try:
        content = absolute.read_bytes()
    except OSError as error:
        raise OwnershipReportError(f"{label} cannot be read") from error
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise OwnershipReportError(f"{label} changed while being read")
    return absolute, content


def _file_record(
    path: str | Path, *, label: str, recorded_path: str | None = None
) -> dict[str, object]:
    absolute, content = _regular_bytes(path, label=label)
    return {
        "path": recorded_path if recorded_path is not None else str(absolute),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OwnershipReportError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise OwnershipReportError(f"{label} must contain a JSON object")
    return value


def _load_bound_json(record: object, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise OwnershipReportError(f"{label} binding schema is invalid")
    path = record.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise OwnershipReportError(f"{label} path must be absolute")
    absolute, content = _regular_bytes(path, label=label)
    observed = {
        "path": str(absolute),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    if dict(record) != observed:
        raise OwnershipReportError(f"{label} binding mismatch")
    return absolute, _json_object(content, label=label)


def _load_relative_json(
    root: Path, record: object, *, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise OwnershipReportError(f"{label} binding schema is invalid")
    relative = record.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise OwnershipReportError(f"{label} path must be local and relative")
    path = root / relative
    observed = _file_record(path, label=label, recorded_path=relative)
    if dict(record) != observed:
        raise OwnershipReportError(f"{label} binding mismatch")
    _, content = _regular_bytes(path, label=label)
    return _json_object(content, label=label), observed


def _relations(payload: object, *, label: str) -> tuple[PairRelation, ...]:
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or not isinstance(payload.get("relations"), list)
    ):
        raise OwnershipReportError(f"{label} relation collection is invalid")
    rows = payload["relations"]
    try:
        relations = tuple(
            PairRelation(**row)
            for row in rows
            if isinstance(row, dict) and set(row) == _RELATION_KEYS
        )
    except (TypeError, ValueError) as error:
        raise OwnershipReportError(f"{label} relation is invalid") from error
    if len(relations) != len(rows):
        raise OwnershipReportError(f"{label} relation schema is invalid")
    return relations


def _finite_metrics(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise OwnershipReportError(f"{label} metrics are invalid")
    metrics: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        if (
            not isinstance(raw_key, str)
            or not raw_key
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            raise OwnershipReportError(f"{label} metrics are invalid")
        metrics[raw_key] = float(raw_value)
    return dict(sorted(metrics.items()))


def _compact_ownership(
    pair: NeuralSampleMap,
    relations: Sequence[PairRelation],
    *,
    variant_id: str,
    source: str,
) -> dict[str, object]:
    source_relations = tuple(relations)
    if pair.entity_semantics:
        supported_entities = {
            (item.visit_id, item.entity_id) for item in pair.entity_semantics
        }
    else:
        supported_entities = set(
            zip(pair.visit_ids.tolist(), pair.token_entity_ids, strict=True)
        )
    applied: list[PairRelation] = []
    excluded: list[str] = []
    for relation in source_relations:
        members = {
            *((0, entity_id) for entity_id in relation.t0_entity_ids),
            *((1, entity_id) for entity_id in relation.t1_entity_ids),
        }
        if members and members <= supported_entities:
            applied.append(relation)
        else:
            excluded.append(str(relation.temporal_query_id))
    readout = build_ownership_readout(pair, applied, variant_id=variant_id)
    identity_sha256 = hashlib.sha256(
        json.dumps(
            readout["identity_groups"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        key: value
        for key, value in readout.items()
        if key not in {"entity_geometry_sha256", "identity_groups"}
    } | {
        "identity_assignment_sha256": identity_sha256,
        "relation_source": source,
        "source_relation_count": len(source_relations),
        "relation_count": len(applied),
        "excluded_out_of_domain_relation_ids": sorted(excluded),
        "relation_domain_policy": "exclude_whole_relation_if_any_endpoint_has_no_token",
    }


def assemble_ownership_completion_report(
    *,
    pair: NeuralSampleMap,
    geometric_relations: Sequence[PairRelation],
    rescene_relations: Sequence[PairRelation],
    b3_payload: Mapping[str, object],
    b7_payload: Mapping[str, object],
    rscan_matrix_payload: Mapping[str, object],
    source_bindings: Mapping[str, object],
) -> dict[str, object]:
    """Assemble measured rows while preserving explicit unavailable denominators."""

    if (
        b3_payload.get("schema_version") != 1
        or b3_payload.get("status") != "PASS"
        or b3_payload.get("variant_id") != "B3"
    ):
        raise OwnershipReportError("B3 metric receipt identity is invalid")
    groups = b3_payload.get("metric_groups")
    if not isinstance(groups, Mapping):
        raise OwnershipReportError("B3 metric groups are invalid")
    merged_b3: dict[str, object] = {}
    for name in ("current_state", "geometry", "systems"):
        group = groups.get(name, {})
        if not isinstance(group, Mapping):
            raise OwnershipReportError("B3 metric groups are invalid")
        merged_b3.update(group)
    if (
        b7_payload.get("schema_version") != 1
        or b7_payload.get("status") != "PASS"
        or b7_payload.get("variant_id") != "B7-G"
    ):
        raise OwnershipReportError("B7-G metric receipt identity is invalid")
    b7_metrics = _finite_metrics(b7_payload.get("metrics"), label="B7-G")
    try:
        b3_metrics = _finite_metrics(
            {name: merged_b3[name] for name in b7_metrics}, label="B3 shared"
        )
    except KeyError as error:
        raise OwnershipReportError("B3 shared metric is missing") from error
    if b3_metrics != b7_metrics:
        raise OwnershipReportError("B7-G B3 reference metrics do not match B3")
    b3_reference = b7_payload.get("b3_reference")
    if not isinstance(b3_reference, Mapping) or any(
        b3_reference.get(name) != value for name, value in b3_metrics.items()
    ):
        raise OwnershipReportError("B7-G B3 reference is invalid")
    diagnostics = b7_payload.get("diagnostics")
    attribution = b7_payload.get("attribution")
    if not isinstance(diagnostics, Mapping) or not isinstance(attribution, Mapping):
        raise OwnershipReportError("B7-G diagnostics or attribution are invalid")
    attribution_summary = attribution.get("summary")
    if not isinstance(attribution_summary, Mapping):
        raise OwnershipReportError("B7-G attribution summary is invalid")
    if (
        rscan_matrix_payload.get("schema_version") != 1
        or rscan_matrix_payload.get("artifact_id")
        != "RSCAN_T2_METHOD_MATRIX_AGGREGATE_V1"
        or rscan_matrix_payload.get("status") != "PASS"
    ):
        raise OwnershipReportError("3RScan method matrix identity is invalid")
    domains = rscan_matrix_payload.get("domain_coverage", {})
    if not isinstance(domains, Mapping):
        raise OwnershipReportError("3RScan domain coverage is invalid")
    transfer_boundary = {
        "D0_NATIVE_PROCESSED": domains.get("D0_NATIVE_PROCESSED", "PASS"),
        "D1_NATIVE_SENSOR_SUPPORT": domains.get(
            "D1_NATIVE_SENSOR_SUPPORT", "MISSING_ASSET"
        ),
        "D2_OVI_RECONSTRUCTION": domains.get(
            "D2_OVI_RECONSTRUCTION", "MISSING_ASSET"
        ),
    }
    if transfer_boundary["D2_OVI_RECONSTRUCTION"] != "MISSING_ASSET":
        raise OwnershipReportError("unexpected 3RScan D2 availability")

    ownership = [
        _compact_ownership(pair, (), variant_id="A0", source="none"),
        _compact_ownership(
            pair,
            geometric_relations,
            variant_id="A1",
            source="Apartment B4 geometric relations",
        ),
        _compact_ownership(
            pair,
            rescene_relations,
            variant_id="A2",
            source="Apartment supported ReScene relations",
        ),
    ]
    if len({row["geometry_sha256"] for row in ownership}) != 1:
        raise OwnershipReportError("ownership variants changed entity geometry")
    completion = [
        {
            "variant_id": "C0",
            "status": "PASS_MEASURED_B3",
            "identity_source": "none",
            "registration_source": "none",
            "oracle_only": False,
            "metrics": b3_metrics,
            "completion_surface": None,
            "unavailable_reason": "NO_INSTANCE_IDENTITY_GT_FOR_OPPORTUNITY_SURFACE",
        },
        {
            "variant_id": "C1",
            "status": "PASS_MEASURED_PRIOR_B7_G",
            "identity_source": "Apartment B4 geometric relations",
            "registration_source": "estimated_trimmed_icp",
            "oracle_only": False,
            "metrics": b7_metrics,
            "completion_surface": None,
            "unavailable_reason": "NO_INSTANCE_IDENTITY_GT_FOR_OPPORTUNITY_SURFACE",
            "diagnostics": dict(diagnostics),
            "attribution": dict(attribution_summary),
        },
        {
            "variant_id": "C2",
            "status": "NOT_COMPUTED_MISSING_R_BOUND_RECOVERY_ARTIFACT",
            "identity_source": "Apartment supported ReScene relations",
            "registration_source": "estimated_trimmed_icp",
            "oracle_only": False,
            "metrics": None,
            "completion_surface": None,
        },
        {
            "variant_id": "O1",
            "status": "NOT_COMPUTED_MISSING_INSTANCE_IDENTITY_GT",
            "identity_source": "ground_truth",
            "registration_source": "estimated_trimmed_icp",
            "oracle_only": True,
            "metrics": None,
            "completion_surface": None,
        },
        {
            "variant_id": "O2",
            "status": "NOT_COMPUTED_MISSING_INSTANCE_IDENTITY_AND_TRANSFORM_GT",
            "identity_source": "ground_truth",
            "registration_source": "ground_truth",
            "oracle_only": True,
            "metrics": None,
            "completion_surface": None,
        },
    ]
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OWNERSHIP_COMPLETION_V1",
        "status": "PASS_WITH_UNAVAILABLE_ROWS",
        "scene": "apartment",
        "pair_sha256": pair.content_sha256(),
        "claim_boundary": (
            "real Apartment ownership organization and prior measured B3/B7-G; "
            "instance AP/PQ and opportunity-surface completion are unavailable "
            "without instance identity GT; 3RScan D2 is unavailable"
        ),
        "shared_geometry_visibility": {
            "status": "PASS",
            "geometry_sha256": ownership[0]["geometry_sha256"],
            "geometry_mutation_count": 0,
            "visibility_policy": "unchanged across A0/A1/A2",
        },
        "ownership": ownership,
        "completion": completion,
        "transfer_boundary": transfer_boundary,
        "source_bindings": dict(source_bindings),
    }


def write_instance_readout_csv(
    path: str | Path,
    rows: tuple[Mapping[str, object], ...],
) -> Path:
    """Write dense instance rows as an atomically replaced LF CSV."""

    if not isinstance(rows, tuple) or not rows:
        raise OwnershipReportError("instance readout rows are empty")
    fields = tuple(rows[0])
    if not fields or any(tuple(row) != fields for row in rows):
        raise OwnershipReportError("instance readout rows use inconsistent fields")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def build_instance_readout_rows(
    pair: OviObjectPairView,
    groupings: Mapping[str, TemporalObjectGrouping],
    ground_truth: GroundTruthPair,
    *,
    evaluated_commit: str,
    config_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Flatten measured dense t1 instance quality for all declared variants."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if tuple(groupings) != _DENSE_VARIANTS:
        raise OwnershipReportError("instance readout variants or order are invalid")
    for value, length, label in (
        (evaluated_commit, 40, "evaluated commit"),
        (config_sha256, 64, "config SHA-256"),
        (checkpoint_sha256, 64, "checkpoint SHA-256"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise OwnershipReportError(f"{label} is invalid")
    current = pair.visits[1]
    owned = current.entity_owner_indices >= 0
    owned_count = int(np.count_nonzero(owned))
    supported_count = int(np.count_nonzero(owned & current.appearance_valid))
    support_rate = None if owned_count == 0 else supported_count / owned_count
    rows: list[dict[str, object]] = []
    for variant_id in _DENSE_VARIANTS:
        grouping = groupings[variant_id]
        if (
            not isinstance(grouping, TemporalObjectGrouping)
            or grouping.variant_id != variant_id
            or grouping.pair_content_sha256 != pair.content_sha256()
        ):
            raise OwnershipReportError("instance grouping identity is invalid")
        metrics = evaluate_current_instance_grouping(grouping, ground_truth)
        current_conflicts = {
            value.entity_id
            for value in grouping.rejected_conflicts
            if value.visit_id == 1
        }
        objects = grouping.objects[1]
        common = {
            "pair_id": pair.pair_id,
            "variant_id": variant_id,
            "evaluated_commit": evaluated_commit,
            "t0_source_candidate_count": len(grouping.candidate_ids[0]),
            "t1_source_candidate_count": len(grouping.candidate_ids[1]),
            "current_composite_object_count": len(objects),
            "current_grouped_source_entity_count": sum(
                len(value.member_entity_ids) for value in objects if value.is_grouped
            ),
            "current_query_supported_object_count": sum(
                bool(value.query_ids) for value in objects
            ),
            "unconfirmed_atomic_object_count": sum(
                not value.query_ids for value in objects
            ),
            "rejected_conflict_count": len(grouping.rejected_conflicts),
            "current_rejected_entity_count": len(current_conflicts),
            "current_owned_point_count": owned_count,
            "appearance_supported_owned_point_count": supported_count,
            "appearance_support_rate": support_rate,
            "geometry_unchanged": (
                grouping.source_xyz_multiset_sha256
                == grouping.output_xyz_multiset_sha256
            ),
            "pair_content_sha256": grouping.pair_content_sha256,
            "grouping_sha256": grouping.content_sha256(),
            "source_xyz_multiset_sha256": grouping.source_xyz_multiset_sha256,
            "output_xyz_multiset_sha256": grouping.output_xyz_multiset_sha256,
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
        for threshold in ("0.50", "0.25"):
            threshold_metrics = metrics[threshold]
            rows.append(
                common
                | {"iou_threshold": float(threshold_metrics["iou_threshold"])}
                | {
                    key: threshold_metrics[key]
                    for key in _INSTANCE_METRIC_KEYS
                }
            )
    return tuple(rows)


def evaluate_from_config(config_path: str | Path) -> tuple[Path, dict[str, object]]:
    """Validate every source binding and publish the configured result once."""

    config_absolute, config_content = _regular_bytes(config_path, label="config")
    config = _json_object(config_content, label="config")
    if (
        set(config) != _CONFIG_KEYS
        or config.get("schema_version") != 1
        or config.get("config_id") != "OVI_RESCENE_OWNERSHIP_COMPLETION_CONFIG_V1"
        or config.get("scene") != "apartment"
    ):
        raise OwnershipReportError("configuration identity is invalid")
    supported_root = config.get("supported_root")
    output = config.get("output")
    if (
        not isinstance(supported_root, str)
        or not Path(supported_root).is_absolute()
        or not isinstance(output, str)
        or not Path(output).is_absolute()
    ):
        raise OwnershipReportError("configuration paths must be absolute")
    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise OwnershipReportError(f"output already exists: {output_path}")
    try:
        audit_supported_input(supported_root)
        pair = load_neural_sample_artifact(Path(supported_root) / "adapter_pair")
    except (OSError, TypeError, ValueError) as error:
        raise OwnershipReportError("supported Apartment input audit failed") from error
    apartment_path, apartment = _load_bound_json(
        config.get("apartment_report"), label="Apartment report"
    )
    if (
        apartment.get("schema_version") != 3
        or apartment.get("artifact_id") != "OVI_RESCENE_APARTMENT_EVIDENCE_V3"
        or apartment.get("status") != "PASS"
        or apartment.get("scene") != "apartment"
        or apartment.get("pair_sha256") != pair.content_sha256()
    ):
        raise OwnershipReportError("Apartment report identity is invalid")
    artifacts = apartment.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OwnershipReportError("Apartment report artifact inventory is invalid")
    b4_payload, b4_record = _load_relative_json(
        apartment_path.parent, artifacts.get("b4_relations"), label="B4 relations"
    )
    r_payload, r_record = _load_relative_json(
        apartment_path.parent,
        artifacts.get("supported_relations"),
        label="supported ReScene relations",
    )
    _b3_path, b3 = _load_bound_json(config.get("b3_metrics"), label="B3 metrics")
    _b7_path, b7 = _load_bound_json(config.get("b7_metrics"), label="B7-G metrics")
    _matrix_path, matrix = _load_bound_json(
        config.get("rscan_matrix"), label="3RScan matrix"
    )
    source_bindings = {
        "config": {
            "path": str(config_absolute),
            "sha256": hashlib.sha256(config_content).hexdigest(),
            "byte_count": len(config_content),
        },
        "supported_input_contract": _file_record(
            Path(supported_root) / "input_contract_v3.json",
            label="supported input contract",
        ),
        "apartment_report": dict(config["apartment_report"]),
        "b4_relations": b4_record,
        "rescene_relations": r_record,
        "b3_metrics": dict(config["b3_metrics"]),
        "b7_metrics": dict(config["b7_metrics"]),
        "rscan_matrix": dict(config["rscan_matrix"]),
    }
    report = assemble_ownership_completion_report(
        pair=pair,
        geometric_relations=_relations(b4_payload, label="B4"),
        rescene_relations=_relations(r_payload, label="supported ReScene"),
        b3_payload=b3,
        b7_payload=b7,
        rscan_matrix_payload=matrix,
        source_bindings=source_bindings,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path, report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        output, report = evaluate_from_config(arguments.config)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "status": report["status"]}, sort_keys=True))
    return 0


__all__ = [
    "OwnershipReportError",
    "assemble_ownership_completion_report",
    "build_instance_readout_rows",
    "evaluate_from_config",
    "main",
    "write_instance_readout_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
