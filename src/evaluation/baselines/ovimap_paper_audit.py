"""OVI-MAP CVPR 2026 Replica protocol and native-artifact audit."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PAPER_PROTOCOL: dict[str, Any] = {
    "name": "ovimap_cvpr2026_replica",
    "scene_ids": (
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
        "room0",
        "room1",
        "room2",
    ),
    "frame_count_per_scene": 200,
    "frame_step": 10,
    "semantic_vocabulary": "Replica-51",
    "semantic_metrics": "per_vertex_miou_macc_and_class_aware_mask_ap",
    "instance_metrics": "class_agnostic_mask_miou_ap25_ap50_ap75",
    "vlm": "siglip-large-patch16-384",
}

PAPER_REPORTED_REFERENCE = {
    "instance_table_2_percent": {
        "miou": 36.3,
        "ap25": 76.7,
        "ap50": 50.8,
        "ap75": 22.0,
    },
    "semantic_table_3_percent": {
        "miou": 26.5,
        "macc": 32.2,
        "ap25": 34.5,
        "ap50": 21.2,
        "apall": 8.5,
    },
}

SUMMARY_FIELDS = (
    "feature_instance_count",
    "eligible_feature_instance_count",
    "query_count",
    "average_queries_per_feature_instance",
    "full_frame_bbox_count",
    "full_frame_bbox_ratio",
)

EVIDENCE_MANIFESTS = (
    "native_mapping_manifest",
    "released_evaluation_manifest",
    "class_agnostic_ap_manifest",
)

NATIVE_INPUT_HASHES = (
    "replica51_vocabulary",
    "siglip_model",
    "replica_semantic_gt",
    "replica_instance_gt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _load_evidence_manifests(
    result: Mapping[str, Any], failures: list[str]
) -> dict[str, dict[str, Any]]:
    evidence = result.get("protocol_evidence", {})
    if not isinstance(evidence, Mapping):
        failures.append("protocol_evidence must be an object")
        evidence = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name in EVIDENCE_MANIFESTS:
        record = evidence.get(name, {})
        if not isinstance(record, Mapping):
            record = {}
        path = Path(str(record.get("path", "")))
        expected_hash = record.get("sha256")
        if not path.is_file():
            failures.append(f"protocol_evidence.{name} file is missing: {path}")
            continue
        if not _is_sha256(expected_hash) or _sha256(path) != expected_hash:
            failures.append(f"protocol_evidence.{name} SHA-256 mismatch")
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"protocol_evidence.{name} is not valid JSON")
            continue
        if not isinstance(document, dict):
            failures.append(f"protocol_evidence.{name} must contain an object")
            continue
        loaded[name] = document
    return loaded


def _validate_evidence_manifests(
    manifests: Mapping[str, Mapping[str, Any]], failures: list[str]
) -> dict[str, float]:
    scenes = list(PAPER_PROTOCOL["scene_ids"])
    validated_metrics: dict[str, float] = {}
    native = manifests.get("native_mapping_manifest")
    if native is not None:
        if native.get("status") != "COMPLETE_NATIVE_MAPPING":
            failures.append("native mapping manifest is not complete")
        if native.get("protocol_name") != PAPER_PROTOCOL["name"]:
            failures.append("native mapping manifest protocol name mismatch")
        if native.get("scene_ids") != scenes:
            failures.append("native mapping manifest scene IDs mismatch")
        expected_frames = list(
            range(
                0,
                PAPER_PROTOCOL["frame_count_per_scene"] * PAPER_PROTOCOL["frame_step"],
                PAPER_PROTOCOL["frame_step"],
            )
        )
        frame_ids = native.get("frame_ids_by_scene", {})
        if not isinstance(frame_ids, Mapping) or any(
            frame_ids.get(scene) != expected_frames for scene in scenes
        ):
            failures.append("native mapping manifest frame provenance mismatch")
        input_hashes = native.get("input_hashes", {})
        if not isinstance(input_hashes, Mapping) or any(
            not _is_sha256(input_hashes.get(name)) for name in NATIVE_INPUT_HASHES
        ):
            failures.append("native mapping manifest input hashes are incomplete")

    released = manifests.get("released_evaluation_manifest")
    if released is not None:
        if released.get("status") != "COMPLETE_RELEASED_EVALUATION":
            failures.append("released evaluation manifest is not complete")
        if released.get("source_protocol") != "released_ovimap_replica51":
            failures.append("released evaluation manifest protocol mismatch")
        if released.get("scene_ids") != scenes:
            failures.append("released evaluation manifest scene IDs mismatch")
        if released.get("frame_count_per_scene") != PAPER_PROTOCOL["frame_count_per_scene"]:
            failures.append("released evaluation manifest frame count mismatch")
        if released.get("semantic_vocabulary") != PAPER_PROTOCOL["semantic_vocabulary"]:
            failures.append("released evaluation manifest vocabulary mismatch")
        availability = released.get("paper_metric_availability", {})
        if not isinstance(availability, Mapping) or availability.get("table_3_semantic") is not True:
            failures.append("released evaluation manifest lacks Table 3 semantic metrics")
        outputs = released.get("output_artifacts", {})
        if not isinstance(outputs, Mapping):
            outputs = {}
        vertex_metrics = outputs.get("semantic_vertex_metrics", {})
        instance_metrics = outputs.get("semantic_instance_metrics", {})
        files = outputs.get("files", {})
        if not isinstance(vertex_metrics, Mapping) or not isinstance(instance_metrics, Mapping):
            failures.append("released evaluation manifest lacks validated output artifacts")
        elif not isinstance(files, Mapping) or not files:
            failures.append("released evaluation manifest lacks validated output artifacts")
        else:
            for output_name, source_name in (
                ("semantic_miou", "miou"),
                ("semantic_macc", "macc"),
            ):
                value = vertex_metrics.get(source_name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    failures.append(f"released evaluation metric is invalid: {source_name}")
                else:
                    validated_metrics[output_name] = float(value)
            for name in ("apall", "ap50", "ap25"):
                value = instance_metrics.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    failures.append(f"released semantic-instance metric is invalid: {name}")
            for name, record in files.items():
                if not isinstance(record, Mapping):
                    failures.append(f"released output artifact record is invalid: {name}")
                    continue
                path = Path(str(record.get("path", "")))
                if not path.is_file() or record.get("sha256") != _sha256(path):
                    failures.append(f"released output artifact hash mismatch: {name}")
            semantic_record = files.get("semantic_stdout", {})
            semantic_path = (
                Path(str(semantic_record.get("path", "")))
                if isinstance(semantic_record, Mapping)
                else Path("")
            )
            parsed_vertex: tuple[float, float] | None = None
            if semantic_path.is_file():
                lines = semantic_path.read_text(encoding="utf-8").splitlines()
                try:
                    header_index = lines.index("mIoU\tmAcc")
                except ValueError:
                    header_index = -1
                for line in lines[header_index + 1 :] if header_index >= 0 else ():
                    if not line.strip():
                        continue
                    try:
                        values = tuple(float(value) for value in line.split("\t"))
                    except ValueError:
                        break
                    if len(values) == 2 and all(math.isfinite(value) for value in values):
                        parsed_vertex = (values[0], values[1])
                    break
            if parsed_vertex != (
                vertex_metrics.get("miou"),
                vertex_metrics.get("macc"),
            ):
                failures.append("released semantic stdout metrics do not match manifest")

            instance_record = files.get("results_replica_json", {})
            instance_path = (
                Path(str(instance_record.get("path", "")))
                if isinstance(instance_record, Mapping)
                else Path("")
            )
            try:
                instance_output = json.loads(instance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                instance_output = {}
            if not isinstance(instance_output, Mapping) or any(
                instance_output.get(source_name) != instance_metrics.get(manifest_name)
                for manifest_name, source_name in (
                    ("apall", "all_ap"),
                    ("ap50", "all_ap_50%"),
                    ("ap25", "all_ap_25%"),
                )
            ):
                failures.append(
                    "released semantic-instance JSON metrics do not match manifest"
                )

    paper_ap = manifests.get("class_agnostic_ap_manifest")
    if paper_ap is not None:
        if paper_ap.get("status") != "COMPLETE_PAPER_AP_EVALUATION":
            failures.append("class-agnostic AP manifest is not complete")
        if paper_ap.get("protocol_name") != PAPER_PROTOCOL["name"]:
            failures.append("class-agnostic AP manifest protocol name mismatch")
        if paper_ap.get("scene_ids") != scenes:
            failures.append("class-agnostic AP manifest scene IDs mismatch")
        if paper_ap.get("metric_contract") != PAPER_PROTOCOL["instance_metrics"]:
            failures.append("class-agnostic AP manifest metric contract mismatch")
        if paper_ap.get("metrics_present") != ["miou", "ap25", "ap50", "ap75"]:
            failures.append("class-agnostic AP manifest metrics are incomplete")
        metrics = paper_ap.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        for name in ("miou", "ap25", "ap50", "ap75"):
            value = metrics.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                failures.append(f"class-agnostic AP metric is invalid: {name}")
            else:
                validated_metrics[f"class_agnostic_{name}"] = float(value)
        outputs = paper_ap.get("output_artifacts", {})
        metrics_record = outputs.get("metrics_json", {}) if isinstance(outputs, Mapping) else {}
        if not isinstance(metrics_record, Mapping):
            metrics_record = {}
        metrics_path = Path(str(metrics_record.get("path", "")))
        if not metrics_path.is_file() or metrics_record.get("sha256") != _sha256(metrics_path):
            failures.append("class-agnostic AP validated output artifact mismatch")
        else:
            try:
                output_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                output_metrics = None
            if output_metrics != dict(metrics):
                failures.append("class-agnostic AP output metrics do not match manifest")
    return validated_metrics


def summarize_feature_file(
    path: str | Path,
    *,
    image_width: int = 1200,
    image_height: int = 680,
) -> dict[str, Any]:
    """Summarize native semantic-view coverage without interpreting labels."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    with Path(path).open("rb") as handle:
        instances = pickle.load(handle)
    if not isinstance(instances, Mapping):
        raise ValueError("OVI-MAP feature file must contain an instance mapping")

    records = list(instances.values())
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("OVI-MAP feature records must be mappings")
    query_count = sum(len(record.get("frame_id", ())) for record in records)
    eligible_count = sum(len(record.get("frame_id", ())) >= 2 for record in records)
    boxes = [box for record in records for box in record.get("box_2d", ())]
    full_frame = 0
    for box in boxes:
        if len(box) != 4:
            raise ValueError("OVI-MAP query boxes must contain four coordinates")
        if (
            int(box[0]) == 0
            and int(box[1]) == 0
            and int(box[2]) >= image_width - 1
            and int(box[3]) >= image_height - 1
        ):
            full_frame += 1
    return {
        "feature_instance_count": len(records),
        "eligible_feature_instance_count": eligible_count,
        "query_count": query_count,
        "average_queries_per_feature_instance": (
            query_count / len(records) if records else 0.0
        ),
        "full_frame_bbox_count": full_frame,
        "full_frame_bbox_ratio": full_frame / len(boxes) if boxes else 0.0,
    }


def audit_paper_parity(
    result: Mapping[str, Any],
    scene_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check whether a result declares the exact paper protocol and inputs."""
    failures: list[str] = []
    method_key = str(result.get("method", {}).get("key", ""))
    if method_key != "OVIMAP":
        failures.append(f"method.key expected OVIMAP, observed {method_key or '<missing>'}")
    if result.get("status") != "VERIFIED":
        failures.append("result status must be VERIFIED before paper-parity audit")

    observed_protocol = result.get("protocol", {})
    if not isinstance(observed_protocol, Mapping):
        observed_protocol = {}
        failures.append("protocol must be an object")
    for field, expected in PAPER_PROTOCOL.items():
        observed = observed_protocol.get(field)
        if field == "scene_ids":
            observed = tuple(str(value) for value in observed or ())
        if observed != expected:
            failures.append(f"protocol.{field} expected {expected!r}, observed {observed!r}")

    replica_metrics = result.get("metrics", {}).get("replica_8_compat", {})
    observed_scenes = tuple(str(value) for value in replica_metrics.get("scene_ids", ()))
    if set(observed_scenes) != set(PAPER_PROTOCOL["scene_ids"]):
        failures.append(
            "metrics.replica_8_compat.scene_ids must contain the paper Replica-8 scenes"
        )
    if int(replica_metrics.get("scene_count", -1)) != len(PAPER_PROTOCOL["scene_ids"]):
        failures.append("metrics.replica_8_compat.scene_count must equal 8")

    evidence_manifests = _load_evidence_manifests(result, failures)
    validated_evidence_metrics = _validate_evidence_manifests(evidence_manifests, failures)

    expected_scene_set = set(PAPER_PROTOCOL["scene_ids"])
    observed_summary_set = {str(value) for value in scene_summaries}
    if observed_summary_set != expected_scene_set:
        missing = sorted(expected_scene_set - observed_summary_set)
        extra = sorted(observed_summary_set - expected_scene_set)
        failures.append(
            f"scene feature summaries must match Replica-8; missing={missing}, extra={extra}"
        )

    normalized_summaries: dict[str, dict[str, Any]] = {}
    for scene_id in sorted(observed_summary_set & expected_scene_set):
        summary = scene_summaries[scene_id]
        missing_fields = [field for field in SUMMARY_FIELDS if field not in summary]
        if missing_fields:
            failures.append(
                f"scene feature summary {scene_id} missing fields: {', '.join(missing_fields)}"
            )
            continue
        normalized_summaries[scene_id] = {
            "feature_instance_count": int(summary["feature_instance_count"]),
            "eligible_feature_instance_count": int(
                summary["eligible_feature_instance_count"]
            ),
            "query_count": int(summary["query_count"]),
            "average_queries_per_feature_instance": float(
                summary["average_queries_per_feature_instance"]
            ),
            "full_frame_bbox_count": int(summary["full_frame_bbox_count"]),
            "full_frame_bbox_ratio": float(summary["full_frame_bbox_ratio"]),
        }

    feature_count = sum(
        summary["feature_instance_count"] for summary in normalized_summaries.values()
    )
    eligible_count = sum(
        summary["eligible_feature_instance_count"]
        for summary in normalized_summaries.values()
    )
    query_count = sum(summary["query_count"] for summary in normalized_summaries.values())
    full_frame_count = sum(
        summary["full_frame_bbox_count"] for summary in normalized_summaries.values()
    )
    aggregate = {
        "feature_instance_count": feature_count,
        "eligible_feature_instance_count": eligible_count,
        "query_count": query_count,
        "average_queries_per_feature_instance": (
            query_count / feature_count if feature_count else 0.0
        ),
        "full_frame_bbox_count": full_frame_count,
        "full_frame_bbox_ratio": full_frame_count / query_count if query_count else 0.0,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "protocol_name": PAPER_PROTOCOL["name"],
        "expected_protocol": {
            **PAPER_PROTOCOL,
            "scene_ids": list(PAPER_PROTOCOL["scene_ids"]),
        },
        "observed_protocol": dict(observed_protocol),
        "paper_reported_reference": PAPER_REPORTED_REFERENCE,
        "paper_values_are_reference_only": True,
        "failures": failures,
        "validated_evidence_manifests": {
            name: {
                "status": manifest.get("status"),
                "scene_ids": manifest.get("scene_ids"),
            }
            for name, manifest in evidence_manifests.items()
        },
        "validated_evidence_metrics": validated_evidence_metrics,
        "scene_artifacts": normalized_summaries,
        "aggregate_artifacts": aggregate,
    }
