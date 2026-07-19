"""OVI-MAP CVPR 2026 Replica protocol and native-artifact audit."""

from __future__ import annotations

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
        "scene_artifacts": normalized_summaries,
        "aggregate_artifacts": aggregate,
    }
