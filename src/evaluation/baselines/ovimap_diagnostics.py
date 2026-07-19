"""Parse released OVI-MAP evaluator output without changing metric meaning."""

from __future__ import annotations

from collections.abc import Sequence
import math


_INSTANCE_HEADER = (
    "mIoU",
    "wIoU",
    "mP@75",
    "mR@75",
    "mP@50",
    "mR@50",
    "mP@25",
    "mR@25",
)
_INSTANCE_NAMES = (
    "mean_iou",
    "weighted_iou",
    "mean_precision_at_75",
    "mean_recall_at_75",
    "mean_precision_at_50",
    "mean_recall_at_50",
    "mean_precision_at_25",
    "mean_recall_at_25",
)
_SEMANTIC_HEADER = ("mIoU", "mAcc")
_SEMANTIC_NAMES = ("mean_iou", "mean_accuracy")


def _parse_one_row(
    text: str,
    *,
    expected_header: Sequence[str],
    metric_names: Sequence[str],
    scene_id: str,
) -> dict:
    normalized_scene = str(scene_id).strip()
    if not normalized_scene:
        raise ValueError("scene_id must be non-empty")
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    header = "\t".join(expected_header)
    try:
        header_index = lines.index(header)
    except ValueError as error:
        raise ValueError(f"released OVI-MAP evaluator header not found: {header}") from error
    rows = lines[header_index + 1 :]
    if len(rows) != 1:
        raise ValueError("released OVI-MAP evaluator must contain exactly one data row")
    values = rows[0].split("\t")
    if len(values) != len(expected_header):
        raise ValueError("released OVI-MAP evaluator data row width does not match header")
    metrics = {}
    for name, raw_value in zip(metric_names, values, strict=True):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("released OVI-MAP evaluator metrics must be finite")
        metrics[name] = value
    return {
        "schema_version": 1,
        "scene_id": normalized_scene,
        "source_protocol": "released_ovimap",
        "source_header": list(expected_header),
        "metrics": metrics,
    }


def parse_official_instance_output(text: str, *, scene_id: str) -> dict:
    return _parse_one_row(
        text,
        expected_header=_INSTANCE_HEADER,
        metric_names=_INSTANCE_NAMES,
        scene_id=scene_id,
    )


def parse_official_semantic_output(text: str, *, scene_id: str) -> dict:
    return _parse_one_row(
        text,
        expected_header=_SEMANTIC_HEADER,
        metric_names=_SEMANTIC_NAMES,
        scene_id=scene_id,
    )
