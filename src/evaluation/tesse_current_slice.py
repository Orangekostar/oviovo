"""Time-domain contracts for the TESSE current-diagonal diagnostic."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

from src.evaluation.khronos_attribution import (
    AttributionRow,
    read_exact_associations,
)

METRIC_TYPES = ("object", "dynamic_object", "change")
METRIC_NAMES = {
    "object": "object_f1",
    "dynamic_object": "dynamic_f1",
    "change": "change_f1",
}


class CurrentSliceError(ValueError):
    """Raised when an input cannot prove current-diagonal membership."""


@dataclass(frozen=True, slots=True)
class MetricCountRow:
    metric_type: str
    map_name: int
    query_time_ns: int
    tp: int
    fp: int
    fn: int
    source_row_ordinal: int


@dataclass(frozen=True, slots=True)
class CurrentSliceRow:
    metric_type: str
    map_name: int
    robot_time_ns: int
    query_time_ns: int
    tp: int
    fp: int
    fn: int
    f1: float | None


@dataclass(frozen=True, slots=True)
class CurrentDiagonalResult:
    rows: tuple[CurrentSliceRow, ...]
    metrics: Mapping[str, float | None]
    raw_row_counts: Mapping[str, int]
    unique_state_counts: Mapping[str, int]
    source_bindings: Mapping[str, Mapping[str, object]] | None = None
    attribution: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "PASS",
            "protocol_id": "TESSE_CURRENT_DIAGONAL",
            "metrics": dict(self.metrics),
            "raw_row_counts": dict(self.raw_row_counts),
            "unique_state_counts": dict(self.unique_state_counts),
            "rows": [
                {
                    "metric_type": row.metric_type,
                    "map_name": row.map_name,
                    "robot_time_ns": row.robot_time_ns,
                    "belief_time_ns": row.query_time_ns,
                    "tp": row.tp,
                    "fp": row.fp,
                    "fn": row.fn,
                    "f1": row.f1,
                }
                for row in self.rows
            ],
            "source_bindings": dict(self.source_bindings or {}),
            "attribution": dict(self.attribution or {}),
        }


def _f1(row: MetricCountRow) -> float | None:
    if row.tp == 0:
        return None
    denominator = 2 * row.tp + row.fp + row.fn
    return None if denominator == 0 else 2 * row.tp / denominator


def evaluate_current_diagonal(
    *, rows: Sequence[MetricCountRow], robot_times: Sequence[int]
) -> CurrentDiagonalResult:
    """Select only states whose belief/query time equals the robot time."""

    if not robot_times or any(
        type(value) is not int or value < 0 for value in robot_times
    ):
        raise CurrentSliceError("robot times must be non-negative integers")
    if any(right <= left for left, right in pairwise(robot_times)):
        raise CurrentSliceError("robot times must be strictly increasing")
    raw_counts = {metric: 0 for metric in METRIC_TYPES}
    unique: dict[tuple[str, int, int], MetricCountRow] = {}
    for row in rows:
        if row.metric_type not in METRIC_TYPES:
            raise CurrentSliceError(f"unknown metric type: {row.metric_type}")
        if not 0 <= row.map_name < len(robot_times):
            raise CurrentSliceError("map name is outside the robot-time domain")
        if row.query_time_ns > robot_times[row.map_name]:
            raise CurrentSliceError("future belief time is not causal")
        if min(row.tp, row.fp, row.fn, row.source_row_ordinal) < 0:
            raise CurrentSliceError("metric counts and row ordinal must be non-negative")
        raw_counts[row.metric_type] += 1
        key = (row.metric_type, row.map_name, row.query_time_ns)
        previous = unique.get(key)
        if previous is not None and (
            previous.tp,
            previous.fp,
            previous.fn,
        ) != (row.tp, row.fp, row.fn):
            raise CurrentSliceError("conflicting duplicate metric row")
        unique[key] = row
    selected: list[CurrentSliceRow] = []
    for metric in METRIC_TYPES:
        for map_name, robot_time in enumerate(robot_times):
            source = unique.get((metric, map_name, robot_time))
            if source is None:
                raise CurrentSliceError(
                    f"missing current-diagonal state for {metric} map {map_name}"
                )
            selected.append(
                CurrentSliceRow(
                    metric,
                    map_name,
                    robot_time,
                    source.query_time_ns,
                    source.tp,
                    source.fp,
                    source.fn,
                    _f1(source),
                )
            )
    metrics: dict[str, float | None] = {}
    for metric in METRIC_TYPES:
        values = [
            row.f1 for row in selected if row.metric_type == metric and row.f1 is not None
        ]
        metrics[METRIC_NAMES[metric]] = sum(values) / len(values) if values else None
    return CurrentDiagonalResult(
        tuple(selected),
        metrics,
        raw_counts,
        {
            metric: sum(key[0] == metric for key in unique)
            for metric in METRIC_TYPES
        },
    )


def _event_signature(row: AttributionRow) -> tuple[object, ...]:
    return (
        row.metric_type,
        row.trajectory_timestamp_ns,
        row.pred_node_id,
        row.gt_node_id,
        row.distance_m,
        row.status,
    )


def validate_attribution_time_domain(
    *,
    attribution_rows: Sequence[AttributionRow],
    metric_rows: Sequence[MetricCountRow],
    robot_times: Sequence[int],
) -> dict[str, int]:
    """Validate attribution causality and duplicate-state event identity."""

    sources = {
        (row.metric_type, row.source_row_ordinal): row for row in metric_rows
    }
    grouped: dict[tuple[str, int, int, int], list[tuple[object, ...]]] = defaultdict(list)
    metric_source = {
        "object": "object",
        "dynamic_object": "dynamic_object",
        "change_appeared": "change",
        "change_disappeared": "change",
    }
    for event in attribution_rows:
        source_type = metric_source.get(event.metric_type)
        source = sources.get((str(source_type), event.metric_row_ordinal))
        if source is None:
            raise CurrentSliceError("attribution row has no official metric row")
        if event.map_name != str(source.map_name) or event.query_time_ns != source.query_time_ns:
            raise CurrentSliceError("attribution row identity disagrees with metric row")
        robot_time = robot_times[source.map_name]
        if event.query_time_ns > robot_time:
            raise CurrentSliceError("attribution has a future belief time")
        if event.trajectory_timestamp_ns > event.query_time_ns:
            raise CurrentSliceError("attribution has a future trajectory timestamp")
        grouped[
            (
                event.metric_type,
                source.map_name,
                source.query_time_ns,
                source.source_row_ordinal,
            )
        ].append(_event_signature(event))
    by_state: dict[tuple[str, int, int], list[tuple[tuple[object, ...], ...]]] = defaultdict(list)
    for source in metric_rows:
        event_types = {
            "object": ("object",),
            "dynamic_object": ("dynamic_object",),
            "change": ("change_appeared", "change_disappeared"),
        }[source.metric_type]
        for event_type in event_types:
            signatures = tuple(
                sorted(
                    grouped.get(
                        (
                            event_type,
                            source.map_name,
                            source.query_time_ns,
                            source.source_row_ordinal,
                        ),
                        (),
                    ),
                    key=repr,
                )
            )
            by_state[(event_type, source.map_name, source.query_time_ns)].append(
                signatures
            )
    for duplicate_groups in by_state.values():
        if any(group != duplicate_groups[0] for group in duplicate_groups[1:]):
            raise CurrentSliceError("duplicate attribution differs across metric rows")
    return {
        "validated_event_count": len(attribution_rows),
        "validated_state_count": len(by_state),
    }


def _integer(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CurrentSliceError(f"invalid integer field {field}") from exc
    if parsed < 0 or str(parsed) != value.strip():
        raise CurrentSliceError(f"invalid integer field {field}")
    return parsed


def _read_csv(path: Path, *, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise CurrentSliceError(f"invalid CSV header: {path.name}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise CurrentSliceError(f"malformed or empty CSV: {path.name}")
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_current_diagonal(
    results_dir: str | Path,
    *,
    map_timestamps: str | Path | None = None,
    object_sidecar: str | Path | None = None,
    dynamic_sidecar: str | Path | None = None,
) -> CurrentDiagonalResult:
    """Load official CSVs, bind their time domain, and evaluate the diagonal."""

    root = Path(results_dir).resolve()
    timestamp_path = (
        Path(map_timestamps).resolve()
        if map_timestamps is not None
        else root / "map_timestamps.txt"
    )
    paths = {
        "static_objects.csv": root / "static_objects.csv",
        "dynamic_objects.csv": root / "dynamic_objects.csv",
        "map_timestamps.txt": timestamp_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise CurrentSliceError("current-diagonal inputs are missing")
    try:
        robot_times = tuple(
            _integer(line, field="map timestamp")
            for line in timestamp_path.read_text(encoding="utf-8").splitlines()
        )
    except UnicodeDecodeError as exc:
        raise CurrentSliceError("map timestamps are not UTF-8") from exc
    static_rows = _read_csv(
        paths["static_objects.csv"],
        required={
            "Name",
            "Query",
            "NumObjDetected",
            "NumObjHallucinated",
            "NumObjMissed",
            "AppearedTP",
            "DisappearedTP",
            "AppearedFP",
            "DisappearedFP",
            "AppearedFN",
            "DisappearedFN",
        },
    )
    dynamic_rows = _read_csv(
        paths["dynamic_objects.csv"],
        required={
            "Name",
            "Query",
            "NumObjDetected",
            "NumObjHallucinated",
            "NumObjMissed",
        },
    )
    metric_rows: list[MetricCountRow] = []
    for ordinal, row in enumerate(static_rows):
        common = (
            _integer(row["Name"], field="Name"),
            _integer(row["Query"], field="Query"),
        )
        metric_rows.append(
            MetricCountRow(
                "object",
                *common,
                _integer(row["NumObjDetected"], field="NumObjDetected"),
                _integer(row["NumObjHallucinated"], field="NumObjHallucinated"),
                _integer(row["NumObjMissed"], field="NumObjMissed"),
                ordinal,
            )
        )
        metric_rows.append(
            MetricCountRow(
                "change",
                *common,
                _integer(row["AppearedTP"], field="AppearedTP")
                + _integer(row["DisappearedTP"], field="DisappearedTP"),
                _integer(row["AppearedFP"], field="AppearedFP")
                + _integer(row["DisappearedFP"], field="DisappearedFP"),
                _integer(row["AppearedFN"], field="AppearedFN")
                + _integer(row["DisappearedFN"], field="DisappearedFN"),
                ordinal,
            )
        )
    for ordinal, row in enumerate(dynamic_rows):
        metric_rows.append(
            MetricCountRow(
                "dynamic_object",
                _integer(row["Name"], field="Name"),
                _integer(row["Query"], field="Query"),
                _integer(row["NumObjDetected"], field="NumObjDetected"),
                _integer(row["NumObjHallucinated"], field="NumObjHallucinated"),
                _integer(row["NumObjMissed"], field="NumObjMissed"),
                ordinal,
            )
        )
    result = evaluate_current_diagonal(rows=metric_rows, robot_times=robot_times)
    attribution: dict[str, object] = {}
    if (object_sidecar is None) != (dynamic_sidecar is None):
        raise CurrentSliceError("both exact attribution sidecars are required")
    if object_sidecar is not None and dynamic_sidecar is not None:
        object_sidecar_path = Path(object_sidecar).resolve()
        dynamic_sidecar_path = Path(dynamic_sidecar).resolve()
        object_events = read_exact_associations(object_sidecar_path)
        dynamic_events = read_exact_associations(dynamic_sidecar_path)
        attribution = {
            "object": validate_attribution_time_domain(
                attribution_rows=object_events,
                metric_rows=[row for row in metric_rows if row.metric_type != "dynamic_object"],
                robot_times=robot_times,
            ),
            "dynamic": validate_attribution_time_domain(
                attribution_rows=dynamic_events,
                metric_rows=[row for row in metric_rows if row.metric_type == "dynamic_object"],
                robot_times=robot_times,
            ),
        }
        paths["object_associations.jsonl"] = object_sidecar_path
        paths["dynamic_associations.jsonl"] = dynamic_sidecar_path
    bindings = {
        name: {"sha256": _sha256(path), "byte_count": path.stat().st_size}
        for name, path in paths.items()
    }
    return replace(result, source_bindings=bindings, attribution=attribution)
