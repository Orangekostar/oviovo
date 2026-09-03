"""Strict contracts for evaluator-native Khronos attribution sidecars."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.json_contracts import loads_strict

OFFICIAL_OUTPUT_NAMES = (
    "background_mesh.csv",
    "dynamic_objects.csv",
    "static_objects.csv",
)
METRIC_TYPES = (
    "object",
    "dynamic_object",
    "change_appeared",
    "change_disappeared",
)
STATUSES = ("TP", "FP", "FN")
_FIELDS = {
    "schema_version",
    "metric_type",
    "map_name",
    "metric_row_ordinal",
    "query_time_ns",
    "trajectory_timestamp_ns",
    "pred_node_id",
    "gt_node_id",
    "distance_m",
    "status",
}
_MAX_LINE_BYTES = 1024 * 1024


class AttributionContractError(ValueError):
    """Raised when an exact-attribution artifact is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class AttributionRow:
    metric_type: str
    map_name: str
    metric_row_ordinal: int
    query_time_ns: int
    trajectory_timestamp_ns: int
    pred_node_id: str | None
    gt_node_id: str | None
    distance_m: float | None
    status: str

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.metric_type,
            self.metric_row_ordinal,
            self.map_name,
            self.query_time_ns,
            self.trajectory_timestamp_ns,
            self.pred_node_id,
            self.gt_node_id,
            self.status,
        )


def _direct_file(path: str | Path, *, label: str) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    try:
        current = Path(value.anchor)
        metadata = current.lstat()
        for component in value.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise AttributionContractError(f"{label} must be a direct regular file")
    except FileNotFoundError as exc:
        raise AttributionContractError(
            f"{label} must be a direct regular file"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise AttributionContractError(f"{label} must be a direct regular file")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AttributionContractError(f"{label} must be a non-negative integer")
    return value


def _node_id(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AttributionContractError(f"{label} must be null or a non-empty string")
    return value


def _parse_row(value: object, *, line_number: int) -> AttributionRow:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise AttributionContractError(
            f"attribution line {line_number} has invalid fields"
        )
    if value["schema_version"] != 1:
        raise AttributionContractError(
            f"attribution line {line_number} has an unsupported schema"
        )
    metric_type = value["metric_type"]
    if metric_type not in METRIC_TYPES:
        raise AttributionContractError(
            f"attribution line {line_number} has an invalid metric type"
        )
    map_name = value["map_name"]
    if not isinstance(map_name, str) or not map_name:
        raise AttributionContractError(
            f"attribution line {line_number} has an invalid map name"
        )
    ordinal = _integer(
        value["metric_row_ordinal"], label="metric_row_ordinal"
    )
    query_time = _integer(value["query_time_ns"], label="query_time_ns")
    trajectory_time = _integer(
        value["trajectory_timestamp_ns"], label="trajectory_timestamp_ns"
    )
    if trajectory_time > query_time:
        raise AttributionContractError(
            f"attribution line {line_number} has a future trajectory timestamp"
        )
    pred = _node_id(value["pred_node_id"], label="pred_node_id")
    gt = _node_id(value["gt_node_id"], label="gt_node_id")
    status_value = value["status"]
    if status_value not in STATUSES:
        raise AttributionContractError(
            f"attribution line {line_number} has an invalid status"
        )
    status = str(status_value)
    if status == "TP" and (pred is None or gt is None):
        raise AttributionContractError("TP attribution requires both node identities")
    if status == "FP" and pred is None:
        raise AttributionContractError("FP attribution requires a predicted node identity")
    if status == "FN" and gt is None:
        raise AttributionContractError("FN attribution requires a GT node identity")
    distance_value = value["distance_m"]
    distance: float | None
    if distance_value is None:
        distance = None
    elif isinstance(distance_value, bool) or not isinstance(
        distance_value, (int, float)
    ):
        raise AttributionContractError("distance_m must be null or finite non-negative")
    else:
        distance = float(distance_value)
        if not math.isfinite(distance) or distance < 0.0:
            raise AttributionContractError(
                "distance_m must be null or finite non-negative"
            )
    return AttributionRow(
        metric_type=str(metric_type),
        map_name=map_name,
        metric_row_ordinal=ordinal,
        query_time_ns=query_time,
        trajectory_timestamp_ns=trajectory_time,
        pred_node_id=pred,
        gt_node_id=gt,
        distance_m=distance,
        status=status,
    )


def read_exact_associations(path: str | Path) -> tuple[AttributionRow, ...]:
    """Read one exact attribution JSONL file and reject ambiguous identities."""

    source = _direct_file(path, label="attribution sidecar")
    rows: list[AttributionRow] = []
    identities: set[tuple[object, ...]] = set()
    with source.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if len(raw_line) > _MAX_LINE_BYTES:
                raise AttributionContractError(
                    f"attribution line {line_number} exceeds the size limit"
                )
            if not raw_line.strip():
                raise AttributionContractError(
                    f"attribution line {line_number} is empty"
                )
            try:
                payload = loads_strict(
                    raw_line.decode("utf-8"),
                    label=f"attribution line {line_number}",
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise AttributionContractError(
                    f"attribution line {line_number} is invalid JSON"
                ) from exc
            row = _parse_row(payload, line_number=line_number)
            if row.identity in identities:
                raise AttributionContractError("duplicate event identity in sidecar")
            identities.add(row.identity)
            rows.append(row)
    return tuple(rows)


def summarize_status(
    rows: Iterable[AttributionRow], *, metric_type: str | None = None
) -> dict[str, int]:
    """Count metric-contributing events in stable TP/FP/FN order."""

    if metric_type is not None and metric_type not in METRIC_TYPES:
        raise AttributionContractError("unknown metric type")
    counts = Counter(
        row.status
        for row in rows
        if metric_type is None or row.metric_type == metric_type
    )
    return {status: counts[status] for status in STATUSES}


def _csv_rows(path: Path, *, required: set[str]) -> list[dict[str, str]]:
    source = _direct_file(path, label=path.name)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise AttributionContractError(f"{path.name} has an invalid schema")
        rows = list(reader)
    if not rows:
        raise AttributionContractError(f"{path.name} has no metric rows")
    return rows


def _csv_count(row: Mapping[str, str], name: str, *, filename: str) -> int:
    raw = row.get(name)
    try:
        value = int(raw) if raw is not None else -1
    except ValueError as exc:
        raise AttributionContractError(
            f"{filename} column {name} is not an integer"
        ) from exc
    if value < 0 or str(value) != raw.strip():
        raise AttributionContractError(
            f"{filename} column {name} is not a canonical non-negative integer"
        )
    return value


def _expected_dynamic(row: Mapping[str, str]) -> dict[str, dict[str, int]]:
    filename = "dynamic_objects.csv"
    return {
        "dynamic_object": {
            "TP": _csv_count(row, "NumObjDetected", filename=filename),
            "FP": _csv_count(row, "NumObjHallucinated", filename=filename),
            "FN": _csv_count(row, "NumObjMissed", filename=filename),
        }
    }


def _expected_static(row: Mapping[str, str]) -> dict[str, dict[str, int]]:
    filename = "static_objects.csv"
    count = lambda name: _csv_count(row, name, filename=filename)
    return {
        "object": {
            "TP": count("NumObjDetected"),
            "FP": count("NumObjHallucinated"),
            "FN": count("NumObjMissed"),
        },
        "change_appeared": {
            "TP": count("AppearedTP"),
            "FP": count("AppearedFP") + count("AppearedHallucinatedP"),
            "FN": count("AppearedFN") + count("AppearedMissedP"),
        },
        "change_disappeared": {
            "TP": count("DisappearedTP"),
            "FP": count("DisappearedFP") + count("DisappearedHallucinatedP"),
            "FN": count("DisappearedFN") + count("DisappearedMissedP"),
        },
    }


def _validate_rows_against_csv(
    *,
    sidecar_rows: Sequence[AttributionRow],
    csv_rows: Sequence[Mapping[str, str]],
    expected_metrics: tuple[str, ...],
    expected_for_row: Any,
    filename: str,
) -> dict[str, dict[str, int]]:
    grouped: dict[tuple[int, str], list[AttributionRow]] = defaultdict(list)
    for row in sidecar_rows:
        if row.metric_type not in expected_metrics:
            raise AttributionContractError(
                f"{filename} sidecar contains metric type {row.metric_type}"
            )
        if row.metric_row_ordinal >= len(csv_rows):
            raise AttributionContractError(f"{filename} metric mass mismatch")
        official = csv_rows[row.metric_row_ordinal]
        try:
            official_query = int(official["Query"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AttributionContractError(
                f"{filename} has an invalid row identity"
            ) from exc
        if row.map_name != official.get("Name") or row.query_time_ns != official_query:
            raise AttributionContractError(f"{filename} sidecar row identity mismatch")
        grouped[(row.metric_row_ordinal, row.metric_type)].append(row)

    totals = {metric: Counter() for metric in expected_metrics}
    for ordinal, official in enumerate(csv_rows):
        expected = expected_for_row(official)
        for metric_type in expected_metrics:
            actual = summarize_status(
                grouped.get((ordinal, metric_type), ()), metric_type=metric_type
            )
            if actual != expected[metric_type]:
                raise AttributionContractError(
                    f"{filename} metric mass mismatch at row {ordinal} "
                    f"for {metric_type}: expected {expected[metric_type]}, got {actual}"
                )
            totals[metric_type].update(actual)
    return {
        metric: {status: totals[metric][status] for status in STATUSES}
        for metric in expected_metrics
    }


def validate_attribution_mass(
    *,
    object_rows: Sequence[AttributionRow],
    dynamic_rows: Sequence[AttributionRow],
    official_results: str | Path,
) -> dict[str, dict[str, int]]:
    """Prove sidecar TP/FP/FN counts equal every official CSV metric row."""

    root = Path(os.path.abspath(os.fspath(official_results)))
    if root.is_symlink() or not root.is_dir():
        raise AttributionContractError("official results must be a direct directory")
    dynamic_csv = _csv_rows(
        root / "dynamic_objects.csv",
        required={
            "Name",
            "Query",
            "NumObjDetected",
            "NumObjMissed",
            "NumObjHallucinated",
        },
    )
    static_csv = _csv_rows(
        root / "static_objects.csv",
        required={
            "Name",
            "Query",
            "AppearedTP",
            "DisappearedTP",
            "AppearedFP",
            "DisappearedFP",
            "AppearedHallucinatedP",
            "DisappearedHallucinatedP",
            "AppearedFN",
            "DisappearedFN",
            "AppearedMissedP",
            "DisappearedMissedP",
            "NumObjDetected",
            "NumObjMissed",
            "NumObjHallucinated",
        },
    )
    dynamic_summary = _validate_rows_against_csv(
        sidecar_rows=dynamic_rows,
        csv_rows=dynamic_csv,
        expected_metrics=("dynamic_object",),
        expected_for_row=_expected_dynamic,
        filename="dynamic_objects.csv",
    )
    static_summary = _validate_rows_against_csv(
        sidecar_rows=object_rows,
        csv_rows=static_csv,
        expected_metrics=("object", "change_appeared", "change_disappeared"),
        expected_for_row=_expected_static,
        filename="static_objects.csv",
    )
    return {**dynamic_summary, **static_summary}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, object]:
    source = _direct_file(path, label=path.name)
    return {
        "sha256": _sha256(source),
        "byte_count": source.stat().st_size,
    }


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AttributionContractError(
            f"Khronos source identity check failed: {detail}"
        ) from exc


def assert_source_patch_identity(
    *,
    source_checkout: str | Path,
    source_commit: str,
    patch_file: str | Path,
    patch_sha256: str,
) -> dict[str, object]:
    """Bind a clean source tree and prove the reviewed patch applies to it."""

    checkout = Path(os.path.abspath(os.fspath(source_checkout)))
    if checkout.is_symlink() or not checkout.is_dir():
        raise AttributionContractError("Khronos source checkout must be a directory")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise AttributionContractError("Khronos source commit must be a full Git SHA")
    if re.fullmatch(r"[0-9a-f]{64}", patch_sha256) is None:
        raise AttributionContractError("patch SHA-256 must be lowercase hexadecimal")
    patch = _direct_file(patch_file, label="attribution patch")
    actual_patch_sha256 = _sha256(patch)
    if actual_patch_sha256 != patch_sha256:
        raise AttributionContractError("attribution patch SHA-256 mismatch")
    head = _git(checkout, "rev-parse", "HEAD").decode().strip()
    if head != source_commit:
        raise AttributionContractError("Khronos source checkout commit mismatch")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise AttributionContractError("Khronos source checkout is not clean")
    _git(checkout, "apply", "--check", str(patch))
    return {
        "repository": "MIT-SPARK/Khronos",
        "commit": source_commit,
        "patch_sha256": actual_patch_sha256,
        "patch_byte_count": patch.stat().st_size,
    }


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def assert_official_outputs_identical(
    unpatched_results: str | Path,
    patched_results: str | Path,
) -> dict[str, dict[str, object]]:
    """Require patched and unpatched official CSV outputs to be byte-identical."""

    left_root = Path(os.path.abspath(os.fspath(unpatched_results)))
    right_root = Path(os.path.abspath(os.fspath(patched_results)))
    receipt: dict[str, dict[str, object]] = {}
    for name in OFFICIAL_OUTPUT_NAMES:
        left = _direct_file(left_root / name, label=f"unpatched {name}")
        right = _direct_file(right_root / name, label=f"patched {name}")
        if not _files_equal(left, right):
            raise AttributionContractError(
                f"official output {name} is not byte-identical"
            )
        left_hash = _sha256(left)
        right_hash = _sha256(right)
        receipt[name] = {
            "byte_count": left.stat().st_size,
            "unpatched_sha256": left_hash,
            "patched_sha256": right_hash,
        }
    return receipt


def audit_exact_attribution(
    *,
    unpatched_results: str | Path,
    patched_results: str | Path,
    object_sidecar: str | Path,
    dynamic_sidecar: str | Path,
    input_map: str | Path,
    expected_input_sha256: str,
    expected_input_byte_count: int,
    source_checkout: str | Path,
    source_commit: str,
    patch_file: str | Path,
    patch_sha256: str,
) -> dict[str, object]:
    """Produce a source-bound non-interference and attribution receipt."""

    source = assert_source_patch_identity(
        source_checkout=source_checkout,
        source_commit=source_commit,
        patch_file=patch_file,
        patch_sha256=patch_sha256,
    )
    map_binding = _file_binding(Path(input_map))
    if (
        map_binding["sha256"] != expected_input_sha256
        or map_binding["byte_count"] != expected_input_byte_count
    ):
        raise AttributionContractError("input map binding mismatch")
    object_path = Path(object_sidecar)
    dynamic_path = Path(dynamic_sidecar)
    object_rows = read_exact_associations(object_path)
    dynamic_rows = read_exact_associations(dynamic_path)
    metric_mass = validate_attribution_mass(
        object_rows=object_rows,
        dynamic_rows=dynamic_rows,
        official_results=patched_results,
    )
    official_outputs = assert_official_outputs_identical(
        unpatched_results,
        patched_results,
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": "KHRONOS_POST_RELEASE_EXACT_ATTRIBUTION_V1",
        "source": source,
        "input_map": map_binding,
        "sidecars": {
            "object": {
                **_file_binding(object_path),
                "event_count": len(object_rows),
            },
            "dynamic": {
                **_file_binding(dynamic_path),
                "event_count": len(dynamic_rows),
            },
        },
        "metric_mass": metric_mass,
        "official_outputs": official_outputs,
    }
