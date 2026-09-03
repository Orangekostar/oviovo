"""Source-bound parity checks for Khronos plotting aggregation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.baselines.tesse_cd import summarize_khronos_official_metrics_partial


class KhronosParityError(ValueError):
    """Raised when a parity input cannot support an auditable comparison."""


@dataclass(frozen=True, slots=True)
class KhronosParityResult:
    row_grid_equal: bool
    duplicate_policy_equal: bool
    slice_values_equal: bool
    metric_values_equal: bool
    local_metrics: Mapping[str, float | None]
    upstream_metrics: Mapping[str, float | None]
    max_abs_delta: float
    raw_row_counts: Mapping[str, int]
    unique_state_counts: Mapping[str, int]
    slice_comparisons: Mapping[str, Mapping[str, Mapping[str, object]]]
    input_sha256: Mapping[str, str]
    upstream_utils_sha256: str

    @property
    def status(self) -> str:
        return (
            "PARITY"
            if all(
                (
                    self.row_grid_equal,
                    self.duplicate_policy_equal,
                    self.slice_values_equal,
                    self.metric_values_equal,
                )
            )
            else "MISMATCH"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "row_grid_equal": self.row_grid_equal,
            "duplicate_policy_equal": self.duplicate_policy_equal,
            "slice_values_equal": self.slice_values_equal,
            "metric_values_equal": self.metric_values_equal,
            "local_metrics": dict(self.local_metrics),
            "upstream_metrics": dict(self.upstream_metrics),
            "max_abs_delta": self.max_abs_delta,
            "raw_row_counts": dict(self.raw_row_counts),
            "unique_state_counts": dict(self.unique_state_counts),
            "slice_comparisons": {
                mode: {metric: dict(detail) for metric, detail in metrics.items()}
                for mode, metrics in self.slice_comparisons.items()
            },
            "input_sha256": dict(self.input_sha256),
            "upstream_utils_sha256": self.upstream_utils_sha256,
        }


@dataclass(frozen=True, slots=True)
class _CsvTable:
    filename: str
    rows: tuple[Mapping[str, str], ...]
    unique: Mapping[tuple[int, ...], Mapping[str, str]]


_STATIC_REQUIRED = (
    "Name",
    "Query",
    "NumGtLoaded",
    "NumDsgLoaded",
    "NumGtNotLoaded",
    "NumDsgNotLoaded",
    "AppearedTP",
    "DisappearedTP",
    "AppearedFP",
    "DisappearedFP",
    "AppearedHallucinatedP",
    "DisappearedHallucinatedP",
    "AppearedFN",
    "DisappearedFN",
    "AppearedTN",
    "DisappearedTN",
    "AppearedMissedP",
    "DisappearedMissedP",
    "NumObjDetected",
    "NumObjMissed",
    "NumObjHallucinated",
)
_DYNAMIC_REQUIRED = (
    "Name",
    "Query",
    "NumGtLoaded",
    "NumDsgLoaded",
    "NumGtNotLoaded",
    "NumDsgNotLoaded",
    "NumObjDetected",
    "NumObjMissed",
    "NumObjHallucinated",
)
_BACKGROUND_REQUIRED = (
    "Name",
    "Accuracy@0.2",
    "Completeness@0.2",
    "NumGTFailed",
    "NumDSGFailed",
)
_INPUT_NAMES = (
    "static_objects.csv",
    "dynamic_objects.csv",
    "background_mesh.csv",
)
_METRIC_NAMES = (
    "object_f1",
    "dynamic_f1",
    "change_f1",
    "background_f1_at_0_2",
)
_UPSTREAM_METRICS = {
    "object_f1": "ObjectF1",
    "dynamic_f1": "DynamicF1",
    "change_f1": "ChangeF1",
    "background_f1_at_0_2": "F1@0.2",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_nonnegative_int(value: str, *, field: str, filename: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise KhronosParityError(
            f"invalid Khronos integer in {filename}: {field}={value!r}"
        ) from error
    if parsed < 0:
        raise KhronosParityError(
            f"negative Khronos integer in {filename}: {field}={parsed}"
        )
    return parsed


def _parse_unit_float(value: str, *, field: str, filename: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise KhronosParityError(
            f"invalid Khronos float in {filename}: {field}={value!r}"
        ) from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise KhronosParityError(
            f"out-of-domain Khronos float in {filename}: {field}={value!r}"
        )
    return parsed


def _read_table(
    data: bytes,
    *,
    filename: str,
    required: Sequence[str],
    key_fields: Sequence[str],
) -> _CsvTable:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KhronosParityError(f"Khronos CSV is not UTF-8: {filename}") from error
    reader = csv.DictReader(text.splitlines())
    fieldnames = reader.fieldnames or []
    if len(fieldnames) != len(set(fieldnames)):
        raise KhronosParityError(f"duplicate Khronos header in {filename}")
    missing = set(required) - set(fieldnames)
    if missing:
        raise KhronosParityError(
            f"Khronos result file {filename} lacks columns: {sorted(missing)}"
        )

    rows: list[Mapping[str, str]] = []
    unique: dict[tuple[int, ...], Mapping[str, str]] = {}
    for row_index, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise KhronosParityError(
                f"malformed Khronos row {row_index} in {filename}"
            )
        normalized = {str(key): str(value) for key, value in row.items()}
        key = tuple(
            _parse_nonnegative_int(
                normalized[field], field=field, filename=filename
            )
            for field in key_fields
        )
        if key in unique and unique[key] != normalized:
            raise KhronosParityError(
                f"conflicting duplicate Khronos row {key} in {filename}"
            )
        unique[key] = normalized
        rows.append(normalized)
    if not rows:
        raise KhronosParityError(f"Khronos result file is empty: {filename}")
    return _CsvTable(filename, tuple(rows), unique)


def _validate_tables(tables: Mapping[str, _CsvTable]) -> None:
    static = tables["static_objects.csv"]
    dynamic = tables["dynamic_objects.csv"]
    background = tables["background_mesh.csv"]
    for table, count_fields in (
        (static, _STATIC_REQUIRED[2:]),
        (dynamic, _DYNAMIC_REQUIRED[2:]),
        (background, ("NumGTFailed", "NumDSGFailed")),
    ):
        for row in table.rows:
            for field in count_fields:
                _parse_nonnegative_int(
                    row[field], field=field, filename=table.filename
                )
    for row in background.rows:
        for field in ("Accuracy@0.2", "Completeness@0.2"):
            _parse_unit_float(row[field], field=field, filename=background.filename)

    if set(static.unique) != set(dynamic.unique):
        raise KhronosParityError("static and dynamic Khronos state keys disagree")
    map_names = sorted({key[0] for key in static.unique})
    if map_names != list(range(len(map_names))):
        raise KhronosParityError("Khronos map names must be contiguous from zero")
    if {key[0] for key in background.unique} != set(map_names):
        raise KhronosParityError("background and object Khronos map names disagree")


def _read_timestamps(data: bytes, *, filename: str) -> list[int]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise KhronosParityError(f"Khronos timestamps are not UTF-8: {filename}") from error
    if not lines or any(not line.strip() for line in lines):
        raise KhronosParityError(f"empty Khronos timestamp in {filename}")
    values = [
        _parse_nonnegative_int(line.strip(), field="timestamp", filename=filename)
        for line in lines
    ]
    if any(right <= left for left, right in pairwise(values)):
        raise KhronosParityError("Khronos map timestamps must be strictly increasing")
    return values


def _validate_time_grid(static: _CsvTable, timestamps: Sequence[int]) -> None:
    map_names = sorted({key[0] for key in static.unique})
    if len(timestamps) != len(map_names):
        raise KhronosParityError(
            "Khronos map timestamp count does not match the map state count"
        )
    for map_name, query in static.unique:
        if query not in timestamps[: map_name + 1]:
            raise KhronosParityError(
                f"Khronos query {query} is not causal for map {map_name}"
            )


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = math.nan if tp + fp == 0 else tp / (tp + fp)
    recall = math.nan if tp + fn == 0 else tp / (tp + fn)
    return (
        math.nan
        if not math.isfinite(precision + recall) or precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )


def _count(row: Mapping[str, str], field: str, filename: str) -> int:
    return _parse_nonnegative_int(row[field], field=field, filename=filename)


def _local_grids(tables: Mapping[str, _CsvTable]) -> dict[str, dict[int, dict[int, float]]]:
    static = tables["static_objects.csv"]
    dynamic = tables["dynamic_objects.csv"]
    background = tables["background_mesh.csv"]
    grids: dict[str, dict[int, dict[int, float]]] = {
        name: {} for name in _METRIC_NAMES
    }
    for (map_name, query), row in static.unique.items():
        grids["object_f1"].setdefault(map_name, {})[query] = _f1(
            _count(row, "NumObjDetected", static.filename),
            _count(row, "NumObjHallucinated", static.filename),
            _count(row, "NumObjMissed", static.filename),
        )
        grids["change_f1"].setdefault(map_name, {})[query] = _f1(
            _count(row, "AppearedTP", static.filename)
            + _count(row, "DisappearedTP", static.filename),
            _count(row, "AppearedFP", static.filename)
            + _count(row, "DisappearedFP", static.filename),
            _count(row, "AppearedFN", static.filename)
            + _count(row, "DisappearedFN", static.filename),
        )
    for (map_name, query), row in dynamic.unique.items():
        grids["dynamic_f1"].setdefault(map_name, {})[query] = _f1(
            _count(row, "NumObjDetected", dynamic.filename),
            _count(row, "NumObjHallucinated", dynamic.filename),
            _count(row, "NumObjMissed", dynamic.filename),
        )
    for (map_name,), row in background.unique.items():
        accuracy = _parse_unit_float(
            row["Accuracy@0.2"], field="Accuracy@0.2", filename=background.filename
        )
        completeness = _parse_unit_float(
            row["Completeness@0.2"],
            field="Completeness@0.2",
            filename=background.filename,
        )
        denominator = accuracy + completeness
        value = (
            math.nan
            if denominator == 0
            else 2.0 * accuracy * completeness / denominator
        )
        grids["background_f1_at_0_2"][map_name] = {
            query: value for query in range(map_name + 1)
        }
    return grids


def _load_upstream(
    upstream_utils: object, expected_sha256: str | None
) -> tuple[object, str, bytes | None, Path | None]:
    if not isinstance(upstream_utils, (str, Path)):
        if expected_sha256 is not None:
            raise KhronosParityError(
                "expected upstream SHA-256 cannot bind an in-memory test double"
            )
        return upstream_utils, "IN_MEMORY_TEST_DOUBLE", None, None

    source_path = Path(upstream_utils).resolve()
    if not source_path.is_file():
        raise KhronosParityError(f"upstream Khronos utils.py is missing: {source_path}")
    source = source_path.read_bytes()
    digest = _sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise KhronosParityError(
            f"upstream Khronos utils.py SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    with tempfile.TemporaryDirectory(prefix="crove-khronos-source-") as temporary:
        snapshot_path = Path(temporary) / "utils.py"
        snapshot_path.write_bytes(source)
        specification = importlib.util.spec_from_file_location(
            f"_crove_khronos_utils_{digest[:12]}", snapshot_path
        )
        if specification is None or specification.loader is None:
            raise KhronosParityError("failed to create upstream Khronos module spec")
        module = importlib.util.module_from_spec(specification)
        try:
            specification.loader.exec_module(module)
        except Exception as error:
            raise KhronosParityError(
                f"failed to load upstream Khronos utils.py: {error}"
            ) from error
    return module, digest, source, source_path


def _require_upstream_api(upstream: object) -> None:
    required = (
        "parse_table_data_simple",
        "augment_object_metrics",
        "augment_dynamic_metrics",
        "augment_bg_metrics",
        "get_4d_data_slice",
    )
    missing = [name for name in required if not callable(getattr(upstream, name, None))]
    if missing:
        raise KhronosParityError(f"upstream Khronos API is incomplete: {missing}")


def _normalize_grid(value: object, *, metric: str) -> dict[int, dict[int, float]]:
    if not isinstance(value, Mapping):
        raise KhronosParityError(f"upstream {metric} grid is not a mapping")
    result: dict[int, dict[int, float]] = {}
    for raw_map, raw_queries in value.items():
        try:
            map_name = int(raw_map)
        except (TypeError, ValueError) as error:
            raise KhronosParityError(f"invalid upstream {metric} map key") from error
        if not isinstance(raw_queries, Mapping):
            raise KhronosParityError(f"upstream {metric} query grid is not a mapping")
        result[map_name] = {}
        for raw_query, raw_value in raw_queries.items():
            try:
                query = int(raw_query)
                number = float(raw_value)
            except (TypeError, ValueError) as error:
                raise KhronosParityError(f"invalid upstream {metric} value") from error
            if math.isinf(number):
                raise KhronosParityError(f"non-finite upstream {metric} value")
            result[map_name][query] = number
    return result


def _grid_equal(
    left: Mapping[int, Mapping[int, float]],
    right: Mapping[int, Mapping[int, float]],
    *,
    atol: float,
) -> bool:
    if set(left) != set(right):
        return False
    for map_name in left:
        if set(left[map_name]) != set(right[map_name]):
            return False
        for query, value in left[map_name].items():
            if not math.isclose(
                value,
                right[map_name][query],
                rel_tol=0.0,
                abs_tol=atol,
            ) and not (math.isnan(value) and math.isnan(right[map_name][query])):
                return False
    return True


def _four_d_values(grid: Mapping[int, Mapping[int, float]]) -> np.ndarray:
    return np.asarray(
        [
            grid[map_name][query]
            for map_name in sorted(grid)
            for query in sorted(grid[map_name])
        ],
        dtype=np.float64,
    )


def _local_slice(
    grid: Mapping[int, Mapping[int, float]],
    map_names: Sequence[int],
    query_times: Sequence[int],
    mode: str,
) -> np.ndarray:
    if mode == "4D":
        return _four_d_values(grid)
    if mode == "Robot":
        return np.asarray(
            [grid[map_name][query_times[0]] for map_name in map_names],
            dtype=np.float64,
        )
    if mode == "Query":
        map_name = map_names[-1]
        return np.asarray(
            [grid[map_name][query_time] for query_time in query_times],
            dtype=np.float64,
        )
    if mode == "Online":
        return np.asarray(
            [grid[map_name][query_times[index]] for index, map_name in enumerate(map_names)],
            dtype=np.float64,
        )
    raise AssertionError(mode)


def _capture_slice(calculate: Any) -> tuple[str, np.ndarray | None]:
    try:
        values = np.asarray(calculate(), dtype=np.float64)
    except (KeyError, IndexError):
        return "MISSING_QUERY_KEY", None
    if values.ndim != 1:
        raise KhronosParityError("Khronos slice is not one-dimensional")
    if np.isinf(values).any():
        raise KhronosParityError("Khronos slice contains infinity")
    return "AVAILABLE", values


def _slice_detail(
    local_status: str,
    local_values: np.ndarray | None,
    upstream_status: str,
    upstream_values: np.ndarray | None,
    *,
    atol: float,
) -> tuple[dict[str, object], bool, float]:
    if local_status != "AVAILABLE" or upstream_status != "AVAILABLE":
        equal = local_status == upstream_status
        return (
            {
                "status": "BOTH_UNAVAILABLE" if equal else "AVAILABILITY_MISMATCH",
                "local": local_status,
                "upstream": upstream_status,
                "count": 0,
                "max_abs_delta": None,
            },
            equal,
            0.0,
        )
    assert local_values is not None and upstream_values is not None
    if local_values.shape != upstream_values.shape:
        return (
            {
                "status": "SHAPE_MISMATCH",
                "local": "AVAILABLE",
                "upstream": "AVAILABLE",
                "count": int(local_values.size),
                "max_abs_delta": None,
            },
            False,
            0.0,
        )
    equal = bool(
        np.allclose(
            local_values,
            upstream_values,
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
    )
    finite = np.isfinite(local_values) & np.isfinite(upstream_values)
    delta = (
        float(np.max(np.abs(local_values[finite] - upstream_values[finite])))
        if finite.any()
        else 0.0
    )
    return (
        {
            "status": "PARITY" if equal else "VALUE_MISMATCH",
            "local": "AVAILABLE",
            "upstream": "AVAILABLE",
            "count": int(local_values.size),
            "max_abs_delta": delta,
        },
        equal,
        delta,
    )


def audit_metric_parity(
    results_dir: Path,
    *,
    upstream_utils: object,
    map_timestamps: Path | None = None,
    expected_upstream_sha256: str | None = None,
    atol: float = 1e-12,
) -> KhronosParityResult:
    """Compare local aggregation with one exact upstream Khronos utils source."""

    if not math.isfinite(atol) or atol < 0:
        raise KhronosParityError("atol must be finite and nonnegative")
    results_dir = Path(results_dir).resolve()
    if not results_dir.is_dir():
        raise KhronosParityError(f"Khronos results directory is missing: {results_dir}")
    timestamp_path = (
        Path(map_timestamps).resolve()
        if map_timestamps is not None
        else results_dir / "map_timestamps.txt"
    )
    source_paths = {name: results_dir / name for name in _INPUT_NAMES}
    source_paths["map_timestamps.txt"] = timestamp_path
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise KhronosParityError(f"Khronos parity inputs are missing: {missing}")
    snapshots = {name: path.read_bytes() for name, path in source_paths.items()}
    input_hashes = {name: _sha256(data) for name, data in snapshots.items()}

    tables = {
        "static_objects.csv": _read_table(
            snapshots["static_objects.csv"],
            filename="static_objects.csv",
            required=_STATIC_REQUIRED,
            key_fields=("Name", "Query"),
        ),
        "dynamic_objects.csv": _read_table(
            snapshots["dynamic_objects.csv"],
            filename="dynamic_objects.csv",
            required=_DYNAMIC_REQUIRED,
            key_fields=("Name", "Query"),
        ),
        "background_mesh.csv": _read_table(
            snapshots["background_mesh.csv"],
            filename="background_mesh.csv",
            required=_BACKGROUND_REQUIRED,
            key_fields=("Name",),
        ),
    }
    _validate_tables(tables)
    timestamps = _read_timestamps(
        snapshots["map_timestamps.txt"], filename="map_timestamps.txt"
    )
    _validate_time_grid(tables["static_objects.csv"], timestamps)
    upstream, upstream_digest, upstream_snapshot, upstream_path = _load_upstream(
        upstream_utils, expected_upstream_sha256
    )
    _require_upstream_api(upstream)

    with tempfile.TemporaryDirectory(prefix="crove-khronos-parity-") as temporary:
        staging = Path(temporary) / "method" / "results"
        staging.mkdir(parents=True)
        for name, data in snapshots.items():
            (staging / name).write_bytes(data)
        try:
            upstream_tables = upstream.parse_table_data_simple(
                str(staging.parents[1]), ["method"]
            )
            if not isinstance(upstream_tables, Sequence) or len(upstream_tables) != 1:
                raise KhronosParityError("upstream Khronos parser returned no unique method")
            upstream_data = upstream_tables[0]
            upstream.augment_object_metrics(upstream_data)
            upstream.augment_dynamic_metrics(upstream_data)
            upstream.augment_bg_metrics(upstream_data)
            local_summary = summarize_khronos_official_metrics_partial(staging)
        except KhronosParityError:
            raise
        except Exception as error:
            raise KhronosParityError(
                f"upstream Khronos aggregation failed: {error}"
            ) from error

    local_grids = _local_grids(tables)
    upstream_grids = {
        local_name: _normalize_grid(
            upstream_data[upstream_name], metric=upstream_name
        )
        for local_name, upstream_name in _UPSTREAM_METRICS.items()
    }
    row_grid_equal = all(
        _grid_equal(local_grids[name], upstream_grids[name], atol=0.0)
        for name in _METRIC_NAMES
    )
    duplicate_policy_equal = row_grid_equal and all(
        len(table.unique) <= len(table.rows) for table in tables.values()
    )

    map_names = sorted({key[0] for key in tables["static_objects.csv"].unique})
    slice_comparisons: dict[str, dict[str, dict[str, object]]] = {}
    slice_values_equal = True
    observed_deltas: list[float] = []
    for mode in ("4D", "Robot", "Query", "Online"):
        slice_comparisons[mode] = {}
        for name in _METRIC_NAMES:
            local_status, local_values = _capture_slice(
                lambda name=name, mode=mode: _local_slice(
                    local_grids[name], map_names, timestamps, mode
                )
            )
            if mode == "4D":
                upstream_calculation = lambda name=name: _four_d_values(
                    upstream_grids[name]
                )
            else:
                upstream_calculation = lambda name=name, mode=mode: upstream.get_4d_data_slice(
                    upstream_grids[name], map_names, timestamps, mode
                )
            upstream_status, upstream_values = _capture_slice(upstream_calculation)
            detail, equal, delta = _slice_detail(
                local_status,
                local_values,
                upstream_status,
                upstream_values,
                atol=atol,
            )
            slice_comparisons[mode][name] = detail
            slice_values_equal = slice_values_equal and equal
            observed_deltas.append(delta)

    local_values = local_summary["metrics"]
    assert isinstance(local_values, Mapping)
    local_metrics = {
        name: None if local_values[name] is None else float(local_values[name])
        for name in _METRIC_NAMES
    }
    upstream_metrics: dict[str, float | None] = {}
    for name in _METRIC_NAMES:
        values = _four_d_values(upstream_grids[name])
        finite = values[np.isfinite(values)]
        upstream_metrics[name] = float(np.mean(finite)) if finite.size else None
    metric_deltas: dict[str, float] = {}
    metric_values_equal = True
    for name in _METRIC_NAMES:
        local_value = local_metrics[name]
        upstream_value = upstream_metrics[name]
        if local_value is None or upstream_value is None:
            metric_values_equal = metric_values_equal and local_value is upstream_value
            metric_deltas[name] = 0.0
            continue
        delta = abs(local_value - upstream_value)
        metric_deltas[name] = delta
        metric_values_equal = metric_values_equal and delta <= atol
    observed_deltas.extend(metric_deltas.values())

    for name, path in source_paths.items():
        if _sha256(path.read_bytes()) != input_hashes[name]:
            raise KhronosParityError(f"Khronos parity input changed during audit: {name}")
    if (
        upstream_path is not None
        and upstream_snapshot is not None
        and upstream_path.read_bytes() != upstream_snapshot
    ):
        raise KhronosParityError("upstream Khronos utils.py changed during audit")

    return KhronosParityResult(
        row_grid_equal=row_grid_equal,
        duplicate_policy_equal=duplicate_policy_equal,
        slice_values_equal=slice_values_equal,
        metric_values_equal=metric_values_equal,
        local_metrics=local_metrics,
        upstream_metrics=upstream_metrics,
        max_abs_delta=max(observed_deltas, default=0.0),
        raw_row_counts={name: len(tables[name].rows) for name in _INPUT_NAMES},
        unique_state_counts={name: len(tables[name].unique) for name in _INPUT_NAMES},
        slice_comparisons=slice_comparisons,
        input_sha256=input_hashes,
        upstream_utils_sha256=upstream_digest,
    )
