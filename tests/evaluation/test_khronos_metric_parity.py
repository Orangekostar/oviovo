from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.audit_khronos_metric_parity import _bind_upstream
from src.evaluation.khronos_metric_parity import (
    KhronosParityError,
    audit_metric_parity,
)

STATIC_FIELDS = [
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
]
DYNAMIC_FIELDS = [
    "Name",
    "Query",
    "NumGtLoaded",
    "NumDsgLoaded",
    "NumGtNotLoaded",
    "NumDsgNotLoaded",
    "NumObjDetected",
    "NumObjMissed",
    "NumObjHallucinated",
]


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _write_triangular_fixture(tmp_path: Path) -> Path:
    results = tmp_path / "method" / "results"
    timestamps = [100, 200, 300]
    static_rows: list[list[object]] = []
    dynamic_rows: list[list[object]] = []
    for map_name, robot_time in enumerate(timestamps):
        for query_index, query_time in enumerate(timestamps[: map_name + 1]):
            value = map_name + query_index + 1
            static_rows.append(
                [
                    map_name,
                    query_time,
                    4,
                    4,
                    0,
                    0,
                    value,
                    1,
                    query_index,
                    0,
                    0,
                    0,
                    map_name,
                    1,
                    0,
                    0,
                    0,
                    0,
                    value + 2,
                    map_name,
                    query_index,
                ]
            )
            dynamic_rows.append(
                [
                    map_name,
                    query_time,
                    3,
                    3,
                    0,
                    0,
                    value + 1,
                    query_index,
                    map_name,
                ]
            )
    static_rows.append(list(static_rows[-1]))
    dynamic_rows.append(list(dynamic_rows[-1]))
    _write_csv(results / "static_objects.csv", STATIC_FIELDS, static_rows)
    _write_csv(results / "dynamic_objects.csv", DYNAMIC_FIELDS, dynamic_rows)
    _write_csv(
        results / "background_mesh.csv",
        [
            "Name",
            "Accuracy@0.2",
            "Completeness@0.2",
            "NumGTFailed",
            "NumDSGFailed",
        ],
        [[0, 0.8, 0.6, 0, 0], [1, 0.7, 0.7, 0, 0], [2, 0.9, 0.6, 0, 0]],
    )
    (results / "map_timestamps.txt").write_text(
        "\n".join(str(value) for value in timestamps) + "\n",
        encoding="utf-8",
    )
    return results


def _convert(value: str) -> int | float | str:
    if value.isnumeric():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _upstream_test_double() -> SimpleNamespace:
    def read_object_data(exp_dir: str, file_name: str = "static_objects.csv") -> dict:
        path = Path(exp_dir) / "results" / file_name
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        metrics = {
            field: {}
            for field in rows[0]
            if field not in {"Name", "Query"}
        }
        for row in rows:
            map_name = _convert(row["Name"])
            query = _convert(row["Query"])
            for field, field_queries in metrics.items():
                field_queries.setdefault(map_name, {})[query] = _convert(row[field])
        return {"._.": metrics}

    def read_dynamic_object_data(exp_dir: str) -> dict:
        raw = read_object_data(exp_dir, "dynamic_objects.csv")["._."]
        return {
            "._.": {
                "DynamicTP": raw["NumObjDetected"],
                "DynamicFP": raw["NumObjHallucinated"],
                "DynamicFN": raw["NumObjMissed"],
            }
        }

    def parse_table_data_simple(data_dir: str, methods: list[str]) -> list[dict]:
        parsed = []
        for method in methods:
            exp_dir = Path(data_dir) / method
            data = read_object_data(str(exp_dir))["._."]
            data.update(read_dynamic_object_data(str(exp_dir))["._."])
            bg_rows = list(
                csv.DictReader(
                    (exp_dir / "results/background_mesh.csv").open(
                        encoding="utf-8", newline=""
                    )
                )
            )
            for field in bg_rows[0]:
                if field == "Name":
                    continue
                data[field] = {}
                for row in bg_rows:
                    map_name = int(row["Name"])
                    value = _convert(row[field])
                    data[field][map_name] = {
                        query: value for query in range(map_name + 1)
                    }
            parsed.append(data)
        return parsed

    def f1(tp: int, fp: int, fn: int) -> float:
        precision = np.nan if tp + fp == 0 else tp / (tp + fp)
        recall = np.nan if tp + fn == 0 else tp / (tp + fn)
        denominator = precision + recall
        return np.nan if denominator == 0 else 2 * precision * recall / denominator

    def add_metric(data: dict, name: str, fields: tuple[str, str, str]) -> None:
        data[name] = {}
        for map_name, queries in data[fields[0]].items():
            data[name][map_name] = {}
            for query in queries:
                data[name][map_name][query] = f1(
                    data[fields[0]][map_name][query],
                    data[fields[1]][map_name][query],
                    data[fields[2]][map_name][query],
                )

    def augment_object_metrics(data: dict) -> None:
        add_metric(
            data,
            "ObjectF1",
            ("NumObjDetected", "NumObjHallucinated", "NumObjMissed"),
        )
        data["ChangeF1"] = {}
        for map_name, queries in data["AppearedTP"].items():
            data["ChangeF1"][map_name] = {}
            for query in queries:
                data["ChangeF1"][map_name][query] = f1(
                    data["AppearedTP"][map_name][query]
                    + data["DisappearedTP"][map_name][query],
                    data["AppearedFP"][map_name][query]
                    + data["DisappearedFP"][map_name][query],
                    data["AppearedFN"][map_name][query]
                    + data["DisappearedFN"][map_name][query],
                )

    def augment_dynamic_metrics(data: dict) -> None:
        add_metric(data, "DynamicF1", ("DynamicTP", "DynamicFP", "DynamicFN"))

    def augment_bg_metrics(data: dict) -> None:
        data["F1@0.2"] = {}
        for map_name, queries in data["Accuracy@0.2"].items():
            data["F1@0.2"][map_name] = {}
            for query in queries:
                accuracy = data["Accuracy@0.2"][map_name][query]
                completeness = data["Completeness@0.2"][map_name][query]
                denominator = accuracy + completeness
                data["F1@0.2"][map_name][query] = (
                    np.nan
                    if denominator == 0
                    else 2 * accuracy * completeness / denominator
                )

    def get_4d_data_slice(
        data: dict, map_names: list[int], query_times: list[int], time_mode: str
    ) -> np.ndarray:
        result = np.zeros(len(map_names))
        if time_mode == "Robot":
            for index, map_name in enumerate(map_names):
                result[index] = data[map_name][query_times[0]]
        elif time_mode == "Query":
            map_name = map_names[-1]
            for index, query_time in enumerate(query_times):
                result[index] = data[map_name][query_time]
        elif time_mode == "Online":
            for index, map_name in enumerate(map_names):
                result[index] = data[map_name][query_times[index]]
        return result

    return SimpleNamespace(
        parse_table_data_simple=parse_table_data_simple,
        augment_object_metrics=augment_object_metrics,
        augment_dynamic_metrics=augment_dynamic_metrics,
        augment_bg_metrics=augment_bg_metrics,
        get_4d_data_slice=get_4d_data_slice,
    )


def test_parity_compares_full_grid_duplicates_slices_and_weighting(tmp_path) -> None:
    results = _write_triangular_fixture(tmp_path)

    result = audit_metric_parity(results, upstream_utils=_upstream_test_double())

    assert result.row_grid_equal
    assert result.duplicate_policy_equal
    assert result.slice_values_equal
    assert result.metric_values_equal
    assert result.max_abs_delta <= 1e-12
    assert result.raw_row_counts == {
        "background_mesh.csv": 3,
        "dynamic_objects.csv": 7,
        "static_objects.csv": 7,
    }
    assert result.unique_state_counts == {
        "background_mesh.csv": 3,
        "dynamic_objects.csv": 6,
        "static_objects.csv": 6,
    }
    assert set(result.slice_comparisons) == {
        "4D",
        "Online",
        "Query",
        "Robot",
    }


def test_conflicting_duplicate_is_reported_as_non_parity(tmp_path) -> None:
    results = _write_triangular_fixture(tmp_path)
    static_path = results / "static_objects.csv"
    rows = list(csv.reader(static_path.open(encoding="utf-8", newline="")))
    conflicting = list(rows[-1])
    conflicting[-1] = str(int(conflicting[-1]) + 1)
    rows.append(conflicting)
    _write_csv(static_path, rows[0], rows[1:])

    with pytest.raises(KhronosParityError, match="conflicting duplicate"):
        audit_metric_parity(results, upstream_utils=_upstream_test_double())


def test_upstream_metric_drift_is_exposed_without_rewriting_local_result(
    tmp_path,
) -> None:
    results = _write_triangular_fixture(tmp_path)
    upstream = _upstream_test_double()
    original = upstream.augment_dynamic_metrics

    def drifted(data: dict) -> None:
        original(data)
        for queries in data["DynamicF1"].values():
            for query in queries:
                queries[query] *= 0.5

    upstream.augment_dynamic_metrics = drifted

    result = audit_metric_parity(results, upstream_utils=upstream)

    assert not result.slice_values_equal
    assert not result.metric_values_equal
    assert result.local_metrics["dynamic_f1"] > result.upstream_metrics["dynamic_f1"]
    assert result.max_abs_delta > 0


def test_all_nan_metric_is_preserved_as_matching_unavailable(tmp_path) -> None:
    results = _write_triangular_fixture(tmp_path)
    dynamic_path = results / "dynamic_objects.csv"
    rows = list(csv.DictReader(dynamic_path.open(encoding="utf-8", newline="")))
    for row in rows:
        row["NumObjDetected"] = "0"
        row["NumObjMissed"] = "0"
        row["NumObjHallucinated"] = "0"
    _write_csv(
        dynamic_path,
        DYNAMIC_FIELDS,
        [[row[field] for field in DYNAMIC_FIELDS] for row in rows],
    )

    result = audit_metric_parity(results, upstream_utils=_upstream_test_double())

    assert result.metric_values_equal
    assert result.local_metrics["dynamic_f1"] is None
    assert result.upstream_metrics["dynamic_f1"] is None
    assert result.slice_comparisons["4D"]["dynamic_f1"]["status"] == "PARITY"


def test_upstream_binding_rejects_worktree_source_drift(tmp_path) -> None:
    checkout = tmp_path / "khronos"
    source = checkout / "khronos_eval/plotting/utils.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=CROVE Test",
            "-c",
            "user.email=crove-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    bound_path, digest = _bind_upstream(checkout, commit)

    assert bound_path == source
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(KhronosParityError, match="differs from the declared commit"):
        _bind_upstream(checkout, commit)
