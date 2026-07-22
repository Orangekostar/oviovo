"""Pure parsers for TESSE-CD metrics emitted by the Khronos evaluator."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean


def _read_unique_rows(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Khronos result file is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(key_fields) - fields
        if missing:
            raise ValueError(
                f"Khronos result file {path} lacks columns: {sorted(missing)}"
            )
        rows = list(reader)
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in unique and unique[key] != row:
            raise ValueError(f"conflicting duplicate Khronos row {key} in {path}")
        unique[key] = row
    if not unique:
        raise ValueError(f"Khronos result file is empty: {path}")
    return unique


def _count(row: dict[str, str], name: str) -> int:
    if name not in row:
        raise ValueError(f"Khronos result row lacks column: {name}")
    try:
        value = int(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Khronos count: {name}={row[name]!r}") from error
    if value < 0:
        raise ValueError(f"negative Khronos count: {name}={value}")
    return value


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = math.nan if tp + fp == 0 else tp / (tp + fp)
    recall = math.nan if tp + fn == 0 else tp / (tp + fn)
    return (
        math.nan
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )


def _finite_mean(values: list[float], *, label: str) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError(f"Khronos {label} has no finite states")
    return fmean(finite)


def summarize_khronos_official_metrics(
    results_dir: Path, *, allow_partial: bool = False
) -> dict[str, object]:
    """Reproduce the upstream plotting script's online 4D aggregation."""

    table_errors: dict[str, ValueError] = {}
    tables: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    for name, keys in (
        ("static_objects.csv", ("Name", "Query")),
        ("dynamic_objects.csv", ("Name", "Query")),
        ("background_mesh.csv", ("Name",)),
    ):
        try:
            tables[name] = _read_unique_rows(results_dir / name, keys)
        except ValueError as error:
            table_errors[name] = error
            tables[name] = {}
    static = tables["static_objects.csv"]
    dynamic = tables["dynamic_objects.csv"]
    background = tables["background_mesh.csv"]

    object_f1 = []
    dynamic_f1 = []
    change_f1 = []
    metric_errors: dict[str, ValueError] = {}
    if "static_objects.csv" not in table_errors:
        try:
            ordered_static = sorted(
                static, key=lambda item: (int(item[0]), int(item[1]))
            )
        except (TypeError, ValueError) as error:
            table_errors["static_objects.csv"] = ValueError(
                "invalid static Khronos state key"
            )
            ordered_static = []
        try:
            for key in ordered_static:
                row = static[key]
                object_f1.append(
                    _f1(
                        _count(row, "NumObjDetected"),
                        _count(row, "NumObjHallucinated"),
                        _count(row, "NumObjMissed"),
                    )
                )
        except ValueError as error:
            metric_errors["object_f1"] = error
            object_f1 = []
        try:
            for key in ordered_static:
                row = static[key]
                change_f1.append(
                    _f1(
                        _count(row, "AppearedTP") + _count(row, "DisappearedTP"),
                        _count(row, "AppearedFP") + _count(row, "DisappearedFP"),
                        _count(row, "AppearedFN") + _count(row, "DisappearedFN"),
                    )
                )
        except ValueError as error:
            metric_errors["change_f1"] = error
            change_f1 = []
    if not ({"static_objects.csv", "dynamic_objects.csv"} & set(table_errors)):
        if set(static) != set(dynamic):
            table_errors["dynamic_objects.csv"] = ValueError(
                "static and dynamic Khronos state keys disagree"
            )
        else:
            try:
                for key in sorted(dynamic, key=lambda item: (int(item[0]), int(item[1]))):
                    row = dynamic[key]
                    dynamic_f1.append(
                        _f1(
                            _count(row, "NumObjDetected"),
                            _count(row, "NumObjHallucinated"),
                            _count(row, "NumObjMissed"),
                        )
                    )
            except ValueError as error:
                metric_errors["dynamic_f1"] = error
                dynamic_f1 = []
    elif "dynamic_objects.csv" not in table_errors:
        metric_errors["dynamic_f1"] = table_errors["static_objects.csv"]

    background_f1: list[float] = []
    if "background_mesh.csv" not in table_errors:
        try:
            for key, row in sorted(
                background.items(), key=lambda item: int(item[0][0])
            ):
                accuracy = float(row["Accuracy@0.2"])
                completeness = float(row["Completeness@0.2"])
                name = int(key[0])
                value = (
                    math.nan
                    if accuracy + completeness == 0
                    else 2.0 * accuracy * completeness / (accuracy + completeness)
                )
                background_f1.extend([value] * (name + 1))
        except (KeyError, TypeError, ValueError) as error:
            detail = error.args[0] if isinstance(error, KeyError) else str(error)
            table_errors["background_mesh.csv"] = ValueError(
                f"invalid Khronos background result: {detail}"
            )
            background_f1 = []

    def _table_metric(
        metric: str, filename: str, values: list[float], label: str
    ) -> float:
        if metric in metric_errors:
            raise metric_errors[metric]
        if filename in table_errors:
            raise table_errors[filename]
        return _finite_mean(values, label=label)

    calculations = (
        (
            "object_f1",
            lambda: _table_metric(
                "object_f1", "static_objects.csv", object_f1, "object F1"
            ),
        ),
        (
            "dynamic_f1",
            lambda: _table_metric(
                "dynamic_f1", "dynamic_objects.csv", dynamic_f1, "dynamic F1"
            ),
        ),
        (
            "change_f1",
            lambda: _table_metric(
                "change_f1", "static_objects.csv", change_f1, "change F1"
            ),
        ),
        (
            "background_f1_at_0_2",
            lambda: _table_metric(
                "background_f1_at_0_2",
                "background_mesh.csv",
                background_f1,
                "background F1@0.2",
            ),
        ),
    )
    metrics: dict[str, float | None] = {}
    unavailable: dict[str, str] = {}
    for name, calculate in calculations:
        try:
            metrics[name] = calculate()
        except ValueError as error:
            metrics[name] = None
            unavailable[name] = str(error)
    if allow_partial:
        return {
            "state_count": len(static),
            "metrics": metrics,
            "unavailable": unavailable,
        }
    if unavailable:
        raise ValueError(next(iter(unavailable.values())))
    return {"state_count": len(static), **metrics}


def summarize_khronos_official_metrics_partial(results_dir: Path) -> dict[str, object]:
    """Return finite official metrics while preserving auditable N/A reasons."""

    return summarize_khronos_official_metrics(results_dir, allow_partial=True)
