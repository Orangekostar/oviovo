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
        rows = list(csv.DictReader(handle))
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
    value = int(row[name])
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


def _finite_mean(
    values: list[float], *, label: str, missed_positive: list[bool] | None = None
) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite and missed_positive and any(missed_positive):
        return 0.0
    if not finite:
        raise ValueError(f"Khronos {label} has no finite states")
    return fmean(finite)


def summarize_khronos_official_metrics(
    results_dir: Path, *, allow_partial: bool = False
) -> dict[str, object]:
    """Reproduce the upstream plotting script's online 4D aggregation."""

    static = _read_unique_rows(
        results_dir / "static_objects.csv", ("Name", "Query")
    )
    dynamic = _read_unique_rows(
        results_dir / "dynamic_objects.csv", ("Name", "Query")
    )
    background = _read_unique_rows(results_dir / "background_mesh.csv", ("Name",))
    if set(static) != set(dynamic):
        raise ValueError("static and dynamic Khronos state keys disagree")

    object_f1 = []
    dynamic_f1 = []
    change_f1 = []
    object_missed_positive = []
    dynamic_missed_positive = []
    change_missed_positive = []
    for key in sorted(static, key=lambda item: (int(item[0]), int(item[1]))):
        row = static[key]
        object_tp = _count(row, "NumObjDetected")
        object_fp = _count(row, "NumObjHallucinated")
        object_fn = _count(row, "NumObjMissed")
        object_f1.append(
            _f1(object_tp, object_fp, object_fn)
        )
        object_missed_positive.append(object_tp == 0 and object_fp == 0 and object_fn > 0)
        tp = _count(row, "AppearedTP") + _count(row, "DisappearedTP")
        fp = _count(row, "AppearedFP") + _count(row, "DisappearedFP")
        fn = _count(row, "AppearedFN") + _count(row, "DisappearedFN")
        change_f1.append(_f1(tp, fp, fn))
        change_missed_positive.append(tp == 0 and fp == 0 and fn > 0)
        dynamic_row = dynamic[key]
        dynamic_tp = _count(dynamic_row, "NumObjDetected")
        dynamic_fp = _count(dynamic_row, "NumObjHallucinated")
        dynamic_fn = _count(dynamic_row, "NumObjMissed")
        dynamic_f1.append(_f1(dynamic_tp, dynamic_fp, dynamic_fn))
        dynamic_missed_positive.append(
            dynamic_tp == 0 and dynamic_fp == 0 and dynamic_fn > 0
        )

    background_f1: list[float] = []
    for key, row in sorted(background.items(), key=lambda item: int(item[0][0])):
        name = int(key[0])
        accuracy = float(row["Accuracy@0.2"])
        completeness = float(row["Completeness@0.2"])
        value = (
            math.nan
            if accuracy + completeness == 0
            else 2.0 * accuracy * completeness / (accuracy + completeness)
        )
        background_f1.extend([value] * (name + 1))

    calculations = (
        (
            "object_f1",
            lambda: _finite_mean(
                object_f1,
                label="object F1",
                missed_positive=object_missed_positive,
            ),
        ),
        (
            "dynamic_f1",
            lambda: _finite_mean(
                dynamic_f1,
                label="dynamic F1",
                missed_positive=dynamic_missed_positive,
            ),
        ),
        (
            "change_f1",
            lambda: _finite_mean(
                change_f1,
                label="change F1",
                missed_positive=change_missed_positive,
            ),
        ),
        (
            "background_f1_at_0_2",
            lambda: _finite_mean(background_f1, label="background F1@0.2"),
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
