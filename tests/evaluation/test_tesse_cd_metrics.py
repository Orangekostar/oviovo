from __future__ import annotations

import csv

import pytest

from src.evaluation.baselines.tesse_cd import (
    summarize_khronos_official_metrics,
    summarize_khronos_official_metrics_partial,
)


def _write_csv(path, headers, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_summarize_khronos_official_metrics_matches_online_4d_slice(tmp_path) -> None:
    results = tmp_path / "results"
    static_headers = [
        "Name",
        "Query",
        "AppearedTP",
        "DisappearedTP",
        "AppearedFP",
        "DisappearedFP",
        "AppearedFN",
        "DisappearedFN",
        "NumObjDetected",
        "NumObjMissed",
        "NumObjHallucinated",
    ]
    static_rows = [
        [0, 10, 1, 1, 1, 0, 0, 1, 8, 2, 2],
        [1, 20, 1, 0, 0, 0, 1, 0, 6, 4, 0],
        [1, 20, 1, 0, 0, 0, 1, 0, 6, 4, 0],
    ]
    _write_csv(results / "static_objects.csv", static_headers, static_rows)
    _write_csv(
        results / "dynamic_objects.csv",
        ["Name", "Query", "NumObjDetected", "NumObjMissed", "NumObjHallucinated"],
        [[0, 10, 4, 1, 0], [1, 20, 3, 1, 1], [1, 20, 3, 1, 1]],
    )
    _write_csv(
        results / "background_mesh.csv",
        ["Name", "Accuracy@0.2", "Completeness@0.2"],
        [[0, 1.0, 0.5], [1, 0.8, 0.8]],
    )

    metrics = summarize_khronos_official_metrics(results)

    assert metrics["state_count"] == 2
    assert metrics["object_f1"] == pytest.approx((0.8 + 0.75) / 2.0)
    assert metrics["dynamic_f1"] == pytest.approx((8 / 9 + 0.75) / 2.0)
    assert metrics["change_f1"] == pytest.approx(2 / 3)
    assert metrics["background_f1_at_0_2"] == pytest.approx((2 / 3 + 0.8 * 2) / 3.0)


def test_summarize_khronos_official_metrics_rejects_conflicting_duplicates(
    tmp_path,
) -> None:
    results = tmp_path / "results"
    _write_csv(
        results / "static_objects.csv",
        [
            "Name",
            "Query",
            "AppearedTP",
            "DisappearedTP",
            "AppearedFP",
            "DisappearedFP",
            "AppearedFN",
            "DisappearedFN",
            "NumObjDetected",
            "NumObjMissed",
            "NumObjHallucinated",
        ],
        [
            [0, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ],
    )
    _write_csv(
        results / "dynamic_objects.csv",
        ["Name", "Query", "NumObjDetected", "NumObjMissed", "NumObjHallucinated"],
        [[0, 10, 1, 0, 0]],
    )
    _write_csv(
        results / "background_mesh.csv",
        ["Name", "Accuracy@0.2", "Completeness@0.2"],
        [[0, 1.0, 1.0]],
    )

    with pytest.raises(ValueError, match="conflicting duplicate"):
        summarize_khronos_official_metrics(results)


def test_summarize_khronos_official_metrics_marks_all_nan_as_unavailable(
    tmp_path,
) -> None:
    results = tmp_path / "results"
    _write_csv(
        results / "static_objects.csv",
        [
            "Name",
            "Query",
            "AppearedTP",
            "DisappearedTP",
            "AppearedFP",
            "DisappearedFP",
            "AppearedFN",
            "DisappearedFN",
            "NumObjDetected",
            "NumObjMissed",
            "NumObjHallucinated",
        ],
        [[0, 10, 0, 0, 0, 0, 1, 0, 0, 1, 0]],
    )
    _write_csv(
        results / "dynamic_objects.csv",
        ["Name", "Query", "NumObjDetected", "NumObjMissed", "NumObjHallucinated"],
        [[0, 10, 0, 1, 0]],
    )
    _write_csv(
        results / "background_mesh.csv",
        ["Name", "Accuracy@0.2", "Completeness@0.2"],
        [[0, 1.0, 1.0]],
    )

    partial = summarize_khronos_official_metrics_partial(results)

    assert partial["metrics"]["object_f1"] is None
    assert partial["metrics"]["dynamic_f1"] is None
    assert partial["metrics"]["change_f1"] is None
    assert set(partial["unavailable"]) == {
        "object_f1",
        "dynamic_f1",
        "change_f1",
    }
    with pytest.raises(ValueError, match="no finite states"):
        summarize_khronos_official_metrics(results)


def test_partial_summary_preserves_finite_metrics_when_change_is_undefined(
    tmp_path,
) -> None:
    results = tmp_path / "results"
    _write_csv(
        results / "static_objects.csv",
        [
            "Name", "Query", "AppearedTP", "DisappearedTP", "AppearedFP",
            "DisappearedFP", "AppearedFN", "DisappearedFN", "NumObjDetected",
            "NumObjMissed", "NumObjHallucinated",
        ],
        [[0, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    )
    _write_csv(
        results / "dynamic_objects.csv",
        ["Name", "Query", "NumObjDetected", "NumObjMissed", "NumObjHallucinated"],
        [[0, 10, 1, 0, 0]],
    )
    _write_csv(
        results / "background_mesh.csv",
        ["Name", "Accuracy@0.2", "Completeness@0.2"],
        [[0, 1.0, 1.0]],
    )

    partial = summarize_khronos_official_metrics_partial(results)

    assert partial["metrics"]["object_f1"] == 1.0
    assert partial["metrics"]["dynamic_f1"] == 1.0
    assert partial["metrics"]["change_f1"] is None
    assert partial["unavailable"] == {
        "change_f1": "Khronos change F1 has no finite states"
    }
    with pytest.raises(ValueError, match="change F1 has no finite states"):
        summarize_khronos_official_metrics(results)


def test_partial_summary_marks_missing_required_column_unavailable(tmp_path) -> None:
    results = tmp_path / "results"
    _write_csv(
        results / "static_objects.csv",
        [
            "Name", "Query", "AppearedTP", "DisappearedTP", "AppearedFP",
            "DisappearedFP", "AppearedFN", "DisappearedFN", "NumObjDetected",
            "NumObjMissed",
        ],
        [[0, 10, 1, 0, 0, 0, 0, 0, 1, 0]],
    )
    _write_csv(
        results / "dynamic_objects.csv",
        ["Name", "Query", "NumObjDetected", "NumObjMissed", "NumObjHallucinated"],
        [[0, 10, 1, 0, 0]],
    )
    _write_csv(
        results / "background_mesh.csv",
        ["Name", "Accuracy@0.2", "Completeness@0.2"],
        [[0, 1.0, 1.0]],
    )

    partial = summarize_khronos_official_metrics_partial(results)

    assert partial["metrics"]["object_f1"] is None
    assert partial["metrics"]["dynamic_f1"] == 1.0
    assert "NumObjHallucinated" in partial["unavailable"]["object_f1"]
    with pytest.raises(ValueError, match="NumObjHallucinated"):
        summarize_khronos_official_metrics(results)
