from __future__ import annotations

import pytest

from src.evaluation.khronos_attribution import AttributionRow
from src.evaluation.tesse_current_slice import (
    CurrentSliceError,
    MetricCountRow,
    evaluate_current_diagonal,
    validate_attribution_time_domain,
)


def _row(
    metric_type: str,
    map_name: int,
    query_time_ns: int,
    tp: int,
    fp: int,
    fn: int,
    ordinal: int,
) -> MetricCountRow:
    return MetricCountRow(
        metric_type=metric_type,
        map_name=map_name,
        query_time_ns=query_time_ns,
        tp=tp,
        fp=fp,
        fn=fn,
        source_row_ordinal=ordinal,
    )


def test_current_slice_requires_equal_robot_and_belief_time() -> None:
    rows = []
    ordinal = 0
    for metric_type in ("object", "dynamic_object", "change"):
        rows.extend(
            [
                _row(metric_type, 0, 10, 1, 0, 0, ordinal),
                _row(metric_type, 1, 10, 0, 1, 1, ordinal + 1),
                _row(metric_type, 1, 20, 1, 1, 0, ordinal + 2),
            ]
        )
        ordinal += 3

    result = evaluate_current_diagonal(rows=rows, robot_times=(10, 20))

    assert len(result.rows) == 6
    assert all(row.robot_time_ns == row.query_time_ns for row in result.rows)
    assert result.metrics["object_f1"] == pytest.approx(5 / 6)
    assert result.metrics["dynamic_f1"] == pytest.approx(5 / 6)
    assert result.metrics["change_f1"] == pytest.approx(5 / 6)


def test_current_slice_rejects_conflicting_duplicate_and_future_query() -> None:
    common = [
        _row(metric, 0, 10, 1, 0, 0, index)
        for index, metric in enumerate(("object", "dynamic_object", "change"))
    ]
    with pytest.raises(CurrentSliceError, match="conflicting duplicate"):
        evaluate_current_diagonal(
            rows=[*common, _row("object", 0, 10, 0, 1, 0, 3)],
            robot_times=(10,),
        )
    with pytest.raises(CurrentSliceError, match="future belief time"):
        evaluate_current_diagonal(
            rows=[*common, _row("object", 0, 11, 1, 0, 0, 3)],
            robot_times=(10,),
        )


def test_zero_true_positive_state_is_missing_like_upstream_khronos() -> None:
    result = evaluate_current_diagonal(
        rows=[
            _row(metric, 0, 10, 0, 1, 1, ordinal)
            for ordinal, metric in enumerate(
                ("object", "dynamic_object", "change")
            )
        ],
        robot_times=(10,),
    )

    assert result.metrics == {
        "object_f1": None,
        "dynamic_f1": None,
        "change_f1": None,
    }


def test_attribution_rejects_future_trajectory_and_duplicate_divergence() -> None:
    metric_rows = [
        _row("dynamic_object", 0, 10, 1, 0, 0, 0),
        _row("dynamic_object", 0, 10, 1, 0, 0, 1),
    ]
    base = {
        "metric_type": "dynamic_object",
        "map_name": "0",
        "query_time_ns": 10,
        "pred_node_id": "O1",
        "gt_node_id": "G1",
        "distance_m": 0.1,
        "status": "TP",
    }
    with pytest.raises(CurrentSliceError, match="future trajectory"):
        validate_attribution_time_domain(
            attribution_rows=(
                AttributionRow(
                    metric_row_ordinal=0,
                    trajectory_timestamp_ns=11,
                    **base,
                ),
            ),
            metric_rows=metric_rows,
            robot_times=(10,),
        )
    with pytest.raises(CurrentSliceError, match="duplicate attribution differs"):
        validate_attribution_time_domain(
            attribution_rows=(
                AttributionRow(
                    metric_row_ordinal=0,
                    trajectory_timestamp_ns=9,
                    **base,
                ),
                AttributionRow(
                    metric_row_ordinal=1,
                    trajectory_timestamp_ns=8,
                    **base,
                ),
            ),
            metric_rows=metric_rows,
            robot_times=(10,),
        )
