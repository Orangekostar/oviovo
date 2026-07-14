"""Current-state query ranking, rejection, and stale-return metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.evaluation.contracts import QueryResult, QueryTarget


def _safe_ratio(numerator: float, denominator: float, *, empty: float) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def evaluate_query_results(
    results: Sequence[QueryResult],
    targets: Sequence[QueryTarget],
) -> dict[str, Any]:
    target_by_id = {target.query_id: target for target in targets}
    result_by_id = {result.query_id: result for result in results}
    if len(target_by_id) != len(targets) or len(result_by_id) != len(results):
        raise ValueError("query ids must be unique")
    if set(target_by_id) != set(result_by_id):
        raise ValueError("query result ids must exactly match target ids")

    present_count = 0
    top1_correct = 0
    actual_absent = 0
    predicted_absent = 0
    absence_true_positive = 0
    stale_false_positive = 0
    latencies: list[float] = []
    for query_id, target in target_by_id.items():
        result = result_by_id[query_id]
        valid_ids = set(target.valid_entity_ids)
        is_absent = not valid_ids
        predicted_not_found = bool(result.rejected_as_not_found)
        latencies.append(float(result.latency_ms))
        if is_absent:
            actual_absent += 1
            if not predicted_not_found and result.returned_entity_id is not None:
                stale_false_positive += 1
        else:
            present_count += 1
            if result.ranked_entity_ids and result.ranked_entity_ids[0] in valid_ids:
                top1_correct += 1
        if predicted_not_found:
            predicted_absent += 1
            if is_absent:
                absence_true_positive += 1

    not_found_precision = _safe_ratio(absence_true_positive, predicted_absent, empty=0.0)
    not_found_recall = _safe_ratio(absence_true_positive, actual_absent, empty=1.0)
    not_found_f1 = _safe_ratio(
        2.0 * not_found_precision * not_found_recall,
        not_found_precision + not_found_recall,
        empty=0.0,
    )
    return {
        "current_r1": _safe_ratio(top1_correct, present_count, empty=1.0),
        "not_found_precision": not_found_precision,
        "not_found_recall": not_found_recall,
        "not_found_f1": not_found_f1,
        "stale_fp": _safe_ratio(stale_false_positive, actual_absent, empty=0.0),
        "query_count": int(len(targets)),
        "present_query_count": int(present_count),
        "absent_query_count": int(actual_absent),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }
