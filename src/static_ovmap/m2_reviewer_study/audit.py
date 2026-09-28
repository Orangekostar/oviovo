"""Exact measured row coverage and released-pool dependency checks."""

import math


def validate_matrix(rows, pools, *, datasets, methods, ranks):
    """Reject duplicated/substituted rows even when aggregate counts agree.

    This checks the measured matrix, not scientific correctness of predictions;
    source, geometry and released-export parity have separate runtime evidence.
    """
    expected = {(dataset, scene, method, rank)
                for dataset, scenes in datasets.items() for scene in scenes
                for method in methods for rank in ranks}
    measured = {}

    def metrics(row):
        if row.get("status") != "COMPLETE":
            raise ValueError("matrix contains a non-measured row")
        values = row["metrics"]
        for key in ("uap", "apall", "ap25", "ap50", "miou", "macc"):
            value = values.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"matrix metric missing or invalid: {key}")
        if values["apall"] != values["uap"]:
            raise ValueError("APall must be the exact uAP alias")

    for row in rows:
        key = tuple(row[k] for k in ("dataset", "scene", "method", "rank_mode"))
        if key in measured:
            raise ValueError(f"duplicate matrix row: {key}")
        metrics(row)
        measured[key] = row
    if set(measured) != expected:
        raise ValueError(f"scene matrix mismatch: missing={sorted(expected - measured.keys())}, unexpected={sorted(measured.keys() - expected)}")
    expected_pools = {(dataset, method, rank) for dataset in datasets for method in methods for rank in ranks}
    seen = set()
    for row in pools:
        key = tuple(row[k] for k in ("dataset", "method", "rank_mode"))
        if key in seen:
            raise ValueError(f"duplicate pooled matrix row: {key}")
        seen.add(key)
        metrics(row)
        if key not in expected_pools:
            raise ValueError(f"unexpected pooled matrix row: {key}")
        dataset, method, rank = key
        scenes = list(datasets[dataset])
        inputs = [measured[(dataset, scene, method, rank)]["evaluation_identity"] for scene in scenes]
        if row.get("scene_order") != scenes or row.get("ordered_inputs") != inputs:
            raise ValueError(f"pool ordered dependencies disagree with measured scene rows: {key}")
    if seen != expected_pools:
        raise ValueError(f"pooled matrix mismatch: missing={sorted(expected_pools - seen)}")
    return {"scene_rows": len(measured), "pooled_rows": len(seen)}
