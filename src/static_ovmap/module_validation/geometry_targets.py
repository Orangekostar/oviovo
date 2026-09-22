"""Annotation-only targets for stored G hypotheses on the frozen projection."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .native_capture import _array_digest
from .partition_quality import LocalPQTarget
from .scannet_runtime import reusable_job


def projected_local_quality(projected_components, component_count, whole_gt_instances):
    """Local PQ with every source component, including zero projected support."""
    from scipy.optimize import linear_sum_assignment

    components = np.asarray(projected_components, np.int64)
    gt = np.asarray(whole_gt_instances, np.int64)
    if components.ndim != 1 or components.shape != gt.shape or component_count <= 0:
        raise ValueError("local quality requires aligned whole GT rows and a nonempty source partition")
    if np.any(components < -1) or np.any(components >= component_count) or np.any(gt < 0):
        raise ValueError("local quality labels are out of range")
    support = components >= 0
    included_ids = np.unique(gt[support & (gt > 0)])
    predicted_sizes = np.bincount(components[support], minlength=component_count)
    matrix = np.zeros((component_count, len(included_ids)), np.float64)
    for column, gt_id in enumerate(included_ids):
        whole = gt == gt_id
        intersection = np.bincount(components[whole & support], minlength=component_count)
        union = predicted_sizes + int(whole.sum()) - intersection
        matrix[:, column] = np.divide(intersection, union, out=np.zeros(component_count, float), where=union > 0)
    if matrix.size:
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        accepted = matrix[rows, columns] > .5
        matched_sum = float(matrix[rows[accepted], columns[accepted]].sum())
        true_positives = int(accepted.sum())
    else:
        matched_sum, true_positives = 0., 0
    false_positives = component_count - true_positives
    false_negatives = len(included_ids) - true_positives
    denominator = true_positives + .5 * false_positives + .5 * false_negatives
    return LocalPQTarget(matched_sum / denominator if denominator else 0., matched_sum,
        true_positives, false_positives, false_negatives, len(included_ids), denominator == 0)


def prepare_geometry_targets(pool, pool_path, targets, output):
    output = Path(output)
    inputs = [file_identity(pool_path), file_identity(Path(__file__))]
    identity = canonical_digest({"inputs": inputs, "gt_instance": _array_digest(targets["gt_instance"]),
        "nearest": _array_digest(targets["nearest"]), "matched": _array_digest(targets["matched"])})
    receipt_path, target_path = output / "receipt.json", output / "targets.json"
    if reusable_job(receipt_path, identity):
        return target_path
    if receipt_path.exists():
        raise ValueError("completed G target inputs changed")
    native_leaves = pool["leaves"]
    projected_leaf = np.full(len(targets["nearest"]), -1, np.int64)
    projected_leaf[targets["matched"]] = native_leaves.row_leaf_ids[targets["nearest"][targets["matched"]]]
    result = {}
    for group, hypotheses in pool["hypotheses"].items():
        group_rows = []
        for hypothesis in hypotheses:
            component_by_leaf = np.full(len(native_leaves.leaves), -1, np.int64)
            component_by_leaf[list(hypothesis.leaf_ids)] = hypothesis.components
            components = np.full(len(projected_leaf), -1, np.int64)
            visible = projected_leaf >= 0
            components[visible] = component_by_leaf[projected_leaf[visible]]
            quality = projected_local_quality(components, hypothesis.component_count, targets["gt_instance"])
            group_rows.append({"hypothesis_id": hypothesis.hypothesis_id, **asdict(quality)})
        result[group] = group_rows
    atomic_write_json(target_path, {"scene_id": pool["receipt"]["scene_id"], "targets": result,
        "definition": "whole positive native GT instance IDs; no GT crop or source-component deletion",
        "oracle_diagnostic_not_method": {group: max(row["quality"] for row in rows) for group, rows in result.items()}})
    atomic_write_json(receipt_path, {"status": "COMPLETE", "input_identity": identity, "inputs": inputs,
        "outputs": [file_identity(target_path)], "prediction_pool_locked_before_GT": True})
    return target_path
