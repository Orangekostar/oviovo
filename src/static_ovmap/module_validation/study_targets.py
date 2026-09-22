"""Evaluation-only ScanNet supervision; never imported by prediction builders."""

from __future__ import annotations

import numpy as np

from src.static_ovmap.attribution_objects import borrowing_diagnosis

from .scannet_study import project_values
from .semantic_selector import semantic_event_label


def semantic_events(native, suggestions: dict, nearest: np.ndarray, matched: np.ndarray,
                    gt_combined: np.ndarray, valid_ids: tuple[int, ...]) -> list[dict]:
    """One event per native object, using the existing independent IoU helper."""
    if not native.locked:
        raise ValueError("semantic supervision requires locked prediction geometry")
    projected = project_values(native.owner_ids, nearest, matched)
    gt_combined = np.asarray(gt_combined).reshape(-1)
    if projected.shape != gt_combined.shape:
        raise ValueError("whole GT and frozen projection must align")
    gt_ids, inverse, gt_sizes = np.unique(gt_combined, return_inverse=True, return_counts=True)
    records, intersections, predicted_sizes = [], [], []
    for owner, suggestion in sorted(suggestions.items()):
        mask = projected == owner
        source = native.owner_ids == owner
        if not np.any(source):
            raise ValueError("suggestion is outside the frozen native registry")
        incumbent = int(np.unique(native.semantic_labels[source])[0])
        records.append({"candidate_id": int(owner), "borrowed_from": "frozen_teacher", "kept": True,
            "original_class_id": incumbent, "final_class_id": int(suggestion["label_id"])})
        intersections.append(np.bincount(inverse[mask], minlength=len(gt_ids)))
        predicted_sizes.append(int(mask.sum()))
    if not records:
        return []
    diagnoses = borrowing_diagnosis(records, np.asarray(intersections), predicted_sizes,
                                    gt_ids, gt_sizes, valid_ids, [0.5])
    result = []
    for row in diagnoses:
        owner = row["candidate_id"]
        match = row["thresholds"]["0.5"]
        unique = match["correspondence"] == "unique"
        available = not suggestions[owner]["technical_fallback"]
        label = int(match["geometry_GT_ids"][0] // 1000) if unique else None
        result.append({"scene_id": native.scene_id, "owner_id": owner,
            "incumbent_label": row["old_class"], "suggestion_label": row["new_class"],
            "technical_available": available, "same_label": row["same_label"],
            "correspondence": match["correspondence"], "gt_label": label,
            "geometry_gt_ids": match["geometry_GT_ids"],
            "geometry_iou": float(match["geometry_IoUs"][0]) if unique else None,
            "event": semantic_event_label(row["old_class"], row["new_class"], label,
                technically_available=available) if unique else None})
    return result
