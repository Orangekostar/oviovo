"""Assemble fixed scalar selector inputs from acquired, model-specific evidence."""

from __future__ import annotations

import numpy as np

from .semantic_selector import (
    SemanticFeatureInputs,
    SemanticRequestStats,
    area_readout,
    build_semantic_features,
)


def _unit(value):
    value = np.asarray(value, np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm <= 0):
        raise ValueError("semantic evidence must have finite nonzero vectors")
    return value / norm


def semantic_feature_rows(native, manifest: dict, suggestions: dict, native_records: dict,
                          teacher_records: dict, native_aggregates: dict, native_text: np.ndarray,
                          teacher_text: np.ndarray, valid_ids: tuple[int, ...], *, teacher_model: str) -> dict:
    """Return features only for differing nontechnical suggestions; never consume GT."""
    if not native.locked or teacher_model not in {"siglip2", "wow"}:
        raise ValueError("semantic features require locked N0 and a listed alternative")
    native_text = _unit(native_text)
    teacher_text = _unit(teacher_text)
    index_by_id = {int(label): index for index, label in enumerate(valid_ids)}
    count = len(valid_ids)
    result = {}
    for owner, suggestion in suggestions.items():
        owner = int(owner)
        owner_mask = native.owner_ids == owner
        keep = int(np.unique(native.semantic_labels[owner_mask])[0])
        proposed = int(suggestion["label_id"])
        if keep == proposed or suggestion["technical_fallback"]:
            continue
        ids = manifest["views"].get(f"owner:{owner}", [])
        if not ids:
            raise ValueError("a successful teacher suggestion has no captured requests")
        native_good = [native_records[key] for key in ids if native_records[key]["status"] == "COMPLETE"]
        teacher_good_ids = [key for key in ids if teacher_records[key]["status"] == "COMPLETE"]
        teacher_good = [teacher_records[key] for key in teacher_good_ids]
        native_views = np.stack([row["feature"] for row in native_good]) if native_good else np.empty((0, native_text.shape[1]))
        native_scores = _unit(native_views) @ native_text.T if len(native_views) else np.empty((0, count))
        aggregate = _unit(native_aggregates[owner]) @ native_text.T
        crop_vectors = np.stack([row["vectors"] for row in native_good]) if native_good else np.empty((0, 9, native_text.shape[1]))
        if crop_vectors.shape != (len(native_good), 9, native_text.shape[1]):
            raise ValueError("native context evidence must preserve all nine crop vectors")
        crop_scores = crop_vectors @ native_text.T
        background = crop_scores[:, 6:9].copy()
        background_usable = []
        for index, row in enumerate(native_good):
            valid = np.asarray(row["background_scale_usable"], bool)
            if valid.shape != (3,):
                raise ValueError("background availability must cover the three native scales")
            background_usable.append(bool(valid.any()))
            if valid.any():
                # Only the per-view background mean is used by the fixed schema.
                # Filling missing scales by the valid-scale mean excludes empty
                # image support without changing that mean or any encoder vector.
                background[index, ~valid] = background[index, valid].mean(axis=0)
        if teacher_model == "siglip2":
            teacher_views = np.stack([row["feature"] for row in teacher_good])
            teacher_scores = _unit(teacher_views) @ teacher_text.T
            teacher_labels = tuple(map(int, np.argmax(teacher_scores, axis=1)))
            areas = [manifest["requests"][key]["visible_target_pixels"] for key in teacher_good_ids]
            teacher_aggregate = np.asarray(area_readout(teacher_views, areas, teacher_text, valid_ids=valid_ids).scores)
            mapping_gaps = ()
        else:
            teacher_scores = np.asarray([row["mapping"]["similarities"] for row in teacher_good])
            teacher_labels = tuple(int(row["mapping"]["class_index"]) for row in teacher_good)
            teacher_aggregate = teacher_scores.mean(axis=0)
            mapping_gaps = tuple(float(row["mapping"]["top1_top2_gap"]) for row in teacher_good)
        stats = []
        for key in ids:
            base = native_records[key]["stats"]
            teacher = teacher_records[key]
            status = teacher["status"]
            stats.append(SemanticRequestStats(**{name: base[name] for name in (
                "request_mask_pixels", "bbox_pixels", "depth_valid_fraction", "local_global_iou")},
                representation_survived=bool(teacher.get("representation_survived", False)),
                teacher_technical_failure=status == "UNAVAILABLE_TECHNICAL_FAILURE",
                unavailable_before_inference=status.startswith("UNAVAILABLE_EMPTY_"),
                native_valid=native_records[key]["status"] == "COMPLETE"))
        result[owner] = build_semantic_features(SemanticFeatureInputs(
            class_count=count, keep_index=index_by_id.get(keep), replace_index=index_by_id[proposed],
            native_aggregate_scores=aggregate, native_view_scores=native_scores,
            native_view_features=native_views, native_requested_views=len(ids),
            teacher_aggregate_scores=teacher_aggregate, teacher_view_labels=teacher_labels,
            teacher_view_strengths=teacher_scores, teacher_requested_views=len(ids), name_mapping_gaps=mapping_gaps,
            source_mask_point_count=int(owner_mask.sum()), request_stats=tuple(stats),
            native_raw_scores=crop_scores[:, (0, 2, 4)], native_foreground_scores=crop_scores[:, (1, 3, 5)],
            native_background_scores=background, background_usable=tuple(background_usable)))
    return result
