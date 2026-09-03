"""Read-only geometry checks for CROVE dense moved-anchor candidates."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.dense_moved_readout import (
    DenseMovedReadoutGateConfig,
    decide_dense_geometry_agreement,
    measure_dense_geometry_agreement,
)

_DENSE_METADATA = {
    "geometry_authority": "ovimap_anchor_template",
    "geometry_source": "causal_ovimap_anchor",
    "state_authority": "crove_temporal",
    "readout_resolution_m": None,
    "readout_resolution_source": "native_ovimap_mesh_not_declared",
}
_TRANSLATION_TOLERANCE_M = 1e-6


def _threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("distance_threshold_m must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("distance_threshold_m must be a finite positive number")
    return normalized


def _entity_map(snapshot: object, *, label: str) -> dict[str, EntityPrediction]:
    if not isinstance(snapshot, MapSnapshot):
        raise TypeError(f"{label} must be a MapSnapshot")
    if snapshot.scene_id != "apartment" or snapshot.scope != "current":
        raise ValueError(f"{label} must be an Apartment current snapshot")
    result = {entity.entity_id: entity for entity in snapshot.entities}
    if len(result) != len(snapshot.entities):
        raise ValueError(f"{label} contains duplicate entity IDs")
    return result


def _moved_ids(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != tuple(sorted(set(value)))
    ):
        raise ValueError("moved_anchor_ids must be a nonempty sorted unique tuple")
    return value


def _metadata_value_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _state_fields_equal(
    compact: EntityPrediction, dense: EntityPrediction
) -> bool:
    scalar_fields = (
        "entity_id",
        "semantic_label",
        "semantic_score",
        "lifecycle_state",
        "first_seen",
        "last_seen",
    )
    if any(getattr(compact, name) != getattr(dense, name) for name in scalar_fields):
        return False
    if not _metadata_value_equal(
        compact.semantic_embedding, dense.semantic_embedding
    ):
        return False
    return all(
        key in dense.metadata and _metadata_value_equal(value, dense.metadata[key])
        for key, value in compact.metadata.items()
    )


def _validate_dense_metadata(entity: EntityPrediction, entity_id: str) -> None:
    metadata = entity.metadata
    if any(metadata.get(key) != value for key, value in _DENSE_METADATA.items()):
        raise ValueError("dense entity geometry authority metadata is invalid")
    if metadata.get("template_anchor_id") != entity_id:
        raise ValueError("dense entity geometry authority template is invalid")
    if metadata.get("transform_source") not in {
        "current_export_centroid_translation",
        "temporal_geometry_centroid_translation",
    }:
        raise ValueError("dense entity transform source is invalid")


def _audit_entity(
    anchor: EntityPrediction,
    compact: EntityPrediction,
    dense: EntityPrediction,
    *,
    threshold: float,
) -> dict[str, object]:
    if not _state_fields_equal(compact, dense):
        raise ValueError("compact and dense state fields differ")
    _validate_dense_metadata(dense, anchor.entity_id)
    anchor_points = np.asarray(anchor.points_xyz, dtype=np.float64)
    compact_points = np.asarray(compact.points_xyz, dtype=np.float64)
    dense_points = np.asarray(dense.points_xyz, dtype=np.float64)
    if anchor_points.shape != dense_points.shape:
        raise ValueError("dense point count must equal anchor point count")
    offsets = dense_points - anchor_points
    translation = offsets.mean(axis=0)
    residuals = np.linalg.norm(offsets - translation, axis=1)
    residual_max = float(residuals.max(initial=0.0))
    if residual_max > _TRANSLATION_TOLERANCE_M:
        raise ValueError("dense geometry is not a constant translation")
    gate_config = DenseMovedReadoutGateConfig(distance_threshold_m=threshold)
    agreement = measure_dense_geometry_agreement(
        compact_points,
        dense_points,
        gate_config,
    )
    decision = decide_dense_geometry_agreement(agreement, gate_config)
    return {
        "anchor_entity_id": anchor.entity_id,
        "anchor_point_count": len(anchor_points),
        "compact_point_count": len(compact_points),
        "dense_point_count": len(dense_points),
        "translation_xyz": translation.tolist(),
        "rigid_translation_residual_max_m": residual_max,
        "dense_to_compact_nn_median_m": agreement.template_to_compact_median_m,
        "dense_to_compact_nn_p90_m": agreement.template_to_compact_p90_m,
        "dense_to_compact_coverage_at_threshold": (
            agreement.template_to_compact_coverage
        ),
        "compact_to_dense_nn_median_m": agreement.compact_to_template_median_m,
        "compact_to_dense_nn_p90_m": agreement.compact_to_template_p90_m,
        "compact_to_dense_coverage_at_threshold": (
            agreement.compact_to_template_coverage
        ),
        "dense_compact_centroid_residual_m": agreement.centroid_residual_m,
        "compact_extent_xyz": list(agreement.compact_extent_xyz),
        "dense_extent_xyz": list(agreement.template_extent_xyz),
        "extent_residual_xyz": list(agreement.extent_residual_xyz),
        "extent_ratio_xyz": list(agreement.extent_ratio_xyz),
        "geometry_gate": decision.to_json_record(),
        "anchor_bbox_min_xyz": anchor_points.min(axis=0).tolist(),
        "anchor_bbox_max_xyz": anchor_points.max(axis=0).tolist(),
        "dense_bbox_min_xyz": dense_points.min(axis=0).tolist(),
        "dense_bbox_max_xyz": dense_points.max(axis=0).tolist(),
        "state_fields_equal": True,
    }


def audit_dense_moved_entities(
    anchor: MapSnapshot,
    compact: MapSnapshot,
    dense: MapSnapshot,
    moved_anchor_ids: tuple[str, ...],
    *,
    distance_threshold_m: float = 0.05,
) -> dict[str, object]:
    """Validate and measure a dense translation candidate without mutation."""

    threshold = _threshold(distance_threshold_m)
    anchor_by_id = _entity_map(anchor, label="anchor")
    compact_by_id = _entity_map(compact, label="compact")
    dense_by_id = _entity_map(dense, label="dense")
    moved_ids = _moved_ids(moved_anchor_ids)
    rows = []
    for entity_id in moved_ids:
        if any(
            entity_id not in entities
            for entities in (anchor_by_id, compact_by_id, dense_by_id)
        ):
            raise ValueError(f"missing moved entity: {entity_id}")
        rows.append(
            _audit_entity(
                anchor_by_id[entity_id],
                compact_by_id[entity_id],
                dense_by_id[entity_id],
                threshold=threshold,
            )
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "scene": "apartment",
        "distance_threshold_m": threshold,
        "moved_entity_count": len(rows),
        "anchor_point_count": sum(int(row["anchor_point_count"]) for row in rows),
        "compact_point_count": sum(
            int(row["compact_point_count"]) for row in rows
        ),
        "dense_point_count": sum(int(row["dense_point_count"]) for row in rows),
        "entities": rows,
    }


__all__ = ["audit_dense_moved_entities"]
