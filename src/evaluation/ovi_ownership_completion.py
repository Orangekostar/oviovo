"""Pure ownership readout and shape-bound completion metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from src.evaluation.ovi_pair_views import OviObjectPairView
from src.evaluation.temporal_object_groups import (
    GroupingConfig,
    TemporalObjectGrouping,
    build_temporal_object_groups,
)
from src.oviv2.two_visit_contracts import NeuralSampleMap, PairRelation


class OwnershipCompletionError(ValueError):
    """Raised when ownership or completion evidence violates its contract."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _canonical_sha256(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def build_ownership_readout(
    pair: NeuralSampleMap,
    relations: Sequence[PairRelation],
    *,
    variant_id: str,
) -> dict[str, object]:
    """Group immutable entity geometry under temporal identities without moving it."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be a NeuralSampleMap")
    if variant_id not in {"A0", "A1", "A2"}:
        raise OwnershipCompletionError("ownership variant must be A0, A1, or A2")
    rows = tuple(relations)
    if any(not isinstance(value, PairRelation) for value in rows):
        raise OwnershipCompletionError("relations contain an invalid value")
    if variant_id == "A0" and rows:
        raise OwnershipCompletionError("A0 cannot consume temporal relations")
    groups: dict[tuple[int, str], list[int]] = {}
    for index, entity_id in enumerate(pair.token_entity_ids):
        groups.setdefault((int(pair.visit_ids[index]), entity_id), []).append(index)
    entity_geometry = {
        f"t{visit_id}:{entity_id}": _array_sha256(
            pair.coordinates_xyzt[np.asarray(indices, dtype=np.int64), :3]
        )
        for (visit_id, entity_id), indices in sorted(groups.items())
    }
    assignments: dict[tuple[int, str], str] = {}
    identity_groups: dict[str, list[str]] = {}
    relation_ids: set[str] = set()
    for relation in rows:
        relation_id = str(relation.temporal_query_id)
        if relation_id in relation_ids:
            raise OwnershipCompletionError("relation assigns entities multiple times")
        relation_ids.add(relation_id)
        members = tuple(
            [(0, entity_id) for entity_id in relation.t0_entity_ids]
            + [(1, entity_id) for entity_id in relation.t1_entity_ids]
        )
        if any(member not in groups for member in members):
            raise OwnershipCompletionError("relation references an unknown entity")
        if any(member in assignments for member in members):
            raise OwnershipCompletionError("entity has multiple temporal owners")
        track_id = f"{variant_id}:relation:{relation_id}"
        for member in members:
            assignments[member] = track_id
        identity_groups[track_id] = [
            f"t{visit_id}:{entity_id}" for visit_id, entity_id in members
        ]
    for member in sorted(groups):
        if member in assignments:
            continue
        visit_id, entity_id = member
        track_id = f"{variant_id}:unmatched:t{visit_id}:{entity_id}"
        assignments[member] = track_id
        identity_groups[track_id] = [f"t{visit_id}:{entity_id}"]
    geometry_sha256 = _canonical_sha256(entity_geometry)
    return {
        "schema_version": 1,
        "status": "PASS",
        "variant_id": variant_id,
        "pair_sha256": pair.content_sha256(),
        "source_entity_count": len(groups),
        "identity_group_count": len(identity_groups),
        "cross_visit_identity_group_count": sum(
            any(value.startswith("t0:") for value in members)
            and any(value.startswith("t1:") for value in members)
            for members in identity_groups.values()
        ),
        "many_sided_identity_group_count": sum(
            len(members) > 2 for members in identity_groups.values()
        ),
        "geometry_mutation_count": 0,
        "geometry_sha256": geometry_sha256,
        "entity_geometry_sha256": entity_geometry,
        "identity_groups": dict(sorted(identity_groups.items())),
        "instance_ap": None,
        "panoptic_quality": None,
        "ground_truth_status": "NOT_COMPUTED_NO_INSTANCE_IDENTITY_GT",
    }


def build_dense_ownership_readout(
    pair: OviObjectPairView,
    relations: Sequence[PairRelation],
    *,
    variant_id: Literal["A_ID", "U0", "U1", "U2", "U3"],
    minimum_query_confidence: float = 0.3,
) -> TemporalObjectGrouping:
    """Build a dense OVI-backed readout while preserving current XYZ coordinates."""

    return build_temporal_object_groups(
        pair,
        relations,
        variant_id=variant_id,
        config=GroupingConfig(
            minimum_query_confidence=minimum_query_confidence,
        ),
    )


def _voxels(value: object, *, voxel_size_m: float, label: str) -> set[tuple[int, int, int]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.all(np.isfinite(array)):
        raise OwnershipCompletionError(f"{label} must have finite shape (N, 3)")
    quantized = np.floor(array / voxel_size_m).astype(np.int64)
    return {tuple(int(axis) for axis in row) for row in quantized}


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_completion_surface(
    *,
    baseline_xyz: object,
    recovered_xyz: object,
    target_xyz: object,
    historical_candidate_xyz: object,
    voxel_size_m: float = 0.05,
) -> dict[str, object]:
    """Measure recovery only against supplied observed shapes on one voxel grid."""

    if (
        isinstance(voxel_size_m, bool)
        or not isinstance(voxel_size_m, (int, float))
        or not math.isfinite(float(voxel_size_m))
        or float(voxel_size_m) <= 0.0
    ):
        raise OwnershipCompletionError("voxel size must be finite and positive")
    voxel_size = float(voxel_size_m)
    baseline = _voxels(baseline_xyz, voxel_size_m=voxel_size, label="baseline")
    recovered = _voxels(recovered_xyz, voxel_size_m=voxel_size, label="recovered")
    target = _voxels(target_xyz, voxel_size_m=voxel_size, label="target")
    candidates = _voxels(
        historical_candidate_xyz,
        voxel_size_m=voxel_size,
        label="historical candidates",
    )
    opportunity = (candidates & target) - baseline
    recovered_new = recovered - baseline
    recovered_target = recovered_new & target
    recoverable = recovered_new & opportunity
    precision = _ratio(len(recovered_target), len(recovered_new))
    recall = _ratio(len(recoverable), len(opportunity))
    f_score = (
        None
        if precision is None or recall is None
        else 0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "voxel_size_m": voxel_size,
        "opportunity_voxel_count": len(opportunity),
        "recoverable_voxel_count": len(recoverable),
        "recovered_new_voxel_count": len(recovered_new),
        "recovered_target_voxel_count": len(recovered_target),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def _total_surface_metrics(
    prediction: set[tuple[int, int, int]],
    target: set[tuple[int, int, int]],
    known_negative: set[tuple[int, int, int]],
) -> dict[str, object]:
    true_positive = len(prediction & target)
    false_positive = len(prediction & known_negative)
    false_negative = len(target - prediction)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f_score = (
        None
        if precision is None or recall is None
        else 0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "true_positive_voxel_count": true_positive,
        "false_positive_voxel_count": false_positive,
        "false_negative_voxel_count": false_negative,
        "unevaluable_prediction_voxel_count": len(
            prediction - target - known_negative
        ),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def evaluate_completion_surface_v2(
    *,
    baseline_xyz: object,
    recovered_xyz: object,
    target_xyz: object,
    historical_candidate_xyz: object,
    opportunity_baseline_xyz: object | None = None,
    known_negative_xyz: object,
    voxel_size_m: float = 0.05,
) -> dict[str, object]:
    """Evaluate completion with explicit positive, negative, and unknown domains."""

    if (
        isinstance(voxel_size_m, bool)
        or not isinstance(voxel_size_m, (int, float))
        or not math.isfinite(float(voxel_size_m))
        or float(voxel_size_m) <= 0.0
    ):
        raise OwnershipCompletionError("voxel size must be finite and positive")
    voxel_size = float(voxel_size_m)
    baseline = _voxels(baseline_xyz, voxel_size_m=voxel_size, label="baseline")
    recovered = _voxels(recovered_xyz, voxel_size_m=voxel_size, label="recovered")
    target = _voxels(target_xyz, voxel_size_m=voxel_size, label="target")
    candidates = _voxels(
        historical_candidate_xyz,
        voxel_size_m=voxel_size,
        label="historical candidates",
    )
    opportunity_baseline = (
        baseline
        if opportunity_baseline_xyz is None
        else _voxels(
            opportunity_baseline_xyz,
            voxel_size_m=voxel_size,
            label="opportunity baseline",
        )
    )
    known_negative = _voxels(
        known_negative_xyz,
        voxel_size_m=voxel_size,
        label="known negative surface",
    )
    if target & known_negative:
        raise OwnershipCompletionError("positive and known-negative domains overlap")
    opportunity = (candidates & target) - opportunity_baseline
    new_surface = recovered - baseline
    deleted_baseline = baseline - recovered
    return {
        "schema_version": 2,
        "status": "PASS_NULL_PRESERVING",
        "voxel_size_m": voxel_size,
        "target_voxel_count": len(target),
        "known_negative_voxel_count": len(known_negative),
        "opportunity_voxel_count": len(opportunity),
        "recovered_opportunity_voxel_count": len(new_surface & opportunity),
        "historical_unevaluable_voxel_count": len(
            candidates - target - known_negative
        ),
        "new_surface_voxel_count": len(new_surface),
        "new_correct_surface_voxel_count": len(new_surface & target),
        "new_wrong_surface_voxel_count": len(new_surface & known_negative),
        "new_unevaluable_surface_voxel_count": len(
            new_surface - target - known_negative
        ),
        "deleted_baseline_voxel_count": len(deleted_baseline),
        "deleted_correct_baseline_voxel_count": len(deleted_baseline & target),
        "baseline_total": _total_surface_metrics(baseline, target, known_negative),
        "recovered_total": _total_surface_metrics(recovered, target, known_negative),
    }


__all__ = [
    "OwnershipCompletionError",
    "build_dense_ownership_readout",
    "build_ownership_readout",
    "evaluate_completion_surface",
    "evaluate_completion_surface_v2",
]
