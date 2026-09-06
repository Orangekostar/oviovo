"""Pure ownership readout and shape-bound completion metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np

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


__all__ = [
    "OwnershipCompletionError",
    "build_ownership_readout",
    "evaluate_completion_surface",
]
