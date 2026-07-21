from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.oviv2_replica import (
    EntityEvaluationInfo,
    ReplicaGroundTruth,
    _average_precision,
    _ground_truth_instance_masks,
)
from src.oviv2.meshing import LabeledMesh


_POPCOUNT_UINT8 = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1)


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_integer(value: Integral, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _read_only_integer_vector(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in {"i", "u"}:
        raise TypeError("vertex_indices must contain integers")
    if source.dtype.kind == "u" and source.size and source.max() > np.iinfo(np.int64).max:
        raise ValueError("vertex_indices must fit signed int64")
    result = np.ascontiguousarray(source, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _read_only_boolean_vector(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.bool_:
        raise TypeError("mask must contain boolean values")
    result = np.ascontiguousarray(source, dtype=np.bool_).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InstanceHeadConfig:
    minimum_component_vertices: int = 200
    child_score_multiplier: float = 2.0
    maximum_projection_distance_m: float = 0.05
    deduplication_iou_threshold: float = 0.95
    view_count_exponent: float = 1.0
    semantic_evidence_exponent: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_component_vertices",
            _positive_integer(
                self.minimum_component_vertices,
                "minimum_component_vertices",
            ),
        )
        for name in (
            "child_score_multiplier",
            "maximum_projection_distance_m",
        ):
            value = _finite_real(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("view_count_exponent", "semantic_evidence_exponent"):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        threshold = _finite_real(
            self.deduplication_iou_threshold,
            "deduplication_iou_threshold",
        )
        if not 0.0 < threshold <= 1.0:
            raise ValueError("deduplication_iou_threshold must lie in (0, 1]")
        object.__setattr__(self, "deduplication_iou_threshold", threshold)


@dataclass(frozen=True)
class InstanceHypothesis:
    hypothesis_id: str
    entity_id: int
    semantic_id: int
    kind: str
    vertex_indices: np.ndarray
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id:
            raise ValueError("hypothesis_id must be a non-empty string")
        object.__setattr__(self, "entity_id", _positive_integer(self.entity_id, "entity_id"))
        object.__setattr__(
            self,
            "semantic_id",
            _positive_integer(self.semantic_id, "semantic_id"),
        )
        if self.kind not in {"parent", "child"}:
            raise ValueError("kind must be parent or child")
        indices = _read_only_integer_vector(self.vertex_indices)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("vertex_indices must be a non-empty vector")
        if np.any(indices < 0):
            raise ValueError("vertex_indices must be non-negative")
        if len(indices) > 1 and np.any(indices[1:] <= indices[:-1]):
            raise ValueError("vertex_indices must be strictly increasing")
        object.__setattr__(self, "vertex_indices", indices)
        score = _finite_real(self.score, "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must lie in [0, 1]")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class ProjectedInstanceHypothesis:
    hypothesis_id: str
    entity_id: int
    semantic_id: int
    kind: str
    score: float
    mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id:
            raise ValueError("hypothesis_id must be a non-empty string")
        object.__setattr__(self, "entity_id", _positive_integer(self.entity_id, "entity_id"))
        object.__setattr__(
            self,
            "semantic_id",
            _positive_integer(self.semantic_id, "semantic_id"),
        )
        if self.kind not in {"parent", "child"}:
            raise ValueError("kind must be parent or child")
        score = _finite_real(self.score, "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must lie in [0, 1]")
        object.__setattr__(self, "score", score)
        mask = _read_only_boolean_vector(self.mask)
        if mask.ndim != 1:
            raise ValueError("mask must be a vector")
        object.__setattr__(self, "mask", mask)


def _entity_components(mesh: LabeledMesh, entity_id: int) -> list[np.ndarray]:
    entity_vertices = np.flatnonzero(mesh.entity_ids == entity_id)
    neighbors = {int(index): set() for index in entity_vertices}
    for triangle in mesh.triangles:
        if np.all(mesh.entity_ids[triangle] == entity_id):
            a, b, c = (int(value) for value in triangle)
            neighbors[a].update((b, c))
            neighbors[b].update((a, c))
            neighbors[c].update((a, b))

    components: list[np.ndarray] = []
    remaining = set(neighbors)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: list[int] = []
        remaining.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            additions = sorted(neighbors[current] & remaining, reverse=True)
            for neighbor in additions:
                remaining.remove(neighbor)
                stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))

    def geometry_key(indices: np.ndarray) -> tuple[float, float, float, int, tuple[int, ...]]:
        points = mesh.vertices_xyz[indices]
        minimum = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))[0]]
        return (
            float(minimum[0]),
            float(minimum[1]),
            float(minimum[2]),
            -len(indices),
            tuple(int(value) for value in indices),
        )

    return sorted(components, key=geometry_key)


def _parent_score(
    info: EntityEvaluationInfo,
    maximum_view_count: int,
    config: InstanceHeadConfig,
) -> float:
    view_score = info.accepted_view_count / maximum_view_count
    return float(
        np.clip(
            view_score**config.view_count_exponent
            * info.semantic_confidence**config.semantic_evidence_exponent,
            0.0,
            1.0,
        )
    )


def build_instance_hypotheses(
    mesh: LabeledMesh,
    entity_info: Sequence[EntityEvaluationInfo],
    config: InstanceHeadConfig = InstanceHeadConfig(),
) -> tuple[InstanceHypothesis, ...]:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    if not isinstance(config, InstanceHeadConfig):
        raise TypeError("config must be an InstanceHeadConfig")
    info_by_id: dict[int, EntityEvaluationInfo] = {}
    for info in entity_info:
        if not isinstance(info, EntityEvaluationInfo):
            raise TypeError("entity_info must contain EntityEvaluationInfo values")
        if info.entity_id in info_by_id:
            raise ValueError(f"duplicate entity info for {info.entity_id}")
        info_by_id[info.entity_id] = info

    mesh_entity_ids = sorted(int(value) for value in np.unique(mesh.entity_ids) if value > 0)
    missing = [entity_id for entity_id in mesh_entity_ids if entity_id not in info_by_id]
    if missing:
        raise ValueError(f"missing entity info for {missing}")
    maximum_view_count = max(
        (info_by_id[entity_id].accepted_view_count for entity_id in mesh_entity_ids),
        default=1,
    )
    maximum_view_count = max(maximum_view_count, 1)

    hypotheses: list[InstanceHypothesis] = []
    for entity_id in mesh_entity_ids:
        info = info_by_id[entity_id]
        vertex_indices = np.flatnonzero(mesh.entity_ids == entity_id)
        parent_score = _parent_score(info, maximum_view_count, config)
        hypotheses.append(
            InstanceHypothesis(
                f"e{entity_id}:parent",
                entity_id,
                info.semantic_id,
                "parent",
                vertex_indices,
                parent_score,
            )
        )
        components = _entity_components(mesh, entity_id)
        if len(components) <= 1:
            continue
        for component_index, component in enumerate(components):
            if len(component) < config.minimum_component_vertices:
                continue
            ratio = len(component) / len(vertex_indices)
            hypotheses.append(
                InstanceHypothesis(
                    f"e{entity_id}:component:{component_index:04d}",
                    entity_id,
                    info.semantic_id,
                    "child",
                    component,
                    float(
                        np.clip(
                            parent_score
                            * config.child_score_multiplier
                            * ratio**2,
                            0.0,
                            1.0,
                        )
                    ),
                )
            )
    return tuple(hypotheses)


def project_instance_hypotheses(
    mesh: LabeledMesh,
    hypotheses: Sequence[InstanceHypothesis],
    target_vertices_xyz: np.ndarray,
    maximum_distance_m: float,
) -> tuple[ProjectedInstanceHypothesis, ...]:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    threshold = _finite_real(maximum_distance_m, "maximum_distance_m")
    if threshold <= 0.0:
        raise ValueError("maximum_distance_m must be positive")
    target = np.asarray(target_vertices_xyz, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != 3 or not np.isfinite(target).all():
        raise ValueError("target_vertices_xyz must be a finite (N, 3) array")

    projected: list[ProjectedInstanceHypothesis] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, InstanceHypothesis):
            raise TypeError("hypotheses must contain InstanceHypothesis values")
        if int(hypothesis.vertex_indices[-1]) >= len(mesh.vertices_xyz):
            raise ValueError("hypothesis vertex index lies outside mesh")
        if len(target):
            distances, _ = cKDTree(mesh.vertices_xyz[hypothesis.vertex_indices]).query(
                target,
                k=1,
                workers=-1,
            )
            mask = np.asarray(distances, dtype=np.float64) < threshold
        else:
            mask = np.zeros(0, dtype=bool)
        projected.append(
            ProjectedInstanceHypothesis(
                hypothesis.hypothesis_id,
                hypothesis.entity_id,
                hypothesis.semantic_id,
                hypothesis.kind,
                hypothesis.score,
                mask,
            )
        )
    return tuple(projected)


def deduplicate_projected_hypotheses(
    hypotheses: Iterable[ProjectedInstanceHypothesis],
    iou_threshold: float,
) -> tuple[ProjectedInstanceHypothesis, ...]:
    threshold = _finite_real(iou_threshold, "iou_threshold")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("iou_threshold must lie in (0, 1]")
    ordered = sorted(hypotheses, key=lambda item: (-item.score, item.hypothesis_id))
    kept: list[ProjectedInstanceHypothesis] = []
    kept_packed: list[np.ndarray] = []
    kept_sizes: list[int] = []
    expected_size: int | None = None
    for hypothesis in ordered:
        if not isinstance(hypothesis, ProjectedInstanceHypothesis):
            raise TypeError(
                "hypotheses must contain ProjectedInstanceHypothesis values"
            )
        if expected_size is None:
            expected_size = len(hypothesis.mask)
        elif len(hypothesis.mask) != expected_size:
            raise ValueError("projected masks must have equal length")
        packed = np.packbits(hypothesis.mask)
        hypothesis_size = int(np.sum(hypothesis.mask))
        duplicate = False
        for accepted_packed, accepted_size in zip(
            kept_packed,
            kept_sizes,
            strict=True,
        ):
            intersection = int(
                _POPCOUNT_UINT8[np.bitwise_and(packed, accepted_packed)].sum(
                    dtype=np.int64
                )
            )
            union = hypothesis_size + accepted_size - intersection
            iou = float(intersection / union) if union else 1.0
            if iou >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(hypothesis)
            kept_packed.append(packed)
            kept_sizes.append(hypothesis_size)
    return tuple(kept)


def evaluate_projected_instance_hypotheses(
    projected: Sequence[ProjectedInstanceHypothesis],
    ground_truth: ReplicaGroundTruth,
    *,
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
    deduplication_iou_threshold: float,
) -> dict[str, object]:
    if not isinstance(ground_truth, ReplicaGroundTruth):
        raise TypeError("ground_truth must be ReplicaGroundTruth")
    minimum = _positive_integer(min_instance_vertices, "min_instance_vertices")
    for item in projected:
        if not isinstance(item, ProjectedInstanceHypothesis):
            raise TypeError(
                "projected must contain ProjectedInstanceHypothesis values"
            )
        if len(item.mask) != len(ground_truth.vertices_xyz):
            raise ValueError("projected mask size must match ground-truth vertices")
    supported = tuple(item for item in projected if int(np.sum(item.mask)) >= minimum)
    deduplicated = deduplicate_projected_hypotheses(
        supported,
        deduplication_iou_threshold,
    )
    gt_masks = _ground_truth_instance_masks(
        ground_truth,
        set(instance_semantic_ids),
        minimum,
    )
    gt_count = len(gt_masks)
    gt_indices = np.full(len(ground_truth.vertices_xyz), -1, dtype=np.int64)
    gt_sizes = np.empty(gt_count, dtype=np.int64)
    for index, (_, mask) in enumerate(gt_masks):
        gt_indices[mask] = index
        gt_sizes[index] = int(np.sum(mask))
    ious = np.zeros((len(deduplicated), gt_count), dtype=np.float64)
    for index, hypothesis in enumerate(deduplicated):
        predicted_size = int(np.sum(hypothesis.mask))
        matched_indices = gt_indices[hypothesis.mask]
        matched_indices = matched_indices[matched_indices >= 0]
        intersections = np.bincount(matched_indices, minlength=gt_count)
        unions = predicted_size + gt_sizes - intersections
        np.divide(
            intersections,
            unions,
            out=ious[index],
            where=unions > 0,
        )

    def threshold_metrics(threshold: float) -> tuple[float, float]:
        matched_gt: set[int] = set()
        tp: list[float] = []
        fp: list[float] = []
        for row in ious:
            available = [index for index in range(gt_count) if index not in matched_gt]
            best_index = max(available, key=lambda index: row[index], default=-1)
            is_match = best_index >= 0 and row[best_index] >= threshold
            tp.append(1.0 if is_match else 0.0)
            fp.append(0.0 if is_match else 1.0)
            if is_match:
                matched_gt.add(best_index)
        return (
            _average_precision(np.asarray(tp), np.asarray(fp), gt_count),
            float(len(matched_gt) / gt_count) if gt_count else 0.0,
        )

    ap25, recall25 = threshold_metrics(0.25)
    ap50, recall50 = threshold_metrics(0.50)
    return {
        "ap25": ap25,
        "ap50": ap50,
        "recall25": recall25,
        "recall50": recall50,
        "raw_hypothesis_count": len(projected),
        "supported_hypothesis_count": len(supported),
        "predicted_instance_count": len(deduplicated),
        "ground_truth_instance_count": len(gt_masks),
        "prediction_hypothesis_ids": [
            item.hypothesis_id for item in deduplicated
        ],
        "prediction_confidences": [item.score for item in deduplicated],
    }


def evaluate_instance_hypotheses(
    mesh: LabeledMesh,
    hypotheses: Sequence[InstanceHypothesis],
    ground_truth: ReplicaGroundTruth,
    *,
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
    config: InstanceHeadConfig = InstanceHeadConfig(),
) -> dict[str, object]:
    if not isinstance(ground_truth, ReplicaGroundTruth):
        raise TypeError("ground_truth must be ReplicaGroundTruth")
    projected = project_instance_hypotheses(
        mesh,
        hypotheses,
        ground_truth.vertices_xyz,
        config.maximum_projection_distance_m,
    )
    return evaluate_projected_instance_hypotheses(
        projected,
        ground_truth,
        instance_semantic_ids=instance_semantic_ids,
        min_instance_vertices=min_instance_vertices,
        deduplication_iou_threshold=config.deduplication_iou_threshold,
    )
