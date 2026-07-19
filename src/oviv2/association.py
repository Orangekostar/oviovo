from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.oviv2.addressing import VoxelKey


def _non_negative_integer(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _unit_interval(value: float, name: str) -> float:
    normalized = _finite_float(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return normalized


def _point3(value: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite coordinates") from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return float(point[0]), float(point[1]), float(point[2])


@dataclass(frozen=True)
class AssociationTarget:
    target_id: int
    voxel_keys: frozenset[VoxelKey]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    semantic_id: int = 0
    semantic_confidence: float = 0.0
    visual_feature: tuple[float, ...] | None = None
    feature_model_id: str | None = None
    free_space_conflict: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _non_negative_integer(self.target_id, "target_id"))
        object.__setattr__(
            self,
            "semantic_id",
            _non_negative_integer(self.semantic_id, "semantic_id"),
        )
        object.__setattr__(
            self,
            "semantic_confidence",
            _unit_interval(self.semantic_confidence, "semantic_confidence"),
        )
        if not isinstance(self.free_space_conflict, bool):
            raise ValueError("free_space_conflict must be a bool")

        centroid = _point3(self.centroid_xyz, "centroid_xyz")
        bounds_min = _point3(self.bounds_min_xyz, "bounds_min_xyz")
        bounds_max = _point3(self.bounds_max_xyz, "bounds_max_xyz")
        if any(lower > upper for lower, upper in zip(bounds_min, bounds_max)):
            raise ValueError("bounds_min_xyz must not exceed bounds_max_xyz")
        object.__setattr__(self, "centroid_xyz", centroid)
        object.__setattr__(self, "bounds_min_xyz", bounds_min)
        object.__setattr__(self, "bounds_max_xyz", bounds_max)

        if not isinstance(self.voxel_keys, frozenset):
            raise ValueError("voxel_keys must be a frozenset")
        normalized_keys: set[VoxelKey] = set()
        for key in self.voxel_keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise ValueError("voxel_keys must contain integer 3-tuples")
            if any(
                not isinstance(component, (int, np.integer))
                or isinstance(component, (bool, np.bool_))
                for component in key
            ):
                raise ValueError("voxel_keys must contain integer 3-tuples")
            normalized_keys.add(tuple(int(component) for component in key))
        object.__setattr__(self, "voxel_keys", frozenset(normalized_keys))

        model_id = self.feature_model_id
        if model_id is not None:
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("feature_model_id must be a non-empty string")
            model_id = model_id.strip()
            object.__setattr__(self, "feature_model_id", model_id)

        if self.visual_feature is None:
            return
        if model_id is None:
            raise ValueError("feature_model_id is required for visual_feature")
        try:
            feature = np.asarray(self.visual_feature, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("visual_feature must be a finite non-zero vector") from exc
        if feature.ndim != 1 or feature.size == 0 or not np.isfinite(feature).all():
            raise ValueError("visual_feature must be a finite non-zero vector")
        scale = float(np.max(np.abs(feature)))
        if scale == 0.0:
            raise ValueError("visual_feature must be a finite non-zero vector")
        scaled = feature / scale
        norm = float(np.linalg.norm(scaled))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("visual_feature must be a finite non-zero vector")
        object.__setattr__(
            self,
            "visual_feature",
            tuple(float(value) for value in scaled / norm),
        )


@dataclass(frozen=True)
class AssociationConfig:
    min_directed_overlap: float = 0.10
    bounds_expansion_m: float = 0.10
    max_centroid_distance_m: float = 0.60
    minimum_score: float = 0.45
    geometry_weight: float = 0.20
    overlap_weight: float = 0.25
    visual_weight: float = 0.30
    semantic_weight: float = 0.20
    temporal_weight: float = 0.05
    semantic_conflict_confidence: float = 0.75
    semantic_conflict_visual_override: float = 0.80

    def __post_init__(self) -> None:
        for name in (
            "min_directed_overlap",
            "minimum_score",
            "semantic_conflict_confidence",
            "semantic_conflict_visual_override",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))

        expansion = _finite_float(self.bounds_expansion_m, "bounds_expansion_m")
        if expansion < 0.0:
            raise ValueError("bounds_expansion_m must be non-negative")
        object.__setattr__(self, "bounds_expansion_m", expansion)

        max_distance = _finite_float(
            self.max_centroid_distance_m,
            "max_centroid_distance_m",
        )
        if max_distance <= 0.0:
            raise ValueError("max_centroid_distance_m must be positive")
        object.__setattr__(self, "max_centroid_distance_m", max_distance)

        for name in (
            "geometry_weight",
            "overlap_weight",
            "visual_weight",
            "semantic_weight",
            "temporal_weight",
        ):
            weight = _finite_float(getattr(self, name), name)
            if weight < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, weight)
        if (
            self.geometry_weight
            + self.overlap_weight
            + self.semantic_weight
            + self.temporal_weight
            <= 0.0
        ):
            raise ValueError("at least one non-visual component weight must be positive")


@dataclass(frozen=True)
class CandidateScore:
    left_id: int
    right_id: int
    score: float
    directed_overlap: float
    bounds_iou: float
    visual_cosine: float | None
    conflict: bool
    accepted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_id", _non_negative_integer(self.left_id, "left_id"))
        object.__setattr__(
            self,
            "right_id",
            _non_negative_integer(self.right_id, "right_id"),
        )
        for name in ("score", "directed_overlap", "bounds_iou"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        if self.visual_cosine is not None:
            visual = _finite_float(self.visual_cosine, "visual_cosine")
            if not -1.0 <= visual <= 1.0:
                raise ValueError("visual_cosine must lie in [-1, 1]")
            object.__setattr__(self, "visual_cosine", visual)
        for name in ("conflict", "accepted"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.conflict and self.accepted:
            raise ValueError("accepted cannot be true for a conflict")


@dataclass(frozen=True)
class Assignment:
    left_id: int
    right_id: int
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_id", _non_negative_integer(self.left_id, "left_id"))
        object.__setattr__(
            self,
            "right_id",
            _non_negative_integer(self.right_id, "right_id"),
        )
        object.__setattr__(self, "score", _unit_interval(self.score, "score"))


def directed_voxel_overlap(
    left: frozenset[VoxelKey],
    right: frozenset[VoxelKey],
) -> float:
    return float(len(left & right) / len(left)) if left else 0.0


def expanded_bounds_iou(
    left: AssociationTarget,
    right: AssociationTarget,
    expansion_m: float,
) -> float:
    if not isinstance(left, AssociationTarget) or not isinstance(right, AssociationTarget):
        raise TypeError("left and right must be AssociationTarget values")
    expansion = _finite_float(expansion_m, "expansion_m")
    if expansion < 0.0:
        raise ValueError("expansion_m must be non-negative")
    left_min = np.asarray(left.bounds_min_xyz) - expansion
    left_max = np.asarray(left.bounds_max_xyz) + expansion
    right_min = np.asarray(right.bounds_min_xyz) - expansion
    right_max = np.asarray(right.bounds_max_xyz) + expansion
    intersection = np.maximum(
        0.0,
        np.minimum(left_max, right_max) - np.maximum(left_min, right_min),
    )
    intersection_volume = float(np.prod(intersection))
    left_volume = float(np.prod(np.maximum(0.0, left_max - left_min)))
    right_volume = float(np.prod(np.maximum(0.0, right_max - right_min)))
    union = left_volume + right_volume - intersection_volume
    if union <= 0.0:
        return 0.0
    return float(np.clip(intersection_volume / union, 0.0, 1.0))


def score_candidate(
    left: AssociationTarget,
    right: AssociationTarget,
    config: AssociationConfig,
) -> CandidateScore | None:
    if not isinstance(left, AssociationTarget) or not isinstance(right, AssociationTarget):
        raise TypeError("left and right must be AssociationTarget values")
    if not isinstance(config, AssociationConfig):
        raise TypeError("config must be an AssociationConfig")
    overlap = max(
        directed_voxel_overlap(left.voxel_keys, right.voxel_keys),
        directed_voxel_overlap(right.voxel_keys, left.voxel_keys),
    )
    bounds_iou = expanded_bounds_iou(left, right, config.bounds_expansion_m)
    if overlap < config.min_directed_overlap and bounds_iou <= 0.0:
        return None

    distance = float(
        np.linalg.norm(np.asarray(left.centroid_xyz) - np.asarray(right.centroid_xyz))
    )
    if distance > config.max_centroid_distance_m and overlap < config.min_directed_overlap:
        return None
    geometry = max(0.0, 1.0 - distance / config.max_centroid_distance_m)

    visual: float | None = None
    if (
        left.visual_feature is not None
        and right.visual_feature is not None
        and left.feature_model_id == right.feature_model_id
        and len(left.visual_feature) == len(right.visual_feature)
    ):
        visual = float(
            np.clip(np.dot(left.visual_feature, right.visual_feature), -1.0, 1.0)
        )

    same_semantic = left.semantic_id > 0 and left.semantic_id == right.semantic_id
    if same_semantic:
        semantic = 1.0
    elif left.semantic_id == 0 or right.semantic_id == 0:
        semantic = 0.5
    else:
        semantic = 0.0

    semantic_conflict = (
        left.semantic_id > 0
        and right.semantic_id > 0
        and left.semantic_id != right.semantic_id
        and min(left.semantic_confidence, right.semantic_confidence)
        >= config.semantic_conflict_confidence
        and (visual is None or visual < config.semantic_conflict_visual_override)
    )
    conflict = bool(
        left.free_space_conflict
        or right.free_space_conflict
        or semantic_conflict
    )

    components = [
        (config.geometry_weight, geometry),
        (config.overlap_weight, max(overlap, bounds_iou)),
        (config.semantic_weight, semantic),
        (config.temporal_weight, 1.0),
    ]
    if visual is not None:
        components.append((config.visual_weight, max(0.0, visual)))
    denominator = sum(weight for weight, _ in components)
    score = sum(weight * value for weight, value in components) / denominator
    score = float(np.clip(score, 0.0, 1.0))
    return CandidateScore(
        left.target_id,
        right.target_id,
        score,
        overlap,
        bounds_iou,
        visual,
        conflict,
        bool(not conflict and score >= config.minimum_score),
    )


def solve_assignment(
    left: tuple[AssociationTarget, ...] | list[AssociationTarget],
    right: tuple[AssociationTarget, ...] | list[AssociationTarget],
    config: AssociationConfig,
) -> tuple[Assignment, ...]:
    if not isinstance(config, AssociationConfig):
        raise TypeError("config must be an AssociationConfig")
    left_values = tuple(left)
    right_values = tuple(right)
    if any(not isinstance(value, AssociationTarget) for value in left_values + right_values):
        raise TypeError("assignment inputs must contain AssociationTarget values")
    left_values = tuple(sorted(left_values, key=lambda value: value.target_id))
    right_values = tuple(sorted(right_values, key=lambda value: value.target_id))
    for name, values in (("left", left_values), ("right", right_values)):
        ids = [value.target_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} target IDs must be unique")
    if not left_values or not right_values:
        return ()

    row_count = len(left_values)
    column_count = len(right_values)
    costs = np.full((row_count, column_count), 1e6, dtype=np.float64)
    scores: dict[tuple[int, int], CandidateScore] = {}
    tie_epsilon = 1e-12 / max(1, row_count, column_count) ** 2
    for row, left_item in enumerate(left_values):
        for column, right_item in enumerate(right_values):
            candidate = score_candidate(left_item, right_item, config)
            if candidate is None or not candidate.accepted:
                continue
            scores[(row, column)] = candidate
            rank_penalty = abs(row - column) * tie_epsilon
            costs[row, column] = 1.0 - candidate.score + rank_penalty

    rows, columns = linear_sum_assignment(costs)
    result = [
        Assignment(
            scores[(row, column)].left_id,
            scores[(row, column)].right_id,
            scores[(row, column)].score,
        )
        for row, column in zip(rows, columns)
        if (row, column) in scores
    ]
    return tuple(sorted(result, key=lambda value: (value.left_id, value.right_id)))
