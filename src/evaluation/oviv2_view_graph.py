from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from src.oviv2.addressing import VoxelKey
from src.oviv2.observations import FrameObservation, ObservationKind


def _integer(value: object, name: str, *, positive: bool) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0 if positive else normalized < 0:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _bounded_real(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not lower <= normalized <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return normalized


def _finite_cosine(left: np.ndarray | tuple[float, float, float], right: np.ndarray | tuple[float, float, float], name: str) -> float:
    cosine = float(np.dot(left, right))
    if not np.isfinite(cosine):
        raise ValueError(f"{name} must be finite")
    if cosine > 1.0 and not np.isclose(cosine, 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name} lies outside [-1, 1]")
    if cosine < -1.0 and not np.isclose(cosine, -1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name} lies outside [-1, 1]")
    return float(np.clip(cosine, -1.0, 1.0))


@dataclass(frozen=True)
class ViewGraphConfig:
    minimum_overlap_voxels: int = 4
    require_semantic_agreement: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_overlap_voxels",
            _integer(self.minimum_overlap_voxels, "minimum_overlap_voxels", positive=True),
        )
        if not isinstance(self.require_semantic_agreement, (bool, np.bool_)):
            raise TypeError("require_semantic_agreement must be a boolean")
        object.__setattr__(self, "require_semantic_agreement", bool(self.require_semantic_agreement))


@dataclass(frozen=True)
class ObservationEdge:
    left_id: int
    right_id: int
    shared_voxels: int
    voxel_iou: float
    left_coverage: float
    right_coverage: float
    feature_cosine: float | None
    view_direction_cosine: float | None

    def __post_init__(self) -> None:
        left_id = _integer(self.left_id, "left_id", positive=False)
        right_id = _integer(self.right_id, "right_id", positive=False)
        if left_id >= right_id:
            raise ValueError("left_id must be less than right_id")
        object.__setattr__(self, "left_id", left_id)
        object.__setattr__(self, "right_id", right_id)
        object.__setattr__(
            self,
            "shared_voxels",
            _integer(self.shared_voxels, "shared_voxels", positive=True),
        )
        for field_name in ("voxel_iou", "left_coverage", "right_coverage"):
            object.__setattr__(
                self,
                field_name,
                _bounded_real(getattr(self, field_name), field_name, 0.0, 1.0),
            )
        for field_name in ("feature_cosine", "view_direction_cosine"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _bounded_real(value, field_name, -1.0, 1.0),
                )


def _shared_voxel_count(left: FrameObservation, right: FrameObservation) -> int:
    return len(left.voxel_keys & right.voxel_keys)


def build_sparse_observation_edges(
    observations: Sequence[FrameObservation],
    config: ViewGraphConfig,
) -> tuple[ObservationEdge, ...]:
    if not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence of FrameObservation values")
    if not isinstance(config, ViewGraphConfig):
        raise TypeError("config must be a ViewGraphConfig")

    by_id: dict[int, FrameObservation] = {}
    for observation in observations:
        if not isinstance(observation, FrameObservation):
            raise TypeError("observations must contain only FrameObservation values")
        if observation.observation_id in by_id:
            raise ValueError(f"duplicate observation_id {observation.observation_id}")
        by_id[observation.observation_id] = observation

    threshold = config.minimum_overlap_voxels
    eligible = tuple(
        sorted(
            (
                observation
                for observation in by_id.values()
                if observation.kind is ObservationKind.OBJECT
                and observation.semantic_id > 0
                and len(observation.voxel_keys) >= threshold
            ),
            key=lambda observation: observation.observation_id,
        )
    )
    document_frequency: dict[VoxelKey, int] = defaultdict(int)
    for observation in eligible:
        for voxel_key in observation.voxel_keys:
            document_frequency[voxel_key] += 1
    token_rank = {
        voxel_key: rank
        for rank, voxel_key in enumerate(
            sorted(document_frequency, key=lambda key: (document_frequency[key], key))
        )
    }

    postings: dict[VoxelKey, list[int]] = defaultdict(list)
    edges: list[ObservationEdge] = []
    for right in eligible:
        ordered_tokens = sorted(right.voxel_keys, key=token_rank.__getitem__)
        prefix_tokens = ordered_tokens[: len(ordered_tokens) - threshold + 1]
        candidate_ids: set[int] = set()
        for voxel_key in prefix_tokens:
            candidate_ids.update(postings[voxel_key])

        for left_id in sorted(candidate_ids):
            left = by_id[left_id]
            if left.frame_id == right.frame_id:
                continue
            if config.require_semantic_agreement and left.semantic_id != right.semantic_id:
                continue
            shared_voxels = _shared_voxel_count(left, right)
            if shared_voxels < threshold:
                continue
            right_id = right.observation_id
            left_size = len(left.voxel_keys)
            right_size = len(right.voxel_keys)
            union_size = left_size + right_size - shared_voxels
            feature_cosine: float | None = None
            if (
                left.image_feature is not None
                and right.image_feature is not None
                and left.feature_model_id == right.feature_model_id
            ):
                if left.image_feature.shape != right.image_feature.shape:
                    raise ValueError("matching feature_model_id values require equal image feature dimensions")
                feature_cosine = _finite_cosine(
                    left.image_feature,
                    right.image_feature,
                    "feature_cosine",
                )
            view_direction_cosine: float | None = None
            if left.view_direction_xyz is not None and right.view_direction_xyz is not None:
                view_direction_cosine = _finite_cosine(
                    left.view_direction_xyz,
                    right.view_direction_xyz,
                    "view_direction_cosine",
                )
            edges.append(
                ObservationEdge(
                    left_id=left_id,
                    right_id=right_id,
                    shared_voxels=shared_voxels,
                    voxel_iou=shared_voxels / union_size,
                    left_coverage=shared_voxels / left_size,
                    right_coverage=shared_voxels / right_size,
                    feature_cosine=feature_cosine,
                    view_direction_cosine=view_direction_cosine,
                )
            )
        for voxel_key in prefix_tokens:
            postings[voxel_key].append(right.observation_id)
    return tuple(sorted(edges, key=lambda edge: (edge.left_id, edge.right_id)))


def select_observation_edges(
    evidence: Sequence[ObservationEdge],
    *,
    minimum_voxel_iou: float,
    minimum_directed_coverage: float,
    minimum_feature_cosine: float,
) -> tuple[ObservationEdge, ...]:
    if not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence of ObservationEdge values")
    minimum_voxel_iou = _bounded_real(
        minimum_voxel_iou,
        "minimum_voxel_iou",
        0.0,
        1.0,
    )
    minimum_directed_coverage = _bounded_real(
        minimum_directed_coverage,
        "minimum_directed_coverage",
        0.0,
        1.0,
    )
    minimum_feature_cosine = _bounded_real(
        minimum_feature_cosine,
        "minimum_feature_cosine",
        -1.0,
        1.0,
    )

    seen_pairs: set[tuple[int, int]] = set()
    selected: list[ObservationEdge] = []
    for edge in evidence:
        if not isinstance(edge, ObservationEdge):
            raise TypeError("evidence must contain only ObservationEdge values")
        pair = edge.left_id, edge.right_id
        if pair in seen_pairs:
            raise ValueError(f"duplicate evidence edge pair {pair}")
        seen_pairs.add(pair)
        geometry_passes = (
            edge.voxel_iou >= minimum_voxel_iou
            or (
                edge.left_coverage >= minimum_directed_coverage
                and edge.right_coverage >= minimum_directed_coverage
            )
        )
        feature_passes = (
            edge.feature_cosine is None
            or edge.feature_cosine >= minimum_feature_cosine
        )
        if geometry_passes and feature_passes:
            selected.append(edge)
    return tuple(sorted(selected, key=lambda edge: (edge.left_id, edge.right_id)))
