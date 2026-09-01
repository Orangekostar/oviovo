"""Causal composition of immutable OVI-MAP anchors and CROVE temporal state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.contracts import EntityPrediction, MapSnapshot


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _point(value: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _points(value: object, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float32, copy=True)
    if array.ndim != 2 or array.shape[1:] != (3,) or not len(array):
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _embedding(value: object | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.array(value, dtype=np.float32, copy=True)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    if float(np.linalg.norm(array)) == 0.0:
        raise ValueError(f"{name} must not be zero")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StaticAnchorConfig:
    minimum_spatial_iou: float
    maximum_centroid_distance_m: float
    minimum_semantic_cosine: float
    moved_displacement_m: float
    background_voxel_size_m: float

    def __post_init__(self) -> None:
        minimum_iou = _finite(self.minimum_spatial_iou, "minimum_spatial_iou")
        maximum_distance = _finite(
            self.maximum_centroid_distance_m, "maximum_centroid_distance_m"
        )
        minimum_cosine = _finite(
            self.minimum_semantic_cosine, "minimum_semantic_cosine"
        )
        moved = _finite(self.moved_displacement_m, "moved_displacement_m")
        voxel = _finite(self.background_voxel_size_m, "background_voxel_size_m")
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_spatial_iou must be in [0, 1]")
        if maximum_distance <= 0.0:
            raise ValueError("maximum_centroid_distance_m must be positive")
        if not -1.0 <= minimum_cosine <= 1.0:
            raise ValueError("minimum_semantic_cosine must be in [-1, 1]")
        if moved <= 0.0 or voxel <= 0.0:
            raise ValueError("movement and voxel thresholds must be positive")
        object.__setattr__(self, "minimum_spatial_iou", minimum_iou)
        object.__setattr__(self, "maximum_centroid_distance_m", maximum_distance)
        object.__setattr__(self, "minimum_semantic_cosine", minimum_cosine)
        object.__setattr__(self, "moved_displacement_m", moved)
        object.__setattr__(self, "background_voxel_size_m", voxel)


@dataclass(frozen=True)
class PrefixIdentitySample:
    temporal_entity_id: int
    frame_index: int
    centroid_xyz: tuple[float, float, float]
    points_xyz: np.ndarray
    semantic_label: str | None
    semantic_embedding: np.ndarray | None
    geometry_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_entity_id",
            _integer(self.temporal_entity_id, "temporal_entity_id", minimum=1),
        )
        object.__setattr__(
            self, "frame_index", _integer(self.frame_index, "frame_index", minimum=0)
        )
        object.__setattr__(
            self, "centroid_xyz", _point(self.centroid_xyz, "centroid_xyz")
        )
        object.__setattr__(self, "points_xyz", _points(self.points_xyz, "points_xyz"))
        label = self.semantic_label
        if label is not None:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("semantic_label must be None or non-empty")
            label = label.strip()
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(
            self,
            "semantic_embedding",
            _embedding(self.semantic_embedding, "semantic_embedding"),
        )
        object.__setattr__(
            self,
            "geometry_epoch",
            _integer(self.geometry_epoch, "geometry_epoch", minimum=0),
        )


@dataclass(frozen=True)
class AnchorOverlayState:
    cutoff_frame: int
    last_frame_index: int
    bindings: tuple[tuple[str, int], ...]
    initial_geometry_epochs: tuple[tuple[int, int], ...]
    removed_anchor_ids: frozenset[str]

    def __post_init__(self) -> None:
        cutoff = _integer(self.cutoff_frame, "cutoff_frame", minimum=0)
        last = _integer(self.last_frame_index, "last_frame_index", minimum=cutoff)
        bindings = tuple(sorted(self.bindings))
        if bindings != self.bindings:
            raise ValueError("bindings must be sorted")
        if any(
            not isinstance(anchor_id, str)
            or not anchor_id
            or _integer(temporal_id, "binding temporal ID", minimum=1) != temporal_id
            for anchor_id, temporal_id in bindings
        ):
            raise ValueError("bindings are invalid")
        if len({item[0] for item in bindings}) != len(bindings) or len(
            {item[1] for item in bindings}
        ) != len(bindings):
            raise ValueError("bindings must be one-to-one")
        epochs = tuple(sorted(self.initial_geometry_epochs))
        if epochs != self.initial_geometry_epochs:
            raise ValueError("initial_geometry_epochs must be sorted")
        if {item[0] for item in epochs} != {item[1] for item in bindings}:
            raise ValueError("initial epochs must cover every bound temporal identity")
        for temporal_id, epoch in epochs:
            _integer(temporal_id, "initial epoch temporal ID", minimum=1)
            _integer(epoch, "initial geometry epoch", minimum=0)
        if not isinstance(self.removed_anchor_ids, frozenset) or any(
            not isinstance(item, str) or not item for item in self.removed_anchor_ids
        ):
            raise TypeError("removed_anchor_ids must be a frozenset of strings")
        if not self.removed_anchor_ids.issubset({item[0] for item in bindings}):
            raise ValueError("only bound anchors may be removed")
        object.__setattr__(self, "cutoff_frame", cutoff)
        object.__setattr__(self, "last_frame_index", last)


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> set[tuple[int, int, int]]:
    indices = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m).astype(
        np.int64
    )
    return {tuple(int(value) for value in row) for row in indices}


def _spatial_iou(
    left: np.ndarray, right: np.ndarray, voxel_size_m: float
) -> float:
    left_keys = _voxel_keys(left, voxel_size_m)
    right_keys = _voxel_keys(right, voxel_size_m)
    union = left_keys | right_keys
    return 0.0 if not union else len(left_keys & right_keys) / len(union)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _normalized_label(value: str | None) -> str | None:
    return None if value is None else value.casefold().replace("_", "-")


def _pair_score(
    anchor: EntityPrediction,
    sample: PrefixIdentitySample,
    config: StaticAnchorConfig,
) -> float | None:
    anchor_centroid = np.asarray(anchor.points_xyz, dtype=np.float64).mean(axis=0)
    distance = float(
        np.linalg.norm(anchor_centroid - np.asarray(sample.centroid_xyz))
    )
    iou = _spatial_iou(
        anchor.points_xyz, sample.points_xyz, config.background_voxel_size_m
    )
    if (
        iou < config.minimum_spatial_iou
        and distance > config.maximum_centroid_distance_m
    ):
        return None

    anchor_label = _normalized_label(anchor.semantic_label)
    sample_label = _normalized_label(sample.semantic_label)
    labels_equal = (
        anchor_label is not None
        and sample_label is not None
        and anchor_label == sample_label
    )
    cosine = _cosine(anchor.semantic_embedding, sample.semantic_embedding)
    if (
        anchor_label is not None
        and sample_label is not None
        and not labels_equal
        and (cosine is None or cosine < config.minimum_semantic_cosine)
    ):
        return None
    if cosine is not None and cosine < config.minimum_semantic_cosine and not labels_equal:
        return None

    centroid_score = max(
        0.0, 1.0 - distance / config.maximum_centroid_distance_m
    )
    semantic_score = 1.0 if labels_equal else (0.5 if cosine is None else (cosine + 1.0) / 2.0)
    return 0.5 * iou + 0.3 * centroid_score + 0.2 * semantic_score


def bind_anchor_identities(
    anchor: MapSnapshot,
    samples: tuple[PrefixIdentitySample, ...],
    config: StaticAnchorConfig,
    *,
    cutoff_frame: int,
) -> AnchorOverlayState:
    """Bind each static anchor to at most one causal-prefix temporal identity."""

    if not isinstance(anchor, MapSnapshot):
        raise TypeError("anchor must be a MapSnapshot")
    if not isinstance(config, StaticAnchorConfig):
        raise TypeError("config must be StaticAnchorConfig")
    if not isinstance(samples, tuple) or any(
        not isinstance(item, PrefixIdentitySample) for item in samples
    ):
        raise TypeError("samples must be a tuple of PrefixIdentitySample values")
    cutoff = _integer(cutoff_frame, "cutoff_frame", minimum=0)
    if any(item.frame_index > cutoff for item in samples):
        raise ValueError("prefix identity sample occurs after causal cutoff")
    temporal_ids = tuple(item.temporal_entity_id for item in samples)
    if len(set(temporal_ids)) != len(temporal_ids):
        raise ValueError("prefix temporal entity IDs must be unique")
    anchors = tuple(sorted(anchor.entities, key=lambda item: item.entity_id))
    ordered_samples = tuple(
        sorted(samples, key=lambda item: item.temporal_entity_id)
    )
    if not anchors or not ordered_samples:
        return AnchorOverlayState(cutoff, cutoff, (), (), frozenset())

    sentinel = 1_000_000.0
    costs = np.full((len(anchors), len(ordered_samples)), sentinel, dtype=np.float64)
    valid = np.zeros_like(costs, dtype=np.bool_)
    for row, anchor_entity in enumerate(anchors):
        for column, sample in enumerate(ordered_samples):
            score = _pair_score(anchor_entity, sample, config)
            if score is None:
                continue
            valid[row, column] = True
            costs[row, column] = (
                -score + row * 1e-9 + column * 1e-12
            )
    row_indices, column_indices = linear_sum_assignment(costs)
    pairs = tuple(
        sorted(
            (
                anchors[int(row)].entity_id,
                ordered_samples[int(column)].temporal_entity_id,
            )
            for row, column in zip(row_indices, column_indices, strict=True)
            if valid[int(row), int(column)]
        )
    )
    sample_by_id = {
        item.temporal_entity_id: item for item in ordered_samples
    }
    epochs = tuple(
        sorted(
            (temporal_id, sample_by_id[temporal_id].geometry_epoch)
            for _, temporal_id in pairs
        )
    )
    return AnchorOverlayState(
        cutoff_frame=cutoff,
        last_frame_index=cutoff,
        bindings=pairs,
        initial_geometry_epochs=epochs,
        removed_anchor_ids=frozenset(),
    )


__all__ = [
    "AnchorOverlayState",
    "PrefixIdentitySample",
    "StaticAnchorConfig",
    "bind_anchor_identities",
]
