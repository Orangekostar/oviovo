from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy import ndimage

from src.core.data_structures import Frame
from src.oviv2.observations import (
    FrameObservation,
    ObservationKind,
    ReplicaVocabulary,
    lift_mask_to_voxels,
)


STRUCTURE_LABELS = ("wall", "floor", "ceiling")


@dataclass(frozen=True)
class DepthStructureConfig:
    enabled: bool = True
    voxel_size_m: float = 0.05
    pixel_stride: int = 4
    min_valid_points: int = 10
    horizontal_threshold: float = 0.6
    wall_vertical_threshold: float = 0.5
    min_component_pixels: int = 500
    min_component_fraction: float = 0.01
    max_components_per_class: int = 5
    object_exclusion_dilation: int = 3
    wall_confidence: float = 0.75
    floor_confidence: float = 0.85
    ceiling_confidence: float = 0.80

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if self.pixel_stride <= 0 or self.min_valid_points <= 0:
            raise ValueError("pixel_stride and min_valid_points must be positive")
        if not 0.0 < self.horizontal_threshold <= 1.0:
            raise ValueError("horizontal_threshold must lie in (0, 1]")
        if not 0.0 <= self.wall_vertical_threshold < self.horizontal_threshold:
            raise ValueError("wall_vertical_threshold must lie in [0, horizontal_threshold)")
        if self.min_component_pixels < 0 or not 0.0 <= self.min_component_fraction <= 1.0:
            raise ValueError("component thresholds must be non-negative")
        if self.max_components_per_class <= 0 or self.object_exclusion_dilation < 0:
            raise ValueError("component count must be positive and dilation non-negative")
        confidences = (self.wall_confidence, self.floor_confidence, self.ceiling_confidence)
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in confidences):
            raise ValueError("structure confidences must lie in [0, 1]")


def classify_structure_masks(
    normal_y: np.ndarray,
    valid: np.ndarray,
    *,
    cy: float,
    horizontal_threshold: float,
    wall_vertical_threshold: float,
) -> dict[str, np.ndarray]:
    normal_y = np.asarray(normal_y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if normal_y.ndim != 2 or normal_y.shape != valid.shape:
        raise ValueError("normal_y and valid must be equally shaped 2D arrays")
    rows = np.arange(normal_y.shape[0], dtype=np.float64)[:, None]
    finite = valid & np.isfinite(normal_y)
    return {
        "wall": finite & (np.abs(normal_y) < wall_vertical_threshold),
        "floor": finite & (rows > cy) & (normal_y > horizontal_threshold),
        "ceiling": finite & (rows < cy) & (normal_y < -horizontal_threshold),
    }


def _camera_normal_y(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame.depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("frame depth must be two dimensional")
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    intrinsics = frame.intrinsics
    points = np.stack(
        (
            (columns - intrinsics.cx) * depth / intrinsics.fx,
            (rows - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        ),
        axis=-1,
    )
    normal_y = np.full(depth.shape, np.nan, dtype=np.float64)
    normal_valid = np.zeros(depth.shape, dtype=bool)
    if min(depth.shape) < 3:
        return normal_y, normal_valid

    horizontal = points[1:-1, 2:] - points[1:-1, :-2]
    vertical = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(normals, axis=2)
    neighbor_valid = (
        valid_depth[1:-1, 1:-1]
        & valid_depth[1:-1, 2:]
        & valid_depth[1:-1, :-2]
        & valid_depth[2:, 1:-1]
        & valid_depth[:-2, 1:-1]
        & np.isfinite(lengths)
        & (lengths > 1e-12)
    )
    interior_y = np.full(lengths.shape, np.nan, dtype=np.float64)
    interior_y[neighbor_valid] = normals[..., 1][neighbor_valid] / lengths[neighbor_valid]
    normal_y[1:-1, 1:-1] = interior_y
    normal_valid[1:-1, 1:-1] = neighbor_valid
    return normal_y, normal_valid


class DepthStructureFrontend:
    def __init__(
        self,
        vocabulary: ReplicaVocabulary,
        config: DepthStructureConfig = DepthStructureConfig(),
    ) -> None:
        if not isinstance(vocabulary, ReplicaVocabulary):
            raise TypeError("vocabulary must be a ReplicaVocabulary")
        if not isinstance(config, DepthStructureConfig):
            raise TypeError("config must be a DepthStructureConfig")
        resolved = {label: vocabulary.resolve(label) for label in STRUCTURE_LABELS}
        missing = [label for label, (_, semantic_id, _) in resolved.items() if semantic_id == 0]
        if missing:
            raise ValueError(f"Replica vocabulary is missing structure labels: {', '.join(missing)}")
        self.vocabulary = vocabulary
        self.config = config
        self._resolved = resolved

    def observe(
        self,
        frame: Frame,
        object_observations: Sequence[FrameObservation],
    ) -> tuple[FrameObservation, ...]:
        if not self.config.enabled:
            return ()
        normal_y, valid = _camera_normal_y(frame)
        masks = classify_structure_masks(
            normal_y,
            valid,
            cy=float(frame.intrinsics.cy),
            horizontal_threshold=self.config.horizontal_threshold,
            wall_vertical_threshold=self.config.wall_vertical_threshold,
        )
        excluded = np.zeros(frame.depth.shape, dtype=bool)
        for observation in object_observations:
            if observation.mask.shape != frame.depth.shape:
                raise ValueError("object observation mask does not match frame depth")
            excluded |= observation.mask
        if self.config.object_exclusion_dilation and excluded.any():
            excluded = ndimage.binary_dilation(
                excluded,
                structure=np.ones((3, 3), dtype=bool),
                iterations=self.config.object_exclusion_dilation,
            )

        confidences = {
            "wall": self.config.wall_confidence,
            "floor": self.config.floor_confidence,
            "ceiling": self.config.ceiling_confidence,
        }
        min_pixels = max(
            self.config.min_component_pixels,
            int(math.ceil(self.config.min_component_fraction * frame.depth.size)),
        )
        observations: list[FrameObservation] = []
        for label_index, label in enumerate(STRUCTURE_LABELS):
            components, count = ndimage.label(masks[label] & ~excluded)
            ranked = sorted(
                (
                    (int((components == component_id).sum()), component_id)
                    for component_id in range(1, count + 1)
                ),
                key=lambda item: (-item[0], item[1]),
            )
            for component_index, (pixel_count, component_id) in enumerate(
                ranked[: self.config.max_components_per_class]
            ):
                if pixel_count < min_pixels:
                    continue
                mask = components == component_id
                lifted = lift_mask_to_voxels(
                    frame,
                    mask,
                    voxel_size_m=self.config.voxel_size_m,
                    pixel_stride=self.config.pixel_stride,
                    min_valid_points=self.config.min_valid_points,
                )
                if lifted is None:
                    continue
                voxel_keys, centroid, bounds_min, bounds_max = lifted
                rows, columns = np.nonzero(mask)
                resolved_label, semantic_id, kind = self._resolved[label]
                observations.append(
                    FrameObservation(
                        observation_id=(
                            int(frame.frame_id) * 1_000_000
                            + 900_000
                            + label_index * 100
                            + component_index
                        ),
                        frame_id=int(frame.frame_id),
                        timestamp=float(frame.timestamp),
                        kind=kind,
                        label=resolved_label,
                        semantic_id=semantic_id,
                        confidence=confidences[label],
                        mask=mask,
                        bbox_xyxy=(
                            float(columns.min()),
                            float(rows.min()),
                            float(columns.max() + 1),
                            float(rows.max() + 1),
                        ),
                        voxel_keys=voxel_keys,
                        centroid_xyz=centroid,
                        bounds_min_xyz=bounds_min,
                        bounds_max_xyz=bounds_max,
                    )
                )
        return tuple(observations)
