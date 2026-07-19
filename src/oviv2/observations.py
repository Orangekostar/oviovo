from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import gzip
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.addressing import VoxelKey


class ObservationKind(str, Enum):
    OBJECT = "object"
    STRUCTURE = "structure"
    UNKNOWN = "unknown"


def _basic_label(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@dataclass(frozen=True)
class ReplicaVocabulary:
    classes: tuple[str, ...]
    aliases: Mapping[str, str]

    def __post_init__(self) -> None:
        classes = tuple(_basic_label(value) for value in self.classes)
        if not classes or any(not value for value in classes) or len(set(classes)) != len(classes):
            raise ValueError("classes must contain unique non-empty labels")
        aliases = {
            _basic_label(source): _basic_label(target)
            for source, target in dict(self.aliases).items()
        }
        if any(target not in classes for target in aliases.values()):
            raise ValueError("alias targets must belong to classes")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "aliases", MappingProxyType(aliases))

    def resolve(self, value: str) -> tuple[str, int, ObservationKind]:
        normalized = _basic_label(value)
        normalized = self.aliases.get(normalized, normalized)
        try:
            semantic_id = self.classes.index(normalized) + 1
        except ValueError:
            return normalized, 0, ObservationKind.UNKNOWN
        kind = (
            ObservationKind.STRUCTURE
            if normalized in {"wall", "floor", "ceiling"}
            else ObservationKind.OBJECT
        )
        return normalized, semantic_id, kind


@dataclass(frozen=True)
class FrameObservation:
    observation_id: int
    frame_id: int
    timestamp: float
    kind: ObservationKind
    label: str
    semantic_id: int
    confidence: float
    mask: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    voxel_keys: frozenset[VoxelKey]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.observation_id < 0 or self.frame_id < 0:
            raise ValueError("observation and frame IDs must be non-negative")
        if not isinstance(self.kind, ObservationKind):
            raise TypeError("kind must be an ObservationKind")
        if self.semantic_id < 0:
            raise ValueError("semantic_id must be non-negative")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        mask = np.ascontiguousarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must be two dimensional")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        if not self.voxel_keys:
            raise ValueError("voxel_keys cannot be empty")


class CachedFrontendAdapter:
    """One-way adapter from frozen YOLO+SAM cache records to OVIV2 observations."""

    def __init__(
        self,
        cache_dir: str | Path,
        vocabulary: ReplicaVocabulary,
        *,
        voxel_size_m: float = 0.05,
        pixel_stride: int = 2,
        min_valid_points: int = 10,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if not isinstance(vocabulary, ReplicaVocabulary):
            raise TypeError("vocabulary must be a ReplicaVocabulary")
        self.vocabulary = vocabulary
        self.voxel_size_m = float(voxel_size_m)
        self.pixel_stride = int(pixel_stride)
        self.min_valid_points = int(min_valid_points)
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if self.pixel_stride <= 0 or self.min_valid_points <= 0:
            raise ValueError("pixel_stride and min_valid_points must be positive")

    def _load(self, cache_frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        path = self.cache_dir / f"frame{int(cache_frame_id):06d}.pkl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("frontend cache payload must be a dictionary")
        try:
            masks = np.asarray(payload["mask"], dtype=bool)
            boxes = np.asarray(payload["xyxy"], dtype=np.float32)
            confidences = np.asarray(payload["confidence"], dtype=np.float32)
            class_ids = np.asarray(payload["class_id"], dtype=np.int64)
            classes = [str(value) for value in payload["classes"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frontend cache payload: {exc}") from exc
        count = masks.shape[0] if masks.ndim == 3 else -1
        if masks.ndim != 3 or boxes.shape != (count, 4):
            raise ValueError("frontend masks or boxes have invalid shapes")
        if confidences.shape != (count,) or class_ids.shape != (count,):
            raise ValueError("frontend cache vector lengths do not match masks")
        if count and (class_ids.min() < 0 or class_ids.max() >= len(classes)):
            raise ValueError("frontend class_id lies outside classes")
        return masks, boxes, confidences, [classes[index] for index in class_ids]

    def observe(self, frame: Frame, cache_frame_id: int) -> tuple[FrameObservation, ...]:
        masks, boxes, confidences, labels = self._load(cache_frame_id)
        if masks.shape[1:] != frame.depth.shape:
            raise ValueError(
                f"frontend mask shape {masks.shape[1:]} does not match frame depth {frame.depth.shape}"
            )
        observations: list[FrameObservation] = []
        for index, (mask, box, confidence, source_label) in enumerate(
            zip(masks, boxes, confidences, labels)
        ):
            lifted = self._lift_voxels(frame, mask)
            if lifted is None:
                continue
            voxel_keys, centroid, bounds_min, bounds_max = lifted
            label, semantic_id, kind = self.vocabulary.resolve(source_label)
            observations.append(
                FrameObservation(
                    observation_id=int(frame.frame_id) * 1_000_000 + index,
                    frame_id=int(frame.frame_id),
                    timestamp=float(frame.timestamp),
                    kind=kind,
                    label=label,
                    semantic_id=semantic_id,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    mask=mask,
                    bbox_xyxy=tuple(float(value) for value in box),
                    voxel_keys=voxel_keys,
                    centroid_xyz=centroid,
                    bounds_min_xyz=bounds_min,
                    bounds_max_xyz=bounds_max,
                )
            )
        return tuple(observations)

    def _lift_voxels(
        self,
        frame: Frame,
        mask: np.ndarray,
    ) -> tuple[
        frozenset[VoxelKey],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None:
        sampled = np.zeros(mask.shape, dtype=bool)
        sampled[:: self.pixel_stride, :: self.pixel_stride] = True
        valid = mask & sampled & np.isfinite(frame.depth) & (frame.depth > 0.0)
        rows, columns = np.nonzero(valid)
        if len(rows) < self.min_valid_points:
            return None
        depth = frame.depth[rows, columns].astype(np.float64)
        intrinsics = frame.intrinsics
        camera = np.column_stack(
            (
                (columns - intrinsics.cx) * depth / intrinsics.fx,
                (rows - intrinsics.cy) * depth / intrinsics.fy,
                depth,
            )
        )
        world = (
            (np.asarray(frame.pose[:3, :3], dtype=np.float64) @ camera.T).T
            + np.asarray(frame.pose[:3, 3], dtype=np.float64)
        )
        integer_keys = np.floor(world / self.voxel_size_m).astype(np.int64)
        voxel_keys = frozenset(tuple(int(value) for value in row) for row in integer_keys)
        if not voxel_keys:
            return None
        return (
            voxel_keys,
            tuple(float(value) for value in world.mean(axis=0)),
            tuple(float(value) for value in world.min(axis=0)),
            tuple(float(value) for value in world.max(axis=0)),
        )
