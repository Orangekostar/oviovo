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


LiftedMask = tuple[
    frozenset[VoxelKey],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def lift_mask_to_voxels(
    frame: Frame,
    mask: np.ndarray,
    *,
    voxel_size_m: float,
    pixel_stride: int,
    min_valid_points: int,
) -> LiftedMask | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != frame.depth.shape:
        raise ValueError(f"mask shape {mask.shape} does not match frame depth {frame.depth.shape}")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    if pixel_stride <= 0 or min_valid_points <= 0:
        raise ValueError("pixel_stride and min_valid_points must be positive")
    pose = np.asarray(frame.pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("frame pose must have shape (4, 4)")
    if not np.all(np.isfinite(pose)):
        raise ValueError("frame pose must contain only finite values")
    intrinsics = frame.intrinsics
    intrinsic_values = np.asarray(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(intrinsic_values)) or intrinsics.fx == 0.0 or intrinsics.fy == 0.0:
        raise ValueError("frame intrinsics must be finite with nonzero focal lengths")

    sampled = np.zeros(mask.shape, dtype=bool)
    sampled[::pixel_stride, ::pixel_stride] = True
    valid = mask & sampled & np.isfinite(frame.depth) & (frame.depth > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < min_valid_points:
        return None
    depth = frame.depth[rows, columns].astype(np.float64)
    camera = np.column_stack(
        (
            (columns - intrinsics.cx) * depth / intrinsics.fx,
            (rows - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        )
    )
    if not np.all(np.isfinite(camera)):
        raise ValueError("lifted camera geometry must contain only finite values")
    world = (
        (pose[:3, :3] @ camera.T).T
        + pose[:3, 3]
    )
    if not np.all(np.isfinite(world)):
        raise ValueError("lifted world geometry must contain only finite values")
    integer_keys = np.floor(world / voxel_size_m).astype(np.int64)
    voxel_keys = frozenset(tuple(int(value) for value in row) for row in integer_keys)
    if not voxel_keys:
        return None
    return (
        voxel_keys,
        tuple(float(value) for value in world.mean(axis=0)),
        tuple(float(value) for value in world.min(axis=0)),
        tuple(float(value) for value in world.max(axis=0)),
    )


class ObservationKind(str, Enum):
    OBJECT = "object"
    STRUCTURE = "structure"
    UNKNOWN = "unknown"


def _basic_label(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _validated_feature_model_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("feature_model_id must be a string or None")
    normalized = value.strip()
    if not normalized:
        raise ValueError("feature_model_id cannot be blank")
    return normalized


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _normalized_feature(value: np.ndarray, field_name: str) -> np.ndarray:
    feature = np.array(value, dtype=np.float64, copy=True, order="C")
    if feature.ndim != 1:
        raise ValueError(f"{field_name} must be one dimensional")
    if not np.all(np.isfinite(feature)):
        raise ValueError(f"{field_name} must contain only finite values")
    scale = float(np.max(np.abs(feature), initial=0.0))
    if scale == 0.0:
        raise ValueError(f"{field_name} must be nonzero")
    scaled = feature / scale
    feature = np.ascontiguousarray(scaled / np.linalg.norm(scaled), dtype=np.float32)
    if not np.all(np.isfinite(feature)):
        raise ValueError(f"normalized {field_name} must contain only finite values")
    output_norm = float(np.linalg.norm(feature.astype(np.float64)))
    if output_norm == 0.0:
        raise ValueError(f"normalized {field_name} must be nonzero")
    if not np.isclose(output_norm, 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"normalized {field_name} must have unit norm")
    return _immutable_array(feature)


def _normalized_direction(value: tuple[float, float, float]) -> tuple[float, float, float]:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("view_direction_xyz must be three dimensional")
    if not np.all(np.isfinite(direction)):
        raise ValueError("view_direction_xyz must contain only finite values")
    scale = float(np.max(np.abs(direction)))
    if scale == 0.0:
        raise ValueError("view_direction_xyz must be nonzero")
    unit = direction / scale
    unit /= np.linalg.norm(unit)
    return tuple(float(value) for value in unit)


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
    image_feature: np.ndarray | None = None
    text_feature: np.ndarray | None = None
    feature_model_id: str | None = None
    view_direction_xyz: tuple[float, float, float] | None = None
    visible_pixel_count: int = 0
    border_contact_fraction: float = 0.0

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
        object.__setattr__(self, "mask", _immutable_array(mask))
        if not self.voxel_keys:
            raise ValueError("voxel_keys cannot be empty")
        for field_name in ("image_feature", "text_feature"):
            feature = getattr(self, field_name)
            if feature is None:
                continue
            object.__setattr__(self, field_name, _normalized_feature(feature, field_name))
        if (
            self.image_feature is not None
            and self.text_feature is not None
            and self.image_feature.shape != self.text_feature.shape
        ):
            raise ValueError("image and text feature dimensions must match")
        model_id = _validated_feature_model_id(self.feature_model_id)
        if (self.image_feature is not None or self.text_feature is not None) and model_id is None:
            raise ValueError("feature_model_id is required when features are present")
        object.__setattr__(self, "feature_model_id", model_id)
        if self.view_direction_xyz is not None:
            object.__setattr__(
                self,
                "view_direction_xyz",
                _normalized_direction(self.view_direction_xyz),
            )
        if isinstance(self.visible_pixel_count, (bool, np.bool_)) or not isinstance(
            self.visible_pixel_count,
            (int, np.integer),
        ):
            raise TypeError("visible_pixel_count must be an integer")
        if self.visible_pixel_count < 0:
            raise ValueError("visible_pixel_count must be non-negative")
        object.__setattr__(self, "visible_pixel_count", int(self.visible_pixel_count))
        if isinstance(self.border_contact_fraction, (bool, np.bool_)) or not isinstance(
            self.border_contact_fraction,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("border_contact_fraction must be numeric")
        border_contact_fraction = float(self.border_contact_fraction)
        if not np.isfinite(border_contact_fraction):
            raise ValueError("border_contact_fraction must be finite")
        if not 0.0 <= border_contact_fraction <= 1.0:
            raise ValueError("border_contact_fraction must lie in [0, 1]")
        object.__setattr__(self, "border_contact_fraction", border_contact_fraction)


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
        feature_model_id: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if not isinstance(vocabulary, ReplicaVocabulary):
            raise TypeError("vocabulary must be a ReplicaVocabulary")
        self.vocabulary = vocabulary
        self.voxel_size_m = float(voxel_size_m)
        self.pixel_stride = int(pixel_stride)
        self.min_valid_points = int(min_valid_points)
        self.feature_model_id = _validated_feature_model_id(feature_model_id)
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if self.pixel_stride <= 0 or self.min_valid_points <= 0:
            raise ValueError("pixel_stride and min_valid_points must be positive")

    def _load(
        self,
        cache_frame_id: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
        np.ndarray | None,
        np.ndarray | None,
    ]:
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
            image_features = (
                None
                if "image_feats" not in payload
                else np.asarray(payload["image_feats"], dtype=np.float64)
            )
            text_features = (
                None
                if "text_feats" not in payload
                else np.asarray(payload["text_feats"], dtype=np.float64)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frontend cache payload: {exc}") from exc
        count = masks.shape[0] if masks.ndim == 3 else -1
        if masks.ndim != 3 or boxes.shape != (count, 4):
            raise ValueError("frontend masks or boxes have invalid shapes")
        if confidences.shape != (count,) or class_ids.shape != (count,):
            raise ValueError("frontend cache vector lengths do not match masks")
        if count and (class_ids.min() < 0 or class_ids.max() >= len(classes)):
            raise ValueError("frontend class_id lies outside classes")
        for field_name, features in (
            ("image_feats", image_features),
            ("text_feats", text_features),
        ):
            if features is None:
                continue
            if features.ndim != 2:
                raise ValueError(f"frontend {field_name} must be two dimensional")
            if features.shape[0] != count:
                raise ValueError(f"frontend {field_name} row count does not match masks")
        if (
            image_features is not None
            and text_features is not None
            and image_features.shape[1] != text_features.shape[1]
        ):
            raise ValueError("frontend image and text feature dimensions must match")
        for field_name, features in (
            ("image_feats", image_features),
            ("text_feats", text_features),
        ):
            if features is None:
                continue
            for index, feature in enumerate(features):
                _normalized_feature(feature, f"frontend {field_name} row {index}")
        if (image_features is not None or text_features is not None) and self.feature_model_id is None:
            raise ValueError("feature_model_id is required when cached features are present")
        return (
            masks,
            boxes,
            confidences,
            [classes[index] for index in class_ids],
            image_features,
            text_features,
        )

    def observe(self, frame: Frame, cache_frame_id: int) -> tuple[FrameObservation, ...]:
        masks, boxes, confidences, labels, image_features, text_features = self._load(
            cache_frame_id
        )
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
            camera_to_centroid = np.asarray(centroid) - np.asarray(frame.pose[:3, 3])
            view_direction = (
                None
                if not np.all(np.isfinite(camera_to_centroid))
                or not np.any(camera_to_centroid)
                else _normalized_direction(camera_to_centroid)
            )
            visible_pixel_count = int(np.count_nonzero(mask))
            border = np.zeros(mask.shape, dtype=bool)
            border[[0, -1], :] = True
            border[:, [0, -1]] = True
            border_contact_fraction = (
                float(np.count_nonzero(mask & border)) / visible_pixel_count
                if visible_pixel_count
                else 0.0
            )
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
                    image_feature=None if image_features is None else image_features[index],
                    text_feature=None if text_features is None else text_features[index],
                    feature_model_id=self.feature_model_id,
                    view_direction_xyz=view_direction,
                    visible_pixel_count=visible_pixel_count,
                    border_contact_fraction=border_contact_fraction,
                )
            )
        return tuple(observations)

    def _lift_voxels(
        self,
        frame: Frame,
        mask: np.ndarray,
    ) -> LiftedMask | None:
        return lift_mask_to_voxels(
            frame,
            mask,
            voxel_size_m=self.voxel_size_m,
            pixel_stride=self.pixel_stride,
            min_valid_points=self.min_valid_points,
        )
