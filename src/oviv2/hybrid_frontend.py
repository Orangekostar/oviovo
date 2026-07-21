from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


_VARIANTS = frozenset({"sam_labeled", "yolo_novel_sam", "quota_nms_ensemble"})
_THRESHOLD_FIELDS = (
    "yolo_match_iou",
    "yolo_match_coverage",
    "novel_iou",
    "duplicate_iou",
    "minimum_area_fraction",
    "maximum_area_fraction",
    "minimum_valid_depth_fraction",
    "minimum_dense_probability",
    "minimum_dense_margin",
    "maximum_structure_probability",
)


def _unit_interval(value: float, field_name: str) -> float:
    message = f"{field_name} must be a finite real number in [0, 1]"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(message)
    return normalized


def _positive_integer(value: int, field_name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{field_name} must be a positive non-bool integer")
    return int(value)


def _non_negative_integer(value: int, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative non-bool integer")
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative non-bool integer") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative non-bool integer")
    return normalized


def _numeric_array(value: np.ndarray, field_name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must have an integer, unsigned, or float dtype")
    return array


def _binary_mask_array(value: np.ndarray, field_name: str) -> np.ndarray:
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a boolean or binary numeric array") from exc
    if mask.dtype.kind == "b":
        return mask
    if mask.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a boolean or binary numeric array")
    try:
        valid = np.isfinite(mask) & ((mask == 0) | (mask == 1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must contain only finite binary values") from exc
    if not np.all(valid):
        raise ValueError(f"{field_name} must contain only finite binary values")
    return mask


def _immutable_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.array(value, dtype=dtype, copy=True, order="C")
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _normalized_features(value: np.ndarray) -> np.ndarray:
    try:
        features = np.array(value, dtype=np.longdouble, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image_features must be a finite numeric array") from exc
    if not np.all(np.isfinite(features)):
        raise ValueError("image_features must contain only finite values")

    scales = np.max(np.abs(features), axis=1)
    if np.any(scales == 0.0):
        raise ValueError("image_features rows must be nonzero")
    scaled = features / scales[:, None]
    squared_norms = np.sum(scaled * scaled, axis=1, dtype=np.longdouble)
    norms = np.sqrt(squared_norms)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("image_features rows must be finite and nonzero")

    normalized = np.ascontiguousarray(scaled / norms[:, None], dtype=np.float32)
    output_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    if (
        not np.all(np.isfinite(normalized))
        or np.any(output_norms == 0.0)
        or not np.allclose(output_norms, 1.0, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("normalized image_features rows must be finite, nonzero, and unit length")
    return _immutable_array(normalized, np.dtype(np.float32))


@dataclass(frozen=True)
class HybridFrontendConfig:
    variant: str
    yolo_match_iou: float = 0.50
    yolo_match_coverage: float = 0.60
    novel_iou: float = 0.80
    duplicate_iou: float = 0.85
    minimum_area_fraction: float = 0.0001
    maximum_area_fraction: float = 0.50
    minimum_valid_depth_fraction: float = 0.50
    minimum_dense_probability: float = 0.25
    minimum_dense_margin: float = 0.05
    maximum_structure_probability: float = 0.50
    maximum_proposals: int = 64
    maximum_per_class: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        for field_name in _THRESHOLD_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _unit_interval(getattr(self, field_name), field_name),
            )
        if self.minimum_area_fraction > self.maximum_area_fraction:
            raise ValueError(
                "minimum_area_fraction must not exceed maximum_area_fraction"
            )
        for field_name in ("maximum_proposals", "maximum_per_class"):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, eq=False)
class FrontendBatch:
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    labels: tuple[str, ...]
    image_features: np.ndarray

    def __post_init__(self) -> None:
        masks = _binary_mask_array(self.masks, "masks")
        boxes = _numeric_array(self.boxes_xyxy, "boxes_xyxy")
        confidences = _numeric_array(self.confidences, "confidences")
        features = _numeric_array(self.image_features, "image_features")

        if masks.ndim != 3:
            raise ValueError("masks must have shape (N, H, W)")
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("boxes_xyxy must have shape (N, 4)")
        if confidences.ndim != 1:
            raise ValueError("confidences must have shape (N,)")
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("image_features must have shape (N, D) with D > 0")
        if not isinstance(self.labels, tuple):
            raise ValueError("labels must be a tuple of non-empty strings")

        batch_size = masks.shape[0]
        if not (
            boxes.shape[0]
            == confidences.shape[0]
            == len(self.labels)
            == features.shape[0]
            == batch_size
        ):
            raise ValueError("frontend batch dimensions must agree")

        if not np.all(np.isfinite(boxes)):
            raise ValueError("boxes_xyxy must contain only finite values")
        if not np.all(np.isfinite(confidences)):
            raise ValueError("confidences must contain only finite values")
        if np.any(confidences < 0.0) or np.any(confidences > 1.0):
            raise ValueError("confidences must lie in [0, 1]")
        if not np.all(np.isfinite(features)):
            raise ValueError("image_features must contain only finite values")

        labels: list[str] = []
        for label in self.labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("labels must contain only non-empty strings")
            labels.append(label.strip())

        try:
            immutable_masks = _immutable_array(masks, np.dtype(np.bool_))
            immutable_boxes = _immutable_array(boxes, np.dtype(np.float32))
            immutable_confidences = _immutable_array(confidences, np.dtype(np.float32))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frontend arrays cannot be converted to required dtypes") from exc
        if not np.all(np.isfinite(immutable_boxes)):
            raise ValueError("boxes_xyxy must remain finite as float32")
        if not np.all(np.isfinite(immutable_confidences)):
            raise ValueError("confidences must remain finite as float32")

        object.__setattr__(self, "masks", immutable_masks)
        object.__setattr__(self, "boxes_xyxy", immutable_boxes)
        object.__setattr__(self, "confidences", immutable_confidences)
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "image_features", _normalized_features(features))


@dataclass(frozen=True)
class MaskOverlap:
    intersection: int
    iou: float
    left_coverage: float
    right_coverage: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intersection",
            _non_negative_integer(self.intersection, "intersection"),
        )
        for field_name in ("iou", "left_coverage", "right_coverage"):
            object.__setattr__(
                self,
                field_name,
                _unit_interval(getattr(self, field_name), field_name),
            )


def mask_overlap(left: np.ndarray, right: np.ndarray) -> MaskOverlap:
    left_mask = _binary_mask_array(left, "left mask")
    right_mask = _binary_mask_array(right, "right mask")
    if left_mask.ndim != 2 or left_mask.shape != right_mask.shape:
        raise ValueError("masks must be two dimensional with equal shape")

    left_mask = np.asarray(left_mask, dtype=bool)
    right_mask = np.asarray(right_mask, dtype=bool)

    intersection = int(np.count_nonzero(left_mask & right_mask))
    left_size = int(np.count_nonzero(left_mask))
    right_size = int(np.count_nonzero(right_mask))
    union = left_size + right_size - intersection
    return MaskOverlap(
        intersection=intersection,
        iou=float(intersection / union) if union else 0.0,
        left_coverage=float(intersection / left_size) if left_size else 0.0,
        right_coverage=float(intersection / right_size) if right_size else 0.0,
    )
