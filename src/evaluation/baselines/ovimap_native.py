"""Strict data contracts for a native OVI-MAP reproduction."""

from __future__ import annotations

from typing import Any

import numpy as np


def _integer_map(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a two-dimensional integer array")
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative IDs")
    return array


def compose_instance_id_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Compose released CropFormer masks so higher scores win overlaps."""
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N,H,W] and boolean dtype")
    if masks.dtype != np.bool_:
        raise ValueError("masks must have shape [N,H,W] and boolean dtype")
    if scores.shape != (masks.shape[0],):
        raise ValueError("scores must contain one finite value per mask")
    try:
        finite = np.all(np.isfinite(scores))
    except TypeError as error:
        raise ValueError("scores must contain one finite value per mask") from error
    if not finite:
        raise ValueError("scores must contain one finite value per mask")
    if masks.shape[0] > np.iinfo(np.uint8).max:
        raise ValueError("too many masks for released uint8 instance IDs")

    output = np.zeros(masks.shape[1:], dtype=np.uint8)
    for index in np.argsort(scores, kind="stable"):
        output[masks[index]] = int(index) + 1
    return output


def summarize_instance_id_mask(mask: np.ndarray) -> dict[str, Any]:
    """Validate and summarize a lossless instance-ID image."""
    mask = _integer_map(mask, name="instance mask")
    ids, counts = np.unique(mask, return_counts=True)
    positive_ids = ids[ids > 0]
    if positive_ids.size and positive_ids.tolist() != list(
        range(1, int(positive_ids[-1]) + 1)
    ):
        raise ValueError("positive instance IDs must be contiguous")
    foreground_counts = counts[ids > 0]
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "instance_ids": positive_ids.astype(int).tolist(),
        "instance_count": int(positive_ids.size),
        "foreground_ratio": float(foreground_counts.sum() / mask.size),
        "largest_instance_ratio": (
            float(foreground_counts.max() / mask.size)
            if foreground_counts.size
            else 0.0
        ),
    }


def normalize_raycast_ids(
    value: object,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Normalize a pybind raycast result without color-channel conversion."""
    if height <= 0 or width <= 0:
        raise ValueError("raycast dimensions must be positive")
    array = np.asarray(value)
    if array.ndim == 1 and array.size == height * width:
        array = array.reshape(height, width)
    if array.shape != (height, width):
        raise ValueError("raycast ID map shape mismatch")
    return _integer_map(array, name="raycast ID map")


def bbox_from_instance_map(ids: np.ndarray, instance_id: int) -> list[int]:
    """Return an inclusive XYXY box for one integer instance ID."""
    ids = _integer_map(ids, name="instance map")
    ys, xs = np.nonzero(ids == instance_id)
    if xs.size == 0:
        raise ValueError(f"instance {instance_id} is absent")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def validate_color_round_trip(
    mapper_rgb: tuple[int, int, int],
    logged_rgb: tuple[int, int, int],
) -> None:
    """Require the mapper and its authoritative log to use identical RGB."""
    for color in (mapper_rgb, logged_rgb):
        if len(color) != 3 or any(
            not isinstance(channel, (int, np.integer))
            or channel < 0
            or channel > 255
            for channel in color
        ):
            raise ValueError("instance colors must be RGB uint8 triples")
    if tuple(mapper_rgb) != tuple(logged_rgb):
        raise ValueError("mapper and logged instance colors differ")


def audit_native_frame(
    mask: np.ndarray,
    raycast: np.ndarray,
    color_pairs: list[dict[str, Any]],
    *,
    raycast_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Audit one frame at the CropFormer-to-mapper output boundary."""
    mask = np.asarray(mask)
    mask_summary = summarize_instance_id_mask(mask)
    raycast_height, raycast_width = raycast_shape or mask.shape
    raycast_ids = normalize_raycast_ids(
        raycast,
        height=raycast_height,
        width=raycast_width,
    )
    for pair in color_pairs:
        try:
            mapper_rgb = tuple(pair["mapper_rgb"])
            logged_rgb = tuple(pair["logged_rgb"])
        except (KeyError, TypeError) as error:
            raise ValueError("color pair requires mapper_rgb and logged_rgb") from error
        validate_color_round_trip(mapper_rgb, logged_rgb)

    boxes = [
        bbox_from_instance_map(raycast_ids, int(instance_id))
        for instance_id in np.unique(raycast_ids)
        if instance_id > 0
    ]
    full_frame = [0, 0, raycast_width - 1, raycast_height - 1]
    full_frame_count = sum(box == full_frame for box in boxes)
    if boxes and full_frame_count == len(boxes):
        raise ValueError("all non-empty raycast boxes are full-frame")
    return {
        "status": "PASS",
        "mask": mask_summary,
        "raycast_shape": [raycast_height, raycast_width],
        "raycast_instance_count": len(boxes),
        "full_frame_bbox_count": full_frame_count,
        "boxes": boxes,
    }
