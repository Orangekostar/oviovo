from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np

from src.oviv2.observations import FrameObservation, ObservationKind


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class TemporalObservationMergeConfig:
    same_semantic_iou_threshold: float = 0.5
    supplement_containment_threshold: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "same_semantic_iou_threshold",
            _unit_interval(
                self.same_semantic_iou_threshold,
                "same_semantic_iou_threshold",
            ),
        )
        object.__setattr__(
            self,
            "supplement_containment_threshold",
            _unit_interval(
                self.supplement_containment_threshold,
                "supplement_containment_threshold",
            ),
        )


_DEFAULT_MERGE_CONFIG = TemporalObservationMergeConfig()


def regularize_temporal_object_extents(
    observations: tuple[FrameObservation, ...],
    voxel_size_m: float,
) -> tuple[FrameObservation, ...]:
    """Give degenerate temporal object point clouds their occupied-cell bounds."""
    if type(observations) is not tuple:
        raise TypeError("observations must be an exact tuple")
    if isinstance(voxel_size_m, (bool, np.bool_)) or not isinstance(
        voxel_size_m, Real
    ):
        raise TypeError("voxel_size_m must be numeric")
    voxel_size = float(voxel_size_m)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")

    result: list[FrameObservation] = []
    for observation in observations:
        if not isinstance(observation, FrameObservation):
            raise TypeError("observations must contain FrameObservation values")
        if observation.kind is not ObservationKind.OBJECT:
            result.append(observation)
            continue
        lower = np.asarray(observation.bounds_min_xyz, dtype=np.float64)
        upper = np.asarray(observation.bounds_max_xyz, dtype=np.float64)
        if not (
            lower.shape == (3,)
            and upper.shape == (3,)
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
            and np.all(upper >= lower)
        ):
            raise ValueError("object bounds must be finite and ordered")
        if np.all(upper - lower > 0.0):
            result.append(observation)
            continue

        keys = np.asarray(tuple(observation.voxel_keys), dtype=np.float64)
        cell_lower = keys.min(axis=0) * voxel_size
        cell_upper = (keys.max(axis=0) + 1.0) * voxel_size
        if np.isfinite(cell_lower).all() and np.isfinite(cell_upper).all():
            collapsed = cell_upper <= cell_lower
            if np.any(collapsed):
                centroid = np.asarray(observation.centroid_xyz, dtype=np.float64)
                if centroid.shape != (3,) or not np.isfinite(centroid).all():
                    raise ValueError("object centroid must be a finite 3D point")
                fallback_lower = np.minimum(
                    np.minimum(cell_lower, cell_upper), centroid
                )
                fallback_upper = np.maximum(
                    np.maximum(cell_lower, cell_upper), centroid
                )
                still_collapsed = fallback_upper <= fallback_lower
                fallback_upper[still_collapsed] = np.nextafter(
                    fallback_lower[still_collapsed], np.inf
                )
                cell_lower = np.where(collapsed, fallback_lower, cell_lower)
                cell_upper = np.where(collapsed, fallback_upper, cell_upper)
        if not (
            np.isfinite(cell_lower).all()
            and np.isfinite(cell_upper).all()
            and np.all(cell_upper > cell_lower)
        ):
            raise ValueError("occupied voxel cells must define finite positive bounds")
        regularized = replace(
            observation,
            bounds_min_xyz=tuple(float(value) for value in cell_lower),
            bounds_max_xyz=tuple(float(value) for value in cell_upper),
        )
        for field_name in (
            "mask",
            "voxel_keys",
            "image_feature",
            "text_feature",
            "view_direction_xyz",
        ):
            object.__setattr__(regularized, field_name, getattr(observation, field_name))
        result.append(regularized)
    return tuple(result)


def _validated_groups(
    primary: object,
    supplements: object,
) -> tuple[tuple[FrameObservation, ...], ...]:
    if type(primary) is not tuple:
        raise TypeError("primary must be an exact tuple")
    if type(supplements) is not tuple:
        raise TypeError("supplements must be an exact tuple")
    if any(type(group) is not tuple for group in supplements):
        raise TypeError("supplements must contain exact tuples")

    groups = (primary, *supplements)
    observations = tuple(item for group in groups for item in group)
    if any(not isinstance(item, FrameObservation) for item in observations):
        raise TypeError("inputs must contain FrameObservation values")
    if not observations:
        return groups

    reference = observations[0]
    observation_ids: set[int] = set()
    for observation in observations:
        if observation.observation_id in observation_ids:
            raise ValueError("observation_id values must be globally unique")
        observation_ids.add(observation.observation_id)
        if observation.frame_id != reference.frame_id:
            raise ValueError("observations must share frame_id")
        if observation.timestamp != reference.timestamp:
            raise ValueError("observations must share timestamp")
        if observation.mask.shape != reference.mask.shape:
            raise ValueError("observations must share mask shape")
    return groups


@dataclass(frozen=True)
class _MatchMask:
    carrier_id: int
    observation: FrameObservation
    area: int


def _matching_carrier_ids(
    candidate: FrameObservation,
    higher_priority: tuple[_MatchMask, ...],
    config: TemporalObservationMergeConfig,
) -> tuple[int, ...]:
    if candidate.kind is not ObservationKind.OBJECT or candidate.semantic_id <= 0:
        return ()
    candidate_area = int(np.count_nonzero(candidate.mask))
    if candidate_area == 0:
        return ()

    carrier_ids: set[int] = set()
    for record in higher_priority:
        retained = record.observation
        if (
            retained.kind is not ObservationKind.OBJECT
            or retained.semantic_id <= 0
            or retained.semantic_id != candidate.semantic_id
        ):
            continue
        intersection = int(np.count_nonzero(candidate.mask & retained.mask))
        union = candidate_area + record.area - intersection
        iou = intersection / union if union else 0.0
        candidate_containment = intersection / candidate_area
        retained_containment = intersection / record.area if record.area else 0.0
        if (
            iou >= config.same_semantic_iou_threshold
            or candidate_containment >= config.supplement_containment_threshold
            or retained_containment >= config.supplement_containment_threshold
        ):
            carrier_ids.add(record.carrier_id)
            if len(carrier_ids) > 1:
                break
    return tuple(sorted(carrier_ids))


def _border_contact_fraction(mask: np.ndarray, area: int) -> float:
    if area == 0:
        return 0.0
    height, width = mask.shape
    border_pixels = int(np.count_nonzero(mask[0]))
    if height > 1:
        border_pixels += int(np.count_nonzero(mask[-1]))
    if height > 2:
        border_pixels += int(np.count_nonzero(mask[1:-1, 0]))
        if width > 1:
            border_pixels += int(np.count_nonzero(mask[1:-1, -1]))
    return border_pixels / area


def _coalesce_geometry(
    carrier: FrameObservation,
    supplement: FrameObservation,
) -> FrameObservation:
    mask = carrier.mask | supplement.mask
    rows, columns = np.nonzero(mask)
    area = len(rows)
    if area == 0:
        raise RuntimeError("coalesced observation mask cannot be empty")
    merged = replace(
        carrier,
        mask=mask,
        bbox_xyxy=(
            float(columns.min()),
            float(rows.min()),
            float(columns.max() + 1),
            float(rows.max() + 1),
        ),
        voxel_keys=carrier.voxel_keys | supplement.voxel_keys,
        bounds_min_xyz=tuple(
            min(left, right)
            for left, right in zip(
                carrier.bounds_min_xyz,
                supplement.bounds_min_xyz,
            )
        ),
        bounds_max_xyz=tuple(
            max(left, right)
            for left, right in zip(
                carrier.bounds_max_xyz,
                supplement.bounds_max_xyz,
            )
        ),
        visible_pixel_count=area,
        border_contact_fraction=_border_contact_fraction(mask, area),
    )
    for field_name in (
        "image_feature",
        "text_feature",
        "view_direction_xyz",
    ):
        object.__setattr__(merged, field_name, getattr(carrier, field_name))
    return merged


def merge_temporal_object_observations(
    primary: tuple[FrameObservation, ...],
    supplements: tuple[tuple[FrameObservation, ...], ...],
    config: TemporalObservationMergeConfig = _DEFAULT_MERGE_CONFIG,
) -> tuple[FrameObservation, ...]:
    if not isinstance(config, TemporalObservationMergeConfig):
        raise TypeError("config must be a TemporalObservationMergeConfig")
    groups = _validated_groups(primary, supplements)
    if not groups:
        return ()

    merged = list(groups[0])
    merged_index = {
        observation.observation_id: index
        for index, observation in enumerate(merged)
    }
    higher_priority = tuple(
        _MatchMask(
            observation.observation_id,
            observation,
            int(np.count_nonzero(observation.mask)),
        )
        for observation in groups[0]
    )
    for group in groups[1:]:
        staged_matches: list[_MatchMask] = []
        for candidate in group:
            carrier_ids = _matching_carrier_ids(
                candidate,
                higher_priority,
                config,
            )
            if not carrier_ids:
                merged_index[candidate.observation_id] = len(merged)
                merged.append(candidate)
                staged_matches.append(
                    _MatchMask(
                        candidate.observation_id,
                        candidate,
                        int(np.count_nonzero(candidate.mask)),
                    )
                )
                continue
            if len(carrier_ids) == 1:
                carrier_id = carrier_ids[0]
                carrier_index = merged_index[carrier_id]
                merged[carrier_index] = _coalesce_geometry(
                    merged[carrier_index],
                    candidate,
                )
                staged_matches.append(
                    _MatchMask(
                        carrier_id,
                        candidate,
                        int(np.count_nonzero(candidate.mask)),
                    )
                )
        higher_priority += tuple(staged_matches)
    return tuple(merged)
