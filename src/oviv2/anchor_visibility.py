"""Causal visibility gating for unbound immutable map anchors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from numbers import Integral, Real

import numpy as np

from src.core.data_structures import Frame
from src.oviv2.addressing import VoxelKey
from src.oviv2.visibility import (
    VisibilityConfig,
    VisibilityStatus,
    VoxelVisibilityProjector,
)

POLICY_ID = "crove_ovimap_unbound_visibility_v1"
VOXEL_SAMPLING_MODE = "sorted_even_spacing"


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


@dataclass(frozen=True)
class AnchorVisibilityConfig:
    voxel_size_m: float
    depth_tolerance_m: float
    depth_max_m: float
    maximum_voxels_per_anchor: int
    minimum_tested_voxels: int
    minimum_absent_fraction: float
    minimum_present_fraction: float
    minimum_absent_observations: int
    minimum_distinct_viewpoints: int
    minimum_viewpoint_baseline_m: float
    minimum_present_streak: int

    def __post_init__(self) -> None:
        for name in (
            "voxel_size_m",
            "depth_tolerance_m",
            "depth_max_m",
            "minimum_viewpoint_baseline_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, positive=True),
            )
        for name in (
            "maximum_voxels_per_anchor",
            "minimum_tested_voxels",
            "minimum_absent_observations",
            "minimum_distinct_viewpoints",
            "minimum_present_streak",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1),
            )
        for name in ("minimum_absent_fraction", "minimum_present_fraction"):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        if self.minimum_tested_voxels > self.maximum_voxels_per_anchor:
            raise ValueError(
                "minimum_tested_voxels cannot exceed maximum_voxels_per_anchor"
            )
        if self.minimum_absent_observations < self.minimum_distinct_viewpoints:
            raise ValueError(
                "minimum_absent_observations cannot be below distinct viewpoints"
            )


class AnchorVisibilityEvidenceKind(str, Enum):
    PRESENT = "present"
    VISIBLE_ABSENT = "visible_absent"
    OCCLUDED = "occluded"
    DEPTH_UNKNOWN = "depth_unknown"


@dataclass(frozen=True)
class AnchorVisibilityEvidence:
    kind: AnchorVisibilityEvidenceKind
    frame_id: int
    timestamp: float
    tested_voxel_count: int
    present_voxel_count: int
    absent_voxel_count: int
    occluded_voxel_count: int
    camera_position_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AnchorVisibilityEvidenceKind):
            raise TypeError("kind must be AnchorVisibilityEvidenceKind")
        object.__setattr__(
            self, "frame_id", _integer(self.frame_id, "frame_id", minimum=0)
        )
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        for name in (
            "tested_voxel_count",
            "present_voxel_count",
            "absent_voxel_count",
            "occluded_voxel_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=0),
            )
        if self.tested_voxel_count != (
            self.present_voxel_count
            + self.absent_voxel_count
            + self.occluded_voxel_count
        ):
            raise ValueError("tested voxel count must equal classified valid voxels")
        object.__setattr__(
            self,
            "camera_position_xyz",
            _point(self.camera_position_xyz, "camera_position_xyz"),
        )


@dataclass(frozen=True)
class AnchorVisibilityState:
    entity_id: str
    active: bool
    last_frame_id: int
    last_timestamp: float
    absence_observation_count: int
    absence_viewpoints_xyz: tuple[tuple[float, float, float], ...]
    present_streak: int

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        if type(self.active) is not bool:
            raise TypeError("active must be bool")
        object.__setattr__(
            self,
            "last_frame_id",
            _integer(self.last_frame_id, "last_frame_id", minimum=0),
        )
        object.__setattr__(
            self, "last_timestamp", _finite(self.last_timestamp, "last_timestamp")
        )
        object.__setattr__(
            self,
            "absence_observation_count",
            _integer(
                self.absence_observation_count,
                "absence_observation_count",
                minimum=0,
            ),
        )
        if not isinstance(self.absence_viewpoints_xyz, tuple):
            raise TypeError("absence_viewpoints_xyz must be a tuple")
        viewpoints = tuple(
            _point(value, "absence viewpoint")
            for value in self.absence_viewpoints_xyz
        )
        if len(viewpoints) > self.absence_observation_count:
            raise ValueError("viewpoint support cannot exceed absence observations")
        object.__setattr__(self, "absence_viewpoints_xyz", viewpoints)
        object.__setattr__(
            self,
            "present_streak",
            _integer(self.present_streak, "present_streak", minimum=0),
        )


def anchor_visibility_config_from_json(
    payload: Mapping[str, object],
) -> AnchorVisibilityConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("visibility policy must be a mapping")
    config_fields = {item.name for item in fields(AnchorVisibilityConfig)}
    expected = config_fields | {"schema_version", "policy_id", "voxel_sampling"}
    if set(payload) != expected:
        raise ValueError("visibility policy fields are invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("policy_id") != POLICY_ID
        or payload.get("voxel_sampling") != VOXEL_SAMPLING_MODE
    ):
        raise ValueError("visibility policy identity is invalid")
    return AnchorVisibilityConfig(
        **{name: payload[name] for name in config_fields}  # type: ignore[arg-type]
    )


def sample_anchor_voxels(
    points_xyz: object,
    config: AnchorVisibilityConfig,
) -> tuple[VoxelKey, ...]:
    if not isinstance(config, AnchorVisibilityConfig):
        raise TypeError("config must be AnchorVisibilityConfig")
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError("points_xyz must have shape (N, 3) with N > 0")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz must contain only finite values")
    quantized = np.floor(points / config.voxel_size_m).astype(np.int64)
    unique = np.unique(quantized, axis=0)
    if len(unique) > config.maximum_voxels_per_anchor:
        indices = np.linspace(
            0,
            len(unique) - 1,
            config.maximum_voxels_per_anchor,
            dtype=np.int64,
        )
        unique = unique[indices]
    return tuple(tuple(int(item) for item in row) for row in unique)


def classify_anchor_visibility(
    voxel_keys: tuple[VoxelKey, ...],
    frame: Frame,
    config: AnchorVisibilityConfig,
) -> AnchorVisibilityEvidence:
    if not isinstance(config, AnchorVisibilityConfig):
        raise TypeError("config must be AnchorVisibilityConfig")
    if not isinstance(frame, Frame):
        raise TypeError("frame must be Frame")
    if not isinstance(voxel_keys, tuple):
        raise TypeError("voxel_keys must be a tuple")
    depth = np.asarray(frame.depth, dtype=np.float64)
    clean_depth = np.array(depth, copy=True)
    valid_depth = (
        np.isfinite(clean_depth)
        & (clean_depth > 0.0)
        & (clean_depth <= config.depth_max_m)
    )
    clean_depth[~valid_depth] = 0.0
    clean = Frame(
        frame_id=frame.frame_id,
        source_frame_id=frame.source_frame_id,
        rgb=frame.rgb,
        depth=clean_depth,
        pose=frame.pose,
        intrinsics=frame.intrinsics,
        timestamp=frame.timestamp,
    )
    grouped = VoxelVisibilityProjector(
        VisibilityConfig(
            voxel_size_m=config.voxel_size_m,
            depth_tolerance_m=config.depth_tolerance_m,
        )
    ).classify_many(voxel_keys, clean)
    present = len(grouped[VisibilityStatus.PRESENT])
    absent = len(grouped[VisibilityStatus.ABSENT])
    occluded = len(grouped[VisibilityStatus.OCCLUDED])
    tested = present + absent + occluded
    absent_fraction = absent / tested if tested else 0.0
    present_fraction = present / tested if tested else 0.0
    if (
        tested >= config.minimum_tested_voxels
        and absent_fraction >= config.minimum_absent_fraction
        and present == 0
    ):
        kind = AnchorVisibilityEvidenceKind.VISIBLE_ABSENT
    elif (
        tested >= config.minimum_tested_voxels
        and present_fraction >= config.minimum_present_fraction
        and absent == 0
    ):
        kind = AnchorVisibilityEvidenceKind.PRESENT
    elif tested:
        kind = AnchorVisibilityEvidenceKind.OCCLUDED
    else:
        kind = AnchorVisibilityEvidenceKind.DEPTH_UNKNOWN
    return AnchorVisibilityEvidence(
        kind=kind,
        frame_id=frame.frame_id,
        timestamp=float(frame.timestamp),
        tested_voxel_count=tested,
        present_voxel_count=present,
        absent_voxel_count=absent,
        occluded_voxel_count=occluded,
        camera_position_xyz=_point(frame.pose[:3, 3], "frame camera position"),
    )


def advance_anchor_visibility(
    state: AnchorVisibilityState,
    evidence: AnchorVisibilityEvidence,
    config: AnchorVisibilityConfig,
) -> AnchorVisibilityState:
    if not isinstance(state, AnchorVisibilityState):
        raise TypeError("state must be AnchorVisibilityState")
    if not isinstance(evidence, AnchorVisibilityEvidence):
        raise TypeError("evidence must be AnchorVisibilityEvidence")
    if not isinstance(config, AnchorVisibilityConfig):
        raise TypeError("config must be AnchorVisibilityConfig")
    if (
        evidence.frame_id <= state.last_frame_id
        or evidence.timestamp <= state.last_timestamp
    ):
        raise ValueError("visibility evidence must increase strictly")

    active = state.active
    absence_count = state.absence_observation_count
    viewpoints = state.absence_viewpoints_xyz
    present_streak = state.present_streak
    if active and evidence.kind is AnchorVisibilityEvidenceKind.PRESENT:
        absence_count = 0
        viewpoints = ()
        present_streak = 0
    elif active and evidence.kind is AnchorVisibilityEvidenceKind.VISIBLE_ABSENT:
        absence_count += 1
        camera = evidence.camera_position_xyz
        if all(
            float(
                np.linalg.norm(
                    np.asarray(camera, dtype=np.float64)
                    - np.asarray(previous, dtype=np.float64)
                )
            )
            >= config.minimum_viewpoint_baseline_m
            for previous in viewpoints
        ):
            viewpoints = (*viewpoints, camera)
        if (
            absence_count >= config.minimum_absent_observations
            and len(viewpoints) >= config.minimum_distinct_viewpoints
        ):
            active = False
            present_streak = 0
    elif not active and evidence.kind is AnchorVisibilityEvidenceKind.PRESENT:
        present_streak += 1
        if present_streak >= config.minimum_present_streak:
            active = True
            absence_count = 0
            viewpoints = ()
            present_streak = 0
    elif not active:
        present_streak = 0

    return AnchorVisibilityState(
        entity_id=state.entity_id,
        active=active,
        last_frame_id=evidence.frame_id,
        last_timestamp=evidence.timestamp,
        absence_observation_count=absence_count,
        absence_viewpoints_xyz=viewpoints,
        present_streak=present_streak,
    )


__all__ = [
    "POLICY_ID",
    "VOXEL_SAMPLING_MODE",
    "AnchorVisibilityConfig",
    "AnchorVisibilityEvidence",
    "AnchorVisibilityEvidenceKind",
    "AnchorVisibilityState",
    "advance_anchor_visibility",
    "anchor_visibility_config_from_json",
    "classify_anchor_visibility",
    "sample_anchor_voxels",
]
