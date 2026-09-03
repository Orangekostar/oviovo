"""Causal visibility gating for unbound immutable map anchors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum, IntEnum
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
LOCALIZED_POLICY_ID = "crove_ovimap_localized_visibility_l1_v1"
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


def localized_anchor_visibility_config_from_json(
    payload: Mapping[str, object],
) -> AnchorVisibilityConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("localized visibility policy must be a mapping")
    config_fields = {item.name for item in fields(AnchorVisibilityConfig)}
    expected = config_fields | {
        "schema_version",
        "policy_id",
        "state_granularity",
        "voxel_sampling",
    }
    if set(payload) != expected:
        raise ValueError("localized visibility policy fields are invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("policy_id") != LOCALIZED_POLICY_ID
        or payload.get("state_granularity") != "per_voxel"
        or payload.get("voxel_sampling") != VOXEL_SAMPLING_MODE
    ):
        raise ValueError("localized visibility policy identity is invalid")
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


class AnchorVoxelVisibilityEvidenceKind(IntEnum):
    PRESENT = 1
    VISIBLE_ABSENT = 2
    OCCLUDED = 3
    UNOBSERVED = 4


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype,
    name: str,
    shape_tail: tuple[int, ...] = (),
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    expected_dimensions = 1 + len(shape_tail)
    if array.ndim != expected_dimensions or array.shape[1:] != shape_tail:
        suffix = "" if not shape_tail else f" with trailing shape {shape_tail}"
        raise ValueError(f"{name} must be a {expected_dimensions}D array{suffix}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AnchorVoxelVisibilityEvidence:
    entity_id: str
    frame_id: int
    timestamp: float
    voxel_indices: np.ndarray
    status_codes: np.ndarray
    camera_position_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        object.__setattr__(
            self, "frame_id", _integer(self.frame_id, "frame_id", minimum=0)
        )
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        indices = _readonly_array(
            self.voxel_indices,
            dtype=np.dtype(np.int32),
            name="voxel_indices",
        )
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError("voxel_indices must be non-empty, sorted, and unique")
        codes = _readonly_array(
            self.status_codes,
            dtype=np.dtype(np.uint8),
            name="status_codes",
        )
        allowed = np.asarray(
            [item.value for item in AnchorVoxelVisibilityEvidenceKind],
            dtype=np.uint8,
        )
        if len(codes) != len(indices) or not np.isin(codes, allowed).all():
            raise ValueError("status_codes must classify every evidence voxel")
        object.__setattr__(self, "voxel_indices", indices)
        object.__setattr__(self, "status_codes", codes)
        object.__setattr__(
            self,
            "camera_position_xyz",
            _point(self.camera_position_xyz, "camera_position_xyz"),
        )


@dataclass(frozen=True)
class AnchorCurrentOwnership:
    entity_id: str
    policy_id: str
    voxel_size_m: float
    voxel_keys: np.ndarray
    evidence_sample_indices: np.ndarray
    current_mask: np.ndarray
    absence_observation_counts: np.ndarray
    absence_viewpoints_xyz: np.ndarray
    absence_viewpoint_counts: np.ndarray
    present_streaks: np.ndarray
    first_absence_frames: np.ndarray
    last_frame_id: int
    last_timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        if self.policy_id != LOCALIZED_POLICY_ID:
            raise ValueError("localized ownership policy identity is invalid")
        object.__setattr__(
            self,
            "voxel_size_m",
            _finite(self.voxel_size_m, "voxel_size_m", positive=True),
        )
        keys = _readonly_array(
            self.voxel_keys,
            dtype=np.dtype(np.int32),
            name="voxel_keys",
            shape_tail=(3,),
        )
        if len(keys) == 0:
            raise ValueError("voxel_keys must be non-empty")
        key_order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        if not np.array_equal(key_order, np.arange(len(keys))) or np.any(
            np.all(keys[1:] == keys[:-1], axis=1)
        ):
            raise ValueError("voxel_keys must be sorted and unique")
        sample = _readonly_array(
            self.evidence_sample_indices,
            dtype=np.dtype(np.int32),
            name="evidence_sample_indices",
        )
        if (
            len(sample) == 0
            or np.any(sample < 0)
            or np.any(sample >= len(keys))
            or np.any(sample[1:] <= sample[:-1])
        ):
            raise ValueError("evidence sample indices are invalid")
        current = _readonly_array(
            self.current_mask,
            dtype=np.dtype(np.bool_),
            name="current_mask",
        )
        absence = _readonly_array(
            self.absence_observation_counts,
            dtype=np.dtype(np.uint16),
            name="absence_observation_counts",
        )
        raw_viewpoints = np.asarray(self.absence_viewpoints_xyz)
        if raw_viewpoints.ndim != 3 or raw_viewpoints.shape[2:] != (3,):
            raise ValueError(
                "absence_viewpoints_xyz must have shape (N, capacity, 3)"
            )
        viewpoints = _readonly_array(
            raw_viewpoints,
            dtype=np.dtype(np.float32),
            name="absence_viewpoints_xyz",
            shape_tail=(raw_viewpoints.shape[1], 3),
        )
        viewpoint_counts = _readonly_array(
            self.absence_viewpoint_counts,
            dtype=np.dtype(np.uint8),
            name="absence_viewpoint_counts",
        )
        streaks = _readonly_array(
            self.present_streaks,
            dtype=np.dtype(np.uint16),
            name="present_streaks",
        )
        first_absence = _readonly_array(
            self.first_absence_frames,
            dtype=np.dtype(np.int32),
            name="first_absence_frames",
        )
        arrays = (current, absence, viewpoint_counts, streaks, first_absence)
        if any(len(array) != len(keys) for array in arrays) or len(viewpoints) != len(
            keys
        ):
            raise ValueError("localized ownership arrays must align with voxel_keys")
        if viewpoints.shape[1] == 0 or not np.isfinite(viewpoints).all():
            raise ValueError("absence viewpoints must have finite positive capacity")
        if np.any(viewpoint_counts > viewpoints.shape[1]):
            raise ValueError("absence viewpoint counts exceed stored capacity")
        if np.any(first_absence < -1):
            raise ValueError("first absence frames must use -1 for no evidence")
        for name, value in (
            ("voxel_keys", keys),
            ("evidence_sample_indices", sample),
            ("current_mask", current),
            ("absence_observation_counts", absence),
            ("absence_viewpoints_xyz", viewpoints),
            ("absence_viewpoint_counts", viewpoint_counts),
            ("present_streaks", streaks),
            ("first_absence_frames", first_absence),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "last_frame_id",
            _integer(self.last_frame_id, "last_frame_id", minimum=0),
        )
        object.__setattr__(
            self,
            "last_timestamp",
            _finite(self.last_timestamp, "last_timestamp"),
        )

    @property
    def current_voxel_count(self) -> int:
        return int(np.count_nonzero(self.current_mask))

    @property
    def suppressed_voxel_count(self) -> int:
        return len(self.current_mask) - self.current_voxel_count

    @property
    def whole_anchor_status(self) -> str:
        current = self.current_voxel_count
        if current == len(self.current_mask):
            return "unchanged"
        return "dormant" if current == 0 else "partially_suppressed"

    @property
    def state_byte_count(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.voxel_keys,
                self.evidence_sample_indices,
                self.current_mask,
                self.absence_observation_counts,
                self.absence_viewpoints_xyz,
                self.absence_viewpoint_counts,
                self.present_streaks,
                self.first_absence_frames,
            )
        )


def initialize_anchor_current_ownership(
    entity_id: str,
    points_xyz: object,
    config: AnchorVisibilityConfig,
    *,
    frame_id: int,
    timestamp: float,
) -> AnchorCurrentOwnership:
    if not isinstance(config, AnchorVisibilityConfig):
        raise TypeError("config must be AnchorVisibilityConfig")
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError("points_xyz must have shape (N, 3) with N > 0")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz must contain only finite values")
    quantized = np.floor(points / config.voxel_size_m)
    limits = np.iinfo(np.int32)
    if np.any(quantized < limits.min) or np.any(quantized > limits.max):
        raise ValueError("anchor voxel keys exceed int32 range")
    keys = np.unique(quantized.astype(np.int32), axis=0)
    sample_count = min(len(keys), config.maximum_voxels_per_anchor)
    sample = np.linspace(0, len(keys) - 1, sample_count, dtype=np.int32)
    voxel_count = len(keys)
    return AnchorCurrentOwnership(
        entity_id=entity_id,
        policy_id=LOCALIZED_POLICY_ID,
        voxel_size_m=config.voxel_size_m,
        voxel_keys=keys,
        evidence_sample_indices=sample,
        current_mask=np.ones(voxel_count, dtype=np.bool_),
        absence_observation_counts=np.zeros(voxel_count, dtype=np.uint16),
        absence_viewpoints_xyz=np.zeros(
            (voxel_count, config.minimum_distinct_viewpoints, 3), dtype=np.float32
        ),
        absence_viewpoint_counts=np.zeros(voxel_count, dtype=np.uint8),
        present_streaks=np.zeros(voxel_count, dtype=np.uint16),
        first_absence_frames=np.full(voxel_count, -1, dtype=np.int32),
        last_frame_id=frame_id,
        last_timestamp=timestamp,
    )


def classify_anchor_voxel_visibility(
    ownership: AnchorCurrentOwnership,
    frame: Frame,
    config: AnchorVisibilityConfig,
) -> AnchorVoxelVisibilityEvidence:
    if not isinstance(ownership, AnchorCurrentOwnership):
        raise TypeError("ownership must be AnchorCurrentOwnership")
    if not isinstance(frame, Frame):
        raise TypeError("frame must be Frame")
    if not isinstance(config, AnchorVisibilityConfig):
        raise TypeError("config must be AnchorVisibilityConfig")
    if ownership.voxel_size_m != config.voxel_size_m:
        raise ValueError("ownership and visibility voxel sizes do not match")
    depth = np.asarray(frame.depth, dtype=np.float64)
    clean_depth = np.array(depth, copy=True)
    valid = (
        np.isfinite(clean_depth)
        & (clean_depth > 0.0)
        & (clean_depth <= config.depth_max_m)
    )
    clean_depth[~valid] = 0.0
    clean = Frame(
        frame_id=frame.frame_id,
        source_frame_id=frame.source_frame_id,
        rgb=frame.rgb,
        depth=clean_depth,
        pose=frame.pose,
        intrinsics=frame.intrinsics,
        timestamp=frame.timestamp,
    )
    sampled_keys = tuple(
        tuple(int(item) for item in row)
        for row in ownership.voxel_keys[ownership.evidence_sample_indices]
    )
    grouped = VoxelVisibilityProjector(
        VisibilityConfig(
            voxel_size_m=config.voxel_size_m,
            depth_tolerance_m=config.depth_tolerance_m,
        )
    ).classify_many(sampled_keys, clean)
    code_by_key = {
        key: AnchorVoxelVisibilityEvidenceKind.PRESENT.value
        for key in grouped[VisibilityStatus.PRESENT]
    }
    code_by_key.update(
        {
            key: AnchorVoxelVisibilityEvidenceKind.VISIBLE_ABSENT.value
            for key in grouped[VisibilityStatus.ABSENT]
        }
    )
    code_by_key.update(
        {
            key: AnchorVoxelVisibilityEvidenceKind.OCCLUDED.value
            for key in grouped[VisibilityStatus.OCCLUDED]
        }
    )
    code_by_key.update(
        {
            key: AnchorVoxelVisibilityEvidenceKind.UNOBSERVED.value
            for key in grouped[VisibilityStatus.UNOBSERVED]
        }
    )
    return AnchorVoxelVisibilityEvidence(
        entity_id=ownership.entity_id,
        frame_id=frame.frame_id,
        timestamp=float(frame.timestamp),
        voxel_indices=ownership.evidence_sample_indices,
        status_codes=np.asarray(
            [code_by_key[key] for key in sampled_keys], dtype=np.uint8
        ),
        camera_position_xyz=_point(frame.pose[:3, 3], "frame camera position"),
    )


def advance_anchor_current_ownership(
    ownership: AnchorCurrentOwnership,
    evidence: AnchorVoxelVisibilityEvidence,
    config: AnchorVisibilityConfig,
) -> AnchorCurrentOwnership:
    if not isinstance(ownership, AnchorCurrentOwnership):
        raise TypeError("ownership must be AnchorCurrentOwnership")
    if not isinstance(evidence, AnchorVoxelVisibilityEvidence):
        raise TypeError("evidence must be AnchorVoxelVisibilityEvidence")
    if not isinstance(config, AnchorVisibilityConfig):
        raise TypeError("config must be AnchorVisibilityConfig")
    if ownership.voxel_size_m != config.voxel_size_m:
        raise ValueError("ownership and visibility voxel sizes do not match")
    if evidence.entity_id != ownership.entity_id or not np.array_equal(
        evidence.voxel_indices, ownership.evidence_sample_indices
    ):
        raise ValueError("voxel evidence does not match localized ownership")
    if (
        evidence.frame_id <= ownership.last_frame_id
        or evidence.timestamp <= ownership.last_timestamp
    ):
        raise ValueError("visibility evidence must increase strictly")

    current = np.array(ownership.current_mask, copy=True)
    absence = np.array(ownership.absence_observation_counts, copy=True)
    viewpoints = np.array(ownership.absence_viewpoints_xyz, copy=True)
    viewpoint_counts = np.array(ownership.absence_viewpoint_counts, copy=True)
    streaks = np.array(ownership.present_streaks, copy=True)
    first_absence = np.array(ownership.first_absence_frames, copy=True)
    indices = np.asarray(evidence.voxel_indices, dtype=np.intp)
    codes = evidence.status_codes
    present_code = AnchorVoxelVisibilityEvidenceKind.PRESENT.value
    absent_code = AnchorVoxelVisibilityEvidenceKind.VISIBLE_ABSENT.value

    present_indices = indices[codes == present_code]
    present_current = present_indices[current[present_indices]]
    if len(present_current):
        absence[present_current] = 0
        viewpoints[present_current] = 0.0
        viewpoint_counts[present_current] = 0
        streaks[present_current] = 0
        first_absence[present_current] = -1
    present_suppressed = present_indices[~current[present_indices]]
    if len(present_suppressed):
        streaks[present_suppressed] = np.minimum(
            streaks[present_suppressed].astype(np.uint32) + 1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        restored = present_suppressed[
            streaks[present_suppressed] >= config.minimum_present_streak
        ]
        if len(restored):
            current[restored] = True
            absence[restored] = 0
            viewpoints[restored] = 0.0
            viewpoint_counts[restored] = 0
            streaks[restored] = 0
            first_absence[restored] = -1

    nonpresent_suppressed = indices[
        (codes != present_code) & (~current[indices])
    ]
    streaks[nonpresent_suppressed] = 0
    absent_indices = indices[(codes == absent_code) & current[indices]]
    if len(absent_indices):
        absence[absent_indices] = np.minimum(
            absence[absent_indices].astype(np.uint32) + 1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        first_absence[absent_indices[first_absence[absent_indices] < 0]] = (
            evidence.frame_id
        )
        camera = np.asarray(evidence.camera_position_xyz, dtype=np.float32)
        for index in absent_indices:
            count = int(viewpoint_counts[index])
            distinct = count == 0 or np.all(
                np.linalg.norm(
                    viewpoints[index, :count].astype(np.float64)
                    - camera.astype(np.float64),
                    axis=1,
                )
                >= config.minimum_viewpoint_baseline_m
            )
            if distinct and count < viewpoints.shape[1]:
                viewpoints[index, count] = camera
                viewpoint_counts[index] = count + 1
        suppressed = absent_indices[
            (absence[absent_indices] >= config.minimum_absent_observations)
            & (
                viewpoint_counts[absent_indices]
                >= config.minimum_distinct_viewpoints
            )
        ]
        current[suppressed] = False
        streaks[suppressed] = 0

    return AnchorCurrentOwnership(
        entity_id=ownership.entity_id,
        policy_id=ownership.policy_id,
        voxel_size_m=ownership.voxel_size_m,
        voxel_keys=ownership.voxel_keys,
        evidence_sample_indices=ownership.evidence_sample_indices,
        current_mask=current,
        absence_observation_counts=absence,
        absence_viewpoints_xyz=viewpoints,
        absence_viewpoint_counts=viewpoint_counts,
        present_streaks=streaks,
        first_absence_frames=first_absence,
        last_frame_id=evidence.frame_id,
        last_timestamp=evidence.timestamp,
    )


def pack_anchor_current_mask(ownership: AnchorCurrentOwnership) -> bytes:
    if not isinstance(ownership, AnchorCurrentOwnership):
        raise TypeError("ownership must be AnchorCurrentOwnership")
    return np.packbits(ownership.current_mask, bitorder="little").tobytes()


__all__ = [
    "LOCALIZED_POLICY_ID",
    "POLICY_ID",
    "VOXEL_SAMPLING_MODE",
    "AnchorCurrentOwnership",
    "AnchorVisibilityConfig",
    "AnchorVisibilityEvidence",
    "AnchorVisibilityEvidenceKind",
    "AnchorVisibilityState",
    "AnchorVoxelVisibilityEvidence",
    "AnchorVoxelVisibilityEvidenceKind",
    "advance_anchor_current_ownership",
    "advance_anchor_visibility",
    "anchor_visibility_config_from_json",
    "classify_anchor_voxel_visibility",
    "classify_anchor_visibility",
    "initialize_anchor_current_ownership",
    "localized_anchor_visibility_config_from_json",
    "pack_anchor_current_mask",
    "sample_anchor_voxels",
]
