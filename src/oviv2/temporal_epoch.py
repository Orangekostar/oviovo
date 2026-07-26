from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_geometry import (
    MotionDecision,
    ObjectMotionEstimate,
    ObjectSubmap,
    _finite_xyz,
    _readonly_array,
    _rigid_transform,
    _translation_pose,
    integrate_object_submap,
)
from src.oviv2.temporal_lifecycle import TemporalEvidenceKind


_MAX_INT64 = int(np.iinfo(np.int64).max)


def _identifier(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > _MAX_INT64:
        raise ValueError(f"{name} must be in [{minimum}, {_MAX_INT64}]")
    return normalized


@dataclass(frozen=True, eq=False)
class GeometryEpoch:
    entity_id: int
    epoch_id: int
    object_to_world: np.ndarray
    submap: ObjectSubmap
    readout_valid: bool
    motion_decision: MotionDecision | None = None
    last_processed_frame_id: int | None = None

    __hash__ = None

    def __post_init__(self) -> None:
        entity_id = _identifier(self.entity_id, "entity_id", minimum=1)
        epoch_id = _identifier(self.epoch_id, "epoch_id", minimum=0)
        pose = _rigid_transform(self.object_to_world, "object_to_world")
        if type(self.submap) is not ObjectSubmap:
            raise TypeError("submap must be an ObjectSubmap")
        if type(self.readout_valid) is not bool:
            raise TypeError("readout_valid must be an exact bool")
        if self.motion_decision is not None and type(self.motion_decision) is not MotionDecision:
            raise TypeError("motion_decision must be a MotionDecision or None")
        maximum_seen = (
            int(self.submap.last_seen_frame_ids.max())
            if self.submap.last_seen_frame_ids.size
            else None
        )
        if self.last_processed_frame_id is None:
            last_processed_frame_id = maximum_seen
        else:
            last_processed_frame_id = _identifier(
                self.last_processed_frame_id,
                "last_processed_frame_id",
                minimum=0,
            )
            if maximum_seen is not None and last_processed_frame_id < maximum_seen:
                raise ValueError(
                    "last_processed_frame_id cannot precede submap observations"
                )
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "epoch_id", epoch_id)
        object.__setattr__(
            self, "last_processed_frame_id", last_processed_frame_id
        )
        object.__setattr__(
            self, "object_to_world", _readonly_array(pose, np.dtype(np.float64))
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not GeometryEpoch:
            return False
        assert isinstance(other, GeometryEpoch)
        return bool(
            self.entity_id == other.entity_id
            and self.epoch_id == other.epoch_id
            and np.array_equal(self.object_to_world, other.object_to_world)
            and self.submap == other.submap
            and self.readout_valid is other.readout_valid
            and self.motion_decision is other.motion_decision
            and self.last_processed_frame_id == other.last_processed_frame_id
        )

    def __deepcopy__(self, memo: dict[int, object]) -> GeometryEpoch:
        copied = GeometryEpoch(
            self.entity_id,
            self.epoch_id,
            self.object_to_world,
            copy.deepcopy(self.submap, memo),
            self.readout_valid,
            self.motion_decision,
            self.last_processed_frame_id,
        )
        memo[id(self)] = copied
        return copied

    def integrate(
        self,
        estimate: ObjectMotionEstimate,
        points_world: np.ndarray,
        frame_id: int,
        config: TemporalGeometryConfig | None = None,
        *,
        entity_id: int | None = None,
    ) -> GeometryEpoch:
        if type(estimate) is not ObjectMotionEstimate:
            raise TypeError("estimate must be an ObjectMotionEstimate")
        if estimate.decision is MotionDecision.REJECTED:
            raise ValueError("rejected motion cannot integrate into an existing epoch")
        normalized_entity_id = (
            self.entity_id
            if entity_id is None
            else _identifier(entity_id, "entity_id", minimum=1)
        )
        if normalized_entity_id != self.entity_id:
            raise ValueError("entity_id does not match geometry epoch")
        if config is None:
            raise TypeError("config must be a TemporalGeometryConfig")
        normalized_frame_id = _identifier(frame_id, "frame_id", minimum=0)
        if (
            self.last_processed_frame_id is not None
            and normalized_frame_id <= self.last_processed_frame_id
        ):
            raise ValueError("frame_id must increase strictly")
        submap = integrate_object_submap(
            self.submap,
            points_world,
            normalized_frame_id,
            config,
            object_to_world=estimate.object_to_world,
        )
        readout_valid = self.readout_valid or submap != self.submap
        return GeometryEpoch(
            self.entity_id,
            self.epoch_id,
            estimate.object_to_world,
            submap,
            readout_valid,
            estimate.decision,
            normalized_frame_id,
        )

    def apply_evidence(self, evidence: TemporalEvidenceKind) -> GeometryEpoch:
        if type(evidence) is not TemporalEvidenceKind:
            raise TypeError("evidence must be a TemporalEvidenceKind")
        if evidence is TemporalEvidenceKind.VISIBLE_ABSENT:
            readout_valid = False
        elif evidence in (
            TemporalEvidenceKind.PRESENT,
            TemporalEvidenceKind.OCCLUDED,
            TemporalEvidenceKind.OUT_OF_VIEW,
            TemporalEvidenceKind.DEPTH_UNKNOWN,
        ):
            readout_valid = self.readout_valid
        else:
            raise ValueError("unsupported evidence")
        if readout_valid is self.readout_valid:
            return self
        return GeometryEpoch(
            self.entity_id,
            self.epoch_id,
            self.object_to_world,
            self.submap,
            readout_valid,
            self.motion_decision,
            self.last_processed_frame_id,
        )


def start_new_epoch(
    previous_epoch: GeometryEpoch,
    estimate: ObjectMotionEstimate,
    points_world: np.ndarray,
    observed_centroid_xyz: tuple[float, float, float],
    frame_id: int,
    config: TemporalGeometryConfig,
    *,
    entity_id: int | None = None,
    initial_object_to_world: np.ndarray | None = None,
) -> GeometryEpoch:
    if type(previous_epoch) is not GeometryEpoch:
        raise TypeError("previous_epoch must be a GeometryEpoch")
    if type(estimate) is not ObjectMotionEstimate:
        raise TypeError("estimate must be an ObjectMotionEstimate")
    normalized_entity_id = (
        previous_epoch.entity_id
        if entity_id is None
        else _identifier(entity_id, "entity_id", minimum=1)
    )
    if normalized_entity_id != previous_epoch.entity_id:
        raise ValueError("entity_id does not match previous geometry epoch")
    if previous_epoch.epoch_id == _MAX_INT64:
        raise OverflowError("epoch_id cannot exceed int64")
    normalized_frame_id = _identifier(frame_id, "frame_id", minimum=0)
    if (
        previous_epoch.last_processed_frame_id is not None
        and normalized_frame_id <= previous_epoch.last_processed_frame_id
    ):
        raise ValueError("frame_id must increase strictly across geometry epochs")

    if initial_object_to_world is None:
        centroid = _finite_xyz(observed_centroid_xyz, "observed_centroid_xyz")
        pose = _translation_pose(centroid)
    else:
        pose = _rigid_transform(initial_object_to_world, "initial_object_to_world")
    reference = tuple(float(value) for value in pose[:3, 3])
    empty = ObjectSubmap(
        reference,
        (),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.int64),
    )
    submap = integrate_object_submap(
        empty,
        points_world,
        normalized_frame_id,
        config,
        object_to_world=pose,
    )
    if submap.local_points_xyz.shape[0] == 0:
        raise ValueError("current observation must contain at least one valid point")
    return GeometryEpoch(
        normalized_entity_id,
        previous_epoch.epoch_id + 1,
        pose,
        submap,
        True,
        estimate.decision,
        normalized_frame_id,
    )
