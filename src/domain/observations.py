from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.domain.arrays import readonly_array


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class ObservationQuality:
    mask_confidence: float = 0.0
    valid_depth_ratio: float = 0.0
    visible_point_ratio: float = 0.0
    view_quality: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "mask_confidence",
            "valid_depth_ratio",
            "visible_point_ratio",
            "view_quality",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))


@dataclass(frozen=True)
class FrameObservation:
    observation_id: str
    frame_id: int
    timestamp: float
    source_proposal_id: int
    source_backend: str
    mask: np.ndarray | None
    bbox_xyxy: np.ndarray
    points_world: np.ndarray
    voxel_keys: np.ndarray
    detector_label: str = ""
    detector_confidence: float = 0.0
    visual_embedding: np.ndarray | None = None
    quality: ObservationQuality = field(default_factory=ObservationQuality)
    refinement_key: str = ""
    parent_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.observation_id):
            raise ValueError("observation_id must be non-empty")
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if int(self.source_proposal_id) < 0:
            raise ValueError("source_proposal_id must be non-negative")
        if not str(self.source_backend):
            raise ValueError("source_backend must be non-empty")

        mask = None
        if self.mask is not None:
            mask = readonly_array(self.mask, dtype=bool, ndim=2)
        bbox = readonly_array(self.bbox_xyxy, dtype=np.float32, ndim=1)
        if bbox.shape != (4,) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy must be an ordered vector with shape (4,)")
        points = readonly_array(
            self.points_world,
            dtype=np.float32,
            ndim=2,
            trailing_shape=(3,),
        )
        voxel_keys = readonly_array(
            self.voxel_keys,
            dtype=np.int64,
            ndim=2,
            trailing_shape=(3,),
        )
        embedding = None
        if self.visual_embedding is not None:
            embedding = readonly_array(self.visual_embedding, dtype=np.float32, ndim=1)
        parent_ids = tuple(str(value) for value in self.parent_observation_ids)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parent_observation_ids must be unique")

        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "source_proposal_id", int(self.source_proposal_id))
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "voxel_keys", voxel_keys)
        object.__setattr__(self, "detector_label", str(self.detector_label).strip())
        object.__setattr__(
            self,
            "detector_confidence",
            _probability(self.detector_confidence, "detector_confidence"),
        )
        object.__setattr__(self, "visual_embedding", embedding)
        object.__setattr__(self, "refinement_key", str(self.refinement_key))
        object.__setattr__(self, "parent_observation_ids", parent_ids)


@dataclass(frozen=True)
class ObservationBatch:
    frame_id: int
    observations: tuple[FrameObservation, ...] = ()

    def __post_init__(self) -> None:
        frame_id = int(self.frame_id)
        observations = tuple(self.observations)
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if any(item.frame_id != frame_id for item in observations):
            raise ValueError("all observations must match batch frame_id")
        observation_ids = [item.observation_id for item in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id values must be unique within a batch")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "observations", observations)
