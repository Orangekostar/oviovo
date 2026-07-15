from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.core.data_structures import Patch3D
from src.domain.observations import FrameObservation, ObservationBatch, ObservationQuality


class LegacyPatchObservationAdapter:
    def __init__(self, *, voxel_size: float) -> None:
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")

    def convert(
        self,
        patches: Sequence[Patch3D],
        *,
        masks_by_patch_id: Mapping[int, np.ndarray] | None = None,
    ) -> ObservationBatch:
        patches = tuple(patches)
        if not patches:
            raise ValueError("legacy adapter requires at least one patch to determine frame_id")
        frame_ids = {int(patch.source_frame_id) for patch in patches}
        if len(frame_ids) != 1:
            raise ValueError("legacy patches must belong to one frame")
        frame_id = next(iter(frame_ids))
        masks = masks_by_patch_id or {}
        observations = tuple(self._convert_one(patch, masks.get(int(patch.patch_id))) for patch in patches)
        return ObservationBatch(frame_id=frame_id, observations=observations)

    def _convert_one(self, patch: Patch3D, mask: np.ndarray | None) -> FrameObservation:
        metadata = patch.metadata
        points = np.asarray(patch.points, dtype=np.float32)
        voxel_keys = np.unique(np.floor(points / self.voxel_size).astype(np.int64), axis=0)
        source_proposal_id = int(metadata.get("source_proposal_id", patch.patch_id))
        parent_ids = tuple(
            f"{int(patch.source_frame_id)}:{int(raw_id)}"
            for raw_id in metadata.get("source_raw_proposal_ids", ())
        )
        geometry = dict(metadata.get("geometric_features", {}) or {})
        return FrameObservation(
            observation_id=f"{int(patch.source_frame_id)}:{source_proposal_id}:{int(patch.patch_id)}",
            frame_id=int(patch.source_frame_id),
            timestamp=float(patch.timestamp),
            source_proposal_id=source_proposal_id,
            source_backend=str(metadata.get("source_backend_name", "legacy_patch")),
            mask=mask,
            bbox_xyxy=np.asarray(
                metadata.get("source_bbox_xyxy", [0.0, 0.0, 0.0, 0.0]),
                dtype=np.float32,
            ),
            points_world=points,
            voxel_keys=voxel_keys,
            detector_label=str(metadata.get("anchor_class_name", "")),
            detector_confidence=float(metadata.get("anchor_confidence", 0.0)),
            quality=ObservationQuality(
                mask_confidence=float(metadata.get("anchor_confidence", 0.0)),
                valid_depth_ratio=float(geometry.get("depth_valid_ratio", 0.0)),
                visible_point_ratio=1.0 if len(points) else 0.0,
                view_quality=float(metadata.get("anchor_view_quality", 0.0)),
            ),
            refinement_key=str(metadata.get("refinement_key", "")),
            parent_observation_ids=parent_ids,
        )
