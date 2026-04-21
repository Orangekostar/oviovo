"""Module 7: Object Update (Layers 4+5).

Responsibility:
  - Integrate patches into global TSDF (Layer 4)
  - Update voxel label support (accumulate/decay, no hard overwrite)
  - Maintain per-instance local_pcd as maintenance memory only (Layer 5)
  - Update soft whole-evidence scores (Rule B)
  - Trigger semantic update hook after association
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import (
    AssociationResult,
    ObjectMap,
    ObjectState,
    ObservationRecord,
    Patch3D,
    SemanticMemory,
    SystemState,
    TSDFInstanceVolume,
    WholeEvidenceScores,
)
from src.modules.tsdf_instance_map import TSDFInstanceMapModule
from src.utils.geometry import compute_bbox, voxel_downsample

logger = logging.getLogger("oviovo.modules.object_update")


class ObjectUpdateModule:
    """Updates object geometry, TSDF ownership, and whole-evidence scores."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.downsample_voxel = config.get("downsample_voxel_size", 0.02)
        self.downsample_interval = config.get("downsample_interval", 5)
        self.max_points = config.get("max_points_per_object", 10000)
        self.whole_evidence_growth = config.get("whole_evidence_growth", 0.05)
        self.part_evidence_growth = config.get("part_evidence_growth", 0.02)
        self.tsdf_module = TSDFInstanceMapModule(config.get("tsdf", {}))
        logger.info("ObjectUpdateModule initialized.")

    def process(
        self,
        association: AssociationResult,
        patches: List[Patch3D],
        state: SystemState,
    ) -> SystemState:
        """Update existing objects and create new ones.

        After association:
          1. Integrate patch into global TSDF (Layer 4)
          2. Update voxel support (accumulate/decay)
          3. Update local_pcd as maintenance memory (Layer 5)
          4. Update soft whole_evidence scores (Rule B)

        Args:
            association: Result from the association module.
            patches: All object patches (needed to look up by patch_id).
            state: Current system state.

        Returns:
            Updated system state.
        """
        patch_map = {p.patch_id: p for p in patches}

        # Update matched objects
        for patch_id, obj_id, score in association.matched:
            patch = patch_map.get(patch_id)
            if patch is None:
                continue
            obj = state.objects.get(obj_id)
            if obj is None:
                continue
            # Step 1: Integrate into global TSDF
            self.tsdf_module.integrate_patch(state.tsdf_volume, patch, obj_id)
            # Step 2+3: Update local geometry + whole evidence
            self._update_object(obj, patch)
            self._refresh_object_debug(obj, state.tsdf_volume)

        # Create new objects
        for patch_id in association.new_object_patches:
            patch = patch_map.get(patch_id)
            if patch is None:
                continue
            new_obj = self._create_object(patch, state.next_object_id)
            state.objects[new_obj.object_id] = new_obj
            # Integrate new object into global TSDF
            self.tsdf_module.integrate_patch(state.tsdf_volume, patch, new_obj.object_id)
            self._refresh_object_debug(new_obj, state.tsdf_volume)
            state.next_object_id += 1

        logger.debug(f"Objects updated. Total objects: {len(state.objects)}")
        return state

    def _update_object(self, obj: ObjectMap, patch: Patch3D) -> None:
        """Merge new patch into an existing object.

        local_pcd is ONLY maintenance memory (Layer 5).
        It does NOT replace the global TSDF instance substrate.
        """
        # Update local_pcd (maintenance memory only)
        obj.local_pcd = np.concatenate([obj.local_pcd, patch.points], axis=0)
        obj.update_count += 1

        # Periodic downsampling of local_pcd
        if obj.update_count % self.downsample_interval == 0:
            obj.local_pcd = voxel_downsample(obj.local_pcd, self.downsample_voxel)

        # Cap point count
        if len(obj.local_pcd) > self.max_points:
            indices = np.random.choice(len(obj.local_pcd), self.max_points, replace=False)
            obj.local_pcd = obj.local_pcd[indices]

        # Update spatial properties from local_pcd
        obj.centroid = obj.local_pcd.mean(axis=0)
        obj.bbox_min, obj.bbox_max = compute_bbox(obj.local_pcd)
        obj.last_seen_frame = patch.source_frame_id
        obj.state = ObjectState.ACTIVE

        # Record observation
        obj.observations.append(ObservationRecord(
            frame_id=patch.source_frame_id,
            patch=patch,
            crop_bbox=self._patch_crop_bbox(patch),
            timestamp=patch.timestamp,
        ))

        # Update soft whole-evidence scores (Rule B: soft only)
        self._update_whole_evidence(obj, patch)

    def _update_whole_evidence(self, obj: ObjectMap, patch: Patch3D) -> None:
        """Update soft whole-evidence scores based on observation patterns.

        No hard is_whole_object labels — only soft evidence.
        """
        we = obj.whole_evidence

        # whole_evidence_score: grows with consistent observations
        obs_count = len(obj.observations)
        we.whole_evidence_score = float(np.clip(
            we.whole_evidence_score + self.whole_evidence_growth * min(obs_count / 5.0, 1.0),
            0.0, 1.0,
        ))

        # part_evidence_score: based on patch-vs-object size ratio
        patch_size = len(patch.points)
        obj_size = max(len(obj.local_pcd), 1)
        size_ratio = patch_size / obj_size
        if size_ratio < 0.3:
            we.part_evidence_score = float(np.clip(
                we.part_evidence_score + self.part_evidence_growth, 0.0, 1.0,
            ))

        # assignment_score: confidence in the current instance assignment
        we.assignment_score = float(np.clip(
            0.5 + 0.1 * min(obs_count, 5), 0.0, 1.0,
        ))

    def _create_object(self, patch: Patch3D, object_id: int) -> ObjectMap:
        """Create a new object from a patch."""
        bbox_min, bbox_max = compute_bbox(patch.points)
        obj = ObjectMap(
            object_id=object_id,
            state=ObjectState.ACTIVE,
            local_pcd=patch.points.copy(),
            centroid=patch.centroid.copy(),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            whole_evidence=WholeEvidenceScores(
                whole_evidence_score=0.1,
                part_evidence_score=0.0,
                assignment_score=0.5,
            ),
            semantic_memory=SemanticMemory(),
            observations=[ObservationRecord(
                frame_id=patch.source_frame_id,
                patch=patch,
                crop_bbox=self._patch_crop_bbox(patch),
                timestamp=patch.timestamp,
            )],
            confidence=1.0,
            last_seen_frame=patch.source_frame_id,
            creation_frame=patch.source_frame_id,
            update_count=1,
        )
        logger.debug(f"Created new object {object_id} with {len(patch.points)} points.")
        return obj

    def _patch_crop_bbox(self, patch: Patch3D) -> np.ndarray | None:
        """Recover the source 2D crop bbox for semantic view selection."""
        bbox = patch.metadata.get("source_bbox_xyxy")
        if bbox is None:
            return None
        return np.asarray(bbox, dtype=np.float32).copy()

    def _refresh_object_debug(self, obj: ObjectMap, volume: TSDFInstanceVolume) -> None:
        """Expose the separation between global TSDF memory and local maintenance memory."""
        support_stats = self.tsdf_module.summarize_instance_support(volume, obj.object_id)
        obj.debug["global_instance_substrate"] = {
            "role": "true_low_level_backbone",
            "owned_voxel_count": int(support_stats["owned_voxel_count"]),
            "support_mass": float(support_stats["support_mass"]),
            "mean_owner_support": float(support_stats["mean_owner_support"]),
            "max_owner_support": float(support_stats["max_owner_support"]),
            "competing_support_mass": float(support_stats["competing_support_mass"]),
            "stability_score": float(support_stats["stability_score"]),
        }
        obj.debug["local_geometry_memory"] = {
            "role": "maintenance_only",
            "local_point_count": int(len(obj.local_pcd)),
            "downsample_voxel_size": float(self.downsample_voxel),
            "max_points_per_object": int(self.max_points),
        }
        obj.debug["whole_evidence"] = {
            "whole_evidence_score": float(obj.whole_evidence.whole_evidence_score),
            "part_evidence_score": float(obj.whole_evidence.part_evidence_score),
            "assignment_score": float(obj.whole_evidence.assignment_score),
        }
