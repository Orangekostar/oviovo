"""Module 9: Background Update.

Responsibility: Update the global background map while avoiding
contamination from foreground object regions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import numpy as np

from src.core.data_structures import BackgroundMap, ObjectMap, Patch3D
from src.utils.geometry import voxel_downsample

logger = logging.getLogger("oviovo.modules.background_update")


class BackgroundUpdateModule:
    """Updates the global background map with background patches."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.voxel_size = config.get("voxel_size", 0.05)
        logger.info("BackgroundUpdateModule initialized.")

    def process(
        self,
        bg_patches: List[Patch3D],
        background: BackgroundMap,
        objects: Dict[int, ObjectMap],
    ) -> BackgroundMap:
        """Integrate background patches into the background map.

        Args:
            bg_patches: Patches classified as background.
            background: Current background map.
            objects: Current objects (for occupancy masking).

        Returns:
            Updated BackgroundMap.
        """
        if not bg_patches:
            return background

        structural_reject_patches = [
            patch for patch in bg_patches
            if bool(patch.metadata.get("surface_gate_structural_reject", False))
        ]
        regular_bg_patches = [
            patch for patch in bg_patches
            if not bool(patch.metadata.get("surface_gate_structural_reject", False))
        ]

        point_chunks: list[np.ndarray] = []
        if regular_bg_patches:
            regular_points = np.concatenate([p.points for p in regular_bg_patches], axis=0)
            # Filter regular background points that overlap known object boxes.
            point_chunks.append(self._filter_object_occupied(regular_points, objects))
        if structural_reject_patches:
            # These points were explicitly rejected by object surface-owner
            # arbitration, so feeding them back strengthens structural support.
            point_chunks.append(np.concatenate([p.points for p in structural_reject_patches], axis=0))
        if not point_chunks:
            return background

        new_points = np.concatenate(point_chunks, axis=0)

        # Merge into background point cloud
        if len(background.point_cloud) > 0:
            background.point_cloud = np.concatenate(
                [background.point_cloud, new_points], axis=0
            )
        else:
            background.point_cloud = new_points

        # Downsample to keep manageable
        background.point_cloud = voxel_downsample(
            background.point_cloud, self.voxel_size
        )
        background.frame_count += 1

        # TODO: Replace point accumulation with proper TSDF fusion.
        # The interface is designed so that a TSDF volume can replace
        # the point_cloud field with tsdf_volume + weight_volume.

        logger.debug(
            f"Background updated: {len(background.point_cloud)} points, "
            f"{background.frame_count} frames integrated."
        )
        return background

    def _filter_object_occupied(
        self,
        points: np.ndarray,
        objects: Dict[int, ObjectMap],
    ) -> np.ndarray:
        """Remove points that fall inside known object bounding boxes.

        TODO: Use more precise occupancy checks (point-in-hull, distance field).
        """
        if len(points) == 0 or not objects:
            return points

        keep = np.ones(len(points), dtype=bool)
        for obj in objects.values():
            if obj.state.value == "removed":
                continue
            inside = np.all(
                (points >= obj.bbox_min) & (points <= obj.bbox_max), axis=1
            )
            keep &= ~inside

        return points[keep]
