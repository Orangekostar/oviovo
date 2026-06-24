"""Module: TSDF Instance Map (Layer 4 backbone).

Global TSDF-based instance map. Each voxel stores:
  - TSDF value + weight
  - owner instance id (argmax of support)
  - per-instance support (accumulated / decayed)

Patch-to-instance association is spatial/geometric first.
Uses sparse hash-map voxel grid for memory efficiency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.data_structures import (
    CameraIntrinsics,
    Patch3D,
    TSDFInstanceVolume,
    VoxelOwnerSupport,
    VoxelVoteResult,
)

logger = logging.getLogger("oviovo.modules.tsdf_instance_map")


def _world_to_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Convert world coordinates to voxel indices."""
    return np.floor(points / voxel_size).astype(np.int64)


class TSDFInstanceMapModule:
    """Global TSDF instance substrate.

    This is the true low-level stable backbone of the system.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.voxel_size = config.get("voxel_size", 0.05)
        self.truncation = config.get("truncation_distance", 0.15)
        self.support_increment = config.get("support_increment", 1.0)
        self.support_decay = config.get("support_decay", 0.95)
        self.w_vote = config.get("vote_weight", 0.6)
        self.w_geom = config.get("geometry_weight", 0.4)
        self.match_threshold = config.get("match_threshold", 0.3)
        self.nearby_radius = config.get("nearby_radius", 3.0)
        logger.info("TSDFInstanceMapModule initialized.")

    def integrate_patch(
        self,
        volume: TSDFInstanceVolume,
        patch: Patch3D,
        instance_id: int,
    ) -> None:
        """Integrate a patch into the global TSDF and mark voxel ownership.

        Args:
            volume: The global TSDF instance volume.
            patch: 3D patch to integrate.
            instance_id: The instance that owns this patch.
        """
        voxel_indices = _world_to_voxel(patch.points, volume.voxel_size)
        unique_voxels = set(map(tuple, voxel_indices.tolist()))

        for vk in unique_voxels:
            # TSDF integration (simplified: binary occupancy + weight)
            volume.tsdf[vk] = volume.tsdf.get(vk, 0.0) * 0.5 + 0.5 * 0.0  # near-surface
            volume.weight[vk] = volume.weight.get(vk, 0.0) + 1.0

            # Owner support accumulation (no hard overwrite)
            if vk not in volume.owner_support:
                volume.owner_support[vk] = VoxelOwnerSupport()
            support = volume.owner_support[vk]

            # Accumulate support for the winning instance
            support.support[instance_id] = (
                support.support.get(instance_id, 0.0) + volume.support_increment
            )

            # Decay support for competing instances
            for oid in list(support.support.keys()):
                if oid != instance_id:
                    support.support[oid] *= volume.support_decay
                    if support.support[oid] < 0.01:
                        del support.support[oid]

    def remove_patch_support(
        self,
        volume: TSDFInstanceVolume,
        patch: Patch3D,
        instance_id: int,
    ) -> None:
        """Remove one patch's owner support while preserving occupancy and weight."""
        voxel_indices = _world_to_voxel(patch.points, volume.voxel_size)
        unique_voxels = set(map(tuple, voxel_indices.tolist()))

        for vk in unique_voxels:
            support = volume.owner_support.get(vk)
            if support is None or instance_id not in support.support:
                continue
            support.support[instance_id] = (
                support.support.get(instance_id, 0.0) - volume.support_increment
            )
            if support.support[instance_id] <= 0.01:
                del support.support[instance_id]
            if not support.support:
                del volume.owner_support[vk]

    def vote_patch_to_instance(
        self,
        volume: TSDFInstanceVolume,
        patch: Patch3D,
    ) -> VoxelVoteResult:
        """Spatial voting: which existing instances own the voxels this patch touches.

        Args:
            volume: The global TSDF instance volume.
            patch: 3D patch to query.

        Returns:
            VoxelVoteResult with all voting metadata exposed.
        """
        voxel_indices = _world_to_voxel(patch.points, volume.voxel_size)
        unique_voxels = set(map(tuple, voxel_indices.tolist()))

        owner_votes: Dict[int, int] = {}
        supported_touched = 0

        for vk in unique_voxels:
            if vk not in volume.owner_support:
                continue
            supported_touched += 1
            owner = volume.owner_support[vk].owner_id
            if owner >= 0:
                owner_votes[owner] = owner_votes.get(owner, 0) + 1

        total_supported = max(supported_touched, 1)
        if owner_votes:
            best_id = max(owner_votes, key=owner_votes.get)
            normalized = owner_votes[best_id] / total_supported
        else:
            best_id = -1
            normalized = 0.0

        return VoxelVoteResult(
            touched_voxel_count=len(unique_voxels),
            supported_voxel_count=supported_touched,
            owner_votes=owner_votes,
            normalized_vote_score=normalized,
            best_instance_id=best_id,
        )

    def query_visible_instances(
        self,
        volume: TSDFInstanceVolume,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Set[int]:
        """Return instance ids with owned voxels projected inside the camera frustum."""
        try:
            world_to_camera = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            logger.warning("Pose inversion failed during visibility query; returning empty visible set.")
            return set()

        visible: Set[int] = set()

        for vk, support in volume.owner_support.items():
            oid = support.owner_id
            if oid < 0:
                continue

            world_pt = (np.array(vk, dtype=np.float64) + 0.5) * volume.voxel_size
            world_pt_h = np.concatenate([world_pt, np.array([1.0], dtype=np.float64)])
            cam_pt = world_to_camera @ world_pt_h
            depth = float(cam_pt[2])
            if depth <= 0.0:
                continue

            u = intrinsics.fx * (cam_pt[0] / depth) + intrinsics.cx
            v = intrinsics.fy * (cam_pt[1] / depth) + intrinsics.cy
            if 0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height:
                visible.add(oid)

        return visible

    def get_instance_voxel_count(
        self,
        volume: TSDFInstanceVolume,
        instance_id: int,
    ) -> int:
        """Count how many voxels are currently owned by this instance."""
        count = 0
        for support in volume.owner_support.values():
            if support.owner_id == instance_id:
                count += 1
        return count

    def summarize_instance_support(
        self,
        volume: TSDFInstanceVolume,
        instance_id: int,
    ) -> Dict[str, float]:
        """Summarize owner-support statistics for one instance."""
        owner_values: List[float] = []
        competing_values: List[float] = []

        for support in volume.owner_support.values():
            if instance_id not in support.support:
                continue
            if support.owner_id == instance_id:
                owner_values.append(float(support.support[instance_id]))
            else:
                competing_values.append(float(support.support[instance_id]))

        owned_voxel_count = len(owner_values)
        support_mass = float(sum(owner_values))
        mean_owner_support = float(support_mass / max(owned_voxel_count, 1))
        max_owner_support = float(max(owner_values)) if owner_values else 0.0
        competing_support_mass = float(sum(competing_values))

        density_term = min(1.0, owned_voxel_count / 20.0)
        support_term = min(1.0, mean_owner_support / max(volume.support_increment * 2.0, 1e-6))
        stability_score = float(np.clip(0.5 * density_term + 0.5 * support_term, 0.0, 1.0))

        return {
            "owned_voxel_count": float(owned_voxel_count),
            "support_mass": support_mass,
            "mean_owner_support": mean_owner_support,
            "max_owner_support": max_owner_support,
            "competing_support_mass": competing_support_mass,
            "stability_score": stability_score,
        }
