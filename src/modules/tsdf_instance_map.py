"""Module: TSDF Instance Map (Layer 4 backbone).

Global TSDF-based instance map. Each voxel stores:
  - TSDF value + weight
  - owner instance id (argmax of support)
  - per-instance support (accumulated / decayed)

Patch-to-instance association is spatial/geometric first.
Uses sparse hash-map voxel grid for memory efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _voxel_key(voxel: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(voxel[0]), int(voxel[1]), int(voxel[2]))


def get_voxel_owner_id(volume: TSDFInstanceVolume, voxel_key: tuple[int, int, int]) -> int:
    """Return cached owner id, lazily rebuilding owner index for legacy volumes."""
    key = _voxel_key(voxel_key)
    if TSDFInstanceMapModule._support_index_missing(volume):
        TSDFInstanceMapModule._rebuild_support_indexes(volume)
    owner_id = int(getattr(volume, "voxel_owner_id", {}).get(key, -1))
    support = volume.owner_support.get(key)
    actual_owner_id = -1 if support is None else int(support.owner_id)
    if owner_id != actual_owner_id:
        TSDFInstanceMapModule._rebuild_support_indexes(volume)
        owner_id = int(volume.voxel_owner_id.get(key, -1))
    return owner_id


owner_id_for_voxel = get_voxel_owner_id


def _world_to_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Convert world coordinates to voxel indices."""
    return np.floor(points / voxel_size).astype(np.int64)


@dataclass(frozen=True)
class PatchVoxelView:
    voxel_size: float
    voxel_indices: np.ndarray
    unique_voxels: np.ndarray
    representative_point_indices: np.ndarray
    inverse: np.ndarray | None = None
    cache_used: bool = False


def get_patch_voxel_view(
    patch: Patch3D,
    voxel_size: float,
    *,
    need_inverse: bool = False,
) -> PatchVoxelView:
    """Return a lazy voxel view for a patch, reusing valid eager metadata when present."""
    point_count = int(len(patch.points))
    metadata = patch.metadata if isinstance(patch.metadata, dict) else {}
    cached_voxel_size = metadata.get("voxel_size")
    cache_matches = False
    try:
        cache_matches = bool(
            cached_voxel_size is not None
            and np.isclose(float(cached_voxel_size), float(voxel_size), rtol=1e-6, atol=1e-9)
        )
    except (TypeError, ValueError):
        cache_matches = False

    if cache_matches:
        try:
            voxel_indices = np.asarray(metadata.get("voxel_indices"), dtype=np.int64)
            unique_voxels = np.asarray(metadata.get("unique_voxel_indices"), dtype=np.int64)
            representative_indices = np.asarray(metadata.get("representative_point_indices"), dtype=np.int64)
        except (TypeError, ValueError):
            voxel_indices = np.empty((0, 3), dtype=np.int64)
            unique_voxels = np.empty((0, 3), dtype=np.int64)
            representative_indices = np.zeros(0, dtype=np.int64)
        valid_shapes = (
            voxel_indices.shape == (point_count, 3)
            and unique_voxels.ndim == 2
            and unique_voxels.shape[1:] == (3,)
            and representative_indices.ndim == 1
            and len(representative_indices) == len(unique_voxels)
            and bool(np.all((representative_indices >= 0) & (representative_indices < max(point_count, 1))))
        )
        if (
            valid_shapes
            and _patch_voxel_cache_fingerprint_matches(patch, metadata)
            and _patch_voxel_cache_voxels_match(patch, voxel_indices, voxel_size)
            and _patch_voxel_cache_unique_matches(voxel_indices, unique_voxels, representative_indices)
        ):
            inverse = None
            if need_inverse:
                inverse = _compute_inverse_from_unique(voxel_indices, unique_voxels)
            return PatchVoxelView(
                voxel_size=float(voxel_size),
                voxel_indices=voxel_indices,
                unique_voxels=unique_voxels,
                representative_point_indices=representative_indices,
                inverse=inverse,
                cache_used=True,
            )

    voxel_indices = _world_to_voxel(patch.points, voxel_size)
    if len(voxel_indices) == 0:
        unique_voxels = np.empty((0, 3), dtype=np.int64)
        representative_indices = np.zeros(0, dtype=np.int64)
        inverse = np.zeros(0, dtype=np.int64) if need_inverse else None
    else:
        if need_inverse:
            unique_voxels, representative_indices, inverse = np.unique(
                voxel_indices,
                axis=0,
                return_index=True,
                return_inverse=True,
            )
            inverse = inverse.astype(np.int64, copy=False)
        else:
            unique_voxels, representative_indices = np.unique(voxel_indices, axis=0, return_index=True)
            inverse = None
        representative_indices = representative_indices.astype(np.int64, copy=False)
    return PatchVoxelView(
        voxel_size=float(voxel_size),
        voxel_indices=voxel_indices,
        unique_voxels=unique_voxels.astype(np.int64, copy=False),
        representative_point_indices=representative_indices,
        inverse=inverse,
        cache_used=False,
    )


def filter_patch_voxel_view(view: PatchVoxelView, keep_mask: np.ndarray) -> PatchVoxelView | None:
    """Filter a voxel view with a point-level mask."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if len(keep_mask) != len(view.voxel_indices) or not np.any(keep_mask):
        return None

    voxel_indices = view.voxel_indices[keep_mask]
    if len(voxel_indices) == 0:
        unique_voxels = np.empty((0, 3), dtype=np.int64)
        representative_indices = np.zeros(0, dtype=np.int64)
        inverse = np.zeros(0, dtype=np.int64) if view.inverse is not None else None
    else:
        unique_voxels, representative_indices, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        representative_indices = representative_indices.astype(np.int64, copy=False)
        inverse = inverse.astype(np.int64, copy=False) if view.inverse is not None else None

    return PatchVoxelView(
        voxel_size=float(view.voxel_size),
        voxel_indices=voxel_indices.astype(np.int64, copy=False),
        unique_voxels=unique_voxels.astype(np.int64, copy=False),
        representative_point_indices=representative_indices,
        inverse=inverse,
        cache_used=False,
    )


def patch_cached_voxels(
    patch: Patch3D,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return per-point voxels, unique voxels, representative point indices, cache_used."""
    view = get_patch_voxel_view(patch, voxel_size)
    return view.voxel_indices, view.unique_voxels, view.representative_point_indices, view.cache_used


def _compute_inverse_from_unique(voxel_indices: np.ndarray, unique_voxels: np.ndarray) -> np.ndarray:
    if len(voxel_indices) == 0:
        return np.zeros(0, dtype=np.int64)
    lookup = {
        (int(voxel[0]), int(voxel[1]), int(voxel[2])): int(index)
        for index, voxel in enumerate(unique_voxels)
    }
    return np.array(
        [lookup[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] for voxel in voxel_indices],
        dtype=np.int64,
    )


def _patch_voxel_cache_voxels_match(
    patch: Patch3D,
    cached_voxel_indices: np.ndarray,
    voxel_size: float,
) -> bool:
    """Reject stale cache when points changed without changing count or bbox."""
    current_voxels = _world_to_voxel(patch.points, voxel_size)
    cached_voxel_indices = np.asarray(cached_voxel_indices, dtype=np.int64)
    return bool(
        current_voxels.shape == cached_voxel_indices.shape
        and np.array_equal(current_voxels, cached_voxel_indices)
    )


def _patch_voxel_cache_unique_matches(
    voxel_indices: np.ndarray,
    cached_unique_voxels: np.ndarray,
    cached_representative_indices: np.ndarray,
) -> bool:
    if len(voxel_indices) == 0:
        return len(cached_unique_voxels) == 0 and len(cached_representative_indices) == 0
    unique_voxels, representative_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return bool(
        np.array_equal(unique_voxels.astype(np.int64, copy=False), cached_unique_voxels)
        and np.array_equal(representative_indices.astype(np.int64, copy=False), cached_representative_indices)
    )


def _patch_voxel_cache_fingerprint_matches(patch: Patch3D, metadata: Dict[str, Any]) -> bool:
    """Reject stale cached voxels after in-place point mutation."""
    points = np.asarray(patch.points, dtype=np.float32)
    cached_count = metadata.get("voxel_cache_point_count")
    if cached_count is not None:
        try:
            if int(cached_count) != int(len(points)):
                return False
        except (TypeError, ValueError):
            return False
    if len(points) == 0:
        return True

    cached_bbox_min = metadata.get("voxel_cache_bbox_min")
    cached_bbox_max = metadata.get("voxel_cache_bbox_max")
    if cached_bbox_min is None or cached_bbox_max is None:
        return True
    try:
        bbox_min = np.asarray(cached_bbox_min, dtype=np.float32)
        bbox_max = np.asarray(cached_bbox_max, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        return False
    return bool(
        np.allclose(bbox_min, points.min(axis=0), rtol=1e-5, atol=1e-6)
        and np.allclose(bbox_max, points.max(axis=0), rtol=1e-5, atol=1e-6)
    )


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
        voxel_view: PatchVoxelView | None = None,
    ) -> None:
        """Integrate a patch into the global TSDF and mark voxel ownership.

        Args:
            volume: The global TSDF instance volume.
            patch: 3D patch to integrate.
            instance_id: The instance that owns this patch.
        """
        self._ensure_support_index(volume)
        view = voxel_view or get_patch_voxel_view(patch, volume.voxel_size)
        unique_voxels = view.unique_voxels

        for voxel in unique_voxels:
            vk = _voxel_key(voxel)
            # TSDF integration (simplified: binary occupancy + weight)
            volume.tsdf[vk] = volume.tsdf.get(vk, 0.0) * 0.5 + 0.5 * 0.0  # near-surface
            volume.weight[vk] = volume.weight.get(vk, 0.0) + 1.0

            # Owner support accumulation (no hard overwrite)
            if vk not in volume.owner_support:
                volume.owner_support[vk] = VoxelOwnerSupport()
            support = volume.owner_support[vk]

            before = dict(support.support)
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
            self._sync_voxel_support_index(volume, vk, before, support.support)

    def remove_patch_support(
        self,
        volume: TSDFInstanceVolume,
        patch: Patch3D,
        instance_id: int,
        voxel_view: PatchVoxelView | None = None,
    ) -> None:
        """Remove one patch's owner support while preserving occupancy and weight."""
        self._ensure_support_index(volume)
        view = voxel_view or get_patch_voxel_view(patch, volume.voxel_size)
        unique_voxels = view.unique_voxels

        for voxel in unique_voxels:
            vk = _voxel_key(voxel)
            support = volume.owner_support.get(vk)
            if support is None or instance_id not in support.support:
                continue
            before = dict(support.support)
            support.support[instance_id] = (
                support.support.get(instance_id, 0.0) - volume.support_increment
            )
            if support.support[instance_id] <= 0.01:
                del support.support[instance_id]
            if not support.support:
                del volume.owner_support[vk]
                self._sync_voxel_support_index(volume, vk, before, {})
            else:
                self._sync_voxel_support_index(volume, vk, before, support.support)

    def vote_patch_to_instance(
        self,
        volume: TSDFInstanceVolume,
        patch: Patch3D,
        voxel_view: PatchVoxelView | None = None,
    ) -> VoxelVoteResult:
        """Spatial voting: which existing instances own the voxels this patch touches.

        Args:
            volume: The global TSDF instance volume.
            patch: 3D patch to query.

        Returns:
            VoxelVoteResult with all voting metadata exposed.
        """
        view = voxel_view or get_patch_voxel_view(patch, volume.voxel_size)
        unique_voxels = view.unique_voxels

        owner_votes: Dict[int, int] = {}
        supported_touched = 0

        self._ensure_support_index(volume)
        for voxel in unique_voxels:
            vk = _voxel_key(voxel)
            owner = get_voxel_owner_id(volume, vk)
            if owner < 0:
                continue
            supported_touched += 1
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

        self._ensure_support_index(volume)
        stale_owner_keys = [
            vk
            for vk, oid in volume.voxel_owner_id.items()
            if vk not in volume.owner_support or int(volume.owner_support[vk].owner_id) != int(oid)
        ]
        if stale_owner_keys:
            self._rebuild_support_indexes(volume)
        for vk, oid in volume.voxel_owner_id.items():
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
        self._ensure_support_index(volume)
        return int(len(volume.instance_owned_support.get(int(instance_id), {})))

    def summarize_instance_support(
        self,
        volume: TSDFInstanceVolume,
        instance_id: int,
    ) -> Dict[str, float]:
        """Summarize owner-support statistics for one instance."""
        self._ensure_support_index(volume)
        owned_support = volume.instance_owned_support.get(int(instance_id), {})
        owner_values = [float(value) for value in owned_support.values()]
        owned_voxel_count = len(owner_values)
        support_mass = float(sum(owner_values))
        mean_owner_support = float(support_mass / max(owned_voxel_count, 1))
        max_owner_support = float(max(owner_values)) if owner_values else 0.0
        competing_support_mass = float(volume.instance_competing_support_mass.get(int(instance_id), 0.0))

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

    def _ensure_support_index(self, volume: TSDFInstanceVolume) -> None:
        if self._support_index_missing(volume):
            self._rebuild_support_indexes(volume)

    @staticmethod
    def _get_cached_owner_arrays(volume: TSDFInstanceVolume) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (keys_1d_sorted, vals_sorted, sorter) for fast vectorized owner lookup.

        Cache is rebuilt when owner_index_revision changes."""
        current_rev = int(getattr(volume, "owner_index_revision", 0))
        cached_rev = int(getattr(volume, "_owner_cache_revision", -1))
        if cached_rev == current_rev and hasattr(volume, "_owner_keys_sorted"):
            return volume._owner_keys_sorted, volume._owner_vals_sorted, volume._owner_sorter

        d = getattr(volume, "voxel_owner_id", {})
        if not d:
            volume._owner_keys_sorted = np.array([], dtype=np.int64)
            volume._owner_vals_sorted = np.array([], dtype=np.int32)
            volume._owner_sorter = np.array([], dtype=np.int64)
        else:
            items = list(d.items())
            voxels = np.array([k for k, _ in items], dtype=np.int64)
            mask = (1 << 21) - 1
            k1d = ((voxels[:, 0] & mask) << 42) | ((voxels[:, 1] & mask) << 21) | (voxels[:, 2] & mask)
            vals = np.array([int(v) for _, v in items], dtype=np.int32)
            sorter = np.argsort(k1d)
            volume._owner_keys_sorted = k1d[sorter]
            volume._owner_vals_sorted = vals[sorter]
            volume._owner_sorter = sorter
        volume._owner_cache_revision = int(current_rev)
        return volume._owner_keys_sorted, volume._owner_vals_sorted, volume._owner_sorter

    @staticmethod
    def _support_index_missing(volume: TSDFInstanceVolume) -> bool:
        if not bool(getattr(volume, "support_index_valid", False)):
            return True
        if not hasattr(volume, "voxel_owner_id"):
            return True
        owner_index = getattr(volume, "voxel_owner_id", {})
        if volume.owner_support and not owner_index:
            return True
        return False

    @staticmethod
    def _rebuild_support_indexes(volume: TSDFInstanceVolume) -> None:
        previous_owner_id = dict(getattr(volume, "voxel_owner_id", {}))
        volume.instance_owned_support = {}
        volume.instance_competing_support_mass = {}
        volume.voxel_owner_id = {}
        for vk, support in volume.owner_support.items():
            TSDFInstanceMapModule._add_voxel_support_to_indexes(volume, vk, support.support)
        volume.support_index_valid = True
        if previous_owner_id != volume.voxel_owner_id:
            volume.owner_index_revision = int(getattr(volume, "owner_index_revision", 0)) + 1

    def _sync_voxel_support_index(
        self,
        volume: TSDFInstanceVolume,
        voxel_key: tuple[int, int, int],
        before_support: Dict[int, float],
        after_support: Dict[int, float],
    ) -> None:
        before = {int(oid): float(value) for oid, value in before_support.items()}
        after = {int(oid): float(value) for oid, value in after_support.items()}
        if not getattr(volume, "support_index_valid", False):
            if before:
                return
            volume.support_index_valid = True

        before_owner = max(before, key=before.get) if before else -1
        after_owner = max(after, key=after.get) if after else -1
        if before_owner != after_owner:
            if after_owner >= 0:
                volume.voxel_owner_id[voxel_key] = int(after_owner)
            else:
                volume.voxel_owner_id.pop(voxel_key, None)
            volume.owner_index_revision = int(getattr(volume, "owner_index_revision", 0)) + 1
        touched_ids = set(before) | set(after)
        self._sync_support_summary_index(volume, voxel_key, before, after, before_owner, after_owner, touched_ids)

    @staticmethod
    def _add_voxel_support_to_indexes(
        volume: TSDFInstanceVolume,
        voxel_key: tuple[int, int, int],
        support: Dict[int, float],
    ) -> None:
        after = {int(oid): float(value) for oid, value in support.items()}
        after_owner = max(after, key=after.get) if after else -1
        if after_owner >= 0:
            volume.voxel_owner_id[voxel_key] = int(after_owner)
        TSDFInstanceMapModule._sync_support_summary_index(
            volume,
            voxel_key,
            {},
            after,
            -1,
            after_owner,
            set(after),
        )

    @staticmethod
    def _sync_support_summary_index(
        volume: TSDFInstanceVolume,
        voxel_key: tuple[int, int, int],
        before: Dict[int, float],
        after: Dict[int, float],
        before_owner: int,
        after_owner: int,
        touched_ids: set[int],
    ) -> None:
        for oid in touched_ids:
            owned = volume.instance_owned_support.setdefault(int(oid), {})
            before_value = float(before.get(oid, 0.0))
            after_value = float(after.get(oid, 0.0))

            if before_owner == oid and before_value > 0.0:
                owned.pop(voxel_key, None)
            elif before_value > 0.0:
                volume.instance_competing_support_mass[int(oid)] = max(
                    0.0,
                    float(volume.instance_competing_support_mass.get(int(oid), 0.0)) - before_value,
                )

            if after_owner == oid and after_value > 0.0:
                owned[voxel_key] = after_value
            elif after_value > 0.0:
                volume.instance_competing_support_mass[int(oid)] = (
                    float(volume.instance_competing_support_mass.get(int(oid), 0.0)) + after_value
                )

            if not owned:
                volume.instance_owned_support.pop(int(oid), None)
            if float(volume.instance_competing_support_mass.get(int(oid), 0.0)) <= 1e-9:
                volume.instance_competing_support_mass.pop(int(oid), None)
