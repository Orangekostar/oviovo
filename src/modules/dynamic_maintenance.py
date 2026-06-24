"""Module 10: Dynamic Maintenance.

Responsibility: TSDF-driven lifecycle maintenance including
moved-object detection, disappearance judgment, new-object candidate
promotion, ghost cleanup, and boundary repair / background reclaim hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.core.data_structures import (
    BackgroundMap,
    CandidateEvidenceEntry,
    CandidateEvidenceGrid,
    ObjectMap,
    ObjectState,
    Patch3D,
    SystemState,
    TSDFInstanceVolume,
)
from src.modules.tsdf_instance_map import TSDFInstanceMapModule

logger = logging.getLogger("oviovo.modules.dynamic_maintenance")


class DynamicMaintenanceModule:
    """Lifecycle-aware dynamic maintenance using TSDF-driven spatial judgments."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.check_interval = config.get("lifecycle_check_interval", 10)
        self.tsdf_module = TSDFInstanceMapModule(config.get("tsdf", {}))
        logger.info("DynamicMaintenanceModule initialized (TSDF-driven).")

    def process(self, state: SystemState) -> SystemState:
        """Run TSDF-driven maintenance: MOVED, DISAPPEARED, NEW OBJECT.

        Only runs at check_interval frequency.

        Args:
            state: Current system state.

        Returns:
            Updated system state.
        """
        if state.frame_count % self.check_interval != 0:
            return state

        volume = state.tsdf_volume
        config = self.config

        # Track objects to remove after iteration
        to_ghost: List[int] = []
        removed_ids: List[int] = []

        for obj_id, obj in state.objects.items():
            if obj.state == ObjectState.REMOVED:
                removed_ids.append(obj_id)
                continue

            if obj.state == ObjectState.GHOST:
                # Fully clean TSDF ownership for ghost objects
                self._clear_object_tsdf_support(volume, obj)
                removed_ids.append(obj_id)
                logger.debug(f"Object {obj_id}: GHOST -> REMOVED")
                continue

            if obj.state != ObjectState.ACTIVE:
                continue

            # --- Three-tier TSDF-driven judgment ---

            # 1. MOVED check (runs first — mutex with DISAPPEARED)
            if self._check_moved(obj, volume, config):
                if len(obj.local_pcd) > 0:
                    # ObjectMap satisfies remove_patch_support's duck-type
                    # requirement for Patch3D via its .points property
                    # (backward-compatible alias for local_pcd).
                    self.tsdf_module.remove_patch_support(
                        volume, obj, obj.object_id
                    )
                obj.state = ObjectState.GHOST
                to_ghost.append(obj_id)
                logger.info(
                    f"Object {obj_id}: MOVED detected, marking GHOST "
                    f"(shift={np.linalg.norm(np.array(obj.centroid) - np.array(obj.creation_centroid)):.2f}m)"
                )
                continue

            # 2. DISAPPEARED check
            if self._check_disappeared(obj, volume, state, config):
                if len(obj.local_pcd) > 0:
                    # ObjectMap satisfies remove_patch_support's duck-type
                    # requirement for Patch3D via its .points property
                    # (backward-compatible alias for local_pcd).
                    self.tsdf_module.remove_patch_support(
                        volume, obj, obj.object_id
                    )
                obj.state = ObjectState.GHOST
                to_ghost.append(obj_id)
                # Log ownership ratio for debugging
                footprint = self._compute_spatial_footprint(obj, volume)
                ratio = footprint["current_voxels"] / max(obj.peak_voxel_count, 1)
                logger.info(
                    f"Object {obj_id}: DISAPPEARED (ownership_ratio={ratio:.2f}), marking GHOST"
                )
                continue

        # 3. NEW OBJECT check — scan CandidateEvidenceGrid
        candidates = self._cluster_and_check_new_objects(state, volume, config)
        for candidate in candidates:
            # Create a provisional object from candidate evidence
            from src.core.data_structures import ProvisionalObject

            provisional = ProvisionalObject(
                provisional_id=state.next_provisional_id,
                anchor_class_name=candidate["label"],
                anchor_confidence=float(candidate["score"]),
                centroid=np.asarray(candidate["centroid"], dtype=np.float32),
                first_seen_frame=state.frame_count,
                last_seen_frame=state.frame_count,
                hit_count=1,
            )
            state.provisional_objects[provisional.provisional_id] = provisional
            state.next_provisional_id += 1
            logger.info(
                f"NEW OBJECT candidate promoted to provisional: "
                f"label={candidate['label']} score={candidate['score']:.2f}"
            )

        # 4. Prune expired candidate evidence entries
        self._prune_expired_candidates(state, config)

        # 5. Remove REMOVED objects from dict
        for obj_id in removed_ids:
            if obj_id in state.objects:
                del state.objects[obj_id]

        logger.debug(
            f"Maintenance complete at frame {state.frame_count}. "
            f"Active: {sum(1 for o in state.objects.values() if o.state == ObjectState.ACTIVE)}, "
            f"Inactive: {sum(1 for o in state.objects.values() if o.state == ObjectState.INACTIVE)}, "
            f"Ghost: {len(to_ghost)}, "
            f"Removed: {len(removed_ids)}"
        )
        return state

    def _clear_object_tsdf_support(
        self, volume: TSDFInstanceVolume, obj: ObjectMap
    ) -> None:
        """Fully remove all TSDF owner_support entries for a given object.

        Scans all voxels in owner_support and removes any belonging to the object.
        Cleanup voxels whose support dict becomes empty.
        """
        to_delete: List[Tuple[int, int, int]] = []
        for vk, support in volume.owner_support.items():
            if obj.object_id in support.support:
                del support.support[obj.object_id]
            if not support.support:
                to_delete.append(vk)
        for vk in to_delete:
            del volume.owner_support[vk]

    # ------------------------------------------------------------------
    # CandidateEvidenceGrid helpers
    # ------------------------------------------------------------------

    def _prune_expired_candidates(
        self, state: SystemState, config: Dict[str, Any]
    ) -> None:
        """Remove CandidateEvidenceGrid entries that haven't been updated
        within candidate_max_idle_frames."""
        max_idle = int(config.get("candidate_max_idle_frames", 60))
        current_frame = state.frame_count
        expired = [
            vk
            for vk, entry in state.candidate_evidence.entries.items()
            if current_frame - entry.last_seen_frame > max_idle
        ]
        for vk in expired:
            del state.candidate_evidence.entries[vk]
        if expired:
            logger.debug(f"Pruned {len(expired)} expired candidate evidence entries.")

    def _cluster_candidate_entries(
        self,
        entries: Dict[Tuple[int, int, int], CandidateEvidenceEntry],
        voxel_size: float,
        cluster_radius: float,
    ) -> List[List[Tuple[int, int, int]]]:
        """Cluster candidate evidence entries by spatial proximity.

        Uses simple single-linkage: entries within cluster_radius are merged.

        Returns:
            List of clusters, each cluster is a list of voxel keys.
        """
        if not entries:
            return []

        voxel_keys = list(entries.keys())
        # Convert to world coordinates
        positions = {
            vk: (np.array(vk, dtype=np.float64) + 0.5) * voxel_size
            for vk in voxel_keys
        }

        # Union-find based clustering
        parent = {vk: vk for vk in voxel_keys}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Check pairs within cluster_radius
        n = len(voxel_keys)
        for i in range(n):
            for j in range(i + 1, n):
                vk_i = voxel_keys[i]
                vk_j = voxel_keys[j]
                dist = float(np.linalg.norm(positions[vk_i] - positions[vk_j]))
                if dist <= cluster_radius:
                    union(vk_i, vk_j)

        # Collect clusters
        cluster_map: Dict[Tuple, List[Tuple[int, int, int]]] = {}
        for vk in voxel_keys:
            root = find(vk)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(vk)

        return list(cluster_map.values())

    def _cluster_and_check_new_objects(
        self,
        state: SystemState,
        volume: TSDFInstanceVolume,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Scan CandidateEvidenceGrid for promotable new object candidates.

        Clusters entries, aggregates label votes, verifies background,
        and returns list of promotion candidates.

        Returns:
            List of dicts with keys: label, score, centroid, voxel_keys
        """
        grid = state.candidate_evidence
        if not grid.entries:
            return []

        cluster_radius = config.get("candidate_cluster_radius", 0.3)
        promotion_threshold = config.get("candidate_promotion_threshold", 3.0)

        clusters = self._cluster_candidate_entries(
            grid.entries, grid.voxel_size, cluster_radius
        )

        promoted: List[Dict[str, Any]] = []

        for cluster in clusters:
            # Aggregate label votes across cluster
            total_votes: Dict[str, float] = {}
            for vk in cluster:
                entry = grid.entries[vk]
                for label, score in entry.label_votes.items():
                    label_norm = str(label).strip().lower()
                    total_votes[label_norm] = total_votes.get(label_norm, 0.0) + score

            if not total_votes:
                continue

            best_label = max(total_votes, key=total_votes.get)
            best_score = total_votes[best_label]

            if best_score < promotion_threshold:
                continue

            # Verify target voxels are still background
            all_background = True
            for vk in cluster:
                support = volume.owner_support.get(vk)
                if support is not None and support.owner_id >= 0:
                    all_background = False
                    break

            if not all_background:
                continue

            # Compute cluster centroid
            positions = [
                (np.array(vk, dtype=np.float64) + 0.5) * grid.voxel_size
                for vk in cluster
            ]
            centroid = np.mean(positions, axis=0)

            promoted.append({
                "label": best_label,
                "score": best_score,
                "centroid": centroid.astype(np.float32),
                "voxel_keys": cluster,
            })

            # Clean up promoted entries from grid
            for vk in cluster:
                if vk in grid.entries:
                    del grid.entries[vk]

        return promoted

    # ------------------------------------------------------------------
    # TSDF spatial footprint query
    # ------------------------------------------------------------------

    def _compute_spatial_footprint(
        self, obj: ObjectMap, volume: TSDFInstanceVolume
    ) -> Dict[str, Any]:
        """Compute the spatial footprint of an object in the TSDF volume.

        Iterates over obj.local_pcd points, queries TSDF voxel ownership,
        and counts how many voxels are still owned by this object.

        Args:
            obj: The object to query.
            volume: The global TSDF instance volume.

        Returns:
            Dict with:
                current_voxels: number of unique voxels owned by this object
                owned_centroid: weighted centroid of owned voxels
                total_queried: number of unique voxels the object's points touched
        """
        if len(obj.local_pcd) == 0:
            return {"current_voxels": 0, "owned_centroid": np.zeros(3, dtype=np.float32), "total_queried": 0}

        points = np.asarray(obj.local_pcd, dtype=np.float32)
        voxel_indices = np.floor(points / volume.voxel_size).astype(np.int64)
        unique_voxels: Set[Tuple[int, int, int]] = set()
        owned_positions: List[np.ndarray] = []

        for i, vk_array in enumerate(voxel_indices):
            vk = (int(vk_array[0]), int(vk_array[1]), int(vk_array[2]))
            if vk in unique_voxels:
                continue
            unique_voxels.add(vk)
            support = volume.owner_support.get(vk)
            if support is not None and support.owner_id == obj.object_id:
                owned_positions.append(points[i])

        current_voxels = len(owned_positions)
        if current_voxels > 0:
            owned_centroid = np.mean(np.stack(owned_positions, axis=0), axis=0)
        else:
            owned_centroid = np.zeros(3, dtype=np.float32)

        return {
            "current_voxels": current_voxels,
            "owned_centroid": owned_centroid,
            "total_queried": len(unique_voxels),
        }

    # ------------------------------------------------------------------
    # TSDF disappearance judgment
    # ------------------------------------------------------------------

    def _check_disappeared(
        self,
        obj: ObjectMap,
        volume: TSDFInstanceVolume,
        state: SystemState,
        config: Dict[str, Any],
    ) -> bool:
        """Check if an object has disappeared from its spatial footprint.

        Returns True if the object should be marked GHOST.
        """
        # Grace period: skip judgment for newly created objects
        frames_since_creation = state.frame_count - obj.creation_frame
        if frames_since_creation < config.get("creation_grace_frames", 5):
            return False

        if obj.peak_voxel_count == 0:
            return False

        footprint = self._compute_spatial_footprint(obj, volume)
        current_voxels = footprint["current_voxels"]
        ownership_ratio = current_voxels / obj.peak_voxel_count

        disappear_threshold = config.get("disappear_threshold", 0.2)

        if ownership_ratio < disappear_threshold:
            if not self._has_matching_proposals_in_region(state, obj):
                return True

        return False

    def _has_matching_proposals_in_region(
        self, state: SystemState, obj: ObjectMap
    ) -> bool:
        """Check if there are unmatched proposals matching this object's class in its region.

        Conservative check to prevent false GHOST when an object is
        heavily occluded but still present.

        Uses state.active_set.new_object_candidate_ids to find unmatched proposals.
        """
        # Get the object's canonical label from semantic memory
        canonical_label = ""
        if obj.semantic_memory.label_hypotheses:
            canonical_label = str(obj.semantic_memory.label_hypotheses[0][0]).strip().lower()
        if not canonical_label:
            return False

        # Check provisional objects in the provisional pool
        for provisional in state.provisional_objects.values():
            prov_label = str(provisional.anchor_class_name).strip().lower()
            if not prov_label or prov_label != canonical_label:
                continue
            # Check spatial overlap: provisional centroid near object bbox
            dist = float(np.linalg.norm(np.array(provisional.centroid) - np.array(obj.centroid)))
            obj_radius = float(np.linalg.norm(np.array(obj.bbox_max) - np.array(obj.bbox_min))) / 2.0
            if dist < max(obj_radius, 0.5):
                return True

        # Check current-frame unmatched proposals from active_set
        for candidate_id in state.active_set.new_object_candidate_ids:
            candidate_obj = state.objects.get(candidate_id)
            if candidate_obj is None:
                continue
            cand_label = ""
            if candidate_obj.semantic_memory.label_hypotheses:
                cand_label = str(candidate_obj.semantic_memory.label_hypotheses[0][0]).strip().lower()
            if not cand_label or cand_label != canonical_label:
                continue
            dist = float(np.linalg.norm(np.array(candidate_obj.centroid) - np.array(obj.centroid)))
            obj_radius = float(np.linalg.norm(np.array(obj.bbox_max) - np.array(obj.bbox_min))) / 2.0
            if dist < max(obj_radius, 0.5):
                return True

        return False

    # ------------------------------------------------------------------
    # TSDF moved-object detection via centroid-shift + direction search
    # ------------------------------------------------------------------

    def _check_moved(
        self,
        obj: ObjectMap,
        volume: TSDFInstanceVolume,
        config: Dict[str, Any],
    ) -> bool:
        """Check if an object has moved by sampling TSDF ownership along
        the direction from creation_centroid to current_centroid.

        Returns True if ownership has migrated to the new half of the path.
        """
        threshold = config.get("moved_centroid_threshold", 0.3)
        search_steps = int(config.get("moved_search_steps", 10))
        ownership_skew = config.get("moved_ownership_skew", 0.6)

        old_centroid = np.asarray(obj.creation_centroid, dtype=np.float32)
        new_centroid = np.asarray(obj.centroid, dtype=np.float32)

        shift_vector = new_centroid - old_centroid
        shift_distance = float(np.linalg.norm(shift_vector))

        if shift_distance < threshold:
            return False

        direction = shift_vector / shift_distance

        old_half_count = 0
        new_half_count = 0
        half = search_steps // 2

        for step in range(search_steps):
            t = step / (search_steps - 1)
            sample_point = old_centroid + direction * shift_distance * t
            vk_array = np.floor(sample_point / volume.voxel_size).astype(np.int64)
            vk = (int(vk_array[0]), int(vk_array[1]), int(vk_array[2]))
            support = volume.owner_support.get(vk)
            if support is not None and support.owner_id == obj.object_id:
                if step < half:
                    old_half_count += 1
                else:
                    new_half_count += 1

        total_owned = old_half_count + new_half_count
        if total_owned == 0:
            return False

        return (new_half_count / total_owned) >= ownership_skew

    # ------------------------------------------------------------------
    # Placeholder hooks for future implementation
    # ------------------------------------------------------------------

    def _boundary_repair(self, state: SystemState) -> None:
        """Repair object boundaries contaminated by background geometry.

        TODO: Detect points in object clouds that are far from object centroid
        and overlap with the background map. Remove or reassign them.
        """
        pass

    def _background_reclaim(self, state: SystemState) -> None:
        """Reclaim background points that were incorrectly assigned to objects.

        TODO: For removed/ghost objects, merge their points back into the
        background map if geometry is consistent.
        """
        pass

    def _re_identification(self, state: SystemState) -> None:
        """Re-identify dormant objects that reappear.

        TODO: Compare new object patches against DORMANT objects using
        spatial + semantic similarity. If match found, reactivate.
        """
        pass
