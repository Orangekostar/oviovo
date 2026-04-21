"""Module 6: Object Association (Layer 4 — spatial voting).

Responsibility: Associate object patches to existing object instances
using TSDF-based spatial voting. Association is geometry-first /
spatial-first — semantics must not drive low-level association (Rule C).

Exposes explicit voting scores:
  - touched voxels, owner votes, normalized vote score
  - optional local geometry consistency
  - final association score
  - whether a new instance is created
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import (
    ActiveSet,
    AssociationResult,
    AssociationScore,
    ObjectMap,
    Patch3D,
    TSDFInstanceVolume,
    VoxelVoteResult,
)
from src.modules.tsdf_instance_map import TSDFInstanceMapModule
from src.utils.geometry import bbox_iou_3d

logger = logging.getLogger("oviovo.modules.association")


class AssociationModule:
    """Geometry-first association of patches to existing objects.

    Uses TSDF spatial voting as primary signal, with optional
    centroid/bbox/geometry fallback for objects not yet in TSDF.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.w_voxel_vote = config.get("voxel_vote_weight", 0.5)
        self.w_centroid = config.get("centroid_distance_weight", 0.2)
        self.w_bbox = config.get("bbox_overlap_weight", 0.2)
        self.w_geometry = config.get("geometry_overlap_weight", 0.1)
        self.match_threshold = config.get("match_threshold", 0.3)
        self.tsdf_module = TSDFInstanceMapModule(config.get("tsdf", {}))
        logger.info("AssociationModule initialized.")

    def process(
        self,
        object_patches: List[Patch3D],
        objects: Dict[int, ObjectMap],
        tsdf_volume: TSDFInstanceVolume,
        active_set: ActiveSet | None = None,
    ) -> AssociationResult:
        """Associate patches to existing objects via spatial voting.

        Args:
            object_patches: 3D patches classified as objects.
            objects: Current object map.
            tsdf_volume: Global TSDF instance volume for spatial voting.

        Returns:
            AssociationResult with matches and new-object requests.
        """
        result = AssociationResult()
        all_candidate_ids = [
            obj_id for obj_id, obj in objects.items()
            if obj.state.value != "removed"
        ]
        active_candidate_ids = self._resolve_candidate_ids(objects, active_set)
        use_active_set = active_set is not None
        result.debug = {
            "used_active_set": use_active_set,
            "active_set_candidate_ids": active_candidate_ids,
            "fallback_to_all_objects": False,
            "candidate_source": "active_set" if use_active_set else "global_object_map",
            "per_patch": {},
        }

        for patch in object_patches:
            # Step 1: TSDF spatial voting
            vote = self.tsdf_module.vote_patch_to_instance(tsdf_volume, patch)
            candidate_ids = active_candidate_ids if use_active_set else all_candidate_ids
            patch_debug = {
                "candidate_object_ids": list(candidate_ids),
                "vote": {
                    "touched_voxel_count": vote.touched_voxel_count,
                    "supported_voxel_count": vote.supported_voxel_count,
                    "owner_votes": dict(vote.owner_votes),
                    "best_instance_id": vote.best_instance_id,
                    "normalized_vote_score": vote.normalized_vote_score,
                },
            }

            best_score = None
            best_obj_id = None

            for obj_id in candidate_ids:
                obj = objects.get(obj_id)
                if obj is None or obj.state.value == "removed":
                    continue
                score = self._compute_score(patch, obj, vote)
                result.scores[(patch.patch_id, obj_id)] = score

                if score.total_score > self.match_threshold:
                    if best_score is None or score.total_score > best_score.total_score:
                        best_score = score
                        best_obj_id = obj_id

            if best_obj_id is not None:
                result.matched.append((patch.patch_id, best_obj_id, best_score))
                patch_debug["selected_object_id"] = best_obj_id
                patch_debug["final_association_score"] = float(best_score.total_score)
                patch_debug["new_instance_created"] = False
            else:
                result.new_object_patches.append(patch.patch_id)
                vote.new_instance_created = True
                patch_debug["selected_object_id"] = None
                patch_debug["final_association_score"] = 0.0
                patch_debug["new_instance_created"] = True

            result.debug["per_patch"][patch.patch_id] = patch_debug

        logger.debug(
            f"Association: {len(result.matched)} matched, "
            f"{len(result.new_object_patches)} new objects requested."
        )
        return result

    def _resolve_candidate_ids(
        self,
        objects: Dict[int, ObjectMap],
        active_set: ActiveSet | None,
    ) -> List[int]:
        """Resolve the local candidate set from the active set."""
        if active_set is None or not active_set.all_candidate_ids:
            return []

        return sorted(
            obj_id
            for obj_id in active_set.all_candidate_ids
            if obj_id in objects and objects[obj_id].state.value != "removed"
        )

    def _compute_score(
        self,
        patch: Patch3D,
        obj: ObjectMap,
        vote: VoxelVoteResult,
    ) -> AssociationScore:
        """Compute association score combining voxel voting + geometric fallback.

        Scoring is spatial-first. Semantics are NOT used here (Rule C).
        """
        # Voxel vote score for this specific object
        vote_denominator = max(vote.supported_voxel_count, 1)
        if vote.owner_votes and obj.object_id in vote.owner_votes:
            obj_votes = vote.owner_votes[obj.object_id]
            voxel_score = obj_votes / vote_denominator
        else:
            voxel_score = 0.0

        # Centroid distance score (inverse distance, normalized)
        dist = float(np.linalg.norm(patch.centroid - obj.centroid))
        centroid_score = max(0.0, 1.0 - dist / 2.0)

        # Bbox IoU
        bbox_score = bbox_iou_3d(patch.bbox_min, patch.bbox_max, obj.bbox_min, obj.bbox_max)

        # Local geometry consistency (nearest-neighbor distance)
        geometry_score = self._geometry_consistency(patch, obj)

        total = (
            self.w_voxel_vote * voxel_score
            + self.w_centroid * centroid_score
            + self.w_bbox * bbox_score
            + self.w_geometry * geometry_score
        )

        return AssociationScore(
            centroid_distance=centroid_score,
            bbox_overlap=bbox_score,
            geometry_overlap=geometry_score,
            voxel_vote_score=voxel_score,
            total_score=total,
            vote_result=VoxelVoteResult(
                touched_voxel_count=vote.touched_voxel_count,
                supported_voxel_count=vote.supported_voxel_count,
                owner_votes=dict(vote.owner_votes),
                normalized_vote_score=voxel_score,
                best_instance_id=vote.best_instance_id,
                geometry_consistency_score=geometry_score,
                final_association_score=total,
                new_instance_created=False,
            ),
        )

    def _geometry_consistency(self, patch: Patch3D, obj: ObjectMap) -> float:
        """Compute local geometry consistency between patch and object local_pcd.

        Uses mean nearest-neighbor distance as a proxy.
        """
        if len(obj.local_pcd) == 0 or len(patch.points) == 0:
            return 0.0

        # Subsample for efficiency
        max_pts = 200
        p_pts = patch.points
        o_pts = obj.local_pcd
        if len(p_pts) > max_pts:
            idx = np.random.choice(len(p_pts), max_pts, replace=False)
            p_pts = p_pts[idx]
        if len(o_pts) > max_pts:
            idx = np.random.choice(len(o_pts), max_pts, replace=False)
            o_pts = o_pts[idx]

        # Mean nearest-neighbor distance (patch -> object)
        diffs = p_pts[:, None, :] - o_pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        mean_nn_dist = float(min_dists.mean())

        # Convert to score: close = high score
        return float(max(0.0, 1.0 - mean_nn_dist / 0.5))
