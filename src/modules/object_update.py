"""Module 7: Object Update (Layers 4+5).

Responsibility:
  - Integrate patches into global TSDF (Layer 4)
  - Update voxel label support (accumulate/decay, no hard overwrite)
  - Maintain per-instance local_pcd as v1 object pool geometry (Layer 5)
  - Update soft whole-evidence scores (Rule B)
  - Trigger semantic update hook after association
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.core.data_structures import (
    AssociationResult,
    CameraIntrinsics,
    CandidateEvidenceEntry,
    CandidateEvidenceGrid,
    ObjectMap,
    ObjectState,
    ObservationRecord,
    Patch3D,
    ProvisionalObject,
    SemanticMemory,
    SystemState,
    TSDFInstanceVolume,
    WholeEvidenceScores,
)
from src.modules.semantic_memory import (
    accumulate_anchor_semantic_vote,
    rebuild_anchor_semantic_votes_from_observations,
)
from src.modules.tsdf_instance_map import TSDFInstanceMapModule
from src.modules.visibility_projector import filter_points_by_depth_consistency
from src.utils.geometry import compute_bbox, voxel_downsample

logger = logging.getLogger("oviovo.modules.object_update")


class ObjectUpdateModule:
    """Updates object geometry, TSDF ownership, and whole-evidence scores."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.downsample_voxel = config.get("downsample_voxel_size", 0.02)
        self.downsample_interval = config.get("downsample_interval", 5)
        self.max_points = int(config.get("max_points_per_object", 10000))
        association_geometry_cfg = config.get("association_geometry", {})
        self.association_geometry_enabled = bool(association_geometry_cfg.get("enabled", False))
        self.association_geometry_voxel = float(
            association_geometry_cfg.get(
                "voxel_size",
                config.get("association_geometry_voxel_size", max(float(self.downsample_voxel), 0.02)),
            )
        )
        self.association_geometry_max_points = int(
            association_geometry_cfg.get(
                "max_points_per_object",
                association_geometry_cfg.get(
                    "max_points",
                    config.get("association_geometry_max_points", min(self.max_points, 2048)),
                ),
            )
        )
        self.whole_evidence_growth = config.get("whole_evidence_growth", 0.05)
        self.part_evidence_growth = config.get("part_evidence_growth", 0.02)
        provisional_cfg = config.get("provisional_pool", {})
        self.provisional_enabled = bool(provisional_cfg.get("enabled", False))
        self.provisional_match_distance = float(provisional_cfg.get("match_distance", 0.35))
        self.provisional_promotion_hits = int(provisional_cfg.get("promotion_hits", 3))
        self.provisional_max_idle_frames = int(provisional_cfg.get("max_idle_frames", 30))
        self.provisional_downsample_voxel = float(
            provisional_cfg.get("downsample_voxel_size", self.downsample_voxel)
        )
        self.provisional_max_points = int(
            provisional_cfg.get("max_points_per_object", max(2000, self.max_points // 2))
        )
        self.provisional_unanchored_objectness_min = float(
            provisional_cfg.get("unanchored_objectness_min", 0.75)
        )
        self.provisional_unanchored_min_points = int(
            provisional_cfg.get("unanchored_min_points", 128)
        )
        contested_cfg = config.get("contested_residual_pool", {})
        self.contested_residual_enabled = bool(contested_cfg.get("enabled", self.provisional_enabled))
        self.last_contested_residual_patch_ids: list[int] = []
        self.last_contested_residual_promoted_object_ids: list[int] = []
        gate_cfg = config.get("surface_owner_gate", {})
        self.surface_owner_gate_enabled = bool(gate_cfg.get("enabled", False))
        self.surface_gate_representative_voxel_mode = bool(gate_cfg.get("representative_voxel_mode", False))
        self.surface_gate_background_enabled = bool(gate_cfg.get("background_enabled", True))
        self.surface_gate_min_accept_points = int(gate_cfg.get("min_accept_points", 20))
        self.surface_gate_min_update_accept_ratio = float(gate_cfg.get("min_update_accept_ratio", 0.30))
        self.surface_gate_min_new_object_accept_ratio = float(gate_cfg.get("min_new_object_accept_ratio", 0.55))
        self.surface_gate_max_foreign_owner_ratio = float(gate_cfg.get("max_foreign_owner_ratio", 0.25))
        self.surface_gate_max_background_owner_ratio = float(gate_cfg.get("max_background_owner_ratio", 0.35))
        self.surface_gate_attached_surface_classes = {
            str(label).strip()
            for label in gate_cfg.get("attached_surface_classes", ["switch", "wall-plug", "vent"])
            if str(label).strip()
        }
        self.surface_gate_attached_max_voxels = int(gate_cfg.get("attached_max_voxels", 40))
        self.surface_gate_self_background_min_score = float(gate_cfg.get("self_background_min_score", 0.65))
        self.surface_gate_self_background_margin = float(gate_cfg.get("self_background_margin", 0.15))
        visibility_cfg = config.get("current_frame_visibility_gate", {})
        self.current_frame_visibility_gate_enabled = bool(visibility_cfg.get("enabled", False))
        self.current_frame_visibility_distance_threshold = float(visibility_cfg.get("distance_threshold", 0.05))
        self.current_frame_visibility_min_accept_points = int(visibility_cfg.get("min_accept_points", 20))
        self.current_frame_visibility_min_accept_ratio = float(visibility_cfg.get("min_accept_ratio", 0.30))
        relocation_cfg = config.get("relocation_split_gate", {})
        self.relocation_split_gate_enabled = bool(relocation_cfg.get("enabled", False))
        self.relocation_split_centroid_distance = float(relocation_cfg.get("centroid_distance", 0.75))
        self.relocation_split_max_voxel_vote_score = float(relocation_cfg.get("max_voxel_vote_score", 0.15))
        refinement_cfg = config.get("async_refinement", {})
        if "replace_coarse_observations" in refinement_cfg:
            refinement_replace_enabled = refinement_cfg["replace_coarse_observations"]
        elif "replace_observations" in refinement_cfg:
            refinement_replace_enabled = refinement_cfg["replace_observations"]
        elif "enabled" in refinement_cfg:
            refinement_replace_enabled = refinement_cfg["enabled"]
        else:
            refinement_replace_enabled = True
        self.refinement_replace_enabled = bool(refinement_replace_enabled)
        self.refinement_rebuild_tsdf_support = bool(refinement_cfg.get("rebuild_tsdf_support", False))
        self.refinement_max_components_per_replacement = int(
            refinement_cfg.get("max_components_per_replacement", 1)
        )
        self.last_async_refinement_summary: dict[str, Any] = {}
        self.last_current_frame_visibility_gate_records: list[dict[str, Any]] = []
        self.last_current_frame_visibility_gate_stats: dict[str, Any] = {}
        self.tsdf_module = TSDFInstanceMapModule(config.get("tsdf", {}))
        self.last_structural_reject_patches: list[Patch3D] = []
        self.last_surface_gate_records: list[dict[str, Any]] = []
        self.last_surface_gate_stats: dict[str, Any] = {}
        self.last_updated_object_ids: list[int] = []
        self.last_created_object_ids: list[int] = []
        logger.info("ObjectUpdateModule initialized.")

    def process(
        self,
        association: AssociationResult,
        patches: List[Patch3D],
        state: SystemState,
        background_patches: List[Patch3D] | None = None,
        *,
        current_depth: np.ndarray | None = None,
        current_pose: np.ndarray | None = None,
        current_intrinsics: CameraIntrinsics | None = None,
    ) -> SystemState:
        """Update existing objects and create new ones.

        After association:
          1. Integrate patch into global TSDF (Layer 4)
          2. Update voxel support (accumulate/decay)
          3. Update local_pcd as pool geometry (Layer 5)
          4. Update soft whole_evidence scores (Rule B)

        Args:
            association: Result from the association module.
            patches: All object patches (needed to look up by patch_id).
            state: Current system state.

        Returns:
            Updated system state.
        """
        patch_map = {p.patch_id: p for p in patches}
        current_frame = int(max((p.source_frame_id for p in patches), default=state.frame_count))
        background_support = self._build_background_voxel_support(state, background_patches or [])
        self.last_structural_reject_patches = []
        self.last_surface_gate_records = []
        self.last_current_frame_visibility_gate_records = []
        self.last_updated_object_ids = []
        self.last_created_object_ids = []
        self.last_contested_residual_patch_ids = []
        self.last_contested_residual_promoted_object_ids = []
        contested_patch_ids = {int(patch_id) for patch_id in association.contested_object_patches}

        # Update matched objects
        for patch_id, obj_id, score in association.matched:
            if self.contested_residual_enabled and int(patch_id) in contested_patch_ids:
                continue
            patch = patch_map.get(patch_id)
            if patch is None:
                continue
            obj = state.objects.get(obj_id)
            if obj is None:
                continue
            patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                patch,
                current_depth=current_depth,
                current_pose=current_pose,
                current_intrinsics=current_intrinsics,
            )
            self._record_current_frame_visibility_gate_debug(visibility_debug)
            if patch is None:
                obj.debug["last_current_frame_visibility_gate"] = visibility_debug
                continue
            patch, structural_reject, gate_debug = self._filter_patch_by_surface_owner(
                patch,
                target_object_id=int(obj_id),
                state=state,
                background_support=background_support,
                new_object=False,
            )
            self._record_surface_gate_debug(gate_debug)
            if structural_reject is not None:
                self.last_structural_reject_patches.append(structural_reject)
            if patch is None:
                obj.debug["last_surface_owner_gate"] = gate_debug
                continue
            if self._should_split_relocated_update(obj, patch, score):
                new_obj = self._create_object(patch, state.next_object_id)
                new_obj.debug["created_by_relocation_split"] = {
                    "source_object_id": int(obj_id),
                    "source_patch_id": int(patch_id),
                    "centroid_distance": float(np.linalg.norm(np.asarray(patch.centroid) - np.asarray(obj.centroid))),
                    "voxel_vote_score": float(getattr(score, "voxel_vote_score", 0.0) or 0.0),
                }
                state.objects[new_obj.object_id] = new_obj
                self.tsdf_module.integrate_patch(state.tsdf_volume, patch, new_obj.object_id)
                self._refresh_object_debug(new_obj, state.tsdf_volume)
                self.last_created_object_ids.append(int(new_obj.object_id))
                self.last_updated_object_ids.append(int(new_obj.object_id))
                state.next_object_id += 1
                continue
            # Step 1: Integrate into global TSDF
            self.tsdf_module.integrate_patch(state.tsdf_volume, patch, obj_id)
            # Step 2+3: Update local geometry + whole evidence
            self._update_object(obj, patch)
            self._refresh_object_debug(obj, state.tsdf_volume)
            obj.debug["last_current_frame_visibility_gate"] = visibility_debug
            obj.debug["last_surface_owner_gate"] = gate_debug
            self.last_updated_object_ids.append(int(obj_id))

        if self.contested_residual_enabled:
            for patch_id in association.contested_object_patches:
                patch = patch_map.get(patch_id)
                if patch is None:
                    continue
                patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                    patch,
                    current_depth=current_depth,
                    current_pose=current_pose,
                    current_intrinsics=current_intrinsics,
                )
                self._record_current_frame_visibility_gate_debug(visibility_debug)
                if patch is None:
                    continue
                _original_patch = patch
                patch, structural_reject, gate_debug = self._filter_patch_by_surface_owner(
                    patch,
                    target_object_id=None,
                    state=state,
                    background_support=background_support,
                    new_object=True,
                    contested_residual=True,
                )
                self._record_surface_gate_debug(gate_debug)
                if structural_reject is not None:
                    self.last_structural_reject_patches.append(structural_reject)
                if patch is None:
                    if not gate_debug.get("passed", True):
                        self._write_candidate_evidence(state, _original_patch)
                    continue
                if not self._should_enter_provisional_pool(patch):
                    continue
                self._upsert_provisional_object(state, patch, contested=True)
                self.last_contested_residual_patch_ids.append(int(patch_id))
            if not self.provisional_enabled:
                self._promote_stable_provisionals(state)
                self._prune_stale_provisionals(state, current_frame)

        # Create new objects or accumulate them in the provisional local pool.
        if self.provisional_enabled:
            for patch_id in association.new_object_patches:
                if self.contested_residual_enabled and int(patch_id) in contested_patch_ids:
                    continue
                patch = patch_map.get(patch_id)
                if patch is None:
                    continue
                patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                    patch,
                    current_depth=current_depth,
                    current_pose=current_pose,
                    current_intrinsics=current_intrinsics,
                )
                self._record_current_frame_visibility_gate_debug(visibility_debug)
                if patch is None:
                    continue
                _original_patch = patch
                patch, structural_reject, gate_debug = self._filter_patch_by_surface_owner(
                    patch,
                    target_object_id=None,
                    state=state,
                    background_support=background_support,
                    new_object=True,
                )
                self._record_surface_gate_debug(gate_debug)
                if structural_reject is not None:
                    self.last_structural_reject_patches.append(structural_reject)
                if patch is None:
                    if not gate_debug.get("passed", True):
                        self._write_candidate_evidence(state, _original_patch)
                    continue
                if not self._should_enter_provisional_pool(patch):
                    continue
                self._upsert_provisional_object(state, patch)
            self._promote_stable_provisionals(state)
            self._prune_stale_provisionals(state, current_frame)
        else:
            for patch_id in association.new_object_patches:
                if self.contested_residual_enabled and int(patch_id) in contested_patch_ids:
                    continue
                patch = patch_map.get(patch_id)
                if patch is None:
                    continue
                patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                    patch,
                    current_depth=current_depth,
                    current_pose=current_pose,
                    current_intrinsics=current_intrinsics,
                )
                self._record_current_frame_visibility_gate_debug(visibility_debug)
                if patch is None:
                    continue
                _original_patch = patch
                patch, structural_reject, gate_debug = self._filter_patch_by_surface_owner(
                    patch,
                    target_object_id=None,
                    state=state,
                    background_support=background_support,
                    new_object=True,
                )
                self._record_surface_gate_debug(gate_debug)
                if structural_reject is not None:
                    self.last_structural_reject_patches.append(structural_reject)
                if patch is None:
                    if not gate_debug.get("passed", True):
                        self._write_candidate_evidence(state, _original_patch)
                    continue
                if self._is_ambiguous_patch(patch):
                    continue
                new_obj = self._create_object(patch, state.next_object_id)
                state.objects[new_obj.object_id] = new_obj
                # Integrate new object into global TSDF
                self.tsdf_module.integrate_patch(state.tsdf_volume, patch, new_obj.object_id)
                self._refresh_object_debug(new_obj, state.tsdf_volume)
                new_obj.debug["last_current_frame_visibility_gate"] = visibility_debug
                new_obj.debug["last_surface_owner_gate"] = gate_debug
                self.last_created_object_ids.append(int(new_obj.object_id))
                self.last_updated_object_ids.append(int(new_obj.object_id))
                state.next_object_id += 1

        self.last_surface_gate_stats = self._summarize_surface_gate_records()
        self.last_current_frame_visibility_gate_stats = self._summarize_current_frame_visibility_gate_records()
        logger.debug(f"Objects updated. Total objects: {len(state.objects)}")
        return state

    def replace_observations(
        self,
        state: SystemState,
        fine_patches: List[Patch3D],
    ) -> dict[str, Any]:
        """Replace coarse local-memory observations with refined observations."""
        summary: dict[str, Any] = {
            "enabled": bool(self.refinement_replace_enabled),
            "fine_patch_count": int(len(fine_patches)),
            "replaced_observation_count": 0,
            "inserted_observation_count": 0,
            "updated_object_count": 0,
            "updated_object_ids": [],
            "skipped_fine_patch_count": 0,
            "rebuild_tsdf_support": bool(self.refinement_rebuild_tsdf_support),
        }
        if not self.refinement_replace_enabled or not fine_patches:
            self.last_async_refinement_summary = summary
            return summary

        updated_ids: set[int] = set()
        prepared_fine_patches, skipped_fine_patch_count = self._prepare_refinement_patches_by_key(fine_patches)
        summary["skipped_fine_patch_count"] = int(skipped_fine_patch_count)
        for fine_patch in prepared_fine_patches:
            refinement_key = str(fine_patch.metadata.get("refinement_key", "")).strip()
            if not refinement_key:
                continue

            target_obj: ObjectMap | None = None
            target_observations: list[ObservationRecord] = []
            for obj in state.objects.values():
                matches = [
                    obs
                    for obs in obj.observations
                    if not obs.replaced_by_refinement
                    and obs.observation_layer == "coarse"
                    and obs.refinement_key == refinement_key
                ]
                if matches:
                    target_obj = obj
                    target_observations = matches
                    break
            if target_obj is None:
                continue

            target_id = int(target_obj.object_id)
            for obs in target_observations:
                obs.replaced_by_refinement = True
                if self.refinement_rebuild_tsdf_support:
                    self.tsdf_module.remove_patch_support(state.tsdf_volume, obs.patch, target_id)

            target_obj.observations = [
                obs
                for obs in target_obj.observations
                if not (
                    obs.observation_layer == "coarse"
                    and obs.refinement_key == refinement_key
                    and obs.replaced_by_refinement
                )
            ]
            target_obj.observations.append(self._observation_from_patch(fine_patch))

            remaining_points = [
                np.asarray(obs.patch.points, dtype=np.float32)
                for obs in target_obj.observations
                if len(obs.patch.points) > 0
            ]
            if remaining_points:
                target_obj.local_pcd = np.concatenate(remaining_points, axis=0)
                if len(target_obj.local_pcd) > self.max_points:
                    target_obj.local_pcd = self._deterministic_spatial_cap(
                        target_obj.local_pcd,
                        self.max_points,
                    )
                target_obj.centroid = target_obj.local_pcd.mean(axis=0)
                target_obj.bbox_min, target_obj.bbox_max = compute_bbox(target_obj.local_pcd)
            else:
                target_obj.local_pcd = np.empty((0, 3), dtype=np.float32)
            target_obj.update_count = int(len(target_obj.observations))
            if target_obj.observations:
                target_obj.last_seen_frame = max(int(obs.frame_id) for obs in target_obj.observations)
            target_obj.state = ObjectState.ACTIVE
            self._refresh_association_geometry(target_obj)

            if self.refinement_rebuild_tsdf_support:
                self.tsdf_module.integrate_patch(state.tsdf_volume, fine_patch, target_id)
            rebuild_anchor_semantic_votes_from_observations(target_obj)
            self._refresh_object_debug(target_obj, state.tsdf_volume)

            replaced_count = int(len(target_observations))
            inserted_count = 1
            target_obj.debug["last_async_refinement"] = {
                "refinement_key": refinement_key,
                "replaced_observation_count": replaced_count,
                "inserted_observation_count": inserted_count,
                "fine_patch_point_count": int(len(fine_patch.points)),
                "rebuild_tsdf_support": bool(self.refinement_rebuild_tsdf_support),
            }
            summary["replaced_observation_count"] += replaced_count
            summary["inserted_observation_count"] += inserted_count
            updated_ids.add(target_id)

        summary["updated_object_count"] = int(len(updated_ids))
        summary["updated_object_ids"] = sorted(int(object_id) for object_id in updated_ids)
        self.last_async_refinement_summary = summary
        return summary

    def _prepare_refinement_patches_by_key(self, fine_patches: List[Patch3D]) -> tuple[list[Patch3D], int]:
        grouped: dict[str, list[Patch3D]] = {}
        passthrough: list[Patch3D] = []
        for patch in fine_patches:
            refinement_key = str(patch.metadata.get("refinement_key", "")).strip()
            if not refinement_key:
                passthrough.append(patch)
                continue
            grouped.setdefault(refinement_key, []).append(patch)

        merged: list[Patch3D] = []
        skipped_count = 0
        for refinement_key in sorted(grouped):
            patches = grouped[refinement_key]
            if (
                self.refinement_max_components_per_replacement > 0
                and len(patches) > self.refinement_max_components_per_replacement
            ):
                skipped_count += int(len(patches))
                continue
            if len(patches) == 1:
                patch = patches[0]
                patch.metadata = dict(patch.metadata)
                patch.metadata.setdefault("merged_refined_patch_count", 1)
                merged.append(patch)
                continue

            patches = sorted(patches, key=lambda item: int(item.patch_id))
            point_chunks = [
                np.asarray(patch.points, dtype=np.float32)
                for patch in patches
                if len(patch.points) > 0
            ]
            points = (
                np.concatenate(point_chunks, axis=0)
                if point_chunks
                else np.empty((0, 3), dtype=np.float32)
            )
            first = patches[0]
            metadata = dict(first.metadata)
            metadata["merged_refined_patch_count"] = int(len(patches))
            metadata["merged_refined_patch_ids"] = [int(patch.patch_id) for patch in patches]
            metadata["source_proposal_id"] = int(first.metadata.get("source_proposal_id", first.patch_id))
            metadata["source_component_patch_ids"] = [int(patch.patch_id) for patch in patches]
            centroid = points.mean(axis=0) if len(points) > 0 else first.centroid.copy()
            if len(points) > 0:
                bbox_min, bbox_max = compute_bbox(points)
            else:
                bbox_min = first.bbox_min.copy()
                bbox_max = first.bbox_max.copy()
            merged.append(
                Patch3D(
                    patch_id=int(first.patch_id),
                    points=points,
                    centroid=centroid,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    normals=None,
                    timestamp=float(first.timestamp),
                    source_frame_id=int(first.source_frame_id),
                    soft_scores=first.soft_scores,
                    metadata=metadata,
                )
            )
        return [*passthrough, *merged], skipped_count

    def _filter_patch_by_current_frame_visibility(
        self,
        patch: Patch3D,
        *,
        current_depth: np.ndarray | None,
        current_pose: np.ndarray | None,
        current_intrinsics: CameraIntrinsics | None,
    ) -> tuple[Patch3D | None, dict[str, Any]]:
        point_count = int(len(patch.points))
        if (
            not self.current_frame_visibility_gate_enabled
            or current_depth is None
            or current_pose is None
            or current_intrinsics is None
            or point_count == 0
        ):
            debug = {
                "enabled": bool(self.current_frame_visibility_gate_enabled),
                "source_patch_id": int(patch.patch_id),
                "source_point_count": point_count,
                "projected_in_bounds_count": point_count,
                "depth_rejected_point_count": 0,
                "invalid_depth_point_count": 0,
                "accepted_point_count": point_count,
                "accepted_ratio": 1.0 if point_count > 0 else 0.0,
                "passed": True,
                "rejection_reasons": [],
            }
            return patch, debug

        keep_mask, diagnostics = filter_points_by_depth_consistency(
            patch.points,
            current_depth,
            current_pose,
            current_intrinsics,
            distance_threshold=self.current_frame_visibility_distance_threshold,
        )
        accepted_count = int(keep_mask.sum())
        accepted_ratio = float(accepted_count / max(point_count, 1))
        rejection_reasons: list[str] = []
        if accepted_count < self.current_frame_visibility_min_accept_points:
            rejection_reasons.append("insufficient_accepted_points")
        if accepted_ratio < self.current_frame_visibility_min_accept_ratio:
            rejection_reasons.append("low_accept_ratio")

        debug = {
            "enabled": True,
            "source_patch_id": int(patch.patch_id),
            "source_point_count": point_count,
            "accepted_point_count": accepted_count,
            "accepted_ratio": accepted_ratio,
            "distance_threshold": float(self.current_frame_visibility_distance_threshold),
            "min_accept_points": int(self.current_frame_visibility_min_accept_points),
            "min_accept_ratio": float(self.current_frame_visibility_min_accept_ratio),
            "passed": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            **diagnostics,
        }
        if rejection_reasons:
            return None, debug
        return self._copy_patch_with_filtered_points(
            patch,
            keep_mask,
            debug,
            metadata_key="current_frame_visibility_gate",
        ), debug

    def _record_current_frame_visibility_gate_debug(self, debug: dict[str, Any]) -> None:
        self.last_current_frame_visibility_gate_records.append(dict(debug))

    def _summarize_current_frame_visibility_gate_records(self) -> dict[str, Any]:
        records = list(self.last_current_frame_visibility_gate_records)
        if not records:
            return {"enabled": bool(self.current_frame_visibility_gate_enabled), "record_count": 0}
        return {
            "enabled": bool(self.current_frame_visibility_gate_enabled),
            "record_count": int(len(records)),
            "rejected_patch_count": int(sum(1 for record in records if not record.get("passed", True))),
            "source_point_count": int(sum(int(record.get("source_point_count", 0)) for record in records)),
            "accepted_point_count": int(sum(int(record.get("accepted_point_count", 0)) for record in records)),
            "depth_rejected_point_count": int(
                sum(int(record.get("depth_rejected_point_count", 0)) for record in records)
            ),
            "invalid_depth_point_count": int(
                sum(int(record.get("invalid_depth_point_count", 0)) for record in records)
            ),
        }

    def _build_background_voxel_support(
        self,
        state: SystemState,
        background_patches: List[Patch3D],
    ) -> set[tuple[int, int, int]]:
        if not self.surface_owner_gate_enabled or not self.surface_gate_background_enabled:
            return set()

        point_chunks: list[np.ndarray] = []
        if len(state.background.point_cloud) > 0:
            point_chunks.append(np.asarray(state.background.point_cloud, dtype=np.float32))
        for patch in background_patches:
            if len(patch.points) > 0:
                point_chunks.append(np.asarray(patch.points, dtype=np.float32))
        if not point_chunks:
            return set()

        points = np.concatenate(point_chunks, axis=0)
        voxels = self._points_to_voxels(points, state.tsdf_volume.voxel_size)
        if len(voxels) == 0:
            return set()
        unique_voxels = np.unique(voxels, axis=0)
        return {
            (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            for voxel in unique_voxels
        }

    def _filter_patch_by_surface_owner(
        self,
        patch: Patch3D,
        *,
        target_object_id: int | None,
        state: SystemState,
        background_support: set[tuple[int, int, int]],
        new_object: bool,
        contested_residual: bool = False,
    ) -> tuple[Patch3D | None, Patch3D | None, dict[str, Any]]:
        point_count = int(len(patch.points))
        mode = "contested_residual" if contested_residual else ("new_object" if new_object else "update")
        if not self.surface_owner_gate_enabled or point_count == 0:
            debug = {
                "enabled": bool(self.surface_owner_gate_enabled),
                "mode": mode,
                "target_object_id": -1 if target_object_id is None else int(target_object_id),
                "source_patch_id": int(patch.patch_id),
                "source_point_count": point_count,
                "accepted_point_count": point_count,
                "accepted_ratio": 1.0 if point_count > 0 else 0.0,
                "passed": True,
                "rejection_reasons": [],
            }
            return patch, None, debug

        points = np.asarray(patch.points, dtype=np.float32)
        voxels = self._points_to_voxels(points, state.tsdf_volume.voxel_size)
        representative_indices = (
            self._first_index_per_voxel(voxels)
            if self.surface_gate_representative_voxel_mode
            else np.arange(point_count, dtype=np.int64)
        )
        decision_voxels = voxels[representative_indices]
        unique_voxel_count = int(len(np.unique(voxels, axis=0))) if len(voxels) > 0 else 0
        anchor_label = str(patch.metadata.get("anchor_class_name", "")).strip()
        attached_override = bool(
            anchor_label in self.surface_gate_attached_surface_classes
            and unique_voxel_count <= self.surface_gate_attached_max_voxels
        )
        self_structural_background = bool(
            self._patch_has_strong_background_evidence(patch)
            and not attached_override
        )

        allowed_owner_id = target_object_id
        if contested_residual:
            metadata_parent_id = patch.metadata.get("contested_parent_object_id", -1)
            try:
                allowed_owner_id = int(metadata_parent_id)
            except (TypeError, ValueError):
                allowed_owner_id = None
            if allowed_owner_id is not None and allowed_owner_id < 0:
                allowed_owner_id = None

        decision_same_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_foreign_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_background_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        conflict_hits: Dict[Tuple[Tuple[int, int, int], int], int] = {}
        for decision_idx, voxel in enumerate(decision_voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            support = state.tsdf_volume.owner_support.get(key)
            owner_id = support.owner_id if support is not None else -1
            if allowed_owner_id is not None and owner_id == int(allowed_owner_id):
                decision_same_owner_mask[decision_idx] = True
            elif owner_id >= 0:
                patch_label = str(patch.metadata.get("anchor_class_name", "")).strip().lower()
                # For updates: same label → same object at different view → ACCEPT
                if patch_label and allowed_owner_id is not None:
                    owner_obj = state.objects.get(int(owner_id))
                    if owner_obj is not None:
                        owner_label = self._get_object_label_from_state(owner_obj)
                        if owner_label and patch_label == owner_label:
                            continue  # ACCEPT: same semantic identity
                # For new objects: same label → already tracked → REJECT
                elif patch_label and allowed_owner_id is None:
                    owner_obj = state.objects.get(int(owner_id))
                    if owner_obj is not None:
                        owner_label = self._get_object_label_from_state(owner_obj)
                        if owner_label and patch_label == owner_label:
                            decision_foreign_owner_mask[decision_idx] = True
                            continue  # REJECT: conflict with same-label existing object

                # Conflict-aware: check per-voxel support (O(1) vs O(N) summarize)
                old_owner_support_val = support.support.get(owner_id, 0.0)
                support_weak = old_owner_support_val < 2.0

                if support_weak:
                    new_id = int(allowed_owner_id) if allowed_owner_id is not None else -1
                    conflict_key = (key, new_id)
                    conflict_hits[conflict_key] = conflict_hits.get(conflict_key, 0) + 1
                    if conflict_hits[conflict_key] >= 2:
                        continue  # ACCEPT: old owner weak, new evidence persistent

                decision_foreign_owner_mask[decision_idx] = True

            if self.surface_gate_background_enabled:
                decision_background_owner_mask[decision_idx] = self_structural_background or key in background_support

        decision_background_reject_mask = (
            np.zeros(len(decision_voxels), dtype=bool)
            if attached_override
            else decision_background_owner_mask.copy()
        )
        decision_accepted_mask = ~(decision_foreign_owner_mask | decision_background_reject_mask)

        representative_count = int(len(decision_voxels))
        decision_accepted_count = int(decision_accepted_mask.sum())
        decision_foreign_count = int(decision_foreign_owner_mask.sum())
        decision_background_count = int(decision_background_reject_mask.sum())
        decision_same_owner_count = int(decision_same_owner_mask.sum())

        if self.surface_gate_representative_voxel_mode:
            rejected_voxel_keys = {
                (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                for voxel, keep in zip(decision_voxels, decision_accepted_mask)
                if not bool(keep)
            }
            accepted_mask = np.array(
                [
                    (int(voxel[0]), int(voxel[1]), int(voxel[2])) not in rejected_voxel_keys
                    for voxel in voxels
                ],
                dtype=bool,
            )
            denominator = max(representative_count, 1)
        else:
            background_reject_mask = decision_background_reject_mask
            accepted_mask = decision_accepted_mask
            denominator = max(point_count, 1)

        if self.surface_gate_representative_voxel_mode:
            background_voxel_keys = {
                (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                for voxel, rejected in zip(decision_voxels, decision_background_reject_mask)
                if bool(rejected)
            }
            background_reject_mask = np.array(
                [
                    (int(voxel[0]), int(voxel[1]), int(voxel[2])) in background_voxel_keys
                    for voxel in voxels
                ],
                dtype=bool,
            )

        accepted_count = int(accepted_mask.sum())
        rejected_count = int(point_count - accepted_count)
        if self.surface_gate_representative_voxel_mode:
            foreign_voxel_keys = {
                (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                for voxel, is_foreign in zip(decision_voxels, decision_foreign_owner_mask)
                if bool(is_foreign)
            }
            same_owner_voxel_keys = {
                (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                for voxel, is_same_owner in zip(decision_voxels, decision_same_owner_mask)
                if bool(is_same_owner)
            }
            foreign_owner_mask = np.array(
                [
                    (int(voxel[0]), int(voxel[1]), int(voxel[2])) in foreign_voxel_keys
                    for voxel in voxels
                ],
                dtype=bool,
            )
            same_owner_mask = np.array(
                [
                    (int(voxel[0]), int(voxel[1]), int(voxel[2])) in same_owner_voxel_keys
                    for voxel in voxels
                ],
                dtype=bool,
            )
        else:
            foreign_owner_mask = decision_foreign_owner_mask
            same_owner_mask = decision_same_owner_mask
        foreign_count = int(foreign_owner_mask.sum())
        background_count = int(background_reject_mask.sum())
        same_owner_count = int(same_owner_mask.sum())
        accepted_ratio = float(decision_accepted_count / denominator)
        foreign_ratio = float(decision_foreign_count / denominator)
        background_ratio = float(decision_background_count / denominator)
        min_accept_ratio = (
            self.surface_gate_min_new_object_accept_ratio
            if new_object
            else self.surface_gate_min_update_accept_ratio
        )
        min_accept_points = 1 if attached_override else self.surface_gate_min_accept_points
        min_accept_count = (
            decision_accepted_count
            if self.surface_gate_representative_voxel_mode
            else accepted_count
        )
        min_accept_count_unit = (
            "decision_voxel"
            if self.surface_gate_representative_voxel_mode
            else "point"
        )

        rejection_reasons: list[str] = []
        # Labeled patches get relaxed foreign_owner threshold (same-label = same object)
        max_foreign = (self.surface_gate_max_foreign_owner_ratio * 2.0
                       if anchor_label else self.surface_gate_max_foreign_owner_ratio)
        if foreign_ratio > max_foreign:
            rejection_reasons.append("foreign_owner_ratio")
        if background_ratio > self.surface_gate_max_background_owner_ratio:
            rejection_reasons.append("background_owner_ratio")
        if min_accept_count < min_accept_points:
            rejection_reasons.append("insufficient_accepted_points")
        if accepted_ratio < min_accept_ratio:
            rejection_reasons.append("low_accept_ratio")

        debug = {
            "enabled": True,
            "mode": mode,
            "target_object_id": -1 if target_object_id is None else int(target_object_id),
            "source_patch_id": int(patch.patch_id),
            "source_point_count": point_count,
            "unique_voxel_count": unique_voxel_count,
            "representative_voxel_mode": bool(self.surface_gate_representative_voxel_mode),
            "decision_voxel_count": representative_count,
            "accepted_decision_voxel_count": decision_accepted_count,
            "foreign_owner_decision_voxel_count": decision_foreign_count,
            "background_owner_decision_voxel_count": decision_background_count,
            "same_owner_decision_voxel_count": decision_same_owner_count,
            "accepted_point_count": accepted_count,
            "rejected_point_count": rejected_count,
            "accepted_ratio": accepted_ratio,
            "foreign_owner_point_count": foreign_count,
            "foreign_owner_ratio": foreign_ratio,
            "background_owner_point_count": background_count,
            "background_owner_ratio": background_ratio,
            "same_owner_point_count": same_owner_count,
            "attached_surface_override": attached_override,
            "min_accept_points": int(min_accept_points),
            "min_accept_count_unit": min_accept_count_unit,
            "self_structural_background": self_structural_background,
            "anchor_class_name": anchor_label,
            "passed": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }

        structural_reject = None
        if background_count > 0:
            structural_reject = self._copy_patch_with_filtered_points(
                patch,
                background_reject_mask,
                dict(
                    debug,
                    split_origin="background",
                    surface_gate_structural_reject=True,
                    surface_gate_reject_of_patch_id=int(patch.patch_id),
                ),
            )

        if rejection_reasons:
            return None, structural_reject, debug

        filtered_patch = self._copy_patch_with_filtered_points(patch, accepted_mask, debug)
        return filtered_patch, structural_reject, debug

    def _copy_patch_with_filtered_points(
        self,
        patch: Patch3D,
        mask: np.ndarray,
        gate_debug: dict[str, Any],
        *,
        metadata_key: str = "surface_owner_gate",
    ) -> Patch3D | None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(patch.points) or not np.any(mask):
            return None

        points = np.asarray(patch.points, dtype=np.float32)[mask]
        normals = None
        if patch.normals is not None and len(patch.normals) == len(mask):
            normals = np.asarray(patch.normals)[mask]
        bbox_min, bbox_max = compute_bbox(points)
        metadata = dict(patch.metadata)
        original_lifted_count = int(metadata.get("lifted_point_count", len(patch.points)))
        metadata[metadata_key] = gate_debug
        metadata[f"{metadata_key}_original_lifted_point_count"] = original_lifted_count
        metadata["lifted_point_count"] = int(len(points))
        if gate_debug.get("split_origin"):
            metadata["split_origin"] = str(gate_debug["split_origin"])
        if gate_debug.get("surface_gate_structural_reject"):
            metadata["surface_gate_structural_reject"] = True
            metadata["surface_gate_reject_of_patch_id"] = int(
                gate_debug.get("surface_gate_reject_of_patch_id", patch.patch_id)
            )

        return Patch3D(
            patch_id=patch.patch_id,
            points=points.astype(np.float32, copy=False),
            centroid=points.mean(axis=0).astype(np.float32, copy=False),
            bbox_min=bbox_min.astype(np.float32, copy=False),
            bbox_max=bbox_max.astype(np.float32, copy=False),
            normals=normals,
            timestamp=patch.timestamp,
            source_frame_id=patch.source_frame_id,
            soft_scores=patch.soft_scores,
            metadata=metadata,
        )

    def _patch_has_strong_background_evidence(self, patch: Patch3D) -> bool:
        bg_score = float(patch.soft_scores.backgroundness_score)
        obj_score = float(patch.soft_scores.objectness_score)
        return bool(
            bg_score >= self.surface_gate_self_background_min_score
            and bg_score >= obj_score + self.surface_gate_self_background_margin
        )

    def _record_surface_gate_debug(self, gate_debug: dict[str, Any]) -> None:
        if not gate_debug:
            return
        self.last_surface_gate_records.append(dict(gate_debug))

    def _summarize_surface_gate_records(self) -> dict[str, Any]:
        records = list(self.last_surface_gate_records)
        if not records:
            return {
                "enabled": bool(self.surface_owner_gate_enabled),
                "checked_patch_count": 0,
                "passed_patch_count": 0,
                "rejected_patch_count": 0,
                "structural_reject_patch_count": int(len(self.last_structural_reject_patches)),
            }

        return {
            "enabled": bool(self.surface_owner_gate_enabled),
            "checked_patch_count": int(len(records)),
            "passed_patch_count": int(sum(1 for record in records if bool(record.get("passed", False)))),
            "rejected_patch_count": int(sum(1 for record in records if not bool(record.get("passed", False)))),
            "structural_reject_patch_count": int(len(self.last_structural_reject_patches)),
            "source_point_count": int(sum(int(record.get("source_point_count", 0)) for record in records)),
            "accepted_point_count": int(sum(int(record.get("accepted_point_count", 0)) for record in records)),
            "foreign_owner_point_count": int(
                sum(int(record.get("foreign_owner_point_count", 0)) for record in records)
            ),
            "background_owner_point_count": int(
                sum(int(record.get("background_owner_point_count", 0)) for record in records)
            ),
        }

    def _write_candidate_evidence(
        self,
        state: SystemState,
        patch: Patch3D,
    ) -> None:
        """Write rejected patch evidence to CandidateEvidenceGrid.

        Only writes if the patch has a class label and its voxels
        are currently background (unowned) in the TSDF volume.

        Args:
            state: Current system state.
            patch: The rejected patch with class label metadata.
        """
        label = str(patch.metadata.get("anchor_class_name", "")).strip()
        if not label:
            return

        # Get confidence from patch metadata
        confidence = float(patch.metadata.get("anchor_confidence", 0.5))
        if confidence <= 0.0:
            return

        points = np.asarray(patch.points, dtype=np.float32)
        if len(points) == 0:
            return

        voxel_size = state.candidate_evidence.voxel_size
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        unique_voxels: Set[Tuple[int, int, int]] = set()

        for vk_array in voxel_indices:
            vk = (int(vk_array[0]), int(vk_array[1]), int(vk_array[2]))
            if vk in unique_voxels:
                continue
            unique_voxels.add(vk)

            # Check if voxel is background in TSDF
            tsdf_owner = state.tsdf_volume.owner_support.get(vk)
            if tsdf_owner is not None and tsdf_owner.owner_id >= 0:
                continue  # Not background — skip this voxel

            # Write to CandidateEvidenceGrid
            grid = state.candidate_evidence
            entry = grid.entries.get(vk)
            if entry is None:
                entry = CandidateEvidenceEntry()
                grid.entries[vk] = entry

            label_norm = label.strip().lower()
            entry.label_votes[label_norm] = entry.label_votes.get(label_norm, 0.0) + confidence
            entry.hit_count += 1
            entry.last_seen_frame = state.frame_count
            if entry.first_seen_frame == 0:
                entry.first_seen_frame = state.frame_count

    @staticmethod
    def _points_to_voxels(points: np.ndarray, voxel_size: float) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.int64)
        safe_voxel_size = max(float(voxel_size), 1e-9)
        return np.floor(np.asarray(points, dtype=np.float32) / safe_voxel_size).astype(np.int64)

    @staticmethod
    def _first_index_per_voxel(voxels: np.ndarray) -> np.ndarray:
        if len(voxels) == 0:
            return np.zeros(0, dtype=np.int64)
        _unique, first_indices = np.unique(voxels, axis=0, return_index=True)
        return np.sort(first_indices.astype(np.int64))

    def _upsert_provisional_object(self, state: SystemState, patch: Patch3D, *, contested: bool = False) -> None:
        provisional = self._match_provisional_object(state, patch)
        if provisional is None:
            provisional = ProvisionalObject(
                provisional_id=int(state.next_provisional_id),
                anchor_class_name=str(patch.metadata.get("anchor_class_name", "")),
                anchor_confidence=float(patch.metadata.get("anchor_confidence", 0.0)),
                first_seen_frame=int(patch.source_frame_id),
                last_seen_frame=int(patch.source_frame_id),
            )
            state.provisional_objects[provisional.provisional_id] = provisional
            state.next_provisional_id += 1
        else:
            if not provisional.anchor_class_name:
                provisional.anchor_class_name = str(patch.metadata.get("anchor_class_name", ""))
            provisional.anchor_confidence = max(
                float(provisional.anchor_confidence),
                float(patch.metadata.get("anchor_confidence", 0.0)),
            )

        point_cloud = np.concatenate([provisional.local_pcd, patch.points], axis=0)
        if len(point_cloud) > 0:
            point_cloud = voxel_downsample(point_cloud, self.provisional_downsample_voxel)
        if len(point_cloud) > self.provisional_max_points:
            point_cloud = self._deterministic_spatial_cap(point_cloud, self.provisional_max_points)

        provisional.local_pcd = point_cloud
        provisional.centroid = provisional.local_pcd.mean(axis=0)
        provisional.bbox_min, provisional.bbox_max = compute_bbox(provisional.local_pcd)
        provisional.last_seen_frame = int(patch.source_frame_id)
        provisional.hit_count += 1
        provisional.observations.append(self._observation_from_patch(patch))
        provisional.debug = {
            "hit_count": int(provisional.hit_count),
            "observation_frame_count": int(len({obs.frame_id for obs in provisional.observations})),
            "local_point_count": int(len(provisional.local_pcd)),
            "anchor_class_name": provisional.anchor_class_name,
            "anchor_confidence": float(provisional.anchor_confidence),
            "semantic_split_candidate_from_object_id": int(
                patch.metadata.get("semantic_split_candidate_from_object_id", -1)
            ),
            "semantic_split_candidate_parent_label": str(
                patch.metadata.get("semantic_split_candidate_parent_label", "")
            ),
            "semantic_split_candidate_new_label": str(
                patch.metadata.get("semantic_split_candidate_new_label", "")
            ),
            "semantic_split_candidate_reason": str(
                patch.metadata.get("semantic_split_candidate_reason", "")
            ),
            "is_contested_residual": bool(
                contested or int(patch.metadata.get("contested_parent_object_id", -1)) != -1
            ),
            "contested_parent_object_id": int(patch.metadata.get("contested_parent_object_id", -1)),
            "contested_parent_label": str(patch.metadata.get("contested_parent_label", "")),
            "contested_patch_label": str(patch.metadata.get("contested_patch_label", "")),
            "contested_reason": str(patch.metadata.get("contested_reason", "")),
        }

    def _match_provisional_object(self, state: SystemState, patch: Patch3D) -> ProvisionalObject | None:
        best = None
        best_distance = float("inf")
        patch_anchor_class = str(patch.metadata.get("anchor_class_name", ""))
        for provisional in state.provisional_objects.values():
            if patch_anchor_class:
                if provisional.anchor_class_name and provisional.anchor_class_name != patch_anchor_class:
                    continue
            elif provisional.anchor_class_name:
                continue
            distance = float(np.linalg.norm(provisional.centroid - patch.centroid))
            if distance > self.provisional_match_distance:
                continue
            if distance < best_distance:
                best_distance = distance
                best = provisional
        return best

    def _promote_stable_provisionals(self, state: SystemState) -> None:
        promotable_ids = []
        for provisional_id, provisional in state.provisional_objects.items():
            seen_frames = len({obs.frame_id for obs in provisional.observations})
            if seen_frames >= self.provisional_promotion_hits:
                promotable_ids.append(int(provisional_id))

        for provisional_id in promotable_ids:
            provisional = state.provisional_objects.pop(provisional_id, None)
            if provisional is None:
                continue
            obj = self._promote_provisional_object(provisional, state.next_object_id)
            state.objects[obj.object_id] = obj
            for observation in provisional.observations:
                self.tsdf_module.integrate_patch(state.tsdf_volume, observation.patch, obj.object_id)
            self._refresh_object_debug(obj, state.tsdf_volume)
            if "promoted_contested_residual" in obj.debug:
                self.last_contested_residual_promoted_object_ids.append(int(obj.object_id))
            self.last_created_object_ids.append(int(obj.object_id))
            self.last_updated_object_ids.append(int(obj.object_id))
            state.next_object_id += 1

    def _promote_provisional_object(self, provisional: ProvisionalObject, object_id: int) -> ObjectMap:
        obj = ObjectMap(
            object_id=object_id,
            state=ObjectState.ACTIVE,
            local_pcd=provisional.local_pcd.copy(),
            centroid=provisional.centroid.copy(),
            bbox_min=provisional.bbox_min.copy(),
            bbox_max=provisional.bbox_max.copy(),
            whole_evidence=WholeEvidenceScores(
                whole_evidence_score=min(1.0, 0.2 * provisional.hit_count),
                part_evidence_score=0.0,
                assignment_score=0.5,
            ),
            semantic_memory=SemanticMemory(),
            observations=list(provisional.observations),
            confidence=1.0,
            last_seen_frame=int(provisional.last_seen_frame),
            creation_frame=int(provisional.last_seen_frame),
            update_count=int(max(provisional.hit_count, 1)),
            creation_centroid=provisional.centroid.copy(),
        )
        self._refresh_association_geometry(obj)
        rebuild_anchor_semantic_votes_from_observations(obj)
        split_parent_id = int(provisional.debug.get("semantic_split_candidate_from_object_id", -1))
        if split_parent_id >= 0:
            obj.debug["promoted_split_candidate"] = {
                "parent_object_id": split_parent_id,
                "parent_label": str(provisional.debug.get("semantic_split_candidate_parent_label", "")),
                "new_label": str(provisional.debug.get("semantic_split_candidate_new_label", "")),
                "reason": str(provisional.debug.get("semantic_split_candidate_reason", "")),
            }
        if bool(provisional.debug.get("is_contested_residual", False)):
            obj.debug["promoted_contested_residual"] = {
                "parent_object_id": int(provisional.debug.get("contested_parent_object_id", -1)),
                "parent_label": str(provisional.debug.get("contested_parent_label", "")),
                "patch_label": str(provisional.debug.get("contested_patch_label", "")),
                "reason": str(provisional.debug.get("contested_reason", "")),
            }
        return obj

    def _refresh_association_geometry(self, obj: ObjectMap) -> None:
        points = np.asarray(obj.local_pcd, dtype=np.float32)
        if not self.association_geometry_enabled or len(points) == 0:
            obj.association_pcd = np.empty((0, 3), dtype=np.float32)
            obj.debug["association_geometry"] = self._association_geometry_debug(
                point_count=0,
                source_point_count=len(points),
            )
            return

        association_points = voxel_downsample(points, self.association_geometry_voxel)
        if (
            self.association_geometry_max_points > 0
            and len(association_points) > self.association_geometry_max_points
        ):
            obj.association_pcd = self._deterministic_spatial_cap(
                association_points,
                self.association_geometry_max_points,
            )
        else:
            obj.association_pcd = association_points.copy()
        obj.debug["association_geometry"] = self._association_geometry_debug(
            point_count=len(obj.association_pcd),
            source_point_count=len(points),
        )

    def _update_association_geometry_from_patch(self, obj: ObjectMap, patch: Patch3D) -> None:
        source_point_count = len(obj.local_pcd)
        if not self.association_geometry_enabled:
            obj.association_pcd = np.empty((0, 3), dtype=np.float32)
            obj.debug["association_geometry"] = self._association_geometry_debug(
                point_count=0,
                source_point_count=source_point_count,
            )
            return

        patch_points = np.asarray(patch.points, dtype=np.float32)
        if len(patch_points) == 0:
            existing_points = np.asarray(
                getattr(obj, "association_pcd", np.empty((0, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            obj.association_pcd = existing_points.copy()
            obj.debug["association_geometry"] = self._association_geometry_debug(
                point_count=len(obj.association_pcd),
                source_point_count=source_point_count,
            )
            return

        patch_rep = voxel_downsample(patch_points, self.association_geometry_voxel)
        existing_points = np.asarray(
            getattr(obj, "association_pcd", np.empty((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        if len(existing_points) == 0:
            combined = np.asarray(patch_rep, dtype=np.float32)
        elif len(patch_rep) == 0:
            combined = existing_points
        else:
            combined = np.concatenate([existing_points, patch_rep], axis=0)

        if (
            self.association_geometry_max_points > 0
            and len(combined) > self.association_geometry_max_points
        ):
            obj.association_pcd = self._deterministic_spatial_cap(
                combined,
                self.association_geometry_max_points,
            )
        else:
            obj.association_pcd = combined.copy()
        obj.debug["association_geometry"] = self._association_geometry_debug(
            point_count=len(obj.association_pcd),
            source_point_count=source_point_count,
        )

    def _association_geometry_debug(self, *, point_count: int, source_point_count: int) -> dict[str, Any]:
        return {
            "role": "bounded_association_geometry",
            "enabled": bool(self.association_geometry_enabled),
            "point_count": int(point_count),
            "source_point_count": int(source_point_count),
            "voxel_size": float(self.association_geometry_voxel),
            "max_points_per_object": int(self.association_geometry_max_points),
            "max_points": int(self.association_geometry_max_points),
        }

    def _should_enter_provisional_pool(self, patch: Patch3D) -> bool:
        if self._is_background_patch(patch):
            return False
        if self._is_ambiguous_patch(patch):
            if len(patch.points) < self.provisional_unanchored_min_points:
                return False
            return float(patch.soft_scores.objectness_score) >= float(patch.soft_scores.backgroundness_score)
        if "anchor_id" not in patch.metadata and "anchor_class_name" not in patch.metadata:
            return True
        anchor_class_name = str(patch.metadata.get("anchor_class_name", ""))
        if anchor_class_name:
            return True
        if len(patch.points) < self.provisional_unanchored_min_points:
            return False
        return float(patch.soft_scores.objectness_score) >= self.provisional_unanchored_objectness_min

    def _prune_stale_provisionals(self, state: SystemState, current_frame: int) -> None:
        stale_ids = [
            provisional_id
            for provisional_id, provisional in state.provisional_objects.items()
            if int(current_frame) - int(provisional.last_seen_frame) > self.provisional_max_idle_frames
        ]
        for provisional_id in stale_ids:
            del state.provisional_objects[provisional_id]

    def _observation_from_patch(self, patch: Patch3D) -> ObservationRecord:
        return ObservationRecord(
            frame_id=patch.source_frame_id,
            patch=patch,
            crop_bbox=self._patch_crop_bbox(patch),
            timestamp=patch.timestamp,
            source_frame_id=int(patch.source_frame_id),
            source_proposal_id=int(patch.metadata.get("source_proposal_id", patch.patch_id)),
            anchor_id=int(patch.metadata.get("anchor_id", -1)),
            observation_layer=str(patch.metadata.get("observation_layer", "")),
            refinement_key=str(patch.metadata.get("refinement_key", "")),
            replaced_by_refinement=False,
        )

    def _should_split_relocated_update(
        self,
        obj: ObjectMap,
        patch: Patch3D,
        score: AssociationScore,
    ) -> bool:
        if not self.relocation_split_gate_enabled:
            return False
        centroid_distance = float(np.linalg.norm(np.asarray(patch.centroid) - np.asarray(obj.centroid)))
        voxel_vote_score = float(getattr(score, "voxel_vote_score", 0.0) or 0.0)
        return bool(
            centroid_distance >= self.relocation_split_centroid_distance
            and voxel_vote_score <= self.relocation_split_max_voxel_vote_score
        )

    def _update_object(self, obj: ObjectMap, patch: Patch3D) -> None:
        """Merge new patch into an existing object.

        local_pcd is the v1 object-level pool geometry (Layer 5).
        It remains separate from the global TSDF instance substrate, which
        stays responsible for owner decisions and support/stability.
        """
        # Update local_pcd as the v1 object pool geometry.
        obj.local_pcd = np.concatenate([obj.local_pcd, patch.points], axis=0)
        obj.update_count += 1

        # Periodically compact the pool geometry to keep the export source bounded.
        if obj.update_count % self.downsample_interval == 0:
            obj.local_pcd = voxel_downsample(obj.local_pcd, self.downsample_voxel)

        # Cap point count
        if len(obj.local_pcd) > self.max_points:
            obj.local_pcd = self._deterministic_spatial_cap(obj.local_pcd, self.max_points)

        self._update_association_geometry_from_patch(obj, patch)

        # Update spatial properties from local_pcd
        obj.centroid = obj.local_pcd.mean(axis=0)
        obj.bbox_min, obj.bbox_max = compute_bbox(obj.local_pcd)
        obj.last_seen_frame = patch.source_frame_id
        obj.state = ObjectState.ACTIVE

        # Record observation
        obj.observations.append(self._observation_from_patch(patch))
        accumulate_anchor_semantic_vote(obj, patch)

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
            observations=[self._observation_from_patch(patch)],
            confidence=1.0,
            last_seen_frame=patch.source_frame_id,
            creation_frame=patch.source_frame_id,
            update_count=1,
            creation_centroid=patch.centroid.copy(),
        )
        self._refresh_association_geometry(obj)
        accumulate_anchor_semantic_vote(obj, patch)
        logger.debug(f"Created new object {object_id} with {len(patch.points)} points.")
        return obj

    def _patch_crop_bbox(self, patch: Patch3D) -> np.ndarray | None:
        """Recover the source 2D crop bbox for semantic view selection."""
        bbox = patch.metadata.get("source_bbox_xyxy")
        if bbox is None:
            return None
        return np.asarray(bbox, dtype=np.float32).copy()

    def _refresh_object_debug(self, obj: ObjectMap, volume: TSDFInstanceVolume) -> None:
        """Expose the separation between the TSDF backbone and the local pool geometry."""
        support_stats = self.tsdf_module.summarize_instance_support(volume, obj.object_id)
        # Update peak voxel count for TSDF-driven dynamic maintenance
        current_owned = int(support_stats["owned_voxel_count"])
        if current_owned > obj.peak_voxel_count:
            obj.peak_voxel_count = current_owned
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
            "role": "object_pool_geometry",
            "local_point_count": int(len(obj.local_pcd)),
            "point_count": int(len(obj.local_pcd)),
            "downsample_voxel_size": float(self.downsample_voxel),
            "max_points_per_object": int(self.max_points),
        }
        association_debug = dict(obj.debug.get("association_geometry", {}))
        association_debug.update(
            self._association_geometry_debug(
                point_count=len(getattr(obj, "association_pcd", [])),
                source_point_count=len(obj.local_pcd),
            )
        )
        obj.debug["association_geometry"] = association_debug
        anchor_state = obj.debug.get("anchor_semantics", {})
        if isinstance(anchor_state, dict):
            anchor_evidence = []
            for item in anchor_state.get("evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                anchor_evidence.append(
                    {
                        "label": str(item.get("label", "")),
                        "confidence": float(item.get("confidence", 0.0)),
                        "frame_id": int(item.get("frame_id", 0)),
                        "view_quality": float(item.get("view_quality", 0.0)),
                        "vote_weight": float(item.get("vote_weight", 0.0)),
                        "weighted_vote": float(item.get("weighted_vote", 0.0)),
                    }
                )
            contextual_evidence = []
            for item in anchor_state.get("contextual_evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                contextual_evidence.append(
                    {
                        "label": str(item.get("label", "")),
                        "confidence": float(item.get("confidence", 0.0)),
                        "frame_id": int(item.get("frame_id", 0)),
                        "anchor_label_strength": str(item.get("anchor_label_strength", "")),
                        "commit_eligible": bool(item.get("commit_eligible", False)),
                    }
                )
            delayed_evidence = []
            for item in anchor_state.get("delayed_evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                delayed_evidence.append(
                    {
                        "label": str(item.get("label", "")),
                        "confidence": float(item.get("confidence", 0.0)),
                        "frame_id": int(item.get("frame_id", 0)),
                        "view_quality": float(item.get("view_quality", 0.0)),
                        "anchor_label_strength": str(item.get("anchor_label_strength", "")),
                        "semantic_commit_allowed": bool(item.get("semantic_commit_allowed", False)),
                        "residual_semantic_policy": str(item.get("residual_semantic_policy", "")),
                        "mask_anchor_relation": str(item.get("mask_anchor_relation", "")),
                        "reason": str(item.get("reason", "")),
                        "commit_eligible": bool(item.get("commit_eligible", False)),
                    }
                )
            obj.debug["anchor_semantics"] = {
                "semantic_state": str(anchor_state.get("semantic_state", "")),
                "committed_label": str(anchor_state.get("committed_label", "")),
                "commit_reason": str(anchor_state.get("commit_reason", "")),
                "canonical_label": str(anchor_state.get("canonical_label", "")),
                "canonical_score": float(anchor_state.get("canonical_score", 0.0)),
                "canonical_frame_hits": int(anchor_state.get("canonical_frame_hits", 0)),
                "canonical_confidence": float(anchor_state.get("canonical_confidence", 0.0)),
                "canonical_best_view_quality": float(anchor_state.get("canonical_best_view_quality", 0.0)),
                "label_score_sum": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("label_score_sum", {}) or {}).items()
                },
                "label_weighted_score": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("label_weighted_score", {}) or {}).items()
                },
                "label_recent_score": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("label_recent_score", {}) or {}).items()
                },
                "label_max_confidence": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("label_max_confidence", {}) or {}).items()
                },
                "label_best_view_quality": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("label_best_view_quality", {}) or {}).items()
                },
                "label_seen_frames": {
                    str(label): [int(frame_id) for frame_id in frame_ids]
                    for label, frame_ids in (anchor_state.get("label_seen_frames", {}) or {}).items()
                },
                "label_frame_hits": {
                    str(label): int(count)
                    for label, count in (anchor_state.get("label_frame_hits", {}) or {}).items()
                },
                "label_high_quality_hits": {
                    str(label): int(count)
                    for label, count in (anchor_state.get("label_high_quality_hits", {}) or {}).items()
                },
                "evidence": anchor_evidence,
                "contextual_label_score_sum": {
                    str(label): float(score)
                    for label, score in (anchor_state.get("contextual_label_score_sum", {}) or {}).items()
                },
                "contextual_evidence": contextual_evidence,
                "delayed_evidence": delayed_evidence,
                "relabel_events": list(anchor_state.get("relabel_events", []) or []),
                "posterior_relabel_events": list(anchor_state.get("posterior_relabel_events", []) or []),
                "ignored_observation_count": int(anchor_state.get("ignored_observation_count", 0)),
                "ignored_observation_reasons": {
                    str(reason): int(count)
                    for reason, count in (anchor_state.get("ignored_observation_reasons", {}) or {}).items()
                },
                "last_frame_id": int(anchor_state.get("last_frame_id", 0)),
                "source": str(anchor_state.get("source", "")),
            }
        obj.debug["whole_evidence"] = {
            "whole_evidence_score": float(obj.whole_evidence.whole_evidence_score),
            "part_evidence_score": float(obj.whole_evidence.part_evidence_score),
            "assignment_score": float(obj.whole_evidence.assignment_score),
        }

    @staticmethod
    def _is_ambiguous_patch(patch: Patch3D) -> bool:
        return str(patch.metadata.get("split_origin", "")) == "ambiguous"

    @staticmethod
    def _is_background_patch(patch: Patch3D) -> bool:
        return str(patch.metadata.get("split_origin", "")) == "background"

    @staticmethod
    def _get_object_label_from_state(obj: ObjectMap) -> str:
        """Extract canonical label from object for gate label matching."""
        if obj.semantic_memory.label_hypotheses:
            return str(obj.semantic_memory.label_hypotheses[0][0]).strip().lower()
        anchor = obj.debug.get("anchor_semantics", {})
        if isinstance(anchor, dict):
            label = anchor.get("canonical_label", "")
            if label:
                return str(label).strip().lower()
        return ""

    @staticmethod
    def _deterministic_spatial_cap(points: np.ndarray, max_points: int) -> np.ndarray:
        if max_points <= 0 or len(points) <= max_points:
            return np.asarray(points, dtype=np.float32, copy=False)

        points = np.asarray(points, dtype=np.float32)
        bbox_min = points.min(axis=0)
        bbox_span = np.maximum(points.max(axis=0) - bbox_min, 1e-6)
        grid_resolution = max(1, int(np.ceil(max_points ** (1.0 / 3.0))))
        normalized = np.clip((points - bbox_min) / bbox_span, 0.0, 1.0 - 1e-6)
        voxel_indices = np.floor(normalized * grid_resolution).astype(np.int32)
        order = np.lexsort(
            (
                points[:, 2],
                points[:, 1],
                points[:, 0],
                voxel_indices[:, 2],
                voxel_indices[:, 1],
                voxel_indices[:, 0],
            )
        )
        sorted_points = points[order]
        sorted_voxels = voxel_indices[order]
        _, start_indices, counts = np.unique(
            sorted_voxels,
            axis=0,
            return_index=True,
            return_counts=True,
        )

        if len(start_indices) >= max_points:
            selected_voxels = np.linspace(0, len(start_indices) - 1, num=max_points, dtype=np.int32)
            return sorted_points[start_indices[selected_voxels]].astype(np.float32, copy=False)

        selected_rows = [int(start_index) for start_index in start_indices.tolist()]
        layer = 1
        max_count = int(counts.max()) if len(counts) > 0 else 0
        while len(selected_rows) < max_points and layer < max_count:
            available = [idx for idx, count in enumerate(counts.tolist()) if count > layer]
            if not available:
                break
            remaining = max_points - len(selected_rows)
            if len(available) > remaining:
                sample_positions = np.linspace(0, len(available) - 1, num=remaining, dtype=np.int32)
                available = [available[int(position)] for position in sample_positions.tolist()]
            for voxel_index in available:
                selected_rows.append(int(start_indices[voxel_index] + layer))
                if len(selected_rows) >= max_points:
                    break
            layer += 1

        selected_rows = np.asarray(selected_rows[:max_points], dtype=np.int32)
        return sorted_points[selected_rows].astype(np.float32, copy=False)
