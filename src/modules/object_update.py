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
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import (
    AssociationResult,
    CameraIntrinsics,
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
from src.modules.tsdf_instance_map import (
    PatchVoxelView,
    TSDFInstanceMapModule,
    filter_patch_voxel_view,
    get_patch_voxel_view,
)
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
        self.local_pcd_incremental_bounds_enabled = bool(
            config.get("local_pcd_incremental_bounds_enabled", True)
        )
        association_geometry_cfg = config.get("association_geometry", {})
        self.association_geometry_enabled = bool(association_geometry_cfg.get("enabled", False))
        self.association_geometry_sketch_enabled = bool(association_geometry_cfg.get("sketch_enabled", False))
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
        self.association_geometry_patch_pre_cap_factor = float(
            association_geometry_cfg.get("patch_pre_cap_factor", 4.0)
        )
        self.association_geometry_materialize_interval = max(
            1,
            int(association_geometry_cfg.get("materialize_interval", self.downsample_interval)),
        )
        self._association_geometry_sketches: dict[int, dict[tuple[int, int, int], np.ndarray]] = {}
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
        self.refresh_object_debug_interval = int(config.get("refresh_object_debug_interval", 1))
        self.refresh_object_debug_on_create = bool(config.get("refresh_object_debug_on_create", True))
        gate_cfg = config.get("surface_owner_gate", {})
        self.surface_owner_gate_enabled = bool(gate_cfg.get("enabled", False))
        self.surface_gate_representative_voxel_mode = bool(gate_cfg.get("representative_voxel_mode", False))
        self.surface_gate_inverse_policy = str(gate_cfg.get("inverse_policy", "lazy")).strip().lower()
        if self.surface_gate_inverse_policy not in {"lazy", "eager", "auto"}:
            logger.warning(
                "Unknown surface_owner_gate.inverse_policy=%r; falling back to lazy.",
                self.surface_gate_inverse_policy,
            )
            self.surface_gate_inverse_policy = "lazy"
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
        self.candidate_evidence_enabled = bool(gate_cfg.get("candidate_evidence_enabled", False))
        self.candidate_evidence_threshold = max(1, int(gate_cfg.get("candidate_evidence_threshold", 2)))
        self.candidate_evidence_decay_interval = max(10, int(gate_cfg.get("candidate_evidence_decay_interval", 100)))
        self.candidate_evidence_max_voxels = max(1000, int(gate_cfg.get("candidate_evidence_max_voxels", 100000)))
        visibility_cfg = config.get("current_frame_visibility_gate", {})
        self.current_frame_visibility_gate_enabled = bool(visibility_cfg.get("enabled", False))
        self.current_frame_visibility_distance_threshold = float(visibility_cfg.get("distance_threshold", 0.05))
        self.current_frame_visibility_min_accept_points = int(visibility_cfg.get("min_accept_points", 20))
        self.current_frame_visibility_min_accept_ratio = float(visibility_cfg.get("min_accept_ratio", 0.30))
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
        self.last_stage_timings: dict[str, float] = {}
        self.last_updated_object_ids: list[int] = []
        self.last_created_object_ids: list[int] = []
        local_pool_cfg = config.get("local_pcd_chunk_pool", {})
        self.local_pcd_chunk_pool_enabled = bool(local_pool_cfg.get("enabled", False))
        self.local_pcd_chunk_pool_bounded_compaction_enabled = bool(
            local_pool_cfg.get("bounded_compaction_enabled", False)
        )
        self.local_pcd_chunk_pool_compact_to_max_points = bool(
            local_pool_cfg.get("compact_to_max_points", False)
        )
        self.local_pcd_chunk_pool_materialize_interval = max(
            1,
            int(local_pool_cfg.get("materialize_interval", self.downsample_interval)),
        )
        self.local_pcd_chunk_pool_max_pending_points = max(
            0,
            int(local_pool_cfg.get("max_pending_points", max(self.max_points // 4, 1))),
        )
        voxel_pool_cfg = config.get("local_pcd_voxel_pool", {})
        self.local_pcd_voxel_pool_enabled = bool(voxel_pool_cfg.get("enabled", False))
        self.local_pcd_voxel_pool_materialize_on_interval = bool(
            voxel_pool_cfg.get("materialize_on_interval", False)
        )
        self.local_pcd_voxel_pool_voxel_size = float(
            voxel_pool_cfg.get("voxel_size", self.downsample_voxel)
        )
        self.local_pcd_voxel_pool_max_keys = int(
            voxel_pool_cfg.get("max_keys_per_object", self.max_points)
        )
        self.association_geometry_patch_budget_points = max(
            0,
            int(association_geometry_cfg.get("patch_budget_points", 0)),
        )
        self.association_geometry_update_budget_new_keys = max(
            0,
            int(association_geometry_cfg.get("update_budget_new_keys", 0)),
        )
        logger.info("ObjectUpdateModule initialized.")

    def _reset_stage_timings(self) -> None:
        self.last_stage_timings = {
            "object_update_current_frame_visibility_gate": 0.0,
            "object_update_surface_owner_gate": 0.0,
            "object_update_tsdf_integrate": 0.0,
            "object_update_local_pcd_update": 0.0,
            "object_update_association_geometry_update": 0.0,
            "object_update_refresh_object_debug": 0.0,
            "object_update_semantic_vote": 0.0,
            "object_update_provisional_pool": 0.0,
            "object_update_surface_gate_summary": 0.0,
        }

    @contextmanager
    def _timed_substage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.last_stage_timings[name] = self.last_stage_timings.get(name, 0.0) + float(
                time.perf_counter() - start
            )

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
        self._reset_stage_timings()
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
            with self._timed_substage("object_update_current_frame_visibility_gate"):
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
            with self._timed_substage("object_update_surface_owner_gate"):
                patch, structural_reject, gate_debug, voxel_view = self._filter_patch_by_surface_owner(
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
            # Step 1: Integrate into global TSDF
            with self._timed_substage("object_update_tsdf_integrate"):
                self.tsdf_module.integrate_patch(state.tsdf_volume, patch, obj_id, voxel_view=voxel_view)
            # Step 2+3: Update local geometry + whole evidence
            self._update_object(obj, patch)
            with self._timed_substage("object_update_refresh_object_debug"):
                self._refresh_object_debug(obj, state.tsdf_volume, current_frame=current_frame, created=False)
            obj.debug["last_current_frame_visibility_gate"] = visibility_debug
            obj.debug["last_surface_owner_gate"] = gate_debug
            self.last_updated_object_ids.append(int(obj_id))

        if self.contested_residual_enabled:
            for patch_id in association.contested_object_patches:
                patch = patch_map.get(patch_id)
                if patch is None:
                    continue
                with self._timed_substage("object_update_current_frame_visibility_gate"):
                    patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                        patch,
                        current_depth=current_depth,
                        current_pose=current_pose,
                        current_intrinsics=current_intrinsics,
                    )
                self._record_current_frame_visibility_gate_debug(visibility_debug)
                if patch is None:
                    continue
                with self._timed_substage("object_update_surface_owner_gate"):
                    patch, structural_reject, gate_debug, _voxel_view = self._filter_patch_by_surface_owner(
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
                    continue
                if not self._should_enter_provisional_pool(patch):
                    continue
                with self._timed_substage("object_update_provisional_pool"):
                    self._upsert_provisional_object(state, patch, contested=True)
                self.last_contested_residual_patch_ids.append(int(patch_id))
            if not self.provisional_enabled:
                with self._timed_substage("object_update_provisional_pool"):
                    self._promote_stable_provisionals(state, current_frame=current_frame)
                    self._prune_stale_provisionals(state, current_frame)

        # Create new objects or accumulate them in the provisional local pool.
        if self.provisional_enabled:
            for patch_id in association.new_object_patches:
                if self.contested_residual_enabled and int(patch_id) in contested_patch_ids:
                    continue
                patch = patch_map.get(patch_id)
                if patch is None:
                    continue
                with self._timed_substage("object_update_current_frame_visibility_gate"):
                    patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                        patch,
                        current_depth=current_depth,
                        current_pose=current_pose,
                        current_intrinsics=current_intrinsics,
                    )
                self._record_current_frame_visibility_gate_debug(visibility_debug)
                if patch is None:
                    continue
                with self._timed_substage("object_update_surface_owner_gate"):
                    patch, structural_reject, gate_debug, _voxel_view = self._filter_patch_by_surface_owner(
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
                    continue
                if not self._should_enter_provisional_pool(patch):
                    continue
                with self._timed_substage("object_update_provisional_pool"):
                    self._upsert_provisional_object(state, patch)
            with self._timed_substage("object_update_provisional_pool"):
                self._promote_stable_provisionals(state, current_frame=current_frame)
                self._prune_stale_provisionals(state, current_frame)
        else:
            for patch_id in association.new_object_patches:
                if self.contested_residual_enabled and int(patch_id) in contested_patch_ids:
                    continue
                patch = patch_map.get(patch_id)
                if patch is None:
                    continue
                with self._timed_substage("object_update_current_frame_visibility_gate"):
                    patch, visibility_debug = self._filter_patch_by_current_frame_visibility(
                        patch,
                        current_depth=current_depth,
                        current_pose=current_pose,
                        current_intrinsics=current_intrinsics,
                    )
                self._record_current_frame_visibility_gate_debug(visibility_debug)
                if patch is None:
                    continue
                with self._timed_substage("object_update_surface_owner_gate"):
                    patch, structural_reject, gate_debug, voxel_view = self._filter_patch_by_surface_owner(
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
                    continue
                if self._is_ambiguous_patch(patch):
                    continue
                with self._timed_substage("object_update_local_pcd_update"):
                    new_obj = self._create_object(patch, state.next_object_id)
                state.objects[new_obj.object_id] = new_obj
                # Integrate new object into global TSDF
                with self._timed_substage("object_update_tsdf_integrate"):
                    self.tsdf_module.integrate_patch(
                        state.tsdf_volume,
                        patch,
                        new_obj.object_id,
                        voxel_view=voxel_view,
                    )
                with self._timed_substage("object_update_refresh_object_debug"):
                    self._refresh_object_debug(new_obj, state.tsdf_volume, current_frame=current_frame, created=True)
                new_obj.debug["last_current_frame_visibility_gate"] = visibility_debug
                new_obj.debug["last_surface_owner_gate"] = gate_debug
                self.last_created_object_ids.append(int(new_obj.object_id))
                self.last_updated_object_ids.append(int(new_obj.object_id))
                state.next_object_id += 1

        with self._timed_substage("object_update_surface_gate_summary"):
            self.last_surface_gate_stats = self._summarize_surface_gate_records()
        self.last_current_frame_visibility_gate_stats = self._summarize_current_frame_visibility_gate_records()
        logger.debug(f"Objects updated. Total objects: {len(state.objects)}")
        self._decay_candidate_evidence(state, current_frame)
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
    ) -> tuple[Patch3D | None, Patch3D | None, dict[str, Any], PatchVoxelView | None]:
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
            return patch, None, debug, None

        inverse_policy = self.surface_gate_inverse_policy
        if inverse_policy == "auto":
            inverse_policy = "eager"
        eager_inverse = bool(
            self.surface_gate_representative_voxel_mode
            and inverse_policy == "eager"
        )
        unique_start = time.perf_counter()
        voxel_view = get_patch_voxel_view(
            patch,
            state.tsdf_volume.voxel_size,
            need_inverse=eager_inverse,
        )
        unique_sec = float(time.perf_counter() - unique_start)
        self.tsdf_module._ensure_support_index(state.tsdf_volume)
        voxel_owner_id = state.tsdf_volume.voxel_owner_id
        voxels = voxel_view.voxel_indices
        unique_voxels = voxel_view.unique_voxels
        cached_representative_indices = voxel_view.representative_point_indices
        cache_used = voxel_view.cache_used
        representative_indices = (
            cached_representative_indices
            if self.surface_gate_representative_voxel_mode
            else np.arange(point_count, dtype=np.int64)
        )
        decision_voxels = voxels[representative_indices]
        unique_voxel_count = int(len(unique_voxels))
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

        candidate_claim = state.tsdf_volume.candidate_claim if self.candidate_evidence_enabled else None
        evidence_threshold = self.candidate_evidence_threshold

        owner_lookup_start = time.perf_counter()
        for decision_idx, voxel in enumerate(decision_voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))

            bypass = False
            if candidate_claim is not None and anchor_label:
                claims = candidate_claim.get(key, {})
                evidence = int(claims.get(anchor_label, 0))
                bypass = evidence >= evidence_threshold

            owner_id = int(voxel_owner_id.get(key, -1))
            if allowed_owner_id is not None and owner_id == int(allowed_owner_id):
                decision_same_owner_mask[decision_idx] = True
            elif owner_id >= 0 and not bypass:
                decision_foreign_owner_mask[decision_idx] = True

            if self.surface_gate_background_enabled and not bypass:
                decision_background_owner_mask[decision_idx] = self_structural_background or key in background_support
        owner_lookup_sec = float(time.perf_counter() - owner_lookup_start)

        if self.candidate_evidence_enabled and new_object and anchor_label:
            self._accumulate_candidate_evidence(
                state.tsdf_volume,
                unique_voxels,
                anchor_label,
            )

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
        denominator = max(representative_count, 1) if self.surface_gate_representative_voxel_mode else max(point_count, 1)
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
            else decision_accepted_count
        )
        min_accept_count_unit = (
            "decision_voxel"
            if self.surface_gate_representative_voxel_mode
            else "point"
        )

        rejection_reasons: list[str] = []
        if foreign_ratio > self.surface_gate_max_foreign_owner_ratio:
            rejection_reasons.append("foreign_owner_ratio")
        if background_ratio > self.surface_gate_max_background_owner_ratio:
            rejection_reasons.append("background_owner_ratio")
        if min_accept_count < min_accept_points:
            rejection_reasons.append("insufficient_accepted_points")
        if accepted_ratio < min_accept_ratio:
            rejection_reasons.append("low_accept_ratio")

        clean_fast_path = bool(
            self.surface_gate_representative_voxel_mode
            and decision_foreign_count == 0
            and decision_background_count == 0
            and not rejection_reasons
        )
        if clean_fast_path:
            debug = {
                "enabled": True,
                "mode": mode,
                "target_object_id": -1 if target_object_id is None else int(target_object_id),
                "source_patch_id": int(patch.patch_id),
                "source_point_count": point_count,
                "unique_voxel_count": unique_voxel_count,
                "cached_voxel_indices_used": bool(cache_used),
                "decision_voxel_size": float(state.tsdf_volume.voxel_size),
                "representative_voxel_mode": True,
                "decision_voxel_count": representative_count,
                "accepted_decision_voxel_count": decision_accepted_count,
                "foreign_owner_decision_voxel_count": 0,
                "background_owner_decision_voxel_count": 0,
                "same_owner_decision_voxel_count": decision_same_owner_count,
                "accepted_point_count": point_count,
                "rejected_point_count": 0,
                "accepted_ratio": accepted_ratio,
                "foreign_owner_point_count": 0,
                "foreign_owner_ratio": foreign_ratio,
                "background_owner_point_count": 0,
                "background_owner_ratio": background_ratio,
                "same_owner_point_count": 0,
                "attached_surface_override": attached_override,
                "min_accept_points": int(min_accept_points),
                "min_accept_count_unit": min_accept_count_unit,
                "self_structural_background": self_structural_background,
                "anchor_class_name": anchor_label,
                "passed": True,
                "rejection_reasons": [],
                "fast_path": True,
                "point_mask_materialized": False,
                "patch_copied": False,
                "surface_gate_decision_lookup_count": representative_count,
                "surface_gate_unique_decision_voxel_count": unique_voxel_count,
                "surface_gate_owner_lookup_sec": owner_lookup_sec,
                "surface_gate_unique_sec": unique_sec,
                "surface_gate_mask_expand_sec": 0.0,
                "surface_gate_patch_copy_sec": 0.0,
                "surface_gate_structural_copy_sec": 0.0,
                "surface_gate_filtered_point_count": point_count,
                "surface_gate_rejected_point_count": 0,
                "surface_gate_partial_filter_patch_count": 0,
                "surface_gate_reject_without_copy_patch_count": 0,
                "surface_gate_inverse_requested_count": 1 if eager_inverse and voxel_view.inverse is not None else 0,
                "surface_gate_inverse_build_sec": 0.0,
                "surface_gate_clean_inverse_skipped_count": 0 if eager_inverse else 1,
                "surface_gate_inverse_policy": self.surface_gate_inverse_policy,
            }
            return self._copy_patch_with_metadata_only(patch, debug), None, debug, voxel_view

        inverse_requested_count = 0
        inverse_build_sec = 0.0
        if self.surface_gate_representative_voxel_mode:
            inverse_requested_count = 1
            inverse_start = time.perf_counter()
            inverse = voxel_view.inverse
            if inverse is None:
                _, _, inverse = np.unique(voxels, axis=0, return_index=True, return_inverse=True)
                inverse = inverse.astype(np.int64, copy=False)
            inverse_build_sec = float(time.perf_counter() - inverse_start)
            mask_expand_start = time.perf_counter()
            voxel_point_counts = np.bincount(inverse, minlength=representative_count).astype(np.int64, copy=False)
            accepted_count = int(voxel_point_counts[decision_accepted_mask].sum())
            rejected_count = int(point_count - accepted_count)
            foreign_count = int(voxel_point_counts[decision_foreign_owner_mask].sum())
            background_count = int(voxel_point_counts[decision_background_reject_mask].sum())
            same_owner_count = int(voxel_point_counts[decision_same_owner_mask].sum())
            accepted_mask = None
            background_reject_mask = None
            denominator = max(representative_count, 1)
        else:
            background_reject_mask = decision_background_reject_mask
            accepted_mask = decision_accepted_mask
            mask_expand_start = time.perf_counter()
            accepted_count = int(accepted_mask.sum())
            rejected_count = int(point_count - accepted_count)
            foreign_owner_mask = decision_foreign_owner_mask
            same_owner_mask = decision_same_owner_mask
            foreign_count = int(foreign_owner_mask.sum())
            background_count = int(background_reject_mask.sum())
            same_owner_count = int(same_owner_mask.sum())
            denominator = max(point_count, 1)
        mask_expand_sec = float(time.perf_counter() - mask_expand_start)

        debug = {
            "enabled": True,
            "mode": mode,
            "target_object_id": -1 if target_object_id is None else int(target_object_id),
            "source_patch_id": int(patch.patch_id),
            "source_point_count": point_count,
            "unique_voxel_count": unique_voxel_count,
            "cached_voxel_indices_used": bool(cache_used),
            "decision_voxel_size": float(state.tsdf_volume.voxel_size),
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
            "fast_path": False,
            "point_mask_materialized": False,
            "patch_copied": False,
            "surface_gate_decision_lookup_count": representative_count,
            "surface_gate_unique_decision_voxel_count": unique_voxel_count,
            "surface_gate_owner_lookup_sec": owner_lookup_sec,
            "surface_gate_unique_sec": unique_sec,
            "surface_gate_mask_expand_sec": mask_expand_sec,
            "surface_gate_patch_copy_sec": 0.0,
            "surface_gate_structural_copy_sec": 0.0,
            "surface_gate_filtered_point_count": accepted_count,
            "surface_gate_rejected_point_count": rejected_count,
            "surface_gate_partial_filter_patch_count": 0,
            "surface_gate_reject_without_copy_patch_count": 0,
            "surface_gate_inverse_requested_count": inverse_requested_count,
            "surface_gate_inverse_build_sec": inverse_build_sec,
            "surface_gate_clean_inverse_skipped_count": 0,
            "surface_gate_inverse_policy": self.surface_gate_inverse_policy,
        }

        structural_reject = None
        if background_count > 0:
            if background_reject_mask is None:
                mask_expand_start = time.perf_counter()
                background_reject_mask = decision_background_reject_mask[inverse]
                debug["surface_gate_mask_expand_sec"] += float(time.perf_counter() - mask_expand_start)
            debug["point_mask_materialized"] = True
            debug["patch_copied"] = True
            structural_copy_start = time.perf_counter()
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
            debug["surface_gate_structural_copy_sec"] = float(time.perf_counter() - structural_copy_start)
            if structural_reject is not None:
                structural_gate_debug = structural_reject.metadata.get("surface_owner_gate")
                if isinstance(structural_gate_debug, dict):
                    structural_gate_debug["surface_gate_structural_copy_sec"] = debug["surface_gate_structural_copy_sec"]

        if rejection_reasons:
            if structural_reject is None:
                debug["surface_gate_reject_without_copy_patch_count"] = 1
            return None, structural_reject, debug, None

        if accepted_mask is None:
            mask_expand_start = time.perf_counter()
            accepted_mask = decision_accepted_mask[inverse]
            debug["surface_gate_mask_expand_sec"] += float(time.perf_counter() - mask_expand_start)
        debug["point_mask_materialized"] = True
        debug["patch_copied"] = True
        debug["surface_gate_partial_filter_patch_count"] = 1 if rejected_count > 0 else 0
        patch_copy_start = time.perf_counter()
        filtered_patch = self._copy_patch_with_filtered_points(patch, accepted_mask, debug)
        debug["surface_gate_patch_copy_sec"] = float(time.perf_counter() - patch_copy_start)
        if filtered_patch is not None:
            filtered_gate_debug = filtered_patch.metadata.get("surface_owner_gate")
            if isinstance(filtered_gate_debug, dict):
                filtered_gate_debug["surface_gate_patch_copy_sec"] = debug["surface_gate_patch_copy_sec"]
        filtered_view = filter_patch_voxel_view(voxel_view, accepted_mask) if filtered_patch is not None else None
        return filtered_patch, structural_reject, debug, filtered_view

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
        for cache_key in (
            "voxel_indices",
            "unique_voxel_indices",
            "representative_point_indices",
            "voxel_size",
            "voxel_cache_point_count",
            "voxel_cache_bbox_min",
            "voxel_cache_bbox_max",
        ):
            metadata.pop(cache_key, None)
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

    def _copy_patch_with_metadata_only(
        self,
        patch: Patch3D,
        gate_debug: dict[str, Any],
        *,
        metadata_key: str = "surface_owner_gate",
    ) -> Patch3D:
        metadata = dict(patch.metadata)
        metadata[metadata_key] = gate_debug
        metadata.setdefault(f"{metadata_key}_original_lifted_point_count", int(len(patch.points)))
        return Patch3D(
            patch_id=patch.patch_id,
            points=patch.points,
            centroid=patch.centroid,
            bbox_min=patch.bbox_min,
            bbox_max=patch.bbox_max,
            normals=patch.normals,
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
                "surface_gate_fast_path_patch_count": 0,
                "surface_gate_point_mask_materialized_count": 0,
                "surface_gate_patch_copy_count": 0,
                "surface_gate_decision_lookup_count": 0,
                "surface_gate_unique_decision_voxel_count": 0,
                "surface_gate_owner_lookup_sec": 0.0,
                "surface_gate_unique_sec": 0.0,
                "surface_gate_mask_expand_sec": 0.0,
                "surface_gate_patch_copy_sec": 0.0,
                "surface_gate_structural_copy_sec": 0.0,
                "surface_gate_filtered_point_count": 0,
                "surface_gate_rejected_point_count": 0,
                "surface_gate_partial_filter_patch_count": 0,
                "surface_gate_reject_without_copy_patch_count": 0,
                "surface_gate_inverse_requested_count": 0,
                "surface_gate_inverse_build_sec": 0.0,
                "surface_gate_clean_inverse_skipped_count": 0,
                "surface_gate_inverse_policy": self.surface_gate_inverse_policy,
            }

        return {
            "enabled": bool(self.surface_owner_gate_enabled),
            "checked_patch_count": int(len(records)),
            "passed_patch_count": int(sum(1 for record in records if bool(record.get("passed", False)))),
            "rejected_patch_count": int(sum(1 for record in records if not bool(record.get("passed", False)))),
            "structural_reject_patch_count": int(len(self.last_structural_reject_patches)),
            "source_point_count": int(sum(int(record.get("source_point_count", 0)) for record in records)),
            "decision_voxel_count": int(sum(int(record.get("decision_voxel_count", 0)) for record in records)),
            "accepted_decision_voxel_count": int(
                sum(int(record.get("accepted_decision_voxel_count", 0)) for record in records)
            ),
            "foreign_owner_decision_voxel_count": int(
                sum(int(record.get("foreign_owner_decision_voxel_count", 0)) for record in records)
            ),
            "background_owner_decision_voxel_count": int(
                sum(int(record.get("background_owner_decision_voxel_count", 0)) for record in records)
            ),
            "accepted_point_count": int(sum(int(record.get("accepted_point_count", 0)) for record in records)),
            "foreign_owner_point_count": int(
                sum(int(record.get("foreign_owner_point_count", 0)) for record in records)
            ),
            "background_owner_point_count": int(
                sum(int(record.get("background_owner_point_count", 0)) for record in records)
            ),
            "surface_gate_fast_path_patch_count": int(sum(1 for record in records if bool(record.get("fast_path", False)))),
            "surface_gate_point_mask_materialized_count": int(
                sum(1 for record in records if bool(record.get("point_mask_materialized", False)))
            ),
            "surface_gate_patch_copy_count": int(sum(1 for record in records if bool(record.get("patch_copied", False)))),
            "surface_gate_decision_lookup_count": int(
                sum(int(record.get("surface_gate_decision_lookup_count", 0)) for record in records)
            ),
            "surface_gate_unique_decision_voxel_count": int(
                sum(int(record.get("surface_gate_unique_decision_voxel_count", 0)) for record in records)
            ),
            "surface_gate_owner_lookup_sec": float(
                sum(float(record.get("surface_gate_owner_lookup_sec", 0.0)) for record in records)
            ),
            "surface_gate_unique_sec": float(
                sum(float(record.get("surface_gate_unique_sec", 0.0)) for record in records)
            ),
            "surface_gate_mask_expand_sec": float(
                sum(float(record.get("surface_gate_mask_expand_sec", 0.0)) for record in records)
            ),
            "surface_gate_patch_copy_sec": float(
                sum(float(record.get("surface_gate_patch_copy_sec", 0.0)) for record in records)
            ),
            "surface_gate_structural_copy_sec": float(
                sum(float(record.get("surface_gate_structural_copy_sec", 0.0)) for record in records)
            ),
            "surface_gate_filtered_point_count": int(
                sum(int(record.get("surface_gate_filtered_point_count", 0)) for record in records)
            ),
            "surface_gate_rejected_point_count": int(
                sum(int(record.get("surface_gate_rejected_point_count", 0)) for record in records)
            ),
            "surface_gate_partial_filter_patch_count": int(
                sum(int(record.get("surface_gate_partial_filter_patch_count", 0)) for record in records)
            ),
            "surface_gate_reject_without_copy_patch_count": int(
                sum(int(record.get("surface_gate_reject_without_copy_patch_count", 0)) for record in records)
            ),
            "surface_gate_inverse_requested_count": int(
                sum(int(record.get("surface_gate_inverse_requested_count", 0)) for record in records)
            ),
            "surface_gate_inverse_build_sec": float(
                sum(float(record.get("surface_gate_inverse_build_sec", 0.0)) for record in records)
            ),
            "surface_gate_clean_inverse_skipped_count": int(
                sum(int(record.get("surface_gate_clean_inverse_skipped_count", 0)) for record in records)
            ),
            "surface_gate_inverse_policy": self.surface_gate_inverse_policy,
        }

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

    @staticmethod
    def _pack_voxel(v: np.ndarray) -> np.ndarray:
        """Pack (N,3) int64 voxel coords → (N,) int64 keys (21 bits/axis)."""
        v = v.astype(np.int64)
        mask = (1 << 21) - 1
        return ((v[:, 0] & mask) << 42) | ((v[:, 1] & mask) << 21) | (v[:, 2] & mask)

    def _local_pcd_voxel_pool_state(self, obj: ObjectMap) -> dict[str, Any]:
        state = obj.debug.get("_local_pcd_voxel_pool")
        if isinstance(state, dict) and "_keys_1d" in state:
            return state

        # Migrate from old tuple-key dict format if present
        old_dict = state.get("point_by_key", None) if isinstance(state, dict) else None
        if isinstance(old_dict, dict) and old_dict:
            order = sorted(old_dict.items(), key=lambda item: item[0])
            k1d = self._pack_voxel(np.array([k for k, _ in order], dtype=np.int64))
            vv = np.array([v for _, v in order], dtype=np.float32)
        else:
            points = np.asarray(obj.local_pcd, dtype=np.float32)
            if len(points) > 0:
                voxels = self._points_to_voxels(points, self.local_pcd_voxel_pool_voxel_size)
                _, fi = np.unique(voxels, axis=0, return_index=True)
                fi = np.sort(fi.astype(np.int64))
                k1d = self._pack_voxel(voxels[fi])
                vv = points[fi].astype(np.float32, copy=True)
            else:
                k1d = np.array([], dtype=np.int64)
                vv = np.empty((0, 3), dtype=np.float32)

        if self.local_pcd_voxel_pool_max_keys > 0 and len(k1d) > self.local_pcd_voxel_pool_max_keys:
            k1d, vv = self._cap_point_by_key_arrays(k1d, vv, self.local_pcd_voxel_pool_max_keys)

        state = {
            "_keys_1d": k1d,
            "_vals": vv,
            "point_sum": vv.sum(axis=0, dtype=np.float64) if len(vv) > 0 else np.zeros(3, dtype=np.float64),
            "revision": 0,
            "materialized_revision": 0,
            "insert_count": 0,
            "update_count": 0,
            "dropped_key_count": 0,
            "materialize_count": 0,
            "materialize_sec": 0.0,
            "cap_applied_count": 0,
            "last_materialized_point_count": int(len(k1d)),
            "voxel_size": float(self.local_pcd_voxel_pool_voxel_size),
        }
        obj.debug["_local_pcd_voxel_pool"] = state
        return state

    @staticmethod
    def _cap_point_by_key_arrays(
        keys_1d: np.ndarray,
        vals: np.ndarray,
        max_keys: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform subsample sorted-by-key arrays down to max_keys."""
        if max_keys <= 0 or len(keys_1d) <= max_keys:
            return keys_1d.copy(), vals.copy()
        order = np.argsort(keys_1d)
        ix = order[np.linspace(0, len(order) - 1, num=max_keys, dtype=np.int64)]
        return keys_1d[ix].copy(), vals[ix].astype(np.float32, copy=True)

    def _update_local_pcd_voxel_pool(self, obj: ObjectMap, patch_points: np.ndarray) -> bool:
        pts = np.asarray(patch_points, dtype=np.float32)
        if len(pts) == 0:
            return False
        state = self._local_pcd_voxel_pool_state(obj)
        k1d = state["_keys_1d"]
        vv = state["_vals"]
        point_sum = np.asarray(state["point_sum"], dtype=np.float64)

        voxels = self._points_to_voxels(pts, self.local_pcd_voxel_pool_voxel_size)
        _, fi = np.unique(voxels, axis=0, return_index=True)
        nk = self._pack_voxel(voxels[fi])
        nv = pts[fi]

        if len(k1d) == 0:
            state["_keys_1d"] = nk
            state["_vals"] = nv.astype(np.float32, copy=True)
            state["point_sum"] = nv.sum(axis=0, dtype=np.float64)
            state["revision"] = int(state.get("revision", 0)) + 1
            state["insert_count"] = int(state.get("insert_count", 0)) + len(nk)
            return True

        upd_mask = np.isin(nk, k1d)
        ins_mask = ~upd_mask
        n_upd = int(upd_mask.sum())
        n_ins = int(ins_mask.sum())

        if n_upd > 0:
            srt = np.argsort(k1d)
            pos = srt[np.searchsorted(k1d, nk[upd_mask], sorter=srt)]
            point_sum -= vv[pos].sum(axis=0, dtype=np.float64)
            point_sum += nv[upd_mask].sum(axis=0, dtype=np.float64)
            vv[pos] = nv[upd_mask]

        if n_ins > 0:
            k1d = np.concatenate([k1d, nk[ins_mask]])
            vv = np.concatenate([vv, nv[ins_mask]], axis=0)
            point_sum += nv[ins_mask].sum(axis=0, dtype=np.float64)

        cap_applied = False
        mk = self.local_pcd_voxel_pool_max_keys
        if mk > 0 and len(k1d) > mk:
            before = len(k1d)
            k1d, vv = self._cap_point_by_key_arrays(k1d, vv, mk)
            point_sum = vv.sum(axis=0, dtype=np.float64)
            cap_applied = True
            state["dropped_key_count"] = int(state.get("dropped_key_count", 0)) + max(0, before - len(k1d))

        state["_keys_1d"] = k1d
        state["_vals"] = vv
        state["point_sum"] = point_sum
        state["revision"] = int(state.get("revision", 0)) + 1
        state["insert_count"] = int(state.get("insert_count", 0)) + n_ins
        state["update_count"] = int(state.get("update_count", 0)) + n_upd
        if cap_applied:
            state["cap_applied_count"] = int(state.get("cap_applied_count", 0)) + 1
        return True

    def _materialize_local_pcd_voxel_pool(self, obj: ObjectMap, *, force: bool = False) -> bool:
        state = self._local_pcd_voxel_pool_state(obj)
        revision = int(state.get("revision", 0))
        materialized_revision = int(state.get("materialized_revision", -1))
        if not force and revision == materialized_revision:
            return False
        start = time.perf_counter()
        k1d = state["_keys_1d"]
        vv = state["_vals"]
        if len(k1d) > 0:
            order = np.argsort(k1d)
            materialized = vv[order].astype(np.float32, copy=True)
        else:
            materialized = np.empty((0, 3), dtype=np.float32)
        if self.max_points > 0 and len(materialized) > self.max_points:
            materialized = self._deterministic_spatial_cap(materialized, self.max_points)
            vx = self._points_to_voxels(materialized, self.local_pcd_voxel_pool_voxel_size)
            _, fi = np.unique(vx, axis=0, return_index=True)
            fi = np.sort(fi.astype(np.int64))
            k1d = self._pack_voxel(vx[fi])
            vv = materialized[fi].astype(np.float32, copy=True)
            state["_keys_1d"] = k1d
            state["_vals"] = vv
            state["point_sum"] = vv.sum(axis=0, dtype=np.float64)
            state["cap_applied_count"] = int(state.get("cap_applied_count", 0)) + 1
        obj.local_pcd = materialized
        elapsed = float(time.perf_counter() - start)
        state["materialized_revision"] = int(state.get("revision", 0))
        state["materialize_count"] = int(state.get("materialize_count", 0)) + 1
        state["materialize_sec"] = float(state.get("materialize_sec", 0.0)) + elapsed
        state["last_materialized_point_count"] = int(len(obj.local_pcd))
        return True

    def _local_pcd_voxel_key(self, point: np.ndarray) -> tuple[int, int, int]:
        voxel = np.floor(
            np.asarray(point, dtype=np.float32) / max(float(self.local_pcd_voxel_pool_voxel_size), 1e-9)
        ).astype(np.int64)
        return (int(voxel[0]), int(voxel[1]), int(voxel[2]))

    @staticmethod
    def _accumulate_candidate_evidence(
        volume: "TSDFInstanceVolume",
        unique_voxels: np.ndarray,
        anchor_label: str,
    ) -> None:
        """Increment candidate claim count for each unique voxel + anchor class.

        Called for ALL new_object patches regardless of gate outcome.
        One claim per unique voxel per frame.
        """
        if not anchor_label or len(unique_voxels) == 0:
            return
        candidate_claim = volume.candidate_claim
        for uv in unique_voxels:
            key = (int(uv[0]), int(uv[1]), int(uv[2]))
            claims = candidate_claim.get(key)
            if claims is None:
                candidate_claim[key] = {anchor_label: 1}
            else:
                claims[anchor_label] = claims.get(anchor_label, 0) + 1

    def _decay_candidate_evidence(self, state: "SystemState", current_frame: int) -> None:
        """Decay candidate claim counts and remove stale entries. Cap total voxels."""
        if not self.candidate_evidence_enabled:
            return
        if current_frame % self.candidate_evidence_decay_interval != 0:
            return

        volume = state.tsdf_volume
        candidate_claim = volume.candidate_claim
        if not candidate_claim:
            return

        decay_factor = 0.9
        keys_to_delete = []

        for voxel_key, claims in candidate_claim.items():
            for cls in list(claims.keys()):
                claims[cls] *= decay_factor
                if claims[cls] < 0.5:
                    del claims[cls]
            if not claims:
                keys_to_delete.append(voxel_key)

        for key in keys_to_delete:
            del candidate_claim[key]

        # Cap total voxels if exceeded
        if len(candidate_claim) > self.candidate_evidence_max_voxels:
            items = sorted(
                candidate_claim.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True,
            )
            for voxel_key, _ in items[self.candidate_evidence_max_voxels:]:
                del candidate_claim[voxel_key]

        volume.candidate_claim_max_voxels = self.candidate_evidence_max_voxels

    def _local_pcd_voxel_pool_debug(
        self,
        obj: ObjectMap,
        *,
        materialized_this_update: bool,
        append_only_update: bool,
    ) -> dict[str, Any]:
        state = self._local_pcd_voxel_pool_state(obj)
        key_count = int(len(state.get("_keys_1d", [])))
        return {
            "chunk_pool_enabled": False,
            "local_pcd_voxel_pool_enabled": True,
            "append_only_update": bool(append_only_update),
            "materialized_this_update": bool(materialized_this_update),
            "materialized_point_count": int(len(obj.local_pcd)),
            "pending_chunk_count": 0,
            "pending_point_count": 0,
            "pending_point_peak": 0,
            "local_pcd_pending_point_count": 0,
            "local_pcd_pending_point_peak": 0,
            "local_point_count": int(key_count),
            "local_pcd_voxel_key_count": int(key_count),
            "local_pcd_voxel_pool_point_count": int(key_count),
            "local_pcd_voxel_insert_count": int(state.get("insert_count", 0)),
            "local_pcd_voxel_update_count": int(state.get("update_count", 0)),
            "local_pcd_voxel_dropped_key_count": int(state.get("dropped_key_count", 0)),
            "local_pcd_materialize_count": int(state.get("materialize_count", 0)),
            "local_pcd_materialize_sec": float(state.get("materialize_sec", 0.0)),
            "local_pcd_cap_applied_count": int(state.get("cap_applied_count", 0)),
            "local_pcd_snapshot_stale": bool(
                int(state.get("revision", 0)) != int(state.get("materialized_revision", -1))
            ),
            "local_pcd_compaction_count": int(state.get("materialize_count", 0)),
            "local_pcd_compaction_sec": float(state.get("materialize_sec", 0.0)),
            "compaction_count": int(state.get("materialize_count", 0)),
            "compaction_sec": float(state.get("materialize_sec", 0.0)),
            "bounded_compaction_enabled": True,
            "compact_to_max_points": True,
        }

    def _local_pcd_pool_state(self, obj: ObjectMap) -> dict[str, Any]:
        state = obj.debug.get("_local_pcd_chunk_pool")
        if not isinstance(state, dict):
            state = {
                "pending_chunks": [],
                "pending_point_count": 0,
                "pending_point_peak": 0,
                "materialized_point_count": int(len(obj.local_pcd)),
                "compaction_count": 0,
                "compaction_sec": 0.0,
                "compacted_point_count_before": 0,
                "compacted_point_count_after": 0,
                "cap_applied_count": 0,
            }
            obj.debug["_local_pcd_chunk_pool"] = state
        return state

    def _reset_local_pcd_pool_state(self, obj: ObjectMap) -> None:
        old_state = obj.debug.get("_local_pcd_chunk_pool")
        old_state = old_state if isinstance(old_state, dict) else {}
        obj.debug["_local_pcd_chunk_pool"] = {
            "pending_chunks": [],
            "pending_point_count": 0,
            "pending_point_peak": int(old_state.get("pending_point_peak", 0)),
            "materialized_point_count": int(len(obj.local_pcd)),
            "compaction_count": int(old_state.get("compaction_count", 0)),
            "compaction_sec": float(old_state.get("compaction_sec", 0.0)),
            "compacted_point_count_before": int(old_state.get("compacted_point_count_before", 0)),
            "compacted_point_count_after": int(old_state.get("compacted_point_count_after", 0)),
            "cap_applied_count": int(old_state.get("cap_applied_count", 0)),
        }

    def _materialize_local_pcd_chunks(self, obj: ObjectMap, *, compact: bool | None = None) -> bool:
        pool = self._local_pcd_pool_state(obj)
        chunks = list(pool.get("pending_chunks", []) or [])
        should_compact = (
            self.local_pcd_chunk_pool_bounded_compaction_enabled
            if compact is None
            else bool(compact)
        )
        if not chunks and not should_compact:
            pool["materialized_point_count"] = int(len(obj.local_pcd))
            pool["pending_point_count"] = 0
            return False
        start = time.perf_counter()
        arrays = [np.asarray(obj.local_pcd, dtype=np.float32)]
        arrays.extend(np.asarray(chunk, dtype=np.float32) for chunk in chunks if len(chunk) > 0)
        materialized = (
            np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
            if arrays
            else np.empty((0, 3), dtype=np.float32)
        )
        before_count = int(len(materialized))
        cap_applied = False
        changed_without_chunks = False
        if should_compact and len(materialized) > 0:
            materialized = voxel_downsample(materialized, self.downsample_voxel)
            if (
                self.local_pcd_chunk_pool_compact_to_max_points
                and self.max_points > 0
                and len(materialized) > self.max_points
            ):
                materialized = self._deterministic_spatial_cap(materialized, self.max_points)
                cap_applied = True
            changed_without_chunks = not chunks and int(len(materialized)) != int(len(obj.local_pcd))
        obj.local_pcd = np.asarray(materialized, dtype=np.float32)
        elapsed = float(time.perf_counter() - start)
        pool["pending_chunks"] = []
        pool["pending_point_count"] = 0
        pool["pending_point_peak"] = max(
            int(pool.get("pending_point_peak", 0)),
            int(sum(len(chunk) for chunk in chunks)),
        )
        pool["materialized_point_count"] = int(len(obj.local_pcd))
        if should_compact:
            pool["compaction_count"] = int(pool.get("compaction_count", 0)) + 1
            pool["compaction_sec"] = float(pool.get("compaction_sec", 0.0)) + elapsed
            pool["compacted_point_count_before"] = before_count
            pool["compacted_point_count_after"] = int(len(obj.local_pcd))
            if cap_applied:
                pool["cap_applied_count"] = int(pool.get("cap_applied_count", 0)) + 1
        return bool(chunks or changed_without_chunks)

    def _append_local_pcd_chunk(self, obj: ObjectMap, patch_points: np.ndarray) -> int:
        if len(patch_points) == 0:
            return 0
        pool = self._local_pcd_pool_state(obj)
        chunks = list(pool.get("pending_chunks", []) or [])
        chunk = np.asarray(patch_points, dtype=np.float32)
        chunks.append(chunk.copy())
        pending_count = int(pool.get("pending_point_count", 0)) + int(len(chunk))
        pool["pending_chunks"] = chunks
        pool["pending_point_count"] = pending_count
        pool["pending_point_peak"] = max(int(pool.get("pending_point_peak", 0)), pending_count)
        pool["materialized_point_count"] = int(len(obj.local_pcd))
        return pending_count

    def _local_pcd_total_count(self, obj: ObjectMap) -> int:
        if self.local_pcd_voxel_pool_enabled:
            state = self._local_pcd_voxel_pool_state(obj)
            return int(len(state.get("_keys_1d", [])))
        if not self.local_pcd_chunk_pool_enabled:
            return int(len(obj.local_pcd))
        pool = self._local_pcd_pool_state(obj)
        return int(len(obj.local_pcd)) + int(pool.get("pending_point_count", 0))

    def _local_pcd_pool_debug(self, obj: ObjectMap, *, materialized_this_update: bool, append_only_update: bool) -> dict[str, Any]:
        pool = self._local_pcd_pool_state(obj)
        pending_chunks = list(pool.get("pending_chunks", []) or [])
        pending_point_count = int(pool.get("pending_point_count", 0))
        pending_point_peak = int(pool.get("pending_point_peak", 0))
        compaction_count = int(pool.get("compaction_count", 0))
        compaction_sec = float(pool.get("compaction_sec", 0.0))
        before_count = int(pool.get("compacted_point_count_before", 0))
        after_count = int(pool.get("compacted_point_count_after", 0))
        cap_count = int(pool.get("cap_applied_count", 0))
        return {
            "chunk_pool_enabled": bool(self.local_pcd_chunk_pool_enabled),
            "append_only_update": bool(append_only_update),
            "materialized_this_update": bool(materialized_this_update),
            "materialized_point_count": int(len(obj.local_pcd)),
            "pending_chunk_count": int(len(pending_chunks)),
            "pending_point_count": pending_point_count,
            "pending_point_peak": pending_point_peak,
            "local_pcd_pending_point_count": pending_point_count,
            "local_pcd_pending_point_peak": pending_point_peak,
            "local_point_count": self._local_pcd_total_count(obj),
            "materialize_interval": int(self.local_pcd_chunk_pool_materialize_interval),
            "max_pending_points": int(self.local_pcd_chunk_pool_max_pending_points),
            "bounded_compaction_enabled": bool(self.local_pcd_chunk_pool_bounded_compaction_enabled),
            "compact_to_max_points": bool(self.local_pcd_chunk_pool_compact_to_max_points),
            "compaction_count": compaction_count,
            "compaction_sec": compaction_sec,
            "compacted_point_count_before": before_count,
            "compacted_point_count_after": after_count,
            "cap_applied_count": cap_count,
            "local_pcd_compaction_count": compaction_count,
            "local_pcd_compaction_sec": compaction_sec,
            "local_pcd_compacted_point_count_before": before_count,
            "local_pcd_compacted_point_count_after": after_count,
            "local_pcd_cap_applied_count": cap_count,
        }

    def flush_deferred_geometry(self, state: SystemState) -> int:
        """Materialize deferred local-pcd chunks before checkpoint/export consumers read arrays."""
        if (
            not self.local_pcd_voxel_pool_enabled
            and not self.local_pcd_chunk_pool_enabled
            and not self.association_geometry_sketch_enabled
        ):
            return 0
        flushed = 0
        for obj in state.objects.values():
            if self.local_pcd_voxel_pool_enabled:
                if self._materialize_local_pcd_voxel_pool(obj, force=True):
                    flushed += 1
            elif self.local_pcd_chunk_pool_enabled and self._materialize_local_pcd_chunks(obj):
                flushed += 1
            if self.association_geometry_enabled:
                if self.association_geometry_sketch_enabled:
                    self._materialize_association_geometry_sketch(
                        obj,
                        source_point_count=self._local_pcd_total_count(obj),
                        force_revision=True,
                    )
                else:
                    self._refresh_association_geometry(obj)
            if self.local_pcd_voxel_pool_enabled:
                local_debug = dict(obj.debug.get("local_geometry_pool", {}) or {})
                local_debug.update(
                    self._local_pcd_voxel_pool_debug(
                        obj,
                        materialized_this_update=False,
                        append_only_update=False,
                    )
                )
                local_debug["flushed_for_export"] = True
                obj.debug["local_geometry_pool"] = local_debug
                obj.debug.pop("_local_pcd_voxel_pool", None)
            elif self.local_pcd_chunk_pool_enabled:
                local_debug = dict(obj.debug.get("local_geometry_pool", {}) or {})
                local_debug.update(
                    self._local_pcd_pool_debug(
                        obj,
                        materialized_this_update=False,
                        append_only_update=False,
                    )
                )
                local_debug["flushed_for_export"] = True
                obj.debug["local_geometry_pool"] = local_debug
                obj.debug.pop("_local_pcd_chunk_pool", None)
        return flushed

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

    def _promote_stable_provisionals(self, state: SystemState, *, current_frame: int | None = None) -> None:
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
                with self._timed_substage("object_update_tsdf_integrate"):
                    self.tsdf_module.integrate_patch(state.tsdf_volume, observation.patch, obj.object_id)
            with self._timed_substage("object_update_refresh_object_debug"):
                self._refresh_object_debug(
                    obj,
                    state.tsdf_volume,
                    current_frame=current_frame,
                    created=True,
                )
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
        if self.association_geometry_sketch_enabled:
            self._reset_association_geometry_sketch(obj)

    def _update_association_geometry_from_patch(self, obj: ObjectMap, patch: Patch3D) -> None:
        source_point_count = self._local_pcd_total_count(obj)
        if not self.association_geometry_enabled:
            obj.association_pcd = np.empty((0, 3), dtype=np.float32)
            obj.debug["association_geometry"] = self._association_geometry_debug(
                point_count=0,
                source_point_count=source_point_count,
            )
            return

        patch_points = np.asarray(patch.points, dtype=np.float32)
        if len(patch_points) == 0:
            obj.debug["association_geometry"] = self._association_geometry_debug(
                point_count=len(obj.association_pcd),
                source_point_count=source_point_count,
            )
            return

        if self.association_geometry_sketch_enabled:
            self._update_association_geometry_sketch_from_patch(obj, patch_points, source_point_count)
            return

        patch_rep = self._association_patch_representatives(patch_points)
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
        debug = {
            "role": "bounded_association_geometry",
            "enabled": bool(self.association_geometry_enabled),
            "point_count": int(point_count),
            "source_point_count": int(source_point_count),
            "voxel_size": float(self.association_geometry_voxel),
            "max_points_per_object": int(self.association_geometry_max_points),
            "max_points": int(self.association_geometry_max_points),
            "sketch_enabled": bool(self.association_geometry_sketch_enabled),
            "materialize_interval": int(self.association_geometry_materialize_interval),
            "patch_budget_points": int(self.association_geometry_patch_budget_points),
            "update_budget_new_keys": int(self.association_geometry_update_budget_new_keys),
        }
        return debug

    def _association_patch_representatives(self, patch_points: np.ndarray) -> np.ndarray:
        patch_points = np.asarray(patch_points, dtype=np.float32)
        if len(patch_points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        if (
            self.association_geometry_patch_budget_points > 0
            and len(patch_points) > self.association_geometry_patch_budget_points
        ):
            patch_points = self._deterministic_spatial_cap(
                patch_points,
                self.association_geometry_patch_budget_points,
            )
        patch_rep = voxel_downsample(patch_points, self.association_geometry_voxel)
        if (
            self.association_geometry_max_points > 0
            and self.association_geometry_patch_pre_cap_factor > 0.0
            and len(patch_rep) > int(self.association_geometry_max_points * self.association_geometry_patch_pre_cap_factor)
        ):
            patch_rep = self._deterministic_spatial_cap(
                patch_rep,
                max(1, int(self.association_geometry_max_points * self.association_geometry_patch_pre_cap_factor)),
            )
        return np.asarray(patch_rep, dtype=np.float32)

    def _association_voxel_key(self, point: np.ndarray) -> tuple[int, int, int]:
        voxel = np.floor(np.asarray(point, dtype=np.float32) / self.association_geometry_voxel).astype(np.int64)
        return (int(voxel[0]), int(voxel[1]), int(voxel[2]))

    def _reset_association_geometry_sketch(self, obj: ObjectMap) -> None:
        association_points = np.asarray(
            getattr(obj, "association_pcd", np.empty((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        point_by_key = {
            self._association_voxel_key(point): np.asarray(point, dtype=np.float32)
            for point in association_points
        }
        debug = obj.debug.setdefault("association_geometry", {})
        revision = int(debug.get("association_geometry_revision", debug.get("revision", 0))) + 1
        self._association_geometry_sketches[int(obj.object_id)] = point_by_key
        debug["association_geometry_revision"] = revision
        debug["revision"] = revision
        debug["sketch_key_count"] = int(len(point_by_key))
        debug["sketch_pending_update_count"] = 0
        debug["materialized_from_sketch"] = True
        debug.setdefault("sketch_update_count", 0)
        debug.setdefault("sketch_new_key_count", 0)
        debug.setdefault("sketch_dropped_key_count", 0)
        debug.setdefault("sketch_materialize_count", 0)
        debug.setdefault("sketch_update_sec", 0.0)
        debug.setdefault("sketch_materialize_sec", 0.0)

    def _update_association_geometry_sketch_from_patch(
        self,
        obj: ObjectMap,
        patch_points: np.ndarray,
        source_point_count: int,
    ) -> None:
        existing_points = np.asarray(
            getattr(obj, "association_pcd", np.empty((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        association_debug = obj.debug.get("association_geometry", {})
        if not isinstance(association_debug, dict) or int(obj.object_id) not in self._association_geometry_sketches:
            self._reset_association_geometry_sketch(obj)
            association_debug = obj.debug.get("association_geometry", {})

        point_by_key = self._association_geometry_sketches.get(int(obj.object_id), {})
        if not isinstance(point_by_key, dict):
            point_by_key = {}
        start = time.perf_counter()
        patch_rep = self._association_patch_representatives(patch_points)
        new_key_count = 0
        dropped_key_count = 0
        key_budget = int(self.association_geometry_update_budget_new_keys)
        for point in patch_rep:
            key = self._association_voxel_key(point)
            if key in point_by_key:
                continue
            if key_budget > 0 and new_key_count >= key_budget:
                dropped_key_count += 1
                continue
            point_by_key[key] = np.asarray(point, dtype=np.float32)
            new_key_count += 1

        self._association_geometry_sketches[int(obj.object_id)] = point_by_key
        pending_update_count = int(association_debug.get("sketch_pending_update_count", 0)) + 1
        previous_update_count = int(association_debug.get("sketch_update_count", 0))
        previous_new_key_count = int(
            association_debug.get(
                "sketch_new_key_count_total",
                association_debug.get("association_geometry_sketch_new_key_count", 0),
            )
        )
        previous_dropped_key_count = int(
            association_debug.get(
                "sketch_dropped_key_count_total",
                association_debug.get("association_geometry_sketch_dropped_key_count", 0),
            )
        )
        previous_update_sec = float(association_debug.get("sketch_update_sec", 0.0))
        update_sec = float(time.perf_counter() - start)
        materialize_due = bool(
            pending_update_count >= self.association_geometry_materialize_interval
            or obj.update_count % self.association_geometry_materialize_interval == 0
        )
        if new_key_count == 0 or not materialize_due:
            existing_revision = int(
                association_debug.get(
                    "association_geometry_revision",
                    association_debug.get("revision", 0),
                )
            )
            obj.debug["association_geometry"] = {
                **self._association_geometry_debug(
                    point_count=len(existing_points),
                    source_point_count=source_point_count,
                ),
                "association_geometry_revision": existing_revision,
                "revision": existing_revision,
                "sketch_key_count": int(len(point_by_key)),
                "sketch_new_key_count": int(new_key_count),
                "sketch_dropped_key_count": int(dropped_key_count),
                "sketch_pending_update_count": int(pending_update_count),
                "sketch_update_count": previous_update_count + 1,
                "sketch_new_key_count_total": previous_new_key_count + new_key_count,
                "sketch_dropped_key_count_total": previous_dropped_key_count + dropped_key_count,
                "sketch_materialize_count": int(association_debug.get("sketch_materialize_count", 0)),
                "sketch_update_sec": previous_update_sec + update_sec,
                "sketch_materialize_sec": float(association_debug.get("sketch_materialize_sec", 0.0)),
                "association_geometry_sketch_update_count": previous_update_count + 1,
                "association_geometry_sketch_new_key_count": previous_new_key_count + new_key_count,
                "association_geometry_sketch_dropped_key_count": previous_dropped_key_count + dropped_key_count,
                "association_geometry_sketch_materialize_count": int(association_debug.get("sketch_materialize_count", 0)),
                "association_geometry_sketch_update_sec": previous_update_sec + update_sec,
                "association_geometry_sketch_materialize_sec": float(association_debug.get("sketch_materialize_sec", 0.0)),
                "materialized_from_sketch": False,
            }
            return

        self._materialize_association_geometry_sketch(
            obj,
            source_point_count=source_point_count,
            new_key_count=new_key_count,
            dropped_key_count=dropped_key_count,
            pending_update_count=pending_update_count,
            update_sec=update_sec,
        )

    def _materialize_association_geometry_sketch(
        self,
        obj: ObjectMap,
        *,
        source_point_count: int,
        new_key_count: int = 0,
        dropped_key_count: int = 0,
        pending_update_count: int | None = None,
        update_sec: float = 0.0,
        force_revision: bool = False,
    ) -> bool:
        start = time.perf_counter()
        association_debug = obj.debug.get("association_geometry", {})
        if not isinstance(association_debug, dict) or int(obj.object_id) not in self._association_geometry_sketches:
            self._reset_association_geometry_sketch(obj)
            association_debug = obj.debug.get("association_geometry", {})

        point_by_key = self._association_geometry_sketches.get(int(obj.object_id), {})
        if not isinstance(point_by_key, dict):
            point_by_key = {}
        sketch_points = (
            np.asarray(list(point_by_key.values()), dtype=np.float32)
            if point_by_key
            else np.empty((0, 3), dtype=np.float32)
        )
        if (
            self.association_geometry_max_points > 0
            and len(sketch_points) > self.association_geometry_max_points
        ):
            sketch_points = self._deterministic_spatial_cap(
                sketch_points,
                self.association_geometry_max_points,
            )
            point_by_key = {
                self._association_voxel_key(point): np.asarray(point, dtype=np.float32)
                for point in sketch_points
            }
        self._association_geometry_sketches[int(obj.object_id)] = point_by_key
        obj.association_pcd = np.asarray(sketch_points, dtype=np.float32, copy=False)
        previous_revision = int(
            association_debug.get(
                "association_geometry_revision",
                association_debug.get("revision", 0),
            )
        )
        revision = previous_revision + 1 if (force_revision or new_key_count > 0) else previous_revision
        materialize_sec = float(time.perf_counter() - start)
        obj.debug["association_geometry"] = {
            **self._association_geometry_debug(
                point_count=len(obj.association_pcd),
                source_point_count=source_point_count,
            ),
            "association_geometry_revision": int(revision),
            "revision": int(revision),
            "sketch_key_count": int(len(point_by_key)),
            "sketch_new_key_count": int(new_key_count),
            "sketch_dropped_key_count": int(dropped_key_count),
            "sketch_pending_update_count": 0,
            "sketch_update_count": int(association_debug.get("sketch_update_count", 0)) + (1 if pending_update_count is not None else 0),
            "sketch_new_key_count_total": int(association_debug.get("sketch_new_key_count_total", association_debug.get("association_geometry_sketch_new_key_count", 0))) + int(new_key_count),
            "sketch_dropped_key_count_total": int(association_debug.get("sketch_dropped_key_count_total", association_debug.get("association_geometry_sketch_dropped_key_count", 0))) + int(dropped_key_count),
            "sketch_materialize_count": int(association_debug.get("sketch_materialize_count", 0)) + 1,
            "sketch_update_sec": float(association_debug.get("sketch_update_sec", 0.0)) + float(update_sec),
            "sketch_materialize_sec": float(association_debug.get("sketch_materialize_sec", 0.0)) + materialize_sec,
            "association_geometry_sketch_update_count": int(association_debug.get("sketch_update_count", 0)) + (1 if pending_update_count is not None else 0),
            "association_geometry_sketch_new_key_count": int(association_debug.get("sketch_new_key_count_total", association_debug.get("association_geometry_sketch_new_key_count", 0))) + int(new_key_count),
            "association_geometry_sketch_dropped_key_count": int(association_debug.get("sketch_dropped_key_count_total", association_debug.get("association_geometry_sketch_dropped_key_count", 0))) + int(dropped_key_count),
            "association_geometry_sketch_materialize_count": int(association_debug.get("sketch_materialize_count", 0)) + 1,
            "association_geometry_sketch_update_sec": float(association_debug.get("sketch_update_sec", 0.0)) + float(update_sec),
            "association_geometry_sketch_materialize_sec": float(association_debug.get("sketch_materialize_sec", 0.0)) + materialize_sec,
            "materialized_from_sketch": True,
        }
        return True

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

    def _update_object(self, obj: ObjectMap, patch: Patch3D) -> None:
        """Merge new patch into an existing object.

        local_pcd is the v1 object-level pool geometry (Layer 5).
        It remains separate from the global TSDF instance substrate, which
        stays responsible for owner decisions and support/stability.
        """
        # Update local_pcd as the v1 object pool geometry.
        with self._timed_substage("object_update_local_pcd_update"):
            old_point_count = self._local_pcd_total_count(obj)
            patch_point_count = int(len(patch.points))
            old_centroid = np.asarray(obj.centroid, dtype=np.float32).copy()
            old_bbox_min = np.asarray(obj.bbox_min, dtype=np.float32).copy()
            old_bbox_max = np.asarray(obj.bbox_max, dtype=np.float32).copy()
            obj.update_count += 1

            local_pcd_materialized = False
            append_only_update = False
            if self.local_pcd_voxel_pool_enabled:
                self._update_local_pcd_voxel_pool(obj, patch.points)
                append_only_update = patch_point_count > 0
                if (
                    self.local_pcd_voxel_pool_materialize_on_interval
                    and obj.update_count % self.local_pcd_chunk_pool_materialize_interval == 0
                ):
                    local_pcd_materialized = self._materialize_local_pcd_voxel_pool(obj)
                    append_only_update = not local_pcd_materialized
            elif self.local_pcd_chunk_pool_enabled:
                pending_after = self._append_local_pcd_chunk(obj, patch.points)
                materialize_due = bool(
                    obj.update_count % self.local_pcd_chunk_pool_materialize_interval == 0
                    or (
                        self.local_pcd_chunk_pool_max_pending_points > 0
                        and pending_after >= self.local_pcd_chunk_pool_max_pending_points
                    )
                )
                if materialize_due:
                    local_pcd_materialized = self._materialize_local_pcd_chunks(obj)
                else:
                    append_only_update = patch_point_count > 0
            else:
                obj.local_pcd = np.concatenate([obj.local_pcd, patch.points], axis=0)

            # Periodically compact the pool geometry to keep the export source bounded.
            if (
                not self.local_pcd_voxel_pool_enabled
                and not append_only_update
                and obj.update_count % self.downsample_interval == 0
            ):
                if self.local_pcd_chunk_pool_enabled:
                    self._materialize_local_pcd_chunks(obj, compact=True)
                else:
                    obj.local_pcd = voxel_downsample(obj.local_pcd, self.downsample_voxel)
                local_pcd_materialized = True

            # Cap point count
            if (
                not self.local_pcd_voxel_pool_enabled
                and not append_only_update
                and len(obj.local_pcd) > self.max_points
            ):
                obj.local_pcd = self._deterministic_spatial_cap(obj.local_pcd, self.max_points)
                self._reset_local_pcd_pool_state(obj)
                local_pcd_materialized = True

            incremental_bounds = (
                self.local_pcd_incremental_bounds_enabled
                and not local_pcd_materialized
                and old_point_count > 0
                and patch_point_count > 0
            )

        with self._timed_substage("object_update_association_geometry_update"):
            self._update_association_geometry_from_patch(obj, patch)

        with self._timed_substage("object_update_local_pcd_update"):
            # Update spatial properties from local_pcd
            if self.local_pcd_voxel_pool_enabled:
                state = self._local_pcd_voxel_pool_state(obj)
                key_count = int(len(state.get("_keys_1d", [])))
                if key_count > 0:
                    point_sum = np.asarray(state["point_sum"], dtype=np.float64)
                    obj.centroid = (point_sum / max(key_count, 1)).astype(np.float32)
                    if old_point_count > 0 and patch_point_count > 0:
                        obj.bbox_min = np.minimum(old_bbox_min, patch.bbox_min).astype(np.float32, copy=False)
                        obj.bbox_max = np.maximum(old_bbox_max, patch.bbox_max).astype(np.float32, copy=False)
                    else:
                        obj.bbox_min, obj.bbox_max = compute_bbox(state["_vals"])
            elif incremental_bounds:
                total_count = old_point_count + patch_point_count
                obj.centroid = (
                    (old_centroid.astype(np.float64) * old_point_count + patch.centroid.astype(np.float64) * patch_point_count)
                    / max(total_count, 1)
                ).astype(np.float32)
                obj.bbox_min = np.minimum(old_bbox_min, patch.bbox_min).astype(np.float32, copy=False)
                obj.bbox_max = np.maximum(old_bbox_max, patch.bbox_max).astype(np.float32, copy=False)
            else:
                if self.local_pcd_chunk_pool_enabled and not local_pcd_materialized:
                    self._materialize_local_pcd_chunks(obj)
                    obj.centroid = obj.local_pcd.mean(axis=0)
                    obj.bbox_min, obj.bbox_max = compute_bbox(obj.local_pcd)
                elif len(obj.local_pcd) > 0:
                    obj.centroid = obj.local_pcd.mean(axis=0)
                    obj.bbox_min, obj.bbox_max = compute_bbox(obj.local_pcd)
            obj.debug["local_geometry_pool"] = {
                "incremental_bounds_enabled": bool(self.local_pcd_incremental_bounds_enabled),
                "incremental_bounds_used": bool(incremental_bounds),
                **(
                    self._local_pcd_voxel_pool_debug(
                        obj,
                        materialized_this_update=bool(local_pcd_materialized),
                        append_only_update=bool(append_only_update),
                    )
                    if self.local_pcd_voxel_pool_enabled
                    else self._local_pcd_pool_debug(
                        obj,
                        materialized_this_update=bool(local_pcd_materialized),
                        append_only_update=bool(append_only_update),
                    )
                ),
            }
            obj.last_seen_frame = patch.source_frame_id
            obj.state = ObjectState.ACTIVE

            # Record observation
            obj.observations.append(self._observation_from_patch(patch))
        with self._timed_substage("object_update_semantic_vote"):
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
        obj_size = max(self._local_pcd_total_count(obj), 1)
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
        )
        self._reset_local_pcd_pool_state(obj)
        with self._timed_substage("object_update_association_geometry_update"):
            self._refresh_association_geometry(obj)
        with self._timed_substage("object_update_semantic_vote"):
            accumulate_anchor_semantic_vote(obj, patch)
        logger.debug(f"Created new object {object_id} with {len(patch.points)} points.")
        return obj

    def _patch_crop_bbox(self, patch: Patch3D) -> np.ndarray | None:
        """Recover the source 2D crop bbox for semantic view selection."""
        bbox = patch.metadata.get("source_bbox_xyxy")
        if bbox is None:
            return None
        return np.asarray(bbox, dtype=np.float32).copy()

    def _refresh_object_debug(
        self,
        obj: ObjectMap,
        volume: TSDFInstanceVolume,
        *,
        current_frame: int | None = None,
        created: bool = False,
    ) -> None:
        """Expose the separation between the TSDF backbone and the local pool geometry."""
        interval = int(self.refresh_object_debug_interval)
        full_refresh = (
            current_frame is None
            or interval <= 1
            or (created and self.refresh_object_debug_on_create)
            or (interval > 0 and int(current_frame) % interval == 0)
        )
        support_stats = self.tsdf_module.summarize_instance_support(volume, obj.object_id)
        obj.debug["global_instance_substrate"] = {
            "role": "true_low_level_backbone",
            "owned_voxel_count": int(support_stats["owned_voxel_count"]),
            "support_mass": float(support_stats["support_mass"]),
            "mean_owner_support": float(support_stats["mean_owner_support"]),
            "max_owner_support": float(support_stats["max_owner_support"]),
            "competing_support_mass": float(support_stats["competing_support_mass"]),
            "stability_score": float(support_stats["stability_score"]),
            "last_full_refresh_frame": (
                -1 if current_frame is None or not full_refresh else int(current_frame)
            ),
            "last_incremental_refresh_frame": (
                -1 if current_frame is None or full_refresh else int(current_frame)
            ),
        }
        local_point_count = self._local_pcd_total_count(obj)
        obj.debug["local_geometry_memory"] = {
            "role": "object_pool_geometry",
            "local_point_count": int(local_point_count),
            "point_count": int(local_point_count),
            "downsample_voxel_size": float(self.downsample_voxel),
            "max_points_per_object": int(self.max_points),
        }
        association_debug = dict(obj.debug.get("association_geometry", {}))
        retained_association_fields = {
            key: association_debug[key]
            for key in (
                "association_geometry_revision",
                "revision",
                "sketch_key_count",
                "sketch_new_key_count",
                "sketch_dropped_key_count",
                "sketch_pending_update_count",
                "sketch_update_count",
                "sketch_new_key_count_total",
                "sketch_dropped_key_count_total",
                "sketch_materialize_count",
                "sketch_update_sec",
                "sketch_materialize_sec",
                "association_geometry_sketch_update_count",
                "association_geometry_sketch_new_key_count",
                "association_geometry_sketch_dropped_key_count",
                "association_geometry_sketch_materialize_count",
                "association_geometry_sketch_update_sec",
                "association_geometry_sketch_materialize_sec",
                "materialized_from_sketch",
            )
            if key in association_debug
        }
        association_debug.update(
            self._association_geometry_debug(
                point_count=len(getattr(obj, "association_pcd", [])),
                source_point_count=local_point_count,
            )
        )
        association_debug.update(retained_association_fields)
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
