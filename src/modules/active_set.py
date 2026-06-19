"""Module: Active Set (Layer 6).

Derive a local active set from the global instance map.
Only used for local reasoning, stability check, split/merge diagnostics.
Active set must NOT replace global memory.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Hashable, Set

import numpy as np

from src.core.data_structures import (
    ActiveSet,
    CameraIntrinsics,
    ObjectMap,
    SystemState,
    TSDFInstanceVolume,
    WholeEvidenceScores,
)
from src.modules.tsdf_instance_map import TSDFInstanceMapModule

logger = logging.getLogger("oviovo.modules.active_set")


class ActiveSetModule:
    """Derives the restricted local active set from global state.

    The active set is:
      - visible instances (in current camera frustum)
      - nearby instances (centroid within radius)
      - whole-prior instances (high whole_evidence_score)
      - new-object candidates (set externally after association)

    This is a local reasoning tool only, not a replacement for global memory.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.nearby_radius = config.get("nearby_radius", 3.0)
        self.whole_prior_threshold = config.get("whole_prior_threshold", 0.3)
        self.max_candidate_ids = int(config.get("max_candidate_ids", 0))
        self.max_nearby_ids = int(config.get("max_nearby_ids", 0))
        self.max_whole_prior_ids = int(config.get("max_whole_prior_ids", 0))
        self.tsdf_module = TSDFInstanceMapModule(config.get("tsdf", {}))
        self.cache_enabled = bool(config.get("cache_enabled", False))
        self.pose_quantization = float(config.get("cache_pose_quantization", 1e-6))
        self.position_quantization = float(config.get("cache_position_quantization", 0.05))
        self.signature_quantization = float(config.get("cache_signature_quantization", 0.0))
        self._visible_cache: Dict[Hashable, Set[int]] = {}
        self._nearby_cache: Dict[Hashable, Set[int]] = {}
        self._whole_prior_cache: Dict[Hashable, Set[int]] = {}
        self.last_debug: Dict[str, Any] = self._empty_debug()
        logger.info("ActiveSetModule initialized.")

    @staticmethod
    def _take_sorted(ids: Set[int], limit: int) -> Set[int]:
        if limit <= 0 or len(ids) <= limit:
            return set(ids)
        return set(sorted(int(item) for item in ids)[: int(limit)])

    def _cap_active_set(self, active_set: ActiveSet) -> ActiveSet:
        visible_ids = set(active_set.visible_ids)
        nearby_ids = self._take_sorted(set(active_set.nearby_ids), self.max_nearby_ids)
        whole_prior_ids = self._take_sorted(set(active_set.whole_prior_ids), self.max_whole_prior_ids)
        new_ids = set(active_set.new_object_candidate_ids)
        if self.max_candidate_ids <= 0:
            return ActiveSet(
                visible_ids=visible_ids,
                nearby_ids=nearby_ids,
                whole_prior_ids=whole_prior_ids,
                new_object_candidate_ids=new_ids,
            )

        selected: list[int] = []
        for group in (visible_ids, new_ids, nearby_ids, whole_prior_ids):
            for object_id in sorted(int(item) for item in group):
                if object_id not in selected:
                    selected.append(object_id)
                if len(selected) >= self.max_candidate_ids:
                    break
            if len(selected) >= self.max_candidate_ids:
                break
        selected_set = set(selected)
        return ActiveSet(
            visible_ids=visible_ids & selected_set,
            nearby_ids=nearby_ids & selected_set,
            whole_prior_ids=whole_prior_ids & selected_set,
            new_object_candidate_ids=new_ids & selected_set,
        )

    def cap_active_set(self, active_set: ActiveSet) -> ActiveSet:
        return self._cap_active_set(active_set)

    @staticmethod
    def _quantized_tuple(values: np.ndarray, quantum: float) -> tuple[int, ...]:
        arr = np.asarray(values, dtype=np.float64)
        if quantum <= 0.0:
            return tuple(float(item) for item in arr.ravel())
        return tuple(int(item) for item in np.rint(arr.ravel() / quantum))

    def _intrinsics_signature(self, intrinsics: CameraIntrinsics) -> tuple[float, float, float, float, int, int]:
        return (
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
            int(intrinsics.width),
            int(intrinsics.height),
        )

    def _tsdf_signature(self, volume: TSDFInstanceVolume) -> tuple[Any, ...]:
        return (
            getattr(volume, "owner_index_revision", None),
            getattr(volume, "support_index_valid", None),
            len(getattr(volume, "owner_support", {})),
            len(getattr(volume, "voxel_owner_id", {})),
        )

    def _object_geometry_signature(self, objects: Dict[int, ObjectMap]) -> tuple[Any, ...]:
        signature: list[Any] = [len(objects)]
        for obj_id, obj in sorted(objects.items()):
            signature.append(
                (
                    int(obj_id),
                    getattr(obj.state, "value", obj.state),
                    int(getattr(obj, "update_count", 0)),
                    int(getattr(obj, "last_seen_frame", 0)),
                    self._quantized_tuple(obj.centroid, self.signature_quantization),
                    self._quantized_tuple(obj.bbox_min, self.signature_quantization),
                    self._quantized_tuple(obj.bbox_max, self.signature_quantization),
                )
            )
        return tuple(signature)

    def _whole_prior_signature(self, objects: Dict[int, ObjectMap]) -> tuple[Any, ...]:
        signature: list[Any] = [len(objects)]
        for obj_id, obj in sorted(objects.items()):
            whole = obj.whole_evidence
            signature.append(
                (
                    int(obj_id),
                    getattr(obj.state, "value", obj.state),
                    int(getattr(obj, "update_count", 0)),
                    int(getattr(obj, "last_seen_frame", 0)),
                    round(float(getattr(whole, "whole_evidence_score", 0.0)), 8),
                    round(float(getattr(whole, "part_evidence_score", 0.0)), 8),
                    round(float(getattr(whole, "assignment_score", 0.0)), 8),
                )
            )
        return tuple(signature)

    def _visible_cache_key(
        self,
        volume: TSDFInstanceVolume,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> tuple[Any, ...]:
        return (
            self._tsdf_signature(volume),
            self._quantized_tuple(pose, self.pose_quantization),
            self._intrinsics_signature(intrinsics),
        )

    def _nearby_cache_key(
        self,
        objects: Dict[int, ObjectMap],
        cam_pos: np.ndarray,
    ) -> tuple[Any, ...]:
        return (
            self._quantized_tuple(cam_pos, self.position_quantization),
            self._object_geometry_signature(objects),
        )

    def _whole_prior_cache_key(self, objects: Dict[int, ObjectMap]) -> tuple[Any, ...]:
        return self._whole_prior_signature(objects)

    def _empty_debug(self) -> Dict[str, Any]:
        return {
            "active_set_cache_enabled": bool(getattr(self, "cache_enabled", False)),
            "visible_cache_hit": 0,
            "visible_cache_miss": 0,
            "nearby_cache_hit": 0,
            "nearby_cache_miss": 0,
            "whole_prior_cache_hit": 0,
            "whole_prior_cache_miss": 0,
        }

    def _query_visible_ids(
        self,
        state: SystemState,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
        debug: Dict[str, Any],
    ) -> Set[int]:
        if not self.cache_enabled:
            return set(self.tsdf_module.query_visible_instances(state.tsdf_volume, pose, intrinsics))
        key = self._visible_cache_key(state.tsdf_volume, pose, intrinsics)
        cached = self._visible_cache.get(key)
        if cached is not None:
            debug["visible_cache_hit"] += 1
            return set(cached)
        debug["visible_cache_miss"] += 1
        visible_ids = set(self.tsdf_module.query_visible_instances(state.tsdf_volume, pose, intrinsics))
        self._visible_cache = {key: set(visible_ids)}
        return visible_ids

    def _query_nearby_ids(
        self,
        objects: Dict[int, ObjectMap],
        cam_pos: np.ndarray,
        debug: Dict[str, Any],
    ) -> Set[int]:
        if self.cache_enabled:
            key = self._nearby_cache_key(objects, cam_pos)
            cached = self._nearby_cache.get(key)
            if cached is not None:
                debug["nearby_cache_hit"] += 1
                return set(cached)
            debug["nearby_cache_miss"] += 1
        else:
            key = None

        nearby_ids: Set[int] = set()
        can_cache = True
        position_margin = math.sqrt(3.0) * self.position_quantization if self.position_quantization > 0.0 else 0.0
        for obj_id, obj in objects.items():
            if obj.state.value == "removed":
                continue
            dist = float(np.linalg.norm(obj.centroid - cam_pos))
            if dist < self.nearby_radius:
                nearby_ids.add(obj_id)
            if abs(dist - self.nearby_radius) <= position_margin:
                can_cache = False
        if self.cache_enabled and key is not None and can_cache:
            self._nearby_cache = {key: set(nearby_ids)}
        return nearby_ids

    def _query_whole_prior_ids(
        self,
        objects: Dict[int, ObjectMap],
        debug: Dict[str, Any],
    ) -> Set[int]:
        if self.cache_enabled:
            key = self._whole_prior_cache_key(objects)
            cached = self._whole_prior_cache.get(key)
            if cached is not None:
                debug["whole_prior_cache_hit"] += 1
                return set(cached)
            debug["whole_prior_cache_miss"] += 1
        else:
            key = None

        whole_prior_ids: Set[int] = set()
        for obj_id, obj in objects.items():
            if obj.state.value == "removed":
                continue
            if obj.whole_evidence.whole_evidence_score >= self.whole_prior_threshold:
                whole_prior_ids.add(obj_id)
        if self.cache_enabled and key is not None:
            self._whole_prior_cache = {key: set(whole_prior_ids)}
        return whole_prior_ids

    def process(
        self,
        state: SystemState,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> ActiveSet:
        """Derive the active set for the current frame.

        Args:
            state: Current system state with global TSDF and objects.
            pose: (4, 4) current camera-to-world pose.
            intrinsics: Camera intrinsics.

        Returns:
            ActiveSet with visible, nearby, whole_prior, and new_object_candidate ids.
        """
        cam_pos = pose[:3, 3]
        debug = self._empty_debug()

        visible_ids = self._query_visible_ids(state, pose, intrinsics, debug)
        nearby_ids = self._query_nearby_ids(state.objects, cam_pos, debug)
        whole_prior_ids = self._query_whole_prior_ids(state.objects, debug)

        active_set = ActiveSet(
            visible_ids=visible_ids,
            nearby_ids=nearby_ids,
            whole_prior_ids=whole_prior_ids,
            new_object_candidate_ids=set(),  # filled after association
        )

        logger.debug(
            "ActiveSet: visible=%d, nearby=%d, whole_prior=%d, total=%d",
            len(visible_ids),
            len(nearby_ids),
            len(whole_prior_ids),
            len(active_set.all_candidate_ids),
        )
        active_set = self.cap_active_set(active_set)
        self.last_debug = debug
        return active_set
