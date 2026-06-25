"""Module: Active Set (Layer 6).

Derive a local active set from the global instance map.
Only used for local reasoning, stability check, split/merge diagnostics.
Active set must NOT replace global memory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Set

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

        # Visible: instances with voxels in frustum
        visible_ids = self.tsdf_module.query_visible_instances(
            state.tsdf_volume, pose, intrinsics,
        )

        # Nearby: instances whose centroid is within radius
        nearby_ids: Set[int] = set()
        for obj_id, obj in state.objects.items():
            if obj.state.value == "removed":
                continue
            dist = float(np.linalg.norm(obj.centroid - cam_pos))
            if dist < self.nearby_radius:
                nearby_ids.add(obj_id)

        # Whole-prior: instances with strong whole_evidence_score
        whole_prior_ids: Set[int] = set()
        for obj_id, obj in state.objects.items():
            if obj.state.value == "removed":
                continue
            if obj.whole_evidence.whole_evidence_score >= self.whole_prior_threshold:
                whole_prior_ids.add(obj_id)

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
        return active_set
