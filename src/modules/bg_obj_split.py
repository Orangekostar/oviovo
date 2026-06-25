"""Module 5: Background / Object Split.

Responsibility: Decide whether each 3D patch belongs to the background,
an object, or is ambiguous, using soft scores from RefinedProposal2D.

Uses soft evidence (objectness_score, backgroundness_score, attachedness_score)
rather than hard heuristics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.data_structures import BackgroundMap, ObjectMap, Patch3D

logger = logging.getLogger("oviovo.modules.bg_obj_split")


class BgObjSplitModule:
    """Classifies 3D patches into background, object, or ambiguous using soft scores."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.object_threshold = config.get("object_threshold", 0.45)
        self.background_threshold = config.get("background_threshold", 0.50)
        self.margin = config.get("class_margin", 0.08)
        # Legacy fallbacks
        self.size_threshold = config.get("size_threshold", 5000)
        self.height_threshold = config.get("height_threshold", 0.1)
        logger.info("BgObjSplitModule initialized.")

    def process(
        self,
        patches: List[Patch3D],
        background: BackgroundMap,
        objects: Dict[int, ObjectMap],
    ) -> Tuple[List[Patch3D], List[Patch3D], List[Patch3D]]:
        """Split patches into background, object, and ambiguous groups.

        Uses soft scores (objectness_score, backgroundness_score) from the
        refined proposal pipeline. Falls back to geometric heuristics
        when soft scores are absent.

        Args:
            patches: 3D patches from the current frame.
            background: Current background map.
            objects: Current object set.

        Returns:
            (background_patches, object_patches, ambiguous_patches)
        """
        bg_patches: List[Patch3D] = []
        obj_patches: List[Patch3D] = []
        amb_patches: List[Patch3D] = []

        for patch in patches:
            category = self._classify_patch(patch, background, objects)
            patch.metadata["split_origin"] = category
            if category == "background":
                bg_patches.append(patch)
            elif category == "object":
                obj_patches.append(patch)
            else:
                amb_patches.append(patch)

        logger.debug(
            f"Split: {len(bg_patches)} bg, {len(obj_patches)} obj, {len(amb_patches)} ambiguous"
        )
        return bg_patches, obj_patches, amb_patches

    def _classify_patch(
        self,
        patch: Patch3D,
        background: BackgroundMap,
        objects: Dict[int, ObjectMap],
    ) -> str:
        """Classify a single patch using soft scores.

        Primary path: use objectness_score and backgroundness_score.
        Fallback: geometric heuristics when soft scores are zero.
        """
        obj_score = patch.soft_scores.objectness_score
        bg_score = patch.soft_scores.backgroundness_score
        attach_score = patch.soft_scores.attachedness_score

        if bool(patch.metadata.get("force_object_candidate", False)):
            return "object"

        # If soft scores are populated, use them
        if obj_score > 0 or bg_score > 0:
            if bg_score >= self.background_threshold and bg_score >= obj_score + self.margin:
                return "background"
            if obj_score >= self.object_threshold and obj_score >= bg_score + self.margin:
                return "object"
            # High attachedness with ambiguous scores -> ambiguous
            if attach_score > 0.5:
                return "ambiguous"
            # Slight lean
            if obj_score > bg_score:
                return "object"
            if bg_score > obj_score:
                return "background"
            return "ambiguous"

        # Fallback: legacy geometric heuristics
        n_points = len(patch.points)
        if n_points > self.size_threshold:
            return "background"
        if patch.centroid[2] < self.height_threshold:
            return "background"
        return "object"
