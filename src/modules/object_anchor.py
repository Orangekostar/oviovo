"""Detector-anchor frontend for YOLO and SAM intersection proposals."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.data_structures import Anchor2D, AnchorAssignment, Proposal2D
from src.models.object_anchor_backend import ObjectAnchorBackend, PlaceholderObjectAnchorBackend

logger = logging.getLogger("oviovo.modules.object_anchor")


class ObjectAnchorModule:
    """Detector-anchor assignment for object-centric local concrete mapping."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.use_boxes_as_primary_proposals = bool(
            config.get("use_boxes_as_primary_proposals", self.enabled)
        )
        self.use_sam_intersection_proposals = bool(
            config.get("use_sam_intersection_proposals", self.enabled)
        )
        self.anchor_primary_mode = bool(config.get("anchor_primary_mode", False))
        self.prefer_largest_covering_box = bool(
            config.get("prefer_largest_covering_box", self.use_sam_intersection_proposals)
        )
        self.assignment_policy = str(
            config.get(
                "assignment_policy",
                "largest_covering" if self.prefer_largest_covering_box else "best_single",
            )
        )
        self.semantic_vote_policy = self.assignment_policy in {"semantic_vote", "vote_only", "sam_mask_semantic_vote"}
        self.preserve_nested_classes = bool(
            config.get("preserve_nested_classes", self.assignment_policy == "preserve_nested_classes")
        )
        self.clip_sam_masks_to_anchor_box = bool(
            config.get("clip_sam_masks_to_anchor_box", self.preserve_nested_classes)
        )
        self.preserve_cross_class_nested = bool(
            config.get("preserve_cross_class_nested", self.preserve_nested_classes)
        )
        self.subtract_cross_class_child_mask_from_parent = bool(
            config.get("subtract_cross_class_child_mask_from_parent", self.preserve_nested_classes)
        )
        self.min_anchor_mask_coverage = float(np.clip(config.get("min_anchor_mask_coverage", 0.20), 0.0, 1.0))
        self.min_protected_anchor_confidence = float(config.get("min_protected_anchor_confidence", 0.25))
        self.requested_backend_name = str(config.get("backend", "placeholder"))
        self.active_backend_name = self.requested_backend_name if self.enabled else "placeholder"
        self.backend_init_error: str | None = None
        self.iou_assign_threshold = float(config.get("iou_assign_threshold", 0.2))
        self.keepalive_iou_threshold = float(config.get("keepalive_iou_threshold", 0.1))
        self.center_assign_enabled = bool(config.get("center_assign_enabled", True))
        self.keepalive_center_inside = bool(config.get("keepalive_center_inside", True))
        self.unanchored_fallback_enabled = bool(config.get("unanchored_fallback_enabled", True))
        self.proposal_min_area = int(config.get("proposal_min_area", 25))
        self.covering_min_proposal_coverage = float(
            np.clip(config.get("covering_min_proposal_coverage", 0.85), 0.0, 1.0)
        )
        self.contained_subproposal_gate_enabled = bool(
            config.get("contained_subproposal_gate_enabled", self.semantic_vote_policy)
        )
        self.contained_max_anchor_coverage = float(
            np.clip(config.get("contained_max_anchor_coverage", self.min_anchor_mask_coverage), 0.0, 1.0)
        )
        self.contained_min_proposal_coverage = float(
            np.clip(config.get("contained_min_proposal_coverage", self.covering_min_proposal_coverage), 0.0, 1.0)
        )
        self.scale_compatible_min_anchor_coverage = float(
            np.clip(config.get("scale_compatible_min_anchor_coverage", self.min_anchor_mask_coverage), 0.0, 1.0)
        )
        self.scale_compatible_min_bbox_iou = float(
            np.clip(config.get("scale_compatible_min_bbox_iou", 0.10), 0.0, 1.0)
        )
        self.weak_structure_overlap_enabled = bool(config.get("weak_structure_overlap_enabled", False))
        self.weak_structure_classes = {
            str(item).strip()
            for item in config.get(
                "weak_structure_classes",
                ["wall", "floor", "ceiling", "blinds", "window"],
            )
            if str(item).strip()
        }
        self.weak_structure_min_proposal_coverage = float(
            np.clip(config.get("weak_structure_min_proposal_coverage", 0.05), 0.0, 1.0)
        )
        self.weak_structure_min_anchor_coverage = float(
            np.clip(config.get("weak_structure_min_anchor_coverage", 0.20), 0.0, 1.0)
        )
        self.supplemental_enabled = bool(config.get("supplemental_enabled", False))
        self.supplemental_iou_threshold = float(config.get("supplemental_iou_threshold", 0.5))
        self.supplemental_overlap_threshold = float(config.get("supplemental_overlap_threshold", 0.6))
        self.supplemental_backend_name = str(
            dict(config.get("supplemental", {})).get("backend", "placeholder")
        )
        self.anchor_classes = [
            str(item).strip()
            for item in config.get(
                "classes",
                [
                    "chair",
                    "table",
                    "cabinet",
                    "sofa",
                    "bed",
                    "plant",
                    "monitor",
                    "lamp",
                    "shelf",
                    "stool",
                    "nightstand",
                ],
            )
            if str(item).strip()
        ]
        self.backend = self._build_backend(config)
        self.supplemental_backend = self._build_supplemental_backend(config)
        self.last_anchors: List[Anchor2D] = []
        self.last_primary_anchors: List[Anchor2D] = []
        self.last_supplemental_anchors: List[Anchor2D] = []
        self.last_generation_timings: dict[str, float] = {}
        self.collect_generation_timings = True
        self._clock = time.perf_counter
        self.last_assignments: List[AnchorAssignment] = []
        logger.info(
            "ObjectAnchorModule initialized with requested backend=%s active backend=%s",
            self.requested_backend_name,
            self.active_backend_name,
        )

    def _create_backend(self, config: Dict[str, Any]) -> ObjectAnchorBackend:
        backend_name = config.get("backend", "placeholder")
        if backend_name == "placeholder":
            return PlaceholderObjectAnchorBackend()
        if backend_name == "yoloworld":
            from src.models.yoloworld_anchor_backend import YOLOWorldAnchorBackend

            return YOLOWorldAnchorBackend()
        if backend_name == "yoloe_seg_pf":
            from src.models.yoloe_seg_anchor_backend import YOLOESegAnchorBackend

            return YOLOESegAnchorBackend()
        logger.warning("Unknown anchor backend '%s', falling back to placeholder.", backend_name)
        return PlaceholderObjectAnchorBackend()

    def _build_backend(self, config: Dict[str, Any]) -> ObjectAnchorBackend:
        if not self.enabled:
            self.active_backend_name = "placeholder"
            return PlaceholderObjectAnchorBackend()

        backend = self._create_backend(config)
        backend_config = dict(config)
        backend_config["classes"] = list(self.anchor_classes)
        try:
            backend.initialize(backend_config)
            self.backend_init_error = None
            self.active_backend_name = self.requested_backend_name
            return backend
        except Exception as exc:
            logger.warning(
                "Anchor backend '%s' unavailable, disabling anchor mode: %s",
                self.requested_backend_name,
                exc,
            )
            self.backend_init_error = str(exc)
            self.enabled = False
            self.active_backend_name = "placeholder"
            return PlaceholderObjectAnchorBackend()

    def _build_supplemental_backend(self, config: Dict[str, Any]) -> ObjectAnchorBackend | None:
        if not self.supplemental_enabled:
            return None
        supplemental_cfg = dict(config.get("supplemental", {}))
        backend_name = str(supplemental_cfg.get("backend", "placeholder"))
        if backend_name == "placeholder":
            return None
        supplemental_cfg["backend"] = backend_name
        if "classes" not in supplemental_cfg:
            supplemental_cfg["classes"] = list(self.anchor_classes)
        backend = self._create_backend(supplemental_cfg)
        backend.initialize(supplemental_cfg)
        return backend

    def process(self, rgb: np.ndarray, proposals: List[Proposal2D]) -> Tuple[List[Anchor2D], List[AnchorAssignment]]:
        """Detect anchors and annotate proposals with the best anchor assignment."""
        self.last_anchors = self._generate_merged_anchors(rgb)
        assignments = [self._assign_single(proposal, self.last_anchors) for proposal in proposals]
        self.last_assignments = assignments

        self._apply_assignments(proposals, assignments)

        return self.last_anchors, assignments

    def generate_proposals(
        self,
        rgb: np.ndarray,
        proposals: List[Proposal2D],
    ) -> Tuple[List[Anchor2D], List[Proposal2D], List[AnchorAssignment]]:
        """Detect anchors and aggregate overlapping SAM proposals into YOLO∩SAM masks."""
        self.last_anchors = self._generate_merged_anchors(rgb)
        if self.semantic_vote_policy:
            voted_proposals, voted_assignments = self._generate_semantic_vote_proposals(proposals)
            self.last_assignments = voted_assignments
            return self.last_anchors, voted_proposals, voted_assignments

        if self.preserve_nested_classes:
            grouped_proposals, grouped_assignments = self._generate_preserved_class_proposals(proposals)
            self.last_assignments = grouped_assignments
            self._apply_assignments(grouped_proposals, grouped_assignments)
            return self.last_anchors, grouped_proposals, grouped_assignments

        if self.prefer_largest_covering_box:
            source_assignments = [self._assign_largest_covering_anchor(proposal, self.last_anchors) for proposal in proposals]
        else:
            source_assignments = [self._assign_single(proposal, self.last_anchors) for proposal in proposals]
        self._apply_assignments(proposals, source_assignments)

        assignments_by_anchor: dict[int, list[tuple[Proposal2D, AnchorAssignment]]] = {}
        for proposal, assignment in zip(proposals, source_assignments):
            if int(assignment.anchor_id) < 0:
                continue
            assignments_by_anchor.setdefault(int(assignment.anchor_id), []).append((proposal, assignment))

        grouped_proposals: list[Proposal2D] = []
        grouped_assignments: list[AnchorAssignment] = []
        for anchor in self.last_anchors:
            proposal = self._anchor_intersection_proposal(
                anchor,
                assignments_by_anchor.get(int(anchor.anchor_id), []),
            )
            if proposal is None:
                continue
            grouped_proposals.append(proposal)
            grouped_assignments.append(
                AnchorAssignment(
                    proposal_id=int(proposal.proposal_id),
                    anchor_id=int(anchor.anchor_id),
                    class_name=str(anchor.class_name),
                    confidence=float(anchor.confidence),
                    bbox_iou=1.0,
                    center_inside=True,
                    keepalive=True,
                )
            )

        self.last_assignments = grouped_assignments
        self._apply_assignments(grouped_proposals, grouped_assignments)
        return self.last_anchors, grouped_proposals, grouped_assignments

    def generate_anchor_box_proposals(
        self,
        rgb: np.ndarray,
    ) -> Tuple[List[Anchor2D], List[Proposal2D], List[AnchorAssignment]]:
        """Generate DualMap-style detector-box primary proposals.

        This fast path intentionally avoids assigning labels to unrelated SAM
        masks. Each detector anchor becomes one rectangular proposal, and SAM2
        high-recall discovery can be layered on separately by the pipeline.
        """
        self.last_anchors = self._generate_merged_anchors(rgb)
        proposals: list[Proposal2D] = []
        assignments: list[AnchorAssignment] = []
        height, width = rgb.shape[:2]
        for anchor in self.last_anchors:
            mask = self._bbox_mask((height, width), np.asarray(anchor.bbox_xyxy, dtype=np.float32))
            area = int(mask.sum())
            if area < self.proposal_min_area:
                continue
            proposal_id = len(proposals)
            proposal = Proposal2D(
                proposal_id=proposal_id,
                mask=mask,
                bbox_xyxy=self._mask_bbox(mask),
                area=area,
                confidence=float(anchor.confidence),
                backend_name="anchor_box_primary",
                metadata={
                    "source": "anchor_box_primary",
                    "requested_backend": self.requested_backend_name,
                    "actual_backend": self.active_backend_name,
                    "anchor_primary_proposal": True,
                    "anchor_id": int(anchor.anchor_id),
                    "anchor_class_name": str(anchor.class_name),
                    "anchor_confidence": float(anchor.confidence),
                    "anchor_bbox_iou": 1.0,
                    "anchor_center_inside": True,
                    "anchor_keepalive": True,
                    "anchor_label_strength": "strong",
                    "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                    "anchor_source_bbox_xyxy": np.asarray(anchor.bbox_xyxy, dtype=np.float32).copy(),
                    "anchor_source_class_name": str(anchor.class_name),
                    "anchor_source_confidence": float(anchor.confidence),
                    "mask_source": "anchor_box",
                    "anchor_assignment_strategy": "anchor_box_primary",
                    "source_raw_proposal_ids": [],
                },
            )
            assignment = AnchorAssignment(
                proposal_id=proposal_id,
                anchor_id=int(anchor.anchor_id),
                class_name=str(anchor.class_name),
                confidence=float(anchor.confidence),
                bbox_iou=1.0,
                center_inside=True,
                keepalive=True,
            )
            proposals.append(proposal)
            assignments.append(assignment)
        self.last_assignments = assignments
        return self.last_anchors, proposals, assignments

    def _generate_semantic_vote_proposals(
        self,
        proposals: List[Proposal2D],
    ) -> tuple[list[Proposal2D], list[AnchorAssignment]]:
        voted_proposals: list[Proposal2D] = []
        voted_assignments: list[AnchorAssignment] = []

        for proposal in proposals:
            assignments, blocked_candidates = self._assign_semantic_vote_candidates(proposal, self.last_anchors)
            if not assignments:
                voted_proposals.append(
                    self._clone_proposal_with_anchor_vote(
                        proposal,
                        None,
                        [],
                        blocked_candidates,
                    )
                )
                voted_assignments.append(AnchorAssignment(proposal_id=int(proposal.proposal_id)))
            else:
                best_assignment = max(assignments, key=self._assignment_score)
                voted_proposals.append(
                    self._clone_proposal_with_anchor_vote(
                        proposal,
                        best_assignment,
                        assignments,
                        blocked_candidates,
                    )
                )
                voted_assignments.append(best_assignment)

            weak_proposals, weak_assignments = self._generate_weak_structure_overlap_proposals(
                proposal,
                self.last_anchors,
                assignments,
            )
            voted_proposals.extend(weak_proposals)
            voted_assignments.extend(weak_assignments)

        return voted_proposals, voted_assignments

    def _semantic_vote_overlap_metrics(
        self,
        proposal: Proposal2D,
        anchor: Anchor2D,
    ) -> dict[str, Any]:
        proposal_bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float32)
        proposal_mask = np.asarray(proposal.mask, dtype=bool)
        proposal_area = max(int(proposal_mask.sum()), 1)
        center = np.asarray(
            [(proposal_bbox[0] + proposal_bbox[2]) * 0.5, (proposal_bbox[1] + proposal_bbox[3]) * 0.5],
            dtype=np.float32,
        )
        anchor_box = np.asarray(anchor.bbox_xyxy, dtype=np.float32)
        box_mask = self._bbox_mask(proposal_mask.shape, anchor_box)
        clipped_mask = proposal_mask & box_mask
        overlap_area = int(clipped_mask.sum())
        anchor_area_px = max(int(box_mask.sum()), 1)
        anchor_coverage = overlap_area / anchor_area_px
        proposal_coverage = overlap_area / proposal_area
        center_inside = self._center_inside(center, anchor_box) if self.center_assign_enabled else False
        bbox_iou = self._bbox_iou(proposal_bbox, anchor_box)
        return {
            "anchor_box": anchor_box,
            "box_mask": box_mask,
            "clipped_mask": clipped_mask,
            "overlap_area": overlap_area,
            "anchor_coverage": anchor_coverage,
            "proposal_coverage": proposal_coverage,
            "center_inside": center_inside,
            "bbox_iou": bbox_iou,
        }

    def _is_scale_compatible_semantic_vote(self, metrics: dict[str, Any]) -> bool:
        anchor_coverage = float(metrics["anchor_coverage"])
        bbox_iou = float(metrics["bbox_iou"])
        return bool(
            anchor_coverage >= self.scale_compatible_min_anchor_coverage
            or bbox_iou >= self.scale_compatible_min_bbox_iou
        )

    def _is_contained_subproposal_without_child_anchor(self, metrics: dict[str, Any]) -> bool:
        if not self.contained_subproposal_gate_enabled:
            return False
        proposal_coverage = float(metrics["proposal_coverage"])
        anchor_coverage = float(metrics["anchor_coverage"])
        if proposal_coverage < self.contained_min_proposal_coverage:
            return False
        if anchor_coverage > self.contained_max_anchor_coverage:
            return False
        return not self._is_scale_compatible_semantic_vote(metrics)

    def _blocked_semantic_vote_candidate(
        self,
        anchor: Anchor2D,
        metrics: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "anchor_id": int(anchor.anchor_id),
            "class_name": str(anchor.class_name),
            "confidence": float(anchor.confidence),
            "reason": str(reason),
            "bbox_iou": float(metrics["bbox_iou"]),
            "center_inside": bool(metrics["center_inside"]),
            "proposal_coverage": float(metrics["proposal_coverage"]),
            "anchor_coverage": float(metrics["anchor_coverage"]),
            "overlap_area": int(metrics["overlap_area"]),
        }

    def _assign_semantic_vote_candidates(
        self,
        proposal: Proposal2D,
        anchors: List[Anchor2D],
    ) -> tuple[list[AnchorAssignment], list[dict[str, Any]]]:
        assignments: list[AnchorAssignment] = []
        blocked_candidates: list[dict[str, Any]] = []
        if not anchors:
            return assignments, blocked_candidates

        for anchor in anchors:
            metrics = self._semantic_vote_overlap_metrics(proposal, anchor)
            if int(metrics["overlap_area"]) < self.proposal_min_area:
                continue

            if float(metrics["proposal_coverage"]) < self.covering_min_proposal_coverage:
                continue

            if self._is_contained_subproposal_without_child_anchor(metrics):
                blocked_candidates.append(
                    self._blocked_semantic_vote_candidate(
                        anchor,
                        metrics,
                        reason="contained_subproposal_without_child_anchor",
                    )
                )
                continue

            bbox_iou = float(metrics["bbox_iou"])
            center_inside = bool(metrics["center_inside"])
            anchor_coverage = float(metrics["anchor_coverage"])
            keepalive = bool(
                bbox_iou >= self.keepalive_iou_threshold
                or (self.keepalive_center_inside and center_inside)
                or anchor_coverage >= self.min_anchor_mask_coverage
            )
            assignments.append(
                AnchorAssignment(
                    proposal_id=int(proposal.proposal_id),
                    anchor_id=int(anchor.anchor_id),
                    class_name=str(anchor.class_name),
                    confidence=float(anchor.confidence),
                    bbox_iou=float(bbox_iou),
                    center_inside=bool(center_inside),
                    keepalive=keepalive,
                )
            )

        return assignments, blocked_candidates

    def _is_weak_structure_class(self, class_name: str) -> bool:
        return str(class_name).strip() in self.weak_structure_classes

    def _weak_structure_overlap_proposal(
        self,
        proposal: Proposal2D,
        anchor: Anchor2D,
        clipped_mask: np.ndarray,
        metrics: dict[str, Any],
        weak_index: int,
    ) -> tuple[Proposal2D, AnchorAssignment] | None:
        clipped_mask = np.asarray(clipped_mask, dtype=bool)
        area = int(clipped_mask.sum())
        if area < self.proposal_min_area:
            return None

        source_proposal_id = int(proposal.proposal_id)
        anchor_id = int(anchor.anchor_id)
        proposal_id = -((source_proposal_id + 1) * 1_000_000 + int(weak_index) + 1)
        bbox_iou = float(metrics["bbox_iou"])
        center_inside = bool(metrics["center_inside"])
        anchor_coverage = float(metrics["anchor_coverage"])
        proposal_coverage = float(metrics["proposal_coverage"])
        keepalive = bool(
            bbox_iou >= self.keepalive_iou_threshold
            or (self.keepalive_center_inside and center_inside)
            or anchor_coverage >= self.min_anchor_mask_coverage
        )
        assignment = AnchorAssignment(
            proposal_id=proposal_id,
            anchor_id=anchor_id,
            class_name=str(anchor.class_name),
            confidence=float(anchor.confidence),
            bbox_iou=float(bbox_iou),
            center_inside=bool(center_inside),
            keepalive=keepalive,
        )
        class_name = str(anchor.class_name)
        metadata = dict(proposal.metadata)
        metadata.update(
            {
                "source": "sam2_anchor_vote",
                "geometry_source": "sam2_anchor_overlap",
                "mask_source": "weak_structure_anchor_box_clip",
                "source_raw_proposal_id": source_proposal_id,
                "source_raw_proposal_ids": [source_proposal_id],
                "anchor_id": anchor_id,
                "anchor_class_name": class_name,
                "anchor_confidence": float(anchor.confidence),
                "anchor_bbox_iou": float(bbox_iou),
                "anchor_center_inside": bool(center_inside),
                "anchor_keepalive": bool(keepalive),
                "anchor_vote_score": float(self._assignment_score(assignment)),
                "anchor_candidate_classes": [class_name],
                "anchor_candidate_ids": [anchor_id],
                "anchor_label_votes": {class_name: float(anchor.confidence)},
                "anchor_label_strength": "weak_overlap",
                "anchor_proposal_coverage": float(proposal_coverage),
                "anchor_anchor_coverage": float(anchor_coverage),
                "anchor_blocked_candidates": [],
            }
        )
        weak_proposal = Proposal2D(
            proposal_id=proposal_id,
            mask=clipped_mask.copy(),
            bbox_xyxy=self._mask_bbox(clipped_mask),
            area=area,
            confidence=max(float(proposal.confidence), float(anchor.confidence)),
            backend_name="sam2_anchor_vote",
            metadata=metadata,
        )
        return weak_proposal, assignment

    def _generate_weak_structure_overlap_proposals(
        self,
        proposal: Proposal2D,
        anchors: List[Anchor2D],
        existing_assignments: list[AnchorAssignment],
    ) -> tuple[list[Proposal2D], list[AnchorAssignment]]:
        if not self.weak_structure_overlap_enabled:
            return [], []

        assigned_anchor_ids = {int(assignment.anchor_id) for assignment in existing_assignments}
        weak_proposals: list[Proposal2D] = []
        weak_assignments: list[AnchorAssignment] = []

        for anchor in anchors:
            if int(anchor.anchor_id) in assigned_anchor_ids:
                continue
            if not self._is_weak_structure_class(str(anchor.class_name)):
                continue
            metrics = self._semantic_vote_overlap_metrics(proposal, anchor)
            if int(metrics["overlap_area"]) < self.proposal_min_area:
                continue
            if float(metrics["proposal_coverage"]) >= self.covering_min_proposal_coverage:
                continue
            if float(metrics["proposal_coverage"]) < self.weak_structure_min_proposal_coverage:
                continue
            if float(metrics["anchor_coverage"]) < self.weak_structure_min_anchor_coverage:
                continue

            weak = self._weak_structure_overlap_proposal(
                proposal,
                anchor,
                np.asarray(metrics["clipped_mask"], dtype=bool),
                metrics,
                weak_index=len(weak_proposals),
            )
            if weak is None:
                continue
            weak_proposal, weak_assignment = weak
            weak_proposals.append(weak_proposal)
            weak_assignments.append(weak_assignment)

        return weak_proposals, weak_assignments

    def _clone_proposal_with_anchor_vote(
        self,
        proposal: Proposal2D,
        assignment: AnchorAssignment | None,
        candidates: list[AnchorAssignment],
        blocked_candidates: list[dict[str, Any]] | None = None,
    ) -> Proposal2D:
        source_backend = str(proposal.backend_name)
        geometry_source = source_backend if source_backend else str(proposal.metadata.get("source", "unknown"))
        metadata = dict(proposal.metadata)
        metadata.update(
            {
                "source": "sam2_anchor_vote",
                "geometry_source": geometry_source,
                "source_raw_proposal_id": int(proposal.proposal_id),
                "source_raw_proposal_ids": [int(proposal.proposal_id)],
                "anchor_candidate_classes": [str(candidate.class_name) for candidate in candidates],
                "anchor_candidate_ids": [int(candidate.anchor_id) for candidate in candidates],
                "anchor_label_votes": self._anchor_label_votes(candidates),
                "anchor_blocked_candidates": list(blocked_candidates or []),
            }
        )

        if assignment is None:
            blocked_items = list(blocked_candidates or [])
            contained_residual = any(
                str(item.get("reason", "")) == "contained_subproposal_without_child_anchor"
                for item in blocked_items
                if isinstance(item, dict)
            )
            metadata.update(
                {
                    "anchor_id": -1,
                    "anchor_class_name": "",
                    "anchor_label_strength": "none",
                    "anchor_confidence": 0.0,
                    "anchor_bbox_iou": 0.0,
                    "anchor_center_inside": False,
                    "anchor_keepalive": False,
                    "anchor_vote_score": 0.0,
                }
            )
            if contained_residual:
                metadata.update(
                    {
                        "semantic_commit_allowed": False,
                        "residual_semantic_policy": "unknown",
                        "mask_anchor_relation": "contained_residual",
                        "semantic_commit_blocked_reason": "contained_residual_identity_guard",
                    }
                )
        else:
            metadata.update(
                {
                    "anchor_id": int(assignment.anchor_id),
                    "anchor_class_name": str(assignment.class_name),
                    "anchor_label_strength": "strong",
                    "anchor_confidence": float(assignment.confidence),
                    "anchor_bbox_iou": float(assignment.bbox_iou),
                    "anchor_center_inside": bool(assignment.center_inside),
                    "anchor_keepalive": bool(assignment.keepalive),
                    "anchor_vote_score": float(self._assignment_score(assignment)),
                }
            )

        return Proposal2D(
            proposal_id=int(proposal.proposal_id),
            mask=np.asarray(proposal.mask, dtype=bool).copy(),
            bbox_xyxy=np.asarray(proposal.bbox_xyxy, dtype=np.float32).copy(),
            area=int(proposal.area),
            confidence=float(proposal.confidence),
            backend_name="sam2_anchor_vote",
            metadata=metadata,
        )

    @staticmethod
    def _anchor_label_votes(candidates: list[AnchorAssignment]) -> dict[str, float]:
        votes: dict[str, float] = {}
        for candidate in candidates:
            class_name = str(candidate.class_name)
            if not class_name:
                continue
            votes[class_name] = max(votes.get(class_name, 0.0), float(candidate.confidence))
        return votes

    def _generate_preserved_class_proposals(
        self,
        proposals: List[Proposal2D],
    ) -> tuple[list[Proposal2D], list[AnchorAssignment]]:
        """Generate one protected proposal per assignable anchor.

        Unlike largest-covering assignment, a SAM mask may support multiple
        anchors. Same-class SAM fragments are still unioned per anchor, while
        nested different-class anchors are preserved as independent proposals.
        """
        anchor_records: dict[int, dict[str, Any]] = {}
        for anchor in self.last_anchors:
            anchor_records[int(anchor.anchor_id)] = {
                "anchor": anchor,
                "mask": None,
                "source_raw_proposal_ids": [],
                "source_confidences": [],
                "coverage_values": [],
                "assignment": None,
            }

        for proposal in proposals:
            assignments = self._assign_preserve_nested_classes(proposal, self.last_anchors)
            if not assignments:
                proposal.metadata = dict(proposal.metadata)
                proposal.metadata["anchor_candidate_classes"] = []
                proposal.metadata["anchor_candidate_ids"] = []
                continue
            proposal.metadata = dict(proposal.metadata)
            proposal.metadata["anchor_candidate_classes"] = [assignment.class_name for assignment, _ in assignments]
            proposal.metadata["anchor_candidate_ids"] = [int(assignment.anchor_id) for assignment, _ in assignments]
            best_assignment = max(assignments, key=lambda item: self._assignment_score(item[0]))[0]
            proposal.metadata["anchor_id"] = int(best_assignment.anchor_id)
            proposal.metadata["anchor_class_name"] = str(best_assignment.class_name)
            proposal.metadata["anchor_confidence"] = float(best_assignment.confidence)
            proposal.metadata["anchor_bbox_iou"] = float(best_assignment.bbox_iou)
            proposal.metadata["anchor_center_inside"] = bool(best_assignment.center_inside)
            proposal.metadata["anchor_keepalive"] = bool(best_assignment.keepalive)

            for assignment, clipped_mask in assignments:
                anchor_id = int(assignment.anchor_id)
                record = anchor_records.get(anchor_id)
                if record is None:
                    continue
                if record["mask"] is None:
                    record["mask"] = np.zeros_like(clipped_mask, dtype=bool)
                record["mask"] |= clipped_mask
                record["source_raw_proposal_ids"].append(int(proposal.proposal_id))
                record["source_confidences"].append(float(proposal.confidence))
                record["coverage_values"].append(float(clipped_mask.sum() / max(int(proposal.mask.sum()), 1)))
                record["assignment"] = assignment

        self._subtract_protected_child_masks(anchor_records)

        grouped_proposals: list[Proposal2D] = []
        grouped_assignments: list[AnchorAssignment] = []
        for anchor in self.last_anchors:
            record = anchor_records.get(int(anchor.anchor_id), {})
            proposal = self._proposal_from_anchor_record(record)
            if proposal is None:
                continue
            grouped_proposals.append(proposal)
            grouped_assignments.append(
                AnchorAssignment(
                    proposal_id=int(proposal.proposal_id),
                    anchor_id=int(anchor.anchor_id),
                    class_name=str(anchor.class_name),
                    confidence=float(anchor.confidence),
                    bbox_iou=1.0,
                    center_inside=True,
                    keepalive=True,
                )
            )

        return grouped_proposals, grouped_assignments

    def _assign_preserve_nested_classes(
        self,
        proposal: Proposal2D,
        anchors: List[Anchor2D],
    ) -> list[tuple[AnchorAssignment, np.ndarray]]:
        assignments: list[tuple[AnchorAssignment, np.ndarray]] = []
        if not anchors:
            return assignments

        proposal_bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float32)
        proposal_mask = np.asarray(proposal.mask, dtype=bool)
        proposal_area = max(int(proposal_mask.sum()), 1)
        center = np.asarray(
            [(proposal_bbox[0] + proposal_bbox[2]) * 0.5, (proposal_bbox[1] + proposal_bbox[3]) * 0.5],
            dtype=np.float32,
        )

        for anchor in anchors:
            anchor_box = np.asarray(anchor.bbox_xyxy, dtype=np.float32)
            box_mask = self._bbox_mask(proposal_mask.shape, anchor_box)
            clipped_mask = proposal_mask & box_mask if self.clip_sam_masks_to_anchor_box else proposal_mask.copy()
            clipped_area = int(clipped_mask.sum())
            if clipped_area < self.proposal_min_area:
                continue

            anchor_area_px = max(int(box_mask.sum()), 1)
            anchor_coverage = clipped_area / anchor_area_px
            proposal_coverage = clipped_area / proposal_area
            center_inside = self._center_inside(center, anchor_box) if self.center_assign_enabled else False
            bbox_iou = self._bbox_iou(proposal_bbox, anchor_box)
            assignable = (
                anchor_coverage >= self.min_anchor_mask_coverage
                or proposal_coverage >= self.covering_min_proposal_coverage
                or center_inside
            )
            if not assignable:
                continue

            keepalive = bool(
                bbox_iou >= self.keepalive_iou_threshold
                or (self.keepalive_center_inside and center_inside)
                or anchor_coverage >= self.min_anchor_mask_coverage
            )
            assignment = AnchorAssignment(
                proposal_id=int(proposal.proposal_id),
                anchor_id=int(anchor.anchor_id),
                class_name=str(anchor.class_name),
                confidence=float(anchor.confidence),
                bbox_iou=float(bbox_iou),
                center_inside=bool(center_inside),
                keepalive=keepalive,
            )
            assignments.append((assignment, clipped_mask))

        return assignments

    def _subtract_protected_child_masks(self, anchor_records: dict[int, dict[str, Any]]) -> None:
        if not self.subtract_cross_class_child_mask_from_parent:
            return
        records = [record for record in anchor_records.values() if record.get("mask") is not None]
        for parent_record in records:
            parent_anchor: Anchor2D = parent_record["anchor"]
            parent_mask = np.asarray(parent_record["mask"], dtype=bool)
            protected_child_ids: list[int] = []
            protected_child_classes: list[str] = []
            for child_record in records:
                child_anchor: Anchor2D = child_record["anchor"]
                if int(child_anchor.anchor_id) == int(parent_anchor.anchor_id):
                    continue
                if str(child_anchor.class_name) == str(parent_anchor.class_name):
                    continue
                if float(child_anchor.confidence) < self.min_protected_anchor_confidence:
                    continue
                if not self._anchor_is_nested_child(child_anchor, parent_anchor):
                    continue
                child_mask = np.asarray(child_record["mask"], dtype=bool)
                if int(child_mask.sum()) >= int(parent_mask.sum()):
                    continue
                parent_mask &= ~child_mask
                protected_child_ids.append(int(child_anchor.anchor_id))
                protected_child_classes.append(str(child_anchor.class_name))
            parent_record["mask"] = parent_mask
            parent_record["protected_child_anchor_ids"] = protected_child_ids
            parent_record["protected_child_classes"] = protected_child_classes

    def _proposal_from_anchor_record(self, record: dict[str, Any]) -> Proposal2D | None:
        anchor = record.get("anchor")
        mask = record.get("mask")
        if anchor is None or mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        if area < self.proposal_min_area:
            return None

        source_confidences = [float(value) for value in record.get("source_confidences", [])]
        confidence = max([float(anchor.confidence), *source_confidences]) if source_confidences else float(anchor.confidence)
        nested_parent_ids = [
            int(other.anchor_id)
            for other in self.last_anchors
            if int(other.anchor_id) != int(anchor.anchor_id)
            and str(other.class_name) != str(anchor.class_name)
            and self._anchor_is_nested_child(anchor, other)
        ]
        protected_small_anchor = bool(
            nested_parent_ids
            and float(anchor.confidence) >= self.min_protected_anchor_confidence
        )
        return Proposal2D(
            proposal_id=int(anchor.anchor_id),
            mask=mask,
            bbox_xyxy=self._mask_bbox(mask),
            area=area,
            confidence=confidence,
            backend_name="anchor_sam_union",
            metadata={
                "source": "anchor_sam_union",
                "requested_backend": self.requested_backend_name,
                "actual_backend": self.active_backend_name,
                "anchor_primary_proposal": True,
                "anchor_source_bbox_xyxy": np.asarray(anchor.bbox_xyxy, dtype=np.float32).copy(),
                "anchor_source_class_name": str(anchor.class_name),
                "anchor_source_confidence": float(anchor.confidence),
                "mask_source": "preserve_nested_classes_anchor_box_clip"
                if self.clip_sam_masks_to_anchor_box
                else "preserve_nested_classes_sam_union",
                "anchor_assignment_strategy": "preserve_nested_classes",
                "source_raw_proposal_ids": [int(value) for value in record.get("source_raw_proposal_ids", [])],
                "source_raw_proposal_confidences": source_confidences,
                "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                "anchor_hit_count": int(len(record.get("source_raw_proposal_ids", []))),
                "protected_child_anchor_ids": [int(value) for value in record.get("protected_child_anchor_ids", [])],
                "protected_child_classes": [str(value) for value in record.get("protected_child_classes", [])],
                "anchor_is_nested_child": bool(nested_parent_ids),
                "nested_parent_anchor_ids": nested_parent_ids,
                "protected_small_anchor": protected_small_anchor,
                "force_object_candidate": protected_small_anchor,
            },
        )

    def _timed_anchor_generation(self, name: str, callback):
        if not self.collect_generation_timings:
            return callback()
        start = self._clock()
        try:
            return callback()
        finally:
            elapsed = max(0.0, float(self._clock() - start))
            self.last_generation_timings[name] = self.last_generation_timings.get(name, 0.0) + elapsed

    def _primary_timing_key(self) -> str:
        backend_name = str(getattr(self, "active_backend_name", "") or "").strip()
        if not backend_name:
            backend_name = str(self.requested_backend_name or "").strip()
        backend_name = backend_name or "anchor"
        if backend_name == "yoloworld":
            return "yoloworld_primary"
        return f"{backend_name}_primary"

    def _supplemental_timing_key(self) -> str:
        if not self.supplemental_enabled or self.supplemental_backend is None:
            return "anchor_supplemental"
        backend_name = str(self.supplemental_backend_name or "").strip() or "anchor"
        if backend_name == "yoloe_seg_pf":
            return "yoloe_supplemental"
        return f"{backend_name}_supplemental"

    def _generate_merged_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        self.last_generation_timings = {}
        primary = self._timed_anchor_generation(
            self._primary_timing_key(),
            lambda: self.backend.generate_anchors(rgb) if self.enabled else [],
        )
        self.last_primary_anchors = list(primary)
        if not self.supplemental_enabled or self.supplemental_backend is None:
            self.last_supplemental_anchors = []
            self.last_generation_timings.setdefault(self._supplemental_timing_key(), 0.0)
            self.last_generation_timings.setdefault("anchor_merge", 0.0)
            return list(primary)
        supplemental = self._timed_anchor_generation(
            self._supplemental_timing_key(),
            lambda: self.supplemental_backend.generate_anchors(rgb),
        )
        self.last_supplemental_anchors = list(supplemental)
        return self._timed_anchor_generation(
            "anchor_merge",
            lambda: self._merge_anchor_sets(primary, supplemental),
        )

    def _merge_anchor_sets(
        self,
        primary_anchors: List[Anchor2D],
        supplemental_anchors: List[Anchor2D],
    ) -> List[Anchor2D]:
        merged = list(primary_anchors)
        for anchor in supplemental_anchors:
            anchor_box = np.asarray(anchor.bbox_xyxy, dtype=np.float32)
            anchor_area = max(self._bbox_area(anchor_box), 1e-6)
            should_keep = True
            for primary in primary_anchors:
                primary_box = np.asarray(primary.bbox_xyxy, dtype=np.float32)
                primary_area = max(self._bbox_area(primary_box), 1e-6)
                inter = self._bbox_intersection_area(anchor_box, primary_box)
                if inter <= 0.0:
                    continue
                iou = self._bbox_iou(anchor_box, primary_box)
                overlap_anchor = inter / anchor_area
                overlap_primary = inter / primary_area
                if (
                    iou > self.supplemental_iou_threshold
                    or overlap_anchor > self.supplemental_overlap_threshold
                    or overlap_primary > self.supplemental_overlap_threshold
                ):
                    should_keep = False
                    break
            if should_keep:
                merged.append(anchor)
        return merged

    def _apply_assignments(
        self,
        proposals: List[Proposal2D],
        assignments: List[AnchorAssignment],
    ) -> None:
        assignment_lookup = {int(assignment.proposal_id): assignment for assignment in assignments}
        for proposal in proposals:
            assignment = assignment_lookup.get(int(proposal.proposal_id), AnchorAssignment(proposal_id=int(proposal.proposal_id)))
            proposal.metadata = dict(proposal.metadata)
            proposal.metadata["anchor_id"] = int(assignment.anchor_id)
            proposal.metadata["anchor_class_name"] = str(assignment.class_name)
            proposal.metadata["anchor_confidence"] = float(assignment.confidence)
            proposal.metadata["anchor_bbox_iou"] = float(assignment.bbox_iou)
            proposal.metadata["anchor_center_inside"] = bool(assignment.center_inside)
            proposal.metadata["anchor_keepalive"] = bool(assignment.keepalive)

    def _anchor_intersection_proposal(
        self,
        anchor: Anchor2D,
        assigned_proposals: list[tuple[Proposal2D, AnchorAssignment]],
    ) -> Proposal2D | None:
        if not assigned_proposals:
            return None

        union_mask = np.zeros_like(np.asarray(assigned_proposals[0][0].mask, dtype=bool), dtype=bool)
        source_raw_proposal_ids: list[int] = []
        source_confidences: list[float] = []
        for proposal, _assignment in assigned_proposals:
            mask = np.asarray(proposal.mask, dtype=bool)
            if not mask.any():
                continue
            union_mask |= mask
            source_raw_proposal_ids.append(int(proposal.proposal_id))
            source_confidences.append(float(proposal.confidence))

        area = int(union_mask.sum())
        if area < self.proposal_min_area:
            return None

        bbox_xyxy = self._mask_bbox(union_mask)
        confidence = max([float(anchor.confidence), *source_confidences]) if source_confidences else float(anchor.confidence)
        return Proposal2D(
            proposal_id=int(anchor.anchor_id),
            mask=union_mask,
            bbox_xyxy=bbox_xyxy,
            area=area,
            confidence=confidence,
            backend_name="anchor_sam_union",
            metadata={
                "source": "anchor_sam_union",
                "requested_backend": self.requested_backend_name,
                "actual_backend": self.active_backend_name,
                "anchor_primary_proposal": True,
                "anchor_source_bbox_xyxy": np.asarray(anchor.bbox_xyxy, dtype=np.float32).copy(),
                "anchor_source_class_name": str(anchor.class_name),
                "anchor_source_confidence": float(anchor.confidence),
                "mask_source": "largest_covering_yolo_box_over_sam_union",
                "anchor_assignment_strategy": "largest_covering_box",
                "source_raw_proposal_ids": source_raw_proposal_ids,
            },
        )

    @staticmethod
    def _bbox_mask(shape: tuple[int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
        height, width = shape
        x1 = int(np.clip(np.floor(float(bbox_xyxy[0])), 0, width))
        y1 = int(np.clip(np.floor(float(bbox_xyxy[1])), 0, height))
        x2 = int(np.clip(np.ceil(float(bbox_xyxy[2])), 0, width))
        y2 = int(np.clip(np.ceil(float(bbox_xyxy[3])), 0, height))
        box_mask = np.zeros((height, width), dtype=bool)
        if x2 <= x1 or y2 <= y1:
            return box_mask
        box_mask[y1:y2, x1:x2] = True
        return box_mask

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            dtype=np.float32,
        )

    def _assign_single(self, proposal: Proposal2D, anchors: List[Anchor2D]) -> AnchorAssignment:
        if not anchors:
            return AnchorAssignment(proposal_id=int(proposal.proposal_id))

        best: AnchorAssignment | None = None
        proposal_bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float32)
        center = np.asarray(
            [(proposal_bbox[0] + proposal_bbox[2]) * 0.5, (proposal_bbox[1] + proposal_bbox[3]) * 0.5],
            dtype=np.float32,
        )

        for anchor in anchors:
            iou = self._bbox_iou(proposal_bbox, anchor.bbox_xyxy)
            center_inside = self._center_inside(center, anchor.bbox_xyxy) if self.center_assign_enabled else False
            assignable = iou >= self.iou_assign_threshold or center_inside
            if not assignable:
                continue
            keepalive = iou >= self.keepalive_iou_threshold or (
                self.keepalive_center_inside and center_inside
            )
            candidate = AnchorAssignment(
                proposal_id=int(proposal.proposal_id),
                anchor_id=int(anchor.anchor_id),
                class_name=str(anchor.class_name),
                confidence=float(anchor.confidence),
                bbox_iou=float(iou),
                center_inside=bool(center_inside),
                keepalive=bool(keepalive),
            )
            if best is None or self._assignment_score(candidate) > self._assignment_score(best):
                best = candidate

        return best or AnchorAssignment(proposal_id=int(proposal.proposal_id))

    def _assign_largest_covering_anchor(self, proposal: Proposal2D, anchors: List[Anchor2D]) -> AnchorAssignment:
        if not anchors:
            return AnchorAssignment(proposal_id=int(proposal.proposal_id))

        proposal_bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float32)
        center = np.asarray(
            [(proposal_bbox[0] + proposal_bbox[2]) * 0.5, (proposal_bbox[1] + proposal_bbox[3]) * 0.5],
            dtype=np.float32,
        )
        proposal_area = max(self._bbox_area(proposal_bbox), 1e-6)
        best: tuple[float, float, AnchorAssignment] | None = None

        for anchor in anchors:
            inter = self._bbox_intersection_area(proposal_bbox, anchor.bbox_xyxy)
            if inter <= 0.0:
                continue
            coverage_ratio = inter / proposal_area
            center_inside = self._center_inside(center, anchor.bbox_xyxy) if self.center_assign_enabled else False
            if coverage_ratio < self.covering_min_proposal_coverage and not center_inside:
                continue

            iou = self._bbox_iou(proposal_bbox, anchor.bbox_xyxy)
            anchor_area = self._bbox_area(np.asarray(anchor.bbox_xyxy, dtype=np.float32))
            keepalive = iou >= self.keepalive_iou_threshold or (
                self.keepalive_center_inside and center_inside
            )
            assignment = AnchorAssignment(
                proposal_id=int(proposal.proposal_id),
                anchor_id=int(anchor.anchor_id),
                class_name=str(anchor.class_name),
                confidence=float(anchor.confidence),
                bbox_iou=float(iou),
                center_inside=bool(center_inside),
                keepalive=bool(keepalive),
            )
            rank = (float(anchor_area), float(coverage_ratio), assignment)
            if best is None or rank[0] > best[0] or (rank[0] == best[0] and rank[1] > best[1]):
                best = rank

        if best is not None:
            return best[2]
        return self._assign_single(proposal, anchors)

    @classmethod
    def _anchor_is_nested_child(cls, child: Anchor2D, parent: Anchor2D) -> bool:
        child_box = np.asarray(child.bbox_xyxy, dtype=np.float32)
        parent_box = np.asarray(parent.bbox_xyxy, dtype=np.float32)
        child_area = max(cls._bbox_area(child_box), 1e-6)
        parent_area = max(cls._bbox_area(parent_box), 1e-6)
        if child_area >= parent_area:
            return False
        inter = cls._bbox_intersection_area(child_box, parent_box)
        return bool(inter / child_area >= 0.80)

    @staticmethod
    def _assignment_score(assignment: AnchorAssignment) -> float:
        return float(assignment.bbox_iou + 0.2 * assignment.center_inside + 0.1 * assignment.confidence)

    @staticmethod
    def _center_inside(center_xy: np.ndarray, bbox_xyxy: np.ndarray) -> bool:
        return bool(
            bbox_xyxy[0] <= center_xy[0] <= bbox_xyxy[2]
            and bbox_xyxy[1] <= center_xy[1] <= bbox_xyxy[3]
        )

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = ObjectAnchorModule._bbox_intersection_area(a, b)
        area_a = ObjectAnchorModule._bbox_area(a)
        area_b = ObjectAnchorModule._bbox_area(b)
        union = area_a + area_b - inter
        if union <= 1e-8:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _bbox_intersection_area(a: np.ndarray, b: np.ndarray) -> float:
        inter_x1 = max(float(a[0]), float(b[0]))
        inter_y1 = max(float(a[1]), float(b[1]))
        inter_x2 = min(float(a[2]), float(b[2]))
        inter_y2 = min(float(a[3]), float(b[3]))
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        return float(inter_w * inter_h)

    @staticmethod
    def _bbox_area(box: np.ndarray) -> float:
        return float(max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])))
