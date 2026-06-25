"""Pure anchor-guided SAM proposal construction logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.core.data_structures import Anchor2D, AnchorAssignment, Frame, Proposal2D


@dataclass
class MaskAnchorRelation:
    """Geometric relation between a SAM proposal mask and detector anchor box."""

    proposal: Proposal2D
    anchor: Anchor2D
    overlap_area: int
    proposal_anchor_coverage: float
    anchor_proposal_coverage: float
    bbox_iou: float
    relation: str


class AnchorGuidedSAMModule:
    """Build labeled proposals from detector anchors and local SAM boundaries."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.proposal_min_area = int(self.config.get("proposal_min_area", 25))
        self.min_proposal_anchor_coverage_for_label = float(
            np.clip(self.config.get("min_proposal_anchor_coverage_for_label", 0.70), 0.0, 1.0)
        )
        self.min_anchor_proposal_coverage_for_label = float(
            np.clip(self.config.get("min_anchor_proposal_coverage_for_label", 0.20), 0.0, 1.0)
        )
        self.contained_residual_enabled = bool(self.config.get("contained_residual_enabled", True))
        self.contained_min_proposal_coverage = float(
            np.clip(self.config.get("contained_min_proposal_coverage", 0.85), 0.0, 1.0)
        )
        self.contained_max_anchor_coverage = float(
            np.clip(self.config.get("contained_max_anchor_coverage", 0.35), 0.0, 1.0)
        )
        self.clip_to_anchor_box = bool(self.config.get("clip_to_anchor_box", True))
        self.include_unknown_residuals = bool(self.config.get("include_unknown_residuals", True))
        self.include_anchor_box_fallbacks = bool(self.config.get("include_anchor_box_fallbacks", True))
        self.max_sam_proposals_per_anchor = int(self.config.get("max_sam_proposals_per_anchor", 4))
        self.structure_classes = [
            str(item).strip()
            for item in self.config.get(
                "structure_classes",
                ["wall", "window", "blinds", "ceiling", "floor", "door"],
            )
            if str(item).strip()
        ]
        self.last_debug: dict[str, Any] = self._debug(0, 0)

    def build_proposals(
        self,
        *,
        frame: Frame,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
    ) -> tuple[list[Proposal2D], list[AnchorAssignment], dict[str, Any]]:
        """Build anchor-labeled proposals and optional unlabeled residual masks."""
        if not self.enabled:
            debug = self._debug(len(anchors), len(sam_proposals), enabled=False)
            self.last_debug = debug
            return [], [], debug

        output_proposals: list[Proposal2D] = []
        assignments: list[AnchorAssignment] = []
        matched_sam_ids: set[int] = set()
        contained_residual_relations: dict[int, MaskAnchorRelation] = {}
        candidate_counts: list[int] = []
        anchored_count = 0
        valid_sam_proposals = [
            proposal
            for proposal in sam_proposals
            if self._is_valid_sam_proposal(proposal, frame)
        ]
        dropped_proposal_count = len(sam_proposals) - len(valid_sam_proposals)
        anchor_masks = {
            int(anchor.anchor_id): self._bbox_mask(frame, anchor.bbox_xyxy)
            for anchor in anchors
        }
        relations_by_anchor: dict[int, list[MaskAnchorRelation]] = {}
        strong_owner_by_proposal_id: dict[int, MaskAnchorRelation] = {}

        for anchor in anchors:
            relations = [
                self._classify_relation(proposal, anchor, anchor_masks[int(anchor.anchor_id)])
                for proposal in valid_sam_proposals
                if int(proposal.area) >= self.proposal_min_area
            ]
            relations_by_anchor[int(anchor.anchor_id)] = relations
            for relation in relations:
                if relation.relation != "scale_compatible":
                    continue
                proposal_id = int(relation.proposal.proposal_id)
                current = strong_owner_by_proposal_id.get(proposal_id)
                if current is None or self._strong_owner_key(relation) > self._strong_owner_key(current):
                    strong_owner_by_proposal_id[proposal_id] = relation

        for anchor in anchors:
            anchor_id = int(anchor.anchor_id)
            anchor_mask = anchor_masks[anchor_id]
            relations = relations_by_anchor[anchor_id]
            selected = [
                item
                for item in relations
                if strong_owner_by_proposal_id.get(int(item.proposal.proposal_id)) is item
            ]
            selected.sort(
                key=lambda item: (
                    item.anchor_proposal_coverage,
                    item.proposal_anchor_coverage,
                    item.bbox_iou,
                ),
                reverse=True,
            )
            selected = selected[: self.max_sam_proposals_per_anchor]
            candidate_counts.append(len(selected))

            for relation in relations:
                if relation.relation != "contained_residual":
                    continue
                proposal_id = int(relation.proposal.proposal_id)
                if proposal_id in strong_owner_by_proposal_id:
                    continue
                current = contained_residual_relations.get(proposal_id)
                if current is None or (
                    relation.proposal_anchor_coverage,
                    relation.anchor_proposal_coverage,
                    relation.bbox_iou,
                ) > (
                    current.proposal_anchor_coverage,
                    current.anchor_proposal_coverage,
                    current.bbox_iou,
                ):
                    contained_residual_relations[proposal_id] = relation

            if selected:
                proposal = self._build_anchored_proposal(anchor, selected, anchor_mask)
                output_proposals.append(proposal)
                assignments.append(self._assignment_for(proposal, anchor, selected[0].bbox_iou))
                matched_sam_ids.update(int(item.proposal.proposal_id) for item in selected)
                anchored_count += 1
                continue

            if self.include_anchor_box_fallbacks:
                proposal = self._build_anchor_box_fallback(anchor, anchor_mask)
                output_proposals.append(proposal)
                assignments.append(self._assignment_for(proposal, anchor, 0.0))
                anchored_count += 1

        residual_count = 0
        semantic_blocked_residual_count = 0
        if self.include_unknown_residuals:
            for proposal_id, relation in contained_residual_relations.items():
                if proposal_id in strong_owner_by_proposal_id:
                    continue
                if proposal_id in matched_sam_ids:
                    continue
                if int(relation.proposal.area) < self.proposal_min_area:
                    continue
                residual = self._build_unknown_residual(relation)
                if residual is None:
                    continue
                output_proposals.append(residual)
                residual_count += 1
                if residual.metadata.get("semantic_commit_allowed") is False:
                    semantic_blocked_residual_count += 1

        debug = self._debug(
            len(anchors),
            len(sam_proposals),
            output_proposal_count=len(output_proposals),
            anchored_proposal_count=anchored_count,
            unknown_residual_count=residual_count,
            dropped_proposal_count=dropped_proposal_count
            + sum(1 for proposal in valid_sam_proposals if int(proposal.area) < self.proposal_min_area),
            matched_sam_proposal_count=len(matched_sam_ids),
            mean_sam_candidates_per_anchor=float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            semantic_blocked_residual_count=semantic_blocked_residual_count,
            assignment_count=len(assignments),
        )
        self.last_debug = debug
        return output_proposals, assignments, debug

    def _is_valid_sam_proposal(self, proposal: Proposal2D, frame: Frame) -> bool:
        return np.asarray(proposal.mask).shape == frame.depth.shape

    def _bbox_mask(self, frame: Frame, bbox_xyxy: np.ndarray) -> np.ndarray:
        height, width = frame.depth.shape
        x1, y1, x2, y2 = self._clipped_bbox(bbox_xyxy, width, height)
        mask = np.zeros((height, width), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask

    def _mask_bbox(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

    def _bbox_iou(self, bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = [float(item) for item in bbox_a]
        bx1, by1, bx2, by2 = [float(item) for item in bbox_b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def _classify_relation(
        self,
        proposal: Proposal2D,
        anchor: Anchor2D,
        anchor_mask: np.ndarray,
    ) -> MaskAnchorRelation:
        proposal_mask = np.asarray(proposal.mask, dtype=bool)
        overlap_area = int(np.logical_and(proposal_mask, anchor_mask).sum())
        proposal_area = int(proposal_mask.sum())
        anchor_area = int(anchor_mask.sum())
        proposal_anchor_coverage = float(overlap_area / proposal_area) if proposal_area > 0 else 0.0
        anchor_proposal_coverage = float(overlap_area / anchor_area) if anchor_area > 0 else 0.0
        bbox_iou = self._bbox_iou(np.asarray(proposal.bbox_xyxy), np.asarray(anchor.bbox_xyxy))

        if (
            self.contained_residual_enabled
            and proposal_anchor_coverage >= self.contained_min_proposal_coverage
            and anchor_proposal_coverage <= self.contained_max_anchor_coverage
        ):
            relation = "contained_residual"
        elif (
            proposal_anchor_coverage >= self.min_proposal_anchor_coverage_for_label
            and anchor_proposal_coverage >= self.min_anchor_proposal_coverage_for_label
        ):
            relation = "scale_compatible"
        else:
            relation = "unmatched"

        return MaskAnchorRelation(
            proposal=proposal,
            anchor=anchor,
            overlap_area=overlap_area,
            proposal_anchor_coverage=proposal_anchor_coverage,
            anchor_proposal_coverage=anchor_proposal_coverage,
            bbox_iou=bbox_iou,
            relation=relation,
        )

    def _build_anchored_proposal(
        self,
        anchor: Anchor2D,
        relations: list[MaskAnchorRelation],
        anchor_mask: np.ndarray,
    ) -> Proposal2D:
        mask = np.zeros_like(anchor_mask, dtype=bool)
        for relation in relations:
            mask = np.logical_or(mask, np.asarray(relation.proposal.mask, dtype=bool))
        if self.clip_to_anchor_box:
            mask = np.logical_and(mask, anchor_mask)

        coverage = self._aggregate_coverages(mask, anchor_mask, relations)
        metadata = self._anchor_metadata(anchor)
        metadata.update(
            {
                "observation_layer": "fine",
                "source_raw_proposal_ids": [int(item.proposal.proposal_id) for item in relations],
                "mask_anchor_relation": "scale_compatible",
                "proposal_anchor_coverage": coverage["proposal_anchor_coverage"],
                "anchor_proposal_coverage": coverage["anchor_proposal_coverage"],
                "mask_source": "anchor_guided_sam_selected",
            }
        )
        return Proposal2D(
            proposal_id=int(anchor.anchor_id),
            mask=mask,
            bbox_xyxy=self._mask_bbox(mask),
            area=int(mask.sum()),
            confidence=float(anchor.confidence),
            backend_name="anchor_guided_sam",
            metadata=metadata,
        )

    def _build_anchor_box_fallback(self, anchor: Anchor2D, anchor_mask: np.ndarray) -> Proposal2D:
        metadata = self._anchor_metadata(anchor)
        metadata.update(
            {
                "observation_layer": "coarse",
                "source_raw_proposal_ids": [],
                "mask_anchor_relation": "anchor_box_fallback",
                "mask_source": "detector_anchor_box",
                "force_object_candidate": True,
            }
        )
        return Proposal2D(
            proposal_id=int(anchor.anchor_id),
            mask=anchor_mask.copy(),
            bbox_xyxy=self._mask_bbox(anchor_mask),
            area=int(anchor_mask.sum()),
            confidence=float(anchor.confidence),
            backend_name="anchor_guided_sam",
            metadata=metadata,
        )

    def _build_unknown_residual(self, relation: MaskAnchorRelation) -> Proposal2D | None:
        proposal = relation.proposal
        mask = np.asarray(proposal.mask, dtype=bool).copy()
        area = int(mask.sum())
        if area < self.proposal_min_area:
            return None
        metadata = dict(proposal.metadata)
        metadata.update(
            {
                "source": "anchor_guided_sam",
                "observation_layer": "residual",
                "anchor_id": -1,
                "anchor_class_name": "",
                "anchor_confidence": 0.0,
                "anchor_label_strength": "none",
                "anchor_keepalive": False,
                "anchor_label_votes": {},
                "anchor_center_inside": False,
                "anchor_bbox_iou": 0.0,
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
                "nearby_anchor_id": int(relation.anchor.anchor_id),
                "nearby_anchor_class_name": str(relation.anchor.class_name),
                "mask_anchor_relation": "contained_residual",
                "mask_source": "sam_contained_residual",
                "source_raw_proposal_ids": [int(proposal.proposal_id)],
            }
        )
        return Proposal2D(
            proposal_id=self._residual_proposal_id(relation),
            mask=mask,
            bbox_xyxy=np.asarray(proposal.bbox_xyxy, dtype=np.float32).copy(),
            area=area,
            confidence=float(proposal.confidence),
            backend_name="anchor_guided_sam",
            metadata=metadata,
        )

    def _residual_proposal_id(self, relation: MaskAnchorRelation) -> int:
        anchor_id = self._signed_int_to_natural(int(relation.anchor.anchor_id))
        proposal_id = self._signed_int_to_natural(int(relation.proposal.proposal_id))
        pair_sum = anchor_id + proposal_id
        return 1_000_000_000_000 + (pair_sum * (pair_sum + 1)) // 2 + proposal_id

    @staticmethod
    def _signed_int_to_natural(value: int) -> int:
        return 2 * value if value >= 0 else -2 * value - 1

    def _strong_owner_key(self, relation: MaskAnchorRelation) -> tuple[float, float, int]:
        return (
            float(relation.anchor_proposal_coverage),
            float(relation.proposal_anchor_coverage),
            -int(relation.anchor.anchor_id),
        )

    def _debug(
        self,
        anchor_count: int,
        source_sam_proposal_count: int,
        *,
        enabled: bool | None = None,
        output_proposal_count: int = 0,
        anchored_proposal_count: int = 0,
        unknown_residual_count: int = 0,
        dropped_proposal_count: int = 0,
        matched_sam_proposal_count: int = 0,
        mean_sam_candidates_per_anchor: float = 0.0,
        semantic_blocked_residual_count: int = 0,
        assignment_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "enabled": self.enabled if enabled is None else enabled,
            "anchor_count": int(anchor_count),
            "source_sam_proposal_count": int(source_sam_proposal_count),
            "output_proposal_count": int(output_proposal_count),
            "anchored_proposal_count": int(anchored_proposal_count),
            "unknown_residual_count": int(unknown_residual_count),
            "dropped_proposal_count": int(dropped_proposal_count),
            "matched_sam_proposal_count": int(matched_sam_proposal_count),
            "mean_sam_candidates_per_anchor": float(mean_sam_candidates_per_anchor),
            "semantic_blocked_residual_count": int(semantic_blocked_residual_count),
            "assignment_count": int(assignment_count),
        }

    def _anchor_metadata(self, anchor: Anchor2D) -> dict[str, Any]:
        return {
            "source": "anchor_guided_sam",
            "anchor_id": int(anchor.anchor_id),
            "anchor_class_name": str(anchor.class_name),
            "anchor_confidence": float(anchor.confidence),
            "anchor_label_strength": "strong",
            "anchor_keepalive": True,
            "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
            "semantic_commit_allowed": True,
        }

    def _assignment_for(
        self,
        proposal: Proposal2D,
        anchor: Anchor2D,
        bbox_iou: float,
    ) -> AnchorAssignment:
        return AnchorAssignment(
            proposal_id=int(proposal.proposal_id),
            anchor_id=int(anchor.anchor_id),
            class_name=str(anchor.class_name),
            confidence=float(anchor.confidence),
            bbox_iou=float(bbox_iou),
            center_inside=True,
            keepalive=True,
        )

    def _aggregate_coverages(
        self,
        mask: np.ndarray,
        anchor_mask: np.ndarray,
        relations: list[MaskAnchorRelation],
    ) -> dict[str, float]:
        overlap_area = int(np.logical_and(mask, anchor_mask).sum())
        proposal_area = int(mask.sum())
        anchor_area = int(anchor_mask.sum())
        if proposal_area > 0 and anchor_area > 0:
            return {
                "proposal_anchor_coverage": float(overlap_area / proposal_area),
                "anchor_proposal_coverage": float(overlap_area / anchor_area),
            }
        if not relations:
            return {"proposal_anchor_coverage": 0.0, "anchor_proposal_coverage": 0.0}
        return {
            "proposal_anchor_coverage": float(relations[0].proposal_anchor_coverage),
            "anchor_proposal_coverage": float(relations[0].anchor_proposal_coverage),
        }

    def _clipped_bbox(self, bbox_xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        raw_x1, raw_y1, raw_x2, raw_y2 = [float(item) for item in bbox_xyxy]
        x1 = math.floor(raw_x1)
        y1 = math.floor(raw_y1)
        x2 = math.ceil(raw_x2)
        y2 = math.ceil(raw_y2)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2
