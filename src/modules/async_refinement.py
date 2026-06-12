"""Delayed SAM refinement for coarse-to-fine mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.data_structures import Anchor2D, Frame, Proposal2D


@dataclass(frozen=True)
class RefinementResult:
    fine_proposals: list[Proposal2D]
    debug: dict[str, Any]


class AsyncRefinementModule:
    """Build fine SAM/anchor proposals that can replace coarse anchor-box observations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        raw_delay_frames = self.config.get("delay_frames", 0)
        if raw_delay_frames != 0:
            raise ValueError("async_refinement.delay_frames must be 0; delayed queueing is not implemented")
        self.delay_frames = 0
        self.proposal_min_area = int(self.config.get("proposal_min_area", 25))
        self.min_proposal_anchor_coverage = float(
            np.clip(self.config.get("min_proposal_anchor_coverage", 0.20), 0.0, 1.0)
        )
        self.min_anchor_proposal_coverage = float(
            np.clip(self.config.get("min_anchor_proposal_coverage", 0.05), 0.0, 1.0)
        )
        self.clip_to_anchor_box = bool(self.config.get("clip_to_anchor_box", False))
        self.last_debug: dict[str, Any] = {}

    def build_fine_proposals(
        self,
        *,
        frame: Frame,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
    ) -> tuple[list[Proposal2D], dict[str, Any]]:
        fine: list[Proposal2D] = []
        matched_sam_ids: set[int] = set()
        matched_anchor_count = 0
        image_shape = frame.rgb.shape[:2]

        if not self.enabled:
            debug = self._debug(
                anchors=anchors,
                sam_proposals=sam_proposals,
                matched_anchor_count=0,
                fine_proposal_count=0,
                matched_sam_ids=matched_sam_ids,
            )
            self.last_debug = debug
            return fine, debug

        for anchor in anchors:
            anchor_mask = self._bbox_mask(image_shape, np.asarray(anchor.bbox_xyxy, dtype=np.float32))
            anchor_area = int(anchor_mask.sum())
            if anchor_area <= 0:
                continue

            selected: list[Proposal2D] = []
            for proposal in sam_proposals:
                proposal_mask = np.asarray(proposal.mask, dtype=bool)
                if proposal_mask.shape != image_shape:
                    continue

                proposal_area = int(proposal_mask.sum())
                if proposal_area <= 0:
                    continue

                overlap_area = int((proposal_mask & anchor_mask).sum())
                if overlap_area < self.proposal_min_area:
                    continue

                proposal_anchor_coverage = overlap_area / max(proposal_area, 1)
                anchor_proposal_coverage = overlap_area / max(anchor_area, 1)
                if (
                    proposal_anchor_coverage < self.min_proposal_anchor_coverage
                    and anchor_proposal_coverage < self.min_anchor_proposal_coverage
                ):
                    continue

                selected.append(proposal)
                matched_sam_ids.add(int(proposal.proposal_id))

            if not selected:
                continue
            selected = sorted(selected, key=lambda proposal: int(proposal.proposal_id))

            union_mask = np.zeros(image_shape, dtype=bool)
            for proposal in selected:
                proposal_mask = np.asarray(proposal.mask, dtype=bool)
                union_mask |= (proposal_mask & anchor_mask) if self.clip_to_anchor_box else proposal_mask

            area = int(union_mask.sum())
            if area < self.proposal_min_area:
                continue

            matched_anchor_count += 1
            source_ids = [int(proposal.proposal_id) for proposal in selected]
            source_confidences = [float(proposal.confidence) for proposal in selected]
            fine.append(
                Proposal2D(
                    proposal_id=int(anchor.anchor_id),
                    mask=union_mask,
                    bbox_xyxy=self._mask_bbox(union_mask),
                    area=area,
                    confidence=max(float(anchor.confidence), max(source_confidences)),
                    backend_name="async_sam_refinement",
                    metadata={
                        "source": "async_sam_refinement",
                        "observation_layer": "fine",
                        "refinement_key": self.refinement_key(frame, anchor),
                        "anchor_id": int(anchor.anchor_id),
                        "anchor_class_name": str(anchor.class_name),
                        "anchor_confidence": float(anchor.confidence),
                        "anchor_label_strength": "strong",
                        "anchor_keepalive": True,
                        "anchor_label_votes": {str(anchor.class_name): float(anchor.confidence)},
                        "source_raw_proposal_ids": source_ids,
                        "source_raw_proposal_confidences": source_confidences,
                        "mask_source": "delayed_sam_union",
                    },
                )
            )

        debug = self._debug(
            anchors=anchors,
            sam_proposals=sam_proposals,
            matched_anchor_count=matched_anchor_count,
            fine_proposal_count=len(fine),
            matched_sam_ids=matched_sam_ids,
        )
        self.last_debug = debug
        return fine, debug

    def process_ready(
        self,
        *,
        frame: Frame,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
    ) -> RefinementResult:
        """Return fine proposals that are ready to apply for this frame."""
        if not self.enabled:
            debug = {
                "enabled": False,
                "anchor_count": int(len(anchors)),
                "sam_proposal_count": int(len(sam_proposals)),
                "matched_anchor_count": 0,
                "fine_proposal_count": 0,
                "matched_sam_proposal_count": 0,
                "unmatched_sam_proposal_count": int(len(sam_proposals)),
                "replaced_observation_count": 0,
                "inserted_observation_count": 0,
                "updated_object_count": 0,
            }
            self.last_debug = debug
            return RefinementResult(fine_proposals=[], debug=debug)

        fine_proposals, debug = self.build_fine_proposals(
            frame=frame,
            anchors=anchors,
            sam_proposals=sam_proposals,
        )
        return RefinementResult(fine_proposals=fine_proposals, debug=debug)

    @staticmethod
    def refinement_key(frame: Frame, anchor: Anchor2D) -> str:
        frame_id = int(frame.source_frame_id if frame.source_frame_id is not None else frame.frame_id)
        return f"{frame_id}:{int(anchor.anchor_id)}"

    @staticmethod
    def _bbox_mask(shape: tuple[int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
        height, width = shape
        x1 = int(np.clip(np.floor(float(bbox_xyxy[0])), 0, width))
        y1 = int(np.clip(np.floor(float(bbox_xyxy[1])), 0, height))
        x2 = int(np.clip(np.ceil(float(bbox_xyxy[2])), 0, width))
        y2 = int(np.clip(np.ceil(float(bbox_xyxy[3])), 0, height))
        mask = np.zeros((height, width), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs) == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

    def _debug(
        self,
        *,
        anchors: list[Anchor2D],
        sam_proposals: list[Proposal2D],
        matched_anchor_count: int,
        fine_proposal_count: int,
        matched_sam_ids: set[int],
    ) -> dict[str, Any]:
        sam_ids = {int(proposal.proposal_id) for proposal in sam_proposals}
        return {
            "enabled": bool(self.enabled),
            "anchor_count": int(len(anchors)),
            "sam_proposal_count": int(len(sam_proposals)),
            "matched_anchor_count": int(matched_anchor_count),
            "fine_proposal_count": int(fine_proposal_count),
            "matched_sam_proposal_count": int(len(matched_sam_ids)),
            "unmatched_sam_proposal_count": int(len(sam_ids - matched_sam_ids)),
        }
