"""Proposal mask cleanup before 3D lifting."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.data_structures import RefinedProposal2D


class ProposalCleanupModule:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.min_area_after_cleanup = int(config.get("min_area_after_cleanup", 50))

    def process(self, proposals: list[RefinedProposal2D]) -> list[RefinedProposal2D]:
        if not self.enabled or not proposals:
            return list(proposals)
        ordered = sorted(proposals, key=self._priority, reverse=True)
        claimed = np.zeros_like(ordered[0].mask, dtype=bool)
        cleaned: list[RefinedProposal2D] = []
        for proposal in ordered:
            mask = np.asarray(proposal.mask, dtype=bool) & ~claimed
            area = int(mask.sum())
            if area < self.min_area_after_cleanup:
                continue
            copied = RefinedProposal2D(
                proposal_id=proposal.proposal_id,
                mask=mask,
                bbox_xyxy=self._bbox_from_mask(mask, proposal.bbox_xyxy),
                area=area,
                confidence=proposal.confidence,
                geometric_features=proposal.geometric_features,
                soft_scores=proposal.soft_scores,
                backend_name=proposal.backend_name,
                metadata=dict(proposal.metadata),
            )
            copied.metadata["mask_cleanup_removed_pixels"] = int(proposal.area - area)
            cleaned.append(copied)
            claimed |= mask
        return sorted(cleaned, key=lambda item: int(item.proposal_id))

    @staticmethod
    def _priority(proposal: RefinedProposal2D) -> tuple[float, float, float]:
        anchor_confidence = float(proposal.metadata.get("anchor_confidence", proposal.confidence))
        objectness = float(proposal.soft_scores.objectness_score)
        small_object_bonus = 1.0 / max(float(proposal.area), 1.0)
        return (anchor_confidence, objectness, small_object_bonus)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.asarray(fallback, dtype=np.float32).copy()
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
