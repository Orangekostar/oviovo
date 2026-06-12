"""Dense structural overlay from existing anchors and precomputed SAM proposals."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.data_structures import Anchor2D, Frame, Proposal2D, StructuralOverlayMap, StructuralOverlayVoxel


class StructuralOverlayModule:
    """Accumulates voxel votes for structure labels without creating object instances."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.classes = {str(label).strip().lower() for label in self.config.get("classes", []) if str(label).strip()}
        self.voxel_size = float(self.config.get("voxel_size", 0.05))
        self.min_overlap_area = int(self.config.get("min_overlap_area", 25))
        self.min_proposal_anchor_coverage = float(self.config.get("min_proposal_anchor_coverage", 0.20))
        self.min_anchor_proposal_coverage = float(self.config.get("min_anchor_proposal_coverage", 0.03))
        self.clip_to_anchor_box = bool(self.config.get("clip_to_anchor_box", True))
        self.pixel_sample_stride = max(1, int(self.config.get("pixel_sample_stride", 1)))
        self.max_pixels_per_pair = max(0, int(self.config.get("max_pixels_per_pair", 0)))
        self.last_summary: dict[str, Any] = {"enabled": self.enabled}

    def process(
        self,
        frame: Frame,
        anchors: list[Anchor2D],
        proposals: list[Proposal2D],
        overlay_map: StructuralOverlayMap,
    ) -> dict[str, Any]:
        summary = {
            "enabled": bool(self.enabled),
            "structure_anchor_count": 0,
            "sam_proposal_count": int(len(proposals)),
            "accepted_pair_count": 0,
            "mismatched_mask_count": 0,
            "voted_pixel_count": 0,
            "new_voxel_count": 0,
            "updated_voxel_count": 0,
            "skip_reason": "",
        }
        if not self.enabled:
            summary["skip_reason"] = "disabled"
            self.last_summary = summary
            return summary

        overlay_map.voxel_size = float(self.voxel_size)
        structure_anchors = [
            anchor for anchor in anchors if str(anchor.class_name).strip().lower() in self.classes
        ]
        summary["structure_anchor_count"] = int(len(structure_anchors))
        if not structure_anchors:
            summary["skip_reason"] = "no_structure_anchors"
            self.last_summary = summary
            return summary
        if not proposals:
            summary["skip_reason"] = "no_sam_proposals"
            self.last_summary = summary
            return summary

        valid_depth = np.isfinite(frame.depth) & (frame.depth > 0)
        valid_proposals: list[Proposal2D] = []
        for proposal in proposals:
            if proposal.mask.shape != frame.depth.shape:
                summary["mismatched_mask_count"] += 1
                continue
            valid_proposals.append(proposal)
        initial_voxel_count = len(overlay_map.voxels)
        touched_voxels: set[tuple[int, int, int]] = set()
        for anchor in structure_anchors:
            anchor_mask = self._anchor_box_mask(anchor.bbox_xyxy, frame.depth.shape)
            anchor_area = int(np.count_nonzero(anchor_mask))
            if anchor_area <= 0:
                continue
            label = str(anchor.class_name).strip().lower()
            for proposal in valid_proposals:
                proposal_mask = np.asarray(proposal.mask, dtype=bool)
                overlap = proposal_mask & anchor_mask
                overlap_area = int(np.count_nonzero(overlap))
                if overlap_area < self.min_overlap_area:
                    continue
                proposal_area = max(int(np.count_nonzero(proposal_mask)), 1)
                proposal_anchor_coverage = overlap_area / float(proposal_area)
                anchor_proposal_coverage = overlap_area / float(anchor_area)
                if proposal_anchor_coverage < self.min_proposal_anchor_coverage:
                    continue
                if anchor_proposal_coverage < self.min_anchor_proposal_coverage:
                    continue

                mask = overlap if self.clip_to_anchor_box else proposal_mask
                voted, pair_voxels = self._vote_mask(frame, mask & valid_depth, overlay_map, label, anchor, proposal)
                if voted <= 0:
                    continue
                touched_voxels.update(pair_voxels)
                summary["accepted_pair_count"] += 1
                summary["voted_pixel_count"] += int(voted)

        summary["updated_voxel_count"] = int(len(touched_voxels))
        summary["new_voxel_count"] = int(max(0, len(overlay_map.voxels) - initial_voxel_count))
        overlay_map.update_count += 1
        if summary["accepted_pair_count"] == 0 and not summary["skip_reason"]:
            summary["skip_reason"] = "no_matching_structure_pairs"
        self.last_summary = summary
        return summary

    def _vote_mask(
        self,
        frame: Frame,
        mask: np.ndarray,
        overlay_map: StructuralOverlayMap,
        label: str,
        anchor: Anchor2D,
        proposal: Proposal2D,
    ) -> tuple[int, set[tuple[int, int, int]]]:
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return 0, set()
        if self.pixel_sample_stride > 1:
            rows = rows[:: self.pixel_sample_stride]
            cols = cols[:: self.pixel_sample_stride]
        if self.max_pixels_per_pair > 0 and len(rows) > self.max_pixels_per_pair:
            indices = np.linspace(0, len(rows) - 1, num=self.max_pixels_per_pair, dtype=np.int64)
            rows = rows[indices]
            cols = cols[indices]

        z = frame.depth[rows, cols].astype(np.float64)
        x = (cols.astype(np.float64) - frame.intrinsics.cx) * z / frame.intrinsics.fx
        y = (rows.astype(np.float64) - frame.intrinsics.cy) * z / frame.intrinsics.fy
        points_cam = np.stack([x, y, z], axis=1)
        points = ((frame.pose[:3, :3] @ points_cam.T).T + frame.pose[:3, 3]).astype(np.float32)
        voxel_indices = np.floor(points / float(overlay_map.voxel_size)).astype(np.int32)
        if len(voxel_indices) == 0:
            return 0, set()
        unique_voxels = np.unique(voxel_indices, axis=0)
        weight = max(float(anchor.confidence), 0.0) * max(float(proposal.confidence), 0.0)
        if weight <= 0.0:
            weight = 1.0
        touched_voxels: set[tuple[int, int, int]] = set()
        for voxel in unique_voxels:
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            entry = overlay_map.voxels.setdefault(key, StructuralOverlayVoxel())
            entry.add_vote(label, weight, frame.frame_id)
            touched_voxels.add(key)
        return int(len(rows)), touched_voxels

    @staticmethod
    def _anchor_box_mask(bbox_xyxy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        left = max(0, min(width, int(np.floor(x1))))
        top = max(0, min(height, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        mask = np.zeros((height, width), dtype=bool)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
        return mask
