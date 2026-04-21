"""Module 3: Depth Refinement (Layer 2).

Responsibility: Refine 2D proposal masks using depth discontinuity
and geometric boundary cues. Output RefinedProposal2D with explicit
geometric features and soft scores. Obvious depth-disconnected regions
are split into separate refined proposals so raw SAM2 masks are never
treated as final objects.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import (
    Proposal2D,
    ProposalGeometricFeatures,
    ProposalSoftScores,
    RefinedProposal2D,
)

logger = logging.getLogger("oviovo.modules.depth_refinement")


class DepthRefinementModule:
    """Refines 2D masks using depth edge information.

    Outputs RefinedProposal2D with:
      - 4 geometric features: depth_valid_ratio, depth_variance,
        border_touch_ratio, planar_fit_residual
      - 3 soft scores: objectness_score, backgroundness_score,
        attachedness_score (weighted functions of geometric features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.depth_edge_threshold = config.get("depth_edge_threshold", 0.05)
        self.min_area = config.get("min_mask_area_after_refine", 50)
        self.max_plane_fit_points = config.get("max_plane_fit_points", 256)
        # Configurable weights for soft scores
        self.w_obj_compact = config.get("w_obj_compact", 0.25)
        self.w_obj_small_extent = config.get("w_obj_small_extent", 0.20)
        self.w_obj_depth_rich = config.get("w_obj_depth_rich", 0.20)
        self.w_obj_low_border = config.get("w_obj_low_border", 0.20)
        self.w_obj_depth_valid = config.get("w_obj_depth_valid", 0.15)
        self.w_bg_large_extent = config.get("w_bg_large_extent", 0.25)
        self.w_bg_border_touch = config.get("w_bg_border_touch", 0.25)
        self.w_bg_planar = config.get("w_bg_planar", 0.25)
        self.w_bg_flat_depth = config.get("w_bg_flat_depth", 0.25)
        self.planar_residual_ref = config.get("planar_residual_ref", 0.02)
        self.depth_variance_ref = config.get("depth_variance_ref", 0.012)
        self.proposal_parallel_enabled = bool(config.get("proposal_parallel_enabled", False))
        self.proposal_parallel_workers = int(config.get("proposal_parallel_workers", 0))
        self.proposal_parallel_min_tasks = int(config.get("proposal_parallel_min_tasks", 8))
        logger.info("DepthRefinementModule initialized.")

    def process(self, depth: np.ndarray, proposals: List[Proposal2D]) -> List[RefinedProposal2D]:
        """Refine proposal masks using depth boundaries and compute features/scores.

        Args:
            depth: (H, W) float32 depth map.
            proposals: List of Proposal2D to refine.

        Returns:
            List of RefinedProposal2D with geometric features and soft scores.
        """
        depth_edges = self._compute_depth_edges(depth)
        if (
            self.proposal_parallel_enabled
            and len(proposals) >= self.proposal_parallel_min_tasks
        ):
            worker_count = self._resolve_worker_count(
                self.proposal_parallel_workers,
                len(proposals),
            )
            if worker_count > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    refined_by_proposal = list(
                        executor.map(
                            lambda proposal: self._refine_single_proposal(depth, depth_edges, proposal),
                            proposals,
                        )
                    )
            else:
                refined_by_proposal = [
                    self._refine_single_proposal(depth, depth_edges, proposal)
                    for proposal in proposals
                ]
        else:
            refined_by_proposal = [
                self._refine_single_proposal(depth, depth_edges, proposal)
                for proposal in proposals
            ]

        refined = []
        next_generated_id = max((p.proposal_id for p in proposals), default=-1) + 1
        for proposal, components in zip(proposals, refined_by_proposal):
            for component in components:
                component_index = component["component_index"]
                component_count = component["component_count"]
                component_mask = component["mask"]
                new_area = component["area"]
                geo = component["geometric_features"]
                scores = component["soft_scores"]
                bbox_xyxy = component["bbox_xyxy"]

                proposal_id = proposal.proposal_id if component_index == 0 else next_generated_id
                if component_index > 0:
                    next_generated_id += 1

                refined.append(RefinedProposal2D(
                    proposal_id=proposal_id,
                    mask=component_mask,
                    bbox_xyxy=bbox_xyxy,
                    area=new_area,
                    confidence=proposal.confidence,
                    backend_name=proposal.backend_name,
                    metadata=dict(
                        proposal.metadata,
                        refined=True,
                        source_raw_proposal_id=proposal.proposal_id,
                        component_index=component_index,
                        split_component_count=component_count,
                        geometric_features={
                            "depth_valid_ratio": geo.depth_valid_ratio,
                            "depth_variance": geo.depth_variance,
                            "border_touch_ratio": geo.border_touch_ratio,
                            "planar_fit_residual": geo.planar_fit_residual,
                        },
                        soft_scores={
                            "objectness_score": scores.objectness_score,
                            "backgroundness_score": scores.backgroundness_score,
                            "attachedness_score": scores.attachedness_score,
                        },
                    ),
                    geometric_features=geo,
                    soft_scores=scores,
                ))
        logger.debug(f"Refined {len(proposals)} -> {len(refined)} proposals.")
        return refined

    def _refine_single_proposal(
        self,
        depth: np.ndarray,
        depth_edges: np.ndarray,
        proposal: Proposal2D,
    ) -> List[Dict[str, Any]]:
        new_mask = proposal.mask & (~depth_edges)
        component_masks = self._split_connected_components(new_mask)
        component_count = len(component_masks)
        refined_components: List[Dict[str, Any]] = []

        for component_index, component_mask in enumerate(component_masks):
            new_area = int(component_mask.sum())
            if new_area < self.min_area:
                continue

            geo = self._compute_geometric_features(depth, component_mask)
            scores = self._compute_soft_scores(depth, component_mask, geo)
            bbox_xyxy = self._mask_bbox(component_mask)
            refined_components.append(
                {
                    "component_index": component_index,
                    "component_count": component_count,
                    "mask": component_mask,
                    "area": new_area,
                    "geometric_features": geo,
                    "soft_scores": scores,
                    "bbox_xyxy": bbox_xyxy,
                }
            )
        return refined_components

    @staticmethod
    def _resolve_worker_count(configured_workers: int, task_count: int) -> int:
        if task_count <= 0:
            return 1
        if configured_workers > 0:
            return max(1, min(configured_workers, task_count))
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count, task_count))

    def _compute_depth_edges(self, depth: np.ndarray) -> np.ndarray:
        """Detect depth discontinuity edges."""
        grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        edges = (grad_x > self.depth_edge_threshold) | (grad_y > self.depth_edge_threshold)
        return edges

    def _compute_geometric_features(
        self, depth: np.ndarray, mask: np.ndarray,
    ) -> ProposalGeometricFeatures:
        """Compute 4 explicit geometric features for a refined proposal."""
        valid = mask & np.isfinite(depth) & (depth > 0)
        depth_values = depth[valid]
        mask_count = max(int(mask.sum()), 1)

        # depth_valid_ratio
        depth_valid_ratio = float(depth_values.size / mask_count)

        # depth_variance (using percentile range)
        if depth_values.size > 2:
            q10 = float(np.percentile(depth_values, 10.0))
            q90 = float(np.percentile(depth_values, 90.0))
            depth_variance = (q90 - q10) ** 2
        else:
            depth_variance = 0.0

        # border_touch_ratio
        border_touch_ratio = self._border_touch_ratio(mask)

        # planar_fit_residual
        planar_fit_residual = self._fit_plane_residual(valid, depth)

        return ProposalGeometricFeatures(
            depth_valid_ratio=depth_valid_ratio,
            depth_variance=depth_variance,
            border_touch_ratio=border_touch_ratio,
            planar_fit_residual=planar_fit_residual,
        )

    def _compute_soft_scores(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        geo: ProposalGeometricFeatures,
    ) -> ProposalSoftScores:
        """Compute 3 explicit soft scores as weighted functions of geometric features."""
        image_area = max(1, int(np.prod(mask.shape)))
        mask_extent = float(mask.sum()) / image_area

        # Compact: high fill ratio within bbox
        ys, xs = np.nonzero(mask)
        if xs.size > 0:
            bbox_area = max(1.0, float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))
            compact = float(mask.sum()) / bbox_area
        else:
            compact = 0.0

        small_extent = 1.0 - min(1.0, mask_extent / 0.18) if mask_extent < 0.18 else 0.0
        depth_rich = float(np.clip(geo.depth_variance / max(self.depth_variance_ref, 1e-6), 0.0, 1.0))
        low_border = 1.0 - geo.border_touch_ratio

        # objectness_score
        objectness = (
            self.w_obj_compact * compact
            + self.w_obj_small_extent * small_extent
            + self.w_obj_depth_rich * depth_rich
            + self.w_obj_low_border * low_border
            + self.w_obj_depth_valid * geo.depth_valid_ratio
        )

        large_extent = float(np.clip(mask_extent / 0.18, 0.0, 1.0))
        if geo.planar_fit_residual < float("inf"):
            planar_like = float(max(0.0, 1.0 - geo.planar_fit_residual / max(self.planar_residual_ref, 1e-6)))
        else:
            planar_like = 0.0
        flat_depth = float(max(0.0, 1.0 - geo.depth_variance / max(self.depth_variance_ref * 0.5, 1e-6)))

        # backgroundness_score
        backgroundness = (
            self.w_bg_large_extent * large_extent
            + self.w_bg_border_touch * geo.border_touch_ratio
            + self.w_bg_planar * planar_like
            + self.w_bg_flat_depth * flat_depth
        )

        # attachedness_score: how much this patch is "stuck" to something else
        attachedness = float(np.clip(
            0.4 * geo.border_touch_ratio
            + 0.3 * (1.0 - compact)
            + 0.3 * (1.0 - geo.depth_valid_ratio),
            0.0, 1.0,
        ))

        return ProposalSoftScores(
            objectness_score=float(np.clip(objectness, 0.0, 1.0)),
            backgroundness_score=float(np.clip(backgroundness, 0.0, 1.0)),
            attachedness_score=float(np.clip(attachedness, 0.0, 1.0)),
        )

    def _border_touch_ratio(self, mask: np.ndarray) -> float:
        """Fraction of mask pixels on image border."""
        if not np.any(mask):
            return 0.0
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        return float((mask & border).sum() / max(int(mask.sum()), 1))

    def _fit_plane_residual(self, valid_mask: np.ndarray, depth: np.ndarray) -> float:
        """Fit a depth plane and return RMSE residual."""
        ys, xs = np.nonzero(valid_mask)
        if xs.size < 8:
            return float("inf")
        if xs.size > self.max_plane_fit_points:
            select = np.linspace(0, xs.size - 1, self.max_plane_fit_points, dtype=np.int32)
            xs, ys = xs[select], ys[select]

        zs = depth[ys, xs].astype(np.float64)
        xs_norm = xs.astype(np.float64) / max(1.0, depth.shape[1] - 1)
        ys_norm = ys.astype(np.float64) / max(1.0, depth.shape[0] - 1)
        A = np.stack([xs_norm, ys_norm, np.ones_like(xs_norm)], axis=1)
        try:
            coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.sqrt(np.mean((A @ coeffs - zs) ** 2)))

    def _split_connected_components(self, mask: np.ndarray) -> List[np.ndarray]:
        """Split a binary mask into 4-connected components."""
        if not np.any(mask):
            return []

        visited = np.zeros_like(mask, dtype=bool)
        components: List[np.ndarray] = []
        height, width = mask.shape

        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            component_pixels = []
            while stack:
                y, x = stack.pop()
                component_pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            component_mask = np.zeros_like(mask, dtype=bool)
            cy, cx = zip(*component_pixels)
            component_mask[np.array(cy), np.array(cx)] = True
            components.append(component_mask)

        components.sort(key=lambda component: int(component.sum()), reverse=True)
        return components

    def _mask_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Compute tight xyxy bbox for a binary mask."""
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            dtype=np.float32,
        )
