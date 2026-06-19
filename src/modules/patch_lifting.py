"""Module 4: Patch Lifting (Layer 3).

Responsibility: Lift refined 2D proposals into 3D object patches
in world coordinates using depth and camera pose.
Each patch is only a partial observation.
"""

from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import (
    CameraIntrinsics,
    Patch3D,
    ProposalSoftScores,
    RefinedProposal2D,
)
from src.utils.geometry import compute_bbox

logger = logging.getLogger("oviovo.modules.patch_lifting")


class PatchLiftingModule:
    """Lifts 2D proposals to 3D patches using depth back-projection.

    Uses explicit RGB-D lifting: depth + intrinsics + pose -> world-frame Patch3D.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.min_points = config.get("min_points", 10)
        self.proposal_parallel_enabled = bool(config.get("proposal_parallel_enabled", True))
        self.proposal_parallel_workers = int(config.get("proposal_parallel_workers", 0))
        self.proposal_parallel_min_tasks = int(config.get("proposal_parallel_min_tasks", 8))
        fg_cfg = config.get("foreground_depth_filter", {})
        self.foreground_depth_filter_enabled = bool(fg_cfg.get("enabled", False))
        self.foreground_front_quantile = float(fg_cfg.get("front_quantile", 0.05))
        self.foreground_depth_band = float(fg_cfg.get("depth_band", 0.08))
        self.foreground_min_component_points = int(fg_cfg.get("min_component_points", self.min_points))
        self.foreground_min_depth_gap = float(fg_cfg.get("min_depth_gap", 0.15))
        self.foreground_max_removed_ratio = float(
            fg_cfg.get("max_removed_ratio", fg_cfg.get("max_foreground_ratio", 0.50))
        )
        self.point_sample_ratio = float(config.get("point_sample_ratio", 1.0))
        self.max_points_per_patch = int(config.get("max_points_per_patch", 0))
        self.voxel_size = float(config.get("voxel_size", 0.05))
        self.voxel_cache_enabled = bool(config.get("voxel_cache_enabled", False))
        self._pixel_grid_cache: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        logger.info("PatchLiftingModule initialized.")

    def process(
        self,
        proposals: List[RefinedProposal2D],
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> List[Patch3D]:
        """Lift refined 2D proposals to 3D patches.

        Args:
            proposals: Refined 2D proposals with geometric features.
            depth: (H, W) float32 depth map.
            pose: (4, 4) camera-to-world transform.
            intrinsics: Camera intrinsics.
            frame_id: Source frame ID.
            timestamp: Frame timestamp.

        Returns:
            List of Patch3D with inherited soft scores.
        """
        valid_depth = np.isfinite(depth) & (depth > 0)
        pixel_u, pixel_v = self._get_pixel_grid(depth.shape)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]

        if (
            self.proposal_parallel_enabled
            and len(proposals) >= self.proposal_parallel_min_tasks
        ):
            worker_count = self._resolve_worker_count(self.proposal_parallel_workers, len(proposals))
            if worker_count > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    patch_results = list(
                        executor.map(
                            lambda proposal: self._lift_single_proposal(
                                proposal=proposal,
                                depth=depth,
                                valid_depth=valid_depth,
                                pixel_u=pixel_u,
                                pixel_v=pixel_v,
                                intrinsics=intrinsics,
                                rotation=rotation,
                                translation=translation,
                                frame_id=frame_id,
                                timestamp=timestamp,
                            ),
                            proposals,
                        )
                    )
            else:
                patch_results = [
                    self._lift_single_proposal(
                        proposal=proposal,
                        depth=depth,
                        valid_depth=valid_depth,
                        pixel_u=pixel_u,
                        pixel_v=pixel_v,
                        intrinsics=intrinsics,
                        rotation=rotation,
                        translation=translation,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                    for proposal in proposals
                ]
        else:
            patch_results = [
                self._lift_single_proposal(
                    proposal=proposal,
                    depth=depth,
                    valid_depth=valid_depth,
                    pixel_u=pixel_u,
                    pixel_v=pixel_v,
                    intrinsics=intrinsics,
                    rotation=rotation,
                    translation=translation,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
                for proposal in proposals
            ]

        patches = [patch for patch in patch_results if patch is not None]
        logger.debug(f"Lifted {len(patches)} patches from {len(proposals)} proposals.")
        return patches

    def _lift_single_proposal(
        self,
        proposal: RefinedProposal2D,
        depth: np.ndarray,
        valid_depth: np.ndarray,
        pixel_u: np.ndarray,
        pixel_v: np.ndarray,
        intrinsics: CameraIntrinsics,
        rotation: np.ndarray,
        translation: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> Patch3D | None:
        valid = proposal.mask & valid_depth
        if not np.any(valid):
            return None

        u = pixel_u[valid].astype(np.float64)
        v = pixel_v[valid].astype(np.float64)
        z = depth[valid].astype(np.float64)
        if z.size < self.min_points:
            return None

        foreground_mask, foreground_debug = self._foreground_depth_mask(z)
        if not np.all(foreground_mask):
            u = u[foreground_mask]
            v = v[foreground_mask]
            z = z[foreground_mask]
        else:
            foreground_debug["kept_point_count"] = int(z.size)
            foreground_debug["removed_point_count"] = int(foreground_debug.get("input_point_count", z.size) - z.size)
        if z.size < self.min_points:
            return None

        u, v, z, sampling_debug = self._sample_pixel_vectors(u, v, z)
        if z.size < self.min_points:
            return None

        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy
        points_cam = np.stack([x, y, z], axis=-1)
        points = ((rotation @ points_cam.T).T + translation).astype(np.float32)

        if len(points) < self.min_points:
            return None

        centroid = points.mean(axis=0)
        bbox_min, bbox_max = compute_bbox(points)
        voxel_metadata = self._build_voxel_metadata(points) if self.voxel_cache_enabled else {}
        return Patch3D(
            patch_id=proposal.proposal_id,
            points=points,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            normals=None,
            timestamp=timestamp,
            source_frame_id=frame_id,
            soft_scores=proposal.soft_scores if hasattr(proposal, "soft_scores") else ProposalSoftScores(),
            metadata={
                "source_proposal_id": proposal.proposal_id,
                "source_bbox_xyxy": proposal.bbox_xyxy.copy(),
                "source_backend_name": proposal.backend_name,
                "geometric_features": asdict(proposal.geometric_features),
                "soft_scores": asdict(proposal.soft_scores),
                "observation_layer": str(proposal.metadata.get("observation_layer", "")),
                "refinement_key": str(proposal.metadata.get("refinement_key", "")),
                "mask_source": str(proposal.metadata.get("mask_source", "")),
                "source_raw_proposal_ids": list(proposal.metadata.get("source_raw_proposal_ids", []) or []),
                "anchor_id": int(proposal.metadata.get("anchor_id", -1)),
                "anchor_class_name": str(proposal.metadata.get("anchor_class_name", "")),
                "anchor_confidence": float(proposal.metadata.get("anchor_confidence", 0.0)),
                "anchor_bbox_iou": float(proposal.metadata.get("anchor_bbox_iou", 0.0)),
                "anchor_center_inside": bool(proposal.metadata.get("anchor_center_inside", False)),
                "anchor_keepalive": bool(proposal.metadata.get("anchor_keepalive", False)),
                "anchor_label_strength": str(proposal.metadata.get("anchor_label_strength", "strong")),
                "anchor_label_votes": dict(proposal.metadata.get("anchor_label_votes", {}) or {}),
                "anchor_blocked_candidates": list(proposal.metadata.get("anchor_blocked_candidates", []) or []),
                "semantic_commit_allowed": bool(proposal.metadata.get("semantic_commit_allowed", True)),
                "residual_semantic_policy": str(proposal.metadata.get("residual_semantic_policy", "")),
                "mask_anchor_relation": str(proposal.metadata.get("mask_anchor_relation", "")),
                "anchor_hit_count": int(proposal.metadata.get("anchor_hit_count", 0)),
                "anchor_ids": list(proposal.metadata.get("anchor_ids", []) or []),
                "force_object_candidate": bool(proposal.metadata.get("force_object_candidate", False)),
                "depth_keepalive_reason": str(proposal.metadata.get("depth_keepalive_reason", "")),
                "protected_small_anchor": bool(proposal.metadata.get("protected_small_anchor", False)),
                "anchor_is_nested_child": bool(proposal.metadata.get("anchor_is_nested_child", False)),
                "nested_parent_anchor_ids": list(proposal.metadata.get("nested_parent_anchor_ids", []) or []),
                "protected_child_anchor_ids": list(proposal.metadata.get("protected_child_anchor_ids", []) or []),
                "protected_child_classes": list(proposal.metadata.get("protected_child_classes", []) or []),
                "lifted_point_count": int(len(points)),
                **voxel_metadata,
                "foreground_depth_filter": foreground_debug,
                "point_sampling": sampling_debug,
                "lifting_mode": "rgbd_world_frame_precomputed_grid",
            },
        )

    def _sample_pixel_vectors(
        self,
        u: np.ndarray,
        v: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        input_count = int(z.size)
        ratio = float(np.clip(self.point_sample_ratio, 0.0, 1.0))
        ratio_limit = input_count if ratio <= 0.0 or ratio >= 1.0 else max(self.min_points, int(np.ceil(input_count * ratio)))
        cap_limit = input_count if self.max_points_per_patch <= 0 else max(self.min_points, int(self.max_points_per_patch))
        target = min(input_count, ratio_limit, cap_limit)
        diagnostics = {
            "enabled": bool(target < input_count),
            "input_point_count": input_count,
            "sampled_point_count": int(target),
            "point_sample_ratio": float(self.point_sample_ratio),
            "max_points_per_patch": int(self.max_points_per_patch),
        }
        if target >= input_count:
            return u, v, z, diagnostics
        indices = np.linspace(0, input_count - 1, num=target, dtype=np.int64)
        return u[indices], v[indices], z[indices], diagnostics

    def _build_voxel_metadata(self, points: np.ndarray) -> dict[str, Any]:
        if len(points) == 0:
            return {
                "voxel_indices": np.empty((0, 3), dtype=np.int64),
                "unique_voxel_indices": np.empty((0, 3), dtype=np.int64),
                "representative_point_indices": np.zeros(0, dtype=np.int64),
                "voxel_size": float(self.voxel_size),
                "voxel_cache_point_count": 0,
                "voxel_cache_bbox_min": np.zeros(3, dtype=np.float32),
                "voxel_cache_bbox_max": np.zeros(3, dtype=np.float32),
            }

        points = np.asarray(points, dtype=np.float32)
        safe_voxel_size = max(float(self.voxel_size), 1e-9)
        voxel_indices = np.floor(points / safe_voxel_size).astype(np.int64)
        unique_voxels, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return {
            "voxel_indices": voxel_indices,
            "unique_voxel_indices": unique_voxels.astype(np.int64, copy=False),
            "representative_point_indices": first_indices.astype(np.int64, copy=False),
            "voxel_size": float(self.voxel_size),
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32, copy=False),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32, copy=False),
        }

    def _foreground_depth_mask(self, z: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        z = np.asarray(z, dtype=np.float64)
        diagnostics = {
            "enabled": bool(self.foreground_depth_filter_enabled),
            "input_point_count": int(z.size),
            "kept_point_count": int(z.size),
            "removed_point_count": 0,
            "front_depth": 0.0,
            "depth_band": float(self.foreground_depth_band),
            "min_depth_gap": float(self.foreground_min_depth_gap),
            "max_removed_ratio": float(self.foreground_max_removed_ratio),
            "split_depth": 0.0,
            "depth_gap": 0.0,
            "foreground_ratio": 1.0,
            "removed_ratio": 0.0,
            "fallback_reason": "",
        }
        if not self.foreground_depth_filter_enabled or z.size == 0:
            return np.ones(z.shape, dtype=bool), diagnostics

        sorted_z = np.sort(z[np.isfinite(z)])
        if sorted_z.size < 2:
            diagnostics["fallback_reason"] = "no_depth_gap"
            return np.ones(z.shape, dtype=bool), diagnostics

        gaps = np.diff(sorted_z)
        max_gap_index = int(np.argmax(gaps))
        max_gap = float(gaps[max_gap_index])
        split_depth = float(sorted_z[max_gap_index])
        foreground_count = max_gap_index + 1
        foreground_ratio = float(foreground_count / sorted_z.size)
        diagnostics["front_depth"] = float(sorted_z[0])
        diagnostics["split_depth"] = split_depth
        diagnostics["depth_gap"] = max_gap
        diagnostics["foreground_ratio"] = foreground_ratio

        min_depth_gap = max(0.0, float(self.foreground_min_depth_gap))
        if max_gap < min_depth_gap:
            diagnostics["fallback_reason"] = "no_depth_gap"
            return np.ones(z.shape, dtype=bool), diagnostics

        keep = z <= split_depth + float(self.foreground_depth_band)
        kept_count = int(keep.sum())
        removed_count = int(z.size - kept_count)
        removed_ratio = float(removed_count / z.size) if z.size else 0.0
        diagnostics["kept_point_count"] = kept_count
        diagnostics["removed_point_count"] = removed_count
        diagnostics["removed_ratio"] = removed_ratio

        min_points = max(int(self.min_points), int(self.foreground_min_component_points))
        if kept_count < min_points:
            diagnostics["kept_point_count"] = int(z.size)
            diagnostics["removed_point_count"] = 0
            diagnostics["fallback_reason"] = "insufficient_foreground_points"
            return np.ones(z.shape, dtype=bool), diagnostics

        max_removed_ratio = float(np.clip(self.foreground_max_removed_ratio, 0.0, 1.0))
        if removed_ratio > max_removed_ratio:
            diagnostics["kept_point_count"] = int(z.size)
            diagnostics["removed_point_count"] = 0
            diagnostics["fallback_reason"] = "removed_ratio_too_large"
            return np.ones(z.shape, dtype=bool), diagnostics

        return keep, diagnostics

    def _get_pixel_grid(self, depth_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        cached = self._pixel_grid_cache.get(depth_shape)
        if cached is not None:
            return cached

        height, width = depth_shape
        pixel_u, pixel_v = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        self._pixel_grid_cache[depth_shape] = (pixel_u, pixel_v)
        return pixel_u, pixel_v

    @staticmethod
    def _resolve_worker_count(configured_workers: int, task_count: int) -> int:
        if task_count <= 0:
            return 1
        if configured_workers > 0:
            return max(1, min(configured_workers, task_count))
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count, task_count))
