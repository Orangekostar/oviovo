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

        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = (v - intrinsics.cy) * z / intrinsics.fy
        points_cam = np.stack([x, y, z], axis=-1)
        points = ((rotation @ points_cam.T).T + translation).astype(np.float32)

        if len(points) < self.min_points:
            return None

        centroid = points.mean(axis=0)
        bbox_min, bbox_max = compute_bbox(points)
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
                "lifted_point_count": int(len(points)),
                "lifting_mode": "rgbd_world_frame_precomputed_grid",
            },
        )

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
