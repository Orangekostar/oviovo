"""Runtime mask consolidation for object-level proposal cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import logging
import math
import os

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame, ObjectMap, Proposal2D, SystemState

logger = logging.getLogger("oviovo.modules.runtime_vis")


@dataclass
class WholeObjectEvidence:
    """Historical signal that an object was previously seen as a coherent whole."""

    object_id: int | None
    source: str
    canonical_bbox_xyxy: Optional[np.ndarray]
    canonical_depth_mean: Optional[float]
    canonical_depth_std: Optional[float]
    whole_observation_count: int
    whole_observation_score: float
    last_seen_frame: int


@dataclass
class RuntimeProposalProfile:
    """Depth-aware per-proposal summary used by debugging and manifests."""

    proposal_id: int
    objectness_score: float
    backgroundness_score: float
    depth_valid_ratio: float
    depth_variance: float
    planar_fit_residual: float
    border_touch_ratio: float
    depth_edge_density: float
    local_closedness: float
    mask_extent_ratio: float
    is_object_like: bool
    is_background_like: bool
    proposal_class: str
    best_prior_object_id: int | None
    best_prior_score: float


@dataclass
class RuntimeMergeDecision:
    """Pairwise merge decision with explicit reasoning."""

    mask_id_a: int
    mask_id_b: int
    adjacency_score: float
    depth_continuity_score: float
    plane_similarity_score: float
    bbox_plausibility_score: float
    containment_score: float
    median_depth_gap: float
    boundary_depth_continuity: float
    plane_compatibility: float
    merged_bbox_compactness: float
    base_score: float
    whole_prior_score: float
    background_conflict_penalty: float
    final_score: float
    accepted: bool
    accepted_reason: str
    boosted_by_whole_prior: bool
    linked_object_id: int | None
    rejected_due_to_background_conflict: bool
    reason_breakdown: Dict[str, float | int | bool | str] = field(default_factory=dict)


@dataclass
class RuntimeMaskGroup:
    """One merged object-level mask candidate for the current frame."""

    group_id: int
    member_mask_ids: List[int]
    merged_mask: np.ndarray
    merged_bbox_xyxy: np.ndarray
    area: int
    confidence: float
    linked_object_id: int | None
    whole_prior_used: bool
    merge_reason: str
    merge_reason_summary: Dict[str, float | int | bool | str] = field(default_factory=dict)


@dataclass
class RuntimeVisOutput:
    """Output package for runtime mask consolidation."""

    raw_proposals: List[Proposal2D]
    merged_proposals: List[Proposal2D]
    groups: List[RuntimeMaskGroup]
    raw_to_group: Dict[int, int]
    group_to_linked_object: Dict[int, int | None]
    proposal_profiles: List[RuntimeProposalProfile]
    merge_decisions: List[RuntimeMergeDecision]
    group_stats: Dict[str, float | int]


@dataclass
class _MaskFeatures:
    """Cached geometric summary for a raw proposal."""

    proposal_id: int
    proposal: Proposal2D
    bbox_xyxy: np.ndarray
    bbox_area: float
    fill_ratio: float
    center_xy: np.ndarray
    mean_depth: float
    median_depth: float
    depth_std: float
    depth_valid_ratio: float
    depth_variance: float
    planar_fit_residual: float
    border_touch_ratio: float
    mask_extent_ratio: float
    depth_edge_density: float
    local_closedness: float
    boundary_mask: np.ndarray
    plane_coeffs: Optional[np.ndarray]
    touches_border: bool
    objectness_score: float = 0.0
    backgroundness_score: float = 0.0
    is_object_like: bool = False
    is_background_like: bool = False
    prior_fit_by_object: Dict[int, float] = field(default_factory=dict)
    best_prior_object_id: int | None = None
    best_prior_score: float = 0.0


class RuntimeVisModule:
    """Merge fragmented raw masks into object-level candidates.

    v1 intentionally uses explainable pairwise scores and connected-components
    grouping. Historical whole-object evidence acts as a significant bonus, but
    it is never a hard binding: current-frame geometry must still be plausible.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = config.get("enabled", True)
        self.merge_score_threshold = float(config.get("merge_score_threshold", 0.65))
        self.base_score_floor = float(config.get("base_score_floor", 0.35))
        self.lambda_whole_prior = float(config.get("lambda_whole_prior", 1.5))
        self.mu_background_conflict = float(config.get("mu_background_conflict", 1.35))
        self.bbox_gap_px = float(config.get("bbox_gap_px", 30))
        self.depth_mean_diff_max = float(config.get("depth_mean_diff_max", 0.18))
        self.prior_depth_diff_max = float(config.get("prior_depth_diff_max", 0.25))
        self.plane_similarity_diff_max = float(config.get("plane_similarity_diff_max", 0.15))
        self.bbox_fill_ratio_ref = float(config.get("bbox_fill_ratio_ref", 0.55))
        self.expanded_prior_bbox_scale = float(config.get("expanded_prior_bbox_scale", 1.15))
        self.recent_history_size = int(config.get("recent_history_size", 10))
        self.min_group_area = int(config.get("min_group_area", 120))
        self.edge_admission_min_adjacency = float(config.get("edge_admission_min_adjacency", 0.55))
        self.edge_admission_min_depth_continuity = float(config.get("edge_admission_min_depth_continuity", 0.55))
        self.edge_admission_min_plane_similarity = float(config.get("edge_admission_min_plane_similarity", 0.60))
        self.edge_admission_min_whole_prior = float(config.get("edge_admission_min_whole_prior", 0.65))
        self.small_object_area_threshold = int(config.get("small_object_area_threshold", 1200))
        self.small_object_depth_gap_min = float(config.get("small_object_depth_gap_min", 0.12))
        self.small_object_bbox_plausibility_margin = float(
            config.get("small_object_bbox_plausibility_margin", 0.08)
        )
        self.small_object_protection_threshold = float(
            config.get("small_object_protection_threshold", 0.60)
        )
        self.small_object_protection_penalty = float(
            config.get("small_object_protection_penalty", 0.25)
        )
        self.max_plane_fit_points = int(config.get("max_plane_fit_points", 256))

        # Depth-aware proposal gating.
        self.depth_edge_threshold = float(config.get("depth_edge_threshold", 0.05))
        self.object_like_threshold = float(config.get("object_like_threshold", 0.55))
        self.background_like_threshold = float(config.get("background_like_threshold", 0.62))
        self.class_margin = float(config.get("class_margin", 0.08))
        self.uncertain_objectness_min = float(config.get("uncertain_objectness_min", 0.45))
        self.background_conflict_reject_threshold = float(
            config.get("background_conflict_reject_threshold", 0.62)
        )
        self.background_planar_residual_ref = float(config.get("background_planar_residual_ref", 0.02))
        self.background_low_variance_ref = float(config.get("background_low_variance_ref", 0.006))
        self.background_large_extent_ref = float(config.get("background_large_extent_ref", 0.18))
        self.object_depth_variance_ref = float(config.get("object_depth_variance_ref", 0.012))
        self.object_edge_density_ref = float(config.get("object_edge_density_ref", 0.45))
        self.small_swallow_area_ratio = float(config.get("small_swallow_area_ratio", 0.22))
        self.min_boundary_depth_points = int(config.get("min_boundary_depth_points", 6))
        self.semantic_class_merge_gate_enabled = bool(config.get("semantic_class_merge_gate_enabled", False))
        self.require_same_anchor_label_for_merge = bool(
            config.get("require_same_anchor_label_for_merge", False)
        )
        self.pairwise_parallel_enabled = bool(config.get("pairwise_parallel_enabled", True))
        self.pairwise_parallel_workers = int(config.get("pairwise_parallel_workers", 0))
        self.pairwise_parallel_min_pairs = int(config.get("pairwise_parallel_min_pairs", 64))

        self.last_output: Optional[RuntimeVisOutput] = None
        logger.info("RuntimeVisModule initialized.")

    def process(
        self,
        frame: Frame,
        raw_proposals: List[Proposal2D],
        state: SystemState,
    ) -> RuntimeVisOutput:
        """Consolidate fragmented raw masks into object-level groups."""
        if not self.enabled:
            output = self._identity_output(raw_proposals)
            self.last_output = output
            return output

        if not raw_proposals:
            output = self._identity_output(raw_proposals)
            self.last_output = output
            return output

        depth_edges = self._compute_depth_edge_map(frame.depth)
        features = [
            self._compute_mask_features(frame.depth, depth_edges, proposal)
            for proposal in raw_proposals
        ]
        evidences = self._build_whole_object_evidence(frame, state)
        self._attach_prior_fits(features, evidences)
        self._apply_depth_aware_gating(features)

        decisions = self._compute_pairwise_decisions(features, evidences, frame.depth)
        groups, raw_to_group, group_to_linked_object = self._build_groups(features, decisions)
        feature_lookup = {feature.proposal_id: feature for feature in features}
        merged_proposals = [
            Proposal2D(
                proposal_id=group.group_id,
                mask=group.merged_mask,
                bbox_xyxy=group.merged_bbox_xyxy.copy(),
                area=group.area,
                confidence=group.confidence,
                backend_name=self._group_backend_name(group, feature_lookup=feature_lookup),
                metadata={
                    "source": "runtime_vis",
                    "member_mask_ids": list(group.member_mask_ids),
                    "linked_object_id": group.linked_object_id,
                    **self._anchor_group_metadata(
                        [feature_lookup[mask_id] for mask_id in group.member_mask_ids if mask_id in feature_lookup],
                        [
                            decision
                            for decision in decisions
                            if decision.mask_id_a in group.member_mask_ids
                            and decision.mask_id_b in group.member_mask_ids
                        ],
                    ),
                },
            )
            for group in groups
        ]

        proposal_profiles = [self._proposal_profile_from_feature(feature) for feature in features]
        whole_prior_boosted = sum(1 for decision in decisions if decision.boosted_by_whole_prior)
        accepted_edge_count = sum(1 for decision in decisions if decision.accepted)
        rejected_edge_count = len(decisions) - accepted_edge_count
        background_blocked_edge_count = sum(
            1
            for decision in decisions
            if decision.rejected_due_to_background_conflict
            or decision.accepted_reason == "background_mask_excluded"
        )
        anchor_label_missing_edge_count = sum(
            1 for decision in decisions if decision.accepted_reason == "missing_anchor_label"
        )
        anchor_label_mismatch_edge_count = sum(
            1 for decision in decisions if decision.accepted_reason == "anchor_label_mismatch"
        )
        anchor_identity_mismatch_edge_count = sum(
            1 for decision in decisions if decision.accepted_reason == "anchor_identity_mismatch"
        )
        semantic_blocked_residual_edge_count = sum(
            1 for decision in decisions if decision.accepted_reason == "semantic_blocked_residual"
        )

        object_like_count = sum(1 for feature in features if feature.is_object_like)
        background_like_count = sum(1 for feature in features if feature.is_background_like)

        output = RuntimeVisOutput(
            raw_proposals=raw_proposals,
            merged_proposals=merged_proposals,
            groups=groups,
            raw_to_group=raw_to_group,
            group_to_linked_object=group_to_linked_object,
            proposal_profiles=proposal_profiles,
            merge_decisions=decisions,
            group_stats={
                "raw_mask_count": len(raw_proposals),
                "merged_group_count": len(groups),
                "object_like_count": object_like_count,
                "background_like_count": background_like_count,
                "uncertain_count": len(raw_proposals) - object_like_count - background_like_count,
                "graph_candidate_count": len(raw_proposals) - background_like_count,
                "accepted_edge_count": accepted_edge_count,
                "rejected_edge_count": rejected_edge_count,
                "background_blocked_edge_count": background_blocked_edge_count,
                "anchor_label_missing_edge_count": anchor_label_missing_edge_count,
                "anchor_label_mismatch_edge_count": anchor_label_mismatch_edge_count,
                "anchor_identity_mismatch_edge_count": anchor_identity_mismatch_edge_count,
                "semantic_blocked_residual_edge_count": semantic_blocked_residual_edge_count,
                "whole_prior_boosted_edge_count": whole_prior_boosted,
                "whole_prior_group_count": sum(1 for group in groups if group.whole_prior_used),
            },
        )
        self.last_output = output
        logger.debug(
            "RuntimeVis: raw=%d merged=%d accepted_edges=%d whole_prior_edges=%d",
            len(raw_proposals),
            len(groups),
            accepted_edge_count,
            whole_prior_boosted,
        )
        return output

    def _identity_output(self, raw_proposals: List[Proposal2D]) -> RuntimeVisOutput:
        groups: List[RuntimeMaskGroup] = []
        raw_to_group: Dict[int, int] = {}
        group_to_linked_object: Dict[int, int | None] = {}
        proposal_profiles: List[RuntimeProposalProfile] = []
        merged: List[Proposal2D] = []
        for group_id, proposal in enumerate(raw_proposals):
            raw_to_group[proposal.proposal_id] = group_id
            group_to_linked_object[group_id] = None
            groups.append(
                RuntimeMaskGroup(
                    group_id=group_id,
                    member_mask_ids=[proposal.proposal_id],
                    merged_mask=proposal.mask.copy(),
                    merged_bbox_xyxy=proposal.bbox_xyxy.copy(),
                    area=proposal.area,
                    confidence=float(proposal.confidence),
                    linked_object_id=None,
                    whole_prior_used=False,
                    merge_reason="identity",
                    merge_reason_summary={
                        "member_count": 1,
                        "accepted_internal_edge_count": 0,
                        "whole_prior_edge_count": 0,
                        "background_conflict_blocked_edges": 0,
                    },
                )
            )
            merged.append(
                Proposal2D(
                    proposal_id=group_id,
                    mask=proposal.mask.copy(),
                    bbox_xyxy=proposal.bbox_xyxy.copy(),
                    area=proposal.area,
                    confidence=float(proposal.confidence),
                    backend_name=proposal.backend_name,
                    metadata=dict(proposal.metadata),
                )
            )

            image_area = max(1, int(np.prod(proposal.mask.shape)))
            proposal_profiles.append(
                RuntimeProposalProfile(
                    proposal_id=proposal.proposal_id,
                    objectness_score=0.0,
                    backgroundness_score=0.0,
                    depth_valid_ratio=0.0,
                    depth_variance=0.0,
                    planar_fit_residual=float("inf"),
                    border_touch_ratio=0.0,
                    depth_edge_density=0.0,
                    local_closedness=0.0,
                    mask_extent_ratio=float(proposal.area) / float(image_area),
                    is_object_like=False,
                    is_background_like=False,
                    proposal_class="uncertain",
                    best_prior_object_id=None,
                    best_prior_score=0.0,
                )
            )

        return RuntimeVisOutput(
            raw_proposals=raw_proposals,
            merged_proposals=merged,
            groups=groups,
            raw_to_group=raw_to_group,
            group_to_linked_object=group_to_linked_object,
            proposal_profiles=proposal_profiles,
            merge_decisions=[],
            group_stats={
                "raw_mask_count": len(raw_proposals),
                "merged_group_count": len(groups),
                "object_like_count": 0,
                "background_like_count": 0,
                "uncertain_count": len(raw_proposals),
                "graph_candidate_count": len(raw_proposals),
                "accepted_edge_count": 0,
                "rejected_edge_count": 0,
                "background_blocked_edge_count": 0,
                "anchor_label_missing_edge_count": 0,
                "anchor_label_mismatch_edge_count": 0,
                "anchor_identity_mismatch_edge_count": 0,
                "semantic_blocked_residual_edge_count": 0,
                "whole_prior_boosted_edge_count": 0,
                "whole_prior_group_count": 0,
            },
        )

    def _compute_mask_features(
        self,
        depth: np.ndarray,
        depth_edges: np.ndarray,
        proposal: Proposal2D,
    ) -> _MaskFeatures:
        mask = proposal.mask.astype(bool)
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            bbox = proposal.bbox_xyxy.copy()
            bbox_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            return _MaskFeatures(
                proposal_id=proposal.proposal_id,
                proposal=proposal,
                bbox_xyxy=bbox,
                bbox_area=bbox_area,
                fill_ratio=0.0,
                center_xy=np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32),
                mean_depth=float("nan"),
                median_depth=float("nan"),
                depth_std=0.0,
                depth_valid_ratio=0.0,
                depth_variance=0.0,
                planar_fit_residual=float("inf"),
                border_touch_ratio=0.0,
                mask_extent_ratio=0.0,
                depth_edge_density=0.0,
                local_closedness=0.0,
                boundary_mask=np.zeros_like(mask, dtype=bool),
                plane_coeffs=None,
                touches_border=False,
            )

        bbox = np.array(
            [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
            dtype=np.float32,
        )
        bbox_area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

        valid = mask & np.isfinite(depth) & (depth > 0)
        depth_values = depth[valid]
        depth_valid_ratio = float(depth_values.size / max(int(mask.sum()), 1))
        if depth_values.size > 0:
            mean_depth = float(depth_values.mean())
            median_depth = float(np.median(depth_values))
            depth_std = float(depth_values.std())
            depth_q10 = float(np.percentile(depth_values, 10.0))
            depth_q90 = float(np.percentile(depth_values, 90.0))
            depth_variance = (depth_q90 - depth_q10) ** 2
            plane_coeffs, planar_fit_residual = self._fit_depth_plane(valid, depth)
        else:
            mean_depth = float("nan")
            median_depth = float("nan")
            depth_std = 0.0
            depth_variance = 0.0
            plane_coeffs = None
            planar_fit_residual = float("inf")

        border_touch_ratio = self._border_touch_ratio(mask)
        touches_border = border_touch_ratio > 0.0
        mask_extent_ratio = float(mask.sum() / max(1, int(np.prod(mask.shape))))
        boundary_mask = self._extract_boundary(mask)
        boundary_count = int(boundary_mask.sum())
        if boundary_count > 0:
            depth_edge_density = float((boundary_mask & depth_edges).sum() / boundary_count)
        else:
            depth_edge_density = 0.0

        local_closedness = float(
            np.clip(
                0.45 * float(mask.sum() / bbox_area)
                + 0.30 * (1.0 - border_touch_ratio)
                + 0.25 * depth_edge_density,
                0.0,
                1.0,
            )
        )

        return _MaskFeatures(
            proposal_id=proposal.proposal_id,
            proposal=proposal,
            bbox_xyxy=bbox,
            bbox_area=bbox_area,
            fill_ratio=float(mask.sum() / bbox_area),
            center_xy=np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32),
            mean_depth=mean_depth,
            median_depth=median_depth,
            depth_std=depth_std,
            depth_valid_ratio=depth_valid_ratio,
            depth_variance=float(depth_variance),
            planar_fit_residual=float(planar_fit_residual),
            border_touch_ratio=float(border_touch_ratio),
            mask_extent_ratio=float(mask_extent_ratio),
            depth_edge_density=float(depth_edge_density),
            local_closedness=local_closedness,
            boundary_mask=boundary_mask,
            plane_coeffs=plane_coeffs,
            touches_border=touches_border,
        )

    def _compute_depth_edge_map(self, depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0)
        edge_x = np.zeros_like(depth, dtype=np.float32)
        edge_y = np.zeros_like(depth, dtype=np.float32)

        valid_x = valid[:, 1:] & valid[:, :-1]
        diff_x = np.abs(depth[:, 1:] - depth[:, :-1]).astype(np.float32)
        edge_x[:, 1:] = np.where(valid_x, diff_x, 0.0)

        valid_y = valid[1:, :] & valid[:-1, :]
        diff_y = np.abs(depth[1:, :] - depth[:-1, :]).astype(np.float32)
        edge_y[1:, :] = np.where(valid_y, diff_y, 0.0)

        edge = (np.maximum(edge_x, edge_y) >= self.depth_edge_threshold) & valid
        return edge

    def _fit_depth_plane(
        self,
        valid_mask: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[Optional[np.ndarray], float]:
        ys, xs = np.nonzero(valid_mask)
        if xs.size < 8:
            return None, float("inf")
        if xs.size > self.max_plane_fit_points:
            select = np.linspace(0, xs.size - 1, self.max_plane_fit_points, dtype=np.int32)
            xs = xs[select]
            ys = ys[select]

        zs = depth[ys, xs].astype(np.float64)
        xs_norm = xs.astype(np.float64) / max(1.0, depth.shape[1] - 1)
        ys_norm = ys.astype(np.float64) / max(1.0, depth.shape[0] - 1)
        A = np.stack([xs_norm, ys_norm, np.ones_like(xs_norm)], axis=1)
        try:
            coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
        except np.linalg.LinAlgError:
            return None, float("inf")

        residual = float(np.sqrt(np.mean((A @ coeffs - zs) ** 2)))
        return coeffs.astype(np.float32), residual

    def _build_whole_object_evidence(self, frame: Frame, state: SystemState) -> List[WholeObjectEvidence]:
        evidences: List[WholeObjectEvidence] = []
        for object_id, obj in state.objects.items():
            bbox_xyxy, depth_mean = self._project_object_bbox(obj, frame.pose, frame.intrinsics)
            if bbox_xyxy is None:
                continue
            frame_gap = max(0, frame.frame_id - obj.last_seen_frame)
            recency = max(0.0, 1.0 - frame_gap / max(1.0, float(self.recent_history_size)))
            whole_obs = max(1, len(obj.observations))
            whole_score = min(1.0, 0.25 * min(whole_obs, 4) + 0.15 * min(obj.update_count, 4))
            whole_score *= max(0.25, recency)
            evidences.append(
                WholeObjectEvidence(
                    object_id=object_id,
                    source="existing_object",
                    canonical_bbox_xyxy=bbox_xyxy,
                    canonical_depth_mean=depth_mean,
                    canonical_depth_std=None,
                    whole_observation_count=whole_obs,
                    whole_observation_score=float(min(1.0, whole_score)),
                    last_seen_frame=obj.last_seen_frame,
                )
            )
        return evidences

    def _project_object_bbox(
        self,
        obj: ObjectMap,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        corners = self._bbox_corners(obj.bbox_min, obj.bbox_max)
        if corners.size == 0:
            return None, None

        pose_inv = np.linalg.inv(pose)
        cam_points = (pose_inv[:3, :3] @ corners.T).T + pose_inv[:3, 3]
        valid = cam_points[:, 2] > 1e-4
        if not np.any(valid):
            return None, None

        cam_points = cam_points[valid]
        u = intrinsics.fx * cam_points[:, 0] / cam_points[:, 2] + intrinsics.cx
        v = intrinsics.fy * cam_points[:, 1] / cam_points[:, 2] + intrinsics.cy
        bbox = np.array(
            [
                float(np.clip(u.min(), 0, intrinsics.width - 1)),
                float(np.clip(v.min(), 0, intrinsics.height - 1)),
                float(np.clip(u.max(), 0, intrinsics.width - 1)),
                float(np.clip(v.max(), 0, intrinsics.height - 1)),
            ],
            dtype=np.float32,
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None, None
        return bbox, float(cam_points[:, 2].mean())

    def _attach_prior_fits(self, features: List[_MaskFeatures], evidences: List[WholeObjectEvidence]) -> None:
        for feature in features:
            best_object_id = None
            best_score = 0.0
            fit_map: Dict[int, float] = {}
            for evidence in evidences:
                if evidence.object_id is None:
                    continue
                score = self._mask_to_prior_fit(feature, evidence)
                fit_map[evidence.object_id] = score
                if score > best_score:
                    best_object_id = evidence.object_id
                    best_score = score
            feature.prior_fit_by_object = fit_map
            feature.best_prior_object_id = best_object_id
            feature.best_prior_score = best_score

    def _mask_to_prior_fit(self, feature: _MaskFeatures, evidence: WholeObjectEvidence) -> float:
        if evidence.canonical_bbox_xyxy is None:
            return 0.0

        expanded_bbox = self._expand_bbox(evidence.canonical_bbox_xyxy, self.expanded_prior_bbox_scale)
        overlap = self._bbox_intersection_area(feature.bbox_xyxy, expanded_bbox)
        bbox_fit = overlap / max(feature.bbox_area, 1.0)
        if bbox_fit <= 0.0:
            return 0.0

        depth_fit = 0.5
        if evidence.canonical_depth_mean is not None and math.isfinite(feature.mean_depth):
            depth_fit = max(
                0.0,
                1.0 - abs(feature.mean_depth - evidence.canonical_depth_mean) / max(self.prior_depth_diff_max, 1e-6),
            )
        return float(np.clip(bbox_fit * depth_fit, 0.0, 1.0))

    def _apply_depth_aware_gating(self, features: List[_MaskFeatures]) -> None:
        for feature in features:
            aspect = self._bbox_aspect_ratio(feature.bbox_xyxy)
            elongated_score = float(np.clip((aspect - 2.0) / 3.0, 0.0, 1.0))
            large_extent_score = self._normalize_minmax(
                feature.mask_extent_ratio,
                self.background_large_extent_ref,
                1.0,
            )

            if math.isfinite(feature.planar_fit_residual):
                planar_like_score = float(
                    max(
                        0.0,
                        1.0 - feature.planar_fit_residual / max(self.background_planar_residual_ref, 1e-6),
                    )
                )
            else:
                planar_like_score = 0.0

            flat_depth_score = float(
                max(0.0, 1.0 - feature.depth_variance / max(self.background_low_variance_ref, 1e-6))
            )
            weak_closed_score = 1.0 - feature.local_closedness
            depth_rich_score = float(
                np.clip(feature.depth_variance / max(self.object_depth_variance_ref, 1e-6), 0.0, 1.0)
            )
            edge_support_score = float(
                np.clip(feature.depth_edge_density / max(self.object_edge_density_ref, 1e-6), 0.0, 1.0)
            )
            compact_score = feature.fill_ratio
            low_border_score = 1.0 - feature.border_touch_ratio
            small_extent_score = 1.0 - large_extent_score
            prior_support_score = float(np.clip(feature.best_prior_score, 0.0, 1.0))

            objectness = (
                0.20 * compact_score
                + 0.18 * small_extent_score
                + 0.16 * depth_rich_score
                + 0.18 * feature.local_closedness
                + 0.12 * low_border_score
                + 0.08 * edge_support_score
                + 0.08 * prior_support_score
            )

            backgroundness = (
                0.24 * large_extent_score
                + 0.20 * feature.border_touch_ratio
                + 0.18 * planar_like_score
                + 0.14 * flat_depth_score
                + 0.14 * elongated_score
                + 0.10 * weak_closed_score
            )

            if feature.depth_valid_ratio < 0.20:
                objectness *= 0.80
                backgroundness = min(1.0, backgroundness + 0.05)

            # TODO: replace this heuristic gate with a learned background-vs-object classifier.
            if (
                feature.mask_extent_ratio > 0.35
                and feature.border_touch_ratio > 0.02
                and planar_like_score > 0.70
                and flat_depth_score > 0.70
            ):
                backgroundness = max(backgroundness, 0.90)

            if (
                feature.proposal.area <= self.small_object_area_threshold
                and feature.local_closedness >= 0.55
                and feature.border_touch_ratio < 0.10
            ):
                objectness = max(objectness, 0.60)

            feature.objectness_score = float(np.clip(objectness, 0.0, 1.0))
            feature.backgroundness_score = float(np.clip(backgroundness, 0.0, 1.0))

            feature.is_background_like = (
                feature.backgroundness_score >= self.background_like_threshold
                and feature.backgroundness_score >= feature.objectness_score + self.class_margin
            )
            feature.is_object_like = (
                feature.objectness_score >= self.object_like_threshold
                and feature.objectness_score >= feature.backgroundness_score + self.class_margin
            )

            if feature.is_background_like and feature.is_object_like:
                if feature.backgroundness_score >= feature.objectness_score:
                    feature.is_object_like = False
                else:
                    feature.is_background_like = False

    def _proposal_profile_from_feature(self, feature: _MaskFeatures) -> RuntimeProposalProfile:
        proposal_class = "uncertain"
        if feature.is_object_like:
            proposal_class = "object_like"
        elif feature.is_background_like:
            proposal_class = "background_like"

        return RuntimeProposalProfile(
            proposal_id=feature.proposal_id,
            objectness_score=float(feature.objectness_score),
            backgroundness_score=float(feature.backgroundness_score),
            depth_valid_ratio=float(feature.depth_valid_ratio),
            depth_variance=float(feature.depth_variance),
            planar_fit_residual=float(feature.planar_fit_residual),
            border_touch_ratio=float(feature.border_touch_ratio),
            depth_edge_density=float(feature.depth_edge_density),
            local_closedness=float(feature.local_closedness),
            mask_extent_ratio=float(feature.mask_extent_ratio),
            is_object_like=feature.is_object_like,
            is_background_like=feature.is_background_like,
            proposal_class=proposal_class,
            best_prior_object_id=feature.best_prior_object_id,
            best_prior_score=float(feature.best_prior_score),
        )

    def _compute_pairwise_decisions(
        self,
        features: List[_MaskFeatures],
        evidences: List[WholeObjectEvidence],
        depth: np.ndarray,
    ) -> List[RuntimeMergeDecision]:
        pair_indices = [
            (i, j)
            for i in range(len(features))
            for j in range(i + 1, len(features))
        ]
        if not pair_indices:
            return []

        if (
            not self.pairwise_parallel_enabled
            or len(pair_indices) < self.pairwise_parallel_min_pairs
        ):
            return [
                self._score_pair(features[i], features[j], evidences, depth)
                for i, j in pair_indices
            ]

        worker_count = self._resolve_worker_count(self.pairwise_parallel_workers, len(pair_indices))
        if worker_count <= 1:
            return [
                self._score_pair(features[i], features[j], evidences, depth)
                for i, j in pair_indices
            ]

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(
                executor.map(
                    lambda pair: self._score_pair(
                        features[pair[0]],
                        features[pair[1]],
                        evidences,
                        depth,
                    ),
                    pair_indices,
                )
            )

    @staticmethod
    def _resolve_worker_count(configured_workers: int, task_count: int) -> int:
        if task_count <= 0:
            return 1
        if configured_workers > 0:
            return max(1, min(configured_workers, task_count))
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count, task_count))

    def _score_pair(
        self,
        feature_a: _MaskFeatures,
        feature_b: _MaskFeatures,
        evidences: List[WholeObjectEvidence],
        depth: np.ndarray,
    ) -> RuntimeMergeDecision:
        adjacency_score = self._adjacency_score(feature_a.bbox_xyxy, feature_b.bbox_xyxy)
        median_depth_gap = self._median_depth_gap(feature_a.median_depth, feature_b.median_depth)
        depth_gap_score = self._depth_gap_score(median_depth_gap)
        boundary_depth_continuity = self._boundary_depth_continuity_score(feature_a, feature_b, depth)
        plane_compatibility = self._plane_compatibility_score(feature_a, feature_b)
        merged_bbox_compactness = self._bbox_plausibility_score(feature_a, feature_b)
        containment_score = self._containment_score(feature_a, feature_b)

        base_score = (
            0.27 * adjacency_score
            + 0.27 * boundary_depth_continuity
            + 0.18 * depth_gap_score
            + 0.14 * plane_compatibility
            + 0.14 * merged_bbox_compactness
        )

        whole_prior_raw, linked_object_id = self._whole_prior_score(feature_a, feature_b, evidences)
        if feature_a.is_background_like or feature_b.is_background_like:
            # Rule 3: whole prior never boosts merges with strong background-like masks.
            whole_prior_score = 0.0
        else:
            whole_prior_score = whole_prior_raw

        background_conflict_penalty = self._background_conflict_penalty(
            feature_a=feature_a,
            feature_b=feature_b,
            adjacency_score=adjacency_score,
            merged_bbox_compactness=merged_bbox_compactness,
            plane_compatibility=plane_compatibility,
            median_depth_gap=median_depth_gap,
        )

        final_score = (
            base_score
            + self.lambda_whole_prior * whole_prior_score
            - self.mu_background_conflict * background_conflict_penalty
        )

        edge_admitted = (
            not (feature_a.is_background_like or feature_b.is_background_like)
            and (
                adjacency_score >= self.edge_admission_min_adjacency
                or boundary_depth_continuity >= self.edge_admission_min_depth_continuity
                or plane_compatibility >= self.edge_admission_min_plane_similarity
                or whole_prior_score >= self.edge_admission_min_whole_prior
            )
        )

        smaller_feature, larger_feature = (
            (feature_a, feature_b)
            if feature_a.proposal.area <= feature_b.proposal.area
            else (feature_b, feature_a)
        )
        cross_class_anchor_conflict = self._anchor_labels_conflict(feature_a.proposal, feature_b.proposal)
        missing_required_anchor_label = self._missing_required_anchor_label(
            feature_a.proposal,
            feature_b.proposal,
        )
        required_anchor_labels_mismatch = self._required_anchor_labels_mismatch(
            feature_a.proposal,
            feature_b.proposal,
        )
        required_anchor_identity_mismatch = self._required_anchor_identity_mismatch(
            feature_a.proposal,
            feature_b.proposal,
        )
        semantic_blocked_residual_merge = self._semantic_blocked_residual_merge(
            feature_a.proposal,
            feature_b.proposal,
        )
        small_object_protection = self._small_object_protection_score(
            smaller_feature,
            larger_feature,
            merged_bbox_compactness=merged_bbox_compactness,
            median_depth_gap=median_depth_gap,
            boundary_depth_continuity=boundary_depth_continuity,
            linked_object_id=linked_object_id,
        )
        effective_threshold = self.merge_score_threshold
        if smaller_feature.proposal.area <= self.small_object_area_threshold:
            effective_threshold += self.small_object_protection_penalty * small_object_protection

        accepted = True
        accepted_reason = "accepted"
        rejected_due_to_background_conflict = False
        if semantic_blocked_residual_merge:
            accepted = False
            accepted_reason = "semantic_blocked_residual"
        elif missing_required_anchor_label:
            accepted = False
            accepted_reason = "missing_anchor_label"
        elif required_anchor_labels_mismatch:
            accepted = False
            accepted_reason = "anchor_label_mismatch"
        elif required_anchor_identity_mismatch:
            accepted = False
            accepted_reason = "anchor_identity_mismatch"
        elif cross_class_anchor_conflict:
            accepted = False
            accepted_reason = "cross_class_anchor_conflict"
        elif background_conflict_penalty >= self.background_conflict_reject_threshold:
            # Rule 2: severe background conflict hard-rejects the edge.
            accepted = False
            accepted_reason = "rejected_due_to_background_conflict"
            rejected_due_to_background_conflict = True
        elif feature_a.is_background_like or feature_b.is_background_like:
            # Rule 1: strong background-like masks do not participate in object grouping.
            accepted = False
            accepted_reason = "background_mask_excluded"
        elif not edge_admitted:
            accepted = False
            accepted_reason = "rejected_before_graph"
        elif base_score < self.base_score_floor:
            accepted = False
            accepted_reason = "base_score_too_low"
        elif (
            smaller_feature.proposal.area <= self.small_object_area_threshold
            and small_object_protection >= self.small_object_protection_threshold
            and boundary_depth_continuity < 0.80
            and whole_prior_score < 0.80
        ):
            accepted = False
            accepted_reason = "small_object_protected"
        elif final_score < effective_threshold:
            accepted = False
            accepted_reason = "final_score_below_threshold"
        elif whole_prior_score > 0.0:
            accepted_reason = "accepted_with_whole_prior_boost"
        else:
            accepted_reason = "accepted_from_base_geometry"

        boosted = bool(whole_prior_score > 0.0 and accepted)
        breakdown: Dict[str, float | int | bool | str] = {
            "edge_admitted": edge_admitted,
            "base_score_floor": round(self.base_score_floor, 4),
            "effective_merge_threshold": round(effective_threshold, 4),
            "raw_whole_prior_score": round(whole_prior_raw, 4),
            "small_object_protection_score": round(small_object_protection, 4),
            "small_object_candidate": smaller_feature.proposal.area <= self.small_object_area_threshold,
            "missing_required_anchor_label": missing_required_anchor_label,
            "required_anchor_labels_mismatch": required_anchor_labels_mismatch,
            "required_anchor_identity_mismatch": required_anchor_identity_mismatch,
            "semantic_blocked_residual_merge": semantic_blocked_residual_merge,
            "cross_class_anchor_conflict": cross_class_anchor_conflict,
            "anchor_class_a": self._proposal_anchor_label(feature_a.proposal),
            "anchor_class_b": self._proposal_anchor_label(feature_b.proposal),
            "anchor_id_a": self._proposal_anchor_id(feature_a.proposal),
            "anchor_id_b": self._proposal_anchor_id(feature_b.proposal),
            "same_best_prior_object": (
                feature_a.best_prior_object_id is not None
                and feature_a.best_prior_object_id == feature_b.best_prior_object_id
            ),
            "mask_a_class": "background_like"
            if feature_a.is_background_like
            else ("object_like" if feature_a.is_object_like else "uncertain"),
            "mask_b_class": "background_like"
            if feature_b.is_background_like
            else ("object_like" if feature_b.is_object_like else "uncertain"),
        }
        return RuntimeMergeDecision(
            mask_id_a=feature_a.proposal_id,
            mask_id_b=feature_b.proposal_id,
            adjacency_score=float(adjacency_score),
            depth_continuity_score=float(boundary_depth_continuity),
            plane_similarity_score=float(plane_compatibility),
            bbox_plausibility_score=float(merged_bbox_compactness),
            containment_score=float(containment_score),
            median_depth_gap=float(median_depth_gap),
            boundary_depth_continuity=float(boundary_depth_continuity),
            plane_compatibility=float(plane_compatibility),
            merged_bbox_compactness=float(merged_bbox_compactness),
            base_score=float(base_score),
            whole_prior_score=float(whole_prior_score),
            background_conflict_penalty=float(background_conflict_penalty),
            final_score=float(final_score),
            accepted=accepted,
            accepted_reason=accepted_reason,
            boosted_by_whole_prior=boosted,
            linked_object_id=linked_object_id,
            rejected_due_to_background_conflict=rejected_due_to_background_conflict,
            reason_breakdown=breakdown,
        )

    def _adjacency_score(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        dx = max(float(max(box_a[0], box_b[0]) - min(box_a[2], box_b[2])), 0.0)
        dy = max(float(max(box_a[1], box_b[1]) - min(box_a[3], box_b[3])), 0.0)
        gap = math.hypot(dx, dy)
        return float(max(0.0, 1.0 - gap / max(self.bbox_gap_px, 1.0)))

    def _median_depth_gap(self, depth_a: float, depth_b: float) -> float:
        if not math.isfinite(depth_a) or not math.isfinite(depth_b):
            return float("inf")
        return float(abs(depth_a - depth_b))

    def _depth_gap_score(self, depth_gap: float) -> float:
        if not math.isfinite(depth_gap):
            return 0.0
        return float(max(0.0, 1.0 - depth_gap / max(self.depth_mean_diff_max, 1e-6)))

    def _boundary_depth_continuity_score(
        self,
        feature_a: _MaskFeatures,
        feature_b: _MaskFeatures,
        depth: np.ndarray,
    ) -> float:
        if feature_a.depth_valid_ratio <= 0.05 or feature_b.depth_valid_ratio <= 0.05:
            return 0.0

        mask_a = feature_a.proposal.mask.astype(bool)
        mask_b = feature_b.proposal.mask.astype(bool)
        contact_a = feature_a.boundary_mask & self._dilate_mask(mask_b)
        contact_b = feature_b.boundary_mask & self._dilate_mask(mask_a)

        valid = np.isfinite(depth) & (depth > 0)
        depth_a = depth[contact_a & valid]
        depth_b = depth[contact_b & valid]

        if depth_a.size >= self.min_boundary_depth_points and depth_b.size >= self.min_boundary_depth_points:
            gap = float(abs(np.median(depth_a) - np.median(depth_b)))
            return float(max(0.0, 1.0 - gap / max(self.depth_mean_diff_max, 1e-6)))

        # Fallback when two masks are near but not touching at pixel-level boundary.
        fallback_gap = self._median_depth_gap(feature_a.median_depth, feature_b.median_depth)
        return float(0.60 * self._depth_gap_score(fallback_gap))

    def _plane_compatibility_score(
        self,
        feature_a: _MaskFeatures,
        feature_b: _MaskFeatures,
    ) -> float:
        if feature_a.plane_coeffs is None or feature_b.plane_coeffs is None:
            return 0.0

        slope_diff = float(np.linalg.norm(feature_a.plane_coeffs[:2] - feature_b.plane_coeffs[:2]))
        slope_score = max(0.0, 1.0 - slope_diff / max(self.plane_similarity_diff_max, 1e-6))

        offset_diff = abs(float(feature_a.plane_coeffs[2] - feature_b.plane_coeffs[2]))
        offset_score = max(0.0, 1.0 - offset_diff / max(self.depth_mean_diff_max, 1e-6))

        if math.isfinite(feature_a.planar_fit_residual) and math.isfinite(feature_b.planar_fit_residual):
            residual_gap = abs(feature_a.planar_fit_residual - feature_b.planar_fit_residual)
            residual_score = max(
                0.0,
                1.0 - residual_gap / max(self.background_planar_residual_ref, 1e-6),
            )
        else:
            residual_score = 0.0

        return float(np.clip(0.55 * slope_score + 0.25 * offset_score + 0.20 * residual_score, 0.0, 1.0))

    def _bbox_plausibility_score(self, feature_a: _MaskFeatures, feature_b: _MaskFeatures) -> float:
        union_bbox = self._union_bbox(feature_a.bbox_xyxy, feature_b.bbox_xyxy)
        union_bbox_area = max(1.0, self._bbox_area(union_bbox))
        union_area_est = feature_a.proposal.area + feature_b.proposal.area
        fill_ratio = min(1.0, union_area_est / union_bbox_area)
        return float(min(1.0, fill_ratio / max(self.bbox_fill_ratio_ref, 1e-6)))

    def _containment_score(self, feature_a: _MaskFeatures, feature_b: _MaskFeatures) -> float:
        inter = self._bbox_intersection_area(feature_a.bbox_xyxy, feature_b.bbox_xyxy)
        min_area = min(feature_a.bbox_area, feature_b.bbox_area)
        containment = inter / max(min_area, 1.0)
        if containment > 0.90 and abs(feature_a.mean_depth - feature_b.mean_depth) > self.depth_mean_diff_max * 0.5:
            return 0.0
        return 1.0

    def _whole_prior_score(
        self,
        feature_a: _MaskFeatures,
        feature_b: _MaskFeatures,
        evidences: List[WholeObjectEvidence],
    ) -> tuple[float, int | None]:
        best_score = 0.0
        best_object_id = None
        for evidence in evidences:
            if evidence.object_id is None:
                continue
            fit_a = feature_a.prior_fit_by_object.get(evidence.object_id, 0.0)
            fit_b = feature_b.prior_fit_by_object.get(evidence.object_id, 0.0)
            shared_fit = min(fit_a, fit_b)
            score = shared_fit * evidence.whole_observation_score
            if score > best_score:
                best_score = score
                best_object_id = evidence.object_id
        return float(best_score), best_object_id

    def _background_conflict_penalty(
        self,
        feature_a: _MaskFeatures,
        feature_b: _MaskFeatures,
        adjacency_score: float,
        merged_bbox_compactness: float,
        plane_compatibility: float,
        median_depth_gap: float,
    ) -> float:
        one_bg = feature_a.is_background_like or feature_b.is_background_like
        both_bg = feature_a.is_background_like and feature_b.is_background_like
        background_strength = max(feature_a.backgroundness_score, feature_b.backgroundness_score)

        union_bbox = self._union_bbox(feature_a.bbox_xyxy, feature_b.bbox_xyxy)
        image_h, image_w = feature_a.proposal.mask.shape
        union_extent_ratio = self._bbox_area(union_bbox) / max(1.0, float(image_h * image_w))

        plane_low_residual = 0.0
        for residual in (feature_a.planar_fit_residual, feature_b.planar_fit_residual):
            if math.isfinite(residual):
                plane_low_residual = max(
                    plane_low_residual,
                    max(0.0, 1.0 - residual / max(self.background_planar_residual_ref, 1e-6)),
                )
        scene_plane_like = plane_compatibility * plane_low_residual

        small_area_ratio = min(feature_a.proposal.area, feature_b.proposal.area) / max(
            feature_a.proposal.area,
            feature_b.proposal.area,
            1,
        )
        swallow_risk = (
            small_area_ratio <= self.small_swallow_area_ratio
            and adjacency_score >= self.edge_admission_min_adjacency
            and merged_bbox_compactness <= 0.78
            and (one_bg or background_strength >= 0.60)
        )

        weak_closedness = 1.0 - min(feature_a.local_closedness, feature_b.local_closedness)
        if math.isfinite(median_depth_gap):
            depth_disagreement = min(1.0, median_depth_gap / max(self.depth_mean_diff_max, 1e-6))
        else:
            depth_disagreement = 1.0

        penalty = (
            0.30 * background_strength
            + 0.20 * (1.0 if (one_bg and not both_bg) else 0.0)
            + 0.20 * min(1.0, 2.0 * union_extent_ratio * scene_plane_like)
            + 0.15 * (1.0 if swallow_risk else 0.0)
            + 0.10 * weak_closedness
            + 0.05 * depth_disagreement
        )

        if both_bg:
            penalty = max(penalty, 0.80)
        elif one_bg and (feature_a.is_object_like or feature_b.is_object_like):
            penalty = max(penalty, 0.68)

        if feature_a.is_object_like and feature_b.is_object_like and not one_bg:
            penalty *= 0.19

        return float(np.clip(penalty, 0.0, 1.0))

    def _small_object_protection_score(
        self,
        smaller_feature: _MaskFeatures,
        larger_feature: _MaskFeatures,
        merged_bbox_compactness: float,
        median_depth_gap: float,
        boundary_depth_continuity: float,
        linked_object_id: int | None,
    ) -> float:
        if smaller_feature.proposal.area > self.small_object_area_threshold:
            return 0.0

        area_ratio = smaller_feature.proposal.area / max(1, larger_feature.proposal.area)
        if (
            smaller_feature.is_object_like
            and larger_feature.is_object_like
            and area_ratio >= 0.55
            and math.isfinite(smaller_feature.median_depth)
            and math.isfinite(larger_feature.median_depth)
            and abs(smaller_feature.median_depth - larger_feature.median_depth) <= self.depth_mean_diff_max * 0.5
        ):
            return 0.0

        contour_independence = smaller_feature.local_closedness
        depth_gap = 1.0 if median_depth_gap >= self.small_object_depth_gap_min else 0.0
        bbox_degrades = (
            1.0
            if merged_bbox_compactness + self.small_object_bbox_plausibility_margin
            < max(smaller_feature.fill_ratio, larger_feature.fill_ratio)
            else 0.0
        )
        background_pressure = larger_feature.backgroundness_score
        historical_self = (
            1.0
            if (
                smaller_feature.best_prior_object_id is not None
                and smaller_feature.best_prior_score >= 0.65
                and smaller_feature.best_prior_object_id != linked_object_id
            )
            else 0.0
        )

        score = (
            0.30 * contour_independence
            + 0.25 * depth_gap
            + 0.20 * bbox_degrades
            + 0.15 * background_pressure
            + 0.10 * historical_self
        )

        if boundary_depth_continuity > 0.85 and smaller_feature.is_object_like and larger_feature.is_object_like:
            score *= 0.70
        return float(np.clip(score, 0.0, 1.0))

    def _build_groups(
        self,
        features: List[_MaskFeatures],
        decisions: List[RuntimeMergeDecision],
    ) -> tuple[List[RuntimeMaskGroup], Dict[int, int], Dict[int, int | None]]:
        parent = list(range(len(features)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        feature_index = {feature.proposal_id: idx for idx, feature in enumerate(features)}
        for decision in decisions:
            if not decision.accepted:
                continue
            union(feature_index[decision.mask_id_a], feature_index[decision.mask_id_b])

        components: Dict[int, List[_MaskFeatures]] = {}
        for idx, feature in enumerate(features):
            components.setdefault(find(idx), []).append(feature)

        groups: List[RuntimeMaskGroup] = []
        raw_to_group: Dict[int, int] = {}
        group_to_linked_object: Dict[int, int | None] = {}
        for group_id, members in enumerate(components.values()):
            merged_mask = np.zeros_like(members[0].proposal.mask, dtype=bool)
            confidences = []
            object_vote: Dict[int, float] = {}
            whole_prior_used = False

            member_ids = [member.proposal_id for member in members]
            relevant_decisions = [
                decision
                for decision in decisions
                if decision.mask_id_a in member_ids and decision.mask_id_b in member_ids and decision.accepted
            ]
            for member in members:
                merged_mask |= member.proposal.mask
                confidences.append(float(member.proposal.confidence))
                if member.best_prior_object_id is not None and member.best_prior_score > 0.0:
                    object_vote[member.best_prior_object_id] = object_vote.get(member.best_prior_object_id, 0.0) + 0.5 * member.best_prior_score
            for decision in relevant_decisions:
                if decision.linked_object_id is not None:
                    object_vote[decision.linked_object_id] = object_vote.get(decision.linked_object_id, 0.0) + decision.final_score
                whole_prior_used = whole_prior_used or decision.boosted_by_whole_prior

            linked_object_id = max(object_vote, key=object_vote.get) if object_vote else None
            ys, xs = np.nonzero(merged_mask)
            if xs.size == 0 or ys.size == 0:
                bbox = members[0].bbox_xyxy.copy()
            else:
                bbox = np.array(
                    [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                    dtype=np.float32,
                )
            area = int(merged_mask.sum())
            confidence = float(np.mean(confidences)) if confidences else 0.0
            merge_reason = "single_raw_mask" if len(members) == 1 else "connected_component_merge"
            if whole_prior_used:
                merge_reason += "+whole_prior"

            groups.append(
                RuntimeMaskGroup(
                    group_id=group_id,
                    member_mask_ids=member_ids,
                    merged_mask=merged_mask,
                    merged_bbox_xyxy=bbox,
                    area=area,
                    confidence=confidence,
                    linked_object_id=linked_object_id,
                    whole_prior_used=whole_prior_used,
                    merge_reason=merge_reason,
                    merge_reason_summary={
                        "member_count": len(member_ids),
                        "accepted_internal_edge_count": len(relevant_decisions),
                        "whole_prior_edge_count": sum(
                            1 for decision in relevant_decisions if decision.boosted_by_whole_prior
                        ),
                        "cross_class_conflict_blocked_edges": sum(
                            1
                            for decision in decisions
                            if (
                                decision.accepted_reason == "cross_class_anchor_conflict"
                                and (
                                    decision.mask_id_a in member_ids
                                    or decision.mask_id_b in member_ids
                                )
                            )
                        ),
                        "background_conflict_blocked_edges": sum(
                            1
                            for decision in decisions
                            if (
                                decision.rejected_due_to_background_conflict
                                and (
                                    decision.mask_id_a in member_ids
                                    or decision.mask_id_b in member_ids
                                )
                            )
                        ),
                    },
                )
            )
            group_to_linked_object[group_id] = linked_object_id
            for member in members:
                raw_to_group[member.proposal_id] = group_id
        return groups, raw_to_group, group_to_linked_object

    def _anchor_labels_conflict(self, proposal_a: Proposal2D, proposal_b: Proposal2D) -> bool:
        if not self.semantic_class_merge_gate_enabled:
            return False
        label_a = self._proposal_anchor_label(proposal_a)
        label_b = self._proposal_anchor_label(proposal_b)
        return bool(label_a and label_b and label_a != label_b)

    def _missing_required_anchor_label(self, proposal_a: Proposal2D, proposal_b: Proposal2D) -> bool:
        if not self.require_same_anchor_label_for_merge:
            return False
        if self._both_semantic_blocked_residuals(proposal_a, proposal_b):
            return False
        label_a = self._proposal_anchor_label(proposal_a)
        label_b = self._proposal_anchor_label(proposal_b)
        return bool(not label_a or not label_b)

    def _required_anchor_labels_mismatch(self, proposal_a: Proposal2D, proposal_b: Proposal2D) -> bool:
        if not self.require_same_anchor_label_for_merge:
            return False
        label_a = self._proposal_anchor_label(proposal_a)
        label_b = self._proposal_anchor_label(proposal_b)
        return bool(label_a and label_b and label_a != label_b)

    def _required_anchor_identity_mismatch(self, proposal_a: Proposal2D, proposal_b: Proposal2D) -> bool:
        if not self.require_same_anchor_label_for_merge:
            return False
        label_a = self._proposal_anchor_label(proposal_a)
        label_b = self._proposal_anchor_label(proposal_b)
        anchor_id_a = self._proposal_anchor_id(proposal_a)
        anchor_id_b = self._proposal_anchor_id(proposal_b)
        return bool(label_a and label_b and label_a == label_b and anchor_id_a >= 0 and anchor_id_b >= 0 and anchor_id_a != anchor_id_b)

    def _semantic_blocked_residual_merge(self, proposal_a: Proposal2D, proposal_b: Proposal2D) -> bool:
        blocked_a = self._proposal_is_semantic_blocked_residual(proposal_a)
        blocked_b = self._proposal_is_semantic_blocked_residual(proposal_b)
        return bool(blocked_a != blocked_b)

    def _both_semantic_blocked_residuals(self, proposal_a: Proposal2D, proposal_b: Proposal2D) -> bool:
        return bool(
            self._proposal_is_semantic_blocked_residual(proposal_a)
            and self._proposal_is_semantic_blocked_residual(proposal_b)
        )

    @staticmethod
    def _proposal_is_semantic_blocked_residual(proposal: Proposal2D) -> bool:
        metadata = proposal.metadata
        label_strength = str(metadata.get("anchor_label_strength", "")).strip().lower()
        residual_policy = str(metadata.get("residual_semantic_policy", "")).strip().lower()
        relation = str(metadata.get("mask_anchor_relation", "")).strip().lower()
        return bool(
            metadata.get("semantic_commit_allowed") is False
            or residual_policy == "unknown"
            or (label_strength == "none" and relation == "contained_residual")
        )

    @staticmethod
    def _proposal_anchor_label(proposal: Proposal2D) -> str:
        return str(proposal.metadata.get("anchor_class_name", "")).strip()

    @staticmethod
    def _proposal_anchor_id(proposal: Proposal2D) -> int:
        try:
            return int(proposal.metadata.get("anchor_id", -1))
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _anchor_group_metadata(
        members: list[_MaskFeatures],
        decisions: list[RuntimeMergeDecision],
    ) -> dict[str, Any]:
        label_votes: dict[str, float] = {}
        anchor_ids: list[int] = []
        anchor_hit_count = 0
        best_label = ""
        best_confidence = 0.0
        best_anchor_id = -1
        force_object_candidate = False
        has_semantic_blocked_residual = False
        for member in members:
            metadata = member.proposal.metadata
            has_semantic_blocked_residual = (
                has_semantic_blocked_residual
                or RuntimeVisModule._proposal_is_semantic_blocked_residual(member.proposal)
            )
            label = str(metadata.get("anchor_class_name", "")).strip()
            confidence = float(metadata.get("anchor_confidence", 0.0))
            anchor_id = int(metadata.get("anchor_id", -1))
            if label:
                anchor_hit_count += 1
                label_votes[label] = float(label_votes.get(label, 0.0) + (confidence if confidence > 0.0 else 1.0))
                if confidence > best_confidence:
                    best_label = label
                    best_confidence = confidence
                    best_anchor_id = anchor_id
            if anchor_id >= 0:
                anchor_ids.append(anchor_id)
            force_object_candidate = force_object_candidate or bool(metadata.get("force_object_candidate", False))

        anchor_conflict_blocked_edges = int(
            sum(1 for decision in decisions if decision.accepted_reason == "cross_class_anchor_conflict")
        )
        if has_semantic_blocked_residual:
            return {
                "anchor_class_name": "",
                "anchor_confidence": 0.0,
                "anchor_id": -1,
                "anchor_label_votes": {},
                "anchor_hit_count": 0,
                "anchor_ids": [],
                "anchor_conflict_blocked_edges": anchor_conflict_blocked_edges,
                "force_object_candidate": bool(force_object_candidate),
                "anchor_label_strength": "none",
                "anchor_keepalive": False,
                "semantic_commit_allowed": False,
                "residual_semantic_policy": "unknown",
            }

        return {
            "anchor_class_name": best_label,
            "anchor_confidence": float(best_confidence),
            "anchor_id": int(best_anchor_id),
            "anchor_label_votes": dict(sorted(label_votes.items())),
            "anchor_hit_count": int(anchor_hit_count),
            "anchor_ids": sorted(set(anchor_ids)),
            "anchor_conflict_blocked_edges": anchor_conflict_blocked_edges,
            "force_object_candidate": bool(force_object_candidate),
        }

    @staticmethod
    def _group_backend_name(
        group: RuntimeMaskGroup,
        feature_lookup: Dict[int, _MaskFeatures],
    ) -> str:
        backend_names = {
            feature_lookup[mask_id].proposal.backend_name
            for mask_id in group.member_mask_ids
            if mask_id in feature_lookup
        }
        if not backend_names:
            return "unknown"
        if len(backend_names) == 1:
            return next(iter(backend_names))
        return "mixed"

    @staticmethod
    def _bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
        if bbox_min.shape[0] != 3 or bbox_max.shape[0] != 3:
            return np.empty((0, 3), dtype=np.float32)
        x0, y0, z0 = bbox_min.tolist()
        x1, y1, z1 = bbox_max.tolist()
        corners = np.array(
            [
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1],
            ],
            dtype=np.float32,
        )
        return corners

    @staticmethod
    def _expand_bbox(bbox: np.ndarray, scale: float) -> np.ndarray:
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        half_w = (bbox[2] - bbox[0]) * 0.5 * scale
        half_h = (bbox[3] - bbox[1]) * 0.5 * scale
        return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32)

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    @staticmethod
    def _union_bbox(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
        return np.array(
            [
                min(float(box_a[0]), float(box_b[0])),
                min(float(box_a[1]), float(box_b[1])),
                max(float(box_a[2]), float(box_b[2])),
                max(float(box_a[3]), float(box_b[3])),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _bbox_intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _extract_boundary(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        up = padded[:-2, 1:-1].astype(bool)
        down = padded[2:, 1:-1].astype(bool)
        left = padded[1:-1, :-2].astype(bool)
        right = padded[1:-1, 2:].astype(bool)
        interior = mask & up & down & left & right
        return mask & (~interior)

    @staticmethod
    def _dilate_mask(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        return (center | up | down | left | right).astype(bool)

    @staticmethod
    def _border_touch_ratio(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        return float((mask & border).sum() / max(int(mask.sum()), 1))

    @staticmethod
    def _normalize_minmax(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    @staticmethod
    def _bbox_aspect_ratio(bbox: np.ndarray) -> float:
        width = max(1e-6, float(bbox[2] - bbox[0]))
        height = max(1e-6, float(bbox[3] - bbox[1]))
        ratio = max(width / height, height / width)
        return float(ratio)
