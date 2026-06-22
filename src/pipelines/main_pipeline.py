"""Main pipeline: wires all modules together in sequence.

Processes one frame at a time through the full OVIOVO pipeline:

Layer 1: SAM2 proposal frontend (raw proposals)
Layer 2: Depth-aware refinement -> RefinedProposal2D
         RuntimeVis (grouping/cleanup only — Rule A: no instance assignment)
Layer 3: Patch lifting -> Patch3D
Layer 4: Global TSDF instance substrate (spatial voting + support stabilization)
Layer 5: Per-instance object pool geometry (`local_pcd`) for object-level export
Layer 6: Active set / local status checks
Layer 7: Semantic layer (post-stabilization only)

Critical rules enforced:
  Rule A: runtime_vis does NOT do final instance assignment
  Rule B: No hard whole-object labels, only soft evidence
  Rule C: Semantics do not enter low-level association
  Rule D: Clear distinction between raw/refined/patch/TSDF backbone/local_pcd pool/semantic
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np

from src.core.data_structures import (
    BackgroundMap,
    CameraIntrinsics,
    Frame,
    Proposal2D,
    SystemState,
    TSDFInstanceVolume,
)
from src.modules.frame_input import FrameInputModule
from src.modules.proposal import ProposalModule
from src.modules.object_anchor import ObjectAnchorModule
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule
from src.modules.runtime_vis import RuntimeVisModule, RuntimeVisOutput
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.patch_lifting import PatchLiftingModule
from src.modules.active_set import ActiveSetModule
from src.modules.bg_obj_split import BgObjSplitModule
from src.modules.association import AssociationModule
from src.modules.async_refinement import AsyncRefinementModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.semantic_memory import SemanticMemoryModule
from src.modules.dense_surface import DenseSurfaceModule
from src.modules.structural_overlay import StructuralOverlayModule
from src.modules.map_tiering import MapTieringModule
from src.modules.background_update import BackgroundUpdateModule
from src.modules.dynamic_maintenance import DynamicMaintenanceModule
from src.pipelines.proposal_bundle import FrameProposalBundle
from src.pipelines.yoloworld_sam_frontend import build_anchor_first_sam_bundle, build_yoloworld_sam_bundle
from src.utils.logging import setup_logging

logger = logging.getLogger("oviovo.pipeline")


class Pipeline:
    """Central pipeline runner for the OVIOVO mapping system."""

    _ASYNC_REFINEMENT_SAM_BACKENDS = {"sam2", "precomputed", "esam", "entitysam"}

    _TOP_LEVEL_STAGE_NAMES = {
        "frame_input",
        "proposal_generation",
        "structural_overlay",
        "runtime_vis",
        "depth_refinement",
        "patch_lifting",
        "active_set",
        "bg_obj_split",
        "association",
        "object_update",
        "async_refinement",
        "semantic_memory",
        "dense_surface",
        "map_tiering",
        "background_update",
        "dynamic_maintenance",
    }

    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        self.config = self._load_config(config_path)

        # Setup logging
        log_cfg = self.config.get("logging", {})
        setup_logging(
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file", ""),
        )

        # Initialize system state with global TSDF instance volume
        bg_cfg = self.config.get("background_update", {})
        tsdf_cfg = self.config.get("tsdf", {})
        self.state = SystemState(
            background=BackgroundMap(voxel_size=bg_cfg.get("voxel_size", 0.05)),
            tsdf_volume=TSDFInstanceVolume(
                voxel_size=tsdf_cfg.get("voxel_size", 0.05),
                truncation_distance=tsdf_cfg.get("truncation_distance", 0.15),
                support_increment=tsdf_cfg.get("support_increment", 1.0),
                support_decay=tsdf_cfg.get("support_decay", 0.95),
            ),
        )

        # Initialize all modules
        self.frame_input = FrameInputModule(self.config.get("frame_input", {}))
        self.proposal = ProposalModule(self.config.get("proposal", {}))
        self.object_anchor = ObjectAnchorModule(self.config.get("anchor_frontend", {}))
        self.anchor_guided_sam = AnchorGuidedSAMModule(self.config.get("anchor_guided_sam", {}))
        self.runtime_vis = RuntimeVisModule(self.config.get("runtime_vis", {}))
        self.depth_refinement = DepthRefinementModule(self.config.get("depth_refinement", {}))
        self.patch_lifting = PatchLiftingModule(self.config.get("patch_lifting", {}))
        self.active_set = ActiveSetModule(self.config.get("active_set", {}))
        self.bg_obj_split = BgObjSplitModule(self.config.get("bg_obj_split", {}))
        self.association = AssociationModule(self.config.get("association", {}))
        object_update_cfg = dict(self.config.get("object_update", {}) or {})
        object_update_cfg.setdefault("async_refinement", self.config.get("async_refinement", {}))
        self.object_update = ObjectUpdateModule(object_update_cfg)
        self.async_refinement = AsyncRefinementModule(self.config.get("async_refinement", {}))
        self.async_refinement_backend = None
        self.structural_overlay = StructuralOverlayModule(self.config.get("structural_overlay", {}))
        self.semantic_memory = SemanticMemoryModule(self.config.get("semantic_memory", {}))
        self.dense_surface = DenseSurfaceModule(self.config.get("dual_map", {}))
        self.map_tiering = MapTieringModule(self.config.get("dual_map", {}))
        self.background_update = BackgroundUpdateModule(self.config.get("background_update", {}))
        self.dynamic_maintenance = DynamicMaintenanceModule(self.config.get("dynamic_maintenance", {}))

        self.verbose = self.config.get("pipeline", {}).get("verbose", True)
        pipeline_cfg = self.config.get("pipeline", {})
        self.collect_stage_timings = bool(pipeline_cfg.get("collect_stage_timings", False))
        self.tsdf_feedback_enabled = bool(pipeline_cfg.get("tsdf_feedback_enabled", False))
        self.yoloworld_sam_parallel_frontend_enabled = bool(
            pipeline_cfg.get("yoloworld_sam_parallel_frontend_enabled", False)
        )
        self.yoloworld_sam_balanced_frontend_enabled = bool(
            pipeline_cfg.get("yoloworld_sam_balanced_frontend_enabled", False)
        )
        self.yoloworld_sam_full_frame_interval = max(
            1,
            int(pipeline_cfg.get("yoloworld_sam_full_frame_interval", 1) or 1),
        )
        self.yoloworld_sam_mode = str(pipeline_cfg.get("yoloworld_sam_mode", "anchor_guided_sam") or "")
        self.object_anchor.collect_generation_timings = self.collect_stage_timings
        self._stage_timings: dict[str, float] = {}
        self.last_raw_proposals = []
        self.last_source_proposals = []
        self.last_anchors = []
        self.last_anchor_assignments = []
        self.last_refined_proposals = []
        self.last_patches = []
        self.last_runtime_vis_output: RuntimeVisOutput | None = None
        self.last_association = None
        self.last_frame_debug: Dict[str, Any] = {}
        logger.info("Pipeline initialized with all modules.")

    @staticmethod
    def _frame_refinement_id(frame: Frame) -> int:
        return int(frame.source_frame_id if frame.source_frame_id is not None else frame.frame_id)

    def _stamp_anchor_primary_coarse_metadata(
        self,
        frame: Frame,
        proposals: list[Proposal2D],
        anchor_assignments: list[Any],
    ) -> None:
        anchor_by_proposal_id = {
            int(assignment.proposal_id): int(assignment.anchor_id)
            for assignment in anchor_assignments
            if int(getattr(assignment, "anchor_id", -1)) >= 0
        }
        frame_id = self._frame_refinement_id(frame)
        for proposal in proposals:
            anchor_id = proposal.metadata.get("anchor_id", anchor_by_proposal_id.get(int(proposal.proposal_id)))
            if anchor_id is None:
                continue
            proposal.metadata["observation_layer"] = "coarse"
            proposal.metadata["refinement_key"] = f"{frame_id}:{int(anchor_id)}"

    def _async_refinement_fallback_allowed(self) -> tuple[bool, str]:
        active_backend = str(getattr(self.proposal, "active_backend_name", "") or "")
        return active_backend in self._ASYNC_REFINEMENT_SAM_BACKENDS, active_backend

    def _load_structural_overlay_proposals(
        self,
        frame: Frame,
        proposal_source: str,
        source_proposals: list[Proposal2D],
        *,
        proposal_bundle_supplied: bool = False,
    ) -> tuple[list[Proposal2D], dict[str, Any]]:
        if not getattr(self.structural_overlay, "enabled", False):
            return [], {"proposal_source": "disabled", "skip_reason": "structural_overlay_disabled"}
        if source_proposals:
            return list(source_proposals), {"proposal_source": proposal_source, "skip_reason": ""}
        active_backend = str(getattr(self.proposal, "active_backend_name", "") or "")
        if proposal_bundle_supplied:
            return [], {
                "proposal_source": "unavailable",
                "active_backend": active_backend,
                "skip_reason": "prefetched_bundle_without_source_proposals",
            }
        if proposal_source == "anchor_box_primary" and active_backend == "precomputed":
            proposals = list(self.proposal.process(frame.rgb, frame.depth, frame=frame))
            return proposals, {"proposal_source": "precomputed_overlay_only", "skip_reason": ""}
        return [], {
            "proposal_source": "unavailable",
            "active_backend": active_backend,
            "skip_reason": "structural_overlay_requires_precomputed_sam_proposals",
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML config file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults.")
            return {}
        with open(path) as f:
            return yaml.safe_load(f)

    @contextmanager
    def _timed_stage(self, name: str):
        if not self.collect_stage_timings:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stage_timings[name] = self._stage_timings.get(name, 0.0) + float(
                time.perf_counter() - start
            )

    def _merge_stage_timings(self, timings: dict[str, float] | None) -> None:
        if not self.collect_stage_timings or not timings:
            return
        for name, value in dict(timings).items():
            try:
                timing_value = float(value)
            except (TypeError, ValueError):
                continue
            if timing_value < 0.0 or not np.isfinite(timing_value):
                continue
            timing_name = str(name)
            if timing_name in self._TOP_LEVEL_STAGE_NAMES:
                timing_name = f"nested_{timing_name}"
            self._stage_timings[timing_name] = self._stage_timings.get(timing_name, 0.0) + timing_value

    def _default_anchor_guided_sam_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.anchor_guided_sam, "enabled", False)),
            "anchor_count": 0,
            "source_sam_proposal_count": 0,
            "output_proposal_count": 0,
            "anchored_proposal_count": 0,
            "unknown_residual_count": 0,
            "dropped_proposal_count": 0,
            "matched_sam_proposal_count": 0,
            "mean_sam_candidates_per_anchor": 0.0,
            "semantic_blocked_residual_count": 0,
            "assignment_count": 0,
        }

    def _should_use_parallel_yoloworld_sam_frontend(self) -> bool:
        anchor_backend = str(getattr(self.object_anchor, "active_backend_name", "") or "")
        proposal_backend = str(getattr(self.proposal, "active_backend_name", "") or "")
        anchor_prompt_enabled = bool(
            getattr(getattr(self.proposal, "backend", None), "anchor_prompt_enabled", False)
        )
        common_enabled = bool(
            self.yoloworld_sam_parallel_frontend_enabled
            and self.object_anchor.enabled
            and getattr(self.object_anchor, "anchor_primary_mode", False)
            and anchor_backend == "yoloworld"
            and proposal_backend == "sam2"
        )
        if not common_enabled:
            return False
        if self.yoloworld_sam_mode == "anchor_first_sam":
            return anchor_prompt_enabled
        return bool(getattr(self.anchor_guided_sam, "enabled", False))

    def _should_run_balanced_sam(self, frame: Frame) -> bool:
        if not self.yoloworld_sam_balanced_frontend_enabled:
            return True
        return int(frame.frame_id) % int(self.yoloworld_sam_full_frame_interval) == 0

    def _build_depth_structure_proposals(
        self, frame: Frame, existing_proposals: list[Any], anchors: list[Any],
    ) -> list[Any]:
        """Generate depth-based masks for structure classes (wall/floor/ceiling).

        Creates clean planar masks from depth surface normals for classes
        where SAM2 oversegmentation is counterproductive.
        """
        from src.core.data_structures import Proposal2D

        # Only generate for structure classes that have YOLO detections
        depth_classes = self.object_anchor.depth_structure_classes
        assigned_classes = {str(a.class_name) for a in anchors if a.class_name in depth_classes}
        # Also check existing proposals — don't duplicate if that class already has proposals
        for p in existing_proposals:
            meta = getattr(p, "metadata", {}) or {}
            cls = str(meta.get("anchor_class_name", ""))
            if cls in assigned_classes:
                assigned_classes.discard(cls)

        if not assigned_classes:
            return []

        intrinsics = (frame.intrinsics.fx, frame.intrinsics.fy,
                       frame.intrinsics.cx, frame.intrinsics.cy)
        h, w = frame.depth.shape

        depth_proposals = []
        for anchor in anchors:
            cls = str(anchor.class_name)
            if cls not in assigned_classes:
                continue
            normal_dir = self.object_anchor.depth_structure_normal.get(cls, "horizontal")
            mask = self.object_anchor._generate_depth_mask(
                frame.depth,
                np.asarray(anchor.bbox_xyxy, dtype=np.float32),
                intrinsics, normal_dir,
            )
            if mask is None:
                continue
            area = int(mask.sum())
            if area < self.object_anchor.proposal_min_area:
                continue
            bbox = self.object_anchor._mask_bbox(mask)
            proposal = Proposal2D(
                proposal_id=-(1000000 + len(depth_proposals)),  # negative id = synthetic
                mask=mask, bbox_xyxy=bbox, area=area,
                confidence=float(anchor.confidence),
                backend_name="depth_structure",
                metadata={
                    "source": "depth_structure",
                    "mask_source": "depth_structure",
                    "anchor_id": int(anchor.anchor_id),
                    "anchor_class_name": cls,
                    "anchor_confidence": float(anchor.confidence),
                    "anchor_label_votes": {cls: float(anchor.confidence)},
                    "anchor_label_strength": "strong",
                    "anchor_keepalive": True,
                    "observation_layer": "coarse",
                    "force_object_candidate": True,
                },
            )
            depth_proposals.append(proposal)

        return depth_proposals

    def _build_tsdf_feedback_proposals(self, frame: Frame) -> list[Any]:
        """Path B: project high-confidence TSDF objects into current frame as proposals.

        Well-established objects (high whole_evidence_score) have their 3D
        geometry projected back into 2D, creating recycled proposals that
        don't require SAM2 re-segmentation.
        """
        from src.core.data_structures import Proposal2D
        proposals: list[Any] = []
        cam_pos = frame.pose[:3, 3]
        fx, fy = frame.intrinsics.fx, frame.intrinsics.fy
        cx, cy = frame.intrinsics.cx, frame.intrinsics.cy
        h, w = frame.depth.shape
        R = frame.pose[:3, :3].T  # world→cam rotation
        t = -R @ cam_pos          # world→cam translation

        for obj in self.state.objects.values():
            if obj.state.value in ("removed", "inactive"):
                continue
            we = obj.whole_evidence.whole_evidence_score
            if we < 0.3:
                continue
            # Skip if already well-represented this frame
            if len(obj.local_pcd) < 50:
                continue

            # Check if object is in front of camera
            centroid_cam = R @ obj.centroid + t
            if centroid_cam[2] <= 0.1:  # behind camera or too close
                continue

            # Project downsampled points to 2D
            pts = np.asarray(obj.local_pcd, dtype=np.float32)
            stride = max(1, len(pts) // 300)
            pts_sample = pts[::stride]
            pts_cam = (R @ pts_sample.T).T + t
            z = pts_cam[:, 2]
            valid = z > 0.01
            if valid.sum() < 5:
                continue
            z = z[valid]
            u = (fx * pts_cam[valid, 0] / z + cx).astype(np.int32)
            v = (fy * pts_cam[valid, 1] / z + cy).astype(np.int32)
            in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if in_bounds.sum() < 20:
                continue
            u, v = u[in_bounds], v[in_bounds]

            # Create mask from projected points (dilate to fill gaps)
            mask = np.zeros((h, w), dtype=bool)
            mask[v, u] = True
            # Simple dilation: 3x3 cross
            from scipy import ndimage as _ndi
            mask = _ndi.binary_dilation(mask, iterations=3)

            area = int(mask.sum())
            if area < 100:
                continue

            bbox_ys, bbox_xs = np.where(mask)
            bbox = np.array([bbox_xs.min(), bbox_ys.min(),
                             bbox_xs.max() + 1, bbox_ys.max() + 1], dtype=np.float32)

            label = str(obj.debug.get("export_semantic_label", ""))
            proposals.append(Proposal2D(
                proposal_id=-(2000000 + int(obj.object_id)),
                mask=mask, bbox_xyxy=bbox, area=area,
                confidence=float(we),
                backend_name="tsdf_feedback",
                metadata={
                    "source": "tsdf_feedback",
                    "mask_source": "tsdf_feedback",
                    "anchor_id": int(obj.object_id),
                    "anchor_class_name": label,
                    "anchor_confidence": float(we),
                    "anchor_label_votes": {label: float(we)} if label else {},
                    "anchor_label_strength": "strong",
                    "anchor_keepalive": True,
                    "observation_layer": "coarse",
                    "force_object_candidate": True,
                    "feedback_object_id": int(obj.object_id),
                },
            ))

        return proposals

    def _build_proposal_bundle(self, frame: Frame) -> FrameProposalBundle:
        if self._should_use_parallel_yoloworld_sam_frontend():
            with self._timed_stage("proposal_generation"):
                if self.yoloworld_sam_mode == "anchor_first_sam":
                    return build_anchor_first_sam_bundle(
                        frame=frame,
                        object_anchor=self.object_anchor,
                        proposal=self.proposal,
                        collect_stage_timings=self.collect_stage_timings,
                    )
                return build_yoloworld_sam_bundle(
                    frame=frame,
                    object_anchor=self.object_anchor,
                    proposal=self.proposal,
                    anchor_guided_sam=self.anchor_guided_sam,
                    collect_stage_timings=self.collect_stage_timings,
                    run_sam=self._should_run_balanced_sam(frame),
                )

        anchor_guided_sam_summary = self._default_anchor_guided_sam_summary()
        proposal_source = "proposal_backend"
        source_proposals: list[Proposal2D] = []
        generation_timings: dict[str, float] = {}
        if hasattr(self.proposal, "last_generation_timings"):
            self.proposal.last_generation_timings = {}
        proposal_generation_timing_before = float(self._stage_timings.get("proposal_generation", 0.0) or 0.0)
        with self._timed_stage("proposal_generation"):
            if self.object_anchor.enabled and getattr(self.object_anchor, "anchor_primary_mode", False):
                anchors, anchor_box_proposals, anchor_assignments = self.object_anchor.generate_anchor_box_proposals(
                    frame.rgb, depth=frame.depth,
                    intrinsics=(frame.intrinsics.fx, frame.intrinsics.fy, frame.intrinsics.cx, frame.intrinsics.cy)
                )
                if getattr(self.anchor_guided_sam, "enabled", False):
                    sam_proposals = self.proposal.process_for_anchors(
                        frame.rgb,
                        frame.depth,
                        anchors,
                        frame=frame,
                    )
                    source_proposals = list(sam_proposals)
                    proposals, anchor_assignments, anchor_guided_sam_summary = self.anchor_guided_sam.build_proposals(
                        frame=frame,
                        anchors=list(anchors),
                        sam_proposals=sam_proposals,
                    )
                    proposal_source = "anchor_guided_sam"
                else:
                    proposals = anchor_box_proposals
                    self._stamp_anchor_primary_coarse_metadata(frame, proposals, anchor_assignments)
                    proposal_source = "anchor_box_primary"
            elif self.object_anchor.enabled and self.object_anchor.use_sam_intersection_proposals:
                sam_proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                source_proposals = list(sam_proposals)
                anchors, proposals, anchor_assignments = self.object_anchor.generate_proposals(frame.rgb, sam_proposals)
                vote_policies = {"semantic_vote", "vote_only", "sam_mask_semantic_vote"}
                assignment_policy = str(getattr(self.object_anchor, "assignment_policy", "") or "")
                proposal_source = "anchor_sam_union"
                if assignment_policy in vote_policies or (
                    proposals
                    and all(
                        proposal.backend_name == "sam2_anchor_vote"
                        or proposal.metadata.get("source") == "sam2_anchor_vote"
                        for proposal in proposals
                    )
                ):
                    proposal_source = "sam2_anchor_vote"
            else:
                proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                source_proposals = list(proposals)
                if self.object_anchor.enabled:
                    anchors, anchor_assignments = self.object_anchor.process(frame.rgb, proposals)
                else:
                    anchors, anchor_assignments = [], []
            generation_timings = dict(getattr(self.object_anchor, "last_generation_timings", {}) or {})
            for timing_name, timing_value in dict(getattr(self.proposal, "last_generation_timings", {}) or {}).items():
                generation_timings[str(timing_name)] = timing_value
        if self.collect_stage_timings:
            proposal_generation_timing_after = float(
                self._stage_timings.get("proposal_generation", proposal_generation_timing_before)
                or proposal_generation_timing_before
            )
            proposal_generation_elapsed = max(
                0.0,
                proposal_generation_timing_after - proposal_generation_timing_before,
            )
            generation_timings["proposal_generation"] = proposal_generation_elapsed
        return FrameProposalBundle(
            frame_id=int(frame.frame_id),
            source_frame_id=frame.source_frame_id,
            source_proposals=list(source_proposals),
            raw_proposals=list(proposals),
            anchors=list(anchors),
            anchor_assignments=list(anchor_assignments),
            proposal_source=proposal_source,
            anchor_guided_sam_summary=dict(anchor_guided_sam_summary),
            generation_timings=generation_timings,
            actual_backend=str(getattr(self.proposal, "active_backend_name", "")),
        )

    def _consume_proposal_bundle(
        self,
        frame: Frame,
        bundle: FrameProposalBundle,
        *,
        merge_top_level_proposal_generation: bool = False,
    ) -> tuple[list[Proposal2D], list[Proposal2D], list[Any], list[Any], str, dict[str, Any]]:
        if int(bundle.frame_id) != int(frame.frame_id):
            raise ValueError(f"Proposal bundle frame_id mismatch: expected {frame.frame_id}, got {bundle.frame_id}")
        if frame.source_frame_id is not None:
            if bundle.source_frame_id is None or int(bundle.source_frame_id) != int(frame.source_frame_id):
                raise ValueError(
                    "Proposal bundle source_frame_id mismatch: "
                    f"expected {frame.source_frame_id}, got {bundle.source_frame_id}"
                )
        self.last_raw_proposals = list(bundle.raw_proposals)
        self.last_source_proposals = list(bundle.source_proposals)
        self.last_anchors = list(bundle.anchors)
        self.last_anchor_assignments = list(bundle.anchor_assignments)
        # Generation timings are produced by the frontend bundle producer; prefetch
        # consumption is not timed as proposal generation here.
        generation_timings = dict(bundle.generation_timings)
        proposal_generation_timing = generation_timings.pop("proposal_generation", None)
        if merge_top_level_proposal_generation and self.collect_stage_timings:
            try:
                timing_value = float(proposal_generation_timing)
            except (TypeError, ValueError):
                timing_value = 0.0
            if timing_value >= 0.0 and np.isfinite(timing_value):
                self._stage_timings["proposal_generation"] = (
                    self._stage_timings.get("proposal_generation", 0.0) + timing_value
                )
        self._merge_stage_timings(generation_timings)
        return (
            self.last_raw_proposals,
            self.last_source_proposals,
            self.last_anchors,
            self.last_anchor_assignments,
            str(bundle.proposal_source),
            dict(bundle.anchor_guided_sam_summary),
        )

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
        timestamp: float = 0.0,
        source_frame_id: int | None = None,
        proposal_bundle: FrameProposalBundle | None = None,
    ) -> SystemState:
        """Run the full pipeline on one frame.

        Flow (aligned with layered roadmap):
          1. Frame Input
          2. Proposal Generation (SAM2 raw proposals — Layer 1)
          3. RuntimeVis (grouping/cleanup only — NOT instance assignment, Rule A)
          4. Depth Refinement -> RefinedProposal2D (Layer 2)
          5. Patch Lifting -> Patch3D (Layer 3)
          6. Active Set derivation (Layer 6 — local reasoning only)
          7. BG/Object Split (using soft scores)
          8. Patch-to-Instance Association (TSDF spatial voting — Layer 4)
          9. Object Update (TSDF integration + local_pcd pool maintenance — Layers 4+5)
         10. Semantic Memory (post-stabilization only — Layer 7)
         11. Background Update
         12. Dynamic Maintenance

        Args:
            rgb: (H, W, 3) uint8 image.
            depth: (H, W) float32 depth in meters.
            pose: (4, 4) camera-to-world transform.
            intrinsics: Camera intrinsic parameters.
            timestamp: Frame timestamp.
            source_frame_id: Optional source/dataset frame index for cache alignment.
            proposal_bundle: Optional precomputed frontend outputs for this frame.

        Returns:
            Updated SystemState.
        """
        self._stage_timings = {}

        # Step 1: Frame Input
        with self._timed_stage("frame_input"):
            frame: Frame = self.frame_input.process(
                rgb,
                depth,
                pose,
                intrinsics,
                timestamp,
                source_frame_id=source_frame_id,
            )
            if self.verbose:
                logger.info(f"=== Frame {frame.frame_id} ===")

        # Step 2: Proposal Generation (Layer 1 — anchor-first when enabled)
        frontend_source = "inline"
        frontend_actual_backend = str(getattr(self.proposal, "active_backend_name", ""))
        proposal_bundle_supplied = proposal_bundle is not None
        if proposal_bundle is None:
            bundle = self._build_proposal_bundle(frame)
        else:
            bundle = proposal_bundle
            frontend_source = "prefetch"
            frontend_actual_backend = str(bundle.actual_backend)
        proposals, source_proposals, anchors, anchor_assignments, proposal_source, anchor_guided_sam_summary = (
            self._consume_proposal_bundle(
                frame,
                bundle,
                merge_top_level_proposal_generation=proposal_bundle_supplied,
            )
        )
        if self.verbose:
            logger.info("  Proposals (%s): %d", proposal_source, len(proposals))
            if self.object_anchor.enabled:
                logger.info(f"  Anchors: {len(anchors)}")

        # Depth-based structure proposals (wall/floor/ceiling from depth, bypass SAM2)
        depth_structure_summary: dict[str, Any] = {"enabled": False, "count": 0}
        if self.object_anchor.depth_structure_enabled:
            with self._timed_stage("depth_structure"):
                depth_proposals = self._build_depth_structure_proposals(frame, proposals, anchors)
                if depth_proposals:
                    proposals = list(proposals) + list(depth_proposals)
                    depth_structure_summary = {"enabled": True, "count": len(depth_proposals)}

        # Path B: TSDF feedback — project established objects back as proposals
        tsdf_feedback_summary: dict[str, Any] = {"enabled": False, "count": 0}
        if self.tsdf_feedback_enabled:
            with self._timed_stage("tsdf_feedback"):
                feedback_proposals = self._build_tsdf_feedback_proposals(frame)
                if feedback_proposals:
                    proposals = list(proposals) + list(feedback_proposals)
                    tsdf_feedback_summary = {"enabled": True, "count": len(feedback_proposals)}

        structural_overlay_summary: dict[str, Any] = {"enabled": bool(self.structural_overlay.enabled)}
        with self._timed_stage("structural_overlay"):
            overlay_source_proposals, overlay_source_debug = self._load_structural_overlay_proposals(
                frame=frame,
                proposal_source=proposal_source,
                source_proposals=source_proposals,
                proposal_bundle_supplied=proposal_bundle_supplied,
            )
            if overlay_source_proposals:
                source_proposals = list(overlay_source_proposals)
                self.last_source_proposals = source_proposals
            structural_overlay_summary = self.structural_overlay.process(
                frame=frame,
                anchors=list(anchors),
                proposals=list(overlay_source_proposals),
                overlay_map=self.state.structural_overlay_map,
            )
            module_skip_reason = str(structural_overlay_summary.get("skip_reason", "") or "")
            structural_overlay_summary.update(overlay_source_debug)
            loader_skip_reason = str(overlay_source_debug.get("skip_reason", "") or "")
            if module_skip_reason and not loader_skip_reason:
                structural_overlay_summary["skip_reason"] = module_skip_reason

        # Step 3: RuntimeVis — grouping/cleanup only (Rule A)
        # RuntimeVis may group fragmented masks but does NOT do final instance assignment.
        with self._timed_stage("runtime_vis"):
            runtime_vis_output = self.runtime_vis.process(frame, proposals, self.state)
            self.last_runtime_vis_output = runtime_vis_output
            merged_proposals = runtime_vis_output.merged_proposals
            if proposal_source in {"anchor_box_primary", "anchor_guided_sam", "anchor_prompted_sam"}:
                source_by_id = {int(proposal.proposal_id): proposal for proposal in proposals}
                for merged_proposal in merged_proposals:
                    member_ids = list(merged_proposal.metadata.get("member_mask_ids", []) or [])
                    if not member_ids:
                        member_ids = [int(merged_proposal.proposal_id)]
                    if len(member_ids) != 1:
                        continue
                    source = source_by_id.get(int(member_ids[0]))
                    if source is None:
                        continue
                    for key in (
                        "observation_layer",
                        "refinement_key",
                        "anchor_keepalive",
                        "anchor_label_strength",
                        "semantic_commit_allowed",
                        "residual_semantic_policy",
                        "mask_anchor_relation",
                        "source_raw_proposal_ids",
                    ):
                        if key in source.metadata and not merged_proposal.metadata.get(key):
                            merged_proposal.metadata[key] = source.metadata[key]
                    if bool(source.metadata.get("anchor_keepalive", False)):
                        merged_proposal.metadata["force_object_candidate"] = True
            if self.verbose:
                logger.info(
                    "  RuntimeVis: raw=%d merged=%d accepted_edges=%d",
                    len(proposals),
                    len(merged_proposals),
                    runtime_vis_output.group_stats.get("accepted_edge_count", 0),
                )

        # Step 4: Depth Refinement -> RefinedProposal2D (Layer 2)
        with self._timed_stage("depth_refinement"):
            refined = self.depth_refinement.process(frame.depth, merged_proposals)
            if proposal_source in {"anchor_box_primary", "anchor_guided_sam", "anchor_prompted_sam"}:
                merged_by_id = {int(proposal.proposal_id): proposal for proposal in merged_proposals}
                for refined_proposal in refined:
                    source_id = int(refined_proposal.metadata.get("source_raw_proposal_id", refined_proposal.proposal_id))
                    source = merged_by_id.get(source_id)
                    if source is not None and bool(source.metadata.get("force_object_candidate", False)):
                        refined_proposal.metadata["force_object_candidate"] = True
            self.last_refined_proposals = refined
            if self.verbose:
                logger.info(f"  Refined proposals: {len(refined)}")
        depth_refinement_debug = {
            "parallel_used": bool(getattr(self.depth_refinement, "last_parallel_used", False)),
            "parallel_worker_count": int(getattr(self.depth_refinement, "last_parallel_worker_count", 1)),
        }

        # Step 5: Patch Lifting (Layer 3 — 2D -> 3D)
        with self._timed_stage("patch_lifting"):
            patches = self.patch_lifting.process(
                refined, frame.depth, frame.pose, frame.intrinsics,
                frame_id=frame.frame_id, timestamp=frame.timestamp,
            )
            self.last_patches = patches
            if self.verbose:
                logger.info(f"  3D patches: {len(patches)}")

        # Step 6: Active Set derivation (Layer 6 — local reasoning only)
        with self._timed_stage("active_set"):
            self.state.active_set = self.active_set.process(
                self.state, frame.pose, frame.intrinsics,
            )
            if self.verbose:
                logger.info(f"  Active set: {len(self.state.active_set.all_candidate_ids)} candidates")

        # Step 7: Background / Object Split (using soft scores)
        with self._timed_stage("bg_obj_split"):
            bg_patches, obj_patches, amb_patches = self.bg_obj_split.process(
                patches, self.state.background, self.state.objects,
            )
            if self.verbose:
                logger.info(f"  Split: bg={len(bg_patches)}, obj={len(obj_patches)}, amb={len(amb_patches)}")

        # Step 8: Object Association (Layer 4 — TSDF spatial voting, geometry-first)
        with self._timed_stage("association"):
            association_patches = [*obj_patches, *amb_patches]
            association = self.association.process(
                association_patches, self.state.objects, self.state.tsdf_volume, active_set=self.state.active_set,
            )
            self.last_association = association
            if self.verbose:
                logger.info(
                    f"  Association: {len(association.matched)} matched, "
                    f"{len(association.new_object_patches)} new"
                )

        # Step 9: Object Update (Layers 4+5 — TSDF integration + local_pcd)
        with self._timed_stage("object_update"):
            self.state = self.object_update.process(
                association,
                association_patches,
                self.state,
                background_patches=bg_patches,
                current_depth=frame.depth,
                current_pose=frame.pose,
                current_intrinsics=frame.intrinsics,
            )
            structural_reject_patches = list(self.object_update.last_structural_reject_patches)
            if self.verbose:
                logger.info(f"  Objects: {len(self.state.objects)}")
        self._merge_stage_timings(getattr(self.object_update, "last_stage_timings", {}) or {})

        # Update active set with new object candidates
        for oid, obj in self.state.objects.items():
            if obj.creation_frame == frame.frame_id:
                self.state.active_set.new_object_candidate_ids.add(oid)
        self.state.active_set = self.active_set.cap_active_set(self.state.active_set)

        async_refinement_summary: dict[str, Any] = {
            "enabled": bool(self.async_refinement.enabled),
            "sam_proposal_count": 0,
            "fine_proposal_count": 0,
            "replaced_observation_count": 0,
            "inserted_observation_count": 0,
            "updated_object_count": 0,
            "updated_object_ids": [],
        }
        if self.async_refinement.enabled:
            with self._timed_stage("async_refinement"):
                if self.async_refinement_backend is not None:
                    refinement_sam_proposals = list(self.async_refinement_backend(frame))
                    fallback_debug: dict[str, Any] = {"proposal_source": "injected_async_refinement_backend"}
                else:
                    fallback_allowed, active_backend = self._async_refinement_fallback_allowed()
                    fallback_debug = {"active_backend": active_backend}
                    if proposal_bundle_supplied:
                        refinement_sam_proposals = list(source_proposals)
                        fallback_debug.update(
                            {
                                "proposal_source": "prefetched_source_proposals",
                                "skip_reason": "prefetched_bundle_without_source_proposals"
                                if not refinement_sam_proposals
                                else "",
                            }
                        )
                    elif fallback_allowed:
                        refinement_sam_proposals = list(self.proposal.process(frame.rgb, frame.depth, frame=frame))
                        fallback_debug["proposal_source"] = "proposal_backend"
                    else:
                        refinement_sam_proposals = []
                        fallback_debug.update(
                            {
                                "proposal_source": "disabled_non_sam_backend",
                                "skip_reason": "async_refinement_requires_sam_backend",
                            }
                        )

                if not refinement_sam_proposals and fallback_debug.get("skip_reason"):
                    refinement_result = self.async_refinement.process_ready(
                        frame=frame,
                        anchors=list(anchors),
                        sam_proposals=[],
                    )
                    fine_refined = []
                    fine_patches = []
                    replacement_summary = self.object_update.replace_observations(self.state, fine_patches)
                else:
                    refinement_result = self.async_refinement.process_ready(
                        frame=frame,
                        anchors=list(anchors),
                        sam_proposals=refinement_sam_proposals,
                    )
                    fine_refined = self.depth_refinement.process(frame.depth, refinement_result.fine_proposals)
                    fine_patches = self.patch_lifting.process(
                        fine_refined,
                        frame.depth,
                        frame.pose,
                        frame.intrinsics,
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                    )
                    replacement_summary = self.object_update.replace_observations(self.state, fine_patches)

            async_refinement_summary.update(refinement_result.debug)
            async_refinement_summary.update(replacement_summary)
            async_refinement_summary.update(fallback_debug)
            async_refinement_summary.update(
                {
                    "sam_proposal_count": int(len(refinement_sam_proposals)),
                    "fine_refined_proposal_count": int(len(fine_refined)),
                    "fine_patch_count": int(len(fine_patches)),
                }
            )

        # Step 10: Semantic Memory Update (Layer 7 — post-stabilization only)
        with self._timed_stage("semantic_memory"):
            updated_ids = list(self.object_update.last_updated_object_ids)
            updated_ids = sorted(
                set(int(object_id) for object_id in updated_ids)
                | {
                    int(object_id)
                    for object_id in async_refinement_summary.get("updated_object_ids", [])
                }
            )
            self.state = self.semantic_memory.process(self.state, frame.rgb, updated_ids)

        # Step 11: Dense semantic surface maintenance (dual-map dense layer)
        with self._timed_stage("dense_surface"):
            self.state = self.dense_surface.process(self.state, frame.frame_id)

        # Step 12: Dense/coarse residency tiering
        with self._timed_stage("map_tiering"):
            self.state = self.map_tiering.process(self.state, frame.frame_id, frame.pose[:3, 3])

        # Step 13: Background Update
        with self._timed_stage("background_update"):
            background_update_patches = [*bg_patches, *structural_reject_patches]
            self.state.background = self.background_update.process(
                background_update_patches, self.state.background, self.state.objects,
            )
            if self.verbose:
                logger.info(f"  Background: {len(self.state.background.point_cloud)} points")

        # Step 14: Dynamic Maintenance
        with self._timed_stage("dynamic_maintenance"):
            self.state = self.dynamic_maintenance.process(self.state)

        # Increment frame counter
        self.state.frame_count += 1

        amb_patch_ids = {int(p.patch_id) for p in amb_patches}
        self.last_frame_debug = {
            "proposal_source": proposal_source,
            "frontend_source": frontend_source,
            "frontend_actual_backend": frontend_actual_backend,
            "raw_proposal_count": len(proposals),
            "anchor_count": len(anchors),
            "anchored_proposal_count": int(sum(1 for assignment in anchor_assignments if assignment.anchor_id >= 0)),
            "refined_proposal_count": len(refined),
            "patch_count": len(patches),
            "bg_patch_ids": [int(p.patch_id) for p in bg_patches],
            "obj_patch_ids": [int(p.patch_id) for p in obj_patches],
            "amb_patch_ids": [int(p.patch_id) for p in amb_patches],
            "ambiguous_patch_count": int(len(amb_patches)),
            "ambiguous_matched_count": int(
                sum(1 for patch_id, _, _ in association.matched if int(patch_id) in amb_patch_ids)
            ),
            "ambiguous_new_patch_count": int(
                sum(1 for patch_id in association.new_object_patches if int(patch_id) in amb_patch_ids)
            ),
            "contested_residual_patch_count": int(len(association.contested_object_patches)),
            "contested_residual_patch_ids": [
                int(patch_id) for patch_id in association.contested_object_patches
            ],
            "contested_residual_promoted_object_ids": list(
                getattr(self.object_update, "last_contested_residual_promoted_object_ids", [])
            ),
            "surface_owner_gate": dict(self.object_update.last_surface_gate_stats),
            "surface_owner_gate_records": list(self.object_update.last_surface_gate_records),
            "surface_owner_gate_structural_reject_patch_count": int(len(structural_reject_patches)),
            "object_update_stage_timings": dict(getattr(self.object_update, "last_stage_timings", {}) or {}),
            "current_frame_visibility_gate": dict(self.object_update.last_current_frame_visibility_gate_stats),
            "current_frame_visibility_gate_records": list(
                self.object_update.last_current_frame_visibility_gate_records
            ),
            "background_update_patch_count": int(len(background_update_patches)),
            "active_set_candidate_ids": sorted(self.state.active_set.all_candidate_ids),
            "active_set_debug": dict(getattr(self.active_set, "last_debug", {}) or {}),
            "association_debug": association.debug,
            "async_refinement": dict(async_refinement_summary),
            "depth_refinement": dict(depth_refinement_debug),
            "structural_overlay": dict(structural_overlay_summary),
            "anchor_guided_sam": dict(anchor_guided_sam_summary),
            "semantic_updated_object_ids": list(self.semantic_memory.last_updated_object_ids),
            "provisional_object_count": int(len(self.state.provisional_objects)),
            "local_memory_point_count_total": int(
                sum(self.object_update._local_pcd_total_count(obj) for obj in self.state.objects.values())
                if hasattr(self.object_update, "_local_pcd_total_count")
                else sum(len(obj.local_pcd) for obj in self.state.objects.values())
            ),
            "dense_surface_resident_point_count": self.dense_surface.resident_point_count(self.state.dense_surface_map),
            "surface_tier_counts": {
                "active": int(sum(obj.surface_tier.value == "active" for obj in self.state.objects.values())),
                "warm": int(sum(obj.surface_tier.value == "warm" for obj in self.state.objects.values())),
                "cold": int(sum(obj.surface_tier.value == "cold" for obj in self.state.objects.values())),
            },
            "stage_timings": dict(self._stage_timings),
        }

        return self.state
