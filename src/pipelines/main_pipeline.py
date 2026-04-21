"""Main pipeline: wires all modules together in sequence.

Processes one frame at a time through the full OVIOVO pipeline:

Layer 1: SAM2 proposal frontend (raw proposals)
Layer 2: Depth-aware refinement -> RefinedProposal2D
         RuntimeVis (grouping/cleanup only — Rule A: no instance assignment)
Layer 3: Patch lifting -> Patch3D
Layer 4: Global TSDF instance substrate (spatial voting + support stabilization)
Layer 5: Per-instance local geometry memory (maintenance only)
Layer 6: Active set / local status checks
Layer 7: Semantic layer (post-stabilization only)

Critical rules enforced:
  Rule A: runtime_vis does NOT do final instance assignment
  Rule B: No hard whole-object labels, only soft evidence
  Rule C: Semantics do not enter low-level association
  Rule D: Clear distinction between raw/refined/patch/instance/local_pcd/semantic
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np

from src.core.data_structures import (
    BackgroundMap,
    CameraIntrinsics,
    Frame,
    SystemState,
    TSDFInstanceVolume,
)
from src.modules.frame_input import FrameInputModule
from src.modules.proposal import ProposalModule
from src.modules.runtime_vis import RuntimeVisModule, RuntimeVisOutput
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.patch_lifting import PatchLiftingModule
from src.modules.active_set import ActiveSetModule
from src.modules.bg_obj_split import BgObjSplitModule
from src.modules.association import AssociationModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.semantic_memory import SemanticMemoryModule
from src.modules.background_update import BackgroundUpdateModule
from src.modules.dynamic_maintenance import DynamicMaintenanceModule
from src.utils.logging import setup_logging

logger = logging.getLogger("oviovo.pipeline")


class Pipeline:
    """Central pipeline runner for the OVIOVO mapping system."""

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
        self.runtime_vis = RuntimeVisModule(self.config.get("runtime_vis", {}))
        self.depth_refinement = DepthRefinementModule(self.config.get("depth_refinement", {}))
        self.patch_lifting = PatchLiftingModule(self.config.get("patch_lifting", {}))
        self.active_set = ActiveSetModule(self.config.get("active_set", {}))
        self.bg_obj_split = BgObjSplitModule(self.config.get("bg_obj_split", {}))
        self.association = AssociationModule(self.config.get("association", {}))
        self.object_update = ObjectUpdateModule(self.config.get("object_update", {}))
        self.semantic_memory = SemanticMemoryModule(self.config.get("semantic_memory", {}))
        self.background_update = BackgroundUpdateModule(self.config.get("background_update", {}))
        self.dynamic_maintenance = DynamicMaintenanceModule(self.config.get("dynamic_maintenance", {}))

        self.verbose = self.config.get("pipeline", {}).get("verbose", True)
        self.last_raw_proposals = []
        self.last_refined_proposals = []
        self.last_patches = []
        self.last_runtime_vis_output: RuntimeVisOutput | None = None
        self.last_association = None
        self.last_frame_debug: Dict[str, Any] = {}
        logger.info("Pipeline initialized with all modules.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML config file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults.")
            return {}
        with open(path) as f:
            return yaml.safe_load(f)

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
        timestamp: float = 0.0,
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
          9. Object Update (TSDF integration + local_pcd maintenance — Layers 4+5)
         10. Semantic Memory (post-stabilization only — Layer 7)
         11. Background Update
         12. Dynamic Maintenance

        Args:
            rgb: (H, W, 3) uint8 image.
            depth: (H, W) float32 depth in meters.
            pose: (4, 4) camera-to-world transform.
            intrinsics: Camera intrinsic parameters.
            timestamp: Frame timestamp.

        Returns:
            Updated SystemState.
        """
        # Step 1: Frame Input
        frame: Frame = self.frame_input.process(rgb, depth, pose, intrinsics, timestamp)
        if self.verbose:
            logger.info(f"=== Frame {frame.frame_id} ===")

        # Step 2: Proposal Generation (Layer 1 — SAM2 raw proposals)
        proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
        self.last_raw_proposals = proposals
        if self.verbose:
            logger.info(f"  Proposals: {len(proposals)}")

        # Step 3: RuntimeVis — grouping/cleanup only (Rule A)
        # RuntimeVis may group fragmented masks but does NOT do final instance assignment.
        runtime_vis_output = self.runtime_vis.process(frame, proposals, self.state)
        self.last_runtime_vis_output = runtime_vis_output
        merged_proposals = runtime_vis_output.merged_proposals
        if self.verbose:
            logger.info(
                "  RuntimeVis: raw=%d merged=%d accepted_edges=%d",
                len(proposals),
                len(merged_proposals),
                runtime_vis_output.group_stats.get("accepted_edge_count", 0),
            )

        # Step 4: Depth Refinement -> RefinedProposal2D (Layer 2)
        refined = self.depth_refinement.process(frame.depth, merged_proposals)
        self.last_refined_proposals = refined
        if self.verbose:
            logger.info(f"  Refined proposals: {len(refined)}")

        # Step 5: Patch Lifting (Layer 3 — 2D -> 3D)
        patches = self.patch_lifting.process(
            refined, frame.depth, frame.pose, frame.intrinsics,
            frame_id=frame.frame_id, timestamp=frame.timestamp,
        )
        self.last_patches = patches
        if self.verbose:
            logger.info(f"  3D patches: {len(patches)}")

        # Step 6: Active Set derivation (Layer 6 — local reasoning only)
        self.state.active_set = self.active_set.process(
            self.state, frame.pose, frame.intrinsics,
        )
        if self.verbose:
            logger.info(f"  Active set: {len(self.state.active_set.all_candidate_ids)} candidates")

        # Step 7: Background / Object Split (using soft scores)
        bg_patches, obj_patches, amb_patches = self.bg_obj_split.process(
            patches, self.state.background, self.state.objects,
        )
        if self.verbose:
            logger.info(f"  Split: bg={len(bg_patches)}, obj={len(obj_patches)}, amb={len(amb_patches)}")

        # Step 8: Object Association (Layer 4 — TSDF spatial voting, geometry-first)
        association = self.association.process(
            obj_patches, self.state.objects, self.state.tsdf_volume, active_set=self.state.active_set,
        )
        self.last_association = association
        if self.verbose:
            logger.info(
                f"  Association: {len(association.matched)} matched, "
                f"{len(association.new_object_patches)} new"
            )

        # Step 9: Object Update (Layers 4+5 — TSDF integration + local_pcd)
        self.state = self.object_update.process(association, obj_patches, self.state)
        if self.verbose:
            logger.info(f"  Objects: {len(self.state.objects)}")

        # Update active set with new object candidates
        for patch_id in association.new_object_patches:
            for oid, obj in self.state.objects.items():
                if obj.creation_frame == frame.frame_id:
                    self.state.active_set.new_object_candidate_ids.add(oid)

        # Step 10: Semantic Memory Update (Layer 7 — post-stabilization only)
        updated_ids = [obj_id for _, obj_id, _ in association.matched]
        updated_ids += [
            oid for oid in self.state.objects
            if self.state.objects[oid].creation_frame == frame.frame_id
        ]
        self.state = self.semantic_memory.process(self.state, frame.rgb, updated_ids)

        # Step 11: Background Update
        self.state.background = self.background_update.process(
            bg_patches, self.state.background, self.state.objects,
        )
        if self.verbose:
            logger.info(f"  Background: {len(self.state.background.point_cloud)} points")

        # Step 12: Dynamic Maintenance
        self.state = self.dynamic_maintenance.process(self.state)

        # Increment frame counter
        self.state.frame_count += 1

        self.last_frame_debug = {
            "raw_proposal_count": len(proposals),
            "refined_proposal_count": len(refined),
            "patch_count": len(patches),
            "bg_patch_ids": [int(p.patch_id) for p in bg_patches],
            "obj_patch_ids": [int(p.patch_id) for p in obj_patches],
            "amb_patch_ids": [int(p.patch_id) for p in amb_patches],
            "active_set_candidate_ids": sorted(self.state.active_set.all_candidate_ids),
            "association_debug": association.debug,
            "semantic_updated_object_ids": list(self.semantic_memory.last_updated_object_ids),
        }

        return self.state
