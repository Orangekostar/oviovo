"""Recovery actions around the original sequential CropFormer mapping path."""

from pathlib import Path

import numpy as np
from PIL import Image

from static_ovmap.backbone_wave1.mapping_hooks import BackboneMappingHook, native_fragments
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _write_npz

from .association import plan_objects
from .binding import ConsumptionIndex, read


class RecoveryMappingHook(BackboneMappingHook):
    def __init__(self, recipe_path, gsm):
        self.recipe = read(recipe_path)
        self.root = Path(self.recipe["diagnostic_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.gsm = gsm
        self.history = []
        gsm.configureRecoveryAssociation(self.recipe["association"] in {"A1", "A2", "A3", "ALL_NATIVE"})
        self.frontend_context = None

    def raster(self, frame_id, original, source_path):
        frontend = self.recipe["frontend"]
        if frontend == "cropformer":
            return original, source_path
        if frontend not in {"S1", "S2"}:
            raise ValueError("unknown recovery frontend")
        path = Path(self.recipe["frontend_root"]) / f"{frame_id:06d}.png"
        if not hasattr(self, "frontend_frames"):
            receipt = read(self.recipe["frontend_receipt"])
            if receipt["status"] != "COMPLETE" or receipt["identity"] != canonical_digest(
                    {k: v for k, v in receipt.items() if k != "identity"}):
                raise ValueError("locked SAM completion receipt changed")
            self.frontend_frames = {row["frame_id"]: row for row in receipt["frames"]}
            self.frontend_index = ConsumptionIndex(Path(self.recipe["frontend_receipt"]).parent / "input_verifications.json")
        self.frontend_index.identity(path, self.frontend_frames[frame_id]["output"])
        raster = np.asarray(Image.open(path), np.int32)
        if raster.shape != original.shape or np.any(raster[original > 0] != original[original > 0]):
            raise ValueError("SAM completion changed a positive CropFormer pixel")
        self.frontend_context = {"frame_id": frame_id, "original": original.copy(), "s1": raster.copy()}
        if frontend == "S2":
            path = self.root / "s2_rasters" / f"{frame_id:06d}.png"
        return raster, str(path)

    def segments(self, generator, depth, intrinsics, pose, frame_id, pano):
        frame_root = self.root / f"{frame_id:06d}"
        frame_root.mkdir()
        if self.recipe["frontend"] == "S2":
            from .sam_completion import causal_s2_raster
            pano[:] = causal_s2_raster(self, frame_id, depth, intrinsics, pose, pano, frame_root)
            path = self.root / "s2_rasters" / f"{frame_id:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pano.astype(np.uint16)).save(path)
        segments = generator.frameToSegmentsCropFormer(depth, intrinsics, pose, frame_id, pano)
        mode = self.recipe["association"]
        if mode in {"A1", "A2", "A3", "ALL_NATIVE"}:
            probe = self.gsm.exportAssociationProbe(pose.astype(np.float32), depth.astype(np.float32))
            if probe["state_before"] != probe["state_after"]:
                raise RuntimeError("own-map prior probe mutated native state")
            plan = plan_objects(native_fragments(segments, intrinsics, pano), depth, probe["prior_owner"],
                                mode="A1" if mode == "ALL_NATIVE" else mode)
            if mode == "ALL_NATIVE":
                plan["actions"] = [{"local_group": group, "action": "USE_NATIVE", "existing_owner": None}
                                   for group in plan["local_groups"]]
                plan["accepted_pairs"] = []
                plan["action_counts"] = {"USE_NATIVE": len(plan["local_groups"]), "ASSIGN_EXISTING": 0, "CREATE_NEW": 0}
            allocation = self.gsm.beginRecoveryAssociation([(a["local_group"], a["action"], a["existing_owner"] or 0)
                                                          for a in plan["actions"]])
            if allocation["allocated_owners"]:
                raise RuntimeError("scientific A arms cannot explicitly allocate a fresh owner")
            if allocation["state_before"] != allocation["state_after"]:
                raise RuntimeError("planning changed native state before insertion")
            _write_npz(frame_root / "probe.npz", {name: np.asarray(probe[name])
                       for name in ("prior_label", "prior_owner", "hit_depth", "validity")})
            atomic_write_json(frame_root / "association_plan.json", {"frame_id": frame_id,
                "prior_state_id": canonical_digest(probe["state_before"]), "plan": plan,
                "probe_state_before": probe["state_before"], "probe_state_after": probe["state_after"],
                "native_allocation": allocation, "GT_input": False})
        return segments

    def after_frame(self, frame_id, depth, intrinsics, pose, pano):
        state = self.gsm.exportStudyFrameState()
        atomic_write_json(self.root / f"{frame_id:06d}" / "realized_state.json", state)
        if self.recipe["frontend"] == "S2":
            from .sam_completion import capture_s2_history
            capture_s2_history(self, frame_id, depth, intrinsics, pose, pano)
