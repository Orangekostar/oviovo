"""Small opt-in hooks around the unchanged native mapping sequence."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from .association import plan_objects
from .fusion import Fragment, canonical_fragments, lift_fragments, mask_key, simultaneous_fusion


def native_fragments(segments, intrinsics, pano):
    height, width = pano.shape
    rows = []
    for segment in segments:
        points = np.asarray(segment.points)
        positive = np.isfinite(points).all(1) & (points[:, 2] > 0)
        points = points[positive]
        projected = np.asarray(intrinsics) @ points.T
        u, v = np.rint(projected[:2] / projected[2]).astype(np.int64)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        support = np.zeros_like(pano, bool)
        support[v[valid], u[valid]] = True
        group = int(segment.instance_label)
        rows.append(Fragment(support, group, mask_key(pano == group) if group > 0 else None,
                             bool(segment.is_thing), float(segment.inst_confidence), float(segment.overlap_ratio)))
    return rows


class BackboneMappingHook:
    def __init__(self, recipe_path, gsm):
        self.recipe = json.loads(Path(recipe_path).read_text())
        self.root = Path(self.recipe["diagnostic_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.gsm = gsm
        self.valid_count = 0
        gsm.configureBackboneAssociation(self.recipe["association"] == "native_ratio_gate",
                                       self.recipe["association"].startswith("object_"))

    def raster(self, frame_id, original, source_path):
        frontend = self.recipe["frontend"]
        if frontend == "cropformer":
            return original, source_path
        name = "raw" if frontend == "sam2_raw" else "geom"
        path = Path(self.recipe["paired_frontend_root"]) / name / f"{frame_id:06d}.png"
        raster = np.asarray(Image.open(path), np.int32)
        if raster.shape != original.shape:
            raise ValueError("SAM input raster changed aligned camera dimensions")
        return raster, str(path)

    def segments(self, generator, depth, intrinsics, pose, frame_id, pano):
        import utils.common_scannet_nyu as native

        native_rows = None
        if self.recipe["depth_fusion"] == "simultaneous":
            rows = simultaneous_fusion(generator.load2DGeometricSegs(frame_id), pano)
            segments = lift_fragments(rows, depth, intrinsics, pose, native)
        else:
            segments = generator.frameToSegmentsCropFormer(depth, intrinsics, pose, frame_id, pano)
            rows = native_fragments(segments, intrinsics, pano)
            native_rows = rows
        frame_root = self.root / f"{frame_id:06d}"
        frame_root.mkdir()
        self.valid_count += 1
        if self.recipe.get("order_diagnostics") and self.valid_count <= 3:
            self._order_diagnostic(native, generator, depth, intrinsics, pose, frame_id, pano, native_rows, frame_root)
        if self.recipe["association"].startswith("object_") and segments:
            probe = self.gsm.exportAssociationProbe(pose.astype(np.float32), depth.astype(np.float32))
            if probe["state_before"] != probe["state_after"]:
                raise RuntimeError("prior probe mutated native state")
            plan = plan_objects(rows, depth, probe["prior_owner"], mode=self.recipe["association"])
            allocated = self.gsm.beginBackboneAssociation([(k, plan.planned_owners[k]) for k in plan.local_groups])
            _write_npz(frame_root / "probe.npz", {k: np.asarray(probe[k]) for k in
                       ("prior_label", "prior_owner", "hit_depth", "validity")})
            atomic_write_json(frame_root / "association_plan.json", {
                "prior_state_id": canonical_digest(probe["state_before"]),
                "probe_state_before": probe["state_before"], "probe_state_after": probe["state_after"],
                "plan": asdict(plan), "native_allocation": allocated,
                "fragments": [{"index": i, "input_group": r.input_group,
                    "support_key": mask_key(r.mask), "pixels": int(r.mask.sum())} for i, r in enumerate(rows)]})
        return segments

    def _order_diagnostic(self, native, generator, depth, k, pose, frame_id, pano, native_rows, frame_root):
        if native_rows is None:
            native_rows = native_fragments(generator.frameToSegmentsCropFormer(depth, k, pose, frame_id, pano), k, pano)
        counter = native.Counter
        class ReversedCounter(counter):
            def __iter__(self):
                return iter(list(dict.keys(self))[::-1])
        try:
            native.Counter = ReversedCounter
            reversed_rows = native_fragments(generator.frameToSegmentsCropFormer(depth, k, pose, frame_id, pano), k, pano)
        finally:
            native.Counter = counter
        groups = [int(g) for g in np.unique(pano) if g > 0]
        renamed = np.zeros_like(pano)
        for group, replacement in zip(groups, groups[::-1]):
            renamed[pano == group] = replacement
        renamed_rows = native_fragments(generator.frameToSegmentsCropFormer(depth, k, pose, frame_id, renamed), k, renamed)
        sync_supports = simultaneous_fusion(generator.load2DGeometricSegs(frame_id), pano)
        sync = native_fragments(lift_fragments(sync_supports, depth, k, pose, native), k, pano)
        collections = {"native": canonical_fragments(native_rows), "reverse": canonical_fragments(reversed_rows),
                       "renamed": canonical_fragments(renamed_rows), "sync": canonical_fragments(sync)}
        atomic_write_json(frame_root / "order_diagnostic.json", {
            "frame_id": frame_id, "canonical_fragments": collections,
            "native_reverse_physical_change": collections["native"] != collections["reverse"],
            "native_rename_physical_change": collections["native"] != collections["renamed"],
            "sync_physical_change": collections["native"] != collections["sync"],
            "native_order": [mask_key(r.mask) for r in native_rows],
            "reverse_order": [mask_key(r.mask) for r in reversed_rows],
            "raw_pano_sha256": _array_digest(pano), "id_bijection": dict(zip(groups, groups[::-1]))})

    def finalize(self, inst_dict, gsm, *, max_top_vis, strategy):
        rows = []
        for owner, info in inst_dict.items():
            rows.append({"owner": int(owner), "frame_id": [int(x) for x in info["frame_id"]],
                "box_2d": [list(map(int, x)) for x in info["box_2d"]],
                "pose": [np.asarray(x).tolist() for x in info["pose"]],
                "vis_area": [int(x) for x in info["vis_area"]],
                "color": np.asarray(gsm.getInstanceColor(int(owner))).tolist()})
        atomic_write_json(self.root / "native_deferred_metadata.json", {
            "status": "DEFERRED_FEATURES_REQUIRED", "owner_insertion_order": [row["owner"] for row in rows],
            "rows": rows, "max_top_views": max_top_vis, "view_select_strategy": strategy,
            "selection": "INHERITED_NP_ARGSORT_AFTER_SUCCESSFUL_FEATURE_FILTER", "completed_native_pickle": False})
