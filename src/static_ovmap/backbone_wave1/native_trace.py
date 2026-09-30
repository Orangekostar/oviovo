"""Three acquired-frame native trace; no synthetic map or neural inference."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.boundary_jobs import file_identity
from .association import plan_objects
from .mapping_hooks import native_fragments


def run(job):
    import consistent_gsm
    from utils.common_scannet_nyu import SegmentsGenerator

    root = Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "log").mkdir(exist_ok=True)
    capture_path = Path(job["capture_manifest"])
    capture = json.loads(capture_path.read_text())
    threads = int(job.get("threads", 8))
    gsm = consistent_gsm.GlobalSegmentMap_py(str(root / "log"), "Nyu40", False, False,
                                            4, 2, threads, False, 3, True, .2, .9)
    if Path(consistent_gsm.__file__).resolve() != Path(job["extension"]).resolve():
        raise ValueError("trace loaded an unexpected native binary")
    mode = job["mode"]
    if job.get("new_binary"):
        gsm.configureBackboneAssociation(False, mode != "native")
    generator = SegmentsGenerator(gsm, None, None, use_geometrics=True,
                                 geometrics_folder=job["geometric_root"])
    k = np.asarray(capture["frames"][0]["intrinsics"], np.float32)
    height, width = capture["frames"][0]["image_size_hw"]
    gsm.initializeCameraRayCaster(k, height, width, .01, 50., threads)
    frames, start = [], time.monotonic()
    for frame in capture["frames"][:3]:
        frame_id = frame["frame_id"]
        with np.load(capture_path.parent / frame["depth_path"], allow_pickle=False) as arrays:
            depth = arrays["depth_m"].astype(np.float32)
        pano = np.asarray(Image.open(capture_path.parent / frame["panoptic_path"]), np.int32)
        pose = np.asarray(frame["pose_c2w"], np.float32)
        probe, plan, allocation = None, None, None
        if job.get("new_binary"):
            probe = gsm.exportAssociationProbe(pose, depth)
            if probe["state_before"] != probe["state_after"]:
                raise ValueError("trace probe mutated native state")
        segments = generator.frameToSegmentsCropFormer(depth, k, pose, frame_id, pano)
        if mode != "native":
            from dataclasses import asdict
            plan = plan_objects(native_fragments(segments, k, pano), depth, probe["prior_owner"], mode=mode)
            allocation = gsm.beginBackboneAssociation([(g, plan.planned_owners[g]) for g in plan.local_groups])
            plan = asdict(plan)
        for segment in segments:
            segment.calculateBBox()
            gsm.insertSegmentsOpen(segment.points, segment.box_points.astype(np.float32),
                int(segment.instance_label), int(segment.class_label), segment.sem_feat.astype(np.float32),
                float(segment.inst_confidence), float(segment.overlap_ratio), pose, bool(segment.is_thing), 0)
        gsm.integrateFrame()
        state = gsm.exportStudyFrameState()
        projected = gsm.raycastInstancePredictions(pose, pano, depth)
        arrays = {"owner_projection": projected}
        if probe is not None:
            arrays.update({name: probe[name] for name in ("prior_label", "prior_owner", "hit_depth", "validity")})
        _write_npz(root / f"{frame_id:06d}.npz", arrays)
        frames.append({"frame_id": frame_id, "state": state, "plan": plan, "allocation": allocation,
            "probe_state_before": None if probe is None else probe["state_before"],
            "probe_state_after": None if probe is None else probe["state_after"]})
        gsm.clearTemporaryMemory()
    tsdf = {str(k): np.asarray(v) for k, v in gsm.exportStudyTsdfState().items()}
    _write_npz(root / "tsdf.npz", tsdf)
    receipt = {"status": "COMPLETE", "extension": consistent_gsm.__file__, "mode": mode,
               "extension_identity": file_identity(consistent_gsm.__file__), "threads": threads,
               "frames": frames, "elapsed_seconds": time.monotonic() - start,
               "tsdf_array_identities": {k: _array_digest(v) for k, v in tsdf.items()},
               "actual_acquired_frames": True, "new_visual_inference": 0}
    atomic_write_json(root / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run(json.loads(Path(parser.parse_args().job).read_text()))
