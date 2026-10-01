"""Serial real-frame parity and a real native many-to-one follower fixture."""

import argparse
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from static_ovmap.backbone_wave1.mapping_hooks import native_fragments
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz

from .association import plan_objects
from .binding import ConsumptionIndex, read


def _integrate(gsm, segments, pose):
    for segment in segments:
        segment.calculateBBox()
        gsm.insertSegmentsOpen(segment.points, segment.box_points.astype(np.float32),
            int(segment.instance_label), int(segment.class_label), segment.sem_feat.astype(np.float32),
            float(segment.inst_confidence), float(segment.overlap_ratio), pose, bool(segment.is_thing), 0)
    gsm.integrateFrame()


def _actions(gsm, plan):
    result = gsm.beginRecoveryAssociation([(row["local_group"], row["action"], row["existing_owner"] or 0)
                                          for row in plan["actions"]])
    if result["allocated_owners"] or result["state_before"] != result["state_after"]:
        raise ValueError("recovery actions preallocated owners or changed native state")
    return result


def run(binding, scene, mode, extension, output):
    import consistent_gsm
    from utils.common_scannet_nyu import SegmentsGenerator
    from utils.common_utils import Segment

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "log").mkdir(exist_ok=True)
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    if Path(consistent_gsm.__file__).resolve() != Path(extension).resolve():
        raise ValueError("native trace imported a different extension")
    index.identity(extension)
    index.identity(__file__)
    started = time.monotonic()
    gsm = consistent_gsm.GlobalSegmentMap_py(str(root / "log"), "Nyu40", False, False,
                                            4, 2, 1, False, 3, True, .2, .9)
    data = binding["scenes"][scene]
    capture_path = Path(data["capture_manifest"])
    index.identity(capture_path)
    capture = read(capture_path)
    enabled = mode not in {"off", "off-parent"}
    if mode != "off-parent":
        gsm.configureRecoveryAssociation(enabled)
    frames = []
    if mode == "follower":
        shape = (80, 80)
        k = np.array([[100., 0., 39.5], [0., 100., 39.5], [0., 0., 1.]], np.float32)
        pose = np.eye(4, dtype=np.float32)
        depth = np.full(shape, 2., np.float32)
        gsm.initializeCameraRayCaster(k, *shape, .01, 50., 1)
        v, u = np.indices(shape)
        xyz = np.stack([(u - k[0, 2]) * depth / k[0, 0],
                        (v - k[1, 2]) * depth / k[1, 1], depth], axis=-1).astype(np.float32)
        make = lambda mask, group, position: Segment(xyz[mask], True, group, 1, 1., 1., pose,
                                                    position, sem_feat=np.array([0., 1.], np.float32))
        whole = np.ones(shape, bool)
        allocation = _actions(gsm, {"actions": [{"local_group": 1, "action": "USE_NATIVE", "existing_owner": None}]})
        _integrate(gsm, [make(whole, 1, 0)], pose)
        frames.append({"frame_id": 0, "state": gsm.exportStudyFrameState(), "allocation": allocation})
        gsm.clearTemporaryMemory()
        probe = gsm.exportAssociationProbe(pose, depth)
        first, second = whole.copy(), whole.copy()
        first[:, 40:], second[:, :40] = False, False
        plan = plan_objects([SimpleNamespace(mask=mask, input_group=group, is_thing=True)
                             for group, mask in ((11, first), (12, second))], depth, probe["prior_owner"], mode="A2")
        if plan["duplicate_owner_followers"] != 1 or len(plan["accepted_pairs"]) != 2:
            raise ValueError("native fixture did not produce a genuine shared prior owner")
        allocation = _actions(gsm, plan)
        _integrate(gsm, [make(first, 11, 0), make(second, 12, 1)], pose)
        state = gsm.exportStudyFrameState()
        desired = dict(plan["accepted_pairs"])
        atomic_write_json(root / "follower_observed.json", {"plan": plan, "state": state, "allocation": allocation})
        for row in state["segments"]:
            if row["realized_frame_assignment"] != desired[row["input_instance_label"]]:
                raise ValueError("real native mode4 restored a one-to-one owner restriction")
        frames.append({"frame_id": 1, "state": state, "plan": plan, "allocation": allocation})
        _write_npz(root / "follower_input.npz", {"depth": depth, "points": xyz,
                                                "first": first, "second": second, "prior_owner": probe["prior_owner"]})
        gsm.clearTemporaryMemory()
    else:
        generator = SegmentsGenerator(gsm, None, None, use_geometrics=True,
                                     geometrics_folder=data["inherited"]["geometric_root"])
        k = np.asarray(capture["frames"][0]["intrinsics"], np.float32)
        height, width = capture["frames"][0]["image_size_hw"]
        gsm.initializeCameraRayCaster(k, height, width, .01, 50., 1)
        for frame in capture["frames"][:3]:
            frame_id = frame["frame_id"]
            for name in ("depth", "panoptic"):
                index.identity(capture_path.parent / frame[name + "_path"], {"sha256": frame[name + "_sha256"]})
            index.identity(Path(data["inherited"]["geometric_root"]) / f"{frame_id:05d}_mask.png")
            with np.load(capture_path.parent / frame["depth_path"], allow_pickle=False) as arrays:
                depth = arrays["depth_m"].astype(np.float32)
            pano = np.asarray(Image.open(capture_path.parent / frame["panoptic_path"]), np.int32)
            pose = np.asarray(frame["pose_c2w"], np.float32)
            segments = generator.frameToSegmentsCropFormer(depth, k, pose, frame_id, pano)
            plan, allocation, probe = None, None, None
            if enabled:
                probe = gsm.exportAssociationProbe(pose, depth)
                if probe["state_before"] != probe["state_after"]:
                    raise ValueError("native prior trace probe mutated state")
                plan = plan_objects(native_fragments(segments, k, pano), depth, probe["prior_owner"],
                                    mode="A1" if mode == "ALL_NATIVE" else mode)
                if mode == "ALL_NATIVE":
                    plan["actions"] = [{"local_group": group, "action": "USE_NATIVE", "existing_owner": None}
                                       for group in plan["local_groups"]]
                    plan["accepted_pairs"] = []
                allocation = _actions(gsm, plan)
            _integrate(gsm, segments, pose)
            state = gsm.exportStudyFrameState()
            projected = gsm.raycastInstancePredictions(pose, pano, depth)
            arrays = {"owner_projection": projected}
            if probe is not None:
                arrays.update({key: probe[key] for key in ("prior_label", "prior_owner", "hit_depth", "validity")})
            _write_npz(root / f"{frame_id:06d}.npz", arrays)
            frames.append({"frame_id": frame_id, "state": state, "plan": plan, "allocation": allocation,
                           "owner_projection_sha256": _array_digest(projected)})
            gsm.clearTemporaryMemory()
    tsdf = {str(key): np.asarray(value) for key, value in gsm.exportStudyTsdfState().items()}
    _write_npz(root / "tsdf.npz", tsdf)
    receipt = {"status": "COMPLETE", "scene": scene, "mode": mode, "threads": 1,
        "frames": frames, "native_extension": index.identity(extension), "inputs": index.entries(),
        "new_visual_inference": 0, "GT_input": False, "synthetic_native_fixture": mode == "follower",
        "actual_acquired_frames": mode != "follower", "elapsed_seconds": time.monotonic() - started,
        "tsdf_array_identities": {key: _array_digest(value) for key, value in tsdf.items()}}
    receipt["identity"] = canonical_digest({key: value for key, value in receipt.items() if key != "elapsed_seconds"})
    atomic_write_json(root / "receipt.json", receipt)
    print(mode, receipt["status"], "frames", len(frames), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", default="scene0056_00")
    parser.add_argument("--mode", required=True, choices=("off-parent", "off", "ALL_NATIVE", "A1", "A2", "A3", "follower"))
    parser.add_argument("--extension", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(read(args.binding), args.scene, args.mode, args.extension, args.output)
