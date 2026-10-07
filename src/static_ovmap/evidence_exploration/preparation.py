"""CPU-only binding of fixed G1 support and complete selected-frame geometry.

No annotation, evaluation projection, or semantic class chooses any evidence.
"""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from static_ovmap.a7_evidence_upgrade.region_adapter import resized_shape, signed_mask
from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import FullSceneProjector, _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.recovery_registry import build_registry

from .evidence import occupancy_trigger, competing_regions


def unpack_mask(request):
    with np.load(request["mask"]["path"], allow_pickle=False) as arrays:
        shape = tuple(map(int, arrays["shape"]))
        mask = np.unpackbits(arrays["packed"], bitorder="little", count=int(np.prod(shape))).reshape(shape).astype(bool)
    if _array_digest(mask) != request["target_mask_sha256"] or int(mask.sum()) != request["visible_pixels"]:
        raise ValueError("fixed G1 mask content changed")
    return mask


def prepare_scene(binding, root, scene):
    root = Path(root)
    dest = root / "prepared" / scene
    dest.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(dest / "input_verifications.json")
    resolver = PathResolver(binding["path_map"])
    row = binding["scenes"][scene]
    producer = {name: index.identity(Path(__file__).with_name(name))["sha256"] for name in ("preparation.py", "evidence.py")}
    key = canonical_digest({"binding": binding["identity"], "scene": scene, "producer": producer})
    receipt_path = dest / "receipt.json"
    if receipt_path.exists():
        previous = read(receipt_path)
        _verified_identity(previous)
        if previous["input_identity"] != key:
            raise ValueError("completed selected-frame evidence changed; preserve and explicitly invalidate descendants")
        for item in previous["outputs"]:
            index.identity(item["path"], item)
        print("PREPARE_REUSE", scene, flush=True)
        return previous
    started = time.monotonic()
    data = resolver.rewrite(read(row["context"]["path"]))
    capture_path = Path(data["capture_manifest"])
    index.identity(capture_path)
    capture = read(capture_path)
    _verified_identity(capture)
    capture = resolver.rewrite(capture)
    surface = capture_path.parent / capture["surface"]["path"]
    index.identity(surface, capture["surface"])
    with np.load(surface, allow_pickle=False) as arrays:
        xyz, faces, raw = arrays["surface_xyz"], arrays["surface_faces"], arrays["original_owner"]
    native = load_prediction(data["predictions"]["NATIVE_READOUT"])
    if _array_digest(xyz) != native.geometry.xyz_sha256 or _array_digest(faces) != native.geometry.faces_sha256:
        raise ValueError("preparation geometry differs from the locked Native geometry")
    registry = build_registry(xyz, raw, native.owner_ids)
    if registry["identity"] != row["registry_identity"]:
        raise ValueError("candidate registry is not the inherited shared registry")
    index.identity(row["view_manifest"])
    manifest = read(row["view_manifest"])
    _verified_identity(manifest)
    manifest = resolver.rewrite(manifest)
    region = resolver.rewrite(read(row["region_receipt"]))
    old = resolver.rewrite(read(Path(binding["reference"]["parent_root"]) / "recovery" / scene / "receipt.json"))
    frames = {int(x["frame_id"]): x for x in capture["frames"]}
    candidates, groups = {}, {}
    for owner, selected in manifest["g1"].items():
        owner = owner.split(":")[1]
        candidates[owner] = {"owner": int(owner), "request_id": selected[0] if selected else None,
                             "observable": bool(selected), "triggered": False, "purity": None, "controls": []}
        if not selected:
            continue
        rid = selected[0]
        request = manifest["requests"][rid]
        index.identity(request["mask"]["path"], request["mask"])
        source = region["requests"][rid]
        old_source = old["requests"].get(rid, {})
        image_key = source.get("image_content_key", old_source.get("image_content_key"))
        dense_path = old["required_dense_receipts"].get(image_key)
        if dense_path is None and "dense_arrays" in source:
            dense_path = str(Path(source["dense_arrays"]["path"]).with_suffix(".json"))
        if dense_path is None:
            raise FileNotFoundError("actual G1 dense FC evidence is missing: " + rid)
        index.identity(dense_path)
        dense_receipt = read(dense_path)
        dense_arrays = resolver.rewrite(dense_receipt["arrays"])
        index.identity(dense_arrays["path"], dense_arrays)
        # Actual cached tensor grid/channels are authoritative, not model-name assumptions.
        with np.load(dense_arrays["path"], allow_pickle=False) as arrays:
            dense = arrays["dense"]
            if dense.dtype != np.float32 or dense.ndim != 4 or dense.shape[0] != 1 or not np.isfinite(dense).all():
                raise ValueError("inherited dense tensor is not finite batch-one FP32")
            dense_shape = list(dense.shape)
        mask = unpack_mask(request)
        size = resized_shape(mask.shape)
        padded = tuple(x + (-x % 32) for x in size)
        signed, support, _ = signed_mask(mask, size, padded, dense_shape[-2:], "cpu")
        h = int(support.sum())
        inherited_h = source["original_support"] if "original_support" in source else source["dense_mask_support"]
        if h != int(inherited_h):
            raise ValueError("original signed-bilinear support differs from B1")
        occupancy = F.interpolate((signed > 0).float(), size=dense_shape[-2:], mode="area")[0, 0].numpy()
        triggered, purity = occupancy_trigger(occupancy, h)
        candidates[owner].update(triggered=triggered, purity=purity, original_support=h,
            baseline_fallback=bool(source.get("fallback", h == 0)), dense_shape=dense_shape,
            original_size=list(mask.shape), resized_size=list(size), padded_size=list(padded),
            output_size=[x // 4 for x in padded], mask_sha256=request["target_mask_sha256"],
            frame_id=int(request["frame_id"]))
        fid = str(request["frame_id"])
        group = groups.setdefault(fid, {"frame": frames[int(fid)], "request_ids": [], "owners": [],
            "dense_arrays": dense_arrays, "dense_receipt": index.identity(dense_path), "dense_shape": dense_shape})
        if group["dense_arrays"] != dense_arrays or group["dense_shape"] != dense_shape:
            raise ValueError("one physical frame refers to conflicting FC tensors")
        group["request_ids"].append(rid)
        group["owners"].append(owner)
    projector = FullSceneProjector(xyz, faces, raw) if groups else None
    outputs = []
    for fid, group in groups.items():
        frame = group["frame"]
        depth_path = capture_path.parent / frame["depth_path"]
        rgb_path = capture_path.parent / frame["rgb_path"]
        index.identity(depth_path, {"sha256": frame["depth_sha256"]})
        index.identity(rgb_path, {"sha256": frame["rgb_sha256"]})
        with np.load(depth_path, allow_pickle=False) as arrays:
            depth = arrays["depth_m"]
        projected = projector.project_frame(np.asarray(frame["intrinsics"]), np.asarray(frame["pose_c2w"]),
                                            depth, [int(x) for x in group["owners"]])
        arrays = {"visible_owners": projected["owners"], "depth_m": depth,
                  "depth_valid": np.isfinite(depth) & (depth > 1e-6)}
        for owner, rid in zip(group["owners"], group["request_ids"], strict=True):
            request = manifest["requests"][rid]
            mask = unpack_mask(request)
            if not np.array_equal(mask, projected["owners"] == int(owner)):
                raise ValueError("complete-scene target mask differs from the original selected G1 mask")
            arrays[rid + "_target"] = mask
            if candidates[owner]["triggered"]:
                for control in competing_regions(mask, projected["owners"], request["canonical_bbox"]):
                    array_key = rid + "_control_" + str(control["owner"])
                    arrays[array_key] = control["mask"]
                    candidates[owner]["controls"].append({"owner": control["owner"], "pixels": control["pixels"],
                        "array_key": array_key, "mask_sha256": _array_digest(control["mask"])})
        path = dest / "frames" / (fid + ".npz")
        _write_npz(path, arrays)
        group.update(evidence=index.identity(path), rgb=index.identity(rgb_path), depth=index.identity(depth_path),
                     triggered_owners=[x for x in group["owners"] if candidates[x]["triggered"]])
        outputs.append(group["evidence"])
    result = seal({"status": "PREPARED", "input_identity": key, "binding_identity": binding["identity"],
        "scene": scene, "registry_identity": registry["identity"], "view_identity": row["view_identity"],
        "candidates": candidates, "frames": groups, "outputs": outputs, "inputs": index.entries(),
        "elapsed_seconds": time.monotonic()-started, "GT_input": False, "new_neural_inference": 0,
        "selected_frames": len(groups), "triggered_frames": sum(bool(x["triggered_owners"]) for x in groups.values()),
        "triggered_candidates": sum(x["triggered"] for x in candidates.values()), "full_scene_projection_calls": len(groups)})
    atomic_write_json(receipt_path, result)
    index.write_memo(dest / "input_verifications.json")
    print("PREPARED", scene, "candidates", len(candidates), "triggered", result["triggered_candidates"], flush=True)
    return result


def prepare(binding, root, scenes=None):
    torch.set_num_threads(4)
    result = [prepare_scene(binding, root, scene) for scene in (scenes or binding["scenes"])]
    if scenes is None:
        atomic_write_json(Path(root) / "prepared" / "summary.json", seal({"status": "PREPARED", "scenes": {
            row["scene"]: row["identity"] for row in result}, "selected_frames": sum(x["selected_frames"] for x in result),
            "triggered_frames": sum(x["triggered_frames"] for x in result),
            "triggered_candidates": sum(x["triggered_candidates"] for x in result), "new_neural_inference": 0}))
    return result
