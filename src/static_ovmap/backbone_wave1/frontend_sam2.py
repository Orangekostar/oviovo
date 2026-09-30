"""Bounded causal SAM2 video assay; paired RAW/GEOM share actual forwards."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _write_npz
from static_ovmap.m2_reviewer_study.binding import InputIndex
from .fusion import mask_key, spatial_key
from .runtime import exclusive_lock


def warp_visible_support(mask, previous_depth, previous_pose, previous_k,
                         current_depth, current_pose, current_k):
    previous_depth, current_depth = np.asarray(previous_depth), np.asarray(current_depth)
    mask = np.asarray(mask, bool)
    if mask.shape != previous_depth.shape or current_depth.ndim != 2:
        raise ValueError("aligned RGB-D supports required")
    y, x = np.nonzero(mask & np.isfinite(previous_depth) & (previous_depth > 0) & (previous_depth < 50))
    result = np.zeros_like(current_depth, bool)
    if not len(x):
        return result
    rays = np.linalg.inv(np.asarray(previous_k, np.float64)) @ np.vstack((x, y, np.ones(len(x))))
    points = rays * previous_depth[y, x]
    transform = np.linalg.solve(np.asarray(current_pose, np.float64), np.asarray(previous_pose, np.float64))
    current = transform[:3, :3] @ points + transform[:3, 3, None]
    valid = np.isfinite(current).all(0) & (current[2] > 0)
    current = current[:, valid]
    if not current.shape[1]:
        return result
    projected = np.asarray(current_k, np.float64) @ current
    u, v = np.rint(projected[:2] / projected[2]).astype(np.int64)
    height, width = current_depth.shape
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    z_buffer = np.full(height * width, np.inf)
    np.minimum.at(z_buffer, v[valid] * width + u[valid], current[2, valid])
    z_buffer = z_buffer.reshape(height, width)
    observed = np.isfinite(current_depth) & (current_depth > 0) & (current_depth < 50)
    return observed & np.isfinite(z_buffer) & (np.abs(z_buffer - current_depth) <= .03 + .01 * current_depth)


def current_output(predictor, state, index):
    outputs = list(predictor.propagate_in_video(
        state, start_frame_idx=index, max_frame_num_to_track=0, reverse=False))
    if len(outputs) != 1 or int(outputs[0][0]) != index:
        raise RuntimeError("pinned SAM2 propagation did not yield only the current frame")
    _, ids, logits = outputs[0]
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    logits = np.asarray(logits, np.float32)
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] != len(ids):
        raise ValueError("SAM2 must return video-resolution object logits")
    return [int(x) for x in ids], logits[:, 0]


def _raster(logits, keys, included, replacements, discoveries, cropformer):
    height, width = cropformer.shape
    raster = np.zeros((height, width), np.uint16)
    remap, next_id = [], 1
    if included:
        order = sorted(included, key=lambda i: keys[i])
        values = logits[order]
        winner = np.argmax(values, axis=0)
        positive = np.max(values, axis=0) > 0
        for index, track in enumerate(order):
            support = positive & (winner == index)
            if support.any():
                raster[support] = next_id
                remap.append({"dense_group": next_id, "kind": "SAM_TRACK", "seed_key": keys[track]})
                next_id += 1
    used = set()
    for iou, group in sorted(replacements, key=lambda row: (-row[0], spatial_key(cropformer == row[1]))):
        if group in used:
            continue
        used.add(group)
        support = (cropformer == group) & (raster == 0)
        if support.any():
            raster[support] = next_id
            remap.append({"dense_group": next_id, "kind": "CROPFORMER_REPLACEMENT",
                          "cropformer_group": int(group), "iou": iou})
            next_id += 1
    for group in discoveries:
        support = (cropformer == group) & (raster == 0)
        if support.any():
            if next_id > np.iinfo(np.uint16).max:
                raise ValueError("dense grouping IDs exceed native uint16")
            raster[support] = next_id
            remap.append({"dense_group": next_id, "kind": "RAW_DISCOVERY", "cropformer_group": int(group)})
            next_id += 1
    return raster, remap


def paired_rasters(logits, seed_keys, cropformer, visible_past):
    logits, cropformer = np.asarray(logits, np.float32), np.asarray(cropformer)
    if logits.ndim != 3 or logits.shape[1:] != cropformer.shape or len(seed_keys) != len(logits):
        raise ValueError("aligned track logits and CropFormer raster required")
    if not np.isfinite(logits).all() or len(visible_past) != len(logits):
        raise ValueError("finite logits and one visible-past support per track required")
    masks = logits > 0
    raw_union = masks.any(0)
    groups = sorted((int(k) for k in np.unique(cropformer) if k > 0),
                    key=lambda k: spatial_key(cropformer == k))
    discovery = [k for k in groups if (raw_union & (cropformer == k)).sum() / (cropformer == k).sum() < .2]
    accepted, replacements, tracks = [], [], []
    for index, (mask, visible) in enumerate(zip(masks, visible_past)):
        visible = np.asarray(visible, bool)
        if visible.shape != mask.shape:
            raise ValueError("visible support shape mismatch")
        area = int(visible.sum())
        retained = float((visible & mask).sum() / area) if area else None
        row = {"seed_key": seed_keys[index], "raw_pixels": int(mask.sum()),
               "visible_past_pixels": area, "retention": retained}
        if area < 100 or retained >= .5:
            accepted.append(index)
            row["decision"] = "ABSTAIN_KEEP_RAW" if area < 100 else "ACCEPT"
        else:
            candidates = []
            for group in groups:
                obj = cropformer == group
                union = int((obj | mask).sum())
                iou = float((obj & mask).sum() / union) if union else 0.
                candidates.append((-iou, spatial_key(obj), group))
            candidates.sort()
            if candidates and -candidates[0][0] >= .3:
                iou, _, group = candidates[0]
                replacements.append((-iou, group))
                row.update(decision="REPLACE", cropformer_group=group, replacement_iou=-iou)
            else:
                row["decision"] = "REJECT_NO_REPLACEMENT"
        tracks.append(row)
    raw, raw_remap = _raster(logits, seed_keys, list(range(len(logits))), [], discovery, cropformer)
    geom, geom_remap = _raster(logits, seed_keys, accepted, replacements, discovery, cropformer)
    return raw, geom, {"discovery_groups": discovery, "geom_discovery_groups": discovery,
                       "tracks": tracks, "raw_remap": raw_remap, "geom_remap": geom_remap,
                       "raw_union_pixels": int(raw_union.sum())}


def run_paired_frontend(job):
    import torch
    import sam2
    from sam2.build_sam import build_sam2_video_predictor

    root, capture_path = Path(job["output_root"]), Path(job["capture_manifest"])
    root.mkdir(parents=True, exist_ok=True)
    capture = json.loads(capture_path.read_text())
    frames = capture["frames"]
    if job.get("preflight"):
        frames = frames[:3]
    index = InputIndex()
    index.identity(Path(__file__))
    index.identity(capture_path)
    index.identity(job["checkpoint"]["path"], job["checkpoint"])
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=job["sam2_repo"], text=True).strip()
    if actual_commit != job["sam2_commit"]:
        raise ValueError("SAM2 repository pin changed")
    if Path(sam2.__file__).resolve().parent != (Path(job["sam2_repo"]) / "sam2").resolve():
        raise ValueError("SAM2 worker loaded a different source tree")
    counters = {"physical_image_encodings": 0, "physical_encoder_calls": 0}
    receipt = {"status": "RUNNING", "scene": capture["scene_id"], "frames": [],
               "code_commit": actual_commit, "precision": "bfloat16_autocast",
               "chunk_valid_frames": 5, "derived_jpeg": {"quality": 100, "subsampling": 0},
               "source_capture": str(capture_path), "output_root": str(root), "counters": counters,
               "optional_connected_components_extension": "AVAILABLE" if importlib.util.find_spec("sam2._C") else "UNAVAILABLE_OFFICIAL_FALLBACK",
               "neural_predictions_shared": True, "standalone_sam_cost_per_arm": "FULL_SHARED_ACTUAL_COST"}
    start = time.monotonic()
    with exclusive_lock(job["gpu_lock"]), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            model_start = time.monotonic()
            predictor = build_sam2_video_predictor(job["sam2_config"], job["checkpoint"]["path"], device="cuda")
            predictor.eval()
            receipt["model_load_seconds"] = time.monotonic() - model_start
            def count_encoder(module, inputs, output):
                counters["physical_encoder_calls"] += 1
                counters["physical_image_encodings"] += int(inputs[0].shape[0])
            handle = predictor.image_encoder.register_forward_hook(count_encoder)
            for chunk_index, first in enumerate(range(0, len(frames), 5)):
                chunk = frames[first:first + 5]
                chunk_root = root / "chunks" / f"{chunk_index:03d}"
                jpeg_root = chunk_root / "jpeg"
                jpeg_root.mkdir(parents=True, exist_ok=True)
                for t, frame in enumerate(chunk):
                    rgb_path = capture_path.parent / frame["rgb_path"]
                    index.identity(rgb_path, {"sha256": frame["rgb_sha256"]})
                    Image.open(rgb_path).convert("RGB").save(jpeg_root / f"{t:05d}.jpg", quality=100, subsampling=0)
                    index.identity(jpeg_root / f"{t:05d}.jpg")
                first_crop = np.asarray(Image.open(capture_path.parent / chunk[0]["panoptic_path"]))
                seeds = sorted((int(k) for k in np.unique(first_crop) if k > 0),
                               key=lambda k: spatial_key(first_crop == k))
                seed_keys = {i + 1: spatial_key(first_crop == group) for i, group in enumerate(seeds)}
                state = None
                if seeds:
                    state = predictor.init_state(str(jpeg_root), offload_video_to_cpu=True, offload_state_to_cpu=True)
                    for obj_id, group in enumerate(seeds, 1):
                        predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=first_crop == group)
                previous_masks, previous_frame, previous_depth = {}, None, None
                for t, frame in enumerate(chunk):
                    frame_start = time.monotonic()
                    crop_path = capture_path.parent / frame["panoptic_path"]
                    depth_path = capture_path.parent / frame["depth_path"]
                    index.identity(crop_path, {"sha256": frame["panoptic_sha256"]})
                    index.identity(depth_path, {"sha256": frame["depth_sha256"]})
                    crop = np.asarray(Image.open(crop_path))
                    with np.load(depth_path, allow_pickle=False) as arrays:
                        depth = arrays["depth_m"]
                    diagnostic = {"state": "EXACT_CHUNK_FIRST" if seeds else "NO_SEED", "tracks": []}
                    if not seeds or t == 0:
                        raw = geom = crop.copy()
                        ids = list(seed_keys)
                        masks = np.stack([first_crop == k for k in seeds]) if seeds else np.empty((0,) + crop.shape, bool)
                    else:
                        ids, logits = current_output(predictor, state, t)
                        visible = [warp_visible_support(previous_masks[obj_id], previous_depth,
                                   previous_frame["pose_c2w"], previous_frame["intrinsics"], depth,
                                   frame["pose_c2w"], frame["intrinsics"]) for obj_id in ids]
                        raw, geom, diagnostic = paired_rasters(logits, [seed_keys[k] for k in ids], crop, visible)
                        masks = logits > 0
                    previous_masks = {obj_id: masks[i] for i, obj_id in enumerate(ids)}
                    previous_frame, previous_depth = frame, depth
                    output_paths = {}
                    for arm, raster in [("raw", raw), ("geom", geom)]:
                        path = root / arm / f"{frame['frame_id']:06d}.png"
                        path.parent.mkdir(exist_ok=True)
                        Image.fromarray(raster.astype(np.uint16)).save(path)
                        output_paths[arm] = index.identity(path)
                    track_path = chunk_root / f"{frame['frame_id']:06d}_tracks.npz"
                    _write_npz(track_path, {"track_ids": np.asarray(ids, np.int64),
                               "shape": np.asarray(crop.shape, np.int64),
                               "raw_masks_bits": np.packbits(masks.reshape(len(ids), -1), axis=1) if ids else np.empty((0, 0), np.uint8)})
                    torch.cuda.synchronize()
                    receipt["frames"].append({"frame_id": frame["frame_id"], "chunk": chunk_index,
                        "current_index": t, "outputs": output_paths, "tracks": index.identity(track_path),
                        "diagnostic": diagnostic, "elapsed_seconds": time.monotonic() - frame_start})
                    atomic_write_json(root / "progress.json", receipt)
                if state is not None:
                    predictor.reset_state(state)
                del state
            handle.remove()
            receipt["status"] = "COMPLETE"
        except BaseException as exc:
            receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - start
            receipt["inputs_and_outputs"] = index.entries()
            receipt["identity"] = canonical_digest({"job": job, "artifacts": index.entries()})
            receipt["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            atomic_write_json(root / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run_paired_frontend(json.loads(Path(parser.parse_args().job).read_text()))
