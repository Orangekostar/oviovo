"""GT-free visibility and actual final-mask requests on the fixed native surface."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .native_capture import RegionRequest, _array_digest, _write_npz
from .region_evidence import stable_target_subset
from .scannet_runtime import reusable_job


def project_visible_rows(surface_xyz, pose_c2w, intrinsics, depth_m, *, depth_tolerance=.05):
    """One nearest depth-consistent source row per pixel, with source-row ties."""
    xyz, pose, camera_matrix, depth = [np.asarray(value, np.float64)
                                     for value in (surface_xyz, pose_c2w, intrinsics, depth_m)]
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or pose.shape != (4, 4) or camera_matrix.shape != (3, 3) or depth.ndim != 2:
        raise ValueError("static visibility inputs do not align")
    if not all(np.isfinite(value).all() for value in (xyz, pose, camera_matrix)):
        raise ValueError("static surface and camera inputs must be finite")
    world_to_camera = np.linalg.inv(pose)
    camera = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    columns = np.rint(camera_matrix[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + camera_matrix[0, 2]).astype(np.int64)
    rows = np.rint(camera_matrix[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + camera_matrix[1, 2]).astype(np.int64)
    inside = (z > 0) & (rows >= 0) & (rows < depth.shape[0]) & (columns >= 0) & (columns < depth.shape[1])
    candidates = np.flatnonzero(inside)
    measured = depth[rows[candidates], columns[candidates]]
    candidates = candidates[np.isfinite(measured) & (measured > 0) & (np.abs(z[candidates] - measured) <= depth_tolerance)]
    if not len(candidates):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    pixels = rows[candidates] * depth.shape[1] + columns[candidates]
    order = np.lexsort((candidates, z[candidates], pixels))
    sorted_pixels = pixels[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    return candidates[order[first]], sorted_pixels[first]


def static_region(owner_raster, local_entities, owner, frame, source_map_version, parent_ids, rank):
    """Construct the final mask's own ROI; degenerate masks remain unavailable."""
    target = np.asarray(owner_raster) == owner
    local_entities = np.asarray(local_entities)
    if local_entities.shape != target.shape or target.ndim != 2:
        raise ValueError("final mask and original local entities must align")
    y, x = np.nonzero(target)
    if not len(x) or x.min() == x.max() or y.min() == y.max():
        return None
    entity_ids, counts = np.unique(local_entities[target], return_counts=True)
    local = local_entities == entity_ids[np.argmax(counts)]
    union = target | local
    target_id = f"owner:{int(owner)}"
    request = RegionRequest(frame["scene_id"], int(frame["frame_id"]), target_id,
        tuple(f"parent:{parent}" for parent in sorted(parent_ids)) + (target_id,), source_map_version,
        _array_digest(target), (int(x.min()), int(y.min()), int(x.max()), int(y.max())),
        _array_digest(union), int(target.sum()), "static_final_mask_native_union_exclusive_upper_v1",
        rank, frame["rgb_sha256"])
    return request, target, union


def visibility_frame(capture_path, frame, xyz, xyz_identity, output):
    capture_path, output = Path(capture_path), Path(output)
    depth_path = capture_path.parent / frame["depth_path"]
    if sha256_file(depth_path) != frame["depth_sha256"]:
        raise ValueError("captured visibility depth changed")
    inputs = [file_identity(path) for path in (capture_path, depth_path, Path(__file__))]
    identity = canonical_digest({"inputs": inputs, "xyz_identity": xyz_identity,
        "frame_id": frame["frame_id"], "pose": frame["pose_c2w"], "intrinsics": frame["intrinsics"], "depth_tolerance": .05})
    receipt_path = output / f"{frame['frame_id']:06d}.json"
    arrays_path = receipt_path.with_suffix(".npz")
    if not reusable_job(receipt_path, identity):
        if receipt_path.exists():
            raise ValueError("static visibility changed; choose a new output root")
        with np.load(depth_path, allow_pickle=False) as arrays:
            source, pixels = project_visible_rows(xyz, frame["pose_c2w"], frame["intrinsics"], arrays["depth_m"])
        _write_npz(arrays_path, {"source_rows": source, "pixel_indices": pixels})
        atomic_write_json(receipt_path, {"status": "COMPLETE", "input_identity": identity,
            "inputs": inputs, "outputs": [file_identity(arrays_path)], "visible_pixels": len(source)})
    with np.load(arrays_path, allow_pickle=False) as arrays:
        return arrays["source_rows"], arrays["pixel_indices"]


def prepare_static_manifest(data, ownership, ancestry, output):
    """Select <=128 masks and <=3 views from all completed scheduled frames."""
    output = Path(output)
    owners = np.asarray(ownership, np.int64)
    native = data["native"]
    if not native.locked or owners.shape != native.owner_ids.shape or np.any(owners < 0):
        raise ValueError("static reread requires aligned final ownership and locked N0")
    if (np.any(owners[native.owner_ids == 0] != 0)
            or _array_digest(data["surface"]["surface_xyz"]) != native.geometry.xyz_sha256):
        raise ValueError("static reread changed owner0 support or source coordinates")
    capture = data["capture"]
    source_map = canonical_digest({"capture": capture["identity"], "geometry": native.geometry.to_dict(),
                                   "owner_ids": _array_digest(owners)})
    inputs = [file_identity(path) for path in (data["capture_path"], data["native_manifest_path"], Path(__file__))]
    identity = canonical_digest({"inputs": inputs, "source_map": source_map,
                                 "ancestry": {str(owner): list(parents) for owner, parents in ancestry.items()}})
    receipt_path, manifest_path = output / "receipt.json", output / "manifest.json"
    if reusable_job(receipt_path, identity):
        return manifest_path
    if receipt_path.exists():
        raise ValueError("completed final-mask requests changed; choose a new output root")
    selected, excluded = stable_target_subset(native.scene_id, [f"owner:{owner}" for owner in sorted(set(owners.tolist()) - {0})])
    chosen = {int(target.split(":")[1]): [] for target in selected}
    unavailable = {str(owner): {"no_visible_pixels": 0, "degenerate_bbox": 0} for owner in chosen}
    visibility_root = data["output"] / "surface_visibility"
    for frame in sorted(capture["frames"], key=lambda row: row["frame_id"]):
        source, pixels = visibility_frame(data["capture_path"], frame, data["surface"]["surface_xyz"],
                                          native.geometry.xyz_sha256, visibility_root)
        raster = np.zeros(frame["image_size_hw"], np.int64)
        raster.flat[pixels] = owners[source]
        local_path = Path(data["capture_path"]).parent / frame["panoptic_path"]
        if sha256_file(local_path) != frame["panoptic_sha256"]:
            raise ValueError("captured local entities changed")
        local = np.asarray(Image.open(local_path))
        visible, counts = np.unique(raster, return_counts=True)
        count_by_owner = dict(zip(map(int, visible), map(int, counts), strict=True))
        for owner, current in chosen.items():
            count = count_by_owner.get(owner, 0)
            if not count:
                unavailable[str(owner)]["no_visible_pixels"] += 1
                continue
            # Frames are ascending; an equal-area later frame cannot displace
            # an earlier retained view under the frozen tie rule.
            if len(current) == 3 and count <= current[-1][0].visible_target_pixels:
                continue
            candidate = static_region(raster, local, owner, frame, source_map, ancestry.get(owner, (owner,)), 0)
            if candidate is None:
                unavailable[str(owner)]["degenerate_bbox"] += 1
                continue
            current.append(candidate)
            current.sort(key=lambda row: (-row[0].visible_target_pixels, row[0].frame_id, row[0].request_id))
            del current[3:]
    requests, views, outputs = {}, {}, []
    for owner, current in chosen.items():
        view_ids = []
        for rank, (original, target, union) in enumerate(current):
            request = replace(original, requested_view_rank=rank)
            arrays_path = output / "masks" / f"{request.request_id}.npz"
            _write_npz(arrays_path, {"target": target, "union": union})
            requests[request.request_id] = {"request": request.to_dict(), "masks": file_identity(arrays_path)}
            view_ids.append(request.request_id)
            outputs.append(file_identity(arrays_path))
        views[f"owner:{owner}"] = view_ids
    manifest = {"artifact_type": "OVIMAP_STATIC_FINAL_MASK_REQUESTS", "scene_id": native.scene_id,
        "capture": file_identity(data["capture_path"]), "source_map_version": source_map,
        "native_record_key": native.record_key, "selected_targets": list(selected), "excluded_targets": list(excluded),
        "requests": requests, "views": views, "unavailable": unavailable, "GT_input": False,
        "view_frame_ids": [frame["frame_id"] for frame in capture["frames"]]}
    manifest["identity"] = canonical_digest(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(receipt_path, {"status": "COMPLETE", "input_identity": identity,
        "inputs": inputs, "outputs": [*outputs, file_identity(manifest_path)]})
    return manifest_path
