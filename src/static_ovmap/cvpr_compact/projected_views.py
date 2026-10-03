"""Full predicted-mesh visibility and independent, geometry-ranked requests."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time

import cv2
import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.recovery_wave2.recovery_registry import build_registry


PROJECTOR = {"schema": "full-predicted-mesh-camera-z-v1", "threads": 4,
             "ray_batch_pixels_max": 65536, "absolute_depth_tolerance_m": .02,
             "relative_depth_tolerance": .02, "minimum_positive_depth_m": 1e-6,
             "minimum_visible_pixels": 100, "minimum_bbox_extent": 2,
             "pixel_coordinates": "integer", "direction_normalized": False,
             "face_owner": "positive_only_if_all_three_raw_vertex_owners_agree"}


def validate_camera(K, pose):
    K, pose = np.asarray(K, np.float64), np.asarray(pose, np.float64)
    if (K.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(K).all()
            or not np.isfinite(pose).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, rtol=0, atol=1e-5)):
        raise ValueError("camera requires original finite intrinsics and a rigid c2w pose")
    return K, pose


def camera_rays(K, pose, image_size, start, count):
    K, pose = validate_camera(K, pose)
    height, width = map(int, image_size)
    if min(height, width) <= 0 or start < 0 or count < 0 or start + count > height * width:
        raise ValueError("invalid camera ray pixel range")
    pixels = np.arange(start, start + count, dtype=np.int64)
    q = np.column_stack((pixels % width, pixels // width, np.ones(count))) @ np.linalg.inv(K).T
    rays = np.empty((count, 6), np.float32)
    rays[:, :3] = pose[:3, 3]
    rays[:, 3:] = q @ pose[:3, :3].T
    return rays


class FullSceneProjector:
    def __init__(self, xyz, faces, raw, *, batch_pixels=65536):
        import open3d as o3d

        xyz, faces, raw = np.asarray(xyz), np.asarray(faces), np.asarray(raw)
        if not 1 <= batch_pixels <= PROJECTOR["ray_batch_pixels_max"]:
            raise ValueError("ray batch exceeds the fixed maximum")
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or raw.shape != (len(xyz),)
                or not np.isfinite(xyz).all() or not np.issubdtype(raw.dtype, np.integer)
                or np.any(raw < 0)):
            raise ValueError("raw owners and finite source geometry must align")
        if (faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
                or np.any(faces < 0) or np.any(faces >= len(xyz))
                or len(xyz) > np.iinfo(np.uint32).max):
            raise ValueError("source mesh contains invalid triangle indices")
        coordinates = np.ascontiguousarray(xyz, np.float32)
        if not np.isfinite(coordinates).all():
            raise ValueError("source coordinates overflow the required FP32 geometry")
        positive = np.zeros(len(faces), np.int64)
        nondegenerate = np.zeros(len(faces), bool)
        for start in range(0, len(faces), 262144):
            sl = slice(start, start + 262144)
            triangles = coordinates[faces[sl]].astype(np.float64)
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            nondegenerate[sl] = np.any(cross != 0, axis=1)
            face_owners = raw[faces[sl]]
            homogeneous = (face_owners[:, 0] > 0) & np.all(face_owners == face_owners[:, :1], axis=1)
            positive[sl] = np.where(homogeneous, face_owners[:, 0], 0)
        self.original_triangle_indices = np.flatnonzero(nondegenerate)
        self.excluded_triangle_indices = np.flatnonzero(~nondegenerate)
        self.face_owners = positive[nondegenerate]
        self.batch_pixels = int(batch_pixels)
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        if nondegenerate.any():
            self.scene.add_triangles(o3d.core.Tensor(coordinates, dtype=o3d.core.Dtype.Float32),
                o3d.core.Tensor(np.ascontiguousarray(faces[nondegenerate], np.uint32), dtype=o3d.core.Dtype.UInt32))
        self.bounds = {}
        self.xyz, self.raw = coordinates, raw

    def _frustum_aabb(self, owner, K, pose, shape):
        if owner not in self.bounds:
            points = self.xyz[self.raw == owner]
            if not len(points):
                raise ValueError("candidate owner is absent from the final raw geometry")
            lo, hi = points.min(axis=0), points.max(axis=0)
            self.bounds[owner] = np.asarray([[x, y, z] for x in (lo[0], hi[0])
                                           for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        camera = (self.bounds[owner] - pose[:3, 3]) @ pose[:3, :3]
        if camera[:, 2].max() <= 1e-6:
            return False
        if camera[:, 2].min() <= 1e-6:
            return True
        uv = camera @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        height, width = shape
        return bool(uv[:, 0].max() >= 0 and uv[:, 0].min() <= width - 1
                    and uv[:, 1].max() >= 0 and uv[:, 1].min() <= height - 1)

    def project_frame(self, K, pose, depth, candidates):
        import open3d as o3d

        K, pose = validate_camera(K, pose)
        depth = np.asarray(depth)
        if depth.ndim != 2 or depth.dtype != np.float32 or not depth.size:
            raise ValueError("projector requires the captured 2D FP32 depth_m without reconversion")
        candidates = tuple(map(int, candidates))
        if any(owner <= 0 for owner in candidates) or len(set(candidates)) != len(candidates):
            raise ValueError("candidate owners must be unique positive raw IDs")
        owners = np.zeros(depth.size, np.int64)
        hit_owners = np.zeros(depth.size, np.int64)
        total_hits = valid_depth_pixels = consistent_hits = 0
        for start in range(0, depth.size, self.batch_pixels):
            count = min(self.batch_pixels, depth.size - start)
            rays = camera_rays(K, pose, depth.shape, start, count)
            hit = self.scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32), nthreads=4)
            t, primitive = hit["t_hit"].numpy(), hit["primitive_ids"].numpy()
            valid_hit = np.isfinite(t) & (t > 1e-6)
            total_hits += int(valid_hit.sum())
            measured = depth.reshape(-1)[start:start + count]
            valid_depth = np.isfinite(measured) & (measured > 1e-6)
            valid_depth_pixels += int(valid_depth.sum())
            face_owner = np.zeros(count, np.int64)
            if valid_hit.any():
                if np.any(primitive[valid_hit] >= len(self.face_owners)):
                    raise ValueError("raycast primitive mapping left the original mesh")
                face_owner[valid_hit] = self.face_owners[primitive[valid_hit]]
            accepted = (valid_hit & valid_depth
                        & (np.abs(t - measured) <= np.maximum(.02, .02 * measured)))
            consistent_hits += int(accepted.sum())
            hit_owners[start:start + count] = face_owner
            owners[start:start + count] = np.where(accepted, face_owner, 0)
        masks, counts = {}, {}
        for owner in candidates:
            mask = (owners == owner).reshape(depth.shape)
            pixels = int(mask.sum())
            bbox = None
            if pixels:
                y, x = np.nonzero(mask)
                bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
            admissible = pixels >= 100 and bbox[2] - bbox[0] >= 2 and bbox[3] - bbox[1] >= 2
            counts[owner] = {"frustum_aabb_intersection": self._frustum_aabb(owner, K, pose, depth.shape),
                "hit_owner_pixels": int((hit_owners == owner).sum()), "depth_consistent_pixels": pixels,
                "support_eligible_pixels": pixels if admissible else 0, "admissible": bool(admissible),
                "canonical_bbox": bbox}
            if admissible:
                masks[owner] = mask
        return {"masks": masks, "owners": owners.reshape(depth.shape), "counts": counts,
                "total_first_hits": total_hits, "valid_measured_depth_pixels": valid_depth_pixels,
                "depth_consistent_first_hits": consistent_hits}


@dataclass(frozen=True)
class ProjectedRegionRequest:
    scene: str
    final_geometry_digest: str
    raw_owner: int
    frame_id: int
    intrinsics_identity: str
    pose_identity: str
    image_sha256: str
    depth_sha256: str
    target_mask_sha256: str
    canonical_bbox: tuple
    legacy_native_bbox: tuple
    visible_pixels: int
    selection_rank: int
    projector_identity: str
    schema: str = "ovimap-projected-region-request-v1"

    @property
    def request_id(self):
        return canonical_digest(asdict(self))

    def to_dict(self):
        return {**asdict(self), "request_id": self.request_id}


def _verified_identity(value):
    if canonical_digest({k: v for k, v in value.items() if k != "identity"}) != value["identity"]:
        raise ValueError("locked projected-view document content changed")


def build_projected_views(capture_path, baseline, output_root, *, index=None):
    capture_path, root = Path(capture_path).resolve(), Path(output_root).resolve()
    index = index or ConsumptionIndex(root / "input_verifications.json")
    index.identity(capture_path)
    capture = read(capture_path)
    _verified_identity(capture)
    if not baseline.locked or capture["scene_id"] != baseline.scene_id:
        raise ValueError("projected views require the locked same-scene native partition")
    surface_path = capture_path.parent / capture["surface"]["path"]
    index.identity(surface_path, capture["surface"])
    frames = capture["frames"]
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    if (len(set(frame_ids)) != len(frame_ids) or frame_ids != capture["completed_frame_ids"]
            or not set(frame_ids) <= set(capture["scheduled_frame_ids"])):
        raise ValueError("completed camera list differs from the original scheduled capture")
    projector_identity = canonical_digest({"config": PROJECTOR, "producer": index.identity(__file__)["sha256"]})
    input_identity = canonical_digest({"surface": capture["surface"]["sha256"],
        "painted": _array_digest(baseline.owner_ids), "frames": [{k: frame[k] for k in (
            "frame_id", "image_size_hw", "intrinsics", "pose_c2w", "rgb_sha256", "depth_sha256")} for frame in frames],
        "projector": projector_identity, "minimum_residual_source_rows": 100, "candidate_cap": 128})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        previous = read(receipt_path)
        _verified_identity(previous)
        if previous["input_identity"] != input_identity:
            raise ValueError("completed projected-view inputs changed; invalidate affected descendants explicitly")
        for item in previous["outputs"]:
            index.identity(item["path"], item)
        for item in previous["frame_inputs"]:
            index.identity(item["path"], item)
        return previous
    started = time.monotonic()
    with np.load(surface_path, allow_pickle=False) as arrays:
        xyz, faces, raw = arrays["surface_xyz"], arrays["surface_faces"], arrays["original_owner"]
    if (_array_digest(xyz) != baseline.geometry.xyz_sha256
            or _array_digest(faces) != baseline.geometry.faces_sha256):
        raise ValueError("visibility geometry differs from the native anchor")
    registry = build_registry(xyz, raw, baseline.owner_ids)
    geometry = canonical_digest({"xyz": _array_digest(xyz), "faces": _array_digest(faces),
        "raw": _array_digest(raw), "tsdf": baseline.geometry.tsdf_sha256})
    candidates = [row["raw_owner"] for row in registry["candidates"]]
    projector = FullSceneProjector(xyz, faces, raw)
    retained = {owner: [] for owner in candidates}
    frame_diagnostics, frame_inputs = [], []
    for frame_number, frame in enumerate(frames):
        rgb_path, depth_path = (capture_path.parent / frame[name] for name in ("rgb_path", "depth_path"))
        frame_inputs.extend((index.identity(rgb_path, {"sha256": frame["rgb_sha256"]}),
                             index.identity(depth_path, {"sha256": frame["depth_sha256"]})))
        with np.load(depth_path, allow_pickle=False) as arrays:
            depth = arrays["depth_m"]
        if list(depth.shape) != frame["image_size_hw"]:
            raise ValueError("captured depth shape differs from the completed camera")
        result = projector.project_frame(frame["intrinsics"], frame["pose_c2w"], depth, candidates)
        frame_diagnostics.append({"frame_id": frame["frame_id"], **{k: v for k, v in result.items() if k not in ("masks", "owners")}})
        for owner, mask in result["masks"].items():
            bbox = result["counts"][owner]["canonical_bbox"]
            request = ProjectedRegionRequest(capture["scene_id"], geometry, owner, int(frame["frame_id"]),
                _array_digest(np.asarray(frame["intrinsics"], np.float64)),
                _array_digest(np.asarray(frame["pose_c2w"], np.float64)), frame["rgb_sha256"], frame["depth_sha256"],
                _array_digest(mask), tuple(bbox), (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1),
                int(mask.sum()), 0, projector_identity)
            values = retained[owner]
            values.append((request, np.packbits(mask.reshape(-1), bitorder="little"), mask.shape))
            values.sort(key=lambda value: (-value[0].visible_pixels, value[0].frame_id, value[0].target_mask_sha256))
            del values[3:]
        if (frame_number + 1) % 25 == 0 or frame_number + 1 == len(frames):
            print(f"Projected {capture['scene_id']}: {frame_number + 1}/{len(frames)} completed cameras", flush=True)
    requests, views, outputs = {}, {}, []
    for owner, values in retained.items():
        views[f"owner:{owner}"] = []
        for rank, (request, packed, shape) in enumerate(values):
            request = replace(request, selection_rank=rank)
            rid = request.request_id
            mask_path = root / "masks" / (rid + ".npz")
            _write_npz(mask_path, {"packed": packed, "shape": np.asarray(shape, np.int64)})
            mask_identity = index.identity(mask_path)
            outputs.append(mask_identity)
            requests[rid] = {**request.to_dict(), "mask": mask_identity, "capture_manifest": str(capture_path),
                             "crop_convention": "native_global_bbox_union_exclusive_upper_v1"}
            views[f"owner:{owner}"].append(rid)
    manifest = {"schema": "ovimap-independent-projected-views-v1", "scene_id": capture["scene_id"],
        "input_identity": input_identity, "capture_manifest": str(capture_path),
        "geometry_identity": geometry, "registry_identity": registry["identity"],
        "requests": requests, "views": views, "g1": {k: v[:1] for k, v in views.items()},
        "projector": PROJECTOR, "projector_identity": projector_identity, "GT_input": False,
        "historical_request_gating": False, "failed_view_replacement": False}
    manifest["identity"] = canonical_digest(manifest)
    for path, value in ((root / "registry.json", registry), (root / "manifest.json", manifest),
                        (root / "frame_diagnostics.json", {"frames": frame_diagnostics}),
                        (root / "primitive_mapping.npz", None)):
        if value is None:
            _write_npz(path, {"original_triangle_indices": projector.original_triangle_indices,
                             "excluded_triangle_indices": projector.excluded_triangle_indices})
        else:
            atomic_write_json(path, value)
        outputs.append(index.identity(path))
    receipt = {"status": "COMPLETE", "scene": capture["scene_id"], "input_identity": input_identity,
        "manifest": str(root / "manifest.json"), "registry": str(root / "registry.json"),
        "candidate_count": len(candidates), "candidates_with_views": sum(bool(v) for v in views.values()),
        "request_count": len(requests), "completed_frame_count": len(frames), "outputs": outputs,
        "frame_inputs": frame_inputs, "inputs": index.entries(), "GT_input": False,
        "new_neural_inference": 0, "elapsed_seconds": time.monotonic() - started}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(receipt_path, receipt)
    index.write_memo(root / "input_verifications.json")
    return receipt


def load_projected_request(manifest_path, request_id, *, index=None):
    manifest = read(manifest_path)
    _verified_identity(manifest)
    request = manifest["requests"][request_id]
    if request["schema"] != "ovimap-projected-region-request-v1":
        raise ValueError("the projected loader requires the independent request schema")
    fields = {name: request[name] for name in ProjectedRegionRequest.__dataclass_fields__}
    for name in ("canonical_bbox", "legacy_native_bbox"):
        fields[name] = tuple(fields[name])
    if ProjectedRegionRequest(**fields).request_id != request_id or request["request_id"] != request_id:
        raise ValueError("projected request content identity changed")
    capture_path = Path(request["capture_manifest"])
    capture = read(capture_path)
    frame = next(row for row in capture["frames"] if row["frame_id"] == request["frame_id"])
    if capture["scene_id"] != request["scene"] or request["scene"] != manifest["scene_id"]:
        raise ValueError("projected request belongs to a different captured scene")
    if (_array_digest(np.asarray(frame["intrinsics"], np.float64)) != request["intrinsics_identity"]
            or _array_digest(np.asarray(frame["pose_c2w"], np.float64)) != request["pose_identity"]):
        raise ValueError("captured projected camera identity changed")
    index = index or ConsumptionIndex()
    index.identity(request["mask"]["path"], request["mask"])
    index.identity(capture_path.parent / frame["rgb_path"], {"sha256": request["image_sha256"]})
    index.identity(capture_path.parent / frame["depth_path"], {"sha256": request["depth_sha256"]})
    with np.load(request["mask"]["path"], allow_pickle=False) as arrays:
        shape = tuple(map(int, arrays["shape"]))
        mask = np.unpackbits(arrays["packed"], bitorder="little", count=int(np.prod(shape))).reshape(shape).astype(bool)
    if (_array_digest(mask) != request["target_mask_sha256"] or int(mask.sum()) != request["visible_pixels"]
            or list(shape) != frame["image_size_hw"]):
        raise ValueError("projected mask readback differs from its selected evidence")
    image = cv2.imread(str(capture_path.parent / frame["rgb_path"]), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (*shape, 3):
        raise ValueError("captured aligned RGB is not an unchanged uint8 three-channel image")
    return {"image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "target": mask,
            "union": mask.copy(), "bbox": request["legacy_native_bbox"], "request": request}
