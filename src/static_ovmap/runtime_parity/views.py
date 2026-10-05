"""Same scientific requests, with honest execution IDs and call-local input reuse."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from static_ovmap.cvpr_compact.projected_views import (
    PROJECTOR, FullSceneProjector, ProjectedRegionRequest, _verified_identity,
    load_projected_request,
)
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.binding import read
from static_ovmap.recovery_wave2.recovery_registry import build_registry

from .binding import file_identity
from .kernels import ExactProjector, VARIANTS, retain_top3


SCIENCE_FIELDS = ("scene","raw_owner","final_geometry_digest","frame_id","intrinsics_identity",
    "pose_identity","image_sha256","depth_sha256","target_mask_sha256","canonical_bbox",
    "legacy_native_bbox","visible_pixels","selection_rank")


def scientific_key(request, valid_ids, model_identity, image_shape):
    return canonical_digest({"request":{key:request[key] for key in SCIENCE_FIELDS},
        "mask_shape":list(image_shape),"mask_dtype":"bool","valid_ids":list(map(int,valid_ids)),
        "model_identity":model_identity,"preprocess":"original_image_tensor_800_1333_RGB_FP32",
        "pooling":"PROJECTED_FC_EMPTY_AREA_FALLBACK_V2"})


def build_views(inputs, output_root, implementation, index, stages):
    if implementation not in VARIANTS:
        raise ValueError("unknown recovery execution implementation")
    number = VARIANTS.index(implementation)
    root, capture_path = Path(output_root),Path(inputs.data["capture_manifest"])
    with stages.span("registry_geometry"):
        index.identity(capture_path)
        capture = read(capture_path)
        _verified_identity(capture)
        if not inputs.baseline.locked or capture["scene_id"] != inputs.scene:
            raise ValueError("visibility requires the fixed same-scene Native geometry")
        frames = capture["frames"]
        frame_ids = [int(frame["frame_id"]) for frame in frames]
        if (len(set(frame_ids)) != len(frame_ids) or frame_ids != capture["completed_frame_ids"]
                or not set(frame_ids) <= set(capture["scheduled_frame_ids"])):
            raise ValueError("visibility changed the original completed frame schedule")
        surface = capture_path.parent/capture["surface"]["path"]
        index.identity(surface,capture["surface"])
        if number == 0:
            with np.load(surface,allow_pickle=False) as arrays:
                xyz,faces,raw = arrays["surface_xyz"],arrays["surface_faces"],arrays["original_owner"]
            stages.counters["mesh_reopens"] += 1
        else:
            xyz,faces,raw = inputs.xyz,inputs.faces,inputs.raw
        if (_array_digest(xyz) != inputs.baseline.geometry.xyz_sha256
                or _array_digest(faces) != inputs.baseline.geometry.faces_sha256):
            raise ValueError("visibility geometry differs from its actual bound anchor")
        registry = build_registry(xyz,raw,inputs.baseline.owner_ids)
        geometry = canonical_digest({"xyz":_array_digest(xyz),"faces":_array_digest(faces),
            "raw":_array_digest(raw),"tsdf":inputs.baseline.geometry.tsdf_sha256})
        candidates = [row["raw_owner"] for row in registry["candidates"]]
        stages.counters.update(raw_candidates=len(candidates),scheduled_frames=len(capture["scheduled_frame_ids"]),completed_frames=len(frames))
        producer = {"views":file_identity(__file__)["sha256"],
            "kernels":file_identity(Path(__file__).with_name("kernels.py"))["sha256"],"implementation":implementation}
        projector_identity = canonical_digest({"config":PROJECTOR,**producer})
        projector = None
        if number == 0 or candidates:
            projector = FullSceneProjector(xyz,faces,raw) if number == 0 else ExactProjector(xyz,faces,raw,roi=number>=2)
            stages.counters["BVH_builds"] += 1
            if number:
                projector.prepare_bounds(candidates)
        retained = {owner:[] for owner in candidates}
    diagnostics = []
    with stages.span("view_search"):
        for frame in frames:
            rgb_path,depth_path = (capture_path.parent/frame[name] for name in ("rgb_path","depth_path"))
            index.identity(rgb_path,{"sha256":frame["rgb_sha256"]})
            index.identity(depth_path,{"sha256":frame["depth_sha256"]})
            with np.load(depth_path,allow_pickle=False) as arrays:
                depth = arrays["depth_m"]
            if list(depth.shape) != frame["image_size_hw"]:
                raise ValueError("original captured depth shape changed")
            stages.counters["full_image_pixels"] += depth.size
            stages.counters["considered_candidate_views"] += len(candidates)
            if projector is None:
                diagnostics.append({"frame_id":frame["frame_id"],"counts":{},"diagnostic_scope":"queried_pixels",
                    "total_first_hits":None,"valid_measured_depth_pixels":None,"depth_consistent_first_hits":None,
                    "queried_rays":0,"query_reason":"ZERO_CANDIDATES"})
                stages.counters["frames_skipped_safe"] += 1
                continue
            result = projector.project_frame(frame["intrinsics"],frame["pose_c2w"],depth,candidates)
            stages.counters["queried_rays"] += result.get("queried_rays",depth.size)
            stages.counters["frames_skipped_safe"] += int(result.get("queried_rays",depth.size)==0)
            omit = {"masks","owners","mask_for_owner","queried_pixel_ids","queried_owners"}
            diagnostics.append({"frame_id":frame["frame_id"],**{key:value for key,value in result.items() if key not in omit}})
            for owner,row in result["counts"].items():
                if not row["admissible"]:
                    continue
                stages.counters["admissible_views"] += 1
                def materialize():
                    mask = result["masks"][owner] if number == 0 else result["mask_for_owner"](owner)
                    bbox = row["canonical_bbox"]
                    request = ProjectedRegionRequest(inputs.scene,geometry,owner,int(frame["frame_id"]),
                        _array_digest(np.asarray(frame["intrinsics"],np.float64)),
                        _array_digest(np.asarray(frame["pose_c2w"],np.float64)),frame["rgb_sha256"],frame["depth_sha256"],
                        _array_digest(mask),tuple(bbox),(bbox[0],bbox[1],bbox[2]-1,bbox[3]-1),
                        int(mask.sum()),0,projector_identity)
                    stages.counters["materialized_masks"] += 1
                    stages.counters["packed_masks"] += 1
                    return request,np.packbits(mask.reshape(-1),bitorder="little"),mask.shape
                values = retained[owner]
                if number:
                    retain_top3(values,row["depth_consistent_pixels"],int(frame["frame_id"]),materialize)
                else:
                    values.append(materialize())
                    values.sort(key=lambda value:(-value[0].visible_pixels,value[0].frame_id,value[0].target_mask_sha256))
                    del values[3:]
    with stages.span("export_bookkeeping"):
        requests,views = {},{}
        for owner,values in retained.items():
            views["owner:"+str(owner)] = []
            for rank,(request,packed,shape) in enumerate(values):
                request = replace(request,selection_rank=rank)
                rid = request.request_id
                path = root/"masks"/(rid+".npz")
                _write_npz(path,{"packed":packed,"shape":np.asarray(shape,np.int64)})
                requests[rid] = {**request.to_dict(),"mask":index.identity(path),"capture_manifest":str(capture_path),
                    "crop_convention":"native_global_bbox_union_exclusive_upper_v1"}
                views["owner:"+str(owner)].append(rid)
        manifest = {"schema":"ovimap-independent-projected-views-v1","scene_id":inputs.scene,
            "capture_manifest":str(capture_path),"geometry_identity":geometry,"registry_identity":registry["identity"],
            "requests":requests,"views":views,"g1":{owner:rids[:1] for owner,rids in views.items()},
            "projector":PROJECTOR,"projector_identity":projector_identity,"execution":producer,
            "GT_input":False,"historical_request_gating":False,"failed_view_replacement":False}
        manifest["identity"] = canonical_digest(manifest)
        atomic_write_json(root/"manifest.json",manifest)
        atomic_write_json(root/"registry.json",registry)
        atomic_write_json(root/"frame_diagnostics.json",{"frames":diagnostics})
        _write_npz(root/"primitive_mapping.npz",{
            "original_triangle_indices":projector.original_triangle_indices if projector is not None else np.empty(0,np.int64),
            "excluded_triangle_indices":projector.excluded_triangle_indices if projector is not None else np.empty(0,np.int64)})
    return registry,manifest


class CallLoader:
    def __init__(self, manifest_path, index, stages, *, reuse):
        self.path,self.index,self.stages,self.reuse = Path(manifest_path),index,stages,reuse
        self.images = {}
        if reuse:
            self.manifest = read(self.path)
            _verified_identity(self.manifest)
            self.capture_path = Path(self.manifest["capture_manifest"])
            self.capture = read(self.capture_path)
            _verified_identity(self.capture)
            self.frames = {int(frame["frame_id"]):frame for frame in self.capture["frames"]}

    def __call__(self, rid):
        with self.stages.span("rgb_preprocess"):
            if not self.reuse:
                value = load_projected_request(self.path,rid,index=self.index)
                self.stages.counters["RGB_decodes"] += 1
                return value
            request = self.manifest["requests"][rid]
            fields = {name:request[name] for name in ProjectedRegionRequest.__dataclass_fields__}
            if ProjectedRegionRequest(**fields).request_id != rid or request["request_id"] != rid:
                raise ValueError("projected request content identity changed")
            frame = self.frames[int(request["frame_id"])]
            if request["scene"] != self.capture["scene_id"] or request["scene"] != self.manifest["scene_id"]:
                raise ValueError("projected request belongs to another scene")
            for field,camera in (("intrinsics_identity","intrinsics"),("pose_identity","pose_c2w")):
                if _array_digest(np.asarray(frame[camera],np.float64)) != request[field]:
                    raise ValueError("captured camera identity changed")
            self.index.identity(request["mask"]["path"],request["mask"])
            self.index.identity(self.capture_path.parent/frame["rgb_path"],{"sha256":request["image_sha256"]})
            self.index.identity(self.capture_path.parent/frame["depth_path"],{"sha256":request["depth_sha256"]})
            with np.load(request["mask"]["path"],allow_pickle=False) as arrays:
                shape = tuple(map(int,arrays["shape"]))
                mask = np.unpackbits(arrays["packed"],bitorder="little",count=int(np.prod(shape))).reshape(shape).astype(bool)
            if (_array_digest(mask) != request["target_mask_sha256"] or int(mask.sum()) != request["visible_pixels"]
                    or list(shape) != frame["image_size_hw"]):
                raise ValueError("full-resolution projected mask changed")
            image_key = (request["image_sha256"],tuple(shape))
            if image_key not in self.images:
                image = cv2.imread(str(self.capture_path.parent/frame["rgb_path"]),cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype != np.uint8 or image.shape != (*shape,3):
                    raise ValueError("original RGB must remain aligned uint8 HWC3")
                self.images[image_key] = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                self.stages.counters["RGB_decodes"] += 1
            return {"image":self.images[image_key],"target":mask,"union":mask.copy(),
                "bbox":request["legacy_native_bbox"],"request":request}
