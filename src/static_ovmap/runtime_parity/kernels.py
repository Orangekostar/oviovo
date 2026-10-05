"""Exact ray arithmetic and conservative queries against the complete parent BVH."""

from itertools import product

import numpy as np

from static_ovmap.cvpr_compact.projected_views import FullSceneProjector, validate_camera


VARIANTS = ("R0_REFERENCE", "R1_IO_EXACT", "R2_ROI_EXACT", "R3_RUNTIME_EXACT")


class PreparedCamera:
    def __init__(self, K, pose, shape):
        self.K, self.pose = validate_camera(K, pose)
        self.shape = tuple(map(int, shape))
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("invalid camera image shape")
        self.inverse = np.linalg.inv(self.K)

    def rays(self, pixels):
        pixels = np.asarray(pixels)
        if (pixels.ndim != 1 or not np.issubdtype(pixels.dtype, np.integer)
                or np.any(pixels < 0) or np.any(pixels >= np.prod(self.shape))):
            raise ValueError("invalid original camera pixel indices")
        width = self.shape[1]
        q = np.column_stack((pixels % width, pixels // width, np.ones(len(pixels)))) @ self.inverse.T
        rays = np.empty((len(pixels), 6), np.float32)
        rays[:, :3] = self.pose[:3, 3]
        rays[:, 3:] = q @ self.pose[:3, :3].T
        return rays

    def query_union(self, boxes):
        height, width = self.shape
        full = lambda: (np.arange(height * width, dtype=np.int64), "UNCERTAIN_FULL_FRAME")
        if np.linalg.cond(self.K) > 1e8 or np.linalg.cond(self.pose[:3,:3]) > 1.0001:
            return full()
        union = np.zeros(self.shape, bool)
        # Ray origins are FP32; the parent pose is only approximately orthogonal.
        origin = self.pose[:3,3].astype(np.float32).astype(np.float64)
        inverse_rotation = np.linalg.inv(self.pose[:3,:3]).T
        for bounds in boxes:
            bounds = np.asarray(bounds, np.float64)
            if bounds.shape == (2,3):
                bounds = np.array(list(product(*zip(bounds[0], bounds[1]))), np.float64)
            if bounds.shape != (8,3) or not np.isfinite(bounds).all():
                return full()
            lo, hi = bounds.min(axis=0), bounds.max(axis=0)
            if np.all(origin >= lo) and np.all(origin <= hi):
                return full()
            camera = (bounds - origin) @ inverse_rotation
            scale = max(1., float(np.abs(camera).max()))
            near_guard = 64 * np.finfo(np.float32).eps * scale
            if camera[:,2].max() < -near_guard:
                continue
            if camera[:,2].min() <= max(1e-6, near_guard):
                return full()
            uvw = camera @ self.K.T
            uv = uvw[:,:2] / uvw[:,2:]
            if not np.isfinite(uv).all():
                return full()
            rounding = 64 * np.finfo(np.float32).eps * np.linalg.norm(self.K, ord=np.inf) * scale / camera[:,2].min()
            if rounding > 16 or np.abs(uv).max() > 1e8:
                return full()
            guard = max(2, int(np.ceil(rounding)))
            x0, y0 = np.floor(uv.min(axis=0)).astype(np.int64) - guard
            x1, y1 = np.ceil(uv.max(axis=0)).astype(np.int64) + guard + 1
            x0, y0 = max(0,int(x0)), max(0,int(y0))
            x1, y1 = min(width,int(x1)), min(height,int(y1))
            if x0 < x1 and y0 < y1:
                union[y0:y1,x0:x1] = True
        pixels = np.flatnonzero(union)
        return pixels, "CONSERVATIVE_UNION" if len(pixels) else "PROVEN_EMPTY"


def grouped_stats(owners, candidates, *, pixels=None, shape=None):
    values = np.asarray(owners)
    if pixels is None:
        if values.ndim != 2:
            raise ValueError("dense grouped owner statistics require a 2D image")
        shape, pixels = values.shape, np.arange(values.size, dtype=np.int64)
    values, pixels = values.reshape(-1), np.asarray(pixels)
    candidates = np.asarray(sorted(map(int,candidates)), np.int64)
    if (len(np.unique(candidates)) != len(candidates) or np.any(candidates <= 0)
            or len(pixels) != len(values)):
        raise ValueError("grouped statistics require distinct positive owners and aligned pixels")
    result = {int(owner): (0,None) for owner in candidates}
    if not len(candidates) or not len(values):
        return result
    positions = np.searchsorted(candidates, values)
    present = positions < len(candidates)
    present[present] &= candidates[positions[present]] == values[present]
    groups, selected = positions[present], pixels[present]
    counts = np.bincount(groups, minlength=len(candidates))
    x, y = selected % shape[1], selected // shape[1]
    min_x = np.full(len(candidates), shape[1], np.int64)
    min_y = np.full(len(candidates), shape[0], np.int64)
    max_x = np.full(len(candidates), -1, np.int64)
    max_y = max_x.copy()
    np.minimum.at(min_x,groups,x)
    np.minimum.at(min_y,groups,y)
    np.maximum.at(max_x,groups,x)
    np.maximum.at(max_y,groups,y)
    for group, owner in enumerate(candidates):
        if counts[group]:
            result[int(owner)] = (int(counts[group]), [int(min_x[group]),int(min_y[group]),int(max_x[group])+1,int(max_y[group])+1])
    return result


def retain_top3(values, pixels, frame_id, materialize):
    if len(values) == 3:
        worst = values[-1][0]
        if (-pixels, frame_id) > (-worst.visible_pixels, worst.frame_id):
            return False
    entry = materialize()
    values.append(entry)
    values.sort(key=lambda value: (-value[0].visible_pixels,value[0].frame_id,value[0].target_mask_sha256))
    kept = any(value is entry for value in values[:3])
    del values[3:]
    return kept


class ExactProjector(FullSceneProjector):
    def __init__(self, xyz, faces, raw, *, roi=False, batch_pixels=65536):
        super().__init__(xyz, faces, raw, batch_pixels=batch_pixels)
        self.roi = bool(roi)
        self.queried_rays = 0

    def prepare_bounds(self, candidates):
        for owner in candidates:
            if owner not in self.bounds:
                points = self.xyz[self.raw == owner]
                if not len(points):
                    raise ValueError("candidate owner is absent from the final raw geometry")
                lo, hi = points.min(axis=0), points.max(axis=0)
                self.bounds[owner] = np.array(list(product(*zip(lo,hi))))

    def project_frame(self, K, pose, depth, candidates):
        import open3d as o3d

        depth = np.asarray(depth)
        if depth.ndim != 2 or depth.dtype != np.float32 or not depth.size:
            raise ValueError("projector requires original nonempty FP32 depth_m")
        candidates = tuple(map(int,candidates))
        if len(set(candidates)) != len(candidates) or any(owner <= 0 for owner in candidates):
            raise ValueError("candidate owners must be unique positive raw IDs")
        camera = PreparedCamera(K,pose,depth.shape)
        self.prepare_bounds(candidates)
        if self.roi:
            pixels, query_reason = camera.query_union([self.bounds[owner] for owner in candidates])
        else:
            pixels, query_reason = np.arange(depth.size,dtype=np.int64), "FULL_FRAME"
        owners = np.zeros(len(pixels),np.int64)
        hit_owners = owners.copy()
        queried_hits = queried_valid_depth = queried_consistent_hits = 0
        for start in range(0,len(pixels),self.batch_pixels):
            selected = pixels[start:start+self.batch_pixels]
            rays = camera.rays(selected)
            hit = self.scene.cast_rays(o3d.core.Tensor(rays,dtype=o3d.core.Dtype.Float32),nthreads=4)
            t, primitive = hit["t_hit"].numpy(), hit["primitive_ids"].numpy()
            valid_hit = np.isfinite(t) & (t > 1e-6)
            measured = depth.reshape(-1)[selected]
            valid_depth = np.isfinite(measured) & (measured > 1e-6)
            face_owner = np.zeros(len(selected),np.int64)
            if valid_hit.any():
                if np.any(primitive[valid_hit] >= len(self.face_owners)):
                    raise ValueError("raycast primitive mapping left the complete original mesh")
                face_owner[valid_hit] = self.face_owners[primitive[valid_hit]]
            accepted = valid_hit & valid_depth & (np.abs(t-measured) <= np.maximum(.02,.02*measured))
            hit_owners[start:start+len(selected)] = face_owner
            owners[start:start+len(selected)] = np.where(accepted,face_owner,0)
            queried_hits += int(valid_hit.sum())
            queried_valid_depth += int(valid_depth.sum())
            queried_consistent_hits += int(accepted.sum())
        self.queried_rays += len(pixels)
        visible = grouped_stats(owners,candidates,pixels=pixels,shape=depth.shape)
        hits = grouped_stats(hit_owners,candidates,pixels=pixels,shape=depth.shape)
        counts = {}
        for owner in candidates:
            area, bbox = visible[owner]
            admissible = area >= 100 and bbox[2]-bbox[0] >= 2 and bbox[3]-bbox[1] >= 2
            counts[owner] = {"frustum_aabb_intersection": self._frustum_aabb(owner,camera.K,camera.pose,depth.shape),
                "hit_owner_pixels": hits[owner][0],"depth_consistent_pixels":area,
                "support_eligible_pixels":area if admissible else 0,"admissible":bool(admissible),"canonical_bbox":bbox}
        def mask_for_owner(owner):
            mask = np.zeros(depth.size,bool)
            mask[pixels] = owners == owner
            return mask.reshape(depth.shape)
        return {"counts":counts,"mask_for_owner":mask_for_owner,"queried_pixel_ids":pixels,
            "queried_owners":owners,"diagnostic_scope":"queried_pixels" if self.roi else "full_frame",
            "query_reason":query_reason,"queried_rays":len(pixels),"full_image_pixels":depth.size,
            "total_first_hits":None if self.roi else queried_hits,
            "valid_measured_depth_pixels":None if self.roi else queried_valid_depth,
            "depth_consistent_first_hits":None if self.roi else queried_consistent_hits,
            "queried_first_hits":queried_hits,"queried_valid_depth_pixels":queried_valid_depth,
            "queried_depth_consistent_first_hits":queried_consistent_hits}
