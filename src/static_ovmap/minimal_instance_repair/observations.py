"""Pose-fixed observation banks and full-mesh, label-independent source-row hits."""

from pathlib import Path
import shutil
import time

import numpy as np

from static_ovmap.cvpr_compact.projected_views import FullSceneProjector, validate_camera
from static_ovmap.runtime_parity.kernels import PreparedCamera


def pose_banks(frames, *, limit=32, translation=.20, degrees=15.):
    ordered = sorted(frames, key=lambda f:int(f['frame_id']))
    ids = [int(f['frame_id']) for f in ordered]
    if len(set(ids)) != len(ids) or limit < 2:
        raise ValueError('pose banks require unique original frames and limit >=2')
    representatives, assignments = [], []
    for frame in ordered:
        pose = np.asarray(frame['pose_c2w'], np.float64)
        validate_camera(np.eye(3), pose)
        selected = None
        for index, previous in enumerate(representatives):
            old = np.asarray(previous['pose_c2w'], np.float64)
            angle = np.rad2deg(np.arccos(np.clip((np.trace(old[:3,:3].T @ pose[:3,:3])-1)/2, -1, 1)))
            if np.linalg.norm(old[:3,3]-pose[:3,3]) <= translation and angle <= degrees:
                selected = index
                break
        if selected is None:
            selected = len(representatives)
            representatives.append(frame)
        assignments.append({'frame_id':int(frame['frame_id']), 'pose_bin':selected,
                            'representative_frame_id':int(representatives[selected]['frame_id'])})
    indices = (np.floor(np.arange(limit)*(len(representatives)-1)/(limit-1)).astype(int)
               if len(representatives) > limit else np.arange(len(representatives)))
    retained = [{**representatives[int(index)], 'pose_bin':int(index), 'selected_index':k,
                 'bank':'proposal' if k % 2 == 0 else 'verification'} for k,index in enumerate(indices)]
    return retained, assignments


def representative_rows(triangle_rows, primitive_uvs, *, tolerance=1e-5):
    faces, uv = np.asarray(triangle_rows), np.asarray(primitive_uvs)
    if faces.ndim != 2 or faces.shape[1] != 3 or uv.shape != (len(faces),2):
        raise ValueError('invalid barycentric source mapping shape')
    weights = np.column_stack((1-uv[:,0]-uv[:,1], uv[:,0], uv[:,1]))
    if (not np.isfinite(weights).all() or np.any(weights < -tolerance)
            or np.any(weights > 1+tolerance)):
        raise ValueError('invalid primitive barycentric weights')
    tied = weights == weights.max(1, keepdims=True)
    return np.where(tied, faces, np.iinfo(np.int64).max).min(1)


class SourceRowProjector:
    def __init__(self, xyz, faces, *, batch_pixels=65536):
        # Zero owners affect only an unused inherited owner lookup; every mesh face
        # still participates in nearest-hit occlusion. No proposal mask enters BVH.
        self.mesh = FullSceneProjector(xyz, faces, np.zeros(len(xyz), np.int64), batch_pixels=batch_pixels)
        self.triangles = np.asarray(faces)[self.mesh.original_triangle_indices]
        self.batch = batch_pixels
        self.queried_rays = 0

    def project(self, intrinsics, pose, depth, *, absolute=.02, relative=.02, minimum=1e-6):
        import open3d as o3d
        depth = np.asarray(depth)
        if depth.ndim != 2 or depth.dtype != np.float32 or not depth.size:
            raise ValueError('observer requires original aligned FP32 metric depth')
        camera = PreparedCamera(intrinsics, pose, depth.shape)
        source = np.full(depth.size, -1, np.int32)
        valid = np.zeros(depth.size, bool)
        for start in range(0, depth.size, self.batch):
            pixels = np.arange(start, min(start+self.batch,depth.size), dtype=np.int64)
            hit = self.mesh.scene.cast_rays(o3d.core.Tensor(camera.rays(pixels), dtype=o3d.core.Dtype.Float32), nthreads=4)
            t, primitive = hit['t_hit'].numpy(), hit['primitive_ids'].numpy()
            measured = depth.reshape(-1)[pixels]
            accepted = np.isfinite(t) & (t > minimum) & np.isfinite(measured) & (measured > minimum)
            accepted &= np.abs(t-measured) <= np.maximum(absolute, relative*measured)
            if accepted.any():
                indices = primitive[accepted]
                if np.any(indices >= len(self.triangles)):
                    raise ValueError('observer primitive index left the original full mesh')
                rows = representative_rows(self.triangles[indices], hit['primitive_uvs'].numpy()[accepted])
                if np.any(rows > np.iinfo(np.int32).max):
                    raise ValueError('source rows exceed exact compact int32 representation')
                source[pixels[accepted]] = rows
                valid[pixels[accepted]] = True
        self.queried_rays += depth.size
        return source.reshape(depth.shape), valid.reshape(depth.shape)


def evidence_counts(visible, labels, ids, counts, *, settings=None):
    cfg = settings or {'min_visible_pixels':25, 'min_labeled_pixels':25,
                      'min_labeled_fraction':.60, 'dominance':.65}
    winner = int(ids[np.lexsort((ids,-counts))[0]]) if len(ids) else 0
    dominance = float(counts.max()/labels) if labels else 0.
    usable = (visible >= cfg['min_visible_pixels'] and labels >= cfg['min_labeled_pixels']
              and labels >= cfg['min_labeled_fraction']*visible and dominance >= cfg['dominance'])
    return {'visible_pixels':int(visible), 'labeled_pixels':int(labels), 'dominant_id':winner,
            'dominance':dominance, 'labeled_fraction':float(labels/visible) if visible else 0.,
            'usable':bool(usable)}


def unit_evidence(mask, panoptic, *, settings=None):
    mask, panoptic = np.asarray(mask), np.asarray(panoptic)
    if mask.dtype != bool or panoptic.shape != mask.shape or not np.issubdtype(panoptic.dtype,np.integer) or np.any(panoptic<0):
        raise ValueError('unit evidence requires aligned masks and nonnegative integer local IDs')
    labels = panoptic[mask]
    ids, counts = np.unique(labels[labels>0],return_counts=True)
    return evidence_counts(int(mask.sum()),int(counts.sum()),ids,counts,settings=settings)


def all_unit_evidence(units, row_count, source_rows, valid, panoptic, *, settings=None):
    names = sorted(units)
    lookup = np.full(row_count, -1, np.int32)
    for slot,name in enumerate(names):
        rows = units[name].rows
        if np.any(lookup[rows] != -1):
            raise ValueError('whole P/K units must be disjoint')
        lookup[rows] = slot
    mapped = lookup[source_rows[valid]]
    labels = panoptic[valid]
    keep = mapped >= 0
    mapped, labels = mapped[keep],labels[keep]
    visible = np.bincount(mapped,minlength=len(names))
    positive = labels > 0
    labeled = np.bincount(mapped[positive],minlength=len(names))
    pairs, counts = np.unique(np.column_stack((mapped[positive],labels[positive])),axis=0,return_counts=True)
    return {name:evidence_counts(visible[i],labeled[i],pairs[pairs[:,0]==i,1],counts[pairs[:,0]==i],settings=settings)
            for i,name in enumerate(names)}


def pair_vote(left, right, *, separation_dominance=.80):
    if not left['usable'] or not right['usable']:
        return None
    if left['dominant_id'] == right['dominant_id']:
        return (1,0)
    return (0,int(min(left['dominance'],right['dominance']) >= separation_dominance))


def pair_rates(observations, *, separation_dominance=.80):
    votes = [v for left,right in observations if (v:=pair_vote(left,right,separation_dominance=separation_dominance)) is not None]
    return {'n':len(votes), 'same':float(np.mean([x[0] for x in votes])) if votes else None,
            'separate':float(np.mean([x[1] for x in votes])) if votes else None}


def support_mask(source_rows, valid, rows):
    return valid & np.isin(source_rows, rows)


def load_panoptic(capture_root, frame, index, path_map):
    import cv2
    from static_ovmap.recovery_wave2.binding import PathResolver
    resolver = PathResolver(path_map)
    path = Path(resolver.resolve(Path(capture_root)/frame['panoptic_path']))
    if not path.is_file():
        sources = frame.get('source_paths',{})
        original = sources.get('panoptic',sources.get('panoptic_source_path'))
        if isinstance(original,dict):
            original = original.get('path')
        if original is not None:
            fallback = Path(resolver.resolve(original))
            if fallback.is_file():
                # Only byte-identical relocation is legal; no re-warp or rerun.
                index.identity(fallback,{'sha256':frame['panoptic_sha256']})
                path = fallback
        if not path.is_file():
            raise FileNotFoundError(f'required pre-insertion panoptic missing: {path}; original={original}')
    identity = index.identity(path,{'sha256':frame['panoptic_sha256']})
    raster = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if (raster is None or raster.ndim != 2 or not np.issubdtype(raster.dtype,np.integer)
            or np.any(raster < 0)):
        raise ValueError('panoptic must be saved integer IDs, never a color visualization')
    return raster,identity


def observe_scene(binding, scene):
    from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
    from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
    from static_ovmap.recovery_wave2.binding import read
    from .binding import load_scene, seal

    root = Path(binding['output_root'])
    cfg = binding['specification']['observer']
    inputs = load_scene(binding,scene)
    selected,bins = pose_banks(inputs.capture['frames'],limit=cfg['max_frames'],
                              translation=cfg['pose_translation_m'],degrees=cfg['pose_rotation_degrees'])
    dest = root/'observations'/scene
    operators = [inputs.index.identity(__file__),inputs.index.identity(Path(__file__).with_name('support_units.py'))]
    dependency = canonical_digest({'capture':inputs.capture['identity'],'geometry':inputs.units.mesh_identity,
        'units':inputs.units.identity,'observer':cfg,'operators':operators})
    atomic_write_json(dest/'pose_roles.json',seal({'scene':scene,'input_identity':dependency,'assignments':bins,
        'selected':[{'frame_id':int(f['frame_id']),'pose_bin':f['pose_bin'],'bank':f['bank'],
                     'selected_index':f['selected_index']} for f in selected]}))
    atomic_write_json(dest/'supports.json',seal(inputs.units.record()))
    preflight = root/'observation_storage_preflight.json'
    if not preflight.exists():
        free = shutil.disk_usage(root).free
        minimum = binding['specification']['resources']['free_space_preflight_min_GiB']*1024**3
        if free < minimum:
            raise OSError(f'new output requires {minimum} free bytes; current {free}; use --storage-root')
        atomic_write_json(preflight,{'free_bytes':free,'minimum_bytes':minimum,'physical_root':str(root),
                                    'logical_root':binding['logical_root']})
    projector, rows, blocked = None,[],[]
    scene_begin = time.perf_counter()
    for frame in selected:
        fid = int(frame['frame_id'])
        receipt_path = dest/'frames'/f'{fid:06d}.json'
        path = receipt_path.with_suffix('.npz')
        rgb_path = inputs.capture_path.parent/frame['rgb_path']
        depth_path = inputs.capture_path.parent/frame['depth_path']
        rgb = inputs.index.identity(rgb_path,{'sha256':frame['rgb_sha256']})
        depth_id = inputs.index.identity(depth_path,{'sha256':frame['depth_sha256']})
        panoptic = None
        try:
            panoptic,panoptic_id = load_panoptic(inputs.capture_path.parent,frame,inputs.index,binding['path_map'])
        except FileNotFoundError as exc:
            # IR02 and ordinary FULL-only IR06 remain independently computable.
            panoptic_id = None
            blocked.append({'frame_id':fid,'reason':str(exc),
                'methods':['IR03_EVIDENCE_ATTACH','IR04_DIRECT_GROUP','IR05_VERIFIED_REPAIR',
                           'IR07_BOUNDARY_STABLE','IR08_COMBINATION']})
        leaf = canonical_digest({'observer':dependency,'frame_id':fid,'bank':frame['bank'],
            'pose':frame['pose_c2w'],'intrinsics':frame['intrinsics'],'rgb':rgb,'depth':depth_id,'panoptic':panoptic_id})
        if receipt_path.is_file():
            previous = read(receipt_path)
            if previous['input_identity'] != leaf:
                raise ValueError('completed observer leaf changed; invalidate only this leaf and descendants')
            inputs.index.identity(path,previous['arrays'])
            rows.append(previous)
            continue
        start = time.perf_counter()
        if projector is None:
            projector = SourceRowProjector(inputs.xyz,inputs.faces,batch_pixels=cfg['ray_batch_max'])
        with np.load(depth_path,allow_pickle=False) as arrays:
            depth = arrays['depth_m']
        if list(depth.shape) != frame['image_size_hw']:
            raise ValueError('captured depth changed original image alignment')
        source,valid = projector.project(frame['intrinsics'],frame['pose_c2w'],depth,
            absolute=cfg['depth_absolute_tolerance_m'],relative=cfg['depth_relative_tolerance'],minimum=cfg['minimum_depth_m'])
        values = {'source_rows':source,'valid':valid}
        stats = None
        if panoptic is not None:
            if panoptic.shape != depth.shape:
                raise ValueError('saved panoptic is not aligned to original captured depth')
            values['panoptic'] = panoptic
            stats = all_unit_evidence(inputs.units.units,len(inputs.raw),source,valid,panoptic,settings=cfg)
        _write_npz(path,values)
        row = seal({'status':'COMPLETE','scene':scene,'frame_id':fid,'bank':frame['bank'],
            'pose_bin':frame['pose_bin'],'input_identity':leaf,'arrays':inputs.index.identity(path),
            'source_row_sha256':_array_digest(source),'valid_sha256':_array_digest(valid),
            'valid_pixels':int(valid.sum()),'queried_rays':int(depth.size),
            'panoptic_input':panoptic_id,'panoptic_available':panoptic is not None,
            'units':stats,'elapsed_seconds':time.perf_counter()-start,'new_neural_inference':0})
        atomic_write_json(receipt_path,row)
        rows.append(row)
        print('OBSERVED',scene,fid,len(rows),'/',len(selected),'pixels',int(valid.sum()),flush=True)
    result = seal({'status':'COMPLETE' if not blocked else 'COMPLETE_GEOMETRY_BLOCKED_FRONTEND_EVIDENCE',
        'scene':scene,'input_identity':dependency,'support_identity':inputs.units.identity,
        'frames':rows,'pose_roles':str(dest/'pose_roles.json'),'support_manifest':str(dest/'supports.json'),
        'blocked':blocked,'successful_unique_frame_count':len(rows),'elapsed_this_invocation':time.perf_counter()-scene_begin,
        'new_maps':0,'new_segmentation_inference':0,'new_FC_inference':0,'new_AnyUp_QK':0})
    atomic_write_json(dest/'receipt.json',result)
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    return result
