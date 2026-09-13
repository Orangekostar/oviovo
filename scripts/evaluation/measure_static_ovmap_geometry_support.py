#!/usr/bin/env python3
"""Render native mesh against raw depth; no GT, semantic labels or feature values."""
import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.baselines.ovimap import parse_instance_color_log
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.geometry_support import supported_owner_pixels, independent_support_frames, native_pixel_rays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-mesh', type=Path, required=True)
    p.add_argument('--color-log', type=Path, required=True)
    p.add_argument('--native-cache', type=Path)
    p.add_argument('--scene-root', type=Path, required=True)
    p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'visible_owners').mkdir()
    start = time.perf_counter()
    camera = json.loads(args.intrinsics.read_text())['camera']
    poses = np.loadtxt(args.scene_root/'traj.txt').reshape(-1,4,4)
    frames = list(range(0,2000,10))
    if len(poses) <= frames[-1]:
        raise ValueError('Replica200 poses incomplete')
    colors = parse_instance_color_log(args.color_log)
    cache_only_colors = []
    if args.native_cache:
        with args.native_cache.open('rb') as handle:
            cached_records = pickle.load(handle)
        for owner, record in cached_records.items():
            color = tuple(map(int, record['color']))
            if int(owner) in colors and colors[int(owner)] != color:
                raise ValueError('native cache and C++ instance color log disagree')
            if int(owner) not in colors:
                cache_only_colors.append(int(owner))
            colors[int(owner)] = color
        del cached_records
    if len(set(colors.values())) != len(colors):
        raise ValueError('different native owners have the same RGB identity')
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    faces = mesh.triangle.indices.numpy()
    rgb = (mesh.vertex.colors.numpy()[faces[:,0]] * 255.).astype(np.uint8)
    owner_faces = np.zeros(len(faces), dtype=np.int64)
    native_owners = np.zeros(len(mesh.vertex.positions), dtype=np.int64)
    for owner, color in colors.items():
        mask = np.all(rgb == np.asarray(color), axis=1)
        owner_faces[mask] = owner
        native_owners[faces[mask].reshape(-1)] = owner
    np.save(args.output/'native_vertex_owners.npy', native_owners)
    np.save(args.output/'native_face_owners.npy', owner_faces)
    scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
    scene.add_triangles(mesh)
    intrinsic = np.array([[camera['fx'],0,camera['cx']], [0,camera['fy'],camera['cy']], [0,0,1.]])
    evidence = {}
    for index, frame in enumerate(frames):
        pose = poses[frame]
        rays = o3d.core.Tensor(native_pixel_rays(intrinsic, pose, camera['w'], camera['h']))
        hit = scene.cast_rays(rays, nthreads=args.threads)
        distances = hit['t_hit'].numpy()
        primitive = hit['primitive_ids'].numpy()
        valid = np.isfinite(distances) & (primitive < len(owner_faces))
        owners = np.zeros(distances.shape, dtype=np.int64)
        owners[valid] = owner_faces[primitive[valid]]
        # Convert the ray parameter to camera-z depth rather than assuming unit rays.
        ray_data = rays.numpy()
        directions = ray_data[...,3:] @ pose[:3,:3]
        predicted_depth = np.full(distances.shape, np.inf)
        predicted_depth[valid] = distances[valid] * directions[...,2][valid]
        depth_path = args.scene_root/'results'/f'depth{frame:06d}.png'
        with Image.open(depth_path) as image:
            depth = np.asarray(image).astype(np.float64) / camera['scale']
        evidence[frame] = supported_owner_pixels(owners, predicted_depth, depth)
        consistent = (np.isfinite(predicted_depth) & (depth>0)
                      & (np.abs(predicted_depth-depth)<.05))
        np.savez_compressed(args.output/'visible_owners'/f'{frame:06d}.npz',
                            owners=np.where(consistent, owners, 0).astype(np.int64))
        if index % 20 == 0:
            print('frame',frame,'supported objects',len(evidence[frame]),flush=True)
    support = independent_support_frames(evidence, dict(enumerate(poses)))
    result = {'mode': 'STATIC_OFFLINE_FINAL_GEOMETRY_VISIBILITY',
        'source': 'native_mesh_render_and_raw_depth_no_GT_no_semantic_query_count',
        'pixel_convention': 'native_integer_pixels_camera_z_depth',
        'objects': {str(k):v for k,v in support.items()}, 'frame_pixel_counts': evidence,
        'input_frame_ids': frames, 'depth_tolerance_m': .05, 'min_supported_pixels': 100,
        'independence_translation_m': .05, 'independence_rotation_degrees': 5.,
        'unassigned_native_vertices': int((native_owners==0).sum()),
        'native_owner_count': int(len(np.unique(native_owners[native_owners>0]))),
        'cache_only_color_ids': cache_only_colors,
        'geometry_seconds': time.perf_counter()-start,
        'native_mesh_sha256': sha256_file(args.native_mesh),
        'color_log_sha256': sha256_file(args.color_log), 'command': sys.argv}
    if args.native_cache:
        result['native_cache_sha256'] = sha256_file(args.native_cache)
    (args.output/'geometry_support.json').write_text(json.dumps(result,indent=2)+'\n')
    print('completed',len(support),'supported objects',result['geometry_seconds'],'seconds',flush=True)


if __name__ == '__main__':
    main()
