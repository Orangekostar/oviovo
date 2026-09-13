#!/usr/bin/env python3
"""Fuse measured RGB-D only, retaining exact frame/input identity for R6."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.rgbd_cloud import RGBDVoxelCloud, measured_world_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene-root', 'intrinsics', 'geometry-support', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output directory required')
    start = time.perf_counter()
    support = json.loads(args.geometry_support.read_text())
    frames = support['input_frame_ids']
    camera = json.loads(args.intrinsics.read_text())['camera']
    k = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1]])
    poses = np.loadtxt(args.scene_root/'traj.txt').reshape(-1, 4, 4)
    cloud = RGBDVoxelCloud(.01)
    inputs = {str(p): sha256_file(p) for p in (args.intrinsics, args.geometry_support, args.scene_root/'traj.txt')}
    counts = []
    for frame in frames:
        rgb_path = args.scene_root/'results'/f'frame{frame:06d}.jpg'
        depth_path = args.scene_root/'results'/f'depth{frame:06d}.png'
        for p in (rgb_path, depth_path):
            inputs[str(p)] = sha256_file(p)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert('RGB'))
        with Image.open(depth_path) as image:
            depth = np.asarray(image).astype(np.float64)/camera['scale']
        xyz, color = measured_world_points(depth, rgb, k, poses[frame])
        cloud.add(xyz, color)
        counts.append({'frame_id': frame, 'valid_depth_samples': len(xyz), 'occupied_voxels': len(cloud.keys)})
        if len(counts) % 20 == 0:
            print(counts[-1], flush=True)
    xyz, rgb, observations = cloud.arrays()
    args.output.mkdir(parents=True)
    payload = args.output/'observed_cloud.npz'
    np.savez_compressed(payload, coord=xyz, color=rgb, support_count=observations)
    receipt = {'status': 'COMPLETE_MEASURED_RGBD_CLOUD', 'input_frame_ids': frames,
        'input_sha256': inputs, 'command': sys.argv, 'voxel_size_m': .01,
        'point_count': len(xyz), 'valid_depth_samples': int(observations.sum()), 'frame_counts': counts,
        'GT_input': False, 'semantic_filter': False, 'XYZ_RGB_reduction': 'sample_count_weighted_mean',
        'pixel_convention': 'native_integer_uv_depth_camera_z',
        'elapsed_seconds': time.perf_counter()-start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'payload_sha256': sha256_file(payload), 'payload_bytes': payload.stat().st_size,
        'source_sha256': {str(p): sha256_file(p) for p in (Path(__file__),
            Path(__file__).resolve().parents[2]/'src/static_ovmap/rgbd_cloud.py')}}
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('cloud complete', len(xyz), 'points', flush=True)


if __name__ == '__main__':
    main()
