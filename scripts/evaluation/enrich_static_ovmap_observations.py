#!/usr/bin/env python3
"""Compute offline RGB-D context quality and native-mesh object view directions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.enrichment import crop_quality, view_direction


def native_centers(path, colors):
    """Stream native ASCII PLY vertices in bounded chunks, no GT or face matrix.

    OVI's unshared triangle vertices make this a triangle-vertex centroid.
    Colors come from the same native cache, never from semantic class labels.
    """
    centers = {}
    wanted = {tuple(map(int, color)): str(instance) for instance, color in colors.items()}
    if len(wanted) != len(colors):
        raise ValueError('ambiguous native instance colors')
    sums = {color: np.zeros(3) for color in wanted}
    counts = defaultdict(int)
    with Path(path).open('rb') as handle:
        header = []
        while True:
            line = handle.readline().decode('ascii').strip()
            if not line:
                raise ValueError('incomplete native PLY header')
            header.append(line)
            if line == 'end_header':
                break
        if 'format ascii 1.0' not in header:
            raise ValueError('this native mesh adapter requires ASCII PLY')
        start = next(i for i, s in enumerate(header) if s.startswith('element vertex '))
        count = int(header[start].split()[-1])
        props = []
        for line in header[start+1:]:
            if not line.startswith('property '):
                break
            props.append(line.split()[-1])
        columns = [props.index(c) for c in ('x', 'y', 'z', 'red', 'green', 'blue')]
        remaining = count
        while remaining:
            size = min(250000, remaining)
            values = np.loadtxt(handle, max_rows=size, usecols=columns, ndmin=2)
            if len(values) != size:
                raise ValueError('truncated vertex data')
            rgb = values[:, 3:].astype(np.uint32)
            codes = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
            keys, inverse = np.unique(codes, return_inverse=True)
            chunk_counts = np.bincount(inverse)
            chunk_sums = np.stack([np.bincount(inverse, weights=values[:, c]) for c in range(3)], axis=1)
            for i, code in enumerate(keys):
                color = (int(code >> 16), int((code >> 8) & 255), int(code & 255))
                if color in wanted:
                    sums[color] += chunk_sums[i]
                    counts[color] += int(chunk_counts[i])
            remaining -= size
    for color, instance in wanted.items():
        if counts[color]:
            centers[instance] = (sums[color] / counts[color]).tolist()
    return centers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-cache', type=Path, required=True)
    parser.add_argument('--native-mesh', type=Path, required=True)
    parser.add_argument('--rgbd-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    with args.native_cache.open('rb') as handle:
        native = pickle.load(handle)
    centers = native_centers(args.native_mesh, {k: v['color'] for k, v in native.items()})
    center_seconds = time.perf_counter() - start
    by_frame = defaultdict(list)
    for instance, record in native.items():
        for index, frame in enumerate(record['frame_id']):
            by_frame[int(frame)].append((int(instance), index, record))
    observations, missing, frame_assets = {}, [], []
    for frame, entries in sorted(by_frame.items()):
        rgb_path = args.rgbd_root / f'frame{frame:06d}.jpg'
        depth_path = args.rgbd_root / f'depth{frame:06d}.png'
        if not rgb_path.is_file() or not depth_path.is_file():
            missing.append(frame)
            continue
        with Image.open(rgb_path) as img:
            rgb = np.asarray(img.convert('RGB'))
        with Image.open(depth_path) as img:
            depth = np.asarray(img)
        frame_assets.append({'frame_id': frame, 'rgb': str(rgb_path), 'depth': str(depth_path),
                             'rgb_bytes': rgb_path.stat().st_size,
                             'depth_bytes': depth_path.stat().st_size})
        for instance, index, record in entries:
            quality = crop_quality(rgb, depth, record['box_2d'][index])
            direction = None
            if str(instance) in centers:
                pose = np.asarray(record['pose'][index])
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError('invalid native camera-to-world pose')
                direction = view_direction(pose[:3, 3], centers[str(instance)])
            view_bin = None
            if direction is not None:
                x, y, z = direction
                azimuth = int(np.floor((np.arctan2(y, x) + np.pi) * 8 / (2*np.pi))) % 8
                view_bin = azimuth + 8 * int(z >= 0)
            observations[f'{instance}:{index}'] = {
                'instance_id': instance, 'frame_id': frame, **quality,
                'camera_direction': direction, 'view_bin': view_bin}
    result = {'mode': 'STATIC_OFFLINE_READOUT',
              'native_cache_sha256': sha256_file(args.native_cache),
              'native_mesh_sha256': sha256_file(args.native_mesh),
              'quality_mode': 'legacy_context_bbox_depth_fraction_and_laplacian_variance',
              'direction_mode': 'final_native_mesh_triangle_vertex_centroid_to_camera',
              'sharpness_scale': 100., 'bbox_policy': 'legacy_exclusive_slice_no_fix',
              'depth_valid_definition': 'finite_and_positive_raw_depth_in_context_bbox',
              'mask_quality_proxy': None, 'source_mask_binding': 'unavailable',
              'geometry_support_frames': None, 'centers': centers,
              'observations': observations, 'frame_assets': frame_assets,
              'missing_frames': missing, 'centroid_seconds': center_seconds,
              'total_preparation_seconds': time.perf_counter()-start,
              'additional_image_queries': 0, 'command': sys.argv}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'observations': len(observations), 'objects_with_centers': len(centers),
                      'missing_frames': missing, 'seconds': result['total_preparation_seconds']}))


if __name__ == '__main__':
    main()
