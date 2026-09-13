#!/usr/bin/env python3
"""Export and verify a native partition without changing its source geometry."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.native_export import write_partition_mesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-mesh', type=Path, required=True)
    p.add_argument('--partition', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('native export output must be new')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    points, faces = mesh.vertex.positions.numpy(), mesh.triangle.indices.numpy()
    owners = np.load(args.partition, allow_pickle=False)
    receipt = write_partition_mesh(points, faces, owners, args.output)
    restored = PlyData.read(args.output, known_list_len={'face': {'vertex_indices': 3}})
    if (any(not np.array_equal(restored['vertex'][name], points[:, i]) for i, name in enumerate(('x','y','z')))
            or not np.array_equal(restored['face']['vertex_indices'], faces)
            or not np.array_equal(restored['vertex']['instance_id'], owners)):
        raise ValueError('native PLY coordinate/triangle/owner roundtrip mismatch')
    receipt.update({'status': 'PASS_FULL_NATIVE_ROUNDTRIP', 'command': sys.argv,
        'seconds': time.perf_counter()-start, 'output_bytes': args.output.stat().st_size,
        'input_sha256': {str(f):sha256_file(f) for f in (args.native_mesh, args.partition)},
        'output_sha256': sha256_file(args.output)})
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(receipt['status'], receipt['vertex_count'], receipt['seconds'], flush=True)


if __name__ == '__main__':
    main()
