#!/usr/bin/env python3
"""Build a GT-free native graph from frozen 200-frame RGB-D entity observations."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.instance_graph import frame_pair_evidence, aggregate_edges, constrained_components
from src.static_ovmap.native_export import remap_partition, restore_partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-mesh', type=Path, required=True)
    p.add_argument('--geometry-support', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--mask-export-source', type=Path, required=True)
    p.add_argument('--mask-run-log', type=Path, required=True)
    p.add_argument('--scene-root', type=Path, required=True)
    p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    support = json.loads(args.geometry_support.read_text())
    if (support['native_mesh_sha256'] != sha256_file(args.native_mesh)
            or support['pixel_convention'] != 'native_integer_pixels_camera_z_depth'):
        raise ValueError('native mesh/support binding mismatch')
    if support['input_frame_ids'] != list(range(0, 2000, 10)):
        raise ValueError('this producer requires explicit Replica200 input schedule')
    # Bind the actual historical exporter and log; cached scores are still unavailable.
    exporter = args.mask_export_source.read_text()
    log = args.mask_run_log.read_text()
    if ('predictions["instances"].pred_masks' not in exporter
            or 'entity_segmentation/cropformer_hornet_3x.yaml' not in log
            or str(args.mask_dir.resolve()) not in log or 'confidence_threshold=0.5' not in log):
        raise ValueError('whole-entity mask provenance is not corroborated by source and run log')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    geometry = args.geometry_support.parent
    native = np.load(geometry/'native_vertex_owners.npy', allow_pickle=False)
    camera = json.loads(args.intrinsics.read_text())['camera']
    poses = np.loadtxt(args.scene_root/'traj.txt').reshape(-1, 4, 4)
    frames, frame_hashes = {}, []
    for i, frame in enumerate(support['input_frame_ids']):
        visible_path = geometry/'visible_owners'/f'{frame:06d}.npz'
        mask_path = args.mask_dir/f'frame{frame:06d}.png'
        depth_path = args.scene_root/'results'/f'depth{frame:06d}.png'
        with np.load(visible_path, allow_pickle=False) as loaded:
            owners = loaded['owners']
        with Image.open(mask_path) as image:
            masks = np.asarray(image)
        with Image.open(depth_path) as image:
            depth = np.asarray(image).astype(np.float64)/camera['scale']
        frames[frame] = frame_pair_evidence(owners, masks, np.isfinite(depth) & (depth > 0),
                                             whole_entity_source=True)
        frame_hashes.append({'frame_id': frame, 'mask_sha256': sha256_file(mask_path),
                             'depth_sha256': sha256_file(depth_path),
                             'visible_owners_sha256': sha256_file(visible_path)})
        if i % 40 == 0:
            print('frame', frame, 'positive', len(frames[frame]['positive']),
                  'negative', len(frames[frame]['negative']), flush=True)
    witness_seconds = time.perf_counter()-start
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    points = mesh.vertex.positions.numpy()
    if len(points) != len(native):
        raise ValueError('native vertex identity misalignment')
    pairs = {tuple(x['owners']) for f in frames.values() for x in f['positive']}
    representatives, trees = {}, {}
    for owner in sorted({o for pair in pairs for o in pair}):
        selected = points[native == owner]
        _, indices = np.unique(np.floor(selected/.02).astype(np.int64), axis=0, return_index=True)
        representatives[owner] = selected[indices]
        trees[owner] = cKDTree(selected[indices])
    distances = {}
    for a, b in sorted(pairs):
        distances[(a, b)] = float(trees[b].query(representatives[a], k=1, workers=1)[0].min())
    edges = aggregate_edges(frames, dict(enumerate(poses)), distances)
    nodes = np.unique(native[native > 0]).astype(int).tolist()
    partition = constrained_components(nodes, edges['positive'], edges['negative'])
    merged, ledger = remap_partition(native, partition['components'])
    if not np.array_equal(restore_partition(merged, ledger), native):
        raise ValueError('native partition restoration failed')
    np.save(args.output/'B0_native_owners.npy', native)
    np.save(args.output/'G1_native_owners.npy', merged)
    np.savez_compressed(args.output/'G1_restoration.npz', **ledger)
    result = {'mode': 'STATIC_OFFLINE_NATIVE_INSTANCE_GRAPH', 'status': 'G1_COMPLETE_G2_PENDING',
        'nodes': nodes, 'edges': edges, 'partition': partition,
        'geometry_coordinates': 'UNCHANGED_NATIVE_MESH', 'gt_used_in_prediction': False,
        'whole_object_evidence': 'historical_CropFormer_entity_masks_plus_independent_bidirectional_RGBD_alignment',
        'mask_limitations': 'visible_score_arbitrated_masks_not_amodal; per_mask_scores_not_cached; evidence_not_object_truth',
        'thresholds': {'min_owner_pixels': 100, 'min_mask_pixels': 500, 'min_depth_ratio': .9,
            'min_geometry_coverage': .7, 'min_owner_purity': .8, 'min_joint_coverage': .5,
            'min_separate_iou': .65, 'min_independent_frames': 3, 'min_positive_agreement': .6,
            'max_surface_distance_m': .05, 'contact_representative_grid_m': .02,
            'independent_translation_m': .05, 'independent_rotation_degrees': 5.},
        'cost': {'witness_seconds': witness_seconds, 'total_seconds': time.perf_counter()-start,
                 'image_encoder_calls': 0, 'graph_device': 'CPU'},
        'command': sys.argv, 'input_sha256': {str(f): sha256_file(f) for f in
            (args.native_mesh, args.geometry_support, geometry/'native_vertex_owners.npy',
             args.mask_export_source, args.mask_run_log, args.scene_root/'traj.txt', args.intrinsics)},
        'frame_hashes': frame_hashes, 'native_id_restore_exact': True,
        'changed_native_vertices': int(len(ledger['positions']))}
    (args.output/'frame_witnesses.json').write_text(json.dumps(frames, indent=2)+'\n')
    (args.output/'graph.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('complete', len(nodes), 'to', len(partition['components']), 'components', result['cost'], flush=True)


if __name__ == '__main__':
    main()
