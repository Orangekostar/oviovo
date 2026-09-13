#!/usr/bin/env python3
"""Lift multiview whole-entity masks to fixed native surface atoms and split G1 parents."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.native_export import partition_change_ledger, restore_partition
from src.static_ovmap.split_evidence import split_from_views


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--native-mesh', type=Path, required=True)
    p.add_argument('--geometry-support', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--scene-root', type=Path, required=True)
    p.add_argument('--intrinsics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    graph = json.loads(args.graph.read_text())
    if any(graph['input_sha256'].get(str(f)) != sha256_file(f) for f in
           (args.native_mesh, args.geometry_support, args.intrinsics, args.scene_root/'traj.txt')):
        raise ValueError('G2 inputs differ from G1 graph evidence')
    frames = json.loads(args.geometry_support.read_text())['input_frame_ids']
    witnesses = json.loads((args.graph.parent/'frame_witnesses.json').read_text())
    frame_hashes = {x['frame_id']: x for x in graph['frame_hashes']}
    native_b0 = np.load(args.graph.parent/'B0_native_owners.npy', allow_pickle=False)
    native_g1 = np.load(args.graph.parent/'G1_native_owners.npy', allow_pickle=False)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    points = mesh.vertex.positions.numpy()
    if len(points) != len(native_g1):
        raise ValueError('native surface and G1 partition do not align')
    vertex_atoms = np.full(len(points), -1, dtype=np.int64)
    atom_points, atom_parents = [], []
    offset = 0
    for parent in np.unique(native_g1[native_g1 > 0]):
        positions = np.flatnonzero(native_g1 == parent)
        _, representatives, inverse = np.unique(np.floor(points[positions]/.02).astype(np.int64),
                                                 axis=0, return_index=True, return_inverse=True)
        vertex_atoms[positions] = offset+inverse
        atom_points.append(points[positions[representatives]])
        atom_parents.append(np.full(len(representatives), parent, dtype=np.int64))
        offset += len(representatives)
    xyz, parents = np.concatenate(atom_points), np.concatenate(atom_parents)
    np.save(args.output/'vertex_to_atom.npy', vertex_atoms)
    np.save(args.output/'atom_parent_ids.npy', parents)
    np.save(args.output/'atom_native_xyz.npy', xyz)
    observations = np.lib.format.open_memmap(args.output/'atom_frame_masks.npy', mode='w+',
                                               dtype=np.int16, shape=(len(frames), len(xyz)))
    poses = np.loadtxt(args.scene_root/'traj.txt').reshape(-1, 4, 4)
    camera = json.loads(args.intrinsics.read_text())['camera']
    owner_to_parent = np.zeros(int(native_b0.max())+1, dtype=np.int64)
    for component in graph['partition']['components']:
        owner_to_parent[component] = min(component)
    for index, frame in enumerate(frames):
        mask_path = args.mask_dir/f'frame{frame:06d}.png'
        depth_path = args.scene_root/'results'/f'depth{frame:06d}.png'
        visible_path = args.geometry_support.parent/'visible_owners'/f'{frame:06d}.npz'
        if any(sha256_file(path) != frame_hashes[frame][key] for path, key in
               [(mask_path, 'mask_sha256'), (depth_path, 'depth_sha256'), (visible_path, 'visible_owners_sha256')]):
            raise ValueError('RGB-D frame evidence changed since G1')
        with Image.open(mask_path) as image:
            masks = np.asarray(image)
        if int(masks.max()) > np.iinfo(np.int16).max:
            raise ValueError('frame-local mask IDs exceed explicit int16 observation format')
        with Image.open(depth_path) as image:
            depth = np.asarray(image).astype(np.float64)/camera['scale']
        with np.load(visible_path, allow_pickle=False) as loaded:
            visible = owner_to_parent[loaded['owners']]
        mask_parent = np.zeros(int(masks.max())+1, dtype=np.int64)
        for mask_id in witnesses[str(frame)]['credible_mask_ids']:
            pixel_mask = masks == mask_id
            ids, counts = np.unique(visible[pixel_mask], return_counts=True)
            counts[ids == 0] = 0
            best = int(counts.argmax())
            if counts[best]/max(int(pixel_mask.sum()), 1) >= .7:
                mask_parent[mask_id] = ids[best]
        pose = poses[frame]
        camera_xyz = (xyz-pose[:3, 3]) @ pose[:3, :3]
        z = camera_xyz[:, 2]
        valid = np.isfinite(camera_xyz).all(axis=1) & (z > 0)
        u = np.zeros(len(xyz), dtype=np.int64)
        v = np.zeros(len(xyz), dtype=np.int64)
        u[valid] = np.rint(camera['fx']*camera_xyz[valid, 0]/z[valid]+camera['cx']).astype(np.int64)
        v[valid] = np.rint(camera['fy']*camera_xyz[valid, 1]/z[valid]+camera['cy']).astype(np.int64)
        valid &= (u >= 0) & (v >= 0) & (u < camera['w']) & (v < camera['h'])
        candidates = np.flatnonzero(valid)
        pixel_depth = depth[v[candidates], u[candidates]]
        consistent = ((pixel_depth > 0) & (np.abs(pixel_depth-z[candidates]) < .05)
                      & (visible[v[candidates], u[candidates]] == parents[candidates]))
        seen = candidates[consistent]
        local_masks = masks[v[seen], u[seen]]
        observations[index] = -1
        observations[index, seen] = np.where(mask_parent[local_masks] == parents[seen], local_masks, 0)
        if index % 40 == 0:
            print('frame', frame, 'visible native atoms', len(seen), flush=True)
    observations.flush()
    atom_children = parents.copy()
    decisions, child_sources = [], {}
    next_id = int(native_b0.max())+1
    for parent in np.unique(parents):
        selected = np.flatnonzero(parents == parent)
        decision = split_from_views(np.asarray(observations[:, selected]), frames, dict(enumerate(poses)))
        children = decision.pop('atom_children')
        decision['parent_id'] = int(parent)
        decision['native_atom_count'] = len(selected)
        if decision['accepted']:
            mapping = {0: int(parent)}
            for child in np.unique(children[children > 0]):
                mapping[int(child)] = next_id
                atom_children[selected[children == child]] = next_id
                child_sources[str(next_id)] = int(parent)
                next_id += 1
            decision['child_to_native_id'] = mapping
        decisions.append(decision)
    native_g2 = native_g1.copy()
    foreground = vertex_atoms >= 0
    native_g2[foreground] = atom_children[vertex_atoms[foreground]]
    ledger = partition_change_ledger(native_b0, native_g2)
    if not np.array_equal(restore_partition(native_g2, ledger), native_b0):
        raise ValueError('G2 cannot restore original B0 native partition')
    np.save(args.output/'G2_native_owners.npy', native_g2)
    np.save(args.output/'atom_child_ids.npy', atom_children)
    np.savez_compressed(args.output/'G2_restoration.npz', **ledger)
    result = {'mode': 'STATIC_OFFLINE_MULTIVIEW_NATIVE_SURFACE_SPLIT', 'status': 'G2_COMPLETE',
        'decisions': decisions, 'child_sources': child_sources, 'native_id_restore_exact': True,
        'accepted_parent_count': sum(x['accepted'] for x in decisions), 'native_atom_count': len(xyz),
        'thresholds': {'surface_atom_grid_m': .02, 'min_parent_mask_purity': .7, 'depth_tolerance_m': .05,
            'min_seed_atoms': 20, 'min_child_atoms': 50, 'min_child_fraction': .1, 'min_seed_agreement': .8,
            'min_independent_frames': 3, 'min_atom_votes': 2, 'min_atom_agreement': .8,
            'min_assigned_fraction': .7, 'max_anchor_children': 4},
        'semantic_rule': 'children_inherit_G1_parent_prediction_no_new_queries',
        'unobserved_or_ambiguous_atoms': 'retain_G1_parent_ID', 'coordinates': 'original_vertices_unchanged',
        'grid_scope': 'evidence_grouping_only_not_mapper_voxel_size_or_geometry_downsampling',
        'cost': {'total_seconds': time.perf_counter()-start, 'image_encoder_calls': 0,
                 'atom_observation_bytes': int(observations.nbytes)},
        'command': sys.argv, 'input_sha256': {str(f): sha256_file(f) for f in
            (args.graph, args.graph.parent/'frame_witnesses.json', args.native_mesh, args.geometry_support,
             args.graph.parent/'B0_native_owners.npy', args.graph.parent/'G1_native_owners.npy',
             args.scene_root/'traj.txt', args.intrinsics)}}
    (args.output/'split.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('complete', result['accepted_parent_count'], 'split parents', result['cost'], flush=True)


if __name__ == '__main__':
    main()
