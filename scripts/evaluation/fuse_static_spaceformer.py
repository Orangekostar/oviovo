#!/usr/bin/env python3
"""T1 proposal union on observed coordinates, with no GT or cross-space feature mixing."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.proposal_fusion import fuse_proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('t0', 'readout', 'native-mesh', 'geometry-support', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    support = json.loads(args.geometry_support.read_text())
    if support['native_mesh_sha256'] != sha256_file(args.native_mesh):
        raise ValueError('native owner identity and mesh differ')
    with np.load(args.t0, allow_pickle=False) as data:
        coord, masks, labels, scores, query_ids = [data[k] for k in ('coord', 'masks', 'class_ids', 'scores', 'query_ids')]
    mesh = o3d.t.io.read_triangle_mesh(str(args.native_mesh))
    native_path = args.geometry_support.parent/'native_vertex_owners.npy'
    native_owners = np.load(native_path, allow_pickle=False)
    if len(native_owners) != len(mesh.vertex.positions):
        raise ValueError('native vertex-owner alignment differs')
    tree = o3d.core.nns.NearestNeighborSearch(mesh.vertex.positions)
    tree.knn_index()
    nearest, squared = tree.knn_search(o3d.core.Tensor(np.ascontiguousarray(coord, dtype=np.float32)), 1)
    nearest, squared = nearest.numpy().reshape(-1), squared.numpy().reshape(-1)
    owners = np.where(squared < .05**2, native_owners[nearest], 0)
    projection_seconds = time.perf_counter()-start
    readout = json.loads(args.readout.read_text())
    fused = fuse_proposals(owners, readout['observations'], masks, labels, scores, query_ids,
        scene_id=readout.get('scene', 'unspecified'), prediction_run=args.t0.parent.name,
        ovi_readout_id='sha256:'+sha256_file(args.readout))
    np.savez_compressed(args.output/'T1.npz', coord=coord, masks=fused['masks'], class_ids=fused['class_ids'],
        scores=fused['scores'], candidate_ids=fused['candidate_ids'], score_definition='common_domain_source_area')
    np.savez_compressed(args.output/'native_projection.npz', owners=owners, nearest=nearest, distance_squared=squared)
    (args.output/'proposal_ledger.json').write_text(json.dumps(fused['ledger'], indent=2)+'\n')
    receipt = {'status': 'COMPLETE_T1_PENDING_EVALUATION', 'GT_input': False, 'extra_3D_training': True,
        'image_encoder_queries_added': 0, 'readout': readout['condition'], 'point_count': len(coord),
        'proposal_count': len(fused['masks']), 'candidate_count': len(fused['ledger']),
        'reuse_iou': .5, 'NMS_iou': .7, 'ordering': 'OVI_area_ownerID_then_T0_released_score',
        'score_definition': 'common_domain_source_area_not_calibrated_cross_model_confidence',
        'feature_space_handling': 'geometric_label_reuse_only_never_average_SigLIP_and_SigLIP2',
        'projection_seconds': projection_seconds, 'elapsed_seconds': time.perf_counter()-start,
        'command': sys.argv, 'input_sha256': {str(p): sha256_file(p) for p in (args.t0, args.readout,
            args.native_mesh, args.geometry_support, native_path)},
        'source_sha256': {str(p): sha256_file(p) for p in (Path(__file__),
            Path(__file__).resolve().parents[2]/'src/static_ovmap/proposal_fusion.py')},
        'output_sha256': sha256_file(args.output/'T1.npz')}
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
