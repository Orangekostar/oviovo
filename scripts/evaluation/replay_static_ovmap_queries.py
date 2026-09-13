#!/usr/bin/env python3
"""Replay cached native geometry to recover query metadata, with no VLM fallback.

Recovered metadata remains provisional until retained-query and final-map parity
are demonstrated. The original source/cache are read-only; results use a new folder.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.native_recording import instrument_original_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--feature-pool', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--scene', default='room0')
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--geometry-dir', type=Path, required=True)
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('raycasts', 'query_masks', 'intermediate', 'input_masks', 'mapping/cropformer_inst'):
        (args.output/name).mkdir(parents=True, exist_ok=True)
    (args.output/'mapping/cropformer_inst/temp_feats').symlink_to(args.feature_pool.resolve())
    (args.output/'input_masks/cropformer').symlink_to(args.mask_dir.resolve())
    current = {'frame_id': None, 'query_count': 0}
    queries = (args.output/'queries.jsonl').open('w')
    def record_frame(frame, ids):
        current['frame_id'] = int(frame)
        np.save(args.output/'raycasts'/f'{frame:06d}.npy', np.asarray(ids))
    def record_query(instance, frame, area, bbox, feature, pose, mask_id, glo_mask, pano_mask):
        index = current['query_count']
        current['query_count'] += 1
        union = np.logical_or(glo_mask, pano_mask)
        mask_path = args.output/'query_masks'/f'{index:06d}.npz'
        np.savez_compressed(mask_path, mask_packed=np.packbits(union), shape=np.array(union.shape))
        x1, y1, x2, y2 = map(int, bbox)
        feature_name = f'siglip-l-16-384_F_{frame}_{x1}-{y1}-{x2-x1}-{y2-y1}.npy'
        if not np.array_equal(feature, np.load(args.feature_pool/feature_name)):
            raise ValueError('recorded query does not match original feature')
        queries.write(json.dumps({'instance_id': int(instance), 'frame_id': int(frame),
            'visible_area_px': int(area), 'crop_bbox_xyxy': [x1,y1,x2,y2],
            'pose': np.asarray(pose).tolist(), 'source_query_id': str(index),
            'source_feature': feature_name, 'source_mask_id': int(mask_id),
            'source_mask_path': str(mask_path)})+'\n')
        queries.flush()
    class CacheOnlyModel:
        def __init__(self, **kwargs):
            pass
        def encode_image_with_bbox(self, rgb, mask, bbox):
            raise RuntimeError(f'CACHE_ONLY_MISS frame={current["frame_id"]} bbox={tuple(map(int,bbox))}')
    module = types.ModuleType('vl_models')
    module.VLModel = CacheOnlyModel
    sys.modules['vl_models'] = module
    source_path = args.source_root/'scripts/panoptic_mapping_.py'
    transformed = instrument_original_source(source_path.read_text())
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root/'scripts'))
    command = list(sys.argv)
    sys.argv = [str(source_path), '--dataset', 'replica', '--task', 'Nyu40', '--scene_num', args.scene,
        '--data_folder', str(args.dataset_root), '--result_folder', str(args.output/'mapping'),
        '--start', '0', '--end', str(args.end), '--step', '10', '--data_association', '2',
        '--inst_association', '4', '--seg_graph_confidence', '3', '--use_temp_results',
        '--intermediate_seg_folder', str(args.output/'intermediate'),
        '--temp_panoptics_folder', str(args.output/'input_masks'), '--use_temp_geometrics',
        '--temp_geometrics_folder', str(args.geometry_dir), '--num_threads', str(args.threads),
        '--log', 'static-cached-query-metadata-replay']
    namespace = {'__name__': 'static_native_replay', '__file__': str(source_path),
                 '_static_record_frame': record_frame, '_static_record_query': record_query}
    start = time.perf_counter()
    status, error = 'COMPLETE_REPLAY_PENDING_PARITY', None
    try:
        exec(compile(transformed, str(source_path), 'exec'), namespace)
        namespace['main'](namespace['parse_args']())
    except Exception:
        status, error = 'FAILED', traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        queries.close()
        receipt = {'status': status, 'error': error, **current, 'command': command,
                   'source_sha256': sha256_file(source_path), 'mapper_argv': sys.argv,
                   'image_queries_added': 0, 'vlm_mode': 'CACHE_ONLY_NO_MODEL_LOADED',
                   'seconds': time.perf_counter()-start,
                   'parity_status': 'NOT_VERIFIED_NO_METADATA_PROMOTED'}
        binding_module = sys.modules.get('consistent_gsm')
        if binding_module is not None:
            binary = Path(binding_module.__file__)
            receipt['native_binding_binary'] = {'path': str(binary), 'sha256': sha256_file(binary)}
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
