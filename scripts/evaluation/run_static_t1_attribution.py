#!/usr/bin/env python3
"""Generate frozen Room0 attribution predictions from two cached T0 runs; no GT loads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.fusion_attribution import (build_frozen_candidates, apply_label_reuse,
    apply_fusion_nms, assignment_scores, resolve_native_owners, fixed_regions, serialize_scores)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checked_file(path, inventory, expected=None):
    path = Path(path)
    key = str(path)
    if key not in inventory:
        inventory[key] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    if expected is not None and inventory[key]['sha256'] != expected:
        raise ValueError('receipt hash mismatch: '+key)
    return inventory[key]['sha256']


def generate_conditions(pool, output, config, native_labels, native_readout_id):
    off, off_events = apply_label_reuse(pool, enabled=False)
    on, on_events = apply_label_reuse(pool)
    kept_nms, edges_nms = apply_fusion_nms(pool)
    kept_all, edges_all = apply_fusion_nms(pool, enabled=False)
    owner_cache = {}
    summaries = []
    for condition in config['prediction_conditions']:
        start = time.perf_counter()
        labels, events = (on.copy(), on_events) if condition in (
            'AT_U10', 'AT_U11', 'AT_ASSIGN_RAW', 'AT_ASSIGN_CLASS_NORM', 'AT_ASSIGN_OVI_FILL',
            'AT_RANK_CLASS_NORM', 'AT_NMS_SF_FIRST', 'AT_NATIVE_U11') else (off.copy(), off_events)
        uses_nms = condition in ('AT_U01', 'AT_U11', 'AT_ASSIGN_RAW', 'AT_ASSIGN_CLASS_NORM',
            'AT_ASSIGN_OVI_FILL', 'AT_RANK_CLASS_NORM', 'AT_NMS_SF_FIRST', 'AT_NATIVE_U11')
        kept, edges = (kept_nms.copy(), edges_nms) if uses_nms else (kept_all.copy(), edges_all)
        readout_id = pool.records[0]['ovi_readout_id'] if pool.records else 'empty'
        if condition.startswith('AT_NATIVE_'):
            if native_labels is None:
                summaries.append({'condition': condition, 'status': 'BLOCKED', 'reason': 'same-generation native readout unavailable'})
                continue
            labels, events = apply_label_reuse(pool, enabled=condition == 'AT_NATIVE_U11', ovi_labels=native_labels)
            readout_id = native_readout_id
        if condition in ('AT_O_AREA', 'AT_NATIVE_O_AREA'):
            kept = kept[kept < pool.ovi_count]
        elif condition in ('AT_S_AREA', 'AT_S_RELEASED'):
            kept = kept[kept >= pool.ovi_count]
        elif condition == 'AT_NMS_SF_FIRST':
            kept, edges = apply_fusion_nms(pool, sf_first=True)
        rank = pool.areas.astype(float)
        assignment = rank.copy()
        rule = 'raw'
        groups = np.zeros(len(rank), dtype=np.int64)
        if condition == 'AT_S_RELEASED':
            rank = pool.released_scores.copy(); assignment = rank.copy(); rule = 'released'
        if condition in ('AT_ASSIGN_CLASS_NORM', 'AT_RANK_CLASS_NORM'):
            normalized = assignment_scores(pool, kept, labels, 'class_norm')
            if condition == 'AT_ASSIGN_CLASS_NORM':
                assignment = normalized; rule = 'class_norm'
            else:
                rank = normalized
        if condition == 'AT_ASSIGN_OVI_FILL':
            groups[pool.ovi_count:] = 1; rule = 'ovi_fill'
        key = hashlib.sha256(kept.tobytes()+assignment.tobytes()+groups.tobytes()).hexdigest()
        assignment_start = time.perf_counter()
        if key not in owner_cache:
            owners = resolve_native_owners(pool.masks, kept, assignment, source_priority=groups)
            path = output/'owners'/f'{key}.npy'; path.parent.mkdir(exist_ok=True)
            np.save(path, owners)
            counts = np.bincount(owners, minlength=len(rank)+1)[1:]
            owner_cache[key] = (str(path), counts)
        owner_path, counts = owner_cache[key]
        assignment_seconds = time.perf_counter()-assignment_start
        strings, parsed = serialize_scores(rank)
        kept_set = set(kept.tolist())
        ledger = []
        for i, metadata in enumerate(pool.records):
            row = dict(metadata)
            row.update(events[i], condition_id=condition, ovi_readout_id=readout_id,
                kept=i in kept_set, condition_selected=i in kept_set,
                nms_visit_rank=edges[i]['nms_visit_rank'] if uses_nms else None,
                suppressed_by=None if edges[i]['suppressed_by'] is None else pool.records[edges[i]['suppressed_by']]['candidate_id'],
                suppression_iou=edges[i]['suppression_iou'],
                rank_score_full_precision=float(rank[i]), rank_score_serialized=strings[i],
                rank_score_parsed=float(parsed[i]), assignment_priority_full_precision=float(assignment[i]),
                assignment_source_priority=int(groups[i]), assignment_rule_id=rule,
                output_owner_id=i+1, owned_source_point_count=int(counts[i]))
            ledger.append(row)
        record = {'status': 'COMPLETE_PREDICTION', 'condition': condition, 'scene': config['scene'],
            'prediction_run_id': pool.records[0]['prediction_run_id'] if pool.records else 'empty',
            'GT_input': False, 'new_model_inferences': 0, 'extra_3D_pretraining': True,
            'ovi_readout_id': readout_id, 'kept': kept.tolist(), 'labels': labels.tolist(),
            'rank_scores': rank.tolist(), 'rank_scores_serialized': strings,
            'assignment_priorities': assignment.tolist(), 'assignment_groups': groups.tolist(),
            'assignment_rule': rule, 'owner_path': owner_path, 'owner_cache_key': key,
            'assignment_seconds_including_first_write': assignment_seconds,
            'condition_seconds': time.perf_counter()-start, 'ledger': ledger}
        write(output/(condition+'.json'), record)
        summaries.append({k:v for k,v in record.items() if k not in ('ledger','labels','rank_scores','rank_scores_serialized','assignment_priorities','assignment_groups')})
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['new_inference_allowed'] or (config['reuse_iou'], config['nms_iou']) != (.5, .7):
        raise ValueError('initial study parameters must remain frozen')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); inventory = {}
    historical = json.loads(Path(config['historical_t1_receipt']).read_text())
    for path, expected in historical['input_sha256'].items():
        checked_file(path, inventory, expected)
    checked_file(config['historical_t1'], inventory, historical['output_sha256'])
    large_index = json.loads((ROOT/'artifacts/static_ovmap/room0_spaceformer/large_artifacts.json').read_text())
    bound = {r['path']:r['sha256'] for r in large_index}
    checked_file(config['native_projection'], inventory, bound[config['native_projection']])
    checked_file(config['historical_ledger'], inventory)
    readout = json.loads(Path(config['readout']).read_text())
    readout_id = 'sha256:'+checked_file(config['readout'], inventory)
    binding_path = Path(config['readout']).parent/'input_binding.json'
    binding = json.loads(binding_path.read_text()); checked_file(binding_path, inventory)
    checked_file(binding['source_path'], inventory, binding['source_sha256'])
    native = Path(config['native_readout']); native_labels = None; native_readout_id = None
    if native.exists():
        checked_file(native, inventory); native_doc = json.loads(native.read_text())
        if native.parent != Path(config['readout']).parent or native_doc['scene'] != readout['scene']:
            raise ValueError('native and S1a must share bound generation')
        if {k for k,v in native_doc['observations'].items() if v is not None} != {k for k,v in readout['observations'].items() if v is not None}:
            raise ValueError('native/S1a eligibility differs')
        native_readout_id = 'sha256:'+inventory[str(native)]['sha256']
    support_path = Path(config['geometry_support'])
    support = json.loads(support_path.read_text())
    if support['native_mesh_sha256'] != inventory[config['native_mesh']]['sha256']:
        raise ValueError('geometry support is from another native mesh')
    native_owners = np.load(support_path.parent/'native_vertex_owners.npy', mmap_mode='r')
    with np.load(config['native_projection'], allow_pickle=False) as data:
        old_projection = {k:data[k] for k in data.files}
    if not np.array_equal(old_projection['owners'], np.where(old_projection['distance_squared'] < .05**2, native_owners[old_projection['nearest']], 0)):
        raise ValueError('cached native owner projection does not follow original strict threshold')
    with np.load(config['historical_t1'], allow_pickle=False) as d:
        reference = {k:d[k] for k in d.files}
    old_ledger = json.loads(Path(config['historical_ledger']).read_text())
    runs = []
    for run in config['runs']:
        run_start = time.perf_counter(); dest = args.output/run['id']; dest.mkdir()
        receipt = json.loads(Path(run['receipt']).read_text())
        checked_file(run['receipt'], inventory)
        checked_file(run['identity'], inventory, receipt['identity_sha256'])
        identity = json.loads(Path(run['identity']).read_text())
        if identity['precision'] != 'fp32':
            raise ValueError('both runs must be independent recorded FP32 predictions')
        checked_file(run['t0'], inventory, receipt['payload_sha256']['T0.npz'])
        with np.load(run['t0'], allow_pickle=False) as data:
            coord, masks, labels, scores, query_ids = [data[k] for k in ('coord','masks','class_ids','scores','query_ids')]
        if coord.shape != (len(old_projection['owners']), 3) or not np.isfinite(coord).all():
            raise ValueError('finite common-domain coordinates required')
        equal = np.array_equal(coord, reference['coord'])
        projection_start = time.perf_counter()
        if equal:
            owners = old_projection['owners']
        else:
            import open3d as o3d
            mesh = o3d.t.io.read_triangle_mesh(config['native_mesh'])
            tree = o3d.core.nns.NearestNeighborSearch(mesh.vertex.positions); tree.knn_index()
            nearest, distance = tree.knn_search(o3d.core.Tensor(np.ascontiguousarray(coord, dtype=np.float32)), 1)
            owners = np.where(distance.numpy().ravel() < .05**2, native_owners[nearest.numpy().ravel()], 0)
            np.savez_compressed(dest/'native_projection.npz', owners=owners, nearest=nearest.numpy().ravel(), distance_squared=distance.numpy().ravel())
        projection_seconds = time.perf_counter()-projection_start
        pool_start = time.perf_counter()
        pool = build_frozen_candidates(owners, readout['observations'], masks, labels, scores, query_ids,
            scene_id=config['scene'], prediction_run=run['id'], ovi_readout_id=readout_id)
        del masks
        preparation_seconds = time.perf_counter()-pool_start
        parity = None
        if run == config['runs'][0]:
            borrowed, events = apply_label_reuse(pool); kept, edges = apply_fusion_nms(pool)
            checks = {'coordinates': equal, 'masks': np.array_equal(pool.masks[kept], reference['masks']),
                'class_ids': np.array_equal(borrowed[kept], reference['class_ids']),
                'scores': np.array_equal(pool.areas[kept].astype(float), reference['scores']),
                'candidate_ids': np.array_equal(kept, reference['candidate_ids']),
                'ledger_count': len(pool.records) == len(old_ledger)}
            checks['ledger_decisions'] = checks['ledger_count'] and all(
                int(pool.areas[i]) == old['area'] and edges[i]['kept'] == old['kept']
                and edges[i]['suppressed_by'] == old['suppressed_by_candidate']
                and (old['source'] != 'SpaCeFormer' or (events[i]['reused_owner'] == old['reused_owner']
                    and pool.records[i]['best_owner_iou'] == old['best_owner_iou'])) for i,old in enumerate(old_ledger))
            write(dest/'historical_parity.json', checks)
            if not all(checks.values()):
                raise ValueError('historical T1 array/ledger parity failed')
            parity = checks
        np.save(dest/'coord.npy', coord); np.save(dest/'masks.npy', pool.masks)
        np.save(dest/'native_owners.npy', pool.native_owners); np.save(dest/'ious.npy', pool.ious)
        write(dest/'candidates.json', [dict(x) for x in pool.records])
        regions, registry, count = fixed_regions(pool)
        np.savez_compressed(dest/'regions.npz', region=regions, multiplicity=count)
        write(dest/'region_registry.json', registry)
        if native.exists():
            native_labels = {r['native_owner_id']: native_doc['observations'][str(r['native_owner_id'])]['class_id'] for r in pool.records[:pool.ovi_count]}
        conditions = generate_conditions(pool, dest, config, native_labels, native_readout_id)
        row = {'id':run['id'], 'status':'COMPLETE_PREDICTION', 't0':run['t0'],
            'coordinate_order_exact_equal_to_historical_primary':equal, 'native_projection_reused':equal,
            'point_count':len(coord), 'ovi_count':pool.ovi_count, 'sf_count':len(labels),
            'pool_preparation_seconds':preparation_seconds, 'projection_seconds':projection_seconds,
            'total_seconds':time.perf_counter()-run_start, 'historical_parity':parity, 'conditions':conditions}
        write(dest/'run.json', row); runs.append(row)
        print(run['id'], 'COMPLETE_PREDICTION', len(pool.records), flush=True)
        del pool
    write(args.output/'prediction_manifest.json', {'status':'COMPLETE_PREDICTION', 'GT_input':False,
        'new_model_inferences':0, 'runs':runs, 'input_inventory':inventory,
        'source_sha256':{str(f):sha256_file(f) for f in [Path(__file__),ROOT/'src/static_ovmap/fusion_attribution.py',args.config]},
        'command':sys.argv, 'elapsed_seconds':time.perf_counter()-start})


if __name__ == '__main__':
    main()
