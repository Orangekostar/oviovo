#!/usr/bin/env python3
"""Stream one released dense branch over fixed native B0 query slots, without GT."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import load_native_cache, sha256_file
from src.static_ovmap.dense_features import pool_owner_features
from src.static_ovmap.feature_refine import geometry_edges, refine_sparse
from src.static_ovmap.gla_backend import GLABackend
from src.static_ovmap.query_scores import score_queries

CONDITIONS = ['D0_ROI', 'D1', 'D2', 'D3_L01', 'D3_L02', 'D3_L03']


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('gla-root', 'dino-root', 'anyup-root', 'weights', 'native-cache', 'baseline-readout',
                 'geometry-support', 'native-text-cache', 'scene-root', 'intrinsics', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit-frames', type=int)
    args = p.parse_args()
    torch.set_num_threads(8)
    baseline = json.loads(args.baseline_readout.read_text())
    geometry = json.loads(args.geometry_support.read_text())
    if geometry['native_cache_sha256'] != sha256_file(args.native_cache):
        raise ValueError('native query cache and geometric owner identity differ')
    native_text = np.load(args.native_text_cache, allow_pickle=False)
    text_metadata = json.loads(args.native_text_cache.with_suffix('.json').read_text())
    bank, _ = load_native_cache(args.native_cache, scene_id=baseline['scene'],
        feature_space_id=str(native_text['feature_space_id']), source_config_hash='immutable_native_slot_selection',
        history_scope='retained_native_top10')
    slots, frame_slots = [], defaultdict(list)
    for owner, observations in sorted(bank.items()):
        previous = baseline['observations'][str(owner)]
        if previous is None:
            continue
        lookup = {o.source_query_id: o for o in observations}
        if not 2 <= len(previous['selected_query_ids']) <= 8:
            raise ValueError('native B0 eligibility/query budget differs')
        for query_id in previous['selected_query_ids']:
            obs = lookup[query_id]
            frame_slots[obs.frame_id].append(len(slots))
            slots.append({'owner': owner, 'frame_id': obs.frame_id, 'source_query_id': query_id,
                          'bbox': list(obs.crop_bbox_xyxy), 'native_area': obs.visible_area_px})
    frames = sorted(frame_slots)
    if not set(frames).issubset(geometry['input_frame_ids']):
        raise ValueError('selected query frames absent from measured geometry inputs')
    start = time.perf_counter()
    model = GLABackend(args.gla_root, args.dino_root, args.anyup_root, args.weights)
    inputs = [args.native_cache, args.baseline_readout, args.geometry_support, args.native_text_cache,
              args.native_text_cache.with_suffix('.json'),
              args.scene_root/'traj.txt', args.intrinsics, Path(__file__),
              Path(__file__).resolve().parents[2]/'src/static_ovmap/feature_refine.py',
              Path(__file__).resolve().parents[2]/'src/static_ovmap/query_scores.py']
    binding = {'input_sha256': {str(f): sha256_file(f) for f in inputs}, 'model_identity': model.identity,
               'feature_space_id': model.space, 'roi_feature_space_id': model.roi_space,
               'slots': slots, 'frames': frames, 'conditions': CONDITIONS}
    if args.output.exists():
        if not args.resume or json.loads((args.output/'binding.json').read_text()) != binding:
            raise ValueError('resume requires exactly matching data, code, query slots and feature spaces')
    else:
        args.output.mkdir(parents=True)
        (args.output/'frames').mkdir()
        write_json(args.output/'binding.json', binding)
    write_json(args.output/'job.json', {'status': 'RUNNING', 'command': sys.argv, 'frame_count': len(frames)})
    labels, valid_ids = text_metadata['labels'], native_text['valid_ids']
    if len(labels) != len(valid_ids):
        raise ValueError('native vocabulary metadata and explicit IDs differ')
    text_features = model.text_features(labels).cpu().numpy()
    np.savez(args.output/'dense_text.npz', text=text_features, valid_ids=valid_ids,
             feature_space_id=model.space, roi_feature_space_id=model.roi_space)
    load_and_text_seconds = time.perf_counter()-start
    camera = json.loads(args.intrinsics.read_text())['camera']
    processed = 0
    for frame in frames:
        payload_path = args.output/'frames'/f'{frame:06d}.npz'
        receipt_path = payload_path.with_suffix('.json')
        image_path = args.scene_root/'results'/f'frame{frame:06d}.jpg'
        depth_path = args.scene_root/'results'/f'depth{frame:06d}.png'
        owner_path = args.geometry_support.parent/'visible_owners'/f'{frame:06d}.npz'
        hashes = {str(f):sha256_file(f) for f in (image_path, depth_path, owner_path)}
        if payload_path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['input_sha256'] != hashes or receipt['payload_sha256'] != sha256_file(payload_path):
                raise ValueError('cached frame artifact changed')
            continue
        selected = frame_slots[frame]
        requested = sorted({slots[i]['owner'] for i in selected})
        image = Image.open(image_path).convert('RGB')
        with Image.open(depth_path) as image_depth:
            depth = np.asarray(image_depth).astype(np.float32)/camera['scale']
        with np.load(owner_path, allow_pickle=False) as loaded:
            owners = loaded['owners']
        features = np.zeros((len(CONDITIONS), len(selected), 512), dtype=np.float32)
        valid = np.zeros((len(CONDITIONS), len(selected)), dtype=bool)
        prototypes = np.zeros((len(CONDITIONS), len(selected), 4, 512), dtype=np.float32)
        prototype_pixels = np.zeros((len(CONDITIONS), len(selected), 4), dtype=np.int64)
        times, peaks = {}, {}
        torch.cuda.reset_peak_memory_stats()
        stage_start = time.perf_counter()
        for local, slot_index in enumerate(selected):
            feature = model.roi(image, slots[slot_index]['bbox']).cpu().numpy()
            features[0, local] = feature
            valid[0, local] = True
            prototypes[0, local, 0] = feature
            prototype_pixels[0, local, 0] = 1
        torch.cuda.synchronize()
        times['roi_seconds'] = time.perf_counter()-stage_start
        peaks['roi'] = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stage_start = time.perf_counter()
        low, info = model.dense(image)
        torch.cuda.synchronize()
        times['dense_seconds'] = time.perf_counter()-stage_start
        peaks['dense'] = torch.cuda.max_memory_allocated()
        pools = {}
        for condition, method in [('D1', 'bilinear'), ('D2', 'anyup')]:
            torch.cuda.reset_peak_memory_stats()
            stage_start = time.perf_counter()
            high = model.upsample(image, low, method)
            torch.cuda.synchronize()
            times[method+'_upsample_seconds'] = time.perf_counter()-stage_start
            peaks[method] = torch.cuda.max_memory_allocated()
            stage_start = time.perf_counter()
            cpu = high[0].permute(1, 2, 0).contiguous().cpu().numpy()
            del high
            pools[condition] = pool_owner_features(cpu, owners, requested)
            times[method+'_transfer_pool_seconds'] = time.perf_counter()-stage_start
            if condition == 'D2':
                stage_start = time.perf_counter()
                sources, targets = geometry_edges(owners, depth, owners > 0)
                neighbor = refine_sparse(cpu.reshape(-1, 512), sources, targets, mixing=1., temperature=1.)
                neighbor_pool = pool_owner_features(neighbor.reshape(cpu.shape), owners, requested)
                for name, mixing in [('D3_L01', .1), ('D3_L02', .2), ('D3_L03', .3)]:
                    pools[name] = {}
                    for owner in requested:
                        original, smooth = pools['D2'][owner], neighbor_pool[owner]
                        if original is None:
                            pools[name][owner] = None
                        else:
                            pools[name][owner] = {'feature': (1-mixing)*original['feature']+mixing*smooth['feature'],
                                'pixels': original['pixels'], 'prototypes': [
                                    {**a, 'feature': (1-mixing)*a['feature']+mixing*b['feature']}
                                    for a, b in zip(original['prototypes'], smooth['prototypes'], strict=True)]}
                times['sparse_neighbor_and_all_lambda_pool_seconds'] = time.perf_counter()-stage_start
                edge_count = len(sources)
                del neighbor, sources, targets, neighbor_pool
            del cpu
        del low
        for c, condition in enumerate(CONDITIONS[1:], start=1):
            for local, slot_index in enumerate(selected):
                pool = pools[condition][slots[slot_index]['owner']]
                if pool is None:
                    continue
                features[c, local] = pool['feature']; valid[c, local] = True
                for prototype in pool['prototypes']:
                    q = prototype['image_quadrant']
                    prototypes[c, local, q] = prototype['feature']
                    prototype_pixels[c, local, q] = prototype['pixels']
        if not np.isfinite(features).all() or not np.isfinite(prototypes).all():
            raise ValueError('nonfinite feature payload')
        np.savez_compressed(payload_path, features=features, valid=valid, prototypes=prototypes,
                            prototype_pixels=prototype_pixels, slot_indices=np.asarray(selected))
        receipt = {'frame_id': frame, 'input_sha256': hashes, 'payload_sha256': sha256_file(payload_path),
            'times': times, 'peak_gpu_allocated_bytes': peaks, 'dense_info': info,
            'sparse_edge_count': edge_count, 'sparse_surface_nodes': int(owners.size),
            'encoder_calls': {'ROI_CLIP': len(selected), 'dense_frames': 1,
                              'dense_CLIP_windows': info['window_count'], 'DINO_windows': info['window_count']},
            'missing_slot_count': {c: int((~valid[i]).sum()) for i, c in enumerate(CONDITIONS)}}
        write_json(receipt_path, receipt)
        processed += 1
        write_json(args.output/'job.json', {'status': 'RUNNING', 'command': sys.argv,
            'last_completed_frame_id': frame, 'completed_frame_count': sum((args.output/'frames'/f'{f:06d}.json').exists() for f in frames),
            'required_frame_count': len(frames)})
        print('frame', frame, times, 'missing', receipt['missing_slot_count'], flush=True)
        if args.limit_frames is not None and processed >= args.limit_frames:
            write_json(args.output/'job.json', {'status': 'PARTIAL_FEATURE_CACHE', 'command': sys.argv,
                'required_frame_count': len(frames), 'new_frames_this_run': processed})
            return
    # Reduce immutable frame caches; no full-resolution field is retained on disk.
    observations = np.zeros((len(CONDITIONS), len(slots), 512), dtype=np.float32)
    obs_valid = np.zeros((len(CONDITIONS), len(slots)), dtype=bool)
    candidates = np.zeros((len(CONDITIONS), len(slots), 4, 512), dtype=np.float32)
    candidate_pixels = np.zeros((len(CONDITIONS), len(slots), 4), dtype=np.int64)
    frame_receipts = []
    for frame in frames:
        path = args.output/'frames'/f'{frame:06d}.npz'
        with np.load(path, allow_pickle=False) as data:
            indices = data['slot_indices']
            observations[:, indices] = data['features']; obs_valid[:, indices] = data['valid']
            candidates[:, indices] = data['prototypes']; candidate_pixels[:, indices] = data['prototype_pixels']
        frame_receipts.append(json.loads(path.with_suffix('.json').read_text()))
    object_ids = sorted({slot['owner'] for slot in slots})
    retained = np.zeros((len(CONDITIONS), len(object_ids), 4, 512), dtype=np.float32)
    retained_valid = np.zeros((len(CONDITIONS), len(object_ids), 4), dtype=bool)
    retained_source_slot = np.full((len(CONDITIONS), len(object_ids), 4), -1, dtype=np.int64)
    retained_quadrant = np.full_like(retained_source_slot, -1)
    readout_dir = args.output/'readout'; readout_dir.mkdir(exist_ok=True)
    shutil.copy2(args.baseline_readout, readout_dir/'B0.json')
    for c, condition in enumerate(CONDITIONS):
        reduction_start = time.perf_counter()
        predictions = {key: None for key in baseline['observations']}
        lost = []
        for oi, owner in enumerate(object_ids):
            chosen = [i for i, slot in enumerate(slots) if slot['owner'] == owner and obs_valid[c, i]]
            if len(chosen) >= 2:
                weights = np.asarray([slots[i]['native_area'] for i in chosen], dtype=np.float64)
                fused = (observations[c, chosen]*weights[:, None]).sum(axis=0)/(weights.sum()+1e-6)
                space = model.roi_space if c == 0 else model.space
                predictions[str(owner)] = {**score_queries(fused, text_features, valid_ids, space, space),
                    'selected_query_ids': [slots[i]['source_query_id'] for i in chosen],
                    'selected_frame_ids': [slots[i]['frame_id'] for i in chosen], 'selected_count': len(chosen)}
            else:
                lost.append(owner)
            options = [(i, q) for i in chosen for q in range(4) if candidate_pixels[c, i, q] > 0]
            if options:
                vectors = np.stack([candidates[c, i, q] for i, q in options])
                unit = vectors/np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
                picked = [max(range(len(options)), key=lambda j: candidate_pixels[c, *options[j]])]
                while len(picked) < min(4, len(options)):
                    similarity = (unit @ unit[picked].T).max(axis=1)
                    similarity[picked] = np.inf
                    picked.append(int(similarity.argmin()))
                for j, choice in enumerate(picked):
                    slot, quadrant = options[choice]
                    retained[c, oi, j] = candidates[c, slot, quadrant]; retained_valid[c, oi, j] = True
                    retained_source_slot[c, oi, j] = slot; retained_quadrant[c, oi, j] = quadrant
        document = {'condition': condition, 'scene': baseline['scene'], 'observations': predictions,
            'readout_seconds': time.perf_counter()-reduction_start, 'query_budget': 8, 'prototype_budget': 4,
            'history_scope': 'fixed_native_B0_selected_slots_from_retained_top10',
            'quality_mode': 'new_encoder_features_no_GT_selection', 'direction_mode': 'native_fixed_slots',
            'mode': 'STATIC_OFFLINE_DENSE_NATIVE_OWNER_READOUT', 'lost_eligible_owners': lost,
            'geometry': 'UNCHANGED_B0_NATIVE_OWNERS', 'single_query_fallback': False,
            'feature_space_id': model.roi_space if c == 0 else model.space,
            'prototype_rule': 'four_diverse_observed_image_quadrant_features_not_canonical_object_parts',
            'AP_confidence': 'released_export_area_score_not_these_cosine_scores'}
        write_json(readout_dir/f'{condition}.json', document)
    np.savez_compressed(args.output/'retained_features.npz', observations=observations, observation_valid=obs_valid,
        prototypes=retained, prototype_valid=retained_valid, prototype_source_slot=retained_source_slot,
        prototype_quadrant=retained_quadrant, object_ids=np.asarray(object_ids))
    totals = {key: sum(f['times'][key] for f in frame_receipts) for key in frame_receipts[0]['times']}
    receipt = {'status': 'COMPLETE_DENSE_FEATURE_READOUT_PENDING_EVALUATION', 'command': sys.argv,
        'binding_sha256': sha256_file(args.output/'binding.json'), 'frame_count': len(frames), 'slot_count': len(slots),
        'model_load_and_text_seconds_this_invocation': load_and_text_seconds, 'stage_total_seconds': totals,
        'peak_gpu_allocated_bytes': max(max(f['peak_gpu_allocated_bytes'].values()) for f in frame_receipts),
        'process_peak_rss_bytes_this_invocation': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'retained_payload_array_bytes': observations.nbytes+obs_valid.nbytes+retained.nbytes+retained_valid.nbytes+retained_source_slot.nbytes+retained_quadrant.nbytes,
        'retained_payload_file_bytes': (args.output/'retained_features.npz').stat().st_size,
        'frame_cache_bytes': sum(p.stat().st_size for p in (args.output/'frames').iterdir()),
        'encoder_calls': {key:sum(f['encoder_calls'][key] for f in frame_receipts) for key in frame_receipts[0]['encoder_calls']},
        'cost_scope': 'new_dense_branch_only_historical_native_frontend_not_timed; D1_D2_D3_share_dense_forward',
        'readout_conditions': CONDITIONS, 'geometry_support_sha256': sha256_file(args.geometry_support)}
    write_json(args.output/'dense_receipt.json', receipt)
    write_json(readout_dir/'input_binding.json', binding)
    write_json(args.output/'job.json', {'status': 'COMPLETE', 'command': sys.argv, 'required_frame_count': len(frames)})
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
