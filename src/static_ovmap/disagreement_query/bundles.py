"""Portable small evidence; large immutable source partitions remain external."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.evaluation import GeometryIdentity, PredictionPayload
from static_ovmap.module_validation.scannet_study import load_prediction

from .common import REPO, _array_digest, read, verified, write


def unpack_json(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream: return json.load(stream)
    return read(path)


def restore_payload(record, baseline):
    if baseline.geometry.to_dict() != record['header']['geometry']:
        raise ValueError('restoration source geometry differs')
    if _array_digest(baseline.owner_ids) != record['owner_partition_digest']:
        raise ValueError('restoration source owner partition differs')
    registry = np.asarray(record['registry'], np.int64)
    owners, labels = registry[:, 0], registry[:, 1]
    active = np.unique(baseline.owner_ids)
    if not np.array_equal(owners, np.unique(np.r_[0, active])):
        raise ValueError('portable registry is incomplete')
    semantic = labels[np.searchsorted(owners, baseline.owner_ids)]
    h = record['header']
    result = PredictionPayload(h['method_id'], h['branch'], h['scene_id'],
        GeometryIdentity(**h['geometry']), baseline.owner_ids, semantic,
        tuple(tuple(r) for r in h['instance_ranks']), h['logical_cost'], h['metadata'])
    result.lock()
    if result.prediction_key != h['prediction_key'] or result.record_key != h['record_key']:
        raise ValueError('portable payload identity reconstruction failed')
    return result


def build_bundle(binding, screen):
    root = Path(binding['output_root'])
    dest = REPO/'artifacts/static_ovmap/disagreement_query_v1/bundles'
    dest.mkdir(parents=True, exist_ok=True)
    files, by_path = {}, {}
    def add(path):
        path = Path(path)
        if str(path) in by_path: return by_path[str(path)]
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        suffix = '.json.gz' if path.suffix == '.json' else path.suffix
        relative = 'files/'+sha+suffix
        output = dest/relative; output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            if path.suffix == '.json':
                with output.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as zipped:
                    zipped.write(raw)
            else: shutil.copyfile(path, output)
        stored = hashlib.sha256(output.read_bytes()).hexdigest()
        files[sha] = {'path': relative, 'source_sha256': sha, 'source_bytes': len(raw),
                      'stored_sha256': stored, 'stored_bytes': output.stat().st_size,
                      'encoding': 'GZIP_JSON' if suffix == '.json.gz' else 'ORIGINAL_BYTES'}
        by_path[str(path)] = relative
        return relative
    scenes = [s for group in binding['screen_cohorts'].values() for s in group]
    evidence, scene_records, predictions = {}, {}, []
    for stage in ('engineering_anchor', 'screen_anchor', 'screen_second'):
        receipt = verified(root/'acquisition'/stage/'receipt.json')
        add(root/'acquisition'/stage/'receipt.json')
        for scene, rows in receipt['records'].items():
            for rid, row in rows.items():
                arrays = add(row['arrays']['path']); mask = add(row['mask']['path'])
                key = row['content_key']
                item = {'scene': scene, 'region_id': rid, 'arrays': arrays, 'mask': mask, 'record': row}
                if key in evidence and evidence[key]['record']['arrays']['sha256'] != row['arrays']['sha256']:
                    raise ValueError('same content key has contradictory feature bytes')
                evidence[key] = item
    for scene in scenes:
        plan = verified(root/'plans'/scene/'manifest.json')
        supports = {owner: add(row['support']['path']) for owner, row in plan['owners'].items()}
        scene_records[scene] = {'plan': add(root/'plans'/scene/'manifest.json'), 'supports': supports,
            'choices': add(root/'choices'/scene/'receipt.json'),
            'surface': add(root/'diagnostics/surface'/scene/'receipt.json'), 'stages': {}}
        for stage in ('screen_A', 'screen_B'):
            lock = verified(root/'predictions'/stage/scene/'receipt.json')
            scene_records[scene]['stages'][stage] = {
                'lock': add(root/'predictions'/stage/scene/'receipt.json'),
                'analysis': add(root/'analysis'/stage/scene/'receipt.json'), 'decisions': {}}
            for method, item in lock['methods'].items():
                payload = load_prediction(item['manifest'])
                labels = owner_labels(payload)
                header = read(item['manifest']); header.pop('arrays')
                predictions.append({'scene': scene, 'stage': stage, 'method': method,
                    'base': 'D2' if method == 'DQ00_D2' else 'G1', 'header': header,
                    'owner_partition_digest': _array_digest(payload.owner_ids),
                    'registry': [[0, 0], *[[o, labels[o]] for o in sorted(labels)]]})
                if method not in ('DQ00_D2', 'DQ01_G1'):
                    decision = verified(root/'decisions'/stage/scene/(method+'.json'))
                    scene_records[scene]['stages'][stage]['decisions'][method] = {
                        'metadata': add(root/'decisions'/stage/scene/(method+'.json')),
                        'posteriors': add(decision['lossless_posteriors']['path'])}
    vocabulary = {}
    for group in binding['cohorts'].values():
        for scene in group:
            folder = root/'diagnostics/vocabulary'/scene
            vocabulary[scene] = {'summary': add(folder/'receipt.json'), 'records': add(folder/'records.json'),
                                 'source_scores': add(folder/'source_scores.npz')}
    score_records = {}
    for row in screen['scene_metrics']:
        path = row['evaluation_receipt']; score = read(path)
        for file in (Path(path), Path(score['manifest']),
                     Path(score['manifest']).with_name('matches.json.gz'), Path(score['manifest']).with_name('trace.json.gz')):
            add(file)
        score_records[row['evaluation_identity']] = by_path[path]
    per_class_pools = {cohort: {m: add(pool['per_class_receipt']) for m, pool in rows.items()}
                       for cohort, rows in screen['pooled_metrics'].items()}
    payload_path = dest/'payload_registries.json.gz'
    raw = json.dumps(predictions, sort_keys=True, separators=(',', ':')).encode()
    with payload_path.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as zipped: zipped.write(raw)
    receipts = {p: add(root/p) for p in (
        'source_binding.json', 'query_kernel_freeze.json', 'screen_prediction_freeze.json',
        'diagnostics/summary.json', 'screen_query_selection.json', 'screen_selection.json',
        'engineering/receipt.json', 'verification/original_scorer_parity.json', 'regression.json', 'timing.json',
        'reference_validation.json', 'reference_tests.log', 'implementation_validation.json',
        'execution/production_tests.json', 'execution/production_tests.log', 'initial_state.json', 'storage_preflight.json')}
    return write(dest/'manifest.json', {'status': 'PORTABLE_SMALL_EVIDENCE_COMPLETE', 'files': files,
        'source_locations': by_path, 'scenes': scene_records, 'vocabulary': vocabulary, 'evidence': evidence,
        'payload_registries': {'path': payload_path.name,
            'sha256': hashlib.sha256(payload_path.read_bytes()).hexdigest(), 'count': len(predictions)},
        'scorer_receipts': score_records, 'per_class_pools': per_class_pools, 'receipts': receipts,
        'external_large_inputs': ['RGB-D', 'full source geometry and G1/D2 owner arrays', 'FC weights', 'dense features'],
        'restoration': 'restore_payload(record, baseline) verifies full prediction and record identities',
        'no_dense_features_or_new_maps_in_git': True})
