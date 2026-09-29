"""Real-output and paired-input checks required before publication."""

import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _array_digest

from .binding import INDEX
from .binding import read_json_or_gzip as read
from .evaluation import RANKS, configurations, probabilities
from .evidence import VARIANTS, unit


def verify(binding):
    started = time.monotonic()
    root = Path(binding['output_root'])
    spec, lock = read(binding['spec']), read(root / 'transfer_lock.json')
    wave = read(binding['wave1_binding'])
    wave_root = Path(wave['output_root'])
    configurations_frozen = configurations(spec, lock['choices'])
    if lock['combination']['status'] == 'MEASURED':
        configurations_frozen.append({k: lock['combination'][k] for k in ('id', 'method', 'parameter')})
    canonical = [c['id'] for c in configurations_frozen]
    checks, pairs_checked, source_errors = [], 0, []
    for scene in spec['datasets']['calibration'] + spec['datasets']['replica']:
        bundle_path = root / 'evidence' / scene / 'frozen.json.gz'
        receipt = read(bundle_path.parent / 'receipt.json')
        INDEX.identity(bundle_path, receipt['bundle'])
        bundle = read(bundle_path)
        if canonical_digest({k: v for k, v in bundle.items() if k != 'identity'}) != bundle['identity']:
            raise ValueError('frozen evidence content changed')
        ids = bundle['valid_ids']
        for method, field in [('AW_E03_FC_FROZEN_A7', 'base'), ('AW_E03_OVR_A7', 'ovr_base')]:
            original = read(wave_root / 'probabilities' / scene / (method + '.json.gz'))
            for owner, obj in bundle['objects'].items():
                if not np.array_equal(obj[field], original[owner]['probabilities']):
                    raise ValueError('original calibrated full probabilities differ')
        configs = list(configurations_frozen)
        if scene in spec['datasets']['calibration']:
            configs += [c for c in configurations(spec) if c['id'] not in canonical]
        for config in configs:
            method = config['id']
            value = read(root / 'locked' / scene / (method + '.json'))
            if set(value['labels']) != set(bundle['objects']):
                raise ValueError('whole-map prediction dropped an owner')
            if config['method'] not in spec['controls']:
                for owner, obj in bundle['objects'].items():
                    p = probabilities(obj, bundle['temperatures'], config['method'], config['parameter'])
                    label = obj['fallback_label'] if p is None else ids[int(np.argmax(p))]
                    if value['labels'][owner] != label:
                        raise ValueError('whole-map label differs from prescribed operator')
            else:
                original = read(wave_root / 'locked' / scene / (method + '.json'))
                if original['labels'] != value['labels']:
                    raise ValueError('original control labels changed')
            if scene in spec['datasets']['replica'] and (root / 'locked' / scene / (method + '.json')).stat().st_mtime_ns < (root / 'transfer_lock.json').stat().st_mtime_ns:
                raise ValueError('Replica predictions predate transfer lock')
            for rank in RANKS:
                row = read(root / 'rows' / scene / method / (rank + '.json'))
                evaluation = read(row['evaluation_receipt'])
                if row['metrics'] != evaluation['metrics'] or not all(evaluation['trace_parity'][k] for k in ('ap_exact', 'pr_and_fn_exact')):
                    raise ValueError('evaluation row/trace parity changed')
                if config['method'] in spec['controls']:
                    parent = read(wave_root / 'rows' / scene / method / (rank + '.json'))
                    if row['metrics'] != parent['metrics']:
                        raise ValueError('original control metrics changed')
                checks.append((scene, method, rank))
        # Independently validate actual common F/O masks and reaggregate source scores.
        manifest = read(wave['scenes'][scene]['static_manifest']['path'])
        sources = {n: read(wave_root / 'e03' / scene / (v + '.json')) for n, v in VARIANTS.items()}
        prototypes, views = {}, {}
        for name, variant in VARIANTS.items():
            tp = sources[name]['text_identity']
            INDEX.identity(tp['path'], tp)
            with np.load(tp['path'], allow_pickle=False) as arrays:
                prototypes[name] = arrays['text_embeddings']
            views[name] = {}
            for obj in bundle['objects'].values():
                for rid in obj['common_requests']:
                    if rid in views[name]:
                        continue
                    record_path = wave_root / 'e03' / scene / variant / 'requests' / (rid + '.json')
                    record = read(record_path)
                    INDEX.expected_output(record['arrays_path'], record)
                    if record['status'] != 'COMPLETE':
                        raise ValueError('paired request unavailable')
                    with np.load(record['arrays_path'], allow_pickle=False) as arrays:
                        shape = tuple(arrays['original_shape'])
                        mask = np.unpackbits(arrays['original_bits'], count=int(np.prod(shape))).reshape(shape).astype(bool)
                        req = manifest['requests'][rid].get('request', manifest['requests'][rid])
                        if _array_digest(mask) != req['target_mask_sha256']:
                            raise ValueError('paired mask differs from original target')
                        views[name][rid] = arrays['feature']
        for obj in bundle['objects'].values():
            common = obj['common_requests']
            if not common:
                continue
            areas = [manifest['requests'][r].get('request', manifest['requests'][r])['visible_target_pixels'] for r in common]
            for name in ('F', 'O'):
                aggregate = unit(np.average([views[name][r] for r in common], axis=0, weights=areas))
                scores = prototypes[name] @ aggregate
                error = float(np.max(np.abs(scores - obj['paired_' + name])))
                source_errors.append(error)
                if error > 1e-12:
                    raise ValueError('paired original score reconstruction differs')
            pairs_checked += 1
    for split in ('cal', 'replica'):
        for method in canonical:
            for rank in RANKS:
                value = read(root / 'pooled' / split / method / (rank + '.json'))
                scenes = spec['datasets']['calibration' if split == 'cal' else 'replica']
                expected = [read(root / 'rows' / scene / method / (rank + '.json'))['evaluation_identity'] for scene in scenes]
                if value['ordered_inputs'] != expected or value['scene_order'] != scenes:
                    raise ValueError('pool is not the declared ordered complete split')
                if method in spec['controls']:
                    old = read(wave_root / 'pooled' / split / method / (rank + '.json'))
                    if value['metrics'] != old['metrics']:
                        raise ValueError('pooled original control metrics differ')
    result = {'status': 'REAL_OUTPUTS_VERIFIED', 'binding': binding['identity'], 'transfer': lock['identity'],
              'canonical_methods': len(canonical), 'scene_rank_rows_checked': len(set(checks)),
              'paired_owners_reconstructed': pairs_checked, 'paired_max_score_error': max(source_errors, default=0),
              'canonical_pools_checked': len(canonical) * 4, 'old_control_label_metric_parity': True,
              'Replica_after_freeze': True, 'new_model_inference': 0,
              'elapsed_seconds': time.monotonic() - started, 'input_manifest': INDEX.entries()}
    result['identity'] = canonical_digest({k: v for k, v in result.items() if k != 'elapsed_seconds'})
    path = root / 'review' / 'outputs.json'
    if path.exists():
        old = read(path)
        if old['identity'] != result['identity']:
            raise ValueError('verified output inputs changed')
        return old
    write_once(path, result)
    return result
