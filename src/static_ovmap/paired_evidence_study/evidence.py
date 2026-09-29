"""Annotation-free frozen-score and observation recovery; no model loads."""

import pickle
import time
from pathlib import Path

import numpy as np

from src.static_ovmap.a7_evidence_upgrade.calibration import base_temperatures
from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.composition_study.object_evidence import (
    native_request_ids,
    owner_labels,
)
from src.static_ovmap.m2_reviewer_study.evaluation import plain, write_gzip
from src.static_ovmap.m2_reviewer_study.scores import read_scene
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _array_digest

from .binding import INDEX
from .binding import read_json_or_gzip as read
from .dependence import solve_variants
from .residuals import candidate_set, paired_residual, pool

B = 'AW_E03_FC_FROZEN_A7'
VARIANTS = {'F': 'AW_E03_FC_FROZEN', 'O': 'AW_E03_OVR'}


def unit(value):
    x = np.asarray(value, np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms <= 1e-12):
        raise ValueError('undefined feature direction')
    return x / norms


class ObservationReader:
    def __init__(self, data, model, decisions, static):
        self.data, self.model, self.static = data, model, static
        self.capture = read(data['capture'])
        INDEX.identity(data['capture'])
        self.frames = {f['frame_id']: f for f in self.capture['frames']}
        self.requests = {r['request_id']: r for f in self.frames.values() for r in f['requests']}
        self.requests.update({r['request']['request_id']: r['request'] for r in decisions['paid_requests']})
        self.masks = {}
        self.frame_arrays = {}

    def atom(self, rid, *, region=False, region_arrays=None):
        req = (self.static['requests'][rid] if region else self.requests[rid])
        req = req.get('request', req)
        mask_name = 'target_mask_sha256' if region else 'native_union_mask_sha256'
        key = (rid, mask_name)
        reason, path = None, None
        if key not in self.masks:
            try:
                if region:
                    path = region_arrays
                    with np.load(path, allow_pickle=False) as arr:
                        shape = tuple(arr['original_shape'])
                        mask = np.unpackbits(arr['original_bits'], count=int(np.prod(shape))).reshape(shape).astype(bool)
                else:
                    frame = self.frames[req['frame_id']]
                    path = Path(self.data['capture']).parent / frame['request_arrays']['path']
                    INDEX.identity(path, frame['request_arrays'])
                    if path not in self.frame_arrays:
                        with np.load(path, allow_pickle=False) as arr:
                            self.frame_arrays[path] = {k: arr[k] for k in arr.files if k.endswith('_union')}
                    mask = self.frame_arrays[path][rid + '_union']
                if _array_digest(mask) != req[mask_name]:
                    raise ValueError('observation support hash differs')
                self.masks[key] = mask
            except (FileNotFoundError, PermissionError, KeyError) as error:
                self.masks[key] = None
                reason = f'{type(error).__name__}: {error}'
        mask = self.masks[key]
        computation = {'model': self.model['identity'] if not region else 'REGION_SET_BY_CALLER',
                       'image': req['image_sha256'], 'bbox': req['bbox_xyxy'], 'mask': req[mask_name],
                       'policy': 'original_region' if region else req['crop_convention'], 'precision': 'float32'}
        return {'request_id': rid, 'image': req['image_sha256'], 'mask': mask,
                'support_status': 'KNOWN_SUPPORT' if mask is not None else 'UNKNOWN_SUPPORT',
                'support_error': reason, 'path': str(path), 'request': req,
                'computation': computation, 'area': req['visible_target_pixels']}


def _source(name, scores, features, areas, text, temperature, atoms, space):
    if len(features) != len(atoms):
        raise ValueError('view/atom alignment changed')
    keep, seen = [], {}
    for i, atom in enumerate(atoms):
        rid = atom['request_id']
        if rid in seen:
            previous = seen[rid]
            if areas[i] != areas[previous] or not np.array_equal(features[i], features[previous]) or atom['computation'] != atoms[previous]['computation']:
                raise ValueError('conflicting repeated observation')
        else:
            seen[rid] = i
            keep.append(i)
    features, areas = np.asarray(features)[keep], np.asarray(areas)[keep]
    atoms = [atoms[i] for i in keep]
    reconstructed = text @ unit(np.average(features, axis=0, weights=areas))
    error = float(np.max(np.abs(reconstructed - scores)))
    if error > 1e-6:
        raise ValueError(f'{name} float64 surrogate error {error} exceeds fixed tolerance')
    exact = canonical_digest({'space': space, 'atoms': [a['computation'] for a in atoms],
                              'areas': areas.tolist(), 'scores': list(scores), 'temperature': temperature})
    return {'name': name, 'scores': scores, 'features': features, 'areas': areas, 'text': text,
            'temperature': temperature, 'atoms': atoms, 'space': space,
            'computation_identity': exact, 'surrogate_max_error': error}


def recover(binding, scene):
    start = time.monotonic()
    output = Path(binding['output_root']) / 'evidence' / scene
    receipt_path = output / 'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        for entry in receipt['inputs']:
            INDEX.identity(entry['path'], entry)
        INDEX.identity(receipt['bundle']['path'], receipt['bundle'])
        return receipt
    wave = read(binding['wave1_binding'])
    root = Path(wave['output_root'])
    parent = read(binding['reviewer_binding'])
    ev = read_scene(parent, scene, INDEX)
    labels, ids = owner_labels(ev.native), ev.sources['N0']['valid_ids']
    config = ev.config
    data, model = config['scenes'][scene], config['models']['native']
    sources = {'N': ev.cosine_native, 'Q': ev.sources['Q_GAIN']}
    temperatures = base_temperatures(wave, scene)
    ts = {'N': temperatures['N0'], 'Q': temperatures['Q_GAIN']}
    region, text = {}, {}
    for name, variant in VARIANTS.items():
        path = root / 'e03' / scene / (variant + '.json')
        INDEX.identity(path)
        sources[name] = read(path)
        if sources[name]['valid_ids'] != ids or sources[name]['native_record_key'] != ev.native.record_key:
            raise ValueError('region vocabulary/geometry mismatch')
        fits_path = root / 'calibration' / (variant + '.json')
        INDEX.identity(fits_path)
        fits = read(fits_path)
        ts[name] = (fits['folds'][scene] if wave['scenes'][scene]['role'] == 'CAL' else fits['final'])['temperature']
        directory = root / 'e03' / scene / variant
        worker = read(directory / 'receipt.json')
        INDEX.identity(directory / 'receipt.json')
        region[name] = {'directory': directory, 'receipt': worker, 'requests': {}, 'features': {}}
        tp = sources[name]['text_identity']
        INDEX.identity(tp['path'], tp)
        with np.load(tp['path'], allow_pickle=False) as arr:
            text[name] = arr['text_embeddings']
            if arr['valid_ids'].tolist() != ids or arr['class_names'].tolist() != model['class_names']:
                raise ValueError('region text policy/order mismatch')
    # Common pipeline code hashes are evidence of matching processing, not model equality.
    operator_hashes = []
    for name in ('F', 'O'):
        operator_hashes.append({Path(r['path']).name: r['sha256'] for r in region[name]['receipt']['inputs']
                               if Path(r['path']).name in ('region_worker.py', 'region_adapter.py', 'recognition_worker.py')})
    if not operator_hashes[0] or operator_hashes[0] != operator_hashes[1]:
        raise ValueError('paired region operator policies differ')
    static_path = wave['scenes'][scene]['static_manifest']
    INDEX.identity(static_path['path'], static_path)
    static = read(static_path['path'])
    query = read(Path(config['attempt_root']) / 'query' / scene / 'Q_GAIN/receipt.json')
    INDEX.expected_output(query['decisions_path'], query)
    decisions = read(query['decisions_path'])
    reader = ObservationReader(data, model, decisions, static)
    with open(data['native_features'], 'rb') as handle:
        saved = pickle.load(handle)
    with np.load(model['text']['path'], allow_pickle=False) as arr:
        native_text = unit(arr['text_embeddings'])
    qpath = root / 'e01' / scene / 'features.npz'
    qfeatures = None
    qerror = None
    try:
        e01receipt = read(qpath.parent / 'receipt.json')
        INDEX.expected_output(qpath, e01receipt)
        qfeatures = np.load(qpath, allow_pickle=False)
    except (FileNotFoundError, PermissionError) as error:
        qerror = str(error)

    def region_view(name, rid):
        cache = region[name]
        if rid not in cache['requests']:
            path = cache['directory'] / 'requests' / (rid + '.json')
            INDEX.expected_output(path, cache['receipt'])
            record = read(path)
            if record['status'] != 'COMPLETE' or record['request_id'] != rid:
                raise ValueError('used region view is not genuinely complete')
            INDEX.expected_output(record['arrays_path'], record)
            with np.load(record['arrays_path'], allow_pickle=False) as arr:
                cache['features'][rid] = arr['feature']
            cache['requests'][rid] = record
        return cache['features'][rid], cache['requests'][rid]

    objects = {}
    operator_seconds = 0.
    for owner, fallback in sorted(labels.items()):
        key = str(owner)
        rows = {n: s['objects'][key] for n, s in sources.items()}
        scores = {n: r['scores'] if r['available'] else None for n, r in rows.items()}
        for n, value in scores.items():
            if value is not None and (len(value) != len(ids) or ids[int(np.argmax(value))] != rows[n]['label']):
                raise ValueError('source label/full-vector parity differs')
        base = pool(scores, ts, ('N', 'Q', 'F'))
        ovr_base = pool(scores, ts, ('N', 'Q', 'O'))
        paired_f = paired_o = None
        used_f, used_o = rows['F']['used_request_ids'], rows['O']['used_request_ids']
        common = [r for r in used_f if r in set(used_o)]
        pairing = 'NO_PAIRED_RESIDUAL'
        pair_error = None
        if common:
            if used_f == used_o:
                paired_f, paired_o = scores['F'], scores['O']
                pairing = 'IDENTICAL_SUCCESSFUL_REQUESTS_AND_WEIGHTS'
            else:
                try:
                    arrays = {n: np.stack([region_view(n, rid)[0] for rid in common]) for n in ('F', 'O')}
                    weights = [static['requests'][r].get('request', static['requests'][r])['visible_target_pixels'] for r in common]
                    paired_f, paired_o = [text[n] @ unit(np.average(arrays[n], axis=0, weights=weights)) for n in ('F', 'O')]
                    pairing = 'COMMON_SUBSET_REAGGREGATED'
                except (FileNotFoundError, PermissionError) as error:
                    pair_error = str(error)
        obs, dependence_error = [], None
        try:
            for n in ('N', 'Q', 'F'):
                if scores[n] is None:
                    continue
                if n == 'N':
                    retained = native_request_ids(saved[owner], reader.frames)
                    used = retained[-8:]
                    if len(retained) < 2 or used != rows[n]['used_request_ids']:
                        raise ValueError('native used last-eight order differs')
                    features = np.asarray(saved[owner]['feat'], np.float64)[-8:]
                    areas = np.asarray(saved[owner]['vis_area'], np.float64)[-8:]
                    atoms = [reader.atom(r) for r in used]
                    prototypes, space = native_text, model['identity']
                elif n == 'Q':
                    used = rows[n]['retained_request_ids']
                    if used != decisions['retained_features'].get(key, []):
                        raise ValueError('query final retained membership changed')
                    if qfeatures is None:
                        raise FileNotFoundError(qerror)
                    features, areas = qfeatures['retained_' + key], qfeatures['areas_' + key]
                    atoms = [reader.atom(r) for r in used]
                    prototypes, space = native_text, model['identity']
                else:
                    views = [region_view(n, r) for r in used_f]
                    features = np.stack([v[0] for v in views])
                    atoms = [reader.atom(r, region=True, region_arrays=v[1]['arrays_path']) for r, v in zip(used_f, views)]
                    areas = np.asarray([a['area'] for a in atoms])
                    prototypes, space = text[n], sources[n]['model_identity']
                    for atom in atoms:
                        atom['computation']['model'] = space
                if not np.array_equal(areas, [a['area'] for a in atoms]):
                    raise ValueError('actual observation area weights differ')
                obs.append(_source(n, scores[n], features, areas, prototypes, ts[n], atoms, space))
        except (FileNotFoundError, PermissionError, KeyError) as error:
            dependence_error = f'{type(error).__name__}: {error}'
        d_results, d_diagnostics = {}, {'status': 'NO_BASE_DISTRIBUTION'}
        if base is not None and dependence_error is None:
            operator_start = time.monotonic()
            d_results, d_diagnostics = solve_variants(obs, base, scene, owner)
            operator_seconds += time.monotonic() - operator_start
            d_diagnostics['status'] = 'COMPUTED'
        elif dependence_error:
            d_diagnostics = {'status': 'MISSING_DEPENDENCE_INPUT', 'error': dependence_error}
        objects[key] = {'scores': scores, 'available': {n: r['available'] for n, r in rows.items()},
            'base': base, 'ovr_base': ovr_base, 'fallback_label': fallback,
            'base_label': fallback if base is None else ids[int(base.argmax())],
            'source_labels': {n: r['label'] if r['available'] else None for n, r in rows.items()},
            'used_requests': {n: r['used_request_ids'] for n, r in rows.items()},
            'pairing': pairing, 'pair_error': pair_error, 'common_requests': common,
            'paired_F': paired_f, 'paired_O': paired_o,
            'candidate': None if base is None else candidate_set(base, paired_o),
            'residual_shared': paired_residual(paired_f, paired_o, ts['F']),
            'residual_caldelta': paired_residual(paired_f, paired_o, ts['F'], ts['O']),
            'dependence_probabilities': d_results, 'dependence': d_diagnostics,
            'observations': {s['name']: {'surrogate_max_error': s['surrogate_max_error'],
                'computation_identity': s['computation_identity'], 'space': s['space'], 'areas': s['areas'],
                'atoms': [{k: v for k, v in atom.items() if k != 'mask'} for atom in s['atoms']]} for s in obs}}
    if qfeatures is not None:
        qfeatures.close()
    for method, distribution in [(B, 'base'), ('AW_E03_OVR_A7', 'ovr_base')]:
        expected_path = root / 'locked' / scene / (method + '.json')
        INDEX.identity(expected_path)
        expected = read(expected_path)['labels']
        actual = {o: r['fallback_label'] if r[distribution] is None else ids[int(np.argmax(r[distribution]))] for o, r in objects.items()}
        if actual != expected:
            raise ValueError(f'original {method} whole-map labels differ')
    bundle = plain({'status': 'FROZEN_EVIDENCE', 'scene': scene, 'binding': binding['identity'],
        'valid_ids': ids, 'temperatures': ts, 'objects': objects, 'native_record_key': ev.native.record_key,
        'source_identities': {n: s.get('identity') for n, s in sources.items()},
        'text_comparison': {'exact_equal': bool(np.array_equal(text['F'], text['O'])),
                            'max_absolute_difference': float(np.abs(text['F'] - text['O']).max())},
        'operator_policy_hashes': operator_hashes[0], 'annotations_opened': False})
    bundle['identity'] = canonical_digest(bundle)
    path = output / 'frozen.json.gz'
    write_gzip(path, bundle)
    receipt = {'status': 'RECOVERED', 'scene': scene, 'binding': binding['identity'], 'bundle': INDEX.identity(path),
               'inputs': [r for r in INDEX.entries() if r['path'] != str(path)],
               'elapsed_seconds': time.monotonic()-start, 'operator_seconds': operator_seconds,
               'owners': len(objects), 'paired_owners': sum(r['pairing'] != 'NO_PAIRED_RESIDUAL' for r in objects.values()),
               'dependence_owners': sum(r['dependence']['status'] == 'COMPUTED' for r in objects.values()),
               'physical_image_forwards': 0, 'physical_text_forwards': 0, 'physical_model_loads': 0}
    write_once(receipt_path, receipt)
    return receipt
