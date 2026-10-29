"""Explicit classification scores replayed from immutable standard predictions."""
from pathlib import Path

import numpy as np

from .common import SEEDS, canonical_digest, digest_file, save_arrays, source_identity, verified, write


def classification_scores(variant, valid_ids, f_raw, f_calibrated, source_loader, temperatures):
    """Keep the existing cosine/probability semantics and source-access boundary."""
    from .standard_export import classify_scores
    from static_ovmap.paired_evidence_study.residuals import pool, simple_dependence
    captured = {}

    def capture(name):
        captured[name] = source_loader(name)
        return captured[name]

    result = classify_scores(variant, valid_ids, f_raw, f_calibrated, capture, temperatures)
    if result['unknown']:
        return None
    f = f_calibrated if result['AC'] else f_raw
    if not result['MF']:
        return np.asarray(f).copy()
    sources = dict(captured, F=f)
    values = simple_dependence(sources, temperatures, pool(sources, temperatures, ('N', 'Q', 'F')),
                               'PE_D2_GROUPED')
    if int(np.asarray(valid_ids)[np.argmax(values)]) != result['label']:
        raise ValueError('Full-score replay differs from the frozen classification operator')
    return values


def _arrays(descriptor):
    path = Path(descriptor['path'])
    if digest_file(path) != descriptor['sha256']:
        raise ValueError('Frozen score input arrays changed: ' + str(path))
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def prepare(binding, replay):
    """Materialize current frozen scene rows; no model, geometry or GT is loaded."""
    from .standard_export import score_F
    from static_ovmap.paired_evidence_study import residuals
    root = Path(binding['output_root'])
    locks = [verified(path) for path in sorted((root/'benchmark/predictions').glob('*/*/receipt.json'))]
    locks = [lock for lock in locks if lock.get('GLOBAL') and lock.get('methods')]
    if not locks:
        return None
    protocol = verified(root/'protocol_lock.json')
    producer = dict(local=source_identity(('score_artifacts', 'standard_export')),
                    original_pool_and_D2=digest_file(residuals.__file__))
    rows, total_owners, total_defined = [], 0, 0
    for lock in locks:
        benchmark, scene = lock['benchmark'], lock['scene']
        scope = protocol['scopes'][benchmark]
        temperatures = replay['standard'][benchmark]['temperatures']
        f = verified(root/'benchmark/F'/benchmark/scene/'receipt.json')
        if f['bundle'] != lock['bundle']:
            raise ValueError('Frozen score F and prediction bundle differ')
        data = _arrays(f['arrays'])
        ids, owners = data['valid_ids'], data['owners']
        if not np.array_equal(ids, scope['text_ids']):
            raise ValueError('Frozen score class axis differs from the locked vocabulary')
        nq, f_scores = None, {}
        for method, descriptor in sorted(lock['methods'].items()):
            prediction = verified(descriptor['path'])
            if prediction['identity'] != descriptor['identity']:
                raise ValueError('Frozen score prediction differs from its scene lock')
            variant, seed = prediction['method'], prediction['seed']
            if method != variant + '_seed' + str(seed) or seed not in SEEDS:
                raise ValueError('Frozen score method or seed differs from its scene key')
            mf, ac = variant in ('U10', 'U11'), variant in ('U01', 'U11')
            provenance = prediction['provenance']
            if (provenance['F'] != f['identity'] or provenance['protocol'] != protocol['identity']
                    or provenance['bundle'] != lock['bundle']):
                raise ValueError('Frozen score source provenance differs')
            if mf and nq is None:
                nq = verified(root/'benchmark/NQ'/benchmark/scene/'sources.json')
                if nq['bundle'] != lock['bundle'] or not np.array_equal(nq['valid_ids'], ids):
                    raise ValueError('Frozen score N/Q bundle or class axis differs')
            if provenance['NQ'] != (nq['identity'] if mf else None):
                raise ValueError('Frozen score N/Q provenance differs')
            frozen = _arrays(prediction['arrays'])
            if not np.array_equal(frozen['owners'], owners):
                raise ValueError('Frozen score owner domain differs from the prediction masks')
            inputs = dict(prediction=prediction['identity'], F=f['identity'], NQ=nq['identity'] if mf else None,
                protocol=protocol['identity'], temperatures=temperatures if mf else None, producer=producer)
            identity = canonical_digest(inputs)
            folder = root/'publication/class_scores'/identity
            receipt_path = folder/'manifest.json'
            if receipt_path.exists():
                row = verified(receipt_path)
                if row['input_identity'] != identity or digest_file(folder/'scores.npz') != row['arrays']['sha256']:
                    raise ValueError('Completed full-score artifact changed; preserve it before repair')
            else:
                key = (seed, ac)
                if key not in f_scores:
                    f_scores[key] = [(None, None) if not data['available'][i] else
                        score_F(data['b'][i], data['z'][SEEDS.index(seed), i], data['text'],
                                scope['mapping']['base_mask'], calibrated=ac)
                        for i in range(len(owners))]
                scores = np.full((len(owners), len(ids)), np.nan, np.float64)
                defined = np.zeros(len(owners), bool)
                labels = np.zeros(len(owners), np.int64)
                for i, owner in enumerate(owners):
                    def source_loader(name):
                        source = nq['sources'][name][str(int(owner))]
                        return source['scores'] if source['available'] else None
                    values = classification_scores(variant, ids, *f_scores[key][i], source_loader, temperatures)
                    if values is not None:
                        scores[i] = values
                        defined[i] = True
                        labels[i] = ids[np.argmax(values)]
                if not np.array_equal(labels, frozen['labels']):
                    raise ValueError('Full-score argmax differs from the actual frozen prediction: ' + method)
                arrays_path = folder/'scores.npz'
                save_arrays(arrays_path, owners=owners, valid_ids=ids, scores=scores, defined=defined,
                    labels=frozen['labels'], confidence=frozen['confidence'], exported=frozen['exported'])
                row = write(receipt_path, dict(status='DONE', input_identity=identity,
                    benchmark=benchmark, scene=scene, method=variant, seed=seed, inputs=inputs,
                    score_kind='D2_PROBABILITY' if mf else 'CAP10_GAMMA025_F_COSINE' if ac else 'RAW_F_COSINE',
                    score_dtype='float64', undefined_rows='ALL_NAN_WITH_DEFINED_FALSE_AND_LABEL_ZERO',
                    all_original_owners_retained=True, owner_count=len(owners), defined_count=int(defined.sum()),
                    class_count=len(ids), original_prediction=dict(path=descriptor['path'], identity=prediction['identity']),
                    original_mask_identity=prediction['proposal_identity'], exact_frozen_label_parity=True,
                    MFoff_NQ_accesses=0 if not mf else None, GT_labels_read=False,
                    arrays=dict(path=str(arrays_path), bytes=arrays_path.stat().st_size, sha256=digest_file(arrays_path)),
                    new_model_forwards=0, new_training_updates=0))
            rows.append(dict(benchmark=benchmark, scene=scene, method=method,
                manifest=str(receipt_path.relative_to(root)), identity=row['identity'],
                arrays=str((folder/'scores.npz').relative_to(root))))
            total_owners += row['owner_count']
            total_defined += row['defined_count']
    return write(root/'publication/class_scores.json', dict(status='DONE',
        scope='FROZEN_STANDARD_PREDICTION_SCORES_ONLY', rows=rows, completed_scene_cells=len(rows),
        expected_original_scene_cells=sum(len(scope['scenes'])*8 for scope in protocol['scopes'].values()),
        retained_owner_rows=total_owners, defined_owner_rows=total_defined,
        undefined_owner_rows=total_owners-total_defined, class_axis='ORIGINAL_TEXT_IDS_IN_ARRAY_ORDER',
        no_cross_kind_score_comparability_claim=True, raw_geometry_RGB_GT_excluded=True,
        new_model_forwards=0, new_training_updates=0))
