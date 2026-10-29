"""Explicit scores for frozen Legacy fixed-eligible readout interventions."""
from pathlib import Path

import numpy as np

from .common import SEEDS, canonical_digest, digest_file, read, save_arrays, source_identity, verified, write
from .score_artifacts import _arrays, classification_scores


def partition_owner_labels(owner_ids, semantic_labels, eligible_owners):
    owners, labels, eligible = (np.asarray(value) for value in (owner_ids, semantic_labels, eligible_owners))
    if (owners.ndim != 1 or labels.shape != owners.shape or eligible.ndim != 1
            or len(np.unique(eligible)) != len(eligible) or (eligible <= 0).any()):
        raise ValueError('Legacy eligible owner mapping is invalid')
    positive = owners > 0
    ids, first, inverse = np.unique(owners[positive], return_index=True, return_inverse=True)
    values = labels[positive]
    per_owner = values[first]
    if not np.array_equal(values, per_owner[inverse]) or not np.isin(eligible, ids).all():
        raise ValueError('Legacy positive owner labels are nonuniform or eligible owners are missing')
    preserved = ~np.isin(ids, eligible)
    return dict(eligible_labels=per_owner[np.searchsorted(ids, eligible)],
        preserved_owners=ids[preserved], preserved_labels=per_owner[preserved], positive_owner_count=len(ids))


def prepare(binding, replay):
    """Read only frozen readouts/predictions; never extend the intervention scope."""
    from .standard_export import score_F
    from static_ovmap.module_validation import scannet_study
    from static_ovmap.paired_evidence_study import residuals
    root = Path(binding['output_root'])
    lock_path = root/'legacy/prediction_lock.json'
    if not lock_path.is_file():
        return None
    lock = verified(lock_path)
    scenes = {}
    for scene, identity in lock['scenes'].items():
        row = verified(root/'legacy/predictions'/scene/'receipt.json')
        if row['identity'] != identity:
            raise ValueError('Legacy score scene differs from its prediction lock')
        if row['methods']:
            scenes[scene] = row
    if not scenes:
        return None
    protocol, model = verified(root/'protocol_lock.json'), verified(root/'model_lock.json')
    calibration = verified(root/'A/import.json')
    producer = dict(local=source_identity(('legacy_score_artifacts', 'score_artifacts', 'standard_export')),
        original_prediction_decoder=digest_file(scannet_study.__file__), original_pool_and_D2=digest_file(residuals.__file__))
    rows, checked_arrays = [], set()
    eligible_total = preserved_total = alias_total = 0
    for scene, scene_row in sorted(scenes.items()):
        context = replay['legacy'][scene]
        if context['prediction_receipt'] != scene_row['identity']:
            raise ValueError('Legacy replay context differs from the frozen scene')
        text_source = context['text']
        text_arrays = _arrays(dict(path=str(root/text_source['source']), sha256=text_source['original']['sha256']))
        text, ids = text_arrays['text_embeddings'], text_arrays['valid_ids']
        if scene_row['cohort'] == 'scannet_cf18':
            base_mask = np.isin(ids, calibration['class_split']['base_ids'])
        elif scene_row['cohort'] == 'replica8':
            if not np.array_equal(ids, protocol['legacy']['old51_text_ids']):
                raise ValueError('Legacy Replica score class axis differs')
            base_mask = protocol['legacy']['replica51']['base_mask']
        else:
            raise ValueError('Unexpected Legacy scope')
        eligible = np.asarray(scene_row['original_eligible_owners'], np.int64)
        objects = {obj['owner']: (key, obj) for key, obj in scene_row['objects'].items()}
        if len(objects) != len(scene_row['objects']) or set(objects) != set(eligible.tolist()):
            raise ValueError('Legacy readout objects differ from the fixed eligible domain')
        for owner, (key, obj) in objects.items():
            for seed, desc in obj['readouts'].items():
                path = Path(desc['arrays']['path'])
                receipt = verified(path.with_suffix('.json'))
                if (receipt['identity'] != desc['readout'] or receipt['object_key'] != key
                        or receipt['seed'] != int(seed) or receipt['arrays'] != desc['arrays']
                        or receipt['weight_sha256'] != model['weights'][seed]['source_sha256']):
                    raise ValueError('Legacy actual readout identity/seed/weight differs')
                if digest_file(path) != desc['arrays']['sha256']:
                    raise ValueError('Legacy actual readout arrays changed')
        nq, scores_by_seed, preserved_reference, owner_partition = None, {}, None, None
        for method, desc in sorted(scene_row['methods'].items()):
            variant, seed_text = method.rsplit('_seed', 1)
            seed = int(seed_text)
            if variant not in ('L00', 'L10', 'L01', 'L11') or seed not in SEEDS or desc['method'] != method:
                raise ValueError('Legacy score method/seed differs')
            mf, ac = variant in ('L10', 'L11'), variant in ('L01', 'L11')
            manifest_path = Path(desc['manifest'])
            prediction = read(manifest_path)
            if (prediction['prediction_key'] != desc['prediction_key'] or prediction['record_key'] != desc['record_key']
                    or prediction['scene_id'] != scene or prediction['metadata']['model_lock'] != model['identity']):
                raise ValueError('Legacy prediction manifest differs from the frozen row')
            arrays_path = manifest_path.parent/prediction['arrays']['path']
            array_key = (str(arrays_path), prediction['arrays']['sha256'])
            if array_key not in checked_arrays:
                if digest_file(arrays_path) != prediction['arrays']['sha256']:
                    raise ValueError('Legacy frozen prediction arrays changed')
                checked_arrays.add(array_key)
            if owner_partition is None:
                owner_partition = prediction['owner_ids']
            elif owner_partition != prediction['owner_ids']:
                raise ValueError('Legacy fixed owner partition differs between methods')
            if mf and nq is None:
                nq = {}
                for name in ('N', 'Q'):
                    source = context['sources'][name]
                    path = root/source['source']
                    if digest_file(path) != source['original']['sha256']:
                        raise ValueError('Legacy source-score snapshot changed')
                    value = verified(path)
                    if value['identity'] != source['original_identity'] or not np.array_equal(value['valid_ids'], ids):
                        raise ValueError('Legacy N/Q source identity or class axis differs')
                    nq[name] = value
            inputs = dict(scene_prediction=scene_row['identity'], logical_method=method,
                original_manifest_sha256=digest_file(manifest_path), text=text_source,
                source_scores=context['sources'] if mf else None, temperatures=context['temperatures'] if mf else None,
                base_mask=base_mask if ac else None, calibration_classes=calibration['identity'] if ac else None,
                protocol=protocol['identity'], model=model['identity'], producer=producer)
            identity = canonical_digest(inputs)
            folder = root/'publication/legacy_class_scores'/identity
            receipt_path = folder/'manifest.json'
            if receipt_path.is_file():
                result = verified(receipt_path)
                if result['input_identity'] != identity:
                    raise ValueError('Legacy full-score artifact identity changed')
                output = _arrays(dict(result['arrays'], path=str(folder/'scores.npz')))
                partition = dict(preserved_owners=output['preserved_owners'], preserved_labels=output['preserved_labels'])
            else:
                payload = scannet_study.load_prediction(manifest_path)
                partition = partition_owner_labels(payload.owner_ids, payload.semantic_labels, eligible)
                if partition['positive_owner_count'] != scene_row['all_positive_G1_owners']:
                    raise ValueError('Legacy full positive owner count differs')
                key = (seed, ac)
                if key not in scores_by_seed:
                    values = []
                    for owner in eligible:
                        arrays = _arrays(objects[int(owner)][1]['readouts'][seed_text]['arrays'])
                        values.append(score_F(arrays['b'], arrays['z'], text, base_mask, calibrated=ac))
                    scores_by_seed[key] = values
                scores = np.empty((len(eligible), len(ids)), np.float64)
                labels = np.empty(len(eligible), np.int64)
                for i, owner in enumerate(eligible):
                    def source_loader(name):
                        source = nq[name]['objects'][str(int(owner))]
                        return source['scores'] if source['available'] else None
                    values = classification_scores('U'+variant[1:], ids, *scores_by_seed[key][i],
                                                   source_loader, context['temperatures'])
                    if values is None:
                        raise ValueError('Frozen Legacy eligible F cannot become undefined')
                    scores[i], labels[i] = values, ids[np.argmax(values)]
                if not np.array_equal(labels, partition['eligible_labels']):
                    raise ValueError('Legacy full-score argmax differs from the frozen labels: '+scene+'/'+method)
                ranks = dict(payload.instance_ranks)
                arrays_path = folder/'scores.npz'
                save_arrays(arrays_path, owners=eligible, valid_ids=ids, scores=scores,
                    defined=np.ones(len(eligible), bool), labels=labels,
                    confidence=np.asarray([ranks[int(owner)] for owner in eligible], np.float64),
                    preserved_owners=partition['preserved_owners'], preserved_labels=partition['preserved_labels'],
                    preserved_confidence=np.asarray([ranks[int(owner)] for owner in partition['preserved_owners']], np.float64))
                result = write(receipt_path, dict(status='DONE', input_identity=identity, inputs=inputs,
                    scene=scene, cohort=scene_row['cohort'], method=variant, seed=seed,
                    scope='FIXED_ELIGIBLE_READOUT_INTERVENTION', full_owner_global_ablation=False,
                    score_kind='D2_PROBABILITY' if mf else 'CAP10_GAMMA025_F_COSINE' if ac else 'RAW_F_COSINE',
                    owner_count=len(eligible), preserved_owner_count=len(partition['preserved_owners']),
                    class_count=len(ids), original_owner_partition=prediction['owner_ids'],
                    original_prediction=dict(path=str(manifest_path), prediction_key=desc['prediction_key'],
                        record_key=desc['record_key'], concrete_method=prediction['method_id'], alias_of=desc.get('alias_of')),
                    arrays=dict(path=str(arrays_path), bytes=arrays_path.stat().st_size, sha256=digest_file(arrays_path)),
                    MFoff_NQ_accesses=0 if not mf else None, exact_eligible_label_parity=True,
                    original_native_ranks_preserved=True, noneligible_score_vectors_not_fabricated=True,
                    GT_labels_read=False, new_model_forwards=0, new_training_updates=0))
            current = (partition['preserved_owners'], partition['preserved_labels'])
            if preserved_reference is None:
                preserved_reference = current
            elif any(not np.array_equal(a, b) for a, b in zip(preserved_reference, current, strict=True)):
                raise ValueError('Legacy noneligible labels differ between interventions')
            rows.append(dict(scene=scene, cohort=scene_row['cohort'], method=method,
                manifest=str(receipt_path.relative_to(root)), arrays=str((folder/'scores.npz').relative_to(root)),
                identity=result['identity']))
            eligible_total += result['owner_count']
            preserved_total += result['preserved_owner_count']
            alias_total += int('alias_of' in desc)
    return write(root/'publication/legacy_class_scores.json', dict(status='DONE',
        scope='FIXED_ELIGIBLE_LEGACY_SCORES_ONLY', rows=rows, completed_scene_cells=len(rows),
        expected_original_scene_cells=lock['expected_logical_scene_rows'], eligible_owner_rows=eligible_total,
        preserved_owner_rows=preserved_total, logical_prediction_aliases_retained=alias_total,
        missing_methods_not_generated=True, original_native_ranks_preserved=True,
        noneligible_score_vectors_not_fabricated=True, new_model_forwards=0, new_training_updates=0))
