"""Explicit, evidence-bound role extensions around unchanged numerical producers."""
from copy import deepcopy
import csv
from pathlib import Path
import shutil

from .common import (ConsumedInputs, Resolver, StageFailure, canonical_digest, digest_file,
                     read, source_identity, verified, write)


def alias_path(binding, lock=None):
    path = binding['asset_config'].get('datasets', {}).get('scannet200_val', {}).get('label_mapping_path')
    if not path and binding.get('structure_root'):
        path = Path(binding['structure_root']) / 'confirmation/raw/scannetv2-labels.combined.tsv'
    if not path and lock and lock.get('semantic_alias_source'):
        path = lock['semantic_alias_source']['path']
    return Path(Resolver(binding.get('path_map')).resolve(path)) if path else None


def check_alias(binding, lock):
    """An ordinary resume cannot silently accept an unrefreshed alias source."""
    path = alias_path(binding, lock)
    old = lock.get('semantic_alias_source')
    if old is None and (path is None or not path.is_file()):
        return
    if (old is None or path is None or not path.is_file() or str(path) != old['path']
            or digest_file(path) != old['sha256']):
        raise ValueError('Semantic alias input changed; use explicit --refresh-assets with preserved old source bytes')


def archive(root, paths, history):
    """Copy a finite set of task products and seal their original-path crosswalk."""
    root, history = Path(root).resolve(), Path(history)
    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(p for p in path.rglob('*') if p.is_file())
        elif path.is_file():
            files.append(path)
    records = {}
    for path in sorted(set(files)):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Refresh history may copy only task-owned regular products')
        target = history / path.relative_to(root)
        sha = digest_file(path)
        if target.exists():
            if digest_file(target) != sha:
                raise ValueError('Immutable refresh history differs: ' + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        records[str(path)] = dict(path=str(target), sha256=sha, bytes=path.stat().st_size)
    return write(history / 'crosswalk.json', dict(status='DONE', files=records))


def _gt_mapping(path):
    # The pinned author GT converter uses the first id for exact raw_category.
    # Category/WordNet evidence may be extended, but this conversion cannot change.
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            result.setdefault(row['raw_category'], int(row['id']))
    return result


def _monotonic(old, new):
    if [r['target_name'] for r in old['rows']] != [r['target_name'] for r in new['rows']]:
        raise ValueError('Refresh changed the frozen target name axis')
    for before, after in zip(old['rows'], new['rows'], strict=True):
        if before['status'] == 'DONE' and (after['status'] != 'DONE'
                or before['candidate_training_ids'] != after['candidate_training_ids']
                or before['base'] != after['base']):
            raise ValueError('Asset refresh cannot reverse a resolved training semantic role')


def prepare_protocol(binding, lock, history):
    """Return a validated candidate; never replace the live protocol here."""
    from .protocol import _aliases, map_training_roles
    root, history = Path(binding['output_root']), Path(history)
    index = ConsumedInputs(root)
    if lock['producer'] != source_identity(('protocol',)):
        raise ValueError('Frozen protocol producer changed')
    old_source = lock.get('semantic_alias_source')
    resolver = Resolver(binding.get('path_map'))
    for descriptor in lock['official_sources'].values():
        index.identity(resolver.resolve(descriptor['path']), descriptor)
    path = alias_path(binding, lock)
    if path is None or not path.is_file():
        raise StageFailure('Declared official semantic-alias TSV is missing', missing=[str(path)])
    if old_source:
        old_path = resolver.resolve(old_source['path'])
        index.identity(old_path, old_source)
        if _gt_mapping(old_path) != _gt_mapping(path):
            raise ValueError('Asset refresh cannot replace the frozen raw-category-to-ID GT conversion')
    elif any((root / 'benchmark/GT/scannet200_val' / scene / 'receipt.json').is_file()
             for scene in lock['scopes']['scannet200_val'].get('scenes', [])):
        raise ValueError('The original alias source is required to preserve existing ScanNet GT')
    class_path = binding['structure_documents'].get('class_split.json')
    original_class_path = class_path or str(Path(binding['semantic_root']) / 'A/import.json')
    class_descriptor = index.rows.get(str(original_class_path))
    class_path = resolver.resolve(class_path) if class_path else None
    classes = verified(class_path) if class_path else verified(Path(binding['semantic_root']) / 'A/import.json')['class_split']
    if class_path:
        index.identity(class_path, class_descriptor)
    ids, names, base_ids = classes['official_ids'], classes['official_names'], classes['base_ids']
    scan = lock['scopes']['scannet200_val']
    if scan['status'] == 'DONE' and (ids != scan['text_ids'] or names != scan['text_names']
            or [label in base_ids for label in ids] != scan['mapping']['base_mask']):
        raise ValueError('Asset refresh cannot replace the frozen training vocabulary or base roles')
    if scan['status'] != 'DONE':
        if class_descriptor is None:
            raise ValueError('A frozen training-class source identity is required independently of the ScanNet evaluator')
        index.identity(resolver.resolve(original_class_path), class_descriptor)
    aliases, descriptor = _aliases(path, ids, names, index)
    result = deepcopy(lock)
    changed = []
    for benchmark, scope in result['scopes'].items():
        if scope['status'] != 'DONE':
            continue
        before = scope['mapping']
        scope['mapping'] = map_training_roles(scope['text_names'], ids, names, base_ids, aliases)
        _monotonic(before, scope['mapping'])
        scope['full_AC_available'] = scope['mapping']['base_mask'] is not None
        if before['base_mask'] != scope['mapping']['base_mask']:
            changed.append(benchmark)
    legacy_changed = False
    if result.get('legacy', {}).get('replica51'):
        before = result['legacy']['replica51']
        after = map_training_roles(result['legacy']['old51_text_names'], ids, names, base_ids, aliases)
        _monotonic(before, after)
        result['legacy']['replica51'] = after
        legacy_changed = before['base_mask'] != after['base_mask']
    previous = history / 'protocol_lock.json'
    if previous.exists() and verified(previous)['identity'] != lock['identity']:
        raise ValueError('Refresh history already binds another protocol')
    write(previous, lock)
    result.update(semantic_alias_source=descriptor, GT_semantic_labels_read=False,
        input_identity=canonical_digest(dict(previous=lock['identity'], alias=descriptor)),
        semantic_refresh=dict(producer=source_identity(('semantic_refresh',)),
            previous_protocol=dict(path=str(previous), identity=lock['identity'], sha256=digest_file(previous)),
            changed_scopes=sorted(changed), legacy_roles_changed=legacy_changed,
            unchanged_raw_category_to_ID_GT_conversion=True if old_source else None,
            no_prior_ScanNet_GT_receipts=True if old_source is None else None, new_training_updates=0))
    result.pop('identity', None)
    result['identity'] = canonical_digest(result)
    index.save()
    return result


def _ancestor(protocol, identity):
    seen = set()
    while protocol['identity'] != identity:
        if protocol['identity'] in seen:
            raise ValueError('Cyclic semantic protocol history')
        seen.add(protocol['identity'])
        revision = protocol.get('semantic_refresh')
        if not revision or revision['producer'] != source_identity(('semantic_refresh',)):
            raise ValueError('Frozen prediction has no validated semantic protocol ancestry')
        descriptor = revision['previous_protocol']
        if digest_file(descriptor['path']) != descriptor['sha256']:
            raise ValueError('Archived semantic protocol bytes changed')
        protocol = verified(descriptor['path'])
        if protocol['identity'] != descriptor['identity']:
            raise ValueError('Archived semantic protocol identity changed')
    return protocol


def _same_scope(old, new, *, calibrated=True):
    def numerical(scope):
        return {key: value for key, value in scope.items() if key not in ('mapping', 'full_AC_available')}
    return (numerical(old) == numerical(new)
            and (not calibrated or old['mapping']['base_mask'] == new['mapping']['base_mask']))


def _validated_methods(lock):
    methods = {}
    for name, descriptor in lock['methods'].items():
        row = verified(descriptor['path'])
        if (row['identity'] != descriptor['identity'] or name != row['method'] + '_seed' + str(row['seed'])
                or row['scene'] != lock['scene'] or row['benchmark'] != lock['benchmark']):
            raise ValueError('Frozen standard method identity changed')
        expected = canonical_digest(dict(prediction=row['prediction_identity'], provenance=row['provenance'],
            method=row['method'], seed=row['seed'], producer=source_identity(('standard_export',))))
        if expected != row['input_identity']:
            raise ValueError('Frozen standard export producer changed')
        if digest_file(row['arrays']['path']) != row['arrays']['sha256']:
            raise ValueError('Frozen standard mask array changed')
        methods[name] = row
    return methods


def resume_standard_scene(binding, benchmark, scene):
    """Return a verified old freeze, or complete only its missing calibrated cells."""
    root = Path(binding['output_root'])
    path = root / 'benchmark/predictions' / benchmark / scene / 'receipt.json'
    protocol = verified(root / 'protocol_lock.json')
    if not protocol.get('semantic_refresh') or not path.is_file():
        return None
    lock = verified(path)
    manifests = _validated_methods(lock)
    scope = protocol['scopes'][benchmark]
    for row in manifests.values():
        ancestor = _ancestor(protocol, row['provenance']['protocol'])
        if not _same_scope(ancestor['scopes'][benchmark], scope, calibrated=row['method'] in ('U01', 'U11')):
            raise ValueError('Existing standard prediction has a different numerical scope')
    expected = {v + '_seed' + str(seed) for v in ('U00', 'U10', 'U01', 'U11') for seed in (17, 29)}
    if set(manifests) == expected or scope['mapping']['base_mask'] is None:
        return lock
    raw = {v + '_seed' + str(seed) for v in ('U00', 'U10') for seed in (17, 29)}
    if set(manifests) != raw:
        raise ValueError('Semantic extension requires the exact frozen raw four-cell grid')
    return _extend_standard(binding, protocol, lock, path)


def _extend_standard(binding, protocol, lock, path):
    import numpy as np
    from .benchmark import template_context
    from .standard_export import classify_scores, save_manifest, score_F
    root, benchmark, scene = Path(binding['output_root']), lock['benchmark'], lock['scene']
    scope = protocol['scopes'][benchmark]
    bundle = verified(root / 'benchmark/bundles' / benchmark / scene / 'bundle.json')
    if bundle['identity'] != lock['bundle']:
        raise ValueError('Semantic extension changed the frozen geometry/source bundle')
    f = verified(root / 'benchmark/F' / benchmark / scene / 'receipt.json')
    nq = verified(root / 'benchmark/NQ' / benchmark / scene / 'sources.json')
    if nq['bundle'] != bundle['identity']:
        raise ValueError('Cached N/Q belongs to another bundle')
    for method, descriptor in lock['methods'].items():
        provenance = verified(descriptor['path'])['provenance']
        if (provenance['bundle'] != bundle['identity'] or provenance['model_lock'] != bundle['model_lock']
                or provenance['F'] != f['identity']
                or method.startswith('U10_') and provenance['NQ'] != nq['identity']):
            raise ValueError('Semantic extension changed original source identities')
    index = ConsumedInputs(root)
    index.identity(f['arrays']['path'], f['arrays'])
    geometry_descriptor = bundle['arrays']['geometry']
    index.identity(geometry_descriptor['path'], geometry_descriptor)
    with np.load(f['arrays']['path'], allow_pickle=False) as arrays:
        fa = {name: arrays[name].copy() for name in arrays.files}
    with np.load(geometry_descriptor['path'], allow_pickle=False) as arrays:
        geometry = {name: arrays[name].copy() for name in ('owners', 'indptr', 'indices', 'target_xyz')}
    if not np.array_equal(geometry['owners'], fa['owners']) or not np.array_equal(fa['valid_ids'], scope['text_ids']):
        raise ValueError('Semantic extension changed the original owner/text axis')
    methods, traces = dict(lock['methods']), dict(lock['traces'])
    temperatures = template_context(binding, benchmark)['temperatures']
    for si, seed in enumerate((17, 29)):
        scores = [(None, None) if not fa['available'][i] else score_F(fa['b'][i], fa['z'][si, i],
            fa['text'], scope['mapping']['base_mask'], calibrated=True) for i in range(len(fa['owners']))]
        for variant in ('U01', 'U11'):
            accesses, labels, audit = [], [], []
            for i, owner in enumerate(geometry['owners']):
                def source_loader(name):
                    accesses.append((int(owner), name))
                    row = nq['sources'][name][str(owner)]
                    return row['scores'] if row['available'] else None
                result = classify_scores(variant, fa['valid_ids'], *scores[i], source_loader, temperatures)
                labels.append(result['label'])
                audit.append(dict(owner=int(owner), **result))
            method = variant + '_seed' + str(seed)
            row = save_manifest(path.parent / method, scene=scene, benchmark=benchmark, method=variant, seed=seed,
                owners=geometry['owners'], indptr=geometry['indptr'], indices=geometry['indices'], labels=labels,
                target_points=len(geometry['target_xyz']), valid_instance_ids=scope['instance_ids'],
                provenance=dict(bundle=bundle['identity'], model_lock=bundle['model_lock'], protocol=protocol['identity'],
                    F=f['identity'], NQ=nq['identity'] if variant == 'U11' else None,
                    MFoff_NQ_accesses=0 if variant == 'U01' else None, calibrator='CAP10_gamma.025',
                    source_availability=audit, exposure_role=scope.get('exposure_role', 'SEE_PROTOCOL_LOCK'),
                    semantic_refresh_producer=source_identity(('semantic_refresh',))))
            methods[method] = dict(path=str(path.parent / method / 'manifest.json'), identity=row['identity'])
            traces[method] = dict(NQ_access_count=len(accesses), availability=audit)
    history = root / 'history/semantic_resume' / protocol['identity'] / benchmark / scene / 'prediction'
    saved = archive(root, [path], history)['files'][str(path)]
    result = write(path, dict(lock, status='DONE', methods=methods, traces=traces, completed_rows=8,
        missing_variants=[], missing_reason=None, semantic_refresh=dict(previous_prediction=saved,
            producer=source_identity(('semantic_refresh',)), new_image_encodings=0, new_head_calls=0,
            preserved_raw_methods=sorted(lock['methods']))))
    index.save()
    return result


def prepare_standard_score(binding, benchmark, scene):
    """Rebind independent full GT only after an exact prediction-superset proof."""
    root = Path(binding['output_root'])
    freeze = verified(root / 'benchmark/predictions' / benchmark / scene / 'receipt.json')
    revision = freeze.get('semantic_refresh')
    if not revision:
        return
    descriptor = revision['previous_prediction']
    if digest_file(descriptor['path']) != descriptor['sha256']:
        raise ValueError('Archived prediction freeze changed')
    old = verified(descriptor['path'])
    if not all(freeze['methods'].get(name) == item for name, item in old['methods'].items()):
        raise ValueError('Semantic score extension changed existing predictions')
    _validated_methods(freeze)
    gt_path = root / 'benchmark/GT' / benchmark / scene / 'receipt.json'
    score_path = root / 'benchmark/evaluation' / benchmark / scene / 'receipt.json'
    history = root / 'history/semantic_resume' / freeze['identity'] / 'score'
    if gt_path.is_file():
        gt = verified(gt_path)
        if gt['prediction_freeze'] != freeze['identity']:
            if gt['prediction_freeze'] != old['identity']:
                raise ValueError('GT freeze has no validated semantic-extension ancestry')
            bundle = verified(root / 'benchmark/bundles' / benchmark / scene / 'bundle.json')
            if gt['geometry'] != bundle['evaluation_geometry'] or digest_file(gt['GT']['path']) != gt['GT']['sha256']:
                raise ValueError('Independent full GT geometry/bytes changed')
            saved = archive(root, [gt_path], history / 'GT')['files'][str(gt_path)]
            write(gt_path, dict(gt, prediction_freeze=freeze['identity'],
                semantic_refresh=dict(previous_GT=saved, unchanged_full_GT=True)))
    if score_path.is_file() and verified(score_path)['prediction_lock'] != freeze['identity']:
        if verified(score_path)['prediction_lock'] != old['identity']:
            raise ValueError('Standard score has no validated semantic-extension ancestry')
        archive(root, [score_path.parent], history / 'evaluation')
        score_path.unlink()
    # OVI uses this independent GT byte stream as well. A larger Ours freeze
    # changes only the GT receipt identity, not OVI's prediction or its matches.
    ovi_path = root / 'external/OVI-MAP/evaluation' / benchmark / scene / 'receipt.json'
    if gt_path.is_file() and ovi_path.is_file():
        gt, ovi = verified(gt_path), verified(ovi_path)
        if ovi['GT'] != gt['identity']:
            previous_gt = gt.get('semantic_refresh', {}).get('previous_GT')
            if not previous_gt or digest_file(previous_gt['path']) != previous_gt['sha256']:
                raise ValueError('OVI score GT has no unchanged-byte refresh proof')
            original_gt = verified(previous_gt['path'])
            producer = source_identity(('ovi_adapter', 'standard_score'))
            expected = canonical_digest(dict(prediction=ovi['prediction'], GT=original_gt['identity'], producer=producer))
            if ovi['GT'] != original_gt['identity'] or ovi['input_identity'] != expected:
                raise ValueError('OVI score producer or original GT changed')
            manifest = verified(root / 'external/OVI-MAP/predictions' / benchmark / scene / 'manifest.json')
            if (manifest['identity'] != ovi['prediction']
                    or digest_file(manifest['arrays']['path']) != manifest['arrays']['sha256']
                    or verified(ovi['matches']['path'])['identity'] != ovi['matches']['identity']):
                raise ValueError('OVI score prediction or matches changed')
            saved = archive(root, [ovi_path], history / 'OVI')['files'][str(ovi_path)]
            write(ovi_path, dict(ovi, GT=gt['identity'],
                input_identity=canonical_digest(dict(prediction=ovi['prediction'], GT=gt['identity'], producer=producer)),
                protocol=verified(root / 'protocol_lock.json')['identity'],
                semantic_refresh=dict(previous_score=saved, unchanged_GT_bytes=True, new_scorer_calls=0)))


def prepare_legacy_prediction(binding):
    """Open only stale Replica receipt gates; preserve all numerical caches/exports."""
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    if not protocol.get('semantic_refresh'):
        return
    from .legacy import numerical_binding
    old = numerical_binding(binding)
    mask = protocol['legacy']['replica51']['base_mask']
    model = verified(root / 'model_lock.json')
    producer = source_identity(('legacy', 'region_inputs'))
    history = root / 'history/semantic_resume' / protocol['identity'] / 'legacy_prediction'
    changed = []
    for scene in old['cohorts']['replica8']:
        path = root / 'legacy/predictions' / scene / 'receipt.json'
        if not path.is_file():
            # Recovery after interruption between opening a scene gate and
            # retiring its old global lock must not return that stale lock.
            if (history / scene / 'crosswalk.json').is_file():
                changed.append(scene)
            continue
        previous = verified(path)
        source = verified(Path(old['lr_parent_root']) / 'regression' / scene / 'manifest.json')
        resolver = Resolver(binding['path_map'])
        context = resolver.rewrite(verified(resolver.resolve(old['scenes'][scene]['context']['path'])))
        variants = ('L00', 'L10', 'L01', 'L11') if mask is not None else ('L00', 'L10')
        identity = canonical_digest(dict(producer=producer, models=model['weights'], scene=source['identity'],
            text=context['FC_text']['sha256'], variants=variants, base_mask=mask))
        if identity == previous['input_identity']:
            continue
        expected = {v + '_seed' + str(seed) for v in ('L00', 'L10') for seed in (17, 29)}
        original_identity = canonical_digest(dict(producer=producer, models=model['weights'],
            scene=source['identity'], text=context['FC_text']['sha256'],
            variants=('L00', 'L10'), base_mask=None))
        if mask is None or set(previous['methods']) != expected or previous['input_identity'] != original_identity:
            raise ValueError('Legacy refresh supports only a proven raw-to-calibrated extension')
        # Validate actual original prediction records before reopening the gate.
        from static_ovmap.module_validation.scannet_study import load_prediction
        for row in previous['methods'].values():
            payload = load_prediction(row['manifest'])
            if payload.prediction_key != row['prediction_key'] or payload.record_key != row['record_key']:
                raise ValueError('Existing Legacy prediction bytes changed')
        archive(root, [path], history / scene)
        path.unlink()
        changed.append(scene)
    global_path = root / 'legacy/prediction_lock.json'
    if changed and global_path.is_file():
        archive(root, [global_path], history / 'global')
        global_path.unlink()


def prepare_legacy_score(binding):
    """Preserve old scores; rebind unaffected scenes and reopen extended scenes."""
    from .scoring import frozen_legacy_inputs
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    if not protocol.get('semantic_refresh'):
        return
    lock, scenes = frozen_legacy_inputs(root)
    path = root / 'legacy/score.json'
    producer = source_identity(('scoring',))
    identity = canonical_digest(dict(predictions=lock['identity'], producer=producer))
    if path.is_file() and verified(path)['input_identity'] == identity:
        return
    history = root / 'history/semantic_resume' / lock['identity'] / 'legacy_score'
    if path.is_file():
        archive(root, [path, root / 'legacy/pools'], history / 'global')
    for scene, freeze in scenes.items():
        receipt = root / 'legacy/evaluation' / scene / 'receipt.json'
        if not receipt.is_file():
            continue
        old = verified(receipt)
        wanted = canonical_digest(dict(global_predictions=lock['identity'], scene=freeze['identity'], producer=producer))
        if old['input_identity'] == wanted:
            continue
        if not old['rows']:
            raise ValueError('Cannot rebind an empty Legacy score')
        for row in old['rows']:
            item = freeze['methods'].get(row['method'])
            if (item is None or row['prediction_identity'] != item['prediction_key']
                    or row['record_identity'] != item['record_key']):
                raise ValueError('Semantic Legacy extension changed an existing scored prediction')
        previous_global = old.get('semantic_refresh', {}).get('new_global_freeze',
            old['rows'][0]['global_prediction_lock'])
        previous_scene = old['rows'][0]['prediction_lock']
        expected_old = canonical_digest(dict(global_predictions=previous_global,
            scene=previous_scene, producer=producer))
        if old['input_identity'] != expected_old or any(row['prediction_lock'] != previous_scene for row in old['rows']):
            raise ValueError('Legacy score producer or original scene freeze changed')
        saved = archive(root, [receipt.parent], history / scene)['files'][str(receipt)]
        if all(row['prediction_lock'] == freeze['identity'] for row in old['rows']):
            if {row['method'] for row in old['rows']} != set(freeze['methods']):
                raise ValueError('Unchanged Legacy score producer or coverage differs')
            write(receipt, dict(old, input_identity=wanted, semantic_refresh=dict(previous_score=saved,
                unchanged_scene_predictions=freeze['identity'], new_global_freeze=lock['identity'],
                scorer_inputs_unchanged=True, new_GT_reads=0, new_scorer_calls=0)))
        else:
            # The original scorer runs only for these extended Replica scenes.
            receipt.unlink()
    if path.is_file():
        path.unlink()
