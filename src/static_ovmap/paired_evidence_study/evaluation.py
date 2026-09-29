"""Locked whole-map decisions and original released dual-rank evaluation."""

from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator
from src.static_ovmap.m2_reviewer_study.evaluation import pool as released_pool
from src.static_ovmap.m2_reviewer_study.scores import read_scene
from src.static_ovmap.module_validation.scannet_study import (
    relabel_prediction,
    save_prediction,
)
from src.static_ovmap.released_loader import load_released_module

from .binding import read_json_or_gzip as read
from .residuals import blend, pool, ratio_update, simple_dependence

RANKS = ('OFFICIAL_CURRENT_CLASS', 'FROZEN_N0')


def bound_context(actual, original):
    current, previous = actual['study_evaluator'], original['study_evaluator']
    if any(current[k] != previous[k] for k in ('bytes', 'sha256')):
        raise ValueError('evaluator bytes changed')
    normalized = {**actual, 'study_evaluator': previous}
    for key, old in original.items():
        new = normalized.get(key)
        if (isinstance(old, dict) and isinstance(new, dict) and {'path', 'sha256'} <= old.keys()
                and Path(old['path']).resolve() == Path(new['path']).resolve()):
            if {k: v for k, v in old.items() if k != 'path'} != {k: v for k, v in new.items() if k != 'path'}:
                raise ValueError('relocated evidence content differs')
            normalized[key] = old
    if normalized != original:
        raise ValueError('evaluation context changed beyond evaluator worktree path')
    return normalized, {'actual_path': current['path'], 'bound_original': previous,
                        'reason': 'EXACT_BYTES_WORKTREE_RELOCATION'}


def configurations(spec, choices=None):
    result = [{'id': m, 'method': m, 'parameter': None} for m in spec['controls']]
    for variant in spec['variants']:
        method = variant['id']
        if choices is not None:
            result.append({'id': method, 'method': method, 'parameter': choices[method]['parameter']})
        else:
            for value in variant.get('grid', [None]):
                identity = method if value is None else f'{method}__{value:g}'
                result.append({'id': identity, 'method': method, 'parameter': value})
    return result


def probabilities(obj, temperatures, method, value):
    base = obj['base']
    if method == 'PE_R1_FOURWAY':
        return pool(obj['scores'], temperatures, ('N', 'Q', 'F', 'O'))
    if base is None:
        return None
    if method == 'PE_R0_BLEND':
        return blend(base, obj['ovr_base'], value)
    if method in ('PE_R2_GLOBAL', 'PE_R3_LOCAL', 'PE_R3_LOCAL_CALDELTA', 'PE_COMBO_D4_R3'):
        residual = obj['residual_caldelta'] if method.endswith('CALDELTA') else obj['residual_shared']
        start = obj['dependence_probabilities'].get('PE_D4_LINEAGE', base) if method == 'PE_COMBO_D4_R3' else base
        return ratio_update(start, residual, value, None if method == 'PE_R2_GLOBAL' else obj['candidate'])
    if method in ('PE_D1_LOGPOOL', 'PE_D1_ANCHORED', 'PE_D2_GROUPED'):
        return simple_dependence(obj['scores'], temperatures, base, method)
    if method in ('PE_D3_DIAGONAL', 'PE_D4_LINEAGE', 'PE_D4_SHUFFLED'):
        return np.asarray(obj['dependence_probabilities'].get(method, base))
    raise ValueError(f'unknown method {method}')


def predict(binding, scene, configs):
    from src.static_ovmap.m2_reviewer_study.evaluation import write_gzip

    root = Path(binding['output_root'])
    frozen = read(root / 'evidence' / scene / 'frozen.json.gz')
    wave_root = Path(read(binding['wave1_binding'])['output_root'])
    spec = read(binding['spec'])
    ev = None
    for config in configs:
        method, parameter, identity = config['method'], config['parameter'], config['id']
        path = root / 'locked' / scene / (identity + '.json')
        if path.exists():
            previous = read(path)
            if previous['configuration'] != config or previous['evidence_identity'] != frozen['identity']:
                raise ValueError('locked prediction definition changed')
            continue
        if method in spec['controls']:
            value = read(wave_root / 'locked' / scene / (method + '.json'))
        else:
            all_probs, labels = {}, {}
            for owner, obj in frozen['objects'].items():
                p = probabilities(obj, frozen['temperatures'], method, parameter)
                labels[int(owner)] = obj['fallback_label'] if p is None else frozen['valid_ids'][int(np.argmax(p))]
                all_probs[owner] = None if p is None else np.asarray(p).tolist()
            if ev is None:
                ev = read_scene(read(binding['reviewer_binding']), scene)
            payload = relabel_prediction(ev.native, identity, 'S', labels, {},
                {'study': binding['identity'], 'evidence': frozen['identity'], 'configuration': config,
                 'geometry': 'UNCHANGED', 'cost_ledger': 'separate logical-operation unions'})
            manifest = save_prediction(payload, root / 'predictions' / scene / identity)
            value = {'labels': labels, 'prediction_identity': payload.prediction_key, 'prediction_manifest': str(manifest)}
            write_gzip(root / 'probabilities' / scene / (identity + '.json.gz'), all_probs)
        value.update(configuration=config, evidence_identity=frozen['identity'])
        write_once(path, value)
    return {'scene': scene, 'locked': len(configs)}


def evaluate(binding, scene, configs):
    root = Path(binding['output_root'])
    if binding['scenes'][scene]['role'] != 'CAL' and not (root / 'transfer_lock.json').exists():
        raise ValueError('Replica evaluation requires frozen transfer choices')
    locked = {c['id']: read(root / 'locked' / scene / (c['id'] + '.json')) for c in configs}
    if all((root / 'rows' / scene / method / (rank + '.json')).exists() for method in locked for rank in RANKS):
        return {'scene': scene, 'status': 'ROWS_REUSED', 'rows': len(locked) * 2}
    ev = read_scene(read(binding['reviewer_binding']), scene)
    evaluator = SceneEvaluator(ev, root)
    wave_root = Path(read(binding['wave1_binding'])['output_root'])
    original_row = read(wave_root / 'rows' / scene / 'N0/OFFICIAL_CURRENT_CLASS.json')
    original_context = read(original_row['evaluation_receipt'])['context']
    evaluator.context, migration = bound_context(evaluator.context, original_context)
    write_once(root / 'evaluation_context' / (scene + '.json'), migration)
    # Only six explicitly bound controls seed the exact evaluation cache.
    spec = read(binding['spec'])
    for method in spec['controls']:
        for rank in RANKS:
            old = read(wave_root / 'rows' / scene / method / (rank + '.json'))
            receipt = read(old['evaluation_receipt'])
            if receipt['context'] != evaluator.context:
                raise ValueError('parent evaluation context differs')
            target = root / 'evaluation_cache' / old['evaluation_identity']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(Path(old['evaluation_receipt']).parent, target_is_directory=True)
    for method, value in locked.items():
        labels = {int(o): c for o, c in value['labels'].items()}
        for rank in RANKS:
            existing = {p.name for p in (root / 'evaluation_cache').iterdir()}
            row = evaluator.evaluate(labels, method, rank, value['prediction_identity'])
            if method in spec['controls']:
                original = read(wave_root / 'rows' / scene / method / (rank + '.json'))
                if row['metrics'] != original['metrics'] or row['evaluation_identity'] != original['evaluation_identity']:
                    raise ValueError('same-source control metric identity changed')
            write_once(root / 'evaluation_aliases' / scene / method / (rank + '.json'),
                {'scene': scene, 'method': method, 'rank': rank, 'evaluation_identity': row['evaluation_identity'],
                 'reused': row['evaluation_identity'] in existing, 'receipt': row['evaluation_receipt']})
    return {'scene': scene, 'status': 'EVALUATED', 'rows': len(locked) * 2,
            'runtime_overlaps': evaluator.context['runtime_overlaps']}


def pool_methods(binding, split, methods, scene_order=None, directory='pooled'):
    root = Path(binding['output_root'])
    spec = read(binding['spec'])
    scenes = scene_order or spec['datasets']['calibration' if split == 'cal' else 'replica']
    config = read(binding['scenes'][scenes[0]]['config'])
    ns = load_released_module(Path(config['runtime']['upstream']) / 'scripts/eval_utils.py')
    ns['init']('Scannet200' if split == 'cal' else 'Replica')
    rows = [read(root / 'rows' / s / m / (r + '.json')) for s in scenes for m in methods for r in RANKS]
    for method in methods:
        for rank in RANKS:
            output = root / directory / split / method / (rank + '.json')
            if output.exists():
                continue
            pooled = released_pool(root, rows, ns, scenes, method, rank)
            value = {**pooled, 'method': method, 'rank_mode': rank, 'split': split,
                     'runtime_overlaps': ns['overlaps'].tolist()}
            write_once(output, value)


def alias_configuration(binding, scene, config, canonical):
    root = Path(binding['output_root'])
    value = read(root / 'locked' / scene / (config + '.json'))
    value = {**value, 'configuration': {**value['configuration'], 'id': canonical}, 'configuration_alias': config}
    write_once(root / 'locked' / scene / (canonical + '.json'), value)
    for rank in RANKS:
        row = read(root / 'rows' / scene / config / (rank + '.json'))
        write_once(root / 'rows' / scene / canonical / (rank + '.json'), {**row, 'method': canonical, 'configuration_alias': config})
    source = root / 'probabilities' / scene / (config + '.json.gz')
    target = root / 'probabilities' / scene / (canonical + '.json.gz')
    if source.exists() and not target.exists():
        target.symlink_to(source)
