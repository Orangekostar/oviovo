"""Prospective practical ties and immutable CAL-to-Replica boundary."""

from pathlib import Path

from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

from .binding import read_json_or_gzip as read
from .evaluation import (
    alias_configuration,
    configurations,
    evaluate,
    pool_methods,
    predict,
)
from .evidence import B


def choose(rows):
    candidates = list(rows)
    stages = []
    for metric, band in [('apall', .05), ('miou', .1), ('ap50', .1)]:
        maximum = max(r['metrics'][metric] * 100 for r in candidates)
        candidates = [r for r in candidates if r['metrics'][metric] * 100 >= maximum - band - 1e-12]
        stages.append({'metric': metric, 'maximum_pp': maximum, 'band_pp': band,
                       'retained': [r['id'] for r in candidates]})
    selected = min(candidates, key=lambda r: (r['tier'], r['parameter'] or 0, r['registry_order']))
    return {**selected, 'ranking_stages': stages}


def freeze(binding):
    root = Path(binding['output_root'])
    if (root / 'transfer_lock.json').exists():
        return read(root / 'transfer_lock.json')
    spec = read(binding['spec'])
    configs = configurations(spec)
    registry = {v['id']: v for v in spec['variants']}
    rows = []
    for order, c in enumerate(configs):
        metrics = read(root / 'pooled/cal' / c['id'] / 'OFFICIAL_CURRENT_CLASS.json')['metrics']
        rows.append({**c, 'metrics': metrics, 'tier': registry.get(c['method'], {}).get('compute_tier', 0),
                     'registry_order': order})
    choices, active = {}, {}
    for method in registry:
        family = [r for r in rows if r['method'] == method]
        choices[method] = choose(family)
        nonzero = [r for r in family if r['parameter'] is not None and r['parameter'] > 0]
        if nonzero:
            active[method] = choose(nonzero)
        selected = choices[method]
        if selected['id'] != method:
            for scene in spec['datasets']['calibration']:
                alias_configuration(binding, scene, selected['id'], method)
    pool_methods(binding, 'cal', list(registry))
    combo = {'status': 'NOT_TRIGGERED_NO_ACTIVE_PAIR'}
    r3 = active['PE_R3_LOCAL']
    changed = {}
    for name in (r3['id'], 'PE_D4_LINEAGE'):
        count = 0
        for scene in spec['datasets']['calibration']:
            base = read(root / 'locked' / scene / (B + '.json'))['labels']
            actual = read(root / 'locked' / scene / (name + '.json'))['labels']
            count += sum(actual[o] != base[o] for o in base)
        changed[name] = count
    if all(changed.values()):
        c = {'id': 'PE_COMBO_D4_R3', 'method': 'PE_COMBO_D4_R3', 'parameter': r3['parameter']}
        for scene in spec['datasets']['calibration']:
            predict(binding, scene, [c])
        for scene in spec['datasets']['calibration']:
            evaluate(binding, scene, [c])
        pool_methods(binding, 'cal', [c['id']])
        metrics = read(root / 'pooled/cal' / c['id'] / 'OFFICIAL_CURRENT_CLASS.json')['metrics']
        combo = {'status': 'MEASURED', **c, 'metrics': metrics, 'tier': 3, 'registry_order': len(rows),
                 'interaction_probe_uses_nonselected_eta': r3['parameter'] != choices['PE_R3_LOCAL']['parameter']}
    combo['component_changed_CAL_labels'] = changed
    eligible = [r for r in rows if r['method'] in spec['controls'] and r['method'] != 'RV_A7_COS_REFIT']
    eligible += [choices[m] for m, v in registry.items() if v['nomination_eligible']]
    if combo['status'] == 'MEASURED':
        eligible.append(combo)
    nominee = choose(eligible)
    # Exact no-op aliases retain the original baseline's identity.
    if all(read(root / 'locked' / s / (nominee['id'] + '.json'))['labels'] ==
           read(root / 'locked' / s / (B + '.json'))['labels'] for s in spec['datasets']['calibration']):
        nominee = {**next(r for r in eligible if r['id'] == B), 'no_op_selected_alias': nominee['id']}
    lock = {'status': 'CAL_FROZEN', 'binding': binding['identity'], 'choices': choices, 'active_choices': active,
            'combination': combo, 'research_nominee': nominee, 'CAL_grid': rows,
            'Replica_used_for_selection': False, 'deployment': 'N0_UNCHANGED'}
    lock['identity'] = canonical_digest(lock)
    write_once(root / 'transfer_lock.json', lock)
    return lock
