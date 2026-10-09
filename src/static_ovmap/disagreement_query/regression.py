"""Conditional fixed six-path regression; a failed screen is a complete stop."""
from pathlib import Path

from .acquisition import acquire, anchor_requests, seal_choices, second_requests
from .analysis import analyze
from .common import ConsumptionIndex, canonical_digest, verified, write
from .evaluation import evaluate
from .outputs import predict
from .query_plan import plan_scene
from .selection import final_gate


def expand(binding):
    root = Path(binding['output_root'])
    selection = verified(root/'screen_selection.json')
    if not selection['run_full_regression']:
        return write(root/'regression.json', {
            'status': 'NOT_RUN_NO_SCREEN_SIGNAL', 'screen_selection_identity': selection['identity'],
            'scene_method_coverage': 0, 'ordered_pool_coverage': 0,
            'selected_method': 'DQ01_G1', 'strict_target_pass': None,
            'material_target_pass': None, 'target_status': 'NOT_EVALUATED_FULL_REGRESSION_NOT_TRIGGERED',
            'deployment': 'N0_UNCHANGED'})
    qs, us = selection['QS'], selection['US']
    paths = {'DQ00_D2': None, 'DQ01_G1': None, 'QS_US': (qs, us),
             'QD_US': ('DISAGREEMENT', us), 'QS_SUPPORT': (qs, 'SUPPORT'),
             'QD_SUPPORT': ('DISAGREEMENT', 'SUPPORT')}
    index = ConsumptionIndex(root/'input_verifications.json')
    operators = [index.identity(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    frozen = {'screen_selection_identity': selection['identity'], 'QS': qs, 'US': us,
              'paths': paths, 'operators': operators, 'spec': binding['spec'],
              'status': 'FROZEN_BEFORE_ADDITIONAL_22_AP'}
    freeze_path = root/'regression_freeze.json'
    if freeze_path.exists():
        if verified(freeze_path)['identity'] != canonical_digest(frozen):
            raise ValueError('full-regression freeze changed')
    else:
        write(freeze_path, frozen)
    pilot = {s for group in binding['screen_cohorts'].values() for s in group}
    names = [s for group in binding['cohorts'].values() for s in group]
    added = {s: plan_scene(binding, s) for s in names if s not in pilot}
    anchor = acquire(binding, 'regression_anchor', added, anchor_requests(added))
    new_choices = seal_choices(binding, added, anchor)
    second = acquire(binding, 'regression_second', added,
                     second_requests(new_choices, (qs, 'DISAGREEMENT')), choices=new_choices)
    plans = {s: verified(root/'plans'/s/'manifest.json') for s in names}
    choices = {s: verified(root/'choices'/s/'receipt.json') for s in names}
    def merge(stage, receipt):
        previous = verified(root/'acquisition'/stage/'receipt.json')
        return write(root/'acquisition'/('merged_'+stage+'.json'), {
            'status': 'COMPLETE', 'records': {**previous['records'], **receipt['records']},
            'input_identities': [previous['identity'], receipt['identity']],
            'no_physical_repeat_for_pilot': True})
    anchors = merge('screen_anchor', anchor)
    seconds = merge('screen_second', second)
    locks = predict(binding, 'regression', binding['cohorts'], plans, choices, anchors, seconds, paths)
    store = evaluate(binding, 'regression', binding['cohorts'], list(paths))
    if store['scene_method_coverage'] != 156 or store['ordered_pool_coverage'] != 12:
        raise ValueError('full realized-branch coverage differs')
    analyze(binding, 'regression', binding['cohorts'], list(paths))
    pools = store['pooled_metrics']
    reference = {c: pools[c]['DQ01_G1']['metrics'] for c in pools}
    d2 = {c: pools[c]['DQ00_D2']['metrics'] for c in pools}
    order = ['QS_US', 'QS_SUPPORT', 'QD_US', 'QD_SUPPORT']
    passed, ranks, material = [], {}, {}
    for i, method in enumerate(order):
        metrics = {c: pools[c][method]['metrics'] for c in pools}
        strict = final_gate(metrics, reference, d2)
        material[method] = bool(strict and metrics['scannet_cf18']['apall']-d2['scannet_cf18']['apall'] >= .001-1e-10)
        if strict:
            passed.append(method)
            ranks[method] = (metrics['scannet_cf18']['apall'], metrics['replica8']['apall'],
                             metrics['scannet_cf18']['miou'], metrics['replica8']['miou'],
                             -locks['costs'][method]['logical_FULL_reads'],
                             -locks['costs'][method]['logical_probe_heads'], -i)
    selected = max(passed, key=lambda m: ranks[m]) if passed else 'DQ01_G1'
    return write(root/'regression.json', {
        'status': 'COMPLETE_TARGET_GAIN' if passed else 'COMPLETE_NO_TARGET_GAIN',
        'store_identity': store['identity'], 'scene_method_coverage': 156, 'ordered_pool_coverage': 12,
        'selected_method': selected, 'strict_target_pass': bool(passed),
        'material_target_pass': material.get(selected, False), 'candidate_material_pass': material,
        'candidate_strict_pass': {m: m in passed for m in order}, 'selection_ranks': ranks,
        'paths': paths, 'costs': locks['costs'], 'screen_selection_identity': selection['identity'],
        'freeze_identity': verified(freeze_path)['identity'], 'deployment': 'N0_UNCHANGED',
        'exposed_regression_not_independent_confirmation': True})
