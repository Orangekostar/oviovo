"""Fixed research gates; DEV nomination, exposed selection and deployment differ."""
import math
from pathlib import Path

from .common import METRICS, verified, write

COHORTS = ('replica8', 'scannet_cf18')
EPSILON = 1e-10


def _metrics(pools, cohort, method):
    values = pools[cohort][method]['metrics']
    if any(not isinstance(values.get(k), (int, float)) or not math.isfinite(values[k])
           or not 0 <= values[k] <= 1 for k in METRICS):
        raise ValueError('Complete finite fractional five-metric results required')
    return values


def gate(pools, method):
    values = {c:_metrics(pools,c,method) for c in COHORTS}
    g1 = {c:_metrics(pools,c,'LR01_G1') for c in COHORTS}
    d2 = _metrics(pools,'scannet_cf18','LR00_D2')
    checks = {c:{k:values[c][k] >= g1[c][k]-EPSILON for k in METRICS} for c in COHORTS}
    crossing = values['scannet_cf18']['apall'] > d2['apall']+EPSILON
    ap50 = values['scannet_cf18']['ap50'] >= d2['ap50']-EPSILON
    passed = all(all(v.values()) for v in checks.values()) and crossing and ap50
    return dict(all5_checks=checks, CF_D2_APall_strict=crossing, CF_D2_AP50_nondecrease=ap50,
                target_met=passed, material_target_met=passed and
                values['scannet_cf18']['apall'] >= d2['apall']+.001-EPSILON)


def select(pools, registry, parameter_counts):
    for cohort in COHORTS:
        for method in ('LR00_D2','LR01_G1'): _metrics(pools,cohort,method)
    candidates = [r for r in registry if r['id'] not in ('LR00_D2','LR01_G1')]
    flags = {r['id']:gate(pools,r['id']) for r in candidates}
    passing = [r for r in candidates if flags[r['id']]['target_met']]
    def order(row):
        m = row['id']
        values = [pools[c][m]['metrics'][k] for c,k in
                  (('scannet_cf18','apall'),('replica8','apall'),('scannet_cf18','miou'),('replica8','miou'))]
        return (*values, not row['trained'], -parameter_counts[m] if row['trained'] else 0, -int(m[2:4]))
    winner = max(passing,key=order)['id'] if passing else None
    return dict(status='COMPLETE_TARGET_MET' if winner else 'COMPLETE_NO_TARGET_GAIN', flags=flags,
                passing_candidates=[r['id'] for r in passing], best_regression_candidate=winner,
                research_retained=winner or 'LR01_G1', target_met=bool(winner),
                material_target_met=bool(winner and flags[winner]['material_target_met']),
                selection_scope='EXPLORATORY_ON_REPEATEDLY_EXPOSED_BENCHMARKS',
                runtime_in_gate=False, deployment='N0_UNCHANGED')


def select_study(binding, store):
    root = Path(binding['output_root']); main = verified(root/'dev_nomination.json')
    repeat = verified(root/'repeat_nomination.json')
    if main['status']!='NOMINATED' or repeat['proposed_architecture']!=main['proposed_architecture']:
        raise ValueError('Pre-regression two-seed architecture nominations required')
    params = {r['id']:verified(root/'training/seed17'/r['id']/'receipt.json')['trainable_parameters']
              for r in binding['specification']['methods'] if r['trained']}
    pools = store['pooled_metrics']; result = select(pools,binding['specification']['methods'],params)
    proposed = main['proposed_architecture']; pair = ('LR05_MA_8',proposed)
    repeat_flags = {m:gate(pools,'R29_'+m) for m in pair}
    repeat_deltas = {c:{m:{k:pools[c]['R29_'+m]['metrics'][k]-pools[c][m]['metrics'][k]
                         for k in METRICS} for m in pair} for c in COHORTS}
    paired = {c:{'seed17':{k:pools[c][proposed]['metrics'][k]-pools[c]['LR05_MA_8']['metrics'][k] for k in METRICS},
                 'seed29':{k:pools[c]['R29_'+proposed]['metrics'][k]-pools[c]['R29_LR05_MA_8']['metrics'][k]
                           for k in METRICS}} for c in COHORTS}
    retained = result['research_retained']
    replication = ('UNTRAINED_CONTROL' if retained not in params else 'SINGLE_SEED_CANDIDATE' if retained not in pair
                   else 'REPEAT_GATE_PASSES' if repeat_flags[retained]['target_met'] else 'REPEAT_GATE_FAILS')
    return write(root/'selection.json',dict(**result,dev_nomination=main['identity'],
                 dev_proposed_architecture=proposed, repeat_nomination=repeat['identity'],
                 repeat_consistency=dict(flags=repeat_flags,seed29_minus_seed17=repeat_deltas,
                                         proposed_minus_MA_by_seed=paired),
                 research_retained_replication=replication,parameter_counts=params,
                 independent_confirmation=False,training_or_seed_selection_uses_regression=False))
