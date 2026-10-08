"""Exact unrounded five-metric gates and prescribed deterministic ties."""

import math

METRICS=('apall','ap50','ap25','miou','macc')


def select(pools,cfg):
    epsilon=cfg['epsilon_fraction'];flags={};passing=[]
    baseline_complete=all(all(m in pools.get(c,{}) for m in ('SU00_D2','SU01_G1'))
                          for c in ('replica8','scannet_cf18'))
    for method in cfg['candidates']:
        complete=baseline_complete and all(method in pools.get(c,{}) and all(
            isinstance(pools[c][method]['metrics'].get(m),(int,float))
            and math.isfinite(pools[c][method]['metrics'][m])
            and 0<=pools[c][method]['metrics'][m]<=1 for m in METRICS)
            for c in ('replica8','scannet_cf18'))
        if not complete:
            flags[method]={'target_met':False,'material_target_met':False,'reason':'INCOMPLETE_METRICS'};continue
        rep=all(pools['replica8'][method]['metrics'][m]>=pools['replica8']['SU01_G1']['metrics'][m]-epsilon for m in METRICS)
        cf=all(pools['scannet_cf18'][method]['metrics'][m]>=pools['scannet_cf18']['SU01_G1']['metrics'][m]-epsilon for m in METRICS)
        x=pools['scannet_cf18'][method]['metrics'];d=pools['scannet_cf18']['SU00_D2']['metrics']
        cross=x['apall']>d['apall']+epsilon and x['ap50']>=d['ap50']-epsilon
        passed=rep and cf and cross
        flags[method]={'replica_all5':rep,'cf_all5':cf,'CF_D2_crossing':cross,'target_met':passed,
            'material_target_met':passed and x['apall']-d['apall']>=cfg['material_marker_cf_apall_delta_fraction']-epsilon}
        if passed:passing.append(method)
    def order(method):
        values=[pools[cohort][method]['metrics'][metric] for cohort,metric in
                (s.split('.') for s in cfg['tie_metrics'])]
        return (*values,-cfg['simplicity_order'].index(method))
    selected=max(passing,key=order) if passing else cfg['fallback']
    all_complete=baseline_complete and all(v.get('reason')!='INCOMPLETE_METRICS' for v in flags.values())
    return {'status':('COMPLETE_TARGET_MET' if passing else 'COMPLETE_NO_TARGET_GAIN') if all_complete else 'PARTIAL_DEPENDENCY_BLOCK',
        'selected':selected,'passing_candidates':passing,'flags':flags,'target_met':bool(passing),
        'material_target_met':bool(passing) and flags[selected]['material_target_met'],
        'deployment':'N0_UNCHANGED','independent_confirmation':False}
