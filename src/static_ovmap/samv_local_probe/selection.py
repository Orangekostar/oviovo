"""Exact exposed-probe gates with tolerance-aware lexicographic research choice."""

from functools import cmp_to_key
import math

METRICS=('apall','ap50','ap25','miou','macc')
CANDIDATES=('SV02_SAMV_GEOM','SV04_SAMV_FC','SV05_COMBINED')


def select(pools,cfg=None):
    cfg=cfg or {};eps=cfg.get('tolerance_fraction',1e-10);flags={};passing=[]
    cohorts=('replica_probe2','cf_probe2')
    def complete(method):
        return all(method in pools.get(c,{}) and all(isinstance(pools[c][method]['metrics'].get(k),(int,float))
            and math.isfinite(pools[c][method]['metrics'][k]) and 0<=pools[c][method]['metrics'][k]<=1 for k in METRICS) for c in cohorts)
    references=complete('SV00_G1') and complete('REF_D2')
    for method in CANDIDATES:
        if not references or not complete(method):flags[method]={'target_met':False,'material_met':False,'reason':'INCOMPLETE_METRICS'};continue
        nondecrease={c:all(pools[c][method]['metrics'][k]>=pools[c]['SV00_G1']['metrics'][k]-eps for k in METRICS) for c in cohorts}
        cf=pools['cf_probe2'][method]['metrics'];d2=pools['cf_probe2']['REF_D2']['metrics']
        cross=cf['apall']>d2['apall']+eps and cf['ap50']>=d2['ap50']-eps
        passed=all(nondecrease.values()) and cross
        flags[method]={'nondecrease':nondecrease,'D2_crossing':cross,'target_met':passed,
            'material_met':passed and cf['apall']-d2['apall']>=cfg.get('material_delta_fraction',.001)}
        if passed:passing.append(method)
    order=cfg.get('lexicographic',['cf_probe2.apall','cf_probe2.ap50','replica_probe2.apall','cf_probe2.miou'])
    simplicity=cfg.get('simplicity_order',list(CANDIDATES))
    def compare(a,b):
        for cell in order:
            c,k=cell.split('.');delta=pools[c][a]['metrics'][k]-pools[c][b]['metrics'][k]
            if abs(delta)>eps:return -1 if delta>0 else 1
        return simplicity.index(a)-simplicity.index(b)
    selected=sorted(passing,key=cmp_to_key(compare))[0] if passing else 'SV00_G1'
    all_complete=references and all(complete(m) for m in CANDIDATES)
    return {'status':('PILOT_TARGET_MET' if passing else 'COMPLETE_NO_PILOT_GAIN') if all_complete else 'PARTIAL_DEPENDENCY_BLOCK',
        'selected':selected,'passing_candidates':passing,'flags':flags,'target_met':bool(passing),
        'material_met':bool(passing) and flags[selected]['material_met'],'deployment':'N0_UNCHANGED',
        'independent_confirmation':False,'automatic_expansion':False}
