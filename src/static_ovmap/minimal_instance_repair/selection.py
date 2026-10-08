"""All-five exposed-cohort target and fixed epsilon-aware simplicity tie-break."""

from functools import cmp_to_key
from pathlib import Path

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import read

from .binding import seal


def select_methods(pooled, spec):
    cfg = spec['selection']
    epsilon = cfg['tolerance_fraction']
    reference = cfg['fallback']
    metrics = spec['evaluation']['metrics']
    outcomes = {}
    for name in cfg['simplicity_order']:
        if any(name not in pooled[cohort] for cohort in spec['cohorts']):
            outcomes[name] = {'status':'BLOCKED_INCOMPLETE_POOL','TARGET_MET':False,'MATERIAL_TARGET_MET':False}
            continue
        checks = {f'{cohort}.{m}.nondecrease':pooled[cohort][name][m] >= pooled[cohort][reference][m]-epsilon
                  for cohort in spec['cohorts'] for m in metrics}
        candidate,baseline = pooled['scannet_cf18'][name],pooled['scannet_cf18']['IR00_D2']
        checks['CF18_APall_strictly_above_D2'] = candidate['apall'] > baseline['apall']+epsilon
        checks['CF18_AP50_at_least_D2'] = candidate['ap50'] >= baseline['ap50']-epsilon
        passed = all(checks.values())
        outcomes[name] = {'status':'COMPLETE','checks':checks,'TARGET_MET':passed,
            'MATERIAL_TARGET_MET':passed and candidate['apall']-baseline['apall'] >= cfg['material_marker_apall_delta_fraction']-epsilon,
            'failed_checks':[key for key,value in checks.items() if not value]}
    candidates = [name for name,item in outcomes.items() if item['TARGET_MET']]
    def compare(left,right):
        for key in cfg['tie_metrics']:
            cohort,metric = key.split('.')
            delta = pooled[cohort][left][metric]-pooled[cohort][right][metric]
            if abs(delta) > epsilon:
                return -1 if delta > 0 else 1
        return cfg['simplicity_order'].index(left)-cfg['simplicity_order'].index(right)
    selected = sorted(candidates,key=cmp_to_key(compare))[0] if candidates else reference
    return {'status':'TARGET_MET' if candidates else 'NO_NEW_METHOD_MEETS_TARGET','selected':selected,
        'TARGET_MET':bool(candidates),'MATERIAL_TARGET_MET':bool(candidates) and outcomes[selected]['MATERIAL_TARGET_MET'],
        'method_outcomes':outcomes,'passing_methods':sorted(candidates,key=cmp_to_key(compare)),
        'deployment':'N0_UNCHANGED','selection_scope':'RETROSPECTIVE_EXPOSED_COHORTS',
        'tolerance_fraction':epsilon,'all_five_gate_unchanged':True}


def select_study(binding):
    root = Path(binding['output_root'])
    store = read(root/'result_store.json')
    values = {cohort:{name:item['metrics'] for name,item in methods.items()}
              for cohort,methods in store['pooled_metrics'].items()}
    result = seal({**select_methods(values,binding['specification']),'result_store_identity':store.get('science_identity',store['identity'])})
    atomic_write_json(root/'selection.json',result)
    return result
