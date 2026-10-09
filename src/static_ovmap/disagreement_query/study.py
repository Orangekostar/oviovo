"""Fixed two-stage screen and frozen control selection."""
from pathlib import Path

from .acquisition import acquire,anchor_requests,seal_choices,second_requests
from .analysis import analyze,contrast
from .common import canonical_digest,read,verified,write
from .evaluation import evaluate
from .outputs import predict
from .runtime import engineering
from .selection import choose_screen,screen_gate


def screen(binding):
    root=Path(binding['output_root']);cohorts=binding['screen_cohorts']
    if verified(root/'diagnostics/summary.json')['status']!='DIAGNOSTICS_COMPLETE':raise ValueError('all prescribed diagnostics required')
    engineer=engineering(binding)
    plans={s:verified(root/'plans'/s/'manifest.json') for subset in cohorts.values() for s in subset}
    anchor=acquire(binding,'screen_anchor',plans,anchor_requests(plans))
    choices=seal_choices(binding,plans,anchor)
    second=acquire(binding,'screen_second',plans,second_requests(choices),choices=choices)
    paths={'DQ00_D2':None,'DQ01_G1':None,
           **{p+'_MEAN':(p,'MEAN') for p in ('AREA','COVERAGE','VERIFY','DISAGREEMENT')}}
    locks_A=predict(binding,'screen_A',cohorts,plans,choices,anchor,second,paths)
    store_A=evaluate(binding,'screen_A',cohorts,list(paths))
    if store_A['scene_method_coverage']!=24 or store_A['ordered_pool_coverage']!=12:raise ValueError('screen A coverage differs')
    simple_methods=['AREA_MEAN','COVERAGE_MEAN','VERIFY_MEAN']
    qs_method,qs_ranks=choose_screen(store_A['pooled_metrics'],simple_methods,locks_A['costs'])
    qs=qs_method.removesuffix('_MEAN')
    query_selection=write(root/'screen_query_selection.json',{'status':'QS_FROZEN_BEFORE_STAGE_B','QS':qs,'method':qs_method,
        'screen_A_identity':store_A['identity'],'ranks':qs_ranks,'fixed_order':simple_methods})
    paths_B={'QS_AREA':(qs,'AREA'),'QS_SUPPORT':(qs,'SUPPORT'),
             'QD_AREA':('DISAGREEMENT','AREA'),'QD_SUPPORT':('DISAGREEMENT','SUPPORT')}
    locks_B=predict(binding,'screen_B',cohorts,plans,choices,anchor,second,paths_B)
    store_B=evaluate(binding,'screen_B',cohorts,list(paths_B))
    if store_B['scene_method_coverage']!=16 or store_B['ordered_pool_coverage']!=8:raise ValueError('screen B coverage differs')
    pools={cohort:{**store_A['pooled_metrics'][cohort],**store_B['pooled_metrics'][cohort]} for cohort in cohorts}
    costs={**locks_A['costs'],**locks_B['costs']}
    us_method,us_ranks=choose_screen(pools,[qs_method,'QS_AREA'],costs)
    us='MEAN' if us_method==qs_method else 'AREA';qd_us='DISAGREEMENT_MEAN' if us=='MEAN' else 'QD_AREA'
    analyze(binding,'screen_A',cohorts,list(paths));analyze(binding,'screen_B',cohorts,list(paths_B))
    all_scenes=[s for subset in cohorts.values() for s in subset]
    pairs=[(qd_us,us_method,'screen_A' if us=='MEAN' else 'screen_B','screen_A' if us=='MEAN' else 'screen_B'),
        ('QS_SUPPORT',us_method,'screen_B','screen_A' if us=='MEAN' else 'screen_B'),
        ('QD_SUPPORT',qd_us,'screen_B','screen_A' if us=='MEAN' else 'screen_B')]
    comparisons=[]
    for candidate,reference,a_stage,b_stage in pairs:
        comparison=contrast(binding,candidate,reference,a_stage,b_stage,all_scenes)
        cm={cohort:pools[cohort][candidate]['metrics'] for cohort in cohorts}
        rm={cohort:pools[cohort][reference]['metrics'] for cohort in cohorts}
        g1={cohort:pools[cohort]['DQ01_G1']['metrics'] for cohort in cohorts}
        passed=screen_gate(cm,rm,g1,changed=comparison['actual_prediction_changed'],
            corrections_net=comparison['corrections_net'],gt50_net=comparison['gt50_net'])
        comparisons.append({**comparison,'passed_screen_extension_gate':bool(passed),
            'candidate_pools':{cohort:pools[cohort][candidate]['identity'] for cohort in cohorts},
            'reference_pools':{cohort:pools[cohort][reference]['identity'] for cohort in cohorts}})
    signal=any(c['passed_screen_extension_gate'] for c in comparisons)
    selection=write(root/'screen_selection.json',{'status':'COMPLETE_SCREEN_SIGNAL' if signal else 'COMPLETE_NO_SCREEN_SIGNAL',
        'input_identity':canonical_digest({'A':store_A['identity'],'B':store_B['identity'],'pairs':comparisons}),
        'QS':qs,'US':us,'QS_US_screen_method':us_method,'QD_US_screen_method':qd_us,
        'QS_choice_identity':query_selection['identity'],'US_ranks':us_ranks,'comparisons':comparisons,
        'run_full_regression':signal,'frozen_before_additional_22_scenes':True,
        'screen_is_development_not_independent_confirmation':True,'deployment':'N0_UNCHANGED'})
    return write(root/'screen_store.json',{'status':'SCREEN_COMPLETE','cohorts':cohorts,
        'scene_metrics':store_A['scene_metrics']+store_B['scene_metrics'],'pooled_metrics':pools,
        'methods':list(paths)+list(paths_B),'paths':{**paths,**paths_B},'costs':costs,
        'scene_method_coverage':40,'ordered_pool_coverage':20,'stage_A_identity':store_A['identity'],
        'stage_B_identity':store_B['identity'],'selection_identity':selection['identity'],'engineering_identity':engineer['identity'],
        'physical_evidence_reused_by_stage_B':True,'stage_B_new_FC_queries':0})


def run_phase(binding,phase):
    if phase=='screen':return screen(binding)
    from .regression import expand
    from .runtime import conditional_timing
    from .verification import original_scorer_parity
    from .reporting import report
    from .compliance import verify
    if phase=='all':
        screen(binding)
        original_scorer_parity(binding)
        expand(binding)
        conditional_timing(binding)
        report(binding)
        result=verify(binding)
        print('ALL_REALIZED_BRANCH_COMPLETE',result['identity'],flush=True)
        return result
    if phase=='expand':return expand(binding)
    if phase=='time':return conditional_timing(binding)
    if phase=='report':
        original_scorer_parity(binding)
        return report(binding)
    if phase=='verify':return verify(binding)
    raise ValueError('unknown lifecycle phase: '+phase)
