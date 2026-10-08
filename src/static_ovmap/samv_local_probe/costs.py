"""Actual stage/call accounting and the fixed eight extra window measurements."""

from collections import Counter
from pathlib import Path
import time

import numpy as np

from .common import canonical_digest,read,verified,write


def _timing_parity(root,row,query,model):
    scientific=verified(root/'segmentation'/query['scene']/model/str(query['owner'])/'receipt.json')
    if row['query_identity']!=scientific['query_identity']:raise ValueError('timing input differs from locked science')
    with np.load(row['raw_arrays']['path'],allow_pickle=False) as a,np.load(scientific['raw_arrays']['path'],allow_pickle=False) as b:
        names=sorted(k for k in a.files if k.startswith(('mask_','canonical_mask_')))
        if names!=sorted(k for k in b.files if k.startswith(('mask_','canonical_mask_'))):raise ValueError('timing frame coverage differs')
        same=all(np.array_equal(a[k],b[k]) for k in names)
    if not same:raise ValueError('timed raw binary masks differ from locked scientific masks')
    return {'raw_binary_masks_exact':True,'scientific_identity':scientific['identity'],
            'outside_timing_boundary':True,'mask_arrays_compared':names}


def measure(binding,assets):
    from .runner import require_freeze
    from .segmentation import run_batch
    require_freeze(binding);root=Path(binding['output_root']);queries=[]
    for cohort in ('replica_probe2','cf_probe2'):
        chosen=None
        for scene in binding['cohorts'][cohort]:
            items=verified(root/'query_plan'/scene/'receipt.json')['queries']
            if items:chosen=items[0];break
        if chosen:queries.append(chosen)
    prior_path=root/'timings/summary.json'
    if prior_path.exists():
        prior=verified(prior_path)
        if prior['actual_extra_calls']!=len(queries)*4:raise ValueError('completed timing block has wrong fixed coverage')
        lookup={q['identity']:q for q in queries}
        for row in prior['rows']:
            if row['query_identity'] not in lookup:raise ValueError('locked timing query changed')
            _timing_parity(root,row,lookup[row['query_identity']],row['model'])
        return prior
    schedule=[('SAM2',1,queries),('SAMV',1,queries),('SAMV',2,list(reversed(queries))),('SAM2',2,list(reversed(queries)))]
    # The two adjacent SAM-V rounds share one resident model invocation.
    batches=[('SAM2',[(1,q) for q in queries]),
             ('SAMV',[(1,q) for q in queries]+[(2,q) for q in reversed(queries)]),
             ('SAM2',[(2,q) for q in reversed(queries)])]
    rows=[];begin=time.perf_counter()
    for model,tasks in batches:
        if not tasks:continue
        destinations=[root/'timings'/model/f'round_{rep}'/q['scene']/str(q['owner']) for rep,q in tasks]
        batch=run_batch(binding,assets,model,[q for rep,q in tasks],kind='EXTRA_TIMING',destinations=destinations)
        lookup={(r['query_identity'],str(Path(r['raw_arrays']['path']).parent)):r for r in batch}
        for (rep,query),dest in zip(tasks,destinations):
            row=lookup[(query['identity'],str(dest))]
            parity=_timing_parity(root,row,query,model)
            rows.append({**row,'round':rep,'parity':parity,'OS_page_cache':'UNCONTROLLED'})
    expected=len(queries)*2*2
    if len(rows)!=expected or len({(r['model'],r['round'],r['query_identity']) for r in rows})!=expected:
        raise ValueError('incomplete or duplicated fixed timing block')
    summaries=[]
    for model in ('SAM2','SAMV'):
        for query in queries:
            values=[r for r in rows if r['model']==model and r['query_identity']==query['identity']]
            summaries.append({'model':model,'scene':query['scene'],'owner':query['owner'],'calls':len(values),
                'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in values])),
                'median_seconds':float(np.median([r['elapsed_seconds'] for r in values])),
                'peak_allocated_GiB':max(r['peak_cuda_allocated_bytes'] for r in values)/2**30,
                'peak_reserved_GiB':max(r['peak_cuda_reserved_bytes'] for r in values)/2**30,
                'window_frames':len(query['frames']),'dtype':'BF16','raw_mask_parity':True})
    return write(root/'timings/summary.json',{'status':'FIXED_TIMING_COMPLETE' if expected==8 else 'ABSENT_COHORT_NO_QUERIABLE_TARGET',
        'rows':rows,'summaries':summaries,'actual_extra_calls':len(rows),'expected_available_calls':expected,
        'absent_cohorts':[c for c,names in binding['cohorts'].items() if not any(q['scene'] in names for q in queries)],
        'schedule':[{'model':m,'round':r,'scenes':[q['scene'] for q in qs]} for m,r,qs in schedule],
        'same_locked_windows':True,'resident_models':True,'cross_query_feature_cache':False,'result_cache':False,
        'OS_page_cache':'UNCONTROLLED','end_to_end_latency':'NOT_MEASURED_THIS_PROBE','elapsed_wall_seconds':time.perf_counter()-begin})


def collect(binding,assets,*,run_timing=True):
    root=Path(binding['output_root'])
    timing=measure(binding,assets) if run_timing else verified(root/'timings/summary.json')
    acquisition={};stages={};science=Counter();engineering=Counter();timed=Counter();failures=[];loads=[]
    folders=[*(root/'segmentation/invocations').glob('*'),*(root/'history').glob('**/segmentation/invocations/*')]
    for folder in sorted(set(folders)):
        process=read(folder/'process.json') if (folder/'process.json').exists() else None
        receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None
        if process and (process['exit_code'] or receipt is None or receipt['status']!='COMPLETE'):
            failures.append({'process':process,'worker_receipt':receipt,'actual_GPU_counts':receipt.get('actual_counts') if receipt else None})
        if receipt and 'model_load' in receipt:loads.append({'invocation':folder.name,'model':receipt['model'],
            'kind':process['kind'] if process else None,'model_load':receipt['model_load']})
        if receipt and receipt['status']=='COMPLETE':
            target=timed if process and process['kind']=='EXTRA_TIMING' else engineering if process and process['kind']=='ENGINEERING_PILOT' else science
            target.update(receipt.get('actual_counts',{}))
    for scene in binding['scenes']:
        plan=verified(root/'query_plan'/scene/'receipt.json');observation=verified(root/'observations'/scene/'receipt.json')
        seg={m:[verified(root/'segmentation'/scene/m/str(q['owner'])/'receipt.json') for q in plan['queries']] for m in ('SAM2','SAMV')}
        acquisition[scene]={m:{'unique_locked_targets':len(rows),'actual_counts':dict(sum((Counter(r['actual_counts']) for r in rows),Counter())),
            'resident_inference_seconds':sum(r['elapsed_seconds'] for r in rows),'pilot_leaves_reused':sum(r['kind']=='ENGINEERING_PILOT' for r in rows)} for m,rows in seg.items()}
        fc=verified(root/'readout'/scene/'receipt.json');output=verified(root/'predictions'/scene/'receipt.json')
        stages[scene]={'observation':observation['elapsed_seconds'],'query_planning_and_canonical_preparation':plan['elapsed_seconds'],
            'SAM2_lifting':verified(root/'lifted'/scene/'SAM2/receipt.json')['elapsed_seconds'],
            'SAMV_lifting':verified(root/'lifted'/scene/'SAMV/receipt.json')['elapsed_seconds'],
            'FC_worker_wall_including_first_load':fc['elapsed_seconds'],'FC_counts':fc['counts'],
            'output_rank_construction':output.get('elapsed_seconds'),
            'released_evaluation_registry_and_export':verified(root/'evaluation'/scene/'receipt.json')['elapsed_seconds'],
            'diagnostics':verified(root/'diagnostics'/scene/'receipt.json')['elapsed_seconds'],
            'new_observer_frames':observation['new_observer_frames'],'new_observer_rays':observation['new_rays']}
    unique={m:sum(v[m]['unique_locked_targets'] for v in acquisition.values()) for m in ('SAM2','SAMV')}
    fc_counts=sum((Counter(s['FC_counts']) for s in stages.values()),Counter())
    if max(unique.values())>32 or fc_counts['new_FC_image_inputs']>64 or fc_counts['region_pool_attempts']>128:
        raise ValueError('fixed scientific budget exceeded')
    return write(root/'costs.json',{'status':'COSTS_COMPLETE','unique_scientific_acquisition':acquisition,'stage_seconds':stages,
        'physical_successful_science_only_counts':dict(science),'engineering_pilot_counts':dict(engineering),
        'extra_timing_counts':dict(timed),'unique_scientific_targets':unique,'FC_counts':dict(fc_counts),
        'model_loads':loads,'FC_model_loads':[verified(p) for p in sorted((root/'readout/model_loads').glob('*.json'))],
        'failed_attempts':failures,'unknown_failed_GPU_work_is_null':True,
        'timing_summary_identity':timing['identity'],'timing':timing,'standalone_complete_map_latency':'NOT_MEASURED_THIS_PROBE',
        'new_maps':0,'new_AnyUp':0,'new_NQ':0,'new_training':0,'online_FPS':None,'budgets_verified':True})
