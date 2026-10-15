"""Four fixed family bootstrap contrasts; seeds are measured separately."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import warnings
import numpy as np
from .common import (verified,write,arrays_record,ConsumptionIndex,read_document,
    unit_key,existing,immutable_record,canonical_digest)
from .metrics import partitions
from .evaluation import released,ap_metrics
from static_ovmap.contribution_attribution_v1.stability import namespace_matches

METRICS=('GT_all_top1','GT_all_balanced_top1','GT_base_top1','GT_novel_top1','GT_H_base_novel','GT_A','GT_M','GT_C','real_apall','real_ap50','real_ap25')

def column(columns,method,seed,profile):
    static=method.startswith('B_')
    return next(i for i,c in enumerate(columns) if c['method']==method and c['prefix']==8 and (static or c['profile']==profile and c['seed']==seed))

def flatten_GT(value):
    return dict(GT_all_top1=value['all']['ordinary_top1'],GT_all_balanced_top1=value['all']['balanced_top1'],GT_base_top1=value['base']['ordinary_top1'],GT_novel_top1=value['novel']['ordinary_top1'],GT_H_base_novel=value['H_base_novel'],GT_A=value['all']['A'],GT_M=value['all']['M'],GT_C=value['all']['C'])

def seed_mean_delta(left,right):
    return {m:float(np.mean([left[s][m]-right[s][m] for s in (17,29)])) if all(left[s][m] is not None and right[s][m] is not None for s in (17,29)) else None for m in METRICS}

def bootstrap_pair(job):
    import time
    binding,pair=job;root=Path(binding['output_root']);gt=verified(root/'confirmation/GT/score.json');real=verified(root/'confirmation/real_score.json');scenes=verified(root/'role_lock.json')['roles']['CONF'];families=[s.split('_')[0] for s in scenes]
    spec=binding['task_specification']['statistics'];identity=unit_key('bootstrap_pair',dict(GT=gt['identity'],real=real['identity']),producers=('statistics','metrics','evaluation'),pair=pair,draws=1000,seed=4314)
    folder=root/'statistics'/('_minus_'.join(pair));path=folder/'receipt.json';old=existing(path,identity)
    if old:return old
    index=ConsumptionIndex(folder/'verifications.json');namespace,_=released(binding,index);GTcols={};realcols={};GTbank={};realbank={}
    for method in pair:
        for seed in (17,29):
            k=(method,seed);gi=column(gt['columns'],method,seed,'matched1000');ri=column(verified(root/'confirmation/real_prediction_lock.json')['columns'],method,seed,'selected');GTcols[k]=gi;realcols[k]=ri
            GTbank[k]={family:[r for r in gt['records'][str(gi)] if r['family']==family] for family in families};realbank[k]={}
            for scene,family in zip(scenes,families):
                row=next(r for r in real['scene_rows'] if r['scene']==scene and r['column']==ri)
                realbank[k][family]=read_document(row['matches']['path'],index,sealed=False,expected=row['matches'])
    calls=0;cache={}
    def measure(sample,method,seed):
        nonlocal calls
        rows=[];matches={};k=(method,seed)
        for slot,j in enumerate(sample):
            family=families[int(j)];rows.extend(dict(r,family=str(slot)+'|'+family) for r in GTbank[k][family]);matches.update(namespace_matches(realbank[k][family],str(slot)))
        metrics=flatten_GT(partitions(rows));cache_key=(realcols[k],tuple(map(int,sample)))
        if cache_key not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);ap=namespace['evaluate_matches'](matches);am,_=ap_metrics(namespace,ap)
            cache[cache_key]=(am,int(np.isfinite(ap[0,:,0]).sum()));calls+=1
        am,support=cache[cache_key];metrics.update({'real_'+m:v for m,v in am.items()})
        return metrics,dict(GT_originals=len(rows),GT_positive_classes=len({r['class_id'] for r in rows}),real_GT_positive_classes=support)
    full=np.arange(len(families));point={};parity=[]
    for method in pair:
        point[method]={}
        for seed in (17,29):
            metrics,support=measure(full,method,seed);point[method][seed]=metrics;expected=next(r for r in real['pools'] if r['column']==realcols[(method,seed)])['metrics']
            for m in ('apall','ap50','ap25'):
                if (metrics['real_'+m] is None)!=(expected[m] is None) or metrics['real_'+m] is not None and abs(metrics['real_'+m]-expected[m])>1e-12:raise ValueError('Released bootstrap full-sequence parity failed')
            parity.append(dict(method=method,seed=seed,all_three_AP_parity=True,tolerance=1e-12))
    rng=np.random.default_rng(4314);draws=rng.integers(0,len(families),size=(1000,len(families)));deltas=[];supports=[];started=time.perf_counter()
    for n,sample in enumerate(draws):
        measures={};support_rows=[]
        for method in pair:
            measures[method]={}
            for seed in (17,29):
                values,support=measure(sample,method,seed);measures[method][seed]=values;support_rows.append([support['GT_originals'],support['GT_positive_classes'],support['real_GT_positive_classes']])
        d=seed_mean_delta(measures[pair[0]],measures[pair[1]]);deltas.append([float('nan') if d[m] is None else d[m] for m in METRICS]);supports.append(support_rows)
        if (n+1)%100==0:print('CONF_BOOTSTRAP',pair,n+1,flush=True)
    intervals={}
    for i,m in enumerate(METRICS):
        values=np.asarray(deltas)[:,i];defined=values[np.isfinite(values)]
        intervals[m]=dict(lower_025=float(np.quantile(defined,.025)) if len(defined) else None,upper_975=float(np.quantile(defined,.975)) if len(defined) else None,defined_draws=len(defined),undefined_draws=1000-len(defined),positive_fraction=float(np.mean(defined>0)) if len(defined) else None)
    support_array=np.asarray(supports,np.int64)
    support_ranges={name:dict(minimum=int(support_array[...,i].min()),maximum=int(support_array[...,i].max())) for i,name in enumerate(('GT_originals','GT_positive_classes','real_GT_positive_classes'))}
    arrays=arrays_record(folder/'bootstrap.npz',dict(sampled_family_indices=draws,family_names=np.asarray(families),metric_names=np.asarray(METRICS),paired_seed_mean_deltas=np.asarray(deltas),support_counts=support_array),index)
    index.write_memo(folder/'verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,pair=pair,draw_sets=1000,seed=4314,arrays=arrays,intervals=intervals,draw_support_ranges=support_ranges,full_delta=seed_mean_delta(point[pair[0]],point[pair[1]]),full_per_seed=point,full_sequence_parity=parity,GT_profile='MATCHED_STEP1000',real_profile='SEED17_DEV_SELECTED_SAME_STEP_SEED29',two_seeds_not_concatenated=True,same_draws_GT_and_real=True,families=families,elapsed_seconds=time.perf_counter()-started,matches_only_evaluate_calls=calls,new_assign_instances_calls=0,undefined_draws_retained=True))

def factorial(binding):
    root=Path(binding['output_root']);selection=verified(root/'model_selection_lock.json');out=[]
    for seed in (17,29):
        for profile in ('matched1000','selected','terminal2000'):
            cells={m:selection['arms'][m+':seed'+str(seed)][profile] for m in ('S_LL','S_UL','S_LU','S_UU')}
            for metric in ('A','M','C'):
                ll,ul,lu,uu=[cells[m][metric] for m in ('S_LL','S_UL','S_LU','S_UU')];inner=.5*((ll-ul)+(lu-uu));outer=.5*((ll-lu)+(ul-uu));interaction=ll-lu-ul+uu
                if abs(inner+outer-(ll-uu))>1e-12:raise ValueError('Factorial arithmetic failed')
                out.append(dict(seed=seed,profile=profile,metric=metric,inner_relative_mean_effect=inner,outer_relative_mean_effect=outer,interaction=interaction,LL_minus_UU=ll-uu,interaction_not_added_twice=True,cells={m:cells[m][metric] for m in cells}))
    return write(root/'statistics/factorial_DEV.json',dict(status='COMPLETE',rows=out,scope='EXPOSED_DEV;PROFILE_RETAINED_SEPARATELY'))

def factorial_confirmation(binding):
    root=Path(binding['output_root']);gt=verified(root/'confirmation/GT/score.json');real=verified(root/'confirmation/real_score.json');real_columns=verified(root/'confirmation/real_prediction_lock.json')['columns'];rows=[]
    methods=('S_LL','S_UL','S_LU','S_UU')
    for seed in (17,29):
        for profile in ('matched1000','selected'):
            gt_cells={m:flatten_GT(gt['metrics'][str(column(gt['columns'],m,seed,profile))]['full_originals']) for m in methods}
            real_cells={m:next(r for r in real['pools'] if r['column']==column(real_columns,m,seed,profile))['metrics'] for m in methods}
            for scope,cells,metrics in (('CONF_GT_FULL_ORIGINALS',gt_cells,METRICS[:8]),('CONF_REAL_SPARSE_AP',real_cells,('apall','ap50','ap25'))):
                for metric in metrics:
                    values={m:cells[m][metric] for m in methods};ll,ul,lu,uu=[values[m] for m in methods]
                    defined=all(v is not None for v in values.values())
                    inner=.5*((ll-ul)+(lu-uu)) if defined else None;outer=.5*((ll-lu)+(ul-uu)) if defined else None
                    interaction=ll-lu-ul+uu if defined else None;total=ll-uu if defined else None
                    if defined and abs(inner+outer-total)>1e-12:raise ValueError('CONF factorial arithmetic failed')
                    rows.append(dict(seed=seed,profile=profile,scope=scope,metric=metric,inner_relative_mean_effect=inner,outer_relative_mean_effect=outer,interaction=interaction,LL_minus_UU=total,cells=values,undefined_retained=True,interaction_not_added_twice=True))
    return write(root/'statistics/factorial_CONF.json',dict(status='COMPLETE',rows=rows,GT=gt['identity'],real=real['identity'],scope='FIXED_CONFIRMATION;POINT_EFFECTS_ONLY;NO_ADDITIONAL_BOOTSTRAP_SEARCH'))

def run(binding,*,phase='statistics'):
    root=Path(binding['output_root']);pairs=binding['task_specification']['statistics']['confirmation_pairs'];gt=verified(root/'confirmation/GT/score.json');real=verified(root/'confirmation/real_score.json')
    identity=unit_key('statistics',dict(GT=gt['identity'],real=real['identity']),producers=('statistics','metrics','evaluation'),pairs=pairs);path=root/'statistics/receipt.json';old=existing(path,identity)
    if old:return old
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(bootstrap_pair,[(binding,p) for p in pairs]))
    if sum(r['draw_sets'] for r in rows)!=4000:raise ValueError('Exactly4000 confirmatory draw sets required')
    factors=factorial(binding);confirm_factors=factorial_confirmation(binding)
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,pairs=rows,draw_sets=4000,factorial=factors['identity'],confirmation_factorial=confirm_factors['identity'],old_attribution_bootstrap_rerun=False,CPU_workers=3,BLAS_threads=1))
