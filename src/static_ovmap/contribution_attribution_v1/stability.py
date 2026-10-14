"""Namespaced repeated matches preserve the original released metric inputs."""
import copy

def namespace_matches(matches,token):
    def rename(value):
        if isinstance(value,dict):return {k:(token+'|'+v if k=='filename' and isinstance(v,str) else rename(v)) for k,v in value.items()}
        if isinstance(value,list):return [rename(v) for v in value]
        return copy.deepcopy(value)
    return {token+'|'+str(scan):rename(row) for scan,row in matches.items()}

def _filenames(value):
    if isinstance(value,dict):
        return ({value['filename']} if isinstance(value.get('filename'),str) else set()).union(*[_filenames(v) for v in value.values()])
    if isinstance(value,list):return set().union(*[_filenames(v) for v in value])
    return set()

def _bootstrap(job):
    from pathlib import Path
    import time
    import warnings
    import numpy as np
    from .binding import load_binding
    from .common import (ConsumptionIndex,unit_identity,cached_record,write,verified,METRICS,plain,family)
    from .records import Records,delta
    from static_ovmap.frozen_fc_routing.common import arrays_record
    from static_ovmap.m2_reviewer_study.evaluation import semantic_summary
    root,cohort,pair=job;root=Path(root);binding=load_binding(root);index=ConsumptionIndex();records=Records(root,binding,index)
    spec=binding['task_specification']['statistics'];scenes=binding['cohorts'][cohort];path=root/'stability'/cohort/('_'.join(pair)+'_bootstrap.json')
    source={m:{s:records.match_input(s,m) for s in scenes} for m in pair}
    inputs={m:{s:dict(score=score['identity'],matches=index.identity(Path(score['manifest']).with_name('matches.json.gz'))) for s,(score,_) in bank.items()} for m,bank in source.items()}
    key=unit_identity('bootstrap',inputs,producers=('common','records','stability'),cohort=cohort,pair=pair,draws=spec['bootstrap_draws'],seed=spec['seed'])
    previous=cached_record(path,key)
    if previous:return previous
    families=sorted({family(s) for s in scenes});bank={f:[s for s in scenes if family(s)==f] for f in families};namespace=records.released(source[pair[0]][scenes[0]][0])
    def measure(sample,method):
        matches={};matrix=None
        for slot,scene in enumerate(sample):
            score,saved=source[method][scene];matches.update(namespace_matches(saved,f'{slot}:{scene}'));m=np.asarray(score['confusion'],np.int64);matrix=m.copy() if matrix is None else matrix+m
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);ap=namespace['evaluate_matches'](matches);averages=plain(namespace['compute_averages'](ap))
        metrics=dict(apall=averages['all_ap'],ap50=averages['all_ap_50%'],ap25=averages['all_ap_25%'],**semantic_summary(matrix))
        return metrics,int(np.isfinite(ap[0,:,0]).sum())
    parity={}
    for method in pair:
        actual,_=measure(scenes,method);expected=records.pool(cohort,method)['metrics']
        for m in METRICS:
            if (actual[m] is None)!=(expected[m] is None) or actual[m] is not None and abs(actual[m]-expected[m])>1e-12:raise ValueError('Full sequence matches-only metric parity failed '+method+' '+m)
        parity[method]=dict(all_five_parity=True,absolute_tolerance=1e-12)
    saved=source[pair[0]][bank[families[0]][0]][1];left=namespace_matches(saved,'DUPLICATE_0');right=namespace_matches(saved,'DUPLICATE_1')
    if set(left)&set(right) or _filenames(left)&_filenames(right):raise ValueError('Repeated family scan/filename namespace collision')
    rng=np.random.default_rng(spec['seed']);draws=[];support=[];indices=[];start=time.perf_counter()
    for draw in range(spec['bootstrap_draws']):
        chosen=rng.integers(0,len(families),size=len(families));sample=[s for j in chosen for s in bank[families[int(j)]]]
        a,ac=measure(sample,pair[0]);b,bc=measure(sample,pair[1]);draws.append(delta(b,a));support.append((ac,bc));indices.append(chosen)
        if (draw+1)%250==0:print('BOOTSTRAP',cohort,pair,draw+1,flush=True)
    intervals={}
    for m in METRICS:
        values=np.asarray([float('nan') if d[m] is None else d[m] for d in draws]);defined=values[np.isfinite(values)]
        intervals[m]=dict(lower_025=float(np.quantile(defined,.025)) if len(defined) else None,upper_975=float(np.quantile(defined,.975)) if len(defined) else None,
                          defined_draws=len(defined),undefined_draws=len(values)-len(defined),positive_fraction=float(np.mean(defined>0)) if len(defined) else None)
    arrays=arrays_record(path.with_suffix('.npz'),dict(metric_names=np.asarray(METRICS),paired_deltas=np.asarray([[float('nan') if d[m] is None else d[m] for m in METRICS] for d in draws]),
                       sampled_family_indices=np.asarray(indices),family_names=np.asarray(families),valid_GT_instance_class_counts=np.asarray(support)),index)
    index.write_memo(path.with_name(path.stem+'_verifications.json'))
    return write(path,dict(status='COMPLETE',input_identity=key,cohort=cohort,pair=pair,draws=len(draws),seed=spec['seed'],intervals=intervals,arrays=arrays,
                 valid_GT_instance_class_count_range=[int(np.min(support)),int(np.max(support))],full_sequence_parity=parity,duplicate_family_namespace_check=True,
                 full_delta=delta(records.pool(cohort,pair[1])['metrics'],records.pool(cohort,pair[0])['metrics']),matches_only_evaluate_calls=2+2*len(draws),new_scene_assign_calls=0,
                 families=families,scope=spec['scope'],undefined_policy='NO_RESAMPLING_NO_ZERO_FILL',target_class_set_changes_with_resampling=True,wall_seconds=time.perf_counter()-start))

def run(root):
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np
    import os
    from .binding import load_binding
    from .common import ConsumptionIndex,unit_identity,cached_record,write,verified,family,METRICS,DependencyError
    from .records import Records,delta
    root=Path(root);binding=load_binding(root);spec=binding['task_specification']['statistics'];index=ConsumptionIndex();records=Records(root,binding,index)
    inputs=dict(main=verified(root/'main_score.json')['identity'],inventory=records.inventory['identity'])
    key=unit_identity('stability',inputs,producers=('common','records','stability'),spec=spec)
    previous=cached_record(root/'stability.json',key)
    if previous:return previous
    lfo=[];missing=[];calls=0
    for cohort,scenes in binding['cohorts'].items():
        families=sorted({family(s) for s in scenes})
        for pair in spec['lfo_pairs']:
            try:
                old,new=[records.pool(cohort,m)['metrics'] for m in pair];full=delta(new,old);rows=[]
                for excluded in families:
                    retained=[s for s in scenes if family(s)!=excluded];a=records.aggregate(retained,pair[0]);b=records.aggregate(retained,pair[1]);calls+=2
                    rows.append(dict(excluded_family=excluded,retained_scenes=retained,delta=delta(b['metrics'],a['metrics']),before=a['metrics'],after=b['metrics']))
                stability={m:dict(positive_fraction=float(np.mean([r['delta'][m]>0 for r in rows if r['delta'][m] is not None])) if any(r['delta'][m] is not None for r in rows) else None,
                                  defined_families=sum(r['delta'][m] is not None for r in rows),minimum=min([r['delta'][m] for r in rows if r['delta'][m] is not None],default=None),maximum=max([r['delta'][m] for r in rows if r['delta'][m] is not None],default=None)) for m in METRICS}
                lfo.append(dict(cohort=cohort,pair=pair,full_delta=full,rows=rows,stability=stability))
            except (DependencyError,FileNotFoundError) as exc:missing.append(dict(cohort=cohort,pair=pair,phase='LFO',reason=str(exc)))
        records.cache.clear()
    bootstrap=[];jobs=[(root,cohort,pair) for cohort in binding['cohorts'] for pair in spec['bootstrap_pairs']]
    workers=int(os.environ.get('OVMAP_ATTRIBUTION_CPU_SLOTS','3'))
    if not 1<=workers<=3:raise ValueError('Task CPU evaluation limit is three')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[(job,executor.submit(_bootstrap,job)) for job in jobs]
        for job,future in futures:
            try:bootstrap.append(future.result())
            except (DependencyError,FileNotFoundError) as exc:missing.append(dict(cohort=job[1],pair=job[2],phase='BOOTSTRAP',status='BLOCKED_MISSING_MATCH_CACHE',reason=str(exc)))
    index.write_memo(root/'stability/verifications.json')
    return write(root/'stability.json',dict(status='COMPLETE' if not missing else 'PARTIAL_DEPENDENCY',input_identity=key,lfo=lfo,bootstrap=bootstrap,missing=missing,
                 matches_only_LFO_calls=calls,matches_only_bootstrap_calls=sum(r['matches_only_evaluate_calls'] for r in bootstrap),bootstrap_draw_pairs=sum(r['draws'] for r in bootstrap),new_model_forwards=0,new_scene_assign_calls=0,scope=spec['scope'],CPU_workers=workers,BLAS_threads=1))
