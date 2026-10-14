"""Explicit crossed-label/rank views do not recalculate current-class ranks."""
import numpy as np

def explicit_view(projected_owners,labels,ranks,minimum):
    ranks={int(k):float(v) for k,v in ranks.items()}
    owners,counts=np.unique(projected_owners,return_counts=True)
    positive=[int(o) for o in owners if o>0]
    if any(o not in ranks for o in positive):raise ValueError('Unsupported missing cross rank for positive owner')
    if any(not np.isfinite(ranks[o]) for o in positive):raise ValueError('Invalid explicit rank')
    return {int(o):dict(label=int(labels[int(o)]),rank=f'{ranks[int(o)]:.6f}',area=int(n)) for o,n in zip(owners,counts,strict=True) if o>0 and n>=minimum and labels[int(o)]!=0}

def score_explicit_view(evaluator,labels,ranks,method,label_score,folder,index,partial_units=None):
    """Direct released matcher with the supplied six-decimal rank manifest."""
    from pathlib import Path
    import time
    from .common import canonical_digest,read_document,write
    from static_ovmap.m2_reviewer_study.evaluation import write_manifest,write_gzip,semantic_summary
    from static_ovmap.released_trace import trace_released_matches
    from static_ovmap.cvpr_compact.evaluation import fraction_metrics
    view=explicit_view(evaluator.owners,labels,ranks,evaluator.minimum)
    identity=canonical_digest(dict(context=evaluator.context,labels=labels,view=view,rank_mode='DIAGNOSTIC_CROSSED_LABEL_RANK'))
    folder=Path(folder)/identity;path=folder/'receipt.json'
    if path.exists():return read_document(path,index),False
    folder.mkdir(parents=True,exist_ok=True);manifest=folder/'pred_inst_sem_mapping.txt';start=time.perf_counter()
    matches_path=folder/'matches.json.gz';trace_path=folder/'trace.json.gz';recovered=False
    if matches_path.exists() and trace_path.exists():
        proof=next((r for r in (partial_units or []) if r['matches']['path']==str(matches_path)),None)
        if proof is None:raise ValueError('Unregistered incomplete rank unit; retain its files before repair')
        index.identity(matches_path,proof['matches']);trace=read_document(trace_path,index,sealed=False,expected=proof['trace'])
        check=manifest.with_name('manifest_check.tmp');write_manifest(check,evaluator.masks,view)
        if check.read_bytes()!=manifest.read_bytes():raise ValueError('Recovered rank manifest differs from explicit current view')
        check.unlink()
        if not trace['parity']['ap_exact'] or not trace['parity']['pr_and_fn_exact']:raise ValueError('Incomplete original rank trace lacks numerical parity')
        shape=(len(evaluator.namespace['dist_threshes']),len(evaluator.namespace['CLASS_LABELS']),len(evaluator.namespace['overlaps']));ap=np.full(shape,np.nan)
        for state in trace['states']:ap[tuple(state[k] for k in ('distance_index','class_index','overlap_index'))]=np.nan if state['ap'] is None else state['ap']
        recovered=True
    else:
        write_manifest(manifest,evaluator.masks,view)
        gt,pred=evaluator.namespace['assign_instances_for_scan'](str(evaluator.root),str(manifest),evaluator.targets['gt_instance_path'])
        matches={str(Path(evaluator.targets['gt_instance_path']).resolve()):dict(gt=gt,pred=pred)}
        ap,trace=trace_released_matches(evaluator.namespace,matches)
        write_gzip(matches_path,matches);write_gzip(trace_path,trace)
    averages=evaluator.namespace['compute_averages'](ap)
    matrix=label_score['confusion'];metrics=fraction_metrics(dict(apall=averages['all_ap'],ap50=averages['all_ap_50%'],ap25=averages['all_ap_25%'],**semantic_summary(matrix)))
    result=write(path,dict(status='COMPLETE',input_identity=identity,metrics=metrics,context=evaluator.context,
                          rank_mode='DIAGNOSTIC_CROSSED_LABEL_RANK',confusion=matrix,view=view,manifest=str(manifest),
                          gt_path=str(evaluator.targets['gt_instance_path']),trace_parity=trace['parity'],elapsed_seconds=None if recovered else time.perf_counter()-start,
                          recovered_completed_matcher_and_trace=recovered,new_assign_this_invocation=not recovered))
    return result,not recovered

def _scene(job):
    from pathlib import Path
    from .binding import load_binding
    from .common import verified,ConsumptionIndex,write,unit_identity,cached_record,_array_digest,canonical_digest
    from .records import Records
    from .scoring import evaluator_for
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.recovery_wave2.evaluation import scorer_context
    root,scene,pair=job;root=Path(root);binding=load_binding(root);a,b=pair;folder=root/'rank'/('_'.join(pair))/scene;index=ConsumptionIndex()
    records=Records(root,binding,index);payloads=[records.payload(scene,m) for m in pair];scores=[records.score(scene,m) for m in pair]
    key=unit_identity('rank_scene',dict(scores=[s['identity'] for s in scores],payloads=[p.prediction_key for p in payloads]),producers=('common','records','scoring','rank'),scene=scene,pair=pair)
    previous=cached_record(folder/'receipt.json',key)
    if previous:return previous
    if payloads[0].geometry!=payloads[1].geometry or not np.array_equal(payloads[0].owner_ids,payloads[1].owner_ids):
        return write(folder/'receipt.json',dict(status='UNSUPPORTED_DIFFERENT_OWNER_DOMAIN',input_identity=key,scene=scene,pair=pair))
    owners=records.projected(scene,a,scores[0]);labels=[owner_labels(p) for p in payloads];ranks=[dict(p.instance_ranks) for p in payloads];views=[]
    try:
        for i in range(2):
            view=explicit_view(owners,labels[i],ranks[i],scores[i]['context']['minimum'])
            if {str(k):v for k,v in view.items()}!=scores[i]['view']:raise ValueError('Explicit diagonal view differs from official receipt')
            views.append(view)
    except ValueError as exc:
        if 'missing cross rank' not in str(exc):raise
        return write(folder/'receipt.json',dict(status='UNSUPPORTED_MISSING_CROSS_RANK',input_identity=key,scene=scene,pair=pair,reason=str(exc)))
    context_record=next(r for r in records.predictions['records'] if r['scene']==scene)
    evaluator=evaluator_for(binding,context_record,payloads[0],folder/'partition',index)
    for score in scores:
        if scorer_context(evaluator.context)!=scorer_context(score['context']):raise ValueError('Crossed scorer context differs')
    repair=root/'repairs/rank_partial_reuse.json';partial_units=verified(repair)['retained_units'] if repair.exists() else []
    cells={};assigns=0;recovered=0
    for l in range(2):
        for r in range(2):
            name=f'D{l}{r}';new=False
            if l==r:score=scores[l]
            else:
                score,new=score_explicit_view(evaluator,labels[l],ranks[r],name,scores[l],folder/'scores',index,partial_units);assigns+=int(new);recovered+=int(score.get('recovered_completed_matcher_and_trace',False))
            if score['confusion']!=scores[l]['confusion']:raise ValueError('Crossed confusion must follow labels only')
            cells[name]=dict(labels=pair[l],ranks=pair[r],metrics=score['metrics'],evaluation_receipt=str(Path(score['manifest']).with_name('receipt.json')),
                             evaluation_identity=score['identity'],rank_mode=score['rank_mode'],measurement_kind='OFFICIAL_DIAGONAL_ALIAS' if l==r else 'DIAGNOSTIC_CROSSED_LABEL_RANK')
    index.write_memo(folder/'verifications.json');print('RANK_SCENE',scene,a,b,flush=True)
    return write(folder/'receipt.json',dict(status='COMPLETE',input_identity=key,scene=scene,pair=pair,cells=cells,physical_scene_assign_calls=assigns+recovered,new_assign_calls_this_invocation=assigns,recovered_completed_scene_assign_calls=recovered,physical_evaluate_matches_calls=3*(assigns+recovered),diagonal_view_parity=True))

def run(root):
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    from .binding import load_binding
    from .common import ConsumptionIndex,unit_identity,cached_record,verified,write,read_document,METRICS
    from .records import Records,aggregate_scores
    from .history import factorial
    import os
    root=Path(root);binding=load_binding(root);spec=binding['task_specification'];key=unit_identity('rank',dict(main=verified(root/'main_score.json')['identity'],inventory=verified(root/'inventory.json')['identity']),producers=('common','records','scoring','rank'),pairs=spec['rank_pairs'])
    previous=cached_record(root/'rank.json',key)
    if previous:return previous
    index=ConsumptionIndex();records=Records(root,binding,index);rows=[];pools={};unsupported=[]
    scenes=[s for v in binding['cohorts'].values() for s in v]
    workers=int(os.environ.get('OVMAP_ATTRIBUTION_CPU_SLOTS','3'))
    if not 1<=workers<=3:raise ValueError('Task CPU scorer limit is three')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for pair in spec['rank_pairs']:
            current=list(executor.map(_scene,[(root,s,pair) for s in scenes]));rows+=current
            bad=[r for r in current if r['status']!='COMPLETE'];unsupported+=bad
            if bad:continue
            name='_'.join(pair);pools[name]={}
            for cohort,ordered in binding['cohorts'].items():
                cells={}
                for l in range(2):
                    for r in range(2):
                        cell=f'D{l}{r}'
                        if l==r:cells[cell]=dict(records.pool(cohort,pair[l]),measurement_kind='OFFICIAL_DIAGONAL_ALIAS')
                        else:
                            scores=[read_document(next(row for row in current if row['scene']==s)['cells'][cell]['evaluation_receipt'],index) for s in ordered]
                            cells[cell]=dict(aggregate_scores(scores,index),measurement_kind='DIAGNOSTIC_CROSSED_LABEL_RANK')
                effects={m:factorial(cells['D00']['metrics'][m],cells['D10']['metrics'][m],cells['D01']['metrics'][m],cells['D11']['metrics'][m]) if all(cells[c]['metrics'].get(m) is not None for c in cells) else None for m in METRICS}
                pools[name][cohort]=dict(cells=cells,effects=effects,E_axis='LABEL',R_axis='RANK',interpretation='EVALUATION_PATH_DIAGNOSTIC_NOT_NEW_OFFICIAL_METHOD')
    assigns=sum(r.get('physical_scene_assign_calls',0) for r in rows)
    if assigns>104:raise ValueError('Rank off-diagonal budget exceeded')
    return write(root/'rank.json',dict(status='COMPLETE' if not unsupported else 'PARTIAL_DEPENDENCY',input_identity=key,scene_rows=rows,pools=pools,unsupported=unsupported,
                 physical_scene_assign_calls=assigns,new_assign_calls_this_invocation=sum(r.get('new_assign_calls_this_invocation',0) for r in rows),recovered_completed_scene_assign_calls=sum(r.get('recovered_completed_scene_assign_calls',0) for r in rows),physical_evaluate_matches_calls=3*assigns,matches_only_pool_calls=8 if not unsupported else sum(4 for v in pools.values()),diagonal_assign_calls=0,CPU_workers=workers,BLAS_threads=1))
