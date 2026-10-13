"""Dynamic coverage using the unchanged official partition/pooling kernels."""
from concurrent.futures import ProcessPoolExecutor
import contextlib
from pathlib import Path
import time
import numpy as np
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.evaluation import equivalent_scoring_input,fraction_metrics,released_pool_with_classes,trace_class_metrics
from static_ovmap.cvpr_compact.protocol import validate_overlaps
from static_ovmap.m2_reviewer_study.evaluation import official_view
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.evaluation import scorer_context
from static_ovmap.released_loader import load_released_module
from static_ovmap.frozen_fc_routing.evaluation import partition_evaluator,required_coverage
from static_ovmap.frozen_fc_routing.binding import load_scene
from .common import REFERENCES, verified,read,write,PathResolver,ConsumptionIndex,canonical_digest,_array_digest,require,unit_identity,cached_unit

OLD={'REF_D2':'REF_D2','REF_G1':'REF_G1','FC8':'FG0_FC8','FG2_seed17':'FG2_SURFACE','FG2_seed29':'S29_FG2_SURFACE'}

def parent_rows(binding):
    path=Path(binding['fg_parent_root'])/'evaluation/store.json'; require([path])
    return PathResolver(binding['path_map']).rewrite(verified(path))

def evaluate_scene(job):
    binding,scene=job; root=Path(binding['output_root']); folder=root/'evaluation'/scene
    global_lock=verified(root/'predictions/summary.json'); methods=global_lock['methods']; lock=verified(root/'predictions'/scene/'receipt.json')
    if global_lock['status']!='ALL_PREDICTIONS_LOCKED' or lock['identity']!=global_lock['scenes'][scene] or set(lock['methods'])!=set(methods): raise ValueError('Global all-arm prediction lock required before GT')
    identity=unit_identity(binding,'evaluate_scene',lock=lock['identity'],global_lock=global_lock['identity']); path=folder/'receipt.json'; previous=cached_unit(path,identity)
    if previous: return dict(previous,scorer_calls_this_invocation=0)
    p,inputs=load_scene(binding,scene); inherited=parent_rows(binding); old_rows={r['method']:r for r in inherited['scene_metrics'] if r['scene']==scene}; scored={}; evaluators={}; rows=[]; checks=[]; start=time.perf_counter()
    for method in methods:
        item=lock['methods'][method]; payload=load_prediction(item['manifest']); labels=owner_labels(payload); partition=_array_digest(payload.owner_ids)
        if payload.prediction_key!=item['prediction_key'] or payload.geometry!=inputs.g1.geometry: raise ValueError('Locked map payload changed')
        if method!='REF_D2' and partition!=_array_digest(inputs.g1.owner_ids): raise ValueError('Readout changed G1 owner geometry')
        if partition not in evaluators: evaluators[partition]=partition_evaluator(inputs,payload,binding,folder)
        evaluator=evaluators[partition]; validate_overlaps(evaluator.namespace['overlaps'])
        if evaluator.minimum!=100 or set(evaluator.masks)!=set(labels): raise ValueError('Official positive registry changed')
        view={str(o):v for o,v in official_view(evaluator.owners,labels,100).items()}; ranks=dict(payload.instance_ranks)
        if set(ranks)!=set(labels) or any(f'{ranks[o]:.6f}'!=view.get(str(o),{'rank':'0.000000'})['rank'] for o in labels): raise ValueError('Official current-class area ranks changed')
        row=None; proof=None
        for reference in REFERENCES:
            old=old_rows[OLD[reference]]; descriptor=lock['methods'][reference]; reference_payload=load_prediction(descriptor['manifest']); receipt=read(old['evaluation_receipt'])
            if equivalent_scoring_input(payload,reference_payload,evaluator.context,receipt['context'],view,receipt['view']):
                row=dict(old,reuse_kind='EXACT_PARENT_ARRAYS_RANKS_SCORER_CONTEXT'); proof=dict(owner_partition=partition,semantics=_array_digest(payload.semantic_labels),ranks=canonical_digest(payload.instance_ranks),scorer_context=canonical_digest(scorer_context(evaluator.context))); break
        if method in REFERENCES and row is None: raise ValueError('Parent reference scorer input cannot be proven identical')
        # Exactly two additional actual baseline checks, in the formal scoring run.
        cohort=binding['scenes'][scene]['cohort']
        if method=='FG2_seed17' and scene==binding['cohorts'][cohort][0]:
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'baseline_check.log').open('a') as log,contextlib.redirect_stdout(log): check=evaluator.evaluate(labels,'ENGINEERING_FG2_SEED17_CHECK','OFFICIAL_CURRENT_CLASS',payload.prediction_key)
            actual=read(check['evaluation_receipt']); historical=read(old_rows[OLD[method]]['evaluation_receipt'])
            if actual['view']!=historical['view'] or actual['confusion']!=historical['confusion'] or scorer_context(actual['context'])!=scorer_context(historical['context']): raise ValueError('Actual baseline scorer parity differs')
            for metric,value in fraction_metrics(actual['metrics']).items():
                if value is not None and abs(value-fraction_metrics(historical['metrics'])[metric])>1e-10: raise ValueError('Historical baseline metrics differ')
            checks.append(dict(scene=scene,method=method,actual_check=check,engineering_only=True))
        if row is None and payload.prediction_key in scored: row=dict(scored[payload.prediction_key],reuse_kind='EXACT_LOCAL_ARRAYS_RANKS_SCORER_CONTEXT')
        if row is None:
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log): row=evaluator.evaluate(labels,method,'OFFICIAL_CURRENT_CLASS',payload.prediction_key)
            row=dict(row,reuse_kind='RELEASED_SCORER_EXECUTED')
        score=read(row['evaluation_receipt'])
        if score['status']!='COMPLETE' or score['view']!=view or scorer_context(score['context'])!=scorer_context(evaluator.context): raise ValueError('Actual scorer input identity differs')
        confusion=np.asarray(score['confusion'],np.int64); valid=int(np.isin(evaluator.targets['gt_semantic'],evaluator.ids).sum())
        if confusion.shape!=(len(evaluator.ids)+1,)*2 or int(confusion.sum())!=valid: raise ValueError('Whole-output confusion support differs')
        row=write(folder/'rows'/(method+'.json'),dict(row,scene=scene,method=method,cohort=cohort,dataset=inputs.data['dataset'],metrics=fraction_metrics(score['metrics']),metric_unit='FRACTION',record_identity=payload.record_key,prediction_identity=payload.prediction_key,owner_partition_digest=partition,per_class=trace_class_metrics(score,index=inputs.index),runtime_overlaps=score['context']['runtime_overlaps'],alias_proof=proof))
        scored[payload.prediction_key]=row; rows.append(row)
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    calls=sum(r['reuse_kind']=='RELEASED_SCORER_EXECUTED' for r in rows)+len(checks)
    return write(path,dict(status='COMPLETE',input_identity=identity,scene=scene,rows=rows,baseline_checks=checks,elapsed_seconds=time.perf_counter()-start,GT_only_after_global_lock=global_lock['identity'],scorer_calls_this_invocation=calls))

def run(binding):
    root=Path(binding['output_root']); require([root/'activation.json',root/'predictions/summary.json']); activation=verified(root/'activation.json'); lock=verified(root/'predictions/summary.json')
    if not activation['expanded']: return write(root/'evaluation/store.json',dict(status='NOT_ACTIVATED_BY_FIXED_RULE',methods=[],scene_metrics=[],pooled_metrics={},new_logical_records=0,new_pool_count=0,baseline_checks=0))
    scenes=[s for names in binding['cohorts'].values() for s in names]; methods=lock['methods']; start=time.perf_counter()
    if set(lock['scenes'])!=set(scenes) or lock['status']!='ALL_PREDICTIONS_LOCKED' or len(lock['new_methods'])>4 or len(scenes)!=26: raise ValueError('Complete dynamic all-arm prediction coverage required')
    with ProcessPoolExecutor(max_workers=3) as pool: receipts=list(pool.map(evaluate_scene,[(binding,s) for s in scenes]))
    rows=[r for receipt in receipts for r in receipt['rows']]; inherited=parent_rows(binding); index=ConsumptionIndex(root/'evaluation/pool_verifications.json'); pools={}
    for cohort,names in binding['cohorts'].items():
        first=next(r for r in rows if r['scene']==names[0] and r['method']=='REF_G1'); score=read(first['evaluation_receipt']); namespace=load_released_module(score['context']['evaluator']['path']); namespace['init']('Replica' if cohort=='replica8' else 'Scannet200'); validate_overlaps(namespace['overlaps']); pools[cohort]={}
        for method in methods:
            ordered=[next(r for r in rows if r['scene']==s and r['method']==method) for s in names]; identities=[r['evaluation_identity'] for r in ordered]
            if method in REFERENCES:
                old=inherited['pooled_metrics'][cohort][OLD[method]]
                if old['ordered_scenes']!=names or old['ordered_inputs']!=identities: raise ValueError('Parent official ordered pool identity differs')
                pooled=dict(old,reuse_kind='EXACT_VERIFIED_PARENT_ORDERED_POOL')
            else:
                alias=next((v for v in pools[cohort].values() if v['ordered_inputs']==identities),None); folder=root/'pools'/cohort; folder.mkdir(parents=True,exist_ok=True)
                if alias: pooled=dict(alias,reuse_kind='EXACT_ORDERED_SCORER_INPUT_ALIAS')
                else:
                    with (folder/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log): pooled,classes=released_pool_with_classes(folder,ordered,namespace,names,method,'OFFICIAL_CURRENT_CLASS',index)
                    pooled=dict(pooled,per_class_receipt=str(folder/(method+'_per_class.json')),per_class_identity=classes['identity'],reuse_kind='RELEASED_ORDERED_POOL_EXECUTED')
            metrics=pooled['metrics'] if pooled.get('metric_unit')=='FRACTION' else fraction_metrics(pooled['metrics'])
            pools[cohort][method]=write(root/'pools'/cohort/(method+'.json'),dict(pooled,method=method,metrics=metrics,metric_unit='FRACTION',ordered_scenes=names,ordered_inputs=identities))
    count,pool_count=required_coverage(binding['cohorts'],methods,rows,pools); checks=[c for r in receipts for c in r['baseline_checks']]
    if len(checks)!=2: raise ValueError('Exactly2 actual baseline scorer checks required')
    gains={}
    for method in activation['expanded']:
        cells=[]
        for cohort in binding['cohorts']:
            for seed in (17,29):
                new=pools[cohort][method+f'_seed{seed}']['metrics']; old=pools[cohort][f'FG2_seed{seed}']['metrics']; cells.append(dict(cohort=cohort,seed=seed,delta_APall=new['apall']-old['apall'],delta_mIoU=new['miou']-old['miou']))
        gains[method]=dict(cells=cells,MAP_GAIN_VS_FG2=all(c['delta_APall']>1e-10 and c['delta_mIoU']>=-.005-1e-10 for c in cells))
    m1='M1_CORRESPONDENCE_GRAPH'; star=activation['C_STAR']; advantage=False
    if m1 in gains and star in gains:
        advantage=all(pools[c][m1+f'_seed{s}']['metrics']['apall']-pools[c][star+f'_seed{s}']['metrics']['apall']>1e-10 and pools[c][m1+f'_seed{s}']['metrics']['miou']-pools[c][star+f'_seed{s}']['metrics']['miou']>=-.005-1e-10 for c in binding['cohorts'] for s in (17,29))
    classes=verified(root/'class_split.json'); split_metrics={}; family_deltas=[]
    for cohort in binding['cohorts']:
        split_metrics[cohort]={}
        for method in methods:
            if cohort=='replica8': split_metrics[cohort][method]='NOT_APPLICABLE'; continue
            details=verified(pools[cohort][method]['per_class_receipt']); subset={}
            for role,ids in (('base',classes['base_ids']),('novel',classes['heldout_ids'])):
                selected=[r for r in details['classes'] if r['class_id'] in ids]
                subset[role]={}
                for metric in ('apall','ap50','ap25','iou','accuracy'):
                    valid=[r[metric] for r in selected if r[metric] is not None]
                    subset[role][metric]=float(np.mean(valid)) if valid else None
                subset[role]['defined_class_counts']={k:sum(r[k] is not None for r in selected) for k in ('apall','ap50','ap25','iou','accuracy')}
            split_metrics[cohort][method]=dict(full200_competition=True,subsets=subset,undefined_classes_not_zero_filled=True)
        for method,arm in lock['arms'].items():
            reference='FG2_seed'+str(arm['seed']); families={}
            for scene in binding['cohorts'][cohort]:
                family=scene.split('_')[0] if cohort=='scannet_cf18' else scene
                new=next(r for r in rows if r['scene']==scene and r['method']==method)['metrics']; old=next(r for r in rows if r['scene']==scene and r['method']==reference)['metrics']
                families.setdefault(family,[]).append({k:new[k]-old[k] if new[k] is not None and old[k] is not None else None for k in new})
            for family,changes in families.items(): family_deltas.append(dict(cohort=cohort,family=family,method=method,descriptive_only_not_official_AP=True,deltas={k:float(np.mean([r[k] for r in changes if r[k] is not None])) if any(r[k] is not None for r in changes) else None for k in changes[0]}))
    elapsed=time.perf_counter()-start
    write(root/'resources/evaluate.json',dict(phase='evaluate',status='COMPLETE',boundary='RELEASED_SCORER_AND_OFFICIAL_POOLS_CPU',wall_seconds=elapsed,scorer_calls_this_invocation=sum(r['scorer_calls_this_invocation'] for r in receipts),CPU_workers=3,new_rgb_encodings=0,new_optimizer_updates=0,new_sam_proposals=0,online_FPS=None))
    index.write_memo(root/'evaluation/pool_verifications.json')
    return write(root/'evaluation/store.json',dict(status='EVALUATION_COMPLETE',cohorts=binding['cohorts'],methods=methods,scene_metrics=rows,pooled_metrics=pools,scene_method_coverage=count,full_cohort_pool_coverage=pool_count,new_logical_records=len(lock['new_methods'])*len(scenes),new_pool_count=len(lock['new_methods'])*len(binding['cohorts']),reference_rows=len(REFERENCES)*len(scenes),reference_pool_count=len(REFERENCES)*len(binding['cohorts']),baseline_checks=checks,map_gains=gains,STRUCTURAL_ADVANTAGE_REGRESSION=advantage,elapsed_seconds=elapsed,CPU_workers=3,metric_unit='FRACTION',base_novel=split_metrics,family_deltas=family_deltas,Replica_base_novel='NOT_APPLICABLE',deployment='N0_UNCHANGED'))
