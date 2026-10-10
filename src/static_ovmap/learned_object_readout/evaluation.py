"""Full globally locked outputs scored by the original released evaluator."""
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
from static_ovmap.source_preserving_update.evaluation import partition_evaluator

from .binding import load_scene
from .common import BASELINES,ConsumptionIndex,canonical_digest,read,verified,write,_array_digest


def required_coverage(cohorts,methods,rows,pools):
    scenes=[s for names in cohorts.values() for s in names]
    expected={(s,m) for s in scenes for m in methods}
    actual={(r['scene'],r['method']) for r in rows if r['status']=='COMPLETE'}
    if actual!=expected or len(rows)!=len(expected): raise ValueError('Incomplete/duplicate scene-method coverage')
    table={(r['scene'],r['method']):r for r in rows}
    for cohort,names in cohorts.items():
        if set(pools[cohort])!=set(methods): raise ValueError('Incomplete ordered pool coverage')
        for method in methods:
            pool=pools[cohort][method]
            inputs=[table[(s,method)]['evaluation_identity'] for s in names]
            if pool['ordered_scenes']!=names or pool['ordered_inputs']!=inputs:
                raise ValueError('Pool replaced exact ordered scenes/scorer inputs')
    return len(expected),len(cohorts)*len(methods)


def evaluate_scene(job):
    binding,scene,methods=job;root=Path(binding['output_root']);folder=root/'evaluation'/scene
    global_lock=verified(root/'predictions/summary.json')
    if global_lock['status']!='ALL_PREDICTIONS_LOCKED' or global_lock['logical_records']!=286:
        raise ValueError('All main/repeat outputs must lock before target labels')
    lock=verified(root/'predictions'/scene/'receipt.json')
    if lock['identity']!=global_lock['scenes'][scene] or set(lock['methods'])!=set(methods):
        raise ValueError('Global/scene prediction lock differs')
    key=canonical_digest(dict(lock=lock['identity'],producer=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
                   baseline_contexts={m:binding['baseline_rows'][s][scene]['evaluation_identity'] for m,s in BASELINES.items()}))
    path=folder/'receipt.json'
    if path.exists():
        previous=verified(path)
        if previous['input_identity']!=key: raise ValueError('Completed evaluation inputs changed')
        return previous
    started=time.perf_counter();p,inputs=load_scene(binding,scene);evaluators={};scored={};rows=[];checks={}
    for method in methods:
        item=lock['methods'][method];payload=load_prediction(item['manifest'])
        if payload.prediction_key!=item['prediction_key'] or payload.geometry!=inputs.g1.geometry:
            raise ValueError('Locked actual source payload changed')
        partition=_array_digest(payload.owner_ids);labels=owner_labels(payload)
        if method not in BASELINES and partition!=_array_digest(inputs.g1.owner_ids):
            raise ValueError('Readout changed fixed G1 owner geometry')
        if partition not in evaluators: evaluators[partition]=partition_evaluator(inputs,payload,binding,folder)
        evaluator=evaluators[partition];validate_overlaps(evaluator.namespace['overlaps'])
        if evaluator.minimum!=100 or set(evaluator.masks)!=set(labels): raise ValueError('Actual positive registry/minimum changed')
        view={str(o):v for o,v in official_view(evaluator.owners,labels,100).items()}
        ranks=dict(payload.instance_ranks)
        if set(ranks)!=set(labels) or any(f'{ranks[o]:.6f}'!=view.get(str(o),{'rank':'0.000000'})['rank'] for o in labels):
            raise ValueError('Official current-class area ranks changed')
        row=None;proof=None
        for source,reference in (('SU00_D2',inputs.d2),('SU01_G1',inputs.g1)):
            parent=binding['baseline_rows'][source][scene];score=read(parent['evaluation_receipt'])
            if equivalent_scoring_input(payload,reference,evaluator.context,score['context'],view,score['view']):
                row={**parent,'reuse_kind':'EXACT_PARENT_ARRAYS_RANKS_AND_SCORER_CONTEXT'}
                proof=dict(owner_partition=partition,semantic_digest=_array_digest(payload.semantic_labels),
                           rank_digest=canonical_digest(payload.instance_ranks),scorer_context=canonical_digest(scorer_context(evaluator.context)))
                break
        if row is None and payload.prediction_key in scored:
            row={**scored[payload.prediction_key],'reuse_kind':'EXACT_LOCAL_ARRAYS_RANKS_AND_SCORER_CONTEXT'}
        if row is None:
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log):
                row=evaluator.evaluate(labels,method,'OFFICIAL_CURRENT_CLASS',payload.prediction_key)
            row={**row,'reuse_kind':'RELEASED_SCORER_EXECUTED'}
        score=read(row['evaluation_receipt'])
        if score['status']!='COMPLETE' or score['view']!=view or scorer_context(score['context'])!=scorer_context(evaluator.context):
            raise ValueError('Actual scorer input identity differs')
        confusion=np.asarray(score['confusion'],np.int64);valid=int(np.isin(evaluator.targets['gt_semantic'],evaluator.ids).sum())
        if confusion.shape!=(len(evaluator.ids)+1,)*2 or int(confusion.sum())!=valid:
            raise ValueError('Whole-output semantic confusion excludes source support')
        row=write(folder/'rows'/(method+'.json'),{**row,'scene':scene,'method':method,
                 'cohort':binding['scenes'][scene]['cohort'],'dataset':inputs.data['dataset'],
                 'metrics':fraction_metrics(score['metrics']),'metric_unit':'FRACTION',
                 'record_identity':payload.record_key,'prediction_identity':payload.prediction_key,
                 'owner_partition_digest':partition,'per_class':trace_class_metrics(score,index=inputs.index),
                 'runtime_overlaps':score['context']['runtime_overlaps'],'alias_proof':proof})
        scored[payload.prediction_key]=row;rows.append(row)
        checks[method]=dict(actual_positive_registry=sorted(labels),scorer_masks=sorted(evaluator.masks),
                           confusion_targets=valid,confusion_count=int(confusion.sum()),all_current_ranks_verified=True)
    result=write(path,dict(status='COMPLETE',scene=scene,input_identity=key,rows=rows,row_count=len(rows),registry_checks=checks,
                  GT_only_after_global_lock=global_lock['identity'],elapsed_seconds=time.perf_counter()-started))
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    print('EVALUATED_LEARNED',scene,len(rows),flush=True)
    return result


def evaluate(binding):
    root=Path(binding['output_root']);lock=verified(root/'predictions/summary.json');methods=lock['methods']
    scenes=[s for names in binding['cohorts'].values() for s in names];started=time.perf_counter()
    if set(lock['scenes'])!=set(scenes) or len(methods)!=11: raise ValueError('Required full main/repeat output coverage absent')
    with ProcessPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(evaluate_scene,[(binding,s,methods) for s in scenes]))
    rows=[r for receipt in receipts for r in receipt['rows']]
    index=ConsumptionIndex(root/'evaluation/pool_verifications.json');pools={}
    for cohort,names in binding['cohorts'].items():
        score=read(binding['baseline_rows']['SU01_G1'][names[0]]['evaluation_receipt'])
        namespace=load_released_module(score['context']['evaluator']['path'])
        namespace['init']('Replica' if cohort=='replica8' else 'Scannet200');validate_overlaps(namespace['overlaps'])
        pools[cohort]={}
        for method in methods:
            ordered=[next(r for r in rows if r['scene']==s and r['method']==method) for s in names]
            identities=[r['evaluation_identity'] for r in ordered]
            alias=next((v for v in pools[cohort].values() if v['ordered_inputs']==identities),None)
            folder=root/'pools'/cohort;folder.mkdir(parents=True,exist_ok=True)
            if alias: pooled={**alias,'method':method,'reuse_kind':'EXACT_ORDERED_SCORER_INPUT_ALIAS'}
            else:
                with (folder/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log):
                    pooled,classes=released_pool_with_classes(folder,ordered,namespace,names,method,'OFFICIAL_CURRENT_CLASS',index)
                pooled={**pooled,'per_class_receipt':str(folder/(method+'_per_class.json')),
                        'per_class_identity':classes['identity'],'reuse_kind':'RELEASED_ORDERED_POOL_EXECUTED'}
            pools[cohort][method]=write(folder/(method+'.json'),{**pooled,'metrics':fraction_metrics(pooled['metrics']),
                                      'metric_unit':'FRACTION','ordered_scenes':names})
    row_count,pool_count=required_coverage(binding['cohorts'],methods,rows,pools)
    if (row_count,pool_count)!=(286,22): raise ValueError('Required286/22 scientific coverage absent')
    result=write(root/'evaluation/store.json',dict(status='EVALUATION_COMPLETE',cohorts=binding['cohorts'],methods=methods,
                 scene_metrics=rows,pooled_metrics=pools,scene_method_coverage=row_count,full_cohort_pool_coverage=pool_count,
                 elapsed_seconds=time.perf_counter()-started,CPU_workers=3,scene_receipts={r['scene']:r['identity'] for r in receipts},
                 deployment='N0_UNCHANGED',metric_unit='FRACTION'))
    index.write_memo(root/'evaluation/pool_verifications.json')
    return result
