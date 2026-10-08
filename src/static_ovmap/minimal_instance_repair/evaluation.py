"""Released evaluation of the actual partition, with original target binding."""

from concurrent.futures import ProcessPoolExecutor
import contextlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.evaluation import (equivalent_scoring_input, fraction_metrics,
    released_pool_with_classes, trace_class_metrics)
from static_ovmap.cvpr_compact.protocol import validate_overlaps
from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, official_view
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.evaluation import scorer_context
from static_ovmap.released_loader import load_released_module

from .binding import load_scene, seal


def partition_evaluator(inputs, payload, binding, root):
    if not payload.locked:
        raise ValueError('actual partition must be locked before targets are loaded')
    config = PathResolver({**binding['reference'].get('path_map',{}),**binding['path_map']}).rewrite(read(inputs.data['config']))
    owners = sorted(map(int,np.unique(payload.owner_ids[payload.owner_ids>0])))
    # Administrative entries enumerate masks. They never create semantic evidence.
    source = {'objects':{str(o):inputs.sources['N']['objects'].get(str(o),
        {'owner_id':o,'available':False,'scores':None,'label':None,
         'reason':'ACTUAL_OUTPUT_MASK_REGISTRY_ONLY'}) for o in owners}}
    evidence = SimpleNamespace(scene=inputs.scene,dataset=inputs.data['dataset'],
                               config=config,native=payload,sources={'N0':source})
    return SceneEvaluator(evidence,Path(root)/'partitions'/_array_digest(payload.owner_ids),index=inputs.index)


def _evaluate_scene(job, *, require_frozen=True):
    binding,scene = job
    if require_frozen:
        from .recognition import require_freeze
        require_freeze(binding)
    root = Path(binding['output_root'])
    lock = read(root/'predictions'/scene/'receipt.json')
    if lock['status']!='PREDICTIONS_LOCKED':
        raise ValueError('all actual scientific predictions must precede annotation access')
    inputs = load_scene(binding,scene)
    target = root/'evaluation'/scene/'receipt.json'
    key = canonical_digest({'prediction_lock':lock['identity'],'producer':inputs.index.identity(__file__),
        'released_adapter':inputs.index.identity(Path(__file__).parents[1]/'m2_reviewer_study/evaluation.py'),
        'base_scoring':{m:binding['baseline_rows'][m][scene]['evaluation_identity'] for m in ('IR00_D2','IR01_G1')}})
    if target.exists():
        old = read(target)
        if old['input_identity']!=key:
            from .resume import invalidate_descendants
            invalidate_descendants(binding,scene,'evaluate','changed actual scoring inputs')
        else:
            return old
    begin = time.perf_counter()
    evaluators,scored,rows,checks = {},{},[],{}
    for method in [r['id'] for r in binding['specification']['methods']]:
        item = lock['methods'][method]
        if item['status']!='PREDICTION_LOCKED':
            rows.append({'status':item['status'],'scene':scene,'method':method,'blocks':item.get('blocks',[])})
            continue
        payload = load_prediction(item['manifest'])
        if payload.prediction_key!=item['prediction_key'] or payload.geometry!=inputs.d2.geometry:
            raise ValueError('actual locked payload changed before released evaluation')
        partition = _array_digest(payload.owner_ids)
        labels = owner_labels(payload)
        if partition not in evaluators:
            evaluators[partition] = partition_evaluator(inputs,payload,binding,target.parent)
        evaluator = evaluators[partition]
        validate_overlaps(evaluator.namespace['overlaps'])
        if evaluator.minimum!=100 or set(evaluator.masks)!=set(labels):
            raise ValueError('actual partition registry or original instance minimum changed')
        view = {str(o):v for o,v in official_view(evaluator.owners,labels,evaluator.minimum).items()}
        ranks = dict(payload.instance_ranks)
        if set(ranks)!=set(labels) or any(f'{ranks[o]:.6f}'!=view.get(str(o),{'rank':'0.000000'})['rank'] for o in labels):
            raise ValueError('actual payload ranks differ from released current-class target-area ranks')
        row = None
        if method in ('IR00_D2','IR01_G1'):
            row = binding['baseline_rows'][method][scene]
        else:
            for baseline,reference in [('IR00_D2',inputs.d2),('IR01_G1',inputs.g1)]:
                imported = binding['baseline_rows'][baseline][scene]
                score = read(imported['evaluation_receipt'])
                if equivalent_scoring_input(payload,reference,evaluator.context,score['context'],view,score['view']):
                    row = {**imported,'reuse_kind':'EXACT_PARENT_ARRAYS_RANKS_AND_SCORER_CONTEXT',
                        'alias_proof':{'prediction_key':reference.prediction_key,
                            'owner_partition':partition,'semantic_digest':_array_digest(payload.semantic_labels),
                            'rank_digest':canonical_digest(payload.instance_ranks),
                            'scorer_context_digest':canonical_digest(scorer_context(evaluator.context))}}
                    break
        if row is None and payload.prediction_key in scored:
            row = {**scored[payload.prediction_key],'reuse_kind':'EXACT_LOCAL_ARRAYS_RANKS_AND_SCORER_CONTEXT'}
        if row is None:
            target.parent.mkdir(parents=True,exist_ok=True)
            with (target.parent/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log):
                row = evaluator.evaluate(labels,method,'OFFICIAL_CURRENT_CLASS',payload.prediction_key)
            row = {**row,'reuse_kind':'RELEASED_SCORER_EXECUTED'}
        score = read(row['evaluation_receipt'])
        if (score['status']!='COMPLETE' or score['view']!=view
                or scorer_context(score['context'])!=scorer_context(evaluator.context)):
            raise ValueError('scorer reuse does not match the actual arrays/ranks/GT protocol')
        matrix = np.asarray(score['confusion'],np.int64)
        valid_count = int(np.isin(evaluator.targets['gt_semantic'],evaluator.ids).sum())
        if matrix.shape!=(len(evaluator.ids)+1,)*2 or int(matrix.sum())!=valid_count:
            raise ValueError('whole-scene semantics omitted small/unknown/added-support target errors')
        classes = trace_class_metrics(score,index=inputs.index)
        row = seal({**row,'scene':scene,'method':method,'dataset':inputs.data['dataset'],
            'cohort':binding['scenes'][scene]['cohort'],'metrics':fraction_metrics(score['metrics']),
            'metric_unit':'FRACTION','record_identity':payload.record_key,
            'prediction_identity':payload.prediction_key,'owner_partition_digest':partition,
            'per_class':classes,'runtime_overlaps':score['context']['runtime_overlaps']})
        atomic_write_json(target.parent/'rows'/(method+'.json'),row)
        rows.append(row)
        scored[payload.prediction_key] = row
        checks[method] = {'partition_root':str(evaluator.root),'mask_registry_owners':sorted(evaluator.masks),
            'positive_output_owners':sorted(labels),'official_view':view,'semantic_valid_targets':valid_count,
            'semantic_confusion_count':int(matrix.sum()),'unknown_semantic_positive_owners':sorted(o for o,l in labels.items() if l==0)}
    result = seal({'status':'COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL_DEPENDENCY_BLOCK',
        'scene':scene,'input_identity':key,'prediction_lock':lock['identity'],'rows':rows,
        'registry_checks':checks,'row_count':sum(r['status']=='COMPLETE' for r in rows),
        'same_single_partition_for_instance_and_semantic':True,'elapsed_seconds':time.perf_counter()-begin})
    atomic_write_json(target,result)
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    print('EVALUATED',scene,result['row_count'],'/ 9',flush=True)
    return result


def evaluate_scene(job, *, require_frozen=True):
    binding,scene = job
    begin = time.perf_counter()
    invocation = {'scene':scene,'status':'RUNNING','seconds':None}
    try:
        result = _evaluate_scene(job,require_frozen=require_frozen)
        invocation.update(status=result['status'],receipt_identity=result['identity'])
        return result
    except BaseException as exc:
        invocation.update(status='FAILED',reason=type(exc).__name__+': '+str(exc))
        raise
    finally:
        invocation['seconds'] = time.perf_counter()-begin
        atomic_write_json(Path(binding['output_root'])/'evaluation'/scene/'invocations'/(str(time.time_ns())+'.json'),seal(invocation))


def evaluate_study(binding):
    root = Path(binding['output_root'])
    scenes = [s for names in binding['cohorts'].values() for s in names]
    with ProcessPoolExecutor(max_workers=binding['specification']['resources']['CPU_evaluation_workers']) as executor:
        receipts = list(executor.map(evaluate_scene,[(binding,s) for s in scenes]))
    rows = [r for receipt in receipts for r in receipt['rows'] if r['status']=='COMPLETE']
    pools = {}
    index = ConsumptionIndex(root/'evaluation/pool_verifications.json')
    for cohort,names in binding['cohorts'].items():
        pools[cohort] = {}
        score = read(binding['baseline_rows']['IR01_G1'][names[0]]['evaluation_receipt'])
        namespace = load_released_module(score['context']['evaluator']['path'])
        namespace['init']('Replica' if cohort=='replica8' else 'Scannet200')
        for method in [r['id'] for r in binding['specification']['methods']]:
            method_rows = [r for r in rows if r['scene'] in names and r['method']==method]
            if len(method_rows)!=len(names):
                continue
            if method in ('IR00_D2','IR01_G1'):
                pooled = binding['baseline_pools'][cohort][method]
            else:
                alias = None
                ordered = [next(r for r in method_rows if r['scene']==s)['evaluation_identity'] for s in names]
                for baseline,parent in pools[cohort].items():
                    if ordered==parent['ordered_inputs']:
                        alias = {**parent,'method':method,
                            'reuse_kind':'EXACT_ORDERED_PARENT_SCORING_ALIAS' if baseline in ('IR00_D2','IR01_G1')
                                else 'EXACT_ORDERED_LOCAL_SCORING_ALIAS',
                            'alias_source_method':baseline}
                        break
                if alias is not None:
                    pooled = alias
                else:
                    output = root/'pools'/cohort
                    output.mkdir(parents=True,exist_ok=True)
                    with (output/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log):
                        pooled,classes = released_pool_with_classes(output,method_rows,namespace,names,method,
                                                                    'OFFICIAL_CURRENT_CLASS',index)
                    pooled = {**pooled,'per_class_receipt':str(output/(method+'_per_class.json')),
                              'per_class_identity':classes['identity']}
            original_identity = pooled.get('source_pool_identity',pooled['identity'])
            if 'per_class_receipt' not in pooled:
                source = Path(pooled['source_receipt'])
                candidates = [source.with_name(source.stem+'_per_class.json'),source.with_name(pooled['source_method']+'_per_class.json'),
                    source.parent/(pooled['source_method']+'_per_class.json')]
                classes_path = next((p for p in candidates if p.is_file()),None)
                if classes_path is None:
                    raise FileNotFoundError('bound parent pool must retain its actual per-class receipt: '+str(source))
                classes = read(classes_path)
                if classes['identity']!=pooled['per_class_identity']:
                    raise ValueError('parent per-class pool identity changed')
                pooled = {**pooled,'per_class_receipt':str(classes_path)}
            pooled = seal({**pooled,'source_pool_identity':original_identity,'method':method,'metrics':fraction_metrics(pooled['metrics']),
                           'metric_unit':'FRACTION'})
            atomic_write_json(root/'pools'/cohort/(method+'.json'),pooled)
            pools[cohort][method] = pooled
    result = seal({'status':'SCIENCE_COMPLETE' if len(rows)==234 and sum(map(len,pools.values()))==18 else 'PARTIAL_DEPENDENCY_BLOCK',
        'binding_identity':binding['identity'],'cohorts':binding['cohorts'],'scene_method_coverage':len(rows),
        'full_cohort_pool_coverage':sum(map(len,pools.values())),'scene_metrics':rows,'pooled_metrics':pools,
        'scene_receipts':{r['scene']:r['identity'] for r in receipts},'deployment':'N0_UNCHANGED',
        'all_cohorts_previously_exposed':True,'metric_unit':'FRACTION',
        'fixed_surface':True,'single_exclusive_partition':True,'new_maps':0,'new_segmentation_inference':0})
    existing = root/'result_store.json'
    if existing.exists() and read(existing).get('science_identity')==result['identity']:
        result = read(existing)
    else:
        atomic_write_json(existing,result)
    index.write_memo(root/'evaluation/pool_verifications.json')
    return result
