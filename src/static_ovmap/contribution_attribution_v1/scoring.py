"""Dynamic official scoring preserves the actual released context and ranks."""
from concurrent.futures import ProcessPoolExecutor
import contextlib
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator,official_view
from static_ovmap.cvpr_compact.evaluation import equivalent_scoring_input,fraction_metrics,released_pool_with_classes,trace_class_metrics
from static_ovmap.cvpr_compact.protocol import validate_overlaps
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.preservation_routing.scorer_compat import adapt
from static_ovmap.released_loader import load_released_module
from static_ovmap.recovery_wave2.evaluation import scorer_context
from .common import verified,read,read_document,write,ConsumptionIndex,PathResolver,unit_identity,cached_record,canonical_digest,_array_digest
from .binding import load_binding

def evaluator_for(binding,record,payload,folder,index):
    data=record['data'];config=PathResolver(binding['path_map']).rewrite(read_document(data['config'],index,sealed=False))
    native=read_document(data['sources']['N']['path'],index)
    owners=owner_labels(payload)
    registry={'objects':{str(o):native['objects'].get(str(o),dict(available=False,scores=None,reason='ACTUAL_OUTPUT_MASK_REGISTRY_ONLY')) for o in owners}}
    evidence=SimpleNamespace(config=config,scene=record['scene'],dataset=data['dataset'],native=payload,sources={'N0':registry})
    evaluator=SceneEvaluator(evidence,folder,index=index);adapt(evaluator.namespace);validate_overlaps(evaluator.namespace['overlaps'])
    if evaluator.minimum!=100 or set(evaluator.masks)!=set(owners):raise ValueError('Actual positive owner registry/minimum differs')
    return evaluator

def _reference_rows(binding,inventory,scene,role,index):
    if role=='REGRESSION':
        return [r for r in inventory['historical_rows']+inventory['FG_rows'] if r['scene']==scene]
    path=Path(binding['backbone_root'])/'readouts'/scene/'BB00_NATIVE/evaluation_rows.json'
    return PathResolver(binding['path_map']).rewrite(read_document(path,index,sealed=False))['rows']

def _reference_manifest(binding,inventory,row,role,index):
    scene=row['scene']
    if role=='REGRESSION' and row['method'].startswith('CT_'):
        return read_document(Path(binding['area_root'])/'predictions'/scene/'receipt.json',index)['predictions'][row['method']]
    if role=='REGRESSION':
        return PathResolver(binding['path_map']).rewrite(read_document(Path(binding['fg_parent_root'])/'predictions'/scene/'receipt.json',index))['methods'][row['method']]['manifest']
    return read_document(Path(binding['backbone_root'])/'readouts'/scene/'BB00_NATIVE/receipt.json',index)['predictions'][row['method']]

def _check_score(evaluator,payload,score):
    labels=owner_labels(payload);view={str(o):v for o,v in official_view(evaluator.owners,labels,evaluator.minimum).items()}
    ranks=dict(payload.instance_ranks)
    if any(f'{ranks[o]:.6f}'!=view.get(str(o),dict(rank='0.000000'))['rank'] for o in labels):raise ValueError('Payload native current-class rank differs')
    if score['view']!=view or scorer_context(score['context'])!=scorer_context(evaluator.context):raise ValueError('Actual original scorer input context/view differs')
    matrix=np.asarray(score['confusion'],np.int64);valid=int(np.isin(evaluator.targets['gt_semantic'],evaluator.ids).sum())
    if matrix.shape!=(len(evaluator.ids)+1,)*2 or int(matrix.sum())!=valid:raise ValueError('Whole semantic target denominator differs')
    return view,valid

def score_scene(job):
    binding,role,scene=job;root=Path(binding['output_root']);inventory=verified(root/'inventory.json')
    predict_phase='fusion_dev_predict' if role=='FUSION_DEV' else 'main_predict';phase='fusion_dev_score' if role=='FUSION_DEV' else 'main_score'
    global_lock=verified(root/(predict_phase+'.json'))
    if global_lock['status'] not in ('ALL_PREDICTIONS_LOCKED','PARTIAL_DEPENDENCY'):raise ValueError('Predictions must globally lock before labels')
    folder=root/phase/scene;index=ConsumptionIndex(folder/'verifications.json')
    locked=[r for r in global_lock['records'] if r['scene']==scene]
    key=unit_identity(phase+'_scene',dict(prediction_lock=global_lock['identity'],records=locked),producers=('common','scoring'),scene=scene,role=role)
    old=cached_record(folder/'receipt.json',key)
    if old:return old
    start=time.perf_counter();evaluators={};rows=[];references=_reference_rows(binding,inventory,scene,role,index);scored={};assigns=0;matches_calls=0;engineering=[]
    for record in locked:
        payload=load_prediction(record['manifest'])
        if payload.prediction_key!=record['prediction_key']:raise ValueError('Locked payload changed')
        partition=_array_digest(payload.owner_ids)
        if partition not in evaluators:evaluators[partition]=evaluator_for(binding,record,payload,folder/'partitions'/partition,index)
        evaluator=evaluators[partition];labels=owner_labels(payload);view={str(o):v for o,v in official_view(evaluator.owners,labels,evaluator.minimum).items()}
        row=None;proof=None
        for reference_row in references:
            if reference_row['rank_mode']!='OFFICIAL_CURRENT_CLASS' or reference_row.get('prediction_identity')!=payload.prediction_key:continue
            oldscore=read_document(reference_row['evaluation_receipt'],index,sealed=False)
            oldpayload=load_prediction(_reference_manifest(binding,inventory,reference_row,role,index))
            if equivalent_scoring_input(payload,oldpayload,evaluator.context,oldscore['context'],view,oldscore['view']):
                row=dict(reference_row,reuse_kind='EXACT_PARENT_ARRAYS_RANKS_SCORER_VIEW')
                proof=dict(owner_partition_digest=partition,semantic_digest=_array_digest(payload.semantic_labels),rank_digest=canonical_digest(payload.instance_ranks),scorer_context=canonical_digest(scorer_context(evaluator.context)))
                break
        if row is None and payload.prediction_key in scored:
            row=dict(scored[payload.prediction_key],reuse_kind='EXACT_LOCAL_ARRAYS_RANKS_SCORER_VIEW')
        if row is None:
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/(record['method']+'.log')).open('a') as log,contextlib.redirect_stdout(log):
                row=evaluator.evaluate(labels,record['method'],'OFFICIAL_CURRENT_CLASS',payload.prediction_key)
            assigns+=1;matches_calls+=3;row=dict(row,reuse_kind='RELEASED_SCENE_ASSIGN_TRACE_AND_SCORE_EXECUTED')
        score=read_document(row['evaluation_receipt'],index,sealed=False);actual_view,valid=_check_score(evaluator,payload,score)
        classes=trace_class_metrics(score,index=index)
        row=write(folder/'rows'/(record['method']+'.json'),dict(row,status='COMPLETE',method=record['method'],scene=scene,role=role,metrics=fraction_metrics(score['metrics']),metric_unit='FRACTION',alias_proof=proof,per_class=classes,positive_owner_count=len(labels),valid_GT_points=valid,owner_partition_digest=partition,semantic_digest=_array_digest(payload.semantic_labels),rank_digest=canonical_digest(payload.instance_ranks),prediction_identity=payload.prediction_key,record_identity=payload.record_key,geometry_identity=payload.geometry.to_dict(),actual_context=score['context'],measurement_kind='REUSED_EXACT' if row['reuse_kind'].startswith('EXACT') else 'NEW_CACHED_ANALYSIS'))
        rows.append(row);scored[payload.prediction_key]=row
    if role=='REGRESSION' and scene in (binding['cohorts']['replica8'][0],binding['cohorts']['scannet_cf18'][0]):
        original=next(r for r in inventory['historical_rows'] if r['scene']==scene and r['method']=='CT_A1_E')
        path=_reference_manifest(binding,inventory,original,role,index);payload=load_prediction(path)
        context_record=next(r for r in locked if not r['method'].startswith('J'))
        e=evaluator_for(binding,context_record,payload,folder/'engineering_A1_once',index)
        with (folder/'engineering_A1.log').open('a') as log,contextlib.redirect_stdout(log):check=e.evaluate(owner_labels(payload),'ENGINEERING_A1','OFFICIAL_CURRENT_CLASS',payload.prediction_key)
        _check_score(e,payload,read(check['evaluation_receipt']))
        for metric,value in original['metrics'].items():
            if metric not in ('apall','ap50','ap25','miou','macc'):continue
            actual=check['metrics'][metric]
            if (value is None)!=(actual is None) or value is not None and abs(value-actual)>1e-12:raise ValueError('Engineering original A1 scorer metric parity failed')
        engineering.append(dict(scene=scene,reference=original['evaluation_identity'],check=check['evaluation_identity'],all_five_metrics_parity=True));assigns+=1;matches_calls+=3
    result=write(folder/'receipt.json',dict(status='COMPLETE',input_identity=key,scene=scene,role=role,rows=rows,physical_scene_assign_calls=assigns,physical_evaluate_matches_calls=matches_calls,engineering_checks=engineering,predictions_locked_before_GT=global_lock['identity'],wall_seconds=time.perf_counter()-start))
    index.write_memo(folder/'verifications.json');print('SCORED',role,scene,len(rows),flush=True);return result

def score(root,role):
    root=Path(root);binding=load_binding(root);p='fusion_dev_predict' if role=='FUSION_DEV' else 'main_predict';phase='fusion_dev_score' if role=='FUSION_DEV' else 'main_score'
    lock=verified(root/(p+'.json'))
    if lock['status']=='NOT_SELECTED_NO_FUSION_DEV':return write(root/(phase+'.json'),dict(status='NOT_SELECTED_NO_FUSION_DEV',role=role,reason=lock['reason'],rows=[],pools={},physical_scene_assign_calls=0))
    key=unit_identity(phase,dict(prediction_lock=lock['identity']),producers=('common','scoring'),role=role)
    old=cached_record(root/(phase+'.json'),key)
    if old:return old
    start=time.perf_counter();scenes=[s for s in lock['scenes'] if any(r['scene']==s for r in lock['records'])]
    with ProcessPoolExecutor(max_workers=3) as executor:receipts=list(executor.map(score_scene,[(binding,role,s) for s in scenes]))
    rows=[r for receipt in receipts for r in receipt['rows']];cohorts={'fusion_dev':binding['task_specification']['fusion_dev']} if role=='FUSION_DEV' else binding['cohorts']
    pools={};index=ConsumptionIndex(root/phase/'pool_verifications.json');pool_calls=pool_assigns=0
    for cohort,ordered in cohorts.items():
        pools[cohort]={}
        for method in lock['methods']:
            selected=[r for r in rows if r['scene'] in ordered and r['method']==method]
            if {r['scene'] for r in selected}!=set(ordered):continue
            first=read(selected[0]['evaluation_receipt']);namespace=load_released_module(first['context']['evaluator']['path']);namespace['init']('Replica' if first['context']['dataset']=='Replica' else 'Scannet200');adapt(namespace);validate_overlaps(namespace['overlaps'])
            identities=[next(r['evaluation_identity'] for r in selected if r['scene']==s) for s in ordered]
            alias=next((v for v in pools[cohort].values() if v['ordered_inputs']==identities),None)
            folder=root/phase/'pools'/cohort;folder.mkdir(parents=True,exist_ok=True)
            if alias:pooled=dict(alias,method=method,reuse_kind='EXACT_ORDERED_SCORER_INPUT_ALIAS')
            else:
                with (folder/(method+'.log')).open('a') as log,contextlib.redirect_stdout(log):pooled,classes=released_pool_with_classes(folder,selected,namespace,ordered,method,'OFFICIAL_CURRENT_CLASS',index)
                pooled=dict(pooled,per_class_receipt=str(folder/(method+'_per_class.json')),reuse_kind='RELEASED_ORDERED_POOL_EXECUTED');pool_calls+=1;pool_assigns+=len(ordered)
            pools[cohort][method]=write(folder/(method+'.json'),dict(pooled,metrics=fraction_metrics(pooled['metrics']),ordered_scenes=ordered,metric_unit='FRACTION'))
    expected=len(lock['methods'])*len(cohorts)
    result=write(root/(phase+'.json'),dict(status='COMPLETE' if sum(map(len,pools.values()))==expected else 'PARTIAL_DEPENDENCY',input_identity=key,role=role,rows=rows,pools=pools,methods=lock['methods'],logical_scene_rows=len(rows),logical_pools=sum(map(len,pools.values())),physical_scene_assign_calls=sum(r['physical_scene_assign_calls'] for r in receipts),physical_scene_evaluate_matches_calls=sum(r['physical_evaluate_matches_calls'] for r in receipts),physical_ordered_pool_calls=pool_calls,nested_pool_assign_calls=pool_assigns,nested_pool_evaluate_matches_calls=pool_calls,engineering_scene_scores=sum(len(r['engineering_checks']) for r in receipts),CPU_workers=3,BLAS_threads=1,wall_seconds=time.perf_counter()-start))
    index.write_memo(root/phase/'pool_verifications.json');return result
