"""Fixed 26-scene G1 applications with original eligibility, N/Q and ranks."""
from concurrent.futures import ProcessPoolExecutor
import contextlib
from pathlib import Path
import time
import numpy as np
from .common import (verified,write,arrays_record,unit_key,existing,immutable_record,
    ConsumptionIndex,PathResolver,read_document,canonical_digest)

def method_name(slot,readout):
    return slot['method']+('_seed'+str(slot['seed']) if slot['seed'] is not None else '')+'__'+readout

def predict(binding,fc,budget):
    import torch
    from .data import SparseInputs
    from .registry import load_registry
    from .grouping import method_inputs
    from static_ovmap.frozen_fc_routing.binding import load_scene
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);predictors=verified(root/'predictor_lock.json')
    prepared=verified(parent/'regression/manifest.json');features=verified(parent/'features/regression.json')
    identity=unit_key('map_predict',dict(predictors=predictors['identity'],inputs=prepared['identity'],features=features['identity']),producers=('mapping','registry','models','grouping','data'))
    path=root/'map/prediction_lock.json';old=existing(path,identity)
    if old:return old
    slots,heads,aliases=load_registry(binding,fc,map_only=True);readouts=binding['task_specification']['map_regression']['readouts']
    methods=[method_name(s,r) for s in slots for r in readouts]
    if len(slots)!=11 or len(methods)!=22:raise ValueError('Fixed map predictor registry differs')
    loader=SparseInputs(fc,features);locks={};start=time.perf_counter();resolver=PathResolver(binding['path_map']);engineering=[]
    for cohort,scenes in binding['cohorts'].items():
        for scene in scenes:
            source=verified(parent/'regression'/scene/'manifest.json');scene_identity=unit_key('map_scene',dict(parent=identity,objects=source['identity']),producers=('mapping',),scene=scene)
            folder=root/'map/predictions'/scene;rp=folder/'receipt.json';prior=existing(rp,scene_identity)
            if prior:
                locks[scene]=prior['identity']
                engineering.extend(prior['engineering_parity'])
                continue
            context=resolver.rewrite(verified(binding['scenes'][scene]['context']['path']));fc.index.identity(context['FC_text']['path'],context['FC_text'])
            if context['FC_physical_model_identity']!=fc.model_key:raise ValueError('Map FC identity differs')
            with np.load(context['FC_text']['path'],allow_pickle=False) as a:text=a['text_embeddings'].copy();ids=a['valid_ids'].copy()
            if len(ids)!=(51 if cohort=='replica8' else 200):raise ValueError('Scene-specific text vocabulary differs')
            records={};relabels={m:{} for m in methods};p,original=load_scene(binding,scene)
            if not np.array_equal(ids,np.asarray(p.valid_ids)):raise ValueError('Map source/text order differs')
            with torch.no_grad():
                for obj in resolver.rewrite(source['objects']):
                    x,_,audit=loader.load(obj,0,8);embeddings=[];cosines=[];group_counts=[]
                    for slot in slots:
                        inputs,_=method_inputs(slot['method'],x,obj['key'],'clean');budget.consume('map_head',persist=False)
                        result=heads[slot['unique_model']](inputs,fc.phi);z=result['embedding']
                        if z is None:raise ValueError('Inherited eligible map object has no valid views')
                        vector=z.cpu().numpy();score=text@vector
                        if not np.isfinite(score).all():raise ValueError('Nonfinite real-map F scores')
                        embeddings.append(vector);cosines.append(score);group_counts.append(result['group_count'])
                        source_scores={n:np.asarray(p.sources[n]['objects'][str(obj['owner'])]['scores'],np.float64) if p.sources[n]['objects'][str(obj['owner'])]['available'] else None for n in ('N','Q','F')}
                        source_scores['F']=score
                        for readout in readouts:
                            if readout=='F_ONLY_FIXED_ELIGIBLE':label=int(ids[int(score.argmax())])
                            else:
                                base=pool(source_scores,p.temperatures,('N','Q','F'));final=simple_dependence(source_scores,p.temperatures,base,'PE_D2_GROUPED')
                                if final is None or not np.isfinite(final).all():raise ValueError('Original D2 fusion undefined')
                                label=int(ids[int(final.argmax())])
                            relabels[method_name(slot,readout)][obj['owner']]=label
                    array=arrays_record(root/'map/readouts'/scene/(str(obj['owner'])+'.npz'),dict(embeddings=np.asarray(embeddings,np.float32),cosines=np.asarray(cosines),valid_ids=ids,slot_ids=np.asarray([s['id'] for s in slots])),fc.index)
                    records[obj['key']]=dict(arrays=array,inputs=obj['inputs'],views=obj['views'],ordinary_pooling=audit,group_counts=group_counts)
            budget.flush();saved={};known={};BFC_payloads={}
            for slot in slots:
                for readout in readouts:
                    method=method_name(slot,readout);payload=build_payload(original.g1,original.d2,original.raw,relabels[method],source['eligible_owners'],original.nearest,original.matched,method,{},dict(fixed_geometry='G1',source_predictors=predictors['identity'],source_manifest=source['identity'],readout=readout))
                    key=payload.prediction_key
                    if key in known:item={**known[key], 'alias_of':known[key]['method'],'alias_basis':'EXACT_OWNER_SEMANTIC_AND_CURRENT_RANK_ARRAYS'}
                    else:
                        manifest=save_prediction(payload,folder/method);item=dict(status='PREDICTION_LOCKED',manifest=str(manifest),prediction_key=key,record_key=payload.record_key,method=method);known[key]=item
                    saved[method]=dict(item,method=method)
                    if slot['method']=='B_FC':BFC_payloads[readout]=payload
            # Exactly one inherited FC8 payload check in each cohort's first scene.
            scene_parity=[]
            if scene==scenes[0]:
                inherited=resolver.rewrite(verified(parent/'predictions'/scene/'receipt.json'))['methods']['LR04_FC_8'];old_payload=load_prediction(inherited['manifest']);new=BFC_payloads['PARENT_D2_FIXED_ELIGIBLE']
                if not(np.array_equal(new.owner_ids,old_payload.owner_ids) and np.array_equal(new.semantic_labels,old_payload.semantic_labels) and new.instance_ranks==old_payload.instance_ranks):raise ValueError('Old B_FC map parity failed: '+scene)
                scene_parity.append(dict(scene=scene,cohort=cohort,source_manifest=inherited['manifest'],bitwise_owner_semantics_ranks=True))
                engineering.extend(scene_parity)
            value=immutable_record(rp,dict(status='COMPLETE',input_identity=scene_identity,scene=scene,cohort=cohort,methods=saved,readouts=readouts,slots=slots,records=records,engineering_parity=scene_parity,eligible_owners=source['eligible_owners'],original_eligibility=source['identity'],owner_partition_unchanged=True,noneligible_recovery_background_unchanged=True,GT_used=False))
            locks[scene]=value['identity'];p.index.write_memo(root/'inputs'/scene/'verifications.json');print('MAP_PREDICT',scene,len(source['objects']),22,flush=True)
    fc.index.write_memo(root/'features/verifications.json')
    if len(engineering)!=2:raise ValueError('Exactly two inherited B_FC parity records required')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,scenes=locks,methods=methods,cohorts=binding['cohorts'],logical_scene_rows=572,slots=slots,source_grid_encodings=fc.images,engineering_parity=engineering,elapsed_seconds=time.perf_counter()-start,all_26_predictions_before_score=True))

def score_scene(job):
    from static_ovmap.frozen_fc_routing.binding import load_scene
    from static_ovmap.frozen_fc_routing.evaluation import partition_evaluator
    from static_ovmap.module_validation.scannet_study import load_prediction
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.cvpr_compact.evaluation import fraction_metrics,trace_class_metrics
    from static_ovmap.module_validation.native_capture import _array_digest
    binding,scene=job;root=Path(binding['output_root']);global_lock=verified(root/'map/prediction_lock.json');lock=verified(root/'map/predictions'/scene/'receipt.json')
    if global_lock['logical_scene_rows']!=572 or len(global_lock['scenes'])!=26:raise ValueError('All required map predictions must precede scoring')
    identity=unit_key('map_score_scene',dict(global_predictions=global_lock['identity'],scene=lock['identity']),producers=('mapping',),scene=scene)
    folder=root/'map/evaluation'/scene;rp=folder/'receipt.json';old=existing(rp,identity)
    if old:return old
    p,inputs=load_scene(binding,scene);rows=[];scored={};evaluator=None
    for method,item in lock['methods'].items():
        payload=load_prediction(item['manifest'])
        if payload.prediction_key!=item['prediction_key'] or not np.array_equal(payload.owner_ids,inputs.g1.owner_ids):raise ValueError('Fixed map payload changed')
        if evaluator is None:evaluator=partition_evaluator(inputs,payload,binding,folder)
        labels=owner_labels(payload)
        if payload.prediction_key in scored:row={**scored[payload.prediction_key], 'reuse_kind':'EXACT_LOCAL_SCORER_INPUT_ALIAS'}
        else:
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/(method+'.log')).open('w') as log,contextlib.redirect_stdout(log):row=evaluator.evaluate(labels,method,'OFFICIAL_CURRENT_CLASS',payload.prediction_key)
        score=read_document(row['evaluation_receipt'],sealed=False);confusion=np.asarray(score['confusion'],np.int64)
        if int(confusion.sum())!=int(np.isin(evaluator.targets['gt_semantic'],evaluator.ids).sum()):raise ValueError('Map confusion omitted valid GT')
        result=write(folder/'rows'/(method+'.json'),dict(row,scene=scene,method=method,status='COMPLETE',cohort=binding['scenes'][scene]['cohort'],metrics=fraction_metrics(score['metrics']),per_class=trace_class_metrics(score,index=inputs.index),owner_partition_digest=_array_digest(payload.owner_ids),record_identity=payload.record_key,prediction_identity=payload.prediction_key,metric_unit='FRACTION'))
        rows.append(result);scored[payload.prediction_key]=result
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    return write(rp,dict(status='COMPLETE',input_identity=identity,scene=scene,rows=rows,row_count=len(rows)))

def score(binding):
    from static_ovmap.released_loader import load_released_module
    from static_ovmap.preservation_routing.scorer_compat import adapt
    from static_ovmap.cvpr_compact.evaluation import released_pool_with_classes,fraction_metrics
    root=Path(binding['output_root']);lock=verified(root/'map/prediction_lock.json');scenes=[s for names in binding['cohorts'].values() for s in names]
    identity=unit_key('map_score',lock['identity'],producers=('mapping',));path=root/'map/score.json';old=existing(path,identity)
    if old:return old
    with ProcessPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(score_scene,[(binding,s) for s in scenes]))
    rows=[row for r in receipts for row in r['rows']];index=ConsumptionIndex(root/'map/pool_verifications.json');pools={}
    for cohort,names in binding['cohorts'].items():
        reference=read_document(binding['baseline_rows']['REF_G1'][names[0]]['evaluation_receipt'],sealed=False);desc=reference['context']['evaluator'];index.identity(desc['path'],desc)
        namespace=load_released_module(desc['path']);namespace['init']('Replica' if cohort=='replica8' else 'Scannet200');adapt(namespace);pools[cohort]={}
        for method in lock['methods']:
            ordered=[next(r for r in rows if r['scene']==s and r['method']==method) for s in names];folder=root/'map/pools'/cohort;folder.mkdir(parents=True,exist_ok=True)
            with (folder/(method+'.log')).open('w') as log,contextlib.redirect_stdout(log):pooled,classes=released_pool_with_classes(folder,ordered,namespace,names,method,'OFFICIAL_CURRENT_CLASS',index)
            pools[cohort][method]=write(folder/(method+'.json'),dict(pooled,per_class_identity=classes['identity'],per_class_receipt=str(folder/(method+'_per_class.json')),metrics=fraction_metrics(pooled['metrics']),metric_unit='FRACTION',ordered_scenes=names))
    if len(rows)!=572 or sum(len(v) for v in pools.values())!=44:raise ValueError('Map572 rows/44 pools incomplete')
    index.write_memo(root/'map/pool_verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,rows=rows,pools=pools,scene_method_coverage=572,full_cohort_pool_coverage=44,CPU_workers=3,BLAS_threads=1,scope='EXPOSED_FIXED_MAP_REGRESSION'))

def run(binding,*,phase):
    if phase=='map_score':return score(binding)
    raise RuntimeError('Map prediction requires the GPU worker')
