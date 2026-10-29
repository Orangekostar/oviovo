"""Legacy26 fixed-eligible intervention; never exported as GLOBAL MF×AC."""
from pathlib import Path
import time
import numpy as np
from .common import (verified,write,canonical_digest,digest_file,save_arrays,SEEDS,
                     source_identity,Resolver,StageFailure)


def numerical_binding(binding):
    if not binding['inherited']:
        raise StageFailure('Actual legacy input lineage is absent')
    old=dict(binding['inherited'])
    old.update(output_root=str(Path(binding['output_root'])/'legacy'),path_map=binding['path_map'])
    return old


def cached_readout(root,obj,source,seed,weight,inputs,model):
    key=canonical_digest(dict(object=obj['key'],inputs=source['arrays']['sha256'],
        weight=weight['source_sha256'],prefix=8,condition='clean'))
    path=Path(root)/'readout_cache'/(key+'.npz');receipt=path.with_suffix('.json')
    if receipt.exists():
        row=verified(receipt)
        if row['input_identity']!=key or digest_file(path)!=row['arrays']['sha256']:
            raise ValueError('Saved real readout input/output identity differs')
        with np.load(path,allow_pickle=False) as arrays:b=arrays['b'];z=arrays['z']
        return b,z,row,False
    result=model(inputs,None)
    if result['embedding'] is None or result['base'] is None:
        raise ValueError('Legacy eligible object unexpectedly has no valid F input')
    b=result['base'].detach().cpu().numpy();z=result['embedding'].detach().cpu().numpy()
    if not np.isfinite(b).all() or not np.isfinite(z).all():
        raise ValueError('Nonfinite supported MV output is a runtime error')
    save_arrays(path,b=b,z=z)
    row=write(receipt,dict(status='DONE',input_identity=key,object_key=obj['key'],seed=seed,
        weight_sha256=weight['source_sha256'],source_inputs=source,
        arrays=dict(path=str(path),bytes=path.stat().st_size,sha256=digest_file(path)),
        actual_head_calls=1,actual_internal_anchor_evaluations=1,
        created_by='LEGACY_MATCHED1000_FORWARD'))
    return b,z,row,True


def predict(binding,models,*,replica_base_mask=None):
    import torch
    from .region_inputs import CachedSparse
    from static_ovmap.frozen_fc_routing.binding import load_scene
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    from static_ovmap.semantic_update_evidence_v1.calibration import angle_cap
    root=Path(binding['output_root']);old=numerical_binding(binding)
    parent=Path(old['lr_parent_root']);struct=Path(binding['structure_root'])
    lock=verified(root/'model_lock.json');resolver=Resolver(binding['path_map'])
    if not lock['region_only_enabled']:
        raise ValueError('This run requires the actual successful region-only parity record')
    features=verified(parent/'features/regression.json');loader=CachedSparse(binding,features)
    classes=verified(Path(binding['semantic_root'])/'A/import.json')['class_split']
    cohorts=old['cohorts'];all_methods=[f'{variant}_seed{seed}' for seed in SEEDS
        for variant in ('L00','L10','L01','L11')]
    locks={};missing=[];engineering=[];head_calls=0;cache_hits=0;start=time.monotonic()
    producer=source_identity(('legacy','region_inputs'))
    for cohort,scenes in cohorts.items():
        for scene in scenes:
            scene_path=parent/'regression'/scene/'manifest.json'
            source=verified(scene_path);loader.index.identity(scene_path)
            ctx_path=resolver.resolve(old['scenes'][scene]['context']['path'])
            context=resolver.rewrite(verified(ctx_path));loader.index.identity(ctx_path)
            loader.index.identity(context['FC_text']['path'],context['FC_text'])
            with np.load(context['FC_text']['path'],allow_pickle=False) as arrays:
                text=arrays['text_embeddings'].copy();ids=arrays['valid_ids'].copy()
            base_mask=np.isin(ids,classes['base_ids']) if cohort=='scannet_cf18' else replica_base_mask
            if base_mask is not None and np.asarray(base_mask).shape!=(len(ids),):
                raise ValueError('Legacy base-role mask has a different vocabulary axis')
            variants=('L00','L10','L01','L11') if base_mask is not None else ('L00','L10')
            methods=[f'{variant}_seed{seed}' for seed in SEEDS for variant in variants]
            identity=canonical_digest(dict(producer=producer,models=lock['weights'],scene=source['identity'],
                text=context['FC_text']['sha256'],variants=variants,base_mask=base_mask))
            folder=root/'legacy/predictions'/scene;receipt=folder/'receipt.json'
            if receipt.exists():
                previous=verified(receipt)
                if previous['input_identity']==identity:
                    locks[scene]=previous;engineering.extend(previous['engineering_parity'])
                    print('LEGACY_REUSED',scene,len(methods),flush=True);continue
                raise ValueError('Completed legacy scene producer/scope changed; preserve it before affected repair')
            p,original=load_scene(old,scene)
            if not np.array_equal(ids,np.asarray(p.valid_ids)):
                raise ValueError('Legacy source/text valid_ids disagree')
            relabels={method:{} for method in methods};records={};checks=[]
            with torch.no_grad():
                for oi,obj in enumerate(source['objects']):
                    inputs,input_record=loader.load(obj,region=True)
                    row_outputs={};shared_b=None
                    for seed in SEEDS:
                        b,z,readout,new=cached_readout(root,obj,input_record,seed,lock['weights'][str(seed)],inputs,models[seed])
                        head_calls+=int(new);cache_hits+=int(not new)
                        if shared_b is None:shared_b=b
                        elif not np.array_equal(b,shared_b):
                            raise ValueError('Same-input anchor changed between MV seeds')
                        raw=text@z
                        calibrated=None
                        if base_mask is not None:
                            cap,_=angle_cap(b[None],z[None],10.)
                            calibrated=text@cap[0];calibrated[np.asarray(base_mask,bool)]-=.025
                        source_scores={name:np.asarray(p.sources[name]['objects'][str(obj['owner'])]['scores'],np.float64)
                            if p.sources[name]['objects'][str(obj['owner'])]['available'] else None for name in ('N','Q')}
                        for variant in variants:
                            f=calibrated if variant in ('L01','L11') else raw
                            if variant in ('L00','L01'):
                                label=int(ids[int(np.argmax(f))])
                            else:
                                values=dict(source_scores,F=f)
                                base=pool(values,p.temperatures,('N','Q','F'))
                                distribution=simple_dependence(values,p.temperatures,base,'PE_D2_GROUPED')
                                if distribution is None or not np.isfinite(distribution).all():
                                    raise ValueError('Legacy supported D2 output is nonfinite')
                                label=int(ids[int(np.argmax(distribution))])
                            relabels[f'{variant}_seed{seed}'][obj['owner']]=label
                        row_outputs[str(seed)]=dict(readout=readout['identity'],arrays=readout['arrays'])
                    if oi==0 and scene==scenes[0]:
                        baseline=struct/'map/readouts'/scene/(str(obj['owner'])+'.npz')
                        loader.index.identity(baseline)
                        with np.load(baseline,allow_pickle=False) as arrays:
                            positions=np.flatnonzero(arrays['slot_ids']=='selected:B_FC')
                            if len(positions)!=1:raise ValueError('Inherited static B_FC slot is ambiguous')
                            reference_b=arrays['embeddings'][positions[0]]
                            reference_s=arrays['cosines'][positions[0]]
                            if not np.array_equal(ids,arrays['valid_ids']):raise ValueError('Old B_FC vocabulary changed')
                        b_error=float(np.max(np.abs(shared_b-reference_b)))
                        s_error=float(np.max(np.abs(text@shared_b-reference_s)))
                        if b_error>1e-6 or s_error>1e-6:
                            raise ValueError('Legacy actual B_FC same-input parity failed')
                        checks.append(dict(scene=scene,cohort=cohort,object_key=obj['key'],
                            actual_same_anchor_max_error=b_error,actual_original_score_max_error=s_error,
                            original_text_dtype=str(text.dtype),original_BFC_arrays=str(baseline),
                            owner_domain_fixed=True,GT_semantic_labels_used=False))
                    records[obj['key']]=dict(owner=obj['owner'],input=input_record,readouts=row_outputs)
            saved={};known={}
            for method in methods:
                payload=build_payload(original.g1,original.d2,original.raw,relabels[method],
                    source['eligible_owners'],original.nearest,original.matched,method,{},
                    dict(scope='FIXED_ELIGIBLE_READOUT_INTERVENTION',source_manifest=source['identity'],
                         model_lock=lock['identity'],variant=method.split('_seed')[0],profile='matched1000'))
                if payload.prediction_key in known:
                    item=dict(known[payload.prediction_key],alias_of=known[payload.prediction_key]['method'],
                        alias_basis='EXACT_MASK_LABEL_RANK_ARRAYS')
                else:
                    manifest=save_prediction(payload,folder/method)
                    item=dict(status='PREDICTION_LOCKED',manifest=str(manifest),
                        prediction_key=payload.prediction_key,record_key=payload.record_key,method=method)
                    known[payload.prediction_key]=item
                saved[method]=dict(item,method=method)
            value=write(receipt,dict(status='DONE',input_identity=identity,scene=scene,cohort=cohort,
                methods=saved,objects=records,original_eligible_owners=source['eligible_owners'],
                eligible_objects=len(source['objects']),all_positive_G1_owners=len(source['full_G1_positive_registry']),
                scope='FIXED_ELIGIBLE_READOUT_INTERVENTION',full_owner_global_ablation=False,
                engineering_parity=checks,owner_partition_unchanged=True,
                noneligible_recovered_background_unchanged=True,GT_used=False))
            locks[scene]=value;engineering.extend(checks)
            loader.flush()
            p.index.write_memo(root/'legacy/inputs'/scene/'verifications.json')
            print('LEGACY_PREDICTED',scene,len(source['objects']),len(saved),'new_heads',head_calls,flush=True)
    loader.flush()
    missing=[dict(scene=scene,method=method,reason='UNRESOLVED_REPLICA_TRAINING_BASE_SEMANTIC_MAPPING')
             for scene,row in locks.items() for method in all_methods if method not in row['methods']]
    return write(root/'legacy/prediction_lock.json',dict(status='DONE' if not missing else 'BLOCKED_DEPENDENCY',
        scenes={scene:row['identity'] for scene,row in locks.items()},cohorts=cohorts,methods=all_methods,
        actual_methods_by_scene={scene:list(row['methods']) for scene,row in locks.items()},
        expected_logical_scene_rows=sum(map(len,cohorts.values()))*len(all_methods),
        completed_logical_scene_rows=sum(len(row['methods']) for row in locks.values()),remaining_jobs=missing,
        engineering_parity=engineering,new_head_calls_this_invocation=head_calls,
        exact_readout_cache_hits_this_invocation=cache_hits,new_RGB_encodings=0,
        all_available_predictions_frozen_before_new_GT_scoring=True,
        elapsed_seconds=time.monotonic()-start,scope='FIXED_ELIGIBLE_READOUT_INTERVENTION'))
