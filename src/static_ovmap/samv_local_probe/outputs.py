"""Actual row-wise fixed-surface partition builder without old whole-unit vetoes."""

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.scannet_study import native_ranks


def build(inputs,method,owners,class_updates,editable,prompts):
    base=inputs.g1;owners=np.asarray(owners,np.int64);editable=np.asarray(editable,bool)
    if owners.shape!=base.owner_ids.shape or editable.shape!=owners.shape or np.any(owners<0):raise ValueError('fixed source row shape required')
    if not np.array_equal(owners[~editable],base.owner_ids[~editable]):raise ValueError('ownership edit outside predeclared local domain')
    classes=owner_labels(base);updates={int(o):int(c) for o,c in class_updates.items()}
    if not set(map(int,np.unique(owners[owners>0])))<=set(classes):raise ValueError('new owner IDs forbidden')
    if not set(updates)<=set(classes) or any(c not in inputs.valid_ids for c in updates.values()):raise ValueError('class update outside original owner/vocabulary')
    classes.update(updates)
    for row,owner in prompts.items():
        if int(base.owner_ids[int(row)])!=int(owner) or int(owners[int(row)])!=int(owner):raise ValueError('original prompt ownership lost')
    semantics=np.zeros(owners.shape,np.int64)
    for owner in np.unique(owners[owners>0]):semantics[owners==owner]=classes[int(owner)]
    ranks=native_ranks(owners,classes,inputs.nearest,inputs.matched)
    payload=PredictionPayload(method,'COMBO',base.scene_id,base.geometry,owners,semantics,ranks,dict(base.logical_cost),
        {'parent_prediction_key':base.prediction_key,'method_recipe':method,'fixed_surface':True,'single_exclusive_partition':True,
         'row_wise_local_intervention':True,'official_current_class_ranks_recomputed':True,
         'owner_semantic_decisions':{str(o):classes[int(o)] for o in np.unique(owners[owners>0])}})
    payload.lock();moved=owners!=base.owner_ids;changed=semantics!=base.semantic_labels
    return payload,{'ownership_edits':int(moved.sum()),'semantic_row_edits':int(changed.sum()),
        'old_raw_zero_ownership_edits':int((moved&(inputs.raw==0)).sum()),
        'old_raw_zero_semantic_edits':int((changed&(inputs.raw==0)).sum()),
        'disappeared_owner_ids':sorted(set(map(int,np.unique(base.owner_ids[base.owner_ids>0])))-set(map(int,np.unique(owners[owners>0])))),
        'prompt_rows_retained':len(prompts),'geometry_identity_unchanged':payload.geometry==base.geometry,
        'class_updates':updates,'rank_changes':[{'owner':o,'before':dict(base.instance_ranks).get(o),'after':v} for o,v in ranks if dict(base.instance_ranks).get(o)!=v]}


def predict(binding):
    from pathlib import Path
    import time
    from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
    from .binding import load_scene
    from .common import METHODS,_array_digest,canonical_digest,verified,write
    from .runner import require_freeze
    require_freeze(binding);root=Path(binding['output_root']);scenes={}
    for scene in binding['scenes']:
        begin=time.perf_counter()
        inputs=load_scene(binding,scene);plan=verified(root/'query_plan'/scene/'receipt.json')
        structural={m:verified(root/'lifted'/scene/m/'receipt.json') for m in ('SAM2','SAMV')}
        decisions=verified(root/'readout'/scene/'decisions.json')
        key=canonical_digest({'plan':plan['identity'],'structural':{m:r['identity'] for m,r in structural.items()},
            'paired_readout':decisions['identity'],'builder':inputs.index.identity(__file__)})
        dest=root/'predictions'/scene/'receipt.json'
        if dest.exists():
            prior=verified(dest)
            if prior['input_identity']!=key:raise ValueError('prediction inputs changed; invalidate only affected outputs/scorers')
            for row in prior['methods'].values():
                payload=load_prediction(row['manifest'])
                if payload.prediction_key!=row['prediction_key']:raise ValueError('locked prediction alias changed')
            scenes[scene]=prior['identity'];continue
        with np.load(plan['domain']['path'],allow_pickle=False) as arrays:editable=arrays['editable']
        owners={}
        for model,row in structural.items():
            inputs.index.identity(row['partition']['path'],row['partition'])
            with np.load(row['partition']['path'],allow_pickle=False) as arrays:owners[model]=arrays['owners']
        updates={name:{int(o):r[name+'_class'] for o,r in decisions['targets'].items()} for name in ('OLD','NEW')}
        prompts={int(r):int(o) for r,o in plan['prompt_owners'].items()};results={};known=[]
        for method in (*METHODS,'REF_D2'):
            if method in ('SV00_G1','REF_D2'):
                original='IR01_G1' if method=='SV00_G1' else 'IR00_D2'
                payload=inputs.g1 if method=='SV00_G1' else inputs.d2
                item=binding['scenes'][scene]['parent_predictions'][original]
                row={'status':'PREDICTION_LOCKED','manifest':item['manifest'],
                    'prediction_key':payload.prediction_key,'record_key':payload.record_key,
                    'source_kind':'EXACT_PUBLISHED_PARENT_BASELINE','audit':None}
            else:
                partition=owners['SAM2'] if method=='SV01_SAM2_GEOM' else owners['SAMV'] if method in ('SV02_SAMV_GEOM','SV05_COMBINED') else inputs.g1.owner_ids
                labels=updates['OLD'] if method=='SV03_OLDMASK_FC' else updates['NEW'] if method in ('SV04_SAMV_FC','SV05_COMBINED') else {}
                payload,audit=build(inputs,method,partition,labels,editable,prompts)
                alias=next((r for p,r in known if p.geometry==payload.geometry and p.instance_ranks==payload.instance_ranks
                    and np.array_equal(p.owner_ids,payload.owner_ids) and np.array_equal(p.semantic_labels,payload.semantic_labels)),None)
                if alias:
                    row={**alias,'source_kind':'EXACT_LOCAL_ARRAYS_CLASSES_RANKS_ALIAS','audit':audit,
                        'alias_proof':{'owners':_array_digest(payload.owner_ids),'semantics':_array_digest(payload.semantic_labels),
                            'ranks':canonical_digest(payload.instance_ranks)}}
                else:
                    saved=save_prediction(payload,root/'predictions'/scene/method)
                    row={'status':'PREDICTION_LOCKED','manifest':str(saved),'prediction_key':payload.prediction_key,
                        'record_key':payload.record_key,'source_kind':'ACTUAL_LOCAL_PARTITION_OR_PAIRED_LABELS','audit':audit}
            row.update(owner_partition_digest=_array_digest(payload.owner_ids),semantic_digest=_array_digest(payload.semantic_labels),
                rank_digest=canonical_digest(payload.instance_ranks));results[method]=row;known.append((payload,row))
        equality={a+'='+b:results[a]['prediction_key']==results[b]['prediction_key']
            for i,a in enumerate(METHODS) for b in METHODS[i+1:]}
        if results['SV02_SAMV_GEOM']['owner_partition_digest']!=results['SV05_COMBINED']['owner_partition_digest']:
            raise ValueError('SV05 must copy exact SV02 partition')
        receipt=write(dest,{'status':'PREDICTIONS_LOCKED','scene':scene,'input_identity':key,
            'methods':results,'equality_flags':equality,'exact_combination_target_class_map':updates['NEW'],
            'SV05_owners_exactly_SV02':True,'SV05_class_map_exactly_SV04':True,
            'one_exclusive_partition_for_AP_and_mIoU':True,'GT_used':False,'new_owner_ids':0,
            'elapsed_seconds':time.perf_counter()-begin})
        scenes[scene]=receipt['identity'];inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
        print('PREDICTIONS_LOCKED',scene,'7 conditions',flush=True)
    return write(root/'predictions/summary.json',{'status':'ALL_PREDICTIONS_LOCKED','scenes':scenes,
        'logical_rows':7*len(scenes),'all_locked_before_new_annotations':True})
