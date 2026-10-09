"""Whole-output homogeneous relabels with unchanged source support and new ranks."""
import numpy as np
from pathlib import Path
import time

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.scannet_study import native_ranks,project_values
from static_ovmap.module_validation.scannet_study import load_prediction,save_prediction
from static_ovmap.m2_reviewer_study.evaluation import official_view

from .binding import load_scene
from .common import BASELINES,_array_digest,arrays_record,canonical_digest,plain,producer,unchanged_receipt,verified,write
from .updates import Evidence,update


def build_payload(g1,d2,raw,relabels,eligible,nearest,matched,method,costs,metadata=None):
    eligible=set(map(int,eligible));sem=g1.semantic_labels.copy();old=owner_labels(g1)
    for owner,label in relabels.items():
        owner=int(owner);mask=g1.owner_ids==owner
        if owner not in eligible or owner not in old or not np.any(d2.owner_ids==owner):raise ValueError('class replacement outside selected original incumbents')
        if np.any(np.asarray(raw)[mask]==0):raise ValueError('raw-zero support cannot be relabeled')
        sem[mask]=int(label)
    allowed=np.isin(g1.owner_ids,list(eligible))
    if not np.array_equal(sem[~allowed],g1.semantic_labels[~allowed]):raise ValueError('unselected class changed')
    original=d2.owner_ids>0
    if not np.array_equal(sem[~original],g1.semantic_labels[~original]):raise ValueError('recovered or unknown class changed')
    labels={o:int(sem[np.flatnonzero(g1.owner_ids==o)[0]]) for o in old}
    ranks=native_ranks(g1.owner_ids,labels,nearest,matched)
    view=official_view(project_values(g1.owner_ids,nearest,matched),labels,100)
    if any(f'{r:.6f}'!=view.get(o,{'rank':'0.000000'})['rank'] for o,r in ranks):raise ValueError('current official ranks differ')
    out=PredictionPayload(method,'COMBO',g1.scene_id,g1.geometry,g1.owner_ids,sem,ranks,costs,
        {'source_record_key':g1.record_key,**(metadata or {})});out.lock()
    if not np.array_equal(out.owner_ids,g1.owner_ids):raise ValueError('owner partition changed')
    return out


def predict_scene(binding,stage,scene,plan,choices,anchor,second,paths):
    root=Path(binding['output_root']);dest=root/'predictions'/stage/scene;p,e=load_scene(binding,scene)
    key=canonical_digest({'plan':plan['identity'],'choices':choices['identity'],'anchor':anchor['identity'],
        'second':second['identity'],'paths':paths,'producer':producer(p.index,'outputs.py','updates.py')})
    old=unchanged_receipt(dest/'receipt.json',key)
    if old:
        for item in old['methods'].values():
            if load_prediction(item['manifest']).prediction_key!=item['prediction_key']:raise ValueError('saved output alias differs')
        return old
    begin=time.perf_counter();records={**anchor['records'][scene],**second['records'][scene]};known={};results={};costs={}
    ordered=list(plan['selected_owner_ids']);eligible=set(ordered);n=len(ordered);k=len(p.valid_ids)
    for method,path in paths.items():
        if method in BASELINES:
            payload=e.d2 if method=='DQ00_D2' else e.g1
            inherited=binding['scenes'][scene]['parent_predictions']['IR00_D2' if method=='DQ00_D2' else 'IR01_G1']
            item={'status':'PREDICTION_LOCKED','manifest':inherited['manifest'],'prediction_key':payload.prediction_key,
                'record_key':payload.record_key,'source_kind':'EXACT_PARENT_BASELINE'}
            results[method]=item;known[payload.prediction_key]=item
            costs[method]={'logical_FULL_reads':0,'logical_probe_heads':0,'selected':n,'query_eligible':0,'updated':0};continue
        policy,kind=path;decision_rows={};relabels={};full_reads=0;probes=0;full_success=0
        p0=np.zeros((n,k),np.float64);pfinal=np.zeros_like(p0);pnew=np.full_like(p0,np.nan);available_new=np.zeros(n,bool)
        view_probabilities=np.full((n,2,k),np.nan,np.float64)
        for j,owner in enumerate(ordered):
            row=plan['owners'][str(owner)];choice=choices['owners'][str(owner)][policy]
            prior=np.asarray(row['p0'],np.float64);p0[j]=prior
            with np.load(row['support']['path'],allow_pickle=False) as arr:
                area=arr['area'];frame_ids=list(map(int,arr['frame_ids']));footprints=arr['O']
            selected=[]
            if row['query_eligible']:
                required=(choice['anchor_region_id'],choice['second_region_id']);full_reads+=2
                if policy=='DISAGREEMENT':probes+=len(row['probe_region_ids'])
                for rid in required:
                    if rid not in records:raise ValueError('required sealed FULL observation absent')
                    observed=records[rid]
                    if observed['valid_ids']!=p.valid_ids:raise ValueError('acquired vocabulary/order differs')
                    selected.append(Evidence(observed['content_key'],np.asarray(observed['scores'],np.float64) if observed['available'] else None,
                        footprints[frame_ids.index(observed['frame_id'])] if observed['available'] else None,
                        bool(observed['available']),observed['reason']))
                full_success+=all(r.available for r in selected)
            decision=update(prior,selected,area,kind,float(p.temperatures['F']),p.valid_ids,row['old_class'],
                required_count=2 if row['query_eligible'] else 0)
            if not row['query_eligible']:decision['reason']=row['reason']
            if decision['changed']:relabels[owner]=decision['label']
            pfinal[j]=decision['probability']
            if decision['p_new'] is not None:pnew[j]=decision['p_new'];available_new[j]=True
            if 'view_probabilities' in decision:
                vp=decision['view_probabilities'];view_probabilities[j,:len(vp)]=vp
            decision_rows[str(owner)]={key:value for key,value in decision.items() if key not in ('probability','p_new','view_probabilities')}
            decision_rows[str(owner)].update(owner=owner,old_class=row['old_class'],policy=policy,updater=kind,
                anchor=row['anchor'],second=choice.get('second'),required_content_keys=[r.key for r in selected],
                before_acquisition_choice_identity=choices['identity'],p0_array_row=j,p_final_array_row=j,
                p_new_array_row=j if available_new[j] else None,
                query_eligible=row['query_eligible'],successful_required_FULL=bool(selected and all(r.available for r in selected)))
        posterior=arrays_record(root/'decisions'/stage/scene/(method+'.npz'),
            {'owner_ids':np.asarray(ordered,np.int64),'valid_ids':np.asarray(p.valid_ids,np.int64),
             'p0':p0,'p_final':pfinal,'p_new':pnew,'p_new_available':available_new,'view_probabilities':view_probabilities},p.index)
        decision=write(root/'decisions'/stage/scene/(method+'.json'),plain({'status':'DECISIONS_LOCKED','scene':scene,
            'stage':stage,'method':method,'input_identity':key,'owners':decision_rows,'lossless_posteriors':posterior,
            'GT_used':False,'no_third_observation':True}))
        cost={'logical_FULL_reads':full_reads,'logical_probe_heads':probes,'selected':n,
            'query_eligible':len(plan['query_eligible_owner_ids']),'required_FULL_success':int(full_success),'updated':len(relabels)}
        payload=build_payload(e.g1,e.d2,e.raw,relabels,eligible,e.nearest,e.matched,method,
            {'FULL_reads':full_reads,'probe_heads':probes},{'decision_identity':decision['identity'],'fixed_source':'G1'})
        payload_key=payload.prediction_key
        alias=known.get(payload_key)
        if alias:
            item={**alias,'source_kind':'EXACT_OWNER_SEMANTIC_CURRENT_RANK_ALIAS','decision_identity':decision['identity'],
                'alias_proof':{'owners':_array_digest(payload.owner_ids),'semantics':_array_digest(payload.semantic_labels),
                    'ranks':canonical_digest(payload.instance_ranks)}}
        else:
            saved=save_prediction(payload,dest/method)
            item={'status':'PREDICTION_LOCKED','manifest':str(saved),'prediction_key':payload_key,'record_key':payload.record_key,
                'source_kind':'REAL_FIXED_G1_REQUERY_LABELS','decision_identity':decision['identity']}
        results[method]=item;known[payload_key]=item;costs[method]=cost
    result=write(dest/'receipt.json',{'status':'PREDICTIONS_LOCKED','scene':scene,'stage':stage,'input_identity':key,
        'methods':results,'costs':costs,'owner_partition_unchanged':True,'recovered_classes_unchanged':True,
        'unknown_and_unselected_classes_unchanged':True,'GT_used':False,'elapsed_seconds':time.perf_counter()-begin})
    p.index.write_memo(root/'inputs'/scene/'verifications.json');print('PREDICTED',stage,scene,len(paths),flush=True)
    return result


def predict(binding,stage,cohorts,plans,choices,anchor,second,paths):
    root=Path(binding['output_root']);locks={};costs={m:{'logical_FULL_reads':0,'logical_probe_heads':0,
        'selected':0,'query_eligible':0,'updated':0,'required_FULL_success':0} for m in paths}
    for group in cohorts.values():
        for scene in group:
            lock=predict_scene(binding,stage,scene,plans[scene],choices[scene],anchor,second,paths)
            locks[scene]=lock['identity']
            for method,c in lock['costs'].items():
                for name,value in c.items():costs[method][name]+=value
    return write(root/'predictions'/stage/'summary.json',{'status':'ALL_STAGE_PREDICTIONS_LOCKED','stage':stage,
        'scenes':locks,'methods':list(paths),'paths':paths,'costs':costs,'cohorts':cohorts,'GT_used':False})
