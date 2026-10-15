"""Common input inference and a global real-prediction barrier before GT."""
from pathlib import Path
import time
import numpy as np
import torch
from .common import (verified,write,arrays_record,unit_key,existing,immutable_record,
    CONDITIONS,canonical_digest,ConsumptionIndex)
from .data import SparseInputs
from .grouping import method_inputs
from .registry import load_registry

def predict(binding,fc,budget,*,domain):
    root=Path(binding['output_root']);real=domain=='REAL'
    input_path=root/('confirmation/real_input_lock.json' if real else 'confirmation/GT/input_lock.json')
    inputs=verified(input_path);features=verified(root/'confirmation/features/manifest.json');lock=verified(root/'predictor_lock.json')
    if not real:verified(root/'confirmation/real_prediction_lock.json')
    identity=unit_key('confirm_'+domain+'_predict',dict(predictors=lock['identity'],inputs=inputs['identity'],features=features['identity']),producers=('prediction','registry','models','grouping','data'),prefixes=[2,4,8])
    folder=root/('confirmation/real_predictions' if real else 'confirmation/GT/predictions');path=root/('confirmation/real_prediction_lock.json' if real else 'confirmation/GT/prediction_lock.json')
    old=existing(path,identity)
    if old:return old
    slots,models,weight_aliases=load_registry(binding,fc);loader=SparseInputs(fc,features);records=[];head_calls=0;start=time.perf_counter()
    columns=[dict(slot=s['id'],method=s['method'],seed=s['seed'],step=s['step'],profile=s['profile'],prefix=p,unique_model=s['unique_model']) for s in slots for p in s['prefixes']]
    ncond=1 if real else 4
    with torch.no_grad():
        for oi,obj in enumerate(inputs['objects']):
            object_id=canonical_digest(dict(key=obj['key'],identity=identity));record_path=folder/(object_id+'.json')
            object_identity=canonical_digest(dict(predictors=identity,object=obj))
            prior=existing(record_path,object_identity)
            if prior:records.append(prior);continue
            arrays=np.full((len(columns),ncond,200),np.nan,np.float32);embeddings=np.full((len(columns),ncond,768),np.nan,np.float32);predictions=np.zeros((len(columns),ncond),np.int64);defined=np.zeros((len(columns),ncond),bool);coverage=[];aliases={};calls=0
            for condition in range(ncond):
                inferred={};xs={}
                for ci,col in enumerate(columns):
                    prefix=col['prefix'];alias_key=(col['unique_model'],prefix)
                    if alias_key in inferred:
                        old_ci=inferred[alias_key];arrays[ci,condition]=arrays[old_ci,condition];embeddings[ci,condition]=embeddings[old_ci,condition];predictions[ci,condition]=predictions[old_ci,condition];defined[ci,condition]=defined[old_ci,condition];aliases[str(ci)+':'+str(condition)]=old_ci;continue
                    inferred[alias_key]=ci
                    if not obj['supported']:
                        coverage.append(dict(column=ci,condition=condition,actual_views=0,local_observations=0,group_count=None,fallback=obj['reason'],head_invoked=False));continue
                    if prefix not in xs:xs[prefix]=loader.load(obj,condition,prefix)[0]
                    x=xs[prefix];x,audit=method_inputs(col['method'],x,obj['key'],CONDITIONS[condition])
                    budget.consume('confirmation_head',persist=False);calls+=1
                    result=models[col['unique_model']](x,fc.phi);z=result['embedding']
                    if z is not None:
                        score=z@fc.text_tensor.T
                        if not bool(torch.isfinite(z).all() and torch.isfinite(score).all()):raise ValueError('Nonfinite confirmation prediction')
                        arrays[ci,condition]=score.cpu().numpy();embeddings[ci,condition]=z.cpu().numpy();predictions[ci,condition]=fc.ids[int(score.argmax())];defined[ci,condition]=True
                    coverage.append(dict(column=ci,condition=condition,actual_views=int(x['view_valid'].sum()),local_observations=int(x['local_valid'][x['view_valid']].sum()),group_count=result['group_count'],fallback=result['fallback'],head_invoked=True,G_audit=audit))
            budget.flush();head_calls+=calls
            record_arrays=arrays_record(record_path.with_suffix('.npz'),dict(full200_cosines=arrays,embeddings=embeddings,predictions=predictions,defined=defined,valid_ids=np.asarray(fc.ids,np.int64)),fc.index)
            records.append(immutable_record(record_path,dict(status='COMPLETE',input_identity=object_identity,key=obj['key'],scene=obj['scene'],family=obj['family'],object_inputs=obj.get('inputs'),support=obj.get('support'),arrays=record_arrays,coverage=coverage,exact_output_aliases=aliases,physical_head_calls=calls,domain=domain,GT_used_by_predictor=False)))
            print('CONF_PREDICT',domain,oi+1,len(inputs['objects']),flush=True)
    fc.index.write_memo(root/'features/verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,domain=domain,inputs=inputs['identity'],predictors=lock['identity'],features=features['identity'],columns=columns,slots=slots,weight_aliases=weight_aliases,records=[dict(key=r['key'],path=str(folder/(canonical_digest(dict(key=r['key'],identity=identity))+'.json')),identity=r['identity']) for r in records],original_objects=len(records),physical_head_calls_this_invocation=head_calls,elapsed_seconds=time.perf_counter()-start,all_12_scenes=verified(root/'role_lock.json')['roles']['CONF'],annotation_files_parsed_by_predictor=0))

def run(binding,*,phase):
    raise RuntimeError('Confirmation prediction must execute in the GPU worker')
