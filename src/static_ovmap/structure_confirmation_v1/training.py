"""Sixteen fresh arms, original integer draws, permanent states and safe resume."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from static_ovmap.learned_object_readout.training import draw_schedule,optimizer_for,learning_rate,model_digest
from static_ovmap.learned_object_readout.losses import base_loss
from static_ovmap.learned_object_readout.engineering import frozen_digest
from .common import (METHODS,STATICS,CHECKPOINTS,CONDITIONS,objects,verified,write,
    immutable_record,canonical_digest,arrays_record,unit_key,existing,reserve)
from .models import make_models,static_model
from .grouping import method_inputs
from .recognition import evaluate
from .checkpoints import save_resume,restore_resume
from .selection import choose_step

PRODUCERS=('common','models','grouping','data','recognition','checkpoints','selection','training')

def model_state(path,head,*,seed,method,step,input_identity,index):
    path=Path(path)
    state=dict(model={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},seed=seed,method=method,step=step,input_identity=input_identity,model_digest=model_digest(head))
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        if any(old[k]!=state[k] for k in ('seed','method','step','input_identity','model_digest')):raise ValueError('Permanent evaluation state changed')
    else:
        temp=path.with_name(path.name+'.tmp');torch.save(state,temp);temp.replace(path)
    return index.identity(path)

def train_arm(binding,fc,loader,budget,train,dev,schedule,seed,method,reference):
    root=Path(binding['output_root']);directory=root/f'training/seed{seed}'/method;directory.mkdir(parents=True,exist_ok=True)
    identity=unit_key('train_arm',dict(binding=binding['identity'],prepare=verified(root/'prepare_cache.json')['identity'],features=loader.feature_identity,FC=fc.model_key,draws=canonical_digest(schedule.tolist())),producers=PRODUCERS,seed=seed,method=method)
    receipt=directory/'receipt.json';old=existing(receipt,identity)
    if old:
        for row in old['validations']:fc.index.identity(row['model']['path'],row['model'])
        return old
    head=make_models(seed)[method].to(fc.device).train();initial=model_digest(head)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    optimizer=optimizer_for(head);validations=[];completed=0;prior_seconds=0.;replayed=0
    ledger=directory/'updates.jsonl'
    if (directory/'last.pt').exists() or (directory/'last.previous.pt').exists():
        state=restore_resume(directory,head,optimizer,input_identity=identity)
        completed=state['step'];validations=state['validations'];prior_seconds=state.get('elapsed_seconds',0.);replayed=state.get('replayed_updates',0)
    partial_path=directory/'failed_partial_update.json'
    partial=verified(partial_path) if partial_path.exists() else None
    partial_replay=int(bool(partial and partial['in_optimizer_update']))
    dropped=[]
    if ledger.exists():
        rows=[json.loads(s) for s in ledger.read_text().splitlines()];dropped=[r for r in rows if r['step']>completed]
        if dropped:
            # Preserve every discarded update; global counters already include its calls.
            write(directory/f'replay_{time.time_ns()}.json',dict(status='COMPLETE',discarded_updates=dropped,restored_step=completed))
            ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=completed))
    if dropped or partial_replay:
        budget.consume('replayed_updates',len(dropped)+partial_replay);replayed+=len(dropped)+partial_replay
    if partial:partial_path.replace(directory/f'processed_failure_{time.time_ns()}.json')
    start=completed;begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def save():
        return save_resume(directory,head,optimizer,seed=seed,method=method,completed=completed,input_identity=identity,validations=validations,initial_model_digest=initial,elapsed_seconds=prior_seconds+time.perf_counter()-begin,replayed_updates=replayed)
    def validate():
        state=model_state(directory/f'model_step{completed:04d}.pt',head,seed=seed,method=method,step=completed,input_identity=identity,index=fc.index)
        path=directory/f'dev_{completed:04d}.json'
        value=evaluate(head,method,loader,dev,fc,budget,save_path=path,reference=reference['records'])
        row=dict(value['metrics'],step=completed,model=state,recognition=value['identity'],path=str(path));validations.append(row)
        print('DEV',seed,method,completed,{k:row[k] for k in ('A','M','C')},flush=True)
    if not (directory/'last.pt').exists():save()
    if completed in CHECKPOINTS and not any(r['step']==completed for r in validations):validate();save()
    print('TRAIN_START',seed,method,'resume',completed,flush=True)
    group_audit=dict(valid_tokens=0,retained_same_slot_tokens=0,calls=0)
    in_update=False
    try:
        with ledger.open('a') as stream:
            for step in range(start,2000):
                in_update=True
                optimizer.zero_grad(set_to_none=True);lr=learning_rate(step)
                for group in optimizer.param_groups:group['lr']=lr
                totals={}
                for oi,prefix,condition in schedule[step*16:(step+1)*16]:
                    obj=train[int(oi)];prefix=int(prefix);condition=int(condition)
                    x,_,_=loader.load(obj,0,prefix);xb,_,_=loader.load(obj,condition,prefix)
                    x,a=method_inputs(method,x,obj['key'],CONDITIONS[0]);xb,b=method_inputs(method,xb,obj['key'],CONDITIONS[condition])
                    for audit in (a,b):
                        if audit:
                            for k in ('valid_tokens','retained_same_slot_tokens'):group_audit[k]+=audit[k]
                            group_audit['calls']+=1
                    budget.consume('main_head',2,persist=False)
                    clean=head(x,fc.phi,training_forward=True,completed_steps=step)
                    corrupt=head(xb,fc.phi,training_forward=True,completed_steps=step)
                    loss,parts=base_loss(clean,corrupt,fc.base_targets[obj['class_id']],fc.base_text)
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
                    (loss/16).backward()
                    parts['total']=float(loss.detach())
                    for k,v in parts.items():totals[k]=totals.get(k,0)+v/16
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite gradient')
                gradients={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None} if step==2 or (step+1)%100==0 else None
                if method!='MV_ATTN' and any(p.grad is not None for p in head.membership.parameters()):raise ValueError('Frozen membership received gradient')
                if step==2 and not any(v>0 for v in gradients.values()):raise ValueError('Effective training step has no nonzero gradient')
                budget.consume('optimizer_updates',persist=False);optimizer.step();completed=step+1;in_update=False
                stream.write(json.dumps(dict(step=completed,components=totals,lr=lr,eta=.5*min(1,step/200),gradient_norm=float(norm),gradients=gradients,sampler_index=completed*16,forward_calls=32))+'\n');stream.flush();budget.flush()
                # Snapshot completed update and RNG before any fallible DEV work.
                pending=completed in CHECKPOINTS
                if completed%100==0 or pending:save()
                if pending:validate();save()
                if completed%100==0:
                    reserve(root);write(receipt,dict(status='RUNNING',input_identity=identity,completed_steps=completed,seed=seed,method=method))
                    print('TRAIN_PROGRESS',seed,method,completed,'seconds',round(time.perf_counter()-begin,1),'loss',round(totals['total'],5),flush=True)
    except BaseException as error:
        # Never publish weights/RNG from an incomplete microbatch as a completed update.
        budget.flush();write(partial_path,dict(status='FAILED',input_identity=identity,in_optimizer_update=in_update,completed_in_memory=completed,error_type=type(error).__name__,error=str(error)));raise
    if [r['step'] for r in validations]!=list(CHECKPOINTS):raise ValueError('Permanent checkpoint coverage incomplete')
    if any(p.grad is not None or p.requires_grad for p in fc.model.parameters()):raise ValueError('Frozen FC acquired gradients')
    return write(receipt,dict(status='COMPLETE',input_identity=identity,seed=seed,method=method,completed_steps=2000,initial_model_digest=initial,validations=validations,
        model_parameters=sum(p.numel() for p in head.parameters()),trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        gradient_parameters_at_last_update=sum(p.numel() for n,p in head.named_parameters() if n in gradients),nonzero_gradient_parameters_at_last_update=sum(p.numel() for n,p in head.named_parameters() if gradients.get(n,0)>0),
        nominal_updates=2000,nominal_head_calls=64000,replayed_updates=replayed,elapsed_seconds=prior_seconds+time.perf_counter()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),G_permutation_audit=group_audit if method=='G_LL' else None))

def train(binding,fc,loader,budget):
    root=Path(binding['output_root']);train_items=objects(binding,'train');dev=objects(binding,'dev')
    path=root/'recognition/dev_B_FC.json';reference=verified(path) if path.exists() else evaluate(static_model('B_FC'),'B_FC',loader,dev,fc,budget,save_path=path)
    for name in STATICS[1:]:
        path=root/('recognition/dev_'+name+'.json')
        if not path.exists():evaluate(static_model(name),name,loader,dev,fc,budget,save_path=path,reference=reference['records'])
    before=frozen_digest(fc);rows=[]
    for seed in (17,29):
        schedule=draw_schedule(train_items,seed,updates=2000);directory=root/f'training/seed{seed}';directory.mkdir(parents=True,exist_ok=True)
        if not (directory/'draws.npz').exists():arrays_record(directory/'draws.npz',dict(draws=schedule),fc.index)
        immutable_record(directory/'draw_contract.json',dict(status='COMPLETE',seed=seed,object_keys=[o['key'] for o in train_items],base_originals=622,draw_sha=canonical_digest(schedule.tolist()),sampling='PARENT_ORIGINAL_INDEX_DOMAIN',prefix_cycle=[2,4,8],scientific_draws=32000))
        for method in METHODS:rows.append(train_arm(binding,fc,loader,budget,train_items,dev,schedule,seed,method,reference))
    after=frozen_digest(fc)
    if before!=after:raise ValueError('Frozen backbone changed')
    for seed in (17,29):
        if len({r['initial_model_digest'] for r in rows if r['seed']==seed and r['method']!='MV_ATTN'})!=1:raise ValueError('FG initialization differs within seed')
    fc.index.write_memo(root/'features/verifications.json')
    return write(root/'training/receipt.json',dict(status='COMPLETE',arms=[dict(seed=r['seed'],method=r['method'],identity=r['identity']) for r in rows],nominal_updates=32000,frozen_before=before,frozen_after=after,source_grid_encodings=fc.images,loader_stats=loader.stats))

def select(binding):
    root=Path(binding['output_root']);training=verified(root/'training/receipt.json')
    arms={};curves={}
    for method in METHODS:
        seed17=verified(root/f'training/seed17/{method}/receipt.json');winner=choose_step(seed17['validations'])
        for seed in (17,29):
            r=verified(root/f'training/seed{seed}/{method}/receipt.json');rows={v['step']:v for v in r['validations']};key=method+':seed'+str(seed)
            curves[key]=r['validations'];arms[key]=dict(method=method,seed=seed,selected=rows[winner['step']],matched1000=rows[1000],terminal2000=rows[2000],seed29_reselected=False)
    return immutable_record(root/'model_selection_lock.json',dict(status='COMPLETE',training=training['identity'],arms=arms,curves=curves,rule='SEED17_A_M_C_EARLIEST_TOL_1E-10;SEED29_SAME_STEP',confirmation_outcomes_used=False))

def freeze(binding,fc):
    root=Path(binding['output_root']);selection=verified(root/'model_selection_lock.json');classes=verified(root/'class_split.json')
    for arm in selection['arms'].values():
        for profile in ('selected','matched1000','terminal2000'):fc.index.identity(arm[profile]['model']['path'],arm[profile]['model'])
    return immutable_record(root/'predictor_lock.json',dict(status='COMPLETE',selection=selection['identity'],arms=selection['arms'],static_methods=list(STATICS),FC=fc.model_key,text=fc.session.text_identity,class_split=classes['identity'],roles=verified(root/'role_lock.json'),numerical_producer=unit_key('predictors',binding['identity'],producers=PRODUCERS),confirmation_outcomes_used=False,selected_slots=19,matched1000_slots=16,all_predictions_before_GT=True))
