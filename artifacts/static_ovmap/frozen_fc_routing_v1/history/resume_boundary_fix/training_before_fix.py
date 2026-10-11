"""Four mandatory matched FG arms; exact inherited draws/schedule/RNG resume."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from static_ovmap.learned_object_readout.training import (
    draw_schedule,optimizer_for,learning_rate,save_checkpoint,restore_checkpoint,model_digest,
)
from static_ovmap.learned_object_readout.engineering import frozen_digest
from .common import ConsumptionIndex,arrays_record,canonical_digest,objects,verified,write
from .models import initialize
from .losses import item_loss
from .recognition import evaluate
from .selection import choose,qualifies

def reset_rng(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def train_arm(binding,fc,loader,train,dev,schedule,seed,cfg,reference):
    root=Path(binding['output_root']);branch=cfg['id'];directory=root/f'training/seed{seed}'/branch
    directory.mkdir(parents=True,exist_ok=True);index=ConsumptionIndex(root/'training/verifications.json')
    config=dict(grouping=cfg['grouping'],direct_routing=cfg['direct_routing'],membership_loss=cfg['membership_loss'])
    code=[index.identity(Path(__file__).with_name(n)) for n in ('models.py','losses.py','training.py','recognition.py','selection.py','data.py')]
    contract=dict(binding=binding['identity'],data=verified(root/'prepare.json')['identity'],
        draws=canonical_digest(schedule.tolist()),seed=seed,branch=branch,config=config,numerical_sources=code)
    identity=canonical_digest(contract);rp=directory/'receipt.json'
    if rp.exists():
        old=verified(rp)
        if old['input_identity']!=identity:raise ValueError('Scientific inputs/code changed')
        if old['status']=='COMPLETE':
            for name in ('selected_checkpoint','last_checkpoint'):index.identity(old[name]['path'],old[name])
            return old
    head=initialize(seed,config);initial=model_digest(head);head.to(fc.device).train();reset_rng(seed);optimizer=optimizer_for(head)
    last=directory/'last.pt';selected=directory/'selected.pt';ledger=directory/'updates.jsonl'
    validations=[];start=0;prior_seconds=0.;discarded=0
    if last.exists():
        state=restore_checkpoint(last,head,optimizer,input_identity=identity)
        if state['config']!=config:raise ValueError('Checkpoint architecture changed')
        start=state['step'];validations=state['validations'];prior_seconds=state['elapsed_seconds']
        discarded=state.get('discarded_uncheckpointed_updates',0)
    elif rp.exists():raise ValueError('Interrupted arm missing checkpoint')
    if ledger.exists():
        rows=[json.loads(s) for s in ledger.read_text().splitlines()];dropped=[r for r in rows if r['step']>start]
        if dropped:
            write(directory/f'replay_{time.time_ns()}.json',dict(discarded_updates=dropped,historical_failure_seconds=None))
            discarded+=len(dropped);ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=start))
    if not (directory/'dev_0000.json').exists():
        value=evaluate(head,loader,dev,fc,reference=reference['records'],save_path=directory/'dev_0000.json',eta_override=0.)
        if value['metrics']!=reference['metrics']:
            # Only correction counts differ if a baseline receipt omitted its own reference.
            for k in ('A','M','C','CE','NLL'):
                if abs(value['metrics'][k]-reference['metrics'][k])>1e-10:raise ValueError('Step0 FC identity differs')
        write(directory/'step0_identity.json',dict(diagnostic_only=True,eta=0.,metrics=value['metrics'],initial_model=initial,eligible=False))
    begin=time.perf_counter();completed=start;torch.cuda.reset_peak_memory_stats()
    print('TRAIN_START',seed,branch,'resume',start,flush=True)
    def save(path):
        save_checkpoint(path,head,optimizer,seed=seed,branch=branch,step=completed,input_identity=identity,config=config,
            validations=validations,initial_model_digest=initial,elapsed_seconds=prior_seconds+time.perf_counter()-begin,
            discarded_uncheckpointed_updates=discarded)
    if not last.exists():save(last)
    try:
        with ledger.open('a') as stream:
            for step in range(start,2000):
                optimizer.zero_grad(set_to_none=True);lr=learning_rate(step)
                for group in optimizer.param_groups:group['lr']=lr
                totals={};denoms={}
                for oi,prefix,condition in schedule[step*16:(step+1)*16]:
                    obj=train[int(oi)];prefix=int(prefix);condition=int(condition)
                    x,_,_=loader.load(obj,0,prefix);xb,_,_=loader.load(obj,condition,prefix)
                    clean=head(x,fc.phi,training_forward=True,completed_steps=step)
                    corrupt=head(xb,fc.phi,training_forward=True,completed_steps=step)
                    memberships=[loader.targets(obj,0,prefix)[1],loader.targets(obj,condition,prefix)[1]]
                    loss,parts=item_loss(clean,corrupt,fc.base_targets[obj['class_id']],fc.base_text,
                                        memberships=memberships,membership_enabled=cfg['membership_loss'])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    (loss/16).backward()
                    for k,v in parts.items():
                        if k in ('positive_tokens','negative_tokens','unknown_tokens','known_tokens','local_observations'):
                            denoms[k]=denoms.get(k,0)+v
                        elif isinstance(v,(float,int)) and not isinstance(v,bool):totals[k]=totals.get(k,0)+v/16
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                gradients={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None} if step==0 or (step+1)%100==0 else None
                if not cfg['membership_loss'] and any(p.grad is not None for p in head.membership.parameters()):raise ValueError('Disabled membership acquired gradient')
                optimizer.step();completed=step+1
                stream.write(json.dumps(dict(step=completed,components=totals,denominators=denoms,lr=lr,eta=.5*min(1,step/200),
                    gradient_norm=float(norm),gradients=gradients,sampler_index=completed*16,original_item_draws=16,forward_calls=32))+'\n');stream.flush()
                if completed in (250,500,1000,1500,2000):
                    value=evaluate(head,loader,dev,fc,reference=reference['records'],save_path=directory/f'dev_{completed:04d}.json')
                    row=dict(value['metrics'],step=completed,recognition=value['identity'],path=str(directory/f'dev_{completed:04d}.json'))
                    validations.append(row);winner=choose(validations,reference['metrics']) or choose(validations)
                    if winner['step']==completed:save(selected)
                    print('DEV',seed,branch,completed,{k:row[k] for k in ('A','M','C','net_corrections')},'passes',qualifies(row,reference['metrics']),flush=True)
                if completed%100==0:
                    save(last);write(rp,dict(status='RUNNING',input_identity=identity,completed_steps=completed,seed=seed,branch=branch,
                         elapsed_seconds=prior_seconds+time.perf_counter()-begin))
                    print('TRAIN_PROGRESS',seed,branch,completed,'loss',round(totals['total'],5),'seconds',round(time.perf_counter()-begin,1),flush=True)
    except BaseException:
        save(last);raise
    winner=choose(validations,reference['metrics']) or choose(validations)
    membership=sum(p.numel() for p in head.membership.parameters())
    result=write(rp,dict(status='COMPLETE',input_identity=identity,input_contract=contract,seed=seed,branch=branch,
         config=config,completed_steps=2000,original_item_draws=32000,clean_corrupt_forward_calls=64000,
         initial_model_digest=initial,selected_step=winner['step'],selected_metrics=winner,qualified=qualifies(winner,reference['metrics']),
         selected_checkpoint=index.identity(selected),last_checkpoint=index.identity(last),validations=validations,
         architecture_parameters=sum(p.numel() for p in head.parameters()),
         observed_gradient_parameters_at_last_update=sum(p.numel() for p in head.parameters() if p.grad is not None),
         observed_nonzero_gradient_parameters_at_last_update=sum(p.numel() for n,p in head.named_parameters() if gradients.get(n,0)>0),
         loss_participating_parameters=216194-(membership if not cfg['membership_loss'] else 0),
         elapsed_seconds=prior_seconds+time.perf_counter()-begin,discarded_uncheckpointed_updates=discarded,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    index.write_memo(root/'training/verifications.json');fc.index.write_memo(root/'features/verifications.json');return result

def train(binding,fc,loader,seed=17):
    root=Path(binding['output_root']);train_items=objects(binding,'train');dev=objects(binding,'dev')
    if seed==29 and verified(root/'nomination_seed17.json')['nominee'] is None:
        return write(root/'nomination_seed29.json',dict(status='NOT_TRIGGERED_NO_FG_SIGNAL',seed=29,nominee=None,
             same_nominee_qualified=False,any_qualified=False,checkpoints={},arms={},qualified={}))
    schedule=draw_schedule(train_items,seed);directory=root/f'training/seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    arrays_record(directory/'draws.npz',dict(draws=schedule),fc.index)
    write(directory/'draw_contract.json',dict(seed=seed,object_keys=[o['key'] for o in train_items],base_originals=622,
        sampling='PARENT_UNIFORM_PRESENT_BASE_CLASS_THEN_ORIGINAL',schedule_digest=canonical_digest(schedule.tolist()),
        prefix_cycle=[2,4,8],scientific_draws=32000))
    path=root/'recognition/dev_FC8.json'
    reference=verified(path) if path.exists() else evaluate(None,loader,dev,fc,save_path=path)
    before=frozen_digest(fc);rows=[]
    for cfg in binding['specification']['methods']:rows.append(train_arm(binding,fc,loader,train_items,dev,schedule,seed,cfg,reference))
    after=frozen_digest(fc)
    if before!=after or any(p.grad is not None or p.requires_grad for p in fc.model.parameters()):raise ValueError('Frozen FC changed')
    if len({r['initial_model_digest'] for r in rows})!=1:raise ValueError('Arms initialized differently')
    return write(directory/'receipt.json',dict(status='COMPLETE',seed=seed,arms=[r['branch'] for r in rows],
        scientific_updates=sum(r['completed_steps'] for r in rows),arm_receipts={r['branch']:r['identity'] for r in rows},
        frozen_before=before,frozen_after=after,all_four_mandatory=True))
