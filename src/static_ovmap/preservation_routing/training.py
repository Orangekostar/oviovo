"""Fixed independent arms, component curves and exact checkpoint/RNG resume."""
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
from .common import ConsumptionIndex,arrays_record,canonical_digest,method,objects,verified,write
from .models import RHead,GHead,teacher
from .losses import item_loss
from .recognition import evaluate,load_head
from .selection import choose,qualifies

def reset_rng(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def initialize(seed,residual):
    reset_rng(seed);return RHead(residual)

def train_arm(binding,fc,loader,teachers,train,dev,schedule,seed,cfg,reference,*,base=None):
    root=Path(binding['output_root']);branch=cfg['id'];directory=root/'training'/f'seed{seed}'/branch
    directory.mkdir(parents=True,exist_ok=True);index=ConsumptionIndex(root/'training/verifications.json')
    config=(dict(stage='R',residual=cfg['residual'],preservation=cfg['preservation']) if base is None else
            dict(stage='G',grouping=cfg['grouping'],direct_routing=cfg['direct_routing'],base_residual=base['config']['residual'],
                 preservation=base['config']['preservation'],base_checkpoint=base['checkpoint']))
    code=[index.identity(Path(__file__).with_name(n)) for n in ('models.py','losses.py','training.py','recognition.py','selection.py')]
    input_contract=dict(binding=binding['identity'],data=verified(root/'prepare.json')['identity'],
                        teachers=teachers['identity'],draws=canonical_digest(schedule.tolist()),seed=seed,branch=branch,
                        config=config,numerical_sources=code)
    identity=canonical_digest(input_contract);rp=directory/'receipt.json'
    if rp.exists():
        previous=verified(rp)
        if previous['input_identity']!=identity:raise ValueError('Scientific arm inputs/code changed')
        if previous['status']=='COMPLETE':
            for name in ('selected_checkpoint','last_checkpoint'):index.identity(previous[name]['path'],previous[name])
            return previous
    if base is None:head=initialize(seed,cfg['residual'])
    else:
        b,_=load_head(base['checkpoint']['path']);reset_rng(seed+1009)
        head=GHead(b,cfg['grouping'],cfg['direct_routing'])
    initial=model_digest(head);ma_initial=model_digest(head.ma) if base is None else model_digest(head.base)
    head.to(fc.device).train();reset_rng(seed);optimizer=optimizer_for(head)
    last=directory/'last.pt';selected=directory/'selected.pt';validations=[];start=0;prior_seconds=0.
    discarded=0;ledger=directory/'updates.jsonl'
    if last.exists():
        state=restore_checkpoint(last,head,optimizer,input_identity=identity)
        if state['config']!=config:raise ValueError('Frozen reference/base dependency changed')
        start=state['step'];validations=state['validations'];prior_seconds=state['elapsed_seconds']
        discarded=state.get('discarded_uncheckpointed_updates',0)
    elif rp.exists():raise ValueError('Interrupted arm has no checkpoint; preserve evidence')
    if ledger.exists():
        rows=[json.loads(s) for s in ledger.read_text().splitlines()]
        dropped=[r for r in rows if r['step']>start]
        if dropped:
            write(directory/f'replay_{time.time_ns()}.json',dict(discarded_updates=dropped,historical_failure_seconds=None))
            discarded+=len(dropped);ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=start))
    if not (directory/'dev_0000.json').exists():
        value=evaluate(head,loader,dev,fc,reference=reference['records'],save_path=directory/'dev_0000.json')
        write(directory/'step0_identity.json',dict(diagnostic_only=True,metrics=value['metrics'],initial_model=initial,
                    initial_trainable_MA=ma_initial,reference_checkpoint_eligible=False))
    # Shared exact initial response cache; keys include immutable input, prefix, FC and reference state.
    cache={};cache_path=root/'reference_cache'/f'seed{seed}.npz'
    cache_identity=canonical_digest(dict(MA_initial=ma_initial,features=verified(root/'features/train-dev.json')['identity'],
                                         FC=fc.model_key,text=fc.session.text_identity,seed=seed))
    if base is None and cfg['residual'] and cache_path.exists():
        memo=verified(cache_path.with_suffix('.json'))
        if memo['input_identity']!=cache_identity:raise ValueError('Frozen initialization cache changed')
        fc.index.identity(memo['arrays']['path'],memo['arrays'])
        with np.load(cache_path,allow_pickle=False) as a:cache={str(k):torch.from_numpy(v.copy()) for k,v in zip(a['keys'],a['vectors'])}
    def forward(x,obj,cond,prefix):
        if base is not None or not cfg.get('residual'):return head(x,fc.phi)
        key=obj['inputs']['sha256']+'|'+str(cond)+'|'+str(prefix)
        if key not in cache:
            with torch.no_grad():cache[key]=head.initial(x,fc.phi)['embedding'].detach().cpu()
        return head(x,fc.phi,reference_embedding=cache[key].to(fc.device))
    begin=time.perf_counter();completed=start;torch.cuda.reset_peak_memory_stats()
    print('TRAIN_START',seed,branch,'resume',start,flush=True)
    def save(path):
        save_checkpoint(path,head,optimizer,seed=seed,branch=branch,step=completed,input_identity=identity,
            config=config,validations=validations,initial_model_digest=initial,initial_trainable_MA=ma_initial,
            elapsed_seconds=prior_seconds+time.perf_counter()-begin,discarded_uncheckpointed_updates=discarded)
    try:
        with ledger.open('a') as stream:
            for step in range(start,2000):
                optimizer.zero_grad(set_to_none=True);lr=learning_rate(step)
                for group in optimizer.param_groups:group['lr']=lr
                totals={};denoms={}
                for oi,prefix,condition in schedule[step*16:(step+1)*16]:
                    obj=train[int(oi)];prefix=int(prefix);condition=int(condition);p=(2,4,8).index(prefix)
                    x,_,_=loader.load(obj,0,prefix);xb,_,_=loader.load(obj,condition,prefix)
                    clean=forward(x,obj,0,prefix);corrupt=forward(xb,obj,condition,prefix)
                    i=teachers['cache']['keys'][obj['key']];t=teachers['cache']['embeddings'][i,p,0]
                    correct=bool(teachers['cache']['correct'][i,p])
                    if not torch.allclose(t,teacher(x),atol=1e-6,rtol=1e-5):raise ValueError('Teacher same-prefix parity failed')
                    memberships=([loader.targets(obj,0,prefix)[1],loader.targets(obj,condition,prefix)[1]] if base is not None else None)
                    loss,parts=item_loss(clean,corrupt,fc.base_targets[obj['class_id']],fc.base_text,t,correct,config['preservation'],
                                         memberships=memberships,config=cfg)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    (loss/16).backward()
                    for k,v in parts.items():
                        if k in ('teacher_correct','teacher_items','positive_tokens','negative_tokens','unknown_tokens','correspondence_pairs'):
                            denoms[k]=denoms.get(k,0)+v
                        elif isinstance(v,(float,int)) and not isinstance(v,bool):totals[k]=totals.get(k,0)+v/16
                norm=torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],1)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
                gradients={name:float(p.grad.norm()) for name,p in head.named_parameters() if p.grad is not None} if (step==0 or (step+1)%100==0) else None
                optimizer.step();completed=step+1
                stream.write(json.dumps(dict(step=completed,components=totals,denominators=denoms,lr=lr,gradient_norm=float(norm),
                                  gradients=gradients,sampler_index=completed*16,original_item_draws=16))+'\n');stream.flush()
                if completed in (250,500,1000,1500,2000):
                    value=evaluate(head,loader,dev,fc,reference=reference['records'],save_path=directory/f'dev_{completed:04d}.json')
                    row=dict(value['metrics'],step=completed,recognition=value['identity'],path=str(directory/f'dev_{completed:04d}.json'))
                    validations.append(row)
                    winner=choose(validations,reference['metrics']) or choose(validations)
                    if winner['step']==completed:save(selected)
                    print('DEV',seed,branch,completed,{k:row[k] for k in ('A','M','C','net_corrections')},'passes',qualifies(row,reference['metrics']),flush=True)
                if completed%100==0 or completed==2000:
                    save(last);write(rp,dict(status='RUNNING',input_identity=identity,completed_steps=completed,seed=seed,branch=branch,
                                           elapsed_seconds=prior_seconds+time.perf_counter()-begin))
                    print('TRAIN_PROGRESS',seed,branch,completed,'loss',round(totals['total'],5),'seconds',round(time.perf_counter()-begin,1),flush=True)
    except BaseException:
        # Save only fully completed optimizer updates; partial microbatches are replayed.
        save(last);raise
    if cache:
        arr=arrays_record(cache_path,dict(keys=np.asarray(list(cache)),vectors=np.stack([v.numpy() for v in cache.values()])),fc.index)
        write(cache_path.with_suffix('.json'),dict(input_identity=cache_identity,arrays=arr,initial_model=ma_initial,
                                                uses_GT=False,entries=len(cache)))
    winner=choose(validations,reference['metrics']) or choose(validations)
    result=write(rp,dict(status='COMPLETE',input_identity=identity,input_contract=input_contract,seed=seed,branch=branch,
            config=config,completed_steps=2000,original_item_draws=32000,initial_model_digest=initial,initial_trainable_MA=ma_initial,
            selected_step=winner['step'],selected_metrics=winner,foundation_pass=qualifies(winner,reference['metrics']),
            selected_checkpoint=index.identity(selected),last_checkpoint=index.identity(last),validations=validations,
            trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
            frozen_small_parameters=sum(p.numel() for p in head.parameters() if not p.requires_grad),
            elapsed_seconds=prior_seconds+time.perf_counter()-begin,discarded_uncheckpointed_updates=discarded,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    index.write_memo(root/'training/verifications.json');fc.index.write_memo(root/'features/verifications.json')
    return result

def train_R(binding,fc,loader,teacher_row,seed=17):
    from .teachers import load
    root=Path(binding['output_root']);train=objects(binding,'train');dev=objects(binding,'dev')
    # Keep the parent's original list ordering; draw_schedule excludes every nonbase original.
    schedule=draw_schedule(train,seed);tc=dict(teacher_row,cache=load(teacher_row,fc.device))
    reference_path=root/'recognition/dev_FC8.json'
    reference=verified(reference_path) if reference_path.exists() else evaluate(None,loader,dev,fc,save_path=reference_path)
    if seed==17:configs=binding['specification']['R_methods']
    else:
        nom=verified(root/'R_nomination_seed17.json')
        if nom['Rstar'] is None:return write(root/'R_nomination_seed29.json',dict(status='NOT_TRIGGERED_NO_FOUNDATION',Rstar=None))
        configs=[method(binding,nom['Rstar'])]
    directory=root/'training'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    arrays_record(directory/'draws.npz',dict(draws=schedule),fc.index)
    write(directory/'draw_contract.json',dict(seed=seed,object_keys=[o['key'] for o in train],base_originals=622,
          sampling='PARENT_UNIFORM_PRESENT_BASE_CLASS_THEN_ORIGINAL',prefix_cycle=[2,4,8],corruption_cycle='parent fixed three variants',
          same_draws_for_all_arms=True,scientific_initialization='FRESH_NO_PARENT_WARMUP'))
    frozen_before=frozen_digest(fc)
    rows=[train_arm(binding,fc,loader,tc,train,dev,schedule,seed,cfg,reference) for cfg in configs]
    frozen_after=frozen_digest(fc)
    if frozen_before!=frozen_after or any(p.grad is not None for p in fc.model.parameters()):raise ValueError('Frozen FC moved')
    if len({r['initial_trainable_MA'] for r in rows})!=1:raise ValueError('Matched R initialization differs')
    return write(directory/'R_receipt.json',dict(status='COMPLETE',seed=seed,arms={r['branch']:r['identity'] for r in rows},
                 scientific_updates=2000*len(rows),frozen_before=frozen_before,frozen_after=frozen_after,physical_image_encodings=0))

def nominate_R(binding,seed):
    root=Path(binding['output_root']);directory=root/'training'/f'seed{seed}';train=verified(directory/'R_receipt.json')
    rows=[verified(directory/m/'receipt.json') for m in train['arms']]
    candidates=[dict(r['selected_metrics'],branch=r['branch'],order=i,checkpoint=r['selected_checkpoint']) for i,r in enumerate(rows) if r['foundation_pass']]
    winner=choose(candidates)
    return write(root/f'R_nomination_seed{seed}.json',dict(status='QUALIFIED' if winner else 'COMPLETE_NO_2D_FOUNDATION' if seed==17 else 'COMPLETE_2D_NOT_REPEATED',
           seed=seed,Rstar=winner['branch'] if winner else None,winner=winner,
           checkpoints={r['branch']:r['selected_checkpoint'] for r in rows},arms={r['branch']:r['selected_metrics'] for r in rows},
           data=verified(root/'prepare.json')['identity'],selection_uses_old_H=False,selection_uses_maps=False,selection_uses_runtime=False))
