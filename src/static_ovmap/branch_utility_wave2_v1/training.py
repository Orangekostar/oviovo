"""Conditional small-controller training; no FC/FG2 objects or forward calls."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from .common import verified,write,arrays_record,ConsumptionIndex,canonical_digest,unit_identity,immutable_write
from .utility_model import Normalizer,UtilityRegressor
ARMS=('WITH_NEW_CORRESPONDENCE_STATS','WITHOUT_NEW_CORRESPONDENCE_STATS')

def make_draws(objects,seed,steps=2000,batch=16):
    rng=np.random.default_rng(seed+3001);classes=sorted({o['class_id'] for o in objects});banks={c:[i for i,o in enumerate(objects) if o['class_id']==c] for c in classes};out=np.empty((steps,batch,3),np.int64)
    for step in range(steps):
        for j in range(batch):
            c=classes[int(rng.integers(len(classes)))];bank=banks[c]
            out[step,j]=(bank[int(rng.integers(len(bank)))],int(rng.choice([4,8])),int(rng.choice(4,p=[.5,1/6,1/6,1/6])))
    return out

def group_loss(predictions,targets,offsets):
    losses=[]
    for a,b in zip(offsets[:-1],offsets[1:]):
        losses.append(F.smooth_l1_loss(predictions[a:b],targets[a:b],beta=.1) if b>a else predictions.sum()*0)
    return torch.stack(losses).mean()

def save_resume(path,model,optimizer,step,extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,extra=extra,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,numpy_rng=np.random.get_state(),python_rng=random.getstate()),temp);temp.replace(path)

def load_resume(path,model,optimizer,expected):
    row=torch.load(path,map_location='cpu',weights_only=False)
    if any(row['extra'].get(k)!=v for k,v in expected.items()):raise ValueError('Resume data/config/normalizer identity differs')
    model.load_state_dict(row['model']);optimizer.load_state_dict(row['optimizer']);torch.set_rng_state(row['torch_rng']);np.random.set_state(row['numpy_rng']);random.setstate(row['python_rng'])
    if row['cuda_rng'] is not None:torch.cuda.set_rng_state_all(row['cuda_rng'])
    return row['step'],row['extra']

def load_bank(root,role,seed):
    root=Path(root);bank=verified(root/'action_bank.json');rows=[];groups={};features=[];targets=[]
    for unit in bank['units']:
        if unit['role']!=role:continue
        record=verified(unit['path']);ConsumptionIndex().identity(record['arrays']['path'],record['arrays'])
        with np.load(record['arrays']['path'],allow_pickle=False) as a:
            for group in record['groups']:
                if group['seed']!=seed or group['prefix'] not in (4,8):continue
                ix=group['indices'][1:];start=len(features)
                for i in ix:features.append(a['features'][i]);targets.append(record['rows'][i]['utility'])
                groups[(record['object']['key'],record['condition'],group['prefix'])]=list(range(start,len(features)))
        rows.append(record['object'])
    objects=sorted({o['key']:o for o in rows}.values(),key=lambda o:o['key'])
    return np.asarray(features,np.float32).reshape(-1,24),np.asarray(targets,np.float32),groups,objects,bank['identity']

def training_gate(root):
    root=Path(root);bank=verified(root/'action_bank.json');sets={seed:{kind:{} for kind in ('positive','negative','branch_preference')} for seed in (17,29)}
    for u in bank['units']:
        if u['role']!='train':continue
        r=verified(u['path']);obj=r['object']
        for g in r['groups']:
            if g['prefix']!=8:continue
            seed=g['seed'];actions=[r['rows'][i] for i in g['indices'][1:]];values=[a['utility'] for a in actions]
            if any(v>=.01 for v in values):sets[seed]['positive'][obj['key']]=obj['family']
            if any(v<=-.01 for v in values):sets[seed]['negative'][obj['key']]=obj['family']
            byframe={}
            for a in actions:byframe.setdefault(a['frame_id'],{})[a['kind']]=a['utility']
            if any(max(v['G'],v['L'])>max(0.,v['B'])+.01 for v in byframe.values()):sets[seed]['branch_preference'][obj['key']]=obj['family']
    summaries={};passes=[]
    for seed,groups in sets.items():
        summaries[str(seed)]={k:dict(originals=len(v),families=len(set(v.values())),keys=sorted(v),family_ids=sorted(set(v.values()))) for k,v in groups.items()}
        counts=summaries[str(seed)];ok=all(counts[k]['originals']>=n and counts[k]['families']>=f for k,n,f in [('positive',40,8),('negative',40,8),('branch_preference',20,6)])
        summaries[str(seed)]['passed']=ok;passes.append(ok)
    return write(root/'training_gate.json',dict(status='COMPLETE',training_activated=all(passes),seeds=summaries,role='TRAIN_ONLY',prefix=8,thresholds_fixed=True,action_bank_identity=bank['identity'],DEV_used=False))

def run(binding):
    root=Path(binding['output_root']);gate=verified(root/'training_gate.json')
    if not gate['training_activated']:return write(root/'training.json',dict(status='NOT_ACTIVATED_BY_TRAIN_GATE',new_optimizer_updates=0,gate=gate['identity'],arms={},frozen_FG2_gradients=False))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(1);index=ConsumptionIndex();arms={}
    for seed in (17,29):
        x,y,groups,objects,data_id=load_bank(root,'train',seed);normal=Normalizer.fit(x);folder=root/'training'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        norm=immutable_write(folder/'normalizer.json',normal.record());draws=make_draws(objects,seed);drawpath=folder/'draws.npz'
        if not drawpath.exists():drawrow=arrays_record(drawpath,dict(draws=draws,object_keys=np.asarray([o['key'] for o in objects])),index)
        else:drawrow=index.identity(drawpath)
        expected=dict(data=data_id,normalizer=norm['identity'],draws=drawrow['sha256'],sources=unit_identity(binding,'training',seed=seed),seed=seed)
        for arm in ARMS:
            directory=folder/arm;receipt=directory/'receipt.json'
            if receipt.exists():
                old=verified(receipt)
                if old['inputs']!=expected:raise ValueError('Completed training identity changed')
                arms[f'{seed}:{arm}']=old;continue
            model=UtilityRegressor(seed).to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
            initial={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()};initial_id=canonical_digest({k:v.tolist() for k,v in initial.items()})
            tx=torch.from_numpy(normal.transform(x,without_new=arm==ARMS[1])).to('cuda');ty=torch.from_numpy(y).to('cuda');step=0;resume=directory/'resume.pt';curve=directory/'curve.jsonl';directory.mkdir(parents=True,exist_ok=True)
            if resume.exists():step,extra=load_resume(resume,model,optimizer,expected)
            existing=[json.loads(line) for line in curve.read_text().splitlines()] if curve.exists() else []
            existing=[r for r in existing if r['step']<=step]
            curve.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in existing))
            with curve.open('a') as stream:
                for it in range(step,2000):
                    ids=[];offsets=[0];empty=0
                    for objidx,prefix,condition in draws[it]:
                        group=groups[(objects[int(objidx)]['key'],int(condition),int(prefix))];ids.extend(group);offsets.append(len(ids));empty+=int(not group)
                    optimizer.zero_grad(set_to_none=True)
                    if ids:
                        pred=model(tx[ids]);loss=group_loss(pred,ty[ids],offsets)
                    else:loss=sum(p.sum()*0 for p in model.parameters())
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    row=dict(step=it+1,loss=float(loss.detach()),grad_norm=float(grad),empty_groups=empty,actions=len(ids),draw_index=it+1,learning_rate=.001)
                    if not np.isfinite([row['loss'],row['grad_norm']]).all():raise ValueError('Nonfinite controller training')
                    stream.write(json.dumps(row,sort_keys=True)+'\n')
                    if (it+1)%100==0:stream.flush();save_resume(resume,model,optimizer,it+1,dict(expected,draw_index=it+1,initial_state=initial_id))
                    if (it+1)%500==0:
                        path=directory/f'step{it+1}.pt';torch.save(dict(model=model.state_dict(),seed=seed,arm=arm,step=it+1,normalizer=norm['identity'],inputs=expected),path)
                    if (it+1)%500==0:print(f'train seed{seed} {arm} step{it+1} loss={row["loss"]:.6f}',flush=True)
            checkpoints={str(step):index.identity(directory/f'step{step}.pt') for step in (500,1000,1500,2000)}
            arms[f'{seed}:{arm}']=write(receipt,dict(status='COMPLETE',seed=seed,arm=arm,updates=2000,parameters=1345,inputs=expected,initial_state=initial_id,normalizer=str(folder/'normalizer.json'),draws=drawrow,checkpoints=checkpoints,curve=index.identity(curve),resume=index.identity(resume),FC_loaded=False,FG2_loaded=False,fp32=True))
    if sum(r['updates'] for r in arms.values())!=8000:raise ValueError('Conditional four-arm budget incomplete')
    for seed in (17,29):
        if arms[f'{seed}:{ARMS[0]}']['initial_state']!=arms[f'{seed}:{ARMS[1]}']['initial_state']:raise ValueError('Paired initialization differs')
    return write(root/'training.json',dict(status='COMPLETE',new_optimizer_updates=8000,gate=gate['identity'],arms=arms,frozen_FG2_gradients=False,paired_draws_and_init=True))
