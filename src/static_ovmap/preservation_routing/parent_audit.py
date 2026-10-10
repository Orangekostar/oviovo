"""Finite retrospective TRAIN/DEV audit; missing historical components stay null."""
import gzip
import json
import shutil
from pathlib import Path
import time
import torch
from static_ovmap.learned_object_readout.model import ReadoutHead
from .common import ConsumptionIndex,PathResolver,objects,verified,write
from .recognition import evaluate

def audit(binding,fc,loader):
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);out=root/'parent_audit';path=out/'receipt.json'
    if path.exists():return verified(path)
    started=time.perf_counter();curves={};meta={};missing=[]
    for directory in sorted((parent/'training').glob('seed*/*')):
        if not directory.is_dir():continue
        ledger=directory/'updates.jsonl';rel=directory.relative_to(parent/'training')
        if ledger.exists():
            rows=[json.loads(line) for line in ledger.read_text().splitlines()]
            dest=out/'curves'/rel/'updates.jsonl.gz';dest.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(dest,'wb') as stream:stream.write(ledger.read_bytes())
            available=sorted({k for r in rows for k in r})
            curves[str(rel)]=dict(updates=len(rows),fields=available,
                    loss_first=rows[0]['loss'],loss_last=rows[-1]['loss'],component_fields=sorted({k for r in rows for k in r.get('auxiliary',{})}))
            missing.append(dict(branch=str(rel),ce_clean=None,ce_corrupt=None,consistency=None,
                                reason='Historical ledger records summed loss, not these scalar components'))
        else:missing.append(dict(branch=str(rel),reason='Historical ledger missing'))
        for p in directory.glob('dev_*.json'):
            dest=out/'curves'/rel/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        if (directory/'receipt.json').exists():meta[str(rel)]=verified(directory/'receipt.json')
    # All retained selected heads + both warmups are evaluated on identical TRAIN/DEV originals.
    resolver=PathResolver(binding['path_map'])
    nominations=[resolver.rewrite(verified(parent/'dev_nomination.json')),resolver.rewrite(verified(parent/'repeat_nomination.json'))]
    heads={'FC8':None}
    for seed,nom in zip((17,29),nominations):
        sources={**nom['checkpoints'],'WARMUP':resolver.rewrite(verified(parent/f'training/seed{seed}/WARMUP/receipt.json')['last_checkpoint'])}
        for branch,record in sources.items():
            fc.index.identity(record['path'],record)
            state=torch.load(record['path'],map_location='cpu',weights_only=False)
            grouping=('VIEW' if branch=='LR06_MV_VIEW' else 'SURFACE' if branch in ('LR07_MV_SURFACE','LR08_MV_AUX') else 'NONE')
            h=ReadoutHead(grouping).to(fc.device);h.load_state_dict(state['model'],strict=True);h.eval().requires_grad_(False)
            heads[f'seed{seed}_{branch}']=h
    results={}
    for role in ('train','dev'):
        items=objects(binding,role);results[role]={};baseline=None
        for name,head in heads.items():
            value=evaluate(head,loader,items,fc,reference=baseline,save_path=out/'recognition'/role/(name+'.json'))
            if head is None:baseline=value['records']
            results[role][name]=dict(metrics=value['metrics'],recognition=value['identity'])
            print('PARENT_AUDIT',role,name,value['metrics']['A'],flush=True)
    # Common group shift cancels; a common absolute membership reduction does not.
    from .models import absolute_route
    q=torch.tensor([-.7,.2,1.1],device=fc.device)
    old_delta=float((q.softmax(0)-(q-10).softmax(0)).abs().max())
    a,n=absolute_route(torch.zeros(4,3,device=fc.device),torch.ones(3,device=fc.device))
    b,nb=absolute_route(torch.zeros(4,3,device=fc.device),torch.full((3,),.01,device=fc.device))
    intervention=dict(old_shift_max_abs=old_delta,old_total_mass=1.,new_local_mass=float(a[0].sum()),
                      new_reduced_local_mass=float(b[0].sum()),new_null_after=float(nb[0]),
                      inference='Algebraic hypothesis only; not explanation of all historical accuracy loss')
    return write(path,dict(status='COMPLETE',parent=binding['parent_store_identity'],curves=curves,
          historical_checkpoint_metadata=meta,missing_components=missing,TRAIN_DEV=results,
          fixed_key_intervention=intervention,old_H_used_for_selection=False,elapsed_seconds=time.perf_counter()-started))
