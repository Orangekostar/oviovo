"""Explicit profile/seed/step registry with only exact weight aliases."""
from pathlib import Path
import torch
from .models import make_models,static_model
from .common import STATICS,verified

def load_registry(binding,fc,*,map_only=False,selected_only=False):
    root=Path(binding['output_root']);lock=verified(root/'predictor_lock.json');slots=[];models={};aliases={}
    for profile in (('selected',) if map_only or selected_only else ('selected','matched1000')):
        for arm in lock['arms'].values():
            if map_only and arm['method'] not in ('S_LL','V_LL','G_LL','MV_ATTN'):continue
            record=arm[profile];weight=record['model'];fc.index.identity(weight['path'],weight)
            state=torch.load(weight['path'],map_location='cpu',weights_only=False)
            if state['method']!=arm['method'] or state['seed']!=arm['seed'] or state['step']!=record['step']:raise ValueError('Weight metadata differs from selection lock')
            unique=arm['method']+':seed'+str(arm['seed'])+':step'+str(record['step']);slot=profile+':'+arm['method']+':seed'+str(arm['seed'])
            if unique not in models:
                head=make_models(arm['seed'])[arm['method']];head.load_state_dict(state['model'],strict=True);models[unique]=head.to(fc.device).eval()
            else:aliases[slot]=unique
            slots.append(dict(id=slot,profile=profile,method=arm['method'],seed=arm['seed'],step=record['step'],weight=weight,unique_model=unique,prefixes=[8] if profile=='matched1000' or map_only else [2,4,8]))
    for name in STATICS:
        models[name]=static_model(name).to(fc.device).eval();slots.append(dict(id='selected:'+name,profile='selected',method=name,seed=None,step=None,weight=None,unique_model=name,prefixes=[8] if map_only else [2,4,8]))
    return slots,models,aliases
