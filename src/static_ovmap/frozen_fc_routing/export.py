"""Portable FG-only states with strict FC/text identity and load parity."""
from pathlib import Path
import torch
from .common import ConsumptionIndex,objects,verified,write
from .models import FGHead

def load_export(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False);cfg=state['config']
    head=FGHead(cfg['grouping'],cfg['direct_routing']);head.load_state_dict(state['model'],strict=True)
    return head.to(device).eval().requires_grad_(False),state

def load_exported(registry_file,name,device='cpu',*,FC_identity,text_identity):
    path=Path(registry_file);registry=verified(path)
    if registry['FC_identity']!=FC_identity or registry['text_identity']!=text_identity:raise ValueError('Frozen FC/text identity differs')
    row=registry['heads'][name];item=row['weights'];p=path.parent/item['relative_path'];ConsumptionIndex().identity(p,item)
    head,state=load_export(p,device)
    if state['config']!=row['config']:raise ValueError('Export config differs')
    return head

def run(binding,fc,loader,heads):
    root=Path(binding['output_root']);folder=root/'export';lock=verified(root/'predictor_lock.json')
    index=ConsumptionIndex();rows={}
    for name,row in lock['heads'].items():
        source=row['checkpoint'];index.identity(source['path'],source)
        state=torch.load(source['path'],map_location='cpu',weights_only=False)
        p=folder/'weights'/(name+'.pt');p.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=state['model'],config=state['config'],seed=row['seed'],step=state['step']),p)
        weights=dict(index.identity(p),relative_path=str(p.relative_to(folder)))
        if weights['bytes']>=100*2**20:raise ValueError('FG head exceeds normal Git size')
        membership=sum(p.numel() for p in heads[name].membership.parameters())
        rows[name]=dict(weights=weights,config=state['config'],seed=row['seed'],selected_step=state['step'],
             training_checkpoint=source,
             observed_nonzero_gradient_parameters_at_last_update=verified(root/f'training/seed{row["seed"]}/{row["branch"]}/receipt.json')['observed_nonzero_gradient_parameters_at_last_update'],
             architecture_parameters=sum(p.numel() for p in heads[name].parameters()),
             loss_participating_parameters=216194-(membership if not state['config']['membership_loss'] else 0),
             unused_membership_parameters=membership if not state['config']['membership_loss'] else 0,dependencies=['FrozenFC','FrozenText'])
    registry=write(folder/'weights_registry.json',dict(status='COMPLETE',heads=rows,predictor_lock=lock['identity'],
          FC_identity=fc.model_key,text_identity=fc.session.text_identity,FC_reference=binding['reference']['fc'],
          frozen_FC_parameters=sum(p.numel() for p in fc.model.parameters()),frozen_text_elements=fc.text_tensor.numel(),
          MA_dependency=False,R_dependency=False,frozen_backbone_included=False,FP32=True,
          loader='static_ovmap.frozen_fc_routing.export.load_exported'))
    sample=sorted([o for o in objects(binding,'dev') if o['base']],key=lambda o:o['key'])[:2];checks={}
    with torch.no_grad():
        for name,head in heads.items():
            restored=load_exported(folder/'weights_registry.json',name,fc.device,FC_identity=fc.model_key,text_identity=fc.session.text_identity)
            errors=[]
            for obj in sample:
                x,_,_=loader.load(obj,0,8);a=head(x,fc.phi)['embedding'];b=restored(x,fc.phi)['embedding']
                torch.testing.assert_close(a,b,rtol=0,atol=0);errors.append(float((a-b).abs().max()))
            checks[name]=dict(sample_keys=[o['key'] for o in sample],max_abs=max(errors),strict_load=True,bitwise=True)
    return write(folder/'roundtrip.json',dict(status='COMPLETE',predictor_lock=lock['identity'],registry=registry['identity'],
          checks=checks,all_selected_heads=len(checks),actual_frozen_phi=True))
