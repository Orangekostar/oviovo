"""Portable small inference weights; shared frozen references/bases are explicit."""
from pathlib import Path
import json
import torch
from static_ovmap.learned_object_readout.training import model_digest
from .common import ConsumptionIndex,canonical_digest,objects,verified,write
from .models import RHead,GHead

def load_exported(registry_file,name,device='cpu',*,FC_identity,text_identity):
    path=Path(registry_file);registry=verified(path);folder=path.parent
    if registry['FC_identity']!=FC_identity:raise ValueError('Frozen FC identity differs')
    if registry['text_identity']!=text_identity:raise ValueError('Frozen text identity differs')
    row=registry['heads'][name];index=ConsumptionIndex()
    item=row['weights'];weight_path=folder/item['relative_path'];index.identity(weight_path,item)
    value=torch.load(weight_path,map_location='cpu',weights_only=False)
    if value['config']!=row['config']:raise ValueError('Inference config differs')
    state=value['model'];cfg=row['config']
    if cfg['stage']=='R':
        head=RHead(cfg['residual'])
        if cfg['residual']:
            dep=row['dependencies']['initial'];p=folder/dep['relative_path'];index.identity(p,dep)
            initial=torch.load(p,map_location='cpu',weights_only=False)['model']
            state={**state,**{'initial.'+k:v for k,v in initial.items()}}
    else:
        base=load_exported(path,row['dependencies']['base'],device='cpu',FC_identity=FC_identity,text_identity=text_identity)
        head=GHead(base,cfg['grouping'],cfg['direct_routing'])
        state={**state,**{'base.'+k:v for k,v in base.state_dict().items()}}
    head.load_state_dict(state,strict=True);return head.to(device).eval().requires_grad_(False)

def run(binding,fc,loader,heads):
    root=Path(binding['output_root']);folder=root/'export';lock=verified(root/'predictor_lock.json')
    path=folder/'weights_registry.json';roundtrip=folder/'roundtrip.json'
    if path.exists() and roundtrip.exists():
        old=verified(roundtrip)
        if old['predictor_lock']!=lock['identity']:raise ValueError('Released head lock changed')
        return old
    index=ConsumptionIndex();rows={};refs={}
    for name,row in lock['heads'].items():
        source=row['checkpoint'];index.identity(source['path'],source)
        checkpoint=torch.load(source['path'],map_location='cpu',weights_only=False);cfg=checkpoint['config']
        state=checkpoint['model'];deps={}
        if cfg['stage']=='R':
            model={k:v for k,v in state.items() if not k.startswith('initial.')}
            if cfg['residual']:
                initial={k.removeprefix('initial.'):v for k,v in state.items() if k.startswith('initial.')}
                digest=model_digest(heads[name].initial)
                if digest not in refs:
                    p=folder/'weights'/f'initial_seed{row["seed"]}.pt';p.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(model=initial,initial_state_digest=digest),p)
                    refs[digest]=dict(index.identity(p),relative_path=str(p.relative_to(folder)))
                deps['initial']=refs[digest]
        else:
            model={k:v for k,v in state.items() if not k.startswith('base.')}
            base_name=('R29_' if row['seed']==29 else '')+lock['Rstar'];deps['base']=base_name
            bound_base=cfg['base_checkpoint'];index.identity(bound_base['path'],bound_base)
            original_base=torch.load(bound_base['path'],map_location='cpu',weights_only=False)['model']
            if any(not torch.equal(v,state['base.'+k]) for k,v in original_base.items()):
                raise ValueError('G checkpoint moved its frozen qualified Rstar')
        p=folder/'weights'/(name+'.pt');p.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=model,config=cfg,seed=row['seed'],selected_step=checkpoint['step']),p)
        weights=dict(index.identity(p),relative_path=str(p.relative_to(folder)))
        if weights['bytes']>=100*2**20:raise ValueError('Inference head exceeds normal Git size limit')
        ledger=root/f'training/seed{row["seed"]}'/row['branch']/'updates.jsonl'
        logged=json.loads(ledger.read_text().splitlines()[-1])
        active=sum(v.numel() for k,v in model.items() if logged['gradients'].get(k,0)>0)
        rows[name]=dict(weights=weights,config=cfg,dependencies=deps,selected_step=checkpoint['step'],
                       training_checkpoint=source,trainable_stored_parameters=sum(v.numel() for v in model.values()),
                       nonzero_gradient_parameters_at_last_logged_update=active,last_logged_gradient_step=logged['step'],
                       frozen_initial_parameters=sum(p.numel() for p in heads[name].initial.parameters()) if cfg['stage']=='R' and cfg['residual'] else 0,
                       frozen_base_parameters=sum(p.numel() for p in heads[name].base.parameters()) if cfg['stage']=='G' else 0)
    registry=write(path,dict(status='COMPLETE',heads=rows,shared_initial_references=refs,predictor_lock=lock['identity'],
                            FC_identity=fc.model_key,text_identity=fc.session.text_identity,frozen_backbone_included=False,
                            frozen_FC_parameters=sum(p.numel() for p in fc.model.parameters()),
                            frozen_text_elements=fc.text_tensor.numel(),
                            loader='static_ovmap.preservation_routing.export.load_exported',FP32=True))
    sample=sorted([o for o in objects(binding,'dev') if o['base']],key=lambda o:o['key'])[:2];checks={}
    with torch.no_grad():
        for name,head in heads.items():
            restored=load_exported(path,name,fc.device,FC_identity=fc.model_key,text_identity=fc.session.text_identity)
            errors=[]
            for obj in sample:
                x,_,_=loader.load(obj,0,8);a=head(x,fc.phi)['embedding'];b=restored(x,fc.phi)['embedding']
                torch.testing.assert_close(a,b,rtol=0,atol=0);errors.append(float((a-b).abs().max()))
            checks[name]=dict(sample_keys=[o['key'] for o in sample],max_abs=max(errors),strict_load=True,bitwise=True)
    return write(roundtrip,dict(status='COMPLETE',predictor_lock=lock['identity'],registry=registry['identity'],checks=checks,
                              all_selected_heads=len(checks),actual_frozen_phi=True,source_code_shape_shared=True))
