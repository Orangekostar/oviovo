"""One real frozen-FC gradient check, without new image encodings."""
from pathlib import Path
import torch
from .common import verified,immutable_record,existing,unit_key,objects
from .models import make_models

def check_gradient(binding,fc,loader,budget):
    root=Path(binding['output_root']);predictors=verified(root/'predictor_lock.json')
    identity=unit_key('real_FC_gradient',predictors['identity'],producers=('engineering','models','data'))
    path=root/'engineering/real_FC_gradient.json';old=existing(path,identity)
    if old:return old
    obj=next(o for o in objects(binding,'train') if o['base']);x,_,_=loader.load(obj,0,8)
    raw=x['local_raw'][x['local_valid']].detach()[:16]
    if len(raw)<2:raise ValueError('Real gradient probe requires two valid cached local observations')
    weights=torch.zeros(len(raw),device=fc.device,requires_grad=True)
    value=fc.phi((torch.softmax(weights,0)@raw)[None])
    probe=(value[0]*fc.base_text[fc.base_targets[obj['class_id']]]).sum();probe.backward()
    if weights.grad is None or not bool(torch.isfinite(weights.grad).all()) or not bool(weights.grad.abs().max()>0):raise ValueError('Real FC phi lost its input gradient')
    row=predictors['arms']['S_LL:seed17']['selected'];state=torch.load(row['model']['path'],map_location='cpu',weights_only=False)
    head=make_models(17)['S_LL'].to(fc.device);head.load_state_dict(state['model'],strict=True)
    budget.consume('engineering_head');out=head(x,fc.phi,training_forward=True,completed_steps=2)
    loss=(out['embedding']*fc.base_text[fc.base_targets[obj['class_id']]]).sum();loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None}
    if not any(n.startswith('quality.') and v>0 for n,v in gradients.items()):raise ValueError('Real FC head quality has no effective gradient')
    if any(p.grad is not None for p in head.membership.parameters()):raise ValueError('Disabled membership acquired a gradient')
    if any(p.grad is not None or p.requires_grad for p in fc.model.parameters()):raise ValueError('Frozen FC changed its gradient boundary')
    if fc.images:raise ValueError('Engineering check encoded an old training image')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,object_key=obj['key'],input=obj['inputs'],model=row['model'],FC=fc.model_key,ordinary_tensor=not torch.is_inference(value),phi_input_gradient_max=float(weights.grad.abs().max()),head_gradients=gradients,frozen_FC_parameter_gradients=False,membership_parameter_gradients=False,head_calls=1,standalone_phi_calls=1,new_image_encodings=0,scientific_result=False))
