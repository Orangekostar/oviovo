"""Two actual TRAIN originals, <=20 discarded updates; reset before science."""
from pathlib import Path
import time
import torch
from static_ovmap.learned_object_readout.engineering import frozen_digest,upstream_head
from static_ovmap.learned_object_readout.training import optimizer_for,model_digest
from .common import objects,verified,write
from .models import RHead,GHead,teacher
from .losses import item_loss
from .training import initialize,reset_rng

def engineer(binding,fc,loader):
    root=Path(binding['output_root']);path=root/'engineering/receipt.json'
    if path.exists():return verified(path)
    selected=[o for o in objects(binding,'train') if o['base']][:2]
    if len(selected)!=2:raise ValueError('Exactly two fixed TRAIN originals needed')
    start=time.perf_counter();before=frozen_digest(fc);head=initialize(17,True).to(fc.device);opt=optimizer_for(head)
    batches=[loader.load(o,0,2)[0] for o in selected]
    x=batches[0]
    if x['raw'].shape[1]!=1536 or x['projected'].shape[1]!=768:raise ValueError('Measured dimensions differ')
    parity=[];first_grad={};losses=[]
    for x in batches:
        with torch.no_grad():
            difference=float((head(x,fc.phi)['embedding']-teacher(x)).abs().max());parity.append(difference)
            if difference>1e-6:raise ValueError('Actual FC residual initialization failed')
    initial_ref=model_digest(head.initial)
    for step in range(16):
        opt.zero_grad(set_to_none=True);obj=selected[step%2];x=batches[step%2]
        xb=loader.load(obj,3,2)[0];a=head(x,fc.phi);b=head(xb,fc.phi);t=teacher(x)
        correct=fc.ids[int((t@fc.text_tensor.T).argmax())]==obj['class_id']
        loss,_=item_loss(a,b,fc.base_targets[obj['class_id']],fc.base_text,t,correct,True)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1)
        if step==0:
            first_grad={n:float(p.grad.norm()) for n,p in head.ma.named_parameters() if p.grad is not None}
            if not any(v>0 for v in first_grad.values()):raise ValueError('MA first backward blocked')
        opt.step();losses.append(float(loss.detach()))
    if model_digest(head.initial)!=initial_ref:raise ValueError('Frozen initial reference moved')
    g=GHead(head,'SURFACE',True).to(fc.device);op=optimizer_for(g);base_before=model_digest(g.base);gradients=[]
    for step in range(4):
        obj=selected[step%2];x=batches[step%2];xb=loader.load(obj,3,2)[0]
        op.zero_grad(set_to_none=True);a=g(x,fc.phi);b=g(xb,fc.phi)
        if step==0:
            with torch.no_grad():
                if not torch.equal(a['embedding'],g.base(x,fc.phi)['embedding']):raise ValueError('G zero-output identity failed')
        t=teacher(x);correct=fc.ids[int((t@fc.text_tensor.T).argmax())]==obj['class_id']
        loss,_=item_loss(a,b,fc.base_targets[obj['class_id']],fc.base_text,t,correct,True,
                        memberships=[loader.targets(obj,0,2)[1],loader.targets(obj,3,2)[1]],
                        config=dict(membership_loss=True,correspondence_loss=True))
        loss.backward();grad={n:float(p.grad.norm()) for n,p in g.named_parameters() if p.grad is not None};gradients.append(grad)
        if step==0 and grad.get('output.weight',0)<=0:raise ValueError('First W0 gradient missing')
        torch.nn.utils.clip_grad_norm_([p for p in g.parameters() if p.requires_grad],1);op.step()
    for name in ('embed.0.weight','quality.0.weight','queries','membership.0.weight'):
        if not any(r.get(name,0)>0 for r in gradients[1:]):raise ValueError('Actual later local gradient missing '+name)
    if model_digest(g.base)!=base_before:raise ValueError('Frozen G base moved')
    # Real upstream source arithmetic and actual feature values on CPU.
    original=upstream_head(root/'external/upstream_sources/MaskAdapter')('convnext_large_d_320',16,256,False,4)
    port=RHead(False).ma.ma;port.load_state_dict(original.state_dict(),strict=True)
    raw=x['projected'][:1].detach().cpu().clone().requires_grad_();other=raw.detach().clone().requires_grad_()
    masks=x['masks'][:1].cpu();a=original(raw,masks);b=port(other,masks)
    torch.testing.assert_close(a,b,rtol=0,atol=0);a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(raw.grad,other.grad,rtol=0,atol=0)
    after=frozen_digest(fc)
    if before!=after or any(p.grad is not None or p.requires_grad for p in fc.model.parameters()):raise ValueError('Frozen backbone changed')
    reset_rng(17);fc.index.write_memo(root/'features/verifications.json')
    return write(path,dict(status='COMPLETE',objects=[o['key'] for o in selected],discarded_updates=20,
            scientific_updates=0,raw_C=1536,projected_D=768,residual_max_abs=parity,first_MA_gradients=first_grad,
            G_gradients=gradients,initial_reference_unchanged=True,frozen_base_unchanged=True,
            frozen_before=before,frozen_after=after,fit_losses=losses,upstream_actual_tensor_CPU_bitwise=True,
            exact_parent_grid_reused=True,new_physical_image_encodings=0,elapsed_seconds=time.perf_counter()-start))
