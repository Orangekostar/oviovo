"""Two real TRAIN objects; twenty discarded active-eta CUDA updates."""
from pathlib import Path
import time
import torch
from static_ovmap.learned_object_readout.training import optimizer_for,model_digest
from static_ovmap.learned_object_readout.engineering import frozen_digest
from .common import objects,verified,write
from .models import initialize,FrozenFCBase
from .losses import item_loss

def engineer(binding,fc,loader):
    root=Path(binding['output_root']);path=root/'engineering/receipt.json'
    if path.exists():return verified(path)
    selected=[o for o in objects(binding,'train') if o['base']][:2]
    if len(selected)!=2:raise ValueError('Two actual TRAIN originals required')
    started=time.perf_counter();before=frozen_digest(fc)
    batches=[(loader.load(o,0,2)[0],loader.load(o,3,2)[0],
              [loader.targets(o,0,2)[1],loader.targets(o,3,2)[1]]) for o in selected]
    report={};initials=[]
    for cfg in binding['specification']['methods']:
        head=initialize(17,cfg).to(fc.device);initials.append(model_digest(head));opt=optimizer_for(head)
        gradients=[];losses=[]
        for step in range(5):
            obj=selected[step%2];x,xb,ys=batches[step%2]
            if x['local_raw'].shape[-1]!=1536 or x['local_unit'].shape[-1]!=768:raise ValueError('Actual FC channels changed')
            with torch.no_grad():
                base=FrozenFCBase()(x)
                for kwargs in ({'eta_override':0.},{'rho_override':0.}):
                    if not torch.equal(head(x,fc.phi,**kwargs)['embedding'],base):raise ValueError('Actual disabled route is not exact FC')
                missing=dict(x,local_valid=torch.zeros_like(x['local_valid']))
                if not torch.equal(head(missing,fc.phi)['embedding'],base):raise ValueError('Actual missing-local FC parity failed')
            opt.zero_grad(set_to_none=True);a=head(x,fc.phi,eta_override=.5);b=head(xb,fc.phi,eta_override=.5)
            loss,parts=item_loss(a,b,fc.base_targets[obj['class_id']],fc.base_text,memberships=ys,membership_enabled=cfg['membership_loss'])
            loss.backward();grad={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None};gradients.append(grad)
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None):raise ValueError('Actual nonfinite gradient/loss')
            if not cfg['membership_loss'] and any(p.grad is not None for p in head.membership.parameters()):raise ValueError('Disabled membership gradient')
            torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step();losses.append(float(loss.detach()))
        for name in ('embed.0.weight','quality.0.weight','queries'):
            if not any(g.get(name,0)>0 for g in gradients):raise ValueError('Actual local input derivative missing: '+name)
        if cfg['membership_loss'] and not any(g.get('membership.2.weight',0)>0 for g in gradients):raise ValueError('Membership derivative missing')
        report[cfg['id']]=dict(gradients=gradients,losses=losses,discarded_updates=5)
    if len(set(initials))!=1:raise ValueError('Actual initial state differs')
    # Explicit vector phi derivative through the real frozen projection.
    x=batches[0][0];raw=x['local_raw'][x['local_valid']][:2].detach().clone().requires_grad_()
    if len(raw)<2:raise ValueError('Pilot lacks two visible observations')
    projected=fc.phi(raw);(projected*torch.arange(projected.shape[-1],device=fc.device)).sum().backward()
    if raw.grad is None or not torch.isfinite(raw.grad).all() or not raw.grad.norm()>0:raise ValueError('Frozen phi input autograd blocked')
    after=frozen_digest(fc)
    if before!=after or any(p.grad is not None or p.requires_grad for p in fc.model.parameters()):raise ValueError('Frozen FC changed')
    fc.index.write_memo(root/'features/verifications.json')
    return write(path,dict(status='COMPLETE',objects=[o['key'] for o in selected],discarded_updates=20,scientific_updates=0,
          arms=report,frozen_before=before,frozen_after=after,actual_phi_input_gradient=float(raw.grad.norm()),
          actual_ordinary_FC_identities=True,matched_initial_state=initials[0],raw_C=1536,projected_D=768,
          new_physical_image_encodings=0,elapsed_seconds=time.perf_counter()-started,FP32=True,TF32=False))

def sparse_parity(binding,fc,loader):
    """Actual output and projection-input gradients; zero optimizer updates."""
    from .data import CachedObjectLoader
    root=Path(binding['output_root']);path=root/'engineering/sparse_loader_parity.json'
    if path.exists():return verified(path)
    cached=CachedObjectLoader(fc,verified(root/'features/train-dev.json'),require_parity=False)
    chosen=[o for o in objects(binding,'train') if o['base']][:2];checks=[]
    head=initialize(17,binding['specification']['methods'][-1]).to(fc.device)
    for obj in chosen:
        for condition in (0,3):
            for prefix in (2,4,8):
                original,_,audit=loader.load(obj,condition,prefix);restored,_,other=cached.load(obj,condition,prefix)
                for k in original:torch.testing.assert_close(original[k],restored[k],atol=0,rtol=0)
                if audit!=other:raise ValueError('Sparse cache changed ordinary region audits')
                head.zero_grad(set_to_none=True);a=head(original,fc.phi);a['embedding'][0].backward()
                grad={n:p.grad.detach().clone() for n,p in head.named_parameters() if p.grad is not None}
                head.zero_grad(set_to_none=True);b=head(restored,fc.phi);b['embedding'][0].backward()
                torch.testing.assert_close(a['embedding'],b['embedding'],atol=0,rtol=0)
                for n,p in head.named_parameters():
                    if n in grad:torch.testing.assert_close(grad[n],p.grad,atol=0,rtol=0)
                # Real nonlinear phi's input derivative also survives CPU/disk transport.
                raw=original['local_raw'][original['local_valid']][:2].detach().clone().requires_grad_()
                copy=restored['local_raw'][restored['local_valid']][:2].detach().clone().requires_grad_()
                weights=torch.arange(768,device=fc.device)
                (fc.phi(raw)*weights).sum().backward();(fc.phi(copy)*weights).sum().backward()
                torch.testing.assert_close(raw.grad,copy.grad,atol=0,rtol=0)
                if not raw.grad.norm()>0:raise ValueError('Sparse phi parity uses a zero derivative')
                checks.append(dict(key=obj['key'],condition=condition,prefix=prefix,bitwise_inputs=True,bitwise_output=True,
                                   bitwise_gradients=True,phi_input_gradient_norm=float(raw.grad.norm())))
    cached.clear();head.zero_grad(set_to_none=True)
    return write(path,dict(status='COMPLETE',FC_identity=fc.model_key,checks=checks,discarded_updates=0,
        original_loader_is_sole_producer=True,shared_all_arms=True,dense_host_GiB=8,sparse_host_GiB=24))
