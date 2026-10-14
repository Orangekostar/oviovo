"""The fixed target-free 16+8 action statistics; copies are normalized only."""
import numpy as np
import torch
from static_ovmap.frozen_fc_routing.models import unit
from .decomposition import local_statistics

def action_features(x,full,action,view_index):
    v=int(view_index);views=torch.nonzero(x['view_valid'],as_tuple=False).flatten();others=views[views!=v]
    regions=unit(x['fc_vectors'][views].detach());position=int(torch.nonzero(views==v).item());rv=regions[position]
    other=unit(x['fc_vectors'][others].detach());pair=other@rv
    loo=unit(x['fc_vectors'][others].detach().mean(0));b=full['base'];z=action['embedding'];za=full['embedding']
    m,g,_,_=local_statistics(full);ma,ga,_,_=local_statistics(action['local'])
    fields=[float(action['kind']==k) for k in ('G','L','B')]+[len(views)/8,float(pair.mean()),float(pair.min()),float(pair.std(unbiased=False)),float(rv@loo),float(action['base']@b),float(z@za),float(torch.linalg.vector_norm(z-za)),m,ma,float(torch.linalg.vector_norm(ga-g))]
    valid=x['local_valid'][v].bool();fi=int(torch.nonzero(full['view_indices']==v).item());q=full['quality_logits'][fi][valid]
    fields += [float(q.mean()) if q.numel() else 0.,float(q.std(unbiased=False)) if q.numel() else 0.]
    corr=[0.]*8
    if bool(valid.any()):
        other_valid=x['local_valid'][others].bool();shared=valid&other_valid.any(0);union=x['local_valid'][views].bool().any(0);exclusive=valid&~other_valid.any(0)
        meta=x['metadata'][v][valid];corr[:5]=[float(valid.sum())/valid.numel(),float(shared.sum())/float(valid.sum()),float(exclusive.sum())/float(union.sum()),float(meta[:,0].mean()),float(meta[:,1].mean())]
        if bool(shared.any()):
            other_units=x['local_unit'][others].detach();ov=other_valid
            # Normalize valid original copies, never write the bank.
            other_norm=torch.where(ov[...,None],other_units,0)
            norms=torch.linalg.vector_norm(other_norm,dim=-1)
            if bool((norms[ov]<=1e-12).any()) or not bool(torch.isfinite(other_norm).all()):raise ValueError('Invalid valid local vector')
            other_norm=other_norm/norms.clamp_min(1e-12)[...,None]
            mean=other_norm.sum(0)/ov.sum(0).clamp_min(1)[...,None]
            support=shared&(torch.linalg.vector_norm(mean,dim=-1)>1e-12)
            if bool(support.any()):
                disagreement=1-(unit(x['local_unit'][v][support].detach())*unit(mean[support])).sum(-1)
                corr[5:7]=[float(disagreement.mean()),float(disagreement.std(unbiased=False))]
            direction=unit(x['metadata'][v,shared,2:5].detach());dirs=x['metadata'][others][:,shared,2:5].detach();mask=ov[:,shared]
            length=torch.linalg.vector_norm(dirs,dim=-1)
            if bool((length[mask]<=1e-12).any()) or not bool(torch.isfinite(dirs[mask]).all()):raise ValueError('Invalid valid view direction')
            cosine=(dirs/length.clamp_min(1e-12)[...,None]*direction[None]).sum(-1).masked_fill(~mask,-torch.inf)
            corr[7]=float(cosine.max(0).values.mean())
    result=np.asarray(fields+corr,np.float32)
    if result.shape!=(24,) or not np.isfinite(result).all():raise ValueError('Nonfinite fixed24 action features')
    return result
