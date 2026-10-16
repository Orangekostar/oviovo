"""Uniform frozen raw-surface evidence, independent of labels and network weights."""
import torch
from static_ovmap.structure_confirmation_v1.models import _base_result
from .constraints import prepare_dictionary

@torch.no_grad()
def build_dictionary(inputs,phi):
    out=_base_result(inputs)
    if out['base'] is None:return dict(base=None,empty=True,zero_norm_groups=0)
    b=out['base'];views=out['view_indices'];valid=out['local_valid'];raw=inputs['local_raw'][views]
    if not torch.isfinite(raw[valid]).all():raise ValueError('Nonfinite valid raw evidence')
    counts=valid.sum(0);present=counts>0
    if not bool(present.any()):return dict(base=b,center=b,empty=True,zero_norm_groups=0)
    pooled=torch.where(valid[...,None],raw,0).sum(0)[present]/counts[present,None]
    u=phi(pooled)
    if not torch.isfinite(u).all():raise ValueError('Nonfinite projected evidence')
    norms=u.norm(dim=-1);keep=norms>0;zero=int((~keep).sum());u=u[keep]/norms[keep,None]
    if not len(u):return dict(base=b,center=b,empty=True,zero_norm_groups=zero)
    D=torch.cat((b[None],u));a=torch.full((len(D),),.02/len(u),device=D.device,dtype=D.dtype);a[0]=.98
    return dict(**prepare_dictionary(D,a),base=b,center=a@D,empty=False,zero_norm_groups=zero)
