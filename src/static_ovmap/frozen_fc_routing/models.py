"""Shared production arithmetic for training, inference and interventions."""
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import FORWARD_KEYS

def unit(x):
    n=torch.linalg.vector_norm(x,dim=-1,keepdim=True)
    if not bool(torch.isfinite(x).all()) or bool((n<=1e-12).any()):raise ValueError('Nonfinite or zero representation')
    return x/n

class FrozenFCBase(nn.Module):
    def forward(self,inputs):
        if 'fc_vectors' not in inputs or 'view_valid' not in inputs:raise ValueError('Ordinary FC vectors/validity required')
        valid=inputs['view_valid'].bool()
        if not bool(valid.any()):raise ValueError('No real FC view')
        return unit(inputs['fc_vectors'][valid].detach().mean(0))

def route(e,rho):
    if e.ndim!=2 or rho.ndim!=1 or e.shape[1]!=rho.numel() or not rho.numel():raise ValueError('Nonempty query/group route required')
    if not bool(torch.isfinite(e).all()) or not bool(torch.isfinite(rho).all()):raise ValueError('Nonfinite route')
    if bool((e.abs()>4+1e-6).any()) or bool(((rho<0)|(rho>1)).any()):raise ValueError('Route outside fixed bounds')
    w=e.exp()*rho[None]/rho.numel();den=1+w.sum(-1,keepdim=True)
    return w/den,den.reciprocal().squeeze(-1)

def matched_mass(e,alpha):
    if e.shape!=alpha.shape:raise ValueError('Intervention group set differs')
    return alpha.sum(-1,keepdim=True)*torch.softmax(e,-1)

def correct(base,values,alpha,eta):
    if not 0<=eta<=.5:raise ValueError('eta outside [0,.5]')
    delta=eta*(alpha@values).mean(0)
    if eta==0 or not bool(alpha.detach().any()):return base+delta*0,delta
    return unit(base+delta),delta

def match_delta_norm(base,original,control,eps=1e-12):
    nr,nc=torch.linalg.vector_norm(original),torch.linalg.vector_norm(control)
    if float(nr)<=eps:return base,dict(status='ORIGINAL_ZERO',delta=torch.zeros_like(base))
    if float(nc)<=eps:return None,dict(status='UNDEFINED_ZERO_CONTROL',delta=None)
    delta=control*(nr/nc)
    return unit(base+delta),dict(status='MATCHED',delta=delta)

class FGHead(nn.Module):
    def __init__(self,grouping='SURFACE',direct_routing=False,*,dim=768,hidden=128):
        super().__init__()
        if grouping not in ('VIEW','SURFACE'):raise ValueError('Unknown grouping')
        self.grouping,self.direct_routing=grouping,bool(direct_routing)
        self.base=FrozenFCBase()
        self.embed=nn.Sequential(nn.Linear(dim+8,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.quality=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.membership=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.group_ff=nn.Sequential(nn.Linear(hidden,hidden*2),nn.GELU(),nn.Linear(hidden*2,hidden))
        self.group_norm=nn.LayerNorm(hidden)
        self.queries=nn.Parameter(torch.empty(4,hidden));nn.init.normal_(self.queries,0,.02)
        nn.init.zeros_(self.membership[-1].weight);nn.init.zeros_(self.membership[-1].bias)

    def forward(self,inputs,phi,*,completed_steps=0,training_forward=False,eta_override=None,rho_override=None):
        if set(inputs)!=FORWARD_KEYS:raise ValueError('Forward requires exactly the seven declared tensor keys')
        if any(not isinstance(v,torch.Tensor) for v in inputs.values()):raise ValueError('Forward accepts tensors only')
        base=self.base(inputs);views=torch.nonzero(inputs['view_valid'],as_tuple=False).flatten()
        valid=inputs['local_valid'][views].bool()
        if valid.ndim!=2 or valid.shape[0]>8 or valid.shape[1]>64:raise ValueError('Local bank dimensions outside fixed bounds')
        raw=torch.where(valid[...,None],inputs['local_raw'][views].detach(),0)
        local=torch.where(valid[...,None],inputs['local_unit'][views].detach(),0)
        meta=torch.where(valid[...,None],inputs['metadata'][views].detach(),0)
        if not all(bool(torch.isfinite(t).all()) for t in (raw,local,meta)):raise ValueError('Nonfinite valid observation')
        h=self.embed(torch.cat((local,meta),-1));q=self.quality(h).squeeze(-1);m=self.membership(h).squeeze(-1)
        eta=(.5*min(1.,completed_steps/200) if training_forward else .5) if eta_override is None else float(eta_override)
        if not 0<=eta<=.5:raise ValueError('eta outside fixed range')
        out=dict(base=base,embedding=base,local_valid=valid,local_hidden=h,membership_logits=m,quality_logits=q,
                 view_indices=views,frame_ids=inputs['frame_ids'][views],group_count=0,fallback=None,eta=eta,
                 delta=torch.zeros_like(base))
        if not bool(valid.any()):
            out['embedding']=base+(h.sum()+q.sum())*0
            out['fallback']='NO_LOCAL_SUPPORT_FC_FALLBACK'
            return out
        nv,ns=valid.shape
        ids=(torch.arange(nv,device=raw.device)[:,None].expand(nv,ns) if self.grouping=='VIEW' else
             torch.arange(ns,device=raw.device)[None,:].expand(nv,ns))
        flat,fv=ids.flatten(),valid.flatten();present=torch.unique(flat[fv],sorted=True)
        members=(present[:,None]==flat[None,:])&fv[None,:]
        a=torch.softmax(q.flatten()[None].expand(len(present),-1).masked_fill(~members,-torch.inf),-1)
        hbar=a@h.reshape(-1,h.shape[-1]);keys=self.group_norm(hbar+self.group_ff(hbar))
        u=unit(phi(a@raw.reshape(-1,raw.shape[-1])))
        values=u-base
        rho=(members.to(m.dtype)@m.sigmoid().flatten())/members.sum(-1)
        e=4*torch.tanh(self.queries@keys.T/math.sqrt(keys.shape[-1]))
        effective=rho if self.direct_routing else torch.ones_like(rho)
        if rho_override is not None:effective=torch.as_tensor(rho_override,dtype=rho.dtype,device=rho.device).expand_as(rho)
        alpha,null=route(e,effective);z,delta=correct(base,values,alpha,eta)
        out.update(embedding=z,delta=delta,rho=rho,effective_rho=effective,alpha=alpha,null=null,energies=e,
                   values=values,group_vectors=u,quality_weights=a,members=members,group_ids=present,group_count=len(present))
        return out

def initialize(seed,config):
    random.seed(seed+1009);np.random.seed(seed+1009);torch.manual_seed(seed+1009)
    return FGHead(config['grouping'],config['direct_routing'])

def interventions(result):
    names=('MATCHED_MASS','MATCHED_MASS_AND_DELTA_NORM','ALL_RHO_ONE','ALL_RHO_ZERO')
    base=result['base']
    if not result['group_count']:
        return {n:dict(embedding=base,delta=torch.zeros_like(base),paired_defined=True,status='NO_LOCAL_SUPPORT',
                       alpha=base.new_zeros(4,0),null=base.new_ones(4)) for n in names}
    e,a,v,eta=result['energies'],result['alpha'],result['values'],result['eta']
    mm=matched_mass(e,a);z,dm=correct(base,v,mm,eta)
    out={'MATCHED_MASS':dict(embedding=z,delta=dm,alpha=mm,null=result['null'],paired_defined=True,status='MATCHED')}
    zn,audit=match_delta_norm(base,result['delta'],dm)
    out[names[1]]=dict(embedding=result['embedding'] if zn is None else zn,delta=audit['delta'],alpha=mm,
                        null=result['null'],paired_defined=zn is not None,status=audit['status'])
    for name,rho in ((names[2],torch.ones_like(result['rho'])),(names[3],torch.zeros_like(result['rho']))):
        aa,nn=route(e,rho);zz,dd=correct(base,v,aa,eta)
        out[name]=dict(embedding=zz,delta=dd,alpha=aa,null=nn,paired_defined=True,status='DEFINED')
    return out
