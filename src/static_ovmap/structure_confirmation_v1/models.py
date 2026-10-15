"""The eight frozen-contract trainable readers and three static controls."""
import copy
import math
import torch
from torch import nn
from static_ovmap.frozen_fc_routing.models import FrozenFCBase,initialize,unit,route,correct
from .common import FORWARD_KEYS,METHODS,STATICS

def _base_result(inputs):
    if set(inputs)!=FORWARD_KEYS or any(not isinstance(v,torch.Tensor) for v in inputs.values()):raise ValueError('Forward requires exactly seven tensor keys')
    views=torch.nonzero(inputs['view_valid'].bool(),as_tuple=False).flatten()
    if len(views)>8 or inputs['local_valid'].ndim!=2 or inputs['local_valid'].shape[1]>64:raise ValueError('Fixed view/slot limits exceeded')
    base=FrozenFCBase()(inputs) if len(views) else None
    return dict(base=base,embedding=base,fallback=None if len(views) else 'NO_OBSERVATION',view_indices=views,frame_ids=inputs['frame_ids'][views],local_valid=inputs['local_valid'][views].bool(),group_count=0)

class FGReadout(nn.Module):
    def __init__(self,method,template):
        super().__init__()
        for name,module in template.named_children():self.add_module(name,copy.deepcopy(module))
        self.queries=nn.Parameter(template.queries.detach().clone())
        self.method=method;self.grouping='VIEW' if method=='V_LL' else 'SURFACE';self.direct_routing=False
        self.membership.requires_grad_(False)
        if method in ('S_UL','S_UU'):self.quality.requires_grad_(False)

    def forward(self,inputs,phi,*,completed_steps=0,training_forward=False,eta_override=None):
        out=_base_result(inputs)
        if out['base'] is None:return out
        base=out['base'];views=out['view_indices'];valid=out['local_valid']
        raw=torch.where(valid[...,None],inputs['local_raw'][views].detach(),0)
        local=torch.where(valid[...,None],inputs['local_unit'][views].detach(),0)
        meta=torch.where(valid[...,None],inputs['metadata'][views].detach(),0)
        if not all(bool(torch.isfinite(t).all()) for t in (raw,local,meta)):raise ValueError('Nonfinite valid local input')
        h=self.embed(torch.cat((local,meta),-1))
        uniform_inner=self.method in ('S_UL','S_UU')
        q=None if uniform_inner else self.quality(h).squeeze(-1)
        eta=(.5*min(1.,completed_steps/200) if training_forward else .5) if eta_override is None else float(eta_override)
        if not 0<=eta<=.5:raise ValueError('eta outside fixed range')
        out.update(eta=eta,local_hidden=h,quality_logits=q,delta=torch.zeros_like(base))
        if not bool(valid.any()):
            out.update(embedding=base+(h.sum()+(q.sum() if q is not None else 0))*0,fallback='NO_LOCAL_SUPPORT_FC_FALLBACK');return out
        nv,ns=valid.shape
        ids=(torch.arange(nv,device=raw.device)[:,None].expand(nv,ns) if self.grouping=='VIEW' else torch.arange(ns,device=raw.device)[None,:].expand(nv,ns))
        flat,fv=ids.flatten(),valid.flatten();present=torch.unique(flat[fv],sorted=True);members=(present[:,None]==flat[None,:])&fv[None,:]
        a=members.to(h.dtype)/members.sum(-1,keepdim=True) if uniform_inner else torch.softmax(q.flatten()[None].expand(len(present),-1).masked_fill(~members,-torch.inf),-1)
        hbar=a@h.reshape(-1,h.shape[-1]);keys=self.group_norm(hbar+self.group_ff(hbar))
        if self.method=='S_PRE':
            rows=raw.reshape(-1,raw.shape[-1]);projected=phi(rows[fv]);scattered=projected.new_zeros(len(rows),projected.shape[-1]);scattered[fv]=projected;u=unit(a@scattered)
        else:u=unit(phi(a@raw.reshape(-1,raw.shape[-1])))
        e=4*torch.tanh(self.queries@keys.T/math.sqrt(keys.shape[-1]));rho=torch.ones(len(present),dtype=h.dtype,device=h.device)
        alpha,null=route(e,rho);mass=alpha.sum(-1,keepdim=True)
        if self.method in ('S_LU','S_UU'):alpha=mass.expand_as(alpha)/len(present)
        values=u-base;z,delta=correct(base,values,alpha,eta)
        out.update(embedding=z,delta=delta,effective_rho=rho,alpha=alpha,null=null,routing_mass=mass,energies=e,values=values,group_vectors=u,quality_weights=a,members=members,group_ids=present,group_count=len(present))
        return out

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(256,eps=1e-5);self.attention=nn.MultiheadAttention(256,8,dropout=0.,bias=True,batch_first=True);self.norm2=nn.LayerNorm(256,eps=1e-5);self.ff=nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))
    def forward(self,h):
        q=self.norm1(h);h=h+self.attention(q,q,q,need_weights=False)[0];return h+self.ff(self.norm2(h))

class MVReadout(nn.Module):
    def __init__(self):
        super().__init__();self.input_projection=nn.Linear(768,256);self.blocks=nn.ModuleList([AttentionBlock() for _ in range(4)]);self.latent=nn.Parameter(torch.empty(1,1,256));nn.init.normal_(self.latent,0,.02)
        self.query_norm=nn.LayerNorm(256,eps=1e-5);self.token_norm=nn.LayerNorm(256,eps=1e-5);self.pool=nn.MultiheadAttention(256,8,dropout=0.,bias=True,batch_first=True);self.output_norm=nn.LayerNorm(256,eps=1e-5);self.output_projection=nn.Linear(256,768);nn.init.zeros_(self.output_projection.weight);nn.init.zeros_(self.output_projection.bias)
    def forward(self,inputs,phi=None,*,completed_steps=0,training_forward=False,eta_override=None):
        out=_base_result(inputs)
        if out['base'] is None:return out
        h=self.input_projection(unit(inputs['fc_vectors'][out['view_indices']].detach()))[None]
        for block in self.blocks:h=block(h)
        q=self.query_norm(self.latent);kv=self.token_norm(h);pooled=self.latent+self.pool(q,kv,kv,need_weights=False)[0];r=self.output_projection(self.output_norm(pooled[0,0]))
        eta=(.5*min(1.,completed_steps/200) if training_forward else .5) if eta_override is None else float(eta_override)
        if not 0<=eta<=.5:raise ValueError('eta outside fixed range')
        delta=eta*r;out.update(embedding=unit(out['base']+delta),delta=delta,eta=eta,group_count=None,group_scope='NOT_APPLICABLE_VIEW_ATTENTION');return out

class StaticReadout(nn.Module):
    def __init__(self,name):
        super().__init__()
        if name not in STATICS:raise ValueError('Unknown static method')
        self.method=name
    def forward(self,inputs,phi=None,**kwargs):
        out=_base_result(inputs)
        if out['base'] is None or self.method=='B_FC':return out
        valid=out['local_valid'];views=out['view_indices']
        if self.method=='B_PFE_ADAPTED':
            visible=valid&(inputs['metadata'][views,...,0]>=.5);present=visible.any(0)
            if not bool(present.any()):out['fallback']='NO_VISIBLE_MASK_SITES_FC_FALLBACK';return out
            f=unit(inputs['fc_vectors'][views].detach());weights=visible[:,present].T.to(f.dtype);pooled=weights@f/weights.sum(-1,keepdim=True)
        else:
            present=valid.any(0)
            if not bool(present.any()):out['fallback']='NO_LOCAL_SUPPORT_FC_FALLBACK';return out
            raw=torch.where(valid[...,None],inputs['local_raw'][views].detach(),0);pooled=phi((raw.sum(0)/valid.sum(0).clamp_min(1)[:,None])[present])
        if not bool(torch.isfinite(pooled).all()):raise ValueError('Nonfinite static representation')
        if bool((torch.linalg.vector_norm(pooled,dim=-1)<=1e-12).any()):out['fallback']='DEGENERATE_GROUP_FC_FALLBACK';return out
        vectors=unit(pooled);mean=vectors.mean(0)
        if float(torch.linalg.vector_norm(mean))<=1e-12:out['fallback']='DEGENERATE_MEAN_FC_FALLBACK';return out
        out.update(embedding=unit(mean),group_count=len(vectors),group_vectors=vectors);return out

def make_models(seed):
    template=initialize(seed,dict(grouping='SURFACE',direct_routing=False))
    result={name:FGReadout(name,template) for name in METHODS if name!='MV_ATTN'}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+701);result['MV_ATTN']=MVReadout()
    return {name:result[name] for name in METHODS}

def static_model(name):return StaticReadout(name)
