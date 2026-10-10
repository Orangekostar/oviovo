"""Shared production kernels for training, inference and numerical checks."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from static_ovmap.learned_object_readout.model import ReadoutHead,unit

def teacher(inputs):
    return unit(inputs['fc_vectors'][inputs['view_valid']].mean(0))

def absolute_route(energies,rho):
    if energies.ndim!=2 or energies.shape[1]!=rho.numel() or not rho.numel():
        raise ValueError('Only nonempty observed groups may enter routing')
    if not torch.isfinite(energies).all() or not torch.isfinite(rho).all() or (rho<0).any():
        raise ValueError('Invalid routing scores')
    logits=energies+rho.log()[None]-math.log(rho.numel())
    a=torch.softmax(torch.cat((energies.new_zeros(energies.shape[0],1),logits),dim=1),dim=1)
    return a[:,1:],a[:,0]

class RHead(nn.Module):
    def __init__(self,residual=False):
        super().__init__();self.residual=bool(residual)
        self.ma=ReadoutHead('NONE')
        self.initial=copy.deepcopy(self.ma).eval().requires_grad_(False) if self.residual else None

    def train(self,mode=True):
        super().train(mode)
        if self.initial is not None:self.initial.eval()
        return self

    def forward(self,inputs,phi,*,reference_embedding=None):
        result=self.ma(inputs,phi)
        result['FC_embedding']=teacher(inputs)
        if self.residual:
            with torch.no_grad():
                initial=self.initial(inputs,phi)['embedding'] if reference_embedding is None else reference_embedding
            result['embedding']=unit(result['FC_embedding']+(result['embedding']-initial))
        return result

class GHead(nn.Module):
    def __init__(self,base,grouping='SURFACE',direct_routing=False):
        super().__init__()
        if grouping not in ('VIEW','SURFACE'):raise ValueError('Fixed grouping required')
        self.base=base.eval().requires_grad_(False);self.grouping=grouping;self.direct_routing=direct_routing
        self.embed=nn.Sequential(nn.Linear(776,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128))
        self.quality=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
        self.membership=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
        self.group_ff=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.group_norm=nn.LayerNorm(128)
        self.queries=nn.Parameter(torch.empty(4,128));nn.init.normal_(self.queries,0,.02)
        self.output=nn.Linear(768,768,bias=False);nn.init.zeros_(self.output.weight)

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,inputs,phi,*,rho_override=None):
        with torch.no_grad():base=self.base(inputs,phi)['embedding']
        views=torch.nonzero(inputs['view_valid'],as_tuple=False).flatten()
        valid=inputs['local_valid'][views].bool()
        raw=torch.where(valid[...,None],inputs['local_raw'][views],0)
        local=torch.where(valid[...,None],inputs['local_unit'][views],0)
        meta=torch.where(valid[...,None],inputs['metadata'][views],0)
        hidden=self.embed(torch.cat((local,meta),-1))
        q=self.quality(hidden).squeeze(-1);m=self.membership(hidden).squeeze(-1)
        result=dict(embedding=base,base_embedding=base,local_hidden=hidden,local_valid=valid,
                    quality_logits=q,membership_logits=m,view_indices=views,frame_ids=inputs['frame_ids'][views],
                    group_count=0,grouping=self.grouping)
        if not valid.any():return result
        nv,ns=valid.shape
        ids=(torch.arange(nv,device=raw.device)[:,None].expand(nv,ns) if self.grouping=='VIEW' else
             torch.arange(ns,device=raw.device)[None,:].expand(nv,ns))
        flat_ids=ids.flatten();fv=valid.flatten();present=torch.unique(flat_ids[fv],sorted=True)
        members=(present[:,None]==flat_ids[None,:])&fv[None,:]
        weights=torch.softmax(q.flatten()[None].expand(len(present),-1).masked_fill(~members,-torch.inf),dim=-1)
        hb=weights@hidden.reshape(-1,128);keys=self.group_norm(hb+self.group_ff(hb))
        values=unit(phi(weights@raw.reshape(-1,1536)))-base
        # logmean(sigmoid(m)) is stable without changing the declared mean.
        log_rho=torch.logsumexp(F.logsigmoid(m.flatten())[None].expand(len(present),-1).masked_fill(~members,-torch.inf),-1)-members.sum(-1).log()
        predicted=log_rho.exp()
        energies=4*torch.tanh(self.queries@keys.T/math.sqrt(128))
        if rho_override is not None:
            rho=torch.as_tensor(rho_override,device=raw.device,dtype=raw.dtype).expand_as(predicted)
            alpha,null=absolute_route(energies,rho)
        else:
            rho=predicted if self.direct_routing else torch.ones_like(predicted)
            logits=energies+(log_rho[None] if self.direct_routing else 0)-math.log(len(present))
            probs=torch.softmax(torch.cat((energies.new_zeros(4,1),logits),-1),-1)
            alpha,null=probs[:,1:],probs[:,0]
        delta_raw=(alpha@values).mean(0);delta=.5*self.output(delta_raw)
        result.update(embedding=unit(base+delta),group_count=len(present),group_ids=present,members=members,
                      quality_weights=weights,predicted_rho=predicted,effective_rho=rho,energies=energies,
                      alpha=alpha,null_weight=null,values=values,delta_raw=delta_raw,delta=delta)
        return result
