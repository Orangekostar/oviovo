"""Common untrained network definition; production training is conditional on A."""
import copy
import torch
from torch import nn
from static_ovmap.structure_confirmation_v1.models import AttentionBlock,_base_result
from .evidence import build_dictionary
from .constraints import radius,evidence

class Readout(nn.Module):
    def __init__(self,method='FREE'):
        super().__init__();self.method=method
        self.input_projection=nn.Linear(768,256);self.type_embedding=nn.Parameter(torch.empty(3,256))
        self.blocks=nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.latent=nn.Parameter(torch.empty(1,1,256));nn.init.normal_(self.latent,0,.02);nn.init.normal_(self.type_embedding,0,.02)
        self.query_norm=nn.LayerNorm(256,eps=1e-5);self.token_norm=nn.LayerNorm(256,eps=1e-5)
        self.pool=nn.MultiheadAttention(256,8,dropout=0.,bias=True,batch_first=True)
        self.output_norm=nn.LayerNorm(256,eps=1e-5);self.output_projection=nn.Linear(256,768)
        nn.init.zeros_(self.output_projection.weight);nn.init.zeros_(self.output_projection.bias)

    def forward(self,inputs,phi,*,dictionary=None,completed_steps=0,training_forward=False):
        out=_base_result(inputs)
        if out['base'] is None:return out
        dictionary=build_dictionary(inputs,phi) if dictionary is None else dictionary
        b=dictionary['base'];regions=torch.nn.functional.normalize(inputs['fc_vectors'][out['view_indices']].detach(),dim=-1)
        local=b.new_empty((0,768)) if dictionary['empty'] else dictionary['D'][1:]
        tokens=torch.cat((b[None],regions,local));types=torch.cat((torch.zeros(1,device=b.device,dtype=torch.long),
            torch.ones(len(regions),device=b.device,dtype=torch.long),torch.full((len(local),),2,device=b.device,dtype=torch.long)))
        h=(self.input_projection(tokens)+self.type_embedding[types])[None]
        for block in self.blocks:h=block(h)
        pooled=self.latent+self.pool(self.query_norm(self.latent),self.token_norm(h),self.token_norm(h),need_weights=False)[0]
        r=self.output_projection(self.output_norm(pooled[0,0]));eta=.5*min(1,completed_steps/200) if training_forward else .5
        q=dictionary['center']+eta*r
        if dictionary['empty']:
            out.update(embedding=b+0*q,fallback='NO_LOCAL_SUPPORT_FC_FALLBACK',preunit=b,q=q);return out
        if self.method=='FREE':y=q
        elif self.method=='RADIUS':y,_=radius(b,q)
        elif self.method=='EVIDENCE':
            projected=evidence(q,dictionary,128);y=projected['preunit'];out.update(projected)
        else:raise ValueError('Undefined output operator')
        out.update(embedding=torch.nn.functional.normalize(y,dim=0),preunit=y,q=q,eta=eta)
        return out

def make_models(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+3701);template=Readout()
    models={m:copy.deepcopy(template) for m in ('FREE','RADIUS','EVIDENCE')}
    for method,model in models.items():model.method=method
    return models
