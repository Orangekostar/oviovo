"""One view-axis transformation for all seven tensors, within one condition."""
import torch
from .common import FORWARD_KEYS,canonical_digest,_array_digest

def validate_inputs(x):
    if set(x)!=FORWARD_KEYS or any(not isinstance(v,torch.Tensor) for v in x.values()):raise ValueError('Exactly seven forward tensors required')
    n=x['view_valid'].numel()
    if not 1<=n<=8 or any(v.shape[0]!=n for v in x.values()):raise ValueError('All seven view axes must agree within1..8')
    if x['view_valid'].ndim!=1 or x['frame_ids'].ndim!=1 or x['local_valid'].ndim!=2:raise ValueError('Invalid tensor axes')
    if x['local_valid'].shape[1]>64 or not bool(x['view_valid'].any()):raise ValueError('Invalid support')
    ids=x['frame_ids'][x['view_valid'].bool()].tolist()
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate physical frame IDs')
    for key in ('local_raw','local_unit','metadata'):
        if x[key].shape[:2]!=x['local_valid'].shape:raise ValueError('Local site axes differ')
    return x

def subset_inputs(x,indices):
    validate_inputs(x);indices=list(indices)
    if not indices or len(indices)!=len(set(indices)) or min(indices)<0 or max(indices)>=len(x['view_valid']):raise ValueError('Invalid or duplicated subset view indices')
    index=torch.tensor(indices,device=x['frame_ids'].device,dtype=torch.long)
    return validate_inputs({k:v.index_select(0,index) for k,v in x.items()})

def state_key(x,condition):
    validate_inputs(x)
    return canonical_digest(dict(condition=condition,tensors={k:_array_digest(v.detach().cpu().numpy()) for k,v in x.items()}))
