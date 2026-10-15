"""Label-independent, prefix-consistent nonphysical matched-size groups."""
import hashlib
import json
import numpy as np

def permute_inputs(x,key,condition):
    import torch
    from .common import FORWARD_KEYS,CONDITIONS
    if set(x)!=FORWARD_KEYS:raise ValueError('Permutation requires exactly seven tensor keys')
    if condition not in CONDITIONS:raise ValueError('Unknown condition name')
    out={k:v.clone() for k,v in x.items()};mappings=[];fixed=total=0
    for view,frame in enumerate(x['frame_ids'].tolist()):
        valid=torch.nonzero(x['local_valid'][view],as_tuple=False).flatten()
        token=json.dumps(['FG2_GROUP_CONTROL_V1',str(key),condition,int(frame),9173],ensure_ascii=False,separators=(',',':'))
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode('utf8')).digest()[:8],'big'))
        source=valid[torch.as_tensor(rng.permutation(len(valid)),device=valid.device,dtype=torch.long)]
        for name in ('local_raw','local_unit','metadata'):out[name][view,valid]=x[name][view,source]
        pairs=list(zip(valid.tolist(),source.tolist()));mappings.append(pairs);fixed+=sum(a==b for a,b in pairs);total+=len(pairs)
    return out,dict(source_slots=mappings,valid_tokens=total,retained_same_slot_tokens=fixed,retained_same_slot_fraction=fixed/total if total else None,group_layout_unchanged=True)

def method_inputs(method,x,key,condition):
    return permute_inputs(x,key,condition) if method=='G_LL' else (x,None)
