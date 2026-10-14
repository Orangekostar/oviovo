"""Native one-view support edits; one local subset shared by L/B."""
import torch
from static_ovmap.frozen_fc_routing.models import FrozenFCBase
from .subsets import validate_inputs,subset_inputs
from .decomposition import compose

def build_actions(x,full,evaluate_subset):
    validate_inputs(x);views=torch.nonzero(x['view_valid'],as_tuple=False).flatten().tolist()
    out=[dict(kind='KEEP',frame_id=-1,view_index=None,edited=False,noop_proof=None,embedding=full['embedding'],base=full['base'],local=full)]
    if len(views)<3:return out
    for v in views:
        sub=subset_inputs(x,[i for i in views if i!=v]);base=FrozenFCBase()(sub)
        unsupported=not bool(x['local_valid'][v].any())
        local=full if unsupported else evaluate_subset(sub)
        for kind in ('G','L','B'):
            supplied=base if kind in ('G','B') else full['base']
            state=full if kind=='G' else local
            noop=kind=='L' and unsupported
            z=full['embedding'] if noop else (local['embedding'] if kind=='B' and not unsupported else compose(supplied,state))
            out.append(dict(kind=kind,frame_id=int(x['frame_ids'][v]),view_index=v,edited=not noop,noop_proof='REMOVED_VIEW_HAS_ZERO_VALID_LOCAL_SITES' if noop else None,embedding=z,base=supplied,local=state))
    return out
