"""Thin compatibility for optional diagnostic fields; parent math is unchanged.

The parent requires site_xyz/site_available even though its forward never reads
them. Complete six-field inputs use the unmodified parent loader. When only
those two fields are absent, use the same parent's signed_mask, sample_raw,
region_vector and phi operations below. No positions or labels are fabricated.
"""
import numpy as np
from static_ovmap.frozen_fc_routing.data import ObjectLoader as ParentLoader
from .common import FORWARD_KEYS

def forward_array_schema(fields):
    required={'masks','uv','metadata','local_valid'}
    allowed=required|{'site_xyz','site_available'}
    if not required<=set(fields) or not set(fields)<=allowed:
        raise ValueError('Forward observations missing or undeclared annotation/semantic field')
    return 'PARENT_EXACT' if set(fields)==allowed else 'DIAGNOSTIC_FIELDS_ABSENT'

class ObjectLoader(ParentLoader):
    def load(self,obj,condition=0,requested_views=8):
        import torch
        from static_ovmap.a7_evidence_upgrade.region_adapter import signed_mask
        from static_ovmap.learned_object_readout.observations import sample_raw
        from static_ovmap.cvpr_compact.area_fallback import region_vector
        obj=self.resolver.rewrite(obj)
        self.fc.index.identity(obj['inputs']['path'],obj['inputs'])
        with np.load(obj['inputs']['path'],allow_pickle=False) as arrays:
            schema=forward_array_schema(arrays.files)
            if schema=='PARENT_EXACT': return super().load(obj,condition,requested_views)
            masks=arrays['masks'][condition,:requested_views].copy()
            uv=arrays['uv'][condition,:requested_views].copy()
            metadata=arrays['metadata'][condition,:requested_views].copy()
            valid=arrays['local_valid'][condition,:requested_views].copy()
        raws,local_raw,local_unit,fc_vectors,fallback=[],[],[],[],[]
        with torch.no_grad():
            for v,frame in enumerate(obj['views'][:requested_views]):
                key=obj['scene']+':'+str(frame['frame_id']); record=self.frames[key]; grid=self._grid(key)
                raw=grid['raw'].to(self.fc.device); raws.append(raw[0])
                signed,_,_=signed_mask(masks[v],record['resized_hw'],record['padded_hw'],record['raw_hw'],self.fc.device)
                sampled,bounds=sample_raw(raw,uv[v],record['original_hw'],record['resized_hw'],record['padded_hw'])
                visibility=torch.from_numpy(valid[v]).to(self.fc.device)&bounds
                sampled=torch.where(visibility[:,None],sampled,0)
                unit=torch.nn.functional.normalize(self.fc.phi(sampled),dim=-1)
                local_raw.append(sampled); local_unit.append(torch.where(visibility[:,None],unit,0)); valid[v]=visibility.cpu().numpy()
                vector,audit=region_vector(self.fc.model,self.fc.operators,raw,signed)
                fc_vectors.append(vector); fallback.append(audit)
        vectors=torch.stack(fc_vectors)
        x=dict(fc_vectors=vectors,local_raw=torch.stack(local_raw),local_unit=torch.stack(local_unit),metadata=torch.from_numpy(metadata).to(self.fc.device),local_valid=torch.from_numpy(valid).to(self.fc.device),view_valid=torch.ones(len(raws),dtype=torch.bool,device=self.fc.device),frame_ids=torch.tensor([v['frame_id'] for v in obj['views'][:requested_views]],device=self.fc.device))
        self.workload['parent_loader_calls']+=1; self.workload['ordinary_FC_view_readouts']+=len(vectors)
        self.workload['local_phi_calls']+=len(vectors); self.workload['local_phi_vectors']+=int(x['local_valid'].numel())
        return {k:x[k] for k in FORWARD_KEYS},vectors,fallback
