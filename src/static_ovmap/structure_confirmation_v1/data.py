"""Exact parent tensor cache with task-owned parity and bounded shared LRUs."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from static_ovmap.surface_bias_wave1_v2.data import ObjectLoader as ParentLoader
from .common import FORWARD_KEYS,canonical_digest,verified,write,arrays_record,objects

class SparseInputs(ParentLoader):
    def __init__(self,fc,features):
        super().__init__(fc,features,maximum_GiB=8)
        self.feature_identity=features['identity']
        self.namespace=canonical_digest(dict(features=self.feature_identity,FC=fc.model_key,projection=fc.dense_operator,format='EXACT_PARENT_TENSORS_V1'))
        self.sparse=OrderedDict();self.sparse_bytes=0;self.sparse_limit=24*2**30
        self.stats=dict(memory_hits=0,parent_disk_hits=0,task_disk_hits=0,parent_tensor_builds=0)

    def load(self,obj,condition=0,requested_views=8):
        if requested_views not in (2,4,8):raise ValueError('Fixed prefix required')
        key=canonical_digest(dict(namespace=self.namespace,object_inputs=obj['inputs']['sha256'],object_key=obj['key'],views=obj['views'],condition=int(condition)))
        if key in self.sparse:
            tensors,audit=self.sparse.pop(key);self.sparse[key]=(tensors,audit);self.stats['memory_hits']+=1
        else:
            roots=[Path(self.fc.binding['output_root']),Path(self.fc.binding['fg_parent_root'])]
            paths=[r/'features/sparse'/self.namespace/(key+'.json') for r in roots]
            path=next((p for p in paths if p.exists()),None)
            if path:
                row=verified(path)
                if row['key']!=key or row['namespace']!=self.namespace or row['feature_identity']!=self.feature_identity or row['source_inputs']['sha256']!=obj['inputs']['sha256']:raise ValueError('Sparse source identity differs')
                self.fc.index.identity(row['arrays']['path'],row['arrays'])
                with np.load(row['arrays']['path'],allow_pickle=False) as a:
                    if set(a.files)!=FORWARD_KEYS:raise ValueError('Undeclared sparse tensor field')
                    tensors={k:torch.from_numpy(a[k].copy()) for k in FORWARD_KEYS}
                audit=row['region_audits'];self.stats['task_disk_hits' if path==paths[0] else 'parent_disk_hits']+=1
            else:
                x,_,audit=super().load(obj,int(condition),8)
                tensors={k:v.detach().cpu().clone() for k,v in x.items()};self.stats['parent_tensor_builds']+=1
                path=paths[0];arrays=arrays_record(path.with_suffix('.npz'),{k:v.numpy() for k,v in tensors.items()},self.fc.index)
                write(path,dict(status='COMPLETE',key=key,arrays=arrays,namespace=self.namespace,feature_identity=self.feature_identity,source_inputs=obj['inputs'],region_audits=audit,new_image_encodings=0,GT_consumed=False))
            size=sum(t.numel()*t.element_size() for t in tensors.values())
            while self.sparse and self.sparse_bytes+size>self.sparse_limit:
                _,(old,_)=self.sparse.popitem(last=False);self.sparse_bytes-=sum(t.numel()*t.element_size() for t in old.values())
            self.sparse[key]=(tensors,audit);self.sparse_bytes+=size
        x={k:v[:requested_views].to(self.fc.device) for k,v in tensors.items()}
        return x,x['fc_vectors'],audit[:requested_views]

    def clear(self):
        self.cache.clear();self.bytes=0;self.sparse.clear();self.sparse_bytes=0

def prepare(binding,fc,loader,budget):
    from .common import unit_key,existing
    from .models import make_models
    from static_ovmap.frozen_fc_routing.models import initialize
    root=Path(binding['output_root']);identity=unit_key('prepare_cache',binding['identity'],producers=('data','models','grouping'),FC=fc.model_key,features=loader.feature_identity)
    path=root/'prepare_cache.json';old=existing(path,identity)
    if old:return old
    obj=next(o for o in objects(binding,'train') if o['base'])
    cached,_,_=loader.load(obj,0,8);reference,_,_=ParentLoader.load(loader,obj,0,8)
    for k in FORWARD_KEYS:
        if not torch.equal(cached[k],reference[k]):raise ValueError('Actual parent tensor parity failed: '+k)
    heads=make_models(17);parity={}
    with torch.no_grad():
        for name,group in [('S_LL','SURFACE'),('V_LL','VIEW')]:
            parent=initialize(17,dict(grouping=group,direct_routing=False)).to(fc.device)
            h=heads[name].to(fc.device);budget.consume('engineering_head',2)
            a=h(cached,fc.phi)['embedding'];b=parent(cached,fc.phi)['embedding']
            if not torch.equal(a,b):raise ValueError('Real FC forward parity failed: '+name)
            parity[name]=dict(bitwise_equal=True,max_absolute_error=float((a-b).abs().max()))
    if fc.images:raise ValueError('Training cache preparation encoded new RGB')
    fc.index.write_memo(root/'features/verifications.json')
    return write(path,dict(status='COMPLETE',input_identity=identity,FC_identity=fc.model_key,features=loader.feature_identity,sparse_namespace=loader.namespace,actual_parent_tensor_parity=True,forward_parity=parity,source_grid_encodings=0,loader_stats=loader.stats))
