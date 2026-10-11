"""Exact parent observations; tensor whitelist and unchanged prelocked plans."""
import shutil
from pathlib import Path
from .common import PathResolver,ConsumptionIndex,objects,verified,write,read,immutable_write
from .common import FORWARD_KEYS
from static_ovmap.preservation_routing.data import ObjectLoader as ParentLoader
from static_ovmap.learned_object_readout.features import FrozenFC,execution_config

class ObjectLoader(ParentLoader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.workload=dict(parent_loader_calls=0,ordinary_FC_view_readouts=0,local_phi_calls=0,local_phi_vectors=0)
    def load(self,obj,condition=0,requested_views=8):
        inputs,vectors,audit=super().load(obj,condition,requested_views)
        self.workload['parent_loader_calls']+=1
        self.workload['ordinary_FC_view_readouts']+=len(vectors)
        self.workload['local_phi_calls']+=len(vectors)
        self.workload['local_phi_vectors']+=int(inputs['local_valid'].numel())
        return {k:inputs[k] for k in FORWARD_KEYS},vectors,audit

def prepare(binding):
    root=Path(binding['output_root']);parent=Path(binding['pr_parent_root']);lr=Path(binding['lr_parent_root'])
    spec=binding['specification'];split=verified(root/'split_manifest.json');classes=verified(root/'class_split.json')
    for role,expected in spec['data']['roles'].items():
        if split['roles']['holdout' if role=='old_H' else role]!=expected:raise ValueError('Role names changed')
    items={r:objects(binding,r) for r in ('train','dev','holdout')}
    counts={r:len(v) for r,v in items.items()};bases={r:sum(o['base'] for o in v) for r,v in items.items()}
    if counts!={'train':785,'dev':87,'holdout':101} or bases['train']!=622 or bases['dev']!=69:raise ValueError('Fixed original support differs')
    observed=sorted({o['class_id'] for o in items['train'] if o['base']})
    if len(observed)!=74 or len({o['class_id'] for o in items['dev'] if o['base']})!=28:raise ValueError('Positive class support differs')
    if len(classes['base_ids'])!=160 or len(classes['heldout_ids'])!=40:raise ValueError('Class partition changed')
    source=parent/'H2_plan.json';dest=root/'H2_plan.json'
    if dest.exists() and dest.read_bytes()!=source.read_bytes():raise ValueError('Prelocked H2/frame plan changed')
    if not dest.exists():shutil.copyfile(source,dest)
    plan=verified(dest)
    # Finite known lineage metadata only. Never parse H2 annotations or new semantic banks here.
    exposure=[]
    lineage_roots=list(dict.fromkeys([parent,lr,Path(binding['lr_source_root']),Path(binding['minimal_root']),Path(binding['observation_root'])]))
    candidates=[Path(binding['prior_data_root'])/'acquisition_lock.json']
    for lineage_root in lineage_roots:
        candidates.extend(lineage_root/name for name in ('source_binding.json','split_manifest.json','data_inventory.json',
                           'holdout2/receipt.json','holdout2/features.json'))
    for p in candidates:
        if p.exists():
            value=read(p);text=str(value)
            exposure.append(dict(source=str(p),identity=ConsumptionIndex().identity(p),
                                 names_present=[s for s in plan['selected'] if s in text],
                                 interpretation='NAME_PRESENCE_NOT_PROOF_OF_SEMANTIC_USE'))
    from static_ovmap.preservation_routing.data import lock_H2_inventory
    # This helper inventories locked names only; it does not choose names or impose an R gate.
    inventory=lock_H2_inventory(binding,plan)
    return immutable_write(root/'prepare.json',dict(status='COMPLETE',source_binding=binding['identity'],split=split['identity'],
        classes=classes['identity'],original_objects=counts,base_original_objects=bases,observed_positive_base_ids=observed,
        zero_positive_base_ids=[i for i in classes['base_ids'] if i not in observed],H2_plan=plan['identity'],
        H2_inventory=inventory['identity'],finite_exposure_checks=exposure,H2_historically_untouched_claim=False,
        new_feature_encodings=0,parent_data_read_only=True))

class CachedObjectLoader(ObjectLoader):
    """Exact tensor-only CPU cache, shared by all arms after real parity checks.

    The existing loader remains the sole producer. Full8 inputs are sliced to
    the inherited2/4/8 prefixes; no raw view, local site or numerical op changes.
    Dense-grid LRU8GiB plus sparse LRU24GiB share the former32GiB host budget.
    """
    def __init__(self,fc,feature_manifest,*,require_parity=True):
        from collections import OrderedDict
        from .common import canonical_digest
        super().__init__(fc,feature_manifest,maximum_GiB=8)
        self.sparse=OrderedDict();self.sparse_bytes=0;self.sparse_limit=24*2**30
        self.feature_identity=feature_manifest['identity'];self.hits=0;self.misses=0
        self.namespace=canonical_digest(dict(features=self.feature_identity,FC=fc.model_key,projection=fc.dense_operator,format='EXACT_PARENT_TENSORS_V1'))
        if require_parity:
            proof=verified(Path(fc.binding['output_root'])/'engineering/sparse_loader_parity.json')
            if proof['status']!='COMPLETE' or proof['FC_identity']!=fc.model_key:raise ValueError('Real sparse-loader parity required')

    def load(self,obj,condition=0,requested_views=8):
        import numpy as np
        import torch
        from .common import canonical_digest,arrays_record
        if requested_views not in (2,4,8):raise ValueError('Fixed parent prefixes required')
        key=canonical_digest(dict(namespace=self.namespace,object_inputs=obj['inputs']['sha256'],
             object_key=obj['key'],views=obj['views'],condition=condition))
        if key in self.sparse:
            tensors,audit=self.sparse.pop(key);self.sparse[key]=(tensors,audit);self.hits+=1
        else:
            path=Path(self.fc.binding['output_root'])/'features/sparse'/self.namespace/(key+'.json')
            if path.exists():
                row=verified(path)
                if row['key']!=key:raise ValueError('Sparse cache identity changed')
                self.fc.index.identity(row['arrays']['path'],row['arrays'])
                with np.load(row['arrays']['path'],allow_pickle=False) as a:
                    if set(a.files)!=set(FORWARD_KEYS):raise ValueError('Sparse cache contains undeclared fields')
                    tensors={k:torch.from_numpy(a[k].copy()) for k in FORWARD_KEYS}
                audit=row['region_audits'];self.hits+=1
            else:
                x,_,audit=super().load(obj,condition,8)
                tensors={k:v.detach().cpu().clone() for k,v in x.items()};self.misses+=1
                arrays=arrays_record(path.with_suffix('.npz'),{k:v.numpy() for k,v in tensors.items()},self.fc.index)
                write(path,dict(status='COMPLETE',key=key,arrays=arrays,namespace=self.namespace,region_audits=audit,
                    source_inputs=obj['inputs'],feature_identity=self.feature_identity,GT_consumed=False,new_image_encodings=0))
            size=sum(t.numel()*t.element_size() for t in tensors.values())
            while self.sparse and self.sparse_bytes+size>self.sparse_limit:
                _,(old,_)=self.sparse.popitem(last=False);self.sparse_bytes-=sum(t.numel()*t.element_size() for t in old.values())
            self.sparse[key]=(tensors,audit);self.sparse_bytes+=size
        x={k:v[:requested_views].to(self.fc.device) for k,v in tensors.items()}
        return x,x['fc_vectors'],audit[:requested_views]

    def clear(self):
        self.cache.clear();self.bytes=0;self.sparse.clear();self.sparse_bytes=0
