"""Real cached seven-field inputs and the equivalent region-only MV view."""
from pathlib import Path
import numpy as np
from .common import verified, canonical_digest, ConsumedInputs, Resolver, StageFailure

FORWARD_KEYS = {'local_raw','local_unit','metadata','local_valid','view_valid','frame_ids','fc_vectors'}


def region_only(inputs):
    """Keep actual regions/validity/order; no local observations were requested."""
    if set(inputs) != FORWARD_KEYS:
        raise ValueError('Expected exactly seven actual input tensors')
    import torch
    v = inputs['fc_vectors'].shape[0]
    if v > 8 or inputs['fc_vectors'].shape != (v,768):
        raise ValueError('Invalid region input dimensions')
    return dict(fc_vectors=inputs['fc_vectors'],view_valid=inputs['view_valid'],
        frame_ids=inputs['frame_ids'],local_raw=inputs['local_raw'].new_empty(v,0,1536),
        local_unit=inputs['local_unit'].new_empty(v,0,768),
        metadata=inputs['metadata'].new_empty(v,0,8),
        local_valid=torch.empty(v,0,dtype=torch.bool,device=inputs['local_valid'].device))


class CachedSparse:
    """Read the already-produced SparseInputs payload, never mutate parent caches."""
    def __init__(self,binding,features):
        self.binding=binding; self.features=features
        self.root=Path(binding['output_root']); self.index=ConsumedInputs(self.root)
        self.resolver=Resolver(binding['path_map']); self.namespaces=[]
        inherited=binding['inherited'] or {}
        for folder in [Path(binding['structure_root'])/'features/sparse'] if binding['structure_root'] else []:
            if not folder.is_dir():continue
            # Only the finite namespace directory roots for this specific feature manifest.
            for directory in sorted(folder.iterdir()):
                if not directory.is_dir():continue
                sample=next(directory.glob('*.json'),None)
                if sample is None:continue
                row=verified(sample)
                if row['feature_identity']==features['identity']:
                    self.namespaces.append(directory)
        self.parent_fallback=None

    def locate(self,obj,condition=0):
        for folder in self.namespaces:
            key=canonical_digest(dict(namespace=folder.name,object_inputs=obj['inputs']['sha256'],
                object_key=obj['key'],views=obj['views'],condition=int(condition)))
            path=folder/(key+'.json')
            if path.is_file():
                self.index.identity(path); row=verified(path)
                if row['key']!=key or row['feature_identity']!=self.features['identity'] or row['source_inputs']['sha256']!=obj['inputs']['sha256']:
                    raise ValueError('Actual SparseInputs cache identity changed')
                return path,row
        raise StageFailure('Exact seven-field parent SparseInputs cache is missing',
            missing=[dict(object_key=obj['key'],feature_identity=self.features['identity'],
                          expected_namespaces=[str(p) for p in self.namespaces])])

    def load(self,obj,*,device='cuda',region=False):
        import torch
        path,row=self.locate(obj)
        arrays_path=Path(self.resolver.resolve(row['arrays']['path']))
        identity=self.index.identity(arrays_path,row['arrays'])
        with np.load(arrays_path,allow_pickle=False) as arrays:
            if set(arrays.files)!=FORWARD_KEYS:
                raise ValueError('SparseInputs contains an unexpected tensor field')
            inputs={k:torch.from_numpy(arrays[k].copy()).to(device) for k in FORWARD_KEYS}
        if not torch.isfinite(inputs['fc_vectors'][inputs['view_valid'].bool()]).all():
            raise ValueError('Nonfinite valid cached region vector')
        return region_only(inputs) if region else inputs,dict(metadata=str(path),arrays=identity,
            namespace=row['namespace'],object_key=obj['key'],inputs=obj['inputs'],region_only=region)

    def flush(self):
        self.index.save()
