"""TRAIN-only correctness sidecars; no correctness parameter exists in models."""
from pathlib import Path
import time
import numpy as np
import torch
from .common import arrays_record,canonical_digest,objects,verified,write
from .models import unit

def build(binding,fc,loader):
    root=Path(binding['output_root']);path=root/'teachers/manifest.json'
    items=[o for o in objects(binding,'train') if o['base']]
    identity=canonical_digest(dict(objects=[o['inputs'] for o in items],features=verified(root/'features/train-dev.json')['identity'],
                                    FC=fc.model_key,text=fc.session.text_identity,prefixes=[2,4,8],pooling='parent-v2-hard-support'))
    if path.exists():
        r=verified(path)
        if r['input_identity']!=identity:raise ValueError('Teacher observation identity changed')
        fc.index.identity(r['arrays']['path'],r['arrays']);return r
    started=time.perf_counter();vectors=np.empty((len(items),3,4,768),np.float32);pred=np.empty((len(items),3,4),np.int64)
    with torch.no_grad():
        for i,obj in enumerate(items):
            for condition in range(4):
                _,v,_=loader.load(obj,condition,8)
                for p,k in enumerate((2,4,8)):
                    z=unit(v[:k].mean(0));vectors[i,p,condition]=z.cpu().numpy()
                    pred[i,p,condition]=fc.ids[int((z@fc.text_tensor.T).argmax())]
            if (i+1)%64==0:print('TEACHERS',i+1,'/',len(items),flush=True)
    labels=np.asarray([o['class_id'] for o in items],np.int64);correct=pred[:,:,0]==labels[:,None]
    arr=arrays_record(root/'teachers/train_prefixes.npz',dict(keys=np.asarray([o['key'] for o in items]),
         class_ids=labels,prefixes=np.asarray([2,4,8]),embeddings=vectors,predictions=pred,clean_correct=correct),fc.index)
    return write(path,dict(status='COMPLETE',input_identity=identity,arrays=arr,TRAIN_only=True,original_objects=len(items),
                 correct_by_prefix={str(k):int(correct[:,p].sum()) for p,k in enumerate((2,4,8))},
                 denominator_per_prefix=len(items),prediction_API_accepts_correctness=False,
                 clean_and_corrupt_embeddings=True,teacher_full_classes=200,elapsed_seconds=time.perf_counter()-started))

def load(row,device):
    with np.load(row['arrays']['path'],allow_pickle=False) as a:
        return dict(keys={str(k):i for i,k in enumerate(a['keys'])},
                    embeddings=torch.from_numpy(a['embeddings'].copy()).to(device),correct=a['clean_correct'].copy())
