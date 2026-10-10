"""Actual embeddings and full-vocabulary scores, preserving correlated originals."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import arrays_record,canonical_digest,objects,verified,write
from .models import teacher,RHead,GHead
from .selection import summarize

def evaluate(head,loader,items,fc,*,reference=None,base_only=True,save_path=None):
    old_mode=head.training if head is not None else None
    if head is not None:head.eval()
    records=[];embeddings=[];scores=[];started=time.perf_counter()
    with torch.no_grad():
        for obj in items:
            if base_only and not obj['base']:continue
            predictions=[];ces=[];nll=[];margins=[];zs=[];ss=[]
            for condition in range(4):
                x,_,_=loader.load(obj,condition,8)
                z=teacher(x) if head is None else head(x,fc.phi)['embedding']
                score=z@fc.text_tensor.T;logits=score/.07;t=fc.ids.index(obj['class_id'])
                pred=int(score.argmax());predictions.append(fc.ids[pred]);nll.append(float(-F.log_softmax(logits,dim=0)[t]))
                other=score.clone();other[t]=-torch.inf;margins.append(float(score[t]-other.max()))
                if obj['base']:
                    ces.append(float(F.cross_entropy((z@fc.base_text.T/.07)[None],torch.tensor([fc.base_targets[obj['class_id']]],device=fc.device))))
                else:ces.append(None)
                zs.append(z.cpu().numpy());ss.append(score.cpu().numpy())
            row=dict(key=obj['key'],family=obj['family'],class_id=obj['class_id'],base=obj['base'],
                     predictions=predictions,CE=ces,NLL=nll,margins=margins,no_op_conditions=obj['no_op_conditions'])
            records.append(row);embeddings.append(zs);scores.append(ss)
    if head is not None:head.train(old_mode)
    base=[r for r in records if r['base']]
    metrics=(summarize(base,reference) if base else dict(A=None,M=None,C=None,CE=None,NLL=None,original_objects=0,
                  variant_records=0,corrections=0,harm=0,net_corrections=0,status='NO_BASE_SUPPORT'))
    result=dict(status='COMPLETE',metrics=metrics,records=records,elapsed_seconds=time.perf_counter()-started,
                original_objects=len(records),variant_records=4*len(records),physical_image_encodings=0,
                condition_order=['clean','truncate','append','truncate_append'],selection_only_base=base_only)
    if save_path:
        result['arrays']=arrays_record(Path(save_path).with_suffix('.npz'),dict(embeddings=np.asarray(embeddings,np.float32),
               full200_cosines=np.asarray(scores,np.float32),class_ids=np.asarray([r['class_id'] for r in records],np.int64),
               object_keys=np.asarray([r['key'] for r in records]),valid_ids=np.asarray(fc.ids,np.int64)),fc.index)
        return write(save_path,result)
    return result

def load_head(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False);cfg=state['config']
    if cfg['stage']=='R':head=RHead(cfg['residual'])
    else:head=GHead(RHead(cfg['base_residual']),cfg['grouping'],cfg['direct_routing'])
    head.load_state_dict(state['model'],strict=True);head.to(device).eval().requires_grad_(False)
    return head,state
