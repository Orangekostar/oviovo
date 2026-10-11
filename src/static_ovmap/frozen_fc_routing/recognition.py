"""Full200 scores from actual fixed observations; labels only score outputs."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import arrays_record,write
from .models import FrozenFCBase
from .selection import summarize
from .export import load_export as load_head

def evaluate(head,loader,items,fc,*,reference=None,base_only=True,save_path=None,eta_override=None,observer=None):
    old_mode=head.training if head is not None else None
    if head is not None:head.eval()
    records=[];embeddings=[];scores=[];started=time.perf_counter();calls=0;frames=set()
    with torch.no_grad():
        for obj in items:
            if base_only and not obj['base']:continue
            predictions=[];ces=[];nll=[];margins=[];gaps=[];winner_scores=[];runner_scores=[];zs=[];ss=[];coverage=[]
            for condition in range(4):
                x,_,audit=loader.load(obj,condition,8)
                result=None if head is None else head(x,fc.phi,eta_override=eta_override)
                z=FrozenFCBase()(x) if result is None else result['embedding'];calls+=1
                score=z@fc.text_tensor.T;logits=score/.07;t=fc.ids.index(obj['class_id'])
                if observer is not None:observer(obj,condition,x,result,score)
                pred=int(score.argmax());predictions.append(fc.ids[pred]);nll.append(float(-F.log_softmax(logits,dim=0)[t]))
                other=score.clone();other[t]=-torch.inf;margins.append(float(score[t]-other.max()))
                top=score.topk(2).values;gaps.append(float(top[0]-top[1]));winner_scores.append(float(top[0]));runner_scores.append(float(top[1]))
                ces.append(float(F.cross_entropy((z@fc.base_text.T/.07)[None],torch.tensor([fc.base_targets[obj['class_id']]],device=fc.device))) if obj['base'] else None)
                zs.append(z.cpu().numpy());ss.append(score.cpu().numpy())
                view_ids=x['frame_ids'][x['view_valid']].tolist();frames.update((obj['scene'],int(v)) for v in view_ids)
                coverage.append(dict(actual_views=len(view_ids),local_observations=int(x['local_valid'][x['view_valid']].sum()),
                     observed_sites=int(x['local_valid'][x['view_valid']].any(0).sum()),frame_ids=view_ids,
                     fallback=result['fallback'] if result else None,region_audits=audit,
                     group_count=result['group_count'] if result else None))
            records.append(dict(key=obj['key'],family=obj['family'],class_id=obj['class_id'],base=obj['base'],
                 predictions=predictions,CE=ces,NLL=nll,margins=margins,winner_runner_gaps=gaps,winner_scores=winner_scores,
                 runner_up_scores=runner_scores,no_op_conditions=obj.get('no_op_conditions'),coverage=coverage))
            embeddings.append(zs);scores.append(ss)
    if head is not None:head.train(old_mode)
    base=[r for r in records if r['base']]
    metrics=summarize(base,reference) if base else dict(A=None,M=None,C=None,CE=None,NLL=None,original_objects=0,variant_records=0,
               corrections=0,harm=0,net_corrections=0,status='NO_BASE_SUPPORT')
    result=dict(status='COMPLETE',metrics=metrics,records=records,elapsed_seconds=time.perf_counter()-started,
          original_objects=len(records),variant_records=4*len(records),physical_image_encodings=0,head_or_base_forward_calls=calls,
          distinct_frames=len(frames),condition_order=['clean','truncate','append','truncate_append'],selection_only_base=base_only,
          eta_override=eta_override)
    if save_path:
        result['arrays']=arrays_record(Path(save_path).with_suffix('.npz'),dict(embeddings=np.asarray(embeddings,np.float32),
             full200_cosines=np.asarray(scores,np.float32),class_ids=np.asarray([r['class_id'] for r in records],np.int64),
             object_keys=np.asarray([r['key'] for r in records]),valid_ids=np.asarray(fc.ids,np.int64)),fc.index)
        return write(save_path,result)
    return result
