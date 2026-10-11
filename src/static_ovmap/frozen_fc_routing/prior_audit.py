"""D0: read published PR outcomes only, with separate GT and GT-free margins."""
from pathlib import Path
import time
import numpy as np
from .common import ConsumptionIndex,PathResolver,verified,write

BINS=[(-np.inf,0.),(0.,.005),(.005,.01),(.01,.02),(.02,.05),(.05,np.inf)]

def distribution(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                quantiles=dict(zip(('p00','p25','p50','p75','p100'),map(float,np.quantile(a,[0,.25,.5,.75,1])))) if len(a) else None)

def audit(binding):
    root=Path(binding['output_root']);parent=Path(binding['pr_parent_root']);path=root/'prior_audit/receipt.json'
    if path.exists():return verified(path)
    started=time.perf_counter();index=ConsumptionIndex();resolver=PathResolver(binding['path_map'])
    observed=set(verified(root/'prepare.json')['observed_positive_base_ids']);all_rows=[];sources={}
    for role in ('train','dev','old_H'):
        data={}
        for method in ('FC8','PR_R1_RESIDUAL','PR_R3_RESIDUAL_KEEP'):
            source=parent/'recognition'/role/(method+'.json');r=verified(source)
            item=resolver.rewrite(r['arrays']);index.identity(item['path'],item)
            with np.load(item['path'],allow_pickle=False) as a:
                data[method]=(r,a['embeddings'].copy(),a['full200_cosines'].copy(),a['object_keys'].tolist(),a['valid_ids'].tolist())
            sources[f'{role}/{method}']=dict(receipt=index.identity(source),arrays=item)
        reference=data['FC8'];ref_records={r['key']:r for r in reference[0]['records']}
        for method,(record,z,s,keys,ids) in data.items():
            if keys!=reference[3] or ids!=reference[4]:raise ValueError('D0 parent paired keys/text order differ')
            for i,obj in enumerate(record['records']):
                if obj['key']!=keys[i] or obj['class_id']!=ref_records[obj['key']]['class_id']:raise ValueError('D0 record keys/labels differ')
                support='OBSERVED_BASE' if obj['class_id'] in observed else 'ZERO_POSITIVE_BASE' if obj['base'] else 'HELDOUT'
                t=ids.index(obj['class_id'])
                for condition in range(4):
                    scores=s[i,condition];order=np.argsort(-scores,kind='stable');other=np.delete(scores,t)
                    pred=ids[int(order[0])];old=reference[0]['records'][i]['predictions'][condition]
                    if pred!=obj['predictions'][condition]:raise ValueError('Parent stored scores disagree with recorded winner')
                    margin=float(scores[t]-other.max());gap=float(scores[order[0]]-scores[order[1]])
                    base_scores=reference[2][i,condition];base_order=np.argsort(-base_scores,kind='stable')
                    baseline_margin=float(base_scores[t]-np.delete(base_scores,t).max())
                    baseline_gap=float(base_scores[base_order[0]]-base_scores[base_order[1]])
                    bi=next(j for j,(lo,hi) in enumerate(BINS) if lo<=baseline_margin<hi)
                    all_rows.append(dict(role=role,method=method,key=obj['key'],family=obj['family'],class_id=obj['class_id'],
                        support=support,condition=condition,GT_margin=margin,baseline_GT_margin=baseline_margin,
                        GT_margin_bin=bi,baseline_winner_runner_gap=baseline_gap,winner_runner_gap=gap,teacher_correct=int(old==obj['class_id']),
                        winner_score=float(scores[order[0]]),runner_up_score=float(scores[order[1]]),prediction=pred,
                        L2_drift=float(np.linalg.norm(z[i,condition]-reference[1][i,condition])),
                        correct=int(pred==obj['class_id']),correction=int(old!=obj['class_id'] and pred==obj['class_id']),
                        harm=int(old==obj['class_id'] and pred!=obj['class_id'])))
    groups={}
    for r in all_rows:
        for scope in ('ALL','FAMILY:'+r['family'],'SUPPORT:'+r['support']):
            groups.setdefault((r['role'],r['method'],scope),[]).append(r)
    summary=[]
    for (role,method,scope),rows in groups.items():
        summary.append(dict(role=role,method=method,scope=scope,original_objects=len({r['key'] for r in rows}),variant_records=len(rows),
            top1=float(np.mean([r['correct'] for r in rows])),corrections=sum(r['correction'] for r in rows),harm=sum(r['harm'] for r in rows),
            baseline_GT_margin=distribution([r['baseline_GT_margin'] for r in rows]),
            baseline_winner_runner_gap=distribution([r['baseline_winner_runner_gap'] for r in rows]),
            GT_margin=distribution([r['GT_margin'] for r in rows]),winner_runner_gap=distribution([r['winner_runner_gap'] for r in rows]),
            L2_drift=distribution([r['L2_drift'] for r in rows]),GT_margin_bins=[sum(r['GT_margin_bin']==i for r in rows) for i in range(6)]))
    write(root/'prior_audit/records.json',dict(status='COMPLETE',rows=all_rows,source=sources,GT_DESCRIPTIVE_ONLY=True))
    return write(path,dict(status='COMPLETE',sources=sources,summary=summary,records=len(all_rows),
        GT_margin_bins=['(-inf,0)','[0,.005)','[.005,.01)','[.01,.02)','[.02,.05)','[.05,inf)'],
        new_training_updates=0,new_image_encodings=0,old_H_exposed=True,used_for_selection=False,elapsed_seconds=time.perf_counter()-started))
