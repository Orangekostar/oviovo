"""D1 fixed GT-only filtering and descriptive recognition strata."""
from pathlib import Path
import time
import numpy as np
import torch
from .common import arrays_record,objects,verified,write
from .models import FrozenFCBase,unit


def score_record(z,fc,obj,condition,method,**extra):
    score=z@fc.text_tensor.T;t=fc.ids.index(obj['class_id']);logits=score/.07
    nll=float(-torch.log_softmax(logits,0)[t]);prediction=fc.ids[int(score.argmax())]
    return dict(key=obj['key'],family=obj['family'],class_id=obj['class_id'],base=obj['base'],condition=condition,
        method=method,prediction=prediction,correct=int(prediction==obj['class_id']),NLL=nll,**extra),score

def metric(rows):
    if not rows:return dict(original_objects=0,variant_records=0,families=0,categories=0,top1=None,macro=None,NLL=None,corrections=0,harm=0,net_corrections=0)
    classes={}
    for r in rows:classes.setdefault(r['class_id'],[]).append(r['correct'])
    corrections=sum(r.get('correction',0) for r in rows);harm=sum(r.get('harm',0) for r in rows)
    counts={k:sum(int(r[k]) for r in rows) for k in ('positive_observations','negative_observations','unknown_observations',
        'observed_sites','known_target_sites','sites_with_positive','sites_with_negative','unknown_only_sites','inconsistent_sites','target_fallback','all_fallback') if all(k in r for r in rows)}
    return dict(counts,original_objects=len({r['key'] for r in rows}),variant_records=len(rows),families=len({r['family'] for r in rows}),
        categories=len(classes),top1=float(np.mean([r['correct'] for r in rows])),macro=float(np.mean([np.mean(v) for v in classes.values()])),
        NLL=float(np.mean([r['NLL'] for r in rows])),corrections=corrections,harm=harm,net_corrections=corrections-harm)

def summarize_rows(rows,*,details=True):
    groups={}
    for r in rows:
        scopes=['ALL','CONDITION:'+str(r['condition']),'BASE' if r['base'] else 'HELDOUT']
        if details:scopes+=['FAMILY:'+r['family'],'CLASS:'+str(r['class_id'])]
        for scope in scopes:groups.setdefault((r['method'],scope),[]).append(r)
    return [dict(method=m,scope=s,**metric(rr)) for (m,s),rr in groups.items()]

def oracle_vectors(x,y,phi):
    views=torch.nonzero(x['view_valid'],as_tuple=False).flatten();valid=x['local_valid'][views];target=y[views]
    raw=torch.where(valid[...,None],x['local_raw'][views],0);count=valid.sum(0);present=count>0
    base=FrozenFCBase()(x)
    positive=(valid&(target==1)).any(0);negative=(valid&(target==0)).any(0)
    target_sites=present&positive&~negative
    vectors={};outputs={'FC8':base};u=None
    if bool(present.any()):u=unit(phi(raw.sum(0)[present]/count[present,None]))
    for method,selection in [('ALL',present),('TARGET_ONLY',target_sites)]:
        chosen=selection[present]
        outputs[method]=unit(.75*base+.25*u[chosen].mean(0)) if bool(chosen.any()) else base
    counts=dict(positive_observations=int((valid&(target==1)).sum()),negative_observations=int((valid&(target==0)).sum()),
        unknown_observations=int((valid&(target<0)).sum()),observed_sites=int(present.sum()),known_target_sites=int(target_sites.sum()),
        sites_with_positive=int((present&positive).sum()),sites_with_negative=int((present&negative).sum()),
        unknown_only_sites=int((present&~positive&~negative).sum()),inconsistent_sites=int((positive&negative).sum()),
        actual_views=len(views),target_fallback=not bool(target_sites.any()),all_fallback=not bool(present.any()),
        target_fallback_reason='NO_KNOWN_TARGET_SITE' if not bool(target_sites.any()) else None,
        all_fallback_reason='NO_LOCAL_SUPPORT' if not bool(present.any()) else None)
    return outputs,counts

def diagnostic_D1(binding,fc,loader):
    root=Path(binding['output_root']);path=root/'diagnostic_D1/receipt.json'
    if path.exists():return refresh_D1(binding)
    rows=[];zs=[];scores=[];started=time.perf_counter()
    with torch.no_grad():
        for role in ('train','dev'):
            selected=[o for o in objects(binding,role) if o['base']]
            for oi,obj in enumerate(selected):
                for condition in range(4):
                    x,_,_=loader.load(obj,condition,8);_,y=loader.targets(obj,condition,8)
                    outputs,counts=oracle_vectors(x,y,fc.phi);ref=None
                    for method,z in outputs.items():
                        row,score=score_record(z,fc,obj,condition,method,role=role,GT_DIAGNOSTIC_ONLY=True,**counts)
                        if method=='FC8':ref=row['correct']
                        row.update(correction=int(not ref and row['correct']),harm=int(ref and not row['correct']))
                        rows.append(row);zs.append(z.cpu().numpy());scores.append(score.cpu().numpy())
                if (oi+1)%100==0:print('D1',role,oi+1,'/',len(selected),flush=True)
    table=[];paired=[]
    for role in ('train','dev'):
        subset=[r for r in rows if r['role']==role]
        table += [dict(role=role,GT_DIAGNOSTIC_ONLY=True,**r) for r in summarize_rows(subset)]
        common=[r for r in subset if not r['target_fallback'] and not r['all_fallback']]
        paired += [dict(role=role,paired_scope='COMMON_NONFALLBACK',GT_DIAGNOSTIC_ONLY=True,**r) for r in summarize_rows(common)]
    arr=arrays_record(root/'diagnostic_D1/scores.npz',dict(embeddings=np.asarray(zs,np.float32),full200_cosines=np.asarray(scores,np.float32),
        keys=np.asarray([r['key'] for r in rows]),methods=np.asarray([r['method'] for r in rows]),conditions=np.asarray([r['condition'] for r in rows]),
        valid_ids=np.asarray(fc.ids)),fc.index)
    write(root/'diagnostic_D1/records.json',dict(status='GT_DIAGNOSTIC_ONLY',rows=rows,arrays=arr))
    return write(path,dict(status='COMPLETE',GT_DIAGNOSTIC_ONLY=True,summary=table,paired_nonfallback=paired,
        original_objects={'train':622,'dev':69},variant_records_per_method=2764,training_scope_affected=False,
        fallback_counts={role:sum(r['target_fallback'] for r in rows if r['role']==role and r['method']=='TARGET_ONLY') for role in ('train','dev')},
        arrays=arr,new_encodings=0,head_calls=0,elapsed_seconds=time.perf_counter()-started))


def refresh_D1(binding):
    """Derive complete count/fallback summaries from already measured fixed rows."""
    import shutil
    root=Path(binding['output_root']);path=root/'diagnostic_D1/receipt.json';old=verified(path)
    if old.get('summary_schema')==2:return old
    source=root/'diagnostic_D1/records.json';records=verified(source)
    archive=root/'history/D1_summary_before_coverage_enrichment';archive.mkdir(parents=True,exist_ok=True)
    for p in (path,source):
        if not (archive/p.name).exists():shutil.copyfile(p,archive/p.name)
    rows=records['rows'];summary=[];paired=[]
    for r in rows:
        r['target_fallback_reason']='NO_KNOWN_TARGET_SITE' if r['target_fallback'] else None
        r['all_fallback_reason']='NO_LOCAL_SUPPORT' if r['all_fallback'] else None
    for role in ('train','dev'):
        rr=[r for r in rows if r['role']==role]
        summary.extend(dict(role=role,GT_DIAGNOSTIC_ONLY=True,**r) for r in summarize_rows(rr))
        common=[r for r in rr if not r['target_fallback'] and not r['all_fallback']]
        paired.extend(dict(role=role,GT_DIAGNOSTIC_ONLY=True,paired_scope='COMMON_NONFALLBACK',**r) for r in summarize_rows(common))
    records.pop('identity',None);write(source,records)
    old.pop('identity',None);old.update(summary=summary,paired_nonfallback=paired,summary_schema=2,
                                      summary_only_refresh=True,additional_forward_calls=0)
    return write(path,old)
