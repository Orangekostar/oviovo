"""Fixed-original recognition arithmetic, retaining abstentions and absent classes."""
import numpy as np

def recognition(rows):
    if not rows:return dict(ordinary_top1=None,balanced_top1=None,A=None,M=None,C=None,original_objects=0,families=0,classes=0,variant_records=0,NLL200=None,NLL200_defined_n=0,CE160=None,CE160_defined_n=0,abstention_n=0)
    fam={};classes={};clean={};nll=[];ce=[];correct=[];abstentions=0
    for row in rows:
        values=np.asarray([int(int(p)==int(row['class_id'])) for p in row['predictions']],np.float64)
        if len(values)!=4:raise ValueError('Four conditions required')
        weighted=float(.5*values[0]+values[1:].sum()/6)
        fam.setdefault(row['family'],[]).append(weighted);classes.setdefault(row['class_id'],[]).append((weighted,float(values[0])));clean.setdefault(row['family'],[]).append(float(values[0]));correct.append(float(values[0]))
        nll.extend(v for v in row.get('NLL',[]) if v is not None);ce.extend(v for v in row.get('CE',[]) if v is not None)
        abstentions+=sum(not d for d in row['defined'])
    return dict(ordinary_top1=float(np.mean(correct)),balanced_top1=float(np.mean([np.mean([v[1] for v in group]) for group in classes.values()])),A=float(np.mean([np.mean(v) for v in fam.values()])),M=float(np.mean([np.mean([v[0] for v in group]) for group in classes.values()])),C=float(np.mean([np.mean(v) for v in clean.values()])),original_objects=len(rows),families=len(fam),classes=len(classes),variant_records=4*len(rows),NLL200=float(np.mean(nll)) if nll else None,NLL200_defined_n=len(nll),CE160=float(np.mean(ce)) if ce else None,CE160_defined_n=len(ce),abstention_n=abstentions)

def partitions(rows):
    out={name:recognition([r for r in rows if name=='all' or bool(r['base'])==(name=='base')]) for name in ('all','base','novel')}
    a=out['base']['ordinary_top1'];b=out['novel']['ordinary_top1'];out['H_base_novel']=None if a is None or b is None else 2*a*b/(a+b) if a+b else 0.
    return out

def score_losses(score,target,base,valid_ids,base_ids):
    if score is None:return None,None
    score=np.asarray(score,np.float64)/.07;t=list(valid_ids).index(int(target));m=float(score.max());nll=float(m+np.log(np.exp(score-m).sum())-score[t]);ce=None
    if base:
        ids=[i for i,c in enumerate(valid_ids) if int(c) in base_ids];v=score[ids];m=float(v.max());ce=float(m+np.log(np.exp(v-m).sum())-score[t])
    return nll,ce
