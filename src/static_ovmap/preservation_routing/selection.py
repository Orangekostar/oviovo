"""Full200 proposal selection on DEV base originals only."""
from functools import cmp_to_key
import numpy as np
TOL=1e-10

def summarize(rows,reference=None):
    if not rows:raise ValueError('Required recognition rows missing')
    fam={};cls={};clean={};ce={};nll={};corrections=harm=0
    ref={r['key']:r for r in reference or []}
    for r in rows:
        correct=[int(p==r['class_id']) for p in r['predictions']]
        a=.5*correct[0]+sum(correct[1:])/6
        f=r['family'];c=r['class_id'];fam.setdefault(f,[]).append(a);cls.setdefault(c,[]).append(a)
        clean.setdefault(f,[]).append(correct[0]);ce.setdefault(f,[]).append(.5*r['CE'][0]+sum(r['CE'][1:])/6)
        nll.setdefault(f,[]).append(.5*r['NLL'][0]+sum(r['NLL'][1:])/6)
        if ref:
            baseline=ref[r['key']]
            if baseline['class_id']!=c:raise ValueError('Paired class identity changed')
            for p,q in zip(correct,baseline['predictions']):
                corrections+=int(p and q!=c);harm+=int(not p and q==c)
    mean=lambda d:float(np.mean([np.mean(v) for v in d.values()]))
    return dict(A=mean(fam),M=mean(cls),C=mean(clean),CE=mean(ce),NLL=mean(nll),
                corrections=corrections,harm=harm,net_corrections=corrections-harm,
                original_objects=len(rows),variant_records=len(rows)*4,families=len(fam),categories=len(cls),
                per_family_A={k:float(np.mean(v)) for k,v in fam.items()},full200_competitors=True)

def qualifies(row,reference):
    return (row['step']>=250 and row['A']>=reference['A']+.005-TOL and
            row['M']>=reference['M']-TOL and row['C']>=reference['C']-TOL and row['net_corrections']>0)

def compare(a,b):
    for k,sign in [('A',-1),('M',-1),('C',-1),('CE',1)]:
        d=a[k]-b[k]
        if abs(d)>TOL:return sign*(1 if d>0 else -1)
    for k in ('step','order'):
        d=a.get(k,0)-b.get(k,0)
        if d:return 1 if d>0 else -1
    return 0

def choose(rows,reference=None):
    candidates=[r for r in rows if r['step']>=250 and (reference is None or qualifies(r,reference))]
    return sorted(candidates,key=cmp_to_key(compare))[0] if candidates else None

def nominate_R(binding,seed):
    from pathlib import Path
    from .common import verified,write
    root=Path(binding['output_root']);directory=root/'training'/f'seed{seed}';train=verified(directory/'R_receipt.json')
    rows=[verified(directory/m/'receipt.json') for m in train['arms']]
    candidates=[dict(r['selected_metrics'],branch=r['branch'],order=i,checkpoint=r['selected_checkpoint']) for i,r in enumerate(rows) if r['foundation_pass']]
    winner=choose(candidates)
    return write(root/f'R_nomination_seed{seed}.json',dict(status='QUALIFIED' if winner else 'COMPLETE_NO_2D_FOUNDATION' if seed==17 else 'COMPLETE_2D_NOT_REPEATED',
           seed=seed,Rstar=winner['branch'] if winner else None,winner=winner,
           checkpoints={r['branch']:r['selected_checkpoint'] for r in rows},arms={r['branch']:r['selected_metrics'] for r in rows},
           data=verified(root/'prepare.json')['identity'],selection_uses_old_H=False,selection_uses_maps=False,selection_uses_runtime=False))
