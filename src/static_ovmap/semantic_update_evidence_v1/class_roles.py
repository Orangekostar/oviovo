"""Label-only deterministic CORE/PSEUDO definitions and original-index draws."""
import hashlib
from collections import Counter
import numpy as np

def split_roles(train,dev,base_ids,novel_ids):
    tr=Counter(int(o['class_id']) for o in train if o.get('base',False))
    dv=Counter(int(o['class_id']) for o in dev if o.get('base',False))
    candidates=[int(c) for c in base_ids if tr[c]>=4 and dv[c]>=1]
    candidates.sort(key=lambda c:(hashlib.sha256(('SUEV1-PSEUDO-20261016|'+str(c)).encode()).hexdigest(),c))
    if len(candidates)<8:return dict(status='B_DATA_LIMITED',reason='FEWER_THAN_8_ELIGIBLE_CLASSES')
    pseudo=candidates[:8];core=[int(c) for c in base_ids if c not in pseudo]
    pdev=[o for o in dev if o['class_id'] in pseudo];cdev=[o for o in dev if o['class_id'] in core]
    ok=len({o['family'] for o in pdev})>=2 and len(cdev)>=8 and len({o['class_id'] for o in cdev})>=8
    return dict(status='COMPLETE' if ok else 'B_DATA_LIMITED',CORE=core,PSEUDO=pseudo,FINAL_NOVEL=list(novel_ids),
                original_index_to_key=[o['key'] for o in train],eligible_counts={c:tr[c] for c in candidates})

def draw_schedule(originals,core_ids,seed,updates=2000):
    classes={}
    for i,o in enumerate(originals):
        if o['class_id'] in core_ids and o.get('supported',bool(o.get('views',True))):
            classes.setdefault(o['class_id'],[]).append(i)
    if not classes:raise ValueError('No usable CORE originals')
    ids=sorted(classes);rng=np.random.default_rng(seed);draws=np.empty((updates*16,3),np.int32)
    for i in range(len(draws)):
        c=ids[int(rng.integers(len(ids)))];rows=classes[c]
        draws[i]=rows[int(rng.integers(len(rows)))],(2,4,8)[i%3],1+(i//3)%3
    return draws
