"""DEV-only lambda selection; seed29 never selects or replaces C_STAR."""
from functools import cmp_to_key
from .common import METHODS, LAMBDAS, verified, write, DependencyError
from static_ovmap.preservation_routing.selection import summarize

TOL=1e-10

def compare(a,b):
    for key in ('A','M','C'):
        difference=a[key]-b[key]
        if abs(difference)>TOL: return -1 if difference>0 else 1
    if a['lambda_']!=b['lambda_']: return -1 if a['lambda_']<b['lambda_'] else 1
    return METHODS.index(a.get('method',METHODS[0]))-METHODS.index(b.get('method',METHODS[0]))

def choose(rows):
    if not rows: raise ValueError('No lambda candidates')
    return sorted(rows,key=cmp_to_key(compare))[0]

def passes(row,baseline):
    return bool(row['lambda_']>0 and row['A']-baseline['A']>=.005-TOL and row['M']-baseline['M']>=-.005-TOL and row['C']-baseline['C']>=-.005-TOL and row['net_corrections']>0 and row['numeric_fallback_rate']<=.01)

def activate(lock,repeat,baseline17,baseline29):
    facts={m:dict(seed17=passes(lock['selected'][m],baseline17),seed29=passes(repeat[m],baseline29)) for m in METHODS}
    both={m:all(r.values()) for m,r in facts.items()}; star=lock['C_STAR']; m1=METHODS[2]
    expanded=[m1,star] if both[m1] else [star] if both[star] else []
    return dict(status='COMPLETE',C_STAR=star,pass_by_seed=facts,pass_both=both,expanded=expanded,locked_lambdas={m:lock['selected'][m]['lambda_'] for m in METHODS},P1_used_for_activation=False)

def lock_selection(root,candidates):
    selected={m:choose(candidates[m]) for m in METHODS}
    star=choose([selected[m] for m in METHODS[:2]])['method']
    value=dict(status='COMPLETE',selected=selected,C_STAR=star,candidates=candidates,seed=17,seed29_seen=False,map_seen=False)
    path=root/'selection_seed17.json'
    if path.exists():
        old=verified(path)
        if {k:v for k,v in old.items() if k!='identity'}!=value: raise ValueError('Locked seed17 selection changed')
        return old
    return write(path,value)
