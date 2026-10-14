"""Available-source probability fusion; no GT, native fallback only for D2."""
import numpy as np
from scipy.special import softmax
from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence

def source_probabilities(sources,temperatures,ids,owner):
    ids=list(map(int,ids))
    if not ids or any(i<=0 for i in ids) or len(set(ids))!=len(ids): raise ValueError('Invalid ordered vocabulary')
    result={}
    for name in ('N','Q','F'):
        source=sources[name]
        if list(map(int,source['valid_ids']))!=ids: raise ValueError('Source ordered vocabulary differs: '+name)
        row=source['objects'][str(owner)]
        temperature=float(temperatures[name])
        if not np.isfinite(temperature) or temperature<=0: raise ValueError('Invalid frozen source temperature')
        if not row['available']:
            result[name]=None; continue
        scores=np.asarray(row['scores'],np.float64)
        if scores.shape!=(len(ids),) or not np.isfinite(scores).all(): raise ValueError('Invalid available source scores: '+name)
        result[name]=softmax(scores/temperature)
    return result

def combine(probabilities,names,operator,weights=None):
    present=[n for n in names if probabilities.get(n) is not None]
    if not present: return None
    if operator=='PARENT_D2':
        groups=[np.mean([probabilities[n] for n in group if n in present],axis=0) for group in (('N','Q'),('F',)) if any(n in present for n in group)]
        p=np.mean(groups,axis=0)
    elif operator=='EQUAL_AVAILABLE':
        p=np.mean([probabilities[n] for n in present],axis=0)
    elif operator=='DEV_SELECTED_WEIGHT':
        w=dict(zip(('N','Q','F'),weights,strict=True))
        if any(not np.isfinite(v) or v<0 for v in w.values()) or not np.isclose(sum(w.values()),1.,rtol=0,atol=1e-12): raise ValueError('Invalid fixed simplex weight')
        selected=[n for n in present if w[n]>0]
        mass=sum(w[n] for n in selected)
        if not mass: return None
        p=sum(w[n]*probabilities[n] for n in selected)/mass
    else: raise ValueError('Unknown source operator: '+operator)
    return p/p.sum()

def classify(probability,ids,native_label=0,operator='EQUAL_AVAILABLE'):
    if probability is None: return int(native_label) if operator=='PARENT_D2' else 0
    return int(ids[int(np.argmax(probability))])

def parent_d2(sources,temperatures,ids,owner,native_label):
    # Validate first, then invoke the parent's actual float64 operation order.
    probabilities=source_probabilities(sources,temperatures,ids,owner)
    scores={n:np.asarray(sources[n]['objects'][str(owner)]['scores']) if probabilities[n] is not None else None for n in probabilities}
    base=pool(scores,temperatures,('N','Q','F'))
    probability=simple_dependence(scores,temperatures,base,'PE_D2_GROUPED')
    return classify(probability,ids,native_label,'PARENT_D2'),probability
