"""The frozen 110-candidate union and exact deterministic selection."""
import numpy as np
from .metrics import predictions

def recipes(spec):
    rows=[]
    for kind,values in [('MIX',spec['alpha_grid']),('CAP',spec['angle_cap_degrees'])]:
        for v in values:
            for gamma in spec['gamma_grid']:
                rows.append(dict(candidate_id=f'{kind}|{v:g}|{gamma:+g}', kind=kind,
                                 alpha=v if kind=='MIX' else None, angle_degrees=v if kind=='CAP' else None,
                                 gamma=gamma, operations=int(v!=1 if kind=='MIX' else v<180)+int(gamma!=0)))
    if len(rows)!=110 or len({r['candidate_id'] for r in rows})!=110:
        raise ValueError('Exactly 110 unique logical candidates required')
    return sorted(rows,key=lambda r:r['candidate_id'])

def angle_cap(b, z, degrees):
    b=np.asarray(b);z=np.asarray(z)
    if b.shape!=z.shape or b.shape[-1]!=768: raise ValueError('True 768-dimensional vectors required')
    result=z.copy();flags=np.zeros(z.shape[:-1],bool)
    if degrees==180:return result,flags
    cosine=np.clip(np.sum(b.astype(np.float64)*z.astype(np.float64),axis=-1),-1.,1.)
    theta=np.arccos(cosine);tau=np.deg2rad(degrees);clip=theta>tau
    tangent=z.astype(np.float64)-cosine[...,None]*b
    norm=np.linalg.norm(tangent,axis=-1);flags=clip & (norm<1e-10)
    normal=clip & ~flags
    result[normal]=(np.cos(tau)*b[normal]+np.sin(tau)*tangent[normal]/norm[normal,None]).astype(z.dtype)
    result[flags]=b[flags]
    return result,flags

def candidate_scores(recipe,b_scores,z_scores,b,z,text,base_mask):
    if recipe['kind']=='MIX':
        a=recipe['alpha']
        s=b_scores.copy() if a==0 else z_scores.copy() if a==1 else (1-a)*b_scores+a*z_scores
        flags=np.zeros(len(s),bool)
    else:
        clipped,flags=angle_cap(b,z,recipe['angle_degrees']);s=clipped@text.T
    # CPU FP32 text/score arithmetic; gamma is in cosine units, not temperature logits.
    s[:,base_mask]-=recipe['gamma']
    return s,flags

def best_candidate(rows,tolerance=1e-10):
    rows=[r for r in rows if r['metrics']['H_macro'] is not None]
    if not rows: raise ValueError('No candidate has both positive class groups')
    for field in ('H_macro','min_macro','B_macro'):
        def value(r):
            m=r['metrics'];return min(m['B_macro'],m['N_macro']) if field=='min_macro' else m[field]
        optimum=max(value(r) for r in rows)
        rows=[r for r in rows if optimum-value(r)<=tolerance]
    return min(rows,key=lambda r:(r['recipe']['operations'],r['recipe']['candidate_id']))

def gate(comparisons, minimum=.01, positive_families=8):
    passed=[]
    for method in ('S_LL','G_LL'):
        rows=comparisons[method]
        if all(r['delta_H_macro'] is not None and r['delta_H_macro']>=minimum-1e-10
               and r['positive_families']>=positive_families for r in rows):passed.append(method)
    reference=max(passed,key=lambda m:(min(r['delta_H_macro'] for r in comparisons[m]),m=='S_LL')) if passed else None
    return dict(open=bool(passed),reference=reference,comparisons=comparisons,
                status='B_GATE_OPEN' if passed else 'COMPLETE_A_NO_RESIDUAL_SIGNAL',
                threshold_H_macro=minimum,threshold_positive_families=positive_families,
                statistical_significance_claim=False)
