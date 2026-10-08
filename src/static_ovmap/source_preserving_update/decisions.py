"""Float64 source updates. No evaluator, target labels or model dependency."""

import numpy as np
from scipy.special import log_softmax, softmax


NEW_METHODS=('SU04_F_REPLACE','SU05_F_BLEND','SU06_F_COARSE',
             'SU07_GLOBAL_BLEND','SU08_PAIRED_DELTA')
MATCHED_METHODS=('SU02_HARD_MATCHED','SU03_STABLE_MATCHED')


def vector(values, size=None):
    result=np.asarray(values,np.float64)
    if (result.ndim!=1 or len(result)<2 or (size is not None and len(result)!=size)
            or not np.isfinite(result).all()):
        raise ValueError('finite full-vocabulary score vector required')
    return result


def source_components(scores, temperatures):
    available=[name for name in ('N','Q','F') if scores.get(name) is not None]
    if not available:
        return None,None,None,0.
    size=len(vector(scores[available[0]]));probabilities={}
    for name in available:
        t=float(temperatures[name])
        if not np.isfinite(t) or t<=0:
            raise ValueError('positive finite frozen temperature required')
        probabilities[name]=softmax(vector(scores[name],size)/t)
    group=[probabilities[n] for n in ('N','Q') if n in probabilities]
    q=np.mean(group,axis=0) if group else None
    if q is not None:q=q/q.sum()
    f=probabilities.get('F')
    groups=[p for p in (q,f) if p is not None]
    p0=np.mean(groups,axis=0);p0=p0/p0.sum()
    return p0,q,f,0. if f is None else 1./len(groups)


def update_components(method,scores,temperatures,a,c):
    if method not in NEW_METHODS:raise ValueError('unknown fixed update rule')
    p0,q,f,w=source_components(scores,temperatures)
    if p0 is None or f is None:raise ValueError('historical F is required')
    a,c=vector(a,len(p0)),vector(c,len(p0));tf=float(temperatures['F'])
    pa,pc=softmax(a/tf),softmax(c/tf);residual=(a-c)/tf;dense=None
    if method=='SU07_GLOBAL_BLEND':
        beta=.5*w;out=(1-beta)*p0+beta*pa
    else:
        if method=='SU04_F_REPLACE':dense=pa
        elif method=='SU05_F_BLEND':dense=.5*f+.5*pa
        elif method=='SU06_F_COARSE':dense=.5*f+.5*pc
        elif np.ptp(residual)==0:
            return p0.copy(),{'p0':p0,'q':q,'pF':f,'wF':w,'pA':pa,'pC':pc,
                             'dense_group':f,'paired_residual':residual,'exact_residual_identity':True}
        else:dense=softmax(log_softmax(vector(scores['F'],len(p0))/tf)+.5*residual)
        out=dense if q is None else (1-w)*q+w*dense
    out=out/out.sum()
    if not np.isfinite(out).all():raise ValueError('nonfinite updated distribution')
    return out,{'p0':p0,'q':q,'pF':f,'wF':w,'pA':pa,'pC':pc,'dense_group':dense,
                'paired_residual':residual,'exact_residual_identity':False}


def update_probability(method,scores,temperatures,a,c):
    return update_components(method,scores,temperatures,a,c)[0]


def decide(method,ids,old_class,scores,temperatures,*,common,a=None,c=None,
           protected=False,hard_class=None,stable_class=None):
    ids=list(map(int,ids))
    if len(ids)<2 or len(set(ids))!=len(ids) or any(i<=0 for i in ids) or old_class not in ids:
        raise ValueError('original ordered positive vocabulary required')
    p0,q,f,w=source_components(scores,temperatures)
    if p0 is not None and len(p0)!=len(ids):raise ValueError('class order/score size differs')
    audit={'p0':p0,'q':q,'pF':f,'wF':w};label=old_class;probability=p0
    reason='KEEP_OUTSIDE_COMMON_DOMAIN'
    if common and not protected:
        if method in MATCHED_METHODS:
            label=hard_class if method==MATCHED_METHODS[0] else stable_class
            if label not in ids:raise ValueError('validated historical decision required')
            probability=None;reason='MATCHED_PARENT_REPLAY'
        else:
            probability,audit=update_components(method,scores,temperatures,a,c)
            label=ids[int(np.argmax(probability))];reason='FULL_DISTRIBUTION_ARGMAX'
    proposed=int(label)
    if protected and label!=old_class:label=old_class;reason='KEEP_PROTECTED_RAW_ZERO'
    a_top=None if a is None else ids[int(np.argmax(vector(a,len(ids))))]
    margins={}
    for name,values in {**scores,'A':a,'C':c}.items():
        if values is not None:
            v=vector(values,len(ids));margins[name]=float(v[ids.index(proposed)]-v[ids.index(old_class)])
    def margin(p):
        if p is None:return None
        top=np.sort(p)[-2:]
        return float(top[-1]-top[-2])
    posterior_delta=None if probability is None or p0 is None else probability-p0
    logits=None if probability is None else log_softmax(np.log(probability,where=probability>0,
        out=np.full_like(probability,-np.inf)))
    return {'old_class':old_class,'proposed_class':proposed,'applied_class':int(label),
        'changed':label!=old_class,'reason':reason,'common_domain':bool(common),
        'probability':probability,'components':audit,'A_top_class':a_top,
        'third_class_winner':proposed not in (old_class,a_top),'source_cosine_margins':margins,
        'old_top2_probability_margin':margin(p0),'new_top2_probability_margin':margin(probability),
        'posterior_delta':posterior_delta,
        'log_probability':None if logits is None else [float(x) if np.isfinite(x) else None for x in logits],
        'posterior_changed':probability is not None and p0 is not None and not np.array_equal(probability,p0)}


def scene_kernel(manifest,eligibility,coarse_records,method):
    """Same resident CPU decision function used for predictions and microtiming."""
    result={}
    for owner,row in manifest['incumbents'].items():
        if method in ('SU00_D2','SU01_G1'):
            result[owner]={'applied_class':row['old_class'],'changed':False,'reason':'BASELINE_PASS_THROUGH'}
            continue
        common=eligibility['incumbents'][owner]['eligible']
        a=c=None
        if common:
            a=np.mean([v['scores'] for v in row['full_views']],axis=0)
            c=np.mean([coarse_records[v['region_id']]['scores'] for v in row['full_views']],axis=0)
        result[owner]=decide(method,manifest['valid_ids'],row['old_class'],row['scores'],
            manifest['temperatures'],common=common,a=a,c=c,protected=bool(row['protected_raw_zero_count']),
            hard_class=row['historical']['IR06_class'],stable_class=row['historical']['IR07_class'])
    return result
