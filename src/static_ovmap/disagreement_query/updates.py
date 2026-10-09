"""Full-vocabulary fixed-mass updates on exact distinct new observations."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class Evidence:
    key:str
    cosine:object
    footprint:object
    available:bool=True
    reason:str='AVAILABLE'


def support_weights(area,footprints,kind):
    a=np.asarray(area,np.float64);o=np.asarray(footprints,bool)
    if o.ndim!=2 or o.shape[1]!=len(a) or not len(o) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('view-by-site footprints required')
    if kind=='MEAN':return np.ones(len(o),np.float64)
    if kind=='AREA':return o.astype(float)@a
    if kind!='SUPPORT':raise ValueError('unknown updater')
    multiplicity=o.sum(axis=0);shared=np.divide(a,multiplicity,out=np.zeros_like(a),where=multiplicity>0)
    weights=o.astype(float)@shared
    if not np.isclose(weights.sum(),a[multiplicity>0].sum(),rtol=1e-12,atol=1e-12):raise ValueError('support mass not conserved')
    return weights


def update(p0,records,area,kind,temperature,ids,old_class,*,required_count=None):
    p0=np.asarray(p0,np.float64);ids=list(map(int,ids))
    if p0.shape!=(len(ids),) or not np.isfinite(p0).all() or np.any(p0<0) or not np.isclose(p0.sum(),1):raise ValueError('immutable full prior required')
    if old_class not in ids or len(set(ids))!=len(ids) or temperature<=0 or not np.isfinite(temperature):raise ValueError('ordered vocabulary and frozen temperature required')
    if required_count is not None and len(records)!=required_count:raise ValueError('required FULL record omitted')
    unique={}
    for r in records:
        if r.key in unique:
            old=unique[r.key]
            if old.available!=r.available or not np.array_equal(old.cosine,r.cosine) or not np.array_equal(old.footprint,r.footprint):raise ValueError('contradictory exact observation identity')
        else:unique[r.key]=r
    base={'probability':p0.copy(),'label':int(old_class),'changed':False,'weights':[],
        'unique_keys':sorted(unique),'duplicate_records':len(records)-len(unique),'p_new':None,
        'prior_mass':.75,'new_mass':.25,'fallback':None,'support_union_area':0.}
    if not records:base['reason']='KEEP_NO_QUERY';return base
    if any(not r.available for r in records):
        base['reason']='KEEP_REQUIRED_FULL_UNAVAILABLE';base['unavailable_reasons']=[r.reason for r in records if not r.available];return base
    rows=[unique[k] for k in sorted(unique)];footprints=np.array([r.footprint for r in rows],bool)
    w=support_weights(area,footprints,kind);base['support_union_area']=float(np.asarray(area)[footprints.any(axis=0)].sum())
    if w.sum()<=0:w=np.ones(len(rows));base['fallback']='ZERO_SUPPORT_FALLBACK_MEAN'
    scores=np.asarray([r.cosine for r in rows],np.float64)
    if scores.shape!=(len(rows),len(ids)) or not np.isfinite(scores).all():raise ValueError('finite full-vocabulary scores required')
    pv=softmax(scores/temperature,axis=1);new=np.average(pv,axis=0,weights=w)
    final=.75*p0+.25*new;final=final/final.sum();label=ids[int(np.argmax(final))]
    return {**base,'probability':final,'p_new':new,'label':label,'changed':label!=old_class,
        'reason':'FULL_VOCABULARY_ARGMAX','weights':w.tolist(),'view_probabilities':pv,
        'identical_supports':bool(len(rows)==2 and np.array_equal(footprints[0],footprints[1])),
        'disjoint_supports':bool(len(rows)==2 and not np.any(footprints[0]&footprints[1]))}
