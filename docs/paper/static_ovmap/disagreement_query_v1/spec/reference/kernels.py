"""Small NumPy reference contracts; not FC/Open3D/production implementations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

METRICS = ('apall', 'ap50', 'ap25', 'miou', 'macc')


def softmax(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('finite class vector required')
    z = np.exp(x - x.max())
    return z / z.sum()


def grouped(scores: Mapping[str, object], temperatures, *, probability=True):
    values = {}
    size = None
    for name in ('N','Q','F'):
        if scores.get(name) is None:
            continue
        v = np.asarray(scores[name], np.float64)
        if v.ndim != 1 or len(v)<2 or not np.isfinite(v).all():
            raise ValueError('finite class vector required')
        size = len(v) if size is None else size
        t = float(temperatures[name])
        if len(v) != size or not np.isfinite(t) or t <= 0:
            raise ValueError('invalid shape or temperature')
        values[name] = softmax(v/t) if probability else v/t
    groups = []
    nq = [values[k] for k in ('N','Q') if k in values]
    if nq: groups.append(np.mean(nq, axis=0))
    if 'F' in values: groups.append(values['F'])
    return np.mean(groups, axis=0) if groups else None


def canonical_sites(xyz, faces, owners):
    """Exact duplicate-face/site reference for tiny fixtures, not a large-mesh engine."""
    xyz=np.asarray(xyz,np.float64).copy(); faces=np.asarray(faces); owners=np.asarray(owners)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or owners.shape!=(len(xyz),) or not np.isfinite(xyz).all():
        raise ValueError('finite source mesh required')
    if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer):
        raise ValueError('indexed triangles required')
    if np.any(faces<0) or np.any(faces>=len(xyz)):
        raise ValueError('face index outside mesh')
    xyz[xyz==0]=0
    unique={}; duplicates=0; ambiguous=set(); degenerates=0
    for f in faces:
        triples=[(tuple(xyz[i]),int(owners[i])) for i in f]
        triples.sort(key=lambda x:x[0])
        key=tuple(t[0] for t in triples); assignment=tuple(t[1] for t in triples)
        points=np.array(key); area=float(np.linalg.norm(np.cross(points[1]-points[0],points[2]-points[0]))/2)
        if area<=0: degenerates+=1; continue
        if key in unique:
            duplicates+=1
            if unique[key][0]!=assignment: ambiguous.add(key)
        else: unique[key]=(assignment,area)
    mass={}
    for key,(assignment,area) in unique.items():
        if key in ambiguous: continue
        for pos,owner in zip(key,assignment):
            if owner>0: mass[(owner,pos)]=mass.get((owner,pos),0)+area/3
    records=sorted(mass)
    return records,np.array([mass[k] for k in records]),dict(duplicates=duplicates,ambiguous=len(ambiguous),degenerates=degenerates)


def quadrature(xyz, area, maximum=4096):
    xyz=np.asarray(xyz,np.float64); area=np.asarray(area,np.float64)
    if xyz.shape!=(len(area),3) or maximum<1 or not np.isfinite(xyz).all() or not np.isfinite(area).all() or np.any(area<=0):
        raise ValueError('positive-area finite canonical sites required')
    order=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0])); x=xyz[order]; a=area[order]
    if len(a)<=maximum: return x.copy(),a.copy()
    total=float(a.sum()); marks=(np.arange(maximum)+.5)*(total/maximum)
    pick=np.minimum(np.searchsorted(np.cumsum(a),marks,side='right'),len(a)-1)
    ids,counts=np.unique(pick,return_counts=True)
    return x[ids],counts.astype(np.float64)*(total/maximum)


def support_weights(area, footprint, kind):
    area=np.asarray(area,np.float64); footprint=np.asarray(footprint,bool)
    if footprint.ndim!=2 or footprint.shape[1]!=len(area) or not len(footprint) or np.any(area<0) or not np.isfinite(area).all():
        raise ValueError('view x site footprint required')
    if kind=='MEAN': return np.ones(len(footprint),np.float64)
    if kind=='AREA': return footprint.astype(float)@area
    if kind!='SUPPORT': raise ValueError('unknown updater')
    m=footprint.sum(axis=0)
    shared=np.divide(area,m,out=np.zeros_like(area),where=m>0)
    return footprint.astype(float)@shared


@dataclass(frozen=True)
class Evidence:
    key: str
    cosine: np.ndarray
    footprint: np.ndarray


def updated(p0, records: Sequence[Evidence], area, kind, temperature=1., beta=.25):
    p0=np.asarray(p0,np.float64)
    if p0.ndim!=1 or not np.isfinite(p0).all() or np.any(p0<0) or not np.isclose(p0.sum(),1):
        raise ValueError('prior must be a probability distribution')
    unique={}
    for r in records:
        if r.key in unique:
            old=unique[r.key]
            if not np.array_equal(old.cosine,r.cosine) or not np.array_equal(old.footprint,r.footprint):
                raise ValueError('contradictory exact observation identity')
        else: unique[r.key]=r
    if not unique: return p0.copy()
    rows=[unique[k] for k in sorted(unique)]
    w=support_weights(area,np.array([r.footprint for r in rows]),kind)
    if w.sum()<=0: w=np.ones(len(rows))
    if temperature<=0 or not 0<=beta<=1: raise ValueError('invalid temperature/beta')
    p=np.array([softmax(np.asarray(r.cosine)/temperature) for r in rows])
    if p.shape[1]!=len(p0): raise ValueError('class order/size mismatch')
    out=(1-beta)*p0+beta*np.average(p,axis=0,weights=w)
    return out/out.sum()


def query_scores(policy, footprints, area, pixels, theta_deg, *, interest=None, bins=None):
    """Anchor is index 0. This API contains no candidate semantic scores."""
    footprints=np.asarray(footprints,bool); area=np.asarray(area,np.float64)
    pixels=np.asarray(pixels,np.float64); theta=np.asarray(theta_deg,np.float64)
    n=len(footprints)
    if n<2 or footprints.shape!=(n,len(area)) or pixels.shape!=(n,) or theta.shape!=(n,):
        raise ValueError('common anchor and candidate geometries required')
    if policy=='AREA': values=pixels.copy()
    elif policy=='COVERAGE':
        if bins is None or len(bins)!=n: raise ValueError('surface direction bins required')
        values=np.array([len(set(b)-set(bins[0]))/len(set(b)) if b else 0. for b in bins])
    else:
        anchor=footprints[0]; denom=float(area[anchor].sum())
        j=(footprints & anchor).astype(float)@area/max(denom,1e-300)
        base=np.sqrt(j*np.clip(theta/30.,0,1))
        if policy=='VERIFY': values=base
        elif policy=='DISAGREEMENT':
            I=np.zeros_like(area) if interest is None else np.asarray(interest,np.float64)
            if I.shape!=area.shape or np.any(I<0) or not np.isfinite(I).all(): raise ValueError('anchor-derived interest required')
            if I.sum()<=1e-12: values=base
            else: values=base*(.5+.5*((footprints.astype(float)@I)/I.sum()))
        else: raise ValueError('unknown query policy')
    values[0]=-np.inf
    return values


def choose_second(policy, frame_ids, footprints, area, pixels, theta_deg, **kwargs):
    ids=list(map(int,frame_ids))
    if len(ids)!=len(set(ids)): raise ValueError('distinct candidate frames required')
    scores=query_scores(policy,footprints,area,pixels,theta_deg,**kwargs)
    k=min(range(1,len(ids)),key=lambda j:(-scores[j],-pixels[j],ids[j]))
    return ids[k]


def final_gate(candidate, g1, d2, eps=1e-10):
    for cohort in ('replica8','scannet_cf18'):
        for metric in METRICS:
            x,y=candidate[cohort][metric],g1[cohort][metric]
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y) or x<y-eps: return False
    c,b=candidate['scannet_cf18'],d2['scannet_cf18']
    return c['apall']>b['apall']+eps and c['ap50']>=b['ap50']-eps


def screen_gate(candidate, control, g1, *, changed, corrections_net, gt50_net, eps=1e-10):
    if not changed: return False
    delta=[]
    for cohort in ('replica_probe2','cf_probe2'):
        c,b,g=candidate[cohort],control[cohort],g1[cohort]
        needed=(c['apall'],c['ap50'],c['miou'],b['apall'],g['apall'],g['ap50'],g['miou'])
        if any(v is None or not np.isfinite(v) for v in needed): return False
        d=c['apall']-b['apall']; delta.append(d)
        if d<-.001-eps: return False
        if c['apall']<g['apall']-.001-eps or c['ap50']<g['ap50']-.001-eps or c['miou']<g['miou']-.002-eps: return False
    return float(np.mean(delta))>=.0005-eps and (corrections_net>=1 or gt50_net>=1)
