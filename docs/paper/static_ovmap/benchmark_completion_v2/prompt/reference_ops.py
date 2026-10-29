"""CPU reference for the new contract. Not a benchmark runner or trained model.

Only checks algebra, aggregation, candidate selection, and manifest planning.
The production implementation must reuse the pinned repository's verified inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import numpy as np


def cap_vectors(b: np.ndarray, z: np.ndarray, degrees: float) -> tuple[np.ndarray,np.ndarray]:
    b, z = np.asarray(b), np.asarray(z)
    if b.ndim != 2 or b.shape != z.shape or not 0 <= degrees <= 180:
        raise ValueError('Aligned [objects,dimensions] and angle in [0,180] required')
    # Dimension-independent reference. Production parent CAP requires dim=768.
    out = z.copy(); flags = np.zeros(len(z), dtype=bool)
    if degrees == 180:
        return out, flags
    finite = np.isfinite(b).all(1) & np.isfinite(z).all(1)
    c = np.clip(np.sum(b.astype(np.float64)*z.astype(np.float64),axis=1),-1.,1.)
    active = finite & (np.arccos(c)>np.deg2rad(degrees))
    tangent = z.astype(np.float64)-c[:,None]*b.astype(np.float64)
    norm = np.linalg.norm(tangent,axis=1)
    flags = active & (norm<1e-10)
    normal = active & ~flags
    tau = np.deg2rad(degrees)
    out[normal]=(np.cos(tau)*b[normal]+np.sin(tau)*tangent[normal]/norm[normal,None]).astype(z.dtype)
    out[flags]=b[flags]
    return out,flags


def normalized_mix(b: np.ndarray,z: np.ndarray,alpha: float) -> tuple[np.ndarray,np.ndarray]:
    b,z=np.asarray(b),np.asarray(z)
    if b.ndim!=2 or b.shape!=z.shape or not 0<=alpha<=1:
        raise ValueError('Aligned vectors and alpha in [0,1] required')
    if alpha==0:return b.copy(),np.zeros(len(b),bool)
    if alpha==1:return z.copy(),np.zeros(len(z),bool)
    w=(1-alpha)*b.astype(np.float64)+alpha*z.astype(np.float64)
    lengths=np.linalg.norm(w,axis=1)
    fallback=np.isfinite(lengths)&(lengths<=1e-12)
    valid=np.isfinite(lengths)&~fallback
    out=np.full_like(z,np.nan)
    out[valid]=(w[valid]/lengths[valid,None]).astype(z.dtype)
    out[fallback]=b[fallback]
    return out,fallback


def offset_scores(scores: np.ndarray, base_mask: np.ndarray, gamma: float) -> np.ndarray:
    scores=np.asarray(scores);base_mask=np.asarray(base_mask,dtype=bool)
    if scores.ndim!=2 or base_mask.shape!=(scores.shape[1],) or not np.isfinite(gamma):
        raise ValueError('Aligned scores/mask and finite cosine offset required')
    out=scores.copy();out[:,base_mask]-=gamma
    return out


def predict(scores: np.ndarray, ids: np.ndarray, defined: np.ndarray|None=None) -> np.ndarray:
    scores=np.asarray(scores);ids=np.asarray(ids)
    if scores.ndim!=2 or len(ids)!=scores.shape[1] or len(np.unique(ids))!=len(ids):
        raise ValueError('Unique aligned class axis required')
    valid=np.isfinite(scores).all(1)
    if defined is not None:
        d=np.asarray(defined,bool)
        if d.shape!=valid.shape:raise ValueError('defined mask shape differs')
        valid &= d
    out=np.zeros(len(scores),np.int64)
    out[valid]=ids[scores[valid].argmax(1)]
    return out


def harmonic(a: float|None,b: float|None) -> float|None:
    if a is None or b is None:return None
    return 2*a*b/(a+b) if a+b else 0.


def recognition(target: np.ndarray,prediction: np.ndarray,base: list[int],novel: list[int]) -> dict[str,Any]:
    target=np.asarray(target);prediction=np.asarray(prediction)
    if target.ndim!=1 or target.shape!=prediction.shape:raise ValueError('One prediction per original required')
    if set(base)&set(novel):raise ValueError('Base/novel labels overlap')
    if not set(target.tolist()) <= set(base)|set(novel):raise ValueError('Unknown target role')
    values={int(c):float(np.mean(prediction[target==c]==c)) for c in np.unique(target)}
    def group(classes):
        supported=[c for c in classes if c in values];mask=np.isin(target,classes)
        macro=float(np.mean([values[c] for c in supported])) if supported else None
        micro=float(np.mean(prediction[mask]==target[mask])) if mask.any() else None
        return macro,micro
    bm,bi=group(base);nm,ni=group(novel)
    return dict(B_macro=bm,N_macro=nm,H_macro=harmonic(bm,nm),
                B_micro=bi,N_micro=ni,H_micro=harmonic(bi,ni),
                originals=len(target),abstentions=int((prediction==0).sum()))


def recipes(spec: dict[str,Any]) -> list[dict[str,Any]]:
    rows=[]
    for kind in ('MIX','CAP','NMIX'):
        strengths=spec['tau_grid'] if kind=='CAP' else spec['alpha_grid']
        for v in strengths:
            for g in spec['gamma_grid']:
                rows.append(dict(kind=kind,value=float(v),gamma=float(g),
                  operations=int(v!=180 if kind=='CAP' else v!=1)+int(g!=0),
                  candidate_id=f'{kind}|{v:g}|{g:+g}'))
    return sorted(rows,key=lambda r:r['candidate_id'])


def best(rows: list[dict[str,Any]], tolerance: float=1e-10) -> dict[str,Any]:
    candidates=[r for r in rows if r['metrics']['H_macro'] is not None]
    if not candidates:raise ValueError('No candidate with both supported groups')
    for f in ('H_macro','min_macro','B_macro'):
        def value(r):
            m=r['metrics'];return min(m['B_macro'],m['N_macro']) if f=='min_macro' else m[f]
        vmax=max(value(r) for r in candidates)
        candidates=[r for r in candidates if vmax-value(r)<=tolerance]
    return min(candidates,key=lambda r:(r['recipe']['operations'],r['recipe']['candidate_id']))


def oof(preds17: np.ndarray, preds29: np.ndarray, target: np.ndarray, families: np.ndarray,
        candidate_recipes: list[dict[str,Any]], base: list[int],novel: list[int]):
    target=np.asarray(target);families=np.asarray(families)
    if preds17.shape!=preds29.shape or preds17.shape!=(len(candidate_recipes),len(target)):
        raise ValueError('candidate by original predictions required')
    out=np.zeros((2,len(target)),np.int64);folds=[]
    for family in sorted(set(families.tolist())):
        hold=families==family;train=~hold
        rows=[dict(recipe=r,metrics=recognition(target[train],preds17[i,train],base,novel))
              for i,r in enumerate(candidate_recipes)]
        chosen=best(rows);i=next(i for i,r in enumerate(candidate_recipes) if r['candidate_id']==chosen['recipe']['candidate_id'])
        out[0,hold]=preds17[i,hold];out[1,hold]=preds29[i,hold]
        folds.append(dict(held_family=family,recipe=chosen['recipe']['candidate_id']))
    return out,folds


def stable_softmax(value: np.ndarray) -> np.ndarray:
    value=np.asarray(value,np.float64);exp=np.exp(value-np.max(value));return exp/exp.sum()


def grouped(scores: dict[str,np.ndarray|None],temperatures: dict[str,float]) -> np.ndarray|None:
    def pool(names):
        vals=[]
        for name in names:
            x=scores.get(name)
            if x is not None:
                temp=temperatures[name]
                if not temp>0:raise ValueError('Positive temperature required')
                vals.append(stable_softmax(np.asarray(x)/temp))
        return None if not vals else np.mean(vals,axis=0)
    groups=[g for g in (pool(('N','Q')),pool(('F',))) if g is not None]
    if not groups:return None
    p=np.mean(groups,axis=0);return p/p.sum()


def global_label(source_scores: dict[str,np.ndarray|None],ids: np.ndarray,
                 temperatures: dict[str,float],multisource: bool) -> int:
    if not multisource:
        f=source_scores.get('F')
        return 0 if f is None else int(ids[np.argmax(f)])
    p=grouped(source_scores,temperatures)
    return 0 if p is None else int(ids[p.argmax()])


def check_dag(dependencies: dict[str,list[str]]) -> list[str]:
    order=[];todo={k:set(v) for k,v in dependencies.items()}
    if any(not v <= set(todo) for v in todo.values()):raise ValueError('Unknown dependency')
    while todo:
        ready=sorted(k for k,v in todo.items() if not v)
        if not ready:raise ValueError('Dependency cycle')
        for k in ready:order.append(k);todo.pop(k)
        for v in todo.values():v.difference_update(ready)
    return order
