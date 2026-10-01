"""Small specification examples, NOT the native mapper or benchmark implementation.

No model loading, no dataset access. Production must validate actual lineage, C++ state
updates and exported masks in addition to these numerical properties.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np


def _prob(p):
    a = np.asarray(p, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any() or not np.isclose(a.sum(), 1):
        raise ValueError('Expected finite one-dimensional probability distribution')
    return a


def grouped_endpoints(sources: Mapping[str, np.ndarray | None]):
    """Inputs here are probabilities, not logits; absent sources stay absent."""
    present = {k: _prob(v) for k, v in sources.items() if v is not None}
    if any(k not in {'N', 'Q', 'F'} for k in present):
        raise ValueError('Unexpected source')
    if not present:
        return None, None
    if len({p.shape for p in present.values()}) != 1:
        raise ValueError('Category order/length mismatch')
    eq = np.mean(list(present.values()), axis=0)
    groups = []
    native = [present[k] for k in ('N', 'Q') if k in present]
    if native:
        groups.append(np.mean(native, axis=0))
    if 'F' in present:
        groups.append(present['F'])
    return eq, np.mean(groups, axis=0)


def weight_readout(sources, gamma):
    if not 1/3 <= gamma <= .5:
        raise ValueError('Gamma outside the fixed screen interval')
    eq, d2 = grouped_endpoints(sources)
    if eq is None:
        return None
    if gamma == 1/3:
        return eq.copy()
    if gamma == .5:
        return d2.copy()
    t = 6 * gamma - 2
    return (1-t)*eq + t*d2


def append_recovered(raw, owners, labels, candidate_labels: Mapping[int, int], *, minimum=100):
    """Toy append-only export. Production adds cap, coordinate hashes and lineage proofs."""
    r, o, y = map(np.asarray, (raw, owners, labels))
    if r.shape != o.shape or y.shape != o.shape or r.ndim != 1:
        raise ValueError('Aligned source arrays required')
    if np.any(y[o == 0] != 0):
        raise ValueError('Base unowned rows must have class zero')
    out, cls = o.copy(), y.copy()
    active = set(o[o > 0].tolist())
    for owner, label in sorted(candidate_labels.items()):
        if owner <= 0 or owner in active or label <= 0:
            raise ValueError('Recovery must target an absent owner with a real positive label')
        mask = (r == owner) & (o == 0)
        if mask.sum() >= minimum:
            out[mask], cls[mask] = owner, label
    return out, cls


def association_actions(intersections, local_areas, global_areas, mode='A1'):
    """Toy owners are positive column+1 IDs; production uses its actual owner registry."""
    from scipy.optimize import linear_sum_assignment
    I = np.asarray(intersections, float)
    L, G = np.asarray(local_areas, float), np.asarray(global_areas, float)
    if I.shape != (len(L), len(G)) or any(not np.isfinite(x).all() or (x < 0).any() for x in (I,L,G)):
        raise ValueError('Invalid support matrix')
    if (I.sum(1) > L+1e-9).any() or (I.sum(0) > G+1e-9).any():
        raise ValueError('Intersections exceed disjoint areas')
    F = np.divide(I, L[:,None], out=np.zeros_like(I), where=L[:,None]>0)
    ok = (I >= 100) & (F > .2)
    chosen = {}
    if mode == 'A1' and I.size:
        n,m = I.shape
        benefits = np.full((n,m+n), -1e6)
        benefits[:,:m] = np.where(ok, F-.2, -1e6)
        benefits[np.arange(n), m+np.arange(n)] = 0
        rows,cols = linear_sum_assignment(-benefits)
        chosen = {int(i):int(j) for i,j in zip(rows,cols) if j<m and ok[i,j]}
    elif mode in ('A2','A3'):
        for i in range(len(L)):
            candidates = np.flatnonzero(ok[i])
            if candidates.size:
                j = int(candidates[np.argmax(F[i,candidates])])
                chosen[i] = j
        if mode == 'A3':
            for i,j in list(chosen.items()):
                denom = I[i].sum()
                second = np.max(np.delete(F[i],j)) if len(G)>1 else 0.
                if denom <= 0 or I[i,j]/denom < .8 or F[i,j]-second < .1-1e-12:
                    del chosen[i]
            for j in set(chosen.values()):
                members = [i for i,k in chosen.items() if k==j]
                # Toy masks are disjoint, so sum(intersections) equals union intersection.
                if G[j] <= 0 or I[members,j].sum()/G[j] <= .2:
                    for i in members:
                        del chosen[i]
    elif mode != 'A1':
        raise ValueError('Unknown mode')
    return [('ASSIGN_EXISTING', chosen[i]+1) if i in chosen else ('USE_NATIVE', None) for i in range(len(L))]


def prepare_actions_without_native_state(actions: Sequence[tuple[str,int|None]], highest_owner: int):
    """Show the explicit allocation contract. This is NOT native integration validation."""
    result = []
    for action,owner in actions:
        if action == 'USE_NATIVE':
            if owner is not None:
                raise ValueError('Fallback must not disguise an owner plan')
            result.append((action,None))
        elif action == 'ASSIGN_EXISTING':
            if owner is None or owner <= 0:
                raise ValueError('Existing positive owner required')
            result.append((action,owner))
        elif action == 'CREATE_NEW':
            highest_owner += 1
            result.append((action,highest_owner))
        else:
            raise ValueError('Unknown action')
    return result, highest_owner


def crop_priority(crop, masks, winner_track):
    """winner_track is predecoded: zero-based track index, -1 for no SAM winner."""
    C = np.asarray(crop, np.int64)
    M = np.asarray(masks, bool)
    W = np.asarray(winner_track, np.int64)
    if C.ndim != 2 or M.ndim != 3 or M.shape[1:] != C.shape or W.shape != C.shape:
        raise ValueError('Aligned masks required')
    if (W < -1).any() or (W >= len(M)).any():
        raise ValueError('Invalid recorded winner')
    for i,m in enumerate(M):
        if np.any((W==i) & ~m):
            raise ValueError('Recorded winner outside its positive binary mask')
    groups = [int(x) for x in np.unique(C) if x>0]
    iou=np.zeros((len(M),len(groups))); cov=np.zeros_like(iou)
    for i,m in enumerate(M):
        for j,g in enumerate(groups):
            cm=C==g
            n=(m&cm).sum(); u=(m|cm).sum()
            iou[i,j]=n/u if u else 0
            cov[i,j]=n/cm.sum()
    attached={}
    for i in range(len(M)):
        if not groups:
            continue
        js=np.flatnonzero(iou[i] == iou[i].max())
        if len(js)!=1:
            continue
        j=int(js[0]); is_=np.flatnonzero(iou[:,j] == iou[:,j].max())
        if len(is_)==1 and is_[0]==i and cov[i,j]>=.8 and iou[i,j]>=.5:
            attached[i]=groups[j]
    out=C.copy(); additions={}; nextid=int(C.max())+1
    for i,m in enumerate(M):
        allow=i in attached or (m.sum()>0 and (m&(C>0)).sum()/m.sum()<.2)
        take=(W==i)&(C==0) if allow else np.zeros_like(C,bool)
        if take.any():
            owner=attached.get(i)
            if owner is None:
                owner=nextid;nextid+=1
            out[take]=owner
        additions[i]=take
    return out,additions,attached


def conflict_suppress(proposed_support, additions, stable_owners, visible_previous, *, minimum=100):
    """Inputs already passed own-map causal stability/depth checks in production."""
    P,A,V = (np.asarray(v,bool) for v in (proposed_support, additions, visible_previous))
    O=np.asarray(stable_owners,np.int64)
    if any(x.shape != O.shape for x in (P,A,V)):
        raise ValueError('Aligned supports required')
    vals,cnts=np.unique(O[V & (O>0)],return_counts=True)
    if cnts.sum()<minimum or not len(cnts) or cnts.max()/cnts.sum()<.8:
        return A.copy(),'ABSTAIN_SELF_UNKNOWN'
    self_owner=int(vals[np.argmax(cnts)])
    known=P&(O>0)
    if known.sum()<minimum:
        return A.copy(),'ABSTAIN_INSUFFICIENT_KNOWN'
    q=np.sum(known&(O!=self_owner))/known.sum()
    if q>.2:
        return np.zeros_like(A),'SUPPRESS_ADDITIONS'
    return A.copy(),'KEEP'


def gain_status(base, candidate):
    delta={k:(candidate[k]-base[k])*100 for k in ('apall','ap50','miou')}
    ok=(delta['apall'] >= .2-1e-10 and delta['ap50'] >= -.1-1e-10 and delta['miou'] >= -.1-1e-10)
    return 'NET_GAIN_WITH_GUARDRAILS' if ok else 'NO_QUALIFYING_NET_GAIN'
