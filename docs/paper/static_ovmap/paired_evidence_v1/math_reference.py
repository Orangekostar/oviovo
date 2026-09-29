"""Synthetic specification checks; NOT a repository adapter or benchmark run.

Run: python math_reference.py
Only numpy is required. No files, network, model, or annotations are loaded.
"""
from __future__ import annotations
import itertools
import json
import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('finite vector required')
    y = np.exp(x - x.max())
    return y / y.sum()


def probability_vector(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError('positive finite probability vector required')
    if not np.isclose(x.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError('probabilities do not sum to one')
    return x


def local_residual(base, residual, indices, eta):
    p = probability_vector(base)
    r = np.asarray(residual, dtype=np.float64)
    ix = np.asarray(indices, dtype=np.int64)
    if r.shape != p.shape or not np.isfinite(r).all() or not np.isfinite(eta) or eta < 0:
        raise ValueError('invalid residual')
    if not len(ix) or len(set(ix.tolist())) != len(ix) or np.any(ix < 0) or np.any(ix >= len(p)):
        raise ValueError('invalid candidate indices')
    if eta == 0 or np.ptp(r[ix]) == 0:
        return p.copy()
    out = p.copy()
    out[ix] = p[ix].sum() * softmax(np.log(p[ix]) + eta*r[ix])
    return out


def common_residual(fc_scores, ovr_scores, temperature):
    a, b = np.asarray(fc_scores, float), np.asarray(ovr_scores, float)
    if a.shape != b.shape or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('invalid paired scores/temperature')
    return (b-a)/temperature


def sensitivity(features, areas, text_delta, temperature):
    f = np.asarray(features, dtype=np.float64)
    alpha = np.asarray(areas, dtype=np.float64)
    delta = np.asarray(text_delta, dtype=np.float64)
    if f.ndim != 2 or alpha.shape != (len(f),) or delta.shape != (f.shape[1],):
        raise ValueError('misaligned sensitivity inputs')
    if np.any(alpha <= 0) or temperature <= 0 or not all(np.isfinite(v).all() for v in (f, alpha, delta)):
        raise ValueError('invalid sensitivity inputs')
    alpha = alpha/alpha.sum()
    a = alpha@f
    norm = np.linalg.norm(a)
    if norm <= 1e-12:
        raise ValueError('undefined mean direction')
    v = a/norm
    j = alpha[:,None]*(delta-v*np.dot(v,delta))[None,:]/(temperature*norm)
    relative_j = j*np.linalg.norm(f,axis=1)[:,None]/np.sqrt(f.shape[1])
    return j, relative_j


def support_kernel(masks, block_ids):
    if len(masks) != len(block_ids):
        raise ValueError('kernel block ids mismatch')
    n = len(masks)
    k = np.eye(n)
    for i in range(n):
        if masks[i] is None:
            continue
        a = np.asarray(masks[i], bool)
        if not a.any():
            raise ValueError('empty support')
        for j in range(i):
            if masks[j] is None or block_ids[i] != block_ids[j]:
                continue
            b = np.asarray(masks[j], bool)
            if a.shape != b.shape or not b.any():
                raise ValueError('masks within same image block must align')
            k[i,j] = k[j,i] = np.count_nonzero(a&b)/np.sqrt(np.count_nonzero(a)*np.count_nonzero(b))
    return k


def covariance_proxy(loadings, kernel):
    """loadings: [sources, atoms, common padded feature space].

    Distinct embedding spaces must be given orthogonal coordinate blocks.
    Production can instead compute the compatible blocks without padding.
    """
    h = np.asarray(loadings, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if h.ndim != 3 or k.shape != (h.shape[1], h.shape[1]):
        raise ValueError('proxy shape mismatch')
    if not np.allclose(k,k.T,atol=1e-12) or np.linalg.eigvalsh(k).min() < -1e-10:
        raise ValueError('kernel is not PSD')
    s = np.einsum('med,ef,nfd->mn',h,k,h)
    s = (s+s.T)/2
    floor = .1*max(float(np.diag(s).mean()),1e-12)
    return s+floor*np.eye(len(s))


def min_variance_weights(covariance):
    s = np.asarray(covariance, dtype=np.float64)
    m = len(s)
    if s.shape != (m,m) or not m or m>3 or not np.isfinite(s).all():
        raise ValueError('expected a finite 1..3 source matrix')
    if not np.allclose(s,s.T,atol=1e-12) or np.linalg.eigvalsh(s).min() <= 0:
        raise ValueError('positive definite covariance required')
    # Normalize only for the solve's conditioning; this does not change alpha.
    t = s/max(float(np.diag(s).mean()),1e-300)
    candidates=[]
    for n in range(1,m+1):
        for active in itertools.combinations(range(m),n):
            ix=np.asarray(active)
            raw=np.linalg.solve(t[np.ix_(ix,ix)],np.ones(n))
            raw=raw/raw.sum()
            if raw.min() < -1e-10:
                continue
            raw=np.maximum(raw,0); raw/=raw.sum()
            w=np.zeros(m); w[ix]=raw
            candidates.append((float(w@t@w),active,w))
    _,_,w=min(candidates,key=lambda item:(item[0],item[1]))
    return w, float(w@s@w)


def pair_graph(differences, weights, base):
    p=probability_vector(base); n=len(p)
    pairs=list(itertools.combinations(range(n),2))
    d=np.asarray(differences,float); w=np.asarray(weights,float)
    if d.shape!=(len(pairs),) or w.shape!=d.shape or np.any(w<=0):
        raise ValueError('complete positive-weight pair graph required')
    if not np.isfinite(d).all() or not np.isfinite(w).all():
        raise ValueError('nonfinite graph input')
    w=w/w.mean()
    L=np.zeros((n,n)); b=np.zeros(n)
    for (a,c),value,weight in zip(pairs,d,w):
        L[a,a]+=weight; L[c,c]+=weight
        L[a,c]-=weight; L[c,a]-=weight
        b[a]+=weight*value; b[c]-=weight*value
    u0=np.log(p); u0-=u0.mean()
    u=np.linalg.solve(L+n*np.eye(n),b+n*u0)
    u-=u.mean()
    return softmax(u),u


def selftest():
    rng=np.random.default_rng(1729); checks=[]
    p=softmax(np.array([.1,-.2,.3,.5,-1,.7,.9])); r=rng.normal(size=len(p)); ix=np.array([0,2,5])
    assert np.array_equal(local_residual(p,r,ix,0),p)
    checks.append('local_zero_identity')
    v=local_residual(p,r,ix,1.0)
    outside=np.setdiff1d(np.arange(len(p)),ix)
    assert np.array_equal(v[outside],p[outside]) and abs(v[ix].sum()-p[ix].sum())<1e-14
    checks.append('local_outside_and_inside_mass')
    assert np.array_equal(local_residual(p,np.ones(len(p)),ix,2),p)
    checks.append('local_constant_residual_identity')
    assert np.array_equal(common_residual(r,r,.02),np.zeros_like(r))
    checks.append('common_temperature_null_residual')
    full,_=pair_graph([0.0]*21,[1.0]*21,p)
    assert abs(full.sum()-1)<1e-12
    # Equivariance of explicit candidate-set transformation (no tie convention).
    perm=rng.permutation(len(p)); inv=np.argsort(perm)
    vp=local_residual(p[perm],r[perm],inv[ix],.5)
    assert np.allclose(vp[inv],local_residual(p,r,ix,.5),atol=1e-14)
    checks.append('local_class_permutation')
    f=rng.normal(size=(3,5)); area=np.array([1.,2.,3.]); delta=rng.normal(size=5)
    j,_=sensitivity(f,area,delta,.7)
    def contrast(ff):
        a=area@ff/area.sum(); return float(delta@(a/np.linalg.norm(a))/.7)
    numerical=np.zeros_like(j); eps=1e-6
    for e in range(3):
        for k in range(5):
            fp=f.copy(); fm=f.copy(); fp[e,k]+=eps; fm[e,k]-=eps
            numerical[e,k]=(contrast(fp)-contrast(fm))/(2*eps)
    assert np.allclose(j,numerical,atol=1e-8,rtol=1e-7)
    checks.append('aggregation_jacobian_finite_difference')
    masks=[np.array([1,1,0,0],bool),np.array([1,0,1,0],bool),np.ones(4,bool),None]
    k=support_kernel(masks,['a','a','b','a'])
    assert np.linalg.eigvalsh(k).min()>-1e-12 and k[0,1]==.5 and k[0,2]==0 and k[3,0]==0
    checks.append('support_kernel_psd_and_isolated_unknown')
    h=rng.normal(size=(3,4,5)); s=covariance_proxy(h,k)
    assert np.linalg.eigvalsh(s).min()>0
    checks.append('proxy_spd_with_relative_floor')
    w,obj=min_variance_weights(s)
    grid=[]
    for a in np.linspace(0,1,101):
        for b in np.linspace(0,1-a,101):
            x=np.array([a,b,1-a-b]); grid.append(float(x@s@x))
    assert np.all(w>=0) and abs(w.sum()-1)<1e-12 and obj<=min(grid)+1e-10
    checks.append('active_set_qp_against_simplex_grid')
    z=rng.normal(size=(3,len(p))); zbar=z.mean(0)
    d=np.array([zbar[a]-zbar[b] for a,b in itertools.combinations(range(len(p)),2)])
    got,u=pair_graph(d,np.ones_like(d),p)
    expected=softmax(.5*zbar+.5*np.log(p))
    assert np.allclose(got,expected,atol=1e-13) and abs(u.sum())<1e-12
    checks.append('anchored_graph_closed_form')
    diagonal=np.diag([1.,2.,3.]); wa,oa=min_variance_weights(diagonal)
    assert np.allclose(wa,np.array([1.,.5,1/3])/sum([1,.5,1/3]))
    checks.append('no_cross_covariance_diagonal_limit')
    # Each shuffled-rho block remains SPD because its magnitude is < 1.
    for rho in [-.8,.1,.7]:
        c=np.diag([.7,3.,1.]); c[0,1]=c[1,0]=rho*np.sqrt(.7*3.)
        assert np.linalg.eigvalsh(c).min()>0
    checks.append('shuffled_rho_preserves_spd')
    # Return value is a synthetic operator check only, not an empirical claim.
    return {'status':'PASS','kind':'SYNTHETIC_MATH_SPECIFICATION_ONLY','checks':checks,
            'check_count':len(checks),'vision_forwards':0,'benchmark_evaluations':0,
            'maximum_jacobian_absolute_error':float(np.max(np.abs(j-numerical)))}


if __name__=='__main__':
    print(json.dumps(selftest(),indent=2))
