"""Spatial cross-fitting, actual bilinear donor purge and matched graph nulls."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from .common import verified, write, require, unit_identity, cached_unit, PathResolver
from .solver import _validate, estimate_view_bias, center_bias, huber_location, SolverConfig

NULL_SEEDS=(101,211,307,401,503)

def spatial_folds(xyz,available):
    xyz=np.asarray(xyz); sites=np.flatnonzero(available)
    if not len(sites): return [np.empty(0,int) for _ in range(4)]
    if xyz.shape!=(len(available),3) or not np.isfinite(xyz[sites]).all(): raise ValueError('Finite per-condition physical positions required')
    axis=int(np.argmax(np.ptp(xyz[sites],axis=0)))
    order=sorted(sites,key=lambda s:(xyz[s,axis],xyz[s,0],xyz[s,1],xyz[s,2],int(s)))
    return np.array_split(np.asarray(order,int),4)

def bilinear_donors(uv,record,valid):
    from static_ovmap.learned_object_readout.observations import normalized_grid
    grid=normalized_grid(uv,record['original_hw'],record['resized_hw'],record['padded_hw']).astype(np.float32).astype(np.float64)
    h,w=record['raw_hw']; point=(grid+1)*np.array([w,h])/2-.5
    donors={}
    for s in np.flatnonzero(valid):
        px,py=point[s]; ix,iy=int(np.floor(px)),int(np.floor(py)); dx,dy=px-ix,py-iy
        donors[int(s)]={(x,y) for x,y,weight in ((ix,iy,(1-dx)*(1-dy)),(ix+1,iy,dx*(1-dy)),(ix,iy+1,(1-dx)*dy),(ix+1,iy+1,dx*dy)) if weight>0 and 0<=x<w and 0<=y<h}
    return donors

def purge_fit_sites(fit,holdout,donors):
    result=np.asarray(fit,bool).copy()
    for view in donors:
        forbidden=set().union(*(view.get(int(s),set()) for s in holdout))
        for s in np.flatnonzero(result):
            if view.get(int(s),set()) & forbidden: result[s]=False
    return result

def independent_bias(raw,valid,fit,estimate,key,condition,fold,seed):
    active=estimate.active_views; positions={v:np.flatnonzero(valid[v]&fit) for v in active}
    lap=np.zeros((len(active),len(active))); rhs=np.zeros((len(active),raw.shape[-1])); indices={v:i for i,v in enumerate(active)}; degenerate=False
    for edge in estimate.edges:
        v,u,n=edge['v'],edge['u'],edge['n_shared']; shared=np.flatnonzero(valid[v]&valid[u]&fit)
        degenerate |= np.array_equal(positions[v],shared) and np.array_equal(positions[u],shared)
        token=json.dumps([key,condition,fold,v,u,seed],ensure_ascii=False,separators=(',',':')).encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'big'))
        sv=rng.choice(positions[v],n,replace=False); su=rng.choice(positions[u],n,replace=False)
        delta=huber_location(raw[v,sv].astype(np.float64)-raw[u,su].astype(np.float64),SolverConfig())
        i,j=indices[v],indices[u]; lap[i,i]+=n; lap[j,j]+=n; lap[i,j]-=n; lap[j,i]-=n; rhs[i]+=n*delta; rhs[j]-=n*delta
    kkt=np.zeros((len(active)+1,len(active)+1)); kkt[:-1,:-1]=lap; kkt[-1,:-1]=1; kkt[:-1,-1]=1
    target=np.zeros((len(active)+1,raw.shape[-1])); target[:-1]=rhs
    try: values=np.linalg.solve(kkt,target)[:-1]
    except np.linalg.LinAlgError: return None,degenerate
    if not np.isfinite(values).all(): return None,degenerate
    values-=values.mean(0); bias=np.zeros((raw.shape[0],raw.shape[-1])); bias[active]=values
    return bias,degenerate

def diagnostic_unit(raw,valid,xyz,available,uv,grids,key,condition):
    raw,valid=_validate(raw,valid); blocks=spatial_folds(xyz,np.asarray(available,bool)&valid.any(0))
    donors=[bilinear_donors(uv[v],grids[v],valid[v]) for v in range(len(valid))]; units=[]
    for fold,holdout in enumerate(blocks):
        fit=np.asarray(available,bool)&valid.any(0); fit[holdout]=False; before=int(fit.sum()); fit=purge_fit_sites(fit,holdout,donors)
        estimate=estimate_view_bias(raw,valid,fit_sites=fit); center=center_bias(raw,valid,fit)
        biases={'ZERO':np.zeros_like(estimate.bias)}
        if estimate.status=='OK': biases['M1']=estimate.bias
        if center.status=='OK': biases['C1']=center.bias
        null_flags={}
        if estimate.status=='OK':
            for seed in NULL_SEEDS:
                value,degenerate=independent_bias(raw,valid,fit,estimate,key,condition,fold,seed)
                null_flags[str(seed)]='DEGENERATE_SUPPORT_NULL' if degenerate else 'OK'
                if value is not None: biases['NULL_'+str(seed)]=value
        support=[]
        for v,u in itertools.combinations(estimate.active_views,2):
            shared=holdout[valid[v,holdout]&valid[u,holdout]]
            if len(shared)>=4: support.append((v,u,shared))
        scores={}
        for method,bias in biases.items():
            numerator=denominator=0.; count=0
            for v,u,sites in support:
                difference=raw[v,sites].astype(np.float64)-raw[u,sites].astype(np.float64)
                numerator+=float(((difference-(bias[v]-bias[u]))**2).sum()); denominator+=float((difference**2).sum()); count+=len(sites)
            scores[method]=dict(N=numerator,D=denominator,support=count,defined=denominator>1e-20)
        units.append(dict(fold=fold,heldout_sites=holdout.tolist(),fit_sites=np.flatnonzero(fit).tolist(),purged_sites=before-int(fit.sum()),status=estimate.status,null_status=null_flags,scores=scores))
    return dict(status='COMPLETE',key=key,condition=condition,folds=units,GT_used=False)

def aggregate(rows,method,paired_with=None):
    families={}; objects=0; N=D=0.; support=0
    for row in rows:
        numerator=denominator=0.; count=0
        for fold in row.get('folds',[]):
            scores=fold['scores']; value=scores.get(method)
            if value is None or not value['defined']: continue
            if paired_with and (paired_with not in scores or not scores[paired_with]['defined']): continue
            numerator+=value['N']; denominator+=value['D']; count+=value['support']
        if denominator<=1e-20: continue
        objects+=1; families.setdefault(row['family'],[]).append(1-numerator/denominator); N+=numerator; D+=denominator; support+=count
    means={family:float(np.mean(values)) for family,values in families.items()}
    return dict(method=method,paired_with=paired_with,R_family_mean=float(np.mean(list(means.values()))) if means else None,objects=objects,families=len(means),per_family_R=means,N=N,D=D,support=support)

def run(binding,fc,loader):
    from static_ovmap.frozen_fc_routing.common import objects
    root=Path(binding['output_root']); lr=Path(binding['lr_parent_root']); resolver=PathResolver(binding['path_map'])
    features=resolver.rewrite(verified(lr/'features/train-dev.json')); results=[]
    for role in ('train','dev'):
        items=sorted([o for o in objects(binding,role) if o['base']],key=lambda o:o['key'])
        expected=622 if role=='train' else 69
        if len(items)!=expected: raise ValueError('Original base support changed')
        for obj in items:
            for condition in range(4):
                path=root/'diagnostics'/role/(obj['key'].replace(':','_')+f'_c{condition}.json')
                identity=unit_identity(binding,'diagnose',inputs=obj['inputs']['sha256'],features=features['identity'],condition=condition,prefix=8)
                row=cached_unit(path,identity)
                if row is None:
                    x,_,_=loader.load(obj,condition,8); raw=x['local_raw'].cpu().numpy(); valid=(x['local_valid']&x['view_valid'][:,None]).cpu().numpy()
                    with np.load(obj['inputs']['path'],allow_pickle=False) as arrays:
                        needed={'site_xyz','site_available','uv'}
                        if not needed<=set(arrays.files): row=dict(status='UNSUPPORTED_STRICT_DIAGNOSTIC',reason='MISSING_DIAGNOSTIC_METADATA',folds=[])
                        else:
                            grids=[features['frames'][obj['scene']+':'+str(v['frame_id'])] for v in obj['views'][:8]]
                            row=diagnostic_unit(raw,valid,arrays['site_xyz'][condition],arrays['site_available'][condition],arrays['uv'][condition,:8],grids,obj['key'],condition)
                    row=write(path,dict(row,input_identity=identity,family=obj['family'],key=obj['key'],condition=condition,role=role))
                results.append(row)
    summary={}; signal=True
    for role in ('train','dev'):
        summary[role]={}
        for condition in range(4):
            subset=[r for r in results if r['role']==role and r['condition']==condition]
            names=('ZERO','M1','C1',*('NULL_'+str(s) for s in NULL_SEEDS))
            summary[role][str(condition)]=dict(own={m:aggregate(subset,m) for m in names},paired_M1={m:dict(M1=aggregate(subset,'M1',m),control=aggregate(subset,m,'M1')) for m in names if m!='M1'},unsupported=sum(r.get('reason')=='MISSING_DIAGNOSTIC_METADATA' for r in subset))
        clean=summary[role]['0']['own']['M1']; floor=(50,8) if role=='train' else (20,3)
        signal &= clean['objects']>=floor[0] and clean['families']>=floor[1] and clean['R_family_mean'] is not None and clean['R_family_mean']>0
    return write(root/'diagnostic_summary.json',dict(status='COMPLETE',summary=summary,mechanism_signal=bool(signal),GT_used=False,not_significance_or_causality=True,scope='SPATIAL_HOLDOUT_WITH_DONOR_PURGE_RECEPTIVE_FIELDS_MAY_OVERLAP',unit_count=len(results),raw_units_retained=True))
