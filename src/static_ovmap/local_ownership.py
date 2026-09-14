"""Neutral source atoms, shared unaries and bounded deterministic local ownership."""
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from .fusion_attribution import resolve_native_owners

METHODS={'LO_U00_OVI_FILL':'global_priority','LO_U00_LOCAL':'local_evidence','LO_U00_SPATIAL':'local_evidence'}


def method_family(method):
    if method not in METHODS:raise ValueError('unknown local ownership method: '+str(method))
    return METHODS[method]


def cache_identity(binding):
    return hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def build_membership_atoms(xyz,masks,kept,fallback_owners,cell_m=.02,connectivity_m=.015,chunk=65536):
    xyz=np.asarray(xyz);n=len(xyz);masks=np.asarray(masks);kept=np.asarray(kept)
    if xyz.shape!=(masks.shape[1],3) or not np.isfinite(xyz).all():raise ValueError('finite aligned metric coordinates required')
    # Exact bits, not a collision-prone hash; one compact signature per original point.
    packed=np.empty((n,(len(kept)+7)//8),np.uint8)
    for start in range(0,n,chunk):packed[start:start+chunk]=np.packbits(masks[kept,start:start+chunk].T,axis=1)
    dtype=np.dtype([('x','<i8'),('y','<i8'),('z','<i8'),('signature',f'V{packed.shape[1]}')])
    keys=np.empty(n,dtype);cells=np.floor(np.asarray(xyz,dtype=np.float64)/cell_m).astype(np.int64)
    for i,k in enumerate(['x','y','z']):keys[k]=cells[:,i]
    keys['signature']=np.ascontiguousarray(packed).view(f'V{packed.shape[1]}').ravel()
    order=np.argsort(keys,kind='stable');starts=np.r_[0,1+np.flatnonzero(keys[order][1:]!=keys[order][:-1]),n]
    point_to_atom=np.empty(n,np.int64);next_id=0
    for start,end in zip(starts[:-1],starts[1:]):
        ids=order[start:end];points=np.asarray(xyz[ids],dtype=np.float64)
        if len(ids)==1 or np.linalg.norm(np.ptp(points,axis=0))<=connectivity_m:
            point_to_atom[ids]=next_id;next_id+=1;continue
        # Radius flood-fill queries one point at a time: no dense local/global distance matrix.
        tree=cKDTree(points);unseen=np.ones(len(ids),bool)
        for seed in range(len(ids)):
            if not unseen[seed]:continue
            pending=[seed];unseen[seed]=False;members=[]
            while pending:
                i=pending.pop();members.append(i)
                neighbors=np.asarray(tree.query_ball_point(points[i],connectivity_m),dtype=np.int64)
                neighbors=neighbors[unseen[neighbors]];unseen[neighbors]=False;pending.extend(neighbors.tolist())
            point_to_atom[ids[members]]=next_id;next_id+=1
    counts=np.bincount(point_to_atom,minlength=next_id)
    centroids=np.column_stack([np.bincount(point_to_atom,weights=xyz[:,i],minlength=next_id)/counts for i in range(3)])
    # The solver's fixed visit order is geometry-derived, independent of candidate/source priority.
    atom_order=np.lexsort((np.arange(next_id),centroids[:,2],centroids[:,1],centroids[:,0]))
    inverse=np.empty(next_id,np.int64);inverse[atom_order]=np.arange(next_id);point_to_atom=inverse[point_to_atom]
    centroids=centroids[atom_order];counts=counts[atom_order]
    representatives=np.full(next_id,n,np.int64);np.minimum.at(representatives,point_to_atom,np.arange(n))
    candidate_blocks=[];degree=np.zeros(next_id,np.int64)
    for start in range(0,next_id,chunk):
        membership=masks[kept[:,None],representatives[start:start+chunk][None,:]].T
        aa,cc=np.nonzero(membership);candidate_blocks.append(kept[cc]);degree[start:start+chunk]=membership.sum(axis=1)
    candidates=np.concatenate(candidate_blocks).astype(np.int64);indptr=np.r_[0,np.cumsum(degree)]
    fallback=np.asarray(fallback_owners[representatives],np.int64)-1
    if not np.array_equal(fallback_owners,fallback[point_to_atom]+1):raise ValueError('fallback varies within exact-membership atom')
    return {'point_to_atom':point_to_atom,'xyz':centroids,'counts':counts,'indptr':indptr,'candidates':candidates,
        'fallback':fallback,'representatives':representatives}


def solve_local_unaries(atoms,views,scores,min_views=2,beta=.05):
    aa=np.repeat(np.arange(len(atoms['counts'])),np.diff(atoms['indptr']));cc=atoms['candidates']
    is_fallback=cc==atoms['fallback'][aa];feasible=(views>=min_views)|is_fallback
    evidence=np.where(views>0,scores,.5)
    if np.any(~np.isfinite(evidence[feasible])):raise ValueError('usable evidence must be finite')
    costs=1-evidence+beta*(~is_fallback);costs[~feasible]=np.inf
    minimum=np.full(len(atoms['counts']),np.inf);np.minimum.at(minimum,aa,costs)
    labels=np.full(len(minimum),np.iinfo(np.int64).max,np.int64)
    tied=feasible&(costs==minimum[aa]);np.minimum.at(labels,aa[tied],cc[tied])
    f=tied&is_fallback;labels[aa[f]]=cc[f];labels[np.diff(atoms['indptr'])==0]=-1
    return {'labels':labels,'costs':costs,'feasible':feasible,'views':views,'scores':scores}


def build_evidence_graph(atoms,histograms,neighbors=6,radius=.03,sigma=.02,epsilon=1e-12,threads=8,chunk=65536):
    xyz=atoms['xyz'];n=len(xyz)
    if not n:return {'edges':np.empty((0,2),np.int64),'weights':np.empty(0),'shared_views':np.empty(0,np.int64),'candidate_edge_count':0}
    distance,index=cKDTree(xyz).query(xyz,k=min(neighbors+1,n+1),distance_upper_bound=radius,workers=threads)
    aa=np.broadcast_to(np.arange(n)[:,None],index.shape)
    valid=np.isfinite(distance)&(index>=0)&(index<n)&(index!=aa)
    # Coincident centers can displace self from the returned k neighbors.
    valid &= np.cumsum(valid,axis=1)<=neighbors
    edges=np.unique(np.sort(np.column_stack([aa[valid],index[valid]]),axis=1),axis=0)
    shared=np.zeros(len(edges),np.int32);qsum=np.zeros(len(edges))
    for hist in histograms:
        area=np.asarray(hist.sum(axis=1)).ravel();normalized=hist.multiply(1/np.maximum(area,1)[:,None]).tocsr()
        for start in range(0,len(edges),chunk):
            end=min(len(edges),start+chunk);a,b=edges[start:end].T;seen=(area[a]>0)&(area[b]>0)
            q=np.asarray(normalized[a].multiply(normalized[b]).sum(axis=1)).ravel()
            shared[start:end]+=seen;qsum[start:end]+=q
    d=np.linalg.norm(xyz[edges[:,0]]-xyz[edges[:,1]],axis=1)
    g=np.exp(-d*d/(2*sigma*sigma))*np.maximum(0,2*qsum/np.maximum(shared,1)-1);g[shared<2]=0
    weighted=np.bincount(edges.ravel(),weights=np.repeat(g,2),minlength=n)
    a,b=edges.T;mass=atoms['counts']/np.maximum(weighted,epsilon)
    weights=g*np.minimum(mass[a],mass[b]);keep=weights>0
    return {'edges':edges[keep],'weights':weights[keep],'shared_views':shared[keep],
        'mean_same_instance':(qsum/np.maximum(shared,1))[keep],'candidate_edge_count':len(edges),
        'unsupported_edge_count':int((~keep).sum())}


def solve_spatial_ownership(atoms,unaries,graph,spatial_lambda=.10,max_sweeps=5):
    labels=unaries['labels'].copy();edges=graph['edges'];weights=graph['weights'];n=len(labels)
    aa=np.repeat(np.arange(n),np.diff(atoms['indptr']));cc=atoms['candidates'];costs=unaries['costs'];mass=atoms['counts']
    adjacency=csr_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
    def energy():
        keep=(cc==labels[aa]);return float(np.sum(mass[aa[keep]]*costs[keep])+spatial_lambda*np.sum(weights*(labels[edges[:,0]]!=labels[edges[:,1]])))
    energies=[energy()];changes=[];eligible=np.flatnonzero(np.bincount(aa,weights=unaries['feasible'],minlength=n)>1)
    for sweep in range(max_sweeps):
        changed=0
        for a in eligible:
            start,end=atoms['indptr'][a:a+2];options=cc[start:end];feasible=unaries['feasible'][start:end]
            left,right=adjacency.indptr[a:a+2];nb=adjacency.indices[left:right];ww=adjacency.data[left:right]
            local=mass[a]*costs[start:end]+spatial_lambda*np.sum(ww[None,:]*(options[:,None]!=labels[nb][None,:]),axis=1)
            current=np.flatnonzero(options==labels[a])[0];minimum=np.min(local[feasible])
            if minimum<local[current]:
                labels[a]=int(options[np.flatnonzero(feasible&(local==minimum))[0]]);changed+=1
        value=energy()
        if value>energies[-1]+1e-9*max(1,abs(energies[-1])):raise ValueError('spatial energy increased')
        energies.append(value);changes.append(changed)
        if not changed:break
    return {'labels':labels,'energy':energies,'sweep_changes':changes,'optimization':'deterministic sequential coordinate descent; no global optimality claim'}


def validate_source_owners(masks,kept,owners,chunk=65536):
    owners=np.asarray(owners);kept=np.asarray(kept)
    if owners.shape!=(masks.shape[1],) or not np.issubdtype(owners.dtype,np.integer) or not np.isin(owners,np.r_[0,kept+1]).all():
        raise ValueError('owner identities outside frozen retained candidates')
    for start in range(0,len(owners),chunk):
        end=min(len(owners),start+chunk);covered=np.any(masks[kept,start:end],axis=0);part=owners[start:end]
        if not np.array_equal(covered,part>0):raise ValueError('owner union coverage differs')
        idx=np.flatnonzero(part)
        if not np.all(masks[part[idx]-1,start+idx]):raise ValueError('owner expands original mask')
    return {'coverage_exact':True,'inside_original_masks':True,'positive_canonical_ids':True}


def validate_assignment(family,masks,kept,owners,nearest,matched,priorities=None,groups=None):
    if family not in ('global_priority','local_evidence'):raise ValueError('unknown assignment family')
    checks=validate_source_owners(masks,kept,owners);projected=np.where(matched,owners[nearest],0)
    if family=='global_priority':
        if priorities is None or groups is None:raise ValueError('global priorities required')
        legacy=resolve_native_owners(masks[:,nearest]&matched,kept,priorities,source_priority=groups)
        if not np.array_equal(legacy,projected):raise ValueError('global projection parity failed')
        checks['global_projection_parity']='EXACT'
    else:checks['local_validation']='source membership/coverage/IDs and direct fixed projection; no global-order comparison'
    return {**checks,'projected_owners':projected}
