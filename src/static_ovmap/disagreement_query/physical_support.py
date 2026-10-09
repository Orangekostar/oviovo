"""Exact geometric sites, bounded area quadrature and full-mesh visibility."""
from pathlib import Path
import time

import numpy as np

from .common import _array_digest,arrays_record,canonical_digest,producer,unchanged_receipt,write


def canonical_sites(xyz,faces,owners):
    x=np.asarray(xyz,np.float64).copy();f=np.asarray(faces);owners=np.asarray(owners)
    if x.ndim!=2 or x.shape[1]!=3 or owners.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError('finite aligned source mesh required')
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer) or np.any(f<0) or np.any(f>=len(x)):
        raise ValueError('valid indexed triangles required')
    x[x==0]=0.
    coordinates,inverse=np.unique(x,axis=0,return_inverse=True)
    ids=inverse[f];order=np.argsort(ids,axis=1,kind='stable')
    triangles=np.take_along_axis(ids,order,axis=1)
    assignments=np.take_along_axis(owners[f],order,axis=1)
    unique,first,group,counts=np.unique(triangles,axis=0,return_index=True,return_inverse=True,return_counts=True)
    points=coordinates[unique]
    areas=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)/2
    disagreements=np.any(assignments!=assignments[first][group],axis=1)
    ambiguous=np.bincount(group,weights=disagreements,minlength=len(unique))>0
    positive=areas>0;keep=positive&~ambiguous
    co=unique[keep].ravel();ow=assignments[first][keep].ravel();mass=np.repeat(areas[keep]/3,3)
    good=ow>0
    pairs,sitegroup=np.unique(np.column_stack((ow[good],co[good])),axis=0,return_inverse=True)
    weights=np.bincount(sitegroup,weights=mass[good],minlength=len(pairs))
    audit={'duplicates':int(np.sum(counts[positive]-1)),'duplicate_area':float(np.sum((counts[positive]-1)*areas[positive])),
        'ambiguous':int(np.sum(ambiguous&positive)),'ambiguous_area':float(areas[ambiguous&positive].sum()),
        'degenerates':int(counts[~positive].sum()),'degenerate_area':0.,
        'raw_triangle_count':len(f),'unique_positive_triangles':int(positive.sum()),
        'unique_positive_triangle_area':float(areas[positive].sum()),
        'retained_triangle_area':float(areas[keep].sum()),'positive_owner_site_area':float(weights.sum()),
        'sites':len(pairs),'exact_coordinates':len(coordinates),'negative_zero_normalized':True,'epsilon_weld_m':0.}
    return {'xyz':coordinates[pairs[:,1]],'owners':pairs[:,0].astype(np.int64),'area':weights,
        'coordinate_ids':pairs[:,1].astype(np.int64),'row_to_coordinate':inverse,
        'coordinates':coordinates,'audit':audit}


def quadrature(xyz,area,maximum=4096):
    x=np.asarray(xyz,np.float64);a=np.asarray(area,np.float64)
    if x.shape!=(len(a),3) or maximum<1 or not np.isfinite(x).all() or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('finite positive-area sites required')
    order=np.lexsort((x[:,2],x[:,1],x[:,0]));x=x[order];a=a[order]
    if len(a)<=maximum:return x.copy(),a.copy()
    total=float(a.sum());marks=(np.arange(maximum)+.5)*(total/maximum)
    picks=np.minimum(np.searchsorted(np.cumsum(a),marks,side='right'),len(a)-1)
    ids,counts=np.unique(picks,return_counts=True)
    return x[ids].copy(),counts.astype(np.float64)*(total/maximum)


def sites_for_owner(sites,owner):
    selected=sites['owners']==owner;x=sites['xyz'][selected];a=sites['area'][selected]
    sample,weights=quadrature(x,a)
    ids=[canonical_digest({'owner':int(owner),'xyz':p.tolist()}) for p in sample]
    return {'xyz':sample,'area':weights,'site_ids':np.asarray(ids,dtype='S64'),
        'audit':{'unsampled_sites':len(a),'sampled_sites':len(weights),'area':float(a.sum()),
                 'sampled_area':float(weights.sum()),'maximum':4096,'convention':'LEXICOGRAPHIC_MIDPOINT_AREA_QUANTILES_RIGHT'}}


def prepare_sites(binding,predictor):
    root=Path(binding['output_root'])/'physical_support'/predictor.scene;index=predictor.index
    key=canonical_digest({'geometry':predictor.geometry.to_dict(),'owners':_array_digest(predictor.owner_ids),
                          'producer':producer(index,'physical_support.py')})
    old=unchanged_receipt(root/'receipt.json',key,index)
    if old:
        with np.load(old['arrays']['path'],allow_pickle=False) as arr:result={k:arr[k] for k in arr.files}
        return {**result,'audit':old['audit']},old
    begin=time.perf_counter();result=canonical_sites(predictor.xyz,predictor.faces,predictor.owner_ids)
    record=arrays_record(root/'sites.npz',{k:v for k,v in result.items() if k!='audit'},index)
    row=write(root/'receipt.json',{'status':'COMPLETE','scene':predictor.scene,'input_identity':key,
        'arrays':record,'dependencies':[record],'audit':result['audit'],'elapsed_seconds':time.perf_counter()-begin,
        'owner_partition_digest':_array_digest(predictor.owner_ids),'representation':'SUPPORT_ONLY_DOES_NOT_REPAINT_FULL'})
    return result,row


def direct_visibility(points,frame,depth,full,scene):
    import open3d as o3d
    x=np.asarray(points,np.float64);depth=np.asarray(depth,np.float64);full=np.asarray(full)
    pose=np.asarray(frame['pose_c2w'],np.float64);k=np.asarray(frame['intrinsics'],np.float64)
    h,w=map(int,frame['image_size_hw'])
    if x.ndim!=2 or x.shape[1]!=3 or depth.shape!=(h,w) or full.shape!=(h,w) or full.dtype!=bool or pose.shape!=(4,4) or k.shape!=(3,3):
        raise ValueError('aligned original camera/depth/FULL required')
    if not np.isfinite(x).all() or not np.isfinite(pose).all() or not np.isfinite(k).all():raise ValueError('finite geometry and camera required')
    inv=np.linalg.inv(pose);y=x@inv[:3,:3].T+inv[:3,3];z=y[:,2]
    q=y@k.T;uv_float=np.full((len(x),2),np.nan)
    positive=np.isfinite(z)&(z>1e-6)
    uv_float[positive]=q[positive,:2]/q[positive,2,None]
    floating=positive&np.isfinite(uv_float).all(axis=1)&(uv_float[:,0]>=0)&(uv_float[:,0]<w)&(uv_float[:,1]>=0)&(uv_float[:,1]<h)
    uv=np.full((len(x),2),-1,np.int64);uv[positive]=np.floor(uv_float[positive]+.5).astype(np.int64)
    inside=floating&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    v=np.zeros(len(x),bool);o=v.copy();eligible=np.flatnonzero(inside);queried=0
    for start in range(0,len(eligible),65536):
        ids=eligible[start:start+65536]
        rays=np.empty((len(ids),6),np.float64);rays[:,:3]=pose[:3,3]
        directions=y[ids]/z[ids,None];rays[:,3:]=directions@pose[:3,:3].T
        hit=scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)),nthreads=4)['t_hit'].numpy().astype(np.float64)
        measured=depth[uv[ids,1],uv[ids,0]];tolerance=np.maximum(.02,.02*z[ids])
        good=np.isfinite(measured)&(measured>1e-6)&np.isfinite(hit)&(hit>1e-6)&(np.abs(measured-z[ids])<=tolerance)&(np.abs(hit-z[ids])<=tolerance)
        v[ids]=good;o[ids]=good&full[uv[ids,1],uv[ids,0]];queried+=len(ids)
    return v,o,uv,{'projected_in_bounds':int(inside.sum()),'geometric_visible':int(v.sum()),
        'mask_supported':int(o.sum()),'visible_outside_full':int((v&~o).sum()),'queried_rays':queried,
        'metadata_dtype':'float64','ray_api_dtype':'float32','rounding':'floor(u+0.5)','clamped_pixels':0}
