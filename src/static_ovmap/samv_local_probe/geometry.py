"""GT-free automatic points, fixed pose windows and local source-row permissions."""

import numpy as np
from scipy.ndimage import distance_transform_edt,label


def interior_points(mask,maximum=3):
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or maximum<1:raise ValueError('2D mask and positive point count required')
    components,count=label(mask,np.ones((3,3),int))
    if not count:return []
    sizes=np.bincount(components.ravel());sizes[0]=0
    candidates=np.flatnonzero(sizes==sizes.max())
    chosen=min(candidates,key=lambda i:int(np.flatnonzero(components==i)[0]))
    distance=distance_transform_edt(np.pad(components==chosen,1))[1:-1,1:-1]
    yx=np.argwhere(distance>=2.)
    if not len(yx):return []
    values=distance[yx[:,0],yx[:,1]];selected=[int(np.flatnonzero(values==values.max())[0])]
    while len(selected)<min(maximum,len(yx)):
        distances=((yx[:,None,:]-yx[np.asarray(selected)][None,:,:])**2).sum(-1).min(1)
        distances[selected]=-1;selected.append(int(np.flatnonzero(distances==distances.max())[0]))
    return [(int(yx[i,1]),int(yx[i,0])) for i in selected]


def transform_points(points,original_hw):
    points=np.asarray(points,np.float64);h,w=original_hw
    if points.ndim!=2 or points.shape[1]!=2 or h<=0 or w<=0 or not np.isfinite(points).all():
        raise ValueError('finite xy points and original H/W required')
    if np.any(points<0) or np.any(points[:,0]>=w) or np.any(points[:,1]>=h):raise ValueError('point outside original image')
    return points*np.array([1024./w,1024./h])


def pose_distance(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    if a.shape!=(4,4) or b.shape!=(4,4) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite SE3 matrices required')
    angle=np.rad2deg(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
    return float(np.linalg.norm(a[:3,3]-b[:3,3])/.20+angle/15.)


def choose_window(frames,qualified,anchor,count=6):
    table={int(f['frame_id']):f for f in frames};qualified=list(map(int,qualified));anchor=int(anchor)
    if len(table)!=len(frames) or len(table)<count or count not in (4,6):raise ValueError('distinct full resource-profile frame window unavailable')
    alternatives=[f for f in qualified if f!=anchor]
    if anchor not in table or any(f not in table for f in qualified) or not alternatives:raise ValueError('two mandatory qualified old-mask views required')
    distance=lambda a,b:pose_distance(table[a]['pose_c2w'],table[b]['pose_c2w'])
    second=min(alternatives,key=lambda f:(-distance(anchor,f),f));chosen=[anchor,second]
    while len(chosen)<count:
        chosen.append(min((f for f in table if f not in chosen),key=lambda f:(-min(distance(f,c) for c in chosen),f)))
    chosen.sort();return chosen,chosen.index(anchor),chosen.index(second)


def select_targets(inventory,quota=4):
    incumbents=sorted((r for r in inventory if r['eligible'] and r['pool']=='INCUMBENT'),key=lambda r:(r['margin'],r['owner']))
    recovered=sorted((r for r in inventory if r['eligible'] and r['pool']=='RECOVERED'),key=lambda r:(-r['maximum_area'],r['owner']))
    chosen=incumbents[:quota]+recovered[:quota]
    if len(incumbents)<quota:chosen+=recovered[quota:quota+quota-len(incumbents)]
    if len(recovered)<quota:chosen+=incumbents[quota:quota+quota-len(recovered)]
    return sorted(chosen,key=lambda r:r['owner'])


def split_panorama(logits,frames):
    values=np.asarray(logits)
    if values.ndim!=2 or frames<1 or values.shape[1]%frames:raise ValueError('panorama requires exact per-view tiles')
    h,width=values.shape;return values.reshape(h,frames,width//frames).transpose(1,0,2).copy()


def boundary_band(owners,faces,hops=2):
    owners=np.asarray(owners);faces=np.asarray(faces)
    if owners.ndim!=1 or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or hops<0:
        raise ValueError('original nonempty triangle mesh required')
    if not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0) or np.any(faces>=len(owners)):raise ValueError('mesh source index outside surface')
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]])
    same=owners[edges[:,0]]==owners[edges[:,1]];band=np.zeros(len(owners),bool)
    band[edges[~same].ravel()]=True
    for _ in range(hops):
        old=band.copy();band[edges[same&old[edges[:,0]],1]]=True;band[edges[same&old[edges[:,1]],0]]=True
    return band


def edit_domain(xyz,faces,owners,targets,padding=.30,hops=2):
    xyz=np.asarray(xyz);owners=np.asarray(owners);targets=list(map(int,targets))
    if xyz.shape!=(len(owners),3) or not np.isfinite(xyz).all() or len(set(targets))!=len(targets) or any(i<=0 for i in targets):
        raise ValueError('fixed surface and unique positive selected owners required')
    band=boundary_band(owners,faces,hops);domains={};union=np.zeros(len(owners),bool)
    for owner in targets:
        support=xyz[owners==owner]
        if not len(support):raise ValueError('selected target absent from actual G1')
        inside=np.all((xyz>=support.min(0)-padding)&(xyz<=support.max(0)+padding),axis=1)
        domains[owner]=inside;union|=inside
    selected=np.isin(owners,targets);protected=(owners>0)&~selected&~band
    return domains,union&(selected|(owners==0)|band),protected


def qualified_mask(mask,minimum=100):
    yx=np.argwhere(mask)
    if len(yx)<minimum:return False,None,int(len(yx))
    lo,hi=yx.min(0),yx.max(0);bbox=[int(lo[1]),int(lo[0]),int(hi[1]+1),int(hi[0]+1)]
    return bbox[2]-bbox[0]>=2 and bbox[3]-bbox[1]>=2,bbox,int(len(yx))
