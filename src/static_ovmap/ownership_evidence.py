"""Shared prediction-side RGB-D instance evidence; no GT or semantic class input."""
import numpy as np
from scipy.sparse import csr_matrix

REASONS={0:'usable',1:'atom_unobserved',2:'empty_candidate_remainder',3:'insufficient_intersection',4:'low_iou',5:'ambiguous_correspondence'}


def select_observation_frames(frame_ids,poses,max_frames=32,translation_m=.05,rotation_degrees=5.):
    selected=[];invalid=[];redundant=[]
    for frame in sorted(frame_ids):
        if frame>=len(poses) or not np.isfinite(poses[frame]).all():invalid.append(int(frame));continue
        pose=poses[frame]
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4):
            invalid.append(int(frame));continue
        distinct=True
        for previous in selected:
            other=poses[previous];distance=np.linalg.norm(pose[:3,3]-other[:3,3])
            angle=np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@other[:3,:3])-1)/2,-1,1)))
            if distance<translation_m and angle<rotation_degrees:distinct=False;break
        (selected if distinct else redundant).append(int(frame))
    indices=np.unique(np.rint(np.linspace(0,len(selected)-1,min(max_frames,len(selected)))).astype(int)) if selected else []
    chosen=[selected[i] for i in indices]
    return chosen,{'original_frame_ids':list(map(int,frame_ids)),'invalid_pose':invalid,'pose_redundant':redundant,
        'pose_distinct_ids':selected,'budget_excluded':[i for i in selected if i not in chosen],'selected_frame_ids':chosen,
        'temporal_sampling':'np.rint linspace indices, ties-to-even, deduplicated','pose_rule':'different from every earlier representative by >=translation OR >=rotation'}


def project_shared_visibility(xyz,pose,camera,depth,masks,tolerance=.05,chunk=65536):
    depth,masks=np.asarray(depth),np.asarray(masks)
    h,w=int(camera['h']),int(camera['w'])
    if depth.shape!=(h,w) or masks.shape!=(h,w) or not np.issubdtype(masks.dtype,np.integer) or np.any(masks<0):
        raise ValueError('registered depth and original nonnegative frame-instance image required; no resize')
    zbuffer=np.full(h*w,np.inf);winners=np.full(h*w,-1,dtype=np.int64)
    for start in range(0,len(xyz),chunk):
        camera_xyz=(np.asarray(xyz[start:start+chunk],dtype=np.float64)-pose[:3,3])@pose[:3,:3]
        z=camera_xyz[:,2];valid=np.isfinite(camera_xyz).all(axis=1)&(z>0)
        ids=np.flatnonzero(valid)
        u=np.rint(camera['fx']*camera_xyz[ids,0]/z[ids]+camera['cx']).astype(np.int64)
        v=np.rint(camera['fy']*camera_xyz[ids,1]/z[ids]+camera['cy']).astype(np.int64)
        keep=(u>=0)&(v>=0)&(u<w)&(v<h);ids=ids[keep];pixels=v[keep]*w+u[keep]
        order=np.lexsort((ids,z[ids],pixels));pixels=pixels[order];ids=ids[order]
        first=np.r_[True,pixels[1:]!=pixels[:-1]] if len(pixels) else np.array([],bool)
        pixels=pixels[first];ids=ids[first];better=z[ids]<zbuffer[pixels]
        # Chunks ascend source row; equal-depth winners retain their smaller original row.
        pixels=pixels[better];ids=ids[better];zbuffer[pixels]=z[ids];winners[pixels]=ids+start
    seen=winners>=0;measured=depth.ravel();labels=masks.ravel()
    valid=seen&np.isfinite(measured)&(measured>0)&(np.abs(zbuffer-measured)<tolerance)
    usable=valid&(labels>0);pixels=np.flatnonzero(usable)
    return {'point_ids':winners[usable],'mask_ids':labels[usable].astype(np.int64),'pixel_ids':pixels,
        'zbuffer_points':winners,'depth_consistent_pixels':int(valid.sum()),'rendered_pixels':int(seen.sum()),
        'positive_instance_pixels':int(usable.sum()),'unknown_consistent_pixels':int((valid&(labels==0)).sum())}


def atom_histogram(point_to_atom,point_ids,mask_ids,atom_count):
    labels,inverse=np.unique(mask_ids,return_inverse=True)
    hist=csr_matrix((np.ones(len(point_ids),np.int64),(point_to_atom[point_ids],inverse)),shape=(atom_count,len(labels)))
    hist.sum_duplicates();return hist,labels


def frame_local_evidence(atoms,hist,mask_ids,candidate_count,chunk=65536,min_intersection=16,min_iou=.20,min_gap=.05):
    atom_ids=np.repeat(np.arange(len(atoms['counts'])),np.diff(atoms['indptr']))
    candidates=atoms['candidates'];n=len(candidates)
    incidence=csr_matrix((np.ones(n,np.int64),(atom_ids,candidates)),shape=(len(atoms['counts']),candidate_count))
    table=(incidence.T@hist).toarray().astype(np.int64)
    candidate_area=table.sum(axis=1,dtype=np.int64);mask_area=np.asarray(hist.sum(axis=0)).ravel().astype(np.int64)
    area=np.asarray(hist.sum(axis=1)).ravel().astype(np.int64)
    r=np.zeros(n);p=np.zeros(n);matched=np.full(n,-1,np.int64);reason=np.ones(n,np.uint8)
    for start in range(0,n,chunk):
        end=min(n,start+chunk);aa=atom_ids[start:end];cc=candidates[start:end];h=hist[aa].toarray()
        remainder=candidate_area[cc]-area[aa]
        if not len(mask_ids):continue
        inter=table[cc]-h;other=mask_area[None,:]-h
        union=remainder[:,None]+other-inter
        iou=np.divide(inter,union,out=np.zeros(inter.shape,float),where=union>0)
        best=iou.argmax(axis=1);rr=iou[np.arange(len(aa)),best]
        second=np.partition(iou,-2,axis=1)[:,-2] if len(mask_ids)>1 else np.zeros(len(aa))
        why=np.zeros(len(aa),np.uint8)
        why[(rr-second)<min_gap]=5;why[rr<min_iou]=4
        why[inter[np.arange(len(aa)),best]<min_intersection]=3
        why[remainder<=0]=2;why[area[aa]==0]=1
        good=why==0;reason[start:end]=why
        r[start:end]=np.where(good,rr,0)
        p[start:end]=np.divide(h[np.arange(len(aa)),best],area[aa],out=np.zeros(len(aa)),where=good)
        matched[start:end]=np.where(good,mask_ids[best],-1)
    return {'r':r,'p':p,'usable':reason==0,'matched_mask':matched,'reason':reason,
        'atom_valid_pixel_count':area,'candidate_mask_table':table,'mask_ids':mask_ids}
