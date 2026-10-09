"""Geometry-only controls and anchor-only response heterogeneity."""
import numpy as np


def mask_info(mask,minimum=100):
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=bool:raise ValueError('original boolean FULL mask required')
    ys,xs=np.nonzero(mask);n=len(xs)
    bbox=None if not n else [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
    qualified=bool(bbox and n>=minimum and bbox[2]-bbox[0]>=2 and bbox[3]-bbox[1]>=2)
    return {'pixels':n,'bbox':bbox,'qualified':qualified}


def quadrants(mask):
    info=mask_info(mask);bbox=info['bbox']
    if bbox is None:return [],None
    x0,y0,x1,y1=bbox;xm=(x0+x1)//2;ym=(y0+y1)//2
    regions=[]
    for i,(l,t,r,b) in enumerate(((x0,y0,xm,ym),(xm,y0,x1,ym),(x0,ym,xm,y1),(xm,ym,x1,y1))):
        tile=np.zeros_like(mask);tile[t:b,l:r]=mask[t:b,l:r]
        regions.append({'tile':i,'mask':tile,'bounds':[l,t,r,b],**mask_info(tile,25)})
    return regions,bbox


def anchor_interest(area,anchor_support,uv,bbox,full_cos,tile_cos,c0,c1,temperature):
    area=np.asarray(area,np.float64);anchor=np.asarray(anchor_support,bool);uv=np.asarray(uv)
    interest=np.zeros_like(area);audit={'successful_tiles':len(tile_cos),'fallback':None,'tile_margins':{},'tile_priorities':{},'tile_sign_conflict':False}
    if full_cos is None:
        audit['fallback']='VERIFY_ANCHOR_FULL_UNAVAILABLE';return interest,audit
    if temperature<=0 or not np.isfinite(temperature):raise ValueError('frozen positive F temperature required')
    full=float((full_cos[c1]-full_cos[c0])/temperature);audit['full_margin']=full
    if bbox is None:raise ValueError('eligible anchor bbox required')
    x0,y0,x1,y1=bbox;xm=(x0+x1)//2;ym=(y0+y1)//2
    labels=(uv[:,0]>=xm).astype(int)+2*(uv[:,1]>=ym).astype(int)
    in_box=(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
    for tile,cos in sorted(tile_cos.items()):
        d=float((cos[c1]-cos[c0])/temperature);priority=abs(d-full)
        audit['tile_margins'][str(tile)]=d;audit['tile_priorities'][str(tile)]=priority
        selected=anchor&in_box&(labels==tile);interest[selected]=area[selected]*priority
    margins=list(audit['tile_margins'].values());audit['tile_sign_conflict']=bool(any(d<0 for d in margins) and any(d>0 for d in margins))
    if len(tile_cos)<2:audit['fallback']='VERIFY_FEWER_THAN_TWO_SUCCESSFUL_TILES';interest[:]=0
    elif interest.sum()<=1e-12:audit['fallback']='VERIFY_ZERO_INTEREST_MASS';interest[:]=0
    audit['interest_mass']=float(interest.sum());return interest,audit


def score_bank(policy,*,xyz,area,footprints,center,cameras,pixels,frame_ids,interest=None):
    x=np.asarray(xyz,np.float64);a=np.asarray(area,np.float64);o=np.asarray(footprints,bool)
    center=np.asarray(center,np.float64);cameras=np.asarray(cameras,np.float64);pixels=np.asarray(pixels,np.float64);ids=list(map(int,frame_ids))
    n=len(ids)
    if n<2 or len(set(ids))!=n or o.shape!=(n,len(a)) or x.shape!=(len(a),3) or cameras.shape!=(n,3) or pixels.shape!=(n,):
        raise ValueError('common anchor-first qualified bank required')
    if policy not in ('AREA','COVERAGE','VERIFY','DISAGREEMENT'):raise ValueError('unknown query policy')
    area_anchor=float(a[o[0]].sum());joint=(o&o[0]).astype(np.float64)@a/max(area_anchor,1e-300)
    rays=cameras-center;norm=np.linalg.norm(rays,axis=1);angles=np.zeros(n);degenerate=norm<=1e-12
    if not degenerate[0]:
        good=~degenerate;angles[good]=np.rad2deg(np.arccos(np.clip((rays[good]@rays[0])/(norm[good]*norm[0]),-1,1)))
    base=np.sqrt(joint*np.clip(angles/30.,0,1));novel=np.zeros(n);bins=None
    if policy=='AREA':values=pixels.copy()
    elif policy=='COVERAGE':
        directions=x-center;length=np.linalg.norm(directions,axis=1);good=length>1e-12
        codes=np.full(len(x),-1,np.int64)
        unit=directions[good]/length[good,None]
        theta=np.minimum((np.arccos(np.clip(unit[:,2],-1,1))/np.pi*180).astype(int),179)
        phi=(np.mod(np.arctan2(unit[:,1],unit[:,0]),2*np.pi)/(2*np.pi)*240).astype(int)
        codes[good]=theta*240+np.minimum(phi,239)
        bins=[set(codes[o[v]&good].tolist()) for v in range(n)]
        novel=np.array([len(b-bins[0])/len(b) if b else 0. for b in bins]);values=novel.copy()
    else:values=base.copy()
    H=np.zeros(n)
    if policy=='DISAGREEMENT':
        I=np.zeros_like(a) if interest is None else np.asarray(interest,np.float64)
        if I.shape!=a.shape or np.any(I<0) or not np.isfinite(I).all():raise ValueError('finite anchor-derived interest required')
        if I.sum()>1e-12:H=o.astype(float)@I/float(I.sum());values=base*(.5+.5*H)
    second=min(range(1,n),key=lambda j:(-values[j],-pixels[j],ids[j]))
    candidates={str(ids[j]):{'score':float(values[j]),'pixels':int(pixels[j]),'J':float(joint[j]),
        'theta_degrees':float(angles[j]),'H':float(H[j]),'coverage_novelty':float(novel[j]),
        'degenerate_camera_center':bool(degenerate[j] or degenerate[0]),
        'occupied_bins':None if bins is None else len(bins[j])} for j in range(1,n)}
    return ids[second],{'policy':policy,'anchor':ids[0],'second':ids[second],'candidates':candidates,
        'tie_order':['score_descending','FULL_area_descending','frame_id_ascending']}
