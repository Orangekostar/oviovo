"""Frozen GT-free target inventory, legacy FULL masks and qualified view bank."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.special import softmax

from static_ovmap.minimal_instance_repair.observations import SourceRowProjector
from static_ovmap.samv_local_probe.query_plan import observations as legacy_observations
from static_ovmap.samv_local_probe.geometry import pose_distance

from .binding import load_scene
from .common import _array_digest,arrays_record,canonical_digest,plain,producer,unchanged_receipt,write
from .physical_support import prepare_sites,sites_for_owner,direct_visibility
from .policies import mask_info,quadrants


def source_scores(predictor,owner):
    key=str(owner);values={}
    for name in ('N','Q','F'):
        row=predictor.sources[name]['objects'][key]
        values[name]=np.asarray(row['scores'],np.float64) if row['available'] else None
    return values


def source_js(scores,temperatures):
    ps=[]
    for name in ('N','Q','F'):
        value=scores.get(name)
        if value is None:continue
        if value.ndim!=1 or not np.isfinite(value).all():raise ValueError('finite source scores required')
        ps.append(softmax(value/float(temperatures[name])))
    if len(ps)<=1:return 0.
    def entropy(p):
        nonzero=p>0;return float(-np.sum(p[nonzero]*np.log(p[nonzero])))
    return entropy(np.mean(ps,axis=0))-float(np.mean([entropy(p) for p in ps]))


def target_inventory(predictor,legacy_views):
    rows=[]
    for owner in np.unique(predictor.d2_owner_ids):
        owner=int(owner)
        if owner<=0:continue
        support=predictor.owner_ids==owner;old=int(predictor.semantic_labels[support][0])
        prior=predictor.probabilities[str(owner)]['probabilities'];scores=source_scores(predictor,owner)
        views=legacy_views.get(owner,{})
        qualified=[int(f) for f,v in views.items() if v['qualified']]
        reason='ELIGIBLE'
        if np.any(predictor.raw[support]==0):reason='EXCLUDE_RAW_ZERO_SUPPORT'
        elif old not in predictor.valid_ids:reason='EXCLUDE_INVALID_FROZEN_CLASS'
        elif prior is None:reason='EXCLUDE_UNAVAILABLE_P0'
        elif scores['F'] is None:reason='EXCLUDE_UNAVAILABLE_HISTORICAL_F'
        elif len(qualified)<2:reason='EXCLUDE_NO_TWO_LEGACY_FULL_VIEWS'
        margin=None
        if prior is not None:
            p=np.asarray(prior,np.float64);ordered=np.sort(p);margin=float(ordered[-1]-ordered[-2])
        rows.append({'owner':owner,'old_class':old,'eligible':reason=='ELIGIBLE','reason':reason,
            'margin':margin,'source_JS':source_js(scores,predictor.temperatures),
            'available_sources':[n for n,s in scores.items() if s is not None],
            'qualified_legacy_frame_ids':sorted(qualified),'support_rows':int(support.sum()),
            'raw_zero_support_rows':int(np.sum(predictor.raw[support]==0))})
    return rows


def select_targets(inventory,maximum=16):
    return sorted((r for r in inventory if r['eligible']),key=lambda r:(r['margin'],-r['source_JS'],r['owner']))[:maximum]


def candidate_bank(frames,qualified,limit=8):
    if len(qualified)<2:raise ValueError('two physically qualified views required')
    table={int(f['frame_id']):f for f in frames}
    if any(fid not in table for fid in qualified) or limit<2:raise ValueError('qualified frame not in original bank')
    anchor=min(qualified,key=lambda f:(-qualified[f]['pixels'],f));chosen=[anchor]
    while len(chosen)<min(limit,len(qualified)):
        remaining=[f for f in qualified if f not in chosen]
        chosen.append(min(remaining,key=lambda f:(-min(pose_distance(table[f]['pose_c2w'],table[c]['pose_c2w']) for c in chosen),f)))
    return anchor,sorted(chosen)


def full_owners(predictor,observation):
    with np.load(observation['arrays']['path'],allow_pickle=False) as arr:
        source,valid=arr['source_rows'],arr['valid']
    raster=np.zeros(source.shape,np.int64);raster[valid]=predictor.owner_ids[source[valid]]
    return raster,source,valid


def plan_scene(binding,scene):
    p,evaluation=load_scene(binding,scene);index=p.index;root=Path(binding['output_root']);dest=root/'plans'/scene
    key=canonical_digest({'binding':binding['identity'],'scene':scene,'geometry':p.geometry.to_dict(),
        'owners':_array_digest(p.owner_ids),'producer':producer(index,'query_plan.py','physical_support.py','policies.py','binding.py')})
    old=unchanged_receipt(dest/'manifest.json',key,index)
    if old:return old
    begin=time.perf_counter()
    adapter=SimpleNamespace(scene=scene,index=index,g1=SimpleNamespace(geometry=p.geometry),capture=p.capture,
        capture_path=p.capture_path,xyz=p.xyz,faces=p.faces)
    frames,observed=legacy_observations(binding,adapter)
    table={int(r['frame_id']):r for r in observed['frames']};incumbents=set(map(int,np.unique(p.d2_owner_ids[p.d2_owner_ids>0])))
    views={o:{} for o in incumbents}
    for frame in frames:
        fid=int(frame['frame_id']);raster,_,_=full_owners(p,table[fid])
        owners,counts=np.unique(raster,return_counts=True)
        for o,n in zip(owners,counts):
            owner=int(o)
            if owner in incumbents and n>=100:views[owner][fid]=mask_info(raster==owner)
    inventory=target_inventory(p,views);selected=select_targets(inventory)
    sites,site_receipt=prepare_sites(binding,p)
    samples={r['owner']:sites_for_owner(sites,r['owner']) for r in selected}
    support={o:{'V':[],'O':[],'uv':[],'frames':[],'views':{},'legacy_row_counts':None,
        'legacy_coordinate_counts':None} for o in samples}
    raw_rows={o:np.flatnonzero(p.owner_ids==o) for o in samples}
    coords={o:np.unique(sites['row_to_coordinate'][raw_rows[o]]) for o in samples}
    row_counts={o:np.zeros(len(raw_rows[o]),np.int16) for o in samples}
    coordinate_counts={o:np.zeros(len(coords[o]),np.int16) for o in samples}
    projector=SourceRowProjector(p.xyz,p.faces) if samples else None;dependencies=[];mask_regions={};ray_counts=0
    for frame in frames:
        fid=int(frame['frame_id']);observation=table[fid]
        raster,source,valid=full_owners(p,observation)
        dependencies.extend([observation['arrays'],observation['rgb'],observation['depth']])
        with np.load(observation['depth']['path'],allow_pickle=False) as arr:depth=arr['depth_m']
        for o,sample in samples.items():
            mask=raster==o;info=mask_info(mask);s=support[o]
            seen=np.unique(source[valid&mask]);row_counts[o][np.searchsorted(raw_rows[o],seen)]+=1
            seen_coords=np.unique(sites['row_to_coordinate'][seen]);coordinate_counts[o][np.searchsorted(coords[o],seen_coords)]+=1
            v,footprint,uv,audit=direct_visibility(sample['xyz'],frame,depth,mask,projector.mesh.scene)
            s['V'].append(v);s['O'].append(footprint);s['uv'].append(uv);s['frames'].append(fid);ray_counts+=audit['queried_rays']
            info={**info,'positive_O_sites':int(footprint.sum()),'V_sites':int(v.sum()),
                'O_area':float(sample['area'][footprint].sum()),'V_area':float(sample['area'][v].sum()),
                'physical_qualified':bool(info['qualified'] and footprint.sum()>=8),'visibility':audit}
            s['views'][fid]=info
            if info['qualified']:
                mask_record=arrays_record(dest/'masks'/str(o)/(str(fid)+'_FULL.npz'),{'mask':mask},index)
                rid=f'{o}:{fid}:FULL'
                region={'region_id':rid,'owner':o,'frame_id':fid,'role':'FULL','tile':None,
                    'mask':mask_record,'mask_digest':_array_digest(mask),'RGB':observation['rgb'],
                    'depth':observation['depth'],'pose_c2w':frame['pose_c2w'],'intrinsics':frame['intrinsics'],
                    'original_size':frame['image_size_hw'],'pixels':info['pixels'],'bbox':info['bbox'],'qualified':True}
                mask_regions[rid]=region;dependencies.append(mask_record)
    owners={}
    for selected_row in selected:
        o=selected_row['owner'];sample=samples[o];s=support[o]
        record=arrays_record(dest/'supports'/(str(o)+'.npz'),{'xyz':sample['xyz'],'area':sample['area'],
            'site_ids':sample['site_ids'],'frame_ids':np.array(s['frames'],np.int64),
            'V':np.asarray(s['V'],bool),'O':np.asarray(s['O'],bool),'uv':np.asarray(s['uv'],np.int64),
            'legacy_source_rows':raw_rows[o],'legacy_row_counts':row_counts[o],
            'legacy_coordinates':sites['coordinates'][coords[o]],'legacy_coordinate_counts':coordinate_counts[o]},index)
        dependencies.append(record);qualified={f:v for f,v in s['views'].items() if v['physical_qualified']}
        rows=raw_rows[o];points=p.xyz[rows].astype(np.float64)
        center=(points.min(axis=0)+points.max(axis=0))/2
        old=selected_row['old_class'];prior=np.asarray(p.probabilities[str(o)]['probabilities'],np.float64)
        candidates=[i for i,c in enumerate(p.valid_ids) if c!=old];other=max(candidates,key=lambda i:prior[i])
        owner={'owner':o,'old_class':old,'other_class':p.valid_ids[other],'p0':prior.tolist(),
            'support':record,'support_hash':canonical_digest({'geometry':p.geometry.to_dict(),'owner':o,'rows':_array_digest(rows)}),
            'sampling':sample['audit'],'center':center.tolist(),'views':{str(f):v for f,v in s['views'].items()},
            'query_eligible':len(qualified)>=2,'reason':'QUERY_ELIGIBLE' if len(qualified)>=2 else 'KEEP_NO_TWO_PHYSICAL_VIEWS',
            'anchor':None,'candidate_frame_ids':[],'probe_region_ids':[]}
        if owner['query_eligible']:
            anchor,bank=candidate_bank(frames,qualified);owner['anchor']=anchor;owner['candidate_frame_ids']=bank
            region=mask_regions[f'{o}:{anchor}:FULL']
            with np.load(region['mask']['path'],allow_pickle=False) as arr:mask=arr['mask']
            tiles,bbox=quadrants(mask);owner['anchor_bbox']=bbox;owner['probe_inventory']=[]
            for tile in tiles:
                owner['probe_inventory'].append({k:v for k,v in tile.items() if k!='mask'})
                if not tile['qualified']:continue
                rid=f"{o}:{anchor}:TILE{tile['tile']}"
                rec=arrays_record(dest/'masks'/str(o)/(str(anchor)+'_TILE'+str(tile['tile'])+'.npz'),{'mask':tile['mask']},index)
                dependencies.append(rec)
                mask_regions[rid]={**region,'region_id':rid,'role':'PROBE','tile':tile['tile'],
                    'mask':rec,'mask_digest':_array_digest(tile['mask']),'pixels':tile['pixels'],'bbox':tile['bbox']}
                owner['probe_region_ids'].append(rid)
        owners[str(o)]=owner
    manifest=write(dest/'manifest.json',plain({'status':'PLAN_LOCKED','scene':scene,'input_identity':key,
        'context':binding['scenes'][scene]['context'],'D2_probability_parity':evaluation.d2_probability_parity,
        'geometry':p.geometry.to_dict(),'owner_partition_digest':_array_digest(p.owner_ids),
        'frames':frames,'observation_identity':observed['identity'],'site_receipt':site_receipt,
        'inventory':inventory,'selected_owner_ids':[r['owner'] for r in selected],
        'query_eligible_owner_ids':[int(o) for o,r in owners.items() if r['query_eligible']],
        'owners':owners,'regions':mask_regions,'dependencies':list({d['path']:d for d in dependencies}.values()),
        'GT_used':False,'legacy_panoptic_pixels_read':False,'direct_site_rays':ray_counts,
        'elapsed_seconds':time.perf_counter()-begin,'selection_before_physical_qualification':True}))
    index.write_memo(root/'inputs'/scene/'verifications.json')
    print('PLAN',scene,len(selected),'selected',len(manifest['query_eligible_owner_ids']),'query eligible',flush=True)
    return manifest
