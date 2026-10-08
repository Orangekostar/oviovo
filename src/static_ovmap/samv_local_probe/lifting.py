"""One depth-validated vote per source row/view and simultaneous ownership updates."""

import numpy as np


def count_view_votes(source_maps,valid_maps,masks,row_count):
    if not (len(source_maps)==len(valid_maps)==len(masks)):raise ValueError('aligned view lists required')
    visible=np.zeros(row_count,np.int64);positive=np.zeros(row_count,np.int64)
    for source,valid,mask in zip(source_maps,valid_maps,masks):
        source,valid,mask=np.asarray(source),np.asarray(valid,bool),np.asarray(mask,bool)
        if source.shape!=valid.shape or source.shape!=mask.shape:raise ValueError('model/observer original image shapes differ')
        ids=source[valid]
        if np.any(ids<0) or np.any(ids>=row_count):raise ValueError('invalid visible source row')
        count=np.bincount(ids,minlength=row_count);hits=np.bincount(ids[mask[valid]],minlength=row_count)
        observed=count>0;visible+=observed;positive+=observed&(2*hits>=count)
    return visible,positive


def arbitrate(base,target_ids,counts,domains,editable,prompts=None):
    base=np.asarray(base,np.int64);editable=np.asarray(editable,bool);ids=sorted(map(int,target_ids))
    if base.ndim!=1 or editable.shape!=base.shape or len(set(ids))!=len(ids) or any(i<=0 for i in ids):raise ValueError('valid original partition and selected owners required')
    winner=np.zeros(len(base),np.int64);bestn=np.zeros(len(base),np.int64);bestk=np.zeros(len(base),np.int64)
    ties=np.zeros(len(base),bool);contenders=np.zeros(len(base),np.uint16)
    for owner in ids:
        n,k=(np.asarray(v,np.int64) for v in counts[owner]);domain=np.asarray(domains[owner],bool)
        if n.shape!=base.shape or k.shape!=base.shape or domain.shape!=base.shape or np.any(k<0) or np.any(n<k):raise ValueError('aligned nonnegative visibility counts required')
        possible=editable&domain&(n>=2)&(3*k>=2*n);contenders+=possible
        better=possible&((winner==0)|(k*bestn>bestk*n));equal=possible&~better&(k*bestn==bestk*n)
        ties[equal]=True;ties[better]=False;winner[better]=owner;bestn[better]=n[better];bestk[better]=k[better]
    out=base.copy();decisive=(winner>0)&~ties;out[decisive]=winner[decisive]
    for owner in ids:
        n,k=(np.asarray(v,np.int64) for v in counts[owner])
        remove=editable&(winner==0)&(base==owner)&domains[owner]&(n>=2)&(3*k<=n);out[remove]=0
    for row,owner in (prompts or {}).items():
        if not 0<=int(row)<len(base) or int(base[int(row)])!=int(owner) or int(owner) not in ids:raise ValueError('prompt must belong to original selected owner')
        out[int(row)]=int(owner)
    moved=out!=base
    return out,{'tie_rows':int(ties.sum()),'conflicting_rows':int((contenders>1).sum()),'moved_rows':int(moved.sum()),
        'added_unowned_rows':int(((base==0)&(out>0)).sum()),'deleted_rows':int(((base>0)&(out==0)).sum()),
        'reassigned_rows':int((moved&(base>0)&(out>0)).sum()),'forced_prompt_rows':len(prompts or {}),
        'out_of_domain_owner_parity':bool(np.array_equal(out[~editable],base[~editable]))}


def lift_scene(binding,scene,model):
    from pathlib import Path
    import time
    from .binding import load_scene
    from .common import arrays_record,canonical_digest,verified,write,_array_digest
    from .query_plan import load_observation
    root=Path(binding['output_root']);inputs=load_scene(binding,scene);plan=verified(root/'query_plan'/scene/'receipt.json')
    index=inputs.index;seg={}
    for query in plan['queries']:
        owner=query['owner'];row=verified(root/'segmentation'/scene/model/str(owner)/'receipt.json')
        if row['status']!='COMPLETE':raise RuntimeError('missing model result cannot become an inactive zero mask')
        index.identity(row['raw_arrays']['path'],row['raw_arrays']);seg[owner]=row
    key=canonical_digest({'plan':plan['identity'],'segmentations':[seg[o]['identity'] for o in plan['selected_owner_ids']],
        'operator':index.identity(__file__)})
    path=root/'lifted'/scene/model/'receipt.json'
    if path.exists():
        old=verified(path)
        if old['input_identity']!=key:raise ValueError('lifting inputs changed; invalidate affected partition descendants')
        index.identity(old['partition']['path'],old['partition'])
        for target in old['targets'].values():index.identity(target['evidence_arrays']['path'],target['evidence_arrays'])
        return old
    begin=time.perf_counter();counts={};domains={};targets={};size=len(inputs.xyz)
    index.identity(plan['domain']['path'],plan['domain'])
    with np.load(plan['domain']['path'],allow_pickle=False) as arrays:
        editable=arrays['editable'];protected=arrays['protected']
        domains={o:arrays[f'owner_{o}'] for o in plan['selected_owner_ids']}
    for query in plan['queries']:
        owner=query['owner'];row=seg[owner];sources=[];valids=[];masks=[]
        with np.load(row['raw_arrays']['path'],allow_pickle=False) as arrays:
            for slot,frame in enumerate(query['frames']):
                source,valid=load_observation(frame,size);sources.append(source);valids.append(valid);masks.append(arrays[f'mask_{slot}'])
        n,k=count_view_votes(sources,valids,masks,size);visible=np.flatnonzero(n).astype(np.int32)
        raw_positive=(n>=2)&(3*k>=2*n);foreground=k>0
        raw_negative=(inputs.g1.owner_ids==owner)&(n>=2)&(3*k<=n)
        admitted=raw_positive&domains[owner]&editable
        active=row['outcome']['anchor_adherent']
        evidence=arrays_record(path.parent/'targets'/f'{owner}.npz',{'source_rows':visible,
            'n':n[visible].astype(np.uint8),'k':k[visible].astype(np.uint8),
            'raw_foreground_rows':np.flatnonzero(foreground).astype(np.int32),
            'raw_positive_proposal_rows':np.flatnonzero(raw_positive).astype(np.int32),
            'raw_negative_proposal_rows':np.flatnonzero(raw_negative).astype(np.int32),
            'admitted_positive_rows':np.flatnonzero(admitted if active else np.zeros(size,bool)).astype(np.int32),
            'suppressed_positive_rows':np.flatnonzero(raw_positive&~(domains[owner]&editable) if active else raw_positive).astype(np.int32)},index)
        targets[str(owner)]={'active':active,'anchor_adherent':active,'segmentation_identity':row['identity'],
            'evidence_arrays':evidence,'visible_source_rows':len(visible),'raw_foreground_rows':int(foreground.sum()),
            'raw_qualified_positive_rows':int(raw_positive.sum()),'raw_qualified_negative_rows':int(raw_negative.sum()),
            'admitted_positive_rows':int(admitted.sum()) if active else 0,
            'suppressed_positive_rows':int((raw_positive&~(domains[owner]&editable)).sum()) if active else int(raw_positive.sum()),
            'suppression_reasons':{'outside_own_AABB':int((raw_positive&~domains[owner]).sum()),
                'protected_unselected_interior':int((raw_positive&domains[owner]&protected).sum()),
                'anchor_failed':int(raw_positive.sum()) if not active else 0},
            'positive_but_n_below_2_rows':int((foreground&(n<2)).sum()),
            'empty_raw_frames':row['outcome']['empty_raw_frames']}
        if not active:n[:]=0;k[:]=0
        counts[owner]=(n,k)
    prompts={int(r):int(o) for r,o in plan['prompt_owners'].items()}
    owners,audit=arbitrate(inputs.g1.owner_ids,plan['selected_owner_ids'],counts,domains,editable,prompts)
    moved=np.flatnonzero(owners!=inputs.g1.owner_ids).astype(np.int32)
    if not np.array_equal(owners[protected],inputs.g1.owner_ids[protected]):raise ValueError('unselected protected interior was changed')
    partition=arrays_record(path.parent/'partition.npz',{'owners':owners,'moved_source_rows':moved,
        'old_owners':inputs.g1.owner_ids[moved],'new_owners':owners[moved]},index)
    for owner,row in targets.items():
        owner=int(owner);row['final_support_rows']=int((owners==owner).sum())
        row['added_rows']=int(((owners==owner)&(inputs.g1.owner_ids!=owner)).sum())
        row['lost_rows']=int(((owners!=owner)&(inputs.g1.owner_ids==owner)).sum())
        row['other_active_query_transfer_from_inactive_old_target']=int(((inputs.g1.owner_ids==owner)&(owners>0)&(owners!=owner)).sum()) if not row['active'] else 0
    donors={}
    for owner in np.unique(inputs.g1.owner_ids[moved]):
        if owner>0:donors[str(int(owner))]={'lost_rows':int(((inputs.g1.owner_ids==owner)&(owners!=owner)).sum()),
            'selected_target':int(owner) in plan['selected_owner_ids'],'remaining_rows':int((owners==owner).sum())}
    result=write(path,{'status':'LIFTED_PARTITION_LOCKED','scene':scene,'model':model,'input_identity':key,
        'partition':partition,'owner_partition_digest':_array_digest(owners),'targets':targets,'audit':audit,
        'donors':donors,'protected_core_preserved':True,'prompt_owners_retained':prompts,
        'old_raw_zero_ownership_edits':int(((owners!=inputs.g1.owner_ids)&(inputs.raw==0)).sum()),
        'elapsed_seconds':time.perf_counter()-begin,'GT_used':False,'new_owner_ids':0})
    index.write_memo(root/'inputs'/scene/'verifications.json');return result


def lift(binding):
    from pathlib import Path
    from .common import write
    rows={s:{m:lift_scene(binding,s,m)['identity'] for m in ('SAM2','SAMV')} for s in binding['scenes']}
    return write(Path(binding['output_root'])/'lifted/summary.json',{'status':'LIFTING_COMPLETE','scenes':rows})
