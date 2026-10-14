"""Fixed residual support and class-agnostic one-to-one geometric diagnostic."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def saved_surface(records,context):
    from pathlib import Path
    from .common import read_document,DependencyError
    if 'surface' in context:descriptor=records.resolver.rewrite(context['surface'])
    else:
        capture_path=Path(records.resolver.resolve(context['capture_manifest']))
        capture=read_document(capture_path,records.index)
        if 'surface' not in capture:raise DependencyError('Missing saved surface descriptor: '+str(capture_path))
        descriptor=records.resolver.rewrite(capture['surface']);path=Path(descriptor['path'])
        if not path.is_absolute():descriptor=dict(descriptor,path=str(capture_path.parent/path))
    records.index.identity(descriptor['path'],descriptor);return descriptor

def validate_residual_support(before,raw,after):
    before,raw,after=map(np.asarray,(before,raw,after))
    if before.shape!=raw.shape or before.shape!=after.shape or not np.array_equal(after[before>0],before[before>0]):raise ValueError('Recovery changed incumbent support')
    changed=(before==0)&(after>0)
    if not np.array_equal(after[changed],raw[changed]):raise ValueError('Recovery is outside its original raw-owner support')

def overlap_matrix(projected_owners,gt_instances,valid_ids,minimum):
    owners,gt=np.asarray(projected_owners),np.asarray(gt_instances)
    if owners.shape!=gt.shape:raise ValueError('Projected owners and GT geometry differ')
    ids,counts=np.unique(owners,return_counts=True)
    pred=[int(o) for o,n in zip(ids,counts,strict=True) if o>0 and n>=minimum]
    gids,gcounts=np.unique(gt,return_counts=True)
    valid=set(map(int,valid_ids))
    ground=[int(g) for g,n in zip(gids,gcounts,strict=True) if g>=1000 and int(g)//1000 in valid and n>=minimum]
    intersections=np.zeros((len(pred),len(ground)),np.int64)
    for i,p in enumerate(pred):
        selected=gt[owners==p]; values,nums=np.unique(selected,return_counts=True);byid=dict(zip(map(int,values),map(int,nums),strict=True))
        for j,g in enumerate(ground):intersections[i,j]=byid.get(g,0)
    pcounts=np.array([np.count_nonzero(owners==p) for p in pred],np.int64)
    gcounts=np.array([np.count_nonzero(gt==g) for g in ground],np.int64)
    union=pcounts[:,None]+gcounts[None,:]-intersections
    iou=np.divide(intersections,union,out=np.zeros(intersections.shape,np.float64),where=union>0)
    return pred,ground,iou,intersections,pcounts,gcounts

def one_to_one(projected_owners,gt_instances,valid_ids,minimum,threshold):
    pred,ground,iou,_,_,_=overlap_matrix(projected_owners,gt_instances,valid_ids,minimum)
    legal=iou>threshold
    weights=np.zeros((len(pred),len(ground)+len(pred)),np.float64)
    weights[:,:len(ground)]=np.where(legal,min(len(pred),len(ground))+1+iou,0.)
    pi,gi=linear_sum_assignment(weights,maximize=True)
    pairs=[dict(owner=pred[i],gt_instance=ground[j],IoU=float(iou[i,j])) for i,j in zip(pi,gi,strict=True) if j<len(ground) and legal[i,j]]
    return dict(measurement_kind='DIAGNOSTIC_ONE_TO_ONE_GEOMETRY',threshold=threshold,strict_IoU=True,matching_count=len(pairs),predicted_owners=len(pred),legal_GT_instances=len(ground),recall=len(pairs)/len(ground) if ground else None,precision=len(pairs)/len(pred) if pred else None,pairs=pairs,semantic_labels_used_for_pairing=False)

def official_entries(trace,threshold):
    """Resolve score-entry ownership from original events, retaining tie ambiguity."""
    import re
    def owner(path):
        match=re.search(r'owner_(\d+)\.npy$',path or '');return int(match.group(1)) if match else None
    winners={};fps=[];ties=[]
    for e in trace['events']:
        if e['distance_index']!=0 or not np.isclose(e['overlap_threshold'],threshold):continue
        key=(e['scene_key'],e['class_index'],e.get('gt_id'))
        if e['event']=='first_match':winners[key]=owner(e['candidate_file'])
        elif e['event']=='duplicate':
            ambiguous=e['score_identity_ambiguous_tie'];winners[key]=None if ambiguous else owner(e['tp_score_owner'])
            fps.append(None if ambiguous else owner(e['fp_score_owner']))
            if ambiguous:ties.append(dict(gt_id=e['gt_id'],candidates=[owner(p) for p in e['score_candidate_files']]))
        elif e['event']=='ignore_test' and e['counted_fp']:fps.append(owner(e['candidate_file']))
    states=[s for s in trace['states'] if s['distance_index']==0 and np.isclose(s['overlap_threshold'],threshold)]
    tp_entries=sum(sum(v==1 for v in s['y_true']) for s in states);fp_entries=sum(sum(v==0 for v in s['y_true']) for s in states)
    return dict(threshold=threshold,TP_score_entries=tp_entries,FP_score_entries=fp_entries,
                definite_TP_owners=[o for o in winners.values() if o is not None],definite_FP_score_entry_owners=[o for o in fps if o is not None],
                unique_GT_matched=sorted({int(k[2]) for k in winners}),ambiguous_ties=ties,hard_FN=sum(s['hard_false_negatives'] for s in states),
                ownership_warning='SCORE_ENTRIES_MAY_REPEAT_OWNER;AMBIGUOUS_TIES_EXCLUDED_FROM_DEFINITE_OWNER_COUNTS')

def run(root):
    from pathlib import Path
    from .binding import load_binding
    from .common import ConsumptionIndex,verified,read_document,write,unit_identity,cached_record,_array_digest,family
    from .records import Records
    from static_ovmap.recovery_wave2.recovery_registry import support_digest
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.branch_utility_wave2_v1.conveyor import geometry_matches
    root=Path(root);binding=load_binding(root);index=ConsumptionIndex();records=Records(root,binding,index)
    key=unit_identity('recovery',dict(inventory=records.inventory['identity']),producers=('common','records','recovery'))
    previous=cached_record(root/'recovery.json',key)
    if previous:return previous
    scene_rows=[];object_rows=[];missing=[]
    for cohort,scenes in binding['cohorts'].items():
        for scene in scenes:
            baseline=records.payload(scene,'A0');after=records.payload(scene,'A3');before=records.payload(scene,'A1')
            if baseline.geometry!=after.geometry or before.geometry!=after.geometry:raise ValueError('Recovery mesh differs')
            context=read_document(Path(binding['compact_root'])/'contexts'/f'{scene}.json',index)
            surface=saved_surface(records,context)
            registry=read_document(Path(binding['compact_root'])/'projected_views'/scene/'registry.json',index)
            with np.load(surface['path'],allow_pickle=False) as arrays:raw=arrays['original_owner'].copy();xyz=arrays['surface_xyz'].copy()
            if _array_digest(raw)!=registry['raw_owner_sha256'] or _array_digest(baseline.owner_ids)!=registry['painted_owner_sha256']:raise ValueError('Residual registry owner identities differ')
            validate_residual_support(baseline.owner_ids,raw,after.owner_ids);validate_residual_support(baseline.owner_ids,raw,records.payload(scene,'A2').owner_ids)
            candidate={r['raw_owner']:r for r in registry['candidates']};incumbents=set(owner_labels(baseline));new=set(owner_labels(after))-incumbents
            source=read_document(Path(binding['area_root'])/'regions'/scene/'G1_FC.json',index)
            for owner in new:
                if owner not in candidate:raise ValueError('Recovery owner outside fixed registry')
                support=(raw==owner)&(baseline.owner_ids==0)
                if int(support.sum())!=candidate[owner]['residual_source_rows'] or support_digest(xyz[support])!=candidate[owner]['support_sha256']:raise ValueError('New support differs from exact residual registry')
                if not np.array_equal(after.owner_ids==owner,support):raise ValueError('Recovery support was truncated or expanded')
            score0=records.score(scene,'A1');score1=records.score(scene,'A3');gt=records.targets(score1);o0=records.projected(scene,'A1',score0);o1=records.projected(scene,'A3',score1)
            labels0=owner_labels(before);labels1=owner_labels(after);namespace=records.released(score1);minimum=score1['context']['minimum']
            geometry={name:[one_to_one(owners,gt,namespace['VALID_CLASS_IDS'],minimum,t) for t in (.25,.5,.75)] for name,owners in [('A1',o0),('A3',o1)]}
            matches=geometry_matches(o1,gt,score1['context']['valid_ids']);official={}
            for method,score in [('A1',score0),('A3',score1)]:
                path=Path(score['manifest']).with_name('trace.json.gz')
                if path.exists():
                    trace=read_document(path,index,sealed=False);official[method]=[official_entries(trace,t) for t in (.25,.5,.75)]
                else:missing.append(dict(scene=scene,method=method,reason='MISSING_OFFICIAL_TRACE'));official[method]=None
            changes=[]
            if all(official.values()):
                for a,b in zip(official['A1'],official['A3'],strict=True):
                    old=set(a['definite_TP_owners']);current=set(b['definite_TP_owners'])
                    changes.append(dict(threshold=a['threshold'],new_definite_TP_score_entries=sum(o in new for o in b['definite_TP_owners']),new_definite_FP_score_entries=sum(o in new for o in b['definite_FP_score_entry_owners']),
                                        new_unique_GT_matched=sorted(set(b['unique_GT_matched'])-set(a['unique_GT_matched'])),old_TP_retained=sorted(old&current),old_TP_lost=sorted(old-current),ambiguous_tie_count=len(b['ambiguous_ties'])))
            projected_ids,counts=np.unique(o1,return_counts=True);areas=dict(zip(map(int,projected_ids),map(int,counts),strict=True));unknown_known=sum(areas.get(o,0) for o in new if labels1[o]!=0)
            for owner in sorted(labels1):
                object_rows.append(dict(cohort=cohort,family=family(scene),scene=scene,owner=owner,domain='NEW_RECOVERY' if owner in new else 'INCUMBENT',old_label=labels0.get(owner,0),new_label=labels1[owner],source_available=source['objects'].get(str(owner),{}).get('available') if owner in new else None,
                                        projected_points=areas.get(owner,0),evaluable=areas.get(owner,0)>=minimum,**matches.get(owner,dict(GT_instance=None,GT_class=None,IoU=None,matched_IoU50=False,owner_points=0))))
            scene_rows.append(dict(cohort=cohort,family=family(scene),scene=scene,registry=registry['identity'],raw_support_validated=True,incumbent_support_unchanged=True,
                                   candidate_count=len(candidate),source_available_candidates=sum(bool(r['available']) for r in source['objects'].values()),new_instances=len(new),new_owners=sorted(new),new_projected_evaluable_owners=sum(areas.get(o,0)>=minimum for o in new),unknown_to_known_target_points=unknown_known,
                                   all_positive_owners=len(labels1),projected_evaluable_owners=sum(areas.get(o,0)>=minimum for o in labels1),point_coverage=sum(areas.get(o,0) for o in labels1),target_points=len(gt),
                                   geometry=geometry,official_entries=official,official_changes=changes,matching_rules=['RELEASED_OFFICIAL_SCORE_ENTRIES','DESCRIPTIVE_MAX_IOU_MULTI_TO_ONE','DIAGNOSTIC_ONE_TO_ONE_GEOMETRY']))
            print('RECOVERY_SCENE',scene,len(new),flush=True);records.cache.clear()
    index.write_memo(root/'recovery/verifications.json')
    return write(root/'recovery.json',dict(status='COMPLETE' if not missing else 'PARTIAL_DEPENDENCY',input_identity=key,scene_rows=scene_rows,object_rows=object_rows,missing=missing,new_model_forwards=0,new_scene_assign_calls=0,
                 controls=dict(A4='NATIVE_LABEL_PLUS_ORIGINAL_RECOVERY',A5='F_ONLY_PLUS_SAME_RECOVERY_NOT_BARE_F',U2='HISTORICAL_DIFFERENT_OBSERVATION_LIBRARY',G3='SAME_TOP3_MULTIVIEW_AGGREGATION'),comparability='R_FIXED_DOMAIN_PRIMARY;CONTROL_OPERATION_LIBRARIES_EXPLICIT'))
