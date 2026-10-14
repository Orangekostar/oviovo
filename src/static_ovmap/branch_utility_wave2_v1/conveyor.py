"""Read-only existing FC8/FG2 conveyor, with fixed-owner descriptive GT matches."""
from pathlib import Path
import numpy as np
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.scannet_study import load_prediction
from .common import verified,read,write,PathResolver,ConsumptionIndex,require

def geometry_matches(owners,gt,valid_ids):
    if owners.shape!=gt.shape:raise ValueError('Existing scorer GT/owner support differs')
    oo,oi,oc=np.unique(owners,return_inverse=True,return_counts=True);gg,gi,gc=np.unique(gt,return_inverse=True,return_counts=True);joint=np.bincount(oi.astype(np.int64)*len(gg)+gi,minlength=len(oo)*len(gg)).reshape(len(oo),len(gg));legal=np.isin(gg//1000,valid_ids)&(gc>=100)&(gg>0);out={}
    for i,owner in enumerate(oo):
        if owner<=0:continue
        overlaps=joint[i]/np.maximum(oc[i]+gc-joint[i],1);overlaps[~legal]=-1;j=int(overlaps.argmax())
        out[int(owner)]=dict(GT_instance=int(gg[j]) if overlaps[j]>=0 else None,GT_class=int(gg[j]//1000) if overlaps[j]>=0 else None,IoU=float(overlaps[j]) if overlaps[j]>=0 else None,matched_IoU50=bool(overlaps[j]>.5),owner_points=int(oc[i]),GT_points=int(gc[j]) if overlaps[j]>=0 else None)
    return out

def scorer_geometry_matches(owners,gt,nearest,matched,valid_ids):
    from static_ovmap.module_validation.scannet_study import project_values
    return geometry_matches(project_values(owners,nearest,matched),gt,valid_ids)

def run(binding):
    root=Path(binding['output_root']);parent=Path(binding['fg_parent_root']);lr=Path(binding['lr_parent_root']);resolver=PathResolver(binding['path_map']);index=ConsumptionIndex(root/'conveyor/verifications.json');missing=[]
    required=[parent/'evaluation/store.json',parent/'recognition/regression.json',lr/'recognition/regression.json']
    for scenes in binding['cohorts'].values():
        for scene in scenes:required += [parent/'predictions'/scene/'receipt.json',parent/'decisions'/scene/'FG2_SURFACE.json',parent/'decisions'/scene/'S29_FG2_SURFACE.json',lr/'regression'/scene/'manifest.json']
    missing=[str(p) for p in required if not p.exists()]
    if missing:return write(root/'conveyor.json',dict(status='ANALYSIS_BLOCKED_DEPENDENCY',missing=missing,new_native_calls=0,new_scorer_calls=0,TRAIN_DEV_not_blocked=True))
    scores=resolver.rewrite(verified(parent/'recognition/regression.json'));base_scores=resolver.rewrite(verified(lr/'recognition/regression.json'));store=resolver.rewrite(verified(parent/'evaluation/store.json'));rows=[];scene_rows=[]
    for cohort,scenes in binding['cohorts'].items():
        for scene in scenes:
            lock=resolver.rewrite(verified(parent/'predictions'/scene/'receipt.json'));base=load_prediction(lock['methods']['FG0_FC8']['manifest']);base_labels=owner_labels(base);baseline_row=next(r for r in store['scene_metrics'] if r['scene']==scene and r['method']=='FG0_FC8');baseline_score=read(baseline_row['evaluation_receipt']);index.identity(baseline_score['context']['gt']['path'],baseline_score['context']['gt']);gt=np.load(baseline_score['context']['gt']['path'],allow_pickle=False);projection=baseline_score['context']['projection_arrays'];index.identity(projection['path'],projection)
            with np.load(projection['path'],allow_pickle=False) as a:matches=scorer_geometry_matches(base.owner_ids,gt,a['nearest'],a['matched'],baseline_score['context']['valid_ids'])
            manifest=resolver.rewrite(verified(lr/'regression'/scene/'manifest.json'));eligible=set(manifest['eligible_owners'])
            for seed,name in ((17,'FG2_SURFACE'),(29,'S29_FG2_SURFACE')):
                decision=resolver.rewrite(verified(parent/'decisions'/scene/(name+'.json')));payload=load_prediction(lock['methods'][name]['manifest']);labels=owner_labels(payload)
                if not np.array_equal(payload.owner_ids,base.owner_ids) or set(labels)!=set(base_labels):raise ValueError('Existing conveyor fixed owner support differs')
                scene_bank=[]
                for owner_text,d in decision['owners'].items():
                    owner=int(owner_text);key=f'{scene}:{owner}';oldrow=base_scores['records'][key]['arrays'];newrow=scores['records'][key]['arrays'];index.identity(oldrow['path'],oldrow);index.identity(newrow['path'],newrow)
                    with np.load(oldrow['path'],allow_pickle=False) as a:
                        pos=list(a['methods']).index('LR04_FC_8');old_z=a['embeddings'][pos].copy();old_f=a['cosines'][pos].copy();ids=a['valid_ids'].copy()
                    with np.load(newrow['path'],allow_pickle=False) as a:
                        pos=list(a['methods']).index(name);new_z=a['embeddings'][pos].copy();new_f=a['cosines'][pos].copy()
                        if not np.array_equal(ids,a['valid_ids']):raise ValueError('Existing conveyor class order differs')
                    np.testing.assert_array_equal(new_f,np.asarray(d['new_F_cosines'],new_f.dtype));
                    # Final decisions and payloads are the already locked native-pool evidence.
                    if labels[owner]!=d['applied_class']:raise ValueError('Existing decision/payload applied class differs')
                    match=matches.get(owner,dict(GT_instance=None,GT_class=None,IoU=None,matched_IoU50=False,owner_points=0,GT_points=None));old_correct=base_labels[owner]==match['GT_class'];new_correct=labels[owner]==match['GT_class'];row=dict(cohort=cohort,scene=scene,seed=seed,owner=owner,eligible=owner in eligible,embedding_difference_norm=float(np.linalg.norm(new_z-old_z)),F_difference_norm=float(np.linalg.norm(new_f-old_f)),FC8_F_top1=int(ids[int(old_f.argmax())]),FG2_F_top1=int(ids[int(new_f.argmax())]),FC8_fused_class=base_labels[owner],FG2_fused_class=labels[owner],F_top1_changed=int(old_f.argmax())!=int(new_f.argmax()),fused_class_changed=base_labels[owner]!=labels[owner],matched_corrected=match['matched_IoU50'] and new_correct and not old_correct,matched_harmed=match['matched_IoU50'] and old_correct and not new_correct,**match)
                    rows.append(row);scene_bank.append(row)
                actual=next(r for r in store['scene_metrics'] if r['scene']==scene and r['method']==name)
                scene_rows.append(dict(cohort=cohort,scene=scene,seed=seed,eligible_owners=len(eligible),readout_owners=len(scene_bank),full_owner_registry=len(labels),F_direction_changed=sum(r['embedding_difference_norm']>0 for r in scene_bank),F_top1_changed=sum(r['F_top1_changed'] for r in scene_bank),fused_class_changed=sum(r['fused_class_changed'] for r in scene_bank),F_top1_change_survived_fusion=sum(r['F_top1_changed'] and r['fused_class_changed'] for r in scene_bank),matched_IoU50_owners=sum(r['matched_IoU50'] for r in scene_bank),matched_corrected=sum(r['matched_corrected'] for r in scene_bank),matched_harmed=sum(r['matched_harmed'] for r in scene_bank),official_metrics=actual['metrics'],FC8_metrics=baseline_row['metrics'],official_deltas={k:actual['metrics'][k]-baseline_row['metrics'][k] if actual['metrics'][k] is not None and baseline_row['metrics'][k] is not None else None for k in ('apall','ap50','ap25','miou','macc')},scorer_receipt=actual['evaluation_receipt'],FC8_scorer_receipt=baseline_row['evaluation_receipt']))
            print('EXISTING_CONVEYOR',scene,flush=True)
    if len(scene_rows)!=52:raise ValueError('Existing26-scene×2-head coverage incomplete')
    pooled=[]
    for cohort in binding['cohorts']:
        for seed,name in ((17,'FG2_SURFACE'),(29,'S29_FG2_SURFACE')):
            old=store['pooled_metrics'][cohort]['FG0_FC8'];new=store['pooled_metrics'][cohort][name];pooled.append(dict(cohort=cohort,seed=seed,FG2=new['metrics'],FC8=old['metrics'],official_deltas={k:new['metrics'][k]-old['metrics'][k] if new['metrics'][k] is not None and old['metrics'][k] is not None else None for k in ('apall','ap50','ap25','miou','macc')}))
    index.write_memo(root/'conveyor/verifications.json')
    return write(root/'conveyor.json',dict(status='COMPLETE',owner_rows=rows,scene_rows=scene_rows,pooled=pooled,new_native_calls=0,new_scorer_calls=0,geometry_unchanged=True,GT_scope='EXISTING_PREDICTIONS_ALREADY_LOCKED_AND_SCORED',matching_rule='DESCRIPTIVE_MAX_GEOMETRIC_IOU;>0.5;GT_AND_OWNER_SUPPORT_FIXED;NOT_OFFICIAL_TP_OR_AP_UPPER_BOUND',no_new_M1_maps=True))
