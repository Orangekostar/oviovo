"""Prediction-only surface/vocabulary diagnostics; no diagnostic main methods."""
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax

from .binding import load_scene
from .common import arrays_record,canonical_digest,plain,producer,unchanged_receipt,verified,write
from .query_plan import source_scores,plan_scene


def grouped(scores,temperatures,probability=True):
    values={n:(softmax(s/float(temperatures[n])) if probability else s/float(temperatures[n]))
        for n,s in scores.items() if s is not None}
    groups=[];nq=[values[n] for n in ('N','Q') if n in values]
    if nq:groups.append(np.mean(nq,axis=0))
    if 'F' in values:groups.append(values['F'])
    return np.mean(groups,axis=0) if groups else None


def _sign(value):
    return 0 if abs(value)<=1e-12 else (1 if value>0 else -1)


def vocabulary_scene(binding,scene):
    root=Path(binding['output_root'])/'diagnostics/vocabulary'/scene
    bound=binding['scenes'][scene]
    key=canonical_digest({'binding':binding['identity'],'scene':scene,'context':bound['context'],
                         'operator':binding['spec']['sha256'],'producer_file':str(Path(__file__).resolve()),
                         'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    # Probe the task-owned receipt before loading a large parent mesh.
    old=unchanged_receipt(root/'receipt.json',key)
    if old:
        p,e=load_scene(binding,scene)
        for dep in old['dependencies']:p.index.identity(dep['path'],dep)
        return old
    begin=time.perf_counter();p,e=load_scene(binding,scene);ids=p.valid_ids;n=len(ids)
    owners=list(map(int,np.unique(p.d2_owner_ids[p.d2_owner_ids>0])))
    score_array=np.full((len(owners),3,n),np.nan,np.float64);available=np.zeros((len(owners),3),bool)
    priors=np.full((len(owners),n),np.nan,np.float64);classes=np.zeros(len(owners),np.int64)
    rows=[];skipped=[];maximum_full_error=0.;source_pair_invariant_checks=0;scaled_invariant_checks=0
    sizes=sorted({min(int(s),n) for s in binding['specification']['diagnostics']['vocabulary_sizes'] if s!='FULL'}|{n})
    for j,owner in enumerate(owners):
        old_class=int(p.semantic_labels[np.flatnonzero(p.owner_ids==owner)[0]])
        scores=source_scores(p,owner);prior=p.probabilities[str(owner)]['probabilities'];classes[j]=old_class
        for q,name in enumerate(('N','Q','F')):
            if scores[name] is not None:available[j,q]=True;score_array[j,q]=scores[name]
        if prior is None or old_class not in ids or not any(s is not None for s in scores.values()):
            skipped.append({'owner':owner,'reason':'NO_FULL_PRIOR_OR_VALID_PAIR'});continue
        prior=np.asarray(prior,np.float64);priors[j]=prior
        full=grouped(scores,p.temperatures);error=float(np.max(np.abs(full-prior)));maximum_full_error=max(maximum_full_error,error)
        if error>1e-12 or ids[int(np.argmax(full))]!=old_class:raise ValueError('full vocabulary diagnostic does not reconstruct D2')
        c0=ids.index(old_class);others=[i for i in range(n) if i!=c0];c1=max(others,key=lambda i:prior[i])
        original_probability_margin=float(full[c1]-full[c0]);full_raw=grouped(scores,p.temperatures,False)
        original_raw_margin=float(full_raw[c1]-full_raw[c0])
        for seed in (0,17,29):
            distractors=sorted((i for i in range(n) if i not in (c0,c1)),
                key=lambda i:hashlib.sha256(f'{seed}|{ids[i]}'.encode()).hexdigest())
            pair_probability_margin=None;pair_raw_margin=None
            for size in sizes:
                subset=sorted([c0,c1,*distractors[:size-2]]);local0=subset.index(c0);local1=subset.index(c1)
                restricted={k:None if s is None else s[subset] for k,s in scores.items()}
                probability=grouped(restricted,p.temperatures);raw=grouped(restricted,p.temperatures,False)
                margin=float(probability[local1]-probability[local0]);raw_margin=float(raw[local1]-raw[local0])
                if size==2:pair_probability_margin=margin;pair_raw_margin=raw_margin
                if not np.isclose(raw_margin,original_raw_margin,rtol=0,atol=1e-12):raise ValueError('scaled pair difference changed with vocabulary')
                scaled_invariant_checks+=1
                for name,s in scores.items():
                    if s is None:continue
                    source_margin=float(s[c1]-s[c0]);restricted_margin=float(restricted[name][local1]-restricted[name][local0])
                    if _sign(source_margin)!=_sign(restricted_margin):raise ValueError('single-source pair ordering changed')
                    source_pair_invariant_checks+=1
                rows.append({'owner':owner,'seed':seed,'size':size,'vocabulary':'FULL' if size==n else str(size),
                    'ordered_ids':[ids[i] for i in subset],'c0':old_class,'c1':ids[c1],
                    'probability_pair_margin':margin,'scaled_score_pair_margin':raw_margin,
                    'probability_pair_tie':_sign(margin)==0,'scaled_pair_tie':_sign(raw_margin)==0,
                    'strict_probability_sign_change_vs_pair':_sign(margin)*_sign(pair_probability_margin)==-1,
                    'strict_probability_sign_change_vs_FULL':_sign(margin)*_sign(original_probability_margin)==-1,
                    'strict_scaled_sign_change_vs_pair':_sign(raw_margin)*_sign(pair_raw_margin)==-1,
                    'probability_winner':ids[subset[int(np.argmax(probability))]],
                    'scaled_score_winner':ids[subset[int(np.argmax(raw))]],
                    'probability_third_class_winner':int(np.argmax(probability)) not in (local0,local1),
                    'full_D2_class':old_class,'FULL_probability_max_error':error})
    arr=arrays_record(root/'source_scores.npz',{'owner_ids':np.asarray(owners,np.int64),'valid_ids':np.asarray(ids,np.int64),
        'scores_NQF':score_array,'available_NQF':available,'p0':priors,'G1_classes':classes},p.index)
    details=write(root/'records.json',{'scene':scene,'records':rows,'skipped':skipped,'GT_used':False,
                                    'AP_on_reduced_vocabulary':'NOT_COMPUTED'})
    detail_file=p.index.identity(root/'records.json')
    summary=write(root/'receipt.json',{'status':'COMPLETE','scene':scene,'input_identity':key,
        'dependencies':[arr,detail_file],'lossless_source_scores':arr,'records_identity':details['identity'],
        'incumbents':len(owners),'diagnosed_owners':len(owners)-len(skipped),'record_count':len(rows),
        'strict_probability_sign_changes_vs_pair':sum(r['strict_probability_sign_change_vs_pair'] for r in rows),
        'strict_probability_sign_changes_vs_FULL':sum(r['strict_probability_sign_change_vs_FULL'] for r in rows),
        'scaled_pair_sign_changes':sum(r['strict_scaled_sign_change_vs_pair'] for r in rows),
        'single_source_pair_invariant_checks':source_pair_invariant_checks,'scaled_pair_invariant_checks':scaled_invariant_checks,
        'FULL_D2_maximum_error':maximum_full_error,'D2_probability_parity':e.d2_probability_parity,
        'new_image_encodings':0,'new_text_encodings':0,'GT_used':False,'elapsed_seconds':time.perf_counter()-begin})
    p.index.write_memo(Path(binding['output_root'])/'inputs'/scene/'verifications.json')
    print('VOCABULARY',scene,summary['diagnosed_owners'],'owners',len(rows),'records',flush=True)
    return summary


def _counts(counts,area=None):
    values=np.asarray(counts)
    result={'locations':len(values),'0':int(np.sum(values==0)),'1':int(np.sum(values==1)),'2+':int(np.sum(values>=2))}
    if area is not None:
        a=np.asarray(area,np.float64);total=float(a.sum())
        result.update(total_area=total,area_0=float(a[values==0].sum()),area_1=float(a[values==1].sum()),
            area_2plus=float(a[values>=2].sum()),repeat_area_fraction=None if not total else float(a[values>=2].sum()/total))
    return result


def surface_scene(binding,scene):
    manifest=plan_scene(binding,scene);root=Path(binding['output_root'])/'diagnostics/surface'/scene
    key=canonical_digest({'plan':manifest['identity'],'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    old=unchanged_receipt(root/'receipt.json',key)
    if old:return old
    begin=time.perf_counter();rows=[]
    dtype=np.dtype([('x',np.float64),('y',np.float64),('z',np.float64)])
    for owner,r in manifest['owners'].items():
        with np.load(r['support']['path'],allow_pickle=False) as arr:
            x=arr['xyz'];a=arr['area'];legacy_x=arr['legacy_coordinates'];legacy_count=arr['legacy_coordinate_counts']
            sample_key=np.ascontiguousarray(x).view(dtype).ravel();legacy_key=np.ascontiguousarray(legacy_x).view(dtype).ravel()
            positions=np.searchsorted(legacy_key,sample_key)
            if np.any(positions>=len(legacy_x)) or not np.array_equal(legacy_x[positions],x):raise ValueError('physical sample not an exact legacy coordinate')
            v=arr['V'];o=arr['O'];alignment=v&~o
            rows.append({'owner':int(owner),'selected':True,'query_eligible':r['query_eligible'],
                'raw_source_rows':_counts(arr['legacy_row_counts']),
                'raw_row_unit':'SOURCE_ROW_COUNT_NOT_PHYSICAL_AREA',
                'co_located_legacy_coordinates':_counts(legacy_count),
                'quadrature_legacy_coordinate_hits':_counts(legacy_count[positions],a),
                'direct_geometric_V':_counts(v.sum(axis=0),a),
                'direct_FULL_supported_O':_counts(o.sum(axis=0),a),
                'sampling':r['sampling'],'V_outside_FULL_site_view_pairs':int(alignment.sum()),
                'V_outside_FULL_area_view_sum':float(np.sum(alignment.astype(float)@a)),
                'support_geometry_changes':0,'legacy_FULL_repainting':False})
    result=write(root/'receipt.json',{'status':'COMPLETE','scene':scene,'input_identity':key,'owners':rows,
        'selected_denominator':len(rows),'site_preparation_audit':manifest['site_receipt']['audit'],
        'sample_convention':'BOUNDED_AREA_QUADRATURE_NOT_EXACT_SURFACE_INTEGRAL',
        'GT_used':False,'new_GPU_inference':0,'elapsed_seconds':time.perf_counter()-begin})
    print('SURFACE',scene,len(rows),'selected owners',flush=True);return result


def diagnose(binding,*,vocabulary=True,surface=True):
    root=Path(binding['output_root']);vocab={};surfaces={}
    if vocabulary:
        for subset in binding['cohorts'].values():
            for scene in subset:vocab[scene]=vocabulary_scene(binding,scene)['identity']
        write(root/'diagnostics/vocabulary/summary.json',{'status':'COMPLETE','scenes':vocab,'expected_scenes':26,
            'new_image_encodings':0,'new_text_encodings':0,'GT_used':False})
    if surface:
        for subset in binding['screen_cohorts'].values():
            for scene in subset:surfaces[scene]=surface_scene(binding,scene)['identity']
        write(root/'diagnostics/surface/summary.json',{'status':'COMPLETE','scenes':surfaces,'expected_scenes':4,'GT_used':False})
    return write(root/'diagnostics/summary.json',{'status':'DIAGNOSTICS_COMPLETE' if vocabulary and surface else 'PARTIAL_DIAGNOSTICS',
        'vocabulary_scenes':vocab,'surface_scenes':surfaces,'new_visual_inference':0,'GT_used':False})
