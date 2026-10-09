"""After-lock fixed-support semantic corrections and unique class-aware matches."""
from pathlib import Path
import time

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.evidence_exploration.analysis import trace_entries,zipped
from static_ovmap.minimal_instance_repair.diagnostics import maximum_matches
from static_ovmap.module_validation.scannet_study import load_prediction,project_values
from static_ovmap.samv_local_probe.diagnostics import eligible_ground_truth

from .binding import load_scene
from .common import _array_digest,canonical_digest,read,unchanged_receipt,verified,write


def fixed_supports(inputs,gt,eligible,minimum):
    projected=project_values(inputs.g1.owner_ids,inputs.nearest,inputs.matched)
    gids,gcounts=np.unique(gt,return_counts=True);sizes=dict(zip(map(int,gids),map(int,gcounts)))
    positive=projected>0
    pairs,counts=np.unique(np.column_stack((projected[positive],gt[positive])),axis=0,return_counts=True)
    by={}
    for (owner,gid),count in zip(pairs,counts):by.setdefault(int(owner),[]).append((int(gid),int(count)))
    supports=[]
    for owner in sorted(map(int,np.unique(inputs.g1.owner_ids[inputs.g1.owner_ids>0]))):
        intersections=by.get(owner,[]);area=sum(n for gid,n in intersections);overlaps=[]
        for gid,n in intersections:
            if gid in eligible:overlaps.append({'gt_id':gid,'class':int(eligible[gid]['label_id']),
                'intersection':n,'iou':n/(area+sizes[gid]-n)})
        overlaps.sort(key=lambda m:(-m['intersection'],m['gt_id']))
        supports.append({'name':str(owner),'owner':owner,'evaluation_points':area,
            'instance_size_eligible':area>=minimum,'overlaps':overlaps})
    return supports


def _class_aware(supports,labels,eligible,threshold):
    restricted=[{**s,'overlaps':[m for m in s['overlaps'] if m['class']==labels[s['owner']]]} for s in supports]
    matching=maximum_matches(restricted,eligible,threshold)
    support_ties=[s['owner'] for s in restricted if sum(m['iou']>threshold for m in s['overlaps'])>1]
    gt_degree={}
    for s in restricted:
        if not s['instance_size_eligible']:continue
        for m in s['overlaps']:
            if m['iou']>threshold:gt_degree[m['gt_id']]=gt_degree.get(m['gt_id'],0)+1
    matching.update(possible_nonunique_supports=support_ties,possible_nonunique_GT_ids=sorted(g for g,n in gt_degree.items() if n>1),
        matching_identity_warning='CANONICAL_MAXIMUM_MATCHING; identity gains/losses are not interpreted as unique physical corrections when graph ties exist')
    return matching


def analyze_scene(binding,stage,scene,methods):
    root=Path(binding['output_root']);global_lock=verified(root/'predictions'/stage/'summary.json')
    if global_lock['status']!='ALL_STAGE_PREDICTIONS_LOCKED':raise ValueError('GT diagnostics require full current-stage lock')
    lock=verified(root/'predictions'/stage/scene/'receipt.json');evaluated=verified(root/'evaluation'/stage/scene/'receipt.json')
    plan=verified(root/'plans'/scene/'manifest.json');choices=verified(root/'choices'/scene/'receipt.json')
    key=canonical_digest({'prediction':lock['identity'],'evaluation':evaluated['identity'],'plan':plan['identity'],
        'producer_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()})
    path=root/'analysis'/stage/scene/'receipt.json';old=unchanged_receipt(path,key)
    if old:return old
    start=time.perf_counter();p,inputs=load_scene(binding,scene)
    gt,eligible,minimum=eligible_ground_truth(binding,inputs)
    if minimum!=100:raise ValueError('fixed-support diagnostic eligibility differs')
    supports=fixed_supports(inputs,gt,eligible,minimum);by={s['owner']:s for s in supports}
    agnostic={str(t):maximum_matches(supports,eligible,t) for t in (.5,.75)}
    owner_rows={};truths={}
    for owner in plan['selected_owner_ids']:
        s=by[owner];overlap=s['overlaps'];best=overlap[0] if overlap and s['instance_size_eligible'] else None
        tied=[m for m in overlap if best and m['intersection']==best['intersection']]
        classes={m['class'] for m in tied};truth=best['class'] if best and len(classes)==1 else None
        truths[owner]=truth
        owner_rows[str(owner)]={'owner':owner,'fixed_original_GT_id':best['gt_id'] if best else None,
            'fixed_class':truth,'tied_GT_ids':[m['gt_id'] for m in tied],
            'reference_ambiguous_class':len(classes)>1,'reference_undefined':truth is None,
            'reference_rule':'MAXIMUM_OLD_G1_INTERSECTION_TIE_MIN_GT_WITH_CLASS_AMBIGUITY_MARKED',
            'old_class':plan['owners'][str(owner)]['old_class'],'query_eligible':plan['owners'][str(owner)]['query_eligible'],
            'selected':True,'geometry_support':s,'choices':choices['owners'][str(owner)],'methods':{}}
    method_rows={};payloads={};trace_cache={}
    for method in methods:
        payload=load_prediction(lock['methods'][method]['manifest']);labels=owner_labels(payload);payloads[method]=payload
        if method=='DQ00_D2':
            # D2 is a literal historical baseline with a different recovered partition.
            method_rows[method]={'reference_only':True,'owner_partition':_array_digest(payload.owner_ids)};continue
        if _array_digest(payload.owner_ids)!=_array_digest(inputs.g1.owner_ids):raise ValueError('class-agnostic support negative control changed')
        aware={str(t):_class_aware(supports,labels,eligible,t) for t in (.5,.75)}
        row=next(r for r in evaluated['rows'] if r['method']==method);score=read(row['evaluation_receipt'])
        trace_path=Path(score['manifest']).with_name('trace.json.gz');inputs.index.identity(trace_path)
        if str(trace_path) not in trace_cache:trace_cache[str(trace_path)]=trace_entries(zipped(trace_path))
        entries=trace_cache[str(trace_path)]
        method_rows[method]={'class_agnostic':agnostic,'class_aware':aware,'current_ranks':list(payload.instance_ranks),
            'released_score_entries':entries,'score_entry_counts':{str(t):sum(abs(r['overlap_threshold']-t)<1e-12 for r in entries) for t in (.5,.75)},
            'raw_score_entries_are_not_unique_GT_counts':True,'owner_partition':_array_digest(payload.owner_ids)}
        for owner in plan['selected_owner_ids']:
            truth=truths[owner];label=labels[owner]
            owner_rows[str(owner)]['methods'][method]={'class':label,'correct_fixed_class':None if truth is None else label==truth,
                'class_changed_from_G1':label!=plan['owners'][str(owner)]['old_class'],
                'current_rank':dict(payload.instance_ranks)[owner]}
    result=write(path,{'status':'AFTER_LOCK_ANALYSIS_COMPLETE','scene':scene,'stage':stage,'input_identity':key,
        'owner_rows':owner_rows,'methods':method_rows,'class_agnostic_fixed_support':agnostic,
        'class_agnostic_new_method_changes':0,'GT_only_after_global_stage_lock':global_lock['identity'],
        'eligible_GT_ids':sorted(eligible),'elapsed_seconds':time.perf_counter()-start})
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json');return result


def analyze(binding,stage,cohorts,methods):
    receipts={s:analyze_scene(binding,stage,s,methods) for group in cohorts.values() for s in group}
    return write(Path(binding['output_root'])/'analysis'/stage/'summary.json',
        {'status':'AFTER_LOCK_ANALYSIS_COMPLETE','stage':stage,'scenes':{s:r['identity'] for s,r in receipts.items()},'GT_used_only_for_analysis':True})


def contrast(binding,candidate,reference,candidate_stage,reference_stage,scenes):
    root=Path(binding['output_root']);wr=rw=undefined=0;gt_net=0;changed=False;rows=[]
    for scene in scenes:
        a=verified(root/'analysis'/candidate_stage/scene/'receipt.json');b=verified(root/'analysis'/reference_stage/scene/'receipt.json')
        pa=verified(root/'predictions'/candidate_stage/scene/'receipt.json')['methods'][candidate]['prediction_key']
        pb=verified(root/'predictions'/reference_stage/scene/'receipt.json')['methods'][reference]['prediction_key'];changed|=pa!=pb
        scene_wr=scene_rw=0
        for owner,arow in a['owner_rows'].items():
            brow=b['owner_rows'][owner]
            if arow['fixed_class']!=brow['fixed_class'] or arow['fixed_original_GT_id']!=brow['fixed_original_GT_id']:
                raise ValueError('candidate/control fixed-support reference changed')
            ca=arow['methods'][candidate]['correct_fixed_class'];cb=brow['methods'][reference]['correct_fixed_class']
            if ca is None or cb is None:undefined+=1;continue
            scene_wr+=cb is False and ca is True;scene_rw+=cb is True and ca is False
        aa=a['methods'][candidate]['class_aware']['0.5'];bb=b['methods'][reference]['class_aware']['0.5']
        gained=sorted(set(aa['gt_ids'])-set(bb['gt_ids']));lost=sorted(set(bb['gt_ids'])-set(aa['gt_ids']))
        net=aa['count']-bb['count']
        if len(gained)-len(lost)!=net:raise ValueError('unique matching difference does not conserve cardinality')
        wr+=scene_wr;rw+=scene_rw;gt_net+=net
        rows.append({'scene':scene,'wrong_to_right':scene_wr,'right_to_wrong':scene_rw,'unique_GT50_gained':gained,
            'unique_GT50_lost':lost,'net_unique_GT50':net,'actual_output_changed':pa!=pb,
            'matching_identity_tie_warning':bool(aa['possible_nonunique_supports'] or aa['possible_nonunique_GT_ids']
                or bb['possible_nonunique_supports'] or bb['possible_nonunique_GT_ids'])})
    return {'candidate':candidate,'reference':reference,'candidate_stage':candidate_stage,'reference_stage':reference_stage,
        'actual_prediction_changed':bool(changed),'wrong_to_right':int(wr),'right_to_wrong':int(rw),
        'corrections_net':int(wr-rw),'undefined_fixed_class_comparisons':undefined,'gt50_net':gt_net,'scenes':rows}
