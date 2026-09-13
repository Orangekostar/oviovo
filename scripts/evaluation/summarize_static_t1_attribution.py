#!/usr/bin/env python3
"""Four evidence tables from completed cached T1 evaluations; no prediction changes."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.static_ovmap.attribution_objects import released_object_outcomes, candidate_ious, borrowing_diagnosis, candidate_event_diagnostics
from src.static_ovmap.attribution_regions import region_transitions
from scripts.evaluation.diagnose_static_t1_attribution import write, write_gzip

CORE=['AT_O_AREA','AT_S_RELEASED','AT_S_AREA','AT_U00','AT_U10','AT_U01','AT_U11']
PAIRS=[('candidate_addition','AT_O_AREA','AT_U00'),('borrow_without_nms','AT_U00','AT_U10'),
    ('borrow_with_nms','AT_U01','AT_U11'),('nms_without_borrow','AT_U00','AT_U01'),
    ('nms_with_borrow','AT_U10','AT_U11'),('fusion_increment_S1a','AT_O_AREA','AT_U11'),
    ('assignment_class_norm','AT_ASSIGN_RAW','AT_ASSIGN_CLASS_NORM'),
    ('assignment_ovi_fill','AT_ASSIGN_RAW','AT_ASSIGN_OVI_FILL'),
    ('rank_class_norm','AT_ASSIGN_RAW','AT_RANK_CLASS_NORM'),
    ('nms_order','AT_U11','AT_NMS_SF_FIRST'),
    ('readout_native_O','AT_NATIVE_O_AREA','AT_O_AREA'),
    ('readout_native_U11','AT_NATIVE_U11','AT_U11'),
    ('fusion_increment_native','AT_NATIVE_O_AREA','AT_NATIVE_U11')]


def load_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def main_metrics(row):
    return {'overlapping_AP':row['overlapping']['released']['all_ap'],
        'overlapping_AP50':row['overlapping']['released']['all_ap_50%'],
        'overlapping_AP25':row['overlapping']['released']['all_ap_25%'],
        'semantic_miou':row['semantic']['semantic_miou'],'semantic_macc':row['semantic']['semantic_macc'],
        'unique_AP':row['unique']['released']['all_ap'],'unique_AP50':row['unique']['released']['all_ap_50%'],
        'unique_AP25':row['unique']['released']['all_ap_25%'],
        'unique_canonical_AP75':row['unique_high_iou_canonical_diagnostic']['ap75']}


def difference(before,after):
    return {k:None if before[k] is None or after[k] is None else after[k]-before[k] for k in before}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','predictions','evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--geometry-bridge',type=Path)
    args=p.parse_args();config=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();protocol=json.loads((args.evaluation/'metric_protocol.json').read_text())
    thresholds=[next(x for x in protocol['overlaps_runtime'] if np.isclose(x,t)) for t in [.25,.5,.75]]
    valid=np.array(protocol['semantic_ids']);instance_ids=protocol['instance_ids']
    sem_gt=PlyData.read(config['gt_semantic_map'])['vertex']['label'];gt=np.load(config['reference_gt_ids'])
    performance=[];effects=[];objects=[];regional_rows=[];confusion_deltas=[];diagnoses={};examples={}
    for run in config['runs']:
        name=run['id'];ev=args.evaluation/name;pred=args.predictions/name
        local={path.parent.name:json.loads(path.read_text()) for path in ev.glob('AT_*/metrics.json')}
        if not set(CORE)<=set(local):raise ValueError('both-run core must finish before attribution report: '+name)
        docs={c:json.loads((pred/(c+'.json')).read_text()) for c in local}
        performance += list(local.values())
        regions=np.load(ev/'regions.npy');registry=json.loads((pred/'region_registry.json').read_text())
        registry['-1']={'source_support':'PROJECTION_UNMATCHED','candidate_multiplicity':None,'class_agreement':None}
        gt_data=np.load(ev/'candidate_gt_overlap.npz');gt_ids=gt_data['gt_ids'];gt_sizes=gt_data['gt_sizes']
        sizes=gt_data['projected_sizes'];intersection=gt_data['intersections'];ious=candidate_ious(intersection,sizes,gt_sizes)
        eligible=(gt_ids>=1000)&(gt_sizes>=100)&np.isin(gt_ids//1000,instance_ids)
        point_gt_index=np.searchsorted(gt_ids,gt)
        all_gt=gt_ids[eligible];semantics={};owners={};outcomes={};coverage={};extent={};event_counts={};ties={};event_details={};tp_ranks={}
        for condition,row in local.items():
            doc=docs[condition];kept=np.array(doc['kept'],dtype=int);labels=np.array(doc['labels'])
            semantics[condition]=np.load(ev/condition/'semantic.npy',mmap_mode='r')
            owners[condition]=np.load(ev/condition/'owners.npy',mmap_mode='r')
            encoded=owners[condition]*len(gt_ids)+point_gt_index
            unique_inter=np.bincount(encoded,minlength=(len(sizes)+1)*len(gt_ids)).reshape(len(sizes)+1,len(gt_ids))[1:]
            unique_sizes=unique_inter.sum(axis=1)
            outcomes[condition]={}
            event_counts[condition]={};event_details[condition]={};tp_ranks[condition]={}
            for representation in ('overlapping','unique'):
                trace=load_gzip(row[representation]['trace_path'])
                outcomes[condition][representation]={str(t):released_object_outcomes(trace,t) for t in thresholds}
                event_details[condition][representation]=candidate_event_diagnostics(trace,
                    unique_inter if representation=='unique' else intersection,
                    unique_sizes if representation=='unique' else sizes,labels,gt_ids,gt_sizes,instance_ids)
                tp_ranks[condition][representation]={}
                rank_values=np.array(list(map(float,doc['rank_scores_serialized'])))
                current_sizes=unique_sizes if representation=='unique' else sizes
                for t in thresholds:
                    tp_ranks[condition][representation][str(t)]={}
                    for gid,match in outcomes[condition][representation][str(t)]['matched'].items():
                        score=match['tp_score'];class_id=int(gid)//1000
                        candidates=kept[(labels[kept]==class_id)&(current_sizes[kept]>=100)]
                        state=next(x for x in trace['states'] if x['overlap_threshold']==t and x['class_index']==instance_ids.index(class_id))
                        event_scores=np.array(state['y_score'])
                        tp_ranks[condition][representation][str(t)][gid]={
                            'candidate_rank_interval':[1+int((rank_values[candidates]>score).sum()),int((rank_values[candidates]>=score).sum())],
                            'event_rank_interval':[1+int((event_scores>score).sum()),int((event_scores>=score).sum())],
                            'tp_score':score,'tp_score_owner':match['tp_score_owner'],
                            'definition':'same-class six-decimal scores; tie intervals, no epsilon; candidate ranks differ from duplicate-expanded AP event ranks; no R@K claim'}
                event_counts[condition][representation]={}
                for t in thresholds:
                    events=[e for e in trace['events'] if e['overlap_threshold']==t]
                    event_counts[condition][representation][str(t)]={kind:sum(e['event']==kind for e in events)
                        for kind in ['first_match','duplicate','hard_fn','ignore_test']}
                    event_counts[condition][representation][str(t)].update(
                        counted_unmatched_fp=sum(e['event']=='ignore_test' and e['counted_fp'] for e in events),
                        ignored_unmatched=sum(e['event']=='ignore_test' and not e['counted_fp'] for e in events))
            coverage[condition]={}
            for t in thresholds:
                valid_pred=kept[sizes[kept]>=100]
                geometric=(ious[valid_pred][:,eligible]>t)
                correct=geometric&(labels[valid_pred,None]==all_gt[None,:]//1000)
                coverage[condition][str(t)]={'denominator':len(all_gt),
                    'geometry_max_IoU_GT_ids':all_gt[geometric.any(axis=0)].tolist(),
                    'class_correct_max_IoU_GT_ids':all_gt[correct.any(axis=0)].tolist(),
                    'definition':'potential coverage; strict runtime threshold, projected pred>=100, evaluable GT; not one-to-one released recall'}
            encoded=owners[condition]*len(gt_ids)+point_gt_index
            unique_inter=np.bincount(encoded,minlength=(len(sizes)+1)*len(gt_ids)).reshape(len(sizes)+1,len(gt_ids))[1:]
            unique_sizes=unique_inter.sum(axis=1);unique_iou=candidate_ious(unique_inter,unique_sizes,gt_sizes)
            extent[condition]=[{'canonical_index':int(i),'candidate_id':doc['ledger'][i]['candidate_id'],
                'before_projected_size':int(sizes[i]),'owned_projected_size':int(unique_sizes[i]),
                'GT_ids_positive_intersection_before':gt_ids[eligible&(intersection[i]>0)].tolist(),
                'GT_ids_positive_intersection_after':gt_ids[eligible&(unique_inter[i]>0)].tolist(),
                'threshold_crossings':{str(t):{'before':gt_ids[eligible&(ious[i]>t)].tolist(),
                    'after':gt_ids[eligible&(unique_iou[i]>t)].tolist()} for t in thresholds},
                'interpretation':'positive multi-GT overlap is an extent diagnostic, not a mutually exclusive overmerge error label'} for i in kept]
            tie_groups={}
            for i in kept:tie_groups.setdefault(doc['rank_scores_serialized'][i],[]).append(int(i))
            ties[condition]={k:v for k,v in tie_groups.items() if len(v)>1}
        for effect,before,after in PAIRS:
            if before not in local or after not in local:
                effects.append({'run':name,'effect':effect,'status':'NOT_RUN','before':before,'after':after});continue
            effects.append({'run':name,'effect':effect,'status':'COMPLETE','before':before,'after':after,
                'delta_proportions':difference(main_metrics(local[before]),main_metrics(local[after]))})
            for representation in ('overlapping','unique'):
                for t in thresholds:
                    old=outcomes[before][representation][str(t)]['matched'];new=outcomes[after][representation][str(t)]['matched']
                    for change,ids in [('gained',set(new)-set(old)),('lost',set(old)-set(new))]:
                        for gid in sorted(ids,key=int):
                            retained=gid in outcomes[after]['unique'][str(t)]['matched']
                            event={'run':name,'effect':effect,'before':before,'after':after,'representation':representation,
                                'threshold':t,'GT_id':int(gid),'transition':change,'old_match':old.get(gid),
                                'new_match':new.get(gid),'matched_after_unique_ownership':retained,
                                'exact_evidence':[local[before][representation]['trace_path'],local[after][representation]['trace_path']]}
                            objects.append(event)
                            category='recovered_object' if change=='gained' else 'lost_object'
                            examples.setdefault(category,event)
            transitions=region_transitions(sem_gt,semantics[before],semantics[after],owners[before],owners[after],regions,valid)
            old_conf=json.loads((ev/before/'regional_confusion.json').read_text());new_conf=json.loads((ev/after/'regional_confusion.json').read_text())
            total_delta=np.zeros((len(valid)+1,len(valid)+1),dtype=np.int64)
            for code,row in transitions.items():
                delta=np.array(new_conf['regions'][code]['confusion'])-np.array(old_conf['regions'][code]['confusion'])
                total_delta+=delta;idx=len(confusion_deltas);confusion_deltas.append(delta)
                regional_rows.append({'run':name,'effect':effect,'before':before,'after':after,'region_id':int(code),
                    **registry[code],**row,'confusion_delta_index':idx})
                if row['same_class_owner_changes']:
                    mask=(regions==int(code))&np.isin(sem_gt,valid)&(semantics[before]==semantics[after])&(owners[before]!=owners[after])
                    examples.setdefault('same_class_ownership_conflict',{'run':name,'before':before,'after':after,
                        'region_id':int(code),'point_indices':np.flatnonzero(mask)[:20].tolist(),'count':int(mask.sum())})
            if not np.array_equal(total_delta,np.array(new_conf['global_confusion'])-np.array(old_conf['global_confusion'])):
                raise ValueError('regional confusion deltas do not sum to global delta')
            if effect=='rank_class_norm' and (not np.array_equal(owners[before],owners[after]) or not np.array_equal(semantics[before],semantics[after])):
                raise ValueError('rank-only control changed map')
            if effect.startswith('assignment_'):
                if docs[before]['kept']!=docs[after]['kept'] or docs[before]['labels']!=docs[after]['labels'] or docs[before]['rank_scores_serialized']!=docs[after]['rank_scores_serialized']:
                    raise ValueError('assignment-only control changed overlapping AP inputs')
        fields=main_metrics(local['AT_U00'])
        interaction={k:main_metrics(local['AT_U11'])[k]-main_metrics(local['AT_U10'])[k]-main_metrics(local['AT_U01'])[k]+fields[k] for k in fields}
        effects.append({'run':name,'effect':'factorial_interaction','status':'COMPLETE','delta_proportions':interaction})
        if {'AT_NATIVE_U11','AT_NATIVE_O_AREA'}<=set(local):
            inter={k:(main_metrics(local['AT_U11'])[k]-main_metrics(local['AT_O_AREA'])[k])-(main_metrics(local['AT_NATIVE_U11'])[k]-main_metrics(local['AT_NATIVE_O_AREA'])[k]) for k in fields}
            effects.append({'run':name,'effect':'readout_fusion_interaction','status':'COMPLETE','delta_proportions':inter})
        borrowing=borrowing_diagnosis(docs['AT_U11']['ledger'],intersection,sizes,gt_ids,gt_sizes,instance_ids,thresholds)
        for row in borrowing:
            for t,event in row['thresholds'].items():
                if event['label_outcome']=='right_to_wrong':examples.setdefault('corrupted_borrowed_label',{'run':name,**row,'threshold':float(t)})
        suppression=[]
        for condition,baseline in [('AT_U01','AT_U00'),('AT_U11','AT_U10'),('AT_NMS_SF_FIRST','AT_U10')]:
            if condition not in docs:continue
            kept=np.array(docs[condition]['kept']);labels=np.array(docs[condition]['labels'])
            candidate_lookup={r['candidate_id']:i for i,r in enumerate(docs[condition]['ledger'])}
            for i,r in enumerate(docs[condition]['ledger']):
                if r['suppressed_by'] is None:continue
                suppressor=candidate_lookup[r['suppressed_by']]
                for t in thresholds:
                    potential=np.flatnonzero(eligible&(ious[i]>t)&(gt_ids//1000==labels[i])) if sizes[i]>=100 else []
                    for gi in potential:
                        alternatives=kept[(sizes[kept]>=100)&(ious[kept,gi]>t)&(labels[kept]==gt_ids[gi]//1000)]
                        gid=str(gt_ids[gi]);lost=gid in outcomes[baseline]['overlapping'][str(t)]['matched'] and gid not in outcomes[condition]['overlapping'][str(t)]['matched']
                        row={'run':name,'condition':condition,'candidate_id':r['candidate_id'],'suppressed_by':r['suppressed_by'],
                            'suppression_iou':r['suppression_iou'],'GT_id':int(gt_ids[gi]),'GT_iou':float(ious[i,gi]),
                            'threshold':t,'suppressor_GT_iou':float(ious[suppressor,gi]),
                            'suppressor_class_correct':bool(labels[suppressor]==gt_ids[gi]//1000),
                            'suppressor_projected_area':int(sizes[suppressor]),
                            'other_retained_covering_candidates':alternatives.tolist(),'actual_released_GT_lost':lost}
                        suppression.append(row)
                        if lost and not len(alternatives):examples.setdefault('harmful_suppression',row)
        # Owner-only policies can change points only where retained U11 multiplicity >=2.
        bank=np.load(pred/'masks.npy',mmap_mode='r');active=np.zeros(bank.shape[1],dtype=np.int32)
        for i in docs['AT_U11']['kept']:active+=bank[i]
        projection=np.load(ev/'projection.npz');active_projected=np.where(projection['matched'],active[projection['nearest']],0)
        for condition in ['AT_ASSIGN_CLASS_NORM','AT_ASSIGN_OVI_FILL']:
            if condition in owners:
                changed=owners[condition]!=owners['AT_U11']
                if np.any(changed&(active_projected<2)):raise ValueError('assignment changed single-candidate point')
        fill_provenance=None
        if 'AT_ASSIGN_OVI_FILL' in docs:
            native_geometry=np.load(pred/'native_owners.npy',mmap_mode='r')
            o_source=np.load(docs['AT_O_AREA']['owner_path'],mmap_mode='r')
            f_source=np.load(docs['AT_ASSIGN_OVI_FILL']['owner_path'],mmap_mode='r')
            filled=(o_source==0)&(f_source>0)
            fill_provenance={'filled_source_points':int(filled.sum()),
                'with_native_geometry_owner_but_no_qualified_OVI_candidate':int((filled&(native_geometry>0)).sum()),
                'without_matched_native_geometry_owner':int((filled&(native_geometry==0)).sum()),
                'interpretation':'lack of a qualified OVI semantic candidate is not unobserved physical space'}
        diagnoses[name]={'OVI_FILL_source_provenance':fill_provenance,'coverage':coverage,'borrowed_labels':borrowing,'suppression':suppression,
            'released_event_counts':event_counts,'candidate_event_diagnostics':event_details,'TP_rank_intervals':tp_ranks,'owned_extent':extent,'rank_score_tie_groups':ties,
            'GT_outcomes':outcomes,'evaluable_GT_ids':all_gt.tolist(),
            'active_U11_multiplicity_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(active_projected,return_counts=True))}}
    repeat_rows=[]
    first,second=[r['id'] for r in config['runs']]
    for condition in sorted(set(diagnoses[first]['GT_outcomes'])&set(diagnoses[second]['GT_outcomes'])):
        for representation in ('overlapping','unique'):
            for t in thresholds:
                a=diagnoses[first]['GT_outcomes'][condition][representation][str(t)]['matched']
                b=diagnoses[second]['GT_outcomes'][condition][representation][str(t)]['matched']
                repeat_rows.append({'condition':condition,'representation':representation,'threshold':t,
                    'common_matched_GT_ids':sorted(map(int,set(a)&set(b))),
                    'primary_only':{k:a[k] for k in sorted(set(a)-set(b),key=int)},
                    'repeat_only':{k:b[k] for k in sorted(set(b)-set(a),key=int)},
                    'comparison_identity':'GT objects, never equal model query indices'})
    if args.geometry_bridge:
        bridge_rows=json.loads((args.geometry_bridge/'performance.json').read_text());performance+=bridge_rows
        for run in config['runs']:
            b={r['condition']:r for r in bridge_rows if r['run']==run['id']}
            for effect,before,after in [('geometry_transfer','AT_GEO_NATIVE','AT_GEO_TRANSFER'),('geometry_rank_area','AT_GEO_TRANSFER','AT_GEO_AREA')]:
                effects.append({'run':run['id'],'effect':effect,'status':'COMPLETE','before':before,'after':after,
                    'delta_proportions':difference(main_metrics(b[before]),main_metrics(b[after])),
                    'semantic_scope':'same frozen historical emitted registry; full semantic-export difference separately reported'})
            core=next(r for r in performance if r['run']==run['id'] and r['condition']=='AT_O_AREA')
            effects.append({'run':run['id'],'effect':'core_registry_eligibility','status':'COMPLETE',
                'before':'AT_GEO_AREA','after':'AT_O_AREA','delta_proportions':difference(main_metrics(b['AT_GEO_AREA']),main_metrics(core))})
        write(args.output/'geometry_export_boundary.json',json.loads((args.geometry_bridge/'registry.json').read_text()))
    write(args.output/'repeat_sensitivity.json',repeat_rows)
    for category in ['recovered_object','lost_object','corrupted_borrowed_label','harmful_suppression','same_class_ownership_conflict']:
        if category not in examples:examples[category]={'status':'NOT_PRESENT_IN_COMPLETED_COMPARISONS'}
    write(args.output/'paired_performance.json',performance);write(args.output/'intervention_effects.json',effects)
    write(args.output/'object_gains_losses.json',objects);write_gzip(args.output/'object_diagnostics.json.gz',diagnoses)
    write(args.output/'regional_losses.json',regional_rows);np.savez_compressed(args.output/'regional_confusion_deltas.npz',deltas=np.stack(confusion_deltas))
    write(args.output/'examples.json',{'selection_rule':'first category occurrence in declared run/pair/representation/threshold order; GT IDs ascending; first 20 vertex indices; no score-based selection','examples':examples})
    write(args.output/'analysis_manifest.json',{'status':'COMPLETE_FOR_AVAILABLE_CONDITIONS','command':sys.argv,
        'conditions_by_run':{r:sorted(diagnoses[r]['coverage']) for r in diagnoses},'seconds':time.perf_counter()-start,
        'repair_decision':'NOT_RUN_PENDING_FULL_DIAGNOSTICS','scope':'paired Room0 development predictions; no CI or independent-scene claim'})
    print(args.output,flush=True)


if __name__=='__main__':main()
