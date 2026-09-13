"""Evaluation-only object correspondences; released events and coverage stay distinct."""
import numpy as np


def released_object_outcomes(trace, nominal_threshold):
    thresholds=sorted({s['overlap_threshold'] for s in trace['states']})
    selected=[x for x in thresholds if np.isclose(x,nominal_threshold)]
    if len(selected)!=1:raise ValueError('one actual runtime threshold must match nominal threshold')
    threshold=selected[0];matched={};hard_fn=[]
    for event in trace['events']:
        if event['overlap_threshold']!=threshold:continue
        kind=event['event']
        if kind=='first_match':
            matched[str(event['gt_id'])]={'first_match_candidate':event['candidate_file'],
                'tp_score_owner':event['candidate_file'],'tp_score':event['confidence'],
                'first_match_overlap':event['overlap'],'duplicate_events':[]}
        elif kind=='duplicate':
            key=str(event['gt_id'])
            if key not in matched:raise ValueError('duplicate event lacks its original GT match')
            matched[key]['duplicate_events'].append(event)
            matched[key]['tp_score_owner']=event['tp_score_owner']
            matched[key]['tp_score']=event['tp_score']
        elif kind=='hard_fn':hard_fn.append(int(event['gt_id']))
    return {'threshold':threshold,'matched':matched,'hard_fn_gt_ids':hard_fn,
        'definition':'actual released first-match/FN events; duplicate max-score owner separately linked'}


def candidate_ious(intersections,predicted_sizes,gt_sizes):
    intersections,predicted_sizes,gt_sizes=map(np.asarray,(intersections,predicted_sizes,gt_sizes))
    if intersections.shape!=(len(predicted_sizes),len(gt_sizes)):
        raise ValueError('aligned candidate-GT intersection matrix required')
    union=predicted_sizes[:,None]+gt_sizes[None,:]-intersections
    return np.divide(intersections,union,out=np.zeros(intersections.shape,dtype=float),where=union>0)


def borrowing_diagnosis(records,intersections,projected_sizes,gt_ids,gt_sizes,instance_ids,thresholds):
    gt_ids,gt_sizes=np.asarray(gt_ids),np.asarray(gt_sizes)
    eligible=(gt_ids>=1000)&(gt_sizes>=100)&np.isin(gt_ids//1000,instance_ids)
    ious=candidate_ious(intersections,projected_sizes,gt_sizes)
    rows=[]
    for i,r in enumerate(records):
        if r.get('borrowed_from') is None:continue
        old,new=int(r['original_class_id']),int(r['final_class_id'])
        old_in,new_in=old in instance_ids,new in instance_ids
        subset='entry' if new_in and not old_in else 'exit' if old_in and not new_in else 'inside' if old_in else 'outside'
        row={'canonical_index':i,'candidate_id':r.get('candidate_id'), 'kept':bool(r['kept']),
            'borrowed_from':r['borrowed_from'],'old_class':old,'new_class':new,
            'same_label':old==new,'instance_class_transition':subset,'thresholds':{}}
        for threshold in thresholds:
            candidates=np.flatnonzero(eligible&(ious[i]>threshold))
            correspondence='unique' if len(candidates)==1 else 'ambiguous' if len(candidates)>1 else 'unmatched'
            old_correct=new_correct=None
            if len(candidates)==1:
                gt_class=int(gt_ids[candidates[0]]//1000)
                old_correct,new_correct=old==gt_class,new==gt_class
            if old==new:outcome='same_label'
            elif correspondence!='unique':outcome=correspondence
            else:outcome=('right' if old_correct else 'wrong')+'_to_'+('right' if new_correct else 'wrong')
            row['thresholds'][str(float(threshold))]={'label_outcome':outcome,
                'correspondence':correspondence,'geometry_GT_ids':gt_ids[candidates].tolist(),
                'geometry_IoUs':ious[i,candidates].tolist(),'old_correct':old_correct,'new_correct':new_correct}
        rows.append(row)
    return rows


def candidate_event_diagnostics(trace,intersections,projected_sizes,labels,gt_ids,gt_sizes,instance_ids):
    """Link actual released events to explicitly nonexclusive geometric error flags."""
    from pathlib import Path
    import re
    gt_ids,gt_sizes,labels=map(np.asarray,(gt_ids,gt_sizes,labels))
    eligible=(gt_ids>=1000)&(gt_sizes>=100)&np.isin(gt_ids//1000,instance_ids)
    ious=candidate_ious(intersections,projected_sizes,gt_sizes)
    rows=[]
    for event_index,event in enumerate(trace['events']):
        if event['event']=='hard_fn':continue
        filename=event.get('fp_score_owner') if event['event']=='duplicate' else event.get('candidate_file')
        if filename is None:continue
        match=re.fullmatch(r'candidate_(\d+)\.npy',Path(filename).name)
        if match is None:raise ValueError('candidate trace filename has no canonical identity')
        index=int(match.group(1));flags=[]
        kind=event['event'];threshold=event['overlap_threshold']
        if kind=='first_match':flags.append('released_first_match')
        elif kind=='duplicate':flags.append('released_duplicate_min_score_fp')
        elif kind=='ignore_test':flags.append('counted_unmatched_fp' if event['counted_fp'] else 'evaluator_ignored')
        positive=eligible&(intersections[index]>0)
        if not positive.any():flags.append('background_diagnostic')
        if positive.sum()>1:flags.append('multiple_evaluable_GT_positive_intersections')
        if kind=='ignore_test' and event['counted_fp'] and positive.any():
            if not np.any(eligible&(ious[index]>threshold)):flags.append('poor_localization_diagnostic')
            elif not np.any(eligible&(ious[index]>threshold)&(gt_ids//1000==labels[index])):flags.append('class_mismatch_at_geometry_coverage')
        if not np.any(eligible&(gt_ids//1000==labels[index])):flags.append('class_has_no_evaluable_GT_AP_null')
        rows.append({'event_index':event_index,'threshold':threshold,'canonical_index':index,
            'candidate_file':filename,'event':kind,'flags':flags,'evaluable_GT_positive_intersections':gt_ids[positive].tolist(),
            'definition':'event-linked nonexclusive flags; background=no overlap with any evaluable GT; poor localization=no strict-threshold coverage; not additive error taxonomy'})
    return rows
