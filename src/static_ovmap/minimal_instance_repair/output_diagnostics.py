"""Post-lock class-agnostic geometry and released class-aware score accounting.

This module is deliberately absent from all prediction-side imports. GT IDs are
always qualified by scene; duplicate score entries retain their multiplicity.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.evidence_exploration.analysis import trace_entries, zipped
from static_ovmap.evidence_exploration.diagnostic_details import compare_entries
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .binding import load_scene, seal
from .diagnostics import eligible_targets, maximum_matches


PAIRS = [('IR03_EVIDENCE_ATTACH','IR02_NEAREST_ATTACH'),
    ('IR05_VERIFIED_REPAIR','IR04_DIRECT_GROUP'),('IR07_BOUNDARY_STABLE','IR06_ANYUP_REREAD'),
    ('IR08_COMBINATION','IR05_VERIFIED_REPAIR'),('IR08_COMBINATION','IR07_BOUNDARY_STABLE')]


def describe_partition(payload, inputs, gt, eligible, minimum):
    projected = np.zeros(len(gt),np.int64)
    projected[inputs.matched] = payload.owner_ids[inputs.nearest[inputs.matched]]
    owners,areas = np.unique(projected[projected>0],return_counts=True)
    gids,sizes = np.unique(gt,return_counts=True)
    gt_sizes = dict(zip(map(int,gids),map(int,sizes),strict=True))
    positive = (projected>0)&np.isin(gt,list(eligible))
    pairs,counts = np.unique(np.stack((projected[positive],gt[positive]),axis=1),axis=0,return_counts=True)
    intersections = {}
    for (owner,gid),n in zip(pairs,counts,strict=True):
        intersections.setdefault(int(owner),[]).append((int(gid),int(n)))
    results = []
    for owner,area in zip(owners,areas,strict=True):
        owner,area = int(owner),int(area)
        overlaps = [{'gt_id':g,'class':int(eligible[g]['label_id']),
            'intersection':n,'iou':n/(area+gt_sizes[g]-n)} for g,n in intersections.get(owner,[])]
        overlaps.sort(key=lambda r:(-r['iou'],r['gt_id']))
        results.append({'name':'OWNER:'+str(owner),'owner':owner,'evaluation_points':area,
            'instance_size_eligible':area>=minimum,'overlaps':overlaps,'best':overlaps[0] if overlaps else None,
            'intersected_eligible_GT_count':len(overlaps)})
    return results


def semantic_transition_counts(a, b, original):
    counts = Counter()
    details = []
    for item in original:
        owner = item['owner']
        if owner not in a or owner not in b or a[owner]==b[owner]:
            continue
        matchable = item['geometrically_matchable50']
        best = item['best']
        before = bool(best and b[owner]==best['class'])
        after = bool(best and a[owner]==best['class'])
        kind = 'matchable' if matchable else 'geometry_insufficient'
        counts[kind+'_changed'] += 1
        counts[kind+'_wrong_to_right'] += int(not before and after)
        counts[kind+'_right_to_wrong'] += int(before and not after)
        details.append({'owner':owner,'reference_class':b[owner],'candidate_class':a[owner],
            'original_P_matchable50':matchable,'reference_correct':before,'candidate_correct':after})
    return dict(counts),details


def output_scene(job):
    binding,scene,rows = job
    root = Path(binding['output_root'])
    lock = read(root/'predictions'/scene/'receipt.json')
    key = canonical_digest({'lock':lock['identity'],'evaluations':{r['method']:r['identity'] for r in rows},
        'producer':ConsumptionIndex().identity(__file__)})
    dest = root/'diagnostics'/scene/'output_comparisons.json'
    if dest.exists() and read(dest)['input_identity']==key:
        return read(dest)
    inputs = load_scene(binding,scene)
    gt,eligible,minimum,_,_ = eligible_targets(binding,inputs)
    opportunity = read(root/'diagnostics'/scene/'opportunity.json')
    methods = [r['id'] for r in binding['specification']['methods']]
    geometry,labels,traces,partitions = {},{},{},{}
    thresholds = rows[0]['runtime_overlaps']
    thresholds = list(map(float,thresholds))
    comparisons = PAIRS+[(m,b) for m in methods for b in ('IR01_G1','IR00_D2')]
    for row in rows:
        method = row['method']
        payload = load_prediction(lock['methods'][method]['manifest'])
        partition = _array_digest(payload.owner_ids)
        if partition not in partitions:
            supports = describe_partition(payload,inputs,gt,eligible,minimum)
            partitions[partition] = {'supports':supports,'matching':{str(t):maximum_matches(supports,eligible,t) for t in thresholds}}
        geometry[method],labels[method] = partitions[partition],owner_labels(payload)
        score = read(row['evaluation_receipt'])
        path = str(Path(score['manifest']).with_name('trace.json.gz'))
        if path not in traces:
            traces[path] = trace_entries(zipped(path))
        labels[method] = {'classes':labels[method],'entries':traces[path],'ranks':dict(payload.instance_ranks)}
    details = []
    for candidate,reference in comparisons:
        a,b = geometry[candidate],geometry[reference]
        values = []
        for t in thresholds:
            ga,gb = set(a['matching'][str(t)]['gt_ids']),set(b['matching'][str(t)]['gt_ids'])
            ea,eb = ([r for r in labels[n]['entries'] if abs(r['overlap_threshold']-t)<1e-12] for n in (candidate,reference))
            score = compare_entries(ea,eb)
            # The reused helper's legacy suffix is renamed for every threshold.
            score = {k.removesuffix('50'):v for k,v in score.items()}
            values.append({'threshold':t,'class_agnostic':{'new_unique_GT_ids':sorted(ga-gb),
                'lost_unique_GT_ids':sorted(gb-ga),'new_unique_GT_matches':len(ga-gb),'lost_unique_GT_matches':len(gb-ga)},
                'released_class_aware':score})
        by_owner_a = {r['owner']:r for r in a['supports']}
        by_owner_b = {r['owner']:r for r in b['supports']}
        ious,ranks = [],[]
        for owner in sorted(by_owner_a.keys()&by_owner_b.keys()):
            aa,bb = by_owner_a[owner],by_owner_b[owner]
            va,vb = (x['best']['iou'] if x['best'] else 0. for x in (aa,bb))
            if va!=vb or aa['evaluation_points']!=bb['evaluation_points']:
                ious.append({'owner':owner,'candidate_best':aa['best'],'reference_best':bb['best'],
                    'best_IoU_delta':va-vb,'candidate_intersected_GT_count':aa['intersected_eligible_GT_count'],
                    'reference_intersected_GT_count':bb['intersected_eligible_GT_count'],
                    'overmerge_indicator_more_GT_intersections_and_lower_best_IoU':
                        aa['intersected_eligible_GT_count']>bb['intersected_eligible_GT_count'] and va<vb})
        for owner in sorted(labels[candidate]['ranks'].keys()&labels[reference]['ranks'].keys()):
            va,vb = labels[candidate]['ranks'][owner],labels[reference]['ranks'][owner]
            if va!=vb:
                ranks.append({'owner':owner,'reference':vb,'candidate':va,'delta':va-vb,
                              'released_reference_six_decimal':f'{vb:.6f}','released_candidate_six_decimal':f'{va:.6f}'})
        semantic,semantic_rows = semantic_transition_counts(labels[candidate]['classes'],labels[reference]['classes'],opportunity['incumbents'])
        details.append({'scene':scene,'cohort':binding['scenes'][scene]['cohort'],'candidate':candidate,'reference':reference,
            'thresholds':values,'existing_owner_IoU_changes':ious,'official_rank_changes':ranks,
            'original_P_semantic_changes':semantic,'semantic_transition_ledger':semantic_rows,
            'candidate_audit':lock['methods'][candidate].get('audit',{}),
            'reference_audit':lock['methods'][reference].get('audit',{})})
    result = seal({'status':'POSTLOCK_OUTPUT_DIAGNOSTICS_COMPLETE','scene':scene,'input_identity':key,
        'thresholds':thresholds,'comparisons':details,'labels_used_after_all_predictions_locked':True,
        'unique_GT_namespace':scene,'duplicate_score_multiplicity_preserved':True})
    atomic_write_json(dest,result)
    print('OUTPUT_DIAGNOSTICS',scene,len(details),flush=True)
    return result


def aggregate_comparisons(details, store):
    result = []
    for cohort,names in store['cohorts'].items():
        pairs = list(dict.fromkeys((d['candidate'],d['reference']) for d in details))
        for candidate,reference in pairs:
            scenes = [d for d in details if d['scene'] in names and (d['candidate'],d['reference'])==(candidate,reference)]
            threshold_rows = []
            for t in scenes[0]['thresholds']:
                threshold = t['threshold']
                counts = {'class_agnostic':Counter(),'released_class_aware':Counter()}
                ids = {'class_agnostic':{'new':[],'lost':[]},'released_class_aware':{'new':[],'lost':[]}}
                for scene in scenes:
                    row = next(x for x in scene['thresholds'] if x['threshold']==threshold)
                    for kind in counts:
                        counts[kind].update({k:v for k,v in row[kind].items() if isinstance(v,int)})
                        for action in ('new','lost'):
                            ids[kind][action].extend({'scene':scene['scene'],'gt_id':gid} for gid in row[kind][action+'_unique_GT_ids'])
                threshold_rows.append({'threshold':threshold,**{k:{'counts':dict(counts[k]),'qualified_GT_sets':ids[k]} for k in counts}})
            semantic = Counter()
            operations = Counter()
            for scene in scenes:
                semantic.update(scene['original_P_semantic_changes'])
                for role in ('candidate','reference'):
                    audit = scene[role+'_audit']
                    operations.update({role+'_applied_operations':len(audit.get('applied',[])),
                        role+'_canceled_unions':len(audit.get('canceled',[])),
                        role+'_class_changes':len(audit.get('class_transitions',[])),
                        role+'_moved_residual_rows':audit.get('moved_residual_rows',0),
                        role+'_moved_incumbent_rows':audit.get('moved_incumbent_rows',0)})
            a,b = (store['pooled_metrics'][cohort][n]['metrics'] for n in (candidate,reference))
            result.append({'cohort':cohort,'candidate':candidate,'reference':reference,
                'metric_deltas_fraction':{k:a[k]-b[k] for k in a},'thresholds':threshold_rows,
                'operations':dict(operations),'original_P_semantic_changes':dict(semantic),
                'overmerge_indicators':sum(x['overmerge_indicator_more_GT_intersections_and_lower_best_IoU']
                    for d in scenes for x in d['existing_owner_IoU_changes']),
                'rank_changes':sum(len(d['official_rank_changes']) for d in scenes),
                'existing_owner_IoU_changes':sum(len(d['existing_owner_IoU_changes']) for d in scenes)})
    return result


def analyze_outputs(binding, store):
    scenes = [s for names in binding['cohorts'].values() for s in names]
    jobs = [(binding,s,[r for r in store['scene_metrics'] if r['scene']==s]) for s in scenes]
    with ProcessPoolExecutor(max_workers=binding['specification']['resources']['CPU_evaluation_workers']) as executor:
        receipts = list(executor.map(output_scene,jobs))
    details = [d for r in receipts for d in r['comparisons']]
    result = seal({'status':'POSTLOCK_OUTPUT_DIAGNOSTICS_COMPLETE','comparisons':aggregate_comparisons(details,store),
        'scene_receipts':{r['scene']:r['identity'] for r in receipts},'scene_count':len(receipts),
        'named_GT_sets_qualified_by_scene':True,'class_agnostic_is_not_released_AP':True,
        'overmerge_indicator_definition':'MORE_ELIGIBLE_GT_INTERSECTIONS_AND_LOWER_BEST_IOU_THAN_REFERENCE',
        'source_science_identity':store.get('science_identity',store['identity'])})
    atomic_write_json(Path(binding['output_root'])/'diagnostics/output_summary.json',result)
    return result
