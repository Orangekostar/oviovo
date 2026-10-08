"""Evaluation-only opportunity and output diagnostics, never predictor inputs."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.evidence_exploration.analysis import trace_entries, zipped
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.binding import read
from static_ovmap.released_loader import load_released_module

from .binding import load_scene, seal
from .reread import select_incumbents


def support_overlap(rows, nearest, matched, gt, eligible, *, minimum=100):
    """Project an arbitrary fixed support using the original target correspondence."""
    rows, nearest, matched, gt = map(np.asarray,(rows,nearest,matched,gt))
    if nearest.shape != matched.shape or gt.shape != nearest.shape:
        raise ValueError('support diagnostic changed original evaluation coordinates')
    mask = matched & np.isin(nearest,rows)
    area = int(mask.sum())
    gids,counts = np.unique(gt,return_counts=True)
    sizes = dict(zip(map(int,gids),map(int,counts),strict=True))
    overlaps = []
    ignored = 0
    for gid,n in zip(*np.unique(gt[mask],return_counts=True),strict=True):
        gid,n = int(gid),int(n)
        if gid not in eligible:
            ignored += n
            continue
        overlaps.append({'gt_id':gid,'class':int(eligible[gid]['label_id']),
                         'intersection':n,'iou':n/(area+sizes[gid]-n)})
    overlaps.sort(key=lambda r:(-r['iou'],r['gt_id']))
    return {'evaluation_points':area,'instance_size_eligible':area>=minimum,
            'ignored_group_void_small_points':ignored,'overlaps':overlaps,
            'best':overlaps[0] if overlaps else None}


def maximum_matches(supports, eligible, threshold):
    """Relaxed unique-GT matching; incompatible proposals are explicitly allowed."""
    names = [r['name'] for r in supports]
    if len(set(names)) != len(names):
        raise ValueError('matching requires uniquely identified fixed supports')
    gids = sorted(eligible)
    positions = {gid:i for i,gid in enumerate(gids)}
    rows,cols = [],[]
    for i,item in enumerate(supports):
        if item['instance_size_eligible']:
            for match in item['overlaps']:
                if match['iou'] > threshold:
                    rows.append(i)
                    cols.append(positions[match['gt_id']])
    graph = csr_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(len(supports),len(gids)))
    matching = maximum_bipartite_matching(graph,perm_type='column')
    pairs = [{'support':names[i],'gt_id':gids[j]} for i,j in enumerate(matching) if j>=0]
    return {'count':len(pairs),'gt_ids':sorted(r['gt_id'] for r in pairs),
            'pairs':pairs,'threshold':threshold,'comparison':'STRICT_GREATER',
            'support_count':len(supports),'eligible_GT_count':len(gids)}


def eligible_targets(binding, inputs):
    """Read only the completed original released evaluator's actual GT eligibility."""
    row = binding['baseline_rows']['IR01_G1'][inputs.scene]
    score = read(row['evaluation_receipt'])
    if score['status']!='COMPLETE' or score['identity']!=row['evaluation_identity']:
        raise ValueError('diagnosis requires locked completed baseline scoring')
    inputs.index.identity(score['gt_path'],score['context']['gt'])
    manifest = Path(score['manifest'])
    matches_path,trace_path = manifest.with_name('matches.json.gz'),manifest.with_name('trace.json.gz')
    inputs.index.identity(matches_path)
    inputs.index.identity(trace_path)
    namespace = load_released_module(score['context']['evaluator']['path'])
    namespace['init']('Replica' if inputs.data['dataset']=='Replica' else 'Scannet200')
    minimum = int(namespace['min_region_sizes'][0])
    distance,confidence = namespace['dist_threshes'][0],namespace['dist_confs'][0]
    matches = next(iter(zipped(matches_path).values()))
    eligible = {int(g['instance_id']):g for values in matches['gt'].values() for g in values
        if g['instance_id']>=1000 and g['vert_count']>=minimum
        and g['med_dist']<=distance and g['dist_conf']>=confidence}
    gt = np.load(score['gt_path'],allow_pickle=False).reshape(-1)
    trace = zipped(trace_path)
    if not all(trace['parity'][k] for k in ('ap_exact','pr_and_fn_exact')):
        raise ValueError('diagnostic baseline trace lacks released-scoring parity')
    return gt,eligible,minimum,score,trace


def diagnose_scene(job):
    binding,scene = job
    root = Path(binding['output_root'])
    proposal = read(root/'proposals'/scene/'receipt.json')
    if proposal['status']!='PROVISIONAL_STRUCTURE_LOCKED' or proposal['class_scores_read']:
        raise ValueError('opportunity diagnosis requires the fixed GT-free proposal library')
    inputs = load_scene(binding,scene)
    key = canonical_digest({'proposal':proposal['identity'],'support':inputs.units.identity,
        'scoring':binding['baseline_rows']['IR01_G1'][scene]['evaluation_identity'],
        'producer':inputs.index.identity(__file__)})
    dest = root/'diagnostics'/scene/'opportunity.json'
    if dest.exists():
        previous = read(dest)
        if previous['input_identity']!=key:
            raise ValueError('changed diagnostic leaf requires scoped descendant invalidation')
        return previous
    begin = time.perf_counter()
    gt,eligible,minimum,score,trace = eligible_targets(binding,inputs)
    def describe(rows):
        return support_overlap(rows,inputs.nearest,inputs.matched,gt,eligible,minimum=minimum)
    baseline_entries = trace_entries(trace)
    incumbents = []
    old_labels = owner_labels(inputs.d2)
    selected = {row['owner'] for row in select_incumbents(inputs.units,inputs.probabilities,
                         limit=binding['specification']['reread']['max_incumbents'])}
    for name,unit in inputs.units.units.items():
        if unit.kind!='I':
            continue
        described = describe(unit.rows)
        best = described['best']
        incumbents.append({'name':name,'owner':unit.owner,'source_rows':len(unit.rows),
            'physical_area':unit.area,'support_hash':unit.support_hash,**described,
            'D2_class':old_labels[unit.owner],'selected_low_margin':unit.owner in selected,
            'geometrically_matchable50':bool(best and best['iou']>.5 and described['instance_size_eligible']),
            'best_GT_class_correct':bool(best and best['class']==old_labels[unit.owner]) if best else None})
    geom_incumbent_gt = {m['gt_id'] for r in incumbents if r['instance_size_eligible']
                         for m in r['overlaps'] if m['iou']>.5}
    official_incumbent_gt = {e['gt_id'] for e in baseline_entries if e['kind']=='TP'
        and abs(e['overlap_threshold']-.5)<1e-12 and e['owner'] in old_labels}
    hypotheses = {h['digest']:h for h in proposal['library']}
    # The locked attachment universe is formed before diagnosis, even when not
    # selected by either attachment arm. Full R is not part of this deployable set.
    for seed,neighbors in proposal['directed_neighbors'].items():
        for item in neighbors:
            if inputs.units.units[item['unit']].kind!='I':
                continue
            names = sorted([seed,item['unit']])
            digest = canonical_digest({'units':names,'supports':[inputs.units.units[n].support_hash for n in names]})
            hypotheses.setdefault(digest,{'digest':digest,'units':names,'host':inputs.units.units[item['unit']].owner,
                                         'origin':'LOCKED_SPATIAL_ATTACHMENT_UNIVERSE'})
    groups = []
    for digest,h in sorted(hypotheses.items()):
        rows = np.unique(np.concatenate([inputs.units.units[n].rows for n in h['units']]))
        groups.append({'name':'H:'+digest,'hypothesis_digest':digest,'units':h['units'],
            'source_rows':len(rows),'physical_area':float(inputs.units.vertex_weights[rows].sum()),
            'in_common_positive_library':any(x['digest']==digest for x in proposal['library']),**describe(rows)})
    recovered = read(binding['scenes'][scene]['G1_source']['path'])['objects']
    residuals = []
    for name,unit in inputs.units.units.items():
        if unit.kind!='C':
            continue
        raw_rows = np.flatnonzero(inputs.raw==unit.owner)
        full,clipped = describe(raw_rows),describe(unit.rows)
        rbest,kbest = full['best'],clipped['best']
        actual = recovered.get(str(unit.owner),{})
        residuals.append({'name':name,'owner':unit.owner,'source_rows_R':len(raw_rows),
            'source_rows_K':len(unit.rows),'physical_area_R':float(inputs.units.vertex_weights[raw_rows].sum()),
            'physical_area_K':unit.area,'clipped_fraction_rows':1-len(unit.rows)/len(raw_rows),
            'support_hash_K':unit.support_hash,'source_rows_R_digest':_array_digest(raw_rows),
            'selected_seed':name in inputs.units.seeds,'R':full,'K':clipped,
            'containing_hypotheses':[g['hypothesis_digest'] for g in groups if name in g['units']],
            'best_R_GT_incumbent_geometry_matchable50':bool(rbest and rbest['gt_id'] in geom_incumbent_gt),
            'best_R_GT_incumbent_released_matched50':bool(rbest and rbest['gt_id'] in official_incumbent_gt),
            'K_geometric_insufficiency50':not bool(kbest and kbest['iou']>.5 and clipped['instance_size_eligible']),
            'R_geometric_insufficiency50':not bool(rbest and rbest['iou']>.5 and full['instance_size_eligible']),
            'clipping_loses_matchable50':bool(rbest and rbest['iou']>.5 and full['instance_size_eligible']
                  and not (kbest and kbest['iou']>.5 and clipped['instance_size_eligible'])),
            'G1_available':bool(actual.get('available',False)),'G1_class':actual.get('label'),
            'baseline_actual_score_entries':[e for e in baseline_entries if e['owner']==unit.owner]})
    current = [{'name':'G1:'+str(o),**describe(np.flatnonzero(inputs.g1.owner_ids==o))}
               for o in sorted(owner_labels(inputs.g1))]
    relaxed = [*incumbents,*[{'name':r['name'],**r['K']} for r in residuals],*groups]
    thresholds = (.25,.5,.75,.9)
    ceilings = {'current_G1_partition':{str(t):maximum_matches(current,eligible,t) for t in thresholds},
        'relaxed_incumbents_residuals_locked_hypotheses':{str(t):maximum_matches(relaxed,eligible,t) for t in thresholds},
        'common_positive_group_library_only':{str(t):maximum_matches(
            [r for r in groups if r['in_common_positive_library']],eligible,t) for t in thresholds}}
    result = seal({'status':'COMPLETE','scene':scene,'cohort':binding['scenes'][scene]['cohort'],
        'input_identity':key,'proposal_identity':proposal['identity'],'GT_evaluation_only':True,
        'predictor_may_read_diagnostic':False,'eligible_GT_ids':sorted(eligible),
        'eligible_GT_count':len(eligible),'incumbents':incumbents,'residuals':residuals,'hypotheses':groups,
        'matching':ceilings,'relaxed_matching_is_AP_bound':False,
        'relaxed_matching_may_use_incompatible_overlapping_hypotheses':True,
        'R_edits_requiring_incumbent_transfer_are_outside_scope':True,
        'baseline_ignored_events':[e for e in trace['events'] if e['event']=='ignore_test'],
        'baseline_ambiguous_score_entries':[e for e in baseline_entries if e['ambiguous_tie']],
        'summary':{'residuals':len(residuals),'incumbents':len(incumbents),
            'K_insufficient50':sum(r['K_geometric_insufficiency50'] for r in residuals),
            'R_insufficient50':sum(r['R_geometric_insufficiency50'] for r in residuals),
            'clipping_loses_matchable50':sum(r['clipping_loses_matchable50'] for r in residuals),
            'selected_incumbents_matchable50':sum(r['selected_low_margin'] and r['geometrically_matchable50'] for r in incumbents),
            'selected_incumbents_matchable50_wrong':sum(r['selected_low_margin'] and r['geometrically_matchable50']
                                                          and not r['best_GT_class_correct'] for r in incumbents)},
        'baseline_scoring_identity':score['identity'],'elapsed_seconds':time.perf_counter()-begin,
        'new_neural_inference':0})
    atomic_write_json(dest,result)
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    print('DIAGNOSED',scene,result['summary'],flush=True)
    return result


def diagnose_study(binding):
    scenes = [s for names in binding['cohorts'].values() for s in names]
    with ProcessPoolExecutor(max_workers=binding['specification']['resources']['CPU_evaluation_workers']) as executor:
        records = list(executor.map(diagnose_scene,[(binding,s) for s in scenes]))
    summary = seal({'status':'COMPLETE','scenes':{r['scene']:r['identity'] for r in records},
        'scene_count':len(records),'cohorts':{c:dict(sum((Counter(r['summary']) for r in records
             if r['cohort']==c),Counter())) for c in binding['cohorts']},
        'fixed_proposals_before_GT':True,'new_neural_inference':0})
    atomic_write_json(Path(binding['output_root'])/'diagnostics/summary.json',summary)
    return summary
