"""Post-lock fixed-support references and released unique-GT/tied-entry attribution."""
from collections import Counter
from pathlib import Path
import time

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.evidence_exploration.analysis import trace_entries, zipped
from static_ovmap.evidence_exploration.diagnostic_details import compare_entries
from static_ovmap.minimal_instance_repair.diagnostics import maximum_matches, support_overlap
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.released_loader import load_released_module

from .binding import load_scene
from .common import METRICS, ConsumptionIndex, canonical_digest, read, verified, write, _array_digest

PAIRS = (('LR05_MA_8','LR04_FC_8'),('LR06_MV_VIEW','LR05_MA_8'),
         ('LR07_MV_SURFACE','LR06_MV_VIEW'),('LR08_MV_AUX','LR07_MV_SURFACE'))


def fixed_support_hash(geometry, rows):
    return canonical_digest(dict(mesh=canonical_digest(geometry.to_dict()),
                                 sorted_source_rows=_array_digest(np.asarray(rows,np.int64))))


def _outcome(old, new, reference):
    if reference is None: return 'UNDEFINED_GT_REFERENCE'
    if old==new: return 'UNCHANGED_RIGHT' if new==reference else 'UNCHANGED_WRONG'
    if old!=reference and new==reference: return 'WRONG_TO_RIGHT'
    if old==reference and new!=reference: return 'RIGHT_TO_WRONG'
    return 'WRONG_TO_WRONG'


def owner_outcomes(decisions, references, threshold):
    refs = {int(r['owner']):r for r in references}; ledger = []; counts = {k:Counter() for k in ('F','final')}
    for owner,row in decisions.items():
        owner = int(owner); item = refs[owner]; best = item['best']
        defined = bool(best and item['instance_size_eligible'] and best['iou']>threshold)
        target = best['class'] if defined else None
        outcomes = dict(F=_outcome(row['old_F_top1'],row['new_F_top1'],target),
                        final=_outcome(row['old_class'],row['applied_class'],target))
        for name,category in outcomes.items(): counts[name][category] += 1
        ledger.append(dict(owner=owner,defined_GT_reference=defined,GT_reference=best if defined else None,
                           closest_eligible_GT=best,outcomes=outcomes,**row))
    return dict(counts={k:dict(v) for k,v in counts.items()},ledger=ledger,
                original_incumbent_denominator=len(ledger),threshold=threshold,
                reference_rule='RELEASED_ELIGIBLE_GT; ORIGINAL_SUPPORT_SIZE>=100; STRICT_IOU>THRESHOLD')


def pairs(methods, proposed):
    return [*PAIRS,*[(m,'LR01_G1') for m in methods if m not in ('LR00_D2','LR01_G1')],
            ('R29_'+proposed,'R29_LR05_MA_8')]


def diagnose_scene(binding, scene, rows, proposed):
    root = Path(binding['output_root']); lock = verified(root/'predictions/summary.json')
    if lock['status']!='ALL_PREDICTIONS_LOCKED' or lock['logical_records']!=286:
        raise ValueError('All outputs must lock before diagnostic annotation reads')
    prediction = verified(root/'predictions'/scene/'receipt.json')
    if prediction['identity']!=lock['scenes'][scene] or len(rows)!=11: raise ValueError('Complete scene evaluation required')
    index = ConsumptionIndex(root/'diagnostics'/scene/'verifications.json'); started = time.perf_counter()
    identity = canonical_digest(dict(lock=prediction['identity'],scores={r['method']:r['identity'] for r in rows},
                                     producer=index.identity(__file__)))
    dest = root/'diagnostics'/scene/'receipt.json'
    if dest.exists():
        old = verified(dest)
        if old['input_identity']!=identity: raise ValueError('Completed diagnostic inputs changed')
        for item in old['dependencies']: index.identity(item['path'],item)
        return old
    _,inputs = load_scene(binding,scene)
    opportunity_path = Path(binding['minimal_root'])/'diagnostics'/scene/'opportunity.json'
    opportunity = verified(opportunity_path); dependencies = [index.identity(opportunity_path)]
    reference = {int(r['owner']):r for r in opportunity['incumbents']}
    decisions = {m:verified(root/'decisions'/scene/(m+'.json'))['owners']
                 for m in prediction['methods'] if m not in ('LR00_D2','LR01_G1')}
    if any(set(map(int,d))!=set(prediction['eligible_owners']) for d in decisions.values()):
        raise ValueError('Eligible owner diagnostic denominator changed')
    for owner in prediction['eligible_owners']:
        rows_source = np.flatnonzero(inputs.d2.owner_ids==owner)
        if (owner not in reference or reference[owner]['support_hash']!=fixed_support_hash(inputs.d2.geometry,rows_source)
                or not np.array_equal(rows_source,np.flatnonzero(inputs.g1.owner_ids==owner))):
            raise ValueError('Original GT reference support changed')
    labels, ranks, traces = {},{},{}; trace_cache = {}
    g1_score = read(next(r for r in rows if r['method']=='LR01_G1')['evaluation_receipt'])
    for row in rows:
        method = row['method']; payload = load_prediction(prediction['methods'][method]['manifest'])
        labels[method] = owner_labels(payload); ranks[method] = dict(payload.instance_ranks)
        if method!='LR00_D2' and (payload.geometry!=inputs.g1.geometry or not np.array_equal(payload.owner_ids,inputs.g1.owner_ids)):
            raise ValueError('Class-agnostic geometry changed during readout')
        score = read(row['evaluation_receipt']); path = Path(score['manifest']).with_name('trace.json.gz')
        if str(path) not in trace_cache:
            dependencies.append(index.identity(path)); raw = zipped(path)
            if not all(raw['parity'][k] for k in ('ap_exact','pr_and_fn_exact')): raise ValueError('Released trace parity absent')
            trace_cache[str(path)] = trace_entries(raw)
        traces[method] = trace_cache[str(path)]
    # Recompute class-agnostic unique matching from the full G1 positive registry.
    # Every learned payload has exactly these source supports and projection.
    gt_path = Path(g1_score['gt_path']); dependencies.append(index.identity(gt_path,g1_score['context']['gt']))
    gt = np.load(gt_path,allow_pickle=False).reshape(-1)
    matched_path = Path(g1_score['manifest']).with_name('matches.json.gz'); dependencies.append(index.identity(matched_path))
    matches = next(iter(zipped(matched_path).values()))
    namespace = load_released_module(g1_score['context']['evaluator']['path'])
    namespace['init']('Replica' if binding['scenes'][scene]['cohort']=='replica8' else 'Scannet200')
    distance,confidence = namespace['dist_threshes'][0],namespace['dist_confs'][0]
    eligible = {int(g['instance_id']):g for values in matches['gt'].values() for g in values
                if g['instance_id']>=1000 and g['vert_count']>=100 and g['med_dist']<=distance and g['dist_conf']>=confidence}
    supports = [dict(name='G1:'+str(o),**support_overlap(np.flatnonzero(inputs.g1.owner_ids==o),
                inputs.nearest,inputs.matched,gt,eligible)) for o in prediction['full_G1_registry']]
    geometry = {str(t):maximum_matches(supports,eligible,t) for t in (.5,.75)}
    outcomes = {m:{str(t):owner_outcomes(d,opportunity['incumbents'],t) for t in (.5,.75)} for m,d in decisions.items()}
    comparisons = []
    for a,b in pairs(list(prediction['methods']),proposed):
        semantic = {}; released = {}
        for t in (.5,.75):
            ledger = {o:{'old_class':labels[b][int(o)],'applied_class':labels[a][int(o)],
                         'old_F_top1':decisions[b][o]['new_F_top1'] if b in decisions else decisions[a][o]['old_F_top1'],
                         'new_F_top1':decisions[a][o]['new_F_top1']} for o in decisions[a]}
            semantic[str(t)] = owner_outcomes(ledger,opportunity['incumbents'],t)
            released[str(t)] = {k.removesuffix('50'):v for k,v in compare_entries(
                [x for x in traces[a] if abs(x['overlap_threshold']-t)<1e-12],
                [x for x in traces[b] if abs(x['overlap_threshold']-t)<1e-12]).items()}
        comparisons.append(dict(candidate=a,reference=b,semantic_outcomes=semantic,released_matches=released,
                      rank_changes=[dict(owner=o,old=ranks[b][o],new=ranks[a][o]) for o in ranks[a] if ranks[a][o]!=ranks[b][o]],
                      output_identity_tie=prediction['methods'][a]['prediction_key']==prediction['methods'][b]['prediction_key']))
    result = write(dest,dict(status='COMPLETE',scene=scene,input_identity=identity,owner_outcomes=outcomes,
                   eligible_original_objects=len(prediction['eligible_owners']),comparisons=comparisons,
                   class_agnostic_matches=geometry,class_agnostic_matches_identical_for_all_fixed_G1_methods=True,
                   unchanged_source_partition=_array_digest(inputs.g1.owner_ids),dependencies=dependencies,
                   GT_after_global_lock=lock['identity'],elapsed_seconds=time.perf_counter()-started))
    index.write_memo(root/'diagnostics'/scene/'verifications.json')
    return result


def diagnose(binding, store):
    root = Path(binding['output_root']); started = time.perf_counter()
    proposed = verified(root/'dev_nomination.json')['proposed_architecture']
    scenes = [diagnose_scene(binding,s,[r for r in store['scene_metrics'] if r['scene']==s],proposed)
              for names in binding['cohorts'].values() for s in names]
    comparisons = []
    for cohort,names in binding['cohorts'].items():
        records = [r for r in scenes if r['scene'] in names]
        for a,b in pairs(store['methods'],proposed):
            details = [(r['scene'],next(d for d in r['comparisons'] if (d['candidate'],d['reference'])==(a,b))) for r in records]
            outcomes,matched = {},{}
            for t in ('0.5','0.75'):
                outcomes[t] = {stream:dict(sum((Counter(d['semantic_outcomes'][t]['counts'][stream]) for _,d in details),Counter()))
                               for stream in ('F','final')}
                counts = Counter(); gained,lost = [],[]
                for scene,d in details:
                    match = d['released_matches'][t]
                    counts.update({k:v for k,v in match.items() if isinstance(v,int)})
                    gained.extend(dict(scene=scene,class_and_GT_id=g) for g in match['new_unique_GT_ids'])
                    lost.extend(dict(scene=scene,class_and_GT_id=g) for g in match['lost_unique_GT_ids'])
                matched[t] = dict(counts=dict(counts),gained=gained,lost=lost)
            comparisons.append(dict(cohort=cohort,candidate=a,reference=b,full_cohort=len(details)==len(names),
                        original_incumbent_denominator=sum(d['semantic_outcomes']['0.5']['original_incumbent_denominator'] for _,d in details),
                        semantic_outcomes=outcomes,released_matches=matched,
                        rank_changes=sum(len(d['rank_changes']) for _,d in details),
                        metric_deltas_fraction={k:store['pooled_metrics'][cohort][a]['metrics'][k]-store['pooled_metrics'][cohort][b]['metrics'][k] for k in METRICS}))
    return write(root/'diagnostics/summary.json',dict(status='DIAGNOSTICS_COMPLETE',scene_receipts={r['scene']:r['identity'] for r in scenes},
                 comparisons=comparisons,class_agnostic_matches_unchanged=True,GT_used_only_after_global_lock=True,
                 undefined_GT_reference_is_retained_in_denominator=True,elapsed_seconds=time.perf_counter()-started))
