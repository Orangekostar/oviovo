"""Evaluation-only outcomes joined after complete-map prediction locks."""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax

from src.static_ovmap.a7_evidence_upgrade.costs import summarize_operations
from src.static_ovmap.a7_evidence_upgrade.operations import method_operations
from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.m2_reviewer_study.attribution import read_trace
from src.static_ovmap.m2_reviewer_study.evaluation import write_gzip

from .binding import INDEX
from .binding import read_json_or_gzip as read
from .evaluation import RANKS, configurations, pool_methods
from .evidence import B

METRICS = ('apall', 'ap50', 'ap25', 'miou', 'macc')


def trace_contrasts(root, scene, methods):
    result = {}
    for rank in RANKS:
        traces = {m: read_trace(root, scene, m, rank)[2] for m in methods}
        for base in (B, 'N0'):
            baseline = traces[base]
            old_states = {(s['overlap_index'], s['class_index']): s for s in baseline['states']}
            for method in methods:
                trace = traces[method]
                records = []
                for overlap in sorted({s['overlap_index'] for s in trace['states']}):
                    old = [e for e in baseline['events'] if e['overlap_index'] == overlap]
                    new = [e for e in trace['events'] if e['overlap_index'] == overlap]
                    old_hits = {(e['class_index'], e['gt_id']): e for e in old if e['event'] == 'first_match'}
                    new_hits = {(e['class_index'], e['gt_id']): e for e in new if e['event'] == 'first_match'}
                    states = [s for s in trace['states'] if s['overlap_index'] == overlap]
                    changes = [{'class_label': s['class_label'], 'base_ap': old_states[overlap, s['class_index']]['ap'],
                                'method_ap': s['ap'], 'base_hard_fn': old_states[overlap, s['class_index']]['hard_false_negatives'],
                                'method_hard_fn': s['hard_false_negatives']} for s in states
                               if s['ap'] != old_states[overlap, s['class_index']]['ap'] or
                               s['hard_false_negatives'] != old_states[overlap, s['class_index']]['hard_false_negatives']]
                    records.append({'overlap': states[0]['overlap_threshold'],
                        'added_gt_matches': [new_hits[k] for k in sorted(new_hits.keys() - old_hits.keys())],
                        'lost_gt_matches': [old_hits[k] for k in sorted(old_hits.keys() - new_hits.keys())],
                        'class_ap_changes': changes,
                        'duplicates': [e for e in new if e['event'] == 'duplicate'],
                        'ignore_and_false_positive_events': [e for e in new if e['event'] == 'ignore_test']})
                result[f'{method}/{base}/{rank}'] = records
    write_gzip(root / 'diagnostics' / scene / 'matcher_contrasts.json.gz', result)
    return {key: [{'overlap': r['overlap'], 'added': len(r['added_gt_matches']), 'lost': len(r['lost_gt_matches']),
                  'changed_classes': len(r['class_ap_changes']), 'duplicates': len(r['duplicates']),
                  'ignored': sum(not e['counted_fp'] for e in r['ignore_and_false_positive_events']),
                  'unmatched_fp': sum(e['counted_fp'] for e in r['ignore_and_false_positive_events'])} for r in rows]
            for key, rows in result.items()}


def method_probability(root, scene, method, objects, temperatures):
    path = root / 'probabilities' / scene / (method + '.json.gz')
    if path.exists():
        return read(path)
    if method == B:
        return {o: r['base'] for o, r in objects.items()}
    if method == 'AW_E03_OVR_A7':
        return {o: r['ovr_base'] for o, r in objects.items()}
    source = {'AW_E03_FC_FROZEN_DIRECT': 'F', 'AW_E03_OVR_DIRECT': 'O'}.get(method)
    if source:
        return {o: softmax(np.asarray(r['scores'][source]) / temperatures[source]).tolist()
                if r['scores'][source] is not None else None for o, r in objects.items()}
    return None  # Original N0 canonical confidence / old A7 not this study's probability scale.


def scene_diagnostics(binding, scene, methods):
    root = Path(binding['output_root'])
    out = root / 'diagnostics' / scene / 'summary.json'
    if out.exists():
        return read(out)
    locked = {m: read(root / 'locked' / scene / (m + '.json')) for m in methods}
    bundle = read(root / 'evidence' / scene / 'frozen.json.gz')
    ids, objects, temperatures = bundle['valid_ids'], bundle['objects'], bundle['temperatures']
    reviewer_root = Path(read(binding['reviewer_binding'])['output_root'])
    geometry_path = reviewer_root / 'diagnostics' / scene / 'objects.json.gz'
    INDEX.identity(geometry_path)
    geometry = read(geometry_path)
    if {str(r['owner_id']) for r in geometry} != set(objects):
        raise ValueError('diagnostic owner registry differs')
    probabilities = {m: method_probability(root, scene, m, objects, temperatures) for m in methods}
    ledger, summaries, blocking = [], {}, []
    counts = Counter()
    common_calibration = []
    for row in sorted(geometry, key=lambda r: r['owner_id']):
        owner = str(row['owner_id'])
        obj = objects[owner]
        identifiable = row['correspondence'] == 'unique' and row['geometry_iou'] > .5 and row['gt_label'] in ids
        gt = row['gt_label'] if identifiable else None
        base_label, n0 = locked[B]['labels'][owner], locked['N0']['labels'][owner]
        paired = obj['paired_F'] is not None and obj['paired_O'] is not None
        category = None
        if paired and identifiable:
            fc_correct = ids[int(np.argmax(obj['paired_F']))] == gt
            ovr_correct = ids[int(np.argmax(obj['paired_O']))] == gt
            category = ('F_right' if fc_correct else 'F_wrong') + '/' + ('O_right' if ovr_correct else 'O_wrong')
            counts[category] += 1
        if identifiable and all(obj['available'][n] for n in ('N', 'Q', 'F')):
            common_calibration.append(owner)
            if base_label != gt and any(obj['source_labels'][n] == gt for n in ('N', 'Q', 'F', 'O')):
                w, g = ids.index(base_label), ids.index(gt)
                nq = [softmax(np.asarray(obj['scores'][n])/temperatures[n]) for n in ('N', 'Q')]
                margin = sum(p[w] - p[g] for p in nq)
                blocking.append({'owner': int(owner), 'gt_label': gt, 'wrong_base_label': base_label,
                                 'NQ_margin': float(margin), 'impossible_for_one_region_to_overcome': bool(margin > 1)})
        record = {**{k: row[k] for k in ('owner_id', 'correspondence', 'geometry_iou', 'gt_label')},
                  'identifiable': identifiable, 'paired_outcome': category, 'base_label': base_label,
                  'N0_label': n0, 'source_labels': obj['source_labels'], 'methods': {}}
        for method in methods:
            label = locked[method]['labels'][owner]
            outcome = {'label': label, 'changed_B': label != base_label, 'changed_N0': label != n0,
                       'corrected_B': identifiable and base_label != gt and label == gt,
                       'harmed_B': identifiable and base_label == gt and label != gt,
                       'corrected_N0': identifiable and n0 != gt and label == gt,
                       'harmed_N0': identifiable and n0 == gt and label != gt,
                       'unused_correct_source': identifiable and label != gt and any(v == gt for v in obj['source_labels'].values()),
                       'extra_O_correction_survived': category == 'F_wrong/O_right' and label == gt,
                       'F_correct_O_wrong_harm': category == 'F_right/O_wrong' and label != gt}
            record['methods'][method] = outcome
        ledger.append(record)
    for method in methods:
        rows = [r['methods'][method] for r in ledger]
        summaries[method] = {key: sum(bool(r[key]) for r in rows) for key in rows[0] if key != 'label'}
        summaries[method]['examples'] = {key: [r['owner_id'] for r in ledger if r['methods'][method][key]][:2]
                                        for key in ('corrected_B', 'harmed_B')}
    # Same fixed common-identifiable support for every defined method probability.
    common_calibration = [o for o in common_calibration if all(p is None or p[o] is not None for p in probabilities.values())]
    gt_by_owner = {str(r['owner_id']): r['gt_label'] for r in ledger}
    calibration = {}
    for method, probs in probabilities.items():
        if probs is None:
            calibration[method] = {'status': 'PROBABILITY_SCALE_NOT_DEFINED', 'support': 0}
            continue
        nll, brier = [], []
        for owner in common_calibration:
            p = np.asarray(probs[owner])
            target = ids.index(gt_by_owner[owner])
            nll.append(-np.log(max(p[target], np.finfo(float).tiny)))
            truth = np.zeros(len(ids)); truth[target] = 1
            brier.append(np.sum((p-truth)**2))
        calibration[method] = {'status': 'MEASURED', 'support': len(nll), 'sum_nll': float(sum(nll)),
                               'sum_brier': float(sum(brier)), 'nll': float(np.mean(nll)) if nll else None,
                               'brier': float(np.mean(brier)) if brier else None}
    local = {}
    for method in methods:
        if not method.startswith(('PE_R3', 'PE_R2', 'PE_COMBO')):
            continue
        probs = probabilities[method]
        changed, full_support, no_pair, coverage = 0, 0, 0, 0
        drift, mass_drift, masses = [], [], []
        for owner, obj in objects.items():
            if obj['base'] is None:
                continue
            ix = np.asarray(obj['candidate'])
            base = np.asarray(obj['base'])
            start = np.asarray(obj['dependence_probabilities'].get('PE_D4_LINEAGE', obj['base'])) if method == 'PE_COMBO_D4_R3' else base
            p = np.asarray(probs[owner])
            outside = np.setdiff1d(np.arange(len(ids)), ix)
            if 'LOCAL' in method or 'COMBO' in method:
                drift.append(float(np.abs(p[outside] - start[outside]).max(initial=0)))
                mass_drift.append(float(abs(p[ix].sum() - start[ix].sum())))
            masses.append(float(start[ix].sum()))
            full_support += obj['paired_F'] is not None
            no_pair += obj['paired_F'] is None
            changed += locked[method]['labels'][owner] != locked[B]['labels'][owner]
            coverage += gt_by_owner[owner] in ids and ids.index(gt_by_owner[owner]) in ix
        local[method] = {'paired_support': full_support, 'no_pair': no_pair, 'changed_labels': changed,
                         'no_new_label': len(objects) - changed, 'maximum_outside_drift': max(drift, default=0),
                         'maximum_inside_mass_drift': max(mass_drift, default=0),
                         'candidate_mass_mean': float(np.mean(masses)), 'candidate_gt_coverage_raw_geometry': coverage}
    write_gzip(root / 'diagnostics' / scene / 'objects.json.gz', ledger)
    dep = [r['dependence'] for r in objects.values()]
    summary = {'scene': scene, 'status': 'POST_LOCK_OUTCOMES', 'owners': len(objects),
               'identifiable': sum(r['identifiable'] for r in ledger), 'paired_outcomes': dict(counts),
               'paired_coverage': sum(r['paired_F'] is not None for r in objects.values()),
               'source_available': {n: sum(r['available'][n] for r in objects.values()) for n in ('N', 'Q', 'F', 'O')},
               'pair_subset_restricted': sum(r['pairing'] == 'COMMON_SUBSET_REAGGREGATED' for r in objects.values()),
               'dependence_coverage': sum(d['status'] == 'COMPUTED' for d in dep),
               'unknown_support_atoms': sum(d.get('unknown_support', 0) for d in dep),
               'shared_atom_pairs': sum(d.get('shared_atom_pairs', 0) for d in dep),
               'NQ_nonzero_covariance_owners': sum(d.get('nonzero_rho_pairs', 0) > 0 for d in dep),
               'D4_equals_D3_owners': sum(d.get('D4_equals_D3', False) for d in dep),
               'rho_range': [min(d.get('rho_min', 0) for d in dep), max(d.get('rho_max', 0) for d in dep)],
               'text_comparison': bundle['text_comparison'], 'blocking': blocking,
               'methods': summaries, 'local': local, 'calibration': calibration,
               'matcher': trace_contrasts(root, scene, methods)}
    write_once(out, summary)
    return summary


def logical_costs(binding, scenes, methods):
    wave_root = Path(read(binding['wave1_binding'])['output_root'])
    all_costs = {}
    for scene in scenes:
        costs = read(wave_root / 'costs' / (scene + '.json'))
        index = costs['operation_manifest']
        INDEX.identity(index['path'], index)
        operations = read(index['path'])['sources']
        records = {}
        for method in methods:
            if method.startswith('PE_'):
                required = ['N0', 'Q_GAIN', 'AW_E03_FC_FROZEN']
                if method.startswith('PE_R') or method == 'PE_COMBO_D4_R3':
                    required.append('AW_E03_OVR')
                union = {}
                for source in required:
                    union.update(operations[source])
            else:
                union = method_operations(operations, method)
            records[method] = summarize_operations(union)
        all_costs[scene] = {'methods': records, 'original_operation_manifest': index}
    return all_costs


def diagnose(binding):
    root = Path(binding['output_root'])
    spec, lock = read(binding['spec']), read(root / 'transfer_lock.json')
    configs = configurations(spec, lock['choices'])
    if lock['combination']['status'] == 'MEASURED':
        configs.append(lock['combination'])
    methods = [c['id'] for c in configs]
    scenes = spec['datasets']['calibration'] + spec['datasets']['replica']
    summaries = {s: scene_diagnostics(binding, s, methods) for s in scenes}
    nominee = lock['research_nominee']['method']
    for omitted in spec['datasets']['replica']:
        kept = [s for s in spec['datasets']['replica'] if s != omitted]
        pool_methods(binding, 'replica', list(dict.fromkeys([nominee, B])), kept,
                     directory='sensitivity/omit_' + omitted)
    result = {'status': 'DIAGNOSTICS_COMPLETE', 'summaries': summaries,
              'logical_costs': logical_costs(binding, scenes, methods),
              'methods': methods, 'physical_image_forwards': 0, 'physical_text_forwards': 0,
              'physical_model_loads': 0, 'new_temperature_fits': 0, 'new_download_bytes': 0,
              'conditional_information_acquisition': 'QUEUED_NOT_EXECUTED'}
    write_gzip(root / 'diagnostics' / 'all.json.gz', result)
    return result
