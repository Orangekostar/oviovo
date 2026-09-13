"""Evaluation-only observers for the unchanged receipt-bound released matcher.

Observe original Python line execution rather than replace its duplicate or AP
rules. A second terminal-only observer verifies PR arrays and FN counts, and a
fully unobserved call verifies every AP entry. No source file is modified.
"""
import inspect
import math
import sys

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def trace_released_matches(evaluator, matches):
    function = evaluator['evaluate_matches']
    lines, start = inspect.getsourcelines(function)
    anchors = {'duplicate': 'cur_score[gti] = max_score',
        'first_match': "pred_visited[pred['filename']] = True",
        'hard_fn': 'hard_false_negatives += 1',
        'ignore_test': 'if proportion_ignore <= overlap_th:',
        'state': 'ap[di, li, oi] = ap_current'}
    locations = {}
    for kind, text in anchors.items():
        found = [start+i for i,line in enumerate(lines) if line.strip() == text]
        if len(found) != 1:
            raise ValueError('released observer anchor changed: '+kind)
        locations[found[0]] = kind

    def observe(detailed):
        events, states = [], []
        winners = {}
        def tracer(frame, event, arg):
            if frame.f_code is not function.__code__:
                return None
            if event != 'line' or frame.f_lineno not in locations:
                return tracer
            kind = locations[frame.f_lineno]
            if not detailed and kind != 'state':
                return tracer
            local = frame.f_locals
            common = {'distance_index':int(local['di']), 'class_index':int(local['li']),
                'class_label':local['label_name'], 'overlap_index':int(local['oi']),
                'overlap_threshold':float(local['overlap_th'])}
            if kind == 'state':
                curve = bool(local['has_gt'] and local['has_pred'])
                states.append(_plain({**common, 'has_gt':bool(local['has_gt']), 'has_pred':bool(local['has_pred']),
                    'hard_false_negatives':int(local['hard_false_negatives']),
                    'y_true':local['y_true'].copy(), 'y_score':local['y_score'].copy(),
                    'precision':local['precision'].copy() if curve else None,
                    'recall':local['recall'].copy() if curve else None,
                    'score_thresholds':local['thresholds'].copy() if curve else None,
                    'ap':float(local['ap_current'])}))
                return tracer
            row = {**common, 'event':kind, 'scene_key':local['m']}
            if kind in ('first_match','duplicate','hard_fn'):
                gt = local['gt']; row['gt_id'] = int(gt['instance_id'])
                key = (local['di'],local['oi'],local['li'],local['m'],row['gt_id'])
            if kind in ('first_match','duplicate'):
                pred = local['pred']; row.update(candidate_file=pred['filename'],
                    confidence=float(pred['confidence']), overlap=float(local['overlap']))
                if kind == 'first_match':
                    winners[key] = pred['filename']
                else:
                    previous = winners.get(key)
                    previous_score = float(local['cur_score'][local['gti']])
                    confidence = float(local['confidence'])
                    row.update(previous_score_owner=previous, previous_score=previous_score,
                        tp_score=float(local['max_score']), fp_score=float(local['min_score']),
                        score_identity_ambiguous_tie=previous_score == confidence,
                        score_candidate_files=[previous,pred['filename']])
                    if confidence > previous_score:
                        winners[key] = pred['filename']
                    row['tp_score_owner'] = winners.get(key)
                    row['fp_score_owner'] = (previous if confidence > previous_score else pred['filename'])
            elif kind == 'ignore_test':
                pred = local['pred']; row.update(candidate_file=pred['filename'],
                    confidence=float(pred['confidence']), ignore_fraction=float(local['proportion_ignore']),
                    counted_fp=bool(local['proportion_ignore'] <= local['overlap_th']),
                    void_intersection=int(pred['void_intersection']), ignored_vertices=int(local['num_ignore']))
            events.append(_plain(row))
            return tracer
        previous = sys.gettrace()
        try:
            sys.settrace(tracer)
            ap = function(matches)
        finally:
            sys.settrace(previous)
        return ap, states, events

    baseline = function(matches)
    actual, states, events = observe(True)
    reference, reference_states, _ = observe(False)
    ap_exact = np.array_equal(actual, baseline, equal_nan=True) and np.array_equal(actual, reference, equal_nan=True)
    pr_exact = states == reference_states
    if not ap_exact or not pr_exact:
        raise ValueError('released observer changed AP/PR/FN behavior')
    return actual, {'parity':{'ap_exact':bool(ap_exact),'pr_and_fn_exact':pr_exact,
        'reference':'unchanged evaluator: unobserved AP and separate terminal-only PR/FN observer'},
        'instrumentation':'sys.settrace scoped to original evaluate_matches code object; source unchanged',
        'states':states, 'events':events,
        'event_warning':'duplicates may emit min-score FP without marking that prediction visited; events are not one-prediction/one-event'}
