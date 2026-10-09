"""Actual scorer parity and realized-branch evidence verification."""
import contextlib
from pathlib import Path
import time

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.evaluation import fraction_metrics
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.samv_local_probe.evaluation import baseline_parity
from static_ovmap.source_preserving_update.evaluation import partition_evaluator

from .binding import load_scene
from .common import ConsumptionIndex, canonical_digest, read, verified, write


def original_scorer_parity(binding):
    root = Path(binding['output_root'])
    lock = verified(root/'predictions/screen_A/summary.json')
    if lock['status'] != 'ALL_STAGE_PREDICTIONS_LOCKED':
        raise ValueError('parity requires the scientific prediction lock')
    screen = verified(root/'screen_store.json')
    parent_path = Path(binding['specification']['parent_probe_root'])/'result_store.json'
    index = ConsumptionIndex(root/'verification/input_verifications.json')
    parent = verified(parent_path)
    key = canonical_digest({'lock': lock['identity'], 'screen': screen['identity'],
                            'parent': parent['identity'], 'producer': index.identity(__file__)})
    dest = root/'verification/original_scorer_parity.json'
    if dest.exists():
        prior = verified(dest)
        if prior['input_identity'] != key:
            raise ValueError('parity dependencies changed')
        return prior
    started = time.perf_counter()
    fresh_rows = []
    for scene in binding['specification']['engineering_scenes']:
        _, inputs = load_scene(binding, scene)
        folder = root/'verification/scorer_parity'/scene
        folder.mkdir(parents=True, exist_ok=True)
        before = set(folder.glob('partitions/*/evaluation_cache/*/receipt.json'))
        evaluator = partition_evaluator(inputs, inputs.g1, binding, folder)
        with (folder/'run.log').open('a') as stream, contextlib.redirect_stdout(stream):
            fresh = evaluator.evaluate(owner_labels(inputs.g1), 'DQ01_G1_PARITY',
                                       'OFFICIAL_CURRENT_CLASS', inputs.g1.prediction_key)
        original = binding['baseline_rows']['SU01_G1'][scene]['evaluation_receipt']
        checks = baseline_parity(read(fresh['evaluation_receipt']), read(original), inputs.index)
        fresh_rows.append({'scene': scene, 'fresh_receipt': fresh['evaluation_receipt'],
                           'parent_receipt': original, 'checks': checks,
                           'new_scorer_execution': Path(fresh['evaluation_receipt']) not in before})
        inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    pool_rows = []
    if parent['cohorts'] != binding['screen_cohorts']:
        raise ValueError('parent probe cohorts/order differ')
    for cohort, scenes in binding['screen_cohorts'].items():
        for method, old_method in (('DQ00_D2', 'REF_D2'), ('DQ01_G1', 'SV00_G1')):
            actual = screen['pooled_metrics'][cohort][method]
            previous = parent['pooled_metrics'][cohort][old_method]
            per_scene = []
            for scene in scenes:
                a = next(r for r in screen['scene_metrics'] if r['scene'] == scene and r['method'] == method)
                b = next(r for r in parent['scene_metrics'] if r['scene'] == scene and r['method'] == old_method)
                check = baseline_parity(read(a['evaluation_receipt']), read(b['evaluation_receipt']), index)
                if a['prediction_identity'] != b['prediction_identity']:
                    raise ValueError('ordered pool source payload differs')
                per_scene.append({'scene': scene, 'checks': check,
                                  'prediction_identity': a['prediction_identity']})
            a_classes = verified(actual['per_class_receipt'])['classes']
            b_classes = verified(previous['per_class_receipt'])['classes']
            if fraction_metrics(actual['metrics']) != fraction_metrics(previous['metrics']) or a_classes != b_classes:
                raise ValueError('exact ordered baseline pool/per-class parity failed')
            pool_rows.append({'cohort': cohort, 'method': method, 'parent_method': old_method,
                              'ordered_scenes': scenes, 'scene_proofs': per_scene,
                              'five_metrics_exact': True, 'per_class_exact': True,
                              'actual_pool_identity': actual['identity'],
                              'parent_pool_identity': previous['identity']})
    index.write_memo(root/'verification/input_verifications.json')
    return write(dest, {'status': 'ORIGINAL_SCORER_AND_ORDERED_POOLS_VERIFIED',
                       'input_identity': key, 'fresh_G1_rows': fresh_rows, 'ordered_pool_checks': pool_rows,
                       'elapsed_seconds': time.perf_counter()-started,
                       'timing_of_check': 'AFTER_SCREEN; NO_SCIENTIFIC_OPERATOR_OR_CHOICE_CHANGED',
                       'mask_directory_normalization_only': True})
