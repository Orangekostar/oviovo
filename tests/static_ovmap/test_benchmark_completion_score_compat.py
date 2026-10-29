import os
from pathlib import Path

import numpy as np
import pytest


def test_absent_first_class_does_not_abort_official_ap(tmp_path):
    checkout = os.environ.get('OVMAP_TEST_OPENMASK3D_CHECKOUT')
    if not checkout:
        pytest.skip('Set OVMAP_TEST_OPENMASK3D_CHECKOUT to the pinned author checkout')
    from static_ovmap.benchmark_completion_v2.common import write, digest_file
    from static_ovmap.benchmark_completion_v2.standard_score import evaluator
    from static_ovmap.benchmark_completion_v2.score_compat import adapt
    source = Path(checkout) / 'openmask3d/evaluation/eval_semantic_instance.py'
    thresholds = np.r_[np.arange(.5, .95, .05), .25].tolist()
    binding = dict(output_root=str(tmp_path))
    write(tmp_path / 'protocol_lock.json', dict(official_sources=dict(evaluator_scan=dict(
        path=str(source), sha256=digest_file(source))), scopes=dict(scannet200_val=dict(
            instance_ids=[3, 5], instance_names=['absent', 'present'],
            IoU_thresholds=thresholds, minimum_region_vertices=100))))
    namespace = evaluator(binding, 'scannet200_val')
    gt = tmp_path / 'gt.txt'
    np.savetxt(gt, np.full(100, 5001), fmt='%d')
    prediction = dict(pred_masks=np.ones((100, 1), bool), pred_classes=np.array([5]),
                      pred_scores=np.array([.875]))
    actual_gt, actual_pred = namespace['assign_instances_for_scan'](prediction, str(gt))
    matches = {'one': dict(gt=actual_gt, pred=actual_pred)}
    with pytest.raises(UnboundLocalError, match='ar_current'):
        namespace['evaluate_matches'](matches)
    evidence = adapt(namespace, source)
    ap, ar, _, _ = namespace['evaluate_matches'](matches)
    assert evidence['AP_AST_unchanged'] is True
    assert np.isnan(ap[0, 0]).all() and np.isnan(ar[0, 0]).all()
    assert np.array_equal(ap[0, 1], np.ones(10))
    assert namespace['compute_averages'](ap)['all_ap'] == 1.0
    # Same author input with present class first has a defined auxiliary value;
    # compare original and adapted AP, including the later absent-class slot.
    namespace = evaluator(binding, 'scannet200_val')
    namespace['CLASS_LABELS'] = ('present', 'absent')
    original = namespace['evaluate_matches'](matches)[0]
    adapt(namespace, source)
    repaired = namespace['evaluate_matches'](matches)[0]
    assert np.array_equal(original, repaired, equal_nan=True)
