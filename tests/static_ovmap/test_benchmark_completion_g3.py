"""G3: scoring must fail before GT access if a frozen scene changes."""
import pytest
from static_ovmap.benchmark_completion_v2.common import write


def test_legacy_freeze_checks_all_scene_identities_before_scoring(tmp_path):
    from static_ovmap.benchmark_completion_v2 import scoring
    scene = write(tmp_path / 'legacy/predictions/one/receipt.json',
                  dict(status='DONE', scene='one', cohort='tiny', methods={'L00_seed17': {}}))
    write(tmp_path / 'legacy/prediction_lock.json', dict(
        status='BLOCKED_DEPENDENCY', cohorts={'tiny': ['one']},
        scenes={'one': scene['identity']}, methods=['L00_seed17', 'L01_seed17'],
        expected_logical_scene_rows=2, completed_logical_scene_rows=1,
        all_available_predictions_frozen_before_new_GT_scoring=True))
    _, scenes = scoring.frozen_legacy_inputs(tmp_path)
    assert set(scenes['one']['methods']) == {'L00_seed17'}
    write(tmp_path / 'legacy/predictions/one/receipt.json', dict(
        status='DONE', scene='one', cohort='tiny', methods={'L00_seed17': {}, 'L01_seed17': {}}))
    with pytest.raises(ValueError, match='frozen scene'):
        scoring.frozen_legacy_inputs(tmp_path)


def test_semantic_role_mapping_does_not_guess_unmatched_or_conflicting_names():
    from static_ovmap.benchmark_completion_v2.protocol import map_training_roles
    result = map_training_roles(['Chair', 'sofa', 'novel object', 'ambiguous'],
        [2, 6], ['chair', 'couch'], [2],
        {'sofa': [dict(class_id=6, evidence='official-alias')],
         'ambiguous': [dict(class_id=2, evidence='one'), dict(class_id=6, evidence='two')]})
    assert [row['base'] for row in result['rows']] == [True, False, None, None]
    assert result['base_mask'] is None
    assert result['unresolved'] == ['novel object', 'ambiguous']


def test_global_source_off_never_reads_nq_and_source_on_uses_available_group():
    import numpy as np
    from static_ovmap.benchmark_completion_v2.standard_export import classify_scores
    calls = []
    def source(name):
        calls.append(name)
        return np.array([0., 1.]) if name == 'N' else None
    ids = np.array([2, 4])
    temperatures = dict(N=1., Q=1., F=1.)
    assert classify_scores('U00', ids, None, None, source, temperatures)['label'] == 0
    assert calls == []
    assert classify_scores('U10', ids, None, None, source, temperatures)['label'] == 4
    assert calls == ['N', 'Q']
    calls.clear()
    assert classify_scores('U01', ids, np.array([1., 0.]), np.array([0., 1.]), source, temperatures)['label'] == 4
    assert calls == []
    with pytest.raises(ValueError, match='Nonfinite supported F'):
        classify_scores('U00', ids, np.array([np.nan, 0.]), None, source, temperatures)


def test_uniform_frame_plan_retains_single_view_and_samples_finite_domain():
    from static_ovmap.benchmark_completion_v2.scene_bundle import uniform_frames
    assert uniform_frames([7]) == [7]
    assert uniform_frames([]) == []
    assert uniform_frames([0, 2, 5, 9]) == [0, 2, 5, 9]
    plan = uniform_frames(list(range(0, 410, 2)))
    assert len(plan) == len(set(plan)) == 200
    assert plan[0] == 0 and plan[50] == 102 and plan[-1] == 408


def test_query_legal_exhaustion_keeps_budget_and_full_replay_barriers():
    from types import SimpleNamespace
    from static_ovmap.benchmark_completion_v2.nq_worker import validate_query_completion
    frames = SimpleNamespace(schedule=[4, 9])
    result = dict(state=SimpleNamespace(logical_ledger=SimpleNamespace(attempts=1)),
        decisions=[dict(frame_id=4, results=[{}]), dict(frame_id=9, results=[])])
    assert validate_query_completion(result, frames, budget=200)['stop_reason'] == 'CANDIDATES_EXHAUSTED'
    result['decisions'].pop()
    with pytest.raises(ValueError, match='barrier'):
        validate_query_completion(result, frames, budget=200)
