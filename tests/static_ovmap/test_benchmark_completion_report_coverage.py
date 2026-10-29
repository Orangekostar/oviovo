"""Reports must describe measured coverage rather than an early execution phase."""
from pathlib import Path

import pytest

from static_ovmap.benchmark_completion_v2 import review
from static_ovmap.benchmark_completion_v2.common import write


@pytest.fixture
def report(tmp_path, monkeypatch):
    root = tmp_path / 'out'
    write(root / 'bootstrap/source_binding.json', dict(
        base='TEST_ONLY_PARENT', archive_sha256='TEST_ONLY_ARCHIVE', source_files=[]))
    monkeypatch.setattr(review, 'REPO', tmp_path / 'repo')
    monkeypatch.setattr(review, '_handoff_evidence', lambda *_: 'Measured ledger retained.\n\n')
    flags = dict(software_complete=False, calibration_complete=True, legacy_complete=False,
                 standard_coverage_complete=False, external_comparisons_complete=True,
                 publication_complete=False, claim_supported=False)
    coverage = dict(completion_flags=flags, standard={
        'scannet200_val': dict(completed_scenes=312, expected_scenes=312,
                              completed_cells=2496, expected_cells=2496),
        'replica48': dict(completed_scenes=8, expected_scenes=8, completed_cells=32, expected_cells=64)},
        external={method: {
            'replica48': dict(complete=True, evidence='LOCAL_FULL', locally_completed_scenes=8),
            'scannet200_val': dict(complete=True, evidence='AUTHOR_REPORTED_COMPATIBLE', locally_completed_scenes=0)}
            for method in ('OVI-MAP', 'OpenMask3D', 'Open3DIS', 'MaskClustering')})
    binding = dict(output_root=str(root), gpu='2')
    folder = tmp_path / 'repo' / review.DOCS
    return binding, coverage, folder


def test_completed_ports_and_missing_native_cells_are_distinguished(report):
    binding, coverage, folder = report
    review._documents(binding, coverage, [dict(kind='STANDARD_SCENE_CELL')] * 65)
    results = (folder / 'RESULTS.md').read_text()
    handoff = (folder / 'HANDOFF.md').read_text()
    assert '| replica48 | 8/8 | 32/64 |' in results
    for method in coverage['external']:
        assert f'| {method} | replica48 | LOCAL_FULL | 8/8 |' in handoff
        assert f'| {method} | scannet200_val | AUTHOR_REPORTED_COMPATIBLE | 0/312 |' in handoff
    assert '65' in handoff
    assert 'missing author instance-GT/point-order' not in (folder / 'CLAIMS.md').read_text()


def test_report_does_not_override_current_gpu_selection_or_claim_live_monitors(report):
    binding, coverage, folder = report
    review._documents(binding, coverage, [])
    handoff = (folder / 'HANDOFF.md').read_text()
    assert 'One inherited physical GPU0' not in handoff
    assert 'single CPU score_recovery monitor repairs' not in handoff
    assert 'resource_ledger.json' in handoff and 'recipe.json' in handoff


def test_review_lists_only_current_incomplete_dimensions(report):
    binding, coverage, folder = report
    review._documents(binding, coverage, [])
    text = (folder / 'REVIEW.md').read_text()
    assert '| external_comparisons_complete | true |' in text
    assert '| standard_coverage_complete | false |' in text
    assert 'remaining complete external method adapters/ports' not in text


def test_incomplete_external_scope_is_reported_from_its_actual_record(report):
    binding, coverage, folder = report
    coverage['external']['Open3DIS']['replica48'].update(
        complete=False, evidence='INCOMPLETE', locally_completed_scenes=1)
    coverage['completion_flags']['external_comparisons_complete'] = False
    review._documents(binding, coverage, [dict(kind='EXTERNAL_SCENE_METHOD')])
    handoff = (folder / 'HANDOFF.md').read_text()
    assert '| Open3DIS | replica48 | INCOMPLETE | 1/8 |' in handoff
    assert '| external_comparisons_complete | false |' in (folder / 'REVIEW.md').read_text()


def test_existing_unfinished_primary_review_is_not_reported_as_final(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import work_state
    write(tmp_path / 'protocol_lock.json', dict(scopes={
        'scannet200_val': dict(scenes=['TEST_ONLY_SCENE'], full_AC_available=True)}))
    write(tmp_path / 'calibration/summary.json', dict(status='DONE'))
    write(tmp_path / 'review/primary_final.json', dict(requirements_review_complete=False))
    monkeypatch.setattr(work_state, 'resource_block', lambda *_: None)
    monkeypatch.setattr(work_state, 'scene_state', lambda *_: dict(
        status='BLOCKED_DEPENDENCY', runnable=False, reason='TEST_ONLY_MISSING_INPUT'))
    monkeypatch.setattr(review, '_documents', lambda *_: None)
    result = review.run(dict(output_root=str(tmp_path), asset_config={}))
    assert result['final_primary_review_complete'] is False
