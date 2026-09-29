import pytest


def test_only_identical_evaluator_bytes_allow_worktree_relocation(tmp_path):
    from src.static_ovmap.paired_evidence_study.evaluation import bound_context

    old = {'study_evaluator': {'path': '/old/evaluation.py', 'sha256': 'same', 'bytes': 10}, 'runtime_overlaps': [.5, .25]}
    new = {'study_evaluator': {'path': '/new/evaluation.py', 'sha256': 'same', 'bytes': 10}, 'runtime_overlaps': [.5, .25]}
    result, migration = bound_context(new, old)
    assert result == old and migration['actual_path'] == '/new/evaluation.py'
    changed = {**new, 'runtime_overlaps': [.5, .75, .25]}
    with pytest.raises(ValueError):
        bound_context(changed, old)
    changed = {**new, 'study_evaluator': {**new['study_evaluator'], 'sha256': 'changed'}}
    with pytest.raises(ValueError):
        bound_context(changed, old)
