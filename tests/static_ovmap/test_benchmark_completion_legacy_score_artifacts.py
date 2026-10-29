from pathlib import Path

import numpy as np
import pytest


def test_partition_owner_labels_preserves_eligible_order_and_unknown_positive_owners():
    from static_ovmap.benchmark_completion_v2.legacy_score_artifacts import partition_owner_labels

    owner_ids = np.array([0, 7, 2, 7, 9, 2, 0, 10])
    semantic_labels = np.array([3, 11, 22, 11, 33, 22, 4, 0])

    result = partition_owner_labels(owner_ids, semantic_labels, [7, 2])

    np.testing.assert_array_equal(result['eligible_labels'], [11, 22])
    np.testing.assert_array_equal(result['preserved_owners'], [9, 10])
    np.testing.assert_array_equal(result['preserved_labels'], [33, 0])


def test_partition_owner_labels_rejects_invalid_eligible_owners():
    from static_ovmap.benchmark_completion_v2.legacy_score_artifacts import partition_owner_labels

    with pytest.raises(ValueError):
        partition_owner_labels(np.array([7, 7]), np.array([11, 12]), [7])
    with pytest.raises(ValueError):
        partition_owner_labels(np.array([7]), np.array([11]), [99])
    with pytest.raises(ValueError):
        partition_owner_labels(np.array([7]), np.array([11]), [7, 7])
    with pytest.raises(ValueError):
        partition_owner_labels(np.array([0, 7]), np.array([0, 11]), [0])


def test_publication_candidates_allowlist_legacy_class_score_artifacts(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates

    allowed = [
        tmp_path / 'publication/legacy_class_scores.json',
        tmp_path / 'publication/legacy_class_scores/test_identity/manifest.json',
        tmp_path / 'publication/legacy_class_scores/test_identity/scores.npz',
    ]
    for path in allowed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    (tmp_path / 'publication/legacy_class_scores/test_identity/raw_GT.json').write_bytes(b'fixture')
    (tmp_path / 'publication/legacy_class_scores/test_identity/points.ply').write_bytes(b'fixture')

    assert {path.relative_to(tmp_path) for path in candidates(tmp_path)} == {
        Path('publication/legacy_class_scores.json'),
        Path('publication/legacy_class_scores/test_identity/manifest.json'),
        Path('publication/legacy_class_scores/test_identity/scores.npz'),
    }
