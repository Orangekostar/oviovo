"""Acquired semantic evidence keeps scalar meaning under class permutations."""

from types import SimpleNamespace

import numpy as np

from src.static_ovmap.module_validation.semantic_features import semantic_feature_rows


def test_features_use_real_strengths_exclude_empty_background_and_ignore_fallback():
    native = SimpleNamespace(locked=True, owner_ids=np.array([7, 7, 8]), semantic_labels=np.array([5, 5, 5]))
    manifest = {"views": {"owner:7": ["r"]}, "requests": {"r": {"visible_target_pixels": 10}}}
    suggestions = {7: {"label_id": 42, "technical_fallback": False},
                   8: {"label_id": 42, "technical_fallback": True}}
    vectors = np.array([[1., 0.], [0., 1.]] * 3 + [[0., 1.], [1., 0.], [1., 0.]])
    records = {"r": {"status": "COMPLETE", "feature": np.array([.6, .4]), "vectors": vectors,
        "background_scale_usable": [True, False, False], "stats": {"request_mask_pixels": 10,
        "bbox_pixels": 20, "depth_valid_fraction": .8, "local_global_iou": .5}}}
    teacher = {"r": {"status": "COMPLETE", "representation_survived": True,
        "mapping": {"similarities": [.2, .8], "class_index": 1, "top1_top2_gap": .6}}}
    args = (native, manifest, suggestions, records, teacher, {7: np.array([1., 0.])})
    rows = semantic_feature_rows(*args, np.eye(2), np.eye(2), (5, 42), teacher_model="wow")
    assert set(rows) == {7}
    row = rows[7]
    np.testing.assert_allclose(row.values[[12, 13, 14, 15, 26, 31]], [.6, .2, .8, .6, 1., 1.])
    assert not row.available[7]  # One view cannot define pair disagreement.
    teacher["r"]["mapping"].update(similarities=[.8, .2], class_index=0)
    permuted = semantic_feature_rows(*args, np.eye(2)[::-1], np.eye(2)[::-1], (42, 5), teacher_model="wow")[7]
    np.testing.assert_allclose(permuted.values, row.values)
    np.testing.assert_array_equal(permuted.available, row.available)
