import numpy as np

from src.static_ovmap.pooled_metrics import pooled_semantic_metrics


def test_pooled_vertex_metrics_are_not_scene_macro_and_ignore_gt_void():
    pairs = [(np.array([1, 0]), np.array([1, 1])),
             (np.ones(9, dtype=int), np.zeros(9, dtype=int))]
    result = pooled_semantic_metrics(pairs, [1, 2])
    assert result['semantic_miou'] == .1
    assert result['semantic_macc'] == .1
    assert result['semantic_miou'] != .5  # scene macro
    assert result['per_class']['1']['gt_vertices'] == 10
    assert '2' not in result['per_class']


def test_no_valid_gt_returns_null_instead_of_zero():
    result = pooled_semantic_metrics([(np.array([0]), np.array([1]))], [1, 2])
    assert result['semantic_miou'] is None
    assert result['semantic_macc'] is None
