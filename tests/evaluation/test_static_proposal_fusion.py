import numpy as np

from src.static_ovmap.proposal_fusion import fuse_proposals


def test_fusion_reuses_only_geometrically_matched_labels_and_preserves_novel_mask():
    owners = np.array([1, 1, 1, 1, 2, 2, 0, 0])
    readout = {'1': {'class_id': 4, 'selected_query_ids': ['a', 'b']}, '2': None}
    masks = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=bool)
    result = fuse_proposals(owners, readout, masks, np.array([7, 8]), np.array([.8, .7]), np.array([11, 12]))
    assert result['class_ids'].tolist() == [4, 8]
    assert result['masks'][1, -1]
    assert result['ledger'][1]['reused_owner'] == 1
    assert result['ledger'][1]['kept'] is False
    assert result['ledger'][2]['reused_owner'] is None


def test_partial_proposal_label_reuse_does_not_force_nms():
    owners = np.array([1, 1, 1, 1, 0])
    result = fuse_proposals(owners, {'1': {'class_id': 4, 'selected_query_ids': ['a', 'b']}},
        np.array([[1, 1, 0, 0, 0]], dtype=bool), np.array([7]), np.array([.8]), np.array([11]))
    assert result['class_ids'].tolist() == [4, 4]
    assert result['ledger'][1]['best_owner_iou'] == .5
