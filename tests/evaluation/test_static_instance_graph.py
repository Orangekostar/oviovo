import numpy as np

from src.static_ovmap.instance_graph import frame_pair_evidence, aggregate_edges, constrained_components


def test_separate_partial_masks_do_not_imply_cannot_link():
    owners = np.array([[1, 1, 1, 1, 2, 2, 2, 2]])
    masks = np.array([[3, 4, 3, 4, 5, 6, 5, 6]])
    evidence = frame_pair_evidence(owners, masks, np.ones_like(owners, dtype=bool),
                                   min_owner_pixels=1, min_mask_pixels=1, whole_entity_source=True)
    assert evidence['negative'] == []


def test_unknown_mask_source_cannot_claim_whole_object_separation():
    evidence = frame_pair_evidence(np.array([[1, 1, 2, 2]]), np.array([[1, 1, 2, 2]]),
                                   np.ones((1, 4), dtype=bool), min_owner_pixels=1, min_mask_pixels=1)
    assert evidence['negative'] == []


def test_whole_entity_alignment_supports_separation_and_shared_mask_supports_merge():
    owners = np.array([[1, 1, 2, 2]])
    valid = np.ones_like(owners, dtype=bool)
    split = frame_pair_evidence(owners, np.array([[7, 7, 9, 9]]), valid,
                               min_owner_pixels=1, min_mask_pixels=1, whole_entity_source=True)
    assert [x['owners'] for x in split['negative']] == [[1, 2]]
    joined = frame_pair_evidence(owners, np.array([[7, 7, 7, 7]]), valid,
                                min_owner_pixels=1, min_mask_pixels=1, whole_entity_source=True)
    assert [x['owners'] for x in joined['positive']] == [[1, 2]]
    assert joined['negative'] == []


def test_repeated_camera_and_distant_surface_cannot_create_merge():
    frame = {'positive': [{'owners': [1, 2]}], 'negative': [], 'covisible': [[1, 2]]}
    frames = {i: frame for i in range(3)}
    poses = {i: np.eye(4) for i in range(3)}
    edges = aggregate_edges(frames, poses, {(1, 2): .01})
    assert edges['positive'] == []
    for i in range(3):
        poses[i][0, 3] = i * .1
    assert len(aggregate_edges(frames, poses, {(1, 2): .01})['positive']) == 1
    assert aggregate_edges(frames, poses, {(1, 2): .06})['positive'] == []


def test_component_wide_prohibition_blocks_transitive_merge():
    positive = [{'owners': [1, 2], 'weight': 5}, {'owners': [2, 3], 'weight': 4}]
    result = constrained_components([1, 2, 3, 70000], positive, [{'owners': [1, 3]}])
    assert result['components'] == [[1, 2], [3], [70000]]
    assert result['rejected'][0]['conflicting_pair'] == [1, 3]
