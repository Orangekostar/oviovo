import numpy as np

from src.static_ovmap.split_evidence import split_from_views


def camera_poses():
    poses = {i: np.eye(4) for i in range(3)}
    for i in poses:
        poses[i][0, 3] = i*.1
    return poses


def test_cross_view_mask_permutations_split_native_atoms_without_changing_residual():
    # Numeric IDs deliberately reverse between frames; the last atom is never seen.
    masks = np.array([[1, 1, 2, 2, -1], [9, 9, 7, 7, -1], [4, 4, 8, 8, -1]])
    split = split_from_views(masks, [0, 1, 2], camera_poses(), min_seed_atoms=1,
                            min_child_atoms=1, min_assigned_fraction=.7)
    assert split['accepted']
    np.testing.assert_array_equal(split['atom_children'], [1, 1, 2, 2, 0])
    assert split['independent_frame_ids'] == [0, 1, 2]


def test_duplicate_views_and_merged_entity_observations_do_not_support_split():
    masks = np.array([[1, 1, 2, 2], [9, 9, 7, 7], [4, 4, 8, 8]])
    duplicate = {i: np.eye(4) for i in range(3)}
    assert not split_from_views(masks, [0, 1, 2], duplicate, min_seed_atoms=1, min_child_atoms=1)['accepted']
    masks[2] = 3
    assert not split_from_views(masks, [0, 1, 2], camera_poses(), min_seed_atoms=1, min_child_atoms=1)['accepted']


def test_ambiguous_atoms_do_not_get_nearest_child_labels():
    masks = np.array([[1, 1, 2, 2, 1], [9, 9, 7, 7, 7], [4, 4, 8, 8, 0]])
    split = split_from_views(masks, [0, 1, 2], camera_poses(), min_seed_atoms=1,
        min_child_atoms=1, min_seed_agreement=.6, min_assigned_fraction=.7)
    assert split['accepted']
    assert split['atom_children'][-1] == 0
