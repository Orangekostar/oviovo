import numpy as np


def test_undersegmented_observer_is_removed_before_consensus():
    from src.oviv2.surface_mask_consensus import construct_mask_graph

    observations = np.array([[1, 1, 1], [1, 1, 1], [1, 2, 2], [1, 2, 2]])
    result = construct_mask_graph(observations, minimum_cloud_size=1)
    merged = result["mask_keys"].index((0, 1))
    assert result["undersegmented"][merged]
    assert result["contained"][:, merged].nnz == 0
    assert not result["visible"][:, 0].any()


def test_consensus_clusters_repeated_regions_without_merging_separated_objects():
    from src.oviv2.surface_mask_consensus import cluster_masks, construct_mask_graph

    labels = np.array([[1, 1, 1], [1, 1, 1], [1, 2, 2], [1, 2, 2]])
    graph = construct_mask_graph(labels, minimum_cloud_size=1)
    assigned, _ = cluster_masks(graph, mode="consensus")
    lookup = dict(zip(graph["mask_keys"], assigned.tolist()))
    assert lookup[(0, 1)] == -1
    assert lookup[(1, 1)] == lookup[(2, 1)]
    assert lookup[(1, 2)] == lookup[(2, 2)]
    assert lookup[(1, 1)] != lookup[(1, 2)]


def test_arbitration_rejects_ties_single_views_and_absent_evidence():
    from src.oviv2.surface_mask_consensus import arbitrate_candidates

    votes = np.array([[0, 0, 1], [0, 1, -1], [0, -1, -1], [-1, -1, -1]])
    got = arbitrate_candidates(votes)
    assert got.tolist() == [0, -1, -1, -1]


def test_owner_readout_keeps_residuals_and_never_renames_single_parent_without_split():
    from src.oviv2.surface_mask_consensus import apply_owner_proposals

    owners = np.array([4, 4, 4, 5, 5, 9, 9])
    proposals = np.array([0, 1, -1, 0, -1, 2, -1])
    result, lineage = apply_owner_proposals(owners, np.arange(7), proposals)
    assert result.tolist() == [10, 11, 4, 10, 5, 9, 9]
    assert lineage[0]["parents"] == [4, 5]
    assert lineage[1]["parents"] == [4]


def test_one_frame_has_at_most_one_support_vote_after_cluster_union():
    from scipy import sparse

    from src.oviv2.surface_mask_consensus import consensus_edges

    # Two common masks in frame 0 are still only one independent supporting view.
    contained = sparse.csr_matrix([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
    visible = np.ones((2, 2), bool)
    edges = consensus_edges(contained, visible, np.array([0, 0, 1, 1]), threshold=0.9)
    assert edges.nnz == 0
