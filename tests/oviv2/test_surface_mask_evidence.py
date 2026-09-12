import numpy as np
from scipy import sparse


def test_mask_boundaries_and_image_edges_do_not_supply_votes():
    from src.oviv2.surface_mask_evidence import interior_labels

    labels = np.ones((7, 7), np.uint16)
    labels[:, 3:] = 2
    got = interior_labels(labels)
    assert got[3, 1] == 1 and got[3, 4] == 2
    assert got[3, 2] == got[3, 3] == got[0, 4] == 0


def test_boundary_graph_needs_independent_joint_observations():
    from src.oviv2.surface_mask_evidence import boundary_graph

    graph = sparse.csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    # Frame 0 sees opposite masks; frame 1 has no usable evidence at node 1.
    one = np.array([[1, 1], [2, 0], [0, 3]], np.uint16)
    got, stats = boundary_graph(graph, one)
    np.testing.assert_array_equal(got.toarray(), graph.toarray())
    assert stats["boundary_edge_count"] == 0
    two = np.column_stack((one, [1, 2, 0]))
    got, stats = boundary_graph(graph, two)
    assert np.isclose(got[0, 1], 0.2)
    assert got[1, 2] == 1.0
    np.testing.assert_array_equal(got.toarray(), got.toarray().T)
    assert stats["boundary_edge_count"] == 1


def test_rgb_term_is_applied_only_to_observed_colors():
    from src.oviv2.surface_mask_evidence import boundary_graph

    graph = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    labels = np.ones((2, 2), np.uint16)
    colors = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    unchanged, _ = boundary_graph(graph, labels, rgb=colors, rgb_valid=[True, False])
    assert unchanged[0, 1] == 1.0
    changed, _ = boundary_graph(graph, labels, rgb=colors, rgb_valid=[True, True])
    assert 0 < changed[0, 1] < 0.01
