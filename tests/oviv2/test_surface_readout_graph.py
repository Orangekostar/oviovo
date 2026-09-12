import numpy as np
from scipy import sparse


def test_patches_weld_triangle_soup_but_do_not_join_disconnected_surfaces():
    from src.oviv2.surface_readout_graph import surface_patches

    xyz = np.array(
        [
            [0.001, 0.001, 0],
            [0.012, 0.001, 0],
            [0.001, 0.012, 0],
            [0.012, 0.001, 0],
            [0.021, 0.012, 0],
            [0.001, 0.012, 0],
            [0.001, 0.001, 0.005],
            [0.012, 0.001, 0.005],
            [0.001, 0.012, 0.005],
        ],
        np.float32,
    )
    result = surface_patches(
        xyz, np.tile([0, 0, 1.0], (9, 1)), np.arange(9).reshape(-1, 3)
    )
    p = result["source_patch"]
    assert p[1] == p[3] and p[2] == p[5]
    assert p[0] != p[6]
    assert p[0] != p[4]
    assert result["physical_samples"] == 7
    assert result["physical_weight"][1] == 0.5
    assert result["adjacency"][p[0], p[4]] > 0
    assert result["adjacency"][p[0], p[6]] == 0


def test_s2_unary_preserves_owner_confidence_and_physical_weighting():
    from src.oviv2.surface_readout_graph import s2_pseudo_unary

    unary, valid, tie = s2_pseudo_unary(
        np.array([0, 0, 0, 1]),
        np.array([1, 1, 2, 0]),
        np.array([0.8, 0.8, 0.5, 0.0]),
        np.array([0.5, 0.5, 1.0, 1.0]),
        np.array([1, 2, 3]),
    )
    np.testing.assert_allclose(unary[0], [0.25, 0.4, 0.65])
    assert valid.tolist() == [True, False]
    assert tie[0] == 0


def test_lambda_zero_is_patch_only_and_zero_support_stays_unchanged():
    from src.oviv2.surface_readout_graph import graph_labels

    unary = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    edges = sparse.csr_matrix(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    )
    got = graph_labels(
        unary, edges, np.array([True, True, False]), np.array([0, 1, 0]), strength=0.0
    )
    assert got.tolist() == [0, 1, -1]


def test_zero_support_node_does_not_send_graph_messages():
    from src.oviv2.surface_readout_graph import graph_labels

    unary = np.array([[0.0, 0.1], [0.0, 0.0]])
    edges = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    got = graph_labels(
        unary, edges, np.array([True, False]), np.array([0, 1]), strength=10.0
    )
    assert got.tolist() == [0, -1]
