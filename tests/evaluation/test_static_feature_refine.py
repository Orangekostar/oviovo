import numpy as np

from src.static_ovmap.feature_refine import geometry_edges, refine_sparse


def test_edges_cannot_cross_owner_unknown_occlusion_or_depth_discontinuity():
    owner = np.array([[1, 1, 2, 0], [1, 1, 2, 2]])
    depth = np.array([[1., 1., 1., 1.], [2., 1., 1., 1.]])
    visible = np.ones_like(owner, dtype=bool)
    visible[1, 2] = False
    src, dst = geometry_edges(owner, depth, visible)
    assert set(zip(src.tolist(), dst.tolist())) == {(0, 1), (1, 0), (1, 5), (5, 1)}


def test_sparse_update_matches_hand_computed_neighbors_and_keeps_unknown_isolated():
    features = np.array([[1., 0.], [0., 1.], [9., 9.]], dtype=np.float32)
    refined = refine_sparse(features, np.array([0, 1]), np.array([1, 0]), mixing=.2)
    np.testing.assert_allclose(refined, [[.8, .2], [.2, .8], [9., 9.]], atol=1e-7)


def test_pooling_linear_update_can_reuse_neighbor_pass_for_all_three_lambdas():
    from src.static_ovmap.dense_features import pool_owner_features
    features = np.array([[[1., 0.], [0., 1.]], [[.5, .5], [2., 0.]]], dtype=np.float32)
    owners = np.ones((2, 2), dtype=np.int64)
    a, b = geometry_edges(owners, np.ones((2, 2)), np.ones((2, 2), dtype=bool))
    neighbors = refine_sparse(features.reshape(-1, 2), a, b, mixing=1.).reshape(2, 2, 2)
    original_pool = pool_owner_features(features, owners, [1])[1]['feature']
    neighbor_pool = pool_owner_features(neighbors, owners, [1])[1]['feature']
    for mixing in (.1, .2, .3):
        explicit = refine_sparse(features.reshape(-1, 2), a, b, mixing=mixing).reshape(2, 2, 2)
        np.testing.assert_allclose(pool_owner_features(explicit, owners, [1])[1]['feature'],
            (1-mixing)*original_pool+mixing*neighbor_pool, atol=1e-7)
