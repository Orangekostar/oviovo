import numpy as np
from scipy.special import softmax


def test_residual_identity_mass_and_class_permutation():
    from src.static_ovmap.paired_evidence_study.residuals import ratio_update

    p = np.array([.1, .2, .3, .15, .25])
    r = np.array([1., -1., 2., 0., 3.])
    ix = np.array([0, 2, 4])
    assert np.array_equal(ratio_update(p, r, 0, ix), p)
    assert np.array_equal(ratio_update(p, np.ones(5), 2, ix), p)
    assert np.array_equal(ratio_update(p, None, 2, ix), p)
    out = ratio_update(p, r, 1, ix)
    assert np.array_equal(out[[1, 3]], p[[1, 3]])
    assert abs(out[ix].sum() - .65) < 1e-14
    permutation = np.array([4, 0, 2, 1, 3])
    inverse = np.argsort(permutation)
    permuted = ratio_update(p[permutation], r[permutation], 1, inverse[ix])
    np.testing.assert_allclose(permuted[inverse], out, atol=1e-14)


def test_graph_anchor_and_simplex_optimum():
    from src.static_ovmap.paired_evidence_study.pair_graph import (
        graph_solve,
        simplex_weights,
    )

    base = np.array([.2, .3, .5])
    z = np.array([1., 2., -1.])
    got, _ = graph_solve(np.array([-1., 2., 3.]), np.ones(3), base)
    np.testing.assert_allclose(got, softmax(.5 * (z + np.log(base))), atol=1e-13)
    s = np.array([[1., .8, .1], [.8, 2., .2], [.1, .2, .7]])
    w, v = simplex_weights(s[None])
    assert abs(w.sum() - 1) < 1e-14 and w.min() >= 0
    grid = [np.array([a, b, 1-a-b]) for a in np.linspace(0, 1, 31)
            for b in np.linspace(0, 1-a, 31)]
    assert v[0] <= min(x @ s @ x for x in grid) + 1e-12


def test_kernel_and_surrogate_jacobian():
    from src.static_ovmap.paired_evidence_study.dependence import (
        sensitivity,
        support_kernel,
    )

    masks = [np.array([1, 1, 0], bool), np.array([1, 0, 1], bool), None]
    k = support_kernel(masks, ['same', 'same', 'same'])
    np.testing.assert_array_equal(k, [[1, .5, 0], [.5, 1, 0], [0, 0, 1]])
    assert np.linalg.eigvalsh(k).min() > 0
    f = np.array([[.4, .3, .2], [.2, -.1, .8]])
    areas, delta, t = np.array([1., 3.]), np.array([.2, -.5, .6]), .7
    j = sensitivity(f, areas, delta, t)
    def value(features):
        a = areas @ features / areas.sum()
        return delta @ a / np.linalg.norm(a) / t
    numerical = np.zeros_like(f)
    for i in range(2):
        for c in range(3):
            plus, minus = f.copy(), f.copy()
            plus[i, c] += 1e-6
            minus[i, c] -= 1e-6
            numerical[i, c] = (value(plus) - value(minus)) / 2e-6
    np.testing.assert_allclose(j, numerical, atol=1e-9)
