import numpy as np
from scipy.special import softmax

from src.static_ovmap.paired_evidence_study.dependence import (
    collapse_sources,
    solve_variants,
)
from src.static_ovmap.paired_evidence_study.residuals import paired_residual, pool
from src.static_ovmap.paired_evidence_study.selection import choose


def test_genuine_availability_and_common_temperature():
    scores = {'N': [1., 0.], 'Q': None, 'F': None, 'O': [0., 2.]}
    t = dict.fromkeys(scores, .5)
    np.testing.assert_array_equal(pool(scores, t, ('N', 'Q', 'F')), softmax([2., 0.]))
    assert pool(scores, t, ('Q', 'F')) is None
    np.testing.assert_array_equal(paired_residual([1, 2], [1, 2], .5), [0, 0])
    assert np.ptp(paired_residual([1, 2], [1, 2], .5, 1)) > 0


def test_no_overlap_d4_equals_d3_and_exact_alias_does_not_gain_weight():
    text = np.eye(3)
    sources = []
    for name, feature, image in [('N', [.8, .1, .3], 'one'), ('Q', [.1, .7, .3], 'two')]:
        feature = np.asarray(feature)
        scores = text @ (feature / np.linalg.norm(feature))
        sources.append({'name': name, 'scores': scores, 'features': feature[None], 'areas': np.array([1.]),
                        'text': text, 'temperature': .7, 'space': 'same-model', 'computation_identity': image,
                        'atoms': [{'image': image, 'mask': np.ones(4, bool)}]})
    base = np.array([.2, .4, .4])
    outputs, detail = solve_variants(sources, base, 'fixture', 1)
    np.testing.assert_array_equal(outputs['PE_D4_LINEAGE'], outputs['PE_D3_DIAGONAL'])
    np.testing.assert_array_equal(outputs['PE_D4_LINEAGE'], outputs['PE_D4_SHUFFLED'])
    assert detail['D4_equals_D3']
    duplicated, _ = solve_variants(sources + [{**sources[0], 'name': 'alias'}], base, 'fixture', 1)
    np.testing.assert_array_equal(outputs['PE_D4_LINEAGE'], duplicated['PE_D4_LINEAGE'])
    assert len(collapse_sources(sources + [sources[0]])) == 2


def test_practical_bands_prefer_miou_then_cost_and_identity():
    rows = [
        {'id': 'strict_ap', 'metrics': {'apall': .1004, 'miou': .20, 'ap50': .3}, 'tier': 0, 'parameter': 0, 'registry_order': 0},
        {'id': 'better_miou', 'metrics': {'apall': .10, 'miou': .21, 'ap50': .3}, 'tier': 2, 'parameter': .5, 'registry_order': 1},
        {'id': 'cheap', 'metrics': {'apall': .10, 'miou': .2095, 'ap50': .3}, 'tier': 1, 'parameter': .25, 'registry_order': 2},
    ]
    assert choose(rows)['id'] == 'cheap'
