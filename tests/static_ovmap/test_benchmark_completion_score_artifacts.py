from pathlib import Path

import numpy as np


def test_classification_scores_freezes_mf_off_cosines_without_loading_sources():
    from static_ovmap.benchmark_completion_v2.score_artifacts import classification_scores

    valid_ids = np.array([1, 2])
    raw = np.array([-.1, .2])
    calibrated = np.array([.5, -.3])
    temperatures = dict(F=1., N=1., Q=1.)

    def forbidden_source_loader(name):
        raise AssertionError(f'unexpected source load: {name}')

    np.testing.assert_array_equal(
        classification_scores('U00', valid_ids, raw, calibrated,
                              forbidden_source_loader, temperatures), raw)
    np.testing.assert_array_equal(
        classification_scores('U01', valid_ids, raw, calibrated,
                              forbidden_source_loader, temperatures), calibrated)
    assert classification_scores('U00', valid_ids, None, calibrated,
                                 forbidden_source_loader, temperatures) is None
    assert classification_scores('U01', valid_ids, raw, None,
                                 forbidden_source_loader, temperatures) is None


def test_classification_scores_uses_grouped_probabilities_and_renormalizes_missing_groups():
    from static_ovmap.benchmark_completion_v2.score_artifacts import classification_scores

    valid_ids = np.array([1, 2])
    f = np.array([np.log(3), 0.])
    n = np.array([0., np.log(3)])
    q = np.array([0., 0.])
    temperatures = dict(F=1., N=1., Q=1.)

    def source_loader(name):
        return {'N': n, 'Q': q}[name]

    for variant in ('U10', 'U11'):
        np.testing.assert_allclose(
            classification_scores(variant, valid_ids, f, f, source_loader, temperatures),
            [.5625, .4375], rtol=1e-12, atol=1e-12)

    def no_f_source_loader(name):
        return {'N': n, 'Q': q}[name]

    np.testing.assert_allclose(
        classification_scores('U10', valid_ids, None, None, no_f_source_loader, temperatures),
        [.375, .625], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        classification_scores('U11', valid_ids, None, None, no_f_source_loader, temperatures),
        [.375, .625], rtol=1e-12, atol=1e-12)

    def no_q_source_loader(name):
        return n if name == 'N' else None

    np.testing.assert_allclose(
        classification_scores('U10', valid_ids, f, f, no_q_source_loader, temperatures),
        [.5, .5], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        classification_scores('U11', valid_ids, f, f, no_q_source_loader, temperatures),
        [.5, .5], rtol=1e-12, atol=1e-12)

    def no_source_loader(name):
        return None

    assert classification_scores('U10', valid_ids, None, None,
                                 no_source_loader, temperatures) is None
    assert classification_scores('U11', valid_ids, None, None,
                                 no_source_loader, temperatures) is None


def test_publication_candidates_allowlist_class_score_artifacts(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates

    allowed = [
        tmp_path / 'publication/class_scores.json',
        tmp_path / 'publication/class_scores/test_identity/manifest.json',
        tmp_path / 'publication/class_scores/test_identity/scores.npz',
    ]
    for path in allowed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    (tmp_path / 'publication/class_scores/test_identity/raw_GT.json').write_bytes(b'fixture')
    (tmp_path / 'publication/class_scores/test_identity/points.ply').write_bytes(b'fixture')

    assert {path.relative_to(tmp_path) for path in candidates(tmp_path)} == {
        Path('publication/class_scores.json'),
        Path('publication/class_scores/test_identity/manifest.json'),
        Path('publication/class_scores/test_identity/scores.npz'),
    }
