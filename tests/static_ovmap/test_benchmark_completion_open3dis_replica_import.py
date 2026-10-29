"""TEST ONLY sealed fixtures; these are never real benchmark evidence."""
import importlib
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from static_ovmap.benchmark_completion_v2.common import digest_file, source_identity, verified, write

VARIANT = 'Open3DIS (Replica port, ISBNet + Grounded SAM-HQ + refinement, author SPP branch)'


def candidate():
    path = REPO / 'src/static_ovmap/benchmark_completion_v2/open3dis_replica_import.py'
    assert path.exists(), 'Validated completed Replica pool importer is missing'
    return importlib.import_module('static_ovmap.benchmark_completion_v2.open3dis_replica_import')


def change(path, **updates):
    return write(path, dict(verified(path), **updates))


def fixture(root):
    scenes = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']
    protocol = write(root / 'protocol_lock.json', dict(TEST_ONLY=True,
        scopes=dict(replica48=dict(scenes=scenes, instance_names=['TEST_CLASS_' + str(i) for i in range(48)]))))
    sources = {}
    folder = root / 'external/Open3DIS'
    for scene in scenes:
        cache = folder / 'predictions/replica48' / scene / 'masks.npz'
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b'TEST ONLY opaque derived prediction bytes, no real GT')
        stages = {name: dict(actual_exit=name + '-actual', result=name + '-result', synchronized=name + '-sync')
                  for name in ['grounding_2d', 'lift3d', 'refine', 'final']}
        prediction = write(cache.with_name('manifest.json'), dict(status='DONE', method='Open3DIS',
            benchmark='replica48', scene=scene, target_points=3, prediction_identity='TEST_prediction_' + scene,
            arrays=dict(path=str(cache), bytes=cache.stat().st_size, sha256=digest_file(cache)),
            provenance=dict(method_variant=VARIANT, full_original_inference_stages=True,
                actual_four_stages=stages, GT_labels_used_for_prediction=False, new_training_updates=0,
                synthetic=False, evaluation_confidence='AUTHOR_inst_run_replica.py_CONSTANT_1',
                archive_confidence_field_used=False)))
        evaluation = folder / 'evaluation/replica48' / scene
        matches_path = evaluation / 'Open3DIS_matches.json.gz'
        matches = write(matches_path, dict(status='DONE', matches={scene: dict(gt={}, pred={})}))
        score = write(evaluation / 'receipt.json', dict(status='DONE', method='Open3DIS', benchmark='replica48',
            scene=scene, variant=VARIANT, protocol=protocol['identity'], full_GT_denominator=True,
            formal_G3_verified=True, synthetic=False, metric_unit='FRACTION', point_count=3,
            prediction=prediction['identity'], prediction_identity=prediction['prediction_identity'],
            matches=dict(path=str(matches_path), identity=matches['identity'])))
        write(evaluation / 'G3_published_raw_GT_agreement.json', dict(status='DONE', method='Open3DIS',
            benchmark='replica48', scene=scene, common_score=score['identity'], prediction=prediction['identity'],
            target_points=3, synthetic=False, formal_G3_verified=True, class_IoU_entries=480,
            AP_array_max_absolute_error=0, all_three_original_AP_arrays_equal=True,
            full_original_GT_denominator=True))
        sources[scene] = dict(score=score['identity'], prediction=prediction['identity'], matches=matches['identity'])
    pool_path = folder / 'pools/replica48.json'
    pool = write(pool_path, dict(status='DONE', method='Open3DIS', benchmark='replica48', variant=VARIANT,
        protocol=protocol['identity'], full_scene_coverage=True, formal_G3_verified=True, scope='FULL_STANDARD',
        completed_scenes=8, expected_scenes=8, ordered_scenes=scenes, missing_scenes=[], source_scenes=sources,
        new_training_updates=0, confidence='UNCHANGED_AUTHOR_VALUES', locally_executed=True,
        producer=source_identity(('open3dis_replica_score', 'replica_baseline_score')),
        aggregation='OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES;NO_SCENE_AP_AVERAGING',
        metrics=dict(apall=.12, ap50=.16, ap25=.18)))
    role = root / 'TEST_ONLY_original_supervisor'
    launch = write(role / 'launch.json', dict(command=['TEST_ONLY_no_execution'], source_sha256='TEST_SOURCE'))
    actual = write(role / 'actual_exit.json', dict(actual_child_exit=0, launch=launch['identity'],
        command=launch['command'], source_sha256=launch['source_sha256']))
    sync = write(role / 'synchronized.json', dict(status='DONE', actual_exit=actual['identity'],
        source_unchanged=True, source_after_sha256=launch['source_sha256']))
    completion = write(role / 'completion.json', dict(status='DONE', method='Open3DIS', benchmark='replica48',
        pool=pool['identity'], completed_scenes=8, expected_scenes=8, metrics=pool['metrics'],
        aggregation=pool['aggregation'], unchanged_original_aggregator=True, new_GPU_calls=0))
    write(pool_path.with_name('replica48_completion.json'), dict(status='DONE', method='Open3DIS',
        benchmark='replica48', pool=pool['identity'], completion=completion, launch=launch,
        actual=actual, synchronized=sync))
    return dict(output_root=str(root), asset_config=dict(external={})), pool_path, pool


def test_complete_pool_is_read_only_and_preserves_original_pooled_metrics(tmp_path):
    module = candidate()
    binding, _, expected = fixture(tmp_path)
    before = {p: (digest_file(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    assert module.pool(binding) == expected
    assert module.pool(binding)['metrics'] == dict(apall=.12, ap50=.16, ap25=.18)
    assert before == {p: (digest_file(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}


def test_absent_or_partial_pool_does_not_create_a_full_table_cell(tmp_path):
    module = candidate()
    assert module.pool(dict(output_root=str(tmp_path))) is None
    binding, path, _ = fixture(tmp_path)
    change(path, full_scene_coverage=False, completed_scenes=7, status='BLOCKED_DEPENDENCY')
    assert module.pool(binding) is None


@pytest.mark.parametrize('update', [dict(ordered_scenes=['room0'] * 8), dict(protocol='wrong'),
    dict(variant='ISBNet only'), dict(completed_scenes=7), dict(metrics=dict(apall=1.2, ap50=.16, ap25=.18))])
def test_incompatible_or_misdeclared_full_pool_is_rejected(tmp_path, update):
    module = candidate()
    binding, path, _ = fixture(tmp_path)
    change(path, **update)
    with pytest.raises(ValueError):
        module.pool(binding)


def test_changed_point_domain_or_raw_GT_agreement_cannot_enter_tables(tmp_path):
    module = candidate()
    binding, _, _ = fixture(tmp_path)
    path = tmp_path / 'external/Open3DIS/evaluation/replica48/room2/G3_published_raw_GT_agreement.json'
    change(path, target_points=2)
    with pytest.raises(ValueError):
        module.pool(binding)
    change(path, target_points=3, AP_array_max_absolute_error=.01)
    with pytest.raises(ValueError):
        module.pool(binding)


def test_changed_frozen_prediction_bytes_are_rejected(tmp_path):
    module = candidate()
    binding, _, _ = fixture(tmp_path)
    path = tmp_path / 'external/Open3DIS/predictions/replica48/room1/masks.npz'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        module.pool(binding)


def test_changed_match_statistics_are_rejected(tmp_path):
    module = candidate()
    binding, _, _ = fixture(tmp_path)
    path = tmp_path / 'external/Open3DIS/evaluation/replica48/office2/Open3DIS_matches.json.gz'
    change(path, matches=dict(wrong_scene={}))
    with pytest.raises(ValueError):
        module.pool(binding)


def test_pooled_metrics_must_match_original_completed_supervisor_proof(tmp_path):
    module = candidate()
    binding, path, _ = fixture(tmp_path)
    change(path, metrics=dict(apall=.13, ap50=.16, ap25=.18))
    with pytest.raises(ValueError):
        module.pool(binding)


def test_incomplete_certificate_never_imports_a_full_result(tmp_path):
    module = candidate()
    binding, path, _ = fixture(tmp_path)
    path.with_name('replica48_completion.json').unlink()
    assert module.pool(binding) is None


def test_nonzero_original_supervisor_exit_is_rejected(tmp_path):
    module = candidate()
    binding, path, _ = fixture(tmp_path)
    certificate_path = path.with_name('replica48_completion.json')
    cert = verified(certificate_path)
    actual = write(tmp_path / 'TEST_ONLY_changed_actual.json', dict(cert['actual'], actual_child_exit=1))
    change(certificate_path, actual=actual)
    with pytest.raises(ValueError):
        module.pool(binding)


def test_tables_include_both_benchmarks_without_replacing_scannet_result(tmp_path, monkeypatch):
    candidate()
    from static_ovmap.benchmark_completion_v2 import tables, open3dis_score
    binding, _, replica = fixture(tmp_path)
    scannet = dict(identity='SCANNET_TEST_POOL', full_scene_coverage=True, metrics=dict(apall=.23))
    monkeypatch.setattr(open3dis_score, 'pool', lambda active: scannet)
    results = {}
    tables.add_open3dis_result(binding, results)
    assert tables.external_cell(results, 'Open3DIS', 'scannet200_val', 'apall')['value'] == .23
    cell = tables.external_cell(results, 'Open3DIS', 'replica48', 'apall')
    assert cell['value'] == .12 and cell['results'] == ['external:Open3DIS:replica48']
    assert results['external:Open3DIS:replica48']['source'] == dict(
        path='external/Open3DIS/pools/replica48.json', identity=replica['identity'])


def test_baseline_recipe_recognizes_completed_full_replica_execution(tmp_path, monkeypatch):
    candidate()
    from static_ovmap.benchmark_completion_v2 import baselines, author_results, external_pooling, open3dis_score
    binding, _, expected = fixture(tmp_path)
    monkeypatch.setattr(baselines, 'PINS', {'Open3DIS': ('TEST_ONLY_REPOSITORY', 'TEST_ONLY_COMMIT')})
    monkeypatch.setattr(author_results, 'build', lambda active: dict(results={}))
    monkeypatch.setattr(external_pooling, 'ovi', lambda active: dict(benchmarks={}))
    monkeypatch.setattr(open3dis_score, 'pool', lambda active: None)
    method = baselines.run(binding)['methods']['Open3DIS']
    scope = method['scopes']['replica48']
    assert scope['status'] == 'DONE' and scope['coverage'] == 'LOCAL_FULL'
    assert scope['local_pool']['identity'] == expected['identity']
    assert scope['locally_inferred'] is True and scope['local_Replica_dataset_port_verified'] is True
    assert scope['variant'] == VARIANT and scope['missing'] == []
    assert method['full_method_not_run'] is False
