"""Recovery uses synthetic identities; these are not formal benchmark results."""
from copy import deepcopy
from pathlib import Path

import pytest

from static_ovmap.benchmark_completion_v2.common import (
    canonical_digest, digest_file, source_identity, verified, write,
)


def fixture(tmp_path, extra_taxonomy='', base_ids=None):
    from static_ovmap.benchmark_completion_v2.common import ConsumedInputs
    from static_ovmap.benchmark_completion_v2.protocol import _aliases, map_training_roles
    old = tmp_path / 'old.tsv'
    new = tmp_path / 'new.tsv'
    header = 'raw_category\tcategory\twnsynsetid\tid\n'
    old.write_text(header + 'chair\tchair\tn1\t1\nseat\tunknown\t\t2\n' + extra_taxonomy)
    new.write_text(header + 'chair\tchair\tn1\t1\nseat\tchair\tn1\t2\n' + extra_taxonomy)
    base_ids = [1] if base_ids is None else base_ids
    classes = dict(official_ids=[1, 3], official_names=['chair', 'table'], base_ids=base_ids)
    class_path = tmp_path / 'class_split.json'
    write(class_path, classes)
    index = ConsumedInputs(tmp_path)
    index.identity(class_path)
    aliases, descriptor = _aliases(old, classes['official_ids'], classes['official_names'], index)
    scope = dict(status='DONE', scenes=['room0'], text_ids=[1], text_names=['seat'],
                 instance_ids=[1], instance_names=['seat'], full_AC_available=False,
                 mapping=map_training_roles(['seat'], [1, 3], ['chair', 'table'], base_ids, aliases))
    scan = dict(scope, scenes=['scene0'], text_ids=[1, 3], text_names=['chair', 'table'],
                mapping=map_training_roles(['chair', 'table'], [1, 3], ['chair', 'table'], base_ids),
                full_AC_available=True)
    lock = write(tmp_path / 'protocol_lock.json', dict(status='DONE', producer=source_identity(('protocol',)),
        official_sources={}, semantic_alias_source=descriptor,
        scopes=dict(replica48=scope, scannet200_val=scan),
        legacy=dict(replica51=deepcopy(scope['mapping']), old51_text_names=['seat']),
        input_identity='original', component_roles={}, overlap={}))
    binding = dict(output_root=str(tmp_path), path_map={}, structure_root=None,
                   structure_documents={'class_split.json': str(class_path)},
                   asset_config=dict(datasets=dict(scannet200_val=dict(label_mapping_path=str(new)))))
    index.save()
    return binding, lock, old, new


def test_alias_refresh_schedules_protocol_and_legacy_descendants():
    from static_ovmap.benchmark_completion_v2.refresh import affected_phases
    old = dict(datasets=dict(scannet200_val=dict(label_mapping_path='/old.tsv')))
    new = dict(datasets=dict(scannet200_val=dict(label_mapping_path='/new.tsv')))
    phases = affected_phases(old, new)
    assert phases[0] == 'protocol'
    assert {'legacy_predict', 'legacy_score', 'benchmark_predict', 'benchmark_score'} <= set(phases)
    assert 'cache_post' not in phases and 'method_lock' not in phases


def role_manifest(binding, tmp_path, ids, *, manual=False, base_ids=None):
    """Explicit synthetic crosswalk; no benchmark labels or scores."""
    csv_path = tmp_path / 'crosswalk.csv'
    csv_path.write_text('id,replicaclass,replicaid\n' + ''.join(str(i) + ',seat,1\n' for i in ids))
    taxonomy = Path(binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'])
    manifest = write(tmp_path / 'roles.json', dict(
        policy='ANY_BASE_IN_CONFIRMED_SYNONYM_GROUP', GT_labels_used=False,
        training_axis=dict(ids=[1, 3], base_ids=[1] if base_ids is None else base_ids),
        taxonomy=dict(path=str(taxonomy), sha256=digest_file(taxonomy)),
        crosswalk=dict(path=str(csv_path), sha256=digest_file(csv_path)),
        user_confirmation='都为同义' if manual else None,
        rows=[dict(target_name='seat', semantic_ids=ids,
                   user_confirmed_synonyms=manual)]))
    binding['asset_config']['datasets']['replica48'] = dict(training_role_aliases_path=str(tmp_path / 'roles.json'))
    return verified(tmp_path / 'protocol_lock.json'), manifest


@pytest.mark.parametrize('ids,manual,base_ids,expected', [([2], False, [1], False),
    ([1], False, [1], True), ([1, 3], False, [1, 3], True), ([1, 3], True, [1], True)])
def test_explicit_global_or_confirmed_synonym_roles_extend_without_changing_training_axis(tmp_path, ids, manual, base_ids, expected):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, _, _, _ = fixture(tmp_path, 'table\ttable\t\t3\n', base_ids=base_ids)
    lock, manifest = role_manifest(binding, tmp_path, ids, manual=manual, base_ids=base_ids)
    # Keep the original alias unresolved so this tests the explicit manifest.
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(tmp_path / 'old.tsv')
    manifest['taxonomy'] = dict(path=str(tmp_path / 'old.tsv'), sha256=digest_file(tmp_path / 'old.tsv'))
    write(tmp_path / 'roles.json', manifest)
    result = prepare_protocol(binding, lock, tmp_path / 'history/role_update')
    assert result['scopes']['replica48']['mapping']['base_mask'] == [expected]
    assert result['legacy']['replica51']['base_mask'] == [expected]
    assert result['scopes']['scannet200_val'] == lock['scopes']['scannet200_val']
    row = result['scopes']['replica48']['mapping']['rows'][0]
    assert row['source_semantic_ids'] == ids
    assert row['candidate_training_ids'] == [i for i in ids if i in (1, 3)]
    assert row['role_evidence']['manual_synonym_group'] is manual
    from static_ovmap.benchmark_completion_v2.semantic_refresh import check_alias
    check_alias(binding, result)


def test_a_declared_manifest_does_not_turn_an_unmatched_target_into_novel(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, _, _, _ = fixture(tmp_path, 'table\ttable\t\t3\n')
    lock, manifest = role_manifest(binding, tmp_path, [2])
    manifest['rows'] = []
    write(tmp_path / 'roles.json', manifest)
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(tmp_path / 'old.tsv')
    manifest['taxonomy'] = dict(path=str(tmp_path / 'old.tsv'), sha256=digest_file(tmp_path / 'old.tsv'))
    write(tmp_path / 'roles.json', manifest)
    result = prepare_protocol(binding, lock, tmp_path / 'history/role_update')
    assert result['scopes']['replica48']['mapping']['base_mask'] is None
    assert result['scopes']['replica48']['mapping']['unresolved'] == ['seat']


def test_crosswalk_can_identify_target_by_official_replica_id_when_name_is_blank(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, _, _, _ = fixture(tmp_path, 'table\ttable\t\t3\n')
    lock, manifest = role_manifest(binding, tmp_path, [2, 3])
    path = Path(manifest['crosswalk']['path'])
    # Raw Replica taxonomy IDs and the evaluator's reindexed text IDs differ.
    path.write_text('id,replicaclass,replicaid\n2,"seat,other","65,99"\n3,,65\n1,other,99\n')
    manifest['crosswalk']['sha256'] = digest_file(path)
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(tmp_path / 'old.tsv')
    manifest['taxonomy'] = dict(path=str(tmp_path / 'old.tsv'), sha256=digest_file(tmp_path / 'old.tsv'))
    write(tmp_path / 'roles.json', manifest)
    result = prepare_protocol(binding, lock, tmp_path / 'history/role_update')
    assert result['scopes']['replica48']['mapping']['base_mask'] == [False]


def test_role_manifest_change_requires_explicit_refresh_even_when_tsv_is_identical(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import check_alias
    binding, _, _, _ = fixture(tmp_path, 'table\ttable\t\t3\n')
    lock, _ = role_manifest(binding, tmp_path, [2])
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(tmp_path / 'old.tsv')
    with pytest.raises(ValueError, match='refresh-assets'):
        check_alias(binding, lock)


@pytest.mark.parametrize('invalid', ['source_bytes', 'unknown_id', 'unconfirmed_conflict', 'changed_axis'])
def test_explicit_role_extension_fails_closed_on_invalid_evidence(tmp_path, invalid):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, _, _, _ = fixture(tmp_path, 'table\ttable\t\t3\n')
    lock, manifest = role_manifest(binding, tmp_path, [1, 3], manual=True)
    if invalid == 'source_bytes':
        Path(manifest['crosswalk']['path']).write_text('tampered')
    elif invalid == 'unknown_id':
        manifest['rows'][0]['semantic_ids'] = [999]
    elif invalid == 'unconfirmed_conflict':
        manifest['rows'][0]['user_confirmed_synonyms'] = False
    else:
        manifest['training_axis']['base_ids'] = [3]
    write(tmp_path / 'roles.json', manifest)
    before = digest_file(tmp_path / 'protocol_lock.json')
    with pytest.raises(ValueError):
        prepare_protocol(binding, lock, tmp_path / 'history/role_update')
    assert digest_file(tmp_path / 'protocol_lock.json') == before


def test_protocol_dispatch_rejects_undeclared_alias_change(tmp_path):
    from static_ovmap.benchmark_completion_v2.runner import dispatch
    binding, _, _, _ = fixture(tmp_path)
    before = digest_file(tmp_path / 'protocol_lock.json')
    with pytest.raises(ValueError, match='refresh-assets'):
        dispatch('protocol', binding, None)
    assert digest_file(tmp_path / 'protocol_lock.json') == before


def test_refresh_resolves_only_evidence_backed_roles_and_preserves_old_lock(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, lock, _, _ = fixture(tmp_path)
    old_bytes = (tmp_path / 'protocol_lock.json').read_bytes()
    result = prepare_protocol(binding, lock, tmp_path / 'history/update')
    assert result['scopes']['replica48']['mapping']['base_mask'] == [True]
    assert result['legacy']['replica51']['base_mask'] == [True]
    assert result['scopes']['scannet200_val']['mapping']['base_mask'] == [True, False]
    assert (tmp_path / 'protocol_lock.json').read_bytes() == old_bytes
    assert verified(result['semantic_refresh']['previous_protocol']['path']) == lock
    assert result['semantic_refresh']['changed_scopes'] == ['replica48']
    assert result['semantic_refresh']['legacy_roles_changed']
    assert not result['GT_semantic_labels_read']


@pytest.mark.parametrize('change', ['gt', 'roles', 'old_bytes'])
def test_refresh_rejects_scientific_change_before_live_write(tmp_path, change):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, lock, old, new = fixture(tmp_path)
    if change == 'gt':
        new.write_text(new.read_text().replace('seat\tchair\tn1\t2', 'seat\tchair\tn1\t3'))
    elif change == 'roles':
        path = binding['structure_documents']['class_split.json']
        write(path, dict(official_ids=[1, 3], official_names=['chair', 'table'], base_ids=[3]))
    else:
        old.write_text(old.read_text() + 'altered\tchair\tn1\t4\n')
    before = digest_file(tmp_path / 'protocol_lock.json')
    with pytest.raises(ValueError):
        prepare_protocol(binding, lock, tmp_path / 'history/update')
    assert digest_file(tmp_path / 'protocol_lock.json') == before


def test_unchanged_standard_scope_reuses_exact_freeze_without_models(tmp_path, monkeypatch):
    import numpy as np
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol, resume_standard_scene
    from static_ovmap.benchmark_completion_v2.standard_export import save_manifest
    binding, old, _, _ = fixture(tmp_path)
    methods = {}
    folder = tmp_path / 'benchmark/predictions/scannet200_val/scene0'
    for variant in ('U00', 'U10', 'U01', 'U11'):
        for seed in (17, 29):
            name = variant + '_seed' + str(seed)
            manifest = save_manifest(folder / name, scene='scene0', benchmark='scannet200_val',
                method=variant, seed=seed, owners=[9], indptr=[0, 2], indices=[0, 1], labels=[1],
                target_points=2, valid_instance_ids=[1], provenance=dict(protocol=old['identity']))
            methods[name] = dict(identity=manifest['identity'], path=str(folder / name / 'manifest.json'))
    freeze = write(folder / 'receipt.json', dict(status='DONE', scene='scene0', benchmark='scannet200_val',
        methods=methods, completed_rows=8, expected_rows=8))
    updated = prepare_protocol(binding, old, tmp_path / 'history/update')
    write(tmp_path / 'protocol_lock.json', updated)
    before = {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}
    from static_ovmap.benchmark_completion_v2 import benchmark
    monkeypatch.setattr(benchmark, 'predict_scene', lambda *a: pytest.fail('unexpected model path'))
    assert resume_standard_scene(binding, 'scannet200_val', 'scene0') == freeze
    assert before == {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}
    npz = Path(next(iter(methods.values()))['path']).with_name('masks.npz')
    with npz.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='array'):
        resume_standard_scene(binding, 'scannet200_val', 'scene0')


def test_partial_standard_grid_adds_only_calibrated_exports_and_rebinds_full_gt(tmp_path, monkeypatch):
    import numpy as np
    from static_ovmap.benchmark_completion_v2 import benchmark, runtime
    from static_ovmap.benchmark_completion_v2.common import save_arrays
    from static_ovmap.benchmark_completion_v2.protocol import map_training_roles
    from static_ovmap.benchmark_completion_v2.semantic_refresh import (
        prepare_protocol, prepare_standard_score, resume_standard_scene,
    )
    from static_ovmap.benchmark_completion_v2.standard_export import save_manifest
    binding, old, _, _ = fixture(tmp_path)
    scope = old['scopes']['replica48']
    scope.update(text_ids=[1, 2], text_names=['seat', 'table'], instance_ids=[1, 2], instance_names=['seat', 'table'])
    scope['mapping'] = map_training_roles(scope['text_names'], [1, 3], ['chair', 'table'], [1])
    old = write(tmp_path / 'protocol_lock.json', old)
    def array_file(name, **arrays):
        path = tmp_path / (name + '.npz')
        save_arrays(path, **arrays)
        return dict(path=str(path), sha256=digest_file(path), bytes=path.stat().st_size)
    geometry = array_file('geometry', owners=[7, 9], indptr=[0, 2, 4], indices=[0, 1, 1, 2], target_xyz=np.zeros((3, 3)))
    bundle = write(tmp_path / 'benchmark/bundles/replica48/room0/bundle.json', dict(
        arrays=dict(geometry=geometry), evaluation_geometry=dict(identity='full-three-points'), model_lock='weights'))
    text = np.eye(2, 768, dtype=np.float32)
    fa = array_file('F', b=np.vstack([text[0], text[0]]),
        z=np.tile(np.vstack([text[1], text[0]])[None], (2, 1, 1)),
        text=text, owners=[7, 9], available=[True, False], valid_ids=[1, 2])
    f = write(tmp_path / 'benchmark/F/replica48/room0/receipt.json', dict(arrays=fa))
    sources = dict(N={'7': dict(available=True, scores=[0., 1.]), '9': dict(available=True, scores=[1., 0.])},
                   Q={'7': dict(available=False, scores=None), '9': dict(available=False, scores=None)})
    nq = write(tmp_path / 'benchmark/NQ/replica48/room0/sources.json', dict(bundle=bundle['identity'], sources=sources))
    folder = tmp_path / 'benchmark/predictions/replica48/room0'
    methods = {}
    for variant in ('U00', 'U10'):
        for seed in (17, 29):
            name = variant + '_seed' + str(seed)
            manifest = save_manifest(folder / name, scene='room0', benchmark='replica48', method=variant, seed=seed,
                owners=[7, 9], indptr=[0, 2, 4], indices=[0, 1, 1, 2], labels=[2, 0 if variant == 'U00' else 1],
                target_points=3, valid_instance_ids=[1, 2], provenance=dict(protocol=old['identity'],
                    bundle=bundle['identity'], model_lock=bundle['model_lock'], F=f['identity'],
                    NQ=nq['identity'] if variant == 'U10' else None))
            methods[name] = dict(path=str(folder / name / 'manifest.json'), identity=manifest['identity'])
    freeze = write(folder / 'receipt.json', dict(status='BLOCKED_DEPENDENCY', scene='room0', benchmark='replica48',
        bundle=bundle['identity'], methods=methods, traces={}, completed_rows=4, expected_rows=8,
        all_available_predictions_frozen_before_new_GT_scoring=True))
    gt_file = tmp_path / 'benchmark/GT/replica48/room0/room0.txt'
    gt_file.parent.mkdir(parents=True)
    gt_file.write_text('1001\n1001\n2001\n')
    gt_path = gt_file.with_name('receipt.json')
    write(gt_path, dict(prediction_freeze=freeze['identity'], geometry=bundle['evaluation_geometry'],
        GT=dict(path=str(gt_file), sha256=digest_file(gt_file)), full_GT_denominator=True))
    score_path = tmp_path / 'benchmark/evaluation/replica48/room0/receipt.json'
    score = write(score_path, dict(prediction_lock=freeze['identity'], rows={'raw': 'unchanged'}))
    original_gt = verified(gt_path)
    raw_manifest = verified(next(iter(methods.values()))['path'])
    ovi_manifest = write(tmp_path / 'external/OVI-MAP/predictions/replica48/room0/manifest.json',
        dict(arrays=raw_manifest['arrays'], synthetic=True))
    ovi_matches_path = tmp_path / 'external/OVI-MAP/evaluation/replica48/room0/matches.json.gz'
    ovi_matches = write(ovi_matches_path, dict(matches={}, synthetic=True))
    ovi_score_path = ovi_matches_path.with_name('receipt.json')
    ovi_producer = source_identity(('ovi_adapter', 'standard_score'))
    write(ovi_score_path, dict(GT=original_gt['identity'], prediction=ovi_manifest['identity'],
        input_identity=canonical_digest(dict(prediction=ovi_manifest['identity'], GT=original_gt['identity'], producer=ovi_producer)),
        matches=dict(path=str(ovi_matches_path), identity=ovi_matches['identity']), metrics={'apall': .123}))
    ovi_matches_before = digest_file(ovi_matches_path), ovi_matches_path.stat().st_mtime_ns
    raw_before = {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file() and p.name != 'receipt.json'}
    gt_before = digest_file(gt_file), gt_file.stat().st_mtime_ns
    candidate = prepare_protocol(binding, old, tmp_path / 'history/update')
    write(tmp_path / 'protocol_lock.json', candidate)
    monkeypatch.setattr(benchmark, 'template_context', lambda *a: dict(temperatures=dict(N=1., Q=1., F=1.)))
    monkeypatch.setattr(benchmark, 'native_query', lambda *a: pytest.fail('unexpected N/Q worker'))
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: pytest.fail('unexpected model worker'))
    result = resume_standard_scene(binding, 'replica48', 'room0')
    assert result['completed_rows'] == 8 and result['status'] == 'DONE'
    assert raw_before == {p: (digest_file(p), Path(p).stat().st_mtime_ns) for p in raw_before}
    for seed in (17, 29):
        with np.load(folder / ('U01_seed' + str(seed)) / 'masks.npz') as arrays:
            assert arrays['labels'].tolist() == [1, 0]
            assert arrays['owners'].tolist() == [7, 9]
            assert arrays['indices'].tolist() == [0, 1, 1, 2]
        with np.load(folder / ('U11_seed' + str(seed)) / 'masks.npz') as arrays:
            assert arrays['labels'][1] == 1
        assert result['traces']['U01_seed' + str(seed)]['NQ_access_count'] == 0
        assert result['traces']['U11_seed' + str(seed)]['NQ_access_count'] == 4
    prepare_standard_score(binding, 'replica48', 'room0')
    assert verified(gt_path)['prediction_freeze'] == result['identity']
    assert (digest_file(gt_file), gt_file.stat().st_mtime_ns) == gt_before
    assert verified(ovi_score_path)['GT'] == verified(gt_path)['identity']
    assert verified(ovi_score_path)['metrics'] == {'apall': .123}
    assert verified(ovi_score_path)['semantic_refresh']['new_scorer_calls'] == 0
    assert (digest_file(ovi_matches_path), ovi_matches_path.stat().st_mtime_ns) == ovi_matches_before
    assert not score_path.exists()
    snapshots = list((tmp_path / 'history/semantic_resume').rglob('evaluation/benchmark/evaluation/replica48/room0/receipt.json'))
    assert len(snapshots) == 1 and verified(snapshots[0]) == score
    before = {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}
    assert resume_standard_scene(binding, 'replica48', 'room0') == result
    prepare_standard_score(binding, 'replica48', 'room0')
    assert before == {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}


def test_legacy_score_refresh_preserves_unaffected_metrics_and_opens_only_extended_scene(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_legacy_score
    binding, old, _, _ = fixture(tmp_path)
    write(tmp_path / 'protocol_lock.json', dict(old, semantic_refresh=dict(authorized=True)))
    methods = ['L00_seed17', 'L01_seed17']
    unchanged = write(tmp_path / 'legacy/predictions/cf/receipt.json', dict(
        status='DONE', scene='cf', cohort='scannet_cf18', methods={m: dict(prediction_key=m, record_key=m) for m in methods}))
    before_replica = write(tmp_path / 'history/old_replica_freeze.json', dict(
        status='DONE', scene='room0', cohort='replica8', methods={methods[0]: dict(prediction_key=methods[0], record_key=methods[0])}))
    extended = write(tmp_path / 'legacy/predictions/room0/receipt.json', dict(before_replica,
        methods={m: dict(prediction_key=m, record_key=m) for m in methods}))
    global_lock = write(tmp_path / 'legacy/prediction_lock.json', dict(cohorts=dict(replica8=['room0'], scannet_cf18=['cf']),
        scenes={'room0': extended['identity'], 'cf': unchanged['identity']}, methods=methods,
        expected_logical_scene_rows=4, completed_logical_scene_rows=4,
        all_available_predictions_frozen_before_new_GT_scoring=True))
    producer = source_identity(('scoring',))
    old_receipts = {}
    for scene, freeze in [('cf', unchanged), ('room0', before_replica)]:
        rows = [dict(method=method, prediction_lock=freeze['identity'], global_prediction_lock='old-global',
                     prediction_identity=method, record_identity=method, metrics={'apall': .123}) for method in freeze['methods']]
        old_receipts[scene] = write(tmp_path / 'legacy/evaluation' / scene / 'receipt.json', dict(status='DONE', rows=rows,
            input_identity=canonical_digest(dict(global_predictions='old-global', scene=freeze['identity'], producer=producer))))
    score = write(tmp_path / 'legacy/score.json', dict(input_identity='old-input', metrics={'apall': .456}))
    prepare_legacy_score(binding)
    reused = verified(tmp_path / 'legacy/evaluation/cf/receipt.json')
    assert reused['rows'] == old_receipts['cf']['rows']
    assert reused['semantic_refresh']['new_scorer_calls'] == 0
    assert reused['input_identity'] == canonical_digest(dict(global_predictions=global_lock['identity'],
        scene=unchanged['identity'], producer=producer))
    assert not (tmp_path / 'legacy/evaluation/room0/receipt.json').exists()
    assert not (tmp_path / 'legacy/score.json').exists()
    history = tmp_path / 'history/semantic_resume' / global_lock['identity'] / 'legacy_score'
    assert verified(history / 'global/legacy/score.json') == score
    assert verified(history / 'room0/legacy/evaluation/room0/receipt.json') == old_receipts['room0']
    before = digest_file(tmp_path / 'legacy/evaluation/cf/receipt.json')
    prepare_legacy_score(binding)
    assert digest_file(tmp_path / 'legacy/evaluation/cf/receipt.json') == before


def test_explicit_refresh_recovers_interruption_after_binding_write(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import binding as binding_module
    from static_ovmap.benchmark_completion_v2.refresh import bind_assets
    from static_ovmap.benchmark_completion_v2.runner import parser
    binding, protocol, old_tsv, _ = fixture(tmp_path)
    old_assets = dict(datasets=dict(scannet200_val=dict(label_mapping_path=str(old_tsv))))
    previous = write(tmp_path / 'binding.json', dict(binding, asset_config=old_assets,
        asset_identity=canonical_digest(old_assets), public_A_identity='original-A', gpu='0'))
    assets_path = tmp_path / 'new_assets.json'
    import json
    assets_path.write_text(json.dumps(binding['asset_config']))
    args = parser().parse_args(['--output-root', str(tmp_path), '--asset-config', str(assets_path), '--refresh-assets'])
    calls = []
    def actual_write_then_interrupt(args):
        result = write(tmp_path / 'binding.json', dict(previous, asset_config=binding['asset_config'],
            asset_identity=canonical_digest(binding['asset_config'])))
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('simulated interruption after atomic binding write')
        return result
    monkeypatch.setattr(binding_module, 'bind', actual_write_then_interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        bind_assets(args)
    assert (tmp_path / 'asset_refresh_pending.json').is_file()
    bind_assets(args)
    refreshed = verified(tmp_path / 'protocol_lock.json')
    assert refreshed['identity'] != protocol['identity']
    assert refreshed['legacy']['replica51']['base_mask'] == [True]
    assert not (tmp_path / 'asset_refresh_pending.json').exists()
    receipt = verified(tmp_path / 'asset_refresh.json')
    assert {'protocol', 'legacy_predict', 'legacy_score'} <= set(receipt['affected_phases'])
    assert receipt['old_binding'] == previous['identity']
    assert verified(Path(receipt['history']) / 'binding.json') == previous


def test_alias_relocation_accepts_only_same_old_bytes_through_explicit_path_map(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, lock, old, new = fixture(tmp_path)
    moved = tmp_path / 'relocated.tsv'
    moved.write_bytes(old.read_bytes())
    binding['path_map'] = {str(old): str(moved)}
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(moved)
    old.unlink()
    candidate = prepare_protocol(binding, lock, tmp_path / 'history/relocation')
    assert candidate['semantic_alias_source']['sha256'] == lock['semantic_alias_source']['sha256']
    assert candidate['semantic_refresh']['changed_scopes'] == []
    assert not candidate['semantic_refresh']['legacy_roles_changed']


def test_replica_alias_refresh_is_independent_of_missing_scannet_evaluator(tmp_path):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, lock, _, _ = fixture(tmp_path)
    lock['scopes']['scannet200_val'] = dict(status='BLOCKED_DEPENDENCY', missing=['evaluator'])
    lock = write(tmp_path / 'protocol_lock.json', lock)
    candidate = prepare_protocol(binding, lock, tmp_path / 'history/independent')
    assert candidate['scopes']['replica48']['mapping']['base_mask'] == [True]
    assert candidate['scopes']['scannet200_val'] == lock['scopes']['scannet200_val']


def test_legacy_gate_interruption_preserves_raw_exports_caches_and_other_cohort(tmp_path, monkeypatch):
    import numpy as np
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_legacy_prediction, prepare_protocol
    from static_ovmap.module_validation.evaluation import GeometryIdentity, PredictionPayload
    from static_ovmap.module_validation.scannet_study import save_prediction
    binding, protocol, _, _ = fixture(tmp_path)
    protocol = prepare_protocol(binding, protocol, tmp_path / 'history/update')
    write(tmp_path / 'protocol_lock.json', protocol)
    parent = tmp_path / 'parent'
    source = write(parent / 'regression/room0/manifest.json', dict(objects=[], synthetic=True))
    context = tmp_path / 'context.json'
    write(context, dict(FC_text=dict(sha256='text-sha')))
    binding['inherited'] = dict(lr_parent_root=str(parent), cohorts=dict(replica8=['room0'], scannet_cf18=['cf']),
        scenes=dict(room0=dict(context=dict(path=str(context)))))
    model = write(tmp_path / 'model_lock.json', dict(weights={'17': {'source_sha256': 'seed17'}, '29': {'source_sha256': 'seed29'}}))
    methods = {}
    for variant in ('L00', 'L10'):
        for seed in (17, 29):
            name = variant + '_seed' + str(seed)
            payload = PredictionPayload(name, 'COMBO', 'room0', GeometryIdentity(*(['0' * 64] * 3), 'fixed', 2),
                np.array([7, 7]), np.array([1, 1]), ((7, 1.),), {}, dict(synthetic=True))
            payload.lock()
            path = save_prediction(payload, tmp_path / 'legacy/predictions/room0' / name)
            methods[name] = dict(manifest=str(path), prediction_key=payload.prediction_key, record_key=payload.record_key)
    scene_path = tmp_path / 'legacy/predictions/room0/receipt.json'
    scene = write(scene_path, dict(status='DONE', scene='room0', methods=methods,
        input_identity=canonical_digest(dict(producer=source_identity(('legacy', 'region_inputs')), models=model['weights'],
            scene=source['identity'], text='text-sha', variants=('L00', 'L10'), base_mask=None))))
    cf_path = tmp_path / 'legacy/predictions/cf/receipt.json'
    write(cf_path, dict(sentinel='unchanged other cohort'))
    cache = tmp_path / 'readout_cache/original.npz'
    cache.parent.mkdir()
    cache.write_bytes(b'existing-readout-not-opened-by-lifecycle')
    global_path = tmp_path / 'legacy/prediction_lock.json'
    write(global_path, dict(scenes={'room0': scene['identity']}))
    untouched = [cache, cf_path] + [p for p in scene_path.parent.rglob('*') if p.is_file() and p != scene_path]
    before = {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in untouched}
    unlink = Path.unlink
    def interrupt_global(path, *args, **kwargs):
        if path == global_path:
            raise RuntimeError('interrupted after scene gate')
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupt_global)
        with pytest.raises(RuntimeError, match='interrupted after scene gate'):
            prepare_legacy_prediction(binding)
    assert not scene_path.exists() and global_path.is_file()
    prepare_legacy_prediction(binding)
    assert not global_path.exists()
    assert before == {str(p): (digest_file(p), p.stat().st_mtime_ns) for p in untouched}
    snapshots = list((tmp_path / 'history/semantic_resume').rglob('room0/legacy/predictions/room0/receipt.json'))
    assert len(snapshots) == 1 and verified(snapshots[0]) == scene


def test_missing_optional_alias_does_not_block_canonical_protocol_or_common_model(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import runner
    from static_ovmap.benchmark_completion_v2.semantic_refresh import check_alias
    binding, protocol, _, _ = fixture(tmp_path)
    protocol['semantic_alias_source'] = None
    protocol = write(tmp_path / 'protocol_lock.json', protocol)
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(tmp_path / 'not-yet-available.tsv')
    check_alias(binding, protocol)
    monkeypatch.setattr(runner, '_model_lock', lambda _: dict(status='DONE', common_weights=True))
    assert runner.dispatch('method_lock', binding, None)['common_weights']


def test_reporting_can_collect_frozen_results_when_alias_source_is_unavailable(tmp_path, monkeypatch):
    from static_ovmap.benchmark_completion_v2 import runner, tables
    binding, _, _, _ = fixture(tmp_path)
    binding['asset_config']['datasets']['scannet200_val']['label_mapping_path'] = str(tmp_path / 'missing.tsv')
    monkeypatch.setattr(tables, 'build', lambda _: dict(status='DONE', frozen_results_only=True))
    assert runner.dispatch('tables', binding, None)['frozen_results_only']


@pytest.mark.parametrize('existing_gt', [False, True])
def test_first_alias_asset_is_accepted_only_before_any_scannet_gt_receipt(tmp_path, existing_gt):
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_protocol
    binding, lock, _, _ = fixture(tmp_path)
    lock = write(tmp_path / 'protocol_lock.json', dict(lock, semantic_alias_source=None))
    if existing_gt:
        write(tmp_path / 'benchmark/GT/scannet200_val/scene0/receipt.json', dict(frozen=True))
        with pytest.raises(ValueError, match='GT'):
            prepare_protocol(binding, lock, tmp_path / 'history/first_alias')
    else:
        result = prepare_protocol(binding, lock, tmp_path / 'history/first_alias')
        assert result['scopes']['replica48']['mapping']['base_mask'] == [True]
        assert result['semantic_refresh']['no_prior_ScanNet_GT_receipts']
