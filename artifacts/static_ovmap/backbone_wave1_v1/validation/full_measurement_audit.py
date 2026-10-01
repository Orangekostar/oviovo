import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001')
REPO = Path('/home/ww/crove/ovimap-backbone-wave1')
SPEC = REPO / 'docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


spec = read(SPEC)
binding = read(ROOT / 'resolved_inputs.json')
selection = read(ROOT / 'selection.json')
freeze = read(ROOT / 'transfer_freeze_commit.json')
development = spec['datasets']['development']
replica = spec['datasets']['replica']
recipes = selection['replica_recipes']
assert selection['replica_scene_order'] == replica
assert selection['Replica_results_read'] is False
assert selection['refit_after_Replica'] is False
assert selection['deployment'] == 'N0_UNCHANGED'
assert freeze['selection_identity'] == selection['identity']
assert binding['identity'] == '003c8dea36098e137e3bfbcadf334dfd23557df2c54c65a70369b3ab9df6d7ed'
expected = {(scene, arm['id']) for scene in development for arm in spec['map_variants']}
expected |= {(scene, arm['id']) for scene in replica for arm in recipes}
if any(arm['id'] == 'BBX_COMPOSE' for arm in recipes):
    expected |= {(scene, 'BBX_COMPOSE') for scene in development}
assert len(expected) == 64
maps = [read(path) for path in (ROOT / 'maps').glob('*/*/map_receipt.json')
        if '.failed_' not in path.parent.name]
assert len(maps) == len(expected)
assert {(row['scene'], row['map_id']) for row in maps} == expected
extensions = {}
q_totals = {'attempts': 0, 'successes': 0, 'failures': 0}
primary_rows = secondary_rows = 0
diagnostics = []
for mapping in maps:
    scene, map_id = mapping['scene'], mapping['map_id']
    assert mapping['status'] == 'COMPLETE'
    assert mapping['scheduled_count'] == mapping['completed_count'] == 200
    assert mapping['geometry_locked_before_semantics_and_labels'] is True
    assert mapping['new_visual_inference'] == 0
    assert mapping['native_build_identity'] == '486649a0beccd3c2b33e415cd0dbeb6699ffa93c98415a1749786d488ced17c3'
    assert mapping.get('post_update_owner_discrepancies', []) == []
    capture = read(mapping['capture_manifest'])
    parent = read(binding['scenes'][scene]['parent_capture'])
    assert capture['scheduled_frame_ids'] == binding['scenes'][scene]['schedule']
    assert capture['completed_frame_ids'] == parent['completed_frame_ids']
    assert [frame['frame_id'] for frame in capture['frames']] == parent['completed_frame_ids']
    assert capture['native_owner_mesh_parity']['exact'] is True
    extension = capture['native_extension']
    assert extension['sha256'] == 'dbf37faaaf6ea8bc49d4dd9cb7008935947cfde649adf66cb543c94e9b52a9da'
    extensions[extension['path']] = extension['sha256']
    readout_root = ROOT / 'readouts' / scene / map_id
    sources = read(readout_root / 'receipt.json')
    assert sources['scene'] == scene and sources['map_id'] == map_id
    nq = read(readout_root / 'native_query/native_query_receipt.json')
    assert nq['status'] == 'COMPLETE' and nq['Q_budget'] == 200
    assert nq['GT_input'] is False and nq['new_temperature_fit'] is False
    assert nq['query_logical_ledger']['attempts'] == 200
    assert nq['query_logical_ledger']['successes'] + nq['query_logical_ledger']['failures'] == 200
    for key in q_totals:
        q_totals[key] += nq['query_logical_ledger'][key]
    evaluation = read(readout_root / 'evaluation_rows.json')
    assert evaluation['status'] == 'COMPLETE'
    assert len(evaluation['rows']) == 6
    assert {(row['method'], row['rank_mode']) for row in evaluation['rows']} == {
        (method, rank) for method in spec['semantics']['readouts']
        for rank in ('OFFICIAL_CURRENT_CLASS', 'FROZEN_N0')}
    for row in evaluation['rows']:
        assert row['status'] == 'COMPLETE' and row['scene'] == scene and row['map_id'] == map_id
        assert Path(row['evaluation_receipt']).is_file()
        primary_rows += row['rank_mode'] == 'OFFICIAL_CURRENT_CLASS'
        secondary_rows += row['rank_mode'] == 'FROZEN_N0'
    raw = read(readout_root / 'raw_geometry_diagnostics.json')
    diagnostics.append({'scene': scene, 'map_id': map_id, 'identity': raw.get('identity')})
assert primary_rows == secondary_rows == 192
for path, expected_sha in extensions.items():
    assert sha256(path) == expected_sha
frontends = []
for scene in development + replica:
    frontend = read(ROOT / 'frontend' / scene / 'SAM2_PAIRED/receipt.json')
    assert frontend['status'] == 'COMPLETE'
    assert frontend['code_commit'] == '2b90b9f5ceec907a1c18123530e92e794ad901a4'
    assert frontend['chunk_valid_frames'] == 5 and frontend['neural_predictions_shared'] is True
    assert frontend['counters']['physical_image_encodings'] == 200
    assert [frame['frame_id'] for frame in frontend['frames']] == binding['scenes'][scene]['parent_completed_frame_ids']
    for index, frame in enumerate(frontend['frames']):
        assert frame['chunk'] == index // 5 and frame['current_index'] == index % 5
        if frame['current_index'] == 0:
            assert frame['diagnostic']['state'] == 'EXACT_CHUNK_FIRST'
            assert frame['outputs']['raw']['sha256'] == frame['outputs']['geom']['sha256']
    frontends.append({'scene': scene, 'identity': frontend['identity'], 'image_encodings': 200})
pools = []
dev_ids = [arm['id'] for arm in spec['map_variants']]
if any(arm['id'] == 'BBX_COMPOSE' for arm in recipes):
    dev_ids.append('BBX_COMPOSE')
for cohort, order, map_ids in (('development', development, dev_ids),
                             ('replica', replica, [arm['id'] for arm in recipes])):
    for map_id in map_ids:
        for method in spec['semantics']['readouts']:
            rank_views = []
            for rank in ('OFFICIAL_CURRENT_CLASS', 'FROZEN_N0'):
                path = ROOT / 'pools' / cohort / map_id / method / (rank + '.json')
                value = read(path)
                assert value['status'] == 'COMPLETE' and value['scene_order'] == order
                assert value['aggregation'] == 'RELEASED_DATASET_POOL'
                assert len(value['ordered_inputs']) == len(order)
                rank_views.append(value)
                pools.append(str(path.relative_to(ROOT)))
            assert all(rank_views[0]['metrics'][key] == rank_views[1]['metrics'][key]
                       for key in ('miou', 'macc'))
assert len(pools) == 72
for leaf in ('candidate_freeze.json', 'selection.json'):
    committed = subprocess.check_output(
        ['git', 'show', freeze['commit'] + ':' + spec['publication']['repo_artifacts'] + '/' + leaf],
        cwd=REPO, text=True)
    assert json.loads(committed) == read(ROOT / leaf)
candidate_time = (ROOT / 'candidate_freeze.json').stat().st_mtime
for path in (ROOT / 'maps').glob('scene*/BBX_COMPOSE/running.json'):
    assert candidate_time < read(path)['started_at_unix']
freeze_commit_time = int(subprocess.check_output(
    ['git', 'show', '-s', '--format=%ct', freeze['commit']], cwd=REPO, text=True).strip())
for scene in replica:
    for path in (ROOT / 'maps' / scene).glob('*/running.json'):
        assert freeze_commit_time < read(path)['started_at_unix']
transfer_runs = [read(path) for path in (ROOT / 'execution').glob('transfer_*.json')
                 if '.resources.' not in path.name and '.console.' not in path.name]
complete_transfer = [run for run in transfer_runs if run.get('phase') == 'transfer' and run['status'] == 'COMPLETE']
assert complete_transfer and all(not run['failures'] for run in complete_transfer)
assert read(ROOT / 'bridge_parity.json')['status'] == 'VERIFIED'
receipt = {
    'status': 'FULL_MEASUREMENTS_VERIFIED_PUBLICATION_PENDING',
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'review_owner': 'primary_agent',
    'script_sha256': sha256(__file__),
    'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'successful_maps': len(maps), 'primary_rows': primary_rows, 'secondary_rows': secondary_rows,
    'ordered_official_and_secondary_pools': len(pools), 'raw_geometry_diagnostics': len(diagnostics),
    'Q_logical_totals': q_totals, 'SAM2_production_image_encodings': sum(row['image_encodings'] for row in frontends),
    'loaded_native_extensions': extensions, 'selection_identity': selection['identity'],
    'selection_freeze_commit': freeze['commit'], 'all_scenes_exposed': True,
    'frontends': frontends, 'pools': pools,
    'publication_verification_is_external': True,
}
(ROOT / 'review/full_measurement_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: value for key, value in receipt.items() if key not in ('frontends', 'pools')}))
