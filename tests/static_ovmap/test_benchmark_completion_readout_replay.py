"""Export must carry the inputs needed to reproduce frozen readout labels."""
import io
from pathlib import Path

import numpy as np
import pytest

from static_ovmap.benchmark_completion_v2.common import write, digest_file, ARTIFACTS
from static_ovmap.benchmark_completion_v2.publication_export import export, decode_record


def replay_fixture(tmp_path):
    out, repo, parent = (tmp_path/name for name in ('out', 'repo', 'parent'))
    repo.mkdir()
    parent.mkdir()
    text = np.eye(2, 768, dtype=np.float64)
    text_path = parent/'text.npz'
    np.savez_compressed(text_path, text_embeddings=text, valid_ids=np.array([1, 2]))
    text_desc = dict(path=str(text_path), bytes=text_path.stat().st_size, sha256=digest_file(text_path))
    n = write(parent/'N.json', dict(objects={'7': dict(available=True, scores=[.4, 0.])}, valid_ids=[1, 2]))
    q = write(parent/'Q.json', dict(objects={'7': dict(available=False, scores=None)}, valid_ids=[1, 2]))
    scenes = {}
    for scene, n_temperature in [('template', .1), ('legacy_scene', 10.)]:
        path = parent/(scene+'.json')
        write(path, dict(FC_text=text_desc, sources={
            'N': dict(path=str(parent/'N.json'), identity=n['identity']),
            'Q': dict(path=str(parent/'Q.json'), identity=q['identity'])},
            temperatures=dict(F=1., N=n_temperature, Q=.2),
            GT_input='/private/raw/GT.txt', raw_files=['/private/raw/color.jpg']))
        scenes[scene] = dict(context=dict(path=str(path), bytes=path.stat().st_size, sha256=digest_file(path)))
    write(out/'benchmark/predictions/scannet200_val/scene1/receipt.json',
          dict(status='DONE', GLOBAL=True, methods={}))
    legacy = write(out/'legacy/predictions/legacy_scene/receipt.json', dict(status='DONE', methods={}))
    write(out/'legacy/prediction_lock.json', dict(scenes={'legacy_scene': legacy['identity']}))
    cache = out/'readout_cache/actual.npz'
    cache.parent.mkdir()
    np.savez_compressed(cache, b=text[1].astype(np.float32), z=text[1].astype(np.float32))
    write(cache.with_suffix('.json'), dict(arrays=dict(path=str(cache), sha256=digest_file(cache))))
    write(out/'A/import.json', dict(class_split=dict(base_ids=[1], heldout_ids=[2])))
    np.savez_compressed(out/'A/import.npz', scores=np.array([[[[0., 1.]]]], np.float32),
                        defined=np.ones((1, 1, 1), bool), embeddings=text[None, 1:2, None],
                        text=text, valid_ids=np.array([1, 2]))
    binding = dict(output_root=str(out), path_map={}, inherited=dict(
        scenes=scenes, cohorts=dict(scannet_cf18=['template'])))
    return binding, repo, parent


def unpack(record, repo):
    result = {}
    for row in record['files']:
        payload = b''.join((repo/part['path']).read_bytes() for part in row['parts'])
        result[row['source']] = (decode_record(payload)['record']
                                if row['encoding']=='GZIP_PATH_REDACTED_RECORD_V1' else payload)
    return result


def test_public_shards_reproduce_labels_with_original_per_scene_temperatures(tmp_path):
    # Omitting the b/z, original text or scene-specific temperatures breaks replay.
    from static_ovmap.benchmark_completion_v2.standard_export import classify_scores
    binding, repo, parent = replay_fixture(tmp_path)
    record = export(binding, repo=repo)
    files = unpack(record, repo)
    assert 'readout_cache/actual.npz' in files
    replay = files['publication/readout_replay.json']
    standard = replay['standard']['scannet200_val']
    legacy = replay['legacy']['legacy_scene']
    assert standard['temperatures'] == dict(F=1., N=.1, Q=.2)
    with np.load(io.BytesIO(files['readout_cache/actual.npz']), allow_pickle=False) as a:
        z = a['z'].copy()
    with np.load(io.BytesIO(files[legacy['text']['source']]), allow_pickle=False) as a:
        raw, ids = a['text_embeddings'] @ z, a['valid_ids'].copy()
        assert a['text_embeddings'].dtype == np.float64
    def source(name):
        row = files[legacy['sources'][name]['source']]['objects']['7']
        return row['scores'] if row['available'] else None
    assert classify_scores('U10', ids, raw, None, source, standard['temperatures'])['label'] == 1
    assert classify_scores('U10', ids, raw, None, source, legacy['temperatures'])['label'] == 2
    assert 'A/import.npz' in files and 'A/import.json' in files
    assert 'GT_input' not in str(replay) and 'raw_files' not in str(replay)
    assert not any('template.json' in name or '/GT' in name for name in files)
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in parent.iterdir()}
    replay_path = Path(binding['output_root'])/'publication/readout_replay.json'
    stamp = replay_path.stat().st_mtime_ns
    assert export(binding, repo=repo)['identity'] == record['identity']
    assert replay_path.stat().st_mtime_ns == stamp
    assert original == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in parent.iterdir()}


@pytest.mark.parametrize('damage', ['changed_context', 'missing_context', 'changed_scores'])
def test_changed_or_missing_replay_input_cannot_replace_valid_publication(tmp_path, damage):
    # Declared input identities must be enforced before a new public index is written.
    binding, repo, parent = replay_fixture(tmp_path)
    export(binding, repo=repo)
    index = repo/ARTIFACTS/'artifact_index.json'
    previous = index.read_bytes(), index.stat().st_mtime_ns
    if damage == 'missing_context':
        (parent/'template.json').unlink()
    elif damage == 'changed_context':
        write(parent/'template.json', dict(temperatures=dict(F=1., N=1., Q=1.)))
    else:
        write(parent/'N.json', dict(objects={'7': dict(available=True, scores=[0., .4])}, valid_ids=[1, 2]))
    with pytest.raises((ValueError, FileNotFoundError)):
        export(binding, repo=repo)
    assert (index.read_bytes(), index.stat().st_mtime_ns) == previous
