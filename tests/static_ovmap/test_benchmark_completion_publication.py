"""Bounded publication checks; the Git remote here is a temporary local bare repo."""
from pathlib import Path
import subprocess
import pytest


def test_public_export_redacts_paths_excludes_raw_gt_and_reassembles_chunks(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import export, decode_record
    from static_ovmap.benchmark_completion_v2.common import write, digest_file
    out, repo = tmp_path/'local-private', tmp_path/'repo'
    repo.mkdir()
    write(out/'result_store.json', dict(status='DONE', synthetic=True,
        private_path=str(out/'benchmark/F/vector.npz'), model='/home/other/private/model.pt'))
    raw=out/'benchmark/GT/scan.txt'
    raw.parent.mkdir(parents=True)
    raw.write_text('unpublishable-raw-GT')
    source=out/'benchmark/F/scannet200_val/scene1/vectors.npz'
    source.parent.mkdir(parents=True)
    source.write_bytes(bytes(range(64)))
    binding=dict(output_root=str(out), semantic_root='/private/semantic', structure_root='/private/structure',
                 inherited={}, interpreters={})
    record=export(binding,repo=repo,blob_limit=32,git_limit=64*1024,chunk_size=16)
    item=next(row for row in record['files'] if row['source']=='benchmark/F/scannet200_val/scene1/vectors.npz')
    reconstructed=b''.join((repo/part['path']).read_bytes() for part in item['parts'])
    assert reconstructed == source.read_bytes()
    assert item['sha256'] == digest_file(source)
    assert not any('benchmark/GT' in row['source'] for row in record['files'])
    wrapped=next(row for row in record['files'] if row['source']=='result_store.json')
    payload=b''.join((repo/part['path']).read_bytes() for part in wrapped['parts'])
    actual=decode_record(payload)
    assert actual['record']['private_path'] == '$OUT/benchmark/F/vector.npz'
    assert actual['record']['model'] == '$LOCAL_ASSET/model.pt'
    assert actual['original_identity'] == __import__('json').loads((out/'result_store.json').read_text())['identity']
    assert record['git_total_derived_bytes'] <= 64*1024


def test_export_overflow_preserves_local_parts_and_total_git_limit(tmp_path):
    import hashlib
    from static_ovmap.benchmark_completion_v2.publication_export import export
    from static_ovmap.benchmark_completion_v2.common import ARTIFACTS
    repo,out=tmp_path/'repo',tmp_path/'local'
    repo.mkdir()
    path=out/'benchmark/F/scannet200_val/scene1/vectors.npz'
    path.parent.mkdir(parents=True)
    payload=b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(20000))
    path.write_bytes(payload)
    record=export(dict(output_root=str(out)),repo=repo,blob_limit=40*1024,git_limit=128*1024,chunk_size=32*1024)
    assert record['git_total_derived_bytes']<=128*1024
    restored=b''
    for part in record['files'][0]['parts']:
        loc=repo/part['path'] if part['distribution']=='GIT' else out/'publication/release_assets'/part['path']
        restored+=loc.read_bytes()
    assert restored==payload and path.read_bytes()==payload
    assert any(p['distribution']=='LOCAL_RETAINED_RELEASE_PENDING' for p in record['files'][0]['parts'])
    assert sum(p.stat().st_size for p in (repo/ARTIFACTS).rglob('*') if p.is_file())<=128*1024


def test_release_urls_are_included_in_final_git_budget(tmp_path):
    import hashlib
    from static_ovmap.benchmark_completion_v2.publication_export import export, finalize_index
    from static_ovmap.benchmark_completion_v2.common import ARTIFACTS
    repo,out=tmp_path/'repo',tmp_path/'local'
    repo.mkdir()
    path=out/'benchmark/F/scannet200_val/scene1/vectors.npz'
    path.parent.mkdir(parents=True)
    payload=b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(10000))
    path.write_bytes(payload)
    binding=dict(output_root=str(out))
    record=export(binding,repo=repo,git_limit=128*1024,chunk_size=32*1024)
    for row in record['files']:
        for part in row['parts']:
            if part['distribution']!='GIT':
                part.update(distribution='GITHUB_RELEASE',url='https://example.invalid/'+('a'*6000))
    final=finalize_index(binding,record,repo=repo)
    actual=sum(p.stat().st_size for p in (repo/ARTIFACTS).rglob('*') if p.is_file())
    assert actual==final['git_total_derived_bytes']<=128*1024
    restored=b''.join((repo/part['path'] if part['distribution']=='GIT'
                      else out/'publication/release_assets'/Path(part['path']).name).read_bytes()
                     for part in final['files'][0]['parts'])
    assert restored==payload and path.read_bytes()==payload


def test_two_commits_read_new_result_from_remote_and_resume_without_commit(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication import publish_transaction
    from static_ovmap.benchmark_completion_v2.common import digest_file
    repo, remote, out = tmp_path/'repo', tmp_path/'remote.git', tmp_path/'out'
    repo.mkdir()
    def git(*args, cwd=repo):
        return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
    git('init','--bare',str(remote))
    git('init','-b','research/ovmap-benchmark-completion-v2')
    git('config','user.email','synthetic@example.invalid')
    git('config','user.name','Synthetic test')
    git('remote','add','origin',str(remote))
    (repo/'base.txt').write_text('base\n')
    git('add','base.txt')
    git('commit','-m','synthetic base')
    artifact='artifacts/result.json'
    (repo/'artifacts').mkdir()
    (repo/artifact).write_text('{"synthetic":true,"measured":false}\n')
    (repo/'HANDOFF.md').write_text('Synthetic publication integration check.\n')
    source_hash=digest_file(repo/artifact)
    args=dict(repo=repo,out=out,files=[artifact,'HANDOFF.md'],
              receipt='publication/C1.json',handoff='HANDOFF.md',release_identity='synthetic-test',
              result_files=[artifact])
    result=publish_transaction(**args)
    assert result['status'] == 'DONE'
    assert result['C1'] != result['C2']
    assert git('rev-parse','refs/heads/research/ovmap-benchmark-completion-v2',cwd=remote) == result['C2']
    changed=set(git('diff-tree','--no-commit-id','--name-only','-r',result['C2']).splitlines())
    assert changed == {'publication/C1.json','HANDOFF.md'}
    assert digest_file(repo/artifact) == source_hash
    assert result['remote_result_read']['sha256'] == source_hash
    before=git('rev-list','--count','HEAD')
    assert publish_transaction(**args)['C2'] == result['C2']
    assert git('rev-list','--count','HEAD') == before


"""Isolated transport fixtures; no real publication, credentials, or science."""
import hashlib
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from static_ovmap.benchmark_completion_v2.common import StageFailure

from static_ovmap.benchmark_completion_v2 import publication as candidate


class PublicReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='public-reader-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.name = 'artifacts/new result.json'
        self.commit = '1'*40
        self.payload = b'{"synthetic":true,"measured":false}\n'
        path = self.repo/self.name
        path.parent.mkdir()
        path.write_bytes(self.payload)
        self.expected_url = ('https://api.github.com/repos/example/fixture/contents/'
                             'artifacts/new%20result.json?ref='+'1'*40)
        patcher = patch.object(candidate, 'git', return_value='git@github.com:example/fixture.git')
        patcher.start()
        self.addCleanup(patcher.stop)

    def response(self, payload):
        def open_request(request, timeout):
            self.assertEqual(request.full_url, self.expected_url)
            self.assertEqual(request.get_header('Accept'), 'application/vnd.github.raw+json')
            self.assertEqual(timeout, 30)
            return io.BytesIO(payload)
        return open_request

    def test_public_object_is_verified_without_gh(self):
        with patch.object(candidate.shutil, 'which', return_value=None), \
             patch('urllib.request.urlopen', side_effect=self.response(self.payload)):
            try:
                result = candidate._read_remote_result(self.repo, self.commit, self.name)
            except StageFailure as error:
                self.fail('Missing gh must allow the verified public HTTP read: '+str(error))
        self.assertEqual(result, dict(path=self.name, commit=self.commit,
            sha256=hashlib.sha256(self.payload).hexdigest(), bytes=len(self.payload),
            mode='GITHUB_CONTENTS_RAW_AT_COMMIT_PUBLIC_HTTP'))
        self.assertEqual((self.repo/self.name).read_bytes(), self.payload)

    def test_wrong_public_bytes_are_never_accepted(self):
        with patch.object(candidate.shutil, 'which', return_value=None), \
             patch('urllib.request.urlopen', side_effect=self.response(b'wrong remote bytes')):
            with self.assertRaises(ValueError):
                try:
                    candidate._read_remote_result(self.repo, self.commit, self.name)
                except StageFailure as error:
                    self.fail('Byte verification was not reached: '+str(error))
        self.assertEqual((self.repo/self.name).read_bytes(), self.payload)

    def test_unavailable_public_object_preserves_dependency_failure(self):
        with patch.object(candidate.shutil, 'which', return_value=None), \
             patch('urllib.request.urlopen', side_effect=urllib.error.URLError('isolated offline fixture')):
            with self.assertRaises(StageFailure) as caught:
                candidate._read_remote_result(self.repo, self.commit, self.name)
        self.assertEqual(caught.exception.status, 'BLOCKED_DEPENDENCY')
        self.assertEqual(caught.exception.missing, [self.name])

    def test_existing_gh_reader_still_checks_exact_bytes(self):
        child = SimpleNamespace(returncode=0, stdout=self.payload)
        with patch.object(candidate.shutil, 'which', return_value='/fixture/gh'), \
             patch.object(candidate.subprocess, 'run', return_value=child):
            result = candidate._read_remote_result(self.repo, self.commit, self.name)
        self.assertEqual(result['sha256'], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(result['mode'], 'GITHUB_CONTENTS_RAW_AT_COMMIT')

    def test_existing_gh_failure_stays_blocked(self):
        with patch.object(candidate.shutil, 'which', return_value='/fixture/gh'), \
             patch.object(candidate.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(StageFailure) as caught:
                candidate._read_remote_result(self.repo, self.commit, self.name)
        self.assertEqual(caught.exception.missing, [self.name])


@pytest.mark.parametrize('numeric_parts_available', [True, False])
def test_publish_includes_renderer_and_capacity_scripts_in_c1_and_keeps_c2_scoped(tmp_path, monkeypatch, numeric_parts_available):
    """Actual two-commit collection in a temporary bare remote; synthetic assets only."""
    from static_ovmap.benchmark_completion_v2 import lifecycle, publication_export
    from static_ovmap.benchmark_completion_v2.common import ARTIFACTS, DOCS, digest_file, write, verified

    repo, remote, out = tmp_path/'repo', tmp_path/'remote.git', tmp_path/'out'
    repo.mkdir()

    def git(*args, cwd=repo):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

    git('init', '--bare', str(remote))
    git('init', '-b', candidate.BRANCH)
    git('config', 'user.email', 'synthetic@example.invalid')
    git('config', 'user.name', 'Synthetic test')
    git('remote', 'add', 'origin', str(remote))
    (repo/'base.txt').write_text('base\n')
    git('add', 'base.txt')
    git('commit', '-m', 'synthetic base')

    renderer = 'scripts/evaluation/render_ovmap_benchmark_completion_supplement.py'
    capacity_scripts = ['scripts/evaluation/monitor_ovmap_benchmark_capacity.py',
                        'scripts/evaluation/test_ovmap_benchmark_capacity_guard.py']
    parallel_delivery = ['scripts/evaluation/run_ovmap_parallel_scenes.py',
                         'scripts/evaluation/test_ovmap_parallel_scenes.py',
                         'docs/superpowers/plans/2026-10-23-ovmap-parallel-scenes.md']
    unrelated = 'scripts/evaluation/unrelated_fixture.py'
    required = [
        'src/static_ovmap/benchmark_completion_v2/fixture.py',
        str(DOCS/'HANDOFF.md'), str(DOCS/'RESULTS.md'),
        'configs/static_ovmap/benchmark_completion_v2.json',
        'scripts/evaluation/run_ovmap_benchmark_completion_v2.py', renderer, *capacity_scripts, *parallel_delivery,
        'docs/superpowers/plans/2026-10-22-ovmap-benchmark-completion-v2.md',
        'tests/static_ovmap/test_benchmark_completion_fixture.py',
    ]
    for name in required + [unrelated]:
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic publication fixture; no model or data.\n')
    renderer_sha = digest_file(repo/renderer)
    capacity_hashes = {name: digest_file(repo/name) for name in capacity_scripts + parallel_delivery}
    import gzip
    import hashlib
    from static_ovmap.benchmark_completion_v2.common import StageFailure
    parent = git('rev-parse', 'HEAD')
    store_value = dict(status='DONE', synthetic=True, metrics=dict(APall=0.12))
    store = write(out/'result_store.json', store_value)
    rows = []
    expected_result_parts = []

    def add_record(source, value, *, split=False, local=False):
        original = write(out/source, value)
        original_path = out/source
        payload = gzip.compress(json.dumps(dict(schema='PATH_REDACTED_RECORD_V1',
            original_identity=original['identity'], original_sha256=digest_file(original_path),
            record=original), sort_keys=True).encode(), mtime=0)
        pieces = [payload] if not split else [payload[:len(payload)//2], payload[len(payload)//2:]]
        parts = []
        offset = 0
        for piece in pieces:
            sha = hashlib.sha256(piece).hexdigest()
            dest = (out/'publication/release_assets' if local else repo/ARTIFACTS/'chunks')/(sha+'.part')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(piece)
            parts.append(dict(path=dest.name if local else str(dest.relative_to(repo)),
                distribution='LOCAL_RETAINED_RELEASE_PENDING' if local else 'GIT',
                bytes=len(piece), sha256=sha, offset=offset))
            offset += len(piece)
        rows.append(dict(source=source, sha256=digest_file(original_path),
                         encoding='GZIP_PATH_REDACTED_RECORD_V1', payload_sha256=hashlib.sha256(payload).hexdigest(), parts=parts))
        return parts

    for source, value in [('result_store.json', store_value),
            ('benchmark/evaluation/scannet200_val/synthetic_scene/receipt.json',
             dict(status='DONE', synthetic=True, metrics=dict(APall=0.34)))]:
        parts = add_record(source, value, local=not numeric_parts_available)
        if numeric_parts_available:
            expected_result_parts.append(parts[0]['path'])
    add_record('review/G4_fixture_metadata.json', dict(status='DONE', synthetic=True, note='metadata_only'))
    add_record('benchmark/evaluation/scannet200_val/split_scene/receipt.json',
               dict(status='DONE', synthetic=True, metrics=dict(APall=0.56)), split=True,
               local=not numeric_parts_available)
    add_record('benchmark/evaluation/scannet200_val/local_scene/receipt.json',
               dict(status='DONE', synthetic=True, metrics=dict(APall=0.78)), local=True)
    index = write(repo/ARTIFACTS/'artifact_index.json', dict(status='DONE', synthetic=True, files=rows))
    write(out/'remaining_jobs.json', dict(jobs=[]))
    write(out/'review/primary_final.json', dict(requirements_review_complete=True,
        reviewed_producer_identity='synthetic-producer', result_store_identity=store['identity']))

    # Real source collection and Git transaction; only the science/export boundary is synthetic.
    monkeypatch.setattr(candidate, 'REPO', repo)
    monkeypatch.setattr(lifecycle, 'producer_identity', lambda: 'synthetic-producer')
    def export_fixture(binding):
        assert binding['output_root'] == str(out)
        return index
    monkeypatch.setattr(publication_export, 'export', export_fixture)
    monkeypatch.setattr(candidate, 'release_overflow', lambda binding, record: record)

    if not numeric_parts_available:
        with pytest.raises(StageFailure, match='complete numerical'):
            candidate.publish(dict(output_root=str(out)), None)
        assert git('rev-parse', 'HEAD') == parent
        assert not (out/'publication/transaction.json').exists()
        return
    result = candidate.publish(dict(output_root=str(out)), None)
    c1_files = set(git('ls-tree', '-r', '--name-only', result['C1']).splitlines())
    assert renderer in c1_files, 'Documented supplementary renderer is missing from the actual C1 commit'
    assert set(capacity_scripts) <= c1_files, 'Capacity monitor and regression source must be present in actual C1'
    assert set(parallel_delivery) <= c1_files, 'Parallel scene entry, regression and plan must be present in actual C1'
    assert set(required) <= c1_files
    assert unrelated not in c1_files
    assert git('show', result['C1']+':'+renderer) == (repo/renderer).read_text().strip()
    state = verified(out/'publication/transaction.json')
    assert next(row for row in state['files'] if row['path'] == renderer)['sha256'] == renderer_sha
    for name, expected in capacity_hashes.items():
        assert git('show', result['C1']+':'+name) == (repo/name).read_text().strip()
        assert next(row for row in state['files'] if row['path'] == name)['sha256'] == expected
    changed = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', result['C2']).splitlines())
    assert changed == {str(DOCS/'PUBLICATION_RECEIPT.json'), str(DOCS/'HANDOFF.md')}
    assert git('rev-parse', 'refs/heads/'+candidate.BRANCH, cwd=remote) == result['C2']
    assert state['result_files'] == sorted(expected_result_parts)
    chosen = result['remote_result_read']['path']
    assert chosen in expected_result_parts
    assert result['remote_result_read']['sha256'] == digest_file(repo/chosen)
    remote_payload = subprocess.check_output(['git', 'show', result['C1']+':'+chosen], cwd=remote)
    assert remote_payload == (repo/chosen).read_bytes()
    restored = json.loads(gzip.decompress(remote_payload))['record']
    assert restored['synthetic'] and restored['metrics']['APall'] in (0.12, 0.34)
    before = git('rev-list', '--count', 'HEAD')
    assert candidate.publish(dict(output_root=str(out)), None)['C2'] == result['C2']
    assert git('rev-list', '--count', 'HEAD') == before
    assert digest_file(repo/renderer) == renderer_sha


import json
import unittest


class TestPublicExportPathPrivacy(unittest.TestCase):
    def setUp(self):
        from static_ovmap.benchmark_completion_v2.publication_export import redact
        self.redact = redact
        self.binding = dict(output_root='/mnt/shared/private-output', semantic_root='/mnt/shared/semantic',
                            structure_root='/home/ww/structure')

    def test_nested_dictionary_keys_and_secrets(self):
        value = {'files': {'/mnt/shared/private-output/scene/receipt.json': {'api_token': 'sensitive',
                  'path': '/home/ww/structure/model.npz'}}}
        actual = self.redact(value, self.binding)
        self.assertEqual(actual, {'files': {'$OUT/scene/receipt.json': {'api_token': 'REDACTED',
                          'path': '$STRUCTURE_ROOT/model.npz'}}})
        self.assertNotIn('/mnt/shared', json.dumps(actual))

    def test_colon_separated_roots_after_known_root(self):
        value = '/home/ww/structure/lib:/home/ww/private-env/lib:/mnt/shared/another/lib'
        actual = self.redact(value, self.binding)
        self.assertEqual(actual, '$STRUCTURE_ROOT/lib:$LOCAL_ASSET/lib:$LOCAL_ASSET/lib')

    def test_unknown_path_keys_preserve_all_entries_with_stable_aliases(self):
        original = {'/home/ww/other/a.npz': 1, '/mnt/shared/another/a.npz': 2}
        actual = self.redact(original, self.binding)
        self.assertEqual(len(actual), 2)
        self.assertEqual(set(actual.values()), {1, 2})
        self.assertEqual(self.redact(dict(reversed(list(original.items()))), self.binding), actual)
        self.assertEqual(self.redact(actual, self.binding), actual)
        self.assertNotIn('/home/ww', json.dumps(actual))
        self.assertNotIn('/mnt/shared', json.dumps(actual))

    def test_conflicting_preexisting_alias_is_rejected(self):
        import hashlib
        key = '/home/ww/other/a.npz'
        alias = '$LOCAL_ASSET/a.npz#' + hashlib.sha256(key.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'collid'):
            self.redact({key: 1, alias: 2}, self.binding)

    def test_compiler_include_and_library_flags(self):
        for flag in ('-I', '-L', '-B', '-isystem', '-iquote', '-idirafter', '-isysroot'):
            with self.subTest(flag=flag):
                self.assertEqual(self.redact(flag + '/home/ww/private/include', self.binding),
                                 flag + '$LOCAL_ASSET/include')

    def test_public_URL_is_preserved(self):
        value = 'https://example.org/home/public/file?download=1'
        self.assertEqual(self.redact(value, self.binding), value)

    def test_redaction_is_idempotent_and_preserves_numbers(self):
        value = {'/mnt/shared/private-output/x': [1, 2.5, None, True, '/home/ww/private/lib']}
        actual = self.redact(value, self.binding)
        self.assertEqual(self.redact(actual, self.binding), actual)
        self.assertEqual(list(actual.values())[0][:4], [1, 2.5, None, True])


def test_export_keeps_complete_numeric_records_in_git_when_vectors_overflow(tmp_path):
    import hashlib
    from static_ovmap.benchmark_completion_v2.publication_export import export, decode_record
    from static_ovmap.benchmark_completion_v2.common import ARTIFACTS, write, digest_file
    repo, out = tmp_path/'repo', tmp_path/'out'
    repo.mkdir()
    vector = out/'benchmark/F/scannet200_val/synthetic_scene/vectors.npz'
    vector.parent.mkdir(parents=True)
    vector.write_bytes(b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(20000)))
    values = {'result_store.json': 0.42,
              'benchmark/evaluation/scannet200_val/synthetic_scene/receipt.json': 0.31}
    for source, ap in values.items():
        write(out/source, dict(status='DONE', synthetic=True, metrics=dict(APall=ap)))
    before = {source: digest_file(out/source) for source in values}
    vector_sha = digest_file(vector)
    index = export(dict(output_root=str(out)), repo=repo, blob_limit=40*1024,
                   git_limit=128*1024, chunk_size=32*1024)
    for source, ap in values.items():
        row = next(item for item in index['files'] if item['source']==source)
        assert len(row['parts'])==1 and row['parts'][0]['distribution']=='GIT'
        restored = decode_record((repo/row['parts'][0]['path']).read_bytes())['record']
        assert restored['synthetic'] and restored['metrics']['APall']==ap
        assert digest_file(out/source)==before[source]
    assert digest_file(vector)==vector_sha
    vector_row = next(item for item in index['files'] if item['source'].endswith('/vectors.npz'))
    assert any(part['distribution']=='LOCAL_RETAINED_RELEASE_PENDING' for part in vector_row['parts'])
    actual = sum(path.stat().st_size for path in (repo/ARTIFACTS).rglob('*') if path.is_file())
    assert actual==index['git_total_derived_bytes']<=128*1024


def test_export_preserves_official_external_match_statistics_and_excludes_raw_gt(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates, export, decode_record
    from static_ovmap.benchmark_completion_v2.common import write, digest_file
    out, repo = tmp_path/'local', tmp_path/'repo'
    repo.mkdir()
    sources = []
    for method in ('OVI-MAP', 'OpenMask3D', 'MaskClustering'):
        benchmark = 'scannet200_val' if method == 'OVI-MAP' else 'replica48'
        path = out/f'external/{method}/evaluation/{benchmark}/scene1/matches.json.gz'
        write(path, dict(status='DONE', synthetic=True, matches={'scene1': {
            'gt': {'chair': [{'vert_count': 12}]}, 'pred': {}}}))
        sources.append(path)
    own = out/'benchmark/evaluation/scannet200_val/scene1/U11_seed17_matches.json.gz'
    write(own, dict(status='DONE', synthetic=True, matches={'scene1': {}}))
    sources.append(own)
    private = [out/'benchmark/GT/scene1/scan.txt',
               out/'external/OVI-MAP/GT/scene1/raw.npy',
               out/'external/OVI-MAP/evaluation/scannet200_val/scene1/raw_GT.npy']
    for path in private:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unpublishable-raw-GT')
    before = {path: (digest_file(path), path.stat().st_mtime_ns) for path in sources+private}
    selected = set(candidates(out))
    assert set(sources) <= selected, 'Completed official external matches must enter the derived export'
    assert not set(private) & selected
    record = export(dict(output_root=str(out)), repo=repo, git_limit=1024*1024)
    for path in sources:
        item = next(row for row in record['files'] if row['source'] == str(path.relative_to(out)))
        assert all(part['distribution'] == 'GIT' for part in item['parts'])
        payload = b''.join((repo/part['path']).read_bytes() for part in item['parts'])
        restored = decode_record(payload)
        import gzip, json
        original = json.loads(gzip.decompress(path.read_bytes()))
        assert restored['original_identity'] == original['identity']
        assert restored['record']['matches'] == original['matches']
    assert not any(row['source'] in {str(path.relative_to(out)) for path in private} for row in record['files'])
    assert {path: (digest_file(path), path.stat().st_mtime_ns) for path in sources+private} == before


def _execution_provenance_fixture(out):
    from static_ovmap.benchmark_completion_v2.common import write, digest_file
    script, log, receipt = (out/'bootstrap'/name for name in ('accepted.py', 'accepted.log', 'accepted_actual_exit.json'))
    script.parent.mkdir(parents=True)
    script.write_text("OUTPUT_ROOT = " + repr(str(out)) + "\n")
    log.write_text('Executed ' + str(script) + '\n')
    actual = write(receipt, dict(status='DONE', actual_child_exit=0, command=['python', str(script)],
        script_sha256=digest_file(script), log=dict(path=str(log), sha256=digest_file(log)), api_token='synthetic-secret'))
    files = [dict(source=str(path.relative_to(out)), kind=kind,
        sha256=digest_file(path), bytes=path.stat().st_size, **extra)
        for path, kind, extra in ((script, 'PYTHON_SOURCE', {}), (log, 'EXECUTION_LOG', {}),
            (receipt, 'SEALED_RECEIPT', dict(identity=actual['identity'])))]
    manifest = out/'publication/execution_provenance.json'
    write(manifest, dict(status='DONE', schema='OVMAP_EXECUTION_PROVENANCE_V1', files=files))
    return manifest, files


def test_execution_provenance_export_roundtrip_redaction_and_idempotence(tmp_path):
    import ast, gzip
    from static_ovmap.benchmark_completion_v2.publication_export import candidates, export, decode_record
    from static_ovmap.benchmark_completion_v2.common import verified, digest_file
    out, repo = tmp_path/'out', tmp_path/'repo'
    repo.mkdir()
    manifest, files = _execution_provenance_fixture(out)
    ignored = out/'bootstrap/unlisted.py'
    ignored.write_text('# excluded by the explicit manifest\n')
    raw = out/'bootstrap/private_raw.npy'
    raw.write_bytes(b'private raw data')
    expected = {out/row['source'] for row in files} | {manifest}
    assert expected <= set(candidates(out)), 'Actual execution source and receipts must be included'
    assert ignored not in candidates(out) and raw not in candidates(out)
    before = {path: (digest_file(path), path.stat().st_mtime_ns) for path in expected | {ignored, raw}}
    binding = dict(output_root=str(out))
    result = export(binding, repo=repo, git_limit=1024*1024)
    for descriptor in files:
        row = next(row for row in result['files'] if row['source'] == descriptor['source'])
        assert row['sha256'] == descriptor['sha256'] and row['original_bytes'] == descriptor['bytes']
        payload = b''.join((repo/part['path']).read_bytes() for part in row['parts'])
        if descriptor['kind'] == 'SEALED_RECEIPT':
            restored = decode_record(payload)
            assert restored['original_identity'] == descriptor['identity']
            assert restored['record']['api_token'] == 'REDACTED'
            assert restored['record']['command'][-1] == '$OUT/bootstrap/accepted.py'
        else:
            assert row['encoding'] == 'GZIP_PATH_REDACTED_TEXT'
            text = gzip.decompress(payload).decode()
            assert str(out) not in text and '$OUT' in text
            if descriptor['kind'] == 'PYTHON_SOURCE':
                ast.parse(text)
    assert {path: (digest_file(path), path.stat().st_mtime_ns) for path in before} == before
    from static_ovmap.benchmark_completion_v2.common import ARTIFACTS
    index = repo/ARTIFACTS/'artifact_index.json'
    index_before = (digest_file(index), index.stat().st_mtime_ns)
    assert export(binding, repo=repo, git_limit=1024*1024)['identity'] == result['identity']
    assert (digest_file(index), index.stat().st_mtime_ns) == index_before
    assert verified(manifest)['schema'] == 'OVMAP_EXECUTION_PROVENANCE_V1'


def test_execution_provenance_rejects_paths_kinds_duplicates_and_raw_data(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates
    from static_ovmap.benchmark_completion_v2.common import verified, write, digest_file
    for case in ('absolute', 'traversal', 'outside_bootstrap', 'raw_suffix', 'wrong_kind', 'duplicate', 'wrong_schema'):
        out = tmp_path/case
        manifest, files = _execution_provenance_fixture(out)
        value = verified(manifest)
        if case == 'absolute': value['files'][0]['source'] = str(out/'bootstrap/accepted.py')
        elif case == 'traversal': value['files'][0]['source'] = 'bootstrap/../private.py'
        elif case == 'outside_bootstrap': value['files'][0]['source'] = 'benchmark/GT/receipt.json'
        elif case == 'raw_suffix':
            raw = out/'bootstrap/raw.npy'; raw.write_bytes(b'private data')
            value['files'][0].update(source='bootstrap/raw.npy', sha256=digest_file(raw), bytes=raw.stat().st_size)
        elif case == 'wrong_kind': value['files'][0]['kind'] = 'SEALED_RECEIPT'
        elif case == 'duplicate': value['files'].append(dict(value['files'][0]))
        elif case == 'wrong_schema': value['schema'] = 'UNDECLARED_SCHEMA'
        write(manifest, value)
        with pytest.raises(ValueError): candidates(out)


def test_execution_provenance_rejects_missing_changed_files_and_wrong_byte_counts(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates
    from static_ovmap.benchmark_completion_v2.common import verified, write
    for case in ('missing', 'changed', 'bytes', 'identity'):
        out = tmp_path/case
        manifest, files = _execution_provenance_fixture(out)
        value = verified(manifest)
        if case == 'missing': (out/files[0]['source']).unlink()
        elif case == 'changed': (out/files[0]['source']).write_text('CHANGED=True\n')
        elif case == 'bytes': value['files'][0]['bytes'] += 1
        elif case == 'identity': value['files'][-1]['identity'] = '0'*64
        write(manifest, value)
        with pytest.raises(ValueError): candidates(out)


def test_execution_provenance_rejects_symlink_files_and_ancestors(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates
    from static_ovmap.benchmark_completion_v2.common import verified, write
    for case in ('file', 'ancestor'):
        out = tmp_path/case
        manifest, files = _execution_provenance_fixture(out)
        value = verified(manifest)
        if case == 'file':
            path = out/files[0]['source']; real = path.with_name('real.py'); path.rename(real); path.symlink_to(real)
        else:
            path = out/'bootstrap'; real = out/'real_bootstrap'; path.rename(real); path.symlink_to(real, target_is_directory=True)
        write(manifest, value)
        with pytest.raises(ValueError): candidates(out)
