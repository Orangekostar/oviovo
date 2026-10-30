"""A later scientific release replaces managed shards and proves a new result."""
import subprocess

import pytest

from static_ovmap.benchmark_completion_v2.common import ARTIFACTS, digest_file
from static_ovmap.benchmark_completion_v2 import publication


def repository(tmp_path):
    repo, remote = tmp_path / 'repo', tmp_path / 'remote.git'
    repo.mkdir()
    def git(*args, cwd=repo):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()
    git('init', '--bare', str(remote))
    git('init', '-b', publication.BRANCH)
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Synthetic test')
    git('remote', 'add', 'origin', str(remote))
    old = str(ARTIFACTS / 'chunks/old.part')
    (repo / old).parent.mkdir(parents=True)
    (repo / old).write_bytes(b'TEST ONLY previous numerical artifact')
    (repo / 'HANDOFF.md').write_text('TEST ONLY historical release.\n')
    git('add', '--', old, 'HANDOFF.md')
    git('commit', '-m', 'TEST ONLY previous release')
    git('push', 'origin', publication.BRANCH)
    return repo, remote, git, old


def test_next_c1_stages_removed_managed_shards_and_c2_stays_receipt_only(tmp_path):
    repo, remote, git, old = repository(tmp_path)
    parent = git('rev-parse', 'HEAD')
    (repo / old).unlink()
    new = str(ARTIFACTS / 'chunks/new.part')
    (repo / new).write_bytes(b'TEST ONLY new numerical artifact')
    args = dict(repo=repo, out=tmp_path / 'out', files=[old, new, 'HANDOFF.md'],
        receipt='publication/C1.json', handoff='HANDOFF.md',
        release_identity='TEST_ONLY_second_scientific_release', result_files=[new])
    result = publication.publish_transaction(**args)
    assert git('rev-parse', result['C1'] + '^') == parent
    assert old not in git('ls-tree', '-r', '--name-only', result['C1']).splitlines()
    assert new in git('ls-tree', '-r', '--name-only', result['C1']).splitlines()
    assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', result['C2']).splitlines()) == {
        'publication/C1.json', 'HANDOFF.md'}
    assert not git('status', '--porcelain')
    assert result['remote_result_read']['sha256'] == digest_file(repo / new)
    assert publication.publish_transaction(**args)['C2'] == result['C2']


def test_remote_numeric_candidates_exclude_files_already_in_parent_and_metadata(tmp_path):
    repo, _, _, old = repository(tmp_path)
    new = str(ARTIFACTS / 'chunks/new.part')
    metadata = str(ARTIFACTS / 'chunks/metadata.part')
    for name in [new, metadata]:
        (repo / name).write_bytes(b'TEST ONLY')
    rows = [dict(source=source, parts=[dict(path=path, distribution='GIT')]) for source, path in [
        ('benchmark/evaluation/replica48/office0/receipt.json', old),
        ('result_store.json', new), ('review/coverage.json', metadata)]]
    assert hasattr(publication, '_new_result_files'), 'Later releases must select newly added numerical shards'
    assert publication._new_result_files(repo, dict(files=rows)) == [new]


def test_deletion_support_does_not_expand_to_untracked_or_unrelated_files(tmp_path):
    repo, _, _, _ = repository(tmp_path)
    for name in [str(ARTIFACTS / 'chunks/untracked.part'), 'user_file.txt']:
        with pytest.raises(ValueError):
            publication.inventory(repo, [name])
