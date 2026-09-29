"""Compact content manifest and verified normal research-branch publication."""

import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.m2_reviewer_study.evaluation import write_gzip
from src.static_ovmap.module_validation.contracts import atomic_write_json

from .binding import INDEX, ROOT
from .binding import read_json_or_gzip as read


def bundle(binding):
    root = Path(binding['output_root'])
    spec = read(binding['spec'])
    destination = ROOT / spec['publication']['artifact_root']
    selected = [root / 'binding.json', root / 'transfer_lock.json', root / 'report/data.json', root / 'review/outputs.json', root / 'review/requirements.json']
    for directory in ('evidence', 'execution', 'locked', 'probabilities', 'rows', 'pooled', 'sensitivity', 'diagnostics', 'evaluation_aliases', 'evaluation_context'):
        selected += sorted(p for p in (root / directory).rglob('*') if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.json.gz')))
    external = {}
    for scene in binding['scenes']:
        receipt = read(root / 'evidence' / scene / 'receipt.json')
        for row in receipt['inputs']:
            external[row['path']] = row
    files = []
    for source in selected:
        relative = source.relative_to(root)
        # Large JSON indices remain lossless, but gzip avoids duplicating bulky formatting.
        compress = source.suffix == '.json' and source.stat().st_size > 100_000
        target = destination / (str(relative) + '.gz' if compress else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        else:
            shutil.copyfile(source, target)
        data = gzip.decompress(target.read_bytes()) if compress else target.read_bytes()
        if data != source.read_bytes():
            raise ValueError('compact publication changed source bytes')
        if b'-----BEGIN PRIVATE KEY-----' in data or b'-----BEGIN OPENSSH PRIVATE KEY-----' in data:
            raise ValueError('private key detected in selected output')
        files.append({'source': INDEX.identity(source), 'published': INDEX.identity(target), 'lossless_gzip': compress})
    write_gzip(destination / 'external_inputs.json.gz', {'files': sorted(external.values(), key=lambda r: r['path']),
               'reconstruction': 'Use committed runner with complete bound shared caches; no inference permitted.'})
    manifest = {'status': 'BYTE_VERIFIED_COMPACT_BUNDLE', 'files': files,
                'excludes': ['original vision weights', 'RGB-D datasets', 'full old traces', 'prediction NPZ arrays', 'full masks']}
    atomic_write_json(destination / 'publication_manifest.json', manifest)
    size = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    if size > 100 * 1024**2:
        raise ValueError(f'compact publication exceeds 100 MiB: {size}')
    atomic_write_json(root / 'publication/bundle.json', {'status': manifest['status'], 'bytes': size, 'files': len(files),
                                                 'directory': str(destination)})
    return destination


def publish(binding):
    start = time.monotonic()
    root = Path(binding['output_root'])
    spec = read(binding['spec'])
    review = read(root / 'review/requirements.json')
    if review['status'] != 'PRIMARY_REVIEW_COMPLETE':
        raise ValueError('primary requirement review not complete')
    verified = read(root / 'review/outputs.json')
    if verified['status'] != 'REAL_OUTPUTS_VERIFIED':
        raise ValueError('real output verification missing')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != spec['branch']:
        raise ValueError('publication branch differs')
    previous_path = root / 'publication/final.json'
    if previous_path.exists():
        previous = read(previous_path)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], cwd=ROOT, text=True).split()[0]
        if head != previous['local_sha'] or remote != head:
            raise ValueError('published branch moved; do not silently replace its receipt')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise ValueError('publication already exists but worktree has new changes')
        return previous
    bundle(binding)
    scopes = ['src/static_ovmap/paired_evidence_study', 'tests/paired_evidence',
              'scripts/evaluation/run_ovimap_paired_evidence.py', 'docs/paper/static_ovmap/paired_evidence_v1',
              spec['publication']['artifact_root']] + ['docs/paper/static_ovmap/' + n for n in spec['reports']]
    # Refuse any pre-existing staged content outside this explicitly authorized task.
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
    if any(not any(p == s or p.startswith(s + '/') for s in scopes) for p in staged):
        raise ValueError('unrelated staged content must be preserved outside task commit')
    subprocess.run(['git', 'add', '--', *scopes], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    names = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
    if any(Path(n).suffix in {'.pt', '.pth', '.safetensors', '.npz', '.npy', '.jpg', '.png'} for n in names):
        raise ValueError('unexpected model/data/large-array file staged')
    if names:
        subprocess.run(['git', 'commit', '-m', 'Implement and publish frozen paired-evidence study'], cwd=ROOT, check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    push = subprocess.run(['git', 'push', '-u', 'origin', 'HEAD:refs/heads/' + branch], cwd=ROOT, text=True, capture_output=True, check=False)
    if push.returncode:
        raise RuntimeError(f'Normal push failed; local commit {sha} preserved: {push.stderr}')
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], cwd=ROOT, text=True).split()[0]
    if sha != remote:
        raise ValueError('local/remote full SHA mismatch')
    receipt = {'status': 'PUSH_VERIFIED', 'branch': branch, 'local_sha': sha, 'remote_sha': remote,
               'push_stdout': push.stdout, 'push_stderr': push.stderr, 'elapsed_seconds': time.monotonic() - start}
    write_once(root / 'publication/final.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'branch', 'local_sha', 'remote_sha')}), flush=True)
    return receipt
