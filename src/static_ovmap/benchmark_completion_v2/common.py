"""Task-owned records; this module and the terminal entry path use stdlib only."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASK = 'ovmap-benchmark-completion-v2'
BASE = '9bbe040570ecbaec854e6d8aa94ba82a3a1afa2a'
DOCS = Path('docs/paper/static_ovmap/benchmark_completion_v2')
ARTIFACTS = Path('artifacts/static_ovmap/benchmark_completion_v2')
SEEDS = (17, 29)
METHODS = ('S_LL', 'G_LL', 'V_LL', 'MV_ATTN')
STATES = ('DONE', 'BLOCKED_DEPENDENCY', 'BLOCKED_RESOURCE', 'FAILED_RUNTIME', 'NOT_APPLICABLE')


class StageFailure(RuntimeError):
    def __init__(self, message, *, status='BLOCKED_DEPENDENCY', missing=()):
        super().__init__(message)
        if status not in STATES:
            raise ValueError('Unknown scientific phase status')
        self.status, self.missing = status, list(missing)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite scalar cannot enter a scientific JSON record')
    return value


def canonical_digest(value):
    return hashlib.sha256(json.dumps(plain(value), ensure_ascii=True, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def verified(path):
    value = read(path)
    expected = value.get('identity')
    if expected != canonical_digest({k: v for k, v in value.items() if k != 'identity'}):
        raise ValueError('Sealed input identity differs: ' + str(path))
    return value


def write(path, value):
    path = Path(path)
    result = plain({k: v for k, v in value.items() if k != 'identity'})
    result['identity'] = canonical_digest(result)
    data = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()
    if path.exists() and path.suffix != '.gz' and path.read_bytes() == data:
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as compressed:
                compressed.write(data)
        else:
            stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    return result


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            digest.update(part)
    return digest.hexdigest()


def source_identity(names):
    folder = Path(__file__).parent
    return canonical_digest({n: digest_file(folder / (n + '.py')) for n in names})


class ConsumedInputs:
    """Memoize exactly consumed files; never recursively scan historical assets."""
    def __init__(self, root):
        self.path = Path(root) / 'input_verifications.json'
        self.rows = verified(self.path)['files'] if self.path.exists() else {}

    def identity(self, path, expected=None):
        path = Path(path)
        stat = path.stat()
        key = str(path)
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev]
        old = self.rows.get(key)
        sha = old['sha256'] if old and old['stamp'] == stamp else digest_file(path)
        if expected and ((expected.get('sha256') and expected['sha256'] != sha)
                         or (expected.get('bytes') is not None and expected['bytes'] != stat.st_size)):
            raise ValueError('Consumed input differs from declared identity: ' + key)
        row = dict(path=key, bytes=stat.st_size, sha256=sha, stamp=stamp)
        self.rows[key] = row
        return row

    def save(self):
        # GPU input capture and CPU protocol/scoring can consume independent
        # assets concurrently. Merge under one task-local lock instead of
        # overwriting whichever process saved its memo most recently.
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = verified(self.path)['files'] if self.path.exists() else {}
            for key, row in self.rows.items():
                prior = current.get(key)
                if prior and prior['sha256'] != row['sha256']:
                    raise ValueError('Consumed input changed between concurrent phases: ' + key)
                current[key] = row
            self.rows = current
            return write(self.path, dict(files=current))


class Resolver:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def resolve(self, value):
        value = str(value)
        for old, new in sorted(self.mapping.items(), key=lambda kv: -len(kv[0])):
            if value == old or value.startswith(old.rstrip('/') + '/'):
                return str(new).rstrip('/') + value[len(old.rstrip('/')):]
        return value

    def rewrite(self, value):
        if isinstance(value, dict):
            return {k: self.rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.rewrite(v) for v in value]
        return self.resolve(value) if isinstance(value, str) and value.startswith('/') else value


def save_arrays(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
